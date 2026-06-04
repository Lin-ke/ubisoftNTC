import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import argparse

from ntc_model import make_model
from dataset import load_material
from ntc_utils import reconstruct_normal, save_image, compute_psnr


_DEFAULT_MODEL_PARAMS = {
    'feature_configs': [
        (512, 8, 3),
        (256, 7, 3),
        (128, 6, 3),
        (64, 5, 3),
    ],
    'hidden_dim': 16,
    'num_layers': 2,
    'filter': 'trilinear',
}


@torch.no_grad()
def infer_from_checkpoint(checkpoint_path, material_dir, target_res=1024, device='cuda'):
    ref = load_material(material_dir, target_res=target_res).to(device)
    output_dim = ref.shape[0]
    h, w = ref.shape[1], ref.shape[2]

    model = make_model(_DEFAULT_MODEL_PARAMS, output_dim=output_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    u = torch.linspace(0, 1, w, device=device)
    v = torch.linspace(0, 1, h, device=device)
    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)

    scale = torch.zeros(1, device=device)
    pred = model(uv, scale)

    os.makedirs('output', exist_ok=True)

    albedo = pred[:, 0:3]
    normal_xy = pred[:, 3:5]
    normal_z_known = pred[:, 5:6]
    ao = pred[:, 6:7]
    roughness = pred[:, 7:8]
    metalness = pred[:, 8:9]

    normal_full = reconstruct_normal(normal_xy)

    save_image(albedo.squeeze(0), 'output/albedo_pred.png')
    save_image(normal_full.squeeze(0), 'output/normal_pred.png', is_normal=True)
    save_image(ao.squeeze(0), 'output/ao_pred.png')
    save_image(roughness.squeeze(0), 'output/roughness_pred.png')
    save_image(metalness.squeeze(0), 'output/metalness_pred.png')

    ref_albedo = ref[0:3].unsqueeze(0)
    ref_normal = ref[3:6].unsqueeze(0)
    ref_ao = ref[6:7].unsqueeze(0)
    ref_roughness = ref[7:8].unsqueeze(0)
    ref_metalness = ref[8:9].unsqueeze(0)

    save_image(ref_albedo.squeeze(0), 'output/albedo_ref.png')
    save_image(ref_normal.squeeze(0), 'output/normal_ref.png', is_normal=True)
    save_image(ref_ao.squeeze(0), 'output/ao_ref.png')
    save_image(ref_roughness.squeeze(0), 'output/roughness_ref.png')
    save_image(ref_metalness.squeeze(0), 'output/metalness_ref.png')

    for name, p, r in [
        ('Albedo', albedo, ref_albedo),
        ('Normal', normal_full, ref_normal),
        ('AO', ao, ref_ao),
        ('Roughness', roughness, ref_roughness),
        ('Metalness', metalness, ref_metalness),
    ]:
        psnr = compute_psnr(p, r)
        print(f"{name:12s} PSNR: {psnr:.2f} dB")

    psnr_total = compute_psnr(pred, ref.unsqueeze(0))
    print(f"{'Total':12s} PSNR: {psnr_total:.2f} dB")
    print("\nSaved all images to output/ directory")


@torch.no_grad()
def infer_mip_comparison(checkpoint_path, material_dir, target_res=1024, device='cuda'):
    ref = load_material(material_dir, target_res=target_res).to(device)
    output_dim = ref.shape[0]

    model = make_model(_DEFAULT_MODEL_PARAMS, output_dim=output_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    from dataset import build_mipmaps
    ref_mips = build_mipmaps(ref.cpu())

    os.makedirs('output', exist_ok=True)
    print(f"{'Mip':>4s}  {'PSNR(dB)':>10s}")
    print('-' * 20)

    for mip_level in range(len(ref_mips)):
        ref_mip = ref_mips[mip_level].to(device)
        size = ref_mip.shape[1]

        u = torch.linspace(0, 1, size, device=device)
        v = torch.linspace(0, 1, size, device=device)
        uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)
        scale = torch.tensor([float(mip_level)], device=device)

        pred = model(uv, scale)
        ref_batch = ref_mip.unsqueeze(0)

        psnr = compute_psnr(pred, ref_batch)
        print(f"{mip_level:>4d}  {psnr:>10.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='output/best_model.pth')
    parser.add_argument('--material-dir', type=str, default='dataset/aerial_beach_02',
                        help='材质目录路径')
    parser.add_argument('--target-res', type=int, default=1024)
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'mips'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Please run training first: python evaluate.py --config configs/bc1_bcf05k.yaml --train-uc")
    else:
        if args.mode == 'full':
            infer_from_checkpoint(args.checkpoint, args.material_dir, args.target_res, device)
        else:
            infer_mip_comparison(args.checkpoint, args.material_dir, args.target_res, device)
