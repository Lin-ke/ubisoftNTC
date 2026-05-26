import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import argparse

from ntc_model import make_model
from ntc_train import load_brick_material


def reconstruct_normal(normal_xy):
    """Reconstruct z component of normal from x,y."""
    xy = normal_xy
    z = torch.sqrt(torch.clamp(1.0 - xy[:, 0:1] ** 2 - xy[:, 1:2] ** 2, min=0))
    return torch.cat([xy, z], dim=1)


def save_image(tensor, path, is_normal=False, is_srgb=True):
    """Save a [C, H, W] tensor as an image."""
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    if is_normal:
        img = (img + 1.0) / 2.0  # [-1,1] -> [0,1]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    Image.fromarray(img).save(path)


@torch.no_grad()
def infer_from_checkpoint(checkpoint_path, device='cuda'):
    # Load reference
    ref = load_brick_material('.', target_res=1024).to(device)
    output_dim = ref.shape[0]
    h, w = 1024, 1024

    # Build model
    model = make_model(reference_resolution=1024, output_dim=output_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Full resolution grid
    u = torch.linspace(0, 1, w, device=device)
    v = torch.linspace(0, 1, h, device=device)
    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)

    scale = torch.zeros(1, device=device)
    pred = model(uv, scale)  # [1, 9, 1024, 1024]

    os.makedirs('output', exist_ok=True)

    # Extract channels: albedo(3), normal(3), ao(1), roughness(1), metalness(1)
    albedo = pred[:, 0:3]
    normal_xy = pred[:, 3:5]
    normal_z_known = pred[:, 5:6]  # target normal_z (may not match XY exactly)
    ao = pred[:, 6:7]
    roughness = pred[:, 7:8]
    metalness = pred[:, 8:9]

    # For display, reconstruct normal Z from XY
    normal_full = reconstruct_normal(normal_xy)

    # Save predicted
    save_image(albedo.squeeze(0), 'output/albedo_pred.png')
    save_image(normal_full.squeeze(0), 'output/normal_pred.png', is_normal=True)
    save_image(ao.squeeze(0), 'output/ao_pred.png')
    save_image(roughness.squeeze(0), 'output/roughness_pred.png')
    save_image(metalness.squeeze(0), 'output/metalness_pred.png')

    # Save reference for comparison
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

    # Compute PSNR per channel
    for name, p, r in [
        ('Albedo', albedo, ref_albedo),
        ('Normal', normal_full, ref_normal),
        ('AO', ao, ref_ao),
        ('Roughness', roughness, ref_roughness),
        ('Metalness', metalness, ref_metalness),
    ]:
        mse = F.mse_loss(p, r).item()
        psnr = -10 * np.log10(mse + 1e-8)
        print(f"{name:12s} PSNR: {psnr:.2f} dB")

    mse_total = F.mse_loss(pred, ref.unsqueeze(0)).item()
    psnr_total = -10 * np.log10(mse_total + 1e-8)
    print(f"{'Total':12s} PSNR: {psnr_total:.2f} dB")

    print("\nSaved all images to output/ directory")


@torch.no_grad()
def infer_mip_comparison(checkpoint_path, device='cuda'):
    """Compare reconstruction across different mip levels."""
    ref = load_brick_material('.', target_res=1024).to(device)
    output_dim = ref.shape[0]

    model = make_model(reference_resolution=1024, output_dim=output_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    from ntc_train import build_mipmaps
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

        mse = F.mse_loss(pred, ref_batch).item()
        psnr = -10 * np.log10(mse + 1e-8)
        print(f"{mip_level:>4d}  {psnr:>10.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='output/best_model.pth')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'mips'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Please run training first: python ntc_train.py")
    else:
        if args.mode == 'full':
            infer_from_checkpoint(args.checkpoint, device)
        else:
            infer_mip_comparison(args.checkpoint, device)
