"""
BC神经材质推理与可视化
===========================

从训练好的BC神经纹理模型checkpoint加载模型，重建所有材质层（反照率、法线、AO、粗糙度、金属度），
可视化推理结果，计算各项PSNR指标，以及评估BC压缩后的模型大小。

支持全部6种 BC 格式 (bc1~bc6)，通过 --bc-format 参数切换。

支持的推理模式（通过 --mode 参数选择）：
  - full   : 从checkpoint加载模型，重建所有材质层，保存预测图像和参考图像，打印各通道PSNR
  - mips   : 跨所有mip级别评估PSNR，对比重建结果与参考在各分辨率下的精度
  - size   : 计算BC压缩后特征网格的大小，与未压缩的fp16/fp32存储对比压缩率
  - compare: 重建结果与参考的逐像素对比，生成误差图（*_diff.png），打印各通道PSNR

各格式每 4x4 块每通道的 bits：
  - BC1: 2端点×5-bit + 16索引×2-bit             = 42 bits
  - BC2: 2端点×(8+5)-bit + 16索引×4-bit          ≈ 74-80 bits (alpha/color 分通道)
  - BC3: 2端点×(8+5)-bit + 16索引×3-bit          ≈ 58-64 bits (alpha/color 分通道)
  - BC4: 2端点×8-bit + 16索引×3-bit               = 64 bits
  - BC5: 2端点×8-bit + 16索引×3-bit               = 64 bits
  - BC6: 4端点×6-bit + 16索引×3-bit + 5-bit分区   = 77 bits

用法：
  python ntc_bc6_inference.py --bc-format bc1 --mode full
  python ntc_bc6_inference.py --bc-format bc6 --checkpoint output_bc6/best_model.pth --mode compare
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import argparse

from ntc_bc_model import make_bc_model, get_bc_format
from ntc_train import load_brick_material, build_mipmaps

# 默认模型配置 (inference 无 config 时)
_DEFAULT_MODEL_PARAMS = {
    'feature_configs': [
        (512, 8, 3),
        (256, 7, 3),
        (128, 6, 3),
        (64, 5, 3),
    ],
    'hidden_dim': 16,
    'num_layers': 1,
    'filter': 'trilinear',
    'half_pixel_offsets': [1, 3],
}


# ============================================================
# 各 BC 格式的每块每通道 bits 计算
# ============================================================

def get_bits_per_block(bc_format_name, feature_dim=3):
    """根据 BC 格式名称计算每个 4x4 块每通道的 bits 数。

    Args:
        bc_format_name: BC 格式名称 ('bc1'~'bc6')
        feature_dim:    每块的特征通道数 (用于 BC2/BC3 的 alpha/color 分通道计算)

    Returns:
        (bits_per_block_per_channel, description) 元组
    """
    fmt = bc_format_name.lower()
    if fmt == 'bc1':
        # 2端点×5-bit + 16索引×2-bit = 42
        bits = 2 * 5 + 16 * 2
        desc = "2eps×5-bit + 16idx×2-bit"
    elif fmt == 'bc2':
        # alpha 通道: 2端点×8-bit + 16索引×4-bit = 80
        # color 通道: 2端点×5-bit + 16索引×4-bit = 74
        split = feature_dim // 2
        alpha_bits = (2 * 8 + 16 * 4) * split
        color_bits = (2 * 5 + 16 * 4) * (feature_dim - split)
        bits = (alpha_bits + color_bits) / feature_dim
        desc = "2eps×(8/5)-bit + 16idx×4-bit (alpha/color split)"
    elif fmt == 'bc3':
        # alpha 通道: 2端点×8-bit + 16索引×3-bit = 64
        # color 通道: 2端点×5-bit + 16索引×3-bit = 58
        split = feature_dim // 2
        alpha_bits = (2 * 8 + 16 * 3) * split
        color_bits = (2 * 5 + 16 * 3) * (feature_dim - split)
        bits = (alpha_bits + color_bits) / feature_dim
        desc = "2eps×(8/5)-bit + 16idx×3-bit (alpha/color split)"
    elif fmt == 'bc4':
        # 2端点×8-bit + 16索引×3-bit = 64
        bits = 2 * 8 + 16 * 3
        desc = "2eps×8-bit + 16idx×3-bit"
    elif fmt == 'bc5':
        # 同 BC4
        bits = 2 * 8 + 16 * 3
        desc = "2eps×8-bit + 16idx×3-bit"
    elif fmt == 'bc6':
        # 4端点×6-bit + 16索引×3-bit + 5-bit分区 = 77
        bits = 4 * 6 + 16 * 3 + 5
        desc = "4eps×6-bit + 16idx×3-bit + 5-bit partition"
    else:
        raise ValueError(f"Unknown BC format: {bc_format_name}")
    return bits, desc


# ============================================================
# 工具函数
# ============================================================

def reconstruct_normal(normal_xy):
    """从XY分量重建法线的Z分量。

    法线是单位向量，满足 x^2 + y^2 + z^2 = 1。
    给定已预测的XY分量，通过 z = sqrt(1 - x^2 - y^2) 反推出Z分量。
    使用torch.clamp避免因浮点误差导致负数开根号。

    Args:
        normal_xy: 形状为 [B, 2, H, W] 的tensor，包含法线的X和Y分量

    Returns:
        形状为 [B, 3, H, W] 的tensor，完整的XYZ法线向量
    """
    xy = normal_xy
    z = torch.sqrt(torch.clamp(1.0 - xy[:, 0:1] ** 2 - xy[:, 1:2] ** 2, min=0))
    return torch.cat([xy, z], dim=1)


def save_image(tensor, path, is_normal=False):
    """将形如 [C, H, W] 的tensor保存为图像文件。

    处理流程：tensor -> numpy数组（HWC格式） -> 值域映射 -> uint8 -> PIL保存。
    对于法线图，需要将 [-1, 1] 范围映射到 [0, 1]。

    Args:
        tensor: 形状为 [C, H, W] 的torch tensor，值在 [0, 1] 之间（法线除外）
        path:   输出图像文件路径
        is_normal: 是否为法线图，法线值域为 [-1, 1]，需要做归一化映射
    """
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    if is_normal:
        img = (img + 1.0) / 2.0
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    Image.fromarray(img).save(path)


def compute_psnr(pred, ref):
    """计算 PSNR (dB)。"""
    mse = F.mse_loss(pred, ref).item()
    return -10 * np.log10(mse + 1e-8)


# ============================================================
# 推理模式实现
# ============================================================

@torch.no_grad()
def infer_from_checkpoint(checkpoint_path, bc_format_name, output_dir, device='cuda'):
    """从BC模型checkpoint加载模型，重建所有材质层并保存图像，打印各通道PSNR。

    Args:
        checkpoint_path: 模型checkpoint文件路径
        bc_format_name:  BC 格式名称 ('bc1'~'bc6')
        output_dir:      输出目录
        device:          'cuda' 或 'cpu'
    """
    ref = load_brick_material('.', target_res=1024).to(device)
    output_dim = ref.shape[0]
    h, w = 1024, 1024

    model = make_bc_model(_DEFAULT_MODEL_PARAMS, output_dim=output_dim,
                          bc_format_name=bc_format_name).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    u = torch.linspace(0, 1, w, device=device)
    v = torch.linspace(0, 1, h, device=device)
    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)

    scale = torch.zeros(1, device=device)
    pred = model(uv, scale)

    os.makedirs(output_dir, exist_ok=True)

    albedo = pred[:, 0:3]
    normal_xy = pred[:, 3:5]
    ao = pred[:, 6:7]
    roughness = pred[:, 7:8]
    metalness = pred[:, 8:9]
    normal_full = reconstruct_normal(normal_xy)

    # 保存预测图像
    save_image(albedo.squeeze(0), f'{output_dir}/albedo_pred.png')
    save_image(normal_full.squeeze(0), f'{output_dir}/normal_pred.png', is_normal=True)
    save_image(ao.squeeze(0), f'{output_dir}/ao_pred.png')
    save_image(roughness.squeeze(0), f'{output_dir}/roughness_pred.png')
    save_image(metalness.squeeze(0), f'{output_dir}/metalness_pred.png')

    # 保存参考图像
    ref_albedo = ref[0:3].unsqueeze(0)
    ref_normal = ref[3:6].unsqueeze(0)
    ref_ao = ref[6:7].unsqueeze(0)
    ref_roughness = ref[7:8].unsqueeze(0)
    ref_metalness = ref[8:9].unsqueeze(0)

    save_image(ref_albedo.squeeze(0), f'{output_dir}/albedo_ref.png')
    save_image(ref_normal.squeeze(0), f'{output_dir}/normal_ref.png', is_normal=True)
    save_image(ref_ao.squeeze(0), f'{output_dir}/ao_ref.png')
    save_image(ref_roughness.squeeze(0), f'{output_dir}/roughness_ref.png')
    save_image(ref_metalness.squeeze(0), f'{output_dir}/metalness_ref.png')

    # 打印各通道PSNR
    print(f"\n[{bc_format_name.upper()}] Inference Results:")
    print(f"{'Channel':>12s}  {'PSNR (dB)':>10s}")
    print('-' * 28)
    for name, p, r in [
        ('Albedo', albedo, ref_albedo),
        ('Normal', normal_full, ref_normal),
        ('AO', ao, ref_ao),
        ('Roughness', roughness, ref_roughness),
        ('Metalness', metalness, ref_metalness),
    ]:
        psnr = compute_psnr(p, r)
        print(f"{name:>12s}  {psnr:>10.2f}")

    psnr_total = compute_psnr(pred, ref.unsqueeze(0))
    print(f"{'Total':>12s}  {psnr_total:>10.2f}")
    print(f"\nSaved all images to {output_dir}/ directory")


@torch.no_grad()
def infer_mip_comparison(checkpoint_path, bc_format_name, output_dir, device='cuda'):
    """跨mip级别PSNR评估：在不同分辨率下评估模型重建精度。

    Args:
        checkpoint_path: 模型checkpoint文件路径
        bc_format_name:  BC 格式名称 ('bc1'~'bc6')
        output_dir:      输出目录
        device:          'cuda' 或 'cpu'
    """
    ref = load_brick_material('.', target_res=1024).to(device)
    output_dim = ref.shape[0]

    model = make_bc_model(_DEFAULT_MODEL_PARAMS, output_dim=output_dim,
                          bc_format_name=bc_format_name).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    ref_mips = build_mipmaps(ref.cpu())

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[{bc_format_name.upper()}] Mip-level PSNR:")
    print(f"{'Mip':>4s}  {'Res':>8s}  {'PSNR(dB)':>10s}")
    print('-' * 30)

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
        print(f"{mip_level:>4d}  {size:>4d}x{size:<4d}  {psnr:>10.2f}")


@torch.no_grad()
def compute_model_size(checkpoint_path, bc_format_name):
    """计算BC压缩后特征网格的大小，并与未压缩存储进行对比。

    根据所选 BC 格式动态计算每块 bits 数。

    Args:
        checkpoint_path: 模型checkpoint文件路径
        bc_format_name:  BC 格式名称 ('bc1'~'bc6')
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']

    # 从 endpoints 的 shape 反推每个 mip 的分辨率
    # endpoints shape: [blocks_h, blocks_w, num_endpoints, feature_dim]
    mip_info = []  # [(resolution, feature_dim), ...]
    seen_res = set()
    for key in sorted(state.keys()):
        if 'endpoints' in key and 'raw' not in key:
            shape = state[key].shape
            if len(shape) == 4:
                bh, bw = shape[0], shape[1]
                fdim = shape[3]
                res = bh * 4
                if res not in seen_res:
                    mip_info.append((res, fdim))
                    seen_res.add(res)

    if not mip_info:
        print("Warning: could not detect block sizes from checkpoint.")
        print("Assuming default resolutions: [512, 256, 128, 64]")
        mip_info = [(512, 3), (256, 3), (128, 3), (64, 3)]

    mip_info.sort(key=lambda x: -x[0])

    feature_dim = mip_info[0][1] if mip_info else 3
    bits_per_ch, bits_desc = get_bits_per_block(bc_format_name, feature_dim)
    bytes_per_block = (int(bits_per_ch * feature_dim) + 7) // 8

    total_bc_bytes = 0
    print(f"\n[{bc_format_name.upper()}] Feature Grid Size:")
    print(f"  Format: {bits_desc}")
    print(f"  Bits per block per channel: {bits_per_ch:.1f}")
    print(f"{'Mip':>4s}  {'Resolution':>10s}  {'Blocks':>10s}  {'Bytes':>12s}")
    print('-' * 50)
    for i, (res, fdim) in enumerate(mip_info):
        blocks = (res // 4) * (res // 4)
        mip_bytes = blocks * bytes_per_block
        total_bc_bytes += mip_bytes
        print(f"  {i}    {res:>5d}x{res:<5d}  {blocks:>10d}  {mip_bytes:>12,d}")

    print('-' * 50)
    print(f"Total {bc_format_name.upper()} features: {total_bc_bytes:>30,d} bytes")

    mlp_params = sum(v.numel() for k, v in state.items() if 'mlp' in k)
    mlp_bytes_fp16 = mlp_params * 2
    print(f"\nMLP weights (fp16): {mlp_params} params x 2 bytes = {mlp_bytes_fp16:,} bytes")

    total_compressed = total_bc_bytes + mlp_bytes_fp16
    print(f"Total compressed size ({bc_format_name.upper()} + MLP fp16): {total_compressed:,} bytes")

    total_pixels = sum(res * res for res, _ in mip_info)
    features_fp16 = total_pixels * feature_dim * 2
    ratio = features_fp16 / max(total_bc_bytes, 1)
    print(f"\nCompression ratio (features only): {bc_format_name.upper()}={total_bc_bytes:,}B vs fp16={features_fp16:,}B = {ratio:.1f}x")

    return total_bc_bytes, total_compressed


@torch.no_grad()
def compare_methods(checkpoint_path, bc_format_name, output_dir, device='cuda'):
    """重建结果与参考的逐像素对比：生成预测图、参考图和误差图。

    Args:
        checkpoint_path: 模型checkpoint文件路径
        bc_format_name:  BC 格式名称 ('bc1'~'bc6')
        output_dir:      输出目录
        device:          'cuda' 或 'cpu'
    """
    ref = load_brick_material('.', target_res=1024).to(device)
    output_dim = ref.shape[0]
    h, w = 1024, 1024

    model = make_bc_model(_DEFAULT_MODEL_PARAMS, output_dim=output_dim,
                          bc_format_name=bc_format_name).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    u = torch.linspace(0, 1, w, device=device)
    v = torch.linspace(0, 1, h, device=device)
    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)

    scale = torch.zeros(1, device=device)
    pred = model(uv, scale)

    os.makedirs(output_dir, exist_ok=True)

    albedo = pred[:, 0:3]
    normal_xy = pred[:, 3:5]
    ao = pred[:, 6:7]
    roughness = pred[:, 7:8]
    metalness = pred[:, 8:9]
    normal_full = reconstruct_normal(normal_xy)

    ref_albedo = ref[0:3].unsqueeze(0)
    ref_normal = ref[3:6].unsqueeze(0)
    ref_ao = ref[6:7].unsqueeze(0)
    ref_roughness = ref[7:8].unsqueeze(0)
    ref_metalness = ref[8:9].unsqueeze(0)

    # 计算并保存误差图
    for name, pred_img, ref_img in [
        ('albedo', albedo, ref_albedo),
        ('normal', normal_full, ref_normal),
        ('ao', ao, ref_ao),
        ('roughness', roughness, ref_roughness),
        ('metalness', metalness, ref_metalness),
    ]:
        diff = (pred_img - ref_img).abs()
        if diff.shape[1] > 1:
            diff = diff.mean(dim=1, keepdim=True)

        save_image(pred_img.squeeze(0), f'{output_dir}/{name}_pred.png',
                   is_normal=(name == 'normal'))
        save_image(ref_img.squeeze(0), f'{output_dir}/{name}_ref.png',
                   is_normal=(name == 'normal'))
        save_image(diff.squeeze(0), f'{output_dir}/{name}_diff.png')

    # 打印各通道PSNR
    print(f"\n[{bc_format_name.upper()}] Compare Results:")
    print(f"{'Channel':>12s}  {'PSNR (dB)':>10s}")
    print('-' * 28)

    for name, p, r in [
        ('Albedo', albedo, ref_albedo),
        ('Normal', normal_full, ref_normal),
        ('AO', ao, ref_ao),
        ('Roughness', roughness, ref_roughness),
        ('Metalness', metalness, ref_metalness),
    ]:
        psnr = compute_psnr(p, r)
        print(f"{name:>12s}  {psnr:>10.2f}")

    psnr_total = compute_psnr(pred, ref.unsqueeze(0))
    print('-' * 28)
    print(f"{'Total':>12s}  {psnr_total:>10.2f}")
    print(f"\nError maps saved as *_diff.png in {output_dir}/")


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BC Neural Texture Inference')
    parser.add_argument('--bc-format', type=str, default='bc6',
                        choices=['bc1', 'bc2', 'bc3', 'bc4', 'bc5', 'bc6'],
                        help='BC 压缩格式 (default: bc6)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型checkpoint文件路径 (默认: output_{bc_format}/best_model.pth)')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'mips', 'size', 'compare'],
                        help='推理模式：full=完整重建, mips=跨mip级PSNR, size=压缩率对比, compare=逐像素误差')
    args = parser.parse_args()

    bc_fmt = args.bc_format
    output_dir = f'output_{bc_fmt}'
    checkpoint = args.checkpoint or f'{output_dir}/best_model.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"BC format: {bc_fmt.upper()}")
    print(f"Loading checkpoint: {checkpoint}")

    if not os.path.exists(checkpoint):
        print(f"\nCheckpoint not found: {checkpoint}")
        print(f"Please run training first:  python ntc_bc6_train.py configs/{bc_fmt}_bcf05k.yaml")
    else:
        if args.mode == 'full':
            infer_from_checkpoint(checkpoint, bc_fmt, output_dir, device)
        elif args.mode == 'mips':
            infer_mip_comparison(checkpoint, bc_fmt, output_dir, device)
        elif args.mode == 'size':
            compute_model_size(checkpoint, bc_fmt)
        elif args.mode == 'compare':
            compare_methods(checkpoint, bc_fmt, output_dir, device)
