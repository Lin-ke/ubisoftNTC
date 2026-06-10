"""NTC 公共工具函数 —— 推理可视化通用接口."""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def _rgb_to_rgb565_batch(rgb):
    """rgb: (N, 3) float [0,1] -> (N,) uint16"""
    r5 = np.clip(rgb[:, 0] * 31 + 0.5, 0, 31).astype(np.uint16)
    g6 = np.clip(rgb[:, 1] * 63 + 0.5, 0, 63).astype(np.uint16)
    b5 = np.clip(rgb[:, 2] * 31 + 0.5, 0, 31).astype(np.uint16)
    return (r5 << 11) | (g6 << 5) | b5


def _rgb565_to_rgb_batch(packed):
    """packed: (N,) uint16 -> (N, 3) float [0,1]"""
    r5 = ((packed >> 11) & 0x1F).astype(np.float32)
    g6 = ((packed >> 5) & 0x3F).astype(np.float32)
    b5 = (packed & 0x1F).astype(np.float32)
    return np.stack([r5 / 31.0, g6 / 63.0, b5 / 31.0], axis=1)


def _compress_bc1_blocks(tensor_np):
    """Vectorized BC1 compression.
    tensor_np: (3, H, W) numpy float32 [0,1], H/W divisible by 4.
    Returns (c0_565, c1_565, indices, blocks_h, blocks_w).
    """
    C, H, W = tensor_np.shape
    assert C == 3 and H % 4 == 0 and W % 4 == 0
    blocks_h, blocks_w = H // 4, W // 4
    N = blocks_h * blocks_w

    blocks = tensor_np.reshape(3, blocks_h, 4, blocks_w, 4).transpose(1, 3, 0, 2, 4)
    blocks = np.ascontiguousarray(blocks.reshape(N, 16, 3))

    pmin = blocks.min(axis=1)
    pmax = blocks.max(axis=1)
    inset = (pmax - pmin) / 4.0
    c0 = np.clip(pmax - inset, 0, 1)
    c1 = np.clip(pmin + inset, 0, 1)

    c0_565 = _rgb_to_rgb565_batch(c0)
    c1_565 = _rgb_to_rgb565_batch(c1)
    swap = c0_565 < c1_565
    c0_565[swap], c1_565[swap] = c1_565[swap].copy(), c0_565[swap].copy()
    c0[swap], c1[swap] = c1[swap].copy(), c0[swap].copy()

    c0_dec = _rgb565_to_rgb_batch(c0_565)
    c1_dec = _rgb565_to_rgb_batch(c1_565)
    palette = np.stack([
        c0_dec,
        c1_dec,
        (2 * c0_dec + c1_dec) / 3,
        (c0_dec + 2 * c1_dec) / 3,
    ], axis=1)

    diff = blocks[:, :, np.newaxis, :] - palette[:, np.newaxis, :, :]
    dists = (diff ** 2).sum(axis=3)
    indices = dists.argmin(axis=2).astype(np.uint8)

    return c0_565, c1_565, indices, blocks_h, blocks_w


def _decompress_bc1_blocks(c0_565, c1_565, indices, blocks_h, blocks_w):
    """Decompress BC1 blocks back to (3, H, W) numpy float32 [0,1]."""
    c0_dec = _rgb565_to_rgb_batch(c0_565)
    c1_dec = _rgb565_to_rgb_batch(c1_565)
    N = c0_565.shape[0]
    palette = np.stack([
        c0_dec,
        c1_dec,
        (2 * c0_dec + c1_dec) / 3,
        (c0_dec + 2 * c1_dec) / 3,
    ], axis=1)
    decoded = palette[np.arange(N)[:, None], indices]
    H, W = blocks_h * 4, blocks_w * 4
    decoded = decoded.reshape(blocks_h, blocks_w, 4, 4, 3).transpose(2, 0, 3, 1, 4)
    decoded = np.ascontiguousarray(decoded.reshape(3, H, W))
    return decoded


def compute_traditional_bc_psnr(ref_tensor, bc_format='bc1'):
    """Compute traditional BC compression PSNR vs original reference.

    ref_tensor: torch.Tensor (C, H, W) on CPU, in original data range.
    bc_format: 'bc1'|'bc2'|'bc3'|'bc4'|'bc5' (currently BC1 impl for RGB).
    Returns: float PSNR (dB) — compares original vs BC-compressed-then-decompressed.
    """
    C, H, W = ref_tensor.shape
    ref_np = ref_tensor.numpy().astype(np.float32)

    n_chan = min(3, C)
    ref_c = ref_np[:n_chan].copy()

    ref_min = float(ref_c.min())
    ref_max = float(ref_c.max())
    has_range = ref_max - ref_min > 1e-8

    if has_range:
        ref_mapped = (ref_c - ref_min) / (ref_max - ref_min)
    else:
        ref_mapped = np.zeros_like(ref_c)

    # BC1 需要 3 通道 RGB; 不足则补零通道, 使用后再切回.
    if n_chan < 3:
        padding = np.zeros((3 - n_chan, H, W), dtype=np.float32)
        ref_mapped = np.concatenate([ref_mapped, padding], axis=0)

    pad_h = (4 - H % 4) % 4
    pad_w = (4 - W % 4) % 4
    if pad_h > 0 or pad_w > 0:
        ref_mapped = np.pad(ref_mapped, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')

    c0, c1, indices, bh, bw = _compress_bc1_blocks(ref_mapped)
    dec_mapped = _decompress_bc1_blocks(c0, c1, indices, bh, bw)
    dec_mapped = dec_mapped[:n_chan, :H, :W]

    if has_range:
        dec_original = dec_mapped * (ref_max - ref_min) + ref_min
    else:
        dec_original = dec_mapped

    mse = np.mean((ref_c - dec_original) ** 2)
    psnr = -10 * np.log10(max(mse, 1e-10))
    return float(psnr)


def reconstruct_normal(normal_xy):
    """从 [0,1] 范围的 XY 分量重建法线 Z 分量.

    先将 [0,1] 映射到 [-1,1]，再按单位向量约束重建 Z。
    """
    xy = normal_xy * 2.0 - 1.0  # [0,1] -> [-1,1]
    z = torch.sqrt(torch.clamp(1.0 - xy[:, 0:1] ** 2 - xy[:, 1:2] ** 2, min=0))
    return torch.cat([xy, z], dim=1)


def save_image(tensor, path):
    """将 [C, H, W] tensor 保存为图像."""
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    Image.fromarray(img).save(path)


def compute_psnr(pred, ref):
    """计算 PSNR (dB)."""
    mse = F.mse_loss(pred, ref).item()
    return -10 * np.log10(mse + 1e-8)


def write_done_json(status, ckpt_path, data):
    """写入训练完成的 JSON 文件."""
    import json
    import os
    
    data['status'] = status
    data['ckpt'] = ckpt_path
    
    json_path = os.path.join("./.loopit", f'done.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)