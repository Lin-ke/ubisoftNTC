"""NTC 公共工具函数 —— 推理可视化通用接口."""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def reconstruct_normal(normal_xy):
    """从 XY 分量重建法线 Z 分量."""
    xy = normal_xy
    z = torch.sqrt(torch.clamp(1.0 - xy[:, 0:1] ** 2 - xy[:, 1:2] ** 2, min=0))
    return torch.cat([xy, z], dim=1)


def save_image(tensor, path, is_normal=False):
    """将 [C, H, W] tensor 保存为图像."""
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    if is_normal:
        img = (img + 1.0) / 2.0
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    Image.fromarray(img).save(path)


def compute_psnr(pred, ref):
    """计算 PSNR (dB)."""
    mse = F.mse_loss(pred, ref).item()
    return -10 * np.log10(mse + 1e-8)
