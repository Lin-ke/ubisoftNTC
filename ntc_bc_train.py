import torch
import math


def sample_lod_vaidyanathan(num_mips, device):
    """Vaidyanathan 2023 风格的 LOD 采样.

    95% 指数分布 (按 mip 面积比例): LOD = floor(-log4(X)), X ~ U(0,1)
    5%  均匀分布 (防止低分辨率 mip 欠采样).

    返回连续 scale，模拟 GPU 硬件 trilinear 采样.
    """
    if torch.rand(1, device=device) < 0.05:
        lod = torch.randint(0, num_mips, (1,), device=device).float()
    else:
        X = torch.rand(1, device=device)
        lod = (-torch.log(X) / math.log(4)).floor().clamp(0, num_mips - 1)
    lod = lod + torch.rand(1, device=device) * 0.999
    return lod.clamp(0, num_mips - 1)
