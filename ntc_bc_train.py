import torch
import math


def sample_lod_vaidyanathan(num_mips, device, max_useful_lod=None):
    """LOD 采样: 整数部分 ~ 指数 (∝ 4^(-k) 面积权重), 小数部分 ~ U[0,1).

    Args:
        num_mips: mip 总数.
        device: 张量设备.
        max_useful_lod: 训练采样的最高整数 LOD (含). None 表示 num_mips - 2.
            高于此 LOD 的 mip (例如 1x1, 2x2) 不参与训练, 把样本预算让给低 LOD.
            注意上限被进一步 clamp 到 num_mips - 2, 以保证 s0+1 仍是合法 mip.

    Returns:
        scale: shape [1] float, ∈ [0, max_int + 1), 可直接用于两 mip 线性混合.
    """
    max_int = num_mips - 2
    if max_useful_lod is not None:
        max_int = min(max_int, int(max_useful_lod))
    max_int = max(max_int, 0)

    X = torch.rand(1, device=device)
    lod_int = (-torch.log(X) / math.log(4)).floor().long().clamp(0, max_int)
    lam = torch.rand(1, device=device)
    return lod_int.float() + lam
