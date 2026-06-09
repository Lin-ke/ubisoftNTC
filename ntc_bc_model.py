"""
BC 块压缩特征模型 —— 支持 BC1~BC5 (LDR 单线段, 无分区).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ntc_visualization import (
    _normalize_channel_names,
    _default_inference_channels,
    _channel_indices,
)


class BCFormat:

    def get_endpoint_init_scale(self, feature_dim):
        return 0.01

    def get_endpoint_count(self):
        return 2

    def get_endpoint_bits(self, feature_dim):
        return [8] * feature_dim

    def get_index_bits(self):
        return 3


# ============================================================
# BC1 格式实现 (DXT1)
# ============================================================

class BC1Format(BCFormat):

    def get_endpoint_bits(self, feature_dim):
        pattern = [5, 6, 5]
        return [pattern[i % 3] for i in range(feature_dim)]

    def get_index_bits(self):
        return 2


# ============================================================
# BC2 格式实现 (DXT3)
# ============================================================

class BC2Format(BCFormat):

    def get_endpoint_bits(self, feature_dim):
        split = feature_dim // 2
        return [8 if i < split else 5 for i in range(feature_dim)]

    def get_index_bits(self):
        return 4


# ============================================================
# BC3 格式实现 (DXT5)
# ============================================================

class BC3Format(BCFormat):

    def get_endpoint_bits(self, feature_dim):
        split = feature_dim // 2
        return [8 if i < split else 5 for i in range(feature_dim)]

    def get_index_bits(self):
        return 3


# ============================================================
# BC4 / BC5 格式实现 (RGTC1 / RGTC2)
# 端点 8-bit / 索引 3-bit 与 BCFormat 默认值一致, 无需重写方法.
# ============================================================

class BC4Format(BCFormat):
    pass


class BC5Format(BCFormat):
    pass


# ============================================================
# 格式注册表
# ============================================================

_FORMATS = {
    'bc1': BC1Format,
    'bc2': BC2Format,
    'bc3': BC3Format,
    'bc4': BC4Format,
    'bc5': BC5Format,
}


def get_bc_format(name: str) -> BCFormat:
    """根据名称获取 BC 格式实例."""
    if name not in _FORMATS:
        raise ValueError(f"未知 BC 格式: '{name}'，支持: {list(_FORMATS.keys())}")
    return _FORMATS[name]()


# ============================================================
# BC 块特征层 —— 通用 4×4 块压缩表示 (单线段, 无分区)
# ============================================================

class BCBlockFeature(nn.Module):

    def __init__(self, resolution_w, resolution_h, feature_dim, bc_format):
        super().__init__()
        assert resolution_w % 4 == 0 and resolution_h % 4 == 0
        self.resolution_w = resolution_w
        self.resolution_h = resolution_h
        self.feature_dim = feature_dim
        self.blocks_w = resolution_w // 4
        self.blocks_h = resolution_h // 4
        self.bc_format = bc_format

        num_eps = bc_format.get_endpoint_count()
        init_scale = bc_format.get_endpoint_init_scale(feature_dim)
        self.endpoints = nn.Parameter(
            torch.randn(self.blocks_h, self.blocks_w, num_eps, feature_dim) * init_scale
        )

        self.raw_indices = nn.Parameter(
            torch.zeros(self.blocks_h, self.blocks_w, 16)
        )

    def _quantize_ste(self, x, levels_minus_one):
        levels = float(levels_minus_one)
        xq = (x * levels).round().clamp(0, levels) / levels
        if not self.training:
            return xq
        return x + (xq - x).detach()

    def forward(self):
        bh, bw = self.blocks_h, self.blocks_w
        C = self.feature_dim

        alpha = torch.sigmoid(self.raw_indices)
        idx_bits = self.bc_format.get_index_bits()
        alpha_q = self._quantize_ste(alpha, (1 << idx_bits) - 1)

        eps = torch.sigmoid(self.endpoints)
        eps_bits = self.bc_format.get_endpoint_bits(C)
        eps_q = eps.clone()
        for ch in range(C):
            b = eps_bits[ch]
            eps_q[..., ch] = self._quantize_ste(eps[..., ch], (1 << b) - 1)

        e0 = eps_q[:, :, 0, :]
        e1 = eps_q[:, :, 1, :]
        x = alpha_q.unsqueeze(-1)
        y = e0.unsqueeze(2) + x * (e1.unsqueeze(2) - e0.unsqueeze(2))

        y = y.reshape(bh, bw, 4, 4, C)
        y = y.permute(0, 2, 1, 3, 4)
        y = y.reshape(self.resolution_h, self.resolution_w, C)
        return y.permute(2, 0, 1).unsqueeze(0)

    @torch.no_grad()
    def init_from_continuous(self, x):
        """从 UC 连续特征 x 初始化 BC 端点/索引 (b. box diagonal 启发式)."""
        assert x.dim() == 4 and x.shape[0] == 1, f"expect [1,C,H,W], got {tuple(x.shape)}"
        C = x.shape[1]
        H, W = x.shape[2], x.shape[3]
        bh, bw = self.blocks_h, self.blocks_w
        assert H == 4 * bh and W == 4 * bw, \
            f"shape mismatch: x {tuple(x.shape)} vs blocks {bh}x{bw}*4"
        assert C == self.feature_dim, \
            f"channel mismatch: x={C} vs feature_dim={self.feature_dim}"

        # [1, C, H, W] -> [bh, bw, 16, C]
        x = x.squeeze(0).to(self.endpoints.device)
        x = x.reshape(C, bh, 4, bw, 4).permute(1, 3, 2, 4, 0).contiguous()
        blocks = x.reshape(bh, bw, 16, C)

        # bbox endpoints per block
        e0 = blocks.min(dim=2).values  # [bh, bw, C]
        e1 = blocks.max(dim=2).values  # [bh, bw, C]

        # 投影 alpha = <P - e0, e1 - e0> / |e1 - e0|^2
        direction = (e1 - e0).unsqueeze(2)                  # [bh, bw, 1, C]
        centered = blocks - e0.unsqueeze(2)                 # [bh, bw, 16, C]
        dot = (centered * direction).sum(dim=-1)            # [bh, bw, 16]
        norm_sq = (direction * direction).sum(dim=-1).clamp(min=1e-8)  # [bh, bw, 1]
        alpha = (dot / norm_sq).clamp(0.0, 1.0)             # [bh, bw, 16]

        # logit 反 sigmoid (forward 时会 sigmoid)
        eps = 1e-3
        e0_c = e0.clamp(eps, 1 - eps)
        e1_c = e1.clamp(eps, 1 - eps)
        a_c = alpha.clamp(eps, 1 - eps)
        eps_endpoints = torch.stack([
            torch.log(e0_c / (1 - e0_c)),
            torch.log(e1_c / (1 - e1_c)),
        ], dim=2)  # [bh, bw, 2, C]
        raw_indices = torch.log(a_c / (1 - a_c))   # [bh, bw, 16]

        self.endpoints.data.copy_(eps_endpoints)
        self.raw_indices.data.copy_(raw_indices)


# ============================================================
# Mipmap 特征网格
# ============================================================

class BCMipmapFeatureGrid(nn.Module):
    """BC 压缩的多分辨率特征金字塔, 支持三线性/三立方插值采样."""

    def __init__(self, base_resolution, num_mips, feature_dim, bc_format,
                 filter_mode='trilinear', half_pixel_offset=False):
        super().__init__()
        self.base_resolution = base_resolution
        self.num_mips = num_mips
        self.feature_dim = feature_dim
        self.filter_mode = filter_mode
        self.half_pixel_offset = half_pixel_offset

        self.mips = nn.ModuleList([
            BCBlockFeature(base_resolution >> i, base_resolution >> i,
                           feature_dim, bc_format)
            for i in range(num_mips)
        ])

    def sample(self, uv, scale):
        B = uv.shape[0]
        s = torch.clamp(scale, 0, self.num_mips - 1)
        s0 = s.long()
        s1 = torch.clamp(s0 + 1, max=self.num_mips - 1)
        lam = (s - s0.float()).view(-1, 1, 1, 1)

        uv_offset = uv
        if self.half_pixel_offset and s0.max() < len(self.mips):
            # 半像素偏移: 当前 mip 的 0.5/分辨率
            res0 = self.base_resolution >> int(s0[0].item())
            uv_offset = uv + 0.5 / res0

        uv_grid = uv_offset * 2 - 1

        # 空间插值模式: trilinear → bilinear, tricubic → bicubic
        spatial_mode = 'bilinear' if self.filter_mode == 'trilinear' else 'bicubic'

        # 仅解压本 batch 实际用到的 mip（s0 ∪ s1），避免每次迭代解压所有 mip。
        needed = set()
        for b in range(B):
            needed.add(int(s0[b].item()))
            needed.add(int(s1[b].item()))
        decompressed = {idx: self.mips[idx]() for idx in needed}

        feat0, feat1 = [], []
        for b in range(B):
            f0 = F.grid_sample(decompressed[int(s0[b].item())], uv_grid[b:b + 1],
                               mode=spatial_mode, padding_mode='border',
                               align_corners=False)
            f1 = F.grid_sample(decompressed[int(s1[b].item())], uv_grid[b:b + 1],
                               mode=spatial_mode, padding_mode='border',
                               align_corners=False)
            feat0.append(f0)
            feat1.append(f1)

        feat0 = torch.cat(feat0, dim=0)
        feat1 = torch.cat(feat1, dim=0)
        return (1 - lam) * feat0 + lam * feat1

    @torch.no_grad()
    def init_from_uc_grid(self, uc_grid):
        """从 UC MipmapFeatureGrid 初始化所有 mip 的 BC 端点/索引."""
        if len(uc_grid.mips) != len(self.mips):
            print(f"  [WARN] UC mips={len(uc_grid.mips)} vs BC mips={len(self.mips)}; "
                  f"only first {min(len(uc_grid.mips), len(self.mips))} will be initialized")
        for j, bc_mip in enumerate(self.mips):
            if j >= len(uc_grid.mips):
                break
            uc_mip = uc_grid.mips[j]  # [1, C, H, W]
            if (uc_mip.shape[1] != bc_mip.feature_dim
                    or uc_mip.shape[2] != bc_mip.resolution_h
                    or uc_mip.shape[3] != bc_mip.resolution_w):
                print(f"  [WARN] UC mip {j} shape {tuple(uc_mip.shape)} != "
                      f"BC mip [{bc_mip.feature_dim},{bc_mip.resolution_h},{bc_mip.resolution_w}], skip")
                continue
            bc_mip.init_from_continuous(uc_mip.detach())


# ============================================================
# 完整神经 BC 材质模型
# ============================================================

class NeuralBCTextureModel(nn.Module):
    """神经 BC 材质模型: 多网格特征 → 拼接 → MLP 解码器."""

    def __init__(self, feature_configs, hidden_dim, output_dim, bc_format,
                 filter_mode='trilinear', half_pixel_offsets=None, num_layers=1,
                 output_activation='none'):
        super().__init__()
        if half_pixel_offsets is None:
            half_pixel_offsets = []
        self.feature_grids = nn.ModuleList([
            BCMipmapFeatureGrid(res, mips, dim, bc_format, filter_mode,
                                half_pixel_offset=(i in half_pixel_offsets))
            for i, (res, mips, dim) in enumerate(feature_configs)
        ])
        total_input_dim = sum(dim for _, _, dim in feature_configs)

        layers = []
        in_dim = total_input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

        self.total_input_dim = total_input_dim
        self.output_dim = output_dim

    def forward(self, uv, scales):
        features = []
        for grid in self.feature_grids:
            features.append(grid.sample(uv, scales))

        x = torch.cat(features, dim=1)
        x = x.permute(0, 2, 3, 1)
        y = self.mlp(x)
        return y.permute(0, 3, 1, 2)

    # ------------------------------------------------------------------
    # 推理通道选择 (eval / 可视化时按 PBR 通道名取子集)
    # ------------------------------------------------------------------

    def set_inference_channels(self, channels=None, ref_dim=9):
        names = _normalize_channel_names(channels)
        if names is None:
            names = _default_inference_channels(self.output_dim, ref_dim=ref_dim)
        self.inference_channels = names
        self.inference_channel_indices = _channel_indices(names, ref_dim=ref_dim)
        return names

    def get_inference_channels(self):
        names = _normalize_channel_names(getattr(self, 'inference_channels', None))
        if names is None:
            names = self.set_inference_channels()
        return names

    # ------------------------------------------------------------------
    # 压缩比统计 (BC features 比特数 + MLP 参数比特数 vs 原始 PNG 字节数)
    # ------------------------------------------------------------------

    def compute_storage_bits(self, mlp_param_bits=16):
        total_bits = 0
        for grid in self.feature_grids:
            for mip in grid.mips:
                fmt = mip.bc_format
                C = mip.feature_dim
                num_eps = fmt.get_endpoint_count()
                sum_eps_bits = sum(fmt.get_endpoint_bits(C))
                idx_bits = fmt.get_index_bits()
                block_bits = num_eps * sum_eps_bits + 16 * idx_bits
                total_bits += mip.blocks_h * mip.blocks_w * block_bits
        mlp_params = sum(p.numel() for p in self.mlp.parameters())
        return total_bits + mlp_params * mlp_param_bits

    @staticmethod
    def compute_reference_bits(dataset_root, material_name):
        mdir = os.path.join(dataset_root, material_name)
        total_bytes = sum(
            os.path.getsize(os.path.join(mdir, f))
            for f in os.listdir(mdir) if f.endswith('.png')
        )
        return total_bytes * 8

    def compute_compression_stats(self, dataset_root, material_name, mlp_param_bits=16):
        bc_bits = self.compute_storage_bits(mlp_param_bits=mlp_param_bits)
        png_bits = self.compute_reference_bits(dataset_root, material_name)
        return {
            'bc_bits': bc_bits,
            'png_bits': png_bits,
            'compression_ratio': round(bc_bits / max(png_bits, 1), 4),
        }

    @torch.no_grad()
    def init_from_uc(self, uc_model):
        """复制 UC MLP 权重并用 UC 特征离线压缩初始化 BC 端点/索引."""
        # 1) MLP
        self.mlp.load_state_dict(uc_model.mlp.state_dict())

        # 2) 特征
        if len(self.feature_grids) != len(uc_model.feature_grids):
            print(f"  [WARN] UC feature_grids={len(uc_model.feature_grids)} "
                  f"vs BC={len(self.feature_grids)}; mismatch may corrupt init")
        for bc_grid, uc_grid in zip(self.feature_grids, uc_model.feature_grids):
            bc_grid.init_from_uc_grid(uc_grid)


# ============================================================
# 工厂函数
# ============================================================

def make_bc_model(model_params, output_dim=9, bc_format_name='bc1'):
    """从配置创建 BC 神经材质模型.

    Args:
        model_params: dict, 由 get_model_params(config) 产生.
            包含: feature_configs, hidden_dim, filter, half_pixel_offsets, num_layers
        output_dim: MLP 输出维度
        bc_format_name: BC 格式名称 (bc1~bc5)
    """
    bc_format = get_bc_format(bc_format_name)

    return NeuralBCTextureModel(
        feature_configs=model_params['feature_configs'],
        hidden_dim=model_params['hidden_dim'],
        output_dim=output_dim,
        bc_format=bc_format,
        filter_mode=model_params.get('filter', 'trilinear'),
        half_pixel_offsets=model_params.get('half_pixel_offsets', []),
        num_layers=model_params.get('num_layers', 1),
        output_activation=model_params.get('output_activation', 'none'),
    )
