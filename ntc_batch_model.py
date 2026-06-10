"""Batched (ensemble) NTC models.

把 M 个材质的同构模型堆叠到一个 leading 维度 M, 用一次大 kernel 替代 M 次小
kernel, 直接攻击 batch=1 导致的 GPU 低利用率 (3-5%).

关键约定 (与 plan 一致):
  * 所有材质共享同一架构 (feature_configs / hidden_dim / num_layers / bc_format),
    仅权重不同.
  * 每个 iteration 共享同一 **连续 LOD scale** (标量); UV tile 仍各自随机.
    因此 grid_sample 的输入分辨率在 batch 内一致, 可一次完成 batch=M 的采样.
  * 训练结束后通过 export_per_material_state_dicts() 拆回 M 个 **标准单材质
    state_dict**, 与 NeuralTextureModel / NeuralBCTextureModel 完全兼容, 供
    eval / inference / compare 零改动复用.

仅支持 encoding='pyramid'. hash_grid 走原有非 batched 路径.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ntc_bc_model import get_bc_format


def _lod_levels(scale, num_mips):
    """共享 LOD 标量 -> (s0, s1, lam): 相邻两 mip 层索引 + 线性混合系数."""
    s = torch.clamp(scale, 0, num_mips - 1)
    s0 = max(0, min(int(s.floor().item()), num_mips - 1))
    s1 = min(s0 + 1, num_mips - 1)
    lam = float(s.item()) - s0
    return s0, s1, lam


# ============================================================
# 批量 MLP (M 套独立线性层, 用 bmm 并行)
# ============================================================

class BatchedMLP(nn.Module):
    """M 套独立 MLP: num_layers 个 (Linear+ReLU) 隐藏层 + 1 个输出 Linear.

    权重布局对齐 nn.Linear: weight [M, out, in], bias [M, out].
    输出层后可选 sigmoid / tanh (output_activation), 与 pyramid 模型一致.
    """

    def __init__(self, M, total_input_dim, hidden_dim, output_dim,
                 num_layers=1, output_activation='none'):
        super().__init__()
        self.M = M
        self.num_layers = num_layers
        self.output_activation = output_activation

        weights = []
        biases = []
        in_dim = total_input_dim
        for _ in range(num_layers):
            weights.append(nn.Parameter(torch.empty(M, hidden_dim, in_dim)))
            biases.append(nn.Parameter(torch.empty(M, hidden_dim)))
            in_dim = hidden_dim
        # 输出层
        weights.append(nn.Parameter(torch.empty(M, output_dim, in_dim)))
        biases.append(nn.Parameter(torch.empty(M, output_dim)))

        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)
        self.reset_parameters()

    def reset_parameters(self):
        # 与 nn.Linear 默认初始化 (Kaiming uniform) 等价, 逐 slice 初始化.
        for w, b in zip(self.weights, self.biases):
            fan_in = w.shape[2]
            bound = 1.0 / math.sqrt(fan_in)
            with torch.no_grad():
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                b.uniform_(-bound, bound)

    def forward(self, x):
        # x: [M, N, total_input_dim]  (N = H*W)
        L = self.num_layers
        for i in range(L):
            w = self.weights[i]          # [M, hidden, in]
            b = self.biases[i]           # [M, hidden]
            x = torch.baddbmm(b.unsqueeze(1), x, w.transpose(1, 2))
            x = F.relu(x)
        w = self.weights[L]
        b = self.biases[L]
        x = torch.baddbmm(b.unsqueeze(1), x, w.transpose(1, 2))
        if self.output_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.output_activation == 'tanh':
            x = torch.tanh(x)
        return x                          # [M, N, output_dim]

    # ---- 与标准 nn.Sequential MLP 的 state_dict 互转 ----

    def linear_mlp_indices(self):
        """返回每个 Linear 在标准 nn.Sequential 中的索引.

        标准结构: [Lin, ReLU, Lin, ReLU, ..., Lin, (out_act?)]
        Linear 位于 0, 2, ..., 2*num_layers.
        """
        return [2 * i for i in range(self.num_layers + 1)]

    def load_from_standard_mlp(self, std_mlp):
        """从标准 nn.Sequential MLP 复制权重到所有 M slice (广播)."""
        idxs = self.linear_mlp_indices()
        with torch.no_grad():
            for k, idx in enumerate(idxs):
                lin = std_mlp[idx]
                self.weights[k].copy_(lin.weight.unsqueeze(0).expand_as(self.weights[k]))
                self.biases[k].copy_(lin.bias.unsqueeze(0).expand_as(self.biases[k]))

    def copy_from_batched(self, other):
        """从另一个 BatchedMLP 复制全部权重 (BC 初始化用)."""
        with torch.no_grad():
            for w_dst, w_src in zip(self.weights, other.weights):
                w_dst.copy_(w_src)
            for b_dst, b_src in zip(self.biases, other.biases):
                b_dst.copy_(b_src)

    def export_state_dict(self, m):
        """导出第 m 个材质的标准 mlp.* state_dict 片段."""
        idxs = self.linear_mlp_indices()
        sd = {}
        for k, idx in enumerate(idxs):
            sd[f'mlp.{idx}.weight'] = self.weights[k][m].detach().clone()
            sd[f'mlp.{idx}.bias'] = self.biases[k][m].detach().clone()
        return sd


# ============================================================
# 批量 UC 特征网格 / 模型
# ============================================================

class BatchedMipmapFeatureGrid(nn.Module):
    def __init__(self, M, base_resolution, num_mips, feature_dim, filter_mode='trilinear'):
        super().__init__()
        self.M = M
        self.base_resolution = base_resolution
        self.num_mips = num_mips
        self.feature_dim = feature_dim
        self.filter_mode = filter_mode
        self.mips = nn.ParameterList([
            nn.Parameter(torch.randn(M, feature_dim,
                                     base_resolution >> i, base_resolution >> i) * 0.01)
            for i in range(num_mips)
        ])

    def _levels(self, scale):
        return _lod_levels(scale, self.num_mips)

    def sample(self, uv, scale):
        # uv: [M, H, W, 2] in [0,1]; scale: 共享标量 tensor
        s0, s1, lam = self._levels(scale)
        spatial_mode = 'bilinear' if self.filter_mode == 'trilinear' else 'bicubic'
        uv_grid = uv * 2 - 1
        m0 = self.mips[s0]
        f0 = F.grid_sample(m0, uv_grid, mode=spatial_mode,
                           padding_mode='border', align_corners=False)
        if s1 == s0 or lam == 0.0:
            return f0
        m1 = self.mips[s1]
        f1 = F.grid_sample(m1, uv_grid, mode=spatial_mode,
                           padding_mode='border', align_corners=False)
        return (1 - lam) * f0 + lam * f1


class BatchedNeuralTextureModel(nn.Module):
    """M 个同构 UC 材质模型的 batched 版本."""

    def __init__(self, M, feature_configs, hidden_dim, output_dim, num_layers=1,
                 filter_mode='trilinear', output_activation='none'):
        super().__init__()
        self.M = M
        self.feature_configs = [tuple(fc) for fc in feature_configs]
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.filter_mode = filter_mode
        self.output_activation = output_activation

        self.feature_grids = nn.ModuleList([
            BatchedMipmapFeatureGrid(M, res, mips, dim, filter_mode)
            for res, mips, dim in self.feature_configs
        ])
        total_input_dim = sum(dim for _, _, dim in self.feature_configs)
        self.total_input_dim = total_input_dim
        self.mlp = BatchedMLP(M, total_input_dim, hidden_dim, output_dim,
                              num_layers=num_layers, output_activation=output_activation)

    def forward(self, uv, scale):
        # uv: [M, H, W, 2]; scale: 共享标量
        M, H, W, _ = uv.shape
        features = [grid.sample(uv, scale) for grid in self.feature_grids]
        x = torch.cat(features, dim=1)           # [M, total_in, H, W]
        x = x.permute(0, 2, 3, 1).reshape(M, H * W, self.total_input_dim)
        y = self.mlp(x)                          # [M, H*W, out]
        y = y.reshape(M, H, W, self.output_dim).permute(0, 3, 1, 2)
        return y                                 # [M, out, H, W]

    def export_state_dict(self, m):
        sd = {}
        for g, grid in enumerate(self.feature_grids):
            for i, mip in enumerate(grid.mips):
                sd[f'feature_grids.{g}.mips.{i}'] = mip[m].detach().unsqueeze(0).clone()
        sd.update(self.mlp.export_state_dict(m))
        return sd

    def export_per_material_state_dicts(self):
        return [self.export_state_dict(m) for m in range(self.M)]


# ============================================================
# 批量 BC 块特征 / 模型
# ============================================================

class BatchedBCBlockFeature(nn.Module):
    """batched 4x4 块压缩特征层 (单线段, STE 量化), leading 维度 M."""

    def __init__(self, M, resolution_w, resolution_h, feature_dim, bc_format):
        super().__init__()
        assert resolution_w % 4 == 0 and resolution_h % 4 == 0
        self.M = M
        self.resolution_w = resolution_w
        self.resolution_h = resolution_h
        self.feature_dim = feature_dim
        self.blocks_w = resolution_w // 4
        self.blocks_h = resolution_h // 4
        self.bc_format = bc_format

        num_eps = bc_format.get_endpoint_count()
        init_scale = bc_format.get_endpoint_init_scale(feature_dim)
        self.endpoints = nn.Parameter(
            torch.randn(M, self.blocks_h, self.blocks_w, num_eps, feature_dim) * init_scale
        )
        self.raw_indices = nn.Parameter(
            torch.zeros(M, self.blocks_h, self.blocks_w, 16)
        )

    def _quantize_ste(self, x, levels_minus_one):
        levels = float(levels_minus_one)
        xq = (x * levels).round().clamp(0, levels) / levels
        if not self.training:
            return xq
        return x + (xq - x).detach()

    def forward(self):
        M = self.M
        bh, bw = self.blocks_h, self.blocks_w
        C = self.feature_dim

        alpha = torch.sigmoid(self.raw_indices)              # [M,bh,bw,16]
        idx_bits = self.bc_format.get_index_bits()
        alpha_q = self._quantize_ste(alpha, (1 << idx_bits) - 1)

        eps = torch.sigmoid(self.endpoints)                  # [M,bh,bw,2,C]
        eps_bits = self.bc_format.get_endpoint_bits(C)
        eps_q = eps.clone()
        for ch in range(C):
            b = eps_bits[ch]
            eps_q[..., ch] = self._quantize_ste(eps[..., ch], (1 << b) - 1)

        e0 = eps_q[:, :, :, 0, :]                            # [M,bh,bw,C]
        e1 = eps_q[:, :, :, 1, :]
        x = alpha_q.unsqueeze(-1)                            # [M,bh,bw,16,1]
        y = e0.unsqueeze(3) + x * (e1.unsqueeze(3) - e0.unsqueeze(3))  # [M,bh,bw,16,C]

        y = y.reshape(M, bh, bw, 4, 4, C)
        y = y.permute(0, 1, 3, 2, 4, 5)
        y = y.reshape(M, self.resolution_h, self.resolution_w, C)
        return y.permute(0, 3, 1, 2)                         # [M,C,H,W]

    @torch.no_grad()
    def init_from_continuous_batch(self, x):
        """从 batched UC 连续特征 x [M,C,H,W] 初始化端点/索引 (bbox 启发式)."""
        M, C, H, W = x.shape
        bh, bw = self.blocks_h, self.blocks_w
        assert H == 4 * bh and W == 4 * bw and C == self.feature_dim

        x = x.to(self.endpoints.device)
        # [M,C,H,W] -> [M,bh,bw,16,C]
        x = x.reshape(M, C, bh, 4, bw, 4).permute(0, 2, 4, 3, 5, 1).contiguous()
        blocks = x.reshape(M, bh, bw, 16, C)

        e0 = blocks.min(dim=3).values                        # [M,bh,bw,C]
        e1 = blocks.max(dim=3).values

        direction = (e1 - e0).unsqueeze(3)                   # [M,bh,bw,1,C]
        centered = blocks - e0.unsqueeze(3)                  # [M,bh,bw,16,C]
        dot = (centered * direction).sum(dim=-1)             # [M,bh,bw,16]
        norm_sq = (direction * direction).sum(dim=-1).clamp(min=1e-8)  # [M,bh,bw,1]
        alpha = (dot / norm_sq).clamp(0.0, 1.0)

        eps = 1e-3
        e0_c = e0.clamp(eps, 1 - eps)
        e1_c = e1.clamp(eps, 1 - eps)
        a_c = alpha.clamp(eps, 1 - eps)
        eps_endpoints = torch.stack([
            torch.log(e0_c / (1 - e0_c)),
            torch.log(e1_c / (1 - e1_c)),
        ], dim=3)                                            # [M,bh,bw,2,C]
        raw_indices = torch.log(a_c / (1 - a_c))             # [M,bh,bw,16]

        self.endpoints.data.copy_(eps_endpoints)
        self.raw_indices.data.copy_(raw_indices)

    def export_state_dict(self, prefix, m):
        return {
            f'{prefix}.endpoints': self.endpoints[m].detach().clone(),
            f'{prefix}.raw_indices': self.raw_indices[m].detach().clone(),
        }


class BatchedBCMipmapFeatureGrid(nn.Module):
    def __init__(self, M, base_resolution, num_mips, feature_dim, bc_format,
                 filter_mode='trilinear', half_pixel_offset=False):
        super().__init__()
        self.M = M
        self.base_resolution = base_resolution
        self.num_mips = num_mips
        self.feature_dim = feature_dim
        self.filter_mode = filter_mode
        self.half_pixel_offset = half_pixel_offset
        self.mips = nn.ModuleList([
            BatchedBCBlockFeature(M, base_resolution >> i, base_resolution >> i,
                                  feature_dim, bc_format)
            for i in range(num_mips)
        ])

    def _levels(self, scale):
        return _lod_levels(scale, self.num_mips)

    def sample(self, uv, scale):
        s0, s1, lam = self._levels(scale)
        uv_offset = uv
        if self.half_pixel_offset:
            res0 = self.base_resolution >> s0
            uv_offset = uv + 0.5 / res0
        uv_grid = uv_offset * 2 - 1
        spatial_mode = 'bilinear' if self.filter_mode == 'trilinear' else 'bicubic'

        f0 = F.grid_sample(self.mips[s0](), uv_grid, mode=spatial_mode,
                           padding_mode='border', align_corners=False)
        if s1 == s0 or lam == 0.0:
            return f0
        f1 = F.grid_sample(self.mips[s1](), uv_grid, mode=spatial_mode,
                           padding_mode='border', align_corners=False)
        return (1 - lam) * f0 + lam * f1


class BatchedNeuralBCTextureModel(nn.Module):
    def __init__(self, M, feature_configs, hidden_dim, output_dim, bc_format,
                 filter_mode='trilinear', half_pixel_offsets=None, num_layers=1,
                 output_activation='none'):
        super().__init__()
        if half_pixel_offsets is None:
            half_pixel_offsets = []
        self.M = M
        self.feature_configs = [tuple(fc) for fc in feature_configs]
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.output_activation = output_activation

        self.feature_grids = nn.ModuleList([
            BatchedBCMipmapFeatureGrid(M, res, mips, dim, bc_format, filter_mode,
                                       half_pixel_offset=(i in half_pixel_offsets))
            for i, (res, mips, dim) in enumerate(self.feature_configs)
        ])
        total_input_dim = sum(dim for _, _, dim in self.feature_configs)
        self.total_input_dim = total_input_dim
        self.mlp = BatchedMLP(M, total_input_dim, hidden_dim, output_dim,
                              num_layers=num_layers, output_activation=output_activation)

    def forward(self, uv, scale):
        M, H, W, _ = uv.shape
        features = [grid.sample(uv, scale) for grid in self.feature_grids]
        x = torch.cat(features, dim=1)
        x = x.permute(0, 2, 3, 1).reshape(M, H * W, self.total_input_dim)
        y = self.mlp(x)
        return y.reshape(M, H, W, self.output_dim).permute(0, 3, 1, 2)

    @torch.no_grad()
    def init_from_uc_batched(self, uc_model):
        """从 batched UC 模型初始化: 复制 MLP + 用 UC 特征 bbox 压缩初始化端点/索引."""
        self.mlp.copy_from_batched(uc_model.mlp)
        for bc_grid, uc_grid in zip(self.feature_grids, uc_model.feature_grids):
            for j, bc_mip in enumerate(bc_grid.mips):
                if j >= len(uc_grid.mips):
                    break
                uc_mip = uc_grid.mips[j]   # [M,C,res,res]
                if (uc_mip.shape[1] != bc_mip.feature_dim
                        or uc_mip.shape[2] != bc_mip.resolution_h
                        or uc_mip.shape[3] != bc_mip.resolution_w):
                    continue
                bc_mip.init_from_continuous_batch(uc_mip.detach())

    def export_state_dict(self, m):
        sd = {}
        for g, grid in enumerate(self.feature_grids):
            for i, mip in enumerate(grid.mips):
                sd.update(mip.export_state_dict(f'feature_grids.{g}.mips.{i}', m))
        sd.update(self.mlp.export_state_dict(m))
        return sd

    def export_per_material_state_dicts(self):
        return [self.export_state_dict(m) for m in range(self.M)]


# ============================================================
# 工厂
# ============================================================

def make_batched_uc_model(M, model_params, output_dim=9):
    if model_params.get('encoding', 'pyramid') != 'pyramid':
        raise ValueError("batched 训练仅支持 encoding='pyramid'")
    return BatchedNeuralTextureModel(
        M,
        feature_configs=model_params['feature_configs'],
        hidden_dim=model_params['hidden_dim'],
        output_dim=output_dim,
        num_layers=model_params.get('num_layers', 1),
        filter_mode=model_params.get('filter', 'trilinear'),
        output_activation=model_params.get('output_activation', 'none'),
    )


def make_batched_bc_model(M, model_params, output_dim=9, bc_format_name='bc1'):
    if model_params.get('encoding', 'pyramid') != 'pyramid':
        raise ValueError("batched 训练仅支持 encoding='pyramid'")
    bc_format = get_bc_format(bc_format_name)
    return BatchedNeuralBCTextureModel(
        M,
        feature_configs=model_params['feature_configs'],
        hidden_dim=model_params['hidden_dim'],
        output_dim=output_dim,
        bc_format=bc_format,
        filter_mode=model_params.get('filter', 'trilinear'),
        half_pixel_offsets=model_params.get('half_pixel_offsets', []),
        num_layers=model_params.get('num_layers', 1),
        output_activation=model_params.get('output_activation', 'none'),
    )
