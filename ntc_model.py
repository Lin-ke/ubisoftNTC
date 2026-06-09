import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HardSwish(nn.Module):
    def forward(self, x):
        # shader: x * saturate((x + 3) / 6)
        return x * torch.clamp((x + 3.0) / 6.0, 0.0, 1.0)


_PRIMES = [1, 2654435761, 805459861, 3674653429,
           2097192037, 1434869437, 2165219737]


@torch.no_grad()
def _fast_hash(ind: torch.Tensor, primes: torch.Tensor, hashmap_size: int):
    """Hash from tiny-cuda-nn/encodings/grid.h. ind: [..., d] int64."""
    d = ind.shape[-1]
    ind = (ind * primes[:d]) & 0xffffffff
    out = ind[..., 0].clone()
    for i in range(1, d):
        out = out ^ ind[..., i]
    return out % hashmap_size


class _HashGridLevel(nn.Module):
    """Single multilinearly-interpolated hash grid level."""

    def __init__(self, dim: int, n_features: int,
                 hashmap_size: int, resolution: int):
        super().__init__()
        self.dim = dim
        self.n_features = n_features
        self.hashmap_size = hashmap_size
        self.resolution = resolution

        if dim > len(_PRIMES):
            raise ValueError(f"HashGrid supports <= {len(_PRIMES)}-D")

        self.embedding = nn.Embedding(hashmap_size, n_features)
        nn.init.uniform_(self.embedding.weight, a=-1e-4, b=1e-4)

        primes = torch.tensor(_PRIMES, dtype=torch.int64)
        self.register_buffer('primes', primes, persistent=False)

        n_neigs = 1 << dim
        neigs = torch.arange(n_neigs, dtype=torch.int64).reshape(-1, 1)
        dims = torch.arange(dim, dtype=torch.int64).reshape(1, -1)
        bin_mask = (neigs & (1 << dims)) == 0
        self.register_buffer('bin_mask', bin_mask, persistent=False)

    def forward(self, x: torch.Tensor):
        # x: [B..., dim], float in [0, 1]
        bdims = len(x.shape[:-1])
        x = torch.clamp(x, 0.0, 1.0) * self.resolution
        xi = x.long()
        xf = x - xi.float().detach()
        xi = xi.unsqueeze(-2)                                 # [B..., 1, dim]
        xf = xf.unsqueeze(-2)
        bin_mask = self.bin_mask.reshape((1,) * bdims + self.bin_mask.shape)
        inds = torch.where(bin_mask, xi, xi + 1)              # [B..., neig, dim]
        ws = torch.where(bin_mask, 1 - xf, xf)
        w = ws.prod(dim=-1, keepdim=True)                     # [B..., neig, 1]
        hash_ids = _fast_hash(inds, self.primes, self.hashmap_size)
        neig_data = self.embedding(hash_ids)                  # [B..., neig, F]
        return (neig_data * w).sum(dim=-2)                    # [B..., F]


class MultiResHashGrid(nn.Module):
    """Instant-NGP style pure-PyTorch multi-resolution hash grid."""

    def __init__(self, dim: int = 2,
                 n_levels: int = 7,
                 n_features_per_level: int = 8,
                 log2_hashmap_size: int = 15,
                 base_resolution: int = 4,
                 finest_resolution: int = 256):
        super().__init__()
        self.dim = dim
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution

        b = math.exp((math.log(finest_resolution) -
                      math.log(base_resolution)) / max(n_levels - 1, 1))

        levels = []
        for l in range(n_levels):
            resolution = math.floor(base_resolution * (b ** l))
            hashmap_size = min(resolution ** dim, 2 ** log2_hashmap_size)
            levels.append(_HashGridLevel(
                dim=dim, n_features=n_features_per_level,
                hashmap_size=hashmap_size, resolution=resolution,
            ))
        self.levels = nn.ModuleList(levels)
        self.output_dim = n_levels * n_features_per_level

    def forward(self, x: torch.Tensor):
        return torch.cat([lvl(x) for lvl in self.levels], dim=-1)

    def iter_param_tensors(self):
        """Iterate over per-level embedding weights."""
        for lvl in self.levels:
            yield lvl.embedding.weight


def _make_activation(name):
    name = (name or 'relu').lower()
    if name == 'relu':
        return nn.ReLU()
    if name == 'leaky_relu':
        return nn.LeakyReLU()
    if name == 'hard_swish':
        return HardSwish()
    raise ValueError(f"Unsupported activation: {name}")


def _make_output_activation(name):
    name = (name or 'none').lower()
    if name == 'none':
        return None
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'tanh':
        return nn.Tanh()
    if name == 'hard_swish':
        return HardSwish()
    raise ValueError(f"Unsupported output activation: {name}")


class MipmapFeatureGrid(nn.Module):
    def __init__(self, base_resolution, num_mips, feature_dim, filter_mode='trilinear'):
        super().__init__()
        self.base_resolution = base_resolution
        self.num_mips = num_mips
        self.feature_dim = feature_dim
        self.filter_mode = filter_mode

        mips = []
        for i in range(num_mips):
            res = base_resolution >> i
            mips.append(nn.Parameter(torch.randn(1, feature_dim, res, res) * 0.01))
        self.mips = nn.ParameterList(mips)

    def sample(self, uv, scale):
        # uv: [B, H, W, 2] in [0, 1]
        # scale: [B] or scalar, continuous mip level
        B = uv.shape[0]
        s = torch.clamp(scale, 0, self.num_mips - 1)
        s0 = s.long()
        s1 = torch.clamp(s0 + 1, max=self.num_mips - 1)
        lam = (s - s0.float()).view(-1, 1, 1, 1)

        uv_grid = uv * 2 - 1

        # 空间插值模式: trilinear → bilinear, tricubic → bicubic
        spatial_mode = 'bilinear' if self.filter_mode == 'trilinear' else 'bicubic'

        feat0, feat1 = [], []
        for b in range(B):
            m0 = self.mips[int(s0[b].item())]
            m1 = self.mips[int(s1[b].item())]
            f0 = F.grid_sample(m0, uv_grid[b:b+1], mode=spatial_mode,
                               padding_mode='border', align_corners=False)
            f1 = F.grid_sample(m1, uv_grid[b:b+1], mode=spatial_mode,
                               padding_mode='border', align_corners=False)
            feat0.append(f0)
            feat1.append(f1)

        feat0 = torch.cat(feat0, dim=0)
        feat1 = torch.cat(feat1, dim=0)
        return (1 - lam) * feat0 + lam * feat1


class NeuralTextureModel(nn.Module):
    def __init__(self, feature_configs, hidden_dim, output_dim, num_layers=1,
                 filter_mode='trilinear', output_activation='none'):
        super().__init__()
        self.feature_grids = nn.ModuleList([
            MipmapFeatureGrid(res, mips, dim, filter_mode)
            for res, mips, dim in feature_configs
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
        # uv: [B, H, W, 2] in [0, 1]
        # scales: [B]
        features = []
        for grid in self.feature_grids:
            feat = grid.sample(uv, scales)
            features.append(feat)
        x = torch.cat(features, dim=1)  # [B, total_input_dim, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, total_input_dim]
        y = self.mlp(x)  # [B, H, W, output_dim]
        y = y.permute(0, 3, 1, 2)  # [B, output_dim, H, W]
        return y


class HashGridTextureModel(nn.Module):
    """Current-project wrapper around pure PyTorch multi-resolution hash grid.

    Keeps the same public surface used by training helpers:
    ``feature_grids``, ``mlp`` and ``forward(uv, scales)``.
    ``scales`` is accepted for interface compatibility; this model samples all
    hash-grid levels and lets the MLP decode the concatenated encoding.
    """

    def __init__(self, hash_grid_config, hidden_dim, output_dim, num_layers=1,
                 activation='leaky_relu', output_activation='hard_swish'):
        super().__init__()
        cfg = dict(hash_grid_config or {})
        self.hash_grid = MultiResHashGrid(
            dim=cfg.get('dim', 2),
            n_levels=cfg.get('n_levels', 7),
            n_features_per_level=cfg.get('n_features_per_level', 8),
            log2_hashmap_size=cfg.get('log2_hashmap_size', 15),
            base_resolution=cfg.get('base_resolution', 4),
            finest_resolution=cfg.get('finest_resolution', 256),
        )
        self.feature_grids = nn.ModuleList([self.hash_grid])
        total_input_dim = self.hash_grid.output_dim

        layers = []
        in_dim = total_input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_make_activation(activation))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        out_act = _make_output_activation(output_activation)
        if out_act is not None:
            layers.append(out_act)
        self.mlp = nn.Sequential(*layers)

        self.total_input_dim = total_input_dim
        self.output_dim = output_dim

    def forward(self, uv, scales):
        # uv: [B, H, W, 2] in [0, 1]; scales is intentionally unused.
        B, H, W, _ = uv.shape
        x = self.hash_grid(uv.reshape(-1, 2))       # [B*H*W, total_input_dim]
        y = self.mlp(x)                             # [B*H*W, output_dim]
        return y.reshape(B, H, W, self.output_dim).permute(0, 3, 1, 2)


def make_model(model_params, output_dim=9):
    """从配置创建 NeuralTextureModel 或 HashGridTextureModel.

    Args:
        model_params: dict, 由 get_model_params(config) 产生.
            pyramid 包含: feature_configs, hidden_dim, num_layers, filter, output_activation
            hash_grid 包含: encoding='hash_grid', hash_grid, hidden_dim, num_layers
        output_dim: MLP 输出维度
    """
    encoding = model_params.get('encoding', 'pyramid')
    if encoding == 'mipmap':
        encoding = 'pyramid'
    if encoding == 'hash_grid':
        return HashGridTextureModel(
            hash_grid_config=model_params.get('hash_grid', {}),
            hidden_dim=model_params['hidden_dim'],
            output_dim=output_dim,
            num_layers=model_params.get('num_layers', 1),
            activation=model_params.get('activation', 'leaky_relu'),
            output_activation=model_params.get('output_activation', 'hard_swish'),
        )
    if encoding == "pyramid":
        return NeuralTextureModel(
            feature_configs=model_params['feature_configs'],
            hidden_dim=model_params['hidden_dim'],
            output_dim=output_dim,
            num_layers=model_params.get('num_layers', 1),
            filter_mode=model_params.get('filter', 'trilinear'),
            output_activation=model_params.get('output_activation', 'none'),
        )
    raise ValueError(f"Unsupported model encoding: {encoding}")
