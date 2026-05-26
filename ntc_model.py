import torch
import torch.nn as nn
import torch.nn.functional as F


class MipmapFeatureGrid(nn.Module):
    def __init__(self, base_resolution, num_mips, feature_dim):
        super().__init__()
        self.base_resolution = base_resolution
        self.num_mips = num_mips
        self.feature_dim = feature_dim

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

        feat0, feat1 = [], []
        for b in range(B):
            m0 = self.mips[int(s0[b].item())]
            m1 = self.mips[int(s1[b].item())]
            f0 = F.grid_sample(m0, uv_grid[b:b+1], mode='bilinear',
                               padding_mode='border', align_corners=False)
            f1 = F.grid_sample(m1, uv_grid[b:b+1], mode='bilinear',
                               padding_mode='border', align_corners=False)
            feat0.append(f0)
            feat1.append(f1)

        feat0 = torch.cat(feat0, dim=0)
        feat1 = torch.cat(feat1, dim=0)
        return (1 - lam) * feat0 + lam * feat1


class NeuralTextureModel(nn.Module):
    def __init__(self, feature_configs, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.feature_grids = nn.ModuleList([
            MipmapFeatureGrid(res, mips, dim)
            for res, mips, dim in feature_configs
        ])
        total_input_dim = sum(dim for _, _, dim in feature_configs)

        layers = []
        in_dim = total_input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
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


def make_model(reference_resolution=1024, output_dim=9):
    feature_configs = [
        (512, 7, 3),
        (256, 6, 3),
        (128, 5, 3),
        (64, 4, 3),
    ]
    return NeuralTextureModel(
        feature_configs=feature_configs,
        hidden_dim=16,
        output_dim=output_dim,
        num_layers=2,
    )
