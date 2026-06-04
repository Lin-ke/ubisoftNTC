import torch
import torch.nn.functional as F
import numpy as np


def build_mipmaps(tensor):
    mips = [tensor]
    h, w = tensor.shape[1], tensor.shape[2]
    while min(h, w) > 1:
        h, w = max(1, h // 2), max(1, w // 2)
        mip = F.interpolate(mips[-1].unsqueeze(0), size=(h, w), mode='area').squeeze(0)
        mips.append(mip)
    return mips


def sample_reference(mips, uv, scale, filter_mode='bicubic'):
    """Sample reference with given spatial filter at two closest mips, linearly mix.

    filter_mode: 'bicubic' (default, backward compat) or 'trilinear' (→ bilinear spatial)
    """
    spatial_mode = 'bilinear' if filter_mode == 'trilinear' else 'bicubic'

    num_mips = len(mips)
    s = torch.clamp(scale, 0, num_mips - 1)
    s0 = s.long()
    s1 = torch.clamp(s0 + 1, max=num_mips - 1)
    lam = (s - s0.float()).view(-1, 1, 1, 1)
    uv_grid = uv * 2 - 1

    results = []
    for b in range(uv.shape[0]):
        m0 = mips[s0[b].item()].unsqueeze(0)
        m1 = mips[s1[b].item()].unsqueeze(0)
        r0 = F.grid_sample(m0, uv_grid[b:b+1], mode=spatial_mode, padding_mode='border', align_corners=False)
        r1 = F.grid_sample(m1, uv_grid[b:b+1], mode=spatial_mode, padding_mode='border', align_corners=False)
        results.append((1 - lam[b]) * r0 + lam[b] * r1)
    return torch.cat(results, dim=0)


@torch.no_grad()
def evaluate_full(model, ref_mips, device, max_res=256):
    """Evaluate PSNR across all mip levels, processing in tiles for high res."""
    mse_per_mip = []
    model.eval()

    for mip_i, ref_mip in enumerate(ref_mips):
        ref_mip = ref_mip.to(device)
        h, w = ref_mip.shape[1], ref_mip.shape[2]

        if h <= max_res and w <= max_res:
            u = torch.linspace(0, 1, w, device=device)
            v = torch.linspace(0, 1, h, device=device)
            uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)
            scale = torch.tensor([float(mip_i)], device=device)
            pred = model(uv, scale)
            mse = F.mse_loss(pred, ref_mip.unsqueeze(0))
        else:
            mse_total = 0.0
            count = 0
            for ty in range(0, h, max_res):
                th = min(max_res, h - ty)
                for tx in range(0, w, max_res):
                    tw = min(max_res, w - tx)
                    u = torch.linspace(tx/w, (tx+tw)/w, tw, device=device)
                    v = torch.linspace(ty/h, (ty+th)/h, th, device=device)
                    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)
                    scale = torch.tensor([float(mip_i)], device=device)
                    pred = model(uv, scale)
                    ref_tile = ref_mip[:, ty:ty+th, tx:tx+tw].unsqueeze(0)
                    mse_total += F.mse_loss(pred, ref_tile).item() * th * tw
                    count += th * tw
            mse = mse_total / count
        mse_per_mip.append(mse)
        model.train()

    avg_mse_val = sum(m.item() if hasattr(m, 'item') else m for m in mse_per_mip) / len(mse_per_mip)
    psnr = -10 * np.log10(max(avg_mse_val, 1e-10))
    return psnr, avg_mse_val
