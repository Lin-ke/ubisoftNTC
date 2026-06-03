import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import time

from ntc_model import make_model


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


def load_brick_material(base_dir, target_res=1024):
    col = np.array(Image.open(os.path.join(base_dir, 'StoneBricksSplitface001_COL_1K.jpg'))
                   .convert('RGB'), dtype=np.float32) / 255.0
    nrm = np.array(Image.open(os.path.join(base_dir, 'StoneBricksSplitface001_NRM_1K.jpg'))
                   .convert('RGB'), dtype=np.float32) / 255.0
    ao = np.array(Image.open(os.path.join(base_dir, 'StoneBricksSplitface001_AO_1K.jpg'))
                  .convert('L'), dtype=np.float32) / 255.0
    gloss = np.array(Image.open(os.path.join(base_dir, 'StoneBricksSplitface001_GLOSS_1K.jpg'))
                     .convert('L'), dtype=np.float32) / 255.0

    nrm = nrm * 2.0 - 1.0
    roughness = 1.0 - gloss
    metalness = np.zeros_like(ao)

    reference = np.concatenate([
        col,
        nrm,
        ao[..., np.newaxis],
        roughness[..., np.newaxis],
        metalness[..., np.newaxis],
    ], axis=-1)

    ref_tensor = torch.from_numpy(reference).permute(2, 0, 1).float()
    if target_res != ref_tensor.shape[1]:
        ref_tensor = F.interpolate(ref_tensor.unsqueeze(0), size=(target_res, target_res),
                                   mode='bicubic', align_corners=False).squeeze(0)
    return ref_tensor


def train(model, reference_mips, total_iterations=50000, warmup_iterations=5000,
          batch_res=384, device='cuda', log_interval=1000, output_dir='output'):
    model = model.to(device)
    reference_mips = [m.to(device) for m in reference_mips]
    num_mips_ref = len(reference_mips)
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: warmup (unconstrained features)
    print(f"\n{'='*50}")
    print(f"Phase 1: Warmup ({warmup_iterations} iterations)")
    print(f"  LR: feat=5e-2, mlp=1e-3, gamma=0.9995")
    print(f"{'='*50}")

    feature_params = list(model.feature_grids.parameters())
    mlp_params = list(model.mlp.parameters())
    optimizer = torch.optim.Adam([
        {'params': feature_params, 'lr': 5e-2},
        {'params': mlp_params, 'lr': 1e-3},
    ])

    gamma = 0.9995 ** (1.0 / 50)  # 0.9995 per iter means decay over 50 iters
    best_psnr = 0.0

    for iteration in range(warmup_iterations):
        model.train()
        B = 1
        H = W = batch_res

        # Random continuous patch + scale
        u0 = torch.rand(1, device=device) * (1 - W / reference_mips[0].shape[1])
        v0 = torch.rand(1, device=device) * (1 - H / reference_mips[0].shape[2])
        u_vals = torch.linspace(u0.item(), u0.item() + W / reference_mips[0].shape[1], W, device=device)
        v_vals = torch.linspace(v0.item(), v0.item() + H / reference_mips[0].shape[2], H, device=device)
        ug, vg = torch.meshgrid(u_vals, v_vals, indexing='xy')
        uv = torch.stack([ug, vg], dim=-1).unsqueeze(0)

        scale = torch.rand(B, device=device) * (num_mips_ref - 1)

        with torch.no_grad():
            ref = sample_reference(reference_mips, uv, scale)

        pred = model(uv, scale)
        loss = F.mse_loss(pred, ref)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for pg in optimizer.param_groups:
            pg['lr'] *= gamma

        if (iteration + 1) % log_interval == 0:
            with torch.no_grad():
                model.eval()
                psnr, _ = evaluate_full(model, reference_mips, device)
                print(f"  Warmup Iter {iteration+1:>6d}/{warmup_iterations} | "
                      f"Loss: {loss.item():.6f} | PSNR: {psnr:.2f} dB | "
                      f"LR_feat: {optimizer.param_groups[0]['lr']:.6f}")

    # Phase 2: main training
    main_iterations = total_iterations - warmup_iterations
    print(f"\n{'='*50}")
    print(f"Phase 2: Main training ({main_iterations} iterations)")
    print(f"  LR: feat=1e-2, mlp=1e-3, gamma=0.9999")
    print(f"{'='*50}")

    optimizer = torch.optim.Adam([
        {'params': feature_params, 'lr': 1e-2},
        {'params': mlp_params, 'lr': 1e-3},
    ])
    gamma = 0.9999 ** (1.0 / 50)

    for iteration in range(warmup_iterations, total_iterations):
        model.train()
        B = 1
        H = W = batch_res

        u0 = torch.rand(1, device=device) * (1 - W / reference_mips[0].shape[1])
        v0 = torch.rand(1, device=device) * (1 - H / reference_mips[0].shape[2])
        u_vals = torch.linspace(u0.item(), u0.item() + W / reference_mips[0].shape[1], W, device=device)
        v_vals = torch.linspace(v0.item(), v0.item() + H / reference_mips[0].shape[2], H, device=device)
        ug, vg = torch.meshgrid(u_vals, v_vals, indexing='xy')
        uv = torch.stack([ug, vg], dim=-1).unsqueeze(0)

        scale = torch.rand(B, device=device) * (num_mips_ref - 1)

        with torch.no_grad():
            ref = sample_reference(reference_mips, uv, scale)

        pred = model(uv, scale)
        loss = F.mse_loss(pred, ref)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for pg in optimizer.param_groups:
            pg['lr'] *= gamma

        if (iteration + 1) % log_interval == 0:
            with torch.no_grad():
                model.eval()
                psnr, _ = evaluate_full(model, reference_mips, device)
                print(f"  Main   Iter {iteration+1:>6d}/{total_iterations} | "
                      f"Loss: {loss.item():.6f} | PSNR: {psnr:.2f} dB | "
                      f"LR_feat: {optimizer.param_groups[0]['lr']:.6f}")

                if psnr > best_psnr:
                    best_psnr = psnr
                    torch.save({
                        'iteration': iteration + 1,
                        'model_state_dict': model.state_dict(),
                        'psnr': best_psnr,
                    }, os.path.join(output_dir, 'best_model.pth'))

        if (iteration + 1) % 10000 == 0:
            torch.save({
                'iteration': iteration + 1,
                'model_state_dict': model.state_dict(),
            }, os.path.join(output_dir, f'checkpoint_{iteration+1}.pth'))

    torch.save({'iteration': total_iterations, 'model_state_dict': model.state_dict()},
               os.path.join(output_dir, 'final_model.pth'))
    return best_psnr


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    torch.manual_seed(42)

    ref_tensor = load_brick_material('.', target_res=1024)
    print(f"Reference shape: {ref_tensor.shape}")

    reference_mips = build_mipmaps(ref_tensor.to('cpu'))
    print(f"Mip levels: {[m.shape for m in reference_mips]}")

    model = make_model(reference_resolution=1024, output_dim=ref_tensor.shape[0])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    start_time = time.time()
    best_psnr = train(
        model, reference_mips,
        total_iterations=20000,
        warmup_iterations=5000,
        batch_res=384,
        device=device,
        log_interval=1000,
        output_dir='output',
    )
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes. Best PSNR: {best_psnr:.2f} dB")
