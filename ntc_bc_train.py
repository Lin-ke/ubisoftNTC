"""
BC 神经材质端到端量化感知训练 (QAT)
=====================================
单阶段训练流程：从初始化起即执行 sigmoid + quant (STE), 模拟 BC 解压,
直接优化量化后的特征参数和 MLP。

论文对齐: NBTC-CooperativeVectors (Belcour & Benyoub, 2025)
  - 单阶段 QAT (消融: 预热无增益)
  - L1 loss (L2 可选)
  - Adam 常数 LR (无 gamma decay)
  - Vaidyanathan 风格的指数 LOD 采样
  - 半像素偏移 (配置化)
  - BC1: 2-bit 索引, RGB565 端点

支持全部 6 种 BC 格式 (bc1~bc6).
"""

import torch
import torch.nn.functional as F
import os
import sys
import time
import math

from ntc_bc_model import make_bc_model, get_bc_format
from ntc_config import load_config, get_training_params, get_model_params
from ntc_train import build_mipmaps, sample_reference, evaluate_full, load_brick_material


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
        # Vaidyanathan 2023: LOD = floor(-log4(X)) = floor(-log(X) / log(4))
        lod = (-torch.log(X) / math.log(4)).floor().clamp(0, num_mips - 1)
    lod = lod + torch.rand(1, device=device) * 0.999
    return lod.clamp(0, num_mips - 1)


def train_bc(reference_mips, config_or_params, device='cuda'):
    """BC 神经材质单阶段 QAT 训练.

    Args:
        reference_mips: list of [C, H_i, W_i], 参考 mipmap 金字塔
        config_or_params: 配置字典或参数字典
        device: 计算设备

    Returns:
        best_psnr: 训练过程中的最佳 PSNR (dB)
    """
    # 解析参数
    if isinstance(config_or_params, dict):
        params = config_or_params
        if 'model' in params:
            # 从完整 config 加载
            model_params = get_model_params(params)
            params = get_training_params(params)
        else:
            # 纯参数字典
            bc_format_name = params.get('bc_format', 'bc6')
            model_params = params.get('model_params', {})
    else:
        raise TypeError("config_or_params must be a dict")

    bc_format_name = params.get('bc_format', 'bc6')
    filter_mode = model_params.get('filter', 'trilinear')
    output_dir = params.get('output_dir', 'output_bc')

    total_iterations = params['total_iterations']
    batch_res = params['batch_res']
    log_interval = params.get('log_interval', 1000)

    lr_feat = params.get('lr_feat', 1e-2)
    lr_mlp = params.get('lr_mlp', 1e-3)
    betas = params.get('betas', [0.9, 0.999])
    loss_fn = params.get('loss_fn', 'l1')

    os.makedirs(output_dir, exist_ok=True)
    reference_mips = [m.to(device) for m in reference_mips]
    num_mips_ref = len(reference_mips)

    output_dim = reference_mips[0].shape[0]

    # 构建模型
    bc_model = make_bc_model(model_params, output_dim=output_dim,
                             bc_format_name=bc_format_name).to(device)

    total_params = sum(p.numel() for p in bc_model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 收集特征参数 + MLP 参数
    feat_params = []
    for grid in bc_model.feature_grids:
        for mip in grid.mips:
            feat_params.append(mip.endpoints)
            feat_params.append(mip.raw_indices)

    optimizer = torch.optim.Adam([
        {'params': feat_params, 'lr': lr_feat},
        {'params': list(bc_model.mlp.parameters()), 'lr': lr_mlp},
    ], betas=betas)

    best_psnr = 0.0
    best_mse = float('inf')

    print(f"\n{'='*60}")
    print(f"Single-phase QAT: {bc_format_name.upper()} ({total_iterations} iters)")
    print(f"  Loss: {loss_fn.upper()}")
    print(f"  LR: feat={lr_feat}, mlp={lr_mlp}, betas={betas}")
    print(f"  Filter: {filter_mode}")
    print(f"{'='*60}")

    for iteration in range(total_iterations):
        bc_model.train()

        H = W = batch_res
        ref_w = reference_mips[0].shape[1]
        ref_h = reference_mips[0].shape[2]

        u0 = torch.rand(1, device=device) * (1.0 - W / ref_w)
        v0 = torch.rand(1, device=device) * (1.0 - H / ref_h)
        u_vals = torch.linspace(u0.item(), u0.item() + W / ref_w, W, device=device)
        v_vals = torch.linspace(v0.item(), v0.item() + H / ref_h, H, device=device)
        ug, vg = torch.meshgrid(u_vals, v_vals, indexing='xy')
        uv = torch.stack([ug, vg], dim=-1).unsqueeze(0)
        scale = sample_lod_vaidyanathan(num_mips_ref, device)

        with torch.no_grad():
            ref = sample_reference(reference_mips, uv, scale, filter_mode)

        pred = bc_model(uv, scale)

        if loss_fn == 'l1':
            loss = F.l1_loss(pred, ref)
        else:
            loss = F.mse_loss(pred, ref)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iteration + 1) % log_interval == 0:
            with torch.no_grad():
                bc_model.eval()
                psnr, mse_val = evaluate_full(bc_model, reference_mips, device)
                print(f"  Iter {iteration+1:>6d}/{total_iterations} | "
                      f"Loss: {loss.item():.6f} | PSNR: {psnr:.2f} dB")
                bc_model.train()

                if mse_val < best_mse:
                    best_mse = mse_val
                    best_psnr = psnr
                    torch.save({
                        'phase': 'qat',
                        'bc_format': bc_format_name,
                        'iteration': iteration + 1,
                        'model_state_dict': bc_model.state_dict(),
                        'psnr': best_psnr,
                    }, os.path.join(output_dir, 'best_model.pth'))

        if (iteration + 1) % 10000 == 0:
            torch.save({
                'phase': 'qat',
                'bc_format': bc_format_name,
                'iteration': iteration + 1,
                'model_state_dict': bc_model.state_dict(),
            }, os.path.join(output_dir, f'checkpoint_{iteration + 1}.pth'))

    torch.save({
        'phase': 'qat',
        'bc_format': bc_format_name,
        'iteration': total_iterations,
        'model_state_dict': bc_model.state_dict(),
    }, os.path.join(output_dir, 'final_model.pth'))

    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")
    return best_psnr


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    config_path = 'configs/bc6_bcf05k.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    if os.path.exists(config_path):
        config = load_config(config_path)
        config['training']['bc_format'] = config['bc_format']
        params = get_training_params(config)
        params['bc_format'] = config['bc_format']
        params['model_params'] = get_model_params(config)
        print(f"Loaded config: {config_path}")
        print(f"  BC format: {config['bc_format']}")
        print(f"  Iterations: {params['total_iterations']}")
    else:
        print(f"Config not found: {config_path}, using defaults")
        params = {
            'bc_format': 'bc6',
            'total_iterations': 206000,
            'batch_res': 512,
            'log_interval': 1000,
            'output_dir': 'output_bc6',
            'lr_feat': 1e-2,
            'lr_mlp': 1e-3,
            'betas': [0.9, 0.999],
            'loss_fn': 'l1',
            'model_params': {
                'feature_configs': [
                    (512, 8, 3),
                    (256, 7, 3),
                    (128, 6, 3),
                    (64, 5, 3),
                ],
                'hidden_dim': 16,
                'num_layers': 1,
                'filter': 'trilinear',
                'half_pixel_offsets': [1, 3],
            },
        }

    torch.manual_seed(42)

    ref_tensor = load_brick_material('.', target_res=1024)
    print(f"Reference shape: {ref_tensor.shape}")

    reference_mips = build_mipmaps(ref_tensor.to('cpu'))
    print(f"Mip levels: {[m.shape for m in reference_mips]}")

    start_time = time.time()
    best_psnr = train_bc(reference_mips, params, device=device)
    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed/60:.1f} minutes. Best PSNR: {best_psnr:.2f} dB")
