import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import argparse
import yaml
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from dataset import MaterialDataset
from ntc_model import make_model
from ntc_bc_model import make_bc_model
from ntc_train import sample_reference, evaluate_full
from ntc_bc_train import sample_lod_vaidyanathan
from ntc_config import (
    load_config,
    get_model_params,
    get_uc_training_params,
    get_bc_training_params,
    get_dataset_params,
    get_benchmark_params,
)


def _timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H%M%S')


# ============================================================
# 可视化辅助函数
# ============================================================

def _safe_font(size=12):
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _vis_psnr(pred, ref):
    mse = F.mse_loss(pred, ref).item()
    return -10 * np.log10(max(mse, 1e-10))


def _tensor_to_hwc(tensor, is_normal=False):
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    if is_normal:
        img = (img + 1.0) / 2.0
    return np.clip(img, 0, 1)


def _to_rgb(img_np):
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    return img_np


def _add_label(img_np, label, psnr=None, font=None):
    h, w = img_np.shape[:2]
    bar_h = 18
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.float32)
    canvas[bar_h:, :] = img_np
    pil = Image.fromarray((canvas * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    text = f"{label}" if psnr is None else f"{label}  PSNR:{psnr:.2f}"
    draw.text((4, 2), text, fill=(255, 255, 255), font=font)
    return np.array(pil, dtype=np.float32) / 255.0


def _make_border(img_np, color, thickness=2):
    h, w = img_np.shape[:2]
    c = img_np.shape[2] if img_np.ndim == 3 else 1
    if c == 1:
        img_np = np.stack([img_np[:, :, 0]] * 3, axis=-1)
    bordered = np.ones((h + thickness * 2, w + thickness * 2, 3), dtype=np.float32)
    bordered[:, :] = color
    bordered[thickness:-thickness, thickness:-thickness] = img_np
    return bordered


@torch.no_grad()
def visualize_comparison(model, material_data, device, output_path, scale=0.0):
    model.eval()
    font = _safe_font()
    ref_tensor = material_data['ref_tensor'].to(device)
    name = material_data['name']
    h, w = ref_tensor.shape[1], ref_tensor.shape[2]

    mip_level = int(scale)
    mipmaps = material_data.get('mipmaps', [])
    if 0 <= mip_level < len(mipmaps):
        ref_tensor = mipmaps[mip_level].to(device)
        h, w = ref_tensor.shape[1], ref_tensor.shape[2]

    u = torch.linspace(0, 1, w, device=device)
    v = torch.linspace(0, 1, h, device=device)
    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)
    scales = torch.tensor([float(scale)], device=device)
    pred = model(uv, scales)

    pred_albedo = pred[0, 0:3]
    pred_normal_xy = pred[0, 3:5]
    pred_ao = pred[0, 6:7]
    pred_rough = pred[0, 7:8]
    pred_metal = pred[0, 8:9]

    z = torch.sqrt(torch.clamp(1.0 - pred_normal_xy[0:1] ** 2 - pred_normal_xy[1:2] ** 2, min=0))
    pred_normal = torch.cat([pred_normal_xy, z], dim=0)

    ref_albedo = ref_tensor[0:3]
    ref_normal = ref_tensor[3:6]
    ref_ao = ref_tensor[6:7]
    ref_rough = ref_tensor[7:8]
    ref_metal = ref_tensor[8:9]

    colors = {
        'Albedo': (0.8, 0.2, 0.2),
        'Normal': (0.2, 0.6, 0.8),
        'AO': (0.6, 0.6, 0.6),
        'Roughness': (0.4, 0.8, 0.4),
        'Metalness': (0.9, 0.7, 0.2),
    }

    rows = []
    for label, pred_ch, ref_ch in [
        ('Albedo', pred_albedo, ref_albedo),
        ('Normal', pred_normal, ref_normal),
        ('AO', pred_ao, ref_ao),
        ('Roughness', pred_rough, ref_rough),
        ('Metalness', pred_metal, ref_metal),
    ]:
        psnr = _vis_psnr(pred_ch.unsqueeze(0), ref_ch.unsqueeze(0))
        pred_np = _to_rgb(_tensor_to_hwc(pred_ch, is_normal=(label == 'Normal')))
        ref_np = _to_rgb(_tensor_to_hwc(ref_ch, is_normal=(label == 'Normal')))
        diff = np.abs(pred_np - ref_np)

        panels = [pred_np, ref_np, diff]
        titles = [f'{label} Pred', f'{label} Ref', f'{label} Diff']

        labeled = []
        for idx, (p, t) in enumerate(zip(panels, titles)):
            p = _add_label(p, t, psnr if idx == 0 else None, font=font)
            p = _make_border(p, colors[label], thickness=2)
            labeled.append(p)

        min_h = min(p.shape[0] for p in labeled)
        labeled = [p[:min_h] for p in labeled]
        row = np.concatenate(labeled, axis=1)
        rows.append(row)

    min_w = min(r.shape[1] for r in rows)
    rows = [r[:, :min_w] for r in rows]
    sep = np.zeros((4, min_w, 3), dtype=np.float32)
    final_rows = []
    for i, r in enumerate(rows):
        final_rows.append(r)
        if i < len(rows) - 1:
            final_rows.append(sep)

    full = np.concatenate(final_rows, axis=0)
    title_h = 22
    title_bar = np.zeros((title_h, full.shape[1], 3), dtype=np.float32)
    title_bar[:, :] = (0.1, 0.1, 0.15)
    full = np.concatenate([title_bar, full], axis=0)

    pil = Image.fromarray((full * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    header = f"Material: {name}  |  Scale (mip): {scale:.1f}"
    draw.text((6, 3), header, fill=(255, 255, 255), font=font)
    pil.save(output_path)


# ============================================================
# UC 训练 / 加载
# ============================================================

def _sample_uv_and_lod(ref_h, ref_w, batch_res, num_mips, device):
    """随机取一个 batch_res × batch_res 的 UV crop + 一个 Vaidyanathan LOD."""
    H = W = batch_res
    u0 = torch.rand(1, device=device) * (1.0 - W / ref_w)
    v0 = torch.rand(1, device=device) * (1.0 - H / ref_h)
    u = torch.linspace(u0.item(), u0.item() + W / ref_w, W, device=device)
    v = torch.linspace(v0.item(), v0.item() + H / ref_h, H, device=device)
    ug, vg = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([ug, vg], dim=-1).unsqueeze(0)
    scale = sample_lod_vaidyanathan(num_mips, device)
    return uv, scale


def train_and_save_uc(ref_mips, output_dim, model_params, uc_params, device, save_path):
    model = make_model(model_params, output_dim=output_dim).to(device)
    ref_mips = [m.to(device) for m in ref_mips]
    num_mips = len(ref_mips)
    ref_h, ref_w = ref_mips[0].shape[1], ref_mips[0].shape[2]

    optimizer = torch.optim.Adam([
        {'params': list(model.feature_grids.parameters()), 'lr': uc_params['lr_feat']},
        {'params': list(model.mlp.parameters()), 'lr': uc_params['lr_mlp']},
    ])
    gamma = uc_params['gamma']
    iterations = uc_params['total_iterations']
    batch_res = uc_params['batch_res']
    filter_mode = model_params['filter']

    for it in range(iterations):
        model.train()
        uv, scale = _sample_uv_and_lod(ref_h, ref_w, batch_res, num_mips, device)

        with torch.no_grad():
            ref = sample_reference(ref_mips, uv, scale, filter_mode)
        pred = model(uv, scale)
        loss = F.mse_loss(pred, ref)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for pg in optimizer.param_groups:
            pg['lr'] *= gamma

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return model


def load_uc(save_path, model_params, output_dim, device):
    model = make_model(model_params, output_dim=output_dim).to(device)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    return model


# ============================================================
# BC QAT 训练
# ============================================================

def train_bc_model(ref_mips, output_dim, model_params, bc_format_name, bc_params, device):
    bc_model = make_bc_model(model_params, output_dim=output_dim,
                             bc_format_name=bc_format_name).to(device)
    ref_mips = [m.to(device) for m in ref_mips]
    num_mips = len(ref_mips)
    ref_h, ref_w = ref_mips[0].shape[1], ref_mips[0].shape[2]

    feat_params = []
    for grid in bc_model.feature_grids:
        for mip in grid.mips:
            feat_params.append(mip.endpoints)
            feat_params.append(mip.raw_indices)

    optimizer = torch.optim.Adam([
        {'params': feat_params, 'lr': bc_params['lr_feat']},
        {'params': list(bc_model.mlp.parameters()), 'lr': bc_params['lr_mlp']},
    ], betas=tuple(bc_params['betas']))

    iterations = bc_params['total_iterations']
    batch_res = bc_params['batch_res']
    loss_fn = bc_params['loss_fn']
    filter_mode = model_params['filter']

    for it in range(iterations):
        bc_model.train()
        uv, scale = _sample_uv_and_lod(ref_h, ref_w, batch_res, num_mips, device)

        with torch.no_grad():
            ref = sample_reference(ref_mips, uv, scale, filter_mode)
        pred = bc_model(uv, scale)
        loss = F.l1_loss(pred, ref) if loss_fn == 'l1' else F.mse_loss(pred, ref)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return bc_model


# ============================================================
# 指标: 推理时间 / 压缩率
# ============================================================

def benchmark_inference_ms(bc_model, ref_h, ref_w, device, warmup_iters=5, timing_iters=20):
    bc_model.eval()
    u = torch.linspace(0.0, 1.0, ref_w, device=device)
    v = torch.linspace(0.0, 1.0, ref_h, device=device)
    ug, vg = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([ug, vg], dim=-1).unsqueeze(0)
    scale = torch.tensor([0.0], device=device)

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = bc_model(uv, scale)

    if str(device) == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(timing_iters):
            _ = bc_model(uv, scale)
    if str(device) == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return (elapsed / timing_iters) * 1000.0


def compute_bc_bits(bc_model, mlp_param_bits=16):
    total_bits = 0
    for grid in bc_model.feature_grids:
        for mip in grid.mips:
            fmt = mip.bc_format
            C = mip.feature_dim
            num_eps = fmt.get_endpoint_count()
            sum_eps_bits = sum(fmt.get_endpoint_bits(C))
            idx_bits = fmt.get_index_bits()

            block_bits = num_eps * sum_eps_bits + 16 * idx_bits
            total_bits += mip.blocks_h * mip.blocks_w * block_bits

    mlp_params = sum(p.numel() for p in bc_model.mlp.parameters())
    total_bits += mlp_params * mlp_param_bits
    return total_bits


def get_png_bytes(dataset_root, material_name):
    mdir = os.path.join(dataset_root, material_name)
    total = 0
    for fname in os.listdir(mdir):
        if fname.endswith('.png'):
            total += os.path.getsize(os.path.join(mdir, fname))
    return total


# ============================================================
# 单材质评估
# ============================================================

def evaluate_one(name, mipmaps, model_params, bc_format_name, device, ckpt_dir,
                 bc_params, dataset_root, bench_params, vis_dir=None):
    output_dim = mipmaps[0].shape[0]
    ref_h, ref_w = mipmaps[0].shape[1], mipmaps[0].shape[2]
    t0 = time.time()

    results = {'name': name, 'channels': output_dim, 'resolution': f'{ref_h}x{ref_w}'}

    # --- UC ---
    uc_path = os.path.join(ckpt_dir, f'{name}.pth')
    if not os.path.exists(uc_path):
        raise FileNotFoundError(
            f"UC checkpoint 不存在: {uc_path}\n请先运行: python evaluate.py --config <yaml> --train-uc"
        )

    uc_model = load_uc(uc_path, model_params, output_dim, device)
    mips_gpu = [m.to(device) for m in mipmaps]
    psnr_uc, _ = evaluate_full(uc_model, mips_gpu, device)
    results['psnr_unconstrained'] = round(psnr_uc, 2)
    del uc_model
    torch.cuda.empty_cache()

    # --- BC QAT ---
    bc_model = train_bc_model(mipmaps, output_dim, model_params, bc_format_name,
                              bc_params, device)
    psnr_bc, _ = evaluate_full(bc_model, mips_gpu, device)
    results['psnr_bc'] = round(psnr_bc, 2)
    results['psnr_drop'] = round(psnr_uc - psnr_bc, 2)

    # --- 推理时间 ---
    inf_ms = benchmark_inference_ms(
        bc_model, ref_h, ref_w, device,
        warmup_iters=bench_params['warmup_iters'],
        timing_iters=bench_params['timing_iters'],
    )
    results['inference_ms'] = round(inf_ms, 3)

    # --- 压缩率 ---
    bc_bits = compute_bc_bits(bc_model, mlp_param_bits=bench_params['mlp_param_bits'])
    png_bytes = get_png_bytes(dataset_root, name)
    png_bits = png_bytes * 8
    results['bc_bits'] = bc_bits
    results['png_bits'] = png_bits
    results['compression_ratio'] = round(bc_bits / max(png_bits, 1), 4)

    # --- 可视化 ---
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, f'{name}.png')
        material_data = {'name': name, 'ref_tensor': mipmaps[0], 'mipmaps': mipmaps}
        visualize_comparison(bc_model, material_data, device, vis_path)
        results['vis_path'] = vis_path

    del bc_model
    torch.cuda.empty_cache()

    results['time_total'] = round(time.time() - t0, 1)
    return results


# ============================================================
# 输出
# ============================================================

def print_header():
    hdr = (f"{'Material':<30s} {'UC(dB)':>7s} {'BC(dB)':>7s} {'Drop':>6s} "
           f"{'Inf(ms)':>8s} {'CR':>7s} {'Time':>6s}")
    print(hdr)
    print('-' * 80)


def print_result(r):
    print(f"{r['name']:<30s} {r['psnr_unconstrained']:>7.2f} "
          f"{r['psnr_bc']:>7.2f} {r['psnr_drop']:>6.2f} "
          f"{r['inference_ms']:>8.3f} {r['compression_ratio']:>7.4f} "
          f"{r['time_total']:>5.1f}s")


def summarize(all_results, params_label):
    print(f"\n{'='*80}")
    print(f"Aggregate over {len(all_results)} materials  "
          f"({all_results[0].get('resolution','?')}, {params_label}):")
    print(f"{'Metric':>28s} {'Mean':>10s} {'Min':>10s} {'Max':>10s} {'Std':>10s}")
    print('-' * 80)

    for k in ['psnr_unconstrained', 'psnr_bc', 'psnr_drop',
              'inference_ms', 'compression_ratio']:
        vals = [r[k] for r in all_results]
        print(f"{k:>28s} {np.mean(vals):>10.4f} {np.min(vals):>10.4f} "
              f"{np.max(vals):>10.4f} {np.std(vals):>10.4f}")

    times = [r['time_total'] for r in all_results]
    print(f"\nTotal: {sum(times):.0f}s  |  Avg: {np.mean(times):.1f}s/material")
    print(f"{'='*80}")


def save_tsv(all_results, ckpt_dir, suffix):
    path = os.path.join(ckpt_dir, f'eval_{suffix}.tsv')
    cols = ['name', 'channels', 'resolution',
            'psnr_unconstrained', 'psnr_bc', 'psnr_drop',
            'inference_ms', 'bc_bits', 'png_bits', 'compression_ratio',
            'time_total']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(cols) + '\n')
        for r in all_results:
            f.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')
    print(f"Saved to {path}")


# ============================================================
# 多进程 Worker
# ============================================================

def _mp_train_uc_worker(args):
    """多进程 UC 训练 worker."""
    gpu_id, names, config_path, ckpt_dir = args
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

    config = load_config(config_path)
    model_params = get_model_params(config)
    uc_params = get_uc_training_params(config)
    dataset_params = get_dataset_params(config)

    ds = MaterialDataset(dataset_params['root'], target_res=dataset_params['target_res'], preload=True)

    results = []
    for i, name in enumerate(names):
        sample = ds.get_by_name(name)
        mipmaps = [m.cpu() for m in sample['mipmaps']]
        output_dim = sample['ref_tensor'].shape[0]
        save_path = os.path.join(ckpt_dir, f'{name}.pth')

        t0 = time.time()
        model = train_and_save_uc(mipmaps, output_dim, model_params, uc_params, device, save_path)
        mips_gpu = [m.to(device) for m in mipmaps]
        psnr_uc, _ = evaluate_full(model, mips_gpu, device)
        el = time.time() - t0
        results.append((name, psnr_uc, el))
        print(f"[{gpu_id}] [{i+1:>2d}/{len(names)}] {name:<30s} {psnr_uc:>7.2f} {el:>5.1f}s")
        del model
        torch.cuda.empty_cache()
    return results


def _mp_eval_worker(args):
    """多进程 Eval worker."""
    gpu_id, names, config_path, ckpt_dir, bench_params, vis_dir = args
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

    config = load_config(config_path)
    bc_format_name = config['bc_format']
    model_params = get_model_params(config)
    bc_params = get_bc_training_params(config)
    dataset_params = get_dataset_params(config)

    ds = MaterialDataset(dataset_params['root'], target_res=dataset_params['target_res'], preload=True)
    ds_root = dataset_params['root']

    # 如果 ckpt_dir 有 config.yaml，恢复模型形状
    ckpt_yaml = os.path.join(ckpt_dir, 'config.yaml')
    if os.path.exists(ckpt_yaml):
        try:
            saved = load_config(ckpt_yaml)
        except ValueError:
            with open(ckpt_yaml, 'r', encoding='utf-8') as f:
                saved = yaml.safe_load(f)
        ckpt_model_params = get_model_params(saved)
        for k in ('feature_configs', 'hidden_dim', 'num_layers', 'half_pixel_offsets'):
            model_params[k] = ckpt_model_params[k]

    all_results = []
    for i, name in enumerate(names):
        sample = ds.get_by_name(name)
        mipmaps = [m.cpu() for m in sample['mipmaps']]
        try:
            r = evaluate_one(name, mipmaps, model_params, bc_format_name,
                             device, ckpt_dir, bc_params, ds_root, bench_params,
                             vis_dir=vis_dir)
            all_results.append(r)
            print(f"[{gpu_id}] [{i+1:>2d}/{len(names)}] {r['name']:<30s} "
                  f"{r['psnr_unconstrained']:>7.2f} {r['psnr_bc']:>7.2f} "
                  f"{r['psnr_drop']:>6.2f} {r['inference_ms']:>8.3f} "
                  f"{r['compression_ratio']:>7.4f} {r['time_total']:>5.1f}s")
        except FileNotFoundError as e:
            print(f"[{gpu_id}] SKIP {name}: {e}")
        except Exception as e:
            print(f"[{gpu_id}] ERROR on {name}: {e}")
            import traceback
            traceback.print_exc()
    return all_results


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='NTC Dataset Quantization Evaluation (yaml-driven)')
    parser.add_argument('--config', type=str, required=True,
                        help='实验 yaml 配置文件路径 (必需)')
    parser.add_argument('--train-uc', action='store_true',
                        help='训练并保存无约束基线模型 (使用 yaml 中的 uc_training 段)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='已有 UC 的 checkpoint 目录 (eval 模式必需)')
    parser.add_argument('--materials', type=str, default=None,
                        help='逗号分隔的材质名, 默认全部')
    parser.add_argument('--vis-dir', type=str, default=None,
                        help='可视化输出目录, 若指定则每个材质生成 BC 预测 vs 参考的对比图')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='并行进程数 (默认 1, 每个进程占一个 GPU)')
    parser.add_argument('--gpus', type=str, default=None,
                        help='指定 GPU ID, 如 "0,1,2", 默认使用所有可用 GPU')
    args = parser.parse_args()

    config = load_config(args.config)
    bc_format_name = config['bc_format']
    model_params = get_model_params(config)
    bc_params = get_bc_training_params(config)
    dataset_params = get_dataset_params(config)
    bench_params = get_benchmark_params(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds_root = dataset_params['root']
    target_res = dataset_params['target_res']

    ds = MaterialDataset(ds_root, target_res=target_res, preload=True)
    print(f"Device: {device}  |  Config: {args.config}  |  Res: {target_res}")
    print(f"BC format: {bc_format_name.upper()}  |  Filter: {model_params.get('filter')}")
    print(f"Loaded {len(ds)} materials from {ds_root}\n")

    names = ([n.strip() for n in args.materials.split(',')]
             if args.materials else ds.material_names)

    # 解析 GPU 列表
    if args.gpus is not None:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    else:
        gpu_ids = list(range(torch.cuda.device_count())) if device == 'cuda' else [0]
    if len(gpu_ids) == 0:
        gpu_ids = [0]

    num_workers = max(1, args.num_workers)
    use_mp = num_workers > 1 and device == 'cuda'

    # 将材料轮询分配给各个 worker
    names_split = [[] for _ in range(num_workers)]
    for i, name in enumerate(names):
        names_split[i % num_workers].append(name)

    # ========================================================
    # 阶段: train-uc
    # ========================================================
    if args.train_uc:
        uc_params = get_uc_training_params(config)
        ckpt_dir = os.path.join('checkpoints', _timestamp())
        os.makedirs(ckpt_dir, exist_ok=True)

        # 把整份 yaml 配置原样保存到 ckpt_dir/config.yaml, 后续 eval 复用
        with open(os.path.join(ckpt_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"Checkpoints → {ckpt_dir}/")
        print(f"=== Train UC ({uc_params['total_iterations']} iters, {num_workers} workers, GPUs={gpu_ids}) ===")
        print(f"{'Material':<30s} {'PSNR':>7s} {'Time':>6s}")
        print('-' * 47)

        if use_mp:
            import torch.multiprocessing as mp
            mp.set_start_method('spawn', force=True)
            task_args = [
                (gpu_ids[i % len(gpu_ids)], names_split[i], args.config, ckpt_dir)
                for i in range(num_workers)
            ]
            with mp.Pool(num_workers) as pool:
                worker_results = pool.map(_mp_train_uc_worker, task_args)
            # 汇总打印
            total_trained = sum(len(wr) for wr in worker_results)
            print(f"\nDone. Trained {total_trained} materials. Use --ckpt {ckpt_dir}/ for eval.")
        else:
            for i, name in enumerate(names):
                sample = ds.get_by_name(name)
                mipmaps = [m.cpu() for m in sample['mipmaps']]
                output_dim = sample['ref_tensor'].shape[0]
                save_path = os.path.join(ckpt_dir, f'{name}.pth')

                t0 = time.time()
                model = train_and_save_uc(mipmaps, output_dim, model_params,
                                          uc_params, device, save_path)
                mips_gpu = [m.to(device) for m in mipmaps]
                psnr_uc, _ = evaluate_full(model, mips_gpu, device)
                el = time.time() - t0
                print(f"[{i+1:>2d}/{len(names)}] {name:<30s} {psnr_uc:>7.2f} {el:>5.1f}s")
                del model
                torch.cuda.empty_cache()

            print(f"\nDone. Use --ckpt {ckpt_dir}/ for eval.")
        return

    # ========================================================
    # 阶段: eval
    # ========================================================
    if not args.ckpt:
        print("ERROR: eval 模式需要 --ckpt <checkpoints/xxx/>")
        sys.exit(1)

    ckpt_dir = args.ckpt
    # 从 ckpt 的 config.yaml 恢复 model 结构 (确保和训练 UC 时形状一致)
    ckpt_yaml = os.path.join(ckpt_dir, 'config.yaml')
    if os.path.exists(ckpt_yaml):
        try:
            saved = load_config(ckpt_yaml)
        except ValueError as e:
            # UC checkpoint 的 config 可能包含实验性的 bc_format（如 bc6），
            # 但模型形状参数与 bc_format 无关，fallback 到 raw yaml
            with open(ckpt_yaml, 'r', encoding='utf-8') as f:
                saved = yaml.safe_load(f)
            print(f"[eval] Config validation warning (ignored): {e}")
        ckpt_model_params = get_model_params(saved)
        # 关键形状字段: feature_configs / hidden_dim / num_layers / half_pixel_offsets
        # 这些必须用 ckpt 的, 否则 load_state_dict 失败
        for k in ('feature_configs', 'hidden_dim', 'num_layers', 'half_pixel_offsets'):
            model_params[k] = ckpt_model_params[k]
        print(f"Loaded model shape from {ckpt_yaml}")
    else:
        print(f"WARNING: {ckpt_yaml} not found, using --config 'model' as-is")

    params_label = (
        f"BC={bc_format_name.upper()} fl={model_params.get('filter')} "
        f"lr=({bc_params['lr_feat']},{bc_params['lr_mlp']}) "
        f"loss={bc_params.get('loss_fn','l1')}"
    )

    print(f"\n{'='*80}")
    print(f"Eval  |  {params_label}")
    print(f"UC loaded from {ckpt_dir}")
    print(f"{'='*80}")
    print_header()

    if use_mp:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        task_args = [
            (gpu_ids[i % len(gpu_ids)], names_split[i], args.config, ckpt_dir, bench_params, args.vis_dir)
            for i in range(num_workers)
        ]
        with mp.Pool(num_workers) as pool:
            worker_results = pool.map(_mp_eval_worker, task_args)
        all_results = []
        for wr in worker_results:
            all_results.extend(wr)
    else:
        all_results = []
        for i, name in enumerate(names):
            sample = ds.get_by_name(name)
            mipmaps = [m.cpu() for m in sample['mipmaps']]

            print(f"[{i+1:>2d}/{len(names)}] ", end='', flush=True)
            try:
                r = evaluate_one(name, mipmaps, model_params, bc_format_name,
                                 device, ckpt_dir, bc_params, ds_root, bench_params,
                                 vis_dir=args.vis_dir)
                all_results.append(r)
                print_result(r)
            except FileNotFoundError as e:
                print(f"\n  SKIP ({e})\n")
                break
            except Exception as e:
                print(f"\n  ERROR on {name}: {e}")
                import traceback; traceback.print_exc()

    if all_results:
        summarize(all_results, params_label)
        cfg_stem = os.path.splitext(os.path.basename(args.config))[0]
        save_tsv(all_results, ckpt_dir, cfg_stem)
        sorted_by_drop = sorted(all_results, key=lambda x: x['psnr_drop'])
        print(f"\nBest  3 (lowest BC drop):")
        for r in sorted_by_drop[:3]:
            print(f"  {r['name']:<30s} drop={r['psnr_drop']:.2f} dB")
        print(f"\nWorst 3 (highest BC drop):")
        for r in sorted_by_drop[-3:]:
            print(f"  {r['name']:<30s} drop={r['psnr_drop']:.2f} dB")


if __name__ == '__main__':
    main()
