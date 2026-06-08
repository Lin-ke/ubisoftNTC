import torch
import torch.nn.functional as F
import os
import sys
import time
import argparse
import types

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
from ntc_visualization import (
    _normalize_channel_names,
    _default_inference_channels,
    _channel_indices,
    visualize_comparison,
)
from ntc_reporting import (
    print_header,
    print_uc_header,
    print_result,
    print_uc_result,
    summarize,
    summarize_uc,
    save_tsv,
    save_uc_tsv,
    _write_eval_done,
    _write_train_done,
)
from ntc_checkpointing import (
    _bc_ckpt_path,
    _append_uc_psnr,
    _restore_model_params_from_ckpt,
    _prepare_train_ckpt,
)


def _model_set_inference_channels(self, channels=None, ref_dim=9):
    names = _normalize_channel_names(channels)
    if names is None:
        names = _default_inference_channels(getattr(self, 'output_dim', 9), ref_dim=ref_dim)
    self.inference_channels = names
    self.inference_channel_indices = _channel_indices(names, ref_dim=ref_dim)
    return names


def _model_get_inference_channels(self):
    names = _normalize_channel_names(getattr(self, 'inference_channels', None))
    if names is None:
        names = self.set_inference_channels()
    return names


def _compute_bc_storage_bits(model, mlp_param_bits=16):
    total_bits = 0
    for grid in model.feature_grids:
        for mip in grid.mips:
            fmt = mip.bc_format
            C = mip.feature_dim
            num_eps = fmt.get_endpoint_count()
            sum_eps_bits = sum(fmt.get_endpoint_bits(C))
            idx_bits = fmt.get_index_bits()
            block_bits = num_eps * sum_eps_bits + 16 * idx_bits
            total_bits += mip.blocks_h * mip.blocks_w * block_bits

    mlp_params = sum(p.numel() for p in model.mlp.parameters())
    return total_bits + mlp_params * mlp_param_bits


def _model_compute_storage_bits(self, mlp_param_bits=16):
    return _compute_bc_storage_bits(self, mlp_param_bits=mlp_param_bits)


def _model_compute_reference_bits(self, dataset_root, material_name):
    return get_png_bytes(dataset_root, material_name) * 8


def _model_compute_compression_stats(self, dataset_root, material_name, mlp_param_bits=16):
    bc_bits = self.compute_storage_bits(mlp_param_bits=mlp_param_bits)
    png_bits = self.compute_reference_bits(dataset_root, material_name)
    return {
        'bc_bits': bc_bits,
        'png_bits': png_bits,
        'compression_ratio': round(bc_bits / max(png_bits, 1), 4),
    }


def _attach_model_interfaces(model, inference_channels=None, ref_dim=9):
    """把通道记录、benchmark、压缩统计挂到模型实例上。"""
    if not hasattr(model, 'set_inference_channels'):
        model.set_inference_channels = types.MethodType(_model_set_inference_channels, model)
    if not hasattr(model, 'get_inference_channels'):
        model.get_inference_channels = types.MethodType(_model_get_inference_channels, model)
    if not hasattr(model, 'compute_storage_bits'):
        model.compute_storage_bits = types.MethodType(_model_compute_storage_bits, model)
    if not hasattr(model, 'compute_reference_bits'):
        model.compute_reference_bits = types.MethodType(_model_compute_reference_bits, model)
    if not hasattr(model, 'compute_compression_stats'):
        model.compute_compression_stats = types.MethodType(_model_compute_compression_stats, model)
    model.set_inference_channels(inference_channels, ref_dim=ref_dim)
    return model


# ============================================================
# UC 训练 / 加载
# ============================================================

def _sample_uv_and_lod(ref_h, ref_w, batch_res, num_mips, device, max_useful_lod=None):
    """随机取一个 batch_res × batch_res 的 UV crop + 一个 Vaidyanathan LOD."""
    H = W = batch_res
    u0 = torch.rand(1, device=device) * (1.0 - W / ref_w)
    v0 = torch.rand(1, device=device) * (1.0 - H / ref_h)
    u = torch.linspace(u0.item(), u0.item() + W / ref_w, W, device=device)
    v = torch.linspace(v0.item(), v0.item() + H / ref_h, H, device=device)
    ug, vg = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([ug, vg], dim=-1).unsqueeze(0)
    scale = sample_lod_vaidyanathan(num_mips, device, max_useful_lod=max_useful_lod)
    return uv, scale


def train_and_save_uc(ref_mips, output_dim, model_params, uc_params, device, save_path):
    model = _attach_model_interfaces(make_model(model_params, output_dim=output_dim).to(device), ref_dim=output_dim)
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
    # GT 滤波永远走 bicubic (论文 Sec 5.1), 与神经特征端的 trilinear 解耦.
    # model.filter 仅作用于神经特征采样 (硬件 sampler 模拟), 不影响 GT.
    gt_filter = 'bicubic'
    loss_channels = uc_params.get('loss_channels', None)
    max_useful_lod = uc_params.get('max_useful_lod', None)

    for it in range(iterations):
        model.train()
        uv, scale = _sample_uv_and_lod(ref_h, ref_w, batch_res, num_mips, device, max_useful_lod)

        with torch.no_grad():
            ref = sample_reference(ref_mips, uv, scale, gt_filter)
        pred = model(uv, scale)
        if loss_channels is not None:
            pred = pred[:, loss_channels, :, :]
            ref = ref[:, loss_channels, :, :]
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
    model = _attach_model_interfaces(make_model(model_params, output_dim=output_dim).to(device), ref_dim=output_dim)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    return model


# ============================================================
# BC 训练 / 加载
# ============================================================

def train_and_save_bc(ref_mips, output_dim, model_params, bc_format_name, bc_params, device, save_path):
    bc_model = train_bc_model(ref_mips, output_dim, model_params, bc_format_name, bc_params, device)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'model_state_dict': bc_model.state_dict()}, save_path)
    return bc_model


def load_bc(save_path, model_params, output_dim, bc_format_name, device):
    bc_model = make_bc_model(model_params, output_dim=output_dim,
                             bc_format_name=bc_format_name).to(device)
    _attach_model_interfaces(bc_model, ref_dim=output_dim)
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    bc_model.load_state_dict(state)
    bc_model.eval()
    return bc_model

def train_bc_model(ref_mips, output_dim, model_params, bc_format_name, bc_params, device):
    bc_model = make_bc_model(model_params, output_dim=output_dim,
                             bc_format_name=bc_format_name).to(device)
    _attach_model_interfaces(bc_model, ref_dim=output_dim)
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
    # GT 滤波永远走 bicubic (论文 Sec 5.1), 与神经特征端的 trilinear 解耦.
    gt_filter = 'bicubic'
    gamma = bc_params.get('gamma', 1.0)
    loss_channels = bc_params.get('loss_channels', None)
    max_useful_lod = bc_params.get('max_useful_lod', None)

    for it in range(iterations):
        bc_model.train()
        uv, scale = _sample_uv_and_lod(ref_h, ref_w, batch_res, num_mips, device, max_useful_lod)

        with torch.no_grad():
            ref = sample_reference(ref_mips, uv, scale, gt_filter)
        pred = bc_model(uv, scale)
        if loss_channels is not None:
            pred = pred[:, loss_channels, :, :]
            ref = ref[:, loss_channels, :, :]
        loss = F.l1_loss(pred, ref) if loss_fn == 'l1' else F.mse_loss(pred, ref)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if gamma < 1.0:
            for pg in optimizer.param_groups:
                pg['lr'] *= gamma

    # ----------------------------------------------------------------
    # 第三阶段 (论文 Sec 6.2): 冻结 BC features, finetune MLP.
    # 此时 _quantize_ste 在 train()/eval() 下都返回硬量化值, MLP 适应
    # 真正的离散特征分布, 修复"训练-导出"行为不一致.
    # ----------------------------------------------------------------
    finetune_iters = bc_params.get('mlp_finetune_iterations', 0)
    if finetune_iters and finetune_iters > 0:
        for p in feat_params:
            p.requires_grad_(False)

        ft_lr = bc_params.get('mlp_finetune_lr', bc_params['lr_mlp'])
        ft_gamma = bc_params.get('mlp_finetune_gamma', 1.0)
        ft_optimizer = torch.optim.Adam(
            list(bc_model.mlp.parameters()),
            lr=ft_lr, betas=tuple(bc_params['betas']),
        )

        for it in range(finetune_iters):
            bc_model.train()
            uv, scale = _sample_uv_and_lod(ref_h, ref_w, batch_res, num_mips, device, max_useful_lod)

            with torch.no_grad():
                ref = sample_reference(ref_mips, uv, scale, gt_filter)
            pred = bc_model(uv, scale)
            if loss_channels is not None:
                pred = pred[:, loss_channels, :, :]
                ref = ref[:, loss_channels, :, :]
            loss = F.l1_loss(pred, ref) if loss_fn == 'l1' else F.mse_loss(pred, ref)

            ft_optimizer.zero_grad()
            loss.backward()
            ft_optimizer.step()
            if ft_gamma < 1.0:
                for pg in ft_optimizer.param_groups:
                    pg['lr'] *= ft_gamma

        # 恢复 requires_grad 状态, 避免影响后续二次训练 / 重载.
        for p in feat_params:
            p.requires_grad_(True)

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


@torch.no_grad()
def _evaluate_full_loss(model, ref_mips, device, loss_fn='mse', max_res=256):
    losses = []
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
            loss = F.l1_loss(pred, ref_mip.unsqueeze(0)) if loss_fn == 'l1' else F.mse_loss(pred, ref_mip.unsqueeze(0))
        else:
            loss_total = 0.0
            count = 0
            for ty in range(0, h, max_res):
                th = min(max_res, h - ty)
                for tx in range(0, w, max_res):
                    tw = min(max_res, w - tx)
                    u = torch.linspace(tx / w, (tx + tw) / w, tw, device=device)
                    v = torch.linspace(ty / h, (ty + th) / h, th, device=device)
                    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)
                    scale = torch.tensor([float(mip_i)], device=device)
                    pred = model(uv, scale)
                    ref_tile = ref_mip[:, ty:ty + th, tx:tx + tw].unsqueeze(0)
                    tile_loss = F.l1_loss(pred, ref_tile) if loss_fn == 'l1' else F.mse_loss(pred, ref_tile)
                    loss_total += tile_loss.item() * th * tw
                    count += th * tw
            loss = loss_total / count
        losses.append(loss.item() if hasattr(loss, 'item') else loss)

    model.train()
    return sum(losses) / len(losses)


def compute_bc_bits(bc_model, mlp_param_bits=16):
    if hasattr(bc_model, 'compute_storage_bits'):
        return bc_model.compute_storage_bits(mlp_param_bits=mlp_param_bits)
    return _compute_bc_storage_bits(bc_model, mlp_param_bits=mlp_param_bits)


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

def get_or_compute_uc_psnr(name, mipmaps, model_params, output_dim, device, ckpt_dir):
    """加载 UC 模型并在传入 mipmaps 分辨率上计算 psnr_uc（不缓存，因为 eval 在 2K 上）。"""
    uc_path = os.path.join(ckpt_dir, f'{name}.pth')
    if not os.path.exists(uc_path):
        return None
    uc_model = load_uc(uc_path, model_params, output_dim, device)
    mips_gpu = [m.to(device) for m in mipmaps]
    psnr_uc, _ = evaluate_full(uc_model, mips_gpu, device)
    del uc_model
    torch.cuda.empty_cache()
    return psnr_uc


def evaluate_uc_one(name, mipmaps, model_params, device, ckpt_dir):
    output_dim = mipmaps[0].shape[0]
    ref_h, ref_w = mipmaps[0].shape[1], mipmaps[0].shape[2]
    t0 = time.time()

    uc_path = os.path.join(ckpt_dir, f'{name}.pth')
    if not os.path.exists(uc_path):
        raise FileNotFoundError(
            f"UC checkpoint not found: {uc_path}\n"
            f"Please run: python Tool.py --config <yaml> --train-uc --ckpt {ckpt_dir}"
        )

    uc_model = load_uc(uc_path, model_params, output_dim, device)
    mips_gpu = [m.to(device) for m in mipmaps]
    psnr_uc, eval_loss = evaluate_full(uc_model, mips_gpu, device)
    _append_uc_psnr(ckpt_dir, name, psnr_uc)
    del uc_model
    torch.cuda.empty_cache()

    return {
        'name': name,
        'channels': output_dim,
        'resolution': f'{ref_h}x{ref_w}',
        'psnr_ori': round(psnr_uc, 2),
        'eval_loss': round(eval_loss, 8),
        'time_total': round(time.time() - t0, 1),
    }


def evaluate_one(name, mipmaps, model_params, bc_format_name, device, ckpt_dir,
                 dataset_root, bench_params, vis_dir=None):
    output_dim = mipmaps[0].shape[0]
    ref_h, ref_w = mipmaps[0].shape[1], mipmaps[0].shape[2]
    t0 = time.time()

    results = {'name': name, 'channels': output_dim, 'resolution': f'{ref_h}x{ref_w}'}

    # 原图质量参考：加载 UC（全精度）模型，计算其对原图的 PSNR
    psnr_uc = get_or_compute_uc_psnr(name, mipmaps, model_params, output_dim, device, ckpt_dir)
    if psnr_uc is not None:
        results['psnr_ori'] = round(psnr_uc, 2)

    bc_path = _bc_ckpt_path(ckpt_dir, bc_format_name, name)
    if not os.path.exists(bc_path):
        raise FileNotFoundError(
            f"BC checkpoint not found: {bc_path}\n"
            f"Please run: python Tool.py --config <yaml> --train-bc --ckpt {ckpt_dir}"
        )
    bc_model = load_bc(bc_path, model_params, output_dim, bc_format_name, device)

    mips_gpu = [m.to(device) for m in mipmaps]
    psnr_bc, _ = evaluate_full(bc_model, mips_gpu, device)
    results['psnr_bc'] = round(psnr_bc, 2)
    # 原图 → BC 的质量 drop
    if psnr_uc is not None:
        results['psnr_drop'] = round(psnr_uc - psnr_bc, 2)

    # --- 推理时间 ---
    inf_ms = benchmark_inference_ms(
        bc_model, ref_h, ref_w, device,
        warmup_iters=bench_params['warmup_iters'],
        timing_iters=bench_params['timing_iters'],
    )
    results['inference_ms'] = round(inf_ms, 3)

    # --- 压缩率 ---
    results.update(bc_model.compute_compression_stats(
        dataset_root, name, mlp_param_bits=bench_params['mlp_param_bits']
    ))

    # --- 可视化 ---
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, f'{name}.png')
        material_data = {
            'name': name,
            'ref_tensor': mipmaps[0],
            'mipmaps': mipmaps,
            'channels': bc_model.get_inference_channels(),
        }
        visualize_comparison(bc_model, material_data, device, vis_path)
        results['vis_path'] = vis_path

    del bc_model
    torch.cuda.empty_cache()

    results['time_total'] = round(time.time() - t0, 1)
    return results


def _train_one_material(mode, name, sample, model_params, train_params,
                        bc_format_name, device, ckpt_dir):
    mipmaps = [m.cpu() for m in sample['mipmaps']]
    output_dim = sample['ref_tensor'].shape[0]
    model = None
    t0 = time.time()

    try:
        if mode == 'train-uc':
            save_path = os.path.join(ckpt_dir, f'{name}.pth')
            model = train_and_save_uc(mipmaps, output_dim, model_params,
                                      train_params, device, save_path)
            mips_gpu = [m.to(device) for m in mipmaps]
            metric, train_loss = evaluate_full(model, mips_gpu, device)
            _append_uc_psnr(ckpt_dir, name, metric)
        else:
            save_path = _bc_ckpt_path(ckpt_dir, bc_format_name, name)
            model = train_and_save_bc(mipmaps, output_dim, model_params,
                                      bc_format_name, train_params, device, save_path)
            mips_gpu = [m.to(device) for m in mipmaps]
            metric, _ = evaluate_full(model, mips_gpu, device)
            train_loss = _evaluate_full_loss(model, mips_gpu, device, train_params['loss_fn'])
    finally:
        elapsed = time.time() - t0

    result = {
        'name': name,
        'train_loss': train_loss,
        'time_total': elapsed,
    }
    if model is not None:
        del model
    torch.cuda.empty_cache()
    return result, metric


def _run_train_materials(mode, names, ds, model_params, train_params,
                         bc_format_name, device, ckpt_dir, prefix=''):
    results = []
    for i, name in enumerate(names):
        sample = ds.get_by_name(name)
        result, metric = _train_one_material(
            mode, name, sample, model_params, train_params,
            bc_format_name, device, ckpt_dir,
        )
        results.append(result)
        metric_str = f"BC {metric:>7.2f}" if mode == 'train-bc' else f"{metric:>7.2f}"
        print(f"{prefix}[{i+1:>2d}/{len(names)}] {name:<30s} {metric_str} {result['time_total']:>5.1f}s", flush=True)
    return results


def _run_mp_train(mode, gpu_ids, names_split, config_path, ckpt_dir, num_workers):
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    task_args = [
        (mode, gpu_ids[i % len(gpu_ids)], names_split[i], config_path, ckpt_dir)
        for i in range(num_workers)
    ]
    with mp.Pool(num_workers) as pool:
        worker_results = pool.map(_mp_train_worker, task_args)

    train_results = []
    for wr in worker_results:
        train_results.extend(wr)
    return train_results


def _print_train_header(mode, train_params, bc_format_name, ckpt_dir,
                        num_workers, gpu_ids):
    if mode == 'train-uc':
        print(f"=== Train UC ({train_params['total_iterations']} iters, {num_workers} workers, GPUs={gpu_ids}) ===")
        print(f"Checkpoints → {ckpt_dir}/")
        print(f"{'Material':<30s} {'PSNR':>7s} {'Time':>6s}")
        print('-' * 47)
    else:
        print(f"=== Train BC ({bc_format_name.upper()}, {train_params['total_iterations']} iters, "
              f"{num_workers} workers, GPUs={gpu_ids}) ===")
        print(f"Checkpoints → {ckpt_dir}/bc_{bc_format_name}/")
        print(f"{'Material':<30s} {'BC PSNR':>7s} {'Time':>6s}")
        print('-' * 49)


def _run_uc_eval(names, ds, model_params, device, ckpt_dir, prefix=''):
    all_results = []
    for i, name in enumerate(names):
        sample = ds.get_by_name(name)
        mipmaps = [m.cpu() for m in sample['mipmaps']]
        try:
            r = evaluate_uc_one(name, mipmaps, model_params, device, ckpt_dir)
            all_results.append(r)
            print(f"{prefix}[{i+1:>2d}/{len(names)}] ", end='', flush=True)
            print_uc_result(r)
        except FileNotFoundError as e:
            print(f"{prefix}SKIP {name}: {e}", flush=True)
        except Exception as e:
            print(f"{prefix}ERROR on {name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
    return all_results


def _run_eval_uc_stage(names, ds, model_params, device, ckpt_dir,
                       config_path, bc_format_name, num_workers,
                       use_mp=False, gpu_ids=None, names_split=None,
                       write_done=True):
    params_label = f"UC fl={model_params.get('filter')}"
    print(f"\n{'='*80}")
    print(f"Eval UC  |  {params_label}")
    print(f"Checkpoints from {ckpt_dir}")
    print(f"{'='*80}")
    print_uc_header()

    if use_mp:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        task_args = [
            (gpu_ids[i % len(gpu_ids)], names_split[i], config_path, ckpt_dir)
            for i in range(num_workers)
        ]
        with mp.Pool(num_workers) as pool:
            worker_results = pool.map(_mp_eval_uc_worker, task_args)
        all_results = []
        for wr in worker_results:
            all_results.extend(wr)
    else:
        all_results = _run_uc_eval(names, ds, model_params, device, ckpt_dir)

    if all_results:
        summarize_uc(all_results, params_label)
        cfg_stem = os.path.splitext(os.path.basename(config_path))[0]
        save_uc_tsv(all_results, ckpt_dir, cfg_stem)

    if write_done:
        _write_eval_done('eval-uc', ckpt_dir, config_path, bc_format_name, num_workers, all_results)
    return all_results


# ============================================================
# 多进程 Worker
# ============================================================

def _make_dataset(dataset_params, target_res):
    """统一构造 MaterialDataset。target_res=None 表示加载原生 2K。"""
    return MaterialDataset(
        dataset_params['root'], target_res=target_res, preload=True,
        output_channels=dataset_params.get('output_channels', 'full'),
    )


def _mp_train_worker(args):
    """多进程训练 worker."""
    mode, gpu_id, names, config_path, ckpt_dir = args
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

    config = load_config(config_path)
    bc_format_name = config['bc_format']
    model_params = get_model_params(config)
    train_params = get_uc_training_params(config) if mode == 'train-uc' else get_bc_training_params(config)
    dataset_params = get_dataset_params(config)

    if mode == 'train-bc':
        _restore_model_params_from_ckpt(ckpt_dir, model_params)

    ds = _make_dataset(dataset_params, dataset_params['target_res'])
    return _run_train_materials(
        mode, names, ds, model_params, train_params,
        bc_format_name, device, ckpt_dir, prefix=f"[{gpu_id}] ",
    )


def _mp_eval_worker(args):
    """多进程 Eval worker."""
    gpu_id, names, config_path, ckpt_dir, bench_params, vis_dir = args
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

    config = load_config(config_path)
    bc_format_name = config['bc_format']
    model_params = get_model_params(config)
    dataset_params = get_dataset_params(config)

    ds = _make_dataset(dataset_params, target_res=None)
    ds_root = dataset_params['root']

    _restore_model_params_from_ckpt(ckpt_dir, model_params)

    all_results = []
    for i, name in enumerate(names):
        sample = ds.get_by_name(name)
        mipmaps = [m.cpu() for m in sample['mipmaps']]
        try:
            r = evaluate_one(name, mipmaps, model_params, bc_format_name,
                             device, ckpt_dir, ds_root, bench_params,
                             vis_dir=vis_dir)
            all_results.append(r)
            drop_str = f"{r['psnr_drop']:>6.2f}" if 'psnr_drop' in r else f"{'N/A':>6s}"
            print(f"[{gpu_id}] [{i+1:>2d}/{len(names)}] {r['name']:<30s} "
                  f"{r['psnr_bc']:>7.2f} {drop_str} "
                  f"{r['inference_ms']:>8.3f} "
                  f"{r['compression_ratio']:>7.4f} {r['time_total']:>5.1f}s", flush=True)
        except FileNotFoundError as e:
            print(f"[{gpu_id}] SKIP {name}: {e}", flush=True)
        except Exception as e:
            print(f"[{gpu_id}] ERROR on {name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
    return all_results


def _mp_eval_uc_worker(args):
    """多进程 UC eval worker."""
    gpu_id, names, config_path, ckpt_dir = args
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

    config = load_config(config_path)
    model_params = get_model_params(config)
    dataset_params = get_dataset_params(config)
    _restore_model_params_from_ckpt(ckpt_dir, model_params)

    ds = _make_dataset(dataset_params, target_res=None)
    return _run_uc_eval(names, ds, model_params, device, ckpt_dir, prefix=f"[{gpu_id}] ")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='NTC: Neural Texture Compression')
    parser.add_argument('--config', type=str, required=True,
                        help='实验 yaml 配置文件路径 (必需)')
    parser.add_argument('--train-uc', action='store_true',
                        help='训练并保存 UC (无约束) 基线模型')
    parser.add_argument('--train-bc', action='store_true',
                        help='训练并保存 BC 压缩模型')
    parser.add_argument('--eval-uc', action='store_true',
                        help='只评估 UC (无约束) 基线模型')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint 目录 (--train-bc 可选, eval 必需)')
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
    target_res = dataset_params['target_res']  # 论文对齐: 永远 None (原生分辨率)

    ds = _make_dataset(dataset_params, target_res)
    res_label = 'native' if target_res is None else str(target_res)
    print(f"Device: {device}  |  Config: {args.config}  |  Res: {res_label}")
    print(f"BC format: {bc_format_name.upper()}  |  Filter: {model_params.get('filter')} (feature)  |  GT filter: bicubic")
    print(f"Loaded {len(ds)} materials from {ds_root}\n")

    names = ([n.strip() for n in args.materials.split(',')]
             if args.materials else ds.material_names)

    if args.gpus is not None:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    else:
        gpu_ids = list(range(torch.cuda.device_count())) if device == 'cuda' else [0]
    if len(gpu_ids) == 0:
        gpu_ids = [0]

    num_workers = max(1, args.num_workers)
    use_mp = num_workers > 1 and device == 'cuda'

    names_split = [[] for _ in range(num_workers)]
    for i, name in enumerate(names):
        names_split[i % num_workers].append(name)

    def _restore_model_params(ckpt_dir):
        _restore_model_params_from_ckpt(ckpt_dir, model_params, verbose=True)

    # ========================================================
    # 阶段1: train-uc
    # ========================================================
    if args.train_uc:
        uc_params = get_uc_training_params(config)
        ckpt_dir = _prepare_train_ckpt('train-uc', args, config, model_params)
        _print_train_header('train-uc', uc_params, bc_format_name, ckpt_dir, num_workers, gpu_ids)
        if use_mp:
            train_results = _run_mp_train('train-uc', gpu_ids, names_split, args.config, ckpt_dir, num_workers)
        else:
            train_results = _run_train_materials(
                'train-uc', names, ds, model_params, uc_params,
                bc_format_name, device, ckpt_dir,
            )
        print(f"\nDone. Trained {len(train_results)} materials.")
        _write_train_done('train-uc', ckpt_dir, args.config, bc_format_name, num_workers, train_results)
        
        print(f"Next: python Tool.py --config {args.config} --train-bc --ckpt {ckpt_dir}/")
        return

    # ========================================================
    # 阶段2: train-bc
    # ========================================================
    if args.train_bc:
        ckpt_dir = _prepare_train_ckpt('train-bc', args, config, model_params)
        _print_train_header('train-bc', bc_params, bc_format_name, ckpt_dir, num_workers, gpu_ids)
        if use_mp:
            train_results = _run_mp_train('train-bc', gpu_ids, names_split, args.config, ckpt_dir, num_workers)
        else:
            train_results = _run_train_materials(
                'train-bc', names, ds, model_params, bc_params,
                bc_format_name, device, ckpt_dir,
            )
        print(f"\nDone. Trained {len(train_results)} BC models.")
        _write_train_done('train-bc', ckpt_dir, args.config, bc_format_name, num_workers, train_results)
        print(f"Next: python Tool.py --config {args.config} --ckpt {ckpt_dir}/")
        return

    # ========================================================
    # 阶段3: eval-uc
    # ========================================================
    if args.eval_uc:
        if not args.ckpt:
            print("ERROR: eval-uc 模式需要 --ckpt <checkpoints/xxx/>")
            sys.exit(1)
        ckpt_dir = args.ckpt
        _restore_model_params(ckpt_dir)
        # Eval 在原生 2K 上进行（而非训练时的 target_res）。
        ds = _make_dataset(dataset_params, target_res=None)
        print(f"[eval-uc] Reloaded dataset at full (2K) resolution")
        _run_eval_uc_stage(
            names, ds, model_params, device, ckpt_dir, args.config,
            bc_format_name, num_workers, use_mp=use_mp,
            gpu_ids=gpu_ids, names_split=names_split,
        )
        return

    # ========================================================
    # 阶段4: eval
    # ========================================================
    if not args.ckpt:
        print("ERROR: eval 模式需要 --ckpt <checkpoints/xxx/>")
        print("Usage:")
        print("  python Tool.py --config <yaml> --train-bc")
        print("  python Tool.py --config <yaml> --train-uc")
        print("  python Tool.py --config <yaml> --ckpt <dir/>")
        sys.exit(1)

    ckpt_dir = args.ckpt
    _restore_model_params(ckpt_dir)

    # Eval 在原生 2K 上进行（而非训练时的 target_res）。
    ds = _make_dataset(dataset_params, target_res=None)
    print(f"[eval] Reloaded dataset at full (2K) resolution")

    params_label = (
        f"BC={bc_format_name.upper()} fl={model_params.get('filter')} "
        f"lr=({bc_params['lr_feat']},{bc_params['lr_mlp']}) "
        f"loss={bc_params.get('loss_fn','l1')}"
    )

    print(f"\n{'='*80}")
    print(f"Eval  |  {params_label}")
    print(f"Checkpoints from {ckpt_dir}")
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
                                 device, ckpt_dir, ds_root, bench_params,
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

    _write_eval_done('eval', ckpt_dir, args.config, bc_format_name, num_workers, all_results)


if __name__ == '__main__':
    main()
