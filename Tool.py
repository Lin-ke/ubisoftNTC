import torch
import torch.nn.functional as F
import os
import sys
import time
import argparse

from dataset import MaterialDataset
from ntc_model import make_model
from ntc_bc_model import make_bc_model
from ntc_train import sample_reference, evaluate_full
from ntc_bc_train import sample_lod_vaidyanathan
from ntc_utils import compute_traditional_bc_psnr
from ntc_config import (
    load_config,
    get_model_params,
    get_uc_training_params,
    get_bc_training_params,
    get_bc_mlp_training_params,
    get_dataset_params,
    get_benchmark_params,
)
from ntc_visualization import visualize_comparison
from ntc_reporting import (
    print_header,
    print_result,
    summarize,
    save_tsv,
    _write_eval_done,
    _write_train_done,
)
from ntc_checkpointing import (
    _bc_ckpt_path,
    _uc_ckpt_path,
    _restore_model_params_from_ckpt,
    _timestamp,
    _write_config_yaml,
)


# ============================================================
# UV + LOD 采样
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


# ============================================================
# 统一训练 / 加载 (UC / BC / BC-MLP)
# ============================================================

def _make_train_model(mode, model_params, output_dim, bc_format_name, device):
    if mode == 'train-uc':
        model = make_model(model_params, output_dim=output_dim)
    else:
        model = make_bc_model(model_params, output_dim=output_dim,
                              bc_format_name=bc_format_name)
    model = model.to(device)
    if hasattr(model, 'set_inference_channels'):
        model.set_inference_channels(ref_dim=output_dim)
    return model


def _with_bc_quant(model_params, train_params):
    """Attach BC-only quantization params without changing UC construction."""
    params = dict(model_params)
    if model_params.get('encoding', 'pyramid') == 'hash_grid':
        params['hash_grid_quant'] = train_params.get('hash_grid_quant', {})
    return params


def _load_state_dict_into(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    return model


def _load_model(mode, save_path, model_params, output_dim, bc_format_name, device):
    model = _make_train_model(mode, model_params, output_dim, bc_format_name, device)
    _load_state_dict_into(model, save_path, device)
    model.eval()
    return model


def _make_train_optimizer(model, mode, train_params):
    feat_params = list(model.feature_grids.parameters())
    mlp_params = list(model.mlp.parameters())
    betas = tuple(train_params.get('betas', [0.9, 0.999]))

    if mode == 'train-bc-mlp':
        # finetune 阶段冻结 feature, 仅优化 MLP.
        for p in feat_params:
            p.requires_grad = False
        return torch.optim.Adam(mlp_params, lr=train_params['lr_mlp'], betas=betas)

    return torch.optim.Adam([
        {'params': feat_params, 'lr': train_params['lr_feat']},
        {'params': mlp_params, 'lr': train_params['lr_mlp']},
    ], betas=betas)


def _run_training_loop(model, ref_mips, device, optimizer, loss_fn,
                       total_iterations, batch_res, gamma=1.0,
                       loss_channels=None, max_useful_lod=None,
                       save_path=None, save_interval=None):
    num_mips = len(ref_mips)
    ref_h, ref_w = ref_mips[0].shape[1], ref_mips[0].shape[2]
    gt_filter = 'bicubic'
    do_periodic_save = save_path is not None and save_interval is not None

    loss_total = 0.0
    for it in range(total_iterations):
        model.train()
        uv, scale = _sample_uv_and_lod(ref_h, ref_w, batch_res, num_mips, device, max_useful_lod)
        with torch.no_grad():
            ref = sample_reference(ref_mips, uv, scale, gt_filter)
        pred = model(uv, scale)
        if loss_channels is not None:
            pred = pred[:, loss_channels, :, :]
            ref = ref[:, loss_channels, :, :]
        loss = F.l1_loss(pred, ref) if loss_fn == 'l1' else F.mse_loss(pred, ref)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        if gamma < 1.0:
            for pg in optimizer.param_groups:
                pg['lr'] *= gamma
        step = it + 1
        if do_periodic_save and step != total_iterations and step % save_interval == 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({'model_state_dict': model.state_dict()}, save_path)
    return model, loss_total / total_iterations


def _train_and_save_model(mode, ref_mips, output_dim, model_params, bc_format_name,
                          train_params, device, save_path,
                          init_from_uc_model=None, existing_model=None,
                          save_interval=None):
    """构建/复用模型并训练.

    - mode='train-uc': 从零构建.
    - 传 init_from_uc_model: 用 UC 模型初始化 BC 端点/索引/MLP.
    - 传 existing_model: 复用已有模型继续训练 (BC-MLP finetune).
    """
    if existing_model is not None:
        model = existing_model
    else:
        build_params = _with_bc_quant(model_params, train_params) if mode != 'train-uc' else model_params
        model = _make_train_model(mode, build_params, output_dim, bc_format_name, device)
        if init_from_uc_model is not None:
            model.init_from_uc(init_from_uc_model)

    ref_mips = [m.to(device) for m in ref_mips]
    optimizer = _make_train_optimizer(model, mode, train_params)
    loss_fn = train_params.get('loss_fn', 'mse')
    model, train_loss = _run_training_loop(
        model, ref_mips, device, optimizer, loss_fn,
        train_params['total_iterations'], train_params['batch_res'],
        gamma=train_params.get('gamma', 1.0),
        loss_channels=train_params.get('loss_channels'),
        max_useful_lod=train_params.get('max_useful_lod'),
        save_path=save_path, save_interval=save_interval,
    )
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'model_state_dict': model.state_dict()}, save_path)
    return model, train_loss


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



# ============================================================
# 单材质评估
# ============================================================

def evaluate_one(name, mipmaps, model_params, bc_format_name, device, ckpt_dir,
                 dataset_root, bench_params, vis_dir=None):
    output_dim = mipmaps[0].shape[0]
    ref_h, ref_w = mipmaps[0].shape[1], mipmaps[0].shape[2]
    t0 = time.time()

    results = {'name': name, 'channels': output_dim, 'resolution': f'{ref_h}x{ref_w}'}

    psnr_bc_ref = compute_traditional_bc_psnr(mipmaps[0], bc_format=bc_format_name)
    results['psnr_bc_ref'] = round(psnr_bc_ref, 2)

    bc_path = _bc_ckpt_path(ckpt_dir, bc_format_name, name)
    if not os.path.exists(bc_path):
        raise FileNotFoundError(
            f"BC checkpoint not found: {bc_path}\n"
            f"Please run: python Tool.py --config <yaml> --train --ckpt {ckpt_dir}"
        )
    bc_train_params = get_bc_training_params(load_config(os.path.join(ckpt_dir, 'config.yaml'))) \
        if os.path.exists(os.path.join(ckpt_dir, 'config.yaml')) else {}
    eval_model_params = _with_bc_quant(model_params, bc_train_params)
    bc_model = _load_model('eval', bc_path, eval_model_params, output_dim, bc_format_name, device)

    mips_gpu = [m.to(device) for m in mipmaps]
    psnr_bc, _ = evaluate_full(bc_model, mips_gpu, device)
    results['psnr_bc'] = round(psnr_bc, 2)
    results['psnr_drop'] = round(psnr_bc_ref - psnr_bc, 2)

    inf_ms = benchmark_inference_ms(
        bc_model, ref_h, ref_w, device,
        warmup_iters=bench_params['warmup_iters'],
        timing_iters=bench_params['timing_iters'],
    )
    results['inference_ms'] = round(inf_ms, 3)

    results.update(bc_model.compute_compression_stats(
        dataset_root, name, mlp_param_bits=bench_params['mlp_param_bits']
    ))

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
                        bc_format_name, device, ckpt_dir,
                        init_from_uc_model=None, existing_model=None):
    """单材质单阶段训练. 返回 (result_dict, model). 调用方负责释放 model."""
    mipmaps = [m.cpu() for m in sample['mipmaps']]
    output_dim = sample['ref_tensor'].shape[0]
    model = None
    t0 = time.time()

    try:
        if mode == 'train-uc':
            save_path = None
            save_interval = None
        elif mode == 'train-bc-mlp':
            save_path = os.path.join(ckpt_dir, f'bc_{bc_format_name}_mlp', f'{name}.pth')
            save_interval = None
        else:  # train-bc
            save_path = _bc_ckpt_path(ckpt_dir, bc_format_name, name)
            save_interval = 10000

        model, train_loss = _train_and_save_model(
            mode, mipmaps, output_dim, model_params,
            bc_format_name, train_params, device, save_path,
            init_from_uc_model=init_from_uc_model,
            existing_model=existing_model,
            save_interval=save_interval,
        )
    finally:
        elapsed = time.time() - t0

    result = {
        'name': name,
        'train_loss': train_loss,
        'time_total': elapsed,
    }
    return result, model


def _run_pipeline_one_material(name, sample, model_params, uc_params, bc_params, mlp_params,
                               bc_format_name, device, ckpt_dir, prefix=''):
    """单材质 UC → BC → BC-MLP 三阶段串行, 模型在内存中传递."""
    uc_result, uc_model = _train_one_material(
        'train-uc', name, sample, model_params, uc_params,
        bc_format_name, device, ckpt_dir,
    )

    bc_result, bc_model = _train_one_material(
        'train-bc', name, sample, model_params, bc_params,
        bc_format_name, device, ckpt_dir,
        init_from_uc_model=uc_model,
    )
    del uc_model
    torch.cuda.empty_cache()

    mlp_result, mlp_model = _train_one_material(
        'train-bc-mlp', name, sample, model_params, mlp_params,
        bc_format_name, device, ckpt_dir,
        existing_model=bc_model,
    )
    del bc_model, mlp_model
    torch.cuda.empty_cache()

    total_t = uc_result['time_total'] + bc_result['time_total'] + mlp_result['time_total']
    print(f"{prefix}{name:<30s}  UC {uc_result['train_loss']:>6.4f}  "
          f"BC {bc_result['train_loss']:>6.4f}  MLP {mlp_result['train_loss']:>6.4f}  "
          f"{total_t:>6.1f}s",
          flush=True)
    return uc_result, bc_result, mlp_result


def _run_pipeline_materials(names, ds, model_params, uc_params, bc_params, mlp_params,
                            bc_format_name, device, ckpt_dir, prefix=''):
    uc_results, bc_results, mlp_results = [], [], []
    for i, name in enumerate(names):
        sample = ds.get_by_name(name)
        ur, br, mr = _run_pipeline_one_material(
            name, sample, model_params, uc_params, bc_params, mlp_params,
            bc_format_name, device, ckpt_dir, prefix=f"{prefix}[{i+1:>2d}/{len(names)}] ",
        )
        uc_results.append(ur)
        bc_results.append(br)
        mlp_results.append(mr)
    return uc_results, bc_results, mlp_results


# ============================================================
# 多进程 Worker
# ============================================================

def _make_dataset(dataset_params, target_res):
    """统一构造 MaterialDataset。target_res=None 表示加载原生 2K。"""
    return MaterialDataset(
        dataset_params['root'], target_res=target_res, preload=True,
        output_channels=dataset_params.get('output_channels', 'full'),
    )


def _mp_pipeline_worker(args):
    """多进程 worker: 分到的材质串行跑完整 UC→BC→BC-MLP 流水线."""
    gpu_id, names, config_path, ckpt_dir = args
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

    config = load_config(config_path)
    bc_format_name = config['bc_format']
    model_params = get_model_params(config)
    uc_params = get_uc_training_params(config)
    bc_params = get_bc_training_params(config)
    mlp_params = get_bc_mlp_training_params(config)
    dataset_params = get_dataset_params(config)

    ds = _make_dataset(dataset_params, dataset_params['target_res'])
    return _run_pipeline_materials(
        names, ds, model_params, uc_params, bc_params, mlp_params,
        bc_format_name, device, ckpt_dir, prefix=f"[{gpu_id}] ",
    )


def _run_mp_pipeline(gpu_ids, names_split, config_path, ckpt_dir, num_workers):
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    task_args = [
        (gpu_ids[i % len(gpu_ids)], names_split[i], config_path, ckpt_dir)
        for i in range(num_workers)
    ]
    with mp.Pool(num_workers) as pool:
        worker_results = pool.map(_mp_pipeline_worker, task_args)
    uc_all, bc_all, mlp_all = [], [], []
    for ur, br, mr in worker_results:
        uc_all.extend(ur)
        bc_all.extend(br)
        mlp_all.extend(mr)
    return uc_all, bc_all, mlp_all


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


# ============================================================
# 一键训练流水线: UC → BC → BC-MLP
# ============================================================

def pipeline_train(config_path, ckpt=None, materials=None, num_workers=1, gpus=None):
    config = load_config(config_path)
    bc_format_name = config['bc_format']
    model_params = get_model_params(config)
    uc_params = get_uc_training_params(config)
    bc_params = get_bc_training_params(config)
    mlp_params = get_bc_mlp_training_params(config)
    dataset_params = get_dataset_params(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = _make_dataset(dataset_params, dataset_params['target_res'])
    names = materials if materials else ds.material_names

    gpu_ids = gpus if gpus else [0]
    use_mp = num_workers > 1 and device == 'cuda'

    names_split = [[] for _ in range(num_workers)]
    for i, name in enumerate(names):
        names_split[i % num_workers].append(name)

    ckpt_dir = ckpt if ckpt is not None else os.path.join('checkpoints', _timestamp())
    os.makedirs(ckpt_dir, exist_ok=True)
    _write_config_yaml(ckpt_dir, config)

    print(f"Device: {device}  |  Config: {config_path}  |  BC: {bc_format_name.upper()}")
    print(f"Checkpoints → {ckpt_dir}/  |  workers={num_workers}, GPUs={gpu_ids}")
    print(f"Iters: UC={uc_params['total_iterations']}  "
          f"BC={bc_params['total_iterations']}  MLP={mlp_params['total_iterations']}")
    print(f"Loaded {len(ds)} materials. Each material: UC → BC → BC-MLP (in-memory handoff).\n")
    print('=' * 78)
    print(f"{'Material':<30s}  {'UC':>7s}  {'BC':>7s}  {'MLP':>7s}  {'Time':>6s}")
    print('=' * 78)

    if use_mp:
        uc_results, bc_results, mlp_results = _run_mp_pipeline(
            gpu_ids, names_split, config_path, ckpt_dir, num_workers,
        )
    else:
        uc_results, bc_results, mlp_results = _run_pipeline_materials(
            names, ds, model_params, uc_params, bc_params, mlp_params,
            bc_format_name, device, ckpt_dir,
        )

    _write_train_done('train-uc', ckpt_dir, config_path, bc_format_name, num_workers, uc_results)
    _write_train_done('train-bc', ckpt_dir, config_path, bc_format_name, num_workers, bc_results)
    _write_train_done('train-bc-mlp', ckpt_dir, config_path, bc_format_name, num_workers, mlp_results)

    print(f"\n{'='*60}")
    print(f"Pipeline complete.  {len(uc_results)} materials × (UC → BC → BC-MLP)")
    print(f"Next: python Tool.py --config {config_path} --ckpt {ckpt_dir}/")
    print(f"{'='*60}")
    return ckpt_dir


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='NTC: Neural Texture Compression')
    parser.add_argument('--config', type=str, required=True,
                        help='实验 yaml 配置文件路径 (必需)')
    parser.add_argument('--train', action='store_true',
                        help='一键流水线: UC → BC → BC-MLP')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint 目录 (eval 必需)')
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
    dataset_params = get_dataset_params(config)
    bench_params = get_benchmark_params(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds_root = dataset_params['root']

    ds = _make_dataset(dataset_params, dataset_params['target_res'])
    print(f"Device: {device}  |  Config: {args.config}  |  Res: native")
    print(f"BC format: {bc_format_name.upper()}  |  Filter: {model_params.get('filter')}  |  GT filter: bicubic")
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

    if args.train:
        pipeline_train(args.config, ckpt=args.ckpt, materials=names,
                       num_workers=num_workers, gpus=gpu_ids)
        return

    if not args.ckpt:
        print("ERROR: eval 模式需要 --ckpt <checkpoints/xxx/>")
        print("Usage:")
        print("  python Tool.py --config <yaml> --train")
        print("  python Tool.py --config <yaml> --ckpt <dir/>")
        sys.exit(1)

    ckpt_dir = args.ckpt
    _restore_model_params_from_ckpt(ckpt_dir, model_params, verbose=True)

    # Eval 在原生 2K 上进行（而非训练时的 target_res）。
    ds = _make_dataset(dataset_params, target_res=None)
    print(f"[eval] Reloaded dataset at full (2K) resolution")

    bc_params = get_bc_training_params(config)

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
