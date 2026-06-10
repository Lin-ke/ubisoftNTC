import torch
import torch.nn.functional as F
import os
import sys
import time
import argparse

from dataset import MaterialDataset
from ntc_model import make_model
from ntc_bc_model import make_bc_model
from ntc_train import sample_reference, evaluate_full, sample_lod_vaidyanathan
from ntc_utils import compute_traditional_bc_psnr
from ntc_batch_model import make_batched_uc_model, make_batched_bc_model, _lod_levels
from ntc_config import (
    load_config,
    get_model_params,
    get_uc_training_params,
    get_bc_training_params,
    get_bc_mlp_training_params,
    get_dataset_params,
    get_benchmark_params,
    get_batch_materials,
)
from ntc_visualization import visualize_comparison
from ntc_reporting import (
    print_header,
    print_result,
    summarize,
    save_tsv,
    _write_eval_done,
)
from ntc_checkpointing import (
    _bc_ckpt_path,
    _uc_ckpt_path,
    _restore_model_params_from_ckpt,
    _timestamp,
    _write_config_yaml,
    _write_config_yaml_safe,
    validate_config_compatible,
    _scan_resume_checkpoints,
    _load_checkpoint_meta,
)


# ============================================================
# UV + LOD 采样
# ============================================================

def _sample_uv_and_lod(ref_h, ref_w, batch_res, num_mips, device, max_useful_lod=None, uv_sampling='tile'):
    """采样 UV 网格 + Vaidyanathan LOD。

    Args:
        uv_sampling: 'tile'  -> 随机抠 batch_res x batch_res 的连续 tile
                     'uniform' -> 整张 grid 铺满 [0,1], 对齐论文 5.1 的 512 全域采样
    """
    H = W = batch_res
    if uv_sampling == 'tile':
        u0 = torch.rand(1, device=device) * (1.0 - W / ref_w)
        v0 = torch.rand(1, device=device) * (1.0 - H / ref_h)
        u = torch.linspace(u0.item(), u0.item() + W / ref_w, W, device=device)
        v = torch.linspace(v0.item(), v0.item() + H / ref_h, H, device=device)
    else:  # uniform
        u = torch.linspace(0, 1, W, device=device)
        v = torch.linspace(0, 1, H, device=device)
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
                       uv_sampling='tile',
                       save_path=None, save_interval=None,
                       tag='', log_interval=None,
                       start_iteration=0, stage='',
                       save_config=None):
    num_mips = len(ref_mips)
    ref_h, ref_w = ref_mips[0].shape[1], ref_mips[0].shape[2]
    gt_filter = 'bicubic'
    do_periodic_save = save_path is not None and save_interval is not None

    if log_interval is None:
        log_interval = max(1, total_iterations // 20)

    loss_total = 0.0
    for it in range(start_iteration, total_iterations):
        model.train()
        uv, scale = _sample_uv_and_lod(ref_h, ref_w, batch_res, num_mips, device,
                                       max_useful_lod, uv_sampling)
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
        if log_interval > 0 and step % log_interval == 0:
            avg_loss = loss_total / (step - start_iteration) if step > start_iteration else loss_total
            lr = optimizer.param_groups[0]['lr']
            tag_str = f"{tag} " if tag else ""
            print(f"{tag_str}[Iter {step:>7d}/{total_iterations}] "
                  f"loss={loss.item():.6f} avg_loss={avg_loss:.6f} lr={lr:.6e}",
                  flush=True)
        if do_periodic_save and step != total_iterations and step % save_interval == 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'stage': stage,
                'iteration': step,
                'config': save_config,
            }, save_path)
    return model, loss_total / max(1, total_iterations - start_iteration)


def _train_and_save_model(mode, ref_mips, output_dim, model_params, bc_format_name,
                          train_params, device, save_path,
                          init_from_uc_model=None, existing_model=None,
                          save_interval=None, tag='',
                          start_iteration=0, stage='',
                          save_config=None):
    """构建/复用模型并训练.

    - mode='train-uc': 从零构建.
    - 传 init_from_uc_model: 用 UC 模型初始化 BC 端点/索引/MLP.
    - 传 existing_model: 复用已有模型继续训练 (BC-MLP finetune / resume).
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
        uv_sampling=train_params.get('uv_sampling', 'tile'),
        save_path=save_path, save_interval=save_interval,
        tag=tag,
        start_iteration=start_iteration,
        stage=stage,
        save_config=save_config,
    )
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'stage': stage,
            'iteration': train_params['total_iterations'],
            'config': save_config,
        }, save_path)
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
                        init_from_uc_model=None, existing_model=None,
                        start_iteration=0, save_config=None):
    """单材质单阶段训练. 返回 (result_dict, model). 调用方负责释放 model."""
    mipmaps = [m.cpu() for m in sample['mipmaps']]
    output_dim = sample['ref_tensor'].shape[0]
    t0 = time.time()

    if mode == 'train-uc':
        save_path = _uc_ckpt_path(ckpt_dir, name)
        save_interval = None
    elif mode == 'train-bc-mlp':
        save_path = os.path.join(ckpt_dir, f'bc_{bc_format_name}_mlp', f'{name}.pth')
        save_interval = None
    else:  # train-bc
        save_path = _bc_ckpt_path(ckpt_dir, bc_format_name, name)
        save_interval = 10000

    stage_str = mode.replace('train-', '')
    tag = f"{name} {stage_str.upper()}"
    model, train_loss = _train_and_save_model(
        mode, mipmaps, output_dim, model_params,
        bc_format_name, train_params, device, save_path,
        init_from_uc_model=init_from_uc_model,
        existing_model=existing_model,
        save_interval=save_interval,
        tag=tag,
        start_iteration=start_iteration,
        stage=stage_str,
        save_config=save_config,
    )

    result = {
        'name': name,
        'train_loss': train_loss,
        'time_total': time.time() - t0,
    }
    return result, model


def _build_model_for_resume(mode, ckpt_meta, model_params, output_dim, bc_format_name, device,
                            train_params=None):
    """根据 checkpoint 元信息构建模型并加载权重（用于 resume）。"""
    build_params = _with_bc_quant(model_params, train_params or {}) if mode != 'train-uc' else model_params
    model = _make_train_model(mode, build_params, output_dim, bc_format_name, device)
    state = ckpt_meta['state_dict']
    model.load_state_dict(state)
    model.train()
    return model


def _run_pipeline_one_material(name, sample, model_params, uc_params, bc_params, mlp_params,
                               bc_format_name, device, ckpt_dir, prefix='',
                               resume_stage=None, resume_iter=0, save_config=None,
                               resume_model=None):
    """单材质 UC → BC → BC-MLP 三阶段串行, 模型在内存中传递.

    resume_stage: 若不为 None，从该阶段开始 resume（'uc'|'bc'|'bc-mlp'）。
    resume_iter:  已完成的迭代数。
    resume_model: 预加载的 checkpoint 模型（用于 resume）。
    """
    # UC stage
    if resume_stage is None or resume_stage == 'uc':
        uc_start = resume_iter if resume_stage == 'uc' else 0
        uc_result, uc_model = _train_one_material(
            'train-uc', name, sample, model_params, uc_params,
            bc_format_name, device, ckpt_dir,
            existing_model=resume_model if resume_stage == 'uc' else None,
            start_iteration=uc_start,
            save_config=save_config,
        )
        resume_stage = None  # UC 跑完后，后续阶段正常从头开始
    else:
        # resume_stage 是 bc 或 bc-mlp，说明 UC 已完成，不需要再训练
        uc_result = {'name': name, 'train_loss': 0.0, 'time_total': 0.0}
        uc_model = None

    # BC stage
    if resume_stage is None or resume_stage == 'bc':
        bc_start = resume_iter if resume_stage == 'bc' else 0
        bc_result, bc_model = _train_one_material(
            'train-bc', name, sample, model_params, bc_params,
            bc_format_name, device, ckpt_dir,
            init_from_uc_model=uc_model,
            existing_model=resume_model if resume_stage == 'bc' else None,
            start_iteration=bc_start,
            save_config=save_config,
        )
        if uc_model is not None:
            del uc_model
            torch.cuda.empty_cache()
        resume_stage = None
    else:
        # resume_stage 是 bc-mlp，说明 BC 已完成
        bc_result = {'name': name, 'train_loss': 0.0, 'time_total': 0.0}
        bc_model = None

    # BC-MLP stage
    if resume_stage is None or resume_stage == 'bc-mlp':
        mlp_start = resume_iter if resume_stage == 'bc-mlp' else 0
        mlp_result, mlp_model = _train_one_material(
            'train-bc-mlp', name, sample, model_params, mlp_params,
            bc_format_name, device, ckpt_dir,
            existing_model=resume_model if resume_stage == 'bc-mlp' else bc_model,
            start_iteration=mlp_start,
            save_config=save_config,
        )
        if bc_model is not None:
            del bc_model
            torch.cuda.empty_cache()
        del mlp_model
        torch.cuda.empty_cache()
    else:
        mlp_result = {'name': name, 'train_loss': 0.0, 'time_total': 0.0}

    total_t = uc_result['time_total'] + bc_result['time_total'] + mlp_result['time_total']
    print(f"{prefix}{name:<30s}  UC {uc_result['train_loss']:>6.4f}  "
          f"BC {bc_result['train_loss']:>6.4f}  MLP {mlp_result['train_loss']:>6.4f}  "
          f"{total_t:>6.1f}s",
          flush=True)
    return uc_result, bc_result, mlp_result


def _run_pipeline_materials(names, ds, model_params, uc_params, bc_params, mlp_params,
                            bc_format_name, device, ckpt_dir, prefix='',
                            resume_map=None, save_config=None):
    """跑多材质流水线。若 resume_map 提供，按每个 material 的 stage/iter 恢复。"""
    uc_results, bc_results, mlp_results = [], [], []
    for i, name in enumerate(names):
        sample = ds.get_by_name(name)
        rstage, riter = None, 0
        resume_model = None
        if resume_map and name in resume_map:
            rstage, riter = resume_map[name]['stage'], resume_map[name]['iteration']
            # 加载 checkpoint 权重到模型
            ckpt_meta = _load_checkpoint_meta(resume_map[name]['path'])
            output_dim = sample['ref_tensor'].shape[0]
            mode = f"train-{rstage}"
            tp = uc_params if rstage == 'uc' else (bc_params if rstage == 'bc' else mlp_params)
            resume_model = _build_model_for_resume(
                mode, ckpt_meta, model_params, output_dim, bc_format_name, device,
                train_params=tp,
            )
        ur, br, mr = _run_pipeline_one_material(
            name, sample, model_params, uc_params, bc_params, mlp_params,
            bc_format_name, device, ckpt_dir, prefix=f"{prefix}[{i+1:>2d}/{len(names)}] ",
            resume_stage=rstage, resume_iter=riter,
            save_config=save_config,
            resume_model=resume_model,
        )
        uc_results.append(ur)
        bc_results.append(br)
        mlp_results.append(mr)
    return uc_results, bc_results, mlp_results


# ============================================================
# Batched (ensemble) 训练: 多材质堆叠到 leading 维度 M
# ============================================================

def _stack_gt_mips(samples, device):
    """把 M 个材质的 mipmap 金字塔按层堆叠.

    Args:
        samples: list of sample dict, 每个含 'mipmaps' (list of [C,res,res]).
    Returns:
        list of [M, C, res_k, res_k], 已搬到 device.
        要求所有材质 mip 层数与各层分辨率一致 (native 2K 满足).
    """
    num_mips = len(samples[0]['mipmaps'])
    for s in samples:
        if len(s['mipmaps']) != num_mips:
            raise ValueError("batched 训练要求所有材质 mip 层数一致")
    stacked = []
    for k in range(num_mips):
        layer = torch.stack([s['mipmaps'][k] for s in samples], dim=0)  # [M,C,res,res]
        stacked.append(layer.to(device))
    return stacked


def _sample_batched_uv(M, ref_h, ref_w, batch_res, device, uv_sampling='tile'):
    """采样 M 个独立 UV tile, 返回 [M, H, W, 2]. (LOD 由调用方共享采样)"""
    H = W = batch_res
    tu = torch.linspace(0, 1, W, device=device)
    tv = torch.linspace(0, 1, H, device=device)
    if uv_sampling == 'tile':
        u0 = torch.rand(M, device=device) * (1.0 - W / ref_w)
        v0 = torch.rand(M, device=device) * (1.0 - H / ref_h)
        u = u0[:, None] + tu[None, :] * (W / ref_w)   # [M,W]
        v = v0[:, None] + tv[None, :] * (H / ref_h)   # [M,H]
    else:  # uniform: 全域铺满, 所有材质相同
        u = tu[None, :].expand(M, W)
        v = tv[None, :].expand(M, H)
    ug = u[:, None, :].expand(M, H, W)                # ug[m,i,j] = u[m,j]
    vg = v[:, :, None].expand(M, H, W)                # vg[m,i,j] = v[m,i]
    return torch.stack([ug, vg], dim=-1)              # [M,H,W,2]


def _batched_sample_reference(gt_mips, uv, scale):
    """共享 LOD 下批量参考采样 (bicubic 空间 + 双 mip 线性混合).

    gt_mips: list of [M,C,res,res]; uv: [M,H,W,2]; scale: 共享标量.
    """
    s0, s1, lam = _lod_levels(scale, len(gt_mips))
    uv_grid = uv * 2 - 1
    r0 = F.grid_sample(gt_mips[s0], uv_grid, mode='bicubic',
                       padding_mode='border', align_corners=False)
    if s1 == s0 or lam == 0.0:
        return r0
    r1 = F.grid_sample(gt_mips[s1], uv_grid, mode='bicubic',
                       padding_mode='border', align_corners=False)
    return (1 - lam) * r0 + lam * r1


def _make_batched_optimizer(model, mode, train_params):
    feat_params = list(model.feature_grids.parameters())
    mlp_params = list(model.mlp.parameters())
    betas = tuple(train_params.get('betas', [0.9, 0.999]))
    if mode == 'train-bc-mlp':
        for p in feat_params:
            p.requires_grad = False
        return torch.optim.Adam(mlp_params, lr=train_params['lr_mlp'], betas=betas)
    return torch.optim.Adam([
        {'params': feat_params, 'lr': train_params['lr_feat']},
        {'params': mlp_params, 'lr': train_params['lr_mlp']},
    ], betas=betas)


def _run_batched_training_loop(model, gt_mips, M, device, optimizer, loss_fn,
                               total_iterations, batch_res, gamma=1.0,
                               loss_channels=None, max_useful_lod=None,
                               uv_sampling='tile', tag='', log_interval=None):
    """batched 训练循环. 每 iteration 共享 LOD, 各材质独立 UV tile.

    loss = Σ_m mean(per-material tile loss), 使每个材质的梯度与单独训练时一致
    (各材质参数互相独立, 无共享).
    """
    num_mips = len(gt_mips)
    ref_h, ref_w = gt_mips[0].shape[2], gt_mips[0].shape[3]
    if log_interval is None:
        log_interval = max(1, total_iterations // 20)

    loss_total = 0.0
    for it in range(total_iterations):
        model.train()
        scale = sample_lod_vaidyanathan(num_mips, device, max_useful_lod=max_useful_lod)
        uv = _sample_batched_uv(M, ref_h, ref_w, batch_res, device, uv_sampling)
        with torch.no_grad():
            ref = _batched_sample_reference(gt_mips, uv, scale)   # [M,C,H,W]
        pred = model(uv, scale)                                   # [M,C,H,W]
        if loss_channels is not None:
            pred = pred[:, loss_channels, :, :]
            ref = ref[:, loss_channels, :, :]
        if loss_fn == 'l1':
            per_el = F.l1_loss(pred, ref, reduction='none')
        else:
            per_el = F.mse_loss(pred, ref, reduction='none')
        per_mat = per_el.mean(dim=[1, 2, 3])                      # [M]
        loss = per_mat.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += per_mat.mean().item()
        if gamma < 1.0:
            for pg in optimizer.param_groups:
                pg['lr'] *= gamma
        step = it + 1
        if log_interval > 0 and step % log_interval == 0:
            avg_loss = loss_total / step
            lr = optimizer.param_groups[0]['lr']
            tag_str = f"{tag} " if tag else ""
            print(f"{tag_str}[Iter {step:>7d}/{total_iterations}] "
                  f"loss={per_mat.mean().item():.6f} avg_loss={avg_loss:.6f} lr={lr:.6e}",
                  flush=True)
    return model, loss_total / max(1, total_iterations)


def _save_batched_checkpoints(model, names, save_path_fn, stage, iteration, save_config):
    """把 batched 模型拆成 M 个标准单材质 checkpoint 并落盘."""
    sds = model.export_per_material_state_dicts()
    for m, name in enumerate(names):
        path = save_path_fn(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cpu_sd = {k: v.cpu() for k, v in sds[m].items()}
        torch.save({
            'model_state_dict': cpu_sd,
            'stage': stage,
            'iteration': iteration,
            'config': save_config,
        }, path)


def _run_batched_pipeline(names, ds, model_params, uc_params, bc_params, mlp_params,
                          bc_format_name, device, ckpt_dir, batch_size=0,
                          save_config=None):
    """batched 三阶段流水线: 按 batch_size 分组, 每组 UC → BC → BC-MLP.

    batch_size=0 表示所有材质一组 (M=全部).
    """
    if model_params.get('encoding', 'pyramid') != 'pyramid':
        raise ValueError("batched 训练仅支持 encoding='pyramid'; hash_grid 走逐材质串行路径 (非 batched)")

    # 先按 GT 参考张量分辨率分桶: batched 要求同组材质 mip 形状一致 (grid_sample 限制).
    # 数据集里大多 2048x2048, 但存在 4096x2048 / 2083x2048 等异类, 必须分开.
    buckets = {}
    bucket_order = []
    for n in names:
        shp = tuple(ds.get_by_name(n)['ref_tensor'].shape)
        if shp not in buckets:
            buckets[shp] = []
            bucket_order.append(shp)
        buckets[shp].append(n)

    if len(bucket_order) > 1:
        print(f"[batched] 检测到 {len(bucket_order)} 种分辨率, 按分辨率分桶训练:")
        for shp in bucket_order:
            print(f"          {shp}: {len(buckets[shp])} 材质")

    groups = []
    for shp in bucket_order:
        bnames = buckets[shp]
        if batch_size and batch_size > 0:
            for i in range(0, len(bnames), batch_size):
                groups.append(bnames[i:i + batch_size])
        else:
            groups.append(bnames)

    uc_results, bc_results, mlp_results = [], [], []

    for gi, group in enumerate(groups):
        M = len(group)
        samples = [ds.get_by_name(n) for n in group]
        output_dim = samples[0]['ref_tensor'].shape[0]
        gt_mips = _stack_gt_mips(samples, device)
        gprefix = f"[grp {gi+1}/{len(groups)} M={M}] "
        print(f"{gprefix}materials: {', '.join(group)}", flush=True)

        # ---- UC ----
        t0 = time.time()
        uc_model = make_batched_uc_model(M, model_params, output_dim=output_dim).to(device)
        opt = _make_batched_optimizer(uc_model, 'train-uc', uc_params)
        uc_model, uc_loss = _run_batched_training_loop(
            uc_model, gt_mips, M, device, opt, uc_params.get('loss_fn', 'l1'),
            uc_params['total_iterations'], uc_params['batch_res'],
            gamma=uc_params.get('gamma', 1.0),
            loss_channels=uc_params.get('loss_channels'),
            max_useful_lod=uc_params.get('max_useful_lod'),
            uv_sampling=uc_params.get('uv_sampling', 'tile'),
            tag=f"{gprefix}UC",
        )
        _save_batched_checkpoints(
            uc_model, group, lambda n: _uc_ckpt_path(ckpt_dir, n),
            'uc', uc_params['total_iterations'], save_config)
        uc_time = time.time() - t0

        # ---- BC (QAT), 从 UC 初始化 ----
        t0 = time.time()
        bc_model = make_batched_bc_model(M, model_params, output_dim=output_dim,
                                         bc_format_name=bc_format_name).to(device)
        bc_model.init_from_uc_batched(uc_model)
        del uc_model
        torch.cuda.empty_cache()
        opt = _make_batched_optimizer(bc_model, 'train-bc', bc_params)
        bc_save_interval = 10000
        bc_total = bc_params['total_iterations']
        # 周期性保存 (crash 安全 + 兼容非 batched resume 扫描)
        done = 0
        bc_loss = 0.0
        while done < bc_total:
            chunk = min(bc_save_interval, bc_total - done)
            bc_model, bc_loss = _run_batched_training_loop(
                bc_model, gt_mips, M, device, opt, bc_params.get('loss_fn', 'l1'),
                chunk, bc_params['batch_res'],
                gamma=bc_params.get('gamma', 1.0),
                loss_channels=bc_params.get('loss_channels'),
                max_useful_lod=bc_params.get('max_useful_lod'),
                uv_sampling=bc_params.get('uv_sampling', 'tile'),
                tag=f"{gprefix}BC",
                log_interval=max(1, chunk // 5),
            )
            done += chunk
            _save_batched_checkpoints(
                bc_model, group,
                lambda n: _bc_ckpt_path(ckpt_dir, bc_format_name, n),
                'bc', done, save_config)
        bc_time = time.time() - t0

        # ---- BC-MLP finetune (冻结特征) ----
        t0 = time.time()
        opt = _make_batched_optimizer(bc_model, 'train-bc-mlp', mlp_params)
        bc_model, mlp_loss = _run_batched_training_loop(
            bc_model, gt_mips, M, device, opt, mlp_params.get('loss_fn', 'l1'),
            mlp_params['total_iterations'], mlp_params['batch_res'],
            gamma=mlp_params.get('gamma', 1.0),
            loss_channels=mlp_params.get('loss_channels'),
            max_useful_lod=mlp_params.get('max_useful_lod'),
            uv_sampling=mlp_params.get('uv_sampling', 'tile'),
            tag=f"{gprefix}MLP",
        )
        _save_batched_checkpoints(
            bc_model, group,
            lambda n: os.path.join(ckpt_dir, f'bc_{bc_format_name}_mlp', f'{n}.pth'),
            'bc-mlp', mlp_params['total_iterations'], save_config)
        mlp_time = time.time() - t0
        del bc_model
        for t in gt_mips:
            del t
        torch.cuda.empty_cache()

        total_t = uc_time + bc_time + mlp_time
        print(f"{gprefix}done  UC {uc_loss:.4f} ({uc_time:.1f}s)  "
              f"BC {bc_loss:.4f} ({bc_time:.1f}s)  MLP {mlp_loss:.4f} ({mlp_time:.1f}s)  "
              f"total {total_t:.1f}s", flush=True)

        for n in group:
            uc_results.append({'name': n, 'train_loss': uc_loss, 'time_total': uc_time / M})
            bc_results.append({'name': n, 'train_loss': bc_loss, 'time_total': bc_time / M})
            mlp_results.append({'name': n, 'train_loss': mlp_loss, 'time_total': mlp_time / M})

    return uc_results, bc_results, mlp_results


# ============================================================
# 数据集构造
# ============================================================

def _make_dataset(dataset_params):
    """统一构造 MaterialDataset, 始终使用原生 2K 分辨率."""
    return MaterialDataset(
        dataset_params['root'], target_res=None, preload=True,
        output_channels=dataset_params.get('output_channels', 'full'),
        srgb_decode=dataset_params.get('srgb_decode', True),
    )


# ============================================================
# 一键训练流水线: UC → BC → BC-MLP
# ============================================================

def pipeline_train(config_path, ckpt=None, materials=None, resume=False):
    config = load_config(config_path)
    bc_format_name = config['bc_format']
    model_params = get_model_params(config)
    uc_params = get_uc_training_params(config)
    bc_params = get_bc_training_params(config)
    mlp_params = get_bc_mlp_training_params(config)
    dataset_params = get_dataset_params(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = _make_dataset(dataset_params)
    names = materials if materials else ds.material_names

    batch_size = get_batch_materials(config)
    # batched (ensemble) 训练: 仅 pyramid + 非 resume 时启用 (默认路径).
    use_batched = (not resume) and model_params.get('encoding', 'pyramid') == 'pyramid'

    resume_map = None
    ckpt_config = None

    if resume:
        if ckpt is None:
            raise ValueError("--resume 必须与 --ckpt <dir> 同时使用")
        ckpt_dir = ckpt
        if not os.path.exists(ckpt_dir):
            raise FileNotFoundError(f"Resume 指定的 ckpt 目录不存在: {ckpt_dir}")

        # 扫描每个 material 可 resume 的状态
        scan = _scan_resume_checkpoints(ckpt_dir, bc_format_name, names)
        resume_map = {}
        for name in names:
            stage, path, iteration = scan[name]
            if stage is None:
                raise FileNotFoundError(
                    f"Resume 失败: material '{name}' 在 {ckpt_dir} 中找不到任何 checkpoint"
                )
            resume_map[name] = {'stage': stage, 'path': path, 'iteration': iteration}

        # 加载第一个 checkpoint 的 config 作为 ckpt_config
        first_meta = _load_checkpoint_meta(resume_map[names[0]]['path'])
        ckpt_config = first_meta.get('config')
        if ckpt_config is None:
            # 旧格式 ckpt 没有内嵌 config，尝试从 config.yaml 读取
            yaml_path = os.path.join(ckpt_dir, 'config.yaml')
            if os.path.exists(yaml_path):
                ckpt_config = load_config(yaml_path)
            else:
                raise ValueError(f"Resume 失败: checkpoint 中无内嵌 config，且 {yaml_path} 不存在")

        # 校验模型结构一致性
        ok, diff_msg = validate_config_compatible(ckpt_config, config)
        if not ok:
            print(f"ERROR: {diff_msg}")
            raise ValueError("Resume 时新 config 与 checkpoint 的模型结构不兼容")

        # 用 ckpt config 恢复模型结构参数，保留训练超参
        model_params = get_model_params(ckpt_config)

        # 保存 resume 配置，不覆盖原 config.yaml
        _write_config_yaml_safe(ckpt_dir, config, suffix='resume')

        print(f"Resume mode: checkpoints from {ckpt_dir}")
        print(f"Loaded {len(resume_map)} materials resume info.")
    else:
        ckpt_dir = ckpt if ckpt is not None else os.path.join('checkpoints', _timestamp())
        os.makedirs(ckpt_dir, exist_ok=True)
        _write_config_yaml(ckpt_dir, config)

    print(f"Device: {device}  |  Config: {config_path}  |  BC: {bc_format_name.upper()}")
    print(f"Checkpoints → {ckpt_dir}/")
    print(f"Iters: UC={uc_params['total_iterations']}  "
          f"BC={bc_params['total_iterations']}  MLP={mlp_params['total_iterations']}")
    print(f"UV: {uc_params.get('uv_sampling','tile')}  batch={uc_params['batch_res']}")
    print(f"Loaded {len(ds)} materials. Each material: UC → BC → BC-MLP (in-memory handoff).\n")
    print('=' * 78)
    print(f"{'Material':<30s}  {'UC':>7s}  {'BC':>7s}  {'MLP':>7s}  {'Time':>6s}")
    print('=' * 78)

    save_config = ckpt_config if ckpt_config is not None else config

    if use_batched:
        grp_desc = 'all-in-one' if not batch_size else f'groups of {batch_size}'
        print(f"[batched] ensemble 训练启用 (M={len(names)}, {grp_desc}). "
              f"单进程单流, 多材质堆叠为一个大 kernel.\n")
        uc_results, bc_results, mlp_results = _run_batched_pipeline(
            names, ds, model_params, uc_params, bc_params, mlp_params,
            bc_format_name, device, ckpt_dir, batch_size=batch_size,
            save_config=save_config,
        )
    else:
        uc_results, bc_results, mlp_results = _run_pipeline_materials(
            names, ds, model_params, uc_params, bc_params, mlp_params,
            bc_format_name, device, ckpt_dir,
            resume_map=resume_map, save_config=save_config,
        )

    # done.json 由后续 eval 阶段统一写入, 避免重复覆盖.

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
    parser.add_argument('--resume', action='store_true',
                        help='从已有 ckpt 恢复训练 (需配合 --ckpt 和 --train)')
    args = parser.parse_args()

    config = load_config(args.config)
    bc_format_name = config['bc_format']
    model_params = get_model_params(config)
    dataset_params = get_dataset_params(config)
    bench_params = get_benchmark_params(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds_root = dataset_params['root']

    ds = _make_dataset(dataset_params)
    print(f"Device: {device}  |  Config: {args.config}  |  Res: native")
    print(f"BC format: {bc_format_name.upper()}  |  Filter: {model_params.get('filter')}  |  GT filter: bicubic")
    print(f"Loaded {len(ds)} materials from {ds_root}\n")

    names = ([n.strip() for n in args.materials.split(',')]
             if args.materials else ds.material_names)

    if args.train:
        ckpt_dir = pipeline_train(args.config, ckpt=args.ckpt, materials=names,
                                  resume=args.resume)
    else:
        ckpt_dir = args.ckpt

    if not ckpt_dir:
        print("ERROR: eval 模式需要 --ckpt <checkpoints/xxx/>")
        print("Usage:")
        print("  python Tool.py --config <yaml> --train")
        print("  python Tool.py --config <yaml> --ckpt <dir/>")
        sys.exit(1)

    _restore_model_params_from_ckpt(ckpt_dir, model_params, verbose=True)

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
        print("\nBest  3 (lowest BC drop):")
        for r in sorted_by_drop[:3]:
            print(f"  {r['name']:<30s} drop={r['psnr_drop']:.2f} dB")
        print("\nWorst 3 (highest BC drop):")
        for r in sorted_by_drop[-3:]:
            print(f"  {r['name']:<30s} drop={r['psnr_drop']:.2f} dB")

    _write_eval_done('eval', ckpt_dir, args.config, bc_format_name, all_results)


if __name__ == '__main__':
    main()
