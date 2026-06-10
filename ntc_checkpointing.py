import os
from datetime import datetime

import torch
import yaml

from ntc_config import load_config, get_model_params


def _timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H%M%S')


def _bc_ckpt_path(ckpt_dir, bc_format_name, material_name):
    return os.path.join(ckpt_dir, f'bc_{bc_format_name}', f'{material_name}.pth')


def _uc_ckpt_path(ckpt_dir, material_name):
    return os.path.join(ckpt_dir, 'uc', f'{material_name}.pth')


def _write_config_yaml(ckpt_dir, config):
    with open(os.path.join(ckpt_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def _restore_model_params_from_ckpt(ckpt_dir, model_params, verbose=False):
    ckpt_yaml = os.path.join(ckpt_dir, 'config.yaml')
    if not os.path.exists(ckpt_yaml):
        return

    try:
        saved = load_config(ckpt_yaml)
    except ValueError as e:
        with open(ckpt_yaml, 'r', encoding='utf-8') as f:
            saved = yaml.safe_load(f)
        if verbose:
            print(f"Config validation warning (ignored): {e}")

    ckpt_model_params = get_model_params(saved)
    for k in ('encoding', 'feature_configs', 'hash_grid', 'hidden_dim', 'num_layers',
              'half_pixel_offsets', 'activation', 'output_activation'):
        if k in ckpt_model_params:
            model_params[k] = ckpt_model_params[k]
    if verbose:
        print(f"Loaded model shape from {ckpt_yaml}")


# ============================================================
# Resume 相关工具
# ============================================================

_MODEL_SHAPE_KEYS = ('encoding', 'feature_configs', 'hash_grid', 'hidden_dim', 'num_layers',
                     'half_pixel_offsets', 'activation', 'output_activation', 'filter')


def _write_config_yaml_safe(ckpt_dir, config, suffix='resume'):
    """Resume 时不覆盖原 config.yaml，写入 config_resume.yaml。"""
    filename = f'config_{suffix}.yaml' if suffix else 'config.yaml'
    path = os.path.join(ckpt_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return path


def validate_config_compatible(ckpt_config, new_config):
    """校验新 config 的模型结构是否与 ckpt 兼容。返回 (ok, diff_msg)。"""
    ckpt_model = get_model_params(ckpt_config)
    new_model = get_model_params(new_config)
    diffs = []
    for k in _MODEL_SHAPE_KEYS:
        v1 = ckpt_model.get(k)
        v2 = new_model.get(k)
        if v1 != v2:
            diffs.append(f"  {k}: ckpt={v1!r} != new={v2!r}")
    # bc_format 也属于结构参数（影响 BC 格式）
    if ckpt_config.get('bc_format') != new_config.get('bc_format'):
        diffs.append(f"  bc_format: ckpt={ckpt_config.get('bc_format')!r} != new={new_config.get('bc_format')!r}")
    if diffs:
        return False, "模型结构参数不一致:\n" + "\n".join(diffs)
    return True, ""


def _scan_resume_checkpoints(ckpt_dir, bc_format_name, material_names):
    """扫描 ckpt 目录，返回每个 material 可 resume 的 (stage, path, iteration)。

    按优先级 bc-mlp > bc > uc 取最新阶段。
    若找不到任何 checkpoint，返回 (None, None, 0)。
    """
    results = {}
    for name in material_names:
        candidates = []
        # BC-MLP
        mlp_path = os.path.join(ckpt_dir, f'bc_{bc_format_name}_mlp', f'{name}.pth')
        if os.path.exists(mlp_path):
            candidates.append(('bc-mlp', mlp_path))
        # BC
        bc_path = _bc_ckpt_path(ckpt_dir, bc_format_name, name)
        if os.path.exists(bc_path):
            candidates.append(('bc', bc_path))
        # UC
        uc_path = _uc_ckpt_path(ckpt_dir, name)
        if os.path.exists(uc_path):
            candidates.append(('uc', uc_path))

        if not candidates:
            results[name] = (None, None, 0)
            continue

        # 默认取最高优先级（列表顺序即优先级）
        stage, path = candidates[0]
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            iteration = ckpt.get('iteration', 0)
            # 如果 ckpt 里有 stage 字段，以 ckpt 里记录的为准
            stage = ckpt.get('stage', stage)
        except Exception:
            iteration = 0
        results[name] = (stage, path, iteration)
    return results


def _load_checkpoint_meta(path):
    """加载 checkpoint 的元信息（不加载模型权重到 GPU）。"""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    return {
        'stage': ckpt.get('stage'),
        'iteration': ckpt.get('iteration', 0),
        'config': ckpt.get('config'),
        'state_dict': ckpt.get('model_state_dict', ckpt),
    }
