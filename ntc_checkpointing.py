import os
from datetime import datetime

import yaml

from ntc_config import load_config, get_model_params


def _timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H%M%S')


def _bc_ckpt_path(ckpt_dir, bc_format_name, material_name):
    return os.path.join(ckpt_dir, f'bc_{bc_format_name}', f'{material_name}.pth')


# UC PSNR 缓存：BC 实验里 UC ckpt 只读、psnr_uc 是常量，
# 没必要每次 eval 都重新加载 UC 模型推理一遍。
# 训完 UC 时写一次 <ckpt>/uc_psnr.tsv，BC eval 直接读。
_UC_PSNR_CACHE = {}  # 进程内 RAM cache：{ckpt_dir: {name: psnr}}


def _uc_psnr_tsv_path(ckpt_dir):
    return os.path.join(ckpt_dir, 'uc_psnr.tsv')


def _load_uc_psnr_table(ckpt_dir):
    if ckpt_dir in _UC_PSNR_CACHE:
        return _UC_PSNR_CACHE[ckpt_dir]
    path = _uc_psnr_tsv_path(ckpt_dir)
    table = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        table[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
    _UC_PSNR_CACHE[ckpt_dir] = table
    return table


def _append_uc_psnr(ckpt_dir, name, psnr):
    """追加一行到 uc_psnr.tsv 并更新 RAM cache."""
    table = _load_uc_psnr_table(ckpt_dir)
    table[name] = psnr
    path = _uc_psnr_tsv_path(ckpt_dir)
    write_header = not os.path.exists(path)
    with open(path, 'a', encoding='utf-8') as f:
        if write_header:
            f.write('name\tpsnr_uc\n')
        f.write(f'{name}\t{psnr:.4f}\n')


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
    for k in ('feature_configs', 'hidden_dim', 'num_layers', 'half_pixel_offsets'):
        model_params[k] = ckpt_model_params[k]
    if verbose:
        print(f"Loaded model shape from {ckpt_yaml}")


def _prepare_train_ckpt(mode, args, config, model_params):
    if args.ckpt:
        ckpt_dir = args.ckpt
        if mode == 'train-bc':
            _restore_model_params_from_ckpt(ckpt_dir, model_params, verbose=True)
        else:
            os.makedirs(ckpt_dir, exist_ok=True)
            _write_config_yaml(ckpt_dir, config)
        return ckpt_dir

    ckpt_dir = os.path.join('checkpoints', _timestamp())
    os.makedirs(ckpt_dir, exist_ok=True)
    _write_config_yaml(ckpt_dir, config)
    return ckpt_dir
