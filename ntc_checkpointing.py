import os
from datetime import datetime

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
    for k in ('feature_configs', 'hidden_dim', 'num_layers', 'half_pixel_offsets'):
        model_params[k] = ckpt_model_params[k]
    if verbose:
        print(f"Loaded model shape from {ckpt_yaml}")



