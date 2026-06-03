"""YAML 配置加载器 —— 全工程唯一配置入口.

新 schema (yaml-driven, 单一事实源):

    bc_format: bc6                # bc1..bc6
    loss: l1                      # l1 | mse
    loss_config: {}               # 可选: 通道权重等

    dataset:
      root: dataset
      target_res: 256

    model:
      feature_configs: [[512,8,3], [256,7,3], [128,6,3], [64,5,3]]
      hidden_dim: 16
      num_layers: 1
      filter: trilinear           # trilinear | tricubic
      half_pixel_offsets: [1, 3]  # 在哪些 feature grid 索引上施加半像素偏移

    uc_training:                  # 训练无约束基线模型 (只在 --train-uc 时使用)
      total_iterations: 10000
      batch_res: 128
      lr_feat: 5.0e-2
      lr_mlp: 1.0e-3
      gamma: 0.9995

    bc_training:                  # BC QAT 阶段
      total_iterations: 10000
      batch_res: 128
      lr_feat: 1.0e-2
      lr_mlp: 1.0e-3
      betas: [0.9, 0.999]

    benchmark:                    # 推理 benchmark (可选, 有缺省值)
      warmup_iters: 5
      timing_iters: 20
      mlp_param_bits: 16          # 假设 MLP 部署精度 (FP16 默认)
"""

import yaml
from typing import Dict, Any


# ---------- 加载/校验 ----------

def load_config(path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件并校验 schema."""
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    __validate(config)
    return config


def __validate(config: Dict[str, Any]):
    """校验 schema 完整性. 兼容新 (bc_training) / 老 (training+optimizer) 两种."""
    required = ['bc_format', 'model']
    for key in required:
        if key not in config:
            raise ValueError(f"缺少必需配置项: '{key}'")

    bc_format = config['bc_format']
    supported = ('bc1', 'bc2', 'bc3', 'bc4', 'bc5', 'bc6')
    if bc_format not in supported:
        raise ValueError(f"不支持的 BC 格式: '{bc_format}', 支持: {', '.join(supported)}")

    model = config['model']
    if 'feature_configs' not in model:
        raise ValueError("model 缺少 'feature_configs'")
    if 'hidden_dim' not in model:
        raise ValueError("model 缺少 'hidden_dim'")

    if 'bc_training' in config:
        bct = config['bc_training']
        for key in ('total_iterations', 'batch_res'):
            if key not in bct:
                raise ValueError(f"bc_training 缺少 '{key}'")
    elif 'training' in config:
        for key in ('total_iterations', 'batch_res'):
            if key not in config['training']:
                raise ValueError(f"training 缺少 '{key}' (legacy schema)")
    else:
        raise ValueError("缺少必需配置项: 'bc_training' (推荐) 或 'training' (legacy)")

    if 'uc_training' in config:
        for key in ('total_iterations', 'batch_res'):
            if key not in config['uc_training']:
                raise ValueError(f"uc_training 缺少 '{key}'")

    loss_fn = config.get('loss', 'l1')
    if loss_fn not in ('l1', 'mse'):
        raise ValueError(f"loss 仅支持 'l1' | 'mse', 收到: '{loss_fn}'")


# ---------- 提取扁平参数字典 ----------

def get_model_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """提取模型参数 (feature grids + MLP 形状)."""
    m = config['model']
    return {
        'feature_configs': [tuple(fc) for fc in m['feature_configs']],
        'hidden_dim': m['hidden_dim'],
        'num_layers': m.get('num_layers', 1),
        'filter': m.get('filter', 'trilinear'),
        'half_pixel_offsets': m.get('half_pixel_offsets', []),
    }


def get_uc_training_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """提取 UC 训练超参."""
    if 'uc_training' not in config:
        raise ValueError("配置缺少 'uc_training' 段, 无法运行 --train-uc")
    t = config['uc_training']
    return {
        'total_iterations': t['total_iterations'],
        'batch_res': t['batch_res'],
        'lr_feat': t.get('lr_feat', 5.0e-2),
        'lr_mlp': t.get('lr_mlp', 1.0e-3),
        'gamma': t.get('gamma', 0.9995),
        'log_interval': t.get('log_interval', 1000),
    }


def get_bc_training_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """提取 BC QAT 训练超参. 兼容老 schema (training + optimizer)."""
    if 'bc_training' in config:
        t = config['bc_training']
        return {
            'total_iterations': t['total_iterations'],
            'batch_res': t['batch_res'],
            'lr_feat': t.get('lr_feat', 1.0e-2),
            'lr_mlp': t.get('lr_mlp', 1.0e-3),
            'betas': t.get('betas', [0.9, 0.999]),
            'log_interval': t.get('log_interval', 1000),
            'output_dir': t.get('output_dir', 'output_bc'),
            'loss_fn': config.get('loss', 'l1'),
            'loss_config': config.get('loss_config', {}),
        }
    # legacy fallback
    t = config.get('training', {})
    opt = config.get('optimizer', {})
    return {
        'total_iterations': t['total_iterations'],
        'batch_res': t['batch_res'],
        'lr_feat': opt.get('lr_feat', 1.0e-2),
        'lr_mlp': opt.get('lr_mlp', 1.0e-3),
        'betas': opt.get('betas', [0.9, 0.999]),
        'log_interval': t.get('log_interval', 1000),
        'output_dir': t.get('output_dir', 'output_bc'),
        'loss_fn': config.get('loss', 'l1'),
        'loss_config': config.get('loss_config', {}),
    }


def get_dataset_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """提取数据集配置."""
    d = config.get('dataset', {})
    return {
        'root': d.get('root', 'dataset'),
        'target_res': d.get('target_res', 256),
    }


def get_benchmark_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """提取推理 benchmark 配置."""
    b = config.get('benchmark', {})
    return {
        'warmup_iters': b.get('warmup_iters', 5),
        'timing_iters': b.get('timing_iters', 20),
        'mlp_param_bits': b.get('mlp_param_bits', 16),
    }


# ---------- 向后兼容 (ntc_bc_train.py / ntc_compare.py 老路径) ----------

def get_training_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """老接口: 默认走 bc_training. 兼容 'training' / 'optimizer' 老 schema."""
    if 'bc_training' in config:
        params = get_bc_training_params(config)
        params['bc_format'] = config.get('bc_format', 'bc6')
        return params
    # legacy
    t = config.get('training', {})
    opt = config.get('optimizer', {})
    return {
        'total_iterations': t.get('total_iterations', 10000),
        'batch_res': t.get('batch_res', 128),
        'log_interval': t.get('log_interval', 1000),
        'output_dir': t.get('output_dir', 'output_bc'),
        'lr_feat': opt.get('lr_feat', 1e-2),
        'lr_mlp': opt.get('lr_mlp', 1e-3),
        'betas': opt.get('betas', [0.9, 0.999]),
        'loss_fn': config.get('loss', 'l1'),
        'loss_config': config.get('loss_config', {}),
        'bc_format': config.get('bc_format', 'bc6'),
    }
