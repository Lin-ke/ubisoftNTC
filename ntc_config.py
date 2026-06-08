"""YAML 配置加载器 —— 全工程唯一配置入口.

新 schema (yaml-driven, 单一事实源):

    bc_format: bc1                # bc1..bc5
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
    required = ['bc_format', 'model', 'bc_training']
    for key in required:
        if key not in config:
            raise ValueError(f"缺少必需配置项: '{key}'")

    bc_format = config['bc_format']
    supported = ('bc1', 'bc2', 'bc3', 'bc4', 'bc5')
    if bc_format not in supported:
        raise ValueError(f"不支持的 BC 格式: '{bc_format}', 支持: {', '.join(supported)}")

    model = config['model']
    for key in ('feature_configs', 'hidden_dim'):
        if key not in model:
            raise ValueError(f"model 缺少 '{key}'")

    for key in ('total_iterations', 'batch_res'):
        if key not in config['bc_training']:
            raise ValueError(f"bc_training 缺少 '{key}'")

    if 'uc_training' in config:
        for key in ('total_iterations', 'batch_res'):
            if key not in config['uc_training']:
                raise ValueError(f"uc_training 缺少 '{key}'")

    loss_fn = config.get('loss', 'l1')
    if loss_fn not in ('l1', 'mse'):
        raise ValueError(f"loss 仅支持 'l1' | 'mse'")

    filt = config['model'].get('filter', 'trilinear')
    if filt != 'trilinear':
        raise ValueError(
            f"model.filter 必须为 'trilinear' (当前: '{filt}'). "
            f"bicubic/tricubic 不对应 GPU 硬件采样行为, 已禁用."
        )


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
    if 'uc_training' not in config:
        raise ValueError("配置缺少 'uc_training' 段")
    t = config['uc_training']
    return {
        'total_iterations': t['total_iterations'],
        'batch_res': t['batch_res'],
        'lr_feat': t.get('lr_feat', 5.0e-2),
        'lr_mlp': t.get('lr_mlp', 1.0e-3),
        'gamma': t.get('gamma', 0.9995),
        'loss_channels': t.get('loss_channels', None),
        'max_useful_lod': t.get('max_useful_lod', None),
    }


def get_bc_training_params(config: Dict[str, Any]) -> Dict[str, Any]:
    t = config['bc_training']
    return {
        'total_iterations': t['total_iterations'],
        'batch_res': t['batch_res'],
        'lr_feat': t.get('lr_feat', 1.0e-2),
        'lr_mlp': t.get('lr_mlp', 1.0e-3),
        'betas': t.get('betas', [0.9, 0.999]),
        'gamma': t.get('gamma', 1.0),
        'loss_fn': config.get('loss', 'l1'),
        'loss_channels': t.get('loss_channels', None),
        'max_useful_lod': t.get('max_useful_lod', None),
    }


def get_dataset_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """提取数据集配置."""
    d = config.get('dataset', {})
    return {
        'root': d.get('root', 'dataset'),
        'target_res': d.get('target_res', 256),
        'output_channels': d.get('output_channels', 'full'),
    }


def get_benchmark_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """提取推理 benchmark 配置."""
    b = config.get('benchmark', {})
    return {
        'warmup_iters': b.get('warmup_iters', 5),
        'timing_iters': b.get('timing_iters', 20),
        'mlp_param_bits': b.get('mlp_param_bits', 16),
    }


