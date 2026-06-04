"""
可视化工具：对比训练好的模型 (UC/BC) 与数据集参考图像

用法:
  # 可视化 UC 模型
  python visualize.py --config configs/bc1_bcf05k.yaml --ckpt checkpoints/xxx/material.pth --material concrete_wall_006

  # 可视化 BC 模型
  python visualize.py --config configs/bc1_bcf05k.yaml --ckpt checkpoints/xxx/material.pth --material concrete_wall_006 --bc-format bc1

  # 指定输出路径和 mip 级别
  python visualize.py ... --output vis.png --scale 2.0
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import argparse

from dataset import MaterialDataset
from ntc_model import make_model
from ntc_bc_model import make_bc_model
from ntc_config import load_config, get_model_params


def _safe_font(size=12):
    """尝试加载默认字体，失败则返回 None。"""
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _compute_psnr(pred, ref):
    mse = F.mse_loss(pred, ref).item()
    return -10 * np.log10(max(mse, 1e-10))


def _tensor_to_hwc(tensor, is_normal=False):
    """[C,H,W] -> [H,W,C] numpy float32 in [0,1]。"""
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    if is_normal:
        img = (img + 1.0) / 2.0
    return np.clip(img, 0, 1)


def _to_rgb(img_np):
    """确保 [H,W,C] 且 C==3。灰度图复制为 RGB。"""
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    return img_np


def _add_label(img_np, label, psnr=None, font=None):
    """在图像顶部添加文字标签栏，返回 numpy 数组。"""
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
    """给图像添加彩色边框。color: (R,G,B) in [0,1]。"""
    h, w = img_np.shape[:2]
    c = img_np.shape[2] if img_np.ndim == 3 else 1
    if c == 1:
        img_np = np.stack([img_np[:, :, 0]] * 3, axis=-1)
    bordered = np.ones((h + thickness * 2, w + thickness * 2, 3), dtype=np.float32)
    bordered[:, :] = color
    bordered[thickness:-thickness, thickness:-thickness] = img_np
    return bordered


@torch.no_grad()
def visualize(model, material_data, device='cuda', output_path='visualization.png',
              scale=0.0, show_error=True, add_borders=True):
    """
    生成模型预测 vs 数据集参考的可视化大图。

    Args:
        model: 已加载的 UC 或 BC 模型 (已 to(device)).
        material_data: dict, 包含 'name', 'ref_tensor', 'mipmaps'.
        device: 'cuda' or 'cpu'.
        output_path: 输出 PNG 路径.
        scale: 连续 mip 级别 (0.0 = 最高分辨率).
        show_error: 是否显示绝对误差图.
        add_borders: 是否为不同通道组添加彩色边框.

    Returns:
        输出文件路径.
    """
    model.eval()
    font = _safe_font()

    ref_tensor = material_data['ref_tensor'].to(device)  # [9, H, W]
    name = material_data['name']
    h, w = ref_tensor.shape[1], ref_tensor.shape[2]

    # 如果 scale 指向更低分辨率的 mip，用对应的参考
    mip_level = int(scale)
    mipmaps = material_data.get('mipmaps', [])
    if 0 <= mip_level < len(mipmaps):
        ref_tensor = mipmaps[mip_level].to(device)
        h, w = ref_tensor.shape[1], ref_tensor.shape[2]

    # ---- 推理 ----
    u = torch.linspace(0, 1, w, device=device)
    v = torch.linspace(0, 1, h, device=device)
    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)
    scales = torch.tensor([float(scale)], device=device)
    pred = model(uv, scales)  # [1, 9, H, W]

    # ---- 拆分通道 ----
    # 模型输出: albedo(3) + normal_xy(2) + ao(1) + roughness(1) + metalness(1)
    # 参考数据: albedo(3) + normal_xyz(3) + ao(1) + roughness(1) + metalness(1)
    pred_albedo = pred[0, 0:3]
    pred_normal_xy = pred[0, 3:5]
    pred_ao = pred[0, 6:7]
    pred_rough = pred[0, 7:8]
    pred_metal = pred[0, 8:9]

    # 重建 normal Z
    z = torch.sqrt(torch.clamp(1.0 - pred_normal_xy[0:1] ** 2 - pred_normal_xy[1:2] ** 2, min=0))
    pred_normal = torch.cat([pred_normal_xy, z], dim=0)

    ref_albedo = ref_tensor[0:3]
    ref_normal = ref_tensor[3:6]
    ref_ao = ref_tensor[6:7]
    ref_rough = ref_tensor[7:8]
    ref_metal = ref_tensor[8:9]

    # 边框颜色 (RGB)
    colors = {
        'Albedo': (0.8, 0.2, 0.2),
        'Normal': (0.2, 0.6, 0.8),
        'AO': (0.6, 0.6, 0.6),
        'Roughness': (0.4, 0.8, 0.4),
        'Metalness': (0.9, 0.7, 0.2),
    }

    rows = []
    psnrs = {}

    for label, pred_ch, ref_ch in [
        ('Albedo', pred_albedo, ref_albedo),
        ('Normal', pred_normal, ref_normal),
        ('AO', pred_ao, ref_ao),
        ('Roughness', pred_rough, ref_rough),
        ('Metalness', pred_metal, ref_metal),
    ]:
        psnr = _compute_psnr(pred_ch.unsqueeze(0), ref_ch.unsqueeze(0))
        psnrs[label] = psnr

        pred_np = _to_rgb(_tensor_to_hwc(pred_ch, is_normal=(label == 'Normal')))
        ref_np = _to_rgb(_tensor_to_hwc(ref_ch, is_normal=(label == 'Normal')))

        panels = [pred_np, ref_np]
        titles = [f'{label} Pred', f'{label} Ref']

        if show_error:
            diff = np.abs(pred_np - ref_np)
            panels.append(diff)
            titles.append(f'{label} Diff')

        # 添加标签栏
        labeled = []
        for idx, (p, t) in enumerate(zip(panels, titles)):
            p = _add_label(p, t, psnr if idx == 0 else None, font=font)
            if add_borders:
                p = _make_border(p, colors[label], thickness=2)
            labeled.append(p)

        # 水平拼接一行
        min_h = min(p.shape[0] for p in labeled)
        labeled = [p[:min_h] for p in labeled]
        row = np.concatenate(labeled, axis=1)
        rows.append(row)

    # 垂直拼接所有行（统一宽度）
    min_w = min(r.shape[1] for r in rows)
    rows = [r[:, :min_w] for r in rows]

    # 行之间添加黑色分隔线
    sep = np.zeros((4, min_w, 3), dtype=np.float32)
    final_rows = []
    for i, r in enumerate(rows):
        final_rows.append(r)
        if i < len(rows) - 1:
            final_rows.append(sep)

    full = np.concatenate(final_rows, axis=0)

    # 顶部全局标题栏
    title_h = 22
    title_bar = np.zeros((title_h, full.shape[1], 3), dtype=np.float32)
    title_bar[:, :] = (0.1, 0.1, 0.15)
    full = np.concatenate([title_bar, full], axis=0)

    pil = Image.fromarray((full * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    header = f"Material: {name}  |  Scale (mip): {scale:.1f}"
    draw.text((6, 3), header, fill=(255, 255, 255), font=font)

    pil.save(output_path)

    # ---- 打印摘要 ----
    print(f"\nVisualization saved to: {output_path}")
    print(f"  Material : {name}")
    print(f"  Scale    : {scale}")
    print(f"  Resolution: {h}x{w}")
    print(f"  {'Channel':>12s}  {'PSNR (dB)':>10s}")
    print(f"  {'-'*26}")
    for label in ['Albedo', 'Normal', 'AO', 'Roughness', 'Metalness']:
        print(f"  {label:>12s}  {psnrs[label]:>10.2f}")
    total_psnr = _compute_psnr(pred, ref_tensor.unsqueeze(0))
    print(f"  {'-'*26}")
    print(f"  {'Total':>12s}  {total_psnr:>10.2f}")

    return output_path


def load_model_for_vis(ckpt_path, model_params, device, bc_format_name=None):
    """从 checkpoint 加载模型（UC 或 BC）。"""
    if bc_format_name:
        model = make_bc_model(model_params, output_dim=9, bc_format_name=bc_format_name)
    else:
        model = make_model(model_params, output_dim=9)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # UC checkpoint 可能直接是 state_dict；BC checkpoint 通常包装在 dict 中
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Visualize model prediction vs dataset reference')
    parser.add_argument('--config', type=str, required=True, help='YAML config path')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint (.pth)')
    parser.add_argument('--material', type=str, required=True, help='Material name in dataset/')
    parser.add_argument('--output', type=str, default='visualization.png', help='Output image path')
    parser.add_argument('--scale', type=float, default=0.0, help='Mipmap level (0.0 = full res)')
    parser.add_argument('--bc-format', type=str, default=None, choices=['bc1', 'bc2', 'bc3', 'bc4', 'bc5'],
                        help='If set, load as BC model. Otherwise UC.')
    parser.add_argument('--no-error', action='store_true', help='Do not show error maps')
    parser.add_argument('--dataset-root', type=str, default='dataset', help='Dataset root directory')
    parser.add_argument('--target-res', type=int, default=256, help='Dataset target resolution')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载配置（visualize 对 bc_format 不敏感，UC/BC 由 --bc-format 控制）
    try:
        config = load_config(args.config)
    except ValueError as e:
        # 如果 config 里有不支持的 bc_format（如 bc6），fallback 到 raw yaml
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"[visualize] Config validation warning (ignored): {e}")

    model_params = get_model_params(config)

    # 加载数据集
    ds = MaterialDataset(args.dataset_root, target_res=args.target_res, preload=True)
    if args.material not in ds.material_names:
        raise ValueError(f"Unknown material '{args.material}'. Available: {ds.material_names}")
    material_data = ds.get_by_name(args.material)
    material_data['name'] = args.material  # 补充名称字段

    # 加载模型
    model = load_model_for_vis(args.ckpt, model_params, device, bc_format_name=args.bc_format)

    # 生成可视化
    visualize(
        model, material_data,
        device=device,
        output_path=args.output,
        scale=args.scale,
        show_error=not args.no_error,
    )


if __name__ == '__main__':
    main()
