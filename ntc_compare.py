"""
BC 格式一键对比工具
===========================

扫描各格式的训练输出目录 (output_bc1/ ~ output_bc5/)，
对所有已训练完成的格式运行推理，生成汇总对比报告。

对比维度：
  - 各通道 PSNR (Albedo / Normal / AO / Roughness / Metalness / Total)
  - 每块每通道 bits 与压缩率
  - 各格式的预测图像 (保存到 output_compare/)

用法：
  python ntc_compare.py                     # 对比所有已有 checkpoint 的格式
  python ntc_compare.py --formats bc1 bc3   # 只对比 BC1 和 BC3
  python ntc_compare.py --output-dir my_cmp # 指定输出目录
"""

import torch
import numpy as np
import os
import sys
import argparse

from ntc_bc_model import make_bc_model, get_bc_format
from ntc_bc_inference import get_bits_per_block
from dataset import load_material, build_mipmaps
from ntc_utils import reconstruct_normal, save_image, compute_psnr

_DEFAULT_MODEL_PARAMS = {
    'feature_configs': [
        (512, 8, 3),
        (256, 7, 3),
        (128, 6, 3),
        (64, 5, 3),
    ],
    'hidden_dim': 16,
    'num_layers': 1,
    'filter': 'trilinear',
    'half_pixel_offsets': [1, 3],
}

ALL_FORMATS = ['bc1', 'bc2', 'bc3', 'bc4', 'bc5']


@torch.no_grad()
def evaluate_format(bc_format_name, checkpoint_path, ref, device='cuda'):
    """对单个 BC 格式运行推理，返回各通道 PSNR 字典。

    Args:
        bc_format_name: BC 格式名称
        checkpoint_path: checkpoint 路径
        ref: 参考材质 tensor [C, H, W]
        device: 计算设备

    Returns:
        dict: {channel_name: psnr_value, ...} 包含 Albedo/Normal/AO/Roughness/Metalness/Total
    """
    output_dim = ref.shape[0]
    h, w = ref.shape[1], ref.shape[2]

    model = make_bc_model(_DEFAULT_MODEL_PARAMS, output_dim=output_dim,
                          bc_format_name=bc_format_name).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    u = torch.linspace(0, 1, w, device=device)
    v = torch.linspace(0, 1, h, device=device)
    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)
    scale = torch.zeros(1, device=device)

    pred = model(uv, scale)

    # 提取各通道
    albedo = pred[:, 0:3]
    normal_xy = pred[:, 3:5]
    ao = pred[:, 6:7]
    roughness = pred[:, 7:8]
    metalness = pred[:, 8:9]
    normal_full = reconstruct_normal(normal_xy)

    ref_albedo = ref[0:3].unsqueeze(0)
    ref_normal = ref[3:6].unsqueeze(0)
    ref_ao = ref[6:7].unsqueeze(0)
    ref_roughness = ref[7:8].unsqueeze(0)
    ref_metalness = ref[8:9].unsqueeze(0)

    results = {}
    for name, p, r in [
        ('Albedo', albedo, ref_albedo),
        ('Normal', normal_full, ref_normal),
        ('AO', ao, ref_ao),
        ('Roughness', roughness, ref_roughness),
        ('Metalness', metalness, ref_metalness),
    ]:
        results[name] = compute_psnr(p, r)

    results['Total'] = compute_psnr(pred, ref.unsqueeze(0))

    return results, pred


def save_comparison_images(predictions, ref, output_dir):
    """保存各格式的预测图像到对比目录。

    Args:
        predictions: {format_name: pred_tensor, ...}
        ref: 参考材质 tensor
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存参考图像（只存一份）
    ref_albedo = ref[0:3]
    ref_normal = ref[3:6]
    ref_ao = ref[6:7]
    ref_roughness = ref[7:8]
    ref_metalness = ref[8:9]

    save_image(ref_albedo, f'{output_dir}/ref_albedo.png')
    save_image(ref_normal, f'{output_dir}/ref_normal.png', is_normal=True)
    save_image(ref_ao, f'{output_dir}/ref_ao.png')
    save_image(ref_roughness, f'{output_dir}/ref_roughness.png')
    save_image(ref_metalness, f'{output_dir}/ref_metalness.png')

    # 保存各格式的预测和误差图
    for fmt_name, pred in predictions.items():
        albedo = pred[:, 0:3]
        normal_xy = pred[:, 3:5]
        ao = pred[:, 6:7]
        roughness = pred[:, 7:8]
        metalness = pred[:, 8:9]
        normal_full = reconstruct_normal(normal_xy)

        for ch_name, p, r in [
            ('albedo', albedo, ref_albedo.unsqueeze(0)),
            ('normal', normal_full, ref_normal.unsqueeze(0)),
            ('ao', ao, ref_ao.unsqueeze(0)),
            ('roughness', roughness, ref_roughness.unsqueeze(0)),
            ('metalness', metalness, ref_metalness.unsqueeze(0)),
        ]:
            save_image(p.squeeze(0), f'{output_dir}/{fmt_name}_{ch_name}_pred.png',
                       is_normal=(ch_name == 'normal'))

            diff = (p - r).abs()
            if diff.shape[1] > 1:
                diff = diff.mean(dim=1, keepdim=True)
            # 放大误差 5x 以便观察
            diff_vis = (diff * 5.0).clamp(0, 1)
            save_image(diff_vis.squeeze(0), f'{output_dir}/{fmt_name}_{ch_name}_diff.png')


def format_report(all_results, output_dir=None):
    """生成并打印（+可选保存）汇总对比报告。

    Args:
        all_results: {format_name: {channel: psnr, ...}, ...}
        output_dir: 若提供，将报告保存为 report.txt
    """
    formats = sorted(all_results.keys())
    channels = ['Albedo', 'Normal', 'AO', 'Roughness', 'Metalness', 'Total']

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("BC FORMAT COMPARISON REPORT")
    lines.append("=" * 80)

    # ---- PSNR 对比表 ----
    lines.append("")
    lines.append("1. PSNR (dB) - higher is better")
    lines.append("-" * 80)

    # 表头
    header = f"{'Channel':>12s}"
    for fmt in formats:
        header += f"  {fmt.upper():>8s}"
    lines.append(header)
    lines.append("-" * (12 + len(formats) * 10))

    # 每行一个通道
    for ch in channels:
        row = f"{ch:>12s}"
        values = [all_results[fmt].get(ch, 0) for fmt in formats]
        best = max(values)
        for fmt in formats:
            val = all_results[fmt].get(ch, 0)
            marker = " *" if val == best and len(formats) > 1 else "  "
            row += f"  {val:>6.2f}{marker}"
        lines.append(row)

    lines.append("")
    lines.append("  (* = best for that channel)")

    # ---- 压缩率对比 ----
    lines.append("")
    lines.append("2. Compression (per 4x4 block per feature channel)")
    lines.append("-" * 80)

    comp_header = f"{'Format':>8s}  {'Bits/Block/Ch':>14s}  {'Description'}"
    lines.append(comp_header)
    lines.append("-" * 80)

    for fmt in formats:
        bits, desc = get_bits_per_block(fmt)
        lines.append(f"{fmt.upper():>8s}  {bits:>14.1f}  {desc}")

    # ---- 排名 ----
    lines.append("")
    lines.append("3. Ranking by Total PSNR")
    lines.append("-" * 40)

    ranked = sorted(formats, key=lambda f: all_results[f].get('Total', 0), reverse=True)
    for i, fmt in enumerate(ranked, 1):
        total_psnr = all_results[fmt].get('Total', 0)
        bits, _ = get_bits_per_block(fmt)
        lines.append(f"  #{i}  {fmt.upper():>4s}  {total_psnr:>7.2f} dB  ({bits:.0f} bits/block/ch)")

    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='BC Format One-Click Comparison')
    parser.add_argument('--formats', nargs='+', default=None,
                        choices=ALL_FORMATS,
                        help='要对比的格式列表 (默认: 自动检测已有 checkpoint)')
    parser.add_argument('--material-dir', type=str, default='dataset/aerial_beach_02',
                        help='材质目录路径')
    parser.add_argument('--target-res', type=int, default=1024,
                        help='目标分辨率')
    parser.add_argument('--output-dir', type=str, default='output_compare',
                        help='对比结果输出目录 (default: output_compare)')
    parser.add_argument('--no-images', action='store_true',
                        help='跳过图像保存，只输出 PSNR 报告')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 确定要对比的格式
    if args.formats:
        candidate_formats = args.formats
    else:
        candidate_formats = ALL_FORMATS

    # 扫描已有 checkpoint
    available = {}
    for fmt in candidate_formats:
        ckpt_path = f'output_{fmt}/best_model.pth'
        if os.path.exists(ckpt_path):
            available[fmt] = ckpt_path

    if not available:
        print("\nNo checkpoints found!")
        print("Expected checkpoint locations:")
        for fmt in candidate_formats:
            print(f"  output_{fmt}/best_model.pth")
        print("\nTrain at least one format first:")
        print("  python evaluate.py --config configs/bc1_bcf05k.yaml --train-uc")
        sys.exit(1)

    print(f"\nFound checkpoints for: {', '.join(f.upper() for f in sorted(available))}")
    missing = set(candidate_formats) - set(available.keys())
    if missing:
        print(f"Missing checkpoints for: {', '.join(f.upper() for f in sorted(missing))}")

    # 加载参考材质（只加载一次）
    print("\nLoading reference material...")
    ref = load_material(args.material_dir, target_res=args.target_res).to(device)
    print(f"Reference shape: {ref.shape}")

    # 逐格式推理
    all_results = {}
    all_predictions = {}

    for fmt, ckpt_path in sorted(available.items()):
        print(f"\n--- Evaluating {fmt.upper()} ---")
        print(f"  Checkpoint: {ckpt_path}")
        results, pred = evaluate_format(fmt, ckpt_path, ref, device)
        all_results[fmt] = results
        if not args.no_images:
            all_predictions[fmt] = pred
        print(f"  Total PSNR: {results['Total']:.2f} dB")

    # 保存图像
    if not args.no_images and all_predictions:
        print(f"\nSaving comparison images to {args.output_dir}/...")
        save_comparison_images(all_predictions, ref, args.output_dir)

    # 生成报告
    format_report(all_results, args.output_dir)


if __name__ == '__main__':
    main()
