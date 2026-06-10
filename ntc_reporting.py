import os

import numpy as np

from ntc_utils import write_done_json


def print_header():
    hdr = (f"{'Material':<30s} {'BC_ref':>7s} {'BC_neur':>7s} {'Drop':>6s} "
           f"{'Inf(ms)':>8s} {'CR':>7s} {'Time':>6s}")
    print(hdr)
    print('-' * 85)


def print_result(r):
    bc_ref_str = f"{r['psnr_bc_ref']:>7.2f}" if 'psnr_bc_ref' in r else f"{'N/A':>7s}"
    drop_str = f"{r['psnr_drop']:>6.2f}" if 'psnr_drop' in r else f"{'N/A':>6s}"
    print(f"{r['name']:<30s} {bc_ref_str} {r['psnr_bc']:>7.2f} {drop_str} "
          f"{r['inference_ms']:>8.3f} {r['compression_ratio']:>7.4f} "
          f"{r['time_total']:>5.1f}s")


def summarize(all_results, params_label):
    print(f"\n{'='*80}")
    print(f"Aggregate over {len(all_results)} materials  "
          f"({all_results[0].get('resolution','?')}, {params_label}):")
    print(f"{'Metric':>28s} {'Mean':>10s} {'Min':>10s} {'Max':>10s} {'Std':>10s}")
    print('-' * 80)

    metrics = ['psnr_bc_ref', 'psnr_bc', 'psnr_drop', 'inference_ms', 'compression_ratio']
    for k in metrics:
        vals = [r[k] for r in all_results if k in r]
        if not vals:
            continue
        print(f"{k:>28s} {np.mean(vals):>10.4f} {np.min(vals):>10.4f} "
              f"{np.max(vals):>10.4f} {np.std(vals):>10.4f}")

    times = [r['time_total'] for r in all_results]
    print(f"\nTotal: {sum(times):.0f}s  |  Avg: {np.mean(times):.1f}s/material")
    print(f"{'='*80}")


def save_tsv(all_results, ckpt_dir, suffix):
    path = os.path.join(ckpt_dir, f'eval_{suffix}.tsv')
    cols = ['name', 'channels', 'resolution',
            'psnr_bc_ref', 'psnr_bc', 'psnr_drop',
            'inference_ms', 'bc_bits', 'png_bits', 'compression_ratio',
            'time_total']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(cols) + '\n')
        for r in all_results:
            f.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')
    print(f"Saved to {path}")


def _summarize_eval_results(all_results):
    data = {'num_materials': len(all_results)}
    for k in ('psnr_bc_ref', 'psnr_bc', 'psnr_drop', 'inference_ms', 'compression_ratio'):
        vals = [r[k] for r in all_results if k in r]
        if vals:
            data[k] = round(float(np.mean(vals)), 6)
            data[f'{k}_min'] = round(float(np.min(vals)), 6)
            data[f'{k}_max'] = round(float(np.max(vals)), 6)
    times = [r['time_total'] for r in all_results if 'time_total' in r]
    if times:
        data['time_total'] = round(float(np.sum(times)), 3)
        data['time_avg'] = round(float(np.mean(times)), 3)
    return data


def _write_eval_done(mode, ckpt_dir, config_path, bc_format_name,
                     num_workers, all_results):
    done_data = _summarize_eval_results(all_results)
    done_data.update({
        'mode': mode,
        'config': config_path,
        'bc_format': bc_format_name,
        'num_workers': num_workers,
    })
    write_done_json("eval", ckpt_dir, done_data)

