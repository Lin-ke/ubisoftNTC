# NTC Auto-Research

Minimize BC quantization PSNR drop across 40 PBR materials, tracking inference time and compression ratio.

## Setup

1. Read fixed files (never modify): `dataset.py`, `ntc_model.py`, `ntc_train.py`, `ntc_bc_inference.py`, `ntc_compare.py`.
2. Read modifiable files: `Tool.py`, `ntc_bc_model.py`, `ntc_bc_train.py`, `ntc_config.py`, `configs/*.yaml`.
3. Confirm `dataset/` contains material subdirectories with `*_arm_2k.png`, `*_diff_2k.png`, `*_nor_dx_2k.png`.
4. `notes.md` is your **freeform notebook**. Read the last 5–10 entries before forming a new hypothesis; append whenever you have an observation, hunch, or dead-end worth remembering. Unlike `results.tsv`, `notes.md` **is committed** — it's cross-session memory.

## Quick Start (One-Click)

Use `train_start.py` / `train_stop.py` instead of calling `Tool.py` directly.

```bash
# UC 训练（自动检测 GPU，每卡 2 worker，日志写到 logs/）
python train_start.py --mode train-uc

# BC 训练
python train_start.py --mode train-bc --ckpt checkpoints/<ts>/

# Eval（复用已有 checkpoint）
python train_start.py --mode eval --ckpt checkpoints/<ts>/ --workers 10

# 后台运行
python train_start.py --mode train-uc --daemon

# 停止（读 pid 文件，优雅退出）
python train_stop.py

# 强制杀死所有 Tool.py 进程
python train_stop.py --all --force
```

如需直接调用 `Tool.py`：

```bash
# Train BC (4 worker parallel)
python Tool.py --config configs/bc1_bcf05k.yaml --train-bc --num-workers 4

# Train UC (optional, for PSNR drop comparison)
python Tool.py --config configs/bc1_bcf05k.yaml --train-uc --num-workers 4

# Eval (4 worker parallel, reuse same --ckpt)
python Tool.py --config configs/bc1_bcf05k.yaml --ckpt checkpoints/<ts>/ --num-workers 4
```

Model-shape changes require a new UC run: `feature_configs`, `hidden_dim`, `num_layers`.

## What you can modify

- **YAML**: `bc_format` (bc1–bc5), `loss`/`loss_config`, `filter`, `half_pixel_offsets`, `hidden_dim`, `num_layers`, `feature_configs`, `lr_feat`, `lr_mlp`, `betas`, `total_iterations`, `batch_res`, `mlp_param_bits`.
- **Python**: `ntc_bc_model.py`, `ntc_bc_train.py`, `Tool.py`, `ntc_config.py`.

## What you CANNOT do

- Modify fixed files. No new packages (torch/numpy/PIL/pyyaml only). Only report via `Tool.py`'s `summarize()`.

## Goal

**Primary**: Minimize mean `psnr_drop = psnr_unconstrained - psnr_bc`. Lower is better.

**Secondary** (tracked, not strictly minimized):

| Metric | Meaning |
|---|---|
| `inference_ms` | BC forward latency on full-res UV grid |
| `compression_ratio` | `bc_bits / png_bits`, lower = stronger compression |

0.05 dB improvement at 3× inference cost is not a win. Simpler is better.

## Dataset

20 PBR materials, each with 3 × 2K PNGs → 9-channel reference:

| Texture | Channels | Range |
|---|---|---|
| `*_diff_2k.png` | RGB Albedo | [0, 1] |
| `*_nor_dx_2k.png` | RGB Normal (DX) | [-1, 1] |
| `*_arm_2k.png` | R=AO, G=Roughness, B=Metalness | [0, 1] |

## Evaluation Pipeline

```
Train BC (standalone)            → checkpoints/<ts>/bc_<format>/<name>.pth + config.yaml
Train UC (optional, for drop)    → checkpoints/<ts>/<name>.pth
Eval (load BC + optional UC)     → psnr_bc, psnr_drop (if UC), inference_ms, compression_ratio
                                   → writes <ckpt>/eval_<config-stem>.tsv
```

### Metrics

| Metric | Description |
|---|---|
| `psnr_unconstrained` | Upper bound (full-precision UC) |
| `psnr_bc` | BC compressed model |
| `psnr_drop` | `psnr_uc - psnr_bc` (primary) |
| `inference_ms` | BC forward latency on target_res grid |
| `compression_ratio` | `bc_bits / png_bits` |

Grep aggregate: `grep -E "^\s*(psnr_drop|inference_ms|compression_ratio)" run.log`

## Logging results (`results.tsv`)

```
commit   mean_psnr_drop  mean_psnr_bc  mean_inference_ms  mean_compression_ratio  status   description
a1b2c3d  1.67            28.45         1.24               0.0832                  keep     baseline: BC1 trilinear 10000 iters
```

- `status`: `keep` | `discard` | `crash`
- Crash row: `99.99 / 0.00 / 0.00 / 0.0000`
- Never commit `results.tsv`.

## Experiment Loop

1. Read `results.tsv` last 20 rows, skim `notes.md` last 5–10 entries → form hypothesis.
2. Edit yaml (or modifiable .py). One variable at a time.
3. `git commit -m "<description>"`.
4. Run: `python Tool.py --config <yaml> --ckpt <$CKPT> --num-workers 4 > run.log 2>&1`
5. Parse results. If crash → `tail -50 run.log`, fix (≤3 attempts) or `git reset --hard HEAD~1` + log crash.
6. If psnr_drop improved and secondary metrics acceptable → keep. Else → `git reset --hard HEAD~1`, log discard.
7. Append to `results.tsv`. Append a short entry to `notes.md` if anything surprised you. Back to step 1.

## Crash handling

- Easy fix (typo, missing key) → fix and re-run.
- Fundamentally broken → log crash, reset, move on.
- ≤3 attempts per crash.

## Constraints

- Time: >2× baseline → kill, discard.
- VRAM: ≤2× increase acceptable for meaningful gains.
- No new packages. All params in yaml.
- One variable per experiment.

## Exploration directions

see Notes.md 

## NEVER STOP

Once the loop begins, do not pause to ask whether to continue. The human may be asleep. You are autonomous — loop runs until manually interrupted.
