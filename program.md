# NTC Auto-Research

Minimize BC quantization PSNR drop across 40 PBR materials, tracking inference time and compression ratio.

## Setup

1. `git checkout -b autoresearch/<date-tag>` from master.
2. Read fixed files (never modify): `dataset.py`, `ntc_model.py`, `ntc_train.py`, `ntc_inference.py`, `ntc_bc_inference.py`, `ntc_compare.py`.
3. Read modifiable files: `evaluate.py`, `ntc_bc_model.py`, `ntc_bc_train.py`, `ntc_config.py`, `configs/*.yaml`.
4. Confirm `dataset/` contains 40 material subdirectories, each with `*_arm_2k.png`, `*_diff_2k.png`, `*_nor_dx_2k.png`.
5. Initialize `results.tsv` with header: `commit\tmean_psnr_drop\tmean_psnr_bc\tmean_inference_ms\tmean_compression_ratio\tstatus\tdescription` (gitignored, never commit).
6. `notes.md` is your **freeform notebook**. Read the last 5–10 entries before forming a new hypothesis; append a new entry whenever you have an observation, hunch, or dead-end worth remembering. Unlike `results.tsv`, `notes.md` **is committed** — it's cross-session memory.

## Usage

All hyper-params in yaml. Only CLI flags:

```
--config <yaml>          # required
--train-uc               # train UC baseline
--ckpt <checkpoints/ts/> # eval mode, load pre-trained UC
--materials <names>      # optional subset
```

```bash
# Train UC once per model-shape change
python evaluate.py --config configs/bc1_bcf05k.yaml --train-uc

# Eval BC (repeatable, reuse same --ckpt)
python evaluate.py --config configs/bc1_bcf05k.yaml --ckpt checkpoints/<ts>/
```

Re-train UC only when changing model shape: `feature_configs`, `hidden_dim`, `num_layers`.

## What you can modify

- **YAML**: `bc_format` (bc1–bc5), `loss`/`loss_config`, `filter` (trilinear/tricubic), `half_pixel_offsets`, `hidden_dim`, `num_layers`, `feature_configs`, `fc_dim`, `lr_feat`, `lr_mlp`, `betas`, `total_iterations`, `batch_res`, `mlp_param_bits`.
- **Python**: `ntc_bc_model.py`, `ntc_bc_train.py`, `evaluate.py`, `ntc_config.py`.

## What you CANNOT do

- Modify fixed files. No new packages (torch/numpy/PIL/pyyaml only). No CLI hyper-params. Only report via `evaluate.py`'s `summarize()`.

## Goal

**Primary**: Minimize mean `psnr_drop = psnr_unconstrained - psnr_bc`. Lower is better.

**Secondary** (tracked, not strictly minimized):
| Metric | Meaning |
|---|---|
| `inference_ms` | BC forward latency on full-res UV grid |
| `compression_ratio` | `bc_bits / png_bits`, lower = stronger compression |

0.05 dB improvement at 3× inference cost is not a win. Simpler is better — removing code for equal results is a simplification win.

## Dataset

40 PBR materials, each with 3 × 2K PNGs concatenated into 9-channel reference:

| Texture | Channels | Range |
|---|---|---|
| `*_diff_2k.png` | RGB Albedo | [0, 1] |
| `*_nor_dx_2k.png` | RGB Normal (DX) | [-1, 1] |
| `*_arm_2k.png` | R=AO, G=Roughness, B=Metalness | [0, 1] |

Channels: `albedo(3) + normal(3) + ao(1) + roughness(1) + metalness(1)`.

## Evaluation Pipeline

```
Train UC (once per model-shape) → checkpoints/<ts>/<name>.pth + config.yaml
Eval BC (repeatable)            → psnr_drop, inference_ms, compression_ratio
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
| Mean/Min/Max/Std | Aggregate across 40 materials |

### Output format (`summarize()`)

```
================================================================================
Aggregate over 40 materials  (256x256, BC=BC1 fl=trilinear ...):
                      Metric       Mean        Min        Max        Std
--------------------------------------------------------------------------------
          psnr_unconstrained    30.1200    25.3400    35.6700     2.5100
                     psnr_bc    28.4500    23.7800    33.8900     2.4700
                   psnr_drop     1.6700     0.8900     2.3400     0.3800
                inference_ms     1.2400     0.9800     1.6500     0.1500
           compression_ratio     0.0832     0.0410     0.1320     0.0200
================================================================================
```

Grep aggregate: `grep -E "^\s*(psnr_drop|inference_ms|compression_ratio)" run.log`

## Logging results (`results.tsv`)

```
commit   mean_psnr_drop  mean_psnr_bc  mean_inference_ms  mean_compression_ratio  status   description
a1b2c3d  1.67            28.45         1.24               0.0832                  keep     baseline: BC1 trilinear 10000 iters
```

- `status`: `keep` | `discard` | `crash`
- Crash row: `99.99 / 0.00 / 0.00 / 0.0000`
- Never commit results.tsv.

## Experiment Loop

LOOP:
1. `git log -5`, read results.tsv last 20 rows, **skim last 5–10 entries of notes.md** → form hypothesis.
2. Edit yaml (or modifiable .py). One variable at a time.
3. `git commit -m "<description>"`.
4. Run: `python evaluate.py --config <yaml> --ckpt <$CKPT> > run.log 2>&1`
5. Parse results. If crash → `tail -50 run.log`, fix (≤3 attempts) or `git reset --hard HEAD~1` + log crash.
6. If psnr_drop improved and secondary metrics acceptable → keep. Else → `git reset --hard HEAD~1`, log discard.
7. Append to results.tsv. **Append a short entry to notes.md** if anything surprised you, any new hunch, or any pattern across recent runs. Back to step 1.

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

- **Filter**: trilinear→tricubic, gradient backprop through BC features.
- **Half-pixel offsets**: try `[]`, `[1]`, `[2,3]`, `[0,2]` subsets.
- **LR**: feat/MLP ratios, warmup, decay schedules.
- **BC format**: bc1–bc5 Pareto frontier.
- **Loss**: L1/L2/Huber, channel-weighted (normal>metalness), FFT loss.
- **MLP**: hidden_dim 16/32/64, num_layers 1/2/3, activations.
- **Feature grids**: fc_dim 3/4/6, fc_levels 4/5, asymmetric mip (vara/varb).
- **Training**: progressive res, batch_res scaling, warmup restart.
- **CR knobs**: mlp_param_bits 16→8 (INT8 deployment).

## NEVER STOP

Once the loop begins, do not pause to ask whether to continue. The human may be asleep. You are autonomous — loop runs until manually interrupted.
