# ntc-autoresearch

NTC (Neural Texture Compression) auto-research — minimize BC quantization PSNR drop across 40 PBR materials, while tracking inference time and compression ratio.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun2`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `program.md` — this file, the experiment protocol.
   - `dataset.py` — PBR material Dataset loader (40 materials, 9-channel). **Do not modify.**
   - `ntc_model.py` — Unconstrained `NeuralTextureModel` + `MipmapFeatureGrid`. **Do not modify.**
   - `ntc_train.py` — Shared training utilities (mipmap / sampling / `evaluate_full`). **Do not modify.**
   - `ntc_inference.py` — UC inference utility. **Do not modify.**
   - `ntc_bc_inference.py` — BC inference + visualization (per-channel PSNR). **Do not modify.**
   - `ntc_bc6_partitions.py` — BC6 32 partition masks (DirectX spec). **Do not modify.**
   - `ntc_compare.py` — UC vs BC visual comparison. **Do not modify.**
   - `evaluate.py` — Dataset evaluation orchestrator. **You may modify this.**
   - `ntc_bc_model.py` — BC compressed feature model (bc1–bc6). **You may modify this.**
   - `ntc_bc_train.py` — BC QAT training routine. **You may modify this.**
   - `ntc_config.py` — YAML config loader. **You may modify this.**
   - `configs/*.yaml` — experiment configs. **You may modify these.**
4. **Verify data exists**: Check that `dataset/` contains 40 material subdirectories, each with 3 PNGs (`*_arm_2k.png`, `*_diff_2k.png`, `*_nor_dx_2k.png`).
5. **Initialize results.tsv**: Create `results.tsv` with the header row only. The baseline will be recorded after the first run. **`results.tsv` is gitignored** — never commit it.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

The pipeline is **strictly yaml-driven**. All hyper-parameters live in a single yaml file. You launch `evaluate.py` with at most 4 CLI flags:

```
--config <path/to/yaml>     # required
--train-uc                  # train the UC baseline (uses uc_training section)
--ckpt <checkpoints/<ts>/>  # load a previously trained UC (eval mode)
--materials <names>         # optional comma-separated subset
```

**No hyper-params on the CLI.** Tuning means editing the yaml.

```bash
# Step 1: Train UC once per model-shape change
python evaluate.py --config configs/bc6_bcf05k.yaml --train-uc

# Step 2: Evaluate BC under that UC (repeatable, edit yaml between runs)
python evaluate.py --config configs/bc6_bcf05k.yaml --ckpt checkpoints/<ts>/
```

### What you CAN do

Edit any yaml under `configs/` — that is the entire surface for tuning. Knobs available:

- **`bc_format`**: `bc1` … `bc6`
- **`loss`** / `loss_config`: `l1` | `mse`, plus optional channel-weighted config
- **`model`**: `feature_configs`, `hidden_dim`, `num_layers`, `filter` (`trilinear` | `tricubic`), `half_pixel_offsets` (which feature-grid indices get +0.5px shift)
- **`uc_training`**: `total_iterations`, `batch_res`, `lr_feat`, `lr_mlp`, `gamma` (LR decay)
- **`bc_training`**: `total_iterations`, `batch_res`, `lr_feat`, `lr_mlp`, `betas`
- **`benchmark`**: `warmup_iters`, `timing_iters`, `mlp_param_bits` (assumed deployment precision for compression-ratio computation)

You may also modify `ntc_bc_model.py`, `ntc_bc_train.py`, `evaluate.py`, `ntc_config.py` for deeper changes (loss formulation, new BC variants, training-loop tweaks).

### What you CANNOT do

- Modify fixed files: `dataset.py`, `ntc_model.py`, `ntc_train.py`, `ntc_inference.py`, `ntc_bc_inference.py`, `ntc_bc6_partitions.py`, `ntc_compare.py`.
- Install new packages or add dependencies. Use only `torch`, `numpy`, `PIL`, `pyyaml`.
- Modify the evaluation metric. `evaluate_full()` in `ntc_train.py` is ground truth for PSNR. Only report via `evaluate.py`'s `summarize()` output.
- Pass any hyper-parameter via CLI. Edit the yaml.

### The goal

**Primary objective**: minimize **mean `psnr_drop`** across 40 materials.
`psnr_drop = psnr_unconstrained − psnr_bc`. Lower is better.

**Secondary objectives** (tracked, not strictly minimized — used to evaluate trade-offs):

| Metric | Meaning |
|---|---|
| `inference_ms` | Mean BC-model forward time on a full-resolution UV grid (ms) |
| `compression_ratio` | `bc_bits / png_bits` — fraction of original PNG storage. **Lower = stronger compression.** |

A change is a **win** if it lowers `psnr_drop` without disproportionately hurting `inference_ms` or `compression_ratio`. A 0.05 dB drop improvement at 3× the inference cost is **not** a win. A 0.05 dB drop improvement that also lowers `compression_ratio` is a clear win.

VRAM is a soft constraint. Some increase is acceptable for meaningful PSNR_drop gains, but it should not blow up dramatically.

### Simplicity criterion

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal-or-better results is a great outcome — that's a simplification win. Weigh the complexity cost against the improvement magnitude. A 0.02 dB improvement that adds 20 lines of hacky code? Probably not worth it. A 0.02 dB improvement from deleting code? Definitely keep.

### The first run

Train the UC baseline once per model-shape, then reuse it for all subsequent BC sweeps:

```bash
# Step 1: Train UC under the chosen yaml (run once per model-shape)
python evaluate.py --config configs/bc6_bcf05k.yaml --train-uc
# → checkpoints/<timestamp>/<name>.pth + a copy of the yaml

# Step 2: Evaluate BC6 baseline (uses the UC checkpoint from step 1)
python evaluate.py --config configs/bc6_bcf05k.yaml --ckpt checkpoints/<timestamp>/
```

**Important**: Subsequent experiments do NOT retrain UC. Edit the yaml (e.g. switch `bc_format`, change `bc_training.lr_feat`, swap `loss`), then re-run with the same `--ckpt`. Retraining UC is required only when you change a yaml field that alters the **learnable parameter count or shape** of the UC model: `feature_configs`, `hidden_dim`, `num_layers`. Other knobs (`filter`, `half_pixel_offsets`, BC-side knobs, learning rates, iters, loss) do not require UC retraining.

> When loading `--ckpt`, `evaluate.py` recovers the model shape from `<ckpt_dir>/config.yaml` so the UC `state_dict` always loads cleanly even if the live yaml differs.

## Dataset

40 PBR materials under `dataset/`, randomly sampled from E:\\poly (739 total).
Each material has 3 textures at 2K resolution:

| Texture | Channels | Range |
|---|---|---|
| `*_diff_2k.png` | RGB Albedo | [0, 1] |
| `*_nor_dx_2k.png` | RGB Normal (DirectX) | [−1, 1] |
| `*_arm_2k.png` | R=AO, G=Roughness, B=Metalness | [0, 1] |

Concatenated into a 9-channel reference: `albedo(3) + normal(3) + ao(1) + roughness(1) + metalness(1)`.

## Evaluation pipeline

UC models are trained once and reused. A single UC checkpoint directory provides pre-trained weights for all 40 materials. BC evaluation loads those weights, computes `psnr_unconstrained`, trains a BC model from scratch, then reports `psnr_drop`, `inference_ms`, and `compression_ratio`.

```
Train UC (once per model-shape)
  python evaluate.py --config <yaml> --train-uc
  → checkpoints/<ts>/<name>.pth  +  config.yaml (snapshot)

Evaluate BC (repeatable — same ckpt, edit yaml between runs)
  python evaluate.py --config <yaml> --ckpt checkpoints/<ts>/
  → psnr_unconstrained, psnr_bc, psnr_drop
  → inference_ms (BC forward latency, ms)
  → compression_ratio (bc_bits / png_bits)
  → writes <ckpt>/eval_<config-stem>.tsv
```

### Metrics

| Metric | Description |
|---|---|
| `psnr_unconstrained` | Upper bound (full-precision UC model) |
| `psnr_bc` | BC compressed model PSNR |
| `psnr_drop` | Quantization loss = `psnr_uc − psnr_bc` (**primary**) |
| `inference_ms` | Mean BC-model forward time on `target_res × target_res` UV grid, after warmup |
| `bc_bits` | Total quantized BC parameter bit-count (endpoints + indices + partitions + MLP@`mlp_param_bits`) |
| `png_bits` | Sum of source PNG file sizes × 8 |
| `compression_ratio` | `bc_bits / png_bits` (lower is more compressed) |
| `Mean / Min / Max / Std` | Aggregate across 40 materials |

## Output format

`summarize()` prints (5-metric aggregate + per-material breakdown):

```
================================================================================
Aggregate over 40 materials  (256x256, BC=BC6 fl=trilinear lr=(0.01,0.001) loss=l1):
                      Metric       Mean        Min        Max        Std
--------------------------------------------------------------------------------
          psnr_unconstrained    30.1200    25.3400    35.6700     2.5100
                     psnr_bc    28.4500    23.7800    33.8900     2.4700
                   psnr_drop     1.6700     0.8900     2.3400     0.3800
                inference_ms     1.2400     0.9800     1.6500     0.1500
           compression_ratio     0.0832     0.0410     0.1320     0.0200

Total: 1200s  |  Avg: 30.0s/material
================================================================================
```

The per-material TSV is written to `<ckpt_dir>/eval_<config-stem>.tsv` with columns:

```
name  channels  resolution  psnr_unconstrained  psnr_bc  psnr_drop
inference_ms  bc_bits  png_bits  compression_ratio  time_total
```

To pluck the aggregate from a run log:

```bash
grep -E "^\s*(psnr_drop|inference_ms|compression_ratio)\s" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, untracked).

The TSV has a header row and 7 columns:

```
commit  mean_psnr_drop  mean_psnr_bc  mean_inference_ms  mean_compression_ratio  status  description
```

- `commit`: short git hash (7 chars)
- `mean_psnr_drop`: mean across 40 materials (e.g. `1.67`) — `99.99` for crashes
- `mean_psnr_bc`: mean (e.g. `28.45`) — `0.00` for crashes
- `mean_inference_ms`: mean BC forward latency (e.g. `1.24`) — `0.00` for crashes
- `mean_compression_ratio`: mean `bc_bits / png_bits` (e.g. `0.083`) — `0.0000` for crashes
- `status`: `keep` | `discard` | `crash`
- `description`: short text describing the change

Example:

```
commit   mean_psnr_drop  mean_psnr_bc  mean_inference_ms  mean_compression_ratio  status   description
a1b2c3d  1.67            28.45         1.24               0.0832                  keep     baseline: BC6 trilinear 10000 iters
b2c3d4e  1.42            28.70         1.31               0.0832                  keep     filter: trilinear → tricubic
c3d4e5f  1.89            28.23         1.22               0.1280                  discard  hidden_dim 16 → 32 (worse drop, worse CR)
d4e5f6g  99.99           0.00          0.00               0.0000                  crash    fc_levels 5 (OOM)
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch / commit.
2. Form a hypothesis — which yaml change might lower mean `psnr_drop` (without blowing up `inference_ms` / `compression_ratio`)?
3. Edit the yaml (or, for deeper changes, the modifiable `.py` files).
4. `git commit` with a descriptive message.
5. Run the experiment:
   ```bash
   python evaluate.py --config <yaml> --ckpt checkpoints/<ts>/ > run.log 2>&1
   ```
   Redirect everything — do NOT use `tee` or let output flood your context.
6. Read results: `grep -E "^\s*(psnr_drop|inference_ms|compression_ratio)" run.log` or open the per-material TSV.
7. If grep is empty, the run crashed. `tail -n 50 run.log` to read the traceback. If you can't fix it in a few attempts, give up.
8. Record results in `results.tsv`. **Do NOT commit `results.tsv`** — it is gitignored.
9. If `mean_psnr_drop` improved (and the secondary metrics are not unacceptable), advance the branch — keep the commit.
10. Otherwise `git reset --hard HEAD~1` to discard.

### Fixing crashes

- Easy fix (typo, missing yaml key, import) → fix it and re-run.
- Idea is fundamentally broken → log `crash` in TSV, `git reset --hard HEAD~1`, move on.
- Don't spend more than 3 attempts fixing the same crash.

### Limitations

- **Time budget**: each experiment should finish within reasonable wall time. If a run exceeds 2× the baseline time, kill and discard.
- **VRAM**: keep a rough eye on GPU memory. 2× increase is acceptable for meaningful gains; 10× blow-up is not.
- **No new packages**: only `torch`, `numpy`, `PIL`, `pyyaml`.

### Exploration directions

- **Filter**: `trilinear` vs `tricubic` — gradient backprop through BC features.
- **Half-pixel offsets**: which subset of feature-grid indices to shift by +0.5px (currently `[1, 3]`); try `[]`, `[1]`, `[2,3]`, `[0,2]` …
- **Learning rates**: feats-vs-MLP LR ratios; constant vs LR-decayed BC training; warmup strategies.
- **BC formats**: `psnr_drop` / `inference_ms` / `compression_ratio` Pareto across `bc1`–`bc6`.
- **Loss function**: L1 vs L2 vs Huber; channel-weighted (normals > metalness); frequency-domain loss.
- **MLP**: `hidden_dim` (16→32→64); `num_layers` (1→2→3); activation functions.
- **Feature grids**: `fc_dim` (3→4→6); `fc_levels` (4→5); asymmetric mip configs (vara/varb).
- **Training**: warmup restart; progressive resolution; `batch_res` scaling.
- **Compression-ratio knobs**: `mlp_param_bits` (16→8) — assume INT8 deployment to push CR down.

### NEVER STOP

Once the experiment loop has begun (after initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from a computer, and expects you to continue working indefinitely until manually stopped. You are autonomous. If you run out of ideas, think harder — read the BC format spec in `task.md`, re-read the modifiable files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
