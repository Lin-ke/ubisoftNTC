# Neural Texture Compression (NTC) Auto-Research

本仓库是一个**持续探索中的实验框架**，基于 PyTorch 研究神经纹理压缩（Neural Texture Compression, NTC）的**更优配置、结构与训练方法**。技术路线、模型架构、压缩格式和训练策略都可能根据实验结果随时调整。当前阶段聚焦于通过神经网络将 PBR 材质贴图压缩为小块压缩特征（Block-Compressed Features）+ 轻量 MLP 解码器，在保持视觉质量的同时实现高压缩比，但未来可能转向其他方向。

---

## 项目概述

### 核心目标（不变）
- **Primary**: 最小化 `psnr_drop = psnr_bc_ref - psnr_bc`（传统BC vs 神经网络BC）。越低越好。
- **Secondary**: 跟踪推理时延 `inference_ms` 和压缩率 `compression_ratio = bc_bits / png_bits`，时延要尽可能低。你需要在保证模型的实验的情况下，尽量提高PSNR；其次是提高压缩率。

### 当前探索方向
技术路线：`--train` 一键执行三阶段流水线 UC → BC QAT → BC-MLP finetune，并在末尾自动追加 Eval，内部自动衔接。

其他见**notes.md**

### 数据集
- 20 个 PBR 材质，位于 `dataset/dataset/` 下（注意是二级目录），每个子目录包含 3 张 PNG：
  - `*_diff_2k.png` → RGB Albedo（sRGB 编码，加载时解码到线性空间 `pow(x, 2.2)`）
  - `*_nor_dx_2k.png` → RGB Normal（DirectX，保持 PNG 原始 [0, 1]，**不再**在加载时映射到 [-1, 1]）
  - `*_arm_2k.png` → R=AO, G=Roughness, B=Metalness（sRGB 编码，解码到线性空间）
- **通道范围**：组装后的 9 通道参考张量全部位于 **[0, 1]**。法线的 Z 分量在 eval/可视化时由 `ntc_utils.reconstruct_normal` 临时把 xy 从 [0,1] 映回 [-1,1] 后重建，参考张量本身不存储 [-1,1]。
- **sRGB 解码** 由 `dataset.srgb_decode` 控制（默认 `true`），设为 `false` 可关闭。
- **分辨率不统一**：18 个为 `2048×2048`，另有 `chinese_hackberry_bark`（4096×2048）和 `crepe_georgette`（2083×2048）。涉及跨材质堆叠（batched 训练）时必须按分辨率分桶，详见下文。

---

## 代码组织与模块划分

```
ntc/
├── dataset.py              # 数据集加载器（Fixed，禁止修改）
├── ntc_model.py            # UC 全精度模型：MipmapFeatureGrid + NeuralTextureModel（Fixed）
├── ntc_train.py            # UC 训练辅助：参考采样、Vaidyanathan LOD 采样、全图 PSNR 评估（Fixed）
├── ntc_bc_model.py         # BC 压缩模型：BCBlockFeature + BCMipmapFeatureGrid + NeuralBCTextureModel（可修改）
├── ntc_batch_model.py      # Batched(ensemble) 模型：多材质堆叠为 leading 维 M（可修改）
├── ntc_bc_inference.py     # 推理与可视化工具（Fixed）
├── ntc_compare.py          # BC 格式一键对比工具（Fixed）
├── ntc_utils.py            # 公共工具：法线重建、图像保存、PSNR 计算（Fixed）
├── ntc_config.py           # YAML 配置加载与校验（可修改）
├── ntc_checkpointing.py    # checkpoint 路径/读写、resume 扫描（可修改）
├── ntc_reporting.py        # 结果打印/汇总/TSV、write_done_json（可修改）
├── ntc_visualization.py    # 预测 vs 参考对比图、通道选择（可修改）
├── Tool.py                 # 主入口：--train 一键流水线 / eval（可修改）
├── train_start.py          # 一键启动包装器（可修改）
├── train_stop.py           # 一键停止包装器（可修改）
├── configs/*.yaml          # 实验配置（可修改）
├── checkpoints/            # 模型存档（按时间戳分子目录，不提交）
├── dataset/                # 材质数据（不提交）
└── notes.md                # 实验自由笔记（追加式，提交到 git）
```

### 关键模块说明

- **`dataset.py`**：`MaterialDataset` 加载所有材质，构建 mipmap 金字塔（`build_mipmaps`），始终使用原生 2K 分辨率。
- **`ntc_model.py`**：
  - `MipmapFeatureGrid`：可学习的多分辨率特征金字塔，支持 `trilinear` / `tricubic` 采样。
  - `NeuralTextureModel`：拼接多个特征网格 → MLP → 输出 9 通道材质。
- **`ntc_bc_model.py`**：
  - `BCFormat` / `BC1Format` / `BC2Format` / `BC3Format` / `BC4Format` / `BC5Format`：定义各 BC 格式的端点位深与索引位深。
  - `BCBlockFeature`：4×4 块级别的可微 BC 压缩特征层，使用 STE（Straight-Through Estimator）量化。
  - `NeuralBCTextureModel`：与 UC 模型结构相同，但特征网格使用 BC 压缩版本。
- **`ntc_batch_model.py`**：UC/BC 的 batched(ensemble) 版本，把 M 个同构材质堆叠到 leading 维 M，用一次大 kernel 替代 M 次小 kernel。提供 `init_from_uc_batched` 与 `export_per_material_state_dicts`（训练后拆回标准单材质 ckpt）。详见「Batched 训练」。
- **`Tool.py`**：主入口 — `--train` 一键流水线 (UC → BC → BC-MLP) → `eval`。
- **`ntc_config.py`**：唯一配置入口，YAML schema 包括 `bc_format`、`model`、`uc_training`、`bc_training`、`bc_mlp_training`、`dataset`、`benchmark`。

---

## 构建与运行命令


### 直接调用 Tool.py

Tool.py的调用指南：
usage: Tool.py [-h] --config CONFIG [--train] [--ckpt CKPT]
               [--materials MATERIALS] [--vis-dir VIS_DIR] [--resume]


```bash
# Train (一键流水线: UC → BC → BC-MLP → Eval)
# 默认走 batched(ensemble) 路径, 按 config 的 batch_materials 分组; 见「Batched 训练」.
# --train 会在三阶段训练完成后, 自动用同一个 ckpt 目录执行 Eval, 仅写一次 .loopit/done.json
python Tool.py --config configs/bc1_bcf05k.yaml --train

# 仅 Eval (复用已有 ckpt, 不重新训练)
python Tool.py --config configs/bc1_bcf05k.yaml --ckpt checkpoints/<ts>/

# Resume (从已有 ckpt 续训; 逐材质串行路径, 非 batched)
# 按材质扫描最新阶段 (bc-mlp > bc > uc), 用 ckpt 内嵌 config 校验模型结构兼容性
python Tool.py --config configs/bc1_bcf05k.yaml --train --resume --ckpt checkpoints/<ts>/
```

> 训练是**单进程单 GPU**：pyramid 走 batched 多材质堆叠，hash_grid / resume 走逐材质串行。已无多进程 / 多 GPU 路径。

### 推理与对比（独立工具）

```bash
# BC 模型推理与可视化
python ntc_bc_inference.py --model-type bc --bc-format bc1 --mode full

# BC 格式对比（需先有各格式的 checkpoint）
python ntc_compare.py --formats bc1 bc3
```

---

## 配置文件（YAML Schema）

示例见 `configs/bc1_bcf05k.yaml`：

```yaml
bc_format: bc1                # bc1 | bc2 | bc3 | bc4 | bc5
loss: l1                      # l1 | mse
loss_config: {}               # 可选: 通道权重等

batch_materials: 10           # batched 分组大小 (0=全部一组); 见「Batched 训练」

dataset:
  root: dataset/dataset
  srgb_decode: true             # sRGB→linear 解码 (albedo/arm); 默认 true

model:
  filter: trilinear           # trilinear | tricubic
  num_layers: 2
  half_pixel_offsets: [1, 3]  # 在哪些 feature grid 索引上施加半像素偏移
  feature_configs:            # 每项 [base_resolution, num_mips, feature_dim]
    - [2048, 10, 3]
    - [2048, 10, 3]
    - [1024, 9, 3]
    - [1024, 9, 3]
  hidden_dim: 16
  output_activation: none     # none | sigmoid | tanh

uc_training:
  total_iterations: 5000
  batch_res: 128
  uv_sampling: tile           # tile (随机抠 tile) | uniform (整张铺满 [0,1])
  lr_feat: 5.0e-2
  lr_mlp: 1.0e-3
  gamma: 0.9995

bc_training:
  total_iterations: 200000
  batch_res: 128
  uv_sampling: tile
  lr_feat: 1.0e-2
  lr_mlp: 1.0e-3
  betas: [0.9, 0.999]

bc_mlp_training:
  total_iterations: 1000
  batch_res: 128
  uv_sampling: tile
  lr_mlp: 1.0e-3
  betas: [0.9, 0.999]

benchmark:
  warmup_iters: 5
  timing_iters: 20
  mlp_param_bits: 16          # 假设 MLP 部署精度 (FP16 默认)
```

**注意**：`feature_configs`、`hidden_dim`、`num_layers` 改变后必须重新跑 UC 训练，因为模型形状变化。

---

## Batched（ensemble）训练 —— 默认路径

`--train` 默认走 batched 路径（仅 `encoding: pyramid`、非 resume 时）：把 M 个**同构**材质堆叠到 leading 维度 M，一次大 kernel 替代 M 次小 kernel，直接解决单材质 `batch=1` 导致的 GPU 低利用率。

- **共享 LOD**：每个 iteration 所有材质共享同一连续 LOD `scale`（UV tile 仍各自随机）。这是硬性前提 —— `grid_sample` 要求 batch 内输入分辨率一致。
- **loss = Σ_m mean(per-material tile loss)**：各材质参数互相独立，求和使每材质梯度与单独训练完全一致（避免 mean 带来的 1/M 等效降 lr）。
- **分辨率分桶**：batched 要求同组 GT mip 形状一致，因此先按 `ref_tensor` 分辨率分桶，再在桶内按 `batch_materials` 分组。当前数据集 → `[2048², 2048²...] + [4096×2048] + [2083×2048]`。
- **checkpoint 兼容**：训练后 `export_per_material_state_dicts` 拆回标准单材质 `.pth`，eval / inference / compare **零改动**复用。

### `batch_materials`（分组大小）

YAML 顶层参数：`0` = 全部一组（M=全部）；`N>0` = 每组 N 个材质。**所有调参放 YAML，不要硬编码**。

**显存是唯一约束**：BC 阶段最坏情况（LOD0 解压全分辨率特征）峰值随 M **线性**增长。RTX 5070（12GB，~10.8GB 可用）实测 `bc1_bcf05k`：

| M | LOD0 峰值 | 状态 |
|---|----------|------|
| 8  | 5.4 GB  | ✅ 安全 |
| 10 | 6.7 GB  | ✅ 安全（当前默认）|
| 12 | 8.0 GB  | ✅ 可用 |
| 16 | 10.7 GB | ⚠️ 临界，无余量 |
| ≥18| ≥12 GB  | ❌ OOM |

**经验法则**：约 `0.67 GB/材质`（针对 2048 特征网格的 bc1_bcf05k；其它 config 需重新估）。换更大模型 / 更高分辨率特征网格时，每材质成本上升，需相应调小 `batch_materials`。OOM 时优先调小它，而不是改模型结构。

> 注意：分辨率分桶会让分组数 ≠ `ceil(20/N)`。例如 `batch_materials: 10` 实际为 `[10, 8, 1, 1]` 共 4 组（18 个 2048² + 2 个异类各自成桶），而非 2 组。

---

## 开发约定与代码风格


### 实验纪律
a- 所有可调参数必须放在 YAML 中，不要硬编码到逻辑里。
b- train和eval的代码最后，必须调用write_done_json！！(`--train` 流程末尾的 eval 阶段会统一写入, train 三阶段不再各自写, 避免覆盖.)

### Git 使用
- `notes.md` 是跨 session 的共享笔记，**追加不写覆盖**。

## 训练与评估流程

```
Train UC
    → checkpoints/<ts>/<name>.pth + config.yaml + uc_psnr.tsv
Train BC (QAT)
    → checkpoints/<ts>/bc_<format>/<name>.pth
Train BC-MLP (finetune)
    → checkpoints/<ts>/bc_<format>_mlp/<name>.pth
Eval
    → psnr_bc_ref, psnr_bc, psnr_drop, inference_ms, compression_ratio
    → <ckpt>/eval_<config-stem>.tsv
```

### 主要指标

| 指标 | 说明 |
|------|------|
| `psnr_bc_ref` | 传统 BC 压缩 PSNR（baseline reference） |
| `psnr_bc` | 神经网络 BC 模型 PSNR |
| `psnr_drop` | `psnr_bc_ref - psnr_bc` |
| `inference_ms` | BC 前向推理时延（全分辨率 UV grid） |
| `compression_ratio` | `bc_bits / png_bits`，越低压缩越强 |
---

## 测试策略

本项目**没有单元测试框架**。验证依赖以下方式：

1. **UC 训练验证**：看单材质 PSNR 是否收敛到合理范围（通常 25~35 dB）。
2. **BC 训练验证**：看 `psnr_drop` 是否为正且不过大（基线约 1~3 dB）。
3. **Eval TSV 校验**：检查 `eval_*.tsv` 中各材质指标是否有异常值（如 `drop` 为负或极大）。
4. **可视化对比**：`--vis-dir` 参数可生成预测 vs 参考的对比图，用于人工检查。
5. **Crash 处理**：
   - 简单错误（typo、缺 key）→ 修复后重跑（≤3 次尝试）。
   - 根本性问题 → 记 crash 到 `results.tsv`，`git reset --hard HEAD~1`，继续下一实验。

---

## 性能与资源约束

- **时间**：BC 训练耗时 > 2× 基线 → kill，discard。
- **VRAM**：允许 ≤2× 增长，但必须有对应收益。OOM 时优先调小 `batch_materials`（见「Batched 训练」），而非改模型结构。
- **推理时延**：0.05 dB 的 PSNR 提升如果代价是 3× 推理耗时，不算胜利。更简单的方案更好。

---
---

## 快速参考：常用文件与命令

| 任务 | 命令/文件 |
|------|----------|
| 启动训练 | `python train_start.py --mode train`（默认后台） |
| 评估 | `python train_start.py --mode eval --ckpt checkpoints/<ts>/` |
| 前台运行 | `python train_start.py --mode train --foreground` |
| 停止训练 | `python train_stop.py` |
| 查看结果 | `grep -E "^\s*(psnr_drop|inference_ms|compression_ratio)" run.log` |
| 实验笔记 | 追加到 `notes.md` |
| 结果记录 | 追加到 `results.tsv`（不提交） |
| 配置文件 | `configs/bc1_bcf05k.yaml` 等 |


## 注意事项！！！
a- `train_start.py` **默认后台运行并自动检查启动日志**（等待 10 秒后扫描日志中的异常），启动后立即返回。Agents 直接运行即可，无需额外监控。完成后我会提醒你。

b- train和eval的代码最后，必须调用write_done_json！！(`--train` 流程末尾的 eval 阶段会统一写入, train 三阶段不再各自写, 避免覆盖.)

c- 禁止行为：不要使用 `tail -f`、`Get-Content -Wait`、循环轮询日志、等待训练完成。`train_start.py` 已内置启动检查。


## 执行流程
Train UC → Train BC → Train BC-MLP → Eval BC。