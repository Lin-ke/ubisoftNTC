# Neural Texture Compression (NTC) Auto-Research

本仓库是一个**持续探索中的实验框架**，基于 PyTorch 研究神经纹理压缩（Neural Texture Compression, NTC）的**更优配置、结构与训练方法**。技术路线、模型架构、压缩格式和训练策略都可能根据实验结果随时调整。当前阶段聚焦于通过神经网络将 PBR 材质贴图压缩为小块压缩特征（Block-Compressed Features）+ 轻量 MLP 解码器，在保持视觉质量的同时实现高压缩比，但未来可能转向其他方向。

---

## 项目概述

### 核心目标（不变）
- **Primary**: 最小化 `psnr_drop = psnr_bc_ref - psnr_bc`（传统BC vs 神经网络BC）。越低越好。
- **Secondary**: 跟踪推理时延 `inference_ms` 和压缩率 `compression_ratio = bc_bits / png_bits`，时延要尽可能低。你需要在保证模型的实验的情况下，尽量提高PSNR；其次是提高压缩率。

### 当前探索方向
技术路线：`--train` 一键执行三阶段流水线 UC → BC QAT → BC-MLP finetune，内部自动衔接。

其他见**notes.md**

### 数据集
- 20 个 PBR 材质，位于 `dataset/` 下，每个子目录包含 3 张 2K PNG：
  - `*_diff_2k.png` → RGB Albedo（范围 [0, 1]）
  - `*_nor_dx_2k.png` → RGB Normal（DirectX，范围 [-1, 1]）
  - `*_arm_2k.png` → R=AO, G=Roughness, B=Metalness（范围 [0, 1]）

---

## 代码组织与模块划分

```
ntc/
├── dataset.py              # 数据集加载器（Fixed，禁止修改）
├── ntc_model.py            # UC 全精度模型：MipmapFeatureGrid + NeuralTextureModel（Fixed）
├── ntc_train.py            # UC 训练辅助：参考采样、全图 PSNR 评估（Fixed）
├── ntc_bc_model.py         # BC 压缩模型：BCBlockFeature + BCMipmapFeatureGrid + NeuralBCTextureModel（可修改）
├── ntc_bc_train.py         # BC 训练辅助：LOD 采样（Vaidyanathan）（可修改）
├── ntc_bc_inference.py     # 推理与可视化工具（Fixed）
├── ntc_compare.py          # BC 格式一键对比工具（Fixed）
├── ntc_utils.py            # 公共工具：法线重建、图像保存、PSNR 计算（Fixed）
├── ntc_config.py           # YAML 配置加载与校验（可修改）
├── Tool.py                 # 主入口：--train 一键流水线 / eval（可修改）
├── train_start.py          # 一键启动包装器（可修改）
├── train_stop.py           # 一键停止包装器（可修改）
├── watchdog.ps1            # PowerShell 守护脚本，防止会话闲置
├── configs/*.yaml          # 实验配置（可修改）
├── checkpoints/            # 模型存档（按时间戳分子目录，不提交）
├── dataset/                # 材质数据（不提交）
├── notes.md                # 实验自由笔记（追加式，提交到 git）
└── results.tsv             # 实验结果汇总（不提交）
```

### 关键模块说明

- **`dataset.py`**：`MaterialDataset` 加载所有材质，构建 mipmap 金字塔（`build_mipmaps`）。`target_res` 默认 256。
- **`ntc_model.py`**：
  - `MipmapFeatureGrid`：可学习的多分辨率特征金字塔，支持 `trilinear` / `tricubic` 采样。
  - `NeuralTextureModel`：拼接多个特征网格 → MLP → 输出 9 通道材质。
- **`ntc_bc_model.py`**：
  - `BCFormat` / `BC1Format` / `BC2Format` / `BC3Format` / `BC4Format` / `BC5Format`：定义各 BC 格式的端点位深与索引位深。
  - `BCBlockFeature`：4×4 块级别的可微 BC 压缩特征层，使用 STE（Straight-Through Estimator）量化。
  - `NeuralBCTextureModel`：与 UC 模型结构相同，但特征网格使用 BC 压缩版本。
- **`Tool.py`**：主入口 — `--train` 一键流水线 (UC → BC → BC-MLP) → `eval`。
- **`ntc_config.py`**：唯一配置入口，YAML schema 包括 `bc_format`、`model`、`uc_training`、`bc_training`、`bc_mlp_training`、`dataset`、`benchmark`。

---

## 构建与运行命令


### 直接调用 Tool.py

Tool.py的调用指南：
usage: Tool.py [-h] --config CONFIG [--train] [--ckpt CKPT]
               [--materials MATERIALS] [--vis-dir VIS_DIR]
               [--num-workers NUM_WORKERS] [--gpus GPUS]


```bash
# Train (一键流水线: UC → BC → BC-MLP, 4 worker parallel)
python Tool.py --config configs/bc1_bcf05k.yaml --train --num-workers 4

# Eval (4 worker parallel, reuse same --ckpt)
python Tool.py --config configs/bc1_bcf05k.yaml --ckpt checkpoints/<ts>/ --num-workers 4
```

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

dataset:
  root: dataset
  target_res: 256

model:
  feature_configs: [[512,8,3], [256,7,3], [128,6,3], [64,5,3]]
  hidden_dim: 16
  num_layers: 1
  filter: trilinear           # trilinear | tricubic
  half_pixel_offsets: [1, 3]  # 在哪些 feature grid 索引上施加半像素偏移

uc_training:
  total_iterations: 5000
  batch_res: 128
  lr_feat: 5.0e-2
  lr_mlp: 1.0e-3
  gamma: 0.9995

bc_training:
  total_iterations: 200000
  batch_res: 128
  lr_feat: 1.0e-2
  lr_mlp: 1.0e-3
  betas: [0.9, 0.999]

bc_mlp_training:
  total_iterations: 1000
  batch_res: 128
  lr_mlp: 1.0e-3
  betas: [0.9, 0.999]

benchmark:
  warmup_iters: 5
  timing_iters: 20
  mlp_param_bits: 16          # 假设 MLP 部署精度 (FP16 默认)
```

**注意**：`feature_configs`、`hidden_dim`、`num_layers` 改变后必须重新跑 UC 训练，因为模型形状变化。

---

## 开发约定与代码风格


### 实验纪律
a- 所有可调参数必须放在 YAML 中，不要硬编码到逻辑里。
b- train和eval的代码最后，必须调用write_done_json！！！

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
- **VRAM**：允许 ≤2× 增长，但必须有对应收益。
- **推理时延**：0.05 dB 的 PSNR 提升如果代价是 3× 推理耗时，不算胜利。更简单的方案更好。
- **基线参考**：约 480 s/material，20 材质 / 2 workers ≈ 80 min wall time per run。

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

b- train和eval的代码最后，必须调用write_done_json！！！

c- 禁止行为：不要使用 `tail -f`、`Get-Content -Wait`、循环轮询日志、等待训练完成。`train_start.py` 已内置启动检查。


## 执行流程
Train UC → Train BC → Train BC-MLP → Eval BC。