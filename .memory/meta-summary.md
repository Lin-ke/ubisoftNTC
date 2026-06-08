# Loop目的与指标

**当前目标**：训练验证探索（train-uc → train-bc → eval），围绕神经纹理压缩（NTC）实验框架，在保证模型可实验的前提下尽量提高PSNR，其次提高压缩率。

**核心指标**：
- `psnr_drop = psnr_原图 - psnr_bc`（**Primary**，越低越好）
- `inference_ms`（推理时延，越低越好）
- `compression_ratio = bc_bits / png_bits`（越低压缩越强）

**技术路线**：Train UC（全精度）→ 量化 → 量化感知训练（QAT，即 Train BC）→ Eval。

# 工作文件/路径/测试命令

| 模块 | 路径 | 说明 |
|------|------|------|
| 配置加载 | `ntc_config.py` | YAML 配置加载与校验（可修改） |
| 数据集 | `dataset.py` | MaterialDataset 加载器（Fixed，禁止修改） |
| UC 模型 | `ntc_model.py` | MipmapFeatureGrid + NeuralTextureModel（Fixed） |
| UC 训练辅助 | `ntc_train.py` | 参考采样、全图 PSNR 评估（Fixed） |
| BC 模型 | `ntc_bc_model.py` | BC 压缩模型（可修改） |
| BC 训练辅助 | `ntc_bc_train.py` | LOD 采样（Vaidyanathan）（可修改） |
| BC 推理 | `ntc_bc_inference.py` | 推理与可视化（Fixed） |
| BC 对比 | `ntc_compare.py` | BC 格式对比（Fixed） |
| 公共工具 | `ntc_utils.py` | 法线重建、图像保存、PSNR 计算（Fixed） |
| 主入口 | `Tool.py` | train-uc / train-bc / eval 三段式入口（可修改） |
| 一键启动 | `train_start.py` | 默认后台运行并自动检查启动日志（可修改） |
| 一键停止 | `train_stop.py` | 停止训练包装器（可修改） |
| 守护脚本 | `watchdog.ps1` | PowerShell 守护防止会话闲置 |
| 配置目录 | `configs/*.yaml` | 实验配置（可修改） |
| 检查点 | `checkpoints/` | 模型存档（按时间戳分子目录，不提交） |
| 数据集 | `dataset/` | 20 个 PBR 材质，不提交 |
| 笔记 | `notes.md` | 实验自由笔记，追加式，提交到 git |
| 结果 | `results.tsv` | 实验结果汇总，不提交 |

**关键测试/运行命令**：
```bash
# Train BC（4 worker 并行）
python Tool.py --config configs/bc1_bcf05k.yaml --train-bc --num-workers 4

# Train UC（可选，供 PSNR drop 比较）
python Tool.py --config configs/bc1_bcf05k.yaml --train-uc --num-workers 4

# Eval（复用 --ckpt）
python Tool.py --config configs/bc1_bcf05k.yaml --ckpt checkpoints/<ts>/ --num-workers 4

# 前台运行（train_start.py 默认后台）
python train_start.py --mode train-uc --foreground
python train_start.py --mode train-bc --ckpt checkpoints/<ts>/
python train_start.py --mode eval --ckpt checkpoints/<ts>/

# BC 推理可视化
python ntc_bc_inference.py --model-type bc --bc-format bc1 --mode full

# BC 格式对比
python ntc_compare.py --formats bc1 bc3

# 查看结果
grep -E "^\s*(psnr_drop|inference_ms|compression_ratio)" run.log
```

**验证方式**：无单元测试框架，依赖 UC 训练 PSNR 收敛（25~35 dB）、BC 训练 psnr_drop 合理（基线约 1~3 dB）、eval TSV 校验、可视化对比图、crash 处理与 results.tsv 记录。

# 开发相关架构

**数据流**：
```
dataset/ (20 PBR材质, 2K PNG: diff/nor_dx/arm)
  → MaterialDataset (build_mipmaps, target_res=256)
  → MipmapFeatureGrid (多分辨率特征金字塔, trilinear/tricubic采样)
  → MLP decoder (hidden_dim=16, num_layers=1)
  → 9通道输出 (RGB Albedo [0,1] + Normal [-1,1] + ARM [0,1])
```

**模型层次**：
- UC 模型：`MipmapFeatureGrid`（全精度可学习特征）+ `NeuralTextureModel`（拼接网格 → MLP）
- BC 模型：`BCBlockFeature`（4×4 块级 STE 量化）+ `BCMipmapFeatureGrid` + `NeuralBCTextureModel`（结构同 UC，特征层替换为 BC 压缩版本）

**BC 格式支持**：bc1 / bc2 / bc3 / bc4 / bc5，定义在 `BCFormat` 子类中，区分端点位深与索引位深。

**Pipeline 流程**：Train UC → checkpoints/\<ts\>/name.pth + config.yaml + uc_psnr.tsv → Train BC (QAT) → checkpoints/\<ts\>/bc_\<format\>/name.pth → Eval → \<ckpt\>/eval_\<config-stem\>.tsv (含 psnr_bc, psnr_drop, inference_ms, compression_ratio)

**配置 Schema**（YAML）：`bc_format`、`loss`（l1/mse）、`dataset`（root + target_res）、`model`（feature_configs + hidden_dim + num_layers + filter + half_pixel_offsets）、`uc_training`（总迭代+批分辨率+学习率+gamma）、`bc_training`（总迭代+批分辨率+学习率+betas）、`benchmark`（预热+计时迭代+mlp_param_bits）。

# 验证流

**图谱状态**：failed（执行超时，无输出超过120秒）。以下验证流基于 AGENTS.md 推断，依赖关系待确认。

**训练验证**：
1. UC 训练 → 检查单材质 PSNR 是否收敛到 25~35 dB 合理范围
2. BC 训练 → 检查 `psnr_drop` 是否为正且不过大（基线约 1~3 dB）
3. Eval TSV → 检查 `eval_*.tsv` 中各材质指标无异常（如 drop 为负或极大）
4. 可视化 → `--vis-dir` 参数生成预测 vs 参考对比图

**Crash 处理**：
- 简单错误（typo、缺 key）→ 修复后重跑（≤3 次尝试）
- 根本性问题 → 记 crash 到 `results.tsv`，`git reset --hard HEAD~1`，继续下一实验

**性能约束**：
- BC 训练耗时 > 2× 基线 → kill 并 discard
- VRAM ≤ 2× 增长，且必须有对应收益
- 0.05 dB PSNR 提升代价为 3× 推理耗时 → 不算胜利
- 基线约 480 s/material，20 材质 / 2 workers ≈ 80 min wall time per run

**启动检查机制**：`train_start.py` 默认后台运行并自动检查启动日志（等待10秒后扫描异常），启动后立即返回，无需额外监控。禁止使用 `tail -f`、`Get-Content -Wait`、循环轮询日志、等待训练完成。

# 项目注意事项

1. **实验纪律**：所有可调参数必须放在 YAML 中，不允许硬编码到逻辑里。
2. **模型变更规则**：`feature_configs`、`hidden_dim`、`num_layers` 改变后必须重新跑 UC 训练，因模型形状变化。
3. **Fixed 文件**（禁止修改）：`dataset.py`、`ntc_model.py`、`ntc_train.py`、`ntc_bc_inference.py`、`ntc_compare.py`、`ntc_utils.py`。
4. **可修改文件**（可实验调整）：`ntc_bc_model.py`、`ntc_bc_train.py`、`Tool.py`、`train_start.py`、`train_stop.py`、`ntc_config.py`、`configs/*.yaml`。
5. **Train/Eval 结尾要求**：train 和 eval 代码最后必须调用 `write_done_json()`。
6. **笔记**：`notes.md` 跨 session 共享，追加不覆盖。
7. **检查点不提交**：`checkpoints/` 目录内容不纳入 git。
8. **数据集不提交**：`dataset/` 目录内容不纳入 git。
9. **交互规范**：`train_start.py` 已内置启动检查，agents 直接运行即可，完成后用户会提醒。
10. **代码风格**：不允许添加任何注释（除非用户要求）。

<!-- META:BEGIN -->
已写入 `D:\ntc\.memory\meta-summary.md`。图谱状态为 failed（超时），已按此状态生成轻量摘要，未知验证流依赖标注为"待确认"。五章节齐全：Loop目的与指标、工作文件/路径/测试命令、开发相关架构、验证流、项目注意事项。
<!-- META:END -->
