# Loop Agent Notebook

自由格式涂鸦本。给跑实验循环的 agent 用，不走 `results.tsv` 那种表格约束 ——
有什么想法、观察、坑、直觉、半成品猜想，都往这里塞。

## 怎么用

- **追加，不覆盖**。最新的写在文件末尾。
- **每次循环开始**：扫一眼最后 5~10 条，看看上几轮你/前任 agent 留了什么线索。
  常见用途：避免重复试已知失败的方向、串联跨实验的观察、回忆"为什么之前放弃了这条路"。
- **每次循环结束**（或想到值得记的事时）：追加一条。
- **不需要客气**。原始思路、半句话、问号都行。这不是给人看的报告，是给下个 LOOP 的你看的备忘。

## 与 results.tsv 的分工

| 写在 results.tsv | 写在 notes.md |
|---|---|
| 结构化指标 (psnr_drop, inference_ms, ...) | 假设、直觉、原因猜测 |
| commit-level 的 keep/discard/crash | "为什么试这个"、"下一步要不要试 X" |
| 一行能说完的 description | 多行的 reasoning chain |
| 必须每轮都有 | 想到才写 |

## 模板（参考用，不强制）

```
## YYYY-MM-DD HH:MM  <topic / commit short hash>
- hypothesis: ...
- observed:   ...
- explain:    ...
- next:       ...
```

---

<!-- 在下面追加新条目 -->

## 2026-06-04 19:03 待试验方向（用户给定）

1. **MLP 加一层/减一层** （注意并行数量可能较低/高一点）
   - 当前默认 `num_layers=1`（部分配置为 2），尝试加深 MLP 看看 PSNR 收益。
   - 注意点：参数量增长有限（MLP 本身很小），主要影响推理速度和显存占用。
1.1 **特征金字塔尺寸**
   - 现在默认是采样4张纹理，12个通道，能否用更大尺寸的特征同时减少采样？这样做对显存占用影响多少？

1. **MLP 激活函数换 hard-Swish**
   - 出处：MobileNetV3 (2024) 提出，兼顾效果与推理效率。
   - 当前用的是 ReLU，hard-Swish 在移动端/低精度部署上可能有优势。
   - 注意点：训练时收敛性可能略有不同，需观察 PSNR 和训练稳定性。

2. **用 ASTC 结构编码**
   - ASTC (Adaptive Scalable Texture Compression) 是移动端 GPU 原生支持的压缩格式。
   - 注意点：ASTC需要考虑

3. **三角波位置编码**
   - 三角波（triangle wave / sawtooth）作为位置编码，可能替代或补充当前的 UV 坐标输入。
   - 直觉：高频细节可能需要周期性的位置信号来辅助 MLP 学习。
   - 注意点：需在哪个阶段注入（UV 采样前？特征拼接后？）、频率如何选择。

4. **独立通道**
   - 把orm的金属度、粗糙度独立出来，只把roughness放到神经纹理流程中。
   - input：（12channel，如无更改），output：6 channel （normal * 2 + diff * 3 + ao * 1）
  

5. 探索最佳的参数集合（例如，UC train iters, BC train iters）。UC train我估计可以从psnr变化趋势得到一个较为肯定的答案。

6. 训练时**采样方法**的影响？三线性插值，如果是各向异性滤波呢？

7. RGBA8量化而非BC量化，考察影响

8. 

你可以参考的论文：
```markdown
## Ubisoft[1]:
1. 格式：神经纹理本身经过BC1压缩和QAT；
2. 参数：
   1. 4个神经纹理，大小；
   2. 两层MLP，12->16->16->6
   3. 网络入参就是4个纹理采样的结果，输出是6（AO*1, Noraml * 2, Diff * 3）。金属度和粗糙度不经过网络。

3. 【实机适用范围】树、桌子

## Intel[4]:
本身是ubisoft早期工作的一个改进，加了bc1、QAT,主要是coorpertive matrix，据称在网络宽度较大（>32）时能得到巨量提升。

## NVIDIA[2]:
hardGELU近似GELU；

做了很多模拟量化的操作

参数：两个金字塔
## AMD[3]:
一个endpoint网络，一个color网络，预测未压缩颜色 

feature texture：FP16，额外多10%做8bit QAT；

3 个隐藏层，每层 64 neurons，SELU 激活，输出 sigmoid（太大了！）

推理后获得BC结果（加载阶段）
## 腾讯[5]:
主要是优化NTC让其能在手机上跑。
技巧：
1. 移动端不用位置编码
2. hard-Swish激活函数，hardSwish(x) = x * saturate(x * (1.0f / 6.0f) + 0.5f)
3. 没有做QAT，训练完了直接量化成ASTC等等，据称没有差异。
4. 网络: 两个rgba8,16 bit + 位置编码等等，一共33->16->channels

结果：据称8gen1全屏NTC材质GPU额外耗时约 0.5ms 以内，视觉上没有较大差异
 

## 腾讯[6]:
Compression后应该做QAT，以解决压缩目标（重建参数图）和解压目标（重建原图）不一致的问题（压缩的好不等于用压缩参数复原的好）。但是对于有分区选择的压缩算法（如BC6），分区后相当于原本参数空间的子空间，不一定能复原的好。


这篇文章试图选择最优分区，方法比较复杂，主要思路就是对于
每种partition都记录一个评分，不断更新评分，并选择几个评分高的训练。因此这对于ASTC几千个分区模式的不太现实。
```

## 2026-06-06 实验1: num_layers 1→2 (BC1, L1)
- hypothesis: MLP depth 从 1 增加到 2 层（12→16→16→9），增加模型容量，PSNR 应有提升。参考 Ubisoft 论文也是 2 层 MLP。
- config: bc1_nl2.yaml（除 num_layers=2 外，其余与 bc1_bcf05k 基线一致）
- plan: Train UC → Train BC QAT → Eval，与 bc1_bcf05k 基线对比 psnr_drop
- status: 已完成 Train UC + Train BC + Eval ✅
- results: checkpoints/2026-06-06_183449/eval_bc1_nl2.tsv
- summary:
  - psnr_bc avg: 30.69 dB
  - psnr_drop avg: -0.49 dB (negative = BC > UC) ← 异常，8/20材质BC反而更好
  - inference_ms avg: 7.57 ms
  - compression_ratio avg: ~0.0057 (约175×压缩)
  - 8/20 drop<0 (BC更好), 12/20 drop>0 (UC更好)
  - Best: crepe_georgette drop=-7.67, Worst: patterned_brick_floor drop=1.74
- observed: QAT训练后，有8个材质的BC PSNR反而高于UC PSNR（drop为负）。这在1层MLP基线（bc1_bcf05k）中也有类似现象吗？需要对比确认。
- explain: 可能原因：(1) BC QAT从UC权重出发继续训练，在量化空间中找到了更好的局部最优；(2) UC训练10000 iter可能不完全收敛，BC再训10000 iter总迭代数多了一倍；(3) 2层MLP比1层容量大，QAT的效果更明显
- next: 与 bc1_bcf05k（num_layers=1）基线对比，看负drop是否是2层独有现象。如果1层也有负drop，说明是QAT普遍特性而非深度带来的。

## 2026-06-08 01:24 BC1 + MSE 基线实验 + train_start.py 配置修复

### train_start.py 配置自动检测修复
- problem: `train_start.py --mode train-bc --ckpt <ckpt>` 时，`--config` 硬编码默认为 `configs/bc1_bcf05k.yaml`（L1），导致 UC=MSE + BC=L1 混搭
- fix: `train_start.py:57-67` — 当 `--mode train-bc/eval/eval-uc` 且 `--ckpt` 提供时，检查 ckpt 目录下的 `config.yaml`，若存在且用户未显式指定 `--config`，自动替换为 ckpt 配置
- 效果: BC 训练自动继承 UC 训练的 loss / 模型参数 / BC 训练参数

### BC1 + MSE 基线实验
- config: bc1_mse.yaml (loss=mse, num_layers=1, hidden_dim=16, 4×feature grid)
- plan: 与 bc1_bcf05k（L1基线）对比，考察 MSE loss 对 PSNR 的影响
- status: UC ✅ → BC 训练中 (18/20, ckpt=2026-06-08_010508)
- hypothesis: MSE 直接优化 PSNR 相关目标，可能比 L1 得到更高 PSNR

## 2026-06-08 23:50 论文对齐重大修复：原图保留 + GT/feature 滤波解耦

### 背景
读 BCf 原文 (Weinreich 2024) Sec 5.1 / 4.3，发现本工程当前训练存在两处与论文不一致：
1. **原图被下采样**：yaml `dataset.target_res=256`，原生 2048×2048 PNG 被 bicubic 下采到 256，再从 256 构 mip 金字塔（仅 9 层）。论文要求**直接从 2K 构金字塔**（12 层：2048→1024→…→1）。
2. **GT 与神经特征滤波被绑定**：`Tool.py` 训练循环把 `model_params['filter']`（=`trilinear`）传给 `sample_reference` 当 GT 滤波器，导致 GT 走 bilinear；论文 Sec 5.1 明确 GT 永远 bicubic（双 mip 线性混合），而 trilinear 仅用于神经特征侧（Sec 4.3，硬件 sampler 模拟）。

### 修复
- `ntc_config.py:get_dataset_params` 强制 `target_res=None`，yaml 中遗留值给 WARN 后忽略
- `Tool.py:train_and_save_uc` / `train_bc_model` 训练循环里硬编码 `gt_filter='bicubic'`，与 `model.filter` 解耦
- `ntc_train.py:sample_reference` 仅加注释明确语义
- 启动信息加 `Filter: trilinear (feature) | GT filter: bicubic` 显示
- `dataset.py` 未改（`target_res=None` 时本就保留原图）

### 影响
- 训练时每条材质会保留 12 层 mip pyramid（2K→1），CPU RAM 占用 ≈3.8GB / 20 mat（preload）
- batch 仍是 `batch_res×batch_res` uv 窗口，但底层 GT 来自更高层的 mip → 高频细节学得到
- 旧 ckpt 不受影响；新跑的实验 PSNR 数值不可与历史 results.tsv 直接对比（GT 变了）
- 烟测 `smoke_10iter` 跑通：UC 10 iter 2.8s + BC 10 iter 2.7s（aerial_beach_02，单卡）

### TODO（后续）
- 重跑 bc1_bcf05k 基线，记录 PSNR 变化作为新 baseline
- batch 采样目前还是 `_sample_uv_and_lod` 的小窗口（`batch_res/ref_w` ≈ 6%），论文是覆盖整个 [0,1]² 的均匀 grid，可能下一步也要修
- max_useful_lod 之类的剪枝逻辑要重新评估：原来按 9 层 mip 调的，现在 12 层
