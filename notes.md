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

1. **MLP 加一层** （注意并行数量可能较低一点）
   - 当前默认 `num_layers=1`（部分配置为 2），尝试加深 MLP 看看 PSNR 收益。
   - 注意点：参数量增长有限（MLP 本身很小），主要影响推理速度和显存占用。
1.1 纹理少一张
   - 现在默认是采样4张纹理，12个通道，考虑下8通道金字塔能行吗


2. **MLP 激活函数换 hard-Swish**
   - 出处：MobileNetV3 (2024) 提出，兼顾效果与推理效率。
   - 当前用的是 ReLU，hard-Swish 在移动端/低精度部署上可能有优势。
   - 注意点：训练时收敛性可能略有不同，需观察 PSNR 和训练稳定性。

3. **用 ASTC 结构编码**
   - ASTC (Adaptive Scalable Texture Compression) 是移动端 GPU 原生支持的压缩格式。
   - 当前实现的是 BC1~BC5，ASTC 支持更多 block size 和通道配置，可能获得更高质量或更高压缩比。
   - 注意点：ASTC 编码/解码逻辑比 BC 复杂得多，需评估实现成本。
   - 注意点：ASTC设计block选择相关的问题，需要先Train一个版本，然后用ASTC编码，后续根据此编码进行训练。所以你需要重写一个训练框架。

4. **三角波位置编码**
   - 三角波（triangle wave / sawtooth）作为位置编码，可能替代或补充当前的 UV 坐标输入。
   - 直觉：高频细节可能需要周期性的位置信号来辅助 MLP 学习。
   - 注意点：需在哪个阶段注入（UV 采样前？特征拼接后？）、频率如何选择。

## 2026-06-05 baseline run
- ckpt 005019 (BC6 UC, but model shape == BC1 yaml: same feat_configs/hidden/layers)
- BC1 baseline: drop=-2.24 (BC>UC because UC undertrained @10k iters w/ gamma=0.9995)
- Time: ~480s/material, 20 mats / 2 workers = ~80min wall per run. Plan accordingly.
- crepe_georgette is huge outlier (BC=40.18 vs UC=28.25 → drop -11.93)
- Game = maximize psnr_bc since UC fixed (we reuse same ckpt).
- Next: try MSE loss (PSNR is MSE-derived, should align gradient with metric).
