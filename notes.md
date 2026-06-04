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
