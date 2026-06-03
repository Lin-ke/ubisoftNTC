# BC 格式推导与神经模拟实现总结

## 1. BC (Block Compression) 格式总览

BC 是 DirectX 标准的 GPU 硬件纹理压缩格式族。所有 BC 格式以固定的 4x4 像素块为基本压缩单元，通过端点和索引的重参数化实现压缩。

### 1.1 六种 BC 格式对比

| 格式  | 别名       | bpp  | 通道         | 块大小(bits) | 端点精度         | 索引精度       | 分区 |
|-------|-----------|------|-------------|-------------|-----------------|---------------|------|
| BC1   | DXT1      | 4    | RGB (+1-bit A) | 64         | RGB565 (5/6/5-bit) | 2-bit (4级)  | 无   |
| BC2   | DXT3      | 8    | RGBA        | 128         | alpha:4-bit/pixel<br>color:RGB565 | alpha:explicit<br>color:2-bit | 无   |
| BC3   | DXT5      | 8    | RGBA        | 128         | alpha:8-bit×2<br>color:RGB565 | alpha:3-bit(8级)<br>color:2-bit(4级) | 无   |
| BC4   | RGTC1/ATI1| 4    | R (单通道)   | 64          | 8-bit UNORM ×2  | 3-bit (8级)   | 无   |
| BC5   | RGTC2/ATI2| 8    | RG (双通道)  | 128         | 8-bit UNORM ×2×2| 3-bit (8级)   | 无   |
| BC6   | BC6H      | 8    | RGB (HDR)   | 128         | FP16 ×4         | 3-bit (8级)   | 32种  |

> **注**: 表中 "×2" 表示每块有2个端点; BC5有2个独立BC4块, 每个有2个8-bit端点, 因此共4个8-bit端点.

### 1.2 通用块压缩框架

所有 BC 格式遵循统一框架: 将4x4像素块参数化为 **端点向量** + **像素索引** + **可选分区**。

**通用解码公式 (单线段)**:
```
pixel_j = endpoint_0 + w_index(j) * (endpoint_1 - endpoint_0)
```
其中 `pixel_j` 为第j个像素(共16个), `w_index(j)` ∈ [0, 1] 由索引值确定。

---

## 2. BC1 (DXT1) — 已实现

### 2.1 位布局 (64 bits = 8 bytes)

```
Byte 0-1:  c0 (RGB565)  — 16 bits
Byte 2-3:  c1 (RGB565)  — 16 bits
Byte 4-7:  16×2-bit 索引 — 32 bits
```

**RGB565 位布局**:
```
c0_lo, c0_hi:
  15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
  [  B[4:0]  ][  G[5:3] ][     R[4:0]     ]
```

解码:
```
R = ((c >> 11) & 0x1F) / 31.0 * 255
G = ((c >> 5)  & 0x3F) / 63.0 * 255
B = (c         & 0x1F) / 31.0 * 255
```

### 2.2 色板解码 (4色模式, c0 > c1)

| index | 权重 w | 公式                    |
|-------|--------|------------------------|
| 0     | 0      | color_0                |
| 1     | 1/3    | 2/3*color_0 + 1/3*color_1 |
| 2     | 2/3    | 1/3*color_0 + 2/3*color_1 |
| 3     | 1      | color_1                |

等价于: `color = lerp(c0, c1, index/3)`

### 2.3 神经模拟 (已实现)

```python
endpoints: [bh, bw, 2, C]    # 2个端点, C为特征维度
indices:  [bh, bw, 16]       # 共享索引, sigmoid → [0,1]

y = e0 + sigmoid(indices) * (e1 - e0)   # 连续可微版本
```

**量化**:
- 端点: 5-bit 均匀量化 (32级): `round(eps_norm * 31) / 31`
- 索引: 2-bit → 4级: `round(prob * 3) / 3` → inverse sigmoid

---

## 3. BC2 (DXT3) — 新实现

### 3.1 硬件位布局 (128 bits = 16 bytes)

```
Alpha 块 (64 bits):
Byte 0-7:  16 pixels × 4-bit alpha 值 (pixel 0: bits 0-3, pixel 1: bits 4-7, ...)

Color 块 (64 bits):
Byte 8-9:   c0 (RGB565)  — 16 bits
Byte 10-11: c1 (RGB565)  — 16 bits
Byte 12-15: 16×2-bit 索引 — 32 bits
```

### 3.2 Alpha 解码

```
alpha_j = a_4bit / 15.0    (显式存储, 无插值)
```
- 每个像素独立存储4-bit alpha: 0, 1, ..., 15
- 无端点, 无插值, 纯粹的4-bit均匀量化

### 3.3 神经模拟

BC2的核心特征: **alpha精度高于BC1**(16级 vs 4级), **但没有插值**。

神经模拟策略:
- **架构**: 2端点 + 共享索引 (与BC1相同的参数结构)
- **端点量化**: 前 C//2 通道 (alpha-like) 用 8-bit (256级), 后 C-C//2 通道 (color-like) 用 5-bit (32级)
- **索引量化**: 4-bit → 16级插值 (精度提升4倍 vs BC1的4级)
- **解码**: 分通道组独立 lerp, 但共享索引

```python
split = C // 2
y_alpha = e0[:split] + x * (e1[:split] - e0[:split])     # 16级索引, 8-bit端点
y_color = e0[split:] + x * (e1[split:] - e0[split:])     # 16级索引, 5-bit端点
```

**设计权衡**: 由于神经参数结构不支持"每像素独立参数", 我们用 **高精度索引 (16级) 弥补缺少的显式alpha独立性**。16级插值近似了BC2中 alpha 的独立4-bit量化精度。

---

## 4. BC3 (DXT5) — 新实现

### 4.1 硬件位布局 (128 bits = 16 bytes)

```
Alpha 块 (64 bits):
Byte 0:     alpha_0 (8-bit)
Byte 1:     alpha_1 (8-bit)
Byte 2-7:   16×3-bit 索引 (共48 bits = 6 bytes)

Color 块 (64 bits):
Byte 8-9:   c0 (RGB565)
Byte 10-11: c1 (RGB565)
Byte 12-15: 16×2-bit 索引
```

### 4.2 Alpha 插值模式

BC3 alpha 块是 BC4 格式 (见第5节) 的子集:

**模式1: alpha_0 > alpha_1 (8级全插值)**

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|-------|---|---|---|---|---|---|---|---|
| alpha | alpha_0 | (6α₀+α₁)/7 | (5α₀+2α₁)/7 | (4α₀+3α₁)/7 | (3α₀+4α₁)/7 | (2α₀+5α₁)/7 | (α₀+6α₁)/7 | alpha_1 |

等价于: `alpha = lerp(alpha_0, alpha_1, index/7)`

**模式2: alpha_0 ≤ alpha_1 (6级插值 + 2特例)**

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|-------|---|---|---|---|---|---|---|---|
| alpha | alpha_0 | (4α₀+α₁)/5 | (3α₀+2α₁)/5 | (2α₀+3α₁)/5 | (α₀+4α₁)/5 | alpha_1 | **0.0** | **1.0** |

前6级等价于: `alpha = lerp(alpha_0, alpha_1, index/5)`
index 6 = 0.0 (完全透明), index 7 = 1.0 (完全不透明)

### 4.3 颜色解码

与 BC1 完全相同: RGB565端点 + 2-bit索引 (4级插值)。

### 4.4 神经模拟

BC3的核心特征: **alpha块 = BC4 (8级插值)**, color块 = BC1 (4级插值)。

神经模拟策略:
- **架构**: 2端点 + 共享索引
- **端点量化**: 前 C//2 (alpha) 用 8-bit, 后 C-C//2 (color) 用 5-bit
- **索引量化**: 3-bit → 8级插值 (匹配 BC4-alpha 的插值范围)
- **解码**: 分通道组独立 lerp

```python
split = C // 2
y_alpha = e0[:split] + x * (e1[:split] - e0[:split])     # 8级索引, 8-bit端点
y_color = e0[split:] + x * (e1[split:] - e0[split:])     # 8级索引, 5-bit端点
```

**简化说明**: 神经模拟中 α₀ ≤ α₁ 的 6+2 特例模式未单独处理 — 3-bit/8级的均匀量化即已覆盖该精度表达力。

---

## 5. BC4 (RGTC1 / ATI1) — 新实现

### 5.1 硬件位布局 (64 bits = 8 bytes)

```
Byte 0:     red_0 (8-bit UNORM)
Byte 1:     red_1 (8-bit UNORM)
Byte 2-7:   16×3-bit 索引 (48 bits)
```

### 5.2 解码公式

与 BC3 alpha 块完全相同的插值逻辑:

| 条件         | 插值模式     | 索引范围   | 输出范围           |
|-------------|-------------|-----------|-------------------|
| red_0 > red_1 | 8级全插值    | 0..7     | [red_0, red_1]   |
| red_0 ≤ red_1 | 6级插值+2特例 | 0..7     | [red_0, red_1] ∪ {0.0, 1.0} |

UNORM 存储: 硬件存储 8-bit 整数 r ∈ [0, 255], 使用时转换 red = r / 255.0

### 5.3 神经模拟

BC4的核心特征: **单通道, 8-bit端点, 3-bit索引 (8级插值)**。

神经模拟策略:
- **架构**: 2端点 + 共享索引
- **端点量化**: 8-bit UNORM (256级): `round(eps_norm * 255) / 255`
- **索引量化**: 3-bit → 8级: `round(prob * 7) / 7`
- **解码**: BC1-style lerp

```python
y = e0 + sigmoid(indices) * (e1 - e0)
```

---

## 6. BC5 (RGTC2 / ATI2) — 新实现

### 6.1 硬件位布局 (128 bits = 16 bytes)

```
Red 块 (64 bits):   与 BC4 完全相同
  Byte 0:     red_0 (8-bit)
  Byte 1:     red_1 (8-bit)
  Byte 2-7:   16×3-bit 红色索引

Green 块 (64 bits): 与 BC4 完全相同
  Byte 8:     green_0 (8-bit)
  Byte 9:     green_1 (8-bit)
  Byte 10-15: 16×3-bit 绿色索引
```

两个 BC4 块完全独立, 各自拥有自己的端点和索引。

### 6.2 神经模拟

BC5的核心特征: **两个独立BC4通道**。

对于神经特征 (任意C维), BC4 和 BC5 在解码逻辑上完全等价:
- 两者都是 2端点 + 3-bit索引 (8级插值)
- 端点都是 8-bit UNORM
- 区别仅在硬件层面: BC4=1通道, BC5=2通道独立编码

神经模拟:
- **端点量化**: 8-bit (256级)
- **索引量化**: 3-bit (8级)
- **解码**: 与 BC4 相同

```python
y = e0 + sigmoid(indices) * (e1 - e0)
```

---

## 7. BC6 (BC6H) — 已实现

### 7.1 硬件架构

BC6 是为 HDR 设计的, 使用 FP16 存储的复杂格式:

- **位布局** (128 bits): Mode bits (2-5 bits) + 端点数据 (FP16) + 分区索引 (5 bits) + 像素索引 (16×3-bit, 可选4-bit)
- **14种模式**: 通过 mode bits 选择, 不同模式有不同端点数量和索引位深
- **32种分区**: 将4x4块分为两个子集
- **双线段**: 每个子集有独立的2个端点

### 7.2 解码步骤

```
Step 1: 读取 mode bits → 确定模式 → 确定各参数位位置
Step 2: 读取分区索引 → 查表得到16个像素中哪些属于子集0, 哪些属于子集1
Step 3: 读取端点原始位 (6-10 bit有符号整数) → 转换为 FP16
Step 4: 读取16个像素索引 (3-bit或4-bit) → 每个子集内做 lerp 插值
Step 5: partition_blend: pk * line1 + (1-pk) * line2
```

### 7.3 神经模拟 (已实现)

```python
# 4端点双线段, 32分区, FP16模拟
y = simulate_fp16(pk * line1 + (1-pk) * line2)

# FP16模拟: 整数位模式 → FP16浮点值
h = clamp(floor((y-1)/1024) - 1, min=0)
w = (2^(h-14)) * (y/1024 - h)

# 量化
endpoints: 6-bit → 64级: round(eps_norm * 63) / 63
indices:   3-bit → 8级:  round(prob * 7) / 7 → inv_sigmoid
```

---

## 8. 神经模拟设计决策总结

### 8.1 统一 lerp 框架

所有 BC 格式的神经模拟都使用统一的连续可微解码:
```
pixel_value = endpoint_0 + sigmoid(index) * (endpoint_1 - endpoint_0)
```

这与硬件 BC1 的 `lerp(c0, c1, idx/3)` 形式一致, 但使用 `sigmoid` 替代离散索引以确保可微性。

### 8.2 精度差异化策略

| 格式 | 端点量化 (bits/levels)              | 索引量化 (bits/levels)    | 解码结构           |
|------|-------------------------------------|--------------------------|-------------------|
| BC1  | 5-bit / 32级 (全通道)              | 2-bit / 4级             | 单线段, 无通道分组  |
| BC2  | 8-bit/256级 (alpha) + 5-bit/32级 (color) | 4-bit / 16级        | 通道分组独立 lerp |
| BC3  | 8-bit/256级 (alpha) + 5-bit/32级 (color) | 3-bit / 8级         | 通道分组独立 lerp |
| BC4  | 8-bit / 256级 (全通道)             | 3-bit / 8级             | 单线段, 无通道分组  |
| BC5  | 8-bit / 256级 (全通道)             | 3-bit / 8级             | 单线段, 无通道分组  |
| BC6  | 6-bit / 64级 (全通道, 4端点)       | 3-bit / 8级             | 双线段+32分区混合  |

### 8.3 通道分组 (BC2/BC3)

BC2 和 BC3 实现了通道分组 (split = C // 2):
- **Alpha 组** (前 C//2 通道): 模仿硬件中的独立 alpha 块 → 更高精度的端点/索引
- **Color 组** (后 C-C//2 通道): 模仿硬件中的 BC1 color 块 → BC1 级精度

两组共享同一个索引参数 `[bh, bw, 16]`, 但端点分别量化。这种设计:
- 保持了参数效率 (不增加额外索引参数)
- 通过端点精度的差异化模拟了双块结构的高精度特征
- 可微训练中梯度可正常流通

### 8.4 BC4 ≈ BC5 (神经模拟层面)

对于神经特征 (任意 C 维), BC4 (1通道) 和 BC5 (2通道独立BC4) 在解码逻辑上等价:
- 两者都是 8-bit端点 + 3-bit索引
- 硬件层面 BC4=4bpp, BC5=8bpp; 但在神经模拟中 bpp 由特征维度和网格分辨率共同决定
- 实现了两个独立类以保持语义清晰, 便于未来发展差异化处理

### 8.5 未实现的硬件细节

以下硬件行为在神经模拟中被简化:
1. **BC1 3色透明模式** (c0 ≤ c1 时 index 3 = transparent): 神经特征不需要透明度模拟
2. **BC3/BC4 6+2 特例模式** (α₀ ≤ α₁): 8级均匀量化已覆盖该精度表达力
3. **RGB565 颜色空间转换**: 神经特征为抽象向量, 不涉及颜色空间
4. **BC6 的14种模式动态选择**: 固定使用 mode-agnostic 的4端点+32分区方案

---

## 9. BC 格式参数对比表

| 属性              | BC1    | BC2         | BC3         | BC4    | BC5    | BC6       |
|-------------------|--------|-------------|-------------|--------|--------|-----------|
| 块大小            | 64 bit | 128 bit     | 128 bit     | 64 bit | 128 bit| 128 bit   |
| bpp               | 4      | 8           | 8           | 4      | 8      | 8         |
| 硬件通道          | RGB    | A(4b/px) + RGB | A(BC4) + RGB | R    | R+G  | RGB(HDR)  |
| 端点数量 (硬件)   | 2      | 2           | 2           | 2      | 4 (2×2)| 4         |
| 端点精度 (硬件)   | 5/6/5b | α:4b, c:5/6/5b | α:8b, c:5/6/5b | 8b | 8b×2 | FP16      |
| 索引精度 (硬件)   | 2b     | α:4b, c:2b  | α:3b, c:2b  | 3b     | 3b×2   | 3b        |
| 插值级别          | 4      | α:16, c:4   | α:8, c:4    | 8      | 8      | 8         |
| 分区数            | 1      | 1           | 1           | 1      | 1      | 32        |
| 神经端点数        | 2      | 2           | 2           | 2      | 2      | 4         |
| 神经端点Lv        | 32 (5b)| α:256, c:32 | α:256, c:32 | 256 (8b)| 256   | 64 (6b)   |
| 神经索引Lv        | 4 (2b) | 16 (4b)     | 8 (3b)      | 8 (3b)  | 8 (3b) | 8 (3b)    |
| 通道分组          | 无     | C//2 拆分   | C//2 拆分   | 无      | 无     | 无        |
| use_partitions    | False  | False       | False       | False   | False  | True      |
| simulate_decode   | 直通   | 直通        | 直通        | 直通    | 直通   | FP16重解释 |

---

## 10. 代码结构

```
ntc_bc_model.py:
  BCFormat                  ← 基类 (协议)
  ├── BC6Format              ← BC6: FP16模拟, 4端点, 32分区
  ├── BC1Format              ← BC1: 5b端点, 2b索引
  ├── BC2Format              ← BC2: 8b/5b端点分通道, 4b索引
  ├── BC3Format              ← BC3: 8b/5b端点分通道, 3b索引
  ├── BC4Format              ← BC4: 8b端点, 3b索引
  └── BC5Format              ← BC5: 8b端点, 3b索引 (等价于BC4)

  BCBlockFeature:
    forward()                ← 路由到 _forward_bc{1..6}
    _forward_bc1()           ← BC1: 单线段2端点 lerp
    _forward_bc2()           ← BC2: 通道分组 lerp (alpha + color)
    _forward_bc3()           ← BC3: 通道分组 lerp (alpha + color)
    _forward_bc4()           ← BC4: 单线段 lerp
    _forward_bc5()           ← BC5: 单线段 lerp
    _forward_bc6()           ← BC6: 双线段 + 分区混合

  _FORMATS = {bc1, bc2, bc3, bc4, bc5, bc6}

ntc_bc6_train.py:
  compress_to_bc1/2/3/4/5/6  ← 各格式的初始化压缩算法
  compress_to_bc()           ← 统一分发器
  init_bc_model_from_unconstrained() ← 预热→BC参数迁移
```

---

## 参考资料

1. Microsoft DirectX BC 格式规范: https://docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression
2. BC6H/BC7 详细位布局: K.Neubelt, "Real-Time BC6H Compression on GPU", GPU Zen 2, 2019
3. "Real-Time Neural Materials using Block-Compressed Features", 2024 (本文工作参考论文)
4. BCn 编码算法综述: J.M.P. van Waveren, "Real-Time DXT Compression", id Software, 2006
