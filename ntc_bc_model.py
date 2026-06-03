"""
BC 块压缩特征模型 —— 支持 BC6 和 BC1 两种格式。

架构:
  BCFormat (protocol)          ← 定义解码行为
    ├── BC6Format               ← BC6H: 4端点, 双线段, 32分区, FP16模拟
    └── BC1Format               ← BC1/DXT1: 2端点, 4级插值, 无分区

  BCBlockFeature (nn.Module)   ← 通用 4×4 块特征层, 组合 BCFormat
  BCMipmapFeatureGrid          ← 多分辨率 mipmap 金字塔
  NeuralBCTextureModel         ← 完整神经材质模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ntc_bc6_partitions import PARTITIONS


# ============================================================
# BC6 专用: FP16 位重解释模拟
# ============================================================

def simulate_fp16(y):
    """BC6 Eq.9: 整数位模式 → FP16 浮点值 (位重解释)."""
    h = torch.clamp(torch.floor((y - 1) / 1024) - 1, min=0)
    w = (2 ** (h - 14)) * (y / 1024 - h)
    return w


def inverse_simulate_fp16(w):
    """simulate_fp16 的逆函数: 给定 FP16 w, 反推 BC6 整数 y."""
    y = torch.zeros_like(w)
    for hi in range(63):
        y_candidate = 1024.0 * (hi + w * (2.0 ** (14 - hi)))
        h_check = torch.clamp(torch.floor((y_candidate - 1) / 1024) - 1, min=0)
        y = torch.where(h_check == hi, y_candidate, y)
    return y


# ============================================================
# BC 格式协议类 —— 定义各 BC 格式的解码行为差异
# ============================================================

class BCFormat:
    """BC 压缩格式的行为协议。

    子类必须实现的方法:
        - simulate_decode(y)       : 位重解释 / 后处理
        - inverse_simulate_decode(w): 反向后处理 (用于初始化)
        - get_endpoint_init_scale() : 端点初始化 scale
        - get_endpoint_count()      : 每块端点数
        - use_partitions()          : 是否使用分区模式
        - quantize_endpoints(eps)   : 端点量化
        - quantize_indices(raw_idx) : 索引量化
        - get_export_params(...)    : 导出参数字典
    """

    def simulate_decode(self, y):
        """解码后处理: BC6 做 FP16 重解释, BC1 直通."""
        return y

    def inverse_simulate_decode(self, w):
        """反向后处理: 用于 warmup→BC init 值域转换."""
        return w

    def get_endpoint_init_scale(self, feature_dim):
        """端点初始化 std 系数."""
        return 0.01

    def get_endpoint_count(self):
        """每个 4×4 块有几条线段的端点."""
        return 2

    def use_partitions(self):
        """是否使用分区模式."""
        return False

    def get_partition_count(self):
        """分区模式数量."""
        return 1

    def get_endpoint_bits(self, feature_dim):
        """每通道的端点量化 bit 数, 返回 list[int] 长度为 feature_dim."""
        return [8] * feature_dim

    def get_index_bits(self):
        """索引量化 bit 数."""
        return 3

    def quantize_endpoints(self, endpoints):
        """量化端点到位宽, 原地修改."""
        raise NotImplementedError

    def quantize_indices(self, raw_indices):
        """量化索引到位宽, 原地修改."""
        raise NotImplementedError

    def get_export_params(self, endpoints, raw_indices, partition_ids, blocks_h, blocks_w):
        """导出量化参数."""
        raise NotImplementedError


# ============================================================
# BC6 格式实现
# ============================================================

class BC6Format(BCFormat):
    """BC6H HDR 格式: 4端点, 双线段, 32分区, FP16 位重解释."""

    def simulate_decode(self, y):
        return simulate_fp16(y)

    def inverse_simulate_decode(self, w):
        return inverse_simulate_fp16(w)

    def get_endpoint_init_scale(self, feature_dim):
        return 1000.0

    def get_endpoint_count(self):
        return 4

    def use_partitions(self):
        return True

    def get_partition_count(self):
        return 32

    def get_endpoint_bits(self, feature_dim):
        return [6] * feature_dim

    def get_index_bits(self):
        return 3

    def quantize_endpoints(self, endpoints):
        eps = endpoints.data
        eps_min, eps_max = eps.min(), eps.max()
        if eps_max - eps_min > 1e-8:
            eps_norm = (eps - eps_min) / (eps_max - eps_min)
            eps_quant = (eps_norm * 63.0).round().clamp(0, 63) / 63.0
            eps.copy_(eps_quant * (eps_max - eps_min) + eps_min)

    def quantize_indices(self, raw_indices):
        prob = torch.sigmoid(raw_indices.data)
        q = (prob * 7.0).round().clamp(0, 7) / 7.0
        safe = q.clamp(1e-6, 1.0 - 1e-6)
        logit = torch.log(safe / (1.0 - safe))
        raw_indices.data.copy_(logit.clamp(-10.0, 10.0))

    def get_export_params(self, endpoints, raw_indices, partition_ids, blocks_h, blocks_w):
        eps = endpoints.data
        eps_min, eps_max = eps.min(), eps.max()
        if eps_max - eps_min < 1e-8:
            eps_quant = torch.zeros_like(eps).to(torch.uint8)
        else:
            eps_norm = (eps - eps_min) / (eps_max - eps_min)
            eps_quant = (eps_norm * 63).round().clamp(0, 63).to(torch.uint8)

        prob = torch.sigmoid(raw_indices.data)
        idx_quant = (prob * 7).round().clamp(0, 7).to(torch.uint8)

        return {
            'endpoints': eps_quant,
            'endpoints_min': eps_min,
            'endpoints_max': eps_max,
            'indices': idx_quant,
            'partition_ids': partition_ids.data.to(torch.uint8),
            'resolution': (
                blocks_w * 4,
                blocks_h * 4,
            ),
        }


# ============================================================
# BC1 格式实现
# ============================================================

class BC1Format(BCFormat):
    """BC1/DXT1 LDR 格式: 2端点, 4级插值, 无分区, 无 FP16 模拟."""

    def simulate_decode(self, y):
        return y

    def inverse_simulate_decode(self, w):
        return w

    def get_endpoint_init_scale(self, feature_dim):
        return 0.01

    def get_endpoint_count(self):
        return 2

    def use_partitions(self):
        return False

    def get_partition_count(self):
        return 1

    def get_endpoint_bits(self, feature_dim):
        # BC1 RGB565: R=5, G=6, B=5, 重复到 feature_dim
        pattern = [5, 6, 5]
        return [(pattern[i % 3]) for i in range(feature_dim)]

    def get_index_bits(self):
        return 2  # BC1: 2-bit, 4 级插值

    def quantize_endpoints(self, endpoints):
        # BC1 RGB565: R=5bit(31级), G=6bit(63级), B=5bit(31级)
        # 逐通道按正确 bit 数量化 (与 QAT 前向路径对齐)
        eps = endpoints.data  # [bh, bw, 2, C]
        C = eps.shape[-1]
        eps_bits = self.get_endpoint_bits(C)  # [5,6,5,5,6,5,...]
        eps_min = eps.min()
        eps_max = eps.max()
        if eps_max - eps_min > 1e-8:
            eps_norm = (eps - eps_min) / (eps_max - eps_min)
            eps_quant = eps_norm.clone()
            for ch in range(C):
                levels = (1 << eps_bits[ch]) - 1  # 5bit→31, 6bit→63
                eps_quant[..., ch] = (eps_norm[..., ch] * levels).round().clamp(0, levels) / levels
            eps.copy_(eps_quant * (eps_max - eps_min) + eps_min)

    def quantize_indices(self, raw_indices):
        prob = torch.sigmoid(raw_indices.data)
        q = (prob * 3.0).round().clamp(0, 3) / 3.0
        safe = q.clamp(1e-6, 1.0 - 1e-6)
        logit = torch.log(safe / (1.0 - safe))
        raw_indices.data.copy_(logit.clamp(-10.0, 10.0))

    def get_export_params(self, endpoints, raw_indices, partition_ids, blocks_h, blocks_w):
        eps = endpoints.data
        eps_min, eps_max = eps.min(), eps.max()
        if eps_max - eps_min < 1e-8:
            eps_quant = torch.zeros_like(eps).to(torch.uint8)
        else:
            eps_norm = (eps - eps_min) / (eps_max - eps_min)
            eps_quant = (eps_norm * 31).round().clamp(0, 31).to(torch.uint8)

        prob = torch.sigmoid(raw_indices.data)
        idx_quant = (prob * 3).round().clamp(0, 3).to(torch.uint8)

        return {
            'endpoints': eps_quant,
            'endpoints_min': eps_min,
            'endpoints_max': eps_max,
            'indices': idx_quant,
            'resolution': (
                blocks_w * 4,
                blocks_h * 4,
            ),
        }


# ============================================================
# BC2 格式实现 (DXT3)
# ============================================================

class BC2Format(BCFormat):
    """BC2/DXT3 RGBA 格式: 双块结构 (显式4-bit alpha + BC1 color).

    硬件结构:
      Alpha块: 64 bits = 16像素 × 4-bit 显式alpha值 (直接存储, 无插值)
      Color块: 64 bits = 2×RGB565端点 + 16×2-bit索引 (同 BC1)

    神经模拟: alpha 通道 4-bit 索引, color 通道 2-bit 索引.
    """

    def simulate_decode(self, y):
        return y

    def inverse_simulate_decode(self, w):
        return w

    def get_endpoint_init_scale(self, feature_dim):
        return 0.01

    def get_endpoint_count(self):
        return 2

    def use_partitions(self):
        return False

    def get_partition_count(self):
        return 1

    def get_endpoint_bits(self, feature_dim):
        # alpha (前一半通道): 8-bit; color (后一半): 5-bit
        split = feature_dim // 2
        bits = []
        for i in range(feature_dim):
            bits.append(8 if i < split else 5)
        return bits

    def get_index_bits(self):
        return 4  # BC2: 4-bit, 16 级插值

    def quantize_endpoints(self, endpoints):
        eps = endpoints.data  # [bh, bw, 2, C]
        C = eps.shape[-1]
        split = C // 2
        eps_min, eps_max = eps.min(), eps.max()
        if eps_max - eps_min > 1e-8:
            eps_norm = (eps - eps_min) / (eps_max - eps_min)
            eps_alpha = (eps_norm[..., :split] * 255.0).round().clamp(0, 255) / 255.0
            eps_color = (eps_norm[..., split:] * 31.0).round().clamp(0, 31) / 31.0
            eps_quant = torch.cat([eps_alpha, eps_color], dim=-1)
            eps.copy_(eps_quant * (eps_max - eps_min) + eps_min)

    def quantize_indices(self, raw_indices):
        prob = torch.sigmoid(raw_indices.data)
        q = (prob * 15.0).round().clamp(0, 15) / 15.0
        safe = q.clamp(1e-6, 1.0 - 1e-6)
        logit = torch.log(safe / (1.0 - safe))
        raw_indices.data.copy_(logit.clamp(-10.0, 10.0))

    def get_export_params(self, endpoints, raw_indices, partition_ids, blocks_h, blocks_w):
        eps = endpoints.data
        eps_min, eps_max = eps.min(), eps.max()
        if eps_max - eps_min < 1e-8:
            eps_quant = torch.zeros_like(eps).to(torch.uint8)
        else:
            C = eps.shape[-1]
            split = C // 2
            eps_norm = (eps - eps_min) / (eps_max - eps_min)
            eps_alpha = (eps_norm[..., :split] * 255).round().clamp(0, 255).to(torch.uint8)
            eps_color = (eps_norm[..., split:] * 31).round().clamp(0, 31).to(torch.uint8)
            eps_quant = torch.cat([eps_alpha, eps_color], dim=-1)

        prob = torch.sigmoid(raw_indices.data)
        idx_quant = (prob * 15).round().clamp(0, 15).to(torch.uint8)

        return {
            'endpoints': eps_quant,
            'endpoints_min': eps_min,
            'endpoints_max': eps_max,
            'indices': idx_quant,
            'resolution': (
                blocks_w * 4,
                blocks_h * 4,
            ),
        }


# ============================================================
# BC3 格式实现 (DXT5)
# ============================================================

class BC3Format(BCFormat):
    """BC3/DXT5 RGBA 格式: 双块结构 (BC4-style alpha + BC1 color).

    Alpha通道 (前 C//2): 端点 8-bit, 索引 3-bit (8级插值)
    Color通道 (后): 端点 5-bit, 索引 3-bit
    """

    def simulate_decode(self, y):
        return y

    def inverse_simulate_decode(self, w):
        return w

    def get_endpoint_init_scale(self, feature_dim):
        return 0.01

    def get_endpoint_count(self):
        return 2

    def use_partitions(self):
        return False

    def get_partition_count(self):
        return 1

    def get_endpoint_bits(self, feature_dim):
        split = feature_dim // 2
        return [8 if i < split else 5 for i in range(feature_dim)]

    def get_index_bits(self):
        return 3  # BC3: 3-bit, 8 级插值

    def quantize_endpoints(self, endpoints):
        eps = endpoints.data
        C = eps.shape[-1]
        split = C // 2
        eps_min, eps_max = eps.min(), eps.max()
        if eps_max - eps_min > 1e-8:
            eps_norm = (eps - eps_min) / (eps_max - eps_min)
            eps_alpha = (eps_norm[..., :split] * 255.0).round().clamp(0, 255) / 255.0
            eps_color = (eps_norm[..., split:] * 31.0).round().clamp(0, 31) / 31.0
            eps_quant = torch.cat([eps_alpha, eps_color], dim=-1)
            eps.copy_(eps_quant * (eps_max - eps_min) + eps_min)

    def quantize_indices(self, raw_indices):
        prob = torch.sigmoid(raw_indices.data)
        q = (prob * 7.0).round().clamp(0, 7) / 7.0
        safe = q.clamp(1e-6, 1.0 - 1e-6)
        logit = torch.log(safe / (1.0 - safe))
        raw_indices.data.copy_(logit.clamp(-10.0, 10.0))

    def get_export_params(self, endpoints, raw_indices, partition_ids, blocks_h, blocks_w):
        eps = endpoints.data
        eps_min, eps_max = eps.min(), eps.max()
        if eps_max - eps_min < 1e-8:
            eps_quant = torch.zeros_like(eps).to(torch.uint8)
        else:
            C = eps.shape[-1]
            split = C // 2
            eps_norm = (eps - eps_min) / (eps_max - eps_min)
            eps_alpha = (eps_norm[..., :split] * 255).round().clamp(0, 255).to(torch.uint8)
            eps_color = (eps_norm[..., split:] * 31).round().clamp(0, 31).to(torch.uint8)
            eps_quant = torch.cat([eps_alpha, eps_color], dim=-1)

        prob = torch.sigmoid(raw_indices.data)
        idx_quant = (prob * 7).round().clamp(0, 7).to(torch.uint8)

        return {
            'endpoints': eps_quant,
            'endpoints_min': eps_min,
            'endpoints_max': eps_max,
            'indices': idx_quant,
            'resolution': (
                blocks_w * 4,
                blocks_h * 4,
            ),
        }


# ============================================================
# BC4 格式实现 (RGTC1 / ATI1)
# ============================================================

class BC4Format(BCFormat):
    """BC4/RGTC1 单通道格式: 8-bit端点 + 3-bit索引 (8级插值)."""

    def simulate_decode(self, y):
        return y

    def inverse_simulate_decode(self, w):
        return w

    def get_endpoint_init_scale(self, feature_dim):
        return 0.01

    def get_endpoint_count(self):
        return 2

    def use_partitions(self):
        return False

    def get_partition_count(self):
        return 1

    def get_endpoint_bits(self, feature_dim):
        return [8] * feature_dim

    def get_index_bits(self):
        return 3

    def quantize_endpoints(self, endpoints):
        eps = endpoints.data
        eps_min, eps_max = eps.min(), eps.max()
        if eps_max - eps_min > 1e-8:
            eps_norm = (eps - eps_min) / (eps_max - eps_min)
            eps_quant = (eps_norm * 255.0).round().clamp(0, 255) / 255.0
            eps.copy_(eps_quant * (eps_max - eps_min) + eps_min)

    def quantize_indices(self, raw_indices):
        prob = torch.sigmoid(raw_indices.data)
        q = (prob * 7.0).round().clamp(0, 7) / 7.0
        safe = q.clamp(1e-6, 1.0 - 1e-6)
        logit = torch.log(safe / (1.0 - safe))
        raw_indices.data.copy_(logit.clamp(-10.0, 10.0))

    def get_export_params(self, endpoints, raw_indices, partition_ids, blocks_h, blocks_w):
        eps = endpoints.data
        eps_min, eps_max = eps.min(), eps.max()
        if eps_max - eps_min < 1e-8:
            eps_quant = torch.zeros_like(eps).to(torch.uint8)
        else:
            eps_norm = (eps - eps_min) / (eps_max - eps_min)
            eps_quant = (eps_norm * 255).round().clamp(0, 255).to(torch.uint8)

        prob = torch.sigmoid(raw_indices.data)
        idx_quant = (prob * 7).round().clamp(0, 7).to(torch.uint8)

        return {
            'endpoints': eps_quant,
            'endpoints_min': eps_min,
            'endpoints_max': eps_max,
            'indices': idx_quant,
            'resolution': (
                blocks_w * 4,
                blocks_h * 4,
            ),
        }


# ============================================================
# BC5 格式实现 (RGTC2 / ATI2)
# ============================================================

class BC5Format(BCFormat):
    """BC5/RGTC2 双通道格式: 两个独立 BC4 块, 8-bit端点 + 3-bit索引."""

    def simulate_decode(self, y):
        return y

    def inverse_simulate_decode(self, w):
        return w

    def get_endpoint_init_scale(self, feature_dim):
        return 0.01

    def get_endpoint_count(self):
        return 2

    def use_partitions(self):
        return False

    def get_partition_count(self):
        return 1

    def get_endpoint_bits(self, feature_dim):
        return [8] * feature_dim

    def get_index_bits(self):
        return 3

    def quantize_endpoints(self, endpoints):
        eps = endpoints.data
        eps_min, eps_max = eps.min(), eps.max()
        if eps_max - eps_min > 1e-8:
            eps_norm = (eps - eps_min) / (eps_max - eps_min)
            eps_quant = (eps_norm * 255.0).round().clamp(0, 255) / 255.0
            eps.copy_(eps_quant * (eps_max - eps_min) + eps_min)

    def quantize_indices(self, raw_indices):
        prob = torch.sigmoid(raw_indices.data)
        q = (prob * 7.0).round().clamp(0, 7) / 7.0
        safe = q.clamp(1e-6, 1.0 - 1e-6)
        logit = torch.log(safe / (1.0 - safe))
        raw_indices.data.copy_(logit.clamp(-10.0, 10.0))

    def get_export_params(self, endpoints, raw_indices, partition_ids, blocks_h, blocks_w):
        eps = endpoints.data
        eps_min, eps_max = eps.min(), eps.max()
        if eps_max - eps_min < 1e-8:
            eps_quant = torch.zeros_like(eps).to(torch.uint8)
        else:
            eps_norm = (eps - eps_min) / (eps_max - eps_min)
            eps_quant = (eps_norm * 255).round().clamp(0, 255).to(torch.uint8)

        prob = torch.sigmoid(raw_indices.data)
        idx_quant = (prob * 7).round().clamp(0, 7).to(torch.uint8)

        return {
            'endpoints': eps_quant,
            'endpoints_min': eps_min,
            'endpoints_max': eps_max,
            'indices': idx_quant,
            'resolution': (
                blocks_w * 4,
                blocks_h * 4,
            ),
        }


# ============================================================
# 格式注册表
# ============================================================

_FORMATS = {
    'bc6': BC6Format,
    'bc1': BC1Format,
    'bc2': BC2Format,
    'bc3': BC3Format,
    'bc4': BC4Format,
    'bc5': BC5Format,
}


def get_bc_format(name: str) -> BCFormat:
    """根据名称获取 BC 格式实例."""
    if name not in _FORMATS:
        raise ValueError(f"未知 BC 格式: '{name}'，支持: {list(_FORMATS.keys())}")
    return _FORMATS[name]()


# ============================================================
# BC 块特征层 —— 通用 4×4 块压缩表示
# ============================================================

class BCBlockFeature(nn.Module):
    """通用 BC 块特征层: 4×4 块参数化为端点 + 索引 (+ 分区)。

    训练时从前向传播即执行 sigmoid + round (STE), 实现量化感知训练 (QAT)。
    解码后通过 BCFormat.simulate_decode 做格式特定的后处理 (如 BC6 FP16)。
    """

    def __init__(self, resolution_w, resolution_h, feature_dim, bc_format, train_qat=True):
        super().__init__()
        assert resolution_w % 4 == 0 and resolution_h % 4 == 0
        self.resolution_w = resolution_w
        self.resolution_h = resolution_h
        self.feature_dim = feature_dim
        self.blocks_w = resolution_w // 4
        self.blocks_h = resolution_h // 4
        self.bc_format = bc_format
        self.train_qat = train_qat

        # 端点: [blocks_h, blocks_w, num_endpoints, feature_dim]
        num_eps = bc_format.get_endpoint_count()
        init_scale = bc_format.get_endpoint_init_scale(feature_dim)
        self.endpoints = nn.Parameter(
            torch.randn(self.blocks_h, self.blocks_w, num_eps, feature_dim) * init_scale
        )

        # 索引: [blocks_h, blocks_w, 16]  (每块 16 个像素)
        self.raw_indices = nn.Parameter(
            torch.zeros(self.blocks_h, self.blocks_w, 16)
        )

        # 分区 (仅 BC6)
        if bc_format.use_partitions():
            self.register_buffer(
                'partitions_lut',
                torch.tensor(PARTITIONS, dtype=torch.float32),
            )
            partition_ids = torch.randint(0, bc_format.get_partition_count(),
                                          (self.blocks_h, self.blocks_w))
            self.register_buffer('partition_ids', partition_ids)

    def _quantize_ste(self, x, levels_minus_one):
        """QAT: 前向 round → 量化 → 反向 STE (直通梯度).

        Args:
            x: 已在 [0, 1] 区间的浮点 tensor
            levels_minus_one: 量化级数-1 (如 BC1 索引: 3 → 4级)
        """
        if not self.training or not self.train_qat:
            return x
        levels = float(levels_minus_one)
        xq = (x * levels).round().clamp(0, levels) / levels
        return x + (xq - x).detach()

    def forward(self):
        """模拟 BC 解压缩, 重建完整特征图 [1, C, H, W].

        训练时: sigmoid → quant(STE) → BC decode
        推理时: sigmoid → BC decode (不量化, 从已量化的参数读取)
        """
        bh, bw = self.blocks_h, self.blocks_w
        C = self.feature_dim

        # Index: sigmoid → [0,1] → quant (QAT)
        alpha = torch.sigmoid(self.raw_indices)
        idx_bits = self.bc_format.get_index_bits()
        alpha_q = self._quantize_ste(alpha, (1 << idx_bits) - 1)

        # Endpoint: sigmoid → [0,1] → quant per-channel (QAT)
        eps = torch.sigmoid(self.endpoints)
        eps_bits = self.bc_format.get_endpoint_bits(C)
        eps_q = eps.clone()
        for ch in range(C):
            b = eps_bits[ch]
            eps_q[..., ch] = self._quantize_ste(eps[..., ch], (1 << b) - 1)

        if self.bc_format.use_partitions():
            y = self._forward_partitioned(alpha_q, eps_q, bh, bw, C)
        else:
            y = self._forward_single_line(alpha_q, eps_q, bh, bw, C)

        y = self.bc_format.simulate_decode(y)
        return y

    def _forward_single_line(self, alpha, eps, bh, bw, C):
        """单线段: y = e0 + alpha * (e1 - e0)."""
        e0 = eps[:, :, 0, :]
        e1 = eps[:, :, 1, :]
        x = alpha.unsqueeze(-1)  # [bh, bw, 16, 1]
        y = e0.unsqueeze(2) + x * (e1.unsqueeze(2) - e0.unsqueeze(2))
        return self._reshape_to_image(y, bh, bw, C)

    def _forward_partitioned(self, alpha, eps, bh, bw, C):
        """BC6 双线段 + 分区: pk * line1 + (1-pk) * line2."""
        e1 = eps[:, :, 0, :]
        e2 = eps[:, :, 1, :]
        e3 = eps[:, :, 2, :]
        e4 = eps[:, :, 3, :]
        x = alpha.unsqueeze(-1)
        line1 = e1.unsqueeze(2) + x * (e2.unsqueeze(2) - e1.unsqueeze(2))
        line2 = e3.unsqueeze(2) + x * (e4.unsqueeze(2) - e3.unsqueeze(2))
        pk = self.partitions_lut[self.partition_ids].unsqueeze(-1)
        y = pk * line1 + (1 - pk) * line2
        return self._reshape_to_image(y, bh, bw, C)

    def _reshape_to_image(self, y, bh, bw, C):
        """将 [bh, bw, 16, C] 重整为 [1, C, H, W]."""
        y = y.reshape(bh, bw, 4, 4, C)
        y = y.permute(0, 2, 1, 3, 4)
        y = y.reshape(self.resolution_h, self.resolution_w, C)
        y = y.permute(2, 0, 1).unsqueeze(0)
        return y

    def get_parameters_for_export(self):
        return self.bc_format.get_export_params(
            self.endpoints, self.raw_indices,
            getattr(self, 'partition_ids', None),
            self.blocks_h, self.blocks_w,
        )

    def quantize_features(self):
        self.bc_format.quantize_endpoints(self.endpoints)
        self.bc_format.quantize_indices(self.raw_indices)


# ============================================================
# Mipmap 特征网格
# ============================================================

class BCMipmapFeatureGrid(nn.Module):
    """BC 压缩的多分辨率特征金字塔, 支持三线性/三立方插值采样."""

    def __init__(self, base_resolution, num_mips, feature_dim, bc_format,
                 filter_mode='trilinear', half_pixel_offset=False):
        super().__init__()
        self.base_resolution = base_resolution
        self.num_mips = num_mips
        self.feature_dim = feature_dim
        self.filter_mode = filter_mode
        self.half_pixel_offset = half_pixel_offset

        self.mips = nn.ModuleList([
            BCBlockFeature(base_resolution >> i, base_resolution >> i,
                           feature_dim, bc_format)
            for i in range(num_mips)
        ])

    def sample(self, uv, scale):
        B = uv.shape[0]
        s = torch.clamp(scale, 0, self.num_mips - 1)
        s0 = s.long()
        s1 = torch.clamp(s0 + 1, max=self.num_mips - 1)
        lam = (s - s0.float()).view(-1, 1, 1, 1)

        uv_offset = uv
        if self.half_pixel_offset and s0.max() < len(self.mips):
            # 半像素偏移: 当前 mip 的 0.5/分辨率
            res0 = self.base_resolution >> int(s0[0].item())
            uv_offset = uv + 0.5 / res0

        uv_grid = uv_offset * 2 - 1

        # 空间插值模式: trilinear → bilinear, tricubic → bicubic
        spatial_mode = 'bilinear' if self.filter_mode == 'trilinear' else 'bicubic'

        decompressed = [mip() for mip in self.mips]

        feat0, feat1 = [], []
        for b in range(B):
            f0 = F.grid_sample(decompressed[int(s0[b].item())], uv_grid[b:b + 1],
                               mode=spatial_mode, padding_mode='border',
                               align_corners=False)
            f1 = F.grid_sample(decompressed[int(s1[b].item())], uv_grid[b:b + 1],
                               mode=spatial_mode, padding_mode='border',
                               align_corners=False)
            feat0.append(f0)
            feat1.append(f1)

        feat0 = torch.cat(feat0, dim=0)
        feat1 = torch.cat(feat1, dim=0)
        return (1 - lam) * feat0 + lam * feat1


# ============================================================
# 完整神经 BC 材质模型
# ============================================================

class NeuralBCTextureModel(nn.Module):
    """神经 BC 材质模型: 多网格特征 → 拼接 → MLP 解码器."""

    def __init__(self, feature_configs, hidden_dim, output_dim, bc_format,
                 filter_mode='trilinear', half_pixel_offsets=None, num_layers=1):
        super().__init__()
        if half_pixel_offsets is None:
            half_pixel_offsets = []
        self.feature_grids = nn.ModuleList([
            BCMipmapFeatureGrid(res, mips, dim, bc_format, filter_mode,
                                half_pixel_offset=(i in half_pixel_offsets))
            for i, (res, mips, dim) in enumerate(feature_configs)
        ])
        total_input_dim = sum(dim for _, _, dim in feature_configs)

        layers = []
        in_dim = total_input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

        self.total_input_dim = total_input_dim
        self.output_dim = output_dim

    def forward(self, uv, scales):
        features = []
        for grid in self.feature_grids:
            features.append(grid.sample(uv, scales))

        x = torch.cat(features, dim=1)
        x = x.permute(0, 2, 3, 1)
        y = self.mlp(x)
        return y.permute(0, 3, 1, 2)


# ============================================================
# 工厂函数
# ============================================================

def make_bc_model(model_params, output_dim=9, bc_format_name='bc6'):
    """从配置创建 BC 神经材质模型.

    Args:
        model_params: dict, 由 get_model_params(config) 产生.
            包含: feature_configs, hidden_dim, filter, half_pixel_offsets, num_layers
        output_dim: MLP 输出维度
        bc_format_name: BC 格式名称
    """
    bc_format = get_bc_format(bc_format_name)

    return NeuralBCTextureModel(
        feature_configs=model_params['feature_configs'],
        hidden_dim=model_params['hidden_dim'],
        output_dim=output_dim,
        bc_format=bc_format,
        filter_mode=model_params.get('filter', 'trilinear'),
        half_pixel_offsets=model_params.get('half_pixel_offsets', []),
        num_layers=model_params.get('num_layers', 1),
    )
