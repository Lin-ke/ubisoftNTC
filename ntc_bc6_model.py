"""
BC6块压缩特征模型的PyTorch实现。

本模块实现了论文 "Neural Texture Compression" (NTC) 中提出的 BC6 压缩特征表示方法,
将 BC6 块压缩格式 (BC6H) 作为可学习的神经特征网格,用于高质量纹理压缩。

核心思路: 利用 BC6 块的端点(endpoints)和索引(indices)来隐式表示特征,而非显式存储
高分辨率特征图。每个 4×4 块的 16 个像素通过两个线段的加权插值来重建特征值,
线段由 4 个端点 (e1,e2,e3,e4) 定义,像素归属于哪条线段由 32 种预定义分区模式决定。

BC6 原本用于 HDR 纹理压缩,这里将其格式结构作为特征表示的载体:
- 端点值在训练中通过梯度下降优化
- 分区模式固定不变(训练期间不更新)
- 通过 sigmoid 将原始索引映射到 [0,1] 区间实现可微的软赋值

训练完成后,端点被量化为 6-bit uint8,索引被量化为 3-bit uint8,与标准 BC6 位布局兼容,
实现高效存储和硬件解压。

参考文献: Neural Texture Compression (NTC), 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ntc_bc6_partitions import PARTITIONS


def simulate_fp16(y):
    """
    模拟 BC6 格式中 half-precision (FP16) 的 "位重解释" (bit reinterpretation) 步骤。

    在 BC6H 解压流程中,解码出的整数值需要先通过逆映射函数还原为 FP16 浮点数,
    具体来说是模拟 HDR 模式下的 10-bit 对数编码 -> FP16 的转换过程。

    Args:
        y: 解压后的特征值张量,形状为 [N, C, H, W]

    Returns:
        w: 经过 FP16 位重解释模拟后的特征值

    Notes:
        此函数模拟了 BC6 解压中 "Unquantize" 步骤,即把 [1, 65535] 范围的整数
        通过非线性映射恢复为 FP16 范围的浮点数。公式源自 BC6 规范的逆量化步骤。
    """
    h = torch.clamp(torch.floor((y - 1) / 1024) - 1, min=0)
    w = (2 ** (h - 14)) * (y / 1024 - h)
    return w


class BC6BlockFeature(nn.Module):
    """
    BC6 块特征层 —— 用 BC6 压缩格式表示可学习的特征网格。

    将整个特征图划分为 resolution_w×resolution_h 的像素网格,每个 4×4 的子块 (block)
    使用 BC6 的双线段模式进行压缩表示:

    结构概览:
    ┌─────────┬─────────┐
    │  e1,e2  │  e3,e4  │   ← 4 个端点 (endpoints), 每个端点 feature_dim 维
    │ (线段1)  │ (线段2)  │      e1/e2 定义线段1, e3/e4 定义线段2
    ├─────────┴─────────┤
    │  16个索引(indices) │   ← 每个像素在所在线段上的插值位置 [0,1]
    ├────────────────────┤
    │  分区ID (0..31)    │   ← 决定 16 个像素各属于哪条线段 (FIXED)
    └────────────────────┘

    重建公式: y_pixel = P * [e1 + idx*(e2-e1)] + (1-P) * [e3 + idx*(e4-e3)]
    其中 P ∈ {0,1} 由预定义的 32 种分区模式决定。

    BC6 标准格式:
    - 每个块 128 bits: 32 bits 存储分区信息和端点索引, 96 bits 存储端点值
    - 端点量化为 6 bits, 索引量化为 3 bits
    - 支持 32 种预定义分区模式 (4×4 块的像素分配到两条线段的掩码)
    """

    def __init__(self, resolution_w, resolution_h, feature_dim=3):
        """
        初始化 BC6 块特征层。

        Args:
            resolution_w: 特征图的宽度 (像素数), 必须能被 4 整除
            resolution_h: 特征图的高度 (像素数), 必须能被 4 整除
            feature_dim: 每个像素的特征通道数, 对应 BC6 每个像素重建出的颜色通道数
                         默认 3 (如 RGB 特征)
        """
        super().__init__()
        assert resolution_w % 4 == 0 and resolution_h % 4 == 0
        self.resolution_w = resolution_w
        self.resolution_h = resolution_h
        self.feature_dim = feature_dim
        self.blocks_w = resolution_w // 4  # 水平方向的 4×4 块数量
        self.blocks_h = resolution_h // 4  # 垂直方向的 4×4 块数量

        # 端点 (endpoints): 每个 4×4 块有 4 个端点, 每个端点 feature_dim 维
        # 形状: [blocks_h, blocks_w, 4, feature_dim]
        # e1/e2 构成线段1, e3/e4 构成线段2
        # 用小方差随机初始化, 避免训练早期梯度爆炸
        self.endpoints = nn.Parameter(
            torch.randn(self.blocks_h, self.blocks_w, 4, feature_dim) * 0.01
        )

        # 原始索引 (raw_indices): 每个 4×4 块有 16 个像素,
        # 每个像素一个索引值表示在线段上的插值位置
        # 形状: [blocks_h, blocks_w, 16]
        # 初始化为 0, 经过 sigmoid 后对应插值位置 0.5
        self.raw_indices = nn.Parameter(
            torch.zeros(self.blocks_h, self.blocks_w, 16)
        )

        # 分区查找表 (LUT): 存储 32 种预定义分区模式的掩码
        # 形状: [32, 16], 每个分区是一组 16 个像素的二值归属 (0或1)
        # register_buffer 表示这是非参数张量, 不参与梯度更新
        self.register_buffer(
            'partitions_lut',
            torch.tensor(PARTITIONS, dtype=torch.float32),
        )

        # 分区 ID: 每个 4×4 块随机分配一个分区模式 (0~31)
        # 训练期间 FIXED (固定不变), 不参与梯度更新
        partition_ids = torch.randint(0, 32, (self.blocks_h, self.blocks_w))
        self.register_buffer('partition_ids', partition_ids)

    def forward(self):
        """
        前向传播: 模拟 BC6 解压缩过程, 从端点、索引和分区重建特征图。

        解压步骤:
        1. 将 raw_indices 通过 sigmoid 映射到 [0,1] 实现可微的软索引
        2. 取出 4 个端点: e1/e2/e3/e4
        3. 对每个像素, 在线段1上做线性插值: line1 = e1 + idx*(e2-e1)
        4. 对每个像素, 在线段2上做线性插值: line2 = e3 + idx*(e4-e3)
        5. 根据分区模式, 选择像素属于线段1还是线段2:
           y = pk * line1 + (1-pk) * line2
        6. 将 16 个像素重塑为 4×4 空间布局
        7. 将所有块拼接成完整的特征图 [1, C, H, W]
        8. 应用 FP16 位重解释模拟 (simulate_fp16)

        Returns:
            y: 重建的特征图, 形状 [1, feature_dim, resolution_h, resolution_w]

        Notes:
            此过程完全可微, 允许梯度过端点值和索引传播, 从而实现端到端的训练优化。
            分区模式 (partition_ids) 是固定不更新的。
        """
        bh, bw = self.blocks_h, self.blocks_w
        C = self.feature_dim

        # 步骤1: 通过 sigmoid 将原始索引映射到 (0,1), 实现可微的软索引
        indices = torch.sigmoid(self.raw_indices)  # [bh, bw, 16]

        # 步骤2: 取出 4 个端点
        e1 = self.endpoints[:, :, 0, :]  # [bh, bw, C] —— 线段1的起点
        e2 = self.endpoints[:, :, 1, :]  # [bh, bw, C] —— 线段1的终点
        e3 = self.endpoints[:, :, 2, :]  # [bh, bw, C] —— 线段2的起点
        e4 = self.endpoints[:, :, 3, :]  # [bh, bw, C] —— 线段2的终点

        x = indices.unsqueeze(-1)  # [bh, bw, 16, 1] — 广播到通道维度

        # 步骤3 & 4: 双线段线性插值
        # 线段1: L1 = e1 + idx*(e2 - e1), 形状 [bh, bw, 16, C]
        line1 = e1.unsqueeze(2) + x * (e2.unsqueeze(2) - e1.unsqueeze(2))
        # 线段2: L2 = e3 + idx*(e4 - e3), 形状 [bh, bw, 16, C]
        line2 = e3.unsqueeze(2) + x * (e4.unsqueeze(2) - e3.unsqueeze(2))

        # 步骤5: 根据分区模式进行软混合
        # partiion_lut[pid] 返回该块的 16 像素二值掩码 (pk ∈ {0,1})
        # 此处实现为可微的混合: y = pk*l1 + (1-pk)*l2
        pk = self.partitions_lut[self.partition_ids]  # [bh, bw, 16]
        pk = pk.unsqueeze(-1)  # [bh, bw, 16, 1] — 广播到通道维度

        y = pk * line1 + (1 - pk) * line2  # [bh, bw, 16, C]

        # 步骤6: 将 16 个像素重塑为 4×4 空间布局
        y = y.reshape(bh, bw, 4, 4, C)
        # 重排维度: 将块的行列与块内像素的行列交错排列
        y = y.permute(0, 2, 1, 3, 4)  # [bh, 4, bw, 4, C]

        # 步骤7: 合并所有块为完整特征图, 并调整为标准格式 [1, C, H, W]
        y = y.reshape(self.resolution_h, self.resolution_w, C)
        y = y.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        # 步骤8: 模拟 BC6 解压中的 FP16 位重解释步骤
        y = simulate_fp16(y)
        return y

    def get_parameters_for_export(self):
        """
        导出量化后的 BC6 参数, 用于存储或硬件解压。

        量化方案:
        - 端点 (endpoints): 归一化到 [0,1] 后量化为 6-bit 无符号整数 (0~63)
          同时保存归一化的 min/max 用于解压时还原
        - 索引 (indices): 先通过 sigmoid 映射到 (0,1), 再量化为 3-bit 无符号整数 (0~7)
        - 分区 ID: 直接转为 uint8 (0~31)

        Returns:
            dict: 包含量化参数的字典
                - 'endpoints':     uint8 [BH, BW, 4, C], 6-bit 量化端点值
                - 'endpoints_min': float, 端点最小值 (用于解压还原)
                - 'endpoints_max': float, 端点最大值 (用于解压还原)
                - 'indices':       uint8 [BH, BW, 16], 3-bit 量化索引值
                - 'partition_ids': uint8 [BH, BW], 分区模式 ID
                - 'resolution':    (int, int), 纹理分辨率

        Notes:
            此量化方案与 BC6H 标准格式兼容:
            - 6-bit 端点 + 3-bit 索引 = 9 bits 精度
            - 32 种分区模式用 5 bits 表示
            - 总存储: 128 bits/block (标准 BC6 块大小)
        """
        endpoints = self.endpoints.data
        eps_min = endpoints.min()
        eps_max = endpoints.max()
        if eps_max - eps_min < 1e-8:
            # 端点值几乎没有变化, 直接设为零
            eps_quant = torch.zeros_like(endpoints).to(torch.uint8)
        else:
            # 归一化到 [0,1], 再量化为 6-bit (0~63)
            eps_norm = (endpoints - eps_min) / (eps_max - eps_min)
            eps_quant = (eps_norm * 63).round().clamp(0, 63).to(torch.uint8)

        raw_indices = self.raw_indices.data
        # sigmoid 映射到 (0,1), 再量化为 3-bit (0~7)
        indices_prob = torch.sigmoid(raw_indices)
        idx_quant = (indices_prob * 7).round().clamp(0, 7).to(torch.uint8)

        return {
            'endpoints': eps_quant,
            'endpoints_min': eps_min,
            'endpoints_max': eps_max,
            'indices': idx_quant,
            'partition_ids': self.partition_ids.data.to(torch.uint8),
            'resolution': (self.resolution_w, self.resolution_h),
        }


class BC6MipmapFeatureGrid(nn.Module):
    """
    BC6 Mipmap 特征网格 —— 多分辨率特征金字塔。

    包含多个 BC6BlockFeature 层级 (mip 级别), 每个级别的分辨率依次减半:
    - mip 0: base_resolution × base_resolution (最高分辨率)
    - mip 1: base_resolution/2 × base_resolution/2
    - mip 2: base_resolution/4 × base_resolution/4
    - ...

    采样时根据 scale 参数在相邻两个 mip 级别之间做三线性插值 (trilinear filtering):
    1. 在两个 mip 级别分别做双线性插值 (bilinear interpolation)
    2. 在 mip 级别之间做线性混合

    这种多分辨率设计使模型能同时捕获不同频率的纹理细节:
    - 低 mip (高分辨率) 捕获高频细节
    - 高 mip (低分辨率) 捕获低频全局信息
    """

    def __init__(self, base_resolution, num_mips, feature_dim=3):
        """
        Args:
            base_resolution: 基础分辨率 (mip 0 的宽高), 必须是 4 的倍数
            num_mips: mip 级别数量
            feature_dim: 每个像素的特征通道数
        """
        super().__init__()
        self.base_resolution = base_resolution
        self.num_mips = num_mips
        self.feature_dim = feature_dim

        # 构建 mip 金字塔: 每个 mip 级别是一个独立的 BC6BlockFeature
        # 分辨率依次减半: res >> i
        self.mips = nn.ModuleList([
            BC6BlockFeature(base_resolution >> i, base_resolution >> i, feature_dim)
            for i in range(num_mips)
        ])

    def sample(self, uv, scale):
        """
        根据 UV 坐标和 mip 级别采样特征。

        执行三线性插值 (trilinear filtering):
        - 空间维度: 在两个最接近的 mip 级别上分别做双线性插值
        - mip 维度: 在两个 mip 级别的结果之间做线性混合

        Args:
            uv: UV 坐标, 形状 [B, H, W, 2], 取值范围 [0,1]
            scale: mip 级别 (浮点数), 形状 [B], 支持每批不同的 mip 级别

        Returns:
            采样得到的特征, 形状 [B, feature_dim, H, W]
        """
        B = uv.shape[0]
        # 将 scale clamp 到 [0, num_mips-1] 范围
        s = torch.clamp(scale, 0, self.num_mips - 1)
        s0 = s.long()  # 较低的 mip 级别 (整数部分)
        s1 = torch.clamp(s0 + 1, max=self.num_mips - 1)  # 较高的 mip 级别
        lam = (s - s0.float()).view(-1, 1, 1, 1)  # mip 间插值权重

        # 将 UV 坐标从 [0,1] 转换为 grid_sample 需要的 [-1, 1] 范围
        uv_grid = uv * 2 - 1

        # 对所有 mip 级别执行 BC6 解压缩 (获得特征图)
        decompressed = [mip() for mip in self.mips]

        # 逐样本在两个 mip 级别上做双线性采样
        feat0, feat1 = [], []
        for b in range(B):
            f0 = F.grid_sample(decompressed[int(s0[b].item())], uv_grid[b:b+1],
                               mode='bilinear', padding_mode='border',
                               align_corners=False)
            f1 = F.grid_sample(decompressed[int(s1[b].item())], uv_grid[b:b+1],
                               mode='bilinear', padding_mode='border',
                               align_corners=False)
            feat0.append(f0)
            feat1.append(f1)

        feat0 = torch.cat(feat0, dim=0)
        feat1 = torch.cat(feat1, dim=0)

        # 三线性插值: 在两个 mip 级别结果间做线性混合
        return (1 - lam) * feat0 + lam * feat1


class NeuralBC6TextureModel(nn.Module):
    """
    神经 BC6 材质模型 —— 完整的端到端神经纹理压缩模型。

    架构: 多分辨率特征网格 + 轻量 MLP 解码器

    1. 特征提取阶段:
       - 多个 BC6MipmapFeatureGrid (多分辨率 mipmap 金字塔) 并行工作
       - 每个 grid 按 UV 坐标采样, 输出 feature_dim 维特征
       - 将所有 grid 的特征在通道维度上拼接 (concatenate)

    2. 解码阶段:
       - 轻量级 MLP (两层全连接 + ReLU 激活)
       - 将拼接后的高维特征映射到输出维度 (如 9 通道: RGB + 6 通道 BRDF)

    整体数据流:
    UV, scale -> [Grid1采样] -> feat1 (3 ch)
              -> [Grid2采样] -> feat2 (3 ch)
              -> [Grid3采样] -> feat3 (3 ch)  -> concat -> MLP -> output
              -> [Grid4采样] -> feat4 (3 ch)

    典型配置 (BCf-0.5K, 适用于 1K 纹理):
    分辨率: 512/256/128/64, 每个 7/6/5/4 个 mip 级别, 全 3 通道
    总参数量约 0.5K 每组特征 (不含 MLP)
    """

    def __init__(self, feature_configs, hidden_dim, output_dim):
        """
        Args:
            feature_configs: 特征网格配置列表, 每项为 (base_resolution, num_mips, feature_dim) 三元组
                             例如 [(512, 7, 3), (256, 6, 3), (128, 5, 3), (64, 4, 3)]
            hidden_dim: MLP 隐藏层维度
            output_dim: 输出维度, 例如 3 (RGB), 9 (RGB + 6 通道 BRDF 参数)
        """
        super().__init__()
        # 构建多个 mipmap 特征网格
        self.feature_grids = nn.ModuleList([
            BC6MipmapFeatureGrid(res, mips, dim)
            for res, mips, dim in feature_configs
        ])
        # 所有网格的特征在通道维度拼接后的总维度
        total_input_dim = sum(dim for _, _, dim in feature_configs)

        # 轻量 MLP 解码器: 两层全连接 + ReLU 激活
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.total_input_dim = total_input_dim
        self.output_dim = output_dim

    def forward(self, uv, scales):
        """
        前向传播: 从 UV 坐标和 mip 级别重建纹理值。

        流程:
        1. 对所有特征网格按 UV 坐标采样 (含 mipmap 三线性插值)
        2. 在通道维度拼接所有网格的特征
        3. 通过 MLP 解码器将特征映射到目标输出维度

        Args:
            uv: UV 坐标, 形状 [B, H, W, 2], 取值范围 [0,1]
            scales: mip 级别, 形状 [B], 决定采样的分辨率层级

        Returns:
            y: 重建的纹理值, 形状 [B, output_dim, H, W]
        """
        features = []
        # 步骤1: 从所有特征网格采样
        for grid in self.feature_grids:
            feat = grid.sample(uv, scales)
            features.append(feat)

        # 步骤2: 在通道维度拼接所有网格的特征
        x = torch.cat(features, dim=1)  # [B, total_input_dim, H, W]

        # 步骤3: MLP 逐像素解码
        # MLP 期望输入形状 [*, total_input_dim], 所以需要将通道维度移到最后
        x = x.permute(0, 2, 3, 1)  # [B, H, W, total_input_dim]
        y = self.mlp(x)  # [B, H, W, output_dim]

        # 恢复为标准的图像格式 [B, C, H, W]
        y = y.permute(0, 3, 1, 2)  # [B, output_dim, H, W]
        return y


def make_bc6_model(reference_resolution=1024, output_dim=9):
    """
    工厂函数: 创建 NTC 论文中的标准 BC6 神经材质模型。

    使用 BCf-0.5K 配置 (论文中适用于 1K 纹理的配置):
    - 4 个 mipmap 特征网格, 分辨率分别为 512、256、128、64
    - mip 级别数: 7, 6, 5, 4 (随分辨率降低而减少)
    - 每个网格输出 3 通道特征, 拼接后得到 12 通道 (4×3) 输入 MLP
    - MLP 隐藏层维度: 16 (轻量级设计, 总共约 0.5K 参数每特征组)
    - BC6BlockFeature 总参数量 ≈ 每组 0.5K (故称 BCf-0.5K)

    多分辨率设计原理:
    高分辨率网格 (512/256) 捕获高频细节 (锐利边缘、细小纹理)
    低分辨率网格 (128/64) 捕获低频全局信息 (光照、颜色渐变)

    Args:
        reference_resolution: 参考纹理分辨率, 默认 1024 (1K)
                              网格的基础分辨率约为其一半, 利用 BC6 的 4×4 块压缩,
                              实际存储量远小于显式存储
        output_dim: MLP 输出维度
                     3 = RGB 颜色
                     9 = RGB + 6 通道 BRDF 参数 (如 SVBRDF: diffuse, specular, roughness 等)

    Returns:
        NeuralBC6TextureModel: 完整的可训练神经纹理压缩模型
    """
    feature_configs = [
        (512, 7, 3),   # 分辨率 512, 7 级 mip, 3 通道特征
        (256, 6, 3),   # 分辨率 256, 6 级 mip, 3 通道特征
        (128, 5, 3),   # 分辨率 128, 5 级 mip, 3 通道特征
        (64, 4, 3),    # 分辨率 64, 4 级 mip, 3 通道特征
    ]
    return NeuralBC6TextureModel(feature_configs, hidden_dim=16, output_dim=output_dim)
