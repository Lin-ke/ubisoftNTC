"""
BC6神经材质三阶段训练流程
===========================
本文件实现了基于BC6块压缩的神经纹理材质训练流程，分为三个阶段：

  阶段1 - 预热训练（~5000步）：使用无约束特征训练的NeuralTextureModel进行预热，
         学习率 feats=5e-2、mlp=1e-3，指数衰减gamma=0.9995，采用随机UV块+随机
         连续mip层级的batch采样策略，损失函数为MSE。

  阶段2 - BC6训练（~19000步）：从预热模型特征中提取BC6块参数（端点、分区、索引）
         作为BC6模型的初始值，然后对BC6参数进行端到端训练，学习率 feats=1e-2、
         mlp=1e-3，gamma=0.9999。

  阶段3 - 量化+精调（~1000步）：将端点量化为6位（64级）、索引量化为3位（8级），
         冻结所有BC6特征参数，仅对MLP进行微调以补偿量化误差。

压缩算法（compress_to_bc6）：对于每个4×4像素块，遍历32种分区模式，每种分区将16个
像素分成两个子集，每个子集用一对min/max端点表示，选择MSE最小的分区作为最优压缩结果。

材质参考通道（共9维）：
  - albedo (RGB, 3维)
  - normal  (RGB, 3维, 范围[-1,1])
  - ambient occlusion (1维)
  - roughness (1维)
  - metalness (1维)

参考文献：Neural Texture Compression using BC6 Block Compression
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import time

from ntc_model import make_model, NeuralTextureModel
from ntc_bc6_model import make_bc6_model, NeuralBC6TextureModel
from ntc_bc6_partitions import PARTITIONS


def build_mipmaps(tensor):
    """构建mipmap金字塔。
    
    从原始分辨率纹理开始，每次将尺寸减半（area平均池化），
    直到最短边不满足继续下采样的条件。返回list，每个元素为一个mip层级，
    按分辨率从高到低排列（mips[0]为原始分辨率）。
    
    参数:
        tensor: [C, H, W] 张量，C为通道数，H、W为空间尺寸
    
    返回:
        list of [C, H_i, W_i]，各mip层级张量
    """
    mips = [tensor]
    h, w = tensor.shape[1], tensor.shape[2]
    while min(h, w) > 1:
        h, w = max(1, h // 2), max(1, w // 2)
        mip = F.interpolate(mips[-1].unsqueeze(0), size=(h, w), mode='area').squeeze(0)
        mips.append(mip)
    return mips


def sample_reference(mips, uv, scale):
    """从参考材质mipmap金字塔中采样，支持连续mip层级的三线性插值。
    
    对于每个batch样本：
    1. 根据scale确定两个相邻的mip层级（s0=floor(scale), s1=s0+1）
    2. 在两个层级上分别进行双三次(bicubic)滤波采样
    3. 按小数部分lam=s-s0进行线性混合（即mip层间的三线性插值）
    
    参数:
        mips: list of [C, H_i, W_i]，mipmap金字塔
        uv:   [B, H', W', 2]，归一化UV坐标，范围[0,1]
        scale: [B]，连续mip层级值（0=最高分辨率，num_mips-1=最低分辨率）
    
    返回:
        [B, C, H', W']，插值后的参考材质值
    """
    num_mips = len(mips)
    s = torch.clamp(scale, 0, num_mips - 1)
    s0 = s.long()
    s1 = torch.clamp(s0 + 1, max=num_mips - 1)
    lam = (s - s0.float()).view(-1, 1, 1, 1)
    uv_grid = uv * 2 - 1

    results = []
    for b in range(uv.shape[0]):
        m0 = mips[s0[b].item()].unsqueeze(0)
        m1 = mips[s1[b].item()].unsqueeze(0)
        r0 = F.grid_sample(m0, uv_grid[b:b+1], mode='bicubic', padding_mode='border', align_corners=False)
        r1 = F.grid_sample(m1, uv_grid[b:b+1], mode='bicubic', padding_mode='border', align_corners=False)
        results.append((1 - lam[b]) * r0 + lam[b] * r1)
    return torch.cat(results, dim=0)


@torch.no_grad()
def evaluate_full(model, ref_mips, device, max_res=256):
    """全mip层级PSNR评估。
    
    对每个mip层级，在完整分辨率上进行逐块推理（每块不超过max_res×max_res），
    计算与参考材质的MSE，最后对全部mip层级的MSE取平均并转换为PSNR。
    
    大分辨率mip层级采用分瓦片(tile)策略：按max_res步长滑动窗口，每个窗口独立
    推理，最后按像素数加权平均计算该层级的MSE。
    
    参数:
        model:     神经材质模型（NeuralTextureModel或NeuralBC6TextureModel）
        ref_mips:  list of [C, H_i, W_i]，参考mipmap金字塔
        device:    计算设备
        max_res:   单次推理的最大分辨率（显存限制）
    
    返回:
        (psnr, avg_mse): 平均PSNR(dB)和平均MSE
    """
    mse_per_mip = []
    model.eval()

    for mip_i, ref_mip in enumerate(ref_mips):
        ref_mip = ref_mip.to(device)
        h, w = ref_mip.shape[1], ref_mip.shape[2]

        if h <= max_res and w <= max_res:
            u = torch.linspace(0, 1, w, device=device)
            v = torch.linspace(0, 1, h, device=device)
            uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)
            scale = torch.tensor([float(mip_i)], device=device)
            pred = model(uv, scale)
            mse = F.mse_loss(pred, ref_mip.unsqueeze(0))
        else:
            mse_total = 0.0
            count = 0
            for ty in range(0, h, max_res):
                th = min(max_res, h - ty)
                for tx in range(0, w, max_res):
                    tw = min(max_res, w - tx)
                    u = torch.linspace(tx/w, (tx+tw)/w, tw, device=device)
                    v = torch.linspace(ty/h, (ty+th)/h, th, device=device)
                    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)
                    scale = torch.tensor([float(mip_i)], device=device)
                    pred = model(uv, scale)
                    ref_tile = ref_mip[:, ty:ty+th, tx:tx+tw].unsqueeze(0)
                    mse_total += F.mse_loss(pred, ref_tile).item() * th * tw
                    count += th * tw
            mse = mse_total / count
        mse_per_mip.append(mse)
        model.train()

    avg_mse_val = sum(m.item() if hasattr(m, 'item') else m for m in mse_per_mip) / len(mse_per_mip)
    psnr = -10 * np.log10(max(avg_mse_val, 1e-10))
    return psnr, avg_mse_val


def load_brick_material(base_dir, target_res=1024):
    """加载StoneBricksSplitface砖墙PBR材质贴图并组合成9通道参考张量。
    
    从base_dir目录读取以下1K贴图：
      - COL:  反照率(albedo), RGB, [0,1]
      - NRM:  法线(normal), RGB, [0,1] → 映射到[-1,1]
      - AO:   环境光遮蔽(ambient occlusion), 灰度, [0,1]
      - GLOSS: 光泽度(gloss), 灰度, [0,1]
    
    处理流程：
      1. roughness = 1 - gloss（光泽度取反得到粗糙度）
      2. metalness设为全0（砖墙非金属材质）
      3. 将9个通道(alb3+nrm3+ao1+rough1+metal1)沿最后一维拼接
      4. 转为[C,H,W]格式并重采样到target_res
    
    参数:
        base_dir:   贴图文件目录路径
        target_res: 目标分辨率（默认1024）
    
    返回:
        [9, target_res, target_res] 参考材质张量
    """
    col = np.array(Image.open(os.path.join(base_dir, 'StoneBricksSplitface001_COL_1K.jpg'))
                   .convert('RGB'), dtype=np.float32) / 255.0
    nrm = np.array(Image.open(os.path.join(base_dir, 'StoneBricksSplitface001_NRM_1K.jpg'))
                   .convert('RGB'), dtype=np.float32) / 255.0
    ao = np.array(Image.open(os.path.join(base_dir, 'StoneBricksSplitface001_AO_1K.jpg'))
                   .convert('L'), dtype=np.float32) / 255.0
    gloss = np.array(Image.open(os.path.join(base_dir, 'StoneBricksSplitface001_GLOSS_1K.jpg'))
                      .convert('L'), dtype=np.float32) / 255.0

    nrm = nrm * 2.0 - 1.0
    roughness = 1.0 - gloss
    metalness = np.zeros_like(ao)

    reference = np.concatenate([
        col,
        nrm,
        ao[..., np.newaxis],
        roughness[..., np.newaxis],
        metalness[..., np.newaxis],
    ], axis=-1)

    ref_tensor = torch.from_numpy(reference).permute(2, 0, 1).float()
    if target_res != ref_tensor.shape[1]:
        ref_tensor = F.interpolate(ref_tensor.unsqueeze(0), size=(target_res, target_res),
                                   mode='bicubic', align_corners=False).squeeze(0)
    return ref_tensor


@torch.no_grad()
def compress_to_bc6(features):
    """将无约束特征压缩为BC6块参数。
    
    BC6压缩算法核心流程：
      1. 将特征划分为4×4的像素块（共BH×BW个块）
      2. 对每个块，遍历全部32种预定义分区模式（PARTITIONS）
      3. 每种分区将16个像素分为两个子集（subset0和subset1）
      4. 每个子集用一对min/max端点(e1,e2)和(e3,e4)表示
      5. 中间像素值 = endpoint_min + weight * (endpoint_max - endpoint_min)
         weight ∈ [0,1] 为标量插值权重
      6. 对两个子集分别计算MSE，按像素数加权得到该分区的总MSE
      7. 选择MSE最小的分区作为最优压缩结果
      8. 所有权重经过反sigmoid转换为logit空间(raw_indices)，便于后续可微优化
    
    参数:
        features: [C, H, W]，无约束连续特征图
    
    返回:
        dict:
          endpoints:     [N, 4, C]，4个端点向量（e1,e2为子集1的min/max，e3,e4为子集0的min/max）
          raw_indices:   [N, 16]，logit空间的16个像素插值权重
          partition_ids: [N]，0-31的整数分区编号
          （其中N = (H/4) * (W/4)，即4×4块的总数）
    """
    C, H, W = features.shape
    BH, BW = H // 4, W // 4

    blocks = features.permute(1, 2, 0).reshape(BH, 4, BW, 4, C)
    blocks = blocks.permute(0, 2, 1, 3, 4).reshape(-1, 16, C)
    N = BH * BW

    if N == 0:
        return {
            'endpoints': torch.empty(0, 4, C, device=features.device),
            'raw_indices': torch.empty(0, 16, device=features.device),
            'partition_ids': torch.empty(0, dtype=torch.long, device=features.device),
        }

    device = features.device
    partitions_t = torch.tensor(PARTITIONS, device=device, dtype=torch.float32)

    best_mse = torch.full((N,), float('inf'), device=device)
    best_endpoints = torch.zeros(N, 4, C, device=device)
    best_raw_indices = torch.zeros(N, 16, device=device)
    best_pids = torch.zeros(N, dtype=torch.long, device=device)

    for pid in range(32):
        mask = partitions_t[pid]          # [16]  {0,1}
        mask1_bool = mask.bool()          # True where pk==1 → subset 1  (line1: e1,e2)
        mask0_bool = ~mask1_bool          # True where pk==0 → subset 0  (line2: e3,e4)

        vals1 = blocks[:, mask1_bool, :]  # [N, k1, C]
        vals0 = blocks[:, mask0_bool, :]  # [N, k0, C]

        k1 = vals1.shape[1]
        k0 = vals0.shape[1]
        if k1 == 0 or k0 == 0:
            continue

        e1 = vals1.min(dim=1).values      # [N, C]  子集1的最小端点
        e2 = vals1.max(dim=1).values      # [N, C]  子集1的最大端点
        e3 = vals0.min(dim=1).values      # [N, C]  子集0的最小端点
        e4 = vals0.max(dim=1).values      # [N, C]  子集0的最大端点

        denom1 = e2 - e1 + 1e-10
        x1_raw = (vals1 - e1.unsqueeze(1)) / denom1.unsqueeze(1)
        x1 = torch.clamp(x1_raw, 0, 1)
        recon1 = e1.unsqueeze(1) + x1 * denom1.unsqueeze(1)

        denom0 = e4 - e3 + 1e-10
        x0_raw = (vals0 - e3.unsqueeze(1)) / denom0.unsqueeze(1)
        x0 = torch.clamp(x0_raw, 0, 1)
        recon0 = e3.unsqueeze(1) + x0 * denom0.unsqueeze(1)

        mse1 = ((recon1 - vals1) ** 2).mean(dim=(1, 2))
        mse0 = ((recon0 - vals0) ** 2).mean(dim=(1, 2))
        mse = (mse1 * k1 + mse0 * k0) / 16.0

        idx1_scalar = x1.mean(dim=-1)     # [N, k1] — 每个像素的标量插值权重
        idx0_scalar = x0.mean(dim=-1)     # [N, k0]

        full_indices = torch.zeros(N, 16, device=device)
        full_indices[:, mask1_bool] = idx1_scalar
        full_indices[:, mask0_bool] = idx0_scalar

        eps_val = 1e-6
        full_indices_clamped = full_indices.clamp(eps_val, 1.0 - eps_val)
        raw_indices = torch.log(full_indices_clamped / (1.0 - full_indices_clamped))
        raw_indices = raw_indices.clamp(-10.0, 10.0)

        endpoints_4 = torch.stack([e1, e2, e3, e4], dim=1)  # [N, 4, C]

        better = mse < best_mse
        best_mse[better] = mse[better]
        best_endpoints[better] = endpoints_4[better]
        best_raw_indices[better] = raw_indices[better]
        best_pids[better] = pid

    return {
        'endpoints': best_endpoints,
        'raw_indices': best_raw_indices,
        'partition_ids': best_pids,
    }


@torch.no_grad()
def init_bc6_model_from_unconstrained(bc6_model, unconstrained_model):
    """从无约束预热模型初始化BC6模型的特征参数。
    
    对每个特征网格(grid)的每个mip层级：
      1. 提取无约束模型的特征张量 [C, H, W]
      2. 通过compress_to_bc6将其压缩为BC6块参数
      3. 将压缩结果(endpoints、raw_indices、partition_ids)复制到BC6模型的对应层级
    
    这样可以继承预热阶段的训练成果，确保BC6训练的初始点质量较好，
    加速后续阶段2的训练收敛。
    
    参数:
        bc6_model:            目标BC6模型（将写入参数）
        unconstrained_model:  已完成预热的无约束模型（读取源参数）
    """
    for bc6_grid, unconstrained_grid in zip(bc6_model.feature_grids,
                                             unconstrained_model.feature_grids):
        for mip_idx in range(bc6_grid.num_mips):
            unconstrained_feat = unconstrained_grid.mips[mip_idx].data.squeeze(0)  # [C, H, W]
            bc6_block = bc6_grid.mips[mip_idx]

            compressed = compress_to_bc6(unconstrained_feat)

            BH = bc6_block.blocks_h
            BW = bc6_block.blocks_w
            C = unconstrained_feat.shape[0]

            bc6_block.endpoints.data.copy_(
                compressed['endpoints'].reshape(BH, BW, 4, C)
            )
            bc6_block.raw_indices.data.copy_(
                compressed['raw_indices'].reshape(BH, BW, 16)
            )
            bc6_block.partition_ids.copy_(
                compressed['partition_ids'].reshape(BH, BW)
            )


def quantize_bc6_features(bc6_model):
    """将BC6特征进行量化（6位端点 + 3位索引），原地修改。
    
    量化策略：
      - 端点(endpoints)：对整个特征网格的值域进行均匀64级量化（6位）
        先将端点值归一化到[0,1]，乘以63后取整（得到0-63的64个离散值），
        再映射回原始值域。这意味着端点最多有64种可能的取值。
      
      - 索引(indices)：对经过sigmoid的插值权重进行均匀8级量化（3位）
        raw_indices → sigmoid → [0,1] → 乘以7取整 → 除以7 → logit空间
        0/7, 1/7, ..., 7/7 共8个离散值。最终映射回logit空间供模型使用。
    
    量化后的值域会被夹紧到安全范围（索引的logit空间夹紧到[-10,10]），
    避免数值不稳定。
    
    参数:
        bc6_model: BC6模型，所有特征网格的端点和索引将被就地量化
    """
    for grid in bc6_model.feature_grids:
        for mip in grid.mips:
            eps = mip.endpoints.data
            eps_min = eps.min()
            eps_max = eps.max()
            if eps_max - eps_min > 1e-8:
                eps_norm = (eps - eps_min) / (eps_max - eps_min)
                eps_quant = (eps_norm * 63.0).round().clamp(0, 63) / 63.0
                mip.endpoints.data.copy_(eps_quant * (eps_max - eps_min) + eps_min)

            indices_prob = torch.sigmoid(mip.raw_indices.data)
            idx_quant = (indices_prob * 7.0).round().clamp(0, 7) / 7.0
            idx_safe = idx_quant.clamp(1e-6, 1.0 - 1e-6)
            idx_logit = torch.log(idx_safe / (1.0 - idx_safe))
            mip.raw_indices.data.copy_(idx_logit.clamp(-10.0, 10.0))


def freeze_bc6_features(bc6_model):
    """冻结BC6模型中所有特征参数（端点和索引），仅保留MLP可训练。
    
    将每个mip层级的endpoints和raw_indices的requires_grad设为False。
    注意partition_ids是整数索引，天然不可导，无需额外处理。
    
    参数:
        bc6_model: BC6模型，其所有块参数的梯度将被关闭
    """
    for grid in bc6_model.feature_grids:
        for mip in grid.mips:
            mip.endpoints.requires_grad_(False)
            mip.raw_indices.requires_grad_(False)


def train_bc6(model, reference_mips, total_iterations=25000, warmup_iterations=5000,
              bc6_iterations=19000, finetune_iterations=1000,
              batch_res=256, device='cuda', log_interval=1000, output_dir='output_bc6'):
    """BC6神经材质三阶段训练主流程。
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ 阶段1 - 无约束预热 (warmup_iterations步)                        │
    │   • 模型: NeuralTextureModel（无约束特征）                       │
    │   • LR:   feats=5e-2, mlp=1e-3                                 │
    │   • γ:    0.9995 (每步衰减，5k步后降至~4e-3)                    │
    │   • 目的: 让MLP和特征协同达到较好初始状态                          │
    │   • 保存: warmup_model.pth                                     │
    ├─────────────────────────────────────────────────────────────────┤
    │ 阶段2 - BC6训练 (bc6_iterations步)                              │
    │   • 模型: NeuralBC6TextureModel（BC6块压缩特征）                 │
    │   • 初始化: init_bc6_model_from_unconstrained从预热模型导入       │
    │   • LR:   feats=1e-2, mlp=1e-3                                 │
    │   • γ:    0.9999 (缓慢衰减，19k步后降至~1.5e-3)                  │
    │   • 训练参数: endpoints(端点) + raw_indices(索引权重) + MLP       │
    │   • 保存: best_model.pth(最优PSNR), checkpoint每10k步            │
    ├─────────────────────────────────────────────────────────────────┤
    │ 阶段3 - 量化+精调 (finetune_iterations步)                        │
    │   • quantize_bc6_features: 端点→6位，索引→3位                    │
    │   • freeze_bc6_features: 冻结所有BC6块参数                       │
    │   • 仅精调MLP, LR=1e-3, 无学习率衰减                             │
    │   • 目的: 补偿量化引入的误差                                     │
    │   • 保存: best_model.pth(如PSNR提升), final_model.pth           │
    └─────────────────────────────────────────────────────────────────┘
    
    批次采样策略（三阶段通用）：
      - 随机选择一个UV矩形块（batch_res×batch_res）
      - 随机选择一个连续的mip层级（scale∈[0, num_mips-1]）
      - 从参考mipmap中通过bicubic+三线性插值获取ground truth
      - 模型仅在采样的块和层级上进行推理，损失为MSE
    
    参数:
        model:              NeuralTextureModel，用于阶段1的初始模型
        reference_mips:     list of [C, H_i, W_i]，参考mipmap金字塔
        total_iterations:   总迭代次数（默认25000）
        warmup_iterations:  阶段1迭代次数（默认5000）
        bc6_iterations:     阶段2迭代次数（默认19000）
        finetune_iterations:阶段3迭代次数（默认1000）
        batch_res:          训练批次的UV采样分辨率（默认256）
        device:             计算设备（默认'cuda'）
        log_interval:       日志输出间隔（默认1000步）
        output_dir:         模型和检查点保存目录
    
    返回:
        best_psnr: 训练过程中达到的最佳PSNR值(dB)
    """
    os.makedirs(output_dir, exist_ok=True)
    reference_mips = [m.to(device) for m in reference_mips]
    num_mips_ref = len(reference_mips)

    # =========================================================================
    # 阶段1 – 无约束特征预热训练
    # 目标：使MLP与特征协同训练到较优状态，为后续BC6压缩提供良好初始点
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Phase 1: Warmup ({warmup_iterations} iterations) — unconstrained")
    print(f"  LR: feat=5e-2, mlp=1e-3, gamma=0.9995")
    print(f"{'='*60}")

    model = model.to(device)
    p1_feature_params = list(model.feature_grids.parameters())
    p1_mlp_params = list(model.mlp.parameters())
    optimizer = torch.optim.Adam([
        {'params': p1_feature_params, 'lr': 5e-2},
        {'params': p1_mlp_params, 'lr': 1e-3},
    ])
    gamma_p1 = 0.9995
    best_psnr = 0.0
    best_mse = float('inf')

    for iteration in range(warmup_iterations):
        model.train()
        B = 1
        H = W = batch_res
        ref_w = reference_mips[0].shape[1]
        ref_h = reference_mips[0].shape[2]

        # 随机采样UV块：在参考纹理范围内随机选取一个batch_res×batch_res的矩形区域
        u0 = torch.rand(1, device=device) * (1.0 - W / ref_w)
        v0 = torch.rand(1, device=device) * (1.0 - H / ref_h)
        u_vals = torch.linspace(u0.item(), u0.item() + W / ref_w, W, device=device)
        v_vals = torch.linspace(v0.item(), v0.item() + H / ref_h, H, device=device)
        ug, vg = torch.meshgrid(u_vals, v_vals, indexing='xy')
        uv = torch.stack([ug, vg], dim=-1).unsqueeze(0)
        # 随机连续mip层级
        scale = torch.rand(B, device=device) * (num_mips_ref - 1)

        with torch.no_grad():
            ref = sample_reference(reference_mips, uv, scale)

        pred = model(uv, scale)
        loss = F.mse_loss(pred, ref)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 学习率指数衰减
        for pg in optimizer.param_groups:
            pg['lr'] *= gamma_p1

        if (iteration + 1) % log_interval == 0:
            with torch.no_grad():
                model.eval()
                psnr, mse_val = evaluate_full(model, reference_mips, device)
                print(f"  Warmup Iter {iteration+1:>6d}/{warmup_iterations} | "
                      f"Loss: {loss.item():.6f} | PSNR: {psnr:.2f} dB | "
                      f"LR_feat: {optimizer.param_groups[0]['lr']:.6f}")
                model.train()

    # 保存阶段1预热模型的检查点
    torch.save({
        'phase': 'warmup',
        'iteration': warmup_iterations,
        'model_state_dict': model.state_dict(),
    }, os.path.join(output_dir, 'warmup_model.pth'))
    print(f"  Warmup complete. Saved to {output_dir}/warmup_model.pth")

    # =========================================================================
    # 阶段2 – BC6块压缩训练
    # 从预热模型中提取压缩特征初始化BC6模型，端到端训练endpoints和indices
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Phase 2: BC6 training ({bc6_iterations} iterations)")
    print(f"  LR: feat=1e-2, mlp=1e-3, gamma=0.9999")
    print(f"{'='*60}")

    output_dim = model.output_dim
    bc6_model = make_bc6_model(reference_resolution=1024, output_dim=output_dim).to(device)

    # 从预热模型导入BC6压缩后的初始参数
    init_bc6_model_from_unconstrained(bc6_model, model)

    # 收集BC6特有的可训练参数：端点和插值权重（分区ID和MLP需分别处理）
    p2_feature_params = []
    for grid in bc6_model.feature_grids:
        for mip in grid.mips:
            p2_feature_params.append(mip.endpoints)
            p2_feature_params.append(mip.raw_indices)
    p2_mlp_params = list(bc6_model.mlp.parameters())

    optimizer = torch.optim.Adam([
        {'params': p2_feature_params, 'lr': 1e-2},
        {'params': p2_mlp_params, 'lr': 1e-3},
    ])
    gamma_p2 = 0.9999

    for iteration in range(bc6_iterations):
        bc6_model.train()
        B = 1
        H = W = batch_res
        ref_w = reference_mips[0].shape[1]
        ref_h = reference_mips[0].shape[2]

        # 相同的随机UV块+随机mip层级采样策略
        u0 = torch.rand(1, device=device) * (1.0 - W / ref_w)
        v0 = torch.rand(1, device=device) * (1.0 - H / ref_h)
        u_vals = torch.linspace(u0.item(), u0.item() + W / ref_w, W, device=device)
        v_vals = torch.linspace(v0.item(), v0.item() + H / ref_h, H, device=device)
        ug, vg = torch.meshgrid(u_vals, v_vals, indexing='xy')
        uv = torch.stack([ug, vg], dim=-1).unsqueeze(0)
        scale = torch.rand(B, device=device) * (num_mips_ref - 1)

        with torch.no_grad():
            ref = sample_reference(reference_mips, uv, scale)

        pred = bc6_model(uv, scale)
        loss = F.mse_loss(pred, ref)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for pg in optimizer.param_groups:
            pg['lr'] *= gamma_p2

        if (iteration + 1) % log_interval == 0:
            with torch.no_grad():
                bc6_model.eval()
                psnr, mse_val = evaluate_full(bc6_model, reference_mips, device)
                print(f"  BC6    Iter {iteration+1:>6d}/{bc6_iterations} | "
                      f"Loss: {loss.item():.6f} | PSNR: {psnr:.2f} dB | "
                      f"LR_feat: {optimizer.param_groups[0]['lr']:.6f}")
                bc6_model.train()

                # 追踪并保存全局最优PSNR的模型
                if mse_val < best_mse:
                    best_mse = mse_val
                    best_psnr = psnr
                    torch.save({
                        'phase': 'bc6',
                        'iteration': warmup_iterations + iteration + 1,
                        'model_state_dict': bc6_model.state_dict(),
                        'psnr': best_psnr,
                    }, os.path.join(output_dir, 'best_model.pth'))

        # 每10000步保存一个检查点用于恢复训练
        if (iteration + 1) % 10000 == 0:
            torch.save({
                'phase': 'bc6',
                'iteration': warmup_iterations + iteration + 1,
                'model_state_dict': bc6_model.state_dict(),
            }, os.path.join(output_dir, f'checkpoint_{warmup_iterations + iteration + 1}.pth'))

    # =========================================================================
    # 阶段3 – 量化BC6特征并精调MLP
    # 将特征量化到目标位宽后冻结，仅通过微调MLP补偿量化带来的精度损失
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Phase 3: Quantize + finetune ({finetune_iterations} iterations)")
    print(f"  LR: mlp=1e-3 (features frozen)")
    print(f"{'='*60}")

    # 先量化（6位端点+3位索引），再冻结全部特征参数
    quantize_bc6_features(bc6_model)
    freeze_bc6_features(bc6_model)

    # 仅MLP参与训练
    p3_mlp_params = list(bc6_model.mlp.parameters())
    optimizer = torch.optim.Adam(p3_mlp_params, lr=1e-3)

    for iteration in range(finetune_iterations):
        bc6_model.train()
        B = 1
        H = W = batch_res
        ref_w = reference_mips[0].shape[1]
        ref_h = reference_mips[0].shape[2]

        u0 = torch.rand(1, device=device) * (1.0 - W / ref_w)
        v0 = torch.rand(1, device=device) * (1.0 - H / ref_h)
        u_vals = torch.linspace(u0.item(), u0.item() + W / ref_w, W, device=device)
        v_vals = torch.linspace(v0.item(), v0.item() + H / ref_h, H, device=device)
        ug, vg = torch.meshgrid(u_vals, v_vals, indexing='xy')
        uv = torch.stack([ug, vg], dim=-1).unsqueeze(0)
        scale = torch.rand(B, device=device) * (num_mips_ref - 1)

        with torch.no_grad():
            ref = sample_reference(reference_mips, uv, scale)

        pred = bc6_model(uv, scale)
        loss = F.mse_loss(pred, ref)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iteration + 1) % log_interval == 0:
            with torch.no_grad():
                bc6_model.eval()
                psnr, mse_val = evaluate_full(bc6_model, reference_mips, device)
                print(f"  FT      Iter {iteration+1:>6d}/{finetune_iterations} | "
                      f"Loss: {loss.item():.6f} | PSNR: {psnr:.2f} dB")
                bc6_model.train()

                if mse_val < best_mse:
                    best_mse = mse_val
                    best_psnr = psnr
                    torch.save({
                        'phase': 'finetune',
                        'iteration': warmup_iterations + bc6_iterations + iteration + 1,
                        'model_state_dict': bc6_model.state_dict(),
                        'psnr': best_psnr,
                    }, os.path.join(output_dir, 'best_model.pth'))

    # 保存最终的量化精调模型
    torch.save({
        'phase': 'finetune',
        'iteration': warmup_iterations + bc6_iterations + finetune_iterations,
        'model_state_dict': bc6_model.state_dict(),
    }, os.path.join(output_dir, 'final_model.pth'))

    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")
    return best_psnr


if __name__ == '__main__':
    """
    主入口：BC6神经材质三阶段训练的完整执行流程。
    
    步骤：
      1. 自动检测设备（CUDA > CPU）
      2. 设置随机种子(42)确保可复现
      3. 加载StoneBricksSplitface砖墙PBR材质 → 9通道参考纹理(1024×1024)
      4. 构建10级mipmap金字塔（1024→512→256→...→2）
      5. 创建NeuralTextureModel作为初始无约束模型
      6. 执行三阶段训练（预热→BC6→量化精调）
      7. 输出总训练时间和最佳PSNR
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    torch.manual_seed(42)

    ref_tensor = load_brick_material('.', target_res=1024)
    print(f"Reference shape: {ref_tensor.shape}")

    reference_mips = build_mipmaps(ref_tensor.to('cpu'))
    print(f"Mip levels: {[m.shape for m in reference_mips]}")

    unconstrained_model = make_model(reference_resolution=1024, output_dim=ref_tensor.shape[0])
    total_params = sum(p.numel() for p in unconstrained_model.parameters())
    print(f"Unconstrained model parameters: {total_params:,}")

    start_time = time.time()
    best_psnr = train_bc6(
        unconstrained_model, reference_mips,
        total_iterations=25000,
        warmup_iterations=5000,
        bc6_iterations=19000,
        finetune_iterations=1000,
        batch_res=384,
        device=device,
        log_interval=1000,
        output_dir='output_bc6',
    )
    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed/60:.1f} minutes. Best PSNR: {best_psnr:.2f} dB")
