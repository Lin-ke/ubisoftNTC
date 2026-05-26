"""
BC6神经材质推理与可视化
===========================

从训练好的BC6神经纹理模型checkpoint加载模型，重建所有材质层（反照率、法线、AO、粗糙度、金属度），
可视化推理结果，计算各项PSNR指标，以及评估BC6压缩后的模型大小。

支持的推理模式（通过 --mode 参数选择）：
  - full   : 从checkpoint加载模型，重建所有材质层，保存预测图像和参考图像，打印各通道PSNR
  - mips   : 跨所有mip级别评估PSNR，对比重建结果与参考在各分辨率下的精度
  - size   : 计算BC6压缩后特征网格的大小，与未压缩的fp16/fp32存储对比压缩率
  - compare: 重建结果与参考的逐像素对比，生成误差图（*_diff.png），打印各通道PSNR

BC6格式（每4x4像素块）：
  - 4个端点（endpoint），每个6位 = 24 bits
  - 16个索引（index），每个3位 = 48 bits
  - 5位分区（partition）= 5 bits
  - 每块总计77 bits ≈ 10 bytes

checkpoint内容：
  - model_state_dict: 包含 'feature_grids.X.mips.Y.raw_endpoints', 'mlp.*' 等参数
  - feature_grids 以 BC6 压缩格式存储（endpoint0, endpoint1, index, partition）

输出目录：output_bc6/
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import argparse

from ntc_bc6_model import make_bc6_model
from ntc_train import load_brick_material, build_mipmaps


def reconstruct_normal(normal_xy):
    """从XY分量重建法线的Z分量。

    法线是单位向量，满足 x² + y² + z² = 1。
    给定已预测的XY分量，通过 z = sqrt(1 - x² - y²) 反推出Z分量。
    使用torch.clamp避免因浮点误差导致负数开根号。

    Args:
        normal_xy: 形状为 [B, 2, H, W] 的tensor，包含法线的X和Y分量

    Returns:
        形状为 [B, 3, H, W] 的tensor，完整的XYZ法线向量
    """
    xy = normal_xy
    # 确保 1 - x² - y² 不小于0，避免NaN
    z = torch.sqrt(torch.clamp(1.0 - xy[:, 0:1] ** 2 - xy[:, 1:2] ** 2, min=0))
    return torch.cat([xy, z], dim=1)


def save_image(tensor, path, is_normal=False):
    """将形如 [C, H, W] 的tensor保存为图像文件。

    处理流程：tensor → numpy数组（HWC格式） → 值域映射 → uint8 → PIL保存。
    对于法线图，需要将 [-1, 1] 范围映射到 [0, 1]。

    Args:
        tensor: 形状为 [C, H, W] 的torch tensor，值在 [0, 1] 之间（法线除外）
        path:   输出图像文件路径
        is_normal: 是否为法线图，法线值域为 [-1, 1]，需要做归一化映射
    """
    # 从GPU移到CPU，转为numpy，并重排为 [H, W, C] 格式
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    if is_normal:
        img = (img + 1.0) / 2.0  # 法线值域映射：[-1, 1] → [0, 1]
    # 裁剪到合法范围，防止保存时溢出
    img = np.clip(img, 0, 1)
    # 转换为8位整数 [0, 255]
    img = (img * 255).astype(np.uint8)
    # 单通道图像去掉最后一维，变为 [H, W]
    if img.shape[2] == 1:
        img = img[:, :, 0]
    Image.fromarray(img).save(path)


@torch.no_grad()
def infer_from_checkpoint(checkpoint_path, device='cuda'):
    """从BC6模型checkpoint加载模型，重建所有材质层并保存图像，打印各通道PSNR。

    加载checkpoint中的model_state_dict到BC6模型，在1024x1024分辨率下推理，
    将预测的反照率、法线、环境光遮蔽(AO)、粗糙度、金属度分别保存为PNG图像。
    同时加载原始参考材质（从.目录的源数据）作为真值参考，计算并打印每层PSNR。

    模型输出9个通道：RGB反照率(0-2)、法线XY(3-4)、未知/法线Z(5)、AO(6)、
    粗糙度(7)、金属度(8)。其中通道5（pred[:, 5:6]）未使用，法线Z由XY分量重建。

    Args:
        checkpoint_path: BC6模型checkpoint文件路径
        device:          'cuda' 或 'cpu'
    """
    # 加载参考材质（真值），分辨率1024x1024
    ref = load_brick_material('.', target_res=1024).to(device)
    output_dim = ref.shape[0]
    h, w = 1024, 1024

    # 创建BC6模型并从checkpoint加载权重
    model = make_bc6_model(reference_resolution=1024, output_dim=output_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 构建 [1, 1024, 1024, 2] 的UV坐标网格
    u = torch.linspace(0, 1, w, device=device)
    v = torch.linspace(0, 1, h, device=device)
    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)

    # scale=0 表示使用最高分辨率（mip level 0）
    scale = torch.zeros(1, device=device)
    # 推理得到 [1, 9, 1024, 1024] 的材质tensor
    pred = model(uv, scale)  # [1, 9, 1024, 1024]

    os.makedirs('output_bc6', exist_ok=True)

    # 从9通道输出中提取各材质层
    albedo = pred[:, 0:3]        # RGB反照率，通道 0-2
    normal_xy = pred[:, 3:5]     # 法线XY分量，通道 3-4
    # pred[:, 5:6] 未使用（占总通道数的一部分）
    ao = pred[:, 6:7]            # 环境光遮蔽，通道 6
    roughness = pred[:, 7:8]     # 粗糙度，通道 7
    metalness = pred[:, 8:9]     # 金属度，通道 8

    # 从XY分量重建完整的XYZ法线
    normal_full = reconstruct_normal(normal_xy)

    # 保存预测图像
    save_image(albedo.squeeze(0), 'output_bc6/albedo_pred.png')
    save_image(normal_full.squeeze(0), 'output_bc6/normal_pred.png', is_normal=True)
    save_image(ao.squeeze(0), 'output_bc6/ao_pred.png')
    save_image(roughness.squeeze(0), 'output_bc6/roughness_pred.png')
    save_image(metalness.squeeze(0), 'output_bc6/metalness_pred.png')

    # 准备参考（真值）各材质层
    ref_albedo = ref[0:3].unsqueeze(0)
    ref_normal = ref[3:6].unsqueeze(0)
    ref_ao = ref[6:7].unsqueeze(0)
    ref_roughness = ref[7:8].unsqueeze(0)
    ref_metalness = ref[8:9].unsqueeze(0)

    # 保存参考图像
    save_image(ref_albedo.squeeze(0), 'output_bc6/albedo_ref.png')
    save_image(ref_normal.squeeze(0), 'output_bc6/normal_ref.png', is_normal=True)
    save_image(ref_ao.squeeze(0), 'output_bc6/ao_ref.png')
    save_image(ref_roughness.squeeze(0), 'output_bc6/roughness_ref.png')
    save_image(ref_metalness.squeeze(0), 'output_bc6/metalness_ref.png')

    # 计算并打印各通道PSNR（峰值信噪比）
    print(f"{'Channel':>12s}  {'PSNR (dB)':>10s}")
    print('-' * 28)
    for name, p, r in [
        ('Albedo', albedo, ref_albedo),
        ('Normal', normal_full, ref_normal),
        ('AO', ao, ref_ao),
        ('Roughness', roughness, ref_roughness),
        ('Metalness', metalness, ref_metalness),
    ]:
        mse = F.mse_loss(p, r).item()
        # PSNR = -10 * log10(MSE)，加1e-8防止log(0)
        psnr = -10 * np.log10(mse + 1e-8)
        print(f"{name:>12s}  {psnr:>10.2f}")

    # 计算整体PSNR（所有9个通道一起）
    mse_total = F.mse_loss(pred, ref.unsqueeze(0)).item()
    psnr_total = -10 * np.log10(mse_total + 1e-8)
    print(f"{'Total':>12s}  {psnr_total:>10.2f}")

    print("\nSaved all images to output_bc6/ directory")


@torch.no_grad()
def infer_mip_comparison(checkpoint_path, device='cuda'):
    """跨mip级别PSNR评估：在不同分辨率下评估模型重建精度。

    对参考材质构建多层mipmap（逐级降采样），在每个mip级别上分别运行模型推理，
    计算预测与参考的PSNR。mip level 0 是最高分辨率（1024x1024），
    后续级别依次降采样（512x512, 256x256, ...）。

    Args:
        checkpoint_path: BC6模型checkpoint文件路径
        device:          'cuda' 或 'cpu'
    """
    # 加载参考材质（最高分辨率），用于构建mipmap链
    ref = load_brick_material('.', target_res=1024).to(device)
    output_dim = ref.shape[0]

    # 创建BC6模型并从checkpoint加载权重
    model = make_bc6_model(reference_resolution=1024, output_dim=output_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 构建mipmap链：从最高分辨率逐级降采样到更低分辨率
    ref_mips = build_mipmaps(ref.cpu())

    os.makedirs('output_bc6', exist_ok=True)
    print(f"{'Mip':>4s}  {'Res':>8s}  {'PSNR(dB)':>10s}")
    print('-' * 30)

    for mip_level in range(len(ref_mips)):
        ref_mip = ref_mips[mip_level].to(device)
        size = ref_mip.shape[1]  # 当前mip级别的分辨率（正方形）

        # 构建当前分辨率下的UV坐标网格
        u = torch.linspace(0, 1, size, device=device)
        v = torch.linspace(0, 1, size, device=device)
        uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)
        # scale 指定mip级别，模型内部会根据scale选择对应的特征网格分辨率
        scale = torch.tensor([float(mip_level)], device=device)

        # 推理并计算PSNR
        pred = model(uv, scale)
        ref_batch = ref_mip.unsqueeze(0)

        mse = F.mse_loss(pred, ref_batch).item()
        psnr = -10 * np.log10(mse + 1e-8)  # 加1e-8防止log(0)
        print(f"{mip_level:>4d}  {size:>4d}x{size:<4d}  {psnr:>10.2f}")


@torch.no_grad()
def compute_model_size(checkpoint_path):
    """计算BC6压缩后特征网格的大小，并与未压缩存储进行对比。

    从checkpoint的model_state_dict中解析出BC6压缩后的各mip层级分辨率，
    按BC6格式的每块位数计算总字节数。同时统计MLP参数大小，最终计算整体压缩率。

    BC6格式（每4x4像素块）：
      - 4个端点(endpoint) × 6 bits = 24 bits
      - 16个索引(index) × 3 bits = 48 bits
      - 5位分区(partition) = 5 bits
      - 总计：77 bits/块（实际存储按10字节/块对齐，即80 bits）

    对于每个mip级别：总块数 = (分辨率/4) × (分辨率/4)，再 × 10 bytes

    Args:
        checkpoint_path: BC6模型checkpoint文件路径
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']

    # 从BC6压缩特征网格的state_dict中提取各mip分辨率
    # BC6模型的feature_grids按mip层级组织，可从endpoint的shape反推分辨率
    mip_resolutions = []
    for key in state:
        if 'endpoint0' in key or 'endpoint1' in key:
            # shape: [channels, num_blocks_flat]，其中 num_blocks_flat = (res/4)^2
            num_blocks_flat = state[key].shape[-1]
            res = int(np.sqrt(num_blocks_flat)) * 4  # sqrt得到每维块数，×4还原分辨率
            if res not in mip_resolutions:
                mip_resolutions.append(res)
    mip_resolutions.sort(reverse=True)  # 从高分辨率到低分辨率排序

    # 如果无法从checkpoint检测到分辨率，使用默认值
    if not mip_resolutions:
        print("Warning: could not detect BC6 block sizes from checkpoint.")
        print("Assuming default resolutions: [512, 256, 128, 64]")
        mip_resolutions = [512, 256, 128, 64]

    # BC6每块位数计算
    # 4个端点 × 6位 + 16个索引 × 3位 + 5位分区 = 77位
    # 实际存储向上取整到整数字节：10字节/块（80位）
    bits_per_block = 4 * 6 + 16 * 3 + 5  # 77 bits
    bytes_per_block = (bits_per_block + 7) // 8  # 向上取整 = 10 bytes

    total_bc6_bytes = 0
    print("\nBC6 Feature Grid Size:")
    print(f"{'Mip:'}  {'Resolution':>10s}  {'Blocks':>10s}  {'Bytes':>12s}")
    print('-' * 50)
    for i, res in enumerate(mip_resolutions):
        blocks = (res // 4) * (res // 4)  # 该分辨率下的4×4块数
        mip_bytes = blocks * bytes_per_block
        total_bc6_bytes += mip_bytes
        print(f"  {i}    {res:>5d}x{res:<5d}  {blocks:>10d}  {mip_bytes:>12,d}")

    print('-' * 50)
    print(f"{'Total BC6 features:'}  {total_bc6_bytes:>34,d} bytes")

    # MLP权重大小：统计所有mlp相关参数的数量
    mlp_params = sum(v.numel() for k, v in state.items() if 'mlp' in k)
    # fp16: 每个参数2字节；fp32: 每个参数4字节
    mlp_bytes_fp16 = mlp_params * 2
    print(f"\nMLP weights (fp16): {mlp_params} params x 2 bytes = {mlp_bytes_fp16:,} bytes")
    print(f"MLP weights (fp32): {mlp_params} params x 4 bytes = {mlp_params * 4:,} bytes")

    # 总压缩大小 = BC6特征 + MLP权重（fp16）
    total_compressed = total_bc6_bytes + mlp_bytes_fp16
    print(f"\n{'Total compressed size (BC6 + MLP fp16):':>38s}  {total_compressed:>12,d} bytes")

    # 与未压缩特征存储对比
    # 未压缩时，每个像素直接存储feature_channels个值
    total_features = sum((res * res) for res in mip_resolutions)
    feature_channels = None
    for key in state:
        if 'endpoint0' in key:
            feature_channels = state[key].shape[0]  # 第一个维度是通道数
            break
    feature_channels = feature_channels or 3  # 默认3通道

    # 计算未压缩存储大小：像素数 × 通道数 × 每值字节数 + MLP权重
    uncomp_fp16 = total_features * feature_channels * 2 + mlp_bytes_fp16
    uncomp_fp32 = total_features * feature_channels * 4 + mlp_params * 4
    print(f"  Uncompressed (fp16): {total_compressed - uncomp_fp16:>14,d} bytes difference ({total_compressed/max(uncomp_fp16,1):.1f}x)")
    print(f"  Uncompressed (fp32): {total_compressed - uncomp_fp32:>14,d} bytes difference ({total_compressed/max(uncomp_fp32,1):.1f}x)")

    # 纯特征部分的压缩率对比
    features_fp16 = total_features * feature_channels * 2
    print(f"\nCompression ratio (features only): BC6={total_bc6_bytes:,}B vs fp16={features_fp16:,}B = {features_fp16/max(total_bc6_bytes,1):.1f}x")


@torch.no_grad()
def compare_methods(checkpoint_path, device='cuda'):
    """重建结果与参考的逐像素对比：生成预测图、参考图和误差图的对比。

    在1024x1024分辨率下推理BC6模型，对每个材质层（反照率、法线、AO、粗糙度、金属度）
    分别保存三张图像：
      - *_pred.png: 模型预测结果
      - *_ref.png:  原始参考（真值）
      - *_diff.png: 逐像素绝对误差图（误差越亮表示偏差越大）

    同时计算并打印各通道及总体的PSNR值。

    Args:
        checkpoint_path: BC6模型checkpoint文件路径
        device:          'cuda' 或 'cpu'
    """
    # 加载参考材质（真值）
    ref = load_brick_material('.', target_res=1024).to(device)
    output_dim = ref.shape[0]
    h, w = 1024, 1024

    # 创建BC6模型并加载权重
    model = make_bc6_model(reference_resolution=1024, output_dim=output_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 构建UV坐标网格
    u = torch.linspace(0, 1, w, device=device)
    v = torch.linspace(0, 1, h, device=device)
    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)

    # scale=0，最高分辨率推理
    scale = torch.zeros(1, device=device)
    pred = model(uv, scale)  # [1, 9, 1024, 1024]

    os.makedirs('output_bc6', exist_ok=True)

    # 提取各材质层
    albedo = pred[:, 0:3]
    normal_xy = pred[:, 3:5]
    ao = pred[:, 6:7]
    roughness = pred[:, 7:8]
    metalness = pred[:, 8:9]

    # 从XY分量重建完整法线
    normal_full = reconstruct_normal(normal_xy)

    # 准备参考各材质层
    ref_albedo = ref[0:3].unsqueeze(0)
    ref_normal = ref[3:6].unsqueeze(0)
    ref_ao = ref[6:7].unsqueeze(0)
    ref_roughness = ref[7:8].unsqueeze(0)
    ref_metalness = ref[8:9].unsqueeze(0)

    # 计算各通道的逐像素绝对误差图
    # 误差图 = |预测 - 参考|，多通道时取均值（对RGB反照率和XYZ法线取通道均值）
    diff_albedo = (albedo - ref_albedo).abs().mean(dim=1, keepdim=True)
    diff_normal = (normal_full - ref_normal).abs().mean(dim=1, keepdim=True)
    diff_ao = (ao - ref_ao).abs()
    diff_roughness = (roughness - ref_roughness).abs()
    diff_metalness = (metalness - ref_metalness).abs()

    # 分别保存预测图、参考图和误差图
    for name, pred_img, ref_img, diff_img in [
        ('albedo', albedo, ref_albedo, diff_albedo),
        ('normal', normal_full, ref_normal, diff_normal),
        ('ao', ao, ref_ao, diff_ao),
        ('roughness', roughness, ref_roughness, diff_roughness),
        ('metalness', metalness, ref_metalness, diff_metalness),
    ]:
        save_image(pred_img.squeeze(0), f'output_bc6/{name}_pred.png',
                   is_normal=(name == 'normal'))
        save_image(ref_img.squeeze(0), f'output_bc6/{name}_ref.png',
                   is_normal=(name == 'normal'))
        save_image(diff_img.squeeze(0), f'output_bc6/{name}_diff.png',
                   is_normal=False)

    # 打印各通道PSNR
    print(f"{'Channel':>12s}  {'PSNR (dB)':>10s}")
    print('-' * 28)

    layer_psnrs = {}
    for name, p, r in [
        ('Albedo', albedo, ref_albedo),
        ('Normal', normal_full, ref_normal),
        ('AO', ao, ref_ao),
        ('Roughness', roughness, ref_roughness),
        ('Metalness', metalness, ref_metalness),
    ]:
        mse = F.mse_loss(p, r).item()
        psnr = -10 * np.log10(mse + 1e-8)
        layer_psnrs[name] = psnr
        print(f"{name:>12s}  {psnr:>10.2f}")

    # 整体PSNR（9个通道一起计算MSE）
    mse_total = F.mse_loss(pred, ref.unsqueeze(0)).item()
    psnr_total = -10 * np.log10(mse_total + 1e-8)
    print('-' * 28)
    print(f"{'Total':>12s}  {psnr_total:>10.2f}")

    print("\nError maps saved as *_diff.png in output_bc6/")


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='BC6 Neural Texture Inference')
    parser.add_argument('--checkpoint', type=str, default='output_bc6/best_model.pth',
                       help='BC6模型checkpoint文件路径')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'mips', 'size', 'compare'],
                       help='推理模式：full=完整重建, mips=跨mip级PSNR, size=压缩率对比, compare=逐像素误差')
    args = parser.parse_args()

    # 自动选择设备：优先使用CUDA，不可用时回退到CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    # 检查checkpoint文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Please run BC6 training first.")
    else:
        # 根据mode参数分发到不同的推理函数
        if args.mode == 'full':
            infer_from_checkpoint(args.checkpoint, device)
        elif args.mode == 'mips':
            infer_mip_comparison(args.checkpoint, device)
        elif args.mode == 'size':
            compute_model_size(args.checkpoint)
        elif args.mode == 'compare':
            compare_methods(args.checkpoint, device)
