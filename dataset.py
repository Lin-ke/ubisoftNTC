"""
PBR 材质数据集加载器
====================
从 dataset/ 目录加载 40 个随机材质，每个材质包含三张 2K PNG：
  - *_arm_2k.png  → R: AO, G: Roughness, B: Metalness
  - *_diff_2k.png → RGB: Albedo (范围 [0, 1])
  - *_nor_dx_2k.png → RGB: Normal (DirectX, 文件范围 [0,1], 转到 [-1,1])

输出 9 通道参考张量:
  albedo(3) + normal(3) + ao(1) + roughness(1) + metalness(1)
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def build_mipmaps(tensor):
    """构建 mipmap 金字塔。

    Args:
        tensor: [C, H, W] 单材质参考张量

    Returns:
        list of [C, H_i, W_i], 从最高分辨率到最低
    """
    mips = [tensor]
    h, w = tensor.shape[1], tensor.shape[2]
    while min(h, w) > 1:
        h, w = max(1, h // 2), max(1, w // 2)
        mip = F.interpolate(mips[-1].unsqueeze(0), size=(h, w), mode='area').squeeze(0)
        mips.append(mip)
    return mips


def load_material(material_dir, target_res=1024, output_channels='full'):
    """从材质目录加载并组装参考张量。

    查找文件:
      - *_arm_2k.png   (AO / Roughness / Metalness)
      - *_diff_2k.png  (Albedo)
      - *_nor_dx_2k.png (Normal, DirectX)

    Args:
        material_dir:      材质子目录路径, 如 d:/ntc/dataset/concrete_wall_006/
        target_res:        目标分辨率 (正方形), 默认 1024
        output_channels:   'full' → 9ch (albedo+normal+ao+roughness+metalness)
                           'compact' → 6ch (albedo+normal_xy+ao)

    Returns:
        ref_tensor: [C, H, W] float32 张量
    """
    pngs = [f for f in os.listdir(material_dir) if f.endswith('.png')]

    arm_path = diff_path = nor_path = None
    for fname in pngs:
        full = os.path.join(material_dir, fname)
        if '_arm' in fname:
            arm_path = full
        elif '_diff' in fname:
            diff_path = full
        elif '_nor' in fname:
            nor_path = full

    if not all([arm_path, diff_path, nor_path]):
        raise FileNotFoundError(
            f"缺少贴图文件: arm={arm_path}, diff={diff_path}, nor={nor_path}"
        )

    # --- Albedo (RGB, [0,1]) ---
    albedo = np.array(Image.open(diff_path).convert('RGB'), dtype=np.float32) / 255.0

    # --- Normal (RGB, [0,1] → [-1,1]) ---
    normal = np.array(Image.open(nor_path).convert('RGB'), dtype=np.float32) / 255.0
    normal = normal * 2.0 - 1.0

    # --- ARM: R=AO, G=Roughness, B=Metalness ---
    arm = np.array(Image.open(arm_path).convert('RGB'), dtype=np.float32) / 255.0
    ao = arm[..., 0:1]         # R → AO
    roughness = arm[..., 1:2]  # G → Roughness
    metalness = arm[..., 2:3]  # B → Metalness

    if output_channels == 'compact':
        normal_xy = normal[..., 0:2]   # 只取 X, Y
        reference = np.concatenate([albedo, normal_xy, ao], axis=-1)  # 3+2+1=6
    else:
        reference = np.concatenate([albedo, normal, ao, roughness, metalness], axis=-1)  # 3+3+1+1+1=9

    ref_tensor = torch.from_numpy(reference).permute(2, 0, 1).float()

    # 统一分辨率
    if target_res is not None:
        h, w = ref_tensor.shape[1], ref_tensor.shape[2]
        if h != target_res or w != target_res:
            ref_tensor = F.interpolate(
                ref_tensor.unsqueeze(0),
                size=(target_res, target_res),
                mode='bicubic', align_corners=False,
            ).squeeze(0)

    return ref_tensor


class MaterialDataset(Dataset):
    """多材质数据集。

    每个样本返回:
      {
          'name': str,          材质名
          'ref_tensor': [9, H, W],  9 通道参考张量 (最高分辨率)
          'mipmaps': list of [9, H_i, W_i],  mipmap 金字塔
      }

    Args:
        root_dir:   dataset 根目录 (如 d:/ntc/dataset/)
        target_res: 统一分辨率, None 保持原分辨率
        preload:    是否在 __init__ 时预加载全部材质 (默认 True)
    """

    def __init__(self, root_dir, target_res=1024, preload=True, output_channels='full'):
        self.root_dir = root_dir
        self.target_res = target_res
        self.output_channels = output_channels

        # 扫描材质子目录 (每个含有 .png 文件的目录)
        self.materials = []
        for d in sorted(os.listdir(root_dir)):
            dpath = os.path.join(root_dir, d)
            if os.path.isdir(dpath):
                has_png = any(f.endswith('.png') for f in os.listdir(dpath))
                if has_png:
                    self.materials.append(d)

        if len(self.materials) == 0:
            raise RuntimeError(f"在 {root_dir} 中未找到材质目录")

        self._data = {}
        if preload:
            for name in self.materials:
                self._load_one(name)

    def _load_one(self, name):
        """加载单个材质并构建 mipmap。"""
        dpath = os.path.join(self.root_dir, name)
        ref = load_material(dpath, self.target_res, self.output_channels)
        mips = build_mipmaps(ref)
        self._data[name] = {'ref_tensor': ref, 'mipmaps': mips}

    def __len__(self):
        return len(self.materials)

    def __getitem__(self, idx):
        name = self.materials[idx]
        if name not in self._data:
            self._load_one(name)
        entry = self._data[name]
        return {
            'name': name,
            'ref_tensor': entry['ref_tensor'],
            'mipmaps': entry['mipmaps'],
        }

    def get_by_name(self, name):
        """按材质名获取。"""
        if name not in self._data:
            self._load_one(name)
        return self._data[name]

    @property
    def material_names(self):
        return self.materials


if __name__ == '__main__':
    import sys

    root = 'dataset'
    if len(sys.argv) > 1:
        root = sys.argv[1]

    ds = MaterialDataset(root, target_res=1024)
    print(f"加载 {len(ds)} 个材质")

    for i in range(min(3, len(ds))):
        sample = ds[i]
        t = sample['ref_tensor']
        mips = sample['mipmaps']
        print(f"  [{i}] {sample['name']}: ref={list(t.shape)}, "
              f"mips={len(mips)} levels, smallest={list(mips[-1].shape)}")
