"""
dataset.py
==========
PyTorch Dataset for MAINet — 加载预生成的 npy 图像、mask、元数据。

目录结构（由 dataset_generator.py 生成）:
  output/{split}/images/  → {id:06d}.npy
  output/{split}/masks/   → {id:06d}_masks.npy, {id:06d}_info.npy
  output/annotations/     → {split}.json

用法:
  from data.dataset import MAINetDataset, collate_fn
  ds = MAINetDataset("output/annotations/train.json",
                     "output/train/images",
                     "output/train/masks",
                     augment=True)
  loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset


class MAINetDataset(Dataset):
    """
    MAINet 训练/测试数据集。

    每次迭代返回:
      img:       [1, 512, 512] float32 tensor（z-score 归一化）
      masks:     [M, 512, 512] float32 tensor（0/1 二值 mask）
      weights:   [M] float32 tensor（SNR 权重）
      gt_params: dict{str → scalar tensor}（物理参数真值，key 与 criterion 对齐）
      image_id:  int
    """

    PARAM_KEYS = ['phi', 'length']  # point 无参数，streak 有 φ/L

    def __init__(self, coco_json, image_dir, mask_dir, augment=False):
        with open(coco_json) as f:
            self.coco = json.load(f)

        self.image_dir = str(image_dir)
        self.mask_dir  = str(mask_dir)
        self.augment   = augment

        self.images = self.coco['images']

        # image_id → image_info 快速查找
        self._img_info = {img['id']: img for img in self.images}

        # image_id → [annotation, ...]
        self._ann_by_image = {}
        for ann in self.coco['annotations']:
            self._ann_by_image.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        return len(self.images)

    # ── 物理参数 ──────────────────────────────────────────────
    def _build_gt_params(self, img_info):
        """从 COCO image 字段提取 gt_params，key 与 param_loss 匹配。
        所有值均为 tensor，避免 train.py 的 device-move 报错。
        """
        params = {}
        for key in self.PARAM_KEYS:
            val = img_info.get(key)
            if val is not None:
                params[key] = torch.tensor(val, dtype=torch.float32)
        # mode → gate_target：point=0, streak=1（与 param_loss 的 BCE target 对齐）
        if 'mode' in img_info:
            params['gate_target'] = torch.tensor(
                1.0 if img_info['mode'] == 'streak' else 0.0, dtype=torch.float32)
        return params

    # ── 数据增强 ──────────────────────────────────────────────
    def _augment(self, img, masks, img_info):
        """
        随机水平/垂直翻转。

        条状图翻转后需同步更新 φ 角度：
          - 水平翻转: φ → π-φ (mod π)
          - 垂直翻转: φ →  -φ (mod π)
        点源图翻转后需同步更新 θ 角度：
          - 水平/垂直翻转: θ → -θ
        """
        mode = img_info.get('mode', 'point')

        # 水平翻转
        if torch.rand(1).item() > 0.5:
            img   = torch.flip(img,   [-1])
            masks = torch.flip(masks, [-1])
            if mode == 'streak' and img_info.get('phi') is not None:
                phi     = img_info['phi']
                new_phi = (np.pi - phi) % np.pi
                img_info = {**img_info, 'phi': float(new_phi)}
            if mode == 'point' and img_info.get('theta') is not None:
                img_info = {**img_info, 'theta': float(-img_info['theta'])}

        # 垂直翻转
        if torch.rand(1).item() > 0.5:
            img   = torch.flip(img,   [-2])
            masks = torch.flip(masks, [-2])
            if mode == 'streak' and img_info.get('phi') is not None:
                phi     = img_info['phi']
                new_phi = (-phi) % np.pi
                img_info = {**img_info, 'phi': float(new_phi)}
            if mode == 'point' and img_info.get('theta') is not None:
                img_info = {**img_info, 'theta': float(-img_info['theta'])}

        return img, masks, img_info

    # ── 主入口 ────────────────────────────────────────────────
    def __getitem__(self, idx):
        img_info = dict(self.images[idx])  # 浅拷贝，防止增强污染原数据
        image_id = img_info['id']
        fname    = img_info['file_name']

        # 图像
        img_path = f"{self.image_dir}/{fname}"
        img = np.load(img_path).astype(np.float32)
        # asinh 归一化（保留暗星细节 + 压缩亮星极端值）
        from data.normalize import normalize_astro
        img = normalize_astro(img)
        img = torch.from_numpy(img).unsqueeze(0)   # [1, H, W]

        # Mask（预解码 npy）
        mask_path = f"{self.mask_dir}/{image_id:06d}_masks.npy"
        masks_np  = np.load(mask_path)
        masks = torch.from_numpy(masks_np.astype(np.float32))  # [M, H, W]

        # 权重（SNR weight，info.npy 第 3 列）
        info_path = f"{self.mask_dir}/{image_id:06d}_info.npy"
        info_np   = np.load(info_path)
        if info_np.shape[0] > 0:
            weights = torch.from_numpy(info_np[:, 2].astype(np.float32))
        else:
            weights = torch.zeros(0, dtype=torch.float32)

        # 数据增强
        if self.augment:
            img, masks, img_info = self._augment(img, masks, img_info)

        # 物理参数真值
        gt_params = self._build_gt_params(img_info)

        return img, masks, weights, gt_params, image_id


# ══════════════════════════════════════════════════════════════
# collate
# ══════════════════════════════════════════════════════════════
def collate_fn(batch):
    """
    自定义 collate：
      - img 堆叠为 [B,1,512,512]
      - masks / weights / params / ids 保持 list-of-B
      （因为每张图的实例数 M 不同，无法堆叠）
    """
    imgs         = torch.stack([item[0] for item in batch])
    masks_list   = [item[1] for item in batch]
    weights_list = [item[2] for item in batch]
    params_list  = [item[3] for item in batch]
    image_ids    = [item[4] for item in batch]
    return imgs, masks_list, weights_list, params_list, image_ids


# ══════════════════════════════════════════════════════════════
# 自检
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    # 需要先生成数据: python dataset_generator.py
    train_json = "output/annotations/train.json"
    train_img  = "output/train/images"
    train_msk  = "output/train/masks"

    import os
    if not os.path.exists(train_json):
        print("请先运行 dataset_generator.py 生成数据")
        sys.exit(0)

    ds = MAINetDataset(train_json, train_img, train_msk, augment=True)
    print(f"Dataset size: {len(ds)}")

    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, num_workers=0)
    imgs, masks_list, weights_list, params_list, ids = next(iter(loader))

    print(f"\nBatch shapes:")
    print(f"  imgs:        {imgs.shape}")          # [4,1,512,512]
    print(f"  masks_list:  {len(masks_list)}, shapes: "
          f"{[m.shape for m in masks_list]}")
    print(f"  weights_list: {[w.shape for w in weights_list]}")
    print(f"  ids:         {ids}")

    # 检查 params 格式与 criterion 兼容
    print(f"\nSample gt_params (image {ids[0]}):")
    for k, v in params_list[0].items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")
        else:
            print(f"  {k}: {v}")

    print("\nOK — Dataset 自检通过")
