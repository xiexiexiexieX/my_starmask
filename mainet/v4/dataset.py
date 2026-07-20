"""
Dataset + asinh 归一化（自包含，零外部依赖除 torch/numpy）
==========================================================
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_astro(img, scale=None):
    """天文图像 asinh 归一化"""
    bg = np.median(img)
    img_sub = img - bg
    if scale is None:
        bg_mask = img < np.percentile(img, 90)
        scale = np.std(img_sub[bg_mask]) + 1e-6
    img_asinh = np.arcsinh(img_sub / scale)
    return ((img_asinh - img_asinh.mean()) / (img_asinh.std() + 1e-6)).astype(np.float32)


class MAINetDataset(Dataset):
    PARAM_KEYS = ['phi', 'length']

    def __init__(self, coco_json, image_dir, mask_dir, augment=False):
        with open(coco_json, encoding='utf-8') as f:
            self.coco = json.load(f)
        self.image_dir = str(image_dir); self.mask_dir = str(mask_dir)
        self.augment = augment
        self.images = self.coco['images']
        self._img_info = {img['id']: img for img in self.images}
        self._ann_by_image = {}
        for ann in self.coco['annotations']:
            self._ann_by_image.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        return len(self.images)

    def _build_gt_params(self, img_info):
        params = {}
        for key in self.PARAM_KEYS:
            val = img_info.get(key)
            if val is not None: params[key] = torch.tensor(val, dtype=torch.float32)
        if 'mode' in img_info:
            params['gate_target'] = torch.tensor(1.0 if img_info['mode'] == 'streak' else 0.0, dtype=torch.float32)
        return params

    def _augment(self, img, masks, img_info):
        mode = img_info.get('mode', 'point')
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, [-1]); masks = torch.flip(masks, [-1])
            if mode == 'streak' and img_info.get('phi') is not None:
                img_info = {**img_info, 'phi': float((np.pi - img_info['phi']) % np.pi)}
            if mode == 'point' and img_info.get('theta') is not None:
                img_info = {**img_info, 'theta': float(-img_info['theta'])}
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, [-2]); masks = torch.flip(masks, [-2])
            if mode == 'streak' and img_info.get('phi') is not None:
                img_info = {**img_info, 'phi': float((-img_info['phi']) % np.pi)}
            if mode == 'point' and img_info.get('theta') is not None:
                img_info = {**img_info, 'theta': float(-img_info['theta'])}
        return img, masks, img_info

    def __getitem__(self, idx):
        img_info = dict(self.images[idx]); image_id = img_info['id']
        fname = img_info['file_name']
        img = np.load(f"{self.image_dir}/{fname}").astype(np.float32)
        if img_info.get('preprocess') in ('real_asinh_0_1', 'real_zscale_0_1', 'fits_zscale_0_1', 'bmp_asinh_0_1'):
            img = np.clip(img, 0.0, 1.0).astype(np.float32)
        else:
            img = normalize_astro(img)
        img = torch.from_numpy(img).unsqueeze(0)
        masks = torch.from_numpy(np.load(f"{self.mask_dir}/{image_id:06d}_masks.npy").astype(np.float32))
        info_np = np.load(f"{self.mask_dir}/{image_id:06d}_info.npy")
        weights = torch.from_numpy(info_np[:, 2].astype(np.float32)) if info_np.shape[0] > 0 else torch.zeros(0)
        if self.augment: img, masks, img_info = self._augment(img, masks, img_info)
        gt_params = self._build_gt_params(img_info)
        return img, masks, weights, gt_params, image_id


def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    masks_list = [item[1] for item in batch]
    weights_list = [item[2] for item in batch]
    params_list = [item[3] for item in batch]
    image_ids = [item[4] for item in batch]
    return imgs, masks_list, weights_list, params_list, image_ids

