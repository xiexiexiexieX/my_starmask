"""MMDetection transform for loading single-channel .npy star images.

This module lives at project root because MMDetection configs import it through
``custom_imports = dict(imports=['npy_loading'])``. Do not import it as
``mmdet.npy_loading``: that name collides with the installed OpenMMLab package
when dataloader workers are spawned on Windows.
"""
import os
import json
from pathlib import Path

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

PREPROCESSED = {'real_asinh_0_1', 'real_zscale_0_1', 'fits_zscale_0_1', 'bmp_asinh_0_1'}
_ANN_CACHE = {}


def _annotation_path_for_image(img_path):
    parts = Path(img_path).parts
    if len(parts) < 3 or parts[-2] != 'images':
        return None
    split = parts[-3]
    data_root = Path(*parts[:-3])
    ann_path = data_root / 'annotations' / f'{split}.json'
    return ann_path if ann_path.exists() else None


def _preprocess_for_image(img_path):
    ann_path = _annotation_path_for_image(img_path)
    if ann_path is None:
        return None
    ann_key = str(ann_path)
    if ann_key not in _ANN_CACHE:
        with open(ann_path, encoding='utf-8-sig') as f:
            data = json.load(f)
        _ANN_CACHE[ann_key] = {
            item.get('file_name'): item.get('preprocess')
            for item in data.get('images', [])
        }
    return _ANN_CACHE[ann_key].get(Path(img_path).name)


def _normalize_astro(img):
    bg = np.median(img)
    img_sub = img - bg
    bg_mask = img < np.percentile(img, 90)
    scale = np.std(img_sub[bg_mask]) + 1e-6
    img_asinh = np.arcsinh(img_sub / scale)
    return ((img_asinh - img_asinh.mean()) / (img_asinh.std() + 1e-6)).astype(np.float32)


@TRANSFORMS.register_module()
class LoadStarNpy(BaseTransform):
    """Load a .npy astronomy image and expose it as a 3-channel HWC array."""

    def transform(self, results):
        if 'img' in results and isinstance(results['img'], str):
            results['img_path'] = results['img']
            data_root = results.get('data_root', '')
            img_prefix = results.get('img_prefix', '') or results.get('data_prefix', {}).get('img', '')
            if not os.path.isabs(results['img_path']) and (data_root or img_prefix):
                results['img_path'] = os.path.join(data_root, img_prefix, results['img'])

        img = np.load(results['img_path']).astype(np.float32)
        preprocess = results.get('preprocess') or _preprocess_for_image(results['img_path'])
        if preprocess in PREPROCESSED:
            img_norm = np.clip(np.where(np.isfinite(img), img, 0.0), 0.0, 1.0).astype(np.float32)
        else:
            img_norm = _normalize_astro(img)
        img_3c = np.stack([img_norm, img_norm, img_norm], axis=-1)

        results['img'] = img_3c
        results['img_shape'] = img_3c.shape[:2]
        results['ori_shape'] = img_3c.shape[:2]
        results['scale_factor'] = np.array([1.0, 1.0], dtype=np.float32)
        results['img_id'] = results.get('img_id', 0)
        return results
