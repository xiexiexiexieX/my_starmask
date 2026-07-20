"""
normalize.py
============
天文图像归一化：asinh 拉伸，统一用于训练和可视化。
"""

import numpy as np


def normalize_astro(img, scale=None):
    """
    天文图像 asinh 归一化。

    行为:
      - 弱信号（暗星/背景）→ 近似线性，保留细节
      - 强信号（亮星）     → 近似对数，压缩但不裁掉

    参数:
      img:   [H, W] float32 原始图像
      scale: asinh 软阈值，默认用背景噪声 std（<=90分位像素）

    返回:
      [H, W] float32, 归一化后（均值为0，std接近1，亮星峰值在个位数σ）
    """
    # 1. 减背景中位数（鲁棒）
    bg = np.median(img)
    img_sub = img - bg

    # 2. asinh 拉伸
    if scale is None:
        # 用背景区域噪声作为尺度（取<=90分位，排除亮星）
        bg_mask = img < np.percentile(img, 90)
        scale = np.std(img_sub[bg_mask]) + 1e-6

    img_asinh = np.arcsinh(img_sub / scale)

    # 3. 归一化到零均值单位方差
    img_norm = (img_asinh - img_asinh.mean()) / (img_asinh.std() + 1e-6)

    return img_norm.astype(np.float32)
