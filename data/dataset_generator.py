"""
dataset_generator.py
====================
批量生成模拟地基望远镜星场数据集，输出COCO格式标注。

目录结构：
  output/
    train/images/  000001.npy ...
    test/images/   002401.npy ...
    annotations/
      train.json
      test.json
    dataset_info.json

运行：
  python data/dataset_generator.py
"""

import os
import json
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import pycocotools.mask as coco_mask_utils


# ══════════════════════════════════════════════════════════════
# 全局配置
# ══════════════════════════════════════════════════════════════
class Config:
    N_TOTAL      = 100
    TRAIN_RATIO  = 0.8
    VAL_RATIO    = 0.1       # test = 1 - train - val
    H, W         = 512, 512

    N_SINGLE     = (10, 51)
    N_BINARY     = (0,  9)
    N_FAINT      = (5,  16)

    PSF_SX       = (1.2, 4.0)
    PSF_SY       = (1.2, 4.0)
    PSF_THETA    = (-0.8, 0.8)

    TRAIL_LENGTH = (8, 25)

    SNR_FULL     = 3.0    # snr >= 3.0      → weight=1.0
    SNR_HALF     = 1.5    # snr in [1.5,3)  → weight=0.5
                          # snr < 1.5       → 不标注

    MASK_SNR_K   = 5.0    # mask边界: 星贡献 > K * bg_noise_std
    MIN_RADIUS   = 0.0    # 无保底

    # 粘连控制
    OVERLAP_RATIO = (0.05, 0.30)  # 双星mask重叠比例区间(轻微粘连)
    SAFE_MARGIN   = 2              # 非粘连星之间的安全边距(px)
    MAX_RETRY     = 30             # 位置重采样最大次数

    CATEGORIES   = [
        {"id": 1, "name": "single"},
        {"id": 2, "name": "binary"},
        {"id": 3, "name": "faint"},
    ]
    CAT_MAP      = {"single": 1, "binary": 2, "faint": 3}

    OUTPUT_DIR   = "output"
    SEED_OFFSET  = 0


# ══════════════════════════════════════════════════════════════
# 背景（全图操作，无法局部化，但只做一次）
# ══════════════════════════════════════════════════════════════
def make_background(rng, H, W):
    y_grid, x_grid = np.mgrid[0:H, 0:W]
    sky_bg   = 300 + 80*np.sin(x_grid/W*np.pi) + 60*np.cos(y_grid/H*np.pi*0.7)
    img      = rng.poisson(sky_bg).astype(float)
    # 纯泊松噪声真值（排除低频渐变），作为后续SNR/mask的统一背景参考
    bg_noise_std = float(np.sqrt(np.mean(sky_bg)))
    stray    = np.zeros((H, W))
    for _ in range(4):
        cx, cy = int(rng.integers(0,W)), int(rng.integers(0,H))
        amp    = rng.uniform(40, 120)
        radius = rng.uniform(80, 200)
        stray += amp * np.exp(-((x_grid-cx)**2+(y_grid-cy)**2)/(2*radius**2))
    atm_turb = gaussian_filter(rng.standard_normal((H,W))*15, sigma=30)
    flat     = 1.0 + 0.03*gaussian_filter(rng.standard_normal((H,W)), sigma=60)
    img      = (img + stray + atm_turb) * flat
    return img, bg_noise_std


# ══════════════════════════════════════════════════════════════
# 局部bbox工具
# ══════════════════════════════════════════════════════════════
def local_bbox(x, y, pad, H, W):
    """返回以(x,y)为中心、pad为半径的局部区域边界（clamp到图像范围）"""
    x1 = max(0, int(x) - pad)
    x2 = min(W, int(x) + pad + 1)
    y1 = max(0, int(y) - pad)
    y2 = min(H, int(y) + pad + 1)
    return x1, x2, y1, y2


# ══════════════════════════════════════════════════════════════
# 星点添加（局部计算）
# ══════════════════════════════════════════════════════════════
def add_point_local(img, x, y, flux, psf_sx, psf_sy, cos_t, sin_t, H, W):
    """只在星点周围局部区域计算高斯，写回img对应区域"""
    pad  = int(psf_sx * 4 + psf_sy * 4) + 5
    x1, x2, y1, y2 = local_bbox(x, y, pad, H, W)
    ys, xs = np.mgrid[y1:y2, x1:x2]
    xr =  (xs - x)*cos_t + (ys - y)*sin_t
    yr = -(xs - x)*sin_t + (ys - y)*cos_t
    img[y1:y2, x1:x2] += flux * np.exp(-0.5*((xr/psf_sx)**2+(yr/psf_sy)**2))


def add_streak_local(img, x, y, flux, psf_sx, psf_sy,
                     trail_length, cos_a, sin_a, H, W, steps=80):
    """
    只在streak覆盖的局部bbox内积分，不对全图操作。
    局部bbox = streak中心 ± (trail_length/2 + psf_sy*4 + margin)
    """
    pad  = int(trail_length / 2 + psf_sx * 4 + psf_sy * 4) + 5
    x1, x2, y1, y2 = local_bbox(x, y, pad, H, W)
    ys, xs = np.mgrid[y1:y2, x1:x2]   # 局部坐标网格，只算一次
    local  = np.zeros((y2-y1, x2-x1), dtype=float)

    for t in np.linspace(-trail_length/2, trail_length/2, steps):
        cx = x + t*cos_a
        cy = y + t*sin_a
        xr =  (xs - cx)*cos_a + (ys - cy)*sin_a
        yr = -(xs - cx)*sin_a + (ys - cy)*cos_a
        local += np.exp(-0.5*((xr/psf_sx)**2+(yr/psf_sy)**2))

    img[y1:y2, x1:x2] += (flux / steps) * local


# ══════════════════════════════════════════════════════════════
# 统一二值 mask 生成（亮度自适应 + 最小半径保底）
# ══════════════════════════════════════════════════════════════
def make_mask_point(H, W, x, y, flux, psf_sx, psf_sy, theta,
                    bg_noise_std, k, min_radius=2.0):
    """
    点源二值mask: 贡献 > K*bg_noise_std 或 距中心≤min_radius。
    三处统一调用(COCO/binary判定/训练npy)。
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    threshold = k * bg_noise_std
    ratio = max(flux / threshold, 1.01)
    r_max = int(np.sqrt(2 * np.log(ratio)) * max(psf_sx, psf_sy)) + 3
    r_max = max(r_max, int(min_radius) + 2)   # 保底窗口
    x1, x2, y1, y2 = local_bbox(x, y, r_max, H, W)
    ys, xs = np.mgrid[y1:y2, x1:x2]
    xr =  (xs-x)*cos_t + (ys-y)*sin_t
    yr = -(xs-x)*sin_t + (ys-y)*cos_t
    contrib = flux * np.exp(-0.5*((xr/psf_sx)**2 + (yr/psf_sy)**2))
    dist2 = (xr/psf_sx)**2 + (yr/psf_sy)**2   # 马氏距离平方
    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2, x1:x2] = (contrib > threshold) | (dist2 <= 1.0)  # 1σ马氏距离≈min_radius效果
    return mask


def make_mask_streak(H, W, x, y, flux, psf_sx, psf_sy,
                     trail_length, trail_angle, bg_noise_std, k,
                     min_radius=2.0, steps=80):
    """
    条状二值mask: 贡献 > K*bg_noise_std 或 到中轴垂直距≤min_radius。
    贡献计算与 add_streak_local 完全一致。
    """
    cos_a, sin_a = np.cos(trail_angle), np.sin(trail_angle)
    threshold = k * bg_noise_std
    ratio = max(flux / threshold, 1.01)
    r_max = int(np.sqrt(2 * np.log(ratio)) * max(psf_sx, psf_sy)) + 3
    r_max = max(r_max, int(min_radius) + 2)
    pad = int(trail_length / 2 + r_max) + 5
    x1, x2, y1, y2 = local_bbox(x, y, pad, H, W)
    ys, xs = np.mgrid[y1:y2, x1:x2]
    # 贡献积分(与add_streak_local一致)
    contrib = np.zeros((y2-y1, x2-x1), dtype=np.float64)
    for t in np.linspace(-trail_length/2, trail_length/2, steps):
        cx_t = x + t*cos_a; cy_t = y + t*sin_a
        xr =  (xs-cx_t)*cos_a + (ys-cy_t)*sin_a
        yr = -(xs-cx_t)*sin_a + (ys-cy_t)*cos_a
        contrib += np.exp(-0.5*((xr/psf_sx)**2 + (yr/psf_sy)**2))
    contrib *= (flux / steps)
    # 到拖尾中轴线的垂直距离: 点(xs,ys)到线段的最短距离
    dx = xs - x; dy = ys - y
    along = dx*cos_a + dy*sin_a               # 沿拖尾投影
    along_clamped = np.clip(along, -trail_length/2, trail_length/2)
    cx_proj = x + along_clamped*cos_a
    cy_proj = y + along_clamped*sin_a
    perp2 = (xs-cx_proj)**2 + (ys-cy_proj)**2  # 垂直距离平方
    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2, x1:x2] = (contrib > threshold) | (perp2 <= min_radius**2)
    return mask


# ══════════════════════════════════════════════════════════════
# SNR
# ══════════════════════════════════════════════════════════════
def compute_snr(flux, bg_noise_std):
    """基于真值计算SNR。add_point/streak_local 的峰值 = flux。"""
    return flux / (bg_noise_std + 1e-6)


# ══════════════════════════════════════════════════════════════
# mask 工具
# ══════════════════════════════════════════════════════════════
def mask_to_rle(mask):
    m   = np.asfortranarray(mask.astype(np.uint8))
    rle = coco_mask_utils.encode(m)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def mask_to_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
    return [cmin, rmin, cmax-cmin+1, rmax-rmin+1]


# ══════════════════════════════════════════════════════════════
# 双星检测: 用亮度mask重叠判断
# ══════════════════════════════════════════════════════════════
def _make_mask(instances, meta, cfg):
    """为instances生成二值mask列表(faint返回None)"""
    H, W = cfg.H, cfg.W
    bg = meta['bg_noise_std']
    masks = []
    for x, y, flux, cat in instances:
        if cat == 'faint':
            masks.append(None)
            continue
        if meta['mode'] == 'point':
            m = make_mask_point(H, W, x, y, flux,
                                0.8, 0.8, 0.0, bg, cfg.MASK_SNR_K, cfg.MIN_RADIUS)
        else:
            m = make_mask_streak(H, W, x, y, flux,
                                 meta.get('sigma_x', 1.0), meta.get('sigma_y', 1.0),
                                 meta['length'], meta['phi'],
                                 bg, cfg.MASK_SNR_K, cfg.MIN_RADIUS)
        masks.append(m)
    return masks


def detect_binaries_by_mask(instances, meta, cfg):
    """
    用亮度mask检测重叠对 → 标为binary。
    三星及以上: 只保留重叠最大的一对, 其余退回single。
    返回更新后的 instances 列表(不修改图像, 纯粹做标注)。
    """
    masks = _make_mask(instances, meta, cfg)

    overlaps = []
    for i in range(len(instances)):
        if masks[i] is None or masks[i].sum() == 0:
            continue
        for j in range(i+1, len(instances)):
            if masks[j] is None or masks[j].sum() == 0:
                continue
            inter = (masks[i] & masks[j]).sum()
            if inter > 0:
                overlaps.append((inter, i, j))

    if not overlaps:
        return [(x, y, f, c) for x, y, f, c in instances]

    overlaps.sort(reverse=True)
    binary_ids = set()
    used = set()

    for area, i, j in overlaps:
        if i in used or j in used:
            continue
        binary_ids.add(i)
        binary_ids.add(j)
        used.add(i)
        used.add(j)

    new_instances = []
    for i, (x, y, flux, cat) in enumerate(instances):
        if i in binary_ids and cat != 'faint':
            new_instances.append((x, y, flux, 'binary'))
        else:
            new_instances.append((x, y, flux, cat))
    return new_instances


def _overlap_ratio(m1, m2):
    """mask重叠比例 = 交集 / min(面积1, 面积2)"""
    a1, a2 = m1.sum(), m2.sum()
    if a1 == 0 or a2 == 0: return 0.0
    return float((m1 & m2).sum()) / min(a1, a2)

def _dilate_mask(mask, margin):
    """膨胀mask (安全边距)"""
    if margin <= 0: return mask
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, iterations=int(margin))


# ══════════════════════════════════════════════════════════════
# 点源图生成
# ══════════════════════════════════════════════════════════════
def gen_point(rng, cfg):
    H, W   = cfg.H, cfg.W
    # 各向异性高斯 PSF（σ_x≠σ_y, θ 随机）→ 模拟真实光学畸变
    sigma_x = rng.uniform(*cfg.PSF_SX)
    sigma_y = rng.uniform(*cfg.PSF_SY)
    theta   = rng.uniform(*cfg.PSF_THETA)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    img, bg_noise_std = make_background(rng, H, W)
    meta = {
        "mode": "point",
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "theta":   float(theta),
        "bg_noise_std": bg_noise_std,
    }

    occupied = np.zeros((H, W), dtype=bool)
    instances = []
    R, lo, hi, sm = cfg.MAX_RETRY, *cfg.OVERLAP_RATIO, cfg.SAFE_MARGIN

    def _mk(x, y, f):
        return make_mask_point(H, W, x, y, f, sigma_x, sigma_y, theta,
                               bg_noise_std, cfg.MASK_SNR_K, cfg.MIN_RADIUS)
    def _add(x, y, f):
        add_point_local(img, x, y, f, sigma_x, sigma_y, cos_t, sin_t, H, W)
    def _isolated(m):
        return not (_dilate_mask(occupied, sm) & m).any()

    # single
    for _ in range(int(rng.integers(*cfg.N_SINGLE))):
        for _ in range(R):
            x, y = rng.uniform(30, W-30), rng.uniform(30, H-30)
            f = rng.uniform(500, 5000); m = _mk(x, y, f)
            if _isolated(m):
                _add(x, y, f); occupied |= m
                instances.append((x, y, f, 'single')); break

    # binary: 配对内允许粘连, 对之间孤立
    n_tgt = int(rng.integers(*cfg.N_BINARY))
    n_done = 0
    for _ in range(n_tgt * 5):
        if n_done >= n_tgt: break
        for _ in range(R):
            x1, y1 = rng.uniform(50, W-50), rng.uniform(50, H-50)
            f1 = rng.uniform(800, 3000); m1 = _mk(x1, y1, f1)
            if not _isolated(m1): continue
            found = False
            for _ in range(R):
                ang = rng.uniform(0, 2*np.pi)
                f2 = f1 * rng.uniform(0.3, 1.0)
                for s in np.linspace(0.3, 6.0, 30):
                    d = s * max(sigma_x, sigma_y)
                    x2 = x1+d*np.cos(ang); y2 = y1+d*np.sin(ang)
                    if not (10<x2<W-10 and 10<y2<H-10): continue
                    m2 = _mk(x2, y2, f2)
                    occ_no_m1 = _dilate_mask(occupied & ~m1, sm)
                    if (occ_no_m1 & m2).any(): continue
                    if lo <= _overlap_ratio(m1, m2) <= hi:
                        _add(x1, y1, f1); _add(x2, y2, f2)
                        occupied |= m1 | m2
                        instances.append((x1, y1, f1, 'single'))
                        instances.append((x2, y2, f2, 'single'))
                        n_done += 1; found = True; break
                if found: break
            if found: break

    # faint
    for _ in range(int(rng.integers(*cfg.N_FAINT))):
        for _ in range(R):
            x, y = rng.uniform(30, W-30), rng.uniform(30, H-30)
            f = rng.uniform(50, 200); m = _mk(x, y, f)
            if _isolated(m):
                _add(x, y, f); occupied |= m
                instances.append((x, y, f, 'faint')); break
    instances = detect_binaries_by_mask(instances, meta, cfg)
    return img, instances, meta


# ══════════════════════════════════════════════════════════════
# 条状图生成
# ══════════════════════════════════════════════════════════════
def gen_streak(rng, cfg):
    H, W         = cfg.H, cfg.W
    streak_w     = rng.uniform(*cfg.PSF_SX)           # 拖尾粗细（各向同性高斯圆）
    trail_angle  = rng.uniform(0, np.pi)
    trail_length = rng.uniform(*cfg.TRAIL_LENGTH)
    cos_a, sin_a = np.cos(trail_angle), np.sin(trail_angle)

    img, bg_noise_std = make_background(rng, H, W)
    meta = {
        "mode": "streak",
        "sigma_x": float(streak_w), "sigma_y": float(streak_w),
        "theta":   None,
        "phi":     float(trail_angle),
        "length":  float(trail_length),
        "bg_noise_std": bg_noise_std,
    }

    occupied = np.zeros((H, W), dtype=bool)
    instances = []
    R, lo, hi, sm = cfg.MAX_RETRY, *cfg.OVERLAP_RATIO, cfg.SAFE_MARGIN

    def _mk(x, y, f):
        return make_mask_streak(H, W, x, y, f, streak_w, streak_w,
                                trail_length, trail_angle,
                                bg_noise_std, cfg.MASK_SNR_K, cfg.MIN_RADIUS)
    def _add(x, y, f):
        add_streak_local(img, x, y, f, streak_w, streak_w,
                         trail_length, cos_a, sin_a, H, W)
    def _isolated(m):
        return not (_dilate_mask(occupied, sm) & m).any()

    # single
    for _ in range(int(rng.integers(*cfg.N_SINGLE))):
        for _ in range(R):
            x, y = rng.uniform(30, W-30), rng.uniform(30, H-30)
            f = rng.uniform(500, 5000); m = _mk(x, y, f)
            if _isolated(m):
                _add(x, y, f); occupied |= m
                instances.append((x, y, f, 'single')); break

    # binary: 配对内允许粘连, 对之间孤立
    n_tgt = int(rng.integers(*cfg.N_BINARY))
    n_done = 0
    for _ in range(n_tgt * 5):
        if n_done >= n_tgt: break
        for _ in range(R):
            x1, y1 = rng.uniform(60, W-60), rng.uniform(60, H-60)
            f1 = rng.uniform(800, 3000); m1 = _mk(x1, y1, f1)
            if not _isolated(m1): continue
            found = False
            for _ in range(R):
                ang = rng.uniform(0, 2*np.pi)
                f2 = f1 * rng.uniform(0.3, 1.0)
                for s in np.linspace(0.3, 6.0, 30):
                    d = s * streak_w
                    x2 = x1+d*np.cos(ang); y2 = y1+d*np.sin(ang)
                    if not (10<x2<W-10 and 10<y2<H-10): continue
                    m2 = _mk(x2, y2, f2)
                    occ_no_m1 = _dilate_mask(occupied & ~m1, sm)
                    if (occ_no_m1 & m2).any(): continue
                    if lo <= _overlap_ratio(m1, m2) <= hi:
                        _add(x1, y1, f1); _add(x2, y2, f2)
                        occupied |= m1 | m2
                        instances.append((x1, y1, f1, 'single'))
                        instances.append((x2, y2, f2, 'single'))
                        n_done += 1; found = True; break
                if found: break
            if found: break

    # faint
    for _ in range(int(rng.integers(*cfg.N_FAINT))):
        for _ in range(R):
            x, y = rng.uniform(30, W-30), rng.uniform(30, H-30)
            f = rng.uniform(50, 200); m = _mk(x, y, f)
            if _isolated(m):
                _add(x, y, f); occupied |= m
                instances.append((x, y, f, 'faint')); break
    instances = detect_binaries_by_mask(instances, meta, cfg)
    return img, instances, meta


# ══════════════════════════════════════════════════════════════
# 构建COCO annotations
# ══════════════════════════════════════════════════════════════
def build_annotations(img, instances, meta, image_id, ann_id_start, cfg):
    H, W        = cfg.H, cfg.W
    annotations = []
    ann_id      = ann_id_start

    for x, y, flux, category in instances:
        snr = compute_snr(flux, meta['bg_noise_std'])
        weight = 1.0 if snr >= cfg.SNR_FULL else 0.5

        if meta['mode'] == 'point':
            sx = meta.get('sigma_x', meta.get('sigma', 0.8))
            sy = meta.get('sigma_y', meta.get('sigma', 0.8))
            th = meta.get('theta', 0.0)
            mask = make_mask_point(
                H, W, x, y, flux, sx, sy, th,
                meta['bg_noise_std'], cfg.MASK_SNR_K, cfg.MIN_RADIUS)
        else:
            mask = make_mask_streak(
                H, W, x, y, flux,
                meta.get('sigma_x', 1.0), meta.get('sigma_y', 1.0),
                meta['length'], meta['phi'],
                meta['bg_noise_std'], cfg.MASK_SNR_K, cfg.MIN_RADIUS)

        if mask.sum() == 0:
            continue

        annotations.append({
            "id":           ann_id,
            "image_id":     image_id,
            "category_id":  cfg.CAT_MAP[category],
            "segmentation": mask_to_rle(mask),
            "area":         int(mask.sum()),
            "bbox":         mask_to_bbox(mask),
            "iscrowd":      0,
            "centroid":     [float(x), float(y)],
            "flux":         float(flux),
            "snr":          float(snr),
            "weight":       float(weight),
        })
        ann_id += 1

    return annotations, ann_id


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════
def generate_dataset(cfg: Config):
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(cfg.OUTPUT_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, 'annotations'), exist_ok=True)

    n_train = int(cfg.N_TOTAL * cfg.TRAIN_RATIO)
    n_val   = int(cfg.N_TOTAL * cfg.VAL_RATIO)
    n_test  = cfg.N_TOTAL - n_train - n_val

    rng_global = np.random.default_rng(cfg.SEED_OFFSET)
    modes      = (['point']  * (cfg.N_TOTAL // 2) +
                  ['streak'] * (cfg.N_TOTAL - cfg.N_TOTAL // 2))
    rng_global.shuffle(modes)
    splits = (['train'] * n_train + ['val'] * n_val + ['test'] * n_test)

    coco = {s: {
        "info":        {"description": "Simulated Star Field", "version": "1.0"},
        "categories":  cfg.CATEGORIES,
        "images":      [],
        "annotations": [],
    } for s in ['train', 'val', 'test']}

    ann_id = 1

    for idx in tqdm(range(cfg.N_TOTAL), desc="Generating"):
        image_id = idx + 1
        split    = splits[idx]
        mode     = modes[idx]
        fname    = f"{image_id:06d}.npy"
        img_path = os.path.join(cfg.OUTPUT_DIR, split, 'images', fname)

        # 断点续传：跳过已存在的图片，但需从已有标注重建 coco
        if os.path.exists(img_path):
            # 重建标注（用确定性种子保证一致）
            rng = np.random.default_rng(cfg.SEED_OFFSET + idx)
            if mode == 'point':
                img, instances, meta = gen_point(rng, cfg)
            else:
                img, instances, meta = gen_streak(rng, cfg)
            anns, ann_id = build_annotations(
                img, instances, meta, image_id, ann_id, cfg)
            coco[split]['images'].append({
                "id": image_id, "file_name": fname,
                "width": cfg.W, "height": cfg.H,
                **meta,
            })
            coco[split]['annotations'].extend(anns)
            continue

        rng = np.random.default_rng(cfg.SEED_OFFSET + idx)

        if mode == 'point':
            img, instances, meta = gen_point(rng, cfg)
        else:
            img, instances, meta = gen_streak(rng, cfg)

        np.save(img_path, img.astype(np.float32))

        anns, ann_id = build_annotations(
            img, instances, meta, image_id, ann_id, cfg)

        coco[split]['images'].append({
            "id": image_id, "file_name": fname,
            "width": cfg.W, "height": cfg.H,
            **meta,
        })
        coco[split]['annotations'].extend(anns)

    # 保存json
    for split in ['train', 'val', 'test']:
        path = os.path.join(cfg.OUTPUT_DIR, 'annotations', f'{split}.json')
        with open(path, 'w') as f:
            json.dump(coco[split], f)
        print(f"{split}: {len(coco[split]['images'])} images, "
              f"{len(coco[split]['annotations'])} annotations → {path}")

    stats = {
        "n_total":  cfg.N_TOTAL,
        "n_train":  n_train,
        "n_val":    n_val,
        "n_test":   n_test,
        "n_point":  modes.count('point'),
        "n_streak": modes.count('streak'),
    }
    with open(os.path.join(cfg.OUTPUT_DIR, 'dataset_info.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nDone → {stats}")


# ══════════════════════════════════════════════════════════════
# 预解码 mask 存 npy（加速训练时的数据加载）
# ══════════════════════════════════════════════════════════════
def save_masks_as_npy(cfg: Config):
    """
    每张图生成两个文件：
      masks/000001_masks.npy  → [N, H, W] uint8 二值mask
      masks/000001_info.npy   → [N, 3] float: [ann_id, label(0-idx), weight]

    二值mask: 与COCO JSON完全一致(亮度自适应+MIN_RADIUS保底)。
    """
    for split in ['train', 'val', 'test']:
        mask_dir = os.path.join(cfg.OUTPUT_DIR, split, 'masks')
        os.makedirs(mask_dir, exist_ok=True)
        ann_path = os.path.join(cfg.OUTPUT_DIR, 'annotations', f'{split}.json')
        with open(ann_path) as f:
            coco = json.load(f)

        ann_by_image = {}
        for ann in coco['annotations']:
            ann_by_image.setdefault(ann['image_id'], []).append(ann)

        print(f"Saving {split} soft masks ({len(coco['images'])} images)...")
        for img_info in tqdm(coco['images'], desc=f"{split}"):
            image_id  = img_info['id']
            mask_path = os.path.join(mask_dir, f"{image_id:06d}_masks.npy")
            info_path = os.path.join(mask_dir, f"{image_id:06d}_info.npy")
            if os.path.exists(mask_path):
                continue
            anns = ann_by_image.get(image_id, [])
            if not anns:
                np.save(mask_path, np.zeros(
                    (0, img_info['height'], img_info['width']), dtype=np.float32))
                np.save(info_path, np.zeros((0, 3), dtype=np.float32))
                continue

            H, W = img_info['height'], img_info['width']

            bg = img_info.get('bg_noise_std', 19.0)
            k = cfg.MASK_SNR_K
            mr = cfg.MIN_RADIUS

            masks_list, info_list = [], []
            for ann in anns:
                cx, cy = ann['centroid']
                flux = ann.get('flux', 0)

                if img_info['mode'] == 'point':
                    sx = img_info.get('sigma_x', img_info.get('sigma', 0.8))
                    sy = img_info.get('sigma_y', img_info.get('sigma', 0.8))
                    th = img_info.get('theta', 0.0)
                    m = make_mask_point(
                        H, W, cx, cy, flux,
                        sx, sy, th,
                        bg, k, mr)
                else:
                    m = make_mask_streak(
                        H, W, cx, cy, flux,
                        img_info.get('sigma_x', 1.0), img_info.get('sigma_y', 1.0),
                        img_info.get('length', 20),
                        img_info.get('phi', 0),
                        bg, k, mr)

                masks_list.append(m.astype(np.uint8))
                info_list.append([ann['id'],
                                   ann['category_id'] - 1,
                                   ann['weight']])
            if masks_list:
                stacked = np.stack(masks_list)
                # 后处理: 如果有像素>2覆盖, 删掉重叠区最小的那颗星
                ov = stacked.sum(axis=0)
                if ov.max() > 2:
                    ys, xs = np.where(ov > 2)
                    for yi, xi in zip(ys, xs):
                        contributors = [j for j in range(len(masks_list))
                                        if masks_list[j][yi, xi]]
                        if len(contributors) > 2:
                            # 删flux最小的
                            fluxes = [(anns[j].get('flux', 0), j) for j in contributors]
                            fluxes.sort()
                            for _, j in fluxes[2:]:  # 保留flux最大的2颗
                                masks_list[j][yi, xi] = 0
                    stacked = np.stack(masks_list)
                np.save(mask_path, stacked)
                np.save(info_path, np.array(info_list, dtype=np.float32))
            else:
                np.save(mask_path, np.zeros(
                    (0, H, W), dtype=np.uint8))
                np.save(info_path, np.zeros((0, 3), dtype=np.float32))
    print("Binary mask npy files saved.")


if __name__ == "__main__":
    cfg = Config()
    generate_dataset(cfg)
    save_masks_as_npy(cfg)