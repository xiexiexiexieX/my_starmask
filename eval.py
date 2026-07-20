"""
Instance segmentation evaluation for MAINet and MMDetection checkpoints.

Real mixed checkpoints:
  python eval.py --real-only --data output_mix --split test
  python eval.py --all --data output_mix --split test
  python eval.py --model work_dirs\real_mixed_baselines\mainet_v4 --data output_mix --split test
  python eval.py --model work_dirs\real_mixed_baselines\mmdet\mask_rcnn --data output_mix --split test
  python eval.py --model work_dirs\real_mixed_baselines\mmdet\condinst --data output_mix --split test
  python eval.py --model work_dirs\real_mixed_baselines\mmdet\mask2former --data output_mix --split test
  python eval.py --model work_dirs\real_mixed_baselines\yolo --data output_mix --split test --force

Moved dataset example:
  python eval.py --all --data E:\codes\query_mask\output_mix --split test

Notes:
  - Use full paths for real mixed checkpoints; --model v4 points to work_dirs/mainet/v4.
  - Use the mmdet environment for MMDetection checkpoints.
  - AP/IoU and F1 use raw masks; cMSE/B-MSE use the largest 8-connected prediction component.
  - F1 uses score >= --score_thr and one-to-one mask matching at IoU >= 0.50.
  - SNR is recomputed from the model-input image and GT masks using local peak contrast / MAD noise.
  - SNR groups are [0,3), [3,4), [4,5), and [5,+inf).
  - Results are saved to work_dirs/comparison_results.json.
"""
import os, sys, json, argparse, hashlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as connected_component_label
from scipy.optimize import linear_sum_assignment

CLI_USAGE = """
MAINet evaluation commands

Real mixed checkpoints:
  python eval.py --real-only --data output_mix --split test
  python eval.py --all --data output_mix --split test
  python eval.py --model work_dirs/real_mixed_baselines/mainet_v4 --data output_mix --split test
  python eval.py --model work_dirs/real_mixed_baselines/mmdet/mask_rcnn --data output_mix --split test
  python eval.py --model work_dirs/real_mixed_baselines/mmdet/condinst --data output_mix --split test
  python eval.py --model work_dirs/real_mixed_baselines/mmdet/mask2former --data output_mix --split test
  python eval.py --model work_dirs/real_mixed_baselines/yolo --data output_mix --split test --force

Legacy/main checkpoints:
  python eval.py --model v4
  python eval.py --model v3
  python eval.py --model work_dirs/mainet/v4
  python eval.py --split val --eval-bs 8
  python eval.py --model v4 --stages

V4 ablation experiment (7 trained variants + external full V4 reference):
  python mainet/v4/ablation_experiment/run.py --eval-only --data output_mix

Full discovery:
  python eval.py --all
"""
# mmdet       
_mmdet_available = False
try:
    from mmdet.apis import init_detector, inference_detector
    _mmdet_available = True
except ImportError:
    pass

_yolo_available = False
try:
    from ultralytics import YOLO
    from yolo.prepare_dataset import (
        npy_to_yolo_rgb, require_yolo_segmentation, yolo_mask_nms_indices,
        yolo_model_display_name,
    )
    _yolo_available = True
except ImportError:
    pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

AP_IOU_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
F1_IOU_THRESHOLD = 0.50
SNR_BINS = (
    ('0-3', 0.0, 3.0),
    ('3-4', 3.0, 4.0),
    ('4-5', 4.0, 5.0),
    ('>=5', 5.0, float('inf')),
)
SNR_LOCAL_PADDING = 12
SNR_MIN_BACKGROUND_PIXELS = 30
SNR_DEFINITION = 'input_local_peak_over_mad_v1'
PREPROCESSED = {'real_asinh_0_1', 'real_zscale_0_1', 'fits_zscale_0_1', 'bmp_asinh_0_1'}
EVAL_SCHEMA_VERSION = 14


CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


def label_mask_components(mask):
    """Return an 8-connected label image and the component count."""
    mask = np.asarray(mask, dtype=bool)
    return connected_component_label(mask, structure=CONNECTIVITY_8)


def component_geometry(mask):
    """Return area, bbox, and centroid for every 8-connected component."""
    labels, count = label_mask_components(mask)
    components = []
    for component_id in range(1, count + 1):
        ys, xs = np.where(labels == component_id)
        components.append({
            'id': component_id,
            'area': int(len(xs)),
            'bbox': (int(xs.min()), int(ys.min()),
                     int(xs.max()), int(ys.max())),
            'center': (float(xs.mean()), float(ys.mean())),
        })
    return components


def largest_connected_component(mask):
    """Keep only the largest 8-connected component of a binary mask."""
    mask = np.asarray(mask, dtype=bool)
    labels, count = label_mask_components(mask)
    if count <= 1:
        return mask.copy()

    areas = np.bincount(labels.ravel(), minlength=count + 1)
    areas[0] = 0
    primary_id = int(np.argmax(areas))
    return labels == primary_id


def raw_mask_centroid(mask):
    """Return the geometric centroid of every foreground pixel as (x, y)."""
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def primary_component_centroid(mask):
    """Return the centroid of the largest 8-connected component as (x, y)."""
    return raw_mask_centroid(largest_connected_component(mask))


def fragmentation_metrics(mask):
    """Return component count and non-primary area ratio for one mask."""
    mask = np.asarray(mask, dtype=bool)
    total_area = int(mask.sum())
    if total_area == 0:
        return 0, 0.0

    components = component_geometry(mask)
    largest_area = max(component['area'] for component in components)
    fragment_area_ratio = (total_area - largest_area) / total_area
    return len(components), float(fragment_area_ratio)


#                                                                                              
#       
#                                                                                              
def _normalize_astro(img):
    """Normalize an astronomy image with asinh scaling."""
    bg = np.median(img)
    img_sub = img - bg
    bg_mask = img < np.percentile(img, 90)
    scale = np.std(img_sub[bg_mask]) + 1e-6
    a = np.arcsinh(img_sub / scale)
    return ((a - a.mean()) / (a.std() + 1e-6)).astype(np.float32)


def _model_input_image(img, img_info):
    if img_info.get('preprocess') in PREPROCESSED:
        img = np.where(np.isfinite(img), img, 0.0)
        return np.clip(img, 0.0, 1.0).astype(np.float32)
    return _normalize_astro(img)


def compute_input_contrast_snr(image, gt_masks,
                               padding=SNR_LOCAL_PADDING,
                               min_background_pixels=SNR_MIN_BACKGROUND_PIXELS):
    """Compute per-instance peak SNR from the actual model-input image.

    For every GT mask, local background pixels are taken from an expanded
    bounding box while excluding all GT masks. Background level is the local
    median and noise is ``1.4826 * MAD``. The returned quantity is therefore
    an input-domain contrast SNR, not a physical aperture-photometry SNR.
    """
    image = np.asarray(image, dtype=np.float32)
    gt_masks = np.asarray(gt_masks, dtype=bool)
    if image.ndim != 2:
        raise ValueError(f'SNR expects a 2-D image, got shape={image.shape}')
    if gt_masks.ndim != 3 or gt_masks.shape[1:] != image.shape:
        raise ValueError(
            f'SNR mask/image shape mismatch: masks={gt_masks.shape}, image={image.shape}')
    if len(gt_masks) == 0:
        return []

    finite = np.isfinite(image)
    all_foreground = gt_masks.any(axis=0)
    global_background = image[finite & ~all_foreground]
    height, width = image.shape
    values = []

    for mask in gt_masks:
        signal_keep = mask & finite
        ys, xs = np.where(signal_keep)
        if len(xs) == 0:
            values.append(float('nan'))
            continue

        x0 = max(0, int(xs.min()) - padding)
        x1 = min(width, int(xs.max()) + padding + 1)
        y0 = max(0, int(ys.min()) - padding)
        y1 = min(height, int(ys.max()) + padding + 1)
        local_keep = finite[y0:y1, x0:x1] & ~all_foreground[y0:y1, x0:x1]
        background = image[y0:y1, x0:x1][local_keep]
        if background.size < min_background_pixels:
            background = global_background
        if background.size == 0:
            values.append(float('nan'))
            continue

        background_level = float(np.median(background))
        mad = float(np.median(np.abs(background - background_level)))
        noise_std = 1.4826 * mad
        if not np.isfinite(noise_std) or noise_std <= 1e-6:
            noise_std = float(np.std(background))
        if not np.isfinite(noise_std) or noise_std <= 1e-6:
            values.append(float('nan'))
            continue

        peak_signal = float(np.max(image[signal_keep])) - background_level
        values.append(max(0.0, peak_signal / noise_std))

    return values


def mask_centroid(mask):
    """Return the raw geometric mask centroid as an ``[x, y]`` array."""
    center = raw_mask_centroid(mask)
    return None if center is None else np.asarray(center, dtype=np.float64)


def primary_mask_centroid(mask):
    """Return the largest 8-connected component centroid as ``[x, y]``."""
    center = primary_component_centroid(mask)
    return None if center is None else np.asarray(center, dtype=np.float64)


# Precompute vectorized IoU matrices once and reuse them across IoU thresholds.
def precompute_iou_matrices(pred_masks_list, pred_scores_list, gt_masks_list):
    """Return per-image pred-by-GT IoU matrices plus flattened prediction scores."""
    from tqdm import tqdm

    all_preds = []
    total_gt = 0
    iou_mats = []
    CHUNK = 50
    pbar = tqdm(range(len(pred_masks_list)), desc="  Precompute IoU", ncols=100, leave=False)
    for img_idx in pbar:
        pms = pred_masks_list[img_idx]
        pss = pred_scores_list[img_idx]
        K = len(pss)
        M = len(gt_masks_list[img_idx])
        total_gt += M

        for i in range(K):
            all_preds.append((float(pss[i]), img_idx, i))

        if K == 0 or M == 0:
            iou_mats.append(np.zeros((K, M)))
            continue

        gt_flat = gt_masks_list[img_idx].reshape(M, -1).astype(np.float32)  # [M, H*W]
        gt_area = gt_flat.sum(1)  # [M]
        iou_mat = np.zeros((K, M), dtype=np.float32)

        for start in range(0, K, CHUNK):
            end = min(start + CHUNK, K)
            chunk = pms[start:end].reshape(end - start, -1).astype(np.float32)  # [c, H*W]
            inter = chunk @ gt_flat.T  # [c, M]
            pred_area = chunk.sum(1, keepdims=True)  # [c, 1]
            union = pred_area + gt_area[None, :] - inter
            iou_mat[start:end] = inter / (union + 1e-12)

        iou_mats.append(iou_mat)

    return iou_mats, total_gt, all_preds


# Compute AP from precomputed IoU matrices.
def compute_ap_from_iou_mats(iou_mats, total_gt, all_preds, iou_thr):
    """Compute 101-point interpolated AP at one IoU threshold."""
    n_recall_points = 101

    if len(all_preds) == 0:
        return 0.0 if total_gt > 0 else float('nan')

    # Sort predictions by confidence.
    all_preds.sort(key=lambda x: x[0], reverse=True)

    gt_matched = [np.zeros(iou_mats[i].shape[1], dtype=bool) for i in range(len(iou_mats))]
    tp = np.zeros(len(all_preds), dtype=int)
    fp = np.zeros(len(all_preds), dtype=int)

    for idx, (score, img_idx, pred_idx) in enumerate(all_preds):
        iou_vec = iou_mats[img_idx][pred_idx]  # [M]
        matched = gt_matched[img_idx]

        if len(iou_vec) == 0:
            fp[idx] = 1
            continue

        iou_vec_masked = np.where(matched, -1.0, iou_vec)
        best_j = int(np.argmax(iou_vec_masked))
        best_iou = iou_vec_masked[best_j]

        if best_iou >= iou_thr:
            tp[idx] = 1
            gt_matched[img_idx][best_j] = True
        else:
            fp[idx] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    fn_cum = total_gt - tp_cum

    prec = tp_cum / (tp_cum + fp_cum + 1e-12)
    rec = tp_cum / (tp_cum + fn_cum + 1e-12)

    recall_points = np.linspace(0, 1, n_recall_points)
    prec_interp = np.zeros(n_recall_points)
    for i, r in enumerate(recall_points):
        mask = rec >= r
        prec_interp[i] = np.max(prec[mask]) if mask.any() else 0.0

    return float(np.mean(prec_interp))


def compute_f1_from_iou_mats(iou_mats, pred_scores_list, score_thr=0.3,
                             iou_thr=F1_IOU_THRESHOLD):
    """Compute dataset-level instance precision, recall, and F1.

    Predictions below ``score_thr`` are discarded. Remaining predictions are
    processed by descending confidence and greedily matched one-to-one to an
    unmatched GT mask from the same image when mask IoU is at least
    ``iou_thr``. This follows the same matching order used by AP evaluation.
    """
    total_gt = sum(mat.shape[1] for mat in iou_mats)
    predictions = []
    for img_idx, scores in enumerate(pred_scores_list):
        predictions.extend(
            (float(score), img_idx, pred_idx)
            for pred_idx, score in enumerate(scores)
            if float(score) >= score_thr)
    predictions.sort(key=lambda item: item[0], reverse=True)

    gt_matched = [np.zeros(mat.shape[1], dtype=bool) for mat in iou_mats]
    tp = 0
    fp = 0
    for _, img_idx, pred_idx in predictions:
        iou_vec = iou_mats[img_idx][pred_idx]
        if iou_vec.size == 0:
            fp += 1
            continue

        available = np.where(gt_matched[img_idx], -1.0, iou_vec)
        gt_idx = int(np.argmax(available))
        if available[gt_idx] >= iou_thr:
            tp += 1
            gt_matched[img_idx][gt_idx] = True
        else:
            fp += 1

    fn = total_gt - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
    }


def compute_coco_ap(pred_masks_list, pred_scores_list, gt_masks_list,
                    iou_mats=None):
    """Compute mean AP over AP_IOU_THRESHOLDS and return per-threshold AP."""
    if iou_mats is None:
        iou_mats, total_gt, all_preds = precompute_iou_matrices(
            pred_masks_list, pred_scores_list, gt_masks_list)
    else:
        total_gt = sum(len(gts) for gts in gt_masks_list)
        all_preds = []
        for img_idx, pss in enumerate(pred_scores_list):
            for i in range(len(pss)):
                all_preds.append((float(pss[i]), img_idx, i))

    ap_iou = {}
    valid_aps = []
    for iou_thr in AP_IOU_THRESHOLDS:
        ap_i = compute_ap_from_iou_mats(iou_mats, total_gt, all_preds, iou_thr)
        key = int(iou_thr * 100)
        ap_iou[key] = round(ap_i, 4) if not np.isnan(ap_i) else float('nan')
        if not np.isnan(ap_i):
            valid_aps.append(ap_i)

    ap = float(np.mean(valid_aps)) if valid_aps else float('nan')
    return ap, ap_iou


def compute_ap_subset_from_iou_mats(iou_mats, pred_scores_list, gt_keep_list, iou_thr):
    """
    AP for a GT subset with ignore behavior.

    GT where gt_keep=True is evaluated. Predictions matching non-target GT are ignored
    instead of counted as false positives, because MAINet predicts one foreground class.
    """
    total_gt = sum(int(np.asarray(k, dtype=bool).sum()) for k in gt_keep_list)
    if total_gt == 0:
        return float('nan')

    all_preds = []
    for img_idx, pss in enumerate(pred_scores_list):
        for pred_idx, score in enumerate(pss):
            all_preds.append((float(score), img_idx, pred_idx))
    if len(all_preds) == 0:
        return 0.0
    all_preds.sort(key=lambda x: x[0], reverse=True)

    gt_matched = [np.zeros(int(np.asarray(k, dtype=bool).sum()), dtype=bool)
                  for k in gt_keep_list]
    tp, fp = [], []

    for score, img_idx, pred_idx in all_preds:
        iou_vec = iou_mats[img_idx][pred_idx]
        keep = np.asarray(gt_keep_list[img_idx], dtype=bool)
        target_ious = iou_vec[keep]
        ignore_ious = iou_vec[~keep]

        if target_ious.size > 0:
            masked = np.where(gt_matched[img_idx], -1.0, target_ious)
            best_j = int(np.argmax(masked))
            best_iou = masked[best_j]
            if best_iou >= iou_thr:
                tp.append(1)
                fp.append(0)
                gt_matched[img_idx][best_j] = True
                continue

        if ignore_ious.size > 0 and float(ignore_ious.max()) >= iou_thr:
            continue

        tp.append(0)
        fp.append(1)

    if len(tp) == 0:
        return 0.0

    tp_cum = np.cumsum(np.asarray(tp, dtype=np.float32))
    fp_cum = np.cumsum(np.asarray(fp, dtype=np.float32))
    rec = tp_cum / (total_gt + 1e-12)
    prec = tp_cum / (tp_cum + fp_cum + 1e-12)

    prec_interp = []
    for r in np.linspace(0, 1, 101):
        mask = rec >= r
        prec_interp.append(float(np.max(prec[mask])) if mask.any() else 0.0)
    return float(np.mean(prec_interp))


def compute_coco_ap_for_gt_subset(iou_mats, pred_scores_list, gt_keep_list):
    ap_iou = {}
    valid_aps = []
    for iou_thr in AP_IOU_THRESHOLDS:
        ap_i = compute_ap_subset_from_iou_mats(
            iou_mats, pred_scores_list, gt_keep_list, iou_thr)
        key = int(iou_thr * 100)
        ap_iou[key] = round(ap_i, 4) if not np.isnan(ap_i) else float('nan')
        if not np.isnan(ap_i):
            valid_aps.append(ap_i)
    ap = float(np.mean(valid_aps)) if valid_aps else float('nan')
    return ap, ap_iou


def match_all_gt(iou_mats, pred_scores_list, score_thr=0.3, iou_thr=0.5):
    """Match exactly like overall cMSE: per-image Hungarian mask-IoU matching."""
    matched_pred = [np.full(mat.shape[1], -1, dtype=np.int64) for mat in iou_mats]
    for img_idx, (iou_mat, scores) in enumerate(zip(iou_mats, pred_scores_list)):
        keep = np.where(np.asarray(scores) >= score_thr)[0]
        if len(keep) == 0 or iou_mat.shape[1] == 0:
            continue
        filtered_iou = iou_mat[keep]
        pred_indices, gt_indices = linear_sum_assignment(-filtered_iou)
        for filtered_pred_idx, gt_idx in zip(pred_indices, gt_indices):
            if filtered_iou[filtered_pred_idx, gt_idx] >= iou_thr:
                matched_pred[img_idx][gt_idx] = int(keep[filtered_pred_idx])
    return matched_pred


def compute_snr_recall_metrics(iou_mats, pred_scores_list, gt_snr_list,
                               score_thr=0.3):
    """Compute GT-conditioned recall after one global match."""
    snr_arrays = [np.asarray(values, dtype=np.float32)
                  for values in gt_snr_list]
    matched_pred = match_all_gt(
        iou_mats, pred_scores_list,
        score_thr=score_thr, iou_thr=F1_IOU_THRESHOLD)

    metrics = {}
    for label, lower, upper in SNR_BINS:
        keep_list = [np.isfinite(values) & (values >= lower) & (values < upper)
                     for values in snr_arrays]
        count = sum(int(keep.sum()) for keep in keep_list)
        tp = sum(int(((assignment >= 0) & keep).sum())
                 for assignment, keep in zip(matched_pred, keep_list))
        fn = count - tp
        recall = tp / count if count > 0 else float('nan')

        metrics[label] = {
            'range': [lower, None if np.isinf(upper) else upper],
            'count': count,
            'tp': int(tp),
            'fn': int(fn),
            'recall': (round(recall, 4)
                       if not np.isnan(recall) else float('nan')),
        }
    return metrics


def subset_gt_by_keep(gt_masks_list, iou_mats, gt_keep_list):
    sub_masks, sub_iou = [], []
    for gts, mat, keep in zip(gt_masks_list, iou_mats, gt_keep_list):
        keep = np.asarray(keep, dtype=bool)
        sub_masks.append(gts[keep])
        sub_iou.append(mat[:, keep])
    return sub_masks, sub_iou


#                                                                                              
# Centroid MSE with IoU matching.
def compute_centroid_mse(pred_masks_list, pred_scores_list, gt_masks_list,
                         iou_mats=None, score_thr=0.3, iou_thr=0.5):
    """Compute matched-TP centroid MSE from each prediction's main component.

    IoU matching still uses the complete raw masks.  Only the prediction
    centroid is made robust to disconnected satellite pixels. GT centroids
    retain their original complete-mask definition.
    """
    total_mse = 0.0
    total_tp = 0

    for img_idx in range(len(pred_masks_list)):
        pms = pred_masks_list[img_idx]
        pss = pred_scores_list[img_idx]
        gts = gt_masks_list[img_idx]

        keep = np.where(pss >= score_thr)[0]
        if len(keep) == 0 or len(gts) == 0:
            continue

        pms_f = pms[keep]

        if iou_mats is not None:
            iou_mat = iou_mats[img_idx][keep]  # [K_f, M]
        else:
            K, M = len(pms_f), len(gts)
            pred_flat = pms_f.reshape(K, -1).astype(np.float32)
            gt_flat = gts.reshape(M, -1).astype(np.float32)
            inter = pred_flat @ gt_flat.T
            pred_area = pred_flat.sum(1, keepdims=True)
            gt_area = gt_flat.sum(1)
            union = pred_area + gt_area[None, :] - inter
            iou_mat = inter / (union + 1e-12)

        pi, gj = linear_sum_assignment(-iou_mat)

        for p, g in zip(pi, gj):
            if iou_mat[p, g] >= iou_thr:
                pc = primary_mask_centroid(pms_f[p])
                gc = mask_centroid(gts[g])
                if pc is not None and gc is not None:
                    total_mse += float(np.sum((pc - gc) ** 2))
                    total_tp += 1

    if total_tp == 0:
        return float('nan'), 0
    return total_mse / total_tp, total_tp


def _find_binary_pairs(gt_masks):
    """Find disjoint GT binary-star pairs from overlapping 2-sigma masks."""
    candidate_indices = range(len(gt_masks))
    pairs = []
    paired = set()
    for i, j1 in enumerate(candidate_indices):
        if j1 in paired:
            continue
        for j2 in range(i + 1, len(gt_masks)):
            if j2 in paired:
                continue
            overlap = np.logical_and(gt_masks[j1], gt_masks[j2]).sum()
            if overlap > 0:
                pairs.append((j1, j2))
                paired.add(j1); paired.add(j2)
                break
    return pairs


def _binary_gt_keep(gt_masks):
    """Return one boolean vector selecting members of overlapping GT pairs."""
    keep = np.zeros(len(gt_masks), dtype=bool)
    for first, second in _find_binary_pairs(gt_masks):
        keep[first] = True
        keep[second] = True
    return keep


def compute_binary_separation_rate(pred_masks_list, pred_scores_list,
                                    gt_masks_list,
                                    iou_mats=None, score_thr=0.3, iou_thr=0.5):
    """Return separated binary-pair rate using IoU matching."""
    total_pairs = 0
    separated_pairs = 0

    for img_idx in range(len(pred_masks_list)):
        pms = pred_masks_list[img_idx]
        pss = pred_scores_list[img_idx]
        gts = gt_masks_list[img_idx]
        pairs = _find_binary_pairs(gts)
        if len(pairs) == 0:
            continue
        total_pairs += len(pairs)

        keep = np.where(pss >= score_thr)[0]
        if len(keep) == 0:
            continue

        if iou_mats is not None:
            iou_mat = iou_mats[img_idx][keep]
        else:
            pms_f = pms[keep]
            K, M = len(pms_f), len(gts)
            pred_flat = pms_f.reshape(K, -1).astype(np.float32)
            gt_flat = gts.reshape(M, -1).astype(np.float32)
            inter = pred_flat @ gt_flat.T
            pred_area = pred_flat.sum(1, keepdims=True)
            gt_area = gt_flat.sum(1)
            union = pred_area + gt_area[None, :] - inter
            iou_mat = inter / (union + 1e-12)

        pi, gj = linear_sum_assignment(-iou_mat)

        gt_to_pred = {}
        for p, g in zip(pi, gj):
            if iou_mat[p, g] >= iou_thr:
                gt_to_pred[g] = p

        for j1, j2 in pairs:
            if j1 in gt_to_pred and j2 in gt_to_pred:
                if gt_to_pred[j1] != gt_to_pred[j2]:
                    separated_pairs += 1

    if total_pairs == 0:
        return float('nan'), 0, 0
    return separated_pairs / total_pairs, separated_pairs, total_pairs


def evaluate_model(model, model_type, data_root, split, device, score_thr=0.3, max_images=0,
                   eval_batch_size=4):
    """Evaluate one model and return AP, cMSE, SNR split, and binary metrics."""
    from tqdm import tqdm

    coco_json = f"{data_root}/annotations/{split}.json"
    with open(coco_json, encoding='utf-8-sig') as f:
        coco = json.load(f)

    images = coco['images']
    if max_images > 0:
        images = images[:max_images]

    mask_dir = f"{data_root}/{split}/masks"
    img_dir = f"{data_root}/{split}/images"

    use_batch = model_type in ('local-rcnn', 'local-query') and eval_batch_size > 1

    if use_batch:
        #                      +     GPU         
        img_tensors = []   # list of [512,512] torch tensors
        gt_masks_all = []
        gt_snr_all = []
        gt_modes_all = []

        for im in tqdm(images, desc="Load data", ncols=100):
            image_id = im['id']
            img_path = f"{img_dir}/{im['file_name']}"
            if not os.path.exists(img_path):
                print(f"  [warn]             : {img_path}")
                continue
            img = np.load(img_path).astype(np.float32)
            img = _model_input_image(img, im)
            img_tensors.append(torch.from_numpy(img))
            gt_masks = np.load(f"{mask_dir}/{image_id:06d}_masks.npy").astype(bool)
            gt_snr = compute_input_contrast_snr(img, gt_masks)
            gt_masks_all.append(gt_masks)
            gt_snr_all.append(gt_snr)
            gt_modes_all.append(str(im.get('mode', '')).lower())

        #       
        pred_masks_all = []
        pred_scores_all = []
        BS = eval_batch_size

        for i in tqdm(range(0, len(img_tensors), BS), desc="Infer", ncols=100):
            batch = img_tensors[i:i + BS]
            B = len(batch)
            x = torch.stack(batch).unsqueeze(1).to(device)  # [B, 1, 512, 512]

            with torch.no_grad():
                if model_type == 'local-rcnn':
                    outputs = model(x)  # list of B dicts
                    for out in outputs:
                        pred_masks_all.append(out['masks'].cpu().numpy().astype(bool))
                        pred_scores_all.append(out['scores'].cpu().numpy())
                else:  # local-query
                    outputs = model(x)
                    for b in range(B):
                        scores = outputs['pred_logits'][b, :, 0].sigmoid().cpu()
                        masks_small = outputs['pred_masks'][b].sigmoid()
                        masks = F.interpolate(masks_small.unsqueeze(1), size=(512, 512),
                                              mode='bilinear', align_corners=False).squeeze(1)
                        pred_masks_all.append((masks > 0.5).cpu().numpy().astype(bool))
                        pred_scores_all.append(scores.numpy())

    else:
        #                 mdet / batch=1     
        pred_masks_all = []
        pred_scores_all = []
        gt_masks_all = []
        gt_snr_all = []
        gt_modes_all = []

        for im in tqdm(images, desc="Infer", ncols=100):
            image_id = im['id']
            img_path = os.path.abspath(f"{img_dir}/{im['file_name']}")

            if not os.path.exists(img_path):
                print(f"  [warn]             : {img_path}")
                continue

            if model_type == 'mmdet':
                result = inference_detector(model, img_path)
                inst = result.pred_instances
                pms = inst.masks.cpu().numpy().astype(bool)
                pss = inst.scores.cpu().numpy()
            elif model_type == 'yolo':
                image = np.load(img_path).astype(np.float32)
                rgb = npy_to_yolo_rgb(image, im)
                result = model.predict(
                    source=rgb,
                    imgsz=1024,
                    conf=0.001,
                    iou=0.80,
                    max_det=50,
                    retina_masks=True,
                    verbose=False,
                    device=0 if str(device).startswith('cuda') else 'cpu')[0]
                if result.masks is None or result.boxes is None:
                    pms = np.zeros((0, rgb.shape[0], rgb.shape[1]), dtype=bool)
                    pss = np.zeros((0,), dtype=np.float32)
                else:
                    pms = result.masks.data.cpu().numpy() > 0.5
                    pss = result.boxes.conf.cpu().numpy()
                    keep = yolo_mask_nms_indices(pms, pss, iou_thr=0.50, max_det=50)
                    pms, pss = pms[keep], pss[keep]
            elif model_type == 'local-query':
                img = np.load(img_path).astype(np.float32)
                x = torch.from_numpy(_model_input_image(img, im))[None, None].to(device)
                with torch.no_grad():
                    outputs = model(x)
                scores = outputs['pred_logits'][0, :, 0].sigmoid().cpu()
                masks_small = outputs['pred_masks'][0].sigmoid()
                masks = F.interpolate(masks_small.unsqueeze(1), size=(512, 512),
                                      mode='bilinear', align_corners=False).squeeze(1)
                pms = (masks > 0.5).cpu().numpy().astype(bool)
                pss = scores.numpy()
            else:
                img = np.load(img_path).astype(np.float32)
                x = torch.from_numpy(_model_input_image(img, im))[None, None].to(device)
                with torch.no_grad():
                    results = model(x)[0]
                pms = results['masks'].cpu().numpy().astype(bool)
                pss = results['scores'].cpu().numpy()

            pred_masks_all.append(pms)
            pred_scores_all.append(pss)
            gt_masks = np.load(f"{mask_dir}/{image_id:06d}_masks.npy").astype(bool)
            snr_image = np.load(img_path).astype(np.float32)
            snr_image = _model_input_image(snr_image, im)
            gt_snr = compute_input_contrast_snr(snr_image, gt_masks)
            gt_masks_all.append(gt_masks)
            gt_snr_all.append(gt_snr)
            gt_modes_all.append(str(im.get('mode', '')).lower())

    print("Precompute IoU ...", end=' ', flush=True)
    iou_mats, _, _ = precompute_iou_matrices(pred_masks_all, pred_scores_all, gt_masks_all)
    print("done", flush=True)

    #                 
    ap, ap_iou = compute_coco_ap(pred_masks_all, pred_scores_all, gt_masks_all,
                                 iou_mats=iou_mats)
    f1_metrics = compute_f1_from_iou_mats(
        iou_mats, pred_scores_all, score_thr=score_thr)
    snr_metrics = compute_snr_recall_metrics(
        iou_mats, pred_scores_all, gt_snr_all, score_thr=score_thr)
    snr_missing = sum(int((~np.isfinite(np.asarray(values))).sum())
                      for values in gt_snr_all)
    cmse, _ = compute_centroid_mse(
        pred_masks_all, pred_scores_all, gt_masks_all,
        iou_mats=iou_mats, score_thr=score_thr)

    #                         GT         
    bright_keep = [np.isfinite(np.asarray(snr, dtype=np.float32)) &
                   (np.asarray(snr, dtype=np.float32) >= 3.0)
                   for snr in gt_snr_all]
    faint_keep = [np.isfinite(np.asarray(snr, dtype=np.float32)) &
                  (np.asarray(snr, dtype=np.float32) < 3.0)
                  for snr in gt_snr_all]
    binary_keep = [_binary_gt_keep(masks) for masks in gt_masks_all]

    bright_ap, bright_ap_iou = compute_coco_ap_for_gt_subset(
        iou_mats, pred_scores_all, bright_keep)
    faint_ap, faint_ap_iou = compute_coco_ap_for_gt_subset(
        iou_mats, pred_scores_all, faint_keep)

    def evaluate_mode(mode):
        indices = [i for i, image_mode in enumerate(gt_modes_all)
                   if image_mode == mode]
        if not indices:
            return float('nan'), {}
        return compute_coco_ap(
            [pred_masks_all[i] for i in indices],
            [pred_scores_all[i] for i in indices],
            [gt_masks_all[i] for i in indices],
            iou_mats=[iou_mats[i] for i in indices])

    point_ap, point_ap_iou = evaluate_mode('point')
    streak_ap, streak_ap_iou = evaluate_mode('streak')

    binary_indices = [i for i, keep in enumerate(binary_keep) if keep.any()]
    binary_pred_masks = [pred_masks_all[i] for i in binary_indices]
    binary_pred_scores = [pred_scores_all[i] for i in binary_indices]
    binary_gt_masks = [gt_masks_all[i] for i in binary_indices]
    binary_iou_mats = [iou_mats[i] for i in binary_indices]

    b_ap = float('nan')
    b_ap_iou = {}
    b_mse = float('nan')
    b_sep_rate = float('nan')
    b_sep_detail = (0, 0)

    if len(binary_gt_masks) > 0:
        binary_gt_keep = [binary_keep[i] for i in binary_indices]
        b_ap, b_ap_iou = compute_coco_ap_for_gt_subset(
            binary_iou_mats, binary_pred_scores, binary_gt_keep)
        binary_only_gt_masks, binary_only_iou_mats = subset_gt_by_keep(
            binary_gt_masks, binary_iou_mats, binary_gt_keep)
        b_mse, _ = compute_centroid_mse(binary_pred_masks, binary_pred_scores,
                                        binary_only_gt_masks, iou_mats=binary_only_iou_mats,
                                        score_thr=score_thr)
        b_sep_rate, b_sep, b_sep_total = compute_binary_separation_rate(
            binary_pred_masks, binary_pred_scores, binary_gt_masks,
            iou_mats=binary_iou_mats, score_thr=score_thr)
        b_sep_detail = (b_sep, b_sep_total)

    return dict(
        ap=round(ap, 4) if not np.isnan(ap) else float('nan'),
        ap_iou=ap_iou,
        precision=round(f1_metrics['precision'], 4),
        recall=round(f1_metrics['recall'], 4),
        f1=round(f1_metrics['f1'], 4),
        f1_tp=f1_metrics['tp'],
        f1_fp=f1_metrics['fp'],
        f1_fn=f1_metrics['fn'],
        f1_score_thr=float(score_thr),
        f1_iou_thr=F1_IOU_THRESHOLD,
        snr_metrics=snr_metrics,
        snr_missing=snr_missing,
        snr_definition=SNR_DEFINITION,
        snr_local_padding=SNR_LOCAL_PADDING,
        snr_matching='hungarian_mask_iou_0.50',
        centroid_mse_reference='gt_mask_centroid',
        cmse=round(cmse, 4) if not np.isnan(cmse) else float('nan'),
        bright_ap=round(bright_ap, 4) if not np.isnan(bright_ap) else float('nan'),
        bright_ap_iou=bright_ap_iou,
        faint_ap=round(faint_ap, 4) if not np.isnan(faint_ap) else float('nan'),
        faint_ap_iou=faint_ap_iou,
        point_ap=round(point_ap, 4) if not np.isnan(point_ap) else float('nan'),
        point_ap_iou=point_ap_iou,
        streak_ap=round(streak_ap, 4) if not np.isnan(streak_ap) else float('nan'),
        streak_ap_iou=streak_ap_iou,
        b_ap=round(b_ap, 4) if not np.isnan(b_ap) else float('nan'),
        b_ap_iou=b_ap_iou,
        b_mse=round(b_mse, 4) if not np.isnan(b_mse) else float('nan'),
        b_sep_rate=round(b_sep_rate, 4) if not np.isnan(b_sep_rate) else float('nan'),
        b_sep_detail=b_sep_detail,
    )


# Checkpoint policy: prefer best_model.pt / YOLO best.pt / best_coco_*.pth.
# last_checkpoint is only a recovery fallback when no best checkpoint exists.
def _find_best_ckpt(ckpt_path):
    """Find the best checkpoint in a file or directory."""
    if os.path.isfile(ckpt_path):
        return ckpt_path
    if not os.path.isdir(ckpt_path):
        return None
    for name in ['best_model.pt', 'best.pt']:
        p = os.path.join(ckpt_path, name)
        if os.path.exists(p):
            return p
    yolo_best = os.path.join(ckpt_path, 'weights', 'best.pt')
    if os.path.exists(yolo_best):
        return yolo_best
    # MMDetection best checkpoints are the model-selection result. They must
    # take precedence over ``last_checkpoint``, which only exists for resume.
    best = sorted(
        [f for f in os.listdir(ckpt_path) if f.startswith('best_coco')],
        key=lambda name: os.path.getmtime(os.path.join(ckpt_path, name)),
        reverse=True)
    if best:
        return os.path.join(ckpt_path, best[0])
    lc = os.path.join(ckpt_path, 'last_checkpoint')
    if os.path.exists(lc):
        with open(lc) as f:
            real = f.read().strip()
        p = real if os.path.exists(real) else os.path.join(ckpt_path, os.path.basename(real))
        if os.path.exists(p):
            return p
    ep = sorted([f for f in os.listdir(ckpt_path)
                 if f.endswith('.pt') and 'epoch' in f],
                key=lambda x: int(x.split('_')[-1].replace('.pt', '')), reverse=True)
    if ep:
        return os.path.join(ckpt_path, ep[0])
    return None


def _find_all_ckpts(ckpt_dir):
    """Find all epoch checkpoints in ascending epoch order."""
    entries = []
    if not os.path.isdir(ckpt_dir):
        return entries

    def _epoch(fname):
        """Extract an epoch number from a checkpoint filename."""
        import re
        # epoch_N.pt / epoch_N.pth / best_coco_...epoch_N.pth
        m = re.search(r'epoch[_\s]*(\d+)', fname)
        return int(m.group(1)) if m else 9999

    seen = set()
    for f in sorted(os.listdir(ckpt_dir), key=_epoch):
        if not (f.endswith('.pt') or f.endswith('.pth')):
            continue
        fpath = os.path.join(ckpt_dir, f)
        if not os.path.isfile(fpath):
            continue
        ep = _epoch(f)
        if ep in seen:
            continue
        seen.add(ep)
        entries.append({'name': f, 'path': fpath, 'epoch': ep})
    return entries


def _resolve_variant(dir_name):
    """Resolve a model variant name from a directory name."""
    name = dir_name.replace('mainet_rcnn_', '').replace('mainet_', '').replace('_star', '')
    return name


def _make_model_name(d, typ='local'):
    """Create a display name for a model."""
    if typ == 'local':
        variant = _resolve_variant(d)
        variant = variant.replace('rcnn_', '').replace('mainet_', '')
        # Examples: v3_none -> V3-none, v3_psf_strip -> V3-psf_strip.
        if variant.startswith('v3_'):
            abl = variant[3:]
            return f'V3-{abl}'
        if variant.startswith('v4_'):
            abl = variant[3:]
            return f'V4-{abl}'
        return f'Mainet-{variant.upper()}' if variant else d.upper()
    elif typ == 'yolo':
        return 'YOLO-seg'
    else:
        return d.replace('_star', '').replace('_', ' ').title().replace(' ', '')


def _import_rcnn_model(variant):
    """Import MAINetRCNN from mainet/<variant>."""
    model_dir = os.path.join(PROJECT_ROOT, 'mainet', variant)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    for mod in list(sys.modules):
        if mod in ('model', 'backbone', 'heads', 'dataset', 'param_loss'):
            sys.modules.pop(mod, None)
    import model as _m
    return _m.MAINetRCNN


def _detect_ablation(variant):
    """Detect legacy ablation variants such as v3_strip."""
    for base in ['v3', 'v4']:
        prefix = f'{base}_'
        if variant.startswith(prefix):
            candidate = variant[len(prefix):]
            #        ablation       
            ablation_path = os.path.join(PROJECT_ROOT, 'mainet', base, 'ablation.py')
            if os.path.exists(ablation_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location(f'ablation_{base}', ablation_path)
                abl_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(abl_mod)
                if candidate in abl_mod.ABLATIONS:
                    return base, candidate
    return None, None


def _resolve_local_variant_and_ablation(ckpt_dir):
    """Resolve normal dirs and nested ablation dirs.

    Normal: work_dirs/mainet/v4
    Legacy ablation: work_dirs/mainet/v4_strip
    Isolated V4 ablation: work_dirs/ablation/mainet_v4/checkpoints/strip
    """
    dir_name = os.path.basename(os.path.normpath(ckpt_dir))
    variant = _resolve_variant(dir_name)
    base_variant, ablation_name = _detect_ablation(variant)
    if base_variant:
        return base_variant, ablation_name

    parent = os.path.basename(os.path.dirname(os.path.normpath(ckpt_dir)))
    if parent in ('v3', 'v4'):
        ablation_path = os.path.join(PROJECT_ROOT, 'mainet', parent, 'ablation.py')
        if os.path.exists(ablation_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location(f'ablation_{parent}', ablation_path)
            abl_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(abl_mod)
            if dir_name in abl_mod.ABLATIONS:
                return parent, dir_name

    experiment_name = os.path.basename(
        os.path.dirname(os.path.dirname(os.path.normpath(ckpt_dir))))
    if parent == 'checkpoints' and experiment_name in (
            'mainet_v4', 'mainet_v4_debug'):
        ablation_path = os.path.join(PROJECT_ROOT, 'mainet', 'v4', 'ablation.py')
        import importlib.util
        spec = importlib.util.spec_from_file_location('ablation_v4', ablation_path)
        abl_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(abl_mod)
        if dir_name in abl_mod.ABLATIONS:
            return 'v4', dir_name

    return variant, None


def _load_local_model(ckpt_dir, device):
    """Load a local MAINet checkpoint."""
    ckpt = _find_best_ckpt(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint: {ckpt_dir}")

    ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
    sd = ckpt_data.get('model_state_dict', ckpt_data)

    variant, ablation_name = _resolve_local_variant_and_ablation(ckpt_dir)
    base_variant = variant if ablation_name else None

    if base_variant:
        variant = base_variant

    # RCNN / Query
    if 'rpn.head.conv.weight' in sd:
        MAINetRCNN = _import_rcnn_model(variant)
        model = MAINetRCNN(in_chans=1, num_classes=1)

        # Swap backbone for ablation variants when needed.
        if ablation_name:
            ablation_path = os.path.join(PROJECT_ROOT, 'mainet', base_variant, 'ablation.py')
            import importlib.util
            spec = importlib.util.spec_from_file_location(f'_abl_{base_variant}', ablation_path)
            abl_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(abl_mod)
            bb_cfg = abl_mod.ABLATIONS[ablation_name]
            model.backbone = abl_mod.DualPathBackboneAblation(
                in_ch=1, stem_ch=32,
                psf_type=bb_cfg['psf'], psf_k=4,
                strip_dirs=bb_cfg['strip'],
                strip_plain=bb_cfg.get('strip_plain', False),
                fusion=bb_cfg['fusion'],
                gate_mode=bb_cfg.get('gate', 'learned'))

        model_type = 'local-rcnn'
    else:
        from mainet.query_head.mainet import MAINet
        model = MAINet(in_chans=1, num_queries=100, num_classes=1)
        model_type = 'local-query'

    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, model_type


def _load_mmdet_model(config_path, ckpt_dir, device):
    """Load an MMDetection model from a config and checkpoint directory."""
    if not _mmdet_available:
        raise RuntimeError("mmdet is not installed or not available in this environment")

    mmdet_dir = os.path.join(PROJECT_ROOT, 'mmdet')
    if mmdet_dir not in sys.path:
        sys.path.insert(0, mmdet_dir)

    ckpt = _find_best_ckpt(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint: {ckpt_dir}")
    model = init_detector(config_path, ckpt, device=device)
    model.eval()
    return model, 'mmdet'


def _load_yolo_model(ckpt_dir, device):
    """Load and validate an Ultralytics instance-segmentation checkpoint."""
    if not _yolo_available:
        raise RuntimeError('ultralytics is not installed in the active environment')
    ckpt = _find_best_ckpt(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(f'No checkpoint: {ckpt_dir}')
    model = YOLO(ckpt)
    require_yolo_segmentation(model)
    return model, 'yolo'


def _mmdet_config_candidates(model_name, ckpt_dir):
    """Return config candidates for an MMDetection baseline."""
    runtime = os.path.join(ckpt_dir, '_runtime_config.py')
    source_real = os.path.join(
        PROJECT_ROOT, 'mmdet', 'configs_real_mixed', f'{model_name}_star.py')
    candidates = [
        runtime,
        os.path.join(ckpt_dir, f'{model_name}.py'),
        source_real,
        os.path.join(PROJECT_ROOT, 'mmdet', 'configs', f'{model_name}.py'),
        os.path.join(PROJECT_ROOT, 'mmdet', 'configs', f'{model_name}_star.py'),
    ]
    return candidates


def discover_models():
    """Discover checkpoints under work_dirs, including the real mixed layout."""
    models = []
    WORK = os.path.join(PROJECT_ROOT, 'work_dirs')
    if not os.path.isdir(WORK):
        return models

    for framework, typ in [('mainet', 'local'), ('mmdet', 'mmdet')]:
        fw_dir = os.path.join(WORK, framework)
        if not os.path.isdir(fw_dir):
            continue
        model_dirs = []
        for entry in sorted(os.listdir(fw_dir)):
            epath = os.path.join(fw_dir, entry)
            if not os.path.isdir(epath):
                continue
            if _find_best_ckpt(epath):
                model_dirs.append((entry, epath))
            else:
                for sub in sorted(os.listdir(epath)):
                    spath = os.path.join(epath, sub)
                    if os.path.isdir(spath) and _find_best_ckpt(spath):
                        model_dirs.append((sub, spath))

        for d, dpath in model_dirs:
            if typ == 'local':
                models.append((_make_model_name(d, 'local'), typ, None, dpath))
            else:
                cfg = next((c for c in _mmdet_config_candidates(d, dpath) if os.path.exists(c)), None)
                if cfg:
                    models.append((_make_model_name(d, 'mmdet'), typ, cfg, dpath))

    real_root = os.path.join(WORK, 'real_mixed_baselines')
    if os.path.isdir(real_root):
        mainet_v4 = os.path.join(real_root, 'mainet_v4')
        if _find_best_ckpt(mainet_v4):
            models.append(('v4-real', 'local', None, mainet_v4))
        real_mmdet = os.path.join(real_root, 'mmdet')
        if os.path.isdir(real_mmdet):
            for d in sorted(os.listdir(real_mmdet)):
                dpath = os.path.join(real_mmdet, d)
                if not os.path.isdir(dpath) or not _find_best_ckpt(dpath):
                    continue
                cfg = next((c for c in _mmdet_config_candidates(d, dpath) if os.path.exists(c)), None)
                if cfg:
                    models.append((f"{_make_model_name(d, 'mmdet')}-real", 'mmdet', cfg, dpath))
        real_yolo = os.path.join(real_root, 'yolo')
        if _find_best_ckpt(real_yolo):
            models.append(('YOLO-seg-real', 'yolo', None, real_yolo))

    for d in sorted(os.listdir(WORK)):
        dpath = os.path.join(WORK, d)
        if not os.path.isdir(dpath) or d in ('mainet', 'mmdet', 'real_mixed_baselines'):
            continue
        if d.startswith('mainet_'):
            ckpt = _find_best_ckpt(dpath)
            if ckpt:
                models.append((_make_model_name(d, 'local'), 'local', None, dpath))
        elif d.endswith('_star'):
            base = d.replace('_star', '')
            cfg = next((c for c in _mmdet_config_candidates(base, dpath) if os.path.exists(c)), None)
            ckpt = _find_best_ckpt(dpath)
            if ckpt and cfg:
                models.append((_make_model_name(d, 'mmdet'), 'mmdet', cfg, dpath))

    return models


def print_table(results):
    """Print overall, SNR-split, and binary-star metrics."""
    SEP = "|"
    COL_W = 10
    IOU_KEYS = [10, 20, 30, 40, 50, 60, 70, 75, 80, 90]

    def _fmt7(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "    N/A"
        return f"{v:>7.4f}"

    # AP-at-IoU plus legacy AP50/AP75 fallback.
    def _get_ap_iou(m, prefix=''):
        """Get AP-at-IoU values from either new or legacy result keys."""
        if f'{prefix}ap_iou' in m:
            d = m[f'{prefix}ap_iou']
            return {int(k): v for k, v in d.items()}
        d = {}
        if f'{prefix}ap50' in m:
            d[50] = m[f'{prefix}ap50']
        if f'{prefix}ap75' in m:
            d[75] = m[f'{prefix}ap75']
        return d

    iou_headers = ''.join(f'{SEP}{"AP"+str(iou):<7}' for iou in IOU_KEYS)
    print("\n" + "=" * (16 + 12 * 8))
    print("Overall (F1: score threshold from --score_thr, mask IoU >= 0.50):")
    print("-" * (16 + 12 * 8))
    print(f"{'Model':<16}{SEP}{'AP':>7}{iou_headers}{SEP}{'F1@50':>7}{SEP}{'cMSE':>7}")
    print("-" * (16 + 12 * 8))
    for name, m in results.items():
        ap_iou = _get_ap_iou(m)
        iou_vals = ''.join(f'{SEP}{_fmt7(ap_iou.get(k))}' for k in IOU_KEYS)
        print(f"{name:<16}{SEP}{_fmt7(m['ap'])}{iou_vals}"
              f"{SEP}{_fmt7(m.get('f1'))}{SEP}{_fmt7(m['cmse'])}")

    print("\nInput-contrast-SNR-stratified recall "
          "(local peak/MAD; score >= --score_thr; Hungarian mask IoU >= 0.50):")
    print("-" * 60)
    print(f"{'Model':<16}{SEP}{'SNR':>7}{SEP}{'GT':>6}{SEP}{'TP':>6}"
          f"{SEP}{'FN':>6}{SEP}{'Recall':>7}")
    print("-" * 60)
    for label, _, _ in SNR_BINS:
        for name, m in results.items():
            group = m.get('snr_metrics', {}).get(label, {})
            count = group.get('count')
            count_text = f"{count:>6d}" if isinstance(count, int) else f"{'N/A':>6}"
            def _count_text(key):
                value = group.get(key)
                return f"{value:>6d}" if isinstance(value, int) else f"{'N/A':>6}"
            print(f"{name:<16}{SEP}{label:>7}{SEP}{count_text}"
                  f"{SEP}{_count_text('tp')}{SEP}{_count_text('fn')}"
                  f"{SEP}{_fmt7(group.get('recall'))}")
        print("-" * 60)
    missing = ', '.join(
        f"{name}={m.get('snr_missing', 'N/A')}" for name, m in results.items())
    print(f"GT without SNR (excluded from SNR groups): {missing}")

    print("\nMorphology split (from image.mode, not training classes):")
    print("-" * (16 + 8 * 7))
    print(f"{'Model':<16}{SEP}{'Pnt-AP':>7}{SEP}{'Pnt75':>7}{SEP}{'Pnt90':>7}"
          f"{SEP}{'Str-AP':>7}{SEP}{'Str75':>7}{SEP}{'Str90':>7}")
    print("-" * (16 + 8 * 7))
    for name, m in results.items():
        point_iou = _get_ap_iou(m, 'point_')
        streak_iou = _get_ap_iou(m, 'streak_')
        print(f"{name:<16}{SEP}{_fmt7(m.get('point_ap'))}"
              f"{SEP}{_fmt7(point_iou.get(75))}{SEP}{_fmt7(point_iou.get(90))}"
              f"{SEP}{_fmt7(m.get('streak_ap'))}"
              f"{SEP}{_fmt7(streak_iou.get(75))}{SEP}{_fmt7(streak_iou.get(90))}")

    b_iou_headers = ''.join(f'{SEP}{"B-AP"+str(iou):<7}' for iou in IOU_KEYS)
    print("\n" + "=" * (16 + 14 * 8))
    print("Binary-star metrics:")
    print("-" * (16 + 14 * 8))
    print(f"{'Model':<16}{SEP}{'B-AP':>7}{b_iou_headers}{SEP}{'B-MSE':>7}{SEP}{'SepRate':>9}")
    print("-" * (16 + 14 * 8))
    for name, m in results.items():
        b_ap_iou = _get_ap_iou(m, 'b_')
        b_iou_vals = ''.join(f'{SEP}{_fmt7(b_ap_iou.get(k))}' for k in IOU_KEYS)
        sep, total = m.get('b_sep_detail', (0, 0))
        if total > 0 and not np.isnan(m['b_sep_rate']):
            sep_str = f"{m['b_sep_rate']:.3f}({sep}/{total})"
        else:
            sep_str = "N/A"
        print(f"{name:<16}{SEP}{_fmt7(m['b_ap'])}{b_iou_vals}{SEP}{_fmt7(m['b_mse'])}{SEP}{sep_str:>9}")

    print("=" * (16 + 13 * 8))


def print_stage_table(model_name, stage_results):
    """Print AP progression across training stages."""
    SEP = "|"

    def _fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "     N/A"
        return f"{v:>8.4f}"

    print(f"\n{'=' * 80}")
    print(f"  {model_name} training-stage AP")
    print(f"{'-' * 80}")
    print(f"{'Stage':<10}{SEP}{'AP':>8}{SEP}{'AP50':>8}{SEP}{'AP75':>8}{SEP}{'F1@50':>8}{SEP}{'cMSE':>8}{SEP}{'B-AP':>8}{SEP}{'SepRate':>8}")
    print(f"{'-' * 80}")
    for stage, m in stage_results.items():
        ap_iou = m.get('ap_iou', {})
        ap50 = ap_iou.get(50, float('nan'))
        ap75 = ap_iou.get(75, float('nan'))
        sep, total = m.get('b_sep_detail', (0, 0))
        sep_str = f"{m['b_sep_rate']:.3f}" if total > 0 and not np.isnan(m['b_sep_rate']) else "N/A"
        print(f"{stage:<10}{SEP}{_fmt(m['ap'])}{SEP}{_fmt(ap50)}{SEP}"
              f"{_fmt(ap75)}{SEP}{_fmt(m.get('f1'))}{SEP}{_fmt(m['cmse'])}"
              f"{SEP}{_fmt(m['b_ap'])}{SEP}{sep_str:>8}")
    print(f"{'=' * 80}")


#                                                                                              
#    
#                                                                                              
def main():
    parser = argparse.ArgumentParser(description='Instance segmentation evaluation: AP, F1, SNR groups, cMSE, and binary-star metrics.')
    parser.description = 'Evaluate MAINet instance segmentation checkpoints.'
    parser.epilog = CLI_USAGE
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument('--data', default='output',
                        help='Dataset root containing annotations/ and split folders. Default: output')
    parser.add_argument('--split', default='test',
                        help='Dataset split to evaluate: train, val, or test. Default: test')
    parser.add_argument('--score_thr', type=float, default=0.3,
                        help='Score threshold used for F1, cMSE, and binary-star separation metrics. Default: 0.3')
    parser.add_argument('--model', default=None,
                        help='Model name or checkpoint directory. Examples: v4, v3, work_dirs/real_mixed_baselines/mainet_v4')
    parser.add_argument('--ablation', default=None,
                        help='Evaluate one ablation under --model, e.g. strip; use all for every ablation')
    parser.add_argument('--all', action='store_true',
                        help='Scan and evaluate all discovered checkpoints')
    parser.add_argument('--real-only', action='store_true',
                        help='Evaluate only work_dirs/real_mixed_baselines models')
    parser.add_argument('--num', type=int, default=0,
                        help='Evaluate the first N images only. 0 means all images.')
    parser.add_argument('--eval-bs', type=int, default=4,
                        help='Batch size for local MAINet evaluation. MMDetection still runs per image.')
    parser.add_argument('--stages', action='store_true',
                        help='Evaluate every epoch checkpoint in the selected directory.')
    parser.add_argument('--force', action='store_true',
                        help='Re-evaluate models even if their names already exist in the output JSON.')
    help_overrides = {
        'score_thr': 'Score threshold used for F1, cMSE, and binary-star separation metrics. Default: 0.3',
        'model': 'Model name or checkpoint directory. Default: v4. Examples: v4, v3, work_dirs/mainet/v4',
        'ablation': 'Ablation key under --model, e.g. strip; use all for every ablation. Default: disabled',
        'all': 'Scan and evaluate all discovered checkpoints instead of only the main v4 model.',
        'real_only': 'Evaluate only checkpoints under work_dirs/real_mixed_baselines.',
        'num': 'Evaluate at most N images. 0 means all images. Default: 0',
        'eval_bs': 'Batch size for local MAINet inference. Default: 4',
        'stages': 'Evaluate every epoch checkpoint in the selected directory.',
        'force': 'Recompute metrics even if cached results exist.',
    }
    for action in parser._actions:
        if action.dest in help_overrides:
            action.help = help_overrides[action.dest]
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    save_path = os.path.join(PROJECT_ROOT, 'work_dirs', 'comparison_results.json')

    annotation_path = os.path.abspath(
        os.path.join(args.data, 'annotations', f'{args.split}.json'))
    annotation_hash = None
    if os.path.exists(annotation_path):
        with open(annotation_path, 'rb') as handle:
            annotation_hash = hashlib.sha256(handle.read()).hexdigest()
    cache_meta = {
        'schema': EVAL_SCHEMA_VERSION,
        'data_root': os.path.abspath(args.data),
        'split': args.split,
        'annotation_sha256': annotation_hash,
        'score_thr': float(args.score_thr),
        'f1_iou_thr': F1_IOU_THRESHOLD,
        'snr_bins': [[label, lower, None if np.isinf(upper) else upper]
                     for label, lower, upper in SNR_BINS],
        'snr_definition': SNR_DEFINITION,
        'snr_local_padding': SNR_LOCAL_PADDING,
        'snr_matching': 'hungarian_mask_iou_0.50',
        'centroid_mse_reference': 'gt_mask_centroid',
    }

    # Cached metrics are valid only for the same evaluator semantics and GT.
    existing = {}
    if os.path.exists(save_path):
        with open(save_path, encoding='utf-8') as f:
            cached = json.load(f)
        if cached.get('__meta__') == cache_meta:
            existing = {k: v for k, v in cached.items() if k != '__meta__'}
        elif not args.force:
            print('Cached metrics use different data/evaluator semantics; recomputing.')

    if args.real_only and args.model is None:
        args.all = True

    if args.all:
        args.model = None
    elif args.model is None:
        args.model = 'v4'

    if args.ablation:
        base = args.model or 'v4'
        ablation_path = os.path.join(PROJECT_ROOT, 'mainet', base, 'ablation.py')
        if not os.path.exists(ablation_path):
            print(f"Ablation config not found: {ablation_path}")
            sys.exit(1)

        import importlib.util
        spec = importlib.util.spec_from_file_location(f'ablation_{base}', ablation_path)
        abl_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(abl_mod)

        keys = list(abl_mod.ABLATIONS.keys()) if args.ablation == 'all' else [args.ablation]
        results = {}
        for key in keys:
            if key not in abl_mod.ABLATIONS:
                print(f"Unknown {base} ablation: {key}")
                print(f"Available: {', '.join(abl_mod.ABLATIONS.keys())}")
                sys.exit(1)

            if hasattr(abl_mod, 'ablation_checkpoint_dir') and base == 'v4':
                dpath = abl_mod.ablation_checkpoint_dir(key, debug=False)
            else:
                dpath = os.path.join(PROJECT_ROOT, 'work_dirs', 'mainet', base, key)
            legacy = os.path.join(PROJECT_ROOT, 'work_dirs', 'mainet', f'{base}_{key}')
            if not _find_best_ckpt(dpath) and _find_best_ckpt(legacy):
                dpath = legacy
            if not _find_best_ckpt(dpath):
                print(f"Checkpoint not found: {dpath}")
                continue

            name = f'{base.upper()}-{key}'
            ckpt = _find_best_ckpt(dpath)
            print(f"Evaluate {name} (split={args.split}, score_thr={args.score_thr}) ...")
            print(f"  ckpt={ckpt}")
            model, model_type = _load_local_model(dpath, device)
            results[name] = evaluate_model(model, model_type, args.data, args.split,
                                           device, args.score_thr, args.num,
                                           eval_batch_size=args.eval_bs)

        if results:
            print_table(results)
    elif args.model:
        model_arg = args.model
        if os.path.isdir(model_arg):
            dpath = model_arg
        elif os.path.isdir(os.path.join(PROJECT_ROOT, model_arg)):
            dpath = os.path.join(PROJECT_ROOT, model_arg)
        else:
            #          
            candidates = [
                os.path.join(PROJECT_ROOT, 'work_dirs', 'mainet', model_arg),
                os.path.join(PROJECT_ROOT, 'work_dirs', 'mmdet', model_arg),
                os.path.join(PROJECT_ROOT, 'work_dirs', 'real_mixed_baselines', model_arg),
                os.path.join(PROJECT_ROOT, 'work_dirs', model_arg),
            ]
            dpath = None
            for c in candidates:
                if os.path.isdir(c):
                    dpath = c
                    break
            if dpath is None:
                print(f"Model path not found: {model_arg}")
                print(f"Tried: {candidates}")
                sys.exit(1)

        ckpt = _find_best_ckpt(dpath)
        if ckpt is None:
            print(f"Checkpoint not found: {dpath}")
            sys.exit(1)
        print(f"Checkpoint: {ckpt}")

        model_arg_norm = model_arg.replace('\\', '/')
        is_yolo_dir = os.path.exists(os.path.join(dpath, 'weights', 'best.pt'))
        if '/yolo/' in model_arg_norm or is_yolo_dir or os.path.basename(os.path.normpath(dpath)) == 'yolo':
            model_type_hint = 'yolo'
        elif 'mmdet/' in model_arg_norm:
            model_type_hint = 'mmdet'
        elif 'mainet' in model_arg_norm:
            model_type_hint = 'local'
        elif dpath.endswith('_star'):
            model_type_hint = 'mmdet'
        else:
            model_type_hint = 'local'

        dname = os.path.basename(os.path.normpath(dpath))

        if model_type_hint == 'mmdet':
            # Resolve mmdet config.
            cfg_candidates = _mmdet_config_candidates(dname, dpath)
            cfg = None
            for c in cfg_candidates:
                if os.path.exists(c):
                    cfg = c
                    break
            if cfg is None:
                print(f"MMDetection config not found. Tried: {cfg_candidates}")
                sys.exit(1)

        if args.stages:
            if model_type_hint == 'yolo':
                print('YOLO keeps best.pt and last.pt only; --stages is not supported.')
                sys.exit(1)
            all_ckpts = _find_all_ckpts(dpath)
            if not all_ckpts:
                print(f"No epoch checkpoint found: {dpath}")
                sys.exit(1)

            print(f"\nEvaluate stages for {dname} ({len(all_ckpts)} checkpoints)")
            stage_results = {}
            for ck in all_ckpts:
                ep = ck['epoch']
                ck_path = ck['path']
                if model_type_hint == 'mmdet':
                    model, model_type = _load_mmdet_model(cfg, dpath, device)
                else:
                    model, model_type = _load_local_model(dpath, device)

                #     checkpoint    
                ckpt_data = torch.load(ck_path, map_location=device, weights_only=False)
                sd = ckpt_data.get('model_state_dict', ckpt_data)
                model.load_state_dict(sd, strict=False)
                model.eval()

                label = f'E{ep}'
                print(f"  epoch {ep} ...", end=' ')
                m = evaluate_model(model, model_type, args.data, args.split, device,
                                   args.score_thr, args.num, eval_batch_size=args.eval_bs)
                stage_results[label] = m
                print(f"AP={m['ap']:.4f} F1={m['f1']:.4f} cMSE={m['cmse']}")

            print_stage_table(dname, stage_results)
            results = {f'{dname}': stage_results}
        else:
            # Evaluate one selected checkpoint.
            if model_type_hint == 'mmdet':
                model, model_type = _load_mmdet_model(cfg, dpath, device)
                name = _make_model_name(dname, 'mmdet')
            elif model_type_hint == 'yolo':
                model, model_type = _load_yolo_model(dpath, device)
                name = yolo_model_display_name(model)
            else:
                model, model_type = _load_local_model(dpath, device)
                name = _make_model_name(dname, 'local')

            print(f"    {name} (split={args.split}, score_thr={args.score_thr}) ...")
            m = evaluate_model(model, model_type, args.data, args.split, device,
                               args.score_thr, args.num, eval_batch_size=args.eval_bs)
            results = {name: m}
            print_table(results)
    else:
        models = discover_models()
        if args.real_only:
            models = [m for m in models if 'real_mixed_baselines' in m[3].replace('\\', '/')]
        if not models:
            print("No checkpoints found.")
            sys.exit(1)

        print(f"\nDiscovered {len(models)} models:")
        for name, typ, _, dpath in models:
            status = " (cached)" if name in existing and not args.force else ""
            ckpt = _find_best_ckpt(dpath)
            print(f"  [{typ}] {name} -> {dpath}{status}")
            print(f"        ckpt={ckpt}")

        results = {}
        skipped = 0
        for name, typ, cfg, dpath in models:
            if name in existing and not args.force:
                print(f"\n>>> Skip {name} (cached; use --force to recompute)")
                results[name] = existing[name]
                skipped += 1
                continue

            ckpt = _find_best_ckpt(dpath)
            print(f"\n>>> Evaluate {name} ...")
            print(f"    ckpt={ckpt}")
            if typ == 'mmdet' and not _mmdet_available:
                print("  Skip: mmdet is not available")
                skipped += 1
                continue
            if typ == 'yolo' and not _yolo_available:
                print("  Skip: ultralytics is not available")
                skipped += 1
                continue
            try:
                if typ == 'mmdet':
                    model, model_type = _load_mmdet_model(cfg, dpath, device)
                elif typ == 'yolo':
                    model, model_type = _load_yolo_model(dpath, device)
                else:
                    model, model_type = _load_local_model(dpath, device)
                result_name = (f'{yolo_model_display_name(model)}-real'
                               if typ == 'yolo' else name)
                results[result_name] = evaluate_model(model, model_type, args.data, args.split,
                                               device, args.score_thr, args.num)
            except Exception as e:
                print(f"  Failed: {e}")
                import traceback
                traceback.print_exc()

        if skipped:
            print(f"\nSkipped {skipped} cached/unavailable models.")

        if results:
            print_table(results)

    # Save/update cached results.
    def _serialize_result(m):
        """Convert result values into JSON-safe data."""
        # Stage-result dict.
        if any(isinstance(v, dict) and 'ap' in v for v in m.values()):
            return {k: _serialize_single(v) for k, v in m.items()}
        return _serialize_single(m)

    def _serialize_single(s):
        d = dict(s)
        sep, total = d.pop('b_sep_detail', (0, 0))
        d['b_sep'] = sep
        d['b_sep_total'] = total
        return d

    serializable = {}
    keep_existing_cache = not (args.force and (args.all or args.real_only))
    if keep_existing_cache and os.path.exists(save_path):
        with open(save_path, encoding='utf-8') as f:
            cached = json.load(f)
        if cached.get('__meta__') == cache_meta:
            serializable = cached
    serializable['__meta__'] = cache_meta
    for name, m in results.items():
        serializable[name] = _serialize_result(m)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved: {save_path}")


if __name__ == '__main__':
    main()


