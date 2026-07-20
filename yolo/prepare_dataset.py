"""Export the project's COCO+npy star data to a cached YOLO-seg dataset.

The source dataset remains untouched.  Images are converted to RGB PNG only
inside the requested cache directory, while COCO RLE masks become one YOLO
polygon label per instance.  Each split list contains the exact source image
set, so old cache files cannot enter a later experiment accidentally.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from pycocotools import mask as mask_utils


PREPROCESSED = {
    'real_asinh_0_1', 'real_zscale_0_1', 'fits_zscale_0_1', 'bmp_asinh_0_1'
}
CLASS_NAMES = {0: 'star'}
EXPORT_VERSION = 2


def npy_to_yolo_rgb(image, image_info):
    """Convert a project npy image to the uint8 RGB representation YOLO sees."""
    image = np.asarray(image, dtype=np.float32)
    if image_info.get('preprocess') in PREPROCESSED:
        normalized = np.clip(np.where(np.isfinite(image), image, 0.0), 0.0, 1.0)
    else:
        finite = np.isfinite(image)
        if not finite.any():
            normalized = np.zeros_like(image, dtype=np.float32)
        else:
            values = image[finite]
            lo, hi = np.percentile(values, [1.0, 99.5])
            if hi <= lo:
                hi = lo + 1.0
            normalized = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
            normalized[~finite] = 0.0

    gray = np.rint(normalized * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def yolo_mask_nms_indices(masks, scores, iou_thr=0.50, max_det=50):
    """Return score-ordered indices after class-agnostic mask NMS.

    Ultralytics suppresses duplicate boxes, not duplicate segmentation masks.
    Every project image contains one morphology, so class-agnostic mask NMS is
    appropriate here.  The 0.50 threshold is safely above observed GT-mask
    overlaps while removing repeated predictions for one star.
    """
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=np.float32)
    if masks.ndim != 3 or len(masks) == 0:
        return np.zeros((0,), dtype=np.int64)

    order = np.argsort(-scores, kind='stable')
    flat_masks = masks.reshape(len(masks), -1)
    kept = []
    for candidate in order:
        if len(kept) >= max_det:
            break
        if kept:
            existing = flat_masks[np.asarray(kept, dtype=np.int64)]
            current = flat_masks[candidate]
            intersections = np.logical_and(existing, current).sum(axis=1)
            unions = np.logical_or(existing, current).sum(axis=1)
            ious = intersections / np.maximum(unions, 1)
            if np.any(ious >= iou_thr):
                continue
        kept.append(int(candidate))
    return np.asarray(kept, dtype=np.int64)


def yolo_model_display_name(model):
    """Return the architecture name embedded in an Ultralytics checkpoint."""
    yaml_cfg = getattr(getattr(model, 'model', None), 'yaml', {}) or {}
    yaml_file = yaml_cfg.get('yaml_file') if isinstance(yaml_cfg, dict) else None
    source = yaml_file or getattr(model, 'ckpt_path', None) or 'yolo-seg'
    name = Path(str(source)).stem
    if name.lower().startswith('yolo'):
        return 'YOLO' + name[4:]
    return name


def require_yolo_segmentation(model):
    """Reject detection-only checkpoints before mask evaluation/visualization."""
    task = str(getattr(model, 'task', '')).lower()
    if task != 'segment':
        raise ValueError(
            f'{yolo_model_display_name(model)} is a {task or "unknown"} checkpoint, '
            'not an instance-segmentation model. Use a *-seg.pt checkpoint.')


def _decode_mask(segmentation, height, width):
    """Decode the project's COCO RLE (or a compatible polygon) to a bitmap."""
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get('counts'), str):
            rle['counts'] = rle['counts'].encode('ascii')
        elif isinstance(rle.get('counts'), list):
            # COCO also permits uncompressed run lengths as a Python list.
            rle = mask_utils.frPyObjects(rle, height, width)
        return mask_utils.decode(rle).astype(np.uint8)
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        return mask_utils.decode(mask_utils.merge(rles)).astype(np.uint8)
    raise TypeError(f'Unsupported COCO segmentation type: {type(segmentation).__name__}')


def _mask_to_polygon(mask):
    """Return the largest exterior contour as a normalized YOLO polygon."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < 3:
        # A target clipped at an image edge can degenerate to one or two
        # pixels. YOLO requires at least three vertices, so retain it as the
        # smallest enclosing rectangle instead of silently dropping the GT.
        x, y, width, height = cv2.boundingRect(mask)
        contour = np.asarray([
            [x, y], [x + width, y], [x + width, y + height], [x, y + height]
        ], dtype=np.float32)
    height, width = mask.shape
    polygon = contour.astype(np.float32)
    polygon[:, 0] /= float(width)
    polygon[:, 1] /= float(height)
    return polygon.reshape(-1)


def _annotation_yolo_class(annotation, image_info):
    """Map every stellar instance to the sole foreground class."""
    return 0


def _load_coco(path):
    with open(path, encoding='utf-8-sig') as handle:
        return json.load(handle)


def _signature(data_root):
    signature = {'version': EXPORT_VERSION, 'data_root': str(data_root.resolve())}
    for split in ('train', 'val', 'test'):
        path = data_root / 'annotations' / f'{split}.json'
        stat = path.stat()
        signature[split] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    return signature


def _is_single_foreground_cache(cache_dir):
    """Return True only when YAML and every cached label use class zero."""
    data_yaml = cache_dir / 'data.yaml'
    if not data_yaml.exists():
        return False
    try:
        config = yaml.safe_load(data_yaml.read_text(encoding='utf-8')) or {}
    except (OSError, yaml.YAMLError):
        return False
    names = config.get('names')
    if names not in ({0: 'star'}, {'0': 'star'}, ['star']):
        return False

    for label_path in (cache_dir / 'labels').glob('*/*.txt'):
        for line in label_path.read_text(encoding='ascii').splitlines():
            fields = line.split()
            if fields and fields[0] != '0':
                return False
    return True


def _clear_generated_cache(cache_dir):
    """Delete only YOLO-derived files, never anything under source data."""
    for name in ('images', 'labels', 'splits'):
        path = cache_dir / name
        if path.exists():
            shutil.rmtree(path)
    for name in ('data.yaml', 'source_manifest.json'):
        path = cache_dir / name
        if path.exists():
            path.unlink()


def prepare_dataset(data_root, cache_dir, force=False):
    """Create or reuse a YOLO dataset cache and return its data.yaml path."""
    data_root = Path(data_root).resolve()
    cache_dir = Path(cache_dir).resolve()
    try:
        cache_dir.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise ValueError(f'YOLO cache must be outside source data_root: {cache_dir}')
    data_yaml = cache_dir / 'data.yaml'
    manifest_path = cache_dir / 'source_manifest.json'
    signature = _signature(data_root)

    if not force and data_yaml.exists() and manifest_path.exists():
        try:
            if (json.loads(manifest_path.read_text(encoding='utf-8')) == signature
                    and _is_single_foreground_cache(cache_dir)):
                return data_yaml
        except json.JSONDecodeError:
            pass

    _clear_generated_cache(cache_dir)

    converted = 0
    skipped_instances = 0
    for split in ('train', 'val', 'test'):
        coco = _load_coco(data_root / 'annotations' / f'{split}.json')
        images = {item['id']: item for item in coco.get('images', [])}
        anns_by_image = {image_id: [] for image_id in images}
        for ann in coco.get('annotations', []):
            if ann['image_id'] not in anns_by_image:
                continue
            anns_by_image[ann['image_id']].append(ann)

        image_dir = cache_dir / 'images' / split
        label_dir = cache_dir / 'labels' / split
        split_dir = cache_dir / 'splits'
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        split_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []

        for image_id, image_info in sorted(images.items()):
            source = data_root / split / 'images' / image_info['file_name']
            if not source.exists():
                raise FileNotFoundError(f'Missing source image: {source}')
            target_image = image_dir / f"{Path(image_info['file_name']).stem}.png"
            target_label = label_dir / f"{Path(image_info['file_name']).stem}.txt"

            image = np.load(source)
            rgb = npy_to_yolo_rgb(image, image_info)
            if not cv2.imwrite(str(target_image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f'Failed to write image: {target_image}')
            converted += 1

            lines = []
            for ann in anns_by_image[image_id]:
                mask = _decode_mask(
                    ann['segmentation'], int(image_info['height']), int(image_info['width']))
                polygon = _mask_to_polygon(mask)
                if polygon is None:
                    skipped_instances += 1
                    continue
                coords = ' '.join(f'{value:.6f}' for value in polygon)
                yolo_class = _annotation_yolo_class(ann, image_info)
                lines.append(f"{yolo_class} {coords}")
            target_label.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='ascii')
            image_paths.append(target_image.as_posix())

        (split_dir / f'{split}.txt').write_text(
            '\n'.join(image_paths) + ('\n' if image_paths else ''), encoding='utf-8')

    yaml_data = {
        'path': cache_dir.as_posix(),
        'train': 'splits/train.txt',
        'val': 'splits/val.txt',
        'test': 'splits/test.txt',
        'names': CLASS_NAMES,
    }
    data_yaml.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding='utf-8')
    manifest_path.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    if not _is_single_foreground_cache(cache_dir):
        raise RuntimeError(f'Invalid YOLO single-foreground cache: {cache_dir}')
    print(f'YOLO dataset ready | cache={cache_dir} | images={converted} | skipped_masks={skipped_instances}')
    return data_yaml


def main():
    parser = argparse.ArgumentParser(description='Export project COCO+npy data for YOLOv8-seg.')
    parser.add_argument('--data-root', default='output_mix')
    parser.add_argument('--cache-dir', default='work_dirs/real_mixed_baselines/yolo/dataset')
    parser.add_argument('--force', action='store_true', help='Rebuild cached PNG images and polygon labels.')
    args = parser.parse_args()
    prepare_dataset(args.data_root, args.cache_dir, force=args.force)


if __name__ == '__main__':
    main()
