"""
Convert existing real MAINet-format datasets to ZScale-stretched images.

This preserves annotations and masks. Existing images are copied to
images_raw/ before images/ is replaced with the ZScale [0, 1] version.

Examples:
  python zscale_real_dataset.py --data-root output_real_10
  python zscale_real_dataset.py --data-root mixed_point_streak_trainable --only-point
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


PREPROCESSED = {"real_zscale_0_1", "fits_zscale_0_1", "real_asinh_0_1", "bmp_asinh_0_1"}
_WARNED_FALLBACK = False


def zscale_stretch(img):
    global _WARNED_FALLBACK
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img, dtype=np.float32)
    vals = img[finite]
    from astropy.visualization import ZScaleInterval

    lo, hi = ZScaleInterval().get_limits(vals)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32)


def image_mode(image):
    return image.get("mode") or image.get("source_dataset") or image.get("target_type")


def source_image_path(data_root, split, image):
    fname = image.get("file_name", f"{int(image['id']):06d}.npy")
    raw_name = image.get("raw_file_name", fname)
    raw_path = data_root / split / "images_raw" / raw_name
    if raw_path.exists():
        return raw_path
    return data_root / split / "images" / fname


def convert_dataset(data_root, only_point=False, force=False):
    data_root = Path(data_root)
    total = 0
    converted = 0
    for split in ("train", "val", "test"):
        ann_path = data_root / "annotations" / f"{split}.json"
        if not ann_path.exists():
            continue
        data = json.loads(ann_path.read_text(encoding="utf-8-sig"))
        changed = False
        for image in data.get("images", []):
            if only_point and image_mode(image) != "point":
                continue
            fname = image.get("file_name", f"{int(image['id']):06d}.npy")
            img_path = data_root / split / "images" / fname
            if not img_path.exists():
                continue
            total += 1
            if image.get("preprocess") in PREPROCESSED and not force:
                continue
            src_path = source_image_path(data_root, split, image)
            raw_dir = data_root / split / "images_raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / image.get("raw_file_name", fname)
            if not raw_path.exists():
                shutil.copy2(img_path, raw_path)
            raw = np.load(src_path).astype(np.float32)
            np.save(img_path, zscale_stretch(raw))
            image["preprocess"] = "real_zscale_0_1"
            image["raw_file_name"] = raw_path.name
            converted += 1
            changed = True
        if changed:
            ann_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="ascii")
    return total, converted


def main():
    parser = argparse.ArgumentParser(description="ZScale existing real MAINet-format images without changing masks.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--only-point", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute even if preprocess is already set")
    args = parser.parse_args()
    total, converted = convert_dataset(args.data_root, only_point=args.only_point, force=args.force)
    print(f"checked={total} converted={converted} data_root={Path(args.data_root).resolve()}")


if __name__ == "__main__":
    main()
