"""Train the YOLOv8-seg baseline as one foreground star class.

Examples:
  python yolo/train.py --epochs 100 --data-root output_mix
  python yolo/train.py --epochs 1 --data-root output_mix --batch-size 1

The default model starts from yolov8n-seg.yaml without COCO weights.  Passing
--pretrained deliberately changes that protocol and downloads/uses YOLO COCO
weights, so do not mix the two result types in one comparison table.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo.prepare_dataset import prepare_dataset


DEFAULT_DATA_ROOT = ROOT / 'output_mix'
DEFAULT_WORK_DIR = ROOT / 'work_dirs' / 'real_mixed_baselines' / 'yolo'


def main():
    parser = argparse.ArgumentParser(description='YOLOv8n-seg star baseline')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data-root', default=str(DEFAULT_DATA_ROOT))
    parser.add_argument('--work-dir', default=str(DEFAULT_WORK_DIR))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--rebuild-data', action='store_true')
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            'Ultralytics is not installed. Run: python -m pip install ultralytics==8.3.0') from exc

    work_dir = Path(args.work_dir).resolve()
    data_yaml = prepare_dataset(
        args.data_root, work_dir / 'dataset', force=args.rebuild_data)
    best_path = work_dir / 'weights' / 'best.pt'
    last_path = work_dir / 'weights' / 'last.pt'

    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f'No YOLO last checkpoint to resume: {last_path}')
        model = YOLO(str(last_path))
    else:
        model = YOLO('yolov8n-seg.pt' if args.pretrained else 'yolov8n-seg.yaml')

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(
        f'YOLOv8n-seg | device={device} | epochs={args.epochs} | '
        f'batch={args.batch_size} | imgsz=1024 | pretrained={args.pretrained} | '
        f'overlap_mask=False')
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=1024,
        batch=args.batch_size,
        workers=args.num_workers,
        patience=args.patience,
        device=device,
        project=str(work_dir.parent),
        name=work_dir.name,
        exist_ok=True,
        resume=args.resume,
        pretrained=args.pretrained,
        seed=42,
        deterministic=True,
        # Morphology is evaluated from image metadata, not learned as a class.
        # Keep this explicit for compatibility across Ultralytics versions.
        single_cls=True,
        optimizer='auto',
        cos_lr=True,
        overlap_mask=False,
        mask_ratio=4,
        fliplr=0.5,
        flipud=0.5,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        iou=0.80,
        save=True,
        save_period=-1,
        plots=True,
    )
    print(f'done | best={best_path if best_path.exists() else "not created"}')


if __name__ == '__main__':
    main()
