r"""
Sequential overnight training for the real mixed point/streak dataset.

Default dataset:
  <project root>\output_mix
Default outputs:
  <project root>\work_dirs\real_mixed_baselines

Examples:
指定路径模型debug
  python train_all_real_baselines.py --debug --models condinst mask2former --data-root output_mix --work-root work_dirs\output_mix_smallobj_debug
正式train
  python train_all_real_baselines.py --models condinst mask2former --epochs 100 --patience 30 --data-root output_mix --work-root work_dirs\output_mix_smallobj
    python train_all_real_baselines.py --epochs 100 --patience 10 --models mainet_v4
  python train_all_real_baselines.py --epochs 100 --patience 30 --models condinst mask2former
  python train_all_real_baselines.py --epochs 100 --models yolo --data-root output_mix
  python train_all_real_baselines.py --epochs 1 --models mainet_v4 mask_rcnn
  python train_all_real_baselines.py --debug
  python train_all_real_baselines.py --debug --compact-log
  python train_all_real_baselines.py --resume
"""
# Notes: mmdet models retain best + one final recovery checkpoint; eval.py
# always selects best. RTX 4080: add --fast --mask2former-bf16 after smoke test.
import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from foreground_coco import prepare_foreground_coco

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "output_mix"
DEFAULT_WORK_ROOT = ROOT / "work_dirs" / "real_mixed_baselines"
DEFAULT_MODELS = ["mainet_v4", "mask_rcnn", "condinst", "mask2former", "yolo"]
ALL_MODELS = DEFAULT_MODELS


def clear_selected_outputs(work_root, models):
    """Remove selected model outputs while keeping source data untouched."""
    work_root = Path(work_root).resolve()
    targets = {
        "mainet_v4": work_root / "mainet_v4",
        "mask_rcnn": work_root / "mmdet" / "mask_rcnn",
        "condinst": work_root / "mmdet" / "condinst",
        "mask2former": work_root / "mmdet" / "mask2former",
        "yolo": work_root / "yolo",
    }
    for model in models:
        target = targets[model].resolve()
        if work_root not in target.parents:
            raise RuntimeError(f"Refusing to remove path outside work root: {target}")
        if target.exists():
            shutil.rmtree(target)
            print(f"OVERWRITE removed {target}")

        log_path = (work_root / "logs" / f"{model}.log").resolve()
        if work_root not in log_path.parents:
            raise RuntimeError(f"Refusing to remove log outside work root: {log_path}")
        if log_path.exists():
            log_path.unlink()


def run_one(name, cmd, log_path, cwd, live_progress=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"START {name}")
    print(f"LOG {log_path}")
    print("=" * 80)
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n\n===== {datetime.now().isoformat(timespec='seconds')} START {name} =====\n")
        log.write(f"CMD {' '.join(cmd)}\n")
        log.flush()
        if live_progress:
            proc = subprocess.Popen(cmd, cwd=str(cwd))
            code = proc.wait()
            log.write(f"===== {datetime.now().isoformat(timespec='seconds')} END {name} code={code} =====\n")
            print(f"END {name}: code={code}")
            return code
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            if should_echo(line):
                print(line, end="")
            log.write(line)
        code = proc.wait()
        log.write(f"===== {datetime.now().isoformat(timespec='seconds')} END {name} code={code} =====\n")
    print(f"END {name}: code={code}")
    return code


def should_echo(line):
    keep_prefix = (
        "MAINet-v4",
        "epoch ",
        "best monitor",
        "done |",
        "END ",
    )
    keep_contains = (
        "mmengine - INFO - Epoch(",
        "mmengine - INFO - Saving checkpoint",
        "best checkpoint",
        "mmengine - WARNING",
        "Traceback",
        "Error",
        "Exception",
        "failed",
    )
    return line.startswith(keep_prefix) or any(k in line for k in keep_contains)

def build_commands(args, annotation_root=None):
    py = sys.executable
    work_root = Path(args.work_root).resolve()
    epochs = 1 if args.debug else args.epochs
    patience = 1 if args.debug else args.patience
    data_root = str(Path(args.data_root).resolve())
    commands = {}
    commands["mainet_v4"] = [py, "mainet/v4/train.py", "--epochs", str(epochs), "--data-root", data_root, "--output-dir", str(work_root / "mainet_v4"), "--log-dir", str(work_root / "mainet_v4" / "runs")]
    if args.compact_log:
        commands["mainet_v4"].append("--clean-log")
    if args.debug:
        commands["mainet_v4"].append("--debug")
    if args.batch_size is not None:
        commands["mainet_v4"] += ["--batch_size", str(args.batch_size)]
    elif args.debug:
        commands["mainet_v4"] += ["--batch_size", "1"]
    if args.num_workers is not None:
        commands["mainet_v4"] += ["--num_workers", str(args.num_workers)]
    elif args.debug:
        commands["mainet_v4"] += ["--num_workers", "0"]
    if args.resume:
        ckpt = work_root / "mainet_v4" / "best_model.pt"
        if ckpt.exists():
            commands["mainet_v4"] += ["--resume", str(ckpt)]
    for model in ["mask_rcnn", "condinst", "mask2former"]:
        commands[model] = [py, f"mmdet/{model}/train.py", "--epochs", str(epochs), "--data-root", data_root, "--work-dir", str(work_root / "mmdet" / model), "--patience", str(patience)]
        if annotation_root is not None:
            commands[model] += ["--annotation-root", str(annotation_root)]
        if model == "mask_rcnn":
            batch_size = args.mask_rcnn_batch_size
            if batch_size is None and args.fast:
                batch_size = 8
        elif model == "condinst":
            batch_size = args.condinst_batch_size
            if batch_size is None and args.fast:
                batch_size = 8
        elif model == "mask2former":
            batch_size = args.mask2former_batch_size
            if batch_size is None and args.fast:
                batch_size = 4
        else:
            batch_size = None
        if batch_size is not None:
            commands[model] += ["--batch-size", str(batch_size)]

        workers = args.mmdet_workers
        if workers is None and args.fast:
            workers = 4
        if workers is not None:
            commands[model] += ["--num-workers", str(workers)]
        if args.fast:
            commands[model].append("--fast")
        if model == "mask2former" and args.mask2former_bf16:
            commands[model].append("--bf16")
        if args.resume:
            commands[model].append("--resume")

    # YOLO owns a generated PNG/polygon cache below its work directory. The
    # project COCO+npy source data remains the single ground-truth authority.
    yolo_batch = args.yolo_batch_size
    if yolo_batch is None:
        yolo_batch = 1 if args.debug else (8 if args.fast else 4)
    commands["yolo"] = [
        py, "yolo/train.py", "--epochs", str(epochs), "--data-root", data_root,
        "--work-dir", str(work_root / "yolo"), "--patience", str(patience),
        "--batch-size", str(yolo_batch), "--num-workers",
        str(4 if args.fast else args.yolo_workers),
    ]
    if args.yolo_pretrained:
        commands["yolo"].append("--pretrained")
    if args.resume:
        commands["yolo"].append("--resume")
    return commands


def main():
    parser = argparse.ArgumentParser(description="Train MAINet v4 and MMDet baselines on the real mixed dataset.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None, help="MAINet v4 batch size only")
    parser.add_argument("--num-workers", type=int, default=None, help="MAINet v4 workers only")
    parser.add_argument("--mmdet-workers", type=int, default=None,
                        help="MMDetection DataLoader workers; default is 0 unless --fast is used")
    parser.add_argument("--mask-rcnn-batch-size", type=int, default=None)
    parser.add_argument("--condinst-batch-size", type=int, default=None)
    parser.add_argument("--mask2former-batch-size", type=int, default=None)
    parser.add_argument("--yolo-batch-size", type=int, default=None,
                        help="YOLOv8-seg batch size; default 4, or 1 in --debug")
    parser.add_argument("--yolo-workers", type=int, default=0,
                        help="YOLOv8-seg DataLoader workers; default 0 for Windows")
    parser.add_argument("--yolo-pretrained", action="store_true",
                        help="Use COCO-pretrained YOLO weights. Disabled by default for a fair comparison.")
    parser.add_argument("--fast", action="store_true",
                        help="Enable 4080-oriented MMDetection throughput settings")
    parser.add_argument("--mask2former-bf16", action="store_true",
                        help="Use BF16 AMP for Mask2Former; smoke-test one epoch first")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--debug", action="store_true", help="Smoke test: run every selected model for 1 epoch")
    parser.add_argument("--compact-log", action="store_true", help="Use compact filtered logs instead of live progress output")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--resume", action="store_true")
    mode_group.add_argument(
        "--overwrite", action="store_true",
        help="Delete outputs for selected models, then train them from scratch")
    parser.add_argument("--keep-going", action="store_true", default=True)
    args = parser.parse_args()
    if args.debug:
        args.work_root = str(Path(args.work_root) / "_debug")
    data_root = Path(args.data_root)
    train_json = data_root / "annotations" / "train.json"
    if not train_json.exists():
        raise FileNotFoundError(f"Missing dataset: {train_json}")
    print(f"DATA_ROOT {data_root.resolve()}")
    print(f"WORK_ROOT {Path(args.work_root).resolve()}")
    work_root = Path(args.work_root)
    if args.overwrite:
        clear_selected_outputs(work_root, args.models)
    mmdet_models = {"mask_rcnn", "condinst", "mask2former"}
    annotation_root = None
    if mmdet_models.intersection(args.models):
        annotation_root = prepare_foreground_coco(
            data_root, work_root / "_prepared_annotations")
        print(f"DERIVED_ANNOTATIONS {annotation_root.resolve()} (source data unchanged)")
    log_dir = work_root / "logs"
    commands = build_commands(args, annotation_root=annotation_root)
    failures = []
    for name in args.models:
        code = run_one(name, commands[name], log_dir / f"{name}.log", ROOT, live_progress=not args.compact_log)
        if code != 0:
            failures.append((name, code))
            if not args.keep_going:
                break
    print("\nSUMMARY")
    if failures:
        for name, code in failures:
            print(f"  {name}: failed/skipped ({code})")
        sys.exit(1)
    print("  all selected models finished")


if __name__ == "__main__":
    main()
