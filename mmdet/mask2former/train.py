"""
Mask2Former (mmdet baseline) — Transformer 通用分割

用法:
  python mmdet/mask2former/train.py                 # 全量训练
  python mmdet/mask2former/train.py --epochs 1      # 冒烟测试

输出: work_dirs/mmdet/mask2former/
"""
# Notes: mmdet keeps the validation-selected best checkpoint and one final
# recovery checkpoint. eval.py always evaluates the best checkpoint.
# RTX 4080 example: --batch-size 4 --num-workers 4 --fast --bf16
import sys, os, argparse
import importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from foreground_coco import prepare_foreground_coco

_TC_PATH = os.path.join(ROOT, "mmdet", "train_compare.py")
_spec = importlib.util.spec_from_file_location("local_train_compare", _TC_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
train_one = _mod.train_one

DEFAULT_DATA_ROOT = os.path.join(ROOT, "output_mix")
DEFAULT_WORK_ROOT = os.path.join(ROOT, "work_dirs/real_mixed_baselines/mmdet")

CONFIG = os.path.join(ROOT, "mmdet/configs_real_mixed/mask2former_star.py")


def make_runtime_config(template, data_root, work_dir, annotation_root=None):
    import re
    os.makedirs(work_dir, exist_ok=True)
    root = os.path.abspath(data_root).replace('\\', '/')
    text = open(template, 'r', encoding='utf-8').read()
    text = re.sub(r"data_root\s*=\s*r?['\"][^'\"]+['\"]", f"data_root=r'{root}'", text)
    annotation_root = (os.path.abspath(annotation_root).replace('\\', '/')
                       if annotation_root else f'{root}/annotations')
    text = re.sub(
        r"ann_file\s*=\s*r?['\"][^'\"]*annotations[\\/](train|val|test)\.json['\"]",
        lambda m: f"ann_file=r'{annotation_root}/{m.group(1)}.json'",
        text)
    out = os.path.abspath(work_dir).replace('\\', '/')
    text = re.sub(r"work_dir\s*=\s*r?['\"][^'\"]+['\"]", f"work_dir=r'{out}'", text)
    runtime = os.path.join(work_dir, '_runtime_config.py')
    with open(runtime, 'w', encoding='utf-8') as f:
        f.write(text)
    return runtime


class Tune:
    max_epochs = 100
    patience   = 10
    use_amp    = True
    work_dir   = os.path.join(ROOT, "work_dirs/real_mixed_baselines/mmdet/mask2former")
    mmdet_batch_size = None
    num_workers = 0
    fast = False
    amp_dtype = None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mask2Former (mmdet baseline)")
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--data-root', type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument('--annotation-root', type=str, default=None,
                    help='Optional derived COCO annotations; source data is read-only.')
    ap.add_argument('--work-dir', type=str, default=os.path.join(DEFAULT_WORK_ROOT, "mask2former"))
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    tune = Tune()
    tune.max_epochs = args.epochs
    tune.patience = args.patience
    tune.work_dir = args.work_dir
    tune.mmdet_batch_size = args.batch_size
    tune.num_workers = args.num_workers
    tune.fast = args.fast
    tune.amp_dtype = 'bfloat16' if args.bf16 else None
    annotation_root = args.annotation_root or prepare_foreground_coco(
        args.data_root, os.path.join(args.work_dir, '_prepared_annotations'))
    config = make_runtime_config(CONFIG, args.data_root, args.work_dir, annotation_root)

    class _A: pass
    train_args = _A()
    train_args.resume_from = None
    train_args.resume = args.resume

    train_one(config, tune, train_args)
