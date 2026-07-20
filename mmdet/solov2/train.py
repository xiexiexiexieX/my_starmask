"""
SOLOv2 (mmdet baseline) — 单阶段实例分割

用法:
  python mmdet/solov2/train.py                 # 全量训练
  python mmdet/solov2/train.py --epochs 1      # 冒烟测试

输出: work_dirs/mmdet/solov2/
"""
import sys, os, argparse
import importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

_TC_PATH = os.path.join(ROOT, "mmdet", "train_compare.py")
_spec = importlib.util.spec_from_file_location("local_train_compare", _TC_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
train_one = _mod.train_one

DEFAULT_DATA_ROOT = os.path.join(ROOT, "mixed_point_streak_trainable")
DEFAULT_WORK_ROOT = os.path.join(ROOT, "work_dirs/real_mixed_baselines/mmdet")

CONFIG = os.path.join(ROOT, "mmdet/configs_real_mixed/solov2_star.py")


def make_runtime_config(template, data_root, work_dir):
    import re
    os.makedirs(work_dir, exist_ok=True)
    root = os.path.abspath(data_root).replace('\\', '/')
    text = open(template, 'r', encoding='utf-8').read()
    text = re.sub(r"data_root\s*=\s*['\"][^'\"]+['\"]", f"data_root = r'{root}'", text, count=1)
    text = re.sub(r"data_root=['\"][^'\"]+['\"]", f"data_root=r'{root}'", text)
    text = re.sub(r"ann_file=['\"][^'\"]*annotations/(train|val|test)\.json['\"]", lambda m: f"ann_file=r'{root}/annotations/{m.group(1)}.json'", text)
    out = os.path.abspath(work_dir).replace('\\', '/')
    text = re.sub(r"work_dir\s*=\s*['\"][^'\"]+['\"]", f"work_dir = r'{out}'", text)
    runtime = os.path.join(work_dir, '_runtime_config.py')
    with open(runtime, 'w', encoding='utf-8') as f:
        f.write(text)
    return runtime


class Tune:
    max_epochs = 100
    patience   = 10
    use_amp    = True
    work_dir   = os.path.join(ROOT, "work_dirs/real_mixed_baselines/mmdet/solov2")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SOLOv2 (mmdet baseline)")
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--data-root', type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument('--work-dir', type=str, default=os.path.join(DEFAULT_WORK_ROOT, "solov2"))
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    tune = Tune()
    tune.max_epochs = args.epochs
    tune.patience = args.patience
    tune.work_dir = args.work_dir
    config = make_runtime_config(CONFIG, args.data_root, args.work_dir)

    class _A: pass
    train_args = _A()
    train_args.resume_from = None
    train_args.resume = args.resume

    train_one(config, tune, train_args)
