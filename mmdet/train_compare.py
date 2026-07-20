"""Local MMDetection training helper for the project baseline wrappers.

This file is loaded by path from ``mmdet/<model>/train.py``. It intentionally is
not imported as ``mmdet.train_compare`` because that name collides with the
installed OpenMMLab ``mmdet`` package.
"""
# Notes: periodic checkpoints are disabled by the real-data configs. This
# helper preserves best + final recovery checkpoint and exposes 4080 speed flags.
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MMDET_DIR = os.path.join(ROOT, "mmdet")
if LOCAL_MMDET_DIR not in sys.path:
    sys.path.insert(0, LOCAL_MMDET_DIR)


def _set_nested(cfg, path, value):
    cur = cfg
    for key in path[:-1]:
        if key not in cur or cur[key] is None:
            return
        cur = cur[key]
    if path[-1] in cur:
        cur[path[-1]] = value


def _tune_schedulers(schedulers, max_epochs):
    if isinstance(schedulers, dict):
        if schedulers.get("type") == "CosineAnnealingLR":
            schedulers["T_max"] = max_epochs
        if "end" in schedulers:
            schedulers["end"] = max(1, max_epochs)
        return schedulers

    if not isinstance(schedulers, list):
        return schedulers

    tuned = []
    for sched in schedulers:
        begin = int(sched.get("begin", 0))
        if begin >= max_epochs:
            continue
        if "end" in sched:
            sched["end"] = max(begin + 1, min(int(sched["end"]), max_epochs))
        if sched.get("type") == "CosineAnnealingLR":
            sched["T_max"] = max(1, max_epochs - begin)
        tuned.append(sched)
    return tuned or None


def _apply_epoch_tuning(cfg, tune):
    max_epochs = int(getattr(tune, "max_epochs", 100))
    patience = int(getattr(tune, "patience", 10))
    num_workers = int(getattr(tune, "num_workers", 0))
    batch_size = getattr(tune, "mmdet_batch_size", None)
    fast = bool(getattr(tune, "fast", False))
    amp_dtype = getattr(tune, "amp_dtype", None)
    cfg.work_dir = getattr(tune, "work_dir", cfg.get("work_dir", None))

    if "train_cfg" in cfg and cfg.train_cfg is not None:
        cfg.train_cfg.max_epochs = max_epochs
        if max_epochs <= 1:
            cfg.train_cfg.val_interval = max_epochs + 1

    cfg.param_scheduler = _tune_schedulers(cfg.get("param_scheduler", None), max_epochs)

    for hook in cfg.get("custom_hooks", []):
        if hook.get("type") == "EarlyStoppingHook":
            hook["patience"] = patience

    # A negative interval means best/last-only checkpointing. Preserve it
    # instead of turning it back into periodic epoch checkpoints.
    checkpoint = cfg.get("default_hooks", {}).get("checkpoint", None)
    if checkpoint is not None and int(checkpoint.get("interval", -1)) > 0:
        checkpoint["interval"] = max(1, min(5, max_epochs))

    if batch_size is not None:
        cfg.train_dataloader["batch_size"] = int(batch_size)

    # Keep the Windows-safe default (0 workers), but allow the training host
    # to overlap npy loading with GPU execution.
    for loader_name in ["train_dataloader", "val_dataloader", "test_dataloader"]:
        loader = cfg.get(loader_name, None)
        if loader is not None:
            loader["num_workers"] = num_workers
            loader["persistent_workers"] = num_workers > 0
            if num_workers > 0:
                loader["pin_memory"] = True
                loader["prefetch_factor"] = 2

    if fast:
        cfg.setdefault("env_cfg", {})["cudnn_benchmark"] = True

    # BF16 is opt-in because the earlier FP16 Mask2Former run was unstable.
    if amp_dtype:
        cfg.optim_wrapper["type"] = "AmpOptimWrapper"
        cfg.optim_wrapper["dtype"] = amp_dtype
        cfg.optim_wrapper["loss_scale"] = "dynamic"


def train_one(config_path, tune, args):
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(config_path)
    _apply_epoch_tuning(cfg, tune)

    if bool(getattr(tune, "fast", False)) and torch.cuda.is_available():
        # TensorFloat-32 accelerates matrix products on Ampere/Ada GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    cfg.resume = bool(getattr(args, "resume", False))
    resume_from = getattr(args, "resume_from", None)
    if resume_from:
        cfg.load_from = resume_from
        cfg.resume = True

    runner = Runner.from_cfg(cfg)
    runner.train()
