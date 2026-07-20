"""
MAINet v4 - PSF-gated residual + Deformable Orientation Strip + residual fusion.
========================================================================== 

自包含训练脚本：依赖同目录下的 model.py / backbone.py / heads.py / dataset.py，
无需项目其他模块。默认路径基于脚本位置推算项目根。

用法:
  python mainet/v4/train.py --debug --epochs 1
  python mainet/v4/train.py --epochs 100

Checkpoint: best_model.pt is selected by validation segmentation mAP at
IoU=0.50:0.05:0.95, matching the MMDetection comparison protocol.

Architecture note: this version adds a geometry-aware image-level morphology
gate and near-collinear deformable strip sampling. Existing pre-change v4
checkpoints are architecture-incompatible; train this version from scratch.
"""
# Notes: MAINet-v4 writes only best_model.pt; eval.py selects it automatically.
# Resume is explicit: --resume <output-dir>/best_model.pt.
import os, sys, math, time, random, argparse, signal
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# 自包含导入：当前目录加入 sys.path，默认路径基于脚本位置推算项目根（向上 2 层）
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path: sys.path.insert(0, _HERE)

try: from torch.utils.tensorboard import SummaryWriter
except ImportError: SummaryWriter = None

os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
os.environ.setdefault('GLOG_minloglevel', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


# ═══════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════
class Config:
    model_type     = "rcnn_v4"
    architecture_version = "v4_geometry_gate_v2_collinear_strip"
    in_chans       = 1; num_classes = 1
    data_root      = os.path.join(_PROJECT_ROOT, "output")
    lr=2e-4; min_lr=1e-6; weight_decay=1e-4; batch_size=2; epochs=100
    grad_clip=1.0; warmup_epochs=5; param_decay_epochs=0
    w_class=1.0; w_mask=5.0; w_dice=5.0; w_param_init=0.1; no_obj_weight=0.1
    gate_loss_weight=0.1; gate_pos_weight=1.0
    cost_class=1.0; cost_mask=5.0; cost_dice=5.0
    patience=10; min_delta=1e-4
    output_dir = os.path.join(_PROJECT_ROOT, "work_dirs/mainet/v4")
    log_dir    = os.path.join(_PROJECT_ROOT, "runs")
    num_workers=4; use_amp=True; seed=42; clean_log=False


# ═══════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _worker_init(worker_id):
    base = torch.initial_seed() % 2**32
    np.random.seed(base + worker_id); random.seed(base + worker_id)

def compute_lr(epoch, cfg):
    if epoch < cfg.warmup_epochs: return cfg.lr*(epoch+1)/cfg.warmup_epochs
    p = min(1.0, (epoch-cfg.warmup_epochs)/max(1, cfg.epochs-cfg.warmup_epochs))
    return cfg.min_lr + 0.5*(cfg.lr-cfg.min_lr)*(1+math.cos(math.pi*p))

def set_lr(opt, lr):
    for pg in opt.param_groups: pg['lr'] = lr

def param_weight_scale(epoch, decay_epochs, floor=0.3):
    return floor if epoch >= decay_epochs else 1.0-(1.0-floor)*(epoch/decay_epochs)

def monitor_metric(v):
    return v.get('segm_mAP', 0.0)


COCO_IOU_THRESHOLDS = tuple(np.arange(0.50, 0.96, 0.05).tolist())


def _update_gate_stats(stats, losses):
    probs = losses.get('_gate_probs')
    targets = losses.get('_gate_targets')
    if probs is None or targets is None:
        return
    probs = probs.detach().float()
    targets = targets.detach().float()
    for name, keep in (('point', targets < 0.5),
                       ('streak', targets >= 0.5)):
        if keep.any():
            stats[f'{name}_sum'] += probs[keep].sum().item()
            stats[f'{name}_count'] += int(keep.sum().item())
    stats['correct'] += ((probs >= 0.5) == (targets >= 0.5)).sum().item()
    stats['count'] += int(targets.numel())


def _finalize_gate_stats(stats, output):
    for name in ('point', 'streak'):
        count = stats.get(f'{name}_count', 0)
        if count:
            output[f'gate_{name}_mean'] = stats[f'{name}_sum'] / count
    if stats.get('count', 0):
        output['gate_accuracy'] = stats['correct'] / stats['count']


def _mask_iou_matrix(pred_masks, gt_masks, chunk_size=32):
    """Compute a prediction-by-GT mask IoU matrix without CPU mask caches."""
    pred_masks = pred_masks > 0.5
    gt_masks = gt_masks > 0.5
    num_pred, num_gt = len(pred_masks), len(gt_masks)
    if num_pred == 0 or num_gt == 0:
        return np.zeros((num_pred, num_gt), dtype=np.float32)

    gt_flat = gt_masks.reshape(num_gt, -1).float()
    gt_area = gt_flat.sum(dim=1)
    chunks = []
    for start in range(0, num_pred, chunk_size):
        pred_flat = pred_masks[start:start + chunk_size].reshape(
            -1, gt_flat.shape[1]).float()
        intersection = pred_flat @ gt_flat.t()
        union = pred_flat.sum(dim=1, keepdim=True) + gt_area[None] - intersection
        chunks.append(intersection / union.clamp(min=1e-6))
    return torch.cat(chunks, dim=0).float().cpu().numpy()


def _segmentation_map(records, total_gt):
    """Compute one-class COCO-style 101-point mask mAP from IoU matrices."""
    if total_gt <= 0:
        return 0.0

    predictions = []
    for image_index, (scores, _) in enumerate(records):
        predictions.extend(
            (float(score), image_index, pred_index)
            for pred_index, score in enumerate(scores))
    predictions.sort(key=lambda item: item[0], reverse=True)
    if not predictions:
        return 0.0

    aps = []
    for threshold in COCO_IOU_THRESHOLDS:
        matched = [np.zeros(ious.shape[1], dtype=bool)
                   for _, ious in records]
        tp = np.zeros(len(predictions), dtype=np.float32)
        fp = np.zeros(len(predictions), dtype=np.float32)
        for index, (_, image_index, pred_index) in enumerate(predictions):
            ious = records[image_index][1][pred_index]
            if ious.size == 0:
                fp[index] = 1.0
                continue
            available = np.where(matched[image_index], -1.0, ious)
            gt_index = int(np.argmax(available))
            if available[gt_index] >= threshold:
                tp[index] = 1.0
                matched[image_index][gt_index] = True
            else:
                fp[index] = 1.0

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / max(total_gt, 1)
        precision = tp / np.maximum(tp + fp, 1e-12)
        interpolated = []
        for recall_level in np.linspace(0.0, 1.0, 101):
            keep = recall >= recall_level
            interpolated.append(float(precision[keep].max()) if keep.any() else 0.0)
        aps.append(float(np.mean(interpolated)))
    return float(np.mean(aps))


# ═══════════════════════════════════════════════════
# 数据
# ═══════════════════════════════════════════════════
def _dataset_modes(dataset):
    if isinstance(dataset, Subset):
        return [dataset.dataset.images[index].get('mode', 'point')
                for index in dataset.indices]
    return [image.get('mode', 'point') for image in dataset.images]


def _stratified_debug_subset(dataset, limit, seed):
    groups = defaultdict(list)
    for index, image in enumerate(dataset.images):
        groups[image.get('mode', 'point')].append(index)
    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)
    modes = [mode for mode in ('point', 'streak') if groups.get(mode)]
    if not modes:
        return Subset(dataset, range(min(limit, len(dataset))))
    selected = []
    per_mode = max(1, limit // len(modes))
    for mode in modes:
        selected.extend(groups[mode][:per_mode])
    if len(selected) < min(limit, len(dataset)):
        selected_set = set(selected)
        remaining = [index for indices in groups.values() for index in indices
                     if index not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[:limit - len(selected)])
    rng.shuffle(selected)
    return Subset(dataset, selected[:limit])


def build_dataloaders(cfg, debug=False):
    from dataset import MAINetDataset, collate_fn
    ds_train = MAINetDataset(f"{cfg.data_root}/annotations/train.json",
        f"{cfg.data_root}/train/images", f"{cfg.data_root}/train/masks", augment=True)
    ds_val = MAINetDataset(f"{cfg.data_root}/annotations/val.json",
        f"{cfg.data_root}/val/images", f"{cfg.data_root}/val/masks", augment=False)
    if debug:
        ds_train = _stratified_debug_subset(
            ds_train, min(50, len(ds_train)), cfg.seed)
        ds_val = _stratified_debug_subset(
            ds_val, min(20, len(ds_val)), cfg.seed + 1)
    train_modes = _dataset_modes(ds_train)
    point_count = train_modes.count('point')
    streak_count = train_modes.count('streak')
    if point_count == 0 or streak_count == 0:
        raise ValueError(
            'MAINet-v4 gate training requires both point and streak images; '
            f'found point={point_count}, streak={streak_count}.')
    cfg.gate_pos_weight = point_count / streak_count
    print(f"Gate labels | point={point_count} streak={streak_count} "
          f"pos_weight={cfg.gate_pos_weight:.3f}")
    g = torch.Generator(); g.manual_seed(cfg.seed)
    t0 = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True,
        worker_init_fn=_worker_init, generator=g)
    v0 = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available())
    return t0, v0


# ═══════════════════════════════════════════════════
# 模型
# ═══════════════════════════════════════════════════
def build_model(cfg, device):
    from model import MAINetRCNN, RCNNCriterion
    model = MAINetRCNN(in_chans=cfg.in_chans, num_classes=cfg.num_classes,
                       score_thr=0.3, nms_thr=0.5, max_per_img=200,
                       gate_pos_weight=cfg.gate_pos_weight,
                       gate_loss_weight=cfg.gate_loss_weight).to(device)
    criterion = RCNNCriterion(model)
    n = sum(p.numel() for p in model.parameters())
    print(f"Model (RCNN/v4 PSF-gate+DeformStrip+ResidualFusion): {n/1e6:.2f}M params")
    return model, criterion


def build_optimizer(cfg, model):
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


# ═══════════════════════════════════════════════════
# Checkpoint
# ═══════════════════════════════════════════════════
def save_ckpt(path, model, optimizer, scaler, epoch, best_monitor,
              patience_counter, global_step, cfg, extra=None):
    ckpt = {'epoch':epoch, 'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scaler':scaler.state_dict() if scaler else None,
            'best_monitor':best_monitor, 'patience_counter':patience_counter,
            'monitor_name':'segm_mAP', 'monitor_mode':'max',
            'global_step':global_step,
            'cfg_dict':{k:v for k,v in vars(cfg).items()
                        if not k.startswith('_') and not callable(v)}}
    if extra: ckpt.update(extra)
    torch.save(ckpt, path)


# ═══════════════════════════════════════════════════
# 训练 / 验证
# ═══════════════════════════════════════════════════
def train_one_epoch(model, criterion, loader, optimizer, device,
                    epoch, cfg, p_scale, scaler, writer, global_step):
    model.train(); meters = defaultdict(float); n_seen = 0
    gate_stats = defaultdict(float)
    use_bar = not getattr(cfg, 'clean_log', False)
    iterator = tqdm(loader, desc=f"Train {epoch+1:3d}/{cfg.epochs}", ncols=100,
                    ascii=True, dynamic_ncols=False, leave=False) if use_bar else loader
    for imgs, masks, wts, params, _ in iterator:
        imgs = imgs.to(device)
        masks = [m.to(device) for m in masks]; wts = [w.to(device) for w in wts]
        params = [{k:v.to(device) if isinstance(v,torch.Tensor) else v
                   for k,v in p.items()} if p else None for p in params]
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                losses = criterion(imgs, masks, wts, gt_params_list=params, param_weight_scale=p_scale)
            scaler.scale(losses['total']).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            losses = criterion(imgs, masks, wts, gt_params_list=params, param_weight_scale=p_scale)
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        _update_gate_stats(gate_stats, losses)
        for k,v in losses.items():
            if not k.startswith('_'):
                meters[k] += v.item()
        n_seen += 1; global_step += 1
        if use_bar:
            iterator.set_postfix({
                'loss': f"{meters['total']/n_seen:.3f}",
                'dice': f"{meters['dice']/n_seen:.3f}",
                'cls': f"{meters['class']/n_seen:.3f}",
                'reg': f"{meters.get('rpn_reg',0)/n_seen:.3f}",
                'bce': f"{meters.get('roi_mask',meters.get('bce',0))/n_seen:.4f}"})
    avg = {k:v/max(1,n_seen) for k,v in meters.items()}
    _finalize_gate_stats(gate_stats, avg)
    if writer:
        for k,v in avg.items(): writer.add_scalar(f'Train/{k}', v, epoch)
        writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)
    return avg, global_step


@torch.no_grad()
def validate(model, criterion, loader, device, epoch, cfg, p_scale, writer):
    model.eval(); model.train()
    meters = defaultdict(float); n_seen = 0
    gate_stats = defaultdict(float)
    map_records = []; total_gt = 0
    use_bar = not getattr(cfg, 'clean_log', False)
    iterator = tqdm(loader, desc=f"Val   {epoch+1:3d}/{cfg.epochs}", ncols=100,
                    ascii=True, dynamic_ncols=False, leave=False) if use_bar else loader
    for imgs, masks, wts, params, _ in iterator:
        imgs = imgs.to(device)
        masks = [m.to(device) for m in masks]; wts = [w.to(device) for w in wts]
        params = [{k:v.to(device) if isinstance(v,torch.Tensor) else v
                   for k,v in p.items()} if p else None for p in params]
        losses, predictions = criterion(
            imgs, masks, wts, gt_params_list=params,
            param_weight_scale=p_scale, return_predictions=True)
        _update_gate_stats(gate_stats, losses)
        for k,v in losses.items():
            if not k.startswith('_'):
                meters[k] += v.item()
        n_seen += 1
        for output, gt_masks in zip(predictions, masks):
            scores = output['scores']
            order = scores.argsort(descending=True)[:100]
            scores = scores[order]
            pred_masks = output['masks'][order]
            ious = _mask_iou_matrix(pred_masks, gt_masks)
            map_records.append((scores.float().cpu().numpy(), ious))
            total_gt += len(gt_masks)
        if use_bar:
            iterator.set_postfix({'loss': f"{meters['total']/n_seen:.3f}", 'dice': f"{meters['dice']/n_seen:.3f}"})
    avg = {k:v/max(1,n_seen) for k,v in meters.items()}
    _finalize_gate_stats(gate_stats, avg)
    avg['segm_mAP'] = _segmentation_map(map_records, total_gt)
    if writer:
        for k,v in avg.items(): writer.add_scalar(f'Val/{k}', v, epoch)
    return avg


# ═══════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════
def run_training(cfg, debug=False, resume=None, force_cpu=False):
    set_seed(cfg.seed)
    device = torch.device('cpu' if force_cpu else 'cuda')
    if device.type=='cuda' and not torch.cuda.is_available():
        print("⚠ CUDA unavailable"); device=torch.device('cpu')
    os.makedirs(cfg.output_dir, exist_ok=True); os.makedirs(cfg.log_dir, exist_ok=True)

    tl, vl = build_dataloaders(cfg, debug=debug)

    model, criterion = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    use_amp = cfg.use_amp and device.type=='cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"MAINet-v4 | device={device} | epochs={cfg.epochs} | batch={cfg.batch_size} | train={len(tl)} | val={len(vl)} | amp={'on' if use_amp else 'off'}")

    writer = SummaryWriter(os.path.join(cfg.log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))) if SummaryWriter else None

    start_epoch=0; best_monitor=float('-inf'); patience_counter=0; global_step=0
    if resume:
        print(f"Resuming: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scaler and ckpt.get('scaler'): scaler.load_state_dict(ckpt['scaler'])
        start_epoch=ckpt['epoch']+1
        if ckpt.get('monitor_name') == 'segm_mAP':
            best_monitor=ckpt.get('best_monitor',float('-inf'))
            patience_counter=ckpt.get('patience_counter',0)
        else:
            print('Resume checkpoint used the old loss monitor; resetting best mAP state.')
            best_monitor=float('-inf'); patience_counter=0
        global_step=ckpt.get('global_step',0)

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        lr = compute_lr(epoch, cfg); set_lr(optimizer, lr)
        p_scale = param_weight_scale(epoch, cfg.param_decay_epochs)
        tl_, global_step = train_one_epoch(model, criterion, tl, optimizer, device,
                                            epoch, cfg, p_scale, scaler, writer, global_step)
        vl_ = validate(model, criterion, vl, device, epoch, cfg, p_scale, writer)
        m = monitor_metric(vl_)
        print(f"epoch {epoch+1:03d}/{cfg.epochs:03d} | {time.time()-t0:.0f}s | lr={lr:.2e} | "
              f"train loss={tl_['total']:.3f} cls={tl_['class']:.3f} dice={tl_['dice']:.3f} "
              f"reg={tl_.get('rpn_reg',0):.3f} bce={tl_.get('roi_mask',0):.4f} "
              f"gate={tl_.get('param_gate',0):.3f} | val loss={vl_['total']:.3f} "
              f"mAP={m:.4f} gate[p={vl_.get('gate_point_mean',float('nan')):.3f},"
              f"s={vl_.get('gate_streak_mean',float('nan')):.3f},"
              f"acc={vl_.get('gate_accuracy',float('nan')):.3f}]")
        gate_gap = (vl_.get('gate_streak_mean', 0.0) -
                    vl_.get('gate_point_mean', 0.0))
        if epoch >= 2 and gate_gap < 0.10:
            print(f"WARNING gate separation is weak (streak-point={gate_gap:.3f}).")
        if m > best_monitor + cfg.min_delta:
            best_monitor=m; patience_counter=0
            save_ckpt(os.path.join(cfg.output_dir, 'best_model.pt'),
                      model, optimizer, scaler, epoch, best_monitor,
                      patience_counter, global_step, cfg, extra={'val_losses':dict(vl_)})
            print(f"best segm_mAP={m:.4f}")
        else:
            patience_counter += 1
        if patience_counter >= cfg.patience:
            print(f"\nEarly stop epoch {epoch+1} (best mAP={best_monitor:.4f})"); break
        if debug and epoch >= 20: print("\nDebug stop @20"); break
    if writer: writer.close()
    print(f"done | best mAP={best_monitor:.4f} | ckpt={cfg.output_dir}/best_model.pt")


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s,f: sys.exit(1))
    import multiprocessing; multiprocessing.freeze_support()
    try: multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass

    ap = argparse.ArgumentParser(description="MAINet v4 - PSF-gate + DeformStrip + ResidualFusion")
    for a in [('--epochs',int,None),('--batch_size',int,None),('--lr',float,None),
              ('--num_workers',int,None),('--patience',int,None)]:
        ap.add_argument(a[0], type=a[1], default=a[2])
    ap.add_argument('--debug', action='store_true'); ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--clean-log', action='store_true')
    ap.add_argument('--resume', type=str, default=None)
    ap.add_argument('--data-root', type=str, default=None, help='dataset root with annotations/train.json')
    ap.add_argument('--output-dir', type=str, default=None, help='checkpoint output directory')
    ap.add_argument('--log-dir', type=str, default=None, help='tensorboard log directory')
    args = ap.parse_args()

    cfg = Config()
    for k in ['epochs','batch_size','lr','num_workers','patience']:
        v = getattr(args, k)
        if v is not None: setattr(cfg, k, v)
    if args.data_root is not None: cfg.data_root = args.data_root
    if args.output_dir is not None: cfg.output_dir = args.output_dir
    if args.log_dir is not None: cfg.log_dir = args.log_dir
    cfg.clean_log = args.clean_log
    if not os.path.exists(f"{cfg.data_root}/annotations/train.json"):
        print("请先运行: python data/dataset_generator.py"); sys.exit(1)
    run_training(cfg, debug=args.debug, resume=args.resume, force_cpu=args.cpu)
