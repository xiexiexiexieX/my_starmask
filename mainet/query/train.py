"""
Query-based MAINet（DETR 风格，Hungarian 匹配 + SetCriterion）
=============================================================

依赖项目根模块 mainet.query_head（需项目根在 sys.path），不能脱离项目目录单独运行。
默认路径基于脚本位置推算项目根。

用法:
  python mainet/query/train.py --debug --epochs 1            # 冒烟测试
  python mainet/query/train.py --epochs 100                  # 全量训练
"""
import os, sys, math, time, random, argparse, signal
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

try: from torch.utils.tensorboard import SummaryWriter
except ImportError: SummaryWriter = None

os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
os.environ.setdefault('GLOG_minloglevel', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


class Config:
    model_type     = "query"
    in_chans       = 1; num_classes = 1; num_queries = 100; d_model = 128
    data_root      = os.path.join(ROOT, "output")
    lr=1e-4; min_lr=1e-6; weight_decay=1e-4; batch_size=4; epochs=100
    grad_clip=1.0; warmup_epochs=5; param_decay_epochs=0
    w_class=1.0; w_mask=5.0; w_dice=5.0; w_param_init=0.1; no_obj_weight=0.1
    cost_class=1.0; cost_mask=5.0; cost_dice=5.0
    patience=10; min_delta=1e-4
    output_dir = os.path.join(ROOT, "work_dirs/mainet/query")
    log_dir    = os.path.join(ROOT, "runs")
    num_workers=4; use_amp=True; seed=42


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
    return floor if epoch >= decay_epochs else 1.0 - (1.0-floor)*(epoch/decay_epochs)

def monitor_metric(v):
    return v.get('dice',0)+v.get('bce',0)


def build_dataloaders(cfg, debug=False):
    from data.dataset import MAINetDataset, collate_fn
    ds_train = MAINetDataset(f"{cfg.data_root}/annotations/train.json",
        f"{cfg.data_root}/train/images", f"{cfg.data_root}/train/masks", augment=True)
    ds_val = MAINetDataset(f"{cfg.data_root}/annotations/val.json",
        f"{cfg.data_root}/val/images", f"{cfg.data_root}/val/masks", augment=False)
    if debug:
        ds_train = torch.utils.data.Subset(ds_train, range(min(50,len(ds_train))))
        ds_val   = torch.utils.data.Subset(ds_val, range(min(20,len(ds_val))))
    g = torch.Generator(); g.manual_seed(cfg.seed)
    t0 = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True,
        worker_init_fn=_worker_init, generator=g)
    v0 = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available())
    return t0, v0


def build_model(cfg, device):
    from mainet.query_head.mainet import MAINet
    from mainet.query_head.criterion import HungarianMatcher, SetCriterion
    model = MAINet(in_chans=cfg.in_chans, num_queries=cfg.num_queries,
                   num_classes=cfg.num_classes, d_model=cfg.d_model).to(device)
    matcher = HungarianMatcher(cost_class=cfg.cost_class, cost_mask=cfg.cost_mask,
                               cost_dice=cfg.cost_dice)
    criterion = SetCriterion(matcher, num_classes=cfg.num_classes,
                             w_class=cfg.w_class, w_mask=cfg.w_mask, w_dice=cfg.w_dice,
                             w_param=cfg.w_param_init, no_obj_weight=cfg.no_obj_weight)
    print(f"Model (Query/MAINet): {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    return model, criterion

def build_optimizer(cfg, model):
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def save_ckpt(path, model, optimizer, scaler, epoch, best_monitor,
              patience_counter, global_step, cfg, extra=None):
    ckpt = {'epoch':epoch, 'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scaler':scaler.state_dict() if scaler else None,
            'best_monitor':best_monitor, 'patience_counter':patience_counter,
            'global_step':global_step,
            'cfg_dict':{k:v for k,v in vars(cfg).items()
                        if not k.startswith('_') and not callable(v)}}
    if extra: ckpt.update(extra)
    torch.save(ckpt, path)


def train_one_epoch(model, criterion, loader, optimizer, device,
                    epoch, cfg, p_scale, scaler, writer, global_step):
    model.train(); meters = defaultdict(float); n_seen = 0
    pbar = tqdm(loader, desc=f"Train {epoch+1:3d}/{cfg.epochs}", ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    for imgs, masks, wts, params, _ in pbar:
        imgs = imgs.to(device)
        masks = [m.to(device) for m in masks]; wts = [w.to(device) for w in wts]
        params = [{k:v.to(device) if isinstance(v,torch.Tensor) else v
                   for k,v in p.items()} if p else None for p in params]
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                losses = criterion(outputs, masks, wts, params, param_weight_scale=p_scale)
            scaler.scale(losses['total']).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            outputs = model(imgs)
            losses = criterion(outputs, masks, wts, params, param_weight_scale=p_scale)
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        for k,v in losses.items(): meters[k] += v.item()
        n_seen += 1; global_step += 1
        pbar.set_postfix({
            'loss':f"{meters['total']/n_seen:.3f}", 'dice':f"{meters['dice']/n_seen:.3f}",
            'cls':f"{meters['class']/n_seen:.3f}", 'noobj':f"{meters['no_obj']/n_seen:.4f}"})
    avg = {k:v/max(1,n_seen) for k,v in meters.items()}
    if writer:
        for k,v in avg.items(): writer.add_scalar(f'Train/{k}', v, epoch)
        writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)
    return avg, global_step


@torch.no_grad()
def validate(model, criterion, loader, device, epoch, cfg, p_scale, writer):
    model.eval()
    meters = defaultdict(float); n_seen = 0
    pbar = tqdm(loader, desc=f"Val   {epoch+1:3d}/{cfg.epochs}", ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    for imgs, masks, wts, params, _ in pbar:
        imgs = imgs.to(device)
        masks = [m.to(device) for m in masks]; wts = [w.to(device) for w in wts]
        params = [{k:v.to(device) if isinstance(v,torch.Tensor) else v
                   for k,v in p.items()} if p else None for p in params]
        outputs = model(imgs)
        losses = criterion(outputs, masks, wts, params, param_weight_scale=p_scale)
        for k,v in losses.items(): meters[k] += v.item()
        n_seen += 1
        pbar.set_postfix({'loss':f"{meters['total']/n_seen:.3f}", 'dice':f"{meters['dice']/n_seen:.3f}"})
    avg = {k:v/max(1,n_seen) for k,v in meters.items()}
    if writer:
        for k,v in avg.items(): writer.add_scalar(f'Val/{k}', v, epoch)
    return avg


def run_training(cfg, debug=False, resume=None, force_cpu=False):
    set_seed(cfg.seed)
    device = torch.device('cpu' if force_cpu else 'cuda')
    if device.type=='cuda' and not torch.cuda.is_available():
        print("⚠ CUDA unavailable, fallback CPU"); device=torch.device('cpu')
    print(f"Device: {device} | Debug: {debug} | Batch: {cfg.batch_size} | Epochs: {cfg.epochs}")
    os.makedirs(cfg.output_dir, exist_ok=True); os.makedirs(cfg.log_dir, exist_ok=True)

    print("\nLoading data...")
    tl, vl = build_dataloaders(cfg, debug=debug)
    print(f"Train: {len(tl)} batches, Val: {len(vl)} batches")

    print("\nBuilding model...")
    model, criterion = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    use_amp = cfg.use_amp and device.type=='cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"AMP: {'on' if use_amp else 'off'}")

    writer = SummaryWriter(os.path.join(cfg.log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))) if SummaryWriter else None

    start_epoch=0; best_monitor=float('inf'); patience_counter=0; global_step=0
    if resume:
        print(f"Resuming: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scaler and ckpt.get('scaler'): scaler.load_state_dict(ckpt['scaler'])
        start_epoch=ckpt['epoch']+1; best_monitor=ckpt.get('best_monitor',float('inf'))
        patience_counter=ckpt.get('patience_counter',0); global_step=ckpt.get('global_step',0)

    print(f"\n{'='*60}\nTraining {cfg.epochs} epochs\n{'='*60}")
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        lr = compute_lr(epoch, cfg); set_lr(optimizer, lr)
        p_scale = param_weight_scale(epoch, cfg.param_decay_epochs)
        tl_, global_step = train_one_epoch(model, criterion, tl, optimizer, device,
                                            epoch, cfg, p_scale, scaler, writer, global_step)
        vl_ = validate(model, criterion, vl, device, epoch, cfg, p_scale, writer)
        m = monitor_metric(vl_)
        print(f"  {time.time()-t0:.0f}s | LR={lr:.2e} | "
              f"T total={tl_['total']:.3f} cls={tl_['class']:.3f} dice={tl_['dice']:.3f} | "
              f"V total={vl_['total']:.3f} dice={vl_['dice']:.3f} | mon={m:.4f}")
        if m < best_monitor - cfg.min_delta:
            best_monitor=m; patience_counter=0
            save_ckpt(os.path.join(cfg.output_dir, 'best_model.pt'),
                      model, optimizer, scaler, epoch, best_monitor,
                      patience_counter, global_step, cfg, extra={'val_losses':dict(vl_)})
            print(f"  → Best (monitor={m:.4f})")
        else:
            patience_counter += 1
        if patience_counter >= cfg.patience:
            print(f"\nEarly stop epoch {epoch+1} (best={best_monitor:.4f})"); break
        if debug and epoch >= 20: print("\nDebug stop @20"); break
    if writer: writer.close()
    print(f"\nDone. Best: {best_monitor:.4f}\nModel: {cfg.output_dir}/best_model.pt")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s,f: sys.exit(1))
    import multiprocessing; multiprocessing.freeze_support()
    try: multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass

    ap = argparse.ArgumentParser(description="Query-based MAINet (DETR-style)")
    for a in [('--epochs',int,None),('--batch_size',int,None),('--lr',float,None),
              ('--num_workers',int,None),('--patience',int,None)]:
        ap.add_argument(a[0], type=a[1], default=a[2])
    ap.add_argument('--debug', action='store_true'); ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--resume', type=str, default=None)
    args = ap.parse_args()

    cfg = Config()
    for k in ['epochs','batch_size','lr','num_workers','patience']:
        v = getattr(args, k)
        if v is not None: setattr(cfg, k, v)
    if not os.path.exists(f"{cfg.data_root}/annotations/train.json"):
        print("请先运行: python data/dataset_generator.py"); sys.exit(1)
    run_training(cfg, debug=args.debug, resume=args.resume, force_cpu=args.cpu)
