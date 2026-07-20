"""
ablation.py — MAINet v3 消融实验
=================================

对 v3 三个核心模块逐一消融:
  A. MoffatPSF: 有无 / 尺度数 / 核类型
  B. MultiOrientStrip: 有无 / 方向数
  C. SKFusion: 有无 / 替代融合方式

用法:
  python mainet/v3/ablation.py --list                        # 列出所有变体
  python mainet/v3/ablation.py --ablation full --epochs 100  # 跑单个
  python mainet/v3/ablation.py --all --epochs 100            # 跑全部
  python mainet/v3/ablation.py --all --epochs 100 --eval-only # 仅评估已有 ckpt
  python mainet/v3/ablation.py --all --debug --epochs 1      # 冒烟测试
"""

import os, sys, math, time, random, argparse
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path: sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── v3 原始模块 ──
from backbone import (Stem, MoffatPSFChannel, MultiOrientStripChannel,
                      SKFusion, SingleBlock)


# ══════════════════════════════════════════════════════════════
# 消融变体模块
# ══════════════════════════════════════════════════════════════

class GaussianPSFKernel(nn.Module):
    """可学习各向异性高斯核（替代 Moffat）"""
    def __init__(self, ch, K=4, max_k=9):
        super().__init__()
        self.K = K; self.max_k = max_k if max_k % 2 == 1 else max_k + 1
        self.sigma = nn.Parameter(torch.rand(K) * 2 + 0.5)
        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, K, 1), nn.Softmax(dim=1))
        self.proj = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        B, C, H, W = x.shape; k = self.max_k
        ys = torch.arange(k, device=x.device).float() - (k - 1) / 2
        xs = torch.arange(k, device=x.device).float() - (k - 1) / 2
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        r2 = gy ** 2 + gx ** 2

        kernels = []
        for i in range(self.K):
            s = F.softplus(self.sigma[i]) + 0.5
            kern = torch.exp(-0.5 * r2 / (s * s))
            kern = kern / kern.sum(); kernels.append(kern)
        kernel = torch.stack(kernels)[None, :, None, :, :]  # [1,K,1,k,k]

        attn = self.scale_attn(x)
        kernel = (kernel * attn[:, :, None, :, :]).sum(dim=1, keepdim=True)
        kernel = kernel.squeeze(2).expand(-1, C, -1, -1).reshape(B * C, 1, k, k)
        x_pad = F.pad(x, [k // 2] * 4, mode='replicate')
        out = F.conv2d(x_pad.reshape(1, B * C, H + k - 1, W + k - 1),
                       kernel, groups=B * C, padding=0)
        out = out.reshape(B, C, H, W)
        return self.proj(out) + x


class PlainChannel(nn.Module):
    """普通卷积通道（无形态先验）—— 替换 PSF/Strip 的占位模块，用于消融实验"""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True))
        self.proj = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.proj(self.conv(x)) + x


class AddFusion(nn.Module):
    """简单相加 + 1×1 conv（替代 SKFusion）"""
    def __init__(self, ch):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.ReLU(inplace=True))
    def forward(self, psf_feat, strip_feat, identity):
        return self.proj(psf_feat + strip_feat) + identity


class ConcatFusion(nn.Module):
    """Concat + 1×1 conv（替代 SKFusion）"""
    def __init__(self, ch):
        super().__init__()
        self.compress = nn.Conv2d(ch * 2, ch, 1)
        self.proj = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.ReLU(inplace=True))
    def forward(self, psf_feat, strip_feat, identity):
        return self.proj(self.compress(torch.cat([psf_feat, strip_feat], dim=1))) + identity


# ══════════════════════════════════════════════════════════════
# 可配置的 DualBlockV3
# ══════════════════════════════════════════════════════════════
class DualBlockV3Ablation(nn.Module):
    """支持消融参数的双路 block

    2³ 全因子设计:
      PSF  ∈ {moffat, gaussian, plain, none}
      Strip ∈ {4dir, 2dir, 8dir, plain(n=0+strip_plain), none(n=0)}
      SK   ∈ {sk, add, concat}
    """
    def __init__(self, in_ch, out_ch, stride=2,
                 psf_type='moffat', psf_k=4,
                 strip_dirs=4, strip_plain=False,
                 fusion='sk'):
        super().__init__()
        # PSF 通道
        if psf_type == 'moffat':
            self.psf = MoffatPSFChannel(in_ch, K=psf_k)
        elif psf_type == 'gaussian':
            self.psf = GaussianPSFKernel(in_ch, K=psf_k)
        elif psf_type == 'plain':
            self.psf = PlainChannel(in_ch)
        elif psf_type == 'none':
            self.psf = None
        else:
            raise ValueError(f"Unknown psf_type: {psf_type}")

        # 条带通道
        if strip_dirs > 0:
            self.strip = self._build_strip(in_ch, strip_dirs)
        elif strip_plain:
            self.strip = PlainChannel(in_ch)
        else:
            self.strip = None

        # 融合
        if fusion == 'sk':
            self.fuse = SKFusion(in_ch)
        elif fusion == 'add':
            self.fuse = AddFusion(in_ch)
        elif fusion == 'concat':
            self.fuse = ConcatFusion(in_ch)
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        self.identity = nn.Conv2d(in_ch, in_ch, 1)
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1), nn.ReLU(inplace=True))

    def _build_strip(self, ch, ndir):
        """构建 N 方向条带通道"""
        half = 7
        pad_h = (half, half, 0, 0)
        pad_w = (0, 0, half, half)
        angles = np.linspace(0, 180, ndir, endpoint=False).tolist()
        strip_h = nn.ModuleList([nn.Conv2d(ch, ch, (5, 15), padding=(2, 0),
                                           groups=ch, bias=False) for _ in range(ndir)])
        strip_v = nn.ModuleList([nn.Conv2d(ch, ch, (15, 5), padding=(0, 2),
                                           groups=ch, bias=False) for _ in range(ndir)])
        attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, ndir, 1),
                             nn.Softmax(dim=1))
        proj = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.ReLU(inplace=True))

        # 把函数封装成类
        class NDirStrip(nn.Module):
            def __init__(self, angles, pad_h, pad_w, strip_h, strip_v, attn, proj):
                super().__init__()
                self.angles = angles; self.pad_h = pad_h; self.pad_w = pad_w
                self.strip_h = strip_h; self.strip_v = strip_v
                self.attn = attn; self.proj = proj

            def _rotate(self, x, ang):
                if ang == 0:
                    return x, lambda y: y
                rad = ang * 3.141592653589793 / 180.0
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
                                     device=x.device, dtype=torch.float32)
                theta = theta.unsqueeze(0).expand(x.shape[0], -1, -1)
                grid = F.affine_grid(theta, x.shape, align_corners=False)
                rot = F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')
                theta_inv = torch.tensor([[cos_a, sin_a, 0], [-sin_a, cos_a, 0]],
                                         device=x.device, dtype=torch.float32)
                theta_inv = theta_inv.unsqueeze(0).expand(x.shape[0], -1, -1)
                def unrot(y):
                    return F.grid_sample(y, F.affine_grid(theta_inv, y.shape,
                                         align_corners=False),
                                         align_corners=False, padding_mode='zeros')
                return rot, unrot

            def forward(self, x):
                outs = []
                for i, ang in enumerate(self.angles):
                    rot, unrot = self._rotate(x, ang)
                    h_out = F.pad(rot, self.pad_h, mode='constant', value=0)
                    h_out = self.strip_h[i](h_out)
                    v_out = F.pad(rot, self.pad_w, mode='constant', value=0)
                    v_out = self.strip_v[i](v_out)
                    outs.append(unrot(h_out + v_out))
                stacked = torch.stack(outs, dim=1)
                a = self.attn(x).unsqueeze(2)
                return self.proj((stacked * a).sum(dim=1)) + x

        return NDirStrip(angles, pad_h, pad_w, strip_h, strip_v, attn, proj)

    def forward(self, x):
        identity = self.identity(x)

        psf_out = self.psf(x) if self.psf is not None else x
        strip_out = self.strip(x) if self.strip is not None else x

        if self.psf is not None and self.strip is not None:
            fused = self.fuse(psf_out, strip_out, identity)
        elif self.psf is not None:
            fused = psf_out + identity
        else:
            fused = strip_out + identity

        return self.down(fused)


class DualPathBackboneAblation(nn.Module):
    """支持消融参数的完整 backbone"""
    def __init__(self, in_ch=1, stem_ch=32, stages=[64, 128, 256, 512],
                 psf_type='moffat', psf_k=4, strip_dirs=4, strip_plain=False, fusion='sk'):
        super().__init__()
        self.stem = Stem(in_ch, stem_ch)
        self.stage1 = DualBlockV3Ablation(stem_ch, stages[0], stride=2,
                                          psf_type=psf_type, psf_k=psf_k,
                                          strip_dirs=strip_dirs, strip_plain=strip_plain, fusion=fusion)
        self.stage2 = DualBlockV3Ablation(stages[0], stages[1], stride=2,
                                          psf_type=psf_type, psf_k=psf_k,
                                          strip_dirs=strip_dirs, strip_plain=strip_plain, fusion=fusion)
        self.stage3 = SingleBlock(stages[1], stages[2], stride=2)
        self.stage4 = SingleBlock(stages[2], stages[3], stride=2)
        self.out_channels = stages

    def forward(self, x):
        f0 = self.stem(x)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4], None, None  # 兼容 MAINetRCNN: (feats, params, gates)


# ══════════════════════════════════════════════════════════════
# 消融配置表 — 2³ 全因子设计
# ========================
# PSF ∈ {plain, moffat}   Strip ∈ {plain, 4dir}   SK ∈ {add, sk}
# 共 8 个实验，可分析主效应和交互效应
# ══════════════════════════════════════════════════════════════
ABLATIONS = {
    # ── 0个创新（全无）──
    'none':        dict(desc='全无（双 PlainConv + Add）',            psf='plain',  strip=0,  strip_plain=True, fusion='add'),

    # ── 1个创新 ──
    'psf':         dict(desc='仅 PSF（Moffat + Plain + Add）',       psf='moffat', strip=0,  strip_plain=True, fusion='add'),
    'strip':       dict(desc='仅 Strip（Plain + 4dir + Add）',       psf='plain',  strip=4,  strip_plain=True, fusion='add'),
    'sk':          dict(desc='仅 SK（双 PlainConv + SKFusion）',      psf='plain',  strip=0,  strip_plain=True, fusion='sk'),

    # ── 2个创新 ──
    'psf_strip':   dict(desc='PSF + Strip（Moffat + 4dir + Add）',  psf='moffat', strip=4,  strip_plain=True, fusion='add'),
    'psf_sk':      dict(desc='PSF + SK（Moffat + Plain + SK）',     psf='moffat', strip=0,  strip_plain=True, fusion='sk'),
    'strip_sk':    dict(desc='Strip + SK（Plain + 4dir + SK）',     psf='plain',  strip=4,  strip_plain=True, fusion='sk'),

    # ── 3个创新（全有）──
    'full':        dict(desc='全有（Moffat + 4dir + SKFusion）✅',    psf='moffat', strip=4,  strip_plain=True, fusion='sk'),
}


def build_ablation_model(name, in_ch=1, stem_ch=32, device='cuda'):
    """构建消融模型"""
    cfg = ABLATIONS[name]
    bb = DualPathBackboneAblation(
        in_ch=in_ch, stem_ch=stem_ch,
        psf_type=cfg['psf'], psf_k=4,
        strip_dirs=cfg['strip'], strip_plain=cfg.get('strip_plain', False),
        fusion=cfg['fusion'])
    from heads import FPN, AnchorGenerator, RPN, ROIHead
    chs = bb.out_channels
    fpn = FPN(in_channels=chs, out_ch=256)
    anchor_gen = AnchorGenerator()
    rpn = RPN(anchor_gen)
    roi_head = ROIHead(in_ch=256, num_classes=1)
    # 包装成 SimpleRCNN（不用 MAINetRCNN 全类，减少依赖）
    model = _SimpleRCNN(bb, fpn, rpn, roi_head)
    model.to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[{name}] {cfg['desc']}: {n/1e6:.2f}M")
    return model, cfg['desc']


class _SimpleRCNN(nn.Module):
    """轻量 RCNN 包装器，与 MAINetRCNN 接口兼容"""
    def __init__(self, backbone, fpn, rpn, roi_head):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.rpn = rpn
        self.roi_head = roi_head

    def forward(self, x, targets=None):
        feats = self.backbone(x)
        fpn_feats = self.fpn(feats)

        if self.training and targets is not None:
            raise NotImplementedError("消融脚本只做推理评估")
        # 推理模式（与 eval.py 兼容）
        img_size = x.shape[-2:]
        img_size = x.shape[-2:]
        proposals = self.rpn(fpn_feats, img_size, train=False)
        results = self.roi_head(fpn_feats, proposals, img_size, train=False)
        return results


# ══════════════════════════════════════════════════════════════
# 训练入口（复用 train.py 的训练循环）
# ══════════════════════════════════════════════════════════════
def run_ablation(name, args):
    """训练单个消融变体"""
    from train import Config as BaseConfig, set_seed, run_training, compute_lr, set_lr
    from model import MAINetRCNN, RCNNCriterion

    cfg = BaseConfig()
    cfg.model_type = f"rcnn_v3_ablation_{name}"
    cfg.output_dir = os.path.join(_PROJECT_ROOT, "work_dirs", "mainet", f"v3_{name}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.epochs = args.epochs
    cfg.batch_size = getattr(args, 'batch_size', 4)
    if args.debug:
        cfg.epochs = 1
        cfg.patience = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 覆盖 build_model：注入消融 backbone
    class AblationMAINetRCNN(MAINetRCNN):
        def __init__(self, **kwargs):
            bb_name = kwargs.pop('_ablation_name', 'full')
            super().__init__(**kwargs)
            bb_cfg = ABLATIONS[bb_name]
            self.backbone = DualPathBackboneAblation(
                in_ch=kwargs.get('in_chans', 1), stem_ch=32,
                psf_type=bb_cfg['psf'], psf_k=4,
                strip_dirs=bb_cfg['strip'],
                strip_plain=bb_cfg.get('strip_plain', False),
                fusion=bb_cfg['fusion'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.seed)

    # 数据
    from train import build_dataloaders
    tl, vl = build_dataloaders(cfg, debug=args.debug)

    # 模型
    model = AblationMAINetRCNN(in_chans=1, num_classes=1,
                               _ablation_name=name).to(device)
    criterion = RCNNCriterion(model)
    n = sum(p.numel() for p in model.parameters())
    print(f"[{name}] {ABLATIONS[name]['desc']}: {n/1e6:.2f}M params")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)
    use_amp = cfg.use_amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 训练
    from train import (monitor_metric, save_ckpt, train_one_epoch, validate,
                       param_weight_scale)
    writer = None
    try: from torch.utils.tensorboard import SummaryWriter
    except ImportError: pass

    start_epoch = 0; best_monitor = float('inf'); pc = 0; gs = 0

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        lr = compute_lr(epoch, cfg); set_lr(optimizer, lr)
        ps = param_weight_scale(epoch, cfg.param_decay_epochs)
        tl_, gs = train_one_epoch(model, criterion, tl, optimizer, device,
                                   epoch, cfg, ps, scaler, writer, gs)
        vl_ = validate(model, criterion, vl, device, epoch, cfg, ps, writer)
        m = monitor_metric(vl_)
        print(f"  [{name}] {time.time()-t0:.0f}s LR={lr:.2e} "
              f"T={tl_['total']:.3f} V={vl_['total']:.3f} mon={m:.4f}")
        if m < best_monitor - cfg.min_delta:
            best_monitor = m; pc = 0
            save_ckpt(os.path.join(cfg.output_dir, 'best_model.pt'),
                      model, optimizer, scaler, epoch, best_monitor, pc, gs, cfg)
        else:
            pc += 1
        if pc >= cfg.patience:
            print(f"  [{name}] Early stop @{epoch+1}"); break
        if args.debug and epoch >= 2: break

    print(f"  [{name}] Done. Best mon={best_monitor:.4f}")
    return best_monitor


# ══════════════════════════════════════════════════════════════
# 评估（调用 eval.py 的 evaluate_model）
# ══════════════════════════════════════════════════════════════
def eval_ablation(name, args):
    """评估单个消融模型的 best checkpoint"""
    ckpt_dir = os.path.join(_PROJECT_ROOT, "work_dirs", "mainet", f"v3_{name}")
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [{name}] 无 checkpoint，跳过评估")
        return None

    # 导入 eval 模块
    sys.path.insert(0, _PROJECT_ROOT)
    from eval import evaluate_model as eval_model, _find_best_ckpt, _resolve_variant

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建消融模型
    from model import MAINetRCNN
    class AblationMAINetRCNN(MAINetRCNN):
        def __init__(self, **kwargs):
            bb_name = kwargs.pop('_ablation_name', 'full')
            super().__init__(**kwargs)
            bb_cfg = ABLATIONS[bb_name]
            self.backbone = DualPathBackboneAblation(
                in_ch=kwargs.get('in_chans', 1), stem_ch=32,
                psf_type=bb_cfg['psf'], psf_k=4,
                strip_dirs=bb_cfg['strip'],
                strip_plain=bb_cfg.get('strip_plain', False),
                fusion=bb_cfg['fusion'])

    model = AblationMAINetRCNN(in_chans=1, num_classes=1,
                               _ablation_name=name).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd.get('model_state_dict', sd), strict=False)
    model.eval()

    result = eval_model(model, 'local-rcnn', args.data, args.split,
                        device, score_thr=0.3,
                        eval_batch_size=getattr(args, 'eval_bs', 4))
    return result


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════
def print_ablations():
    print(f"\n{'='*70}")
    print(f"  MAINet v3 消融实验 — 2³ 全因子设计")
    print(f"  PSF ∈ {{plain, moffat}}  |  Strip ∈ {{plain, 4dir}}  |  SK ∈ {{add, sk}}")
    print(f"{'='*70}")
    print(f"  {'Key':<12} {'PSF':<8} {'Strip':<8} {'SK':<6} {'描述':<36}")
    print(f"  {'-'*66}")
    for key, cfg in ABLATIONS.items():
        psf_label = 'moffat' if cfg['psf'] == 'moffat' else 'plain'
        strip_label = f"{cfg['strip']}dir" if cfg['strip'] > 0 else 'plain'
        sk_label = cfg['fusion']
        print(f"  {key:<12} {psf_label:<8} {strip_label:<8} {sk_label:<6} {cfg['desc']:<36}")
    print(f"{'='*70}")
    print(f"  主效应分析: PSF主效应 = avg(有PSF) - avg(无PSF)")
    print(f"              Strip主效应 = avg(有Strip) - avg(无Strip)")
    print(f"              SK主效应 = avg(有SK) - avg(无SK)")
    print(f"  交互效应: PSF×Strip, PSF×SK, Strip×SK, PSF×Strip×SK")
    print(f"{'='*70}\n")


def final_eval_and_report(keys, args):
    """对所有变体做最终评估，汇总对比表格"""
    print(f"\n{'='*70}")
    print(f"  最终评估汇总")
    print(f"{'='*70}\n")

    results = {}
    for k in keys:
        r = eval_ablation(k, args)
        if r:
            results[f"v3_{k}"] = r

    if not results:
        print("  无可用结果")
        return results

    # 调用 eval.py 的打印
    try:
        from eval import print_table
        print_table(results)
    except (ImportError, AttributeError):
        # 回退：简单打印
        header = f"  {'Model':<14} {'AP':>6} {'AP50':>6} {'AP75':>6} {'cMSE':>6}"
        print(header)
        print(f"  {'-'*42}")
        for name, r in results.items():
            print(f"  {name:<14} {r.get('ap',0):>6.3f} {r.get('ap50',0):>6.3f} "
                  f"{r.get('ap75',0):>6.3f} {r.get('cmse',0):>6.2f}")

    # 2³ 主效应分析
    print(f"\n{'='*70}")
    print(f"  2³ 主效应分析")
    print(f"{'='*70}")
    # 按 2³ 因子分组
    def _avg(ks):
        vals = [results[f"v3_{k}"].get('ap', 0) for k in ks if f"v3_{k}" in results]
        return sum(vals) / len(vals) if vals else 0

    psf_on  = ['psf', 'psf_strip', 'psf_sk', 'full']
    psf_off = ['none', 'strip', 'sk', 'strip_sk']
    strip_on  = ['strip', 'psf_strip', 'strip_sk', 'full']
    strip_off = ['none', 'psf', 'sk', 'psf_sk']
    sk_on  = ['sk', 'psf_sk', 'strip_sk', 'full']
    sk_off = ['none', 'psf', 'strip', 'psf_strip']

    psf_eff = _avg(psf_on) - _avg(psf_off)
    strip_eff = _avg(strip_on) - _avg(strip_off)
    sk_eff = _avg(sk_on) - _avg(sk_off)

    print(f"  PSF  主效应 = avg(有PSF) - avg(无PSF)  = {psf_eff:+.4f}")
    print(f"  Strip 主效应 = avg(有Strip) - avg(无Strip) = {strip_eff:+.4f}")
    print(f"  SK   主效应 = avg(有SK) - avg(无SK)   = {sk_eff:+.4f}")

    # 交互效应（简化：2路交互）
    psf_strip_both = _avg(['psf_strip', 'full'])      # PSF=on, Strip=on
    psf_strip_neither = _avg(['none', 'sk'])            # PSF=off, Strip=off
    psf_only_eff = _avg(['psf', 'psf_sk'])              # PSF=on, Strip=off
    strip_only_eff = _avg(['strip', 'strip_sk'])         # PSF=off, Strip=on
    interaction_ps = (psf_strip_both + psf_strip_neither) - (psf_only_eff + strip_only_eff)
    print(f"  PSF×Strip 交互 = {interaction_ps:+.4f}  (>0表示协同增益)")

    return results


def main():
    parser = argparse.ArgumentParser(description='MAINet v3 消融实验')
    parser.add_argument('--list', action='store_true', help='列出所有消融变体')
    parser.add_argument('--ablation', default=None, help='单个消融 key')
    parser.add_argument('--all', action='store_true', help='跑全部消融')
    parser.add_argument('--eval-only', action='store_true', help='仅评估已有 ckpt')
    parser.add_argument('--data', default='output')
    parser.add_argument('--split', default='test')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--eval-bs', type=int, default=4,
                        help='评估时批量推理大小（1=逐张，默认4）')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.list:
        print_ablations()
        return

    if args.ablation:
        keys = [args.ablation]
    elif args.all:
        keys = list(ABLATIONS.keys())
    else:
        parser.print_help()
        return

    for k in keys:
        if k not in ABLATIONS:
            print(f"未知消融: {k}"); continue

    if args.eval_only:
        # 仅评估模式：评估已有 ckpt + 出汇总报告
        final_eval_and_report(keys, args)
    else:
        # 训练模式：逐个训练
        for k in keys:
            print(f"\n{'='*60}")
            print(f"消融: {k} — {ABLATIONS[k]['desc']}")
            print(f"{'='*60}")
            run_ablation(k, args)

        # 全部训练完成后，自动最终评估
        print(f"\n{'='*70}")
        print(f"  全部训练完成，开始最终评估...")
        print(f"{'='*70}")
        final_eval_and_report(keys, args)


if __name__ == "__main__":
    main()
