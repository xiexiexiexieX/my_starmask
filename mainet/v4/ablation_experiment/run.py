"""MAINet-v4 three-factor ablation experiment.

Innovation factors:
  P: PSFGatedEnhancement
  S: DeformableOrientationStrip
  M: MorphologyFusion, which combines the learned global morphology gate and
     the local gated residual fusion. Gate and fusion are one innovation.

Only the seven non-full combinations are trained. After ``--all`` finishes,
the script evaluates those seven checkpoints and the already-trained full V4
checkpoint from ``work_dirs/real_mixed_baselines/mainet_v4/best_model.pt``.
The final report contains exactly eight rows and only AP/MSE.

Commands:
  python mainet/v4/ablation_experiment/run.py --list
  python mainet/v4/ablation_experiment/run.py --all --data output_mix --epochs 100
  python mainet/v4/ablation_experiment/run.py --all --data output_mix --debug
  python mainet/v4/ablation_experiment/run.py --eval-only --data output_mix

Experiment artifacts:
  work_dirs/ablation/mainet_v4/checkpoints/<variant>/best_model.pt
  work_dirs/ablation/mainet_v4/ablation_results.json
  work_dirs/ablation/mainet_v4/ablation_results.txt

RPN, ROI heads, losses, and training settings stay fixed in all combinations.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_V4_DIR = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_V4_DIR))
if _V4_DIR not in sys.path:
    sys.path.insert(0, _V4_DIR)

_EXPERIMENT_ROOT = os.path.join(
    _PROJECT_ROOT, 'work_dirs', 'ablation', 'mainet_v4')
_DEBUG_EXPERIMENT_ROOT = os.path.join(
    _PROJECT_ROOT, 'work_dirs', 'ablation', 'mainet_v4_debug')
_DEFAULT_FULL_CKPT = os.path.join(
    'work_dirs', 'real_mixed_baselines', 'mainet_v4', 'best_model.pt')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#        v4             
from backbone import (Stem, GlobalMorphologyContext, PSFGatedEnhancement,
                      DeformableOrientationStrip, GatedResidualFusion,
                      SingleBlock)


#                                                                                                                                            ?
#               
#                                                                                                                                            ?

class GaussianPSFKernel(nn.Module):
    """Learnable Gaussian PSF ablation branch."""
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
    """Identity branch used when a morphology module is disabled."""
    def __init__(self, ch):
        super().__init__()

    def forward(self, x):
        return x


class AddFusion(nn.Module):
    """Parameter-free residual addition used when local fusion is disabled."""
    def __init__(self, ch):
        super().__init__()

    def forward(self, psf_feat, strip_feat, identity, mode_gate=None):
        psf_delta = psf_feat - identity
        strip_delta = strip_feat - identity
        if mode_gate is None:
            psf_prior = strip_prior = 1.0
        else:
            gate = mode_gate[:, :, None, None]
            psf_prior = 0.05 + 0.95 * (1.0 - gate)
            strip_prior = 0.05 + 0.95 * gate
        return (identity + psf_prior * psf_delta +
                strip_prior * strip_delta)


class ConcatFusion(nn.Module):
    """Concat fusion baseline."""
    def __init__(self, ch):
        super().__init__()
        self.compress = nn.Conv2d(ch * 2, ch, 1)
        self.proj = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.ReLU(inplace=True))
    def forward(self, psf_feat, strip_feat, identity):
        return self.proj(self.compress(torch.cat([psf_feat, strip_feat], dim=1))) + identity


#                                                                                                                                            ?
#           DualBlockV3
#                                                                                                                                            ?
class DualBlockV3Ablation(nn.Module):
    """Configurable v4 dual block for backbone ablations."""
    def __init__(self, in_ch, out_ch, stride=2,
                 psf_type='psf_gate', psf_k=4,
                 strip_dirs=4, strip_plain=False,
                 fusion='residual'):
        super().__init__()
        # PSF     ?
        if psf_type == 'psf_gate':
            self.psf = PSFGatedEnhancement(in_ch, K=psf_k)
        elif psf_type == 'gaussian':
            self.psf = GaussianPSFKernel(in_ch, K=psf_k)
        elif psf_type == 'plain':
            self.psf = PlainChannel(in_ch)
        elif psf_type == 'none':
            self.psf = None
        else:
            raise ValueError(f"Unknown psf_type: {psf_type}")

        #          ?
        if strip_dirs > 0:
            self.strip = self._build_strip(in_ch, strip_dirs)
        elif strip_plain:
            self.strip = PlainChannel(in_ch)
        else:
            self.strip = None

        #     ?
        if fusion == 'residual':
            self.fuse = GatedResidualFusion(in_ch)
        elif fusion == 'add':
            self.fuse = AddFusion(in_ch)
        elif fusion == 'concat':
            self.fuse = ConcatFusion(in_ch)
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        gn = max(1, min(8, out_ch))
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(gn, out_ch),
            nn.ReLU(inplace=True))

    def _build_strip(self, ch, ndir):
        return DeformableOrientationStrip(ch)

    def forward(self, x, mode_gate=None, global_direction=None):
        identity = x

        psf_out = self.psf(x) if self.psf is not None else x
        if isinstance(self.strip, DeformableOrientationStrip):
            strip_out = self.strip(x, global_direction=global_direction)
        else:
            strip_out = self.strip(x) if self.strip is not None else x

        if self.psf is not None and self.strip is not None:
            if isinstance(self.fuse, (GatedResidualFusion, AddFusion)):
                fused = self.fuse(
                    psf_out, strip_out, identity, mode_gate=mode_gate)
            else:
                fused = self.fuse(psf_out, strip_out, identity)
        elif self.psf is not None:
            fused = psf_out + identity
        else:
            fused = strip_out + identity

        return self.down(fused)


class DualPathBackboneAblation(nn.Module):
    """Complete ablation backbone."""
    def __init__(self, in_ch=1, stem_ch=32, stages=[64, 128, 256, 512],
                 psf_type='psf_gate', psf_k=4, strip_dirs=4,
                 strip_plain=False, fusion='residual', gate_mode='learned'):
        super().__init__()
        if gate_mode not in ('learned', 'fixed'):
            raise ValueError(f"Unknown gate_mode: {gate_mode}")
        self.gate_mode = gate_mode
        self.stem = Stem(in_ch, stem_ch)
        self.context = GlobalMorphologyContext(stem_ch)
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
        gate_logit, direction = self.context(f0)
        if self.gate_mode == 'fixed':
            # MorphFusion-off uses plain residual addition with no global
            # routing prior and no gate supervision.
            routing_gate = None
            supervised_gate_logit = None
        else:
            mode_gate = torch.sigmoid(gate_logit).unsqueeze(1)
            routing_gate = mode_gate.detach() if self.training else mode_gate
            supervised_gate_logit = gate_logit
        f1 = self.stage1(f0, mode_gate=routing_gate,
                         global_direction=direction)
        f2 = self.stage2(f1, mode_gate=routing_gate,
                         global_direction=direction)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        aux = {'gate_logit': supervised_gate_logit, 'direction': direction}
        return [f1, f2, f3, f4], aux, None  # stride 2/4/8/16


# Full 2^3 design. The 111 cell is the external full V4 reference and is not
# part of ABLATIONS because this script must never retrain the complete model.
# Factor order in ``bits`` is P/S/M:
#   P = PSFGatedEnhancement
#   S = DeformableOrientationStrip
#   M = learned GlobalMorphologyContext + local GatedResidualFusion
def _factorial_cfg(psf, strip, morph_fusion):
    factors = dict(psf=bool(psf), strip=bool(strip),
                   morph_fusion=bool(morph_fusion))
    labels = [label for label, enabled in (
        ('PSF', psf), ('Strip', strip),
        ('MorphFusion', morph_fusion)) if enabled]
    return dict(
        desc=' + '.join(labels) if labels else 'Baseline',
        psf='psf_gate' if psf else 'plain',
        strip=4 if strip else 0,
        strip_plain=True,
        fusion='residual' if morph_fusion else 'add',
        gate='learned' if morph_fusion else 'fixed',
        factors=factors,
        bits=''.join('1' if factors[name] else '0'
                     for name in ('psf', 'strip', 'morph_fusion')))


ABLATIONS = {
    'none':         _factorial_cfg(0, 0, 0),
    'psf':          _factorial_cfg(1, 0, 0),
    'strip':        _factorial_cfg(0, 1, 0),
    'fusion':       _factorial_cfg(0, 0, 1),
    'psf_strip':    _factorial_cfg(1, 1, 0),
    'psf_fusion':   _factorial_cfg(1, 0, 1),
    'strip_fusion': _factorial_cfg(0, 1, 1),
}

FULL_CONFIG = _factorial_cfg(1, 1, 1)
ALL_RESULT_CONFIGS = {**ABLATIONS, 'full': FULL_CONFIG}


def build_ablation_model(name, in_ch=1, stem_ch=32, device='cuda'):
    """Build an ablation model."""
    cfg = ABLATIONS[name]
    bb = DualPathBackboneAblation(
        in_ch=in_ch, stem_ch=stem_ch,
        psf_type=cfg['psf'], psf_k=4,
        strip_dirs=cfg['strip'], strip_plain=cfg.get('strip_plain', False),
        fusion=cfg['fusion'], gate_mode=cfg['gate'])
    from heads import FPN, AnchorGenerator, RPN, ROIHead
    chs = bb.out_channels
    fpn = FPN(in_channels=chs, out_ch=256)
    strides = (2, 4, 8, 16)
    anchor_gen = AnchorGenerator(strides=strides)
    rpn = RPN(anchor_gen)
    roi_head = ROIHead(in_ch=256, num_classes=1, strides=strides)
    #       ?SimpleRCNN      ?MAINetRCNN                   
    model = _SimpleRCNN(bb, fpn, rpn, roi_head)
    model.to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[{name}] {cfg['desc']}: {n/1e6:.2f}M")
    return model, cfg['desc']


class _SimpleRCNN(nn.Module):
    """Lightweight RCNN wrapper compatible with MAINetRCNN evaluation."""
    def __init__(self, backbone, fpn, rpn, roi_head):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.rpn = rpn
        self.roi_head = roi_head

    def forward(self, x, targets=None):
        feats, _, _ = self.backbone(x)
        fpn_feats = self.fpn(feats)

        if self.training and targets is not None:
            raise NotImplementedError("Ablation wrapper is for inference/evaluation only.")
        img_size = x.shape[-2:]
        proposals, _ = self.rpn(fpn_feats, None, img_size, train=False)
        results = self.roi_head(fpn_feats, proposals, img_size, train=False)
        return results


def experiment_root(debug=False):
    """Return the isolated artifact directory for this experiment."""
    return _DEBUG_EXPERIMENT_ROOT if debug else _EXPERIMENT_ROOT


def ablation_checkpoint_dir(name, debug=False):
    return os.path.join(experiment_root(debug), 'checkpoints', name)


def run_ablation(name, args):
    """Train one v4 ablation variant."""
    from train import Config as BaseConfig, set_seed, compute_lr, set_lr
    from model import MAINetRCNN, RCNNCriterion

    cfg = BaseConfig()
    cfg.model_type = f"rcnn_v4_ablation_{name}"
    cfg.data_root = (args.data if os.path.isabs(args.data)
                     else os.path.join(_PROJECT_ROOT, args.data))
    cfg.output_dir = ablation_checkpoint_dir(name, debug=args.debug)
    cfg.log_dir = os.path.join(cfg.output_dir, 'runs')
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.epochs = args.epochs
    cfg.batch_size = getattr(args, 'batch_size', 2)
    if args.debug:
        cfg.epochs = 1
        cfg.patience = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #     ?build_model          ?backbone
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
                fusion=bb_cfg['fusion'],
                gate_mode=bb_cfg.get('gate', 'learned'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.seed)

    #     ?
    from train import build_dataloaders
    tl, vl = build_dataloaders(cfg, debug=args.debug)

    #      ?
    model = AblationMAINetRCNN(in_chans=1, num_classes=1,
                               gate_pos_weight=cfg.gate_pos_weight,
                               gate_loss_weight=cfg.gate_loss_weight,
                               _ablation_name=name).to(device)
    criterion = RCNNCriterion(model)
    n = sum(p.numel() for p in model.parameters())
    print(f"[{name}] {ABLATIONS[name]['desc']}: {n/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    use_amp = cfg.use_amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    #     ?
    from train import (monitor_metric, save_ckpt, train_one_epoch, validate,
                       param_weight_scale)
    writer = None
    try: from torch.utils.tensorboard import SummaryWriter
    except ImportError: pass

    start_epoch = 0; best_monitor = float('-inf'); pc = 0; gs = 0

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        lr = compute_lr(epoch, cfg); set_lr(optimizer, lr)
        ps = param_weight_scale(epoch, cfg.param_decay_epochs)
        tl_, gs = train_one_epoch(model, criterion, tl, optimizer, device,
                                   epoch, cfg, ps, scaler, writer, gs)
        vl_ = validate(model, criterion, vl, device, epoch, cfg, ps, writer)
        m = monitor_metric(vl_)
        print(f"  [{name}] {time.time()-t0:.0f}s LR={lr:.2e} "
              f"T={tl_['total']:.3f} V={vl_['total']:.3f} mAP={m:.4f}")
        if m > best_monitor + cfg.min_delta:
            best_monitor = m; pc = 0
            save_ckpt(os.path.join(cfg.output_dir, 'best_model.pt'),
                      model, optimizer, scaler, epoch, best_monitor, pc, gs, cfg)
        else:
            pc += 1
        if pc >= cfg.patience:
            print(f"  [{name}] Early stop @{epoch+1}"); break
        if args.debug and epoch >= 2: break

    print(f"  [{name}] Done. Best mAP={best_monitor:.4f}")
    return best_monitor


#                                                                                                                                            ?
#           ?eval.py  ?evaluate_model ?#                                                                                                                                            ?
def eval_ablation(name, args):
    """Evaluate one ablation checkpoint."""
    ckpt_dir = ablation_checkpoint_dir(name, debug=args.debug)
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Ablation checkpoint not found: {ckpt_path}")

    sys.path.insert(0, _PROJECT_ROOT)
    from eval import evaluate_model as eval_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                fusion=bb_cfg['fusion'],
                gate_mode=bb_cfg.get('gate', 'learned'))

    model = AblationMAINetRCNN(in_chans=1, num_classes=1,
                               _ablation_name=name).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd.get('model_state_dict', sd), strict=True)
    model.eval()

    data_root = (args.data if os.path.isabs(args.data)
                 else os.path.join(_PROJECT_ROOT, args.data))
    max_images = 20 if args.debug else 0
    result = eval_model(model, 'local-rcnn', data_root, args.split,
                        device, score_thr=0.3,
                        max_images=max_images,
                        eval_batch_size=getattr(args, 'eval_bs', 4))
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def _resolve_full_checkpoint(args):
    path = args.full_ckpt
    if not os.path.isabs(path):
        path = os.path.join(_PROJECT_ROOT, path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Full V4 checkpoint not found: {path}\n"
            "Train the formal V4 first or pass --full-ckpt explicitly.")
    return path


def eval_full_model(args):
    """Evaluate the formal full V4 without training or copying its weights."""
    sys.path.insert(0, _PROJECT_ROOT)
    from eval import evaluate_model as eval_model
    from model import MAINetRCNN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = _resolve_full_checkpoint(args)
    model = MAINetRCNN(in_chans=1, num_classes=1).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd.get('model_state_dict', sd), strict=True)
    model.eval()

    data_root = (args.data if os.path.isabs(args.data)
                 else os.path.join(_PROJECT_ROOT, args.data))
    max_images = 20 if args.debug else 0
    result = eval_model(model, 'local-rcnn', data_root, args.split,
                        device, score_thr=0.3,
                        max_images=max_images,
                        eval_batch_size=getattr(args, 'eval_bs', 4))
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def print_ablations():
    print(f"\n{'='*92}")
    print("  MAINet-v4 2^3 ablation: train 7 variants + reuse 1 formal full model")
    print("  Bits=P/S/M: PSF | DeformStrip | MorphFusion(GlobalGate + ResidualFusion)")
    print(f"{'='*92}")
    print(f"  {'Key':<16} {'Bits':<5} {'PSF':<10} {'Strip':<10} "
          f"{'MorphFusion':<14} {'Source':<12} {'Description':<28}")
    print(f"  {'-'*88}")
    for key, cfg in ALL_RESULT_CONFIGS.items():
        psf_label = 'gated' if cfg['psf'] == 'psf_gate' else 'identity'
        strip_label = 'deform' if cfg['strip'] > 0 else 'identity'
        morph_label = 'learned' if cfg['factors']['morph_fusion'] else 'off'
        source = 'work_dirs' if key == 'full' else 'train'
        print(f"  {key:<16} {cfg['bits']:<5} {psf_label:<10} {strip_label:<10} "
              f"{morph_label:<14} {source:<12} {cfg['desc']:<28}")
    print(f"{'='*92}")
    print("  Seven non-full variants are trained; full=111 is evaluation-only.")
    print(f"{'='*92}\n")


def _json_metric(value):
    value = float(value)
    return None if not math.isfinite(value) else round(value, 4)


def final_eval_and_report(keys, args):
    """Evaluate selected ablations plus formal V4 and save AP/MSE only."""
    results = {}
    for key in keys:
        print(f"\nEvaluate ablation: {key}")
        results[key] = eval_ablation(key, args)

    print("\nEvaluate formal full V4 (not retrained)")
    print(f"  checkpoint: {_resolve_full_checkpoint(args)}")
    results['full'] = eval_full_model(args)

    simple = {
        key: {
            'AP': _json_metric(result.get('ap', float('nan'))),
            'MSE': _json_metric(result.get('cmse', float('nan'))),
        }
        for key, result in results.items()
    }

    report_dir = experiment_root(args.debug)
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, 'ablation_results.json')
    txt_path = os.path.join(report_dir, 'ablation_results.txt')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(simple, f, indent=2, ensure_ascii=False)

    lines = [f"{'Model':<18} {'AP':>10} {'MSE':>10}", '-' * 40]
    for key, metrics in simple.items():
        ap = 'N/A' if metrics['AP'] is None else f"{metrics['AP']:.4f}"
        mse = 'N/A' if metrics['MSE'] is None else f"{metrics['MSE']:.4f}"
        lines.append(f"{key:<18} {ap:>10} {mse:>10}")
    report = '\n'.join(lines)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report + '\n')

    print(f"\n{'='*40}\n{report}\n{'='*40}")
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")
    return simple


def main():
    parser = argparse.ArgumentParser(
        description='Train seven MAINet-v4 ablations, then evaluate them with the formal full V4.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mainet/v4/ablation_experiment/run.py --list
  python mainet/v4/ablation_experiment/run.py --all --data output_mix --epochs 100
  python mainet/v4/ablation_experiment/run.py --all --data output_mix --debug
  python mainet/v4/ablation_experiment/run.py --eval-only --data output_mix

Artifact layout:
  work_dirs/ablation/mainet_v4/checkpoints/<ablation>/best_model.pt
  work_dirs/ablation/mainet_v4/ablation_results.{json,txt}
""")
    parser.add_argument('--list', action='store_true',
                        help='List all ablation variants and exit.')
    parser.add_argument('--ablation', default=None,
                        help='Train one non-full key, e.g. none, psf, strip, fusion.')
    parser.add_argument('--all', action='store_true',
                        help='Train all seven non-full variants and report all eight combinations.')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training and evaluate saved ablations plus the formal full V4.')
    parser.add_argument('--data', default='output_mix',
                        help='Dataset root for both training and evaluation. Default: output_mix')
    parser.add_argument('--split', default='test',
                        help='Evaluation split used after training. Default: test')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs. Default: 100')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Training batch size. Default: 2')
    parser.add_argument('--eval-bs', type=int, default=4,
                        help='Evaluation batch size. Default: 4')
    parser.add_argument('--full-ckpt', default=_DEFAULT_FULL_CKPT,
                        help='Existing formal V4 best_model.pt used as the 111 reference.')
    parser.add_argument('--debug', action='store_true',
                        help='Use a small debug subset and force epochs=1 for a smoke test.')
    args = parser.parse_args()

    if args.list:
        print_ablations()
        return

    if args.eval_only and not args.ablation:
        keys = list(ABLATIONS.keys())
    elif args.ablation:
        keys = [args.ablation]
    elif args.all:
        keys = list(ABLATIONS.keys())
    else:
        parser.print_help()
        return

    for k in keys:
        if k not in ABLATIONS:
            print(f"Unknown ablation key: {k}")
            return

    if args.eval_only:
        final_eval_and_report(keys, args)
        return

    for k in keys:
        print(f"\n{'='*60}")
        print(f"Ablation: {k} - {ABLATIONS[k]['desc']}")
        print(f"{'='*60}")
        run_ablation(k, args)

    print("\nTraining finished. Starting the unified AP/MSE evaluation...")
    final_eval_and_report(keys, args)


if __name__ == "__main__":
    main()
