"""
plot_ap.py — 多模型 PR 曲线对比 & AP
====================================

用法:
  python plot_ap.py                                             # 自动扫描全部
  python plot_ap.py --models v2 mask_rcnn                       # 指定模型
  python plot_ap.py --n 50                                      # 只用 50 张图
  python plot_ap.py --split val --n 30

输出: work_dirs/ap_comparison.png
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

_mmdet_available = False
try:
    from mmdet.apis import init_detector, inference_detector
    _mmdet_available = True
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WORK_DIRS = 'work_dirs'

# ── 颜色方案 ──
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']
MODEL_COLORS = {}  # 按模型名固定颜色

# ══════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════

def _find_ckpt(dpath):
    if os.path.isfile(dpath):
        return dpath
    if not os.path.isdir(dpath):
        return None
    lc = os.path.join(dpath, 'last_checkpoint')
    if os.path.exists(lc):
        with open(lc) as f:
            real = f.read().strip()
        p = real if os.path.exists(real) else os.path.join(dpath, os.path.basename(real))
        if os.path.exists(p):
            return p
    best = sorted([f for f in os.listdir(dpath) if f.startswith('best_coco')], reverse=True)
    if best:
        return os.path.join(dpath, best[0])
    for name in ['best_model.pt', 'best.pt']:
        p = os.path.join(dpath, name)
        if os.path.exists(p):
            return p
    ep = sorted([f for f in os.listdir(dpath)
                 if (f.endswith('.pt') or f.endswith('.pth')) and 'epoch' in f],
                key=lambda x: int(x.split('_')[-1].replace('.pt', '').replace('.pth', '')),
                reverse=True)
    if ep:
        return os.path.join(dpath, ep[0])
    return None


def mask_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union > 0 else 0.0


def _normalize_astro(img):
    bg = np.median(img)
    img_sub = img - bg
    bg_mask = img < np.percentile(img, 90)
    scale = np.std(img_sub[bg_mask]) + 1e-6
    a = np.arcsinh(img_sub / scale)
    return ((a - a.mean()) / (a.std() + 1e-6)).astype(np.float32)


def match_at_threshold(pred_masks, pred_scores, gt_masks, score_thr, iou_thr):
    keep = pred_scores >= score_thr
    masks = pred_masks[keep]
    K, M = len(masks), len(gt_masks)
    if K == 0:
        return 0, 0, M
    if M == 0:
        return 0, K, 0
    iou_mat = np.zeros((K, M))
    for i in range(K):
        for j in range(M):
            iou_mat[i, j] = mask_iou(masks[i], gt_masks[j])
    pi, gj = linear_sum_assignment(-iou_mat)
    tp = sum(1 for p, g in zip(pi, gj) if iou_mat[p, g] >= iou_thr)
    return tp, K - tp, M - tp


def discover_models():
    """扫描 work_dirs 下所有可用模型（新旧两种目录结构）。"""
    models = []
    added = set()  # 去重

    def _try_add(name, typ, cfg, dpath):
        ckpt = _find_ckpt(dpath)
        if ckpt is None:
            return
        key = os.path.normpath(dpath)
        if key not in added:
            added.add(key)
            models.append((name, typ, cfg, dpath, ckpt))

    # 新结构: work_dirs/mainet/<name>/ 和 work_dirs/mmdet/<name>/
    for framework, typ in [('mainet', 'local'), ('mmdet', 'mmdet')]:
        fw_dir = os.path.join(WORK_DIRS, framework)
        if os.path.isdir(fw_dir):
            for d in sorted(os.listdir(fw_dir)):
                dpath = os.path.join(fw_dir, d)
                if not os.path.isdir(dpath):
                    continue
                if typ == 'local':
                    name = f'MAINet-{d.upper()}'
                    _try_add(name, typ, None, dpath)
                else:
                    cfg = os.path.join(os.path.dirname(__file__), 'mmdet', 'configs', f'{d}_star.py')
                    if os.path.exists(cfg):
                        name = d.replace('_', ' ').title().replace(' ', '')
                        _try_add(name, typ, cfg, dpath)

    # 旧扁平结构: work_dirs/mainet_*/ 和 work_dirs/*_star/（向后兼容）
    for d in sorted(os.listdir(WORK_DIRS)):
        dpath = os.path.join(WORK_DIRS, d)
        if not os.path.isdir(dpath) or d in ('mainet', 'mmdet'):
            continue
        if d.endswith('_star'):
            cfg = os.path.join(os.path.dirname(__file__), 'mmdet', 'configs', f'{d}.py')
            if os.path.exists(cfg):
                name = d.replace('_star', '').replace('_', ' ').title().replace(' ', '')
                _try_add(name, 'mmdet', cfg, dpath)
        elif d.startswith('mainet_'):
            variant = d.replace('mainet_', '')
            name = f'MAINet-{variant.replace("rcnn_", "").upper()}'
            _try_add(name, 'local', None, dpath)
    return models


# ══════════════════════════════════════════════════════════════
# 单模型推理
# ══════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_variant(dir_name):
    name = dir_name.replace('mainet_rcnn_', '').replace('mainet_', '')
    return name


def _import_rcnn_model(variant):
    model_dir = os.path.join(PROJECT_ROOT, 'mainet', variant)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    for mod in list(sys.modules):
        if mod in ('model', 'backbone', 'heads', 'dataset', 'param_loss'):
            sys.modules.pop(mod, None)
    import model as _m
    return _m.MAINetRCNN, _m.RCNNCriterion


def load_model(name, typ, cfg, ckpt, device):
    if typ == 'mmdet':
        if not _mmdet_available:
            return None
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mmdet'))
        model = init_detector(cfg, ckpt, device=device)
        model.eval()
        return model

    ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
    sd = ckpt_data.get('model_state_dict', ckpt_data)
    # 从目录名解析变体并导入对应模型
    variant = _resolve_variant(os.path.basename(os.path.normpath(os.path.dirname(ckpt))))
    MAINetRCNN, _ = _import_rcnn_model(variant)
    model = MAINetRCNN(in_chans=1, num_classes=1).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def run_model(model, typ, img_path, device):
    if typ == 'mmdet':
        result = inference_detector(model, img_path)
        inst = result.pred_instances
        masks = inst.masks.cpu().numpy().astype(bool)
        scores = inst.scores.cpu().numpy()
    else:
        img = np.load(img_path).astype(np.float32)
        x = torch.from_numpy(_normalize_astro(img))[None, None].to(device)
        results = model(x)[0]
        masks = results['masks'].cpu().numpy().astype(bool)
        scores = results['scores'].cpu().numpy()
    return masks, scores


# ══════════════════════════════════════════════════════════════
# PR 计算
# ══════════════════════════════════════════════════════════════

def compute_pr_curve(all_preds, all_gts, iou_thr, n_points=100):
    """从所有预测计算 PR 曲线点。"""
    all_scores = np.concatenate([s for _, s in all_preds if len(s) > 0])
    if len(all_scores) == 0:
        return np.array([0]), np.array([0]), 0.0

    thr_min = max(float(all_scores.min()), 0.01)
    thr_max = float(all_scores.max())
    thresholds = np.linspace(thr_min, thr_max, n_points)[::-1]

    tp_arr = np.zeros(n_points)
    fp_arr = np.zeros(n_points)
    fn_arr = np.zeros(n_points)

    for (pms, pss), gms in zip(all_preds, all_gts):
        for i, thr in enumerate(thresholds):
            tp, fp, fn = match_at_threshold(pms, pss, gms, thr, iou_thr)
            tp_arr[i] += tp; fp_arr[i] += fp; fn_arr[i] += fn

    prec = np.divide(tp_arr, tp_arr + fp_arr, out=np.zeros_like(tp_arr),
                     where=(tp_arr + fp_arr) > 0)
    rec = np.divide(tp_arr, tp_arr + fn_arr, out=np.zeros_like(tp_arr),
                    where=(tp_arr + fn_arr) > 0)

    # 按 recall 排序
    order = np.argsort(rec)
    r_sort, p_sort = rec[order], prec[order]

    # all-point interpolated: precision 单调递减
    for i in range(len(p_sort) - 2, -1, -1):
        p_sort[i] = max(p_sort[i], p_sort[i + 1])

    # 补端点
    r_plot = np.concatenate([[0.0], r_sort, [1.0]])
    p_plot = np.concatenate([[1.0], p_sort, [0.0]])

    ap = float(np.trapz(p_plot, r_plot))
    return r_plot, p_plot, ap


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='多模型 PR 曲线对比')
    ap.add_argument('--models', nargs='*', default=None,
                    help='模型目录名（work_dirs/ 下），不传则自动扫描全部')
    ap.add_argument('--data', default='output')
    ap.add_argument('--split', default='test')
    ap.add_argument('--n', type=int, default=0,
                    help='最多使用图片数（0=全部）')
    ap.add_argument('--iou_thr', type=float, default=0.5,
                    help='匹配 IoU 阈值')
    ap.add_argument('--n_points', type=int, default=100,
                    help='score 采样点数')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f"Device: {device}")

    # ── 确定模型列表 ──
    if args.models:
        target_set = set(args.models)
        all_models = discover_models()
        selected = [(n, t, c, d, ck) for n, t, c, d, ck in all_models
                    if os.path.basename(d) in target_set]
        # 保持用户输入顺序
        ordered = []
        for m in args.models:
            for n, t, c, d, ck in selected:
                if os.path.basename(d) == m:
                    ordered.append((n, t, c, d, ck))
                    break
        selected = ordered
    else:
        selected = discover_models()

    if not selected:
        print("未找到可评估模型"); sys.exit(1)

    print(f"\nModels ({len(selected)}):")
    for name, typ, _, dpath, _ in selected:
        if typ == 'mmdet' and not _mmdet_available:
            print(f"  [skip] {name} — mmdet 未安装")
            continue
        print(f"  [{typ}] {name}  ← {dpath}")

    # ── 数据 ──
    coco_json = f"{args.data}/annotations/{args.split}.json"
    with open(coco_json) as f:
        coco = json.load(f)
    ann_by_img = {}
    for ann in coco['annotations']:
        ann_by_img.setdefault(ann['image_id'], []).append(ann)

    images = coco['images']
    n_use = min(args.n, len(images)) if args.n > 0 else len(images)
    images = images[:n_use]
    print(f"Images: {n_use}  |  IoU threshold: {args.iou_thr}")

    mask_dir = f"{args.data}/{args.split}/masks"

    # 预加载 GT（所有模型共享）
    all_gts = []
    print(f"\nLoading GT masks...")
    for im in images:
        gt_masks = np.load(f"{mask_dir}/{im['id']:06d}_masks.npy").astype(bool)
        all_gts.append(gt_masks)

    # ── 逐模型评估 ──
    results = []  # [(name, r_plot, p_plot, ap)]

    for name, typ, cfg, dpath, ckpt in selected:
        if typ == 'mmdet' and not _mmdet_available:
            continue

        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

        model = load_model(name, typ, cfg, ckpt, device)
        if model is None:
            print(f"  [fail] 无法加载模型")
            continue

        all_preds = []
        print(f"  Running {len(images)} images...")
        for im in images:
            img_path = f"{args.data}/{args.split}/images/{im['file_name']}"
            try:
                masks, scores = run_model(model, typ, img_path, device)
                all_preds.append((masks, scores))
            except Exception as e:
                print(f"  [warn] {im['file_name']}: {e}")
                all_preds.append((np.zeros((0, 512, 512), dtype=bool), np.array([])))

        r_plot, p_plot, ap = compute_pr_curve(all_preds, all_gts, args.iou_thr, args.n_points)
        results.append((name, r_plot, p_plot, ap))
        print(f"  AP@0.5 = {ap:.4f}")

    if not results:
        print("无结果"); sys.exit(1)

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (name, r_plot, p_plot, ap) in enumerate(results):
        color = COLORS[i % len(COLORS)]
        ax.plot(r_plot, p_plot, '-', color=color, linewidth=2,
                label=f'{name}  (AP={ap:.4f})')
        ax.fill_between(r_plot, 0, p_plot, alpha=0.05, color=color)

    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.2)

    # 图例右下
    ax.legend(loc='lower left', fontsize=11,
              framealpha=0.9, edgecolor='gray', fancybox=True)

    title = f"PR Curves — {args.split} (n={n_use}, IoU={args.iou_thr})"
    ax.set_title(title, fontsize=15, fontweight='bold')

    # AP 排序表（左上角）
    sorted_results = sorted(results, key=lambda x: x[3], reverse=True)
    txt_lines = ['AP@0.5 ranking:']
    for i, (name, _, _, ap) in enumerate(sorted_results):
        marker = '★' if i == 0 else f'{i+1}.'
        txt_lines.append(f'  {marker} {ap:.4f}  {name}')
    ax.text(0.60, 0.55, '\n'.join(txt_lines), fontsize=9, family='monospace',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fafafa',
                       edgecolor='#cccccc', alpha=0.95))

    fig.tight_layout()
    save_path = os.path.join(WORK_DIRS, 'ap_comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n{'='*50}")
    print(f"Saved: {save_path}")
    print(f"\nAP Ranking:")
    for i, (name, _, _, ap) in enumerate(sorted_results):
        print(f"  {i+1}. {name:<20s}  AP={ap:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
