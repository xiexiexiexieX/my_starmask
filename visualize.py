"""
Interactive visualization for MAINet and MMDetection checkpoints.

Real mixed checkpoints:
  python visualize.py --ckpt work_dirs\real_mixed_baselines\mainet_v4 --data output_mix --split test
  python visualize.py --ckpt work_dirs\real_mixed_baselines\mmdet\mask_rcnn --data output_mix --split test
  python visualize.py --ckpt work_dirs\real_mixed_baselines\mmdet\condinst --data output_mix --split test
  python visualize.py --ckpt work_dirs\real_mixed_baselines\mmdet\mask2former --data output_mix --split test
  python visualize.py --ckpt work_dirs\real_mixed_baselines\yolo --data output_mix --split test

Morphology filtering:
  --mode streak
  --mode point

Moved dataset example:
  python visualize.py --ckpt E:\codes\query_mask\work_dirs\real_mixed_baselines\mainet_v4 --data E:\codes\query_mask\output_mix --split test

Notes:
  - Use the mmdet environment for MMDetection and YOLO checkpoints.
  - Paths containing mmdet are loaded with MMDetection.
  - Keys: Space next, b previous, s save, Esc quit.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.optimize import linear_sum_assignment

from eval import (
    primary_component_centroid,
    raw_mask_centroid,
)

PREPROCESSED = {'real_asinh_0_1', 'real_zscale_0_1', 'fits_zscale_0_1', 'bmp_asinh_0_1'}
COLORS = np.array([
    [1.00, 0.82, 0.10],
    [0.00, 0.85, 1.00],
    [0.85, 0.35, 1.00],
    [0.25, 1.00, 0.35],
    [1.00, 0.45, 0.15],
    [0.40, 0.60, 1.00],
    [1.00, 0.25, 0.55],
], dtype=np.float32)


#      checkpoint                                                                                   
def _find_ckpt(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        # best_model.pt / best.pt (   )
        for name in ['best_model.pt', 'best.pt']:
            p = os.path.join(path, name)
            if os.path.exists(p):
                return p
        yolo_best = os.path.join(path, 'weights', 'best.pt')
        if os.path.exists(yolo_best):
            return yolo_best
        # best_coco (mmdet)
        best = sorted([f for f in os.listdir(path) if f.startswith('best_coco')], reverse=True)
        if best:
            return os.path.join(path, best[0])
        # epoch    
        ep = sorted([f for f in os.listdir(path)
                     if (f.endswith('.pt') or f.endswith('.pth')) and 'epoch' in f],
                    key=lambda x: int(x.split('_')[-1].replace('.pt','').replace('.pth','')), reverse=True)
        if ep:
            return os.path.join(path, ep[0])
    return None


#                                                                                                              
def normalize_astro(img):
    bg = np.median(img)
    img_sub = img - bg
    bg_mask = img < np.percentile(img, 90)
    scale = np.std(img_sub[bg_mask]) + 1e-6
    a = np.arcsinh(img_sub / scale)
    return ((a - a.mean()) / (a.std() + 1e-6)).astype(np.float32)


def normalize_image_for_display(img, already_stretched=False):
    img = img.astype(np.float32)
    if already_stretched:
        img = np.where(np.isfinite(img), img, 0.0)
        return np.clip(img, 0.0, 1.0).astype(np.float32)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img, dtype=np.float32)
    vals = img[finite]
    lo, hi = np.percentile(vals, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32)


def model_input_image(img, img_info):
    if img_info.get('preprocess') in PREPROCESSED:
        img = np.where(np.isfinite(img), img, 0.0)
        return np.clip(img, 0.0, 1.0).astype(np.float32)
    return normalize_astro(img)


def mask_rgb(masks, shape):
    h, w = shape
    if len(masks) == 0:
        return np.zeros((h, w, 3), dtype=np.float32), np.zeros((h, w), dtype=bool)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    coverage = np.asarray(masks).astype(bool).sum(axis=0)
    for i, mask in enumerate(np.asarray(masks).astype(bool)):
        color = COLORS[i % len(COLORS)]
        rgb[mask] = 0.35 * rgb[mask] + 0.65 * color
    overlap = coverage > 1
    rgb[overlap] = [1.0, 1.0, 1.0]
    return rgb, overlap


def mask_iou(p, g):
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return inter / union if union > 0 else 0.0


def mask_centroid(m):
    """Return the centroid of the largest 8-connected mask component."""
    return primary_component_centroid(m)


def evaluate_frame(pred_masks, gt_masks, iou_thr=0.5):
    K, M = len(pred_masks), len(gt_masks)
    if K == 0 or M == 0:
        return dict(tp=0, fp=K, fn=M, precision=0., recall=0., f1=0.,
                    mean_iou=float('nan'), centroid_mse=float('nan'))

    iou_mat = np.zeros((K, M))
    for i in range(K):
        for j in range(M):
            iou_mat[i, j] = mask_iou(pred_masks[i], gt_masks[j])
    pi, gj = linear_sum_assignment(-iou_mat)
    pairs = [(p, g, iou_mat[p, g]) for p, g in zip(pi, gj) if iou_mat[p, g] >= iou_thr]
    tp = len(pairs)

    ious = [i for _, _, i in pairs]
    sq_errs = []
    for p, g, _ in pairs:
        pc = mask_centroid(pred_masks[p])
        gc = raw_mask_centroid(gt_masks[g])
        if pc is not None and gc is not None:
            sq_errs.append((pc[0] - gc[0]) ** 2 + (pc[1] - gc[1]) ** 2)

    prec = tp / K if K > 0 else 0.
    rec = tp / M if M > 0 else 0.
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.
    return dict(tp=tp, fp=K - tp, fn=M - tp, precision=prec, recall=rec, f1=f1,
                mean_iou=np.mean(ious) if ious else float('nan'),
                centroid_mse=np.mean(sq_errs) if sq_errs else float('nan'))


#                                                                                                              
class StarDataset:
    def __init__(self, data_root, split, mode=None):
        with open(f"{data_root}/annotations/{split}.json", encoding='utf-8-sig') as f:
            self.coco = json.load(f)
        self.images = self.coco['images']
        if mode:
            mode = str(mode).strip().lower()
            self.images = [
                image for image in self.images
                if str(image.get('mode') or image.get('source_dataset') or
                       image.get('target_type') or '').strip().lower() == mode
            ]
            if not self.images:
                raise ValueError(
                    f'No images with mode={mode!r} in '
                    f'{data_root}/annotations/{split}.json')
        self.img_dir = f"{data_root}/{split}/images"
        self.mask_dir = f"{data_root}/{split}/masks"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx]
        iid = im['id']
        img_path = f"{self.img_dir}/{im['file_name']}"
        mask_file = f"{self.mask_dir}/{iid:06d}_masks.npy"
        gt_masks = np.load(mask_file).astype(bool) if os.path.exists(mask_file) else np.zeros((0, 512, 512), dtype=bool)
        return img_path, iid, gt_masks, im


#          ?                                                                                                
class Visualizer:
    def __init__(self, model, model_name, model_type, dataset, device,
                 score_thr=0.3, iou_thr=0.5, save_dir='vis_out'):
        self.model = model
        self.model_name = model_name
        self.model_type = model_type  # 'local', 'mmdet', or 'yolo'
        self.dataset = dataset
        self.device = device
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.save_dir = save_dir
        self.idx = 0
        self.total = len(dataset)
        os.makedirs(save_dir, exist_ok=True)

        self.fig, self.axes = plt.subplots(2, 3, figsize=(24, 12))
        self.fig.subplots_adjust(bottom=0.10)
        self.axes = self.axes.flatten()
        self.axes[5].axis('off')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self._init_buttons()
        print(f"Model: {model_name} [{model_type}] | Images: {self.total} | Score_thr={score_thr}")
        print("Keys: Space/right=next, b/left=prev, s=save, Esc=quit. Buttons are also available.")
        self.update()
        plt.show(block=False)

    def _init_buttons(self):
        self.button_axes = [
            self.fig.add_axes([0.34, 0.02, 0.07, 0.04]),
            self.fig.add_axes([0.42, 0.02, 0.07, 0.04]),
            self.fig.add_axes([0.50, 0.02, 0.07, 0.04]),
            self.fig.add_axes([0.58, 0.02, 0.07, 0.04]),
        ]
        self.buttons = [
            Button(self.button_axes[0], 'Prev'),
            Button(self.button_axes[1], 'Next'),
            Button(self.button_axes[2], 'Save'),
            Button(self.button_axes[3], 'Quit'),
        ]
        self.buttons[0].on_clicked(lambda event: self.prev())
        self.buttons[1].on_clicked(lambda event: self.next())
        self.buttons[2].on_clicked(lambda event: self.save_current())
        self.buttons[3].on_clicked(lambda event: plt.close(self.fig))

    @torch.no_grad()
    def infer(self, img_path, img_np, img_info):
        if self.model_type == 'mmdet':
            from mmdet.apis import inference_detector
            result = inference_detector(self.model, img_path)
            inst = result.pred_instances
            keep = inst.scores > self.score_thr
            masks = inst.masks[keep].cpu().numpy().astype(bool)
            scores = inst.scores[keep].cpu().numpy()
        elif self.model_type == 'yolo':
            from yolo.prepare_dataset import npy_to_yolo_rgb, yolo_mask_nms_indices
            rgb = npy_to_yolo_rgb(img_np, img_info)
            result = self.model.predict(
                source=rgb,
                imgsz=1024,
                conf=0.001,
                iou=0.80,
                max_det=50,
                retina_masks=True,
                verbose=False,
                device=0 if self.device.type == 'cuda' else 'cpu')[0]
            if result.masks is None or result.boxes is None:
                masks = np.zeros((0, rgb.shape[0], rgb.shape[1]), dtype=bool)
                scores = np.zeros((0,), dtype=np.float32)
            else:
                masks = result.masks.data.cpu().numpy() > 0.5
                scores = result.boxes.conf.cpu().numpy()
                keep = yolo_mask_nms_indices(masks, scores, iou_thr=0.50, max_det=50)
                masks, scores = masks[keep], scores[keep]
            keep = scores >= self.score_thr
            masks, scores = masks[keep], scores[keep]
        elif self.model_type in ('local-rcnn', 'local-query'):
            x = torch.from_numpy(model_input_image(img_np, img_info))[None, None].to(self.device)
            output = self.model(x)
            if self.model_type == 'local-rcnn':
                # RCNN: output is list of dict per image
                results = output[0]
                masks = results['masks'].cpu().numpy().astype(bool)
                scores = results['scores'].cpu().numpy()
            else:
                # Query: output is dict with pred_masks/pred_logits
                pred_masks = output['pred_masks'][0].cpu()              # [N,256,256]
                pred_logits = output['pred_logits'][0].cpu()            # [N,2]
                scores = torch.softmax(pred_logits, dim=-1)[:, 1].numpy()  # foreground prob
                #        512x512
                pred_masks = F.interpolate(
                    pred_masks[None].float(), size=(512, 512), mode='bilinear',
                    align_corners=False)[0] > 0.5
                masks = pred_masks.numpy().astype(bool)
            keep = scores >= self.score_thr
            masks, scores = masks[keep], scores[keep]
        return masks, scores

    def get_sample(self, idx):
        img_path, iid, gt_masks, img_info = self.dataset[idx]
        img_np = np.load(img_path).astype(np.float32)
        pred_masks, pred_scores = self.infer(img_path, img_np, img_info)
        metrics = evaluate_frame(list(pred_masks), list(gt_masks), self.iou_thr)
        return img_np, iid, gt_masks, pred_masks, pred_scores, metrics, img_info

    def update(self):
        idx = self.idx % self.total
        img_np, iid, gt_masks, pred_masks, pred_scores, metrics, img_info = self.get_sample(idx)
        H, W = img_np.shape

        for ax in self.axes:
            ax.clear(); ax.axis('off')

        already_stretched = img_info.get('preprocess') in PREPROCESSED
        img_show = normalize_image_for_display(img_np, already_stretched=already_stretched)

        self.axes[0].imshow(img_show, cmap='gray', origin='upper')
        self.axes[0].set_title(f'Image (N_gt={len(gt_masks)})', fontsize=11)

        gt_rgb, gt_overlap = mask_rgb(gt_masks, (H, W))
        self.axes[1].imshow(gt_rgb, origin='upper')
        self.axes[1].set_title(f'GT Masks (overlap={int(gt_overlap.sum())})', fontsize=11)

        self.axes[2].text(0.02, 0.98, self._fmt(metrics, pred_scores, img_info),
                          va='top', ha='left', fontsize=8, family='monospace',
                          transform=self.axes[2].transAxes,
                          bbox=dict(facecolor='#f5f5f5', alpha=0.95, edgecolor='gray'))

        n_pred = len(pred_masks)
        pred_rgb, pred_overlap = mask_rgb(pred_masks, (H, W))
        pred_overlay = np.dstack([img_show] * 3)
        pred_overlay = np.clip(0.55 * pred_overlay + 0.75 * pred_rgb, 0.0, 1.0)
        pred_overlay[pred_overlap] = [1.0, 1.0, 1.0]
        self.axes[3].imshow(pred_overlay, origin='upper')
        self.axes[3].set_title(f'Pred Overlay (K={n_pred})', fontsize=11)

        self.axes[4].imshow(pred_rgb, origin='upper')
        self.axes[4].set_title(f'Pred Masks (overlap={int(pred_overlap.sum())})', fontsize=11)

        self.fig.suptitle(f"[{idx+1}/{self.total}] id={iid}  {self.model_name}  thr={self.score_thr}",
                          fontsize=12, y=0.97)
        self.fig.canvas.draw(); self.fig.canvas.flush_events()

    def _fmt(self, m, scores, img_info):
        mi = f"{m['mean_iou']:.4f}" if not np.isnan(m['mean_iou']) else "nan"
        ms = f"{m['centroid_mse']:.4f}" if not np.isnan(m['centroid_mse']) else "nan"
        mode = img_info.get('mode', 'unknown')
        txt = (f"Model: {self.model_name}\nMode: {mode}\n-- Global --\n"
               f"Prec={m['precision']:.3f} Rec={m['recall']:.3f} F1={m['f1']:.3f}\n"
               f"IoU={mi} MSE={ms}\nTP={m['tp']} FP={m['fp']} FN={m['fn']}\n")
        if len(scores) > 0:
            txt += f"Score: min={scores.min():.2f} max={scores.max():.2f}\n"
        return txt

    def on_key(self, event):
        if event.key in [' ', 'right']:
            self.next()
        elif event.key in ['b', 'left']:
            self.prev()
        elif event.key == 'escape':
            plt.close(self.fig)
        elif event.key == 's':
            self.save_current()

    def next(self):
        self.idx += 1
        self.update()

    def prev(self):
        self.idx -= 1
        self.update()

    def save_current(self):
        path = os.path.join(self.save_dir, f"vis_{self.idx % self.total:04d}.png")
        self.fig.savefig(path, bbox_inches='tight', dpi=130)
        print(f"Saved: {path}")

    def run(self):
        while plt.fignum_exists(self.fig.number):
            self.fig.canvas.start_event_loop(timeout=0.05)


#                                                                                                         
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_variant(dir_name):
    """                key    mainet_rcnn_v2  ?v2, v3  ?v3"""
    name = dir_name.replace('mainet_rcnn_', '').replace('mainet_', '')
    return name


def _import_rcnn_model(variant):
    """ ?mainet/<variant>/     MAINetRCNN"""
    model_dir = os.path.join(PROJECT_ROOT, 'mainet', variant)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    for mod in list(sys.modules):
        if mod in ('model', 'backbone', 'heads', 'dataset', 'param_loss'):
            sys.modules.pop(mod, None)
    import model as _m
    return _m.MAINetRCNN, _m.RCNNCriterion


#                                                                                                              
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='       (work_dirs/xxx)  ?.pt    ')
    ap.add_argument('--data', default='output')
    ap.add_argument('--split', default='test')
    ap.add_argument('--score_thr', type=float, default=0.3)
    ap.add_argument('--iou_thr', type=float, default=0.5)
    ap.add_argument('--vis_dir', default='vis_out')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--mode', choices=('point', 'streak'), default=None,
                    help='Only visualize one morphology; default uses all images')
    args = ap.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f"Device: {device}")

    ckpt_path = _find_ckpt(args.ckpt)
    if ckpt_path is None:
        print(f"    ?checkpoint: {args.ckpt}"); sys.exit(1)
    print(f"Checkpoint: {ckpt_path}")

    #                    
    dir_name = os.path.basename(os.path.dirname(ckpt_path)
                                if os.path.isfile(ckpt_path) else ckpt_path)

    ckpt_norm = args.ckpt.replace('\\', '/').strip('/')
    is_yolo = ('/yolo/' in f'/{ckpt_norm}/' or
               os.path.basename(os.path.dirname(ckpt_path)) == 'weights')
    if is_yolo:
        try:
            from ultralytics import YOLO
        except ImportError:
            print('ultralytics is not installed: python -m pip install ultralytics==8.3.0')
            sys.exit(1)
        from yolo.prepare_dataset import require_yolo_segmentation, yolo_model_display_name
        model = YOLO(ckpt_path)
        try:
            require_yolo_segmentation(model)
        except ValueError as exc:
            print(f'Invalid YOLO checkpoint: {exc}')
            sys.exit(1)
        model_type = 'yolo'
        model_name = yolo_model_display_name(model)
    elif '_star' in args.ckpt or '_star' in dir_name or '/mmdet/' in ckpt_norm or ckpt_norm.startswith('mmdet/'):
        # mmdet    
        try:
            from mmdet.apis import init_detector
        except ImportError:
            print("mmdet              mmdet    : conda activate mmdet")
            sys.exit(1)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mmdet'))
        model_name = dir_name.replace('_star', '')
        base_dir = os.path.dirname(ckpt_path) if os.path.isfile(ckpt_path) else args.ckpt
        runtime_config = os.path.join(base_dir, '_runtime_config.py')
        project_config = os.path.join(
            os.path.dirname(__file__), 'mmdet', 'configs_real_mixed',
            f'{model_name}_star.py')
        # The project Mask2Former config includes the local mask-NMS head.
        # Use it for old checkpoints too; the head adds no learned weights.
        config = project_config if model_name == 'mask2former' else runtime_config
        if not os.path.exists(config):
            config = runtime_config if os.path.exists(runtime_config) else project_config
        if not os.path.exists(config):
            config = os.path.join(os.path.dirname(__file__), 'mmdet', 'configs', f'{model_name}_star.py')
        if not os.path.exists(config):
            print(f"       ? {config}"); sys.exit(1)
        model = init_detector(config, ckpt_path, device=device)
        model.eval()
        model_type = 'mmdet'
        model_name = model_name.replace('_', ' ').title().replace(' ', '')
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        cfg_dict = ckpt.get('cfg_dict', {})
        mt = cfg_dict.get('model_type', '')

        has_roi = any('roi_head' in k for k in sd.keys())
        has_pixel_decoder = any('pixel_decoder' in k for k in sd.keys())

        if has_roi:
            variant = _resolve_variant(dir_name)
            MAINetRCNN, _ = _import_rcnn_model(variant)
            model = MAINetRCNN(
                in_chans=cfg_dict.get('in_chans', 1),
                num_classes=cfg_dict.get('num_classes', 1)).to(device)

            missing, unexpected = model.load_state_dict(sd, strict=False)
            model.eval()
            model_type = 'local-rcnn'
            model_name = f"MAINet-{variant.upper()}"
            if missing:
                print(f"  [warn] {len(missing)} missing keys")
            if unexpected:
                print(f"  [warn] {len(unexpected)} unexpected keys")
        elif has_pixel_decoder:
            # Query-based MAINet
            from mainet.query_head.mainet import MAINet
            model = MAINet(
                in_chans=cfg_dict.get('in_chans', 1),
                num_queries=cfg_dict.get('num_queries', 100),
                num_classes=cfg_dict.get('num_classes', 1),
                d_model=cfg_dict.get('d_model', 128)).to(device)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            model.eval()
            model_type = 'local-query'
            model_name = f"MAINet-Q"
            if missing:
                print(f"  [warn] {len(missing)} missing keys (architecture drift)")
            if unexpected:
                print(f"  [warn] {len(unexpected)} unexpected keys")
        else:
            print(f"  [error] Cannot determine model architecture from state_dict keys")
            sys.exit(1)

    dataset = StarDataset(args.data, args.split, mode=args.mode)
    viewer = Visualizer(model, model_name, model_type, dataset, device,
                        score_thr=args.score_thr, iou_thr=args.iou_thr,
                        save_dir=args.vis_dir)
    viewer.run()
