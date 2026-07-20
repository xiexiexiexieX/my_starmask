"""
heads.py — FPN + AnchorGenerator + RPN + BoxUtils + ROIHead（合并，零外部依赖）
================================================================================
Mask R-CNN 后半段完整实现。仅依赖 torch + torchvision.ops。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, box_iou, nms


# ═══════════════════════════════════════════════════════
# Box utils
# ═══════════════════════════════════════════════════════
def masks_to_boxes(masks):
    M = masks.shape[0]
    if M == 0:
        return torch.zeros((0, 4), device=masks.device)
    m = masks > 0.5
    rows = m.any(dim=2); cols = m.any(dim=1)
    boxes = torch.zeros((M, 4), device=masks.device)
    H, W = masks.shape[1], masks.shape[2]
    ar = torch.arange(H, device=masks.device); ac = torch.arange(W, device=masks.device)
    for_empty = ~m.flatten(1).any(dim=1)
    y1 = torch.where(rows, ar[None, :], torch.full_like(ar[None, :], H)).min(dim=1).values
    y2 = torch.where(rows, ar[None, :], torch.full_like(ar[None, :], -1)).max(dim=1).values
    x1 = torch.where(cols, ac[None, :], torch.full_like(ac[None, :], W)).min(dim=1).values
    x2 = torch.where(cols, ac[None, :], torch.full_like(ac[None, :], -1)).max(dim=1).values
    boxes[:, 0] = x1.float(); boxes[:, 1] = y1.float()
    boxes[:, 2] = (x2 + 1).float(); boxes[:, 3] = (y2 + 1).float()
    boxes[for_empty] = 0
    return boxes


def encode_boxes(gt, anchors, weights=(1., 1., 1., 1.)):
    aw = anchors[:, 2] - anchors[:, 0]; ah = anchors[:, 3] - anchors[:, 1]
    ax = anchors[:, 0] + 0.5 * aw; ay = anchors[:, 1] + 0.5 * ah
    gw = gt[:, 2] - gt[:, 0]; gh = gt[:, 3] - gt[:, 1]
    gx = gt[:, 0] + 0.5 * gw; gy = gt[:, 1] + 0.5 * gh
    eps = 1e-6
    aw = aw.clamp(min=eps); ah = ah.clamp(min=eps)
    gw = gw.clamp(min=eps); gh = gh.clamp(min=eps)
    wx, wy, ww, wh = weights
    dx = wx * (gx - ax) / aw; dy = wy * (gy - ay) / ah
    dw = ww * torch.log(gw / aw); dh = wh * torch.log(gh / ah)
    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(deltas, anchors, weights=(1., 1., 1., 1.), max_dwh=4.135):
    aw = anchors[:, 2] - anchors[:, 0]; ah = anchors[:, 3] - anchors[:, 1]
    ax = anchors[:, 0] + 0.5 * aw; ay = anchors[:, 1] + 0.5 * ah
    wx, wy, ww, wh = weights
    dx = deltas[:, 0] / wx; dy = deltas[:, 1] / wy
    dw = (deltas[:, 2] / ww).clamp(max=max_dwh); dh = (deltas[:, 3] / wh).clamp(max=max_dwh)
    cx = dx * aw + ax; cy = dy * ah + ay
    pw = torch.exp(dw) * aw; ph = torch.exp(dh) * ah
    x1 = cx - 0.5 * pw; y1 = cy - 0.5 * ph
    x2 = cx + 0.5 * pw; y2 = cy + 0.5 * ph
    return torch.stack([x1, y1, x2, y2], dim=1)


# ═══════════════════════════════════════════════════════
# FPN
# ═══════════════════════════════════════════════════════
class FPN(nn.Module):
    def __init__(self, in_channels=(64, 128, 256, 512), out_ch=256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_channels])
        self.smooth = nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, padding=1) for _ in in_channels])

    def forward(self, feats):
        laterals = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest')
            laterals[i - 1] = laterals[i - 1] + up
        return [s(l) + l for s, l in zip(self.smooth, laterals)]


# ═══════════════════════════════════════════════════════
# AnchorGenerator
# ═══════════════════════════════════════════════════════
class AnchorGenerator:
    def __init__(self, strides=(2, 4, 8, 16), base_sizes=(4, 8, 16, 32),
                 scales=(1.0, 1.26, 1.587), ratios=(0.5, 1.0, 2.0)):
        self.strides = strides; self.base_sizes = base_sizes
        self.scales = torch.tensor(scales); self.ratios = torch.tensor(ratios)
        self.num_base = len(scales) * len(ratios)

    def _base_anchors(self, base_size):
        h_ratios = torch.sqrt(self.ratios); w_ratios = 1.0 / h_ratios
        ws = (base_size * self.scales[:, None] * w_ratios[None, :]).reshape(-1)
        hs = (base_size * self.scales[:, None] * h_ratios[None, :]).reshape(-1)
        return torch.stack([-ws / 2, -hs / 2, ws / 2, hs / 2], dim=1)

    def grid_anchors(self, feat_sizes, device):
        all_anchors = []
        for lvl, (H, W) in enumerate(feat_sizes):
            stride = self.strides[lvl]
            base = self._base_anchors(self.base_sizes[lvl]).to(device)
            shift_x = (torch.arange(W, device=device) + 0.5) * stride
            shift_y = (torch.arange(H, device=device) + 0.5) * stride
            sy, sx = torch.meshgrid(shift_y, shift_x, indexing='ij')
            shifts = torch.stack([sx.reshape(-1), sy.reshape(-1), sx.reshape(-1), sy.reshape(-1)], dim=1)
            anchors = (base[None, :, :] + shifts[:, None, :]).reshape(-1, 4)
            all_anchors.append(anchors)
        return all_anchors


# ═══════════════════════════════════════════════════════
# RPN
# ═══════════════════════════════════════════════════════
class RPNHead(nn.Module):
    def __init__(self, in_ch=256, num_anchors=15):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.cls = nn.Conv2d(in_ch, num_anchors, 1)
        self.reg = nn.Conv2d(in_ch, num_anchors * 4, 1)
        for layer in [self.conv, self.cls, self.reg]:
            nn.init.normal_(layer.weight, std=0.01); nn.init.constant_(layer.bias, 0)

    def forward(self, feats):
        cls_outs, reg_outs = [], []
        for f in feats:
            t = F.relu(self.conv(f))
            cls_outs.append(self.cls(t)); reg_outs.append(self.reg(t))
        return cls_outs, reg_outs


class RPN(nn.Module):
    def __init__(self, anchor_gen, in_ch=256, pos_iou=0.7, neg_iou=0.3,
                 n_sample=256, pos_frac=0.5,
                 pre_nms_topk=1000, post_nms_topk=256, nms_thr=0.7):
        super().__init__()
        self.anchor_gen = anchor_gen; self.head = RPNHead(in_ch, anchor_gen.num_base)
        self.pos_iou=pos_iou; self.neg_iou=neg_iou; self.n_sample=n_sample
        self.pos_frac=pos_frac; self.pre_nms_topk=pre_nms_topk
        self.post_nms_topk=post_nms_topk; self.nms_thr=nms_thr
        self._cached_anchors = None; self._cached_feat_sizes = None

    def _get_anchors(self, feat_sizes, device):
        key = tuple((h, w) for h, w in feat_sizes)
        if self._cached_feat_sizes != key:
            anchors_per_lvl = self.anchor_gen.grid_anchors(feat_sizes, device)
            self._cached_anchors = torch.cat(anchors_per_lvl, 0)
            self._cached_feat_sizes = key
        return self._cached_anchors.to(device)

    def _cat_levels(self, cls_outs, reg_outs):
        B = cls_outs[0].shape[0]; cls_flat, reg_flat = [], []
        for c, r in zip(cls_outs, reg_outs):
            A = self.anchor_gen.num_base; Hc, Wc = c.shape[-2:]
            c = c.permute(0, 2, 3, 1).reshape(B, Hc * Wc * A, 1)
            r = r.permute(0, 2, 3, 1).reshape(B, Hc * Wc * A, 4)
            cls_flat.append(c); reg_flat.append(r)
        return torch.cat(cls_flat, 1), torch.cat(reg_flat, 1)

    def forward(self, feats, gt_boxes_list, img_size, train=True):
        device = feats[0].device
        feat_sizes = [f.shape[-2:] for f in feats]
        anchors = self._get_anchors(feat_sizes, device)
        cls_outs, reg_outs = self.head(feats)
        cls_flat, reg_flat = self._cat_levels(cls_outs, reg_outs)
        B = cls_flat.shape[0]

        proposals = []
        with torch.no_grad():
            for b in range(B):
                scores = cls_flat[b, :, 0].sigmoid(); deltas = reg_flat[b]
                boxes = decode_boxes(deltas, anchors)
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_size[1])
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_size[0])
                keep_size = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
                boxes, scores = boxes[keep_size], scores[keep_size]
                keep_score = scores > 0.01
                if keep_score.sum() > 0: boxes, scores = boxes[keep_score], scores[keep_score]
                if scores.numel() > self.pre_nms_topk:
                    scores, idx = scores.topk(self.pre_nms_topk); boxes = boxes[idx]
                keep = nms(boxes, scores, self.nms_thr)[:self.post_nms_topk]
                proposals.append(boxes[keep])

        losses = {}
        if train:
            losses = self._loss(cls_flat, reg_flat, anchors, gt_boxes_list)
        return proposals, losses

    def _loss(self, cls_flat, reg_flat, anchors, gt_boxes_list):
        B = cls_flat.shape[0]
        cls_loss_total = cls_flat.new_tensor(0.); reg_loss_total = cls_flat.new_tensor(0.)
        valid_imgs = 0
        for b in range(B):
            gt = gt_boxes_list[b]; logits = cls_flat[b, :, 0]; deltas = reg_flat[b]
            if gt.numel() == 0:
                labels = torch.zeros_like(logits)
                cls_loss_total = cls_loss_total + F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
                valid_imgs += 1; continue
            with torch.no_grad():
                ious = box_iou(anchors, gt)
                max_iou, argmax = ious.max(dim=1)
                labels = torch.full_like(max_iou, -1)
                labels[max_iou < self.neg_iou] = 0
                labels[max_iou >= self.pos_iou] = 1
                gt_best = ious.argmax(dim=0); labels[gt_best] = 1
                del ious
                pos_idx = torch.where(labels == 1)[0]; neg_idx = torch.where(labels == 0)[0]
                n_pos = min(int(self.n_sample * self.pos_frac), pos_idx.numel())
                n_neg = min(self.n_sample - n_pos, neg_idx.numel())
                if pos_idx.numel() > n_pos:
                    pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[:n_pos]]
                if neg_idx.numel() > n_neg:
                    neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[:n_neg]]
                samp = torch.cat([pos_idx, neg_idx])
                samp_labels = torch.cat([torch.ones(pos_idx.numel(), device=samp.device),
                                         torch.zeros(neg_idx.numel(), device=samp.device)])
            cls_loss_total = cls_loss_total + F.binary_cross_entropy_with_logits(logits[samp], samp_labels, reduction='mean')
            if pos_idx.numel() > 0:
                with torch.no_grad():
                    reg_targets = encode_boxes(gt[argmax[pos_idx]], anchors[pos_idx])
                reg_loss_total = reg_loss_total + F.smooth_l1_loss(deltas[pos_idx], reg_targets, beta=1.0/9, reduction='mean')
            valid_imgs += 1
        n = max(1, valid_imgs)
        return {'rpn_cls': cls_loss_total / n, 'rpn_reg': reg_loss_total / n}


# ═══════════════════════════════════════════════════════
# ROIHead
# ═══════════════════════════════════════════════════════
class BoxHead(nn.Module):
    def __init__(self, in_ch=256, roi=7, hidden=1024, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(in_ch * roi * roi, hidden); self.fc2 = nn.Linear(hidden, hidden)
        self.cls = nn.Linear(hidden, num_classes + 1); self.reg = nn.Linear(hidden, num_classes * 4)
        self.num_classes = num_classes

    def forward(self, x):
        x = x.flatten(1); x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.cls(x), self.reg(x)


class MaskHead(nn.Module):
    def __init__(self, in_ch=256, num_classes=1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True))
        self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.pred = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        x = self.convs(x); x = F.relu(self.deconv(x))
        return self.pred(x)


class ROIHead(nn.Module):
    def __init__(self, in_ch=256, num_classes=1, strides=(2, 4, 8, 16),
                 roi_box=7, roi_mask=14, pos_iou=0.5, neg_iou=0.5,
                 n_sample=256, pos_frac=0.25,
                 score_thr=0.3, nms_thr=0.5, max_per_img=200):
        super().__init__()
        self.box_head = BoxHead(in_ch, roi_box, num_classes=num_classes)
        self.mask_head = MaskHead(in_ch, num_classes=num_classes)
        self.strides = strides; self.roi_box = roi_box; self.roi_mask = roi_mask
        self.num_classes = num_classes
        self.pos_iou=pos_iou; self.neg_iou=neg_iou; self.n_sample=n_sample
        self.pos_frac=pos_frac; self.score_thr=score_thr
        self.nms_thr=nms_thr; self.max_per_img=max_per_img

    def _roi_align_multi(self, feats, boxes_with_idx, out_size):
        if boxes_with_idx.shape[0] == 0:
            return feats[0].new_zeros((0, feats[0].shape[1], out_size, out_size))
        boxes = boxes_with_idx[:, 1:]
        w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1); h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1)
        scale = torch.sqrt(w * h)
        target_lvl = torch.floor(2 + torch.log2(scale / 16 + 1e-6)).clamp(0, len(feats)-1).long()
        out = boxes_with_idx.new_zeros((boxes_with_idx.shape[0], feats[0].shape[1], out_size, out_size))
        for lvl in range(len(feats)):
            mask = target_lvl == lvl
            if mask.sum() == 0: continue
            out[mask] = roi_align(feats[lvl], boxes_with_idx[mask],
                                  output_size=(out_size, out_size),
                                  spatial_scale=1.0/self.strides[lvl],
                                  sampling_ratio=2, aligned=True).float()
        return out

    def forward(self, feats, proposals, img_size,
                gt_boxes_list=None, gt_masks_list=None, gt_weights_list=None, train=True):
        if train:
            return self._forward_train(feats, proposals, img_size, gt_boxes_list, gt_masks_list, gt_weights_list)
        return self._forward_test(feats, proposals, img_size)

    def _sample(self, proposals_b, gt_boxes):
        device = proposals_b.device
        cand = torch.cat([proposals_b, gt_boxes], 0) if gt_boxes.numel() > 0 else proposals_b
        if cand.numel() == 0:
            return cand, torch.zeros(0, device=device, dtype=torch.long), torch.zeros(0, device=device, dtype=torch.long)
        if gt_boxes.numel() == 0:
            labels = torch.zeros(cand.shape[0], device=device, dtype=torch.long)
            return cand, labels, torch.zeros(cand.shape[0], device=device, dtype=torch.long)
        with torch.no_grad():
            ious = box_iou(cand, gt_boxes); max_iou, gt_assign = ious.max(dim=1); del ious
            labels = (max_iou >= self.pos_iou).long()
            pos_idx = torch.where(labels == 1)[0]
            neg_idx = torch.where((max_iou < self.neg_iou))[0]
            n_pos = min(int(self.n_sample * self.pos_frac), pos_idx.numel())
            n_neg = min(self.n_sample - n_pos, neg_idx.numel())
            if pos_idx.numel() > n_pos: pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=device)[:n_pos]]
            if neg_idx.numel() > n_neg: neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=device)[:n_neg]]
            keep = torch.cat([pos_idx, neg_idx])
        return cand[keep], labels[keep], gt_assign[keep]

    def _crop_resize_masks(self, masks, boxes, out_size):
        if masks.shape[0] == 0: return masks.new_zeros((0, out_size, out_size))
        m = masks.unsqueeze(1).float()
        idx = torch.arange(boxes.shape[0], device=boxes.device).float().unsqueeze(1)
        rois = torch.cat([idx, boxes], 1)
        out = roi_align(m, rois, output_size=(out_size, out_size), spatial_scale=1.0, sampling_ratio=2, aligned=True)
        return (out[:, 0] >= 0.5).float()

    def _forward_train(self, feats, proposals, img_size, gt_boxes_list, gt_masks_list, gt_weights_list):
        B = len(proposals)
        all_rois, all_labels, all_reg_t = [], [], []
        all_mask_rois, all_mask_t, all_mask_w = [], [], []
        for b in range(B):
            gt_boxes = gt_boxes_list[b]
            samp_box, labels, gt_assign = self._sample(proposals[b], gt_boxes)
            if samp_box.shape[0] == 0: continue
            batch_col = torch.full((samp_box.shape[0], 1), b, device=samp_box.device, dtype=samp_box.dtype)
            all_rois.append(torch.cat([batch_col, samp_box], 1)); all_labels.append(labels)
            reg_t = samp_box.new_zeros((samp_box.shape[0], 4))
            pos = labels == 1
            if pos.sum() > 0 and gt_boxes.numel() > 0:
                reg_t[pos] = encode_boxes(gt_boxes[gt_assign[pos]], samp_box[pos])
            all_reg_t.append(reg_t)
            if pos.sum() > 0 and gt_boxes.numel() > 0:
                pos_box = samp_box[pos]; pos_gt = gt_assign[pos]
                mb = torch.cat([torch.full((pos_box.shape[0], 1), b, device=pos_box.device, dtype=pos_box.dtype), pos_box], 1)
                all_mask_rois.append(mb)
                gtm = gt_masks_list[b][pos_gt]
                tgt = self._crop_resize_masks(gtm, pos_box, self.roi_mask * 2)
                all_mask_t.append(tgt)
                all_mask_w.append(gt_weights_list[b][pos_gt] if gt_weights_list is not None and gt_weights_list[b].numel() > 0
                                  else torch.ones(pos_box.shape[0], device=pos_box.device))

        device = feats[0].device
        losses = {'roi_cls': torch.tensor(0., device=device),
                  'roi_reg': torch.tensor(0., device=device),
                  'roi_mask': torch.tensor(0., device=device)}
        if not all_rois: return losses

        rois = torch.cat(all_rois, 0); labels = torch.cat(all_labels, 0); reg_t = torch.cat(all_reg_t, 0)
        feat_box = self._roi_align_multi(feats, rois, self.roi_box)
        cls_logits, reg_pred = self.box_head(feat_box)
        losses['roi_cls'] = F.cross_entropy(cls_logits, labels)
        pos = labels == 1
        if pos.sum() > 0:
            losses['roi_reg'] = F.smooth_l1_loss(reg_pred[pos], reg_t[pos], beta=1.0, reduction='mean')
        if all_mask_rois:
            mask_rois = torch.cat(all_mask_rois, 0); mask_t = torch.cat(all_mask_t, 0); mask_w = torch.cat(all_mask_w, 0)
            feat_mask = self._roi_align_multi(feats, mask_rois, self.roi_mask)
            mask_pred = self.mask_head(feat_mask)[:, 0]
            bce = F.binary_cross_entropy_with_logits(mask_pred, mask_t, reduction='none').mean(dim=(1, 2))
            w = mask_w.clamp(min=0)
            losses['roi_mask'] = (bce * w).sum() / (w.sum() + 1e-6)
        return losses

    @torch.no_grad()
    def _forward_test(self, feats, proposals, img_size):
        results = []
        for b in range(len(proposals)):
            boxes = proposals[b]
            if boxes.shape[0] == 0:
                results.append(dict(boxes=boxes.new_zeros((0, 4)), scores=boxes.new_zeros((0,)),
                                    masks=boxes.new_zeros((0, img_size[0], img_size[1]))))
                continue
            rois = torch.cat([boxes.new_full((boxes.shape[0], 1), b), boxes], 1)
            feat_box = self._roi_align_multi(feats, rois, self.roi_box)
            cls_logits, reg_pred = self.box_head(feat_box)
            scores = F.softmax(cls_logits, dim=1)[:, 1]
            refined = decode_boxes(reg_pred, boxes)
            refined[:, 0::2] = refined[:, 0::2].clamp(0, img_size[1])
            refined[:, 1::2] = refined[:, 1::2].clamp(0, img_size[0])
            keep = scores >= self.score_thr
            refined, scores = refined[keep], scores[keep]
            if refined.shape[0] == 0:
                results.append(dict(boxes=refined, scores=scores,
                                    masks=refined.new_zeros((0, img_size[0], img_size[1]))))
                continue
            nms_keep = nms(refined, scores, self.nms_thr)[:self.max_per_img]
            refined, scores = refined[nms_keep], scores[nms_keep]
            mrois = torch.cat([refined.new_full((refined.shape[0], 1), b), refined], 1)
            feat_mask = self._roi_align_multi(feats, mrois, self.roi_mask)
            mask_logits = self.mask_head(feat_mask)[:, 0].sigmoid()
            full_masks = self._paste_masks(mask_logits, refined, img_size)
            results.append(dict(boxes=refined, scores=scores, masks=full_masks))
        return results

    def _paste_masks(self, mask_logits, boxes, img_size):
        N = boxes.shape[0]; H, W = img_size
        out = mask_logits.new_zeros((N, H, W))
        for i in range(N):
            x1, y1, x2, y2 = boxes[i].round().long().tolist()
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
            if x2 <= x1 or y2 <= y1: continue
            m = mask_logits[i][None, None]
            m = F.interpolate(m, size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False)[0, 0]
            out[i, y1:y2, x1:x2] = (m >= 0.5).float()
        return out
