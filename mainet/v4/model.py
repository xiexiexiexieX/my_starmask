"""
MAINetRCNN v4 - PSF-gated residual + Deformable Orientation Strip + residual fusion.
================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import DualPathBackbone
from heads import FPN, AnchorGenerator, RPN, ROIHead, masks_to_boxes


class MAINetRCNN(nn.Module):
    def __init__(self, in_chans=1, num_classes=1, fpn_out=256,
                 score_thr=0.3, nms_thr=0.5, max_per_img=200,
                 gate_pos_weight=1.0, gate_loss_weight=0.1):
        super().__init__()
        self.backbone_type = 'dual_path_v4_deform'
        self.backbone = DualPathBackbone(in_chans=in_chans)
        self.strides = (2, 4, 8, 16)
        self.fpn = FPN(in_channels=(64, 128, 256, 512), out_ch=fpn_out)
        anchor_gen = AnchorGenerator(strides=self.strides)
        self.rpn = RPN(anchor_gen, in_ch=fpn_out)
        self.roi_head = ROIHead(in_ch=fpn_out, num_classes=num_classes, strides=self.strides,
                                score_thr=score_thr, nms_thr=nms_thr, max_per_img=max_per_img)
        self.gate_pos_weight = float(gate_pos_weight)
        self.lw = dict(rpn_cls=1.0, rpn_reg=1.0, roi_cls=1.0,
                       roi_reg=1.0, roi_mask=1.0,
                       param_gate=float(gate_loss_weight))

    def forward(self, imgs, masks_list=None, weights_list=None,
                gt_params_list=None, param_weight_scale=1.0,
                return_predictions=False):
        img_size = imgs.shape[-2:]
        feats, aux, _ = self.backbone(imgs)
        feats = self.fpn(feats)

        if self.training:
            gt_boxes_list = [masks_to_boxes(m) for m in masks_list]
            proposals, rpn_losses = self.rpn(feats, gt_boxes_list, img_size, train=True)
            roi_losses = self.roi_head(feats, proposals, img_size,
                                       gt_boxes_list=gt_boxes_list, gt_masks_list=masks_list,
                                       gt_weights_list=weights_list, train=True)
            losses = {**rpn_losses, **roi_losses}
            losses['param_angle'] = imgs.new_tensor(0.0)
            losses['param_phi']   = imgs.new_tensor(0.0)
            gate_logit = aux.get('gate_logit') if aux else None
            gate_indices, gate_targets = [], []
            for index, params in enumerate(gt_params_list or []):
                if params is not None and 'gate_target' in params:
                    gate_indices.append(index)
                    gate_targets.append(params['gate_target'].to(imgs.device))
            if gate_logit is not None and gate_targets:
                targets = torch.stack(gate_targets).to(gate_logit.dtype)
                losses['param_gate'] = F.binary_cross_entropy_with_logits(
                    gate_logit[gate_indices], targets,
                    pos_weight=gate_logit.new_tensor(self.gate_pos_weight))
            else:
                losses['param_gate'] = imgs.new_tensor(0.0)
            total = imgs.new_tensor(0.0)
            for k, v in losses.items():
                total = total + self.lw.get(k, 1.0) * v
            losses['total'] = total
            if gate_logit is not None and gate_targets:
                losses['_gate_probs'] = torch.sigmoid(
                    gate_logit[gate_indices].detach())
                losses['_gate_targets'] = targets.detach()
            if return_predictions:
                eval_proposals, _ = self.rpn(
                    feats, None, img_size, train=False)
                predictions = self.roi_head(
                    feats, eval_proposals, img_size, train=False)
                return losses, predictions
            return losses
        else:
            proposals, _ = self.rpn(feats, None, img_size, train=False)
            return self.roi_head(feats, proposals, img_size, train=False)


class RCNNCriterion(nn.Module):
    """适配训练循环的 criterion 接口"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, imgs, masks_list, weights_list,
                gt_params_list=None, param_weight_scale=1.0,
                return_predictions=False):
        raw = self.model(
            imgs, masks_list, weights_list,
            gt_params_list=gt_params_list,
            param_weight_scale=param_weight_scale,
            return_predictions=return_predictions)
        predictions = None
        if return_predictions:
            raw, predictions = raw
        formatted = {
            'total':  raw['total'],
            'class':  raw.get('roi_cls', imgs.new_tensor(0.)),
            'bce':    raw.get('roi_mask', imgs.new_tensor(0.)),
            'dice':   raw.get('roi_mask', imgs.new_tensor(0.)),
            'no_obj': raw.get('rpn_cls', imgs.new_tensor(0.)),
            'param_angle': raw.get('param_angle', imgs.new_tensor(0.)),
            'param_gate':  raw.get('param_gate', imgs.new_tensor(0.)),
            'param_phi':   raw.get('param_phi', imgs.new_tensor(0.)),
            'rpn_cls': raw.get('rpn_cls', imgs.new_tensor(0.)),
            'rpn_reg': raw.get('rpn_reg', imgs.new_tensor(0.)),
            'roi_cls': raw.get('roi_cls', imgs.new_tensor(0.)),
            'roi_reg': raw.get('roi_reg', imgs.new_tensor(0.)),
            'roi_mask': raw.get('roi_mask', imgs.new_tensor(0.)),
        }
        if '_gate_probs' in raw:
            formatted['_gate_probs'] = raw['_gate_probs']
            formatted['_gate_targets'] = raw['_gate_targets']
        return (formatted, predictions) if return_predictions else formatted
