"""MAINetRCNN + ResNet-18 — 消融 baseline"""
import torch
import torch.nn as nn
from backbone import ResNetBackbone
from heads import FPN, AnchorGenerator, RPN, ROIHead, masks_to_boxes


class MAINetRCNN(nn.Module):
    def __init__(self, in_chans=1, num_classes=1, fpn_out=256,
                 score_thr=0.3, nms_thr=0.5, max_per_img=200):
        super().__init__()
        self.backbone_type = 'resnet'
        self.backbone = ResNetBackbone(in_chans=in_chans)
        self.strides = (4, 8, 16, 32)
        self.fpn = FPN(in_channels=(64, 128, 256, 512), out_ch=fpn_out)
        anchor_gen = AnchorGenerator(strides=self.strides)
        self.rpn = RPN(anchor_gen, in_ch=fpn_out)
        self.roi_head = ROIHead(in_ch=fpn_out, num_classes=num_classes, strides=self.strides,
                                score_thr=score_thr, nms_thr=nms_thr, max_per_img=max_per_img)
        self.lw = dict(rpn_cls=1.0, rpn_reg=1.0, roi_cls=1.0, roi_reg=1.0, roi_mask=1.0)

    def forward(self, imgs, masks_list=None, weights_list=None,
                gt_params_list=None, param_weight_scale=1.0):
        img_size = imgs.shape[-2:]
        feats, _, _ = self.backbone(imgs)
        feats = self.fpn(feats)
        if self.training:
            gt_boxes_list = [masks_to_boxes(m) for m in masks_list]
            proposals, rpn_losses = self.rpn(feats, gt_boxes_list, img_size, train=True)
            roi_losses = self.roi_head(feats, proposals, img_size,
                                       gt_boxes_list=gt_boxes_list, gt_masks_list=masks_list,
                                       gt_weights_list=weights_list, train=True)
            losses = {**rpn_losses, **roi_losses}
            losses['param_angle'] = imgs.new_tensor(0.0)
            losses['param_gate']  = imgs.new_tensor(0.0)
            losses['param_phi']   = imgs.new_tensor(0.0)
            total = imgs.new_tensor(0.0)
            for k, v in losses.items():
                total = total + self.lw.get(k, 1.0) * v
            losses['total'] = total
            return losses
        else:
            proposals, _ = self.rpn(feats, None, img_size, train=False)
            return self.roi_head(feats, proposals, img_size, train=False)


class RCNNCriterion(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, imgs, masks, wts, gt_params_list=None, param_weight_scale=1.0):
        raw = self.model(imgs, masks, wts)
        return {
            'total': raw['total'], 'class': raw.get('roi_cls', imgs.new_tensor(0.)),
            'bce': raw.get('roi_mask', imgs.new_tensor(0.)), 'dice': raw.get('roi_mask', imgs.new_tensor(0.)),
            'no_obj': raw.get('rpn_cls', imgs.new_tensor(0.)),
            'param_angle': imgs.new_tensor(0.), 'param_gate': imgs.new_tensor(0.), 'param_phi': imgs.new_tensor(0.),
            'rpn_cls': raw.get('rpn_cls', imgs.new_tensor(0.)), 'rpn_reg': raw.get('rpn_reg', imgs.new_tensor(0.)),
            'roi_cls': raw.get('roi_cls', imgs.new_tensor(0.)), 'roi_reg': raw.get('roi_reg', imgs.new_tensor(0.)),
            'roi_mask': raw.get('roi_mask', imgs.new_tensor(0.)),
        }
