"""Project-local Mask2Former instance post-processing for stellar masks.

The stock MMDetection MaskFormer fusion head returns top query/class pairs for
instance segmentation without an instance-level de-duplication step.  For
tiny stellar objects, several queries can converge on one star and become
high-scoring false positives.  This head keeps the original scoring rule and
adds conservative mask NMS at inference/validation time only.
"""

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData

from mmdet.models.seg_heads.panoptic_fusion_heads.maskformer_fusion_head import (
    MaskFormerFusionHead,
)
from mmdet.registry import MODELS
from mmdet.structures.mask import mask2bbox


@MODELS.register_module()
class StarMaskFormerFusionHead(MaskFormerFusionHead):
    """MaskFormer instance head with score filtering and conservative mask NMS."""

    def instance_postprocess(self, mask_cls, mask_pred):
        """Return de-duplicated instance masks for one image.

        The real-data config uses ``mask_nms_iou_thr=0.50``.  Its GT masks
        overlap far less than this, so it removes duplicate queries while
        retaining physically distinct stars.
        """
        max_per_image = int(self.test_cfg.get('max_per_image', 100))
        pre_nms_topk = int(self.test_cfg.get('pre_nms_topk', max_per_image))
        score_thr = float(self.test_cfg.get('score_thr', 0.001))
        nms_iou_thr = float(self.test_cfg.get('mask_nms_iou_thr', 0.80))
        min_mask_pixels = int(self.test_cfg.get('min_mask_pixels', 4))
        class_agnostic_nms = bool(
            self.test_cfg.get('class_agnostic_mask_nms', True))

        num_queries = mask_cls.shape[0]
        class_scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(
            self.num_classes, device=mask_cls.device).unsqueeze(0).repeat(
                num_queries, 1).flatten(0, 1)

        topk = min(pre_nms_topk, class_scores.numel())
        scores, top_indices = class_scores.flatten(0, 1).topk(
            topk, sorted=True)
        labels = labels[top_indices]
        query_indices = top_indices // self.num_classes
        masks = mask_pred[query_indices]

        # This project uses thing classes only, but retain the base head's
        # behavior if a config later introduces stuff classes.
        is_thing = labels < self.num_things_classes
        scores = scores[is_thing]
        labels = labels[is_thing]
        masks = masks[is_thing]

        masks_binary = masks > 0
        mask_areas = masks_binary.flatten(1).sum(1)
        mask_quality = (masks.sigmoid() * masks_binary).flatten(1).sum(1)
        mask_quality = mask_quality / (mask_areas + 1e-6)
        det_scores = scores * mask_quality

        valid = (det_scores >= score_thr) & (mask_areas >= min_mask_pixels)
        det_scores = det_scores[valid]
        labels = labels[valid]
        masks_binary = masks_binary[valid]

        keep = self._mask_nms(
            masks_binary, det_scores, labels, max_per_image, nms_iou_thr,
            class_agnostic_nms)
        masks_binary = masks_binary[keep]
        det_scores = det_scores[keep]
        labels = labels[keep]

        results = InstanceData()
        results.bboxes = mask2bbox(masks_binary)
        results.labels = labels
        results.scores = det_scores
        results.masks = masks_binary
        return results

    @staticmethod
    def _mask_nms(masks, scores, labels, max_per_image, iou_thr,
                  class_agnostic):
        """Greedy NMS over binary masks, returning score-descending indices."""
        if masks.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=masks.device)

        order = scores.argsort(descending=True)
        kept = []
        flat_masks = masks.flatten(1)

        for candidate in order:
            if len(kept) >= max_per_image:
                break

            if kept:
                kept_indices = torch.as_tensor(
                    kept, dtype=torch.long, device=masks.device)
                overlaps = (flat_masks[kept_indices] & flat_masks[candidate]).sum(1)
                unions = (flat_masks[kept_indices] | flat_masks[candidate]).sum(1)
                ious = overlaps.float() / (unions.float() + 1e-6)
                if not class_agnostic:
                    same_label = labels[kept_indices] == labels[candidate]
                    ious = ious * same_label.float()
                if torch.any(ious >= iou_thr):
                    continue

            kept.append(int(candidate))

        return torch.as_tensor(kept, dtype=torch.long, device=masks.device)
