"""COCO metric that reports a truthful zero score for all-empty predictions."""

from collections import OrderedDict

from mmengine.logging import MMLogger

from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class EmptySafeCocoMetric(CocoMetric):
    """Treat an all-background validation result as AP=0, not a metric error."""

    def compute_metrics(self, results):
        if any(len(pred.get('scores', ())) for _, pred in results):
            return super().compute_metrics(results)

        logger = MMLogger.get_current_instance()
        logger.warning('All validation predictions are empty; reporting COCO AP as 0.0.')
        metrics = OrderedDict()
        for metric in self.metrics:
            if metric in ('bbox', 'segm'):
                for item in ('mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'):
                    metrics[f'{metric}_{item}'] = 0.0
        return metrics
