from typing import List

from mmdet.registry import METRICS
from mmdet.evaluation.metrics import CocoMetric


@METRICS.register_module()
class CocoCariesMetric(CocoMetric):

    def __init__(
        self,
        classes: List[str],
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        cat_name2id = {cat['name']: cat_id for cat_id, cat in self._coco_api.cats.items()}
        self.cat_ids = [cat_name2id[cat_name] for cat_name in classes]
