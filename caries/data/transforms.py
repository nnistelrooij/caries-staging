import numpy as np

from mmdet.datasets.transforms import (
    LoadAnnotations,
    PackDetInputs,
)
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadScoreAnnotations(LoadAnnotations):

    def __init__(
        self,
        with_score: bool = True,
        *args,
        **kwargs,
    ):
        self.with_score = with_score

        super().__init__(*args, **kwargs)

    def _load_scores(self, results: dict) -> None:
        """Private function to load score annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded score annotations.
        """
        gt_scores = []
        for instance in results.get('instances', []):
            gt_scores.append(instance['score'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_scores'] = np.array(gt_scores, dtype=np.float32)

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_score:
            self._load_scores(results)
        return results


@TRANSFORMS.register_module()
class PackDetScoreInputs(PackDetInputs):
    
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_scores': 'scores',
    }
