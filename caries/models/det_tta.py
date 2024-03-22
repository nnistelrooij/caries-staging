# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmcv.ops import batched_nms
from mmengine.registry import MODELS
from mmengine.structures import InstanceData
import torch

from mmdet.models import DetTTAModel
from mmdet.structures import DetDataSample


@MODELS.register_module()
class InstSegTTAModel(DetTTAModel):
    """Merge augmented detection results, only bboxes corresponding score under
    flipping and multi-scale resizing can be processed now.

    Examples:
        >>> tta_model = dict(
        >>>     type='DetTTAModel',
        >>>     tta_cfg=dict(nms=dict(
        >>>                     type='nms',
        >>>                     iou_threshold=0.5),
        >>>                     max_per_img=100))
        >>>
        >>> tta_pipeline = [
        >>>     dict(type='LoadImageFromFile',
        >>>          file_client_args=dict(backend='disk')),
        >>>     dict(
        >>>         type='TestTimeAug',
        >>>         transforms=[[
        >>>             dict(type='Resize',
        >>>                  scale=(1333, 800),
        >>>                  keep_ratio=True),
        >>>         ], [
        >>>             dict(type='RandomFlip', prob=1.),
        >>>             dict(type='RandomFlip', prob=0.)
        >>>         ], [
        >>>             dict(
        >>>                 type='PackDetInputs',
        >>>                 meta_keys=('img_id', 'img_path', 'ori_shape',
        >>>                         'img_shape', 'scale_factor', 'flip',
        >>>                         'flip_direction'))
        >>>         ]])]
    """

    def merge_preds(self, data_samples_list: List[List[DetDataSample]]):
        """Merge batch predictions of enhanced data.

        Args:
            data_samples_list (List[List[DetDataSample]]): List of predictions
                of all enhanced data. The outer list indicates images, and the
                inner list corresponds to the different views of one image.
                Each element of the inner list is a ``DetDataSample``.
        Returns:
            List[DetDataSample]: Merged batch prediction.
        """
        merged_data_samples = []
        for data_samples in data_samples_list:
            merged_data_samples.append(self._merge_single_sample(data_samples))
        return merged_data_samples    

    def _merge_single_sample(
            self, data_samples: List[DetDataSample]) -> DetDataSample:
        """Merge predictions which come form the different views of one image
        to one prediction.

        Args:
            data_samples_list (List[DetDataSample]): List of predictions
            of enhanced data which come form one image.
        Returns:
            List[DetDataSample]: Merged prediction.
        """
        aug_bboxes = []
        aug_scores = []
        aug_progs = []
        aug_probs = []
        aug_labels = []
        aug_masks = []
        img_metas = []
        convert_to_cpu = True
        for data_sample in data_samples:
            masks = data_sample.pred_instances.masks
            x1 = (masks.sum(dim=1) > 0).int().argmax(dim=-1)
            y1 = (masks.sum(dim=2) > 0).int().argmax(dim=-1)
            x2 = masks.shape[2] - (torch.flip(masks, dims=(2,)).sum(dim=1) > 0).int().argmax(dim=-1)
            y2 = masks.shape[1] - (torch.flip(masks, dims=(1,)).sum(dim=2) > 0).int().argmax(dim=-1)
            bboxes = torch.column_stack((x1, y1, x2, y2))

            if bboxes.device == 'cpu':
                convert_to_cpu = True

            aug_bboxes.append(bboxes.float())
            aug_scores.append(data_sample.pred_instances.scores)
            if not hasattr(data_sample.pred_instances, 'progs'):
                aug_progs.append(torch.zeros_like(data_sample.pred_instances.scores))
                aug_probs.append(torch.zeros_like(data_sample.pred_instances.scores))
            else:
                aug_progs.append(data_sample.pred_instances.progs)
                aug_probs.append(data_sample.pred_instances.probs)
            img_metas.append(data_sample.metainfo)

            if not data_sample.flip:
                aug_labels.append(data_sample.pred_instances.labels)
                aug_masks.append(data_sample.pred_instances.masks)
                continue

            labels = data_sample.pred_instances.labels
            aug_labels.append(labels)

            masks = data_sample.pred_instances.masks
            flipped_masks = torch.flip(masks, dims=(-1,))
            aug_masks.append(flipped_masks)

        if convert_to_cpu:
            aug_bboxes = [bboxes.cpu() for bboxes in aug_bboxes]
            aug_scores = [scores.cpu() for scores in aug_scores]
            aug_labels = [labels.cpu() for labels in aug_labels]
            aug_masks = [masks.cpu() for masks in aug_masks]

        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_progs = torch.cat(aug_progs, dim=0)
        merged_probs = torch.cat(aug_probs, dim=0)
        merged_labels = torch.cat(aug_labels, dim=0)
        merged_masks = torch.cat(aug_masks, dim=0)

        if merged_bboxes.numel() == 0:
            det_bboxes = torch.cat([merged_bboxes, merged_scores[:, None]], -1)
            return [
                (det_bboxes, merged_labels),
            ]

        det_bboxes, keep_idxs = batched_nms(merged_bboxes, merged_scores,
                                            merged_labels, self.tta_cfg.nms)

        det_bboxes = det_bboxes[:self.tta_cfg.max_per_img]
        det_progs = merged_progs[keep_idxs][:self.tta_cfg.max_per_img]
        det_probs = merged_probs[keep_idxs][:self.tta_cfg.max_per_img]
        det_labels = merged_labels[keep_idxs][:self.tta_cfg.max_per_img]
        det_masks = merged_masks[keep_idxs][:self.tta_cfg.max_per_img]

        results = InstanceData()
        _det_bboxes = det_bboxes.clone()
        results.bboxes = _det_bboxes[:, :-1]
        results.scores = _det_bboxes[:, -1]
        results.progs = det_progs
        results.probs = det_probs
        results.labels = det_labels
        results.masks = det_masks
        det_results = data_samples[0]
        det_results.pred_instances = results
        
        return det_results
