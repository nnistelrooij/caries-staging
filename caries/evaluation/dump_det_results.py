from typing import Sequence

from mmengine.evaluator.metric import _to_cpu

from mmdet.evaluation.metrics import DumpDetResults
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results


@METRICS.register_module()
class DumpGTPredDetResults(DumpDetResults):
    """Dump model predictions to a pickle file for offline evaluation.

    Different from `DumpResults` in MMEngine, it compresses instance
    segmentation masks into RLE format.

    Args:
        out_file_path (str): Path of the dumped file. Must end with '.pkl'
            or '.pickle'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
    """

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """transfer tensors in predictions to CPU."""
        data_samples = _to_cpu(data_samples)
        for data_sample in data_samples:
            # remove gt
            data_sample.pop('ignored_instances', None)
            data_sample.pop('gt_panoptic_seg', None)

            if 'gt_instances' in data_sample:
                gt = data_sample['gt_instances']
                # encode mask to RLE
                if 'masks' in gt:
                    gt['masks'] = encode_mask_results(gt['masks'].to_ndarray())

            if 'pred_instances' in data_sample:
                pred = data_sample['pred_instances']
                # encode mask to RLE
                if 'masks' in pred:
                    pred['masks'] = encode_mask_results(pred['masks'].numpy())

        self.results.extend(data_samples)
