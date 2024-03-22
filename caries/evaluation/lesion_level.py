from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import scipy
import torch
from tqdm import tqdm

from caries.evaluation.tooth_level import match_lesion_restoration


def eval_image(
    coco,
    img_dict,
    gt_instances,
    pred_instances,
    score_thr: float,
    iou_thr: float,
    caries_tag: str,
):
    gt_progs = gt_instances['scores']
    gt_masks = gt_instances['masks']

    if caries_tag:
        gt_labels = match_lesion_restoration(coco, img_dict, gt_masks)
        gt_progs = gt_progs[gt_labels == caries_tag]
        gt_masks = [gt_masks[idx] for idx in np.nonzero(gt_labels == caries_tag)[0]]

    pred_scores = pred_instances['scores']
    pred_progs = pred_instances['progs']
    pred_masks = pred_instances['masks']

    if caries_tag:
        pred_labels = match_lesion_restoration(coco, img_dict, pred_masks)
        pred_scores = pred_scores[pred_labels == caries_tag]
        pred_progs = pred_progs[pred_labels == caries_tag]
        pred_masks = [pred_masks[i] for i in np.nonzero(pred_labels == caries_tag)[0]]

    ious = maskUtils.iou(gt_masks, pred_masks, [0]*len(pred_masks))

    gts, preds = torch.zeros((2, 0))
    for gt_idx, gt_ious in enumerate(ious):
        if not np.any(gt_ious > 0.1):
            continue

        pos_scores = pred_scores * (gt_ious >= iou_thr)
        pred_idx = pos_scores.argmax()
        if pos_scores[pred_idx] < score_thr:
            continue

        gts = torch.cat((gts, torch.tensor([gt_progs[gt_idx]])))
        preds = torch.cat((preds, torch.tensor([pred_progs[pred_idx]])))

    return gts, preds


def eval_split(coco, results, score_thr, iou_thr, caries_tag):
    img_name2id = {img['file_name']: img_id for img_id, img in coco.imgs.items()}

    gts, preds = torch.zeros((2, 0))
    for result in results:
        file_name = Path(result['img_path']).name
        img_dict = coco.imgs[img_name2id[file_name]]
        result_gts, result_preds = eval_image(
            coco, img_dict,
            result['gt_instances'], result['pred_instances'],
            score_thr, iou_thr, caries_tag=caries_tag,
        )

        gts = torch.cat((gts, result_gts))
        preds = torch.cat((preds, result_preds))

    return gts, preds


def eval_lesion_level(coco, score_thrs, iou_thr, caries_tag):
    gts, preds = torch.zeros((2, 0))
    for split, score_thr in enumerate(tqdm(score_thrs)):
        with open(f'work_dirs/stage2_{split}/detections.pkl', 'rb') as f:
            results = pickle.load(f)

        split_gts, split_preds = eval_split(
            coco, results, score_thr, iou_thr, caries_tag,
        )
        gts = torch.cat((gts, split_gts))
        preds = torch.cat((preds, split_preds))


    slope, intercept, r_value, _, _ = scipy.stats.linregress(gts, preds)

    file_suffix = f'_model_consensus{"_" + caries_tag if caries_tag else ""}'
    plt.scatter(gts, preds, alpha=0.8)
    plt.plot([0, 1], [intercept, intercept + slope], color='k', linewidth=2, label=f'Pearson\'s $r$ = {r_value:.3f}')
    plt.xlabel('Reference severity score')
    plt.ylabel('Predicted severity score')
    plt.legend()
    plt.savefig(f'scatter{file_suffix}.png', dpi=800, bbox_inches='tight', pad_inches=None)
    plt.show()

    plt.scatter(np.mean([gts, preds], axis=0), gts - preds, alpha=0.8)
    plt.axhline(0, c='k')
    plt.axhline(np.quantile(gts - preds, 0.025), c='k', linestyle='--')
    plt.axhline(np.quantile(gts - preds, 0.975), c='k', linestyle='--')
    plt.xlabel('Mean')
    plt.ylabel('Difference')
    plt.ylim(-0.7, 0.55)
    plt.savefig(f'blandaltman{file_suffix}.png', dpi=800, bbox_inches='tight', pad_inches=None)
    plt.show()


if __name__ == '__main__':
    coco = COCO(Path('splits/all.json'))

    eval_lesion_level(
        coco=coco,
        score_thrs=[0.9041, 0.9092, 0.8944, 0.7912, 0.9116, 0.8991, 0.9361, 0.9189, 0.7006, 0.9292],
        iou_thr=0.5,
        caries_tag='Secondary',
    )
