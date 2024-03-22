from pathlib import Path
import pickle
from typing import Callable, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from scipy import ndimage
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
import torch
from tqdm import tqdm


def coco_to_rle(ann, h, w):
    if isinstance(ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(ann, h, w)
        return maskUtils.merge(rles)
    
    if isinstance(ann['counts'], list):
        # uncompressed RLE
        return maskUtils.frPyObjects(ann, h, w)
    
    # rle
    return ann


def draw_confusion_matrix(
    cm,
    labels,
    ax,
):
    norm_cm = cm / cm.sum()

    disp = ConfusionMatrixDisplay(norm_cm, display_labels=labels)
    disp.plot(cmap='magma', ax=ax, colorbar=False)

    # draw colorbar according to largest non-TN value
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
    disp.ax_.images[0].set_norm(normalize)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            disp.text_[i, j].set_text(cm[i, j])
    
    # draw y ticklabels vertically
    offset = matplotlib.transforms.ScaledTranslation(-0.1, 0, disp.figure_.dpi_scale_trans)
    for label in disp.ax_.get_yticklabels():
        label.set_rotation(90)
        label.set_transform(label.get_transform() + offset)
        label.set_ha('center')
        label.set_rotation_mode('anchor')


def draw_rocs(rocs, ax, aucs, name, c):
    fprs = np.linspace(0, 1, 1000)
    interp_rocs = np.zeros((0, 1000))
    for roc in rocs:
        if np.any(np.isnan(roc[1])):
            interp_roc = [np.nan]*1000
        else:
            interp_roc = np.interp(fprs, roc[0], roc[1])
        interp_rocs = np.concatenate((interp_rocs, [interp_roc]))

    mean_roc = np.nanmean(interp_rocs, axis=0)
    std_roc = np.nanstd(interp_rocs, axis=0)
    
    ax.plot([0, 1], [0, 1], c='k', label='Random', ls='--')
    ax.plot(fprs, mean_roc, c=c, label=f'{name} (AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f})')
    ax.plot(fprs, np.minimum(mean_roc + std_roc, 1), c=c)
    ax.plot(fprs, mean_roc - std_roc, c=c)
    ax.fill_between(fprs, mean_roc - std_roc, np.minimum(mean_roc + std_roc, 1), color=c, alpha=0.2)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.grid()
    ax.legend()


def match_lesion_restoration(
    coco: COCO,
    img_dict: dict,
    lesion_rles: List[dict],
):
    if not lesion_rles:
        return np.array([])

    cat_id2name = {cat_id: cat['name'] for cat_id, cat in coco.cats.items()}
    anns = coco.imgToAnns[img_dict['id']]

    # determine RLEs, areas, and FDIs of teeth
    tooth_anns = [
        ann for ann in anns
        if cat_id2name[ann['category_id']].startswith('TOOTH')
    ]
    tooth_rles = [coco_to_rle(ann['segmentation'], img_dict['height'], img_dict['width']) for ann in tooth_anns]
    tooth_masks = [maskUtils.decode(rle) for rle in tooth_rles]
    tooth_masks = [ndimage.binary_dilation(mask) for mask in tooth_masks]
    tooth_rles = [maskUtils.encode(np.asfortranarray(mask)) for mask in tooth_masks]
    tooth_areas = np.array([ann['area'] for ann in tooth_anns])
    tooth_fdis = np.array([cat_id2name[ann['category_id']][-2:] for ann in tooth_anns])

    # determine FDI of each caries lesion
    lesion_areas = np.array([maskUtils.area(rle) for rle in lesion_rles])
    optimal = lesion_areas[:, None] / tooth_areas[None]
    ious = maskUtils.iou(lesion_rles, tooth_rles, [0]*len(tooth_rles)) / optimal
    lesion_fdis = tooth_fdis[ious.argmax(axis=1)]
    
    # determine RLEs and FDIs of restorations
    restoration_anns = [
        ann for ann in anns
        if cat_id2name[ann['category_id']].startswith('fillings')
        or cat_id2name[ann['category_id']].startswith('crowns')
        or cat_id2name[ann['category_id']].startswith('pontic')
    ]
    restoration_rles = [coco_to_rle(ann['segmentation'], img_dict['height'], img_dict['width']) for ann in restoration_anns]
    restoration_masks = [maskUtils.decode(rle) for rle in restoration_rles]
    restoration_masks = [ndimage.binary_dilation(mask) for mask in restoration_masks]
    restoration_rles = [maskUtils.encode(np.asfortranarray(mask)) for mask in restoration_masks]
    restoration_fdis = np.array([cat_id2name[ann['category_id']][-2:] for ann in restoration_anns])
    
    # match lesion to restoration with overlap and same FDI
    ious = maskUtils.iou(lesion_rles, restoration_rles, [0]*len(restoration_rles))
    out = []
    for fdi, restoration_ious in zip(lesion_fdis, ious):
        overlaps = (restoration_ious > 0) & (restoration_fdis == fdi)
        out.append('Secondary' if np.any(overlaps) else 'Primary')

    return np.array(out)


def match_tooth_restoration(
    coco: COCO,
    img_dict: dict,
):
    cat_id2name = {cat_id: cat['name'] for cat_id, cat in coco.cats.items()}
    anns = coco.imgToAnns[img_dict['id']]
    labels = [cat_id2name[ann['category_id']] for ann in anns]

    fdi_restored = set()
    for label in labels:
        if label[:-3] not in ['fillings', 'crowns', 'pontic']:
            continue

        fdi_restored.add(label[-2:])

    tooth_fdis = [
        cat_id2name[ann['category_id']][-2:] for ann in anns
        if cat_id2name[ann['category_id']].startswith('TOOTH')
    ]
    tooth_restored = np.array([fdi in fdi_restored for fdi in tooth_fdis])

    return tooth_restored


def eval_image(
    coco: COCO,
    img_dict: dict,
    gt_instances: List[dict],
    pred_instances: List[dict],
    tooth_tag: str,
    caries_tag: str,
    prog_thr: float,
    score_fn: Callable[[List[float]], float],
):
    # collect and filter ground-truth lesions
    keep_gt = torch.ones(gt_instances['labels'].shape, dtype=torch.bool)
    if caries_tag:
        tags = match_lesion_restoration(coco, img_dict, gt_instances['masks'])
        keep_gt[tags != caries_tag] = False
    gt_masks = [gt_instances['masks'][idx] for idx in torch.nonzero(keep_gt)[:, 0]]
    gt_progs = gt_instances['scores'][keep_gt]
    gt_areas = np.array([maskUtils.area(rle) for rle in gt_masks])

    # collect and filter predicted lesions
    keep_pred = torch.ones(pred_instances['labels'].shape, dtype=torch.bool)
    if caries_tag:
        tags = match_lesion_restoration(coco, img_dict, pred_instances['masks'])
        keep_pred[tags != caries_tag] = False
    pred_masks = [pred_instances['masks'][idx] for idx in torch.nonzero(keep_pred)[:, 0]]
    pred_probs = pred_instances['probs'][keep_pred]
    pred_areas = np.maximum(1, np.array([maskUtils.area(rle) for rle in pred_masks]))
    
    # determine teeth to be ignored
    cat_id2name = {cat_id: cat['name'] for cat_id, cat in coco.cats.items()}
    cat_name2id = {cat['name']: cat_id for cat_id, cat in coco.cats.items()}
    anns = coco.imgToAnns[img_dict['id']]
    caries_anns = [
        ann for ann in coco.imgToAnns[img_dict['id']]
        if cat_id2name[ann['category_id']].startswith('Primary Caries')
        or cat_id2name[ann['category_id']].startswith('Secondary Caries')
    ]
    ignore_anns = [
        ann for ann in caries_anns
        if 'Occlusal' in ann['extra']['attributes']
        or 'Occluded' in ann['extra']['attributes']
    ]
    ignore_fdis = [cat_id2name[ann['category_id']][-2:] for ann in ignore_anns]
    ignore_ids = [cat_name2id[f'TOOTH_{fdi}'] for fdi in ignore_fdis]

    # prepare tooth annotations and tooth ignore filter
    tooth_anns = [
        ann for ann in anns
        if cat_id2name[ann['category_id']].startswith('TOOTH')
    ]
    tooth_filter = np.array([ann['category_id'] not in ignore_ids for ann in tooth_anns])
    if tooth_tag == 'restoration':
        tooth_filter = tooth_filter & match_tooth_restoration(coco, img_dict)
    elif tooth_tag:
        raise ValueError('Either empty or "restoration"')
    tooth_masks = [ann['segmentation'] for ann in tooth_anns]
    tooth_rles = [coco_to_rle(mask, img_dict['height'], img_dict['width']) for mask in tooth_masks]
    tooth_areas = np.array([maskUtils.area(rle) for rle in tooth_rles])
    tooth_idx_map = np.zeros(len(tooth_masks), dtype=int)
    tooth_idx_map[np.nonzero(tooth_filter)[0]] = np.arange(tooth_filter.sum())

    # get the binary tooth-level ground-truth labels for caries
    gt_tooth_ious = maskUtils.iou(tooth_rles, gt_masks, [0]*len(gt_masks))
    if isinstance(gt_tooth_ious, list):
        gt_tooth_ious = np.zeros((1, len(tooth_rles)))
    else:
        gt_tooth_ious /= gt_areas / tooth_areas[:, None]
    gt_teeth = gt_tooth_ious.argmax(0)
    gt_caries = torch.zeros(tooth_filter.sum())
    for tooth_idx, prog in zip(gt_teeth, gt_progs):
        if not tooth_filter[tooth_idx]:
            continue

        tooth_idx = tooth_idx_map[tooth_idx]
        gt_caries[tooth_idx] = max(gt_caries[tooth_idx], prog >= prog_thr)

    # get the tooth-level prediction scores for caries
    pred_tooth_ious = maskUtils.iou(tooth_rles, pred_masks, [0]*len(pred_masks))
    if isinstance(pred_tooth_ious, list):
        pred_tooth_ious = np.zeros((1, len(tooth_rles)))
    else:
        pred_tooth_ious /= pred_areas / tooth_areas[:, None]
    pred_teeth = pred_tooth_ious.argmax(0)
    pred_scores = torch.zeros(tooth_filter.sum())
    for tooth_idx, probs in zip(pred_teeth, pred_probs):
        if not tooth_filter[tooth_idx]:
            continue

        tooth_idx = tooth_idx_map[tooth_idx]
        pred_scores[tooth_idx] = max(pred_scores[tooth_idx], score_fn(probs))

    return gt_caries, pred_scores


def optimal_threshold(
    gts,
    scores,
):
    f1s = []
    thresholds = np.linspace(
        scores[scores > 0].min(),
        scores.max(),
        1000,
    )
    for thr in thresholds:
        preds = scores >= thr

        tp = ((gts > 0) & (preds > 0)).sum()
        fp = (preds > 0).sum() - tp
        fn = (gts > 0).sum() - tp
        f1 = 2 * tp / (2 * tp + fp + fn)

        f1s.append(f1)

    thr = thresholds[np.argmax(f1s)]

    return thr


def compute_metrics(cm):
    tn = cm[0, 0]
    tp = cm[1:, 1:].sum()
    fp = cm[0, 1:].sum()
    fn = cm[1:, 0].sum()
    
    return {
        'F1': 2 * tp / (2 * tp + fp + fn),
        'precision': tp / (tp + fp),
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
    }


def eval_split(
    coco: COCO,
    results: List[dict],
    tooth_tag: str,
    caries_tag: str,
    prog_thr: float,
    score_fn: Callable[[List[float]], float],
):   
    img_name2id = {img['file_name']: img_id for img_id, img in coco.imgs.items()}

    gts_list, scores_list = torch.zeros((2, 0))
    for result in results:
        file_name = Path(result['img_path']).name
        img_dict = coco.imgs[img_name2id[file_name]]
        gts, scores = eval_image(
            coco, img_dict,
            result['gt_instances'], result['pred_instances'],
            tooth_tag, caries_tag, prog_thr, score_fn,
        )

        gts_list = torch.cat((gts_list, gts))
        scores_list = torch.cat((scores_list, scores))

    score_thr = optimal_threshold(gts_list, scores_list)
    preds_list = scores_list >= score_thr

    cm = confusion_matrix(gts_list, preds_list)
    roc = roc_curve(gts_list, scores_list)
    metrics = compute_metrics(cm)
    metrics['auc'] = roc_auc_score(gts_list, scores_list)

    return cm, roc, metrics


def eval_tooth_level(
    coco: COCO,
    splits: List[int],
    tooth_tag: str='',
    caries_tag: str='',
    prog_thr: float=0.0,
    score_fn: Callable[[List[float]], float]=lambda probs: sum(probs[:-1]),
    labels: List[str]=['No caries', 'caries'],
):
    cms, rocs, metrics_list = [], [], []
    for split in tqdm(list(splits)):
        with open(f'work_dirs/stage2_{split}/detections.pkl', 'rb') as f:
            results = pickle.load(f)

        cm, roc, metrics = eval_split(
            coco, results, tooth_tag, caries_tag, prog_thr, score_fn,
        )

        cms.append(cm)
        rocs.append(roc)
        metrics_list.append(metrics)

    metrics_means = {
        k: np.nanmean([ms[k] for ms in metrics_list])
        for k in metrics_list[0]
    }
    metrics_stds = {
        k: np.nanstd([ms[k] for ms in metrics_list])
        for k in metrics_list[0]
    }

    print('Macro-averaged')
    for k in metrics_means:
        print(f'{k}: {metrics_means[k]:.3f}±{metrics_stds[k]:.3f}')

    _, ax = plt.subplots(1, 1, figsize=(7.0, 5.2))
    draw_confusion_matrix(sum(cms), labels, ax)
    file_suffix = (
        ('_' + tooth_tag if tooth_tag else '')
        + ('_' + caries_tag if caries_tag else '')
    )
    plt.savefig(f'cm{file_suffix}.png', dpi=800, bbox_inches='tight', pad_inches=None)
    plt.show()

    _, ax = plt.subplots(1, 1)
    draw_rocs(rocs, ax, [ms['auc'] for ms in metrics_list], f'{caries_tag + " c" if caries_tag else "C"}aries', c='g')
    plt.savefig(f'roc{file_suffix}.png', dpi=800, bbox_inches='tight', pad_inches=None)
    plt.show()


if __name__ == '__main__':
    coco = COCO(Path('annotations/coco_scores.json'))
    
    # restored teeth with all secondary caries lesions
    eval_tooth_level(
        coco=coco,
        splits=range(10),
        tooth_tag='restoration',
        caries_tag='Secondary',
        labels=['No secondary caries', 'Secondary caries'],
    )
    
    # restored teeth with secondary dentin caries lesions
    eval_tooth_level(
        coco=coco,
        splits=range(10),
        tooth_tag='restoration',
        caries_tag='Secondary',
        prog_thr=0.33,
        score_fn=lambda probs: sum(probs[:-1]) * sum(probs[1:-1]),
        labels=['No secondary dentin caries', 'Secondary dentin caries'],
    )
