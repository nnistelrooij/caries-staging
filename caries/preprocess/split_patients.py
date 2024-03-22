from collections import defaultdict
import copy
from pathlib import Path
import tempfile
from typing import List

import json
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from skmultilearn.model_selection import IterativeStratification



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


def duplicate_groups(items, pairs):
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)

    duplicates = []
    for idx1, idx2 in pairs:
        new = True
        for group in duplicates:
            if idx1 not in group and idx2 not in group:
                continue
            
            group.add(idx1)
            group.add(idx2)
            new = False
            break

        if new:
            duplicates.append(set([idx1, idx2]))

    groups = list(map(list, duplicates))

    unique_idxs = [idx for group in groups for idx in group]
    groups += [[i] for i, _ in enumerate(items) if i not in unique_idxs]

    return groups


def determine_labels(
    coco: COCO,
    classes: list[str]=[
        'caries', 'restoration', 'tooth_confounder', 'overlap',
        'restoration_confounder', 'undercontour', 'overhang',
    ],
) -> np.ndarray[np.bool_]:
    cat_id2label = defaultdict(lambda: -1)
    for cat in coco.cats.values():
        if cat['name'] not in classes:
            continue

        cat_id2label[cat['id']] = len(cat_id2label)

    onehots = np.zeros((len(coco.imgs), len(cat_id2label)), dtype=int)
    for i, img_id in enumerate(coco.imgs):
        for ann in coco.imgToAnns[img_id]:
            label = cat_id2label[ann['category_id']]
            if label != -1:
                onehots[i, label] = 1

    return onehots


def determine_groups(
    coco: COCO,
) -> np.ndarray[np.int64]:
    groups = []
    for img_dict in coco.imgs.values():
        patient = img_dict['file_name'].split('-')[0]
        groups.append(patient)

    _, groups = np.unique(groups, return_inverse=True)

    return groups


def cluster_patient_images(patient_groups, folds):
    fold_groups = np.repeat(np.arange(len(folds)), [idxs.shape[0] for idxs in folds])
    multi_groups = np.nonzero(np.unique(patient_groups, return_counts=True)[1] > 1)[0]
    for group in multi_groups:
        first_idx = (patient_groups == group).argmax()
        first_fold = fold_groups[np.concatenate(folds) == first_idx][0]

        for i, fold in enumerate(folds):
            if i == first_fold:
                other_patient_images = np.nonzero(patient_groups == group)[0]
                folds[i] = np.concatenate((fold, other_patient_images))
                folds[i] = np.unique(folds[i])
                continue

            other_patient_images = patient_groups[fold] == group
            folds[i] = folds[i][~other_patient_images]


def split_patients(
    coco: COCO,
    n_folds: int=5,
) -> dict[str, np.ndarray[np.int64]]:
    labels = determine_labels(coco)
    groups = determine_groups(coco)

    splits = {}
    splitter = IterativeStratification(n_folds, order=2, shuffle=True, random_state=1234)
    folds = [split[1] for split in list(splitter.split(coco.imgs, labels))]
    cluster_patient_images(groups, folds)

    splits['all'] = np.concatenate(folds)
    for i in range(len(folds)):
        splits[f'train{i}'] = np.concatenate(
            folds[max(0, i - len(folds) + 1):max(0, i - 2)] +
            folds[i:i + len(folds) - 2]
        )
        splits[f'val{i}'] = folds[(i + len(folds) - 2) % len(folds)]
        splits[f'test{i}'] = folds[(i + len(folds) - 1) % len(folds)]

    return splits


def add_segmentation_group(
    eduardo_coco: COCO,
    name: str,
    group: List[str],
    merge: str='none',
):    
    coco_dict = copy.deepcopy(eduardo_coco.dataset)

    max_cat_id = max([cat['id'] for cat in coco_dict['categories']])
    max_ann_id = max([ann['id'] for ann in coco_dict['annotations']])
    coco_dict['categories'].append({
        'id': max_cat_id + 1, 'name': name,
    })

    cat_id2name = {cat_id: cat['name'] for cat_id, cat in eduardo_coco.cats.items()}
    for img_id, img_dict in eduardo_coco.imgs.items():
        h, w = img_dict['height'], img_dict['width']
        anns = eduardo_coco.imgToAnns[img_id]

        if merge == 'fdi':
            merged_anns = []
            fdis = np.array([cat_id2name[ann['category_id']][-2:] for ann in anns])
            for fdi in np.unique(fdis):
                ann_idxs = np.nonzero(fdis == fdi)[0]
                cat_names = [cat_id2name[anns[i]['category_id']] for i in ann_idxs]
                ann_idxs = [i for i, cat_name in zip(ann_idxs, cat_names) if cat_name[:-3] in group]
                if not ann_idxs:
                    continue
                merged_ann = copy.deepcopy(anns[ann_idxs[0]])
                fdi_rles = [coco_to_rle(anns[i]['segmentation'], h, w) for i in ann_idxs]
                merged_rle = maskUtils.merge(fdi_rles)
                merged_ann['segmentation'] = {
                    'size': merged_rle['size'],
                    'counts': merged_rle['counts'].decode(),
                }
                merged_ann['area'] = maskUtils.area(merged_rle).item()
                merged_ann['bbox'] = maskUtils.toBbox(merged_rle).tolist()
                merged_anns.append(merged_ann)
            anns = merged_anns
        elif merge == 'overlap':
            merged_anns = []
            cat_names = [cat_id2name[ann['category_id']] for ann in anns]
            keep = [cat_name[:-3] in group for cat_name in cat_names]
            anns = [ann for ann, k in zip(anns, keep) if k]
            if not anns:
                continue
            rles = [coco_to_rle(ann['segmentation'], h, w) for ann in anns]
            ious = maskUtils.iou(rles, rles, [0]*len(rles))
            ious[np.diag_indices_from(ious)] = 0.0
            overlap_pairs = np.column_stack(np.nonzero(ious > 0))
            overlap_groups = duplicate_groups(rles, overlap_pairs)
            for overlap_group in overlap_groups:
                merged_ann = copy.deepcopy(anns[overlap_group[0]])
                merged_rle = maskUtils.merge([rles[i] for i in overlap_group])
                merged_ann['segmentation'] = {
                    'size': merged_rle['size'],
                    'counts': merged_rle['counts'].decode(),
                }
                merged_ann['area'] = maskUtils.area(merged_rle).item()
                merged_ann['bbox'] = maskUtils.toBbox(merged_rle).tolist()
                merged_anns.append(merged_ann)
            anns = merged_anns



        for ann in anns:
            cat_name = cat_id2name[ann['category_id']]
            if cat_name[:-3] not in group:
                continue

            ann['id'] += max_ann_id + 1
            ann['category_id'] = max_cat_id + 1
            coco_dict['annotations'].append(ann)
    
    with tempfile.NamedTemporaryFile() as f:
        with open(f.name, 'w') as fp:
            json.dump(coco_dict, fp)
        coco = COCO(f.name)

    return coco


def attribute_as_class(coco: COCO):
    cat_id2name = {cat_id: cat['name'] for cat_id, cat in coco.cats.items()}
    max_cat_id = max(coco.cats)
    max_ann_id = max(coco.anns)

    coco_dict = copy.deepcopy(coco.dataset)
    coco_dict['categories'].extend([
        {'id': max_cat_id + 1, 'name': 'Initial'},
        {'id': max_cat_id + 2, 'name': 'Moderate'},
        {'id': max_cat_id + 3, 'name': 'Severe'},
        {'id': max_cat_id + 4, 'name': 'Leave'},
        {'id': max_cat_id + 5, 'name': 'Treat'},
    ])
    for img_id, img_dict in coco.imgs.items():
        h, w = img_dict['height'], img_dict['width']
        anns = coco.imgToAnns[img_id]

        tooth_anns = [ann for ann in anns if cat_id2name[ann['category_id']].startswith('TOOTH')]
        fdi2tooth = {cat_id2name[ann['category_id']][-2:]: ann for ann in tooth_anns}

        dentin_anns = [ann for ann in anns if cat_id2name[ann['category_id']].startswith('Dentin')]
        dentin_rles = [coco_to_rle(ann['segmentation'], h, w) for ann in dentin_anns]
        for ann in anns:
            cat_name = cat_id2name[ann['category_id']].lower()
            if 'caries' not in cat_name or 'residual' in cat_name:
                continue

            if 'Occlusal' in ann['extra']['attributes']:
                continue

            ann_id = ann['id']
            prog_score = ann['score']
            ann['id'] = 2 * (ann_id + max_ann_id + 1)
            if prog_score < 0.25:
                ann['category_id'] = max_cat_id + 1
            elif prog_score >= 0.25 and prog_score < 0.75:
                ann['category_id'] = max_cat_id + 2
            else:
                ann['category_id'] = max_cat_id + 3
            coco_dict['annotations'].append(ann)

            ann = copy.deepcopy(ann)
            ann['id'] = 2 * (ann_id + max_ann_id + 1) + 1
            if prog_score < 0.5:
                ann['category_id'] = max_cat_id + 4
            else:
                ann['category_id'] = max_cat_id + 5
            coco_dict['annotations'].append(ann)
            

    with tempfile.NamedTemporaryFile() as f:
        with open(f.name, 'w') as fp:
            json.dump(coco_dict, fp)
        coco = COCO(f.name)

    return coco


def filter_coco(
    coco: COCO,
    idxs: np.ndarray[np.int64],
) -> dict[str, list]:
    coco_dict = {
        'images': [],
        'annotations': [],
        'categories': coco.dataset['categories'],
        'tag_categories': coco.dataset['tag_categories'],
    }
    for i, img_dict in enumerate(coco.imgs.values()):
        if i not in idxs:
            continue

        coco_dict['images'].append(img_dict)

        anns = coco.imgToAnns[img_dict['id']]
        coco_dict['annotations'].extend(anns)

    return coco_dict


def split(coco: COCO, splits_dir: Path):
    coco = attribute_as_class(coco)

    coco = add_segmentation_group(coco, name='tooth', group=['TOOTH'])
    coco = add_segmentation_group(coco, name='caries', group=[
        'Primary Caries', 'Secondary Caries',
    ])
    coco = add_segmentation_group(coco, name='residual', group=['Residual Caries'])
    coco = add_segmentation_group(coco, name='restoration', merge='fdi', group=[
        'fillings', 'crowns', 'pontic',
    ])    
    
    splits = split_patients(coco, n_folds=10)

    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, idxs in splits.items():
        coco_dict = filter_coco(coco, idxs)
        with open(splits_dir / f'{name}.json', 'w') as f:
            json.dump(coco_dict, f, indent=2)
