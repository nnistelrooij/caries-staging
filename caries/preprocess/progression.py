import copy
import json
from pathlib import Path
from typing import List

import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from skimage.segmentation import find_boundaries
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


def wrong_order(coco, img_dict, caries_ann, dentin_ann):
    cat_id2name = {cat_id: cat['name'] for cat_id, cat in coco.cats.items()}

    fdi = cat_id2name[caries_ann['category_id']][-2:]
    anns = coco.imgToAnns[img_dict['id']]
    tooth_ann = [
        ann for ann in anns
        if 'TOOTH' in cat_id2name[ann['category_id']]
        and fdi in cat_id2name[ann['category_id']]
    ][0]

    enamel_x = dentin_ann[0]['x']

    x1, width = tooth_ann['bbox'][::2]

    if (
        enamel_x > x1 + 0.4 * width
        and enamel_x < x1 + 0.6 * width
    ):
        return True
    
    return False


def project_to_line(
    img_dict,
    caries_ann,
    dentin_ann,
    num_interp: int=1000,
):
    # determine yx pixels of dentin skeleton
    p0 = np.array([dentin_ann[0][k] for k in 'yx'])
    p1 = np.array([dentin_ann[1][k] for k in 'yx'])
    interp_x = np.linspace(p0[1], p1[1], num=num_interp)
    interp_y = np.linspace(p0[0], p1[0], num=num_interp)
    interp_pixels = np.column_stack((interp_y, interp_x)).astype(int)
    line_pixels = np.unique(interp_pixels, axis=0)

    # determine yx pixels of caries boundary
    h, w = [img_dict[k] for k in ['height', 'width']]
    caries_rle = coco_to_rle(caries_ann['segmentation'], h, w)
    caries_mask = maskUtils.decode(caries_rle)
    caries_boundary = find_boundaries(caries_mask)
    caries_pixels = np.column_stack(np.nonzero(caries_boundary))

    # determine yx pixels of intersections between dentin and caries
    pixels = np.concatenate((line_pixels, caries_pixels))
    unique, counts = np.unique(pixels, return_counts=True, axis=0)
    inters_pixels = unique[counts == 2]

    # determine line pixel index of caries boundary closest to pulpa
    pixels = np.concatenate((interp_pixels, inters_pixels))
    unique, index, inverse = np.unique(
        pixels, return_index=True, return_inverse=True, axis=0,
    )
    pixel_idx = index[inverse[num_interp:]].max()

    # score is proportion of dentin affected by caries
    score = (pixel_idx + 1) / num_interp
    
    return score


def determine_scores(coco, img_dict, caries_anns, dentin_anns):
    caries_bboxes = np.array([ann['bbox'] for ann in caries_anns])
    dentin_bboxes = np.zeros((0, 4))
    for ann in dentin_anns:
        assert ann[0]['name'] == 'Enamel'
        x1 = min([node['x'] for node in ann])
        x2 = max([node['x'] for node in ann])
        y1 = min([node['y'] for node in ann])
        y2 = max([node['y'] for node in ann])
        bbox = [x1, y1, x2 - x1, y2 - y1]
        dentin_bboxes = np.concatenate((dentin_bboxes, [bbox]))

    ious = maskUtils.iou(caries_bboxes, dentin_bboxes, [0]*dentin_bboxes.shape[0])
    if isinstance(ious, list):
        ious = np.zeros((len(caries_anns), 1))

    out_anns = []
    for caries_ann, caries_ious in zip(caries_anns, ious):
        if not np.any(caries_ious > 0):
            try:
                score = caries_ann['extra']['attributes'][0] == 'C'
            except:
                print(img_dict['file_name'])
                score = 0.0
        else:
            scores = []
            for pos_idx in np.nonzero(caries_ious)[0]:
                dentin_ann = dentin_anns[pos_idx]
                if wrong_order(coco, img_dict, caries_ann, dentin_ann):
                    print('Wrong point order ' + img_dict['file_name'])
                try:
                    score = project_to_line(img_dict, caries_ann, dentin_ann)
                    scores.append(score)
                except:
                    pass
            
            if len(scores) != 1:
                print('Two dentin skeletons for one caries lesion! ' + img_dict['file_name'])
            score = scores[0] if scores else 0.0

        out_ann = copy.deepcopy(caries_ann)
        out_ann['score'] = float(score)
        out_anns.append(out_ann)

    return out_anns


def progression_scores(coco, darwin_root):
    out_dict = copy.deepcopy(coco.dataset)
    out_dict['annotations'] = []

    cat_id2name = {cat_id: cat['name'] for cat_id, cat in coco.cats.items()}
    for img_dict in tqdm(list(coco.imgs.values())):
        img_stem = Path(img_dict["file_name"]).stem
        with open(darwin_root / f'{img_stem}.json', 'r') as f:
            darwin_anns = json.load(f)['annotations']
        dentin_anns = [ann['skeleton']['nodes'] for ann in darwin_anns if 'skeleton' in ann]

        coco_anns = coco.imgToAnns[img_dict['id']]
        is_caries = np.array([
            'Caries' in cat_id2name[ann['category_id']]
            and 'Residual' not in cat_id2name[ann['category_id']]
            for ann in coco_anns
        ])
        non_caries_anns = [coco_anns[idx] for idx in np.nonzero(~is_caries)[0]]
        caries_anns = [coco_anns[idx] for idx in np.nonzero(is_caries)[0]]
        caries_anns = determine_scores(coco, img_dict, caries_anns, dentin_anns)

        out_dict['annotations'].extend(non_caries_anns)
        out_dict['annotations'].extend(caries_anns)

    return out_dict


def progression(
    coco: COCO,
    annotator_roots: List[Path],
    out_file,
):
    img_id2name = {img_id: img['file_name'] for img_id, img in coco.imgs.items()}

    coco_dicts = []
    for  root in annotator_roots:
        coco_dict = progression_scores(coco, root)
        coco_dicts.append(coco_dict['annotations'])

    out_dict = copy.deepcopy(coco_dicts[0])
    out_dict['annotations'] = []
    for anns in zip(*coco_dicts):
        if 'score' not in anns[0]:
            out_dict['annotations'].append(anns[0])
            continue

        scores = np.array([ann['score'] for ann in anns])
        score = scores.mean()

        if np.any(np.abs(scores - score) > 0.4):
            score = scores[scores != scores.round()].mean()
            print(np.abs(scores - score).argmax())
            print(img_id2name[anns[0]['image_id']])

        ann = copy.deepcopy(anns[0])
        ann['score'] = score.item()
        out_dict['annotations'].append(ann)
    
    with open(out_file, 'w') as f:
        json.dump(out_dict, f, indent=2)
