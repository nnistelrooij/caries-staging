import copy
import json
from pathlib import Path
import re
import tempfile

import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from scipy import ndimage


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


def add_fdi_categories(coco: COCO):
    catids = set([cat['id'] for cat in coco.dataset['categories']])
    catnames = [cat['name'] for cat in coco.dataset['categories']]
    out_dict = copy.deepcopy(coco.dataset)
    offset = 1
    for cat in coco.dataset['categories']:
        if re.match('\d+', cat['name'].split('_')[-1]):
            continue

        for q in '1234':
            for e in '12345678':
                name = f'{cat["name"]}_{q}{e}'
                if name in catnames:
                    continue

                while offset in catids:
                    offset += 1

                out_dict['categories'].append({'id': offset, 'name': name})
                offset += 1

    with tempfile.NamedTemporaryFile('w') as f:
        json.dump(out_dict, f)
        f.flush()
        coco = COCO(f.name)

    return coco


def determine_tooth_annotations(coco, anns):
    if not anns:
        return []
    
    h, w = [coco.imgs[anns[0]['image_id']][k] for k in ['height', 'width']]
    catid2name = {cat_id: cat['name'] for cat_id, cat in coco.cats.items()}
    catname2id = {cat['name']: cat_id for cat_id, cat in coco.cats.items()}

    catnames = [catid2name[ann['category_id']] for ann in anns]
    is_tooths = np.nonzero(['TOOTH' in cat for cat in catnames])[0]
    no_fdis = np.nonzero([re.match('\d+', cat.split('_')[-1]) is None for cat in catnames])[0]
    no_fdis = np.array([i for i in no_fdis if anns[i]['area'] >= 4])

    out_anns = [ann for i, ann in enumerate(anns) if i not in no_fdis]
    if no_fdis.shape[0] == 0:
        return out_anns

    rles = [coco_to_rle(ann['segmentation'], h, w) for ann in anns]
    tooth_masks = [maskUtils.decode(rle) for i, rle in enumerate(rles) if i in is_tooths]
    tooth_wide_masks = [ndimage.binary_dilation(mask) for mask in tooth_masks]
    tooth_rles = [maskUtils.encode(np.asfortranarray(mask)) for mask in tooth_wide_masks]
    non_tooth_rles = [rle for i, rle in enumerate(rles) if i in no_fdis]

    tooth_areas = np.array([anns[i]['area'] for i in is_tooths])
    non_tooth_areas = np.array([anns[i]['area'] for i in no_fdis])

    optimal = non_tooth_areas[:, None] / tooth_areas[None]
    ious = maskUtils.iou(non_tooth_rles, tooth_rles, np.zeros_like(is_tooths)) / optimal
    match_scores = ious / optimal
    for i, non_tooth_match_scores in enumerate(match_scores):
        if not np.any(non_tooth_match_scores):
            non_tooth_ann = anns[no_fdis[i]]
            img_name = coco.imgs[non_tooth_ann['image_id']]['file_name']
            catname = coco.cats[non_tooth_ann['category_id']]['name']
            print(img_name, catname)
            continue

        tooth_ann = anns[is_tooths[non_tooth_match_scores.argmax()]]
        non_tooth_ann = copy.deepcopy(anns[no_fdis[i]])
        fdi = catid2name[tooth_ann['category_id']][-2:]
        
        label = f'{catid2name[non_tooth_ann["category_id"]]}_{fdi}'
        non_tooth_ann['category_id'] = catname2id[label]

        out_anns.append(non_tooth_ann)

    return out_anns


def assign_fdis(coco, out_file):
    coco = add_fdi_categories(coco)
    
    out_anns = []
    for img_id in coco.imgs:
        anns = determine_tooth_annotations(coco, coco.imgToAnns[img_id])
        out_anns.extend(anns)

    coco.dataset['annotations'] = out_anns
    with open(out_file, 'w') as f:
        json.dump(coco.dataset, f, indent=2)
