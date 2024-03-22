import copy
from pathlib import Path
from typing import List, Union

from mmengine.fileio import get_local_path
import numpy as np
from tqdm import tqdm

from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class CocoCariesDataset(CocoDataset):

    def __init__(
        self,
        ignore_border_teeth: bool=False,
        *args,
        **kwargs,
    ):
        self.ignore_border_teeth = ignore_border_teeth

        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        cat_name2id = {cat['name']: cat_id for cat_id, cat in self.coco.cats.items()}

        self.cat_ids = [cat_name2id[cat_name] for cat_name in self.metainfo['classes']]

        cat_id2idx = {cat_id: i for i, cat_id in enumerate(self.coco.cats)}
        cat_idxs = np.argsort([cat_id2idx[cat_id] for cat_id in self.cat_ids])
        if not all([i == j for i, j in enumerate(cat_idxs)]):
            cat_names = [self.coco.cats[self.cat_ids[i]]['name'] for i in cat_idxs]
            raise ValueError(f'Please specify classes as {cat_names}')
        
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in tqdm(img_ids):
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = Path(self.data_prefix['img']) / img_info['file_name']
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = None
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if 'attributes' in ann['extra'] and (
                'Occlusal' in ann['extra']['attributes']
                or (
                    self.ignore_border_teeth
                    and 'Occluded' in ann['extra']['attributes']
                )
            ):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instance['score'] = ann.get('score', 0.0)

            instances.append(instance)
        data_info['instances'] = instances
        return data_info
