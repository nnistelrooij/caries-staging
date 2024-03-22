from pathlib import Path

from pycocotools.coco import COCO

from caries.preprocess import (
    assign_fdis,
    progression,
    split,
)


if __name__ == '__main__':
    root = Path('annotations')

    # match each tooth finding to a tooth
    coco = COCO(root / 'coco.json')
    assign_fdis(coco=coco, out_file=root / 'coco_fdis.json')

    # determine the consensus severity scores
    coco = COCO(root / 'coco_fdis.json')
    progression(
        coco=coco,
        annotator_roots=[
            root / 'Eduardo_Lingyun',
            root / 'Max',
            root / 'Vitor',
        ],
        out_file=root / 'coco_scores.json',
    )

    # split the data for 10-fold cross-validation
    coco = COCO(root / 'coco_scores.json')
    split(coco=coco, splits_dir=root.parent / 'splits')
