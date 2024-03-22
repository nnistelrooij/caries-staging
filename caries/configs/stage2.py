_base_ = './base.py'

classes = ['Initial', 'Moderate', 'Severe'] 
scores = 'head'

data_root = './'
split = 1
work_dir = f'work_dirs/stage2_{split}/'

train_dataloader = dict(dataset=dict(
    data_root=data_root,
    metainfo=dict(classes=classes),
    ann_file=data_root + f'splits/train{split}.json',
    data_prefix=dict(img=data_root + 'images'),
))

val_dataloader = dict(dataset=dict(
    data_root=data_root,
    metainfo=dict(classes=classes),
    ann_file=data_root + f'splits/val{split}.json',
    data_prefix=dict(img=data_root + 'images'),
))
val_evaluator = dict(
    classes=classes,
    ann_file=data_root + f'splits/val{split}.json',
)

test_dataloader = dict(dataset=dict( 
    data_root=data_root,   
    metainfo=dict(classes=classes),
    ann_file=data_root + f'splits/test{split}.json',
    data_prefix=dict(img=data_root + 'images'),
))
test_evaluator = dict(
    out_file_path=work_dir + 'detections.pkl',
)

model = dict(roi_head=dict(
    type='ScoreRoiHead',
    bbox_head=dict(
        type='ProgressionBBoxHead',
        num_classes=len(classes),
    ),
))
load_from = f'work_dirs/stage1_{split}/epoch_24.pth'
