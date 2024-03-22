_base_ = './base.py'

classes = ['tooth', 'caries', 'residual', 'restoration'] 
scores = 'none'

data_root = './'
split = 1
work_dir = f'work_dirs/stage1_{split}/'

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

model = dict(roi_head=dict(bbox_head=dict(
    num_classes=len(classes),
)))
load_from = 'checkpoints/mask-rcnn_swin-t.pth'
