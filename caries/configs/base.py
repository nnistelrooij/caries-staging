_base_ = 'mmdet::swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'

custom_imports = dict(
    imports=[
        'caries.data',
        'caries.evaluation',
        'caries.models',
    ],
    allow_failed_imports=False,
)


filter_empty = False
ignore_border_teeth = True

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadScoreAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[{
                'type':
                'RandomChoiceResize',
                'scales': [(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                'keep_ratio':
                True
            }],
            [{
                'type': 'RandomChoiceResize',
                'scales': [(400, 4200), (500, 4200), (600, 4200)],
                'keep_ratio': True
            }, {
                'type': 'RandomCrop',
                'crop_type': 'absolute_range',
                'crop_size': (384, 600),
                'allow_negative_crop': True
            }, {
                'type':
                'RandomChoiceResize',
                'scales':
                [(480, 1333), (512, 1333), (544, 1333),
                    (576, 1333), (608, 1333), (640, 1333),
                    (672, 1333), (704, 1333), (736, 1333),
                    (768, 1333), (800, 1333)],
                'keep_ratio':
                True
            }],
        ]),
    dict(type='PackDetScoreInputs'),
]

workers = 0
train_dataloader = dict(
    batch_size=4,
    num_workers=workers,
    persistent_workers=False,
    dataset=dict(
        _delete_=True,
        type='CocoCariesDataset',
        ignore_border_teeth=ignore_border_teeth,
        filter_cfg=dict(filter_empty_gt=filter_empty),
        serialize_data=False,
        pipeline=train_pipeline,
    ),
)

val_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(1333, 800),
        keep_ratio=True),
    dict(type='LoadScoreAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetScoreInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

val_dataloader = dict(
    num_workers=workers,
    persistent_workers=False,
    dataset=dict(
        type='CocoCariesDataset',
        ignore_border_teeth=ignore_border_teeth,
        pipeline=val_pipeline,
    ),
)
val_evaluator = dict(
    _delete_=True,
    type='CocoCariesMetric',
    classwise=True,
    metric=['bbox', 'segm'],
)

test_dataloader = dict(
    num_workers=workers,
    persistent_workers=False,
    dataset=dict(
        type='CocoCariesDataset',
        ignore_border_teeth=ignore_border_teeth,
        pipeline=val_pipeline,
    ),
)
test_evaluator = dict(
    type='DumpGTPredDetResults',
)

model = dict(
    type='MaskRCNN',
    backbone=dict(drop_path_rate=0.2),
    rpn_head=dict(type='RPNHead'),
    test_cfg=dict(rcnn=dict(score_thr=0.0)),
)

max_epochs = 24
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
)
param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    begin=0,
    end=max_epochs,
    by_epoch=True,
    milestones=[20],
    gamma=0.1,
)

optim_wrapper = dict(    
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        interval=12,
        by_epoch=True,
        max_keep_ckpts=2,
        save_best='coco/segm_mAP',
        rule='greater',
    ),
    visualization=dict(
        draw=True,
        interval=50,
    ),
)

visualizer = dict(
    type='DetLocalProgressionVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
)

tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TestTimeAug', transforms=[
        [
            {
                'type': 'Resize',
                'scale': scale,
                'keep_ratio': True,
            } for scale in [
                (1333, 640), (1333, 672), (1333, 704),
                (1333, 736), (1333, 768), (1333, 800),
            ]
        ],
        [
            {'type': 'RandomFlip', 'prob': 0.0},
            {'type': 'RandomFlip', 'prob': 1.0},
        ],
        [{
            'type': 'LoadScoreAnnotations', 'with_bbox': True, 'with_mask': True,
        }],
        [{
            'type': 'PackDetScoreInputs',
            'meta_keys': [
                'img_id', 'img_path', 'ori_shape', 'img_shape',
                'scale_factor', 'flip', 'flip_direction',
            ]
        }],
    ]),
]

tta_model = dict(
    type='InstSegTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
    ),
)
