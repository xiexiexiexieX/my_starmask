CLASSES = ('point', 'streak')
auto_scale_lr = dict(base_batch_size=16, enable=False)
custom_hooks = [
    dict(
        min_delta=0.001,
        monitor='coco/segm_mAP',
        patience=10,
        rule='greater',
        type='EarlyStoppingHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'npy_loading',
    ])
data_root = r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=5,
        max_keep_ckpts=3,
        rule='greater',
        save_best='coco/segm_mAP',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor')

model = dict(
    _scope_='mmdet',
    type='SOLO',
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[0.0, 0.0, 0.0],
        pad_mask=True,
        pad_size_divisor=32,
        std=[1.0, 1.0, 1.0],
        type='DetDataPreprocessor'),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=None),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    mask_head=dict(
        type='SOLOV2Head',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        ins_out_channels=256,
        mask_feature_head=dict(
            type='MaskFeatHead',
            in_channels=256,
            feat_channels=128,
            start_level=0,
            end_level=3,
            out_channels=256,
            mask_stride=4,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        loss_mask=dict(type='DiceLoss', use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)),
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.1,
        mask_thr=0.5,
        filter_thr=0.05,
        kernel='gaussian',
        sigma=2.0,
        max_per_img=200))

num_classes = 2
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    type='AmpOptimWrapper')
param_scheduler = dict(
    T_max=100, by_epoch=True, eta_min=1e-06, type='CosineAnnealingLR')
resume = False

test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable/annotations/test.json',
        data_prefix=dict(img='test/images/'),
        data_root=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable',
        metainfo=dict(classes=('point', 'streak')),
        pipeline=[
            dict(type='LoadStarNpy'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmdet',
    ann_file=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable/annotations/test.json',
    format_only=False,
    metric=['segm'],
    type='CocoMetric')

test_pipeline = [
    dict(type='LoadStarNpy'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]

train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        _scope_='mmdet',
        ann_file=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable/annotations/train.json',
        data_prefix=dict(img='train/images/'),
        data_root=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=('point', 'streak')),
        pipeline=[
            dict(type='LoadStarNpy'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(direction='vertical', prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadStarNpy'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(direction='vertical', prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]

val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable/annotations/val.json',
        data_prefix=dict(img='val/images/'),
        data_root=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable',
        metainfo=dict(classes=('point', 'streak')),
        pipeline=[
            dict(type='LoadStarNpy'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmdet',
    ann_file=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable/annotations/val.json',
    format_only=False,
    metric=['segm'],
    type='CocoMetric')

vis_backends = [
    dict(_scope_='mmdet', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = r'D:/CodeSpace/Python/MY_query_mask/work_dirs/real_mixed_baselines/mmdet/solov2'
