CLASSES = ('star',)
auto_scale_lr = dict(base_batch_size=16, enable=False)
backend = 'pillow'
backend_args = None
custom_hooks = [
    dict(
        min_delta=0.001,
        monitor='coco/segm_mAP',
        patience=30,
        rule='greater',
        type='EarlyStoppingHook'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'npy_loading',
    ])
data_root = r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        # Keep the metric-selected best checkpoint plus one final checkpoint
        # for recovery. Do not save fixed-interval epoch checkpoints.
        interval=-1,
        max_keep_ckpts=1,
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
max_iter = 90000
model = dict(
    _scope_='mmdet',
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=None,
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        center_sampling=True,
        centerness_on_reg=True,
        conv_bias=True,
        dcn_on_last_conv=False,
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='GIoULoss'),
        loss_centerness=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        norm_on_bbox=True,
        num_classes = 1,
        num_params=169,
        stacked_convs=4,
        strides=[
            # Stars occupy only a few pixels. P2 keeps a 7 px target at
            # roughly two feature cells instead of losing it on P3.
            4,
            8,
            16,
            32,
            64,
        ],
        # Assign small stars to P2 rather than the default P3/P4 FCOS bins.
        regress_ranges=[(-1, 32), (16, 64), (32, 128), (64, 256),
                        (128, 100000000.0)],
        type='CondInstBboxHead'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='DetDataPreprocessor'),
    mask_head=dict(
        feat_channels=8,
        loss_mask=dict(
            activate=True,
            eps=5e-06,
            loss_weight=1.0,
            type='DiceLoss',
            use_sigmoid=True),
        mask_feature_head=dict(
            end_level=2,
            feat_channels=128,
            in_channels=256,
            # This branch now starts from P2, whose spatial stride is 4.
            mask_stride=4,
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_stacked_convs=4,
            out_channels=8,
            start_level=0),
        mask_out_stride=4,
        max_masks_to_train=300,
        num_layers=3,
        size_of_interest=8,
        type='CondInstMaskHead'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        # Use C2/P2 as the first FPN level. The stock CondInst setup starts
        # at P3 (stride 8), which is too coarse for 7-15 px stellar masks.
        start_level=0,
        type='FPN'),
    test_cfg=dict(
        mask_thr=0.5,
        max_per_img=200,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        # Keep low-confidence candidates during validation; AP ranks them and
        # should not receive an empty prediction set in early epochs.
        score_thr=0.001),
    type='CondInst')
num_classes = 1
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
        backend_args=None,
        data_prefix=dict(img='test/images/'),
        data_root=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable',
        metainfo=dict(classes=('star',)),
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
    backend_args=None,
    format_only=False,
    metric=[
        'segm',
    ],
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
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=2)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        _scope_='mmdet',
        ann_file=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable/annotations/train.json',
        backend_args=None,
        data_prefix=dict(img='train/images/'),
        data_root=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=('star',)),
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
        backend_args=None,
        data_prefix=dict(img='val/images/'),
        data_root=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable',
        metainfo=dict(classes=('star',)),
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
    backend_args=None,
    format_only=False,
    metric=[
        'segm',
    ],
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
work_dir = r'D:/CodeSpace/Python/MY_query_mask/work_dirs/real_mixed_baselines/mmdet/condinst'
