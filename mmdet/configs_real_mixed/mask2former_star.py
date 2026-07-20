CLASSES = ('star',)
auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
batch_augments = []
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
        'star_mask2former_fusion',
        'star_coco_metric',
    ])
data_preprocessor = dict(
    _scope_='mmdet',
    batch_augments=[
        dict(
            img_pad_value=0,
            mask_pad_value=0,
            pad_mask=True,
            pad_seg=False,
            size=(
                1024,
                1024,
            ),
            type='BatchFixedSizePad'),
    ],
    bgr_to_rgb=True,
    mask_pad_value=0,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_mask=True,
    pad_seg=False,
    pad_size_divisor=32,
    seg_pad_value=255,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='DetDataPreprocessor')
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
dynamic_intervals = [
    (
        365001,
        368750,
    ),
]
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
image_size = (
    512,
    512,
)
interval = 5000
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor')
max_iters = 368750
model = dict(
    _scope_='mmdet',
    backbone=dict(
        depth=50,
        frozen_stages=-1,
        init_cfg=None,
        norm_cfg=dict(requires_grad=False, type='BN'),
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
    data_preprocessor=dict(
        batch_augments=[],
        bgr_to_rgb=False,
        mask_pad_value=0,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        pad_mask=True,
        pad_seg=False,
        pad_size_divisor=32,
        seg_pad_value=255,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='DetDataPreprocessor'),
    init_cfg=None,
    panoptic_fusion_head=dict(
        init_cfg=None,
        loss_panoptic=None,
        num_stuff_classes=0,
        num_things_classes = 1,
        # The stock instance path emits raw query/class pairs.  Use the local
        # head so duplicate queries on one tiny star are removed before the
        # validation metric, early stopping, visualization, and eval.py.
        type='StarMaskFormerFusionHead'),
    panoptic_head=dict(
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        loss_cls=dict(
            class_weight=[1.0, 0.1],
            loss_weight=2.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=2.0,
            naive_dice=True,
            reduction='mean',
            type='DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=2.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        # The local dataset currently has fewer than 20 instances/image, but
        # retain the standard capacity for denser star fields and avoid a
        # query-count bottleneck when another split is used.
        num_queries=100,
        num_stuff_classes=0,
        num_things_classes = 1,
        num_transformer_feat_level=3,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        embed_dims=256,
                        feedforward_channels=1024,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dropout=0.0,
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4)),
                num_layers=6),
            norm_cfg=dict(num_groups=32, type='GN'),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=128),
            type='MSDeformAttnPixelDecoder'),
        positional_encoding=dict(normalize=True, num_feats=128),
        strides=[
            4,
            8,
            16,
            32,
        ],
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8)),
            num_layers=9,
            return_intermediate=True),
        type='Mask2FormerHead'),
    test_cfg=dict(
        # The real-data GT masks overlap by at most about 0.17.  A 0.50
        # class-agnostic mask-IoU threshold therefore removes duplicate
        # query masks for one star while retaining physically distinct stars.
        class_agnostic_mask_nms=True,
        instance_on=True,
        mask_nms_iou_thr=0.50,
        max_per_image=50,
        min_mask_pixels=1,
        panoptic_on=False,
        pre_nms_topk=200,
        score_thr=0.001,
        semantic_on=False),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', use_sigmoid=True, weight=2.0),
                dict(eps=1.0, pred_act=True, type='DiceCost', weight=2.0),
            ],
            type='HungarianAssigner'),
        # A 7-15 px mask occupies very few pixels of a 512x512 image. The
        # default 12,544 sampled points often contains no foreground pixels
        # for a matched query, so increase the supervision density.
        importance_sample_ratio=0.9,
        num_points=32768,
        oversample_ratio=4.0,
        sampler=dict(type='MaskPseudoSampler')),
    type='Mask2Former')
num_classes = 1
num_stuff_classes = 0
num_things_classes = 1
optim_wrapper = dict(
    _scope_='mmdet',
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=2e-05,
        type='AdamW',
        weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(decay_mult=1.0, lr_mult=0.1),
            level_embed=dict(decay_mult=0.0, lr_mult=1.0),
            query_embed=dict(decay_mult=0.0, lr_mult=1.0),
            query_feat=dict(decay_mult=0.0, lr_mult=1.0)),
        norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=5, start_factor=0.1, type='LinearLR'),
    dict(
        T_max=100,
        begin=5,
        by_epoch=True,
        end=50,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable/annotations/test.json',
        backend_args=None,
        data_prefix=dict(
            img='test/images/', seg='annotations/panoptic_val2017/'),
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
    num_workers=4,
    persistent_workers=False,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmdet',
    ann_file=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable/annotations/test.json',
    backend_args=None,
    format_only=False,
    metric=[
        'segm',
    ],
    type='EmptySafeCocoMetric')
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
    # Denser point supervision costs memory; batch 2 is stable on modest GPUs.
    batch_size=2,
    dataset=dict(
        _scope_='mmdet',
        ann_file=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable/annotations/train.json',
        backend_args=None,
        data_prefix=dict(
            img='train/images/', seg='annotations/panoptic_train2017/'),
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
    persistent_workers=False,
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
        data_prefix=dict(
            img='val/images/', seg='annotations/panoptic_val2017/'),
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
    num_workers=4,
    persistent_workers=False,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmdet',
    ann_file=r'D:/CodeSpace/Python/MY_query_mask/mixed_point_streak_trainable/annotations/val.json',
    backend_args=None,
    format_only=False,
    metric=[
        'segm',
    ],
    type='EmptySafeCocoMetric')
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
work_dir = r'D:/CodeSpace/Python/MY_query_mask/work_dirs/real_mixed_baselines/mmdet/mask2former'
