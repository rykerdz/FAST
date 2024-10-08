model = dict(
    type='FAST',
    backbone=dict(
        type='fast_backbone',
        config='config/fast/nas-configs/fast_base.config'
    ),
    neck=dict(
        type='fast_neck',
        config='config/fast/nas-configs/fast_base.config'
    ),
    detection_head=dict(
        type='fast_head',
        config='config/fast/nas-configs/fast_base.config',
        pooling_size=6,
        dropout_ratio=0.1,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=0.5
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        loss_emb=dict(
            type='EmbLoss_v1',
            feature_dim=4,
            loss_weight=0.25
        )
    )
)
repeat_times = 1
data = dict(
    batch_size=8,
    train=dict(
        type='FAST_IC15',
        split='train',
        is_transform=True,
        img_size=1024,
        short_size=1280,
        pooling_size=6,
        read_type='cv2',
        repeat_times=repeat_times
    ),
    test=dict(
        type='FAST_IC15',
        split='test',
        short_size=1280,
        read_type='cv2'
    )
)
train_cfg = dict(
    lr=2e-3,
    schedule='polylr',
    epoch=30 // repeat_times,
    optimizer='Adam',
    pretrain='pretrained/fast_base_ic17mlt_640.pth',
    # https://github.com/czczup/FAST/releases/download/release/fast_base_ic17mlt_640.pth
    save_interval=8 // repeat_times,
)
test_cfg = dict(
    min_score=0.88,
    min_area=200,
    bbox_type='rect',
    result_path='outputs/submit_ic15.zip'
)
