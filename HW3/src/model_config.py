"""
Model configuration for Cascade Mask R-CNN + ConvNeXt-V2-Base + FPN.

Provides functions to build the full MMDetection config dict.
All hyperparameters are exposed via function arguments for argparse integration.
"""


def _bbox_loss_cfg(bbox_loss):
    """Return loss_bbox dict for Shared2FCBBoxHead stages."""
    if bbox_loss == "giou":
        return dict(type="GIoULoss", loss_weight=2.0)
    return dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0)


def get_model_config(
    backbone_name="convnextv2_base",
    pretrained=True,
    fpn_out_channels=256,
    num_classes=4,
    drop_path_rate=0.4,
    cascade_iou_thresholds=(0.5, 0.6, 0.7),
    cascade_stage_weights=(1.0, 0.5, 0.25),
    bbox_loss="smoothl1",
    mask_head_convs=4,
    mask_roi_size=14,
):
    """Build the model config dict for Cascade Mask R-CNN.

    Args:
        backbone_name: timm model name for ConvNeXt-V2.
        pretrained: Whether to use ImageNet pretrained weights.
        fpn_out_channels: FPN output channels.
        num_classes: Number of foreground classes (default 4).
        drop_path_rate: Stochastic depth rate.
        cascade_iou_thresholds: IoU thresholds for cascade stages.
        cascade_stage_weights: Loss weights for cascade stages.
        bbox_loss: Loss type for Cascade bbox heads — "smoothl1" or "giou".
        mask_head_convs: Number of conv layers in FCNMaskHead (default 4).
        mask_roi_size: RoIAlign output size for mask branch (default 14).

    Returns:
        dict: MMDetection model config.
    """
    # ConvNeXt-V2-Base output channels per stage
    backbone_channels = {
        "convnextv2_base": [128, 256, 512, 1024],
        "convnextv2_large": [192, 384, 768, 1536],
        "convnextv2_tiny": [96, 192, 384, 768],
        "convnextv2_small": [96, 192, 384, 768],  # actually same as tiny for some variants
    }

    in_channels = backbone_channels.get(backbone_name, [128, 256, 512, 1024])

    model = dict(
        type="CascadeRCNN",
        data_preprocessor=dict(
            type="DetDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_mask=True,
            pad_size_divisor=32,
        ),
        backbone=dict(
            type="ConvNeXtV2Backbone",
            model_name=backbone_name,
            pretrained=pretrained,
            out_indices=(0, 1, 2, 3),
            drop_path_rate=drop_path_rate,
        ),
        neck=dict(
            type="FPN",
            in_channels=in_channels,
            out_channels=fpn_out_channels,
            num_outs=5,
        ),
        rpn_head=dict(
            type="RPNHead",
            in_channels=fpn_out_channels,
            feat_channels=fpn_out_channels,
            anchor_generator=dict(
                type="AnchorGenerator",
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
            ),
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
        ),
        roi_head=dict(
            type="CascadeRoIHead",
            num_stages=len(cascade_iou_thresholds),
            stage_loss_weights=list(cascade_stage_weights),
            bbox_roi_extractor=dict(
                type="SingleRoIExtractor",
                roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                out_channels=fpn_out_channels,
                featmap_strides=[4, 8, 16, 32],
            ),
            bbox_head=[
                dict(
                    type="Shared2FCBBoxHead",
                    in_channels=fpn_out_channels,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.1, 0.1, 0.2, 0.2],
                    ),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                    ),
                    loss_bbox=_bbox_loss_cfg(bbox_loss),
                ),
                dict(
                    type="Shared2FCBBoxHead",
                    in_channels=fpn_out_channels,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.05, 0.05, 0.1, 0.1],
                    ),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                    ),
                    loss_bbox=_bbox_loss_cfg(bbox_loss),
                ),
                dict(
                    type="Shared2FCBBoxHead",
                    in_channels=fpn_out_channels,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.033, 0.033, 0.067, 0.067],
                    ),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                    ),
                    loss_bbox=_bbox_loss_cfg(bbox_loss),
                ),
            ],
            mask_roi_extractor=dict(
                type="SingleRoIExtractor",
                roi_layer=dict(type="RoIAlign", output_size=mask_roi_size, sampling_ratio=0),
                out_channels=fpn_out_channels,
                featmap_strides=[4, 8, 16, 32],
            ),
            mask_head=dict(
                type="FCNMaskHead",
                num_convs=mask_head_convs,
                in_channels=fpn_out_channels,
                conv_out_channels=fpn_out_channels,
                num_classes=num_classes,
                loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
            ),
        ),
        # Training config
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False,
                ),
                allowed_border=0,
                pos_weight=-1,
                debug=False,
            ),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=2000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=[
                dict(
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=cascade_iou_thresholds[0],
                        neg_iou_thr=cascade_iou_thresholds[0],
                        min_pos_iou=cascade_iou_thresholds[0],
                        match_low_quality=False,
                        ignore_iof_thr=-1,
                    ),
                    sampler=dict(
                        type="RandomSampler",
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True,
                    ),
                    mask_size=mask_roi_size * 2,  # FCNMaskHead upsamples RoI by 2× via ConvTranspose
                    pos_weight=-1,
                    debug=False,
                ),
                dict(
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=cascade_iou_thresholds[1],
                        neg_iou_thr=cascade_iou_thresholds[1],
                        min_pos_iou=cascade_iou_thresholds[1],
                        match_low_quality=False,
                        ignore_iof_thr=-1,
                    ),
                    sampler=dict(
                        type="RandomSampler",
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True,
                    ),
                    mask_size=mask_roi_size * 2,
                    pos_weight=-1,
                    debug=False,
                ),
                dict(
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=cascade_iou_thresholds[2],
                        neg_iou_thr=cascade_iou_thresholds[2],
                        min_pos_iou=cascade_iou_thresholds[2],
                        match_low_quality=False,
                        ignore_iof_thr=-1,
                    ),
                    sampler=dict(
                        type="RandomSampler",
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True,
                    ),
                    mask_size=mask_roi_size * 2,
                    pos_weight=-1,
                    debug=False,
                ),
            ],
        ),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type="nms", iou_threshold=0.5),
                max_per_img=300,
                mask_thr_binary=0.5,
            ),
        ),
    )

    return model
