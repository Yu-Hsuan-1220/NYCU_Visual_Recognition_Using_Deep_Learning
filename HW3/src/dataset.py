"""
Dataset utilities: ConvNeXtV2 backbone registration and augmentation pipelines.

This module registers the ConvNeXtV2 backbone with MMDetection and provides
train/val/test augmentation pipeline configurations.
"""
import torch.nn as nn
import numpy as np
import tifffile
from mmdet.registry import MODELS, TRANSFORMS
from mmcv.transforms import BaseTransform
from mmengine.model import BaseModule

@TRANSFORMS.register_module()
class LoadTifImageFromFile(BaseTransform):
    """Load .tif images using tifffile, handling RGBA and ensuring contiguity."""

    def __init__(self, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def transform(self, results):
        filename = results['img_path']
        img = tifffile.imread(filename)

        # Handle RGBA → RGB
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        # Handle grayscale → 3-channel
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        # RGB → BGR for mmdet (OpenCV convention)
        if img.ndim == 3 and img.shape[2] == 3:
            img = img[:, :, ::-1]

        # CRITICAL: ensure contiguous uint8 array.
        # The [::-1] above creates negative strides which corrupt tensor conversion.
        img = np.ascontiguousarray(img, dtype=np.uint8)

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


# ---------------------------------------------------------------------------
# ConvNeXt-V2 Backbone (via timm)
# ---------------------------------------------------------------------------
@MODELS.register_module()
class ConvNeXtV2Backbone(BaseModule):
    """ConvNeXt-V2 backbone using timm for feature extraction.

    Includes per-stage LayerNorm on output features to prevent extreme
    magnitudes (ConvNeXt-V2's GRN can produce values up to ±4000) from
    causing NaN in downstream FPN/RoI heads.

    Args:
        model_name (str): timm model name. Default: 'convnextv2_base'.
        pretrained (bool): Use ImageNet pretrained weights. Default: True.
        out_indices (tuple): Indices of stages to output. Default: (0, 1, 2, 3).
        drop_path_rate (float): Stochastic depth rate. Default: 0.4.
    """

    def __init__(
        self,
        model_name="convnextv2_base",
        pretrained=True,
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.4,
    ):
        super().__init__()
        import timm

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            drop_path_rate=drop_path_rate,
        )
        # Get output channels for each stage
        self._out_channels = self.model.feature_info.channels()

        # Add LayerNorm for each output stage to normalize extreme values
        # This is critical for ConvNeXt-V2 where GRN can produce very large features
        self.norms = nn.ModuleList([
            nn.LayerNorm(ch) for ch in self._out_channels
        ])

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        features = self.model(x)
        # Apply LayerNorm per stage (channel-last, then back to channel-first)
        out = []
        for i, feat in enumerate(features):
            # feat: (B, C, H, W) → permute to (B, H, W, C) for LayerNorm
            feat = feat.permute(0, 2, 3, 1)
            feat = self.norms[i](feat)
            feat = feat.permute(0, 3, 1, 2).contiguous()
            out.append(feat)
        return tuple(out)


# ---------------------------------------------------------------------------
# Augmentation Pipelines
# ---------------------------------------------------------------------------
def get_train_pipeline(img_scale=(1024, 1024), multiscale_mode=True):
    """Get training augmentation pipeline.

    Strong augmentation for small medical dataset:
    - Multi-scale resize
    - Random crop
    - Random flips (H + V)
    - Rotation via RandomChoiceResize
    - Color jitter via Albu
    """
    pipeline = [
        dict(type="LoadTifImageFromFile"),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    ]

    if multiscale_mode:
        # Multi-scale training: resize then crop
        pipeline.extend([
            dict(
                type="RandomResize",
                scale=img_scale,
                ratio_range=(0.5, 2.0),
                keep_ratio=True,
            ),
            dict(
                type="RandomCrop",
                crop_size=img_scale,
                crop_type="absolute",
                recompute_bbox=True,
                allow_negative_crop=True,
            ),
        ])
    else:
        pipeline.append(
            dict(type="Resize", scale=img_scale, keep_ratio=True)
        )

    pipeline.extend([
        dict(type="RandomFlip", prob=0.5, direction="horizontal"),
        dict(type="RandomFlip", prob=0.5, direction="vertical"),
        # Filter out degenerate GT bboxes (zero width/height) created by RandomCrop
        # Without this, DeltaXYWH encoding produces log(0) = -inf → NaN loss
        dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1)),
        # Pad to divisible size
        dict(type="Pad", size=img_scale, pad_val=dict(img=(114, 114, 114))),
        dict(type="PackDetInputs"),
    ])

    return pipeline


def get_val_pipeline(img_scale=(1024, 1024)):
    """Get validation pipeline (no augmentation)."""
    return [
        dict(type="LoadTifImageFromFile"),
        dict(type="Resize", scale=img_scale, keep_ratio=True),
        dict(type="Pad", size=img_scale, pad_val=dict(img=(114, 114, 114))),
        # Load annotations for evaluation
        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
        dict(
            type="PackDetInputs",
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        ),
    ]


def get_test_pipeline(img_scale=(1024, 1024)):
    """Get test pipeline (no annotations)."""
    return [
        dict(type="LoadTifImageFromFile"),
        dict(type="Resize", scale=img_scale, keep_ratio=True),
        dict(type="Pad", size=img_scale, pad_val=dict(img=(114, 114, 114))),
        dict(
            type="PackDetInputs",
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        ),
    ]
