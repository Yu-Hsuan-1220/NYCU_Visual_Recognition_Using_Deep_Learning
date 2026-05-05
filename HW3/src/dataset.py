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


# ---------------------------------------------------------------------------
# Custom Augmentation Transforms
# ---------------------------------------------------------------------------
@TRANSFORMS.register_module()
class GaussianNoise(BaseTransform):
    """Add Gaussian pixel noise to the image.

    Args:
        std (float): Standard deviation of noise (in uint8 pixel range 0-255).
        prob (float): Probability of applying the transform.
    """

    def __init__(self, std: float = 15.0, prob: float = 0.5):
        self.std = std
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if np.random.random() < self.prob:
            img = results['img'].astype(np.float32)
            noise = np.random.randn(*img.shape).astype(np.float32) * self.std
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            results['img'] = img
        return results


@TRANSFORMS.register_module()
class GaussianBlurAug(BaseTransform):
    """Apply Gaussian blur with a randomly sampled odd kernel size.

    Uses cv2.GaussianBlur — no albumentations required.

    Args:
        max_kernel (int): Upper bound for the kernel size (must be odd; if even,
            the next odd value is used).  Default: 7.
        prob (float): Probability of applying the transform.  Default: 0.3.
    """

    def __init__(self, max_kernel: int = 7, prob: float = 0.3):
        import cv2 as _cv2  # lazy import — avoids hard dep at module load time
        self.max_kernel = max_kernel if max_kernel % 2 == 1 else max_kernel + 1
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if np.random.random() < self.prob:
            import cv2
            # Sample a random odd kernel ∈ {3, 5, …, max_kernel}
            k = np.random.choice(range(3, self.max_kernel + 1, 2))
            results['img'] = cv2.GaussianBlur(results['img'], (k, k), 0)
        return results


@TRANSFORMS.register_module()
class CLAHEAug(BaseTransform):
    """Contrast-Limited Adaptive Histogram Equalisation applied per channel.

    Uses cv2.createCLAHE — no albumentations required.  Very useful for
    microscopy images where staining intensity varies across experiments.

    Args:
        clip_limit (float): Threshold for contrast limiting.  Default: 4.0.
        tile_grid_size (tuple): Size of the grid for histogram equalisation.
            Default: (8, 8).
        prob (float): Probability of applying the transform.  Default: 0.3.
    """

    def __init__(self, clip_limit: float = 4.0,
                 tile_grid_size: tuple = (8, 8), prob: float = 0.3):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if np.random.random() < self.prob:
            import cv2
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size,
            )
            img = results['img']
            orig_dtype = img.dtype
            # clahe.apply() requires CV_8UC1; PhotoMetricDistortion leaves float32
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            for c in range(img.shape[2]):
                img[:, :, c] = clahe.apply(np.ascontiguousarray(img[:, :, c]))
            results['img'] = img.astype(orig_dtype)
        return results

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
def get_train_pipeline(
    img_scale=(1024, 1024),
    multiscale_mode=True,
    color_jitter=False,
    diagonal_flip=False,
    resize_ratio_min=0.5,
    resize_ratio_max=2.0,
    random_rotate=False,
    rotate_max_angle=30.0,
    gridmask=False,
    gaussian_noise=False,
    noise_std=15.0,
    albu=False,
):
    """Get training augmentation pipeline.

    Pipeline order:
      Load → (Resize+Crop) → [RandomRotate] → [PhotoMetricDistortion]
      → [Albu] → [GaussianNoise] → Flip H/V → [DiagonalFlip]
      → [GridMask] → Filter → Pad → Pack
    """
    pipeline = [
        dict(type="LoadTifImageFromFile"),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    ]

    if multiscale_mode:
        pipeline.extend([
            dict(
                type="RandomResize",
                scale=img_scale,
                ratio_range=(resize_ratio_min, resize_ratio_max),
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

    # Spatial augmentation: random rotation.
    # Placed after crop so it operates on the fixed-size window; border pixels are
    # filled with 114 (same as our pad value) to avoid introducing black regions.
    if random_rotate:
        pipeline.append(dict(
            type="RandomRotate",
            prob=0.5,
            angle=rotate_max_angle,
            img_border_value=(114, 114, 114),
            mask_border_value=0,
            interpolation="bilinear",
            border_mode="constant",
        ))

    # Color / intensity augmentation (before flips — flips are geometry-only).
    if color_jitter:
        pipeline.append(dict(
            type="PhotoMetricDistortion",
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18,
        ))

    # Albu-style pixel-level transforms implemented with cv2 (no albumentations dep).
    # Equivalent to: GaussNoise + GaussianBlur + CLAHE from albumentations.
    if albu:
        pipeline.extend([
            dict(type="GaussianBlurAug", max_kernel=7, prob=0.3),
            dict(type="CLAHEAug", clip_limit=4.0, tile_grid_size=(8, 8), prob=0.3),
        ])

    # Custom Gaussian noise (alternative to Albu; no extra dependency).
    if gaussian_noise:
        pipeline.append(dict(
            type="GaussianNoise",
            std=noise_std,
            prob=0.5,
        ))

    pipeline.extend([
        dict(type="RandomFlip", prob=0.5, direction="horizontal"),
        dict(type="RandomFlip", prob=0.5, direction="vertical"),
    ])

    # Diagonal flip = simultaneous H+V = transpose; combined with H/V gives D4.
    if diagonal_flip:
        pipeline.append(dict(type="RandomFlip", prob=0.5, direction="diagonal"))

    # GridMask: randomly masks rectangular grid regions for regularization.
    # Placed after flips so the mask pattern is not systematically aligned.
    if gridmask:
        pipeline.append(dict(
            type="GridMask",
            use_h=True,
            use_w=True,
            max_rotate=1,
            ratio=0.5,
            mode=1,
            prob=0.7,
        ))

    pipeline.extend([
        # Filter out degenerate GT bboxes (zero width/height) created by RandomCrop
        # or RandomRotate; without this, DeltaXYWH encoding produces log(0) = NaN.
        dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1)),
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
