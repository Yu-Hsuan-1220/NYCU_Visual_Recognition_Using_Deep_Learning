"""COCO-format dataset and transforms for DETR digit detection."""

import json
import os
import random

import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


def box_xyxy_to_cxcywh(boxes):
    # Convert (x1, y1, x2, y2) to (cx, cy, w, h)
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1
    )


class CocoDetection(data.Dataset):
    """COCO-format detection dataset for training and validation."""

    def __init__(self, img_dir, ann_file, transforms=None):
        with open(ann_file, "r") as f:
            coco = json.load(f)
        self.img_dir = img_dir
        self.transforms = transforms
        self.images = sorted(coco["images"], key=lambda x: x["id"])
        self.img_to_anns = {}
        for ann in coco["annotations"]:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img = Image.open(
            os.path.join(self.img_dir, img_info["file_name"])
        ).convert("RGB")
        anns = self.img_to_anns.get(img_info["id"], [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            # Map category_id from 1-10 to 0-9
            labels.append(ann["category_id"] - 1)
        target = {
            "boxes": torch.as_tensor(
                boxes, dtype=torch.float32
            ).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": img_info["id"],
            "orig_size": torch.tensor(
                [img_info["height"], img_info["width"]]
            ),
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


class TestDataset(data.Dataset):
    """Test dataset without annotations."""

    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.images = []
        for fname in sorted(os.listdir(img_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img_id = int(os.path.splitext(fname)[0])
                self.images.append({"id": img_id, "file_name": fname})
        self.images.sort(key=lambda x: x["id"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img = Image.open(
            os.path.join(self.img_dir, info["file_name"])
        ).convert("RGB")
        w, h = img.size
        target = {"image_id": info["id"], "orig_size": torch.tensor([h, w])}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


# ---------------------------------------------------------------------------
# Transform helpers (operate on PIL images + target dicts)
# ---------------------------------------------------------------------------

def _resize(image, target, size, max_size):
    w, h = image.size
    min_side = float(min(w, h))
    max_side = float(max(w, h))
    scale = size / min_side
    if max_size is not None and max_side * scale > max_size:
        scale = max_size / max_side
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    image = TF.resize(image, [new_h, new_w])
    if "boxes" in target and len(target["boxes"]):
        ratio_w = new_w / w
        ratio_h = new_h / h
        target["boxes"] = target["boxes"] * torch.as_tensor(
            [ratio_w, ratio_h, ratio_w, ratio_h]
        )
    target["size"] = torch.tensor([new_h, new_w])
    return image, target


def _crop(image, target, region):
    top, left, h, w = region
    image = TF.crop(image, top, left, h, w)
    if "boxes" in target and len(target["boxes"]):
        boxes = target["boxes"] - torch.as_tensor(
            [left, top, left, top], dtype=torch.float32
        )
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 2] > boxes[:, 0] + 1) & \
               (boxes[:, 3] > boxes[:, 1] + 1)
        target["boxes"] = boxes[keep]
        target["labels"] = target["labels"][keep]
    target["size"] = torch.tensor([h, w])
    return image, target


def _hflip(image, target):
    image = TF.hflip(image)
    w = image.width
    if "boxes" in target and len(target["boxes"]):
        boxes = target["boxes"]
        target["boxes"] = torch.stack(
            [w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]],
            dim=-1
        )
    return image, target


# ---------------------------------------------------------------------------
# Transform classes
# ---------------------------------------------------------------------------

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return _hflip(img, target)
        return img, target


class RandomResize:
    def __init__(self, sizes, max_size=None):
        self.sizes = (
            list(sizes)
            if not isinstance(sizes, (list, tuple))
            else sizes
        )
        self.max_size = max_size

    def __call__(self, img, target):
        size = random.choice(self.sizes)
        return _resize(img, target, size, self.max_size)


class RandomSizeCrop:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, target):
        w = random.randint(
            self.min_size, min(img.width, self.max_size)
        )
        h = random.randint(
            self.min_size, min(img.height, self.max_size)
        )
        region = T.RandomCrop.get_params(img, (h, w))
        return _crop(img, target, region)


class RandomResizedCropFixed:
    """Resize to slightly larger than target.

    Then random crop to exact (H, W).
    """

    def __init__(
        self, target_h, target_w, scale_min=0.9, scale_max=1.0
    ):
        self.target_h = target_h
        self.target_w = target_w
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, img, target):
        w, h = img.size
        # Compute scale so that after resize the image covers target_h x
        # target_w
        scale_h = self.target_h / h
        scale_w = self.target_w / w
        base_scale = max(scale_h, scale_w)
        # Random extra zoom in [1/scale_max, 1/scale_min] to simulate
        # crop range
        extra = random.uniform(
            1.0 / self.scale_max, 1.0 / self.scale_min
        )
        scale = base_scale * extra
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        # Ensure at least target size
        new_w = max(new_w, self.target_w)
        new_h = max(new_h, self.target_h)
        img, target = _resize(
            img, target, min(new_h, new_w), max_size=None
        )
        # Force exact size via resize if aspect changed
        cur_w, cur_h = img.size
        if cur_w < self.target_w or cur_h < self.target_h:
            new_w2 = max(cur_w, self.target_w)
            new_h2 = max(cur_h, self.target_h)
            img = TF.resize(img, [new_h2, new_w2])
            if "boxes" in target and len(target["boxes"]):
                rw = new_w2 / cur_w
                rh = new_h2 / cur_h
                target["boxes"] = target["boxes"] * torch.as_tensor(
                    [rw, rh, rw, rh], dtype=torch.float32
                )
            cur_w, cur_h = new_w2, new_h2
        # Random crop to exact target size
        top = random.randint(0, max(cur_h - self.target_h, 0))
        left = random.randint(0, max(cur_w - self.target_w, 0))
        return _crop(
            img, target, (top, left, self.target_h, self.target_w)
        )


class RandomSelect:
    """Randomly selects between two transforms.

    Selects between two transforms with probability p.
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target):
        return self.jitter(img), target


class RandomGrayscale:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = TF.to_grayscale(img, num_output_channels=3)
        return img, target


class RandomGaussianBlur:
    """Apply Gaussian blur with a random kernel size."""

    def __init__(self, p=0.1, kernel_sizes=(3, 5)):
        self.p = p
        self.kernel_sizes = kernel_sizes

    def __call__(self, img, target):
        if random.random() < self.p:
            k = random.choice(self.kernel_sizes)
            img = TF.gaussian_blur(img, kernel_size=k)
        return img, target


class RandomISOSNoise:
    """Add ISO-like Gaussian noise to simulate camera sensor noise."""

    def __init__(self, p=0.2, intensity=0.05):
        self.p = p
        self.intensity = intensity

    def __call__(self, img, target):
        if random.random() < self.p:
            import numpy as np
            img_array = np.array(img, dtype=np.float32) / 255.0
            noise = np.random.randn(*img_array.shape) * self.intensity
            img_array = np.clip(img_array + noise, 0, 1)
            img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        return img, target


class RandomTranslation:
    """Random translation with bounding box adjustment and filtering."""

    def __init__(self, p=0.3, max_shift=0.1, min_area_ratio=0.25):
        """
        Args:
            p: probability of applying translation
            max_shift: maximum shift as fraction of image size
            min_area_ratio: drop bbox if remaining area < ratio * original area
        """
        self.p = p
        self.max_shift = max_shift
        self.min_area_ratio = min_area_ratio

    def __call__(self, img, target):
        if random.random() >= self.p:
            return img, target

        w, h = img.size
        # Random shift in pixels
        shift_x = int(
            random.uniform(-self.max_shift, self.max_shift) * w
        )
        shift_y = int(
            random.uniform(-self.max_shift, self.max_shift) * h
        )

        # Translate image using affine transform
        img = TF.affine(
            img, angle=0, translate=(shift_x, shift_y),
            scale=1.0, shear=0, fill=0
        )

        if "boxes" not in target or len(target["boxes"]) == 0:
            return img, target

        boxes = target["boxes"].clone()
        labels = target["labels"]

        # Calculate original areas
        orig_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Shift boxes
        boxes[:, 0] += shift_x  # x1
        boxes[:, 2] += shift_x  # x2
        boxes[:, 1] += shift_y  # y1
        boxes[:, 3] += shift_y  # y2

        # Clip to image boundaries
        boxes[:, 0].clamp_(min=0, max=w)
        boxes[:, 2].clamp_(min=0, max=w)
        boxes[:, 1].clamp_(min=0, max=h)
        boxes[:, 3].clamp_(min=0, max=h)

        # Calculate new areas
        new_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Keep boxes that have sufficient remaining area and are not
        # degenerate
        keep = (
            (new_areas >= self.min_area_ratio * orig_areas)
            & (boxes[:, 2] > boxes[:, 0] + 1)
            & (boxes[:, 3] > boxes[:, 1] + 1)
        )

        target["boxes"] = boxes[keep]
        target["labels"] = labels[keep]

        return img, target


class RandomExpand:
    """Randomly expand (zoom-out) the image.

    Places it on a larger canvas filled with ImageNet mean pixel value.
    Bounding boxes are shifted accordingly. This effectively simulates
    objects appearing smaller with more surrounding context.

    Args:
        p: probability of applying the expand.
        max_ratio: maximum expansion ratio. Canvas side length is
            ``original_side * uniform(1, 1 + max_ratio)``.
    """

    MEAN_PIXEL = (124, 116, 104)  # BGR-ish ImageNet mean as uint8

    def __init__(self, p=0.3, max_ratio=0.2):
        self.p = p
        self.max_ratio = max_ratio

    def __call__(self, img, target):
        if random.random() >= self.p:
            return img, target

        w, h = img.size
        ratio = random.uniform(1.0, 1.0 + self.max_ratio)
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))

        # Random placement of the original image on the canvas
        left = random.randint(0, new_w - w)
        top = random.randint(0, new_h - h)

        # Create canvas filled with mean pixel
        canvas = Image.new("RGB", (new_w, new_h), self.MEAN_PIXEL)
        canvas.paste(img, (left, top))

        # Shift bounding boxes
        if "boxes" in target and len(target["boxes"]):
            boxes = target["boxes"].clone()
            boxes[:, 0] += left  # x1
            boxes[:, 1] += top  # y1
            boxes[:, 2] += left  # x2
            boxes[:, 3] += top  # y2
            target["boxes"] = boxes

        target["size"] = torch.tensor([new_h, new_w])
        return canvas, target


class RandomRotation:
    """Small-angle rotation with bounding box adjustment."""

    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, img, target):
        angle = random.uniform(-self.max_angle, self.max_angle)
        w, h = img.size
        img = TF.rotate(img, angle, expand=False, fill=0)

        if "boxes" not in target or len(target["boxes"]) == 0:
            return img, target

        # Rotate box corners and take axis-aligned bounding box
        import math
        cx_img, cy_img = w / 2.0, h / 2.0
        # PIL rotates counter-clockwise
        rad = math.radians(-angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        boxes = target["boxes"]
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        # 4 corners per box
        corners_x = torch.stack([x1, x2, x2, x1], dim=1)  # (N, 4)
        corners_y = torch.stack([y1, y1, y2, y2], dim=1)  # (N, 4)
        # Translate to origin, rotate, translate back
        rx = (
            cos_a * (corners_x - cx_img)
            - sin_a * (corners_y - cy_img)
            + cx_img
        )
        ry = (
            sin_a * (corners_x - cx_img)
            + cos_a * (corners_y - cy_img)
            + cy_img
        )
        new_x1 = rx.min(dim=1).values.clamp(min=0, max=w)
        new_y1 = ry.min(dim=1).values.clamp(min=0, max=h)
        new_x2 = rx.max(dim=1).values.clamp(min=0, max=w)
        new_y2 = ry.max(dim=1).values.clamp(min=0, max=h)
        new_boxes = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)
        # Filter degenerate
        keep = (
            (new_boxes[:, 2] > new_boxes[:, 0] + 1)
            & (new_boxes[:, 3] > new_boxes[:, 1] + 1)
        )
        target["boxes"] = new_boxes[keep]
        target["labels"] = target["labels"][keep]
        return img, target


class ToTensor:
    def __call__(self, img, target):
        return TF.to_tensor(img), target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = TF.normalize(img, mean=self.mean, std=self.std)
        h, w = img.shape[-2:]
        if "boxes" in target and len(target["boxes"]):
            boxes = box_xyxy_to_cxcywh(target["boxes"])
            boxes = boxes / torch.tensor(
                [w, h, w, h], dtype=torch.float32
            )
            target["boxes"] = boxes
        target["size"] = torch.tensor([h, w])
        return img, target


# ---------------------------------------------------------------------------
# Build transforms
# ---------------------------------------------------------------------------

def _resize_fixed(image, target, target_h, target_w):
    """Resize image to exact (target_h, target_w)."""
    w, h = image.size
    image = TF.resize(image, [target_h, target_w])
    if "boxes" in target and len(target["boxes"]):
        ratio_w = target_w / w
        ratio_h = target_h / h
        target["boxes"] = target["boxes"] * torch.as_tensor(
            [ratio_w, ratio_h, ratio_w, ratio_h], dtype=torch.float32
        )
    target["size"] = torch.tensor([target_h, target_w])
    return image, target


class ResizeFixed:
    """Resize to exact (H, W)."""

    def __init__(self, target_h, target_w):
        self.target_h = target_h
        self.target_w = target_w

    def __call__(self, img, target):
        return _resize_fixed(img, target, self.target_h, self.target_w)


def make_transforms(split, args):
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    target_h = getattr(args, 'fixed_h', 320)
    target_w = getattr(args, 'fixed_w', 640)

    if split == "train":
        transforms_list = [
            ColorJitter(
                args.color_jitter, args.color_jitter,
                args.color_jitter, args.color_jitter * 0.25
            ),
            RandomGrayscale(p=0.05),
            RandomGaussianBlur(
                p=getattr(args, 'gaussian_blur_p', 0.0)
            ),
        ]

        # ISO Noise augmentation
        if getattr(args, 'aug_iso_noise', False):
            transforms_list.append(
                RandomISOSNoise(
                    p=getattr(args, 'aug_iso_noise_p', 0.2),
                    intensity=getattr(
                        args, 'aug_iso_noise_intensity', 0.05
                    ),
                )
            )

        # Random translation augmentation (before crop to allow more
        # variation)
        if getattr(args, 'aug_translation', False):
            transforms_list.append(
                RandomTranslation(
                    p=getattr(args, 'aug_translation_p', 0.3),
                    max_shift=getattr(
                        args, 'aug_translation_max_shift', 0.1
                    ),
                    min_area_ratio=getattr(
                        args, 'aug_translation_min_area_ratio', 0.25
                    ),
                )
            )

        # Random expand (zoom-out) augmentation
        if getattr(args, 'aug_expand', False):
            transforms_list.append(
                RandomExpand(
                    p=getattr(args, 'aug_expand_p', 0.3),
                    max_ratio=getattr(
                        args, 'aug_expand_max_ratio', 0.2
                    ),
                )
            )

        transforms_list.extend([
            RandomResizedCropFixed(
                target_h, target_w,
                scale_min=0.9, scale_max=1.0
            ),
            normalize,
        ])
        return Compose(transforms_list)

    return Compose([
        ResizeFixed(target_h, target_w),
        normalize,
    ])


# ---------------------------------------------------------------------------
# Collate function (pads images to max size in batch, creates masks)
# ---------------------------------------------------------------------------

def collate_fn(batch):
    images, targets = list(zip(*batch))
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    padded = torch.zeros(len(images), 3, max_h, max_w)
    masks = torch.ones(len(images), max_h, max_w, dtype=torch.bool)
    for i, img in enumerate(images):
        padded[i, :, : img.shape[1], : img.shape[2]] = img
        masks[i, : img.shape[1], : img.shape[2]] = False
    return padded, masks, list(targets)
