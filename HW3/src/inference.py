"""
Inference script: run predictions on test images and output test-results.json.

Supports Test-Time Augmentation (TTA) with horizontal and vertical flips.

Usage:
    python inference.py --checkpoint ../work_dirs/fold0/best.pth
    python inference.py --checkpoint ../work_dirs/fold0/best.pth --tta
    python inference.py --checkpoints ../work_dirs/fold0/best.pth ../work_dirs/fold1/best.pth  # Ensemble
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on test set")

    # Input / Output
    parser.add_argument("--data_root", type=str, default="../dataset",
                        help="Root directory of the dataset")
    parser.add_argument("--test_dir", type=str, default="../dataset/test_release",
                        help="Test images directory")
    parser.add_argument("--id_mapping", type=str,
                        default="../dataset/test_image_name_to_ids.json",
                        help="Path to test_image_name_to_ids.json")
    parser.add_argument("--output", type=str, default="../test-results.json",
                        help="Output JSON file path")

    # Model (single checkpoint OR multiple for ensemble — exactly one required)
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoint", type=str,
                            help="Path to a single model checkpoint (.pth)")
    ckpt_group.add_argument("--checkpoints", type=str, nargs="+",
                            help="Multiple checkpoints for WBF ensemble inference")
    parser.add_argument("--backbone", type=str, default="convnextv2_base",
                        help="Backbone model name (must match training)")
    parser.add_argument("--fpn_channels", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--drop_path_rate", type=float, default=0.4)
    parser.add_argument("--bbox_loss", type=str, default="smoothl1",
                        choices=["smoothl1", "giou"],
                        help="Must match training value")
    parser.add_argument("--mask_head_convs", type=int, default=4,
                        help="Must match training value")
    parser.add_argument("--mask_roi_size", type=int, default=14,
                        help="CRITICAL: must match training value for checkpoint compatibility")

    # Inference settings
    parser.add_argument("--img_scale", type=int, nargs=2, default=[1024, 1024],
                        help="Inference image scale (H, W)")
    parser.add_argument("--score_threshold", type=float, default=0.05,
                        help="Final score threshold applied to merged predictions")
    parser.add_argument("--nms_threshold", type=float, default=0.5,
                        help="Per-model NMS IoU threshold")
    parser.add_argument("--max_det", type=int, default=300,
                        help="Maximum detections per image")

    # TTA
    parser.add_argument("--tta", action="store_true", default=False,
                        help="Enable test-time augmentation (H+V flip)")
    parser.add_argument("--tta_rotation", action="store_true", default=False,
                        help="Extend TTA with 90°CW + 90°CCW rotation views (requires --tta)")

    # Ensemble (only used when --checkpoints is set)
    parser.add_argument("--wbf_iou_threshold", type=float, default=0.55,
                        help="Bbox IoU threshold for WBF clustering across folds")
    parser.add_argument("--mask_vote_threshold", type=float, default=0.5,
                        help="Threshold to binarize the score-weighted average mask")
    parser.add_argument("--per_model_score_threshold", type=float, default=0.03,
                        help="Per-model score floor before WBF clustering (looser than final)")
    parser.add_argument("--load_all_models", action="store_true", default=False,
                        help="Keep all checkpoints in GPU memory at once (faster, more VRAM). "
                             "Default: load one fold at a time and cache predictions in RAM.")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")

    return parser.parse_args()


def encode_binary_mask(binary_mask):
    """Encode a binary mask to COCO RLE format."""
    arr = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(arr)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def build_model(args, checkpoint_path=None, score_threshold=None):
    """Build the model from config and load checkpoint.

    Args:
        args: Parsed CLI args.
        checkpoint_path: Override args.checkpoint when running ensemble. Falls
            back to args.checkpoint for backward compatibility.
        score_threshold: Override args.score_threshold (used by ensemble path
            to feed a looser per-model score floor into the rcnn head).
    """
    import dataset  # noqa: F401 -- registers ConvNeXtV2Backbone

    from mmdet.apis import init_detector
    from mmengine.config import Config
    from model_config import get_model_config

    model_cfg = get_model_config(
        backbone_name=args.backbone,
        pretrained=False,  # We load from checkpoint
        fpn_out_channels=args.fpn_channels,
        num_classes=args.num_classes,
        drop_path_rate=args.drop_path_rate,
        bbox_loss=args.bbox_loss,
        mask_head_convs=args.mask_head_convs,
        mask_roi_size=args.mask_roi_size,
    )

    # Update test_cfg thresholds
    score_thr = args.score_threshold if score_threshold is None else score_threshold
    model_cfg["test_cfg"]["rcnn"]["score_thr"] = score_thr
    model_cfg["test_cfg"]["rcnn"]["nms"]["iou_threshold"] = args.nms_threshold
    model_cfg["test_cfg"]["rcnn"]["max_per_img"] = args.max_det

    # Build a minimal config for init_detector
    # init_detector requires test_dataloader.dataset for the pipeline
    from dataset import get_test_pipeline
    test_pipeline = get_test_pipeline(img_scale=tuple(args.img_scale))

    full_cfg = dict(
        model=model_cfg,
        default_scope="mmdet",
        test_dataloader=dict(
            dataset=dict(
                type="CocoDataset",
                pipeline=test_pipeline,
            ),
        ),
        test_cfg=dict(type="TestLoop"),
        test_evaluator=dict(type="CocoMetric", metric=["segm"]),
    )
    cfg = Config(full_cfg)

    ckpt = checkpoint_path if checkpoint_path is not None else args.checkpoint
    model = init_detector(cfg, ckpt, device=args.device)
    model.eval()
    return model


def load_test_image(img_path):
    """Load a test .tif image, handle RGBA → BGR conversion."""
    import tifffile
    img = tifffile.imread(img_path)

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]  # RGBA → RGB

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, ::-1]  # RGB → BGR

    # Ensure contiguous uint8 (critical for correct tensor conversion)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    return img


def single_image_inference(model, img, img_scale):
    """Run inference on a single image (numpy array, BGR)."""
    from mmdet.apis import inference_detector
    result = inference_detector(model, img)
    return result


def flip_image(img, direction):
    """Flip image horizontally or vertically."""
    if direction == "horizontal":
        return cv2.flip(img, 1)
    elif direction == "vertical":
        return cv2.flip(img, 0)
    return img


def flip_masks(masks, direction, h, w):
    """Flip masks back to original orientation."""
    flipped = []
    for mask in masks:
        if direction == "horizontal":
            flipped.append(np.flip(mask, axis=1).copy())
        elif direction == "vertical":
            flipped.append(np.flip(mask, axis=0).copy())
        else:
            flipped.append(mask)
    return flipped


def tta_inference(model, img, img_scale, tta_rotation=False):
    """Test-time augmentation: original + H-flip + V-flip (+ optional 90°CW/CCW).

    With tta_rotation=False (default): 3 views — backward-compatible.
    With tta_rotation=True:            5 views — adds 90°CW and 90°CCW.

    Each rotation entry is a tuple (tag, cv2_rotate_code, inv_k) where inv_k is
    the k argument to np.rot90 that inverts the rotation.
    """
    from mmdet.apis import inference_detector

    h, w = img.shape[:2]
    all_bboxes = []
    all_labels = []
    all_scores = []
    all_masks = []

    # Build augmentation list.  String entries are flip directions; tuple entries
    # are rotation specs: ("rot", cv2_code, inv_k_for_np_rot90).
    tta_configs = [None, "horizontal", "vertical"]
    if tta_rotation:
        tta_configs += [
            ("rot", cv2.ROTATE_90_CLOCKWISE, 1),         # invert with np.rot90(k=1) = CCW
            ("rot", cv2.ROTATE_90_COUNTERCLOCKWISE, 3),  # invert with np.rot90(k=3) = CW
        ]

    for aug in tta_configs:
        # --- Apply augmentation ---
        if aug is None:
            aug_img = img
        elif isinstance(aug, str):
            aug_img = flip_image(img, aug)
        else:
            _, cv2_code, _ = aug
            aug_img = cv2.rotate(img, cv2_code)

        result = inference_detector(model, aug_img)
        pred = result.pred_instances

        bboxes = pred.bboxes.cpu().numpy()
        labels = pred.labels.cpu().numpy()
        scores = pred.scores.cpu().numpy()
        masks = pred.masks.cpu().numpy()

        # --- Inverse transform: map predictions back to original orientation ---
        if aug is None:
            pass  # no transform needed

        elif aug == "horizontal":
            if len(masks) > 0:
                mask_w = masks.shape[2]
                bboxes_inv = bboxes.copy()
                bboxes_inv[:, 0] = mask_w - bboxes[:, 2]
                bboxes_inv[:, 2] = mask_w - bboxes[:, 0]
                bboxes = bboxes_inv
                masks = np.array([np.flip(m, axis=1).copy() for m in masks])

        elif aug == "vertical":
            if len(masks) > 0:
                mask_h = masks.shape[1]
                bboxes_inv = bboxes.copy()
                bboxes_inv[:, 1] = mask_h - bboxes[:, 3]
                bboxes_inv[:, 3] = mask_h - bboxes[:, 1]
                bboxes = bboxes_inv
                masks = np.array([np.flip(m, axis=0).copy() for m in masks])

        else:
            # Rotation inverse transform.
            # After rotating original (H×W) image 90° CW or CCW, the rotated image
            # has shape (W×H).  mask_H_rot = W_orig and mask_W_rot = H_orig in the
            # model's output coordinate space.
            _, cv2_code, inv_k = aug
            if len(masks) > 0:
                mask_H_rot = masks.shape[1]  # = orig W
                mask_W_rot = masks.shape[2]  # = orig H
                nx1 = bboxes[:, 0]
                ny1 = bboxes[:, 1]
                nx2 = bboxes[:, 2]
                ny2 = bboxes[:, 3]
                bboxes_inv = bboxes.copy()
                if cv2_code == cv2.ROTATE_90_CLOCKWISE:
                    # Forward CW: (orig_x, orig_y) → (orig_H - orig_y, orig_x)
                    # Inverse:    orig_x1=ny1, orig_y1=mask_W_rot-nx2,
                    #             orig_x2=ny2, orig_y2=mask_W_rot-nx1
                    bboxes_inv[:, 0] = ny1
                    bboxes_inv[:, 1] = mask_W_rot - nx2
                    bboxes_inv[:, 2] = ny2
                    bboxes_inv[:, 3] = mask_W_rot - nx1
                else:  # ROTATE_90_COUNTERCLOCKWISE
                    # Forward CCW: (orig_x, orig_y) → (orig_y, orig_W - orig_x)
                    # Inverse:    orig_x1=mask_H_rot-ny2, orig_y1=nx1,
                    #             orig_x2=mask_H_rot-ny1, orig_y2=nx2
                    bboxes_inv[:, 0] = mask_H_rot - ny2
                    bboxes_inv[:, 1] = nx1
                    bboxes_inv[:, 2] = mask_H_rot - ny1
                    bboxes_inv[:, 3] = nx2
                bboxes = bboxes_inv
                masks = np.stack([np.rot90(m, k=inv_k, axes=(0, 1)).copy() for m in masks])

        all_bboxes.append(bboxes)
        all_labels.append(labels)
        all_scores.append(scores)
        all_masks.append(masks)

    # Concatenate all predictions
    if all(len(b) == 0 for b in all_bboxes):
        return np.array([]), np.array([]), np.array([]), np.array([])

    all_bboxes = np.concatenate(all_bboxes, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    # Apply class-wise NMS to merge TTA predictions
    keep_indices = []
    for cls_id in range(4):
        cls_mask = all_labels == cls_id
        if not cls_mask.any():
            continue
        cls_indices = np.where(cls_mask)[0]
        cls_bboxes = all_bboxes[cls_indices]
        cls_scores = all_scores[cls_indices]

        bboxes_tensor = torch.from_numpy(cls_bboxes).float()
        scores_tensor = torch.from_numpy(cls_scores).float()
        from torchvision.ops import nms
        keep = nms(bboxes_tensor, scores_tensor, iou_threshold=0.5)
        keep_indices.extend(cls_indices[keep.numpy()].tolist())

    keep_indices = sorted(keep_indices)
    return (
        all_bboxes[keep_indices],
        all_labels[keep_indices],
        all_scores[keep_indices],
        all_masks[keep_indices],
    )


def process_results(result, ori_h, ori_w):
    """Extract predictions from MMDetection result object."""
    pred = result.pred_instances

    bboxes = pred.bboxes.cpu().numpy()
    labels = pred.labels.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    masks = pred.masks.cpu().numpy()  # (N, H, W) binary masks

    return bboxes, labels, scores, masks


def resize_masks_to_ori(masks, ori_h, ori_w):
    """Resize a stack of binary masks to (ori_h, ori_w) with nearest interp."""
    if len(masks) == 0:
        return np.zeros((0, ori_h, ori_w), dtype=np.uint8)
    if masks.shape[1] == ori_h and masks.shape[2] == ori_w:
        return masks.astype(np.uint8)
    out = np.empty((len(masks), ori_h, ori_w), dtype=np.uint8)
    for i, m in enumerate(masks):
        out[i] = cv2.resize(m.astype(np.uint8), (ori_w, ori_h),
                            interpolation=cv2.INTER_NEAREST)
    return out


def gather_predictions_for_image(model, img, ori_h, ori_w, args, model_idx):
    """Run a single model (optionally with TTA) on one image and return
    predictions resized to the original image size, tagged with model_idx."""
    if args.tta:
        bboxes, labels, scores, masks = tta_inference(
            model, img, tuple(args.img_scale),
            tta_rotation=getattr(args, "tta_rotation", False),
        )
    else:
        result = single_image_inference(model, img, tuple(args.img_scale))
        bboxes, labels, scores, masks = process_results(result, ori_h, ori_w)

    bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    masks = resize_masks_to_ori(np.asarray(masks), ori_h, ori_w)

    model_ids = np.full(len(bboxes), model_idx, dtype=np.int32)
    return bboxes, labels, scores, masks, model_ids


def _bbox_iou(box, others):
    """IoU between a single box and an array of boxes. Boxes are xyxy."""
    if len(others) == 0:
        return np.zeros(0, dtype=np.float32)
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    area_b = (others[:, 2] - others[:, 0]).clip(min=0) * (others[:, 3] - others[:, 1]).clip(min=0)
    union = area_a + area_b - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def wbf_with_masks(bboxes, labels, scores, masks, model_ids,
                   num_models, num_classes, iou_thr=0.55, mask_thr=0.5):
    """Weighted Box Fusion + score-weighted mask averaging across models.

    For each class, greedily cluster detections by bbox IoU. Within a cluster:
        merged_bbox  = sum(s_i * bbox_i) / sum(s_i)
        merged_mask  = (sum(s_i * mask_i) / sum(s_i)) >= mask_thr
        merged_score = mean(s_i) * unique_models / num_models

    Detections seen by more folds get a score boost; single-fold detections
    are downweighted, which is what we want for ensemble robustness.

    All masks must already share spatial shape (ori_h, ori_w).
    """
    if len(bboxes) == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
                np.zeros((0,) + (masks.shape[1:] if masks.ndim == 3 else (0, 0)),
                         dtype=np.uint8))

    out_bboxes, out_labels, out_scores, out_masks = [], [], [], []

    for cls in range(num_classes):
        cls_mask = labels == cls
        if not cls_mask.any():
            continue
        cls_bboxes = bboxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_masks = masks[cls_mask]
        cls_model_ids = model_ids[cls_mask]

        # High-score-first ordering for greedy cluster seeding
        order = np.argsort(-cls_scores)
        cls_bboxes = cls_bboxes[order]
        cls_scores = cls_scores[order]
        cls_masks = cls_masks[order]
        cls_model_ids = cls_model_ids[order]

        used = np.zeros(len(cls_bboxes), dtype=bool)
        for i in range(len(cls_bboxes)):
            if used[i]:
                continue
            # Seed cluster with detection i; greedily attach later detections
            # whose IoU with the seed exceeds iou_thr.
            remaining = np.where(~used[i + 1:])[0] + (i + 1)
            ious = _bbox_iou(cls_bboxes[i], cls_bboxes[remaining]) if len(remaining) else np.zeros(0)
            members = [i] + remaining[ious >= iou_thr].tolist()
            for m in members:
                used[m] = True

            m_scores = cls_scores[members]
            m_bboxes = cls_bboxes[members]
            m_masks = cls_masks[members]
            m_model_ids = cls_model_ids[members]

            score_sum = float(m_scores.sum())
            if score_sum <= 0:
                continue

            weights = m_scores / score_sum  # (k,)
            merged_bbox = (m_bboxes * weights[:, None]).sum(axis=0)
            avg_mask = (m_masks.astype(np.float32) * weights[:, None, None]).sum(axis=0)
            merged_mask = (avg_mask >= mask_thr).astype(np.uint8)

            unique_models = len(np.unique(m_model_ids))
            merged_score = float(m_scores.mean()) * unique_models / max(num_models, 1)

            out_bboxes.append(merged_bbox)
            out_labels.append(cls)
            out_scores.append(merged_score)
            out_masks.append(merged_mask)

    if not out_bboxes:
        h, w = masks.shape[1:] if masks.ndim == 3 else (0, 0)
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
                np.zeros((0, h, w), dtype=np.uint8))

    return (np.stack(out_bboxes).astype(np.float32),
            np.array(out_labels, dtype=np.int64),
            np.array(out_scores, dtype=np.float32),
            np.stack(out_masks).astype(np.uint8))


def predictions_to_coco_results(image_id, ori_h, ori_w,
                                bboxes, labels, scores, masks,
                                score_threshold):
    """Turn (bboxes, labels, scores, masks) into a list of COCO RLE result dicts."""
    out = []
    for i in range(len(scores)):
        score = float(scores[i])
        if score < score_threshold:
            continue

        category_id = int(labels[i]) + 1  # mmdet 0-indexed → submission 1-indexed

        binary_mask = masks[i].astype(np.uint8)
        if binary_mask.shape[0] != ori_h or binary_mask.shape[1] != ori_w:
            binary_mask = cv2.resize(
                binary_mask, (ori_w, ori_h),
                interpolation=cv2.INTER_NEAREST,
            )

        rle = encode_binary_mask(binary_mask)
        out.append({
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": {
                "size": [ori_h, ori_w],
                "counts": rle["counts"],
            },
            "score": round(score, 6),
        })
    return out


def run_single_model(args, name_to_info, test_files):
    """Original single-checkpoint path (kept intact for backward compat)."""
    print("Building model...")
    model = build_model(args)
    print("Model loaded successfully!")

    img_scale = tuple(args.img_scale)
    results = []

    for img_name in tqdm(test_files, desc="Inference"):
        if img_name not in name_to_info:
            print(f"WARNING: {img_name} not found in id mapping, skipping!")
            continue

        img_info = name_to_info[img_name]
        image_id = img_info["id"]
        ori_h = img_info["height"]
        ori_w = img_info["width"]

        img = load_test_image(os.path.join(args.test_dir, img_name))

        if args.tta:
            bboxes, labels, scores, masks = tta_inference(
                model, img, img_scale,
                tta_rotation=getattr(args, "tta_rotation", False),
            )
        else:
            result = single_image_inference(model, img, img_scale)
            bboxes, labels, scores, masks = process_results(result, ori_h, ori_w)

        results.extend(predictions_to_coco_results(
            image_id, ori_h, ori_w, bboxes, labels, scores, masks,
            args.score_threshold,
        ))

    return results


def run_ensemble_load_all(args, name_to_info, test_files, checkpoints):
    """Ensemble path: keep all models in GPU memory, run per-image."""
    num_models = len(checkpoints)
    print(f"Building {num_models} models (all loaded simultaneously)...")
    models = []
    for idx, ckpt in enumerate(checkpoints):
        print(f"  [{idx+1}/{num_models}] Loading {ckpt}")
        models.append(build_model(args, checkpoint_path=ckpt,
                                  score_threshold=args.per_model_score_threshold))
    print("All models loaded.")

    results = []
    for img_name in tqdm(test_files, desc="Ensemble inference"):
        if img_name not in name_to_info:
            print(f"WARNING: {img_name} not found in id mapping, skipping!")
            continue

        img_info = name_to_info[img_name]
        image_id = img_info["id"]
        ori_h = img_info["height"]
        ori_w = img_info["width"]

        img = load_test_image(os.path.join(args.test_dir, img_name))

        all_b, all_l, all_s, all_m, all_mid = [], [], [], [], []
        for idx, model in enumerate(models):
            b, l, s, m, mid = gather_predictions_for_image(
                model, img, ori_h, ori_w, args, idx,
            )
            all_b.append(b); all_l.append(l); all_s.append(s)
            all_m.append(m); all_mid.append(mid)

        bboxes = np.concatenate(all_b) if all_b else np.zeros((0, 4), np.float32)
        labels = np.concatenate(all_l) if all_l else np.zeros(0, np.int64)
        scores = np.concatenate(all_s) if all_s else np.zeros(0, np.float32)
        masks = np.concatenate(all_m) if all_m else np.zeros((0, ori_h, ori_w), np.uint8)
        model_ids = np.concatenate(all_mid) if all_mid else np.zeros(0, np.int32)

        b, l, s, m = wbf_with_masks(
            bboxes, labels, scores, masks, model_ids,
            num_models=num_models, num_classes=args.num_classes,
            iou_thr=args.wbf_iou_threshold, mask_thr=args.mask_vote_threshold,
        )

        results.extend(predictions_to_coco_results(
            image_id, ori_h, ori_w, b, l, s, m, args.score_threshold,
        ))

    return results


def run_ensemble_sequential(args, name_to_info, test_files, checkpoints):
    """Ensemble path: load one fold at a time, cache predictions, then merge.

    Lower GPU memory usage: only one model resident at a time. Trades VRAM for
    host RAM (cached masks at original resolution).
    """
    num_models = len(checkpoints)

    # cache[image_id] = list of (bboxes, labels, scores, masks, model_ids) tuples
    cache = {}

    for fold_idx, ckpt in enumerate(checkpoints):
        print(f"\n[Fold {fold_idx+1}/{num_models}] Loading {ckpt}")
        model = build_model(args, checkpoint_path=ckpt,
                            score_threshold=args.per_model_score_threshold)

        for img_name in tqdm(test_files, desc=f"Fold {fold_idx} inference"):
            if img_name not in name_to_info:
                continue

            img_info = name_to_info[img_name]
            image_id = img_info["id"]
            ori_h = img_info["height"]
            ori_w = img_info["width"]

            img = load_test_image(os.path.join(args.test_dir, img_name))
            b, l, s, m, mid = gather_predictions_for_image(
                model, img, ori_h, ori_w, args, fold_idx,
            )
            cache.setdefault(image_id, []).append((b, l, s, m, mid))

        # Free GPU memory before loading next fold
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nMerging predictions across folds via WBF...")
    results = []
    # Build image_id -> ori shape lookup
    id_to_info = {info["id"]: info for info in name_to_info.values()}

    for image_id, fold_preds in tqdm(cache.items(), desc="WBF merge"):
        info = id_to_info[image_id]
        ori_h, ori_w = info["height"], info["width"]

        bboxes = np.concatenate([p[0] for p in fold_preds]) if fold_preds else np.zeros((0, 4), np.float32)
        labels = np.concatenate([p[1] for p in fold_preds]) if fold_preds else np.zeros(0, np.int64)
        scores = np.concatenate([p[2] for p in fold_preds]) if fold_preds else np.zeros(0, np.float32)
        masks = np.concatenate([p[3] for p in fold_preds]) if fold_preds else np.zeros((0, ori_h, ori_w), np.uint8)
        model_ids = np.concatenate([p[4] for p in fold_preds]) if fold_preds else np.zeros(0, np.int32)

        b, l, s, m = wbf_with_masks(
            bboxes, labels, scores, masks, model_ids,
            num_models=num_models, num_classes=args.num_classes,
            iou_thr=args.wbf_iou_threshold, mask_thr=args.mask_vote_threshold,
        )

        results.extend(predictions_to_coco_results(
            image_id, ori_h, ori_w, b, l, s, m, args.score_threshold,
        ))

    return results


def main():
    args = parse_args()

    with open(args.id_mapping, "r") as f:
        id_mapping = json.load(f)
    name_to_info = {item["file_name"]: item for item in id_mapping}

    test_files = sorted(
        [f for f in os.listdir(args.test_dir) if f.endswith(".tif")]
    )
    print(f"Found {len(test_files)} test images")

    checkpoints = [args.checkpoint] if args.checkpoint else args.checkpoints

    if len(checkpoints) == 1:
        results = run_single_model(args, name_to_info, test_files)
    else:
        print(f"Ensemble mode: {len(checkpoints)} checkpoints, "
              f"load_all_models={args.load_all_models}, tta={args.tta}, "
              f"wbf_iou={args.wbf_iou_threshold}, mask_thr={args.mask_vote_threshold}")
        if args.load_all_models:
            results = run_ensemble_load_all(args, name_to_info, test_files, checkpoints)
        else:
            results = run_ensemble_sequential(args, name_to_info, test_files, checkpoints)

    with open(args.output, "w") as f:
        json.dump(results, f)

    print(f"\nSaved {len(results)} predictions to {args.output}")
    print(f"Unique image_ids: {len(set(r['image_id'] for r in results))}")


if __name__ == "__main__":
    main()
