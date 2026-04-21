"""Inference script for Deformable DETR digit detection."""

import argparse
import json
import types

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from criterion import box_cxcywh_to_xyxy
from dataset import TestDataset, collate_fn
from model import DeformableDETR


def get_args():
    p = argparse.ArgumentParser(
        description="Deformable DETR Digit Detection Inference"
    )
    p.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    p.add_argument("--test_img_dir", default="../dataset/test")
    p.add_argument("--output_file", default="pred.json")
    p.add_argument("--score_threshold", type=float, default=0.01)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_size", type=int, default=800)
    p.add_argument("--max_size", type=int, default=1333)
    p.add_argument("--fixed_h", type=int, default=320,
                   help="Fixed image height (overrides val_size/max_size)")
    p.add_argument("--fixed_w", type=int, default=640,
                   help="Fixed image width (overrides val_size/max_size)")
    p.add_argument("--tta", action="store_true",
                   help="Test-time augmentation (horizontal flip)")
    p.add_argument("--nms_iou", type=float, default=0.6,
                   help="NMS IoU threshold (used with TTA)")
    p.add_argument("--use_ema", action="store_true", default=True,
                   help="Use EMA weights if available")
    return p.parse_args()


def build_model_from_checkpoint(ckpt, device):
    """Reconstruct Deformable DETR model from saved args in checkpoint."""
    saved = ckpt.get("args", {})
    model = DeformableDETR(
        num_classes=saved.get("num_classes", 10),
        hidden_dim=saved.get("hidden_dim", 256),
        nheads=saved.get("nheads", 8),
        num_encoder_layers=saved.get("num_encoder_layers", 6),
        num_decoder_layers=saved.get("num_decoder_layers", 6),
        dim_feedforward=saved.get("dim_feedforward", 1024),
        dropout=saved.get("dropout", 0.1),
        num_queries=saved.get("num_queries", 50),
        num_feature_levels=saved.get("num_feature_levels", 4),
        enc_n_points=saved.get("enc_n_points", 4),
        dec_n_points=saved.get("dec_n_points", 4),
        aux_loss=False,
        with_box_refine=saved.get("with_box_refine", False),
        pretrained_backbone=False,
    )
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
        print("Loaded model weights")
    return model.to(device).eval()


def make_test_transforms(args):
    """Create a minimal args namespace for make_transforms."""
    ns = types.SimpleNamespace(
        val_size=args.val_size,
        max_size=args.max_size,
        min_size=args.val_size,
        max_size_train=args.val_size,
        color_jitter=0,
        fixed_h=args.fixed_h,
        fixed_w=args.fixed_w,
    )
    from dataset import make_transforms as _mt
    return _mt("val", ns)


def _decode_single_image(pred_logits, pred_boxes, orig_h, orig_w,
                         score_thresh, focal_loss):
    """Top-1 per query decoding for a single image."""
    if focal_loss:
        probs = pred_logits[:, :-1].sigmoid()
    else:
        probs = pred_logits.softmax(-1)[:, :-1]

    max_scores, max_labels = probs.max(dim=-1)

    keep = max_scores > score_thresh
    max_scores = max_scores[keep]
    max_labels = max_labels[keep]
    boxes = pred_boxes[keep]

    if len(max_scores) == 0:
        return None

    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    scale = torch.tensor([orig_w, orig_h, orig_w, orig_h],
                         device=boxes.device, dtype=boxes.dtype)
    boxes_xyxy = boxes_xyxy * scale
    boxes_xyxy[:, 0::2].clamp_(min=0, max=orig_w)
    boxes_xyxy[:, 1::2].clamp_(min=0, max=orig_h)

    w_box = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    h_box = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    valid = (w_box >= 1) & (h_box >= 1)
    boxes_xyxy = boxes_xyxy[valid]
    max_scores = max_scores[valid]
    max_labels = max_labels[valid]

    boxes_xywh = boxes_xyxy.clone()
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]

    return boxes_xywh.cpu(), max_scores.cpu(), max_labels.cpu()


def _apply_per_class_nms(boxes_xyxy, scores, labels, iou_thresh):
    keep = torchvision.ops.batched_nms(boxes_xyxy, scores, labels, iou_thresh)
    return keep


@torch.no_grad()
def run_inference(model, data_loader, device, score_thresh, num_classes,
                  focal_loss=False, tta=False, nms_iou=0.5):
    results = []
    for images, masks, targets in tqdm(data_loader, desc="Inference"):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images, masks)
        all_logits = [outputs["pred_logits"]]
        all_boxes = [outputs["pred_boxes"]]

        if tta:
            flip_out = model(torch.flip(images, [-1]), torch.flip(masks, [-1]))
            flip_boxes = flip_out["pred_boxes"].clone()
            flip_boxes[..., 0] = 1 - flip_boxes[..., 0]
            all_logits.append(flip_out["pred_logits"])
            all_boxes.append(flip_boxes)

        pred_logits = torch.cat(all_logits, dim=1)
        pred_boxes = torch.cat(all_boxes, dim=1)

        for idx in range(len(targets)):
            img_id = int(targets[idx]["image_id"])
            orig_h, orig_w = targets[idx]["orig_size"].tolist()
            decoded = _decode_single_image(
                pred_logits[idx], pred_boxes[idx],
                orig_h, orig_w, score_thresh, focal_loss,
            )
            if decoded is None:
                continue
            boxes_xywh, scores, labels = decoded
            if tta:
                boxes_xyxy_nms = boxes_xywh.clone()
                boxes_xyxy_nms[:, 2] += boxes_xyxy_nms[:, 0]
                boxes_xyxy_nms[:, 3] += boxes_xyxy_nms[:, 1]
                keep = _apply_per_class_nms(
                    boxes_xyxy_nms, scores, labels, nms_iou,
                )
                boxes_xywh, scores, labels = \
                    boxes_xywh[keep], scores[keep], labels[keep]

            for j in range(len(scores)):
                results.append({
                    "image_id": img_id,
                    "category_id": labels[j].item() + 1,
                    "bbox": [round(v, 2) for v in boxes_xywh[j].tolist()],
                    "score": round(scores[j].item(), 6),
                })
    return results


def main():
    args = get_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    num_classes = saved_args.get("num_classes", 10)
    focal_loss = saved_args.get("focal_loss", False)

    if "val_size" in saved_args:
        args.val_size = saved_args["val_size"]
    if "max_size" in saved_args:
        args.max_size = saved_args["max_size"]

    model = build_model_from_checkpoint(ckpt, device)
    transforms = make_test_transforms(args)
    test_ds = TestDataset(args.test_img_dir, transforms=transforms)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    results = run_inference(
        model, test_loader, device, args.score_threshold,
        num_classes, focal_loss, tta=args.tta, nms_iou=args.nms_iou,
    )

    with open(args.output_file, "w") as f:
        json.dump(results, f)

    import zipfile
    zip_name = args.output_file.replace(".json", ".zip")
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output_file, "pred.json")


if __name__ == "__main__":
    main()
