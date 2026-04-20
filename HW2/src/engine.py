"""Training and evaluation engine for DETR."""

import copy
import math
import sys
import time

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from criterion import box_cxcywh_to_xyxy


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class ModelEMA:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model, decay=0.9998):
        import io
        # Clone model via serialize/deserialize to avoid deepcopy non-leaf tensor error
        buf = io.BytesIO()
        torch.save(model, buf)
        buf.seek(0)
        self.module = torch.load(buf, weights_only=False)
        self.module.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_v, model_v in zip(
            self.module.state_dict().values(), model.state_dict().values()
        ):
            ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))


# ---------------------------------------------------------------------------
# Learning rate warmup
# ---------------------------------------------------------------------------

def adjust_learning_rate(optimizer, epoch, args):
    """LR schedule with linear warmup."""
    if epoch < args.warmup_epochs:
        # Linear warmup: (epoch+1) so epoch 0 gets a non-zero LR
        alpha = (epoch + 1) / max(args.warmup_epochs, 1)
        for pg in optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * alpha
    elif args.lr_scheduler == "step":
        if epoch in args.lr_drop_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = pg["lr"] * 0.1
    elif args.lr_scheduler == "cosine":
        progress = (epoch - args.warmup_epochs) / max(
            args.epochs - args.warmup_epochs, 1
        )
        alpha = 0.5 * (1.0 + math.cos(math.pi * progress))
        alpha = max(alpha, args.lr_min_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * alpha


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                    args, scaler=None, ema=None):
    model.train()
    criterion.train()

    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}", leave=True)
    for i, (images, masks, targets) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images, masks)
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                loss = sum(
                    loss_dict[k] * weight_dict[k]
                    for k in loss_dict if k in weight_dict
                )
                loss = loss / args.accumulate_steps
            scaler.scale(loss).backward()
            if (i + 1) % args.accumulate_steps == 0:
                if args.clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_max_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)
        else:
            outputs = model(images, masks)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict if k in weight_dict
            )
            loss = loss / args.accumulate_steps
            loss.backward()
            if (i + 1) % args.accumulate_steps == 0:
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_max_norm,
                    )
                optimizer.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

        total_loss += loss.item() * args.accumulate_steps
        num_batches += 1

        avg = total_loss / num_batches
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}")

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Evaluation (COCO mAP)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data_loader, device, coco_gt, num_classes,
             focal_loss=False, score_thresh=0.001):
    """Evaluate using top-1 per query decoding (aligned with inference)."""
    model.eval()
    results = []

    for images, masks, targets in tqdm(data_loader, desc="Evaluating", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images, masks)
        pred_logits = outputs["pred_logits"]  # (B, Q, C+1) or (B, Q, C)
        pred_boxes = outputs["pred_boxes"]    # (B, Q, 4)

        if focal_loss:
            probs = pred_logits[:, :, :-1].sigmoid()
        else:
            probs = pred_logits.softmax(-1)[:, :, :-1]

        # Top-1 per query
        max_scores, max_labels = probs.max(dim=-1)  # (B, Q)

        for idx in range(len(targets)):
            img_id = int(targets[idx]["image_id"])
            orig_h, orig_w = targets[idx]["orig_size"].tolist()

            scores = max_scores[idx]       # (Q,)
            labels = max_labels[idx]       # (Q,)
            boxes = pred_boxes[idx]        # (Q, 4) cxcywh normalized

            keep = scores > score_thresh
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            if len(scores) == 0:
                continue

            # Convert to absolute xyxy
            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            scale = torch.tensor(
                [orig_w, orig_h, orig_w, orig_h],
                device=boxes.device, dtype=boxes.dtype,
            )
            boxes_xyxy = boxes_xyxy * scale
            boxes_xyxy[:, 0::2].clamp_(min=0, max=orig_w)
            boxes_xyxy[:, 1::2].clamp_(min=0, max=orig_h)

            # Filter degenerate boxes
            w_box = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
            h_box = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
            valid = (w_box >= 1) & (h_box >= 1)
            boxes_xyxy = boxes_xyxy[valid]
            scores = scores[valid]
            labels = labels[valid]

            # Convert to xywh
            boxes_xywh = boxes_xyxy.clone()
            boxes_xywh[:, 2] -= boxes_xywh[:, 0]
            boxes_xywh[:, 3] -= boxes_xywh[:, 1]

            for j in range(len(scores)):
                results.append({
                    "image_id": img_id,
                    "category_id": labels[j].item() + 1,
                    "bbox": boxes_xywh[j].cpu().tolist(),
                    "score": scores[j].cpu().item(),
                })

    if not results:
        print("Warning: no predictions generated!")
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP": coco_eval.stats[0],
        "mAP_50": coco_eval.stats[1],
        "mAP_75": coco_eval.stats[2],
    }
