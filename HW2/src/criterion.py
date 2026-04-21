"""Hungarian Matcher and Set Criterion for DETR training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
    )


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6), union


def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / (area + 1e-6)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2.0):
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


# ---------------------------------------------------------------------------
# Hungarian Matcher
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and ground truth."""

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0,
                 focal_loss=False, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        B, Q = outputs["pred_logits"].shape[:2]

        if self.focal_loss:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets])

        if len(tgt_ids) == 0:
            return [(torch.as_tensor([], dtype=torch.int64),
                     torch.as_tensor([], dtype=torch.int64))] * B

        # Classification cost
        if self.focal_loss:
            neg_cost = (1 - self.focal_alpha) * (
                out_prob ** self.focal_gamma
            ) * (-(1 - out_prob + 1e-8).log())
            pos_cost = self.focal_alpha * (
                (1 - out_prob) ** self.focal_gamma
            ) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]
        else:
            cost_class = -out_prob[:, tgt_ids]

        # BBox L1 cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU cost
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox),
        )

        C = (self.cost_class * cost_class
             + self.cost_bbox * cost_bbox
             + self.cost_giou * cost_giou)
        C = C.view(B, Q, -1).cpu()

        sizes = [len(t["labels"]) for t in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (torch.as_tensor(i, dtype=torch.int64),
             torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


# ---------------------------------------------------------------------------
# Set Criterion
# ---------------------------------------------------------------------------

class SetCriterion(nn.Module):
    """DETR loss: classification + bbox (L1 + GIoU) with optional aux loss."""

    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1,
                 focal_loss=False, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        if not focal_loss:
            empty_weight = torch.ones(num_classes + 1)
            empty_weight[-1] = eos_coef
            self.register_buffer("empty_weight", empty_weight)

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][j] for t, (_, j) in zip(targets, indices)]
        )

        if self.focal_loss:
            target_onehot = torch.zeros(
                src_logits.shape, dtype=src_logits.dtype,
                device=src_logits.device,
            )
            target_onehot[idx[0], idx[1], target_classes_o] = 1
            loss_ce = sigmoid_focal_loss(
                src_logits, target_onehot, num_boxes,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
            )
        else:
            target_classes = torch.full(
                src_logits.shape[:2], self.num_classes,
                dtype=torch.int64, device=src_logits.device,
            )
            target_classes[idx] = target_classes_o
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight,
            )
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0,
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        return {
            "loss_bbox": loss_bbox.sum() / num_boxes,
            "loss_giou": loss_giou.sum() / num_boxes,
        }

    def forward(self, outputs, targets):
        out_no_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(out_no_aux, targets)

        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux, targets)
                aux_loss_dict = self.loss_labels(
                    aux, targets, aux_indices, num_boxes
                )
                aux_loss_dict.update(
                    self.loss_boxes(aux, targets, aux_indices, num_boxes)
                )
                losses.update({
                    k + f"_{i}": v for k, v in aux_loss_dict.items()
                })

        return losses


def build_criterion(args):
    matcher = HungarianMatcher(
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )

    weight_dict = {
        "loss_ce": args.loss_ce_coef,
        "loss_bbox": args.loss_bbox_coef,
        "loss_giou": args.loss_giou_coef,
    }
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.num_decoder_layers - 1):
            aux_weight_dict[f"loss_ce_{i}"] = args.loss_ce_coef
            aux_weight_dict[f"loss_bbox_{i}"] = args.loss_bbox_coef
            aux_weight_dict[f"loss_giou_{i}"] = args.loss_giou_coef
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        num_classes=args.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    return criterion
