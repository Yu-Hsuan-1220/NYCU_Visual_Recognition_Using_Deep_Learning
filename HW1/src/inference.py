import os
import csv
import argparse
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.transforms import functional as TF

from dataset import get_val_transforms
from model import get_model


class TestDataset(Dataset):
    """Dataset for unlabeled test images."""

    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_names = sorted([
            name for name in os.listdir(test_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.test_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, image_name


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for HW1 image classification")
    parser.add_argument("--test_dir", type=str, default="../dataset/data/test", help="Path to test image directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output_csv", type=str, default="prediction.csv", help="Output CSV path")

    parser.add_argument("--backbone", type=str, default="resnet101", choices=["resnet50", "resnet101", "resnet152"])
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use_deeper_fc", action="store_true", help="Set this if checkpoint was trained with --use_deeper_fc")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--use_tta", action="store_true", help="Use 7-view TTA (base + 6 transforms)")

    return parser.parse_args()


def get_tta_transforms(img_size):
    """Return 7 transforms: base + 6 additional TTA transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    resize_size = int(img_size * 1.14)

    base = get_val_transforms(img_size=img_size)

    tta_1 = v2.Compose([
        v2.Resize((resize_size, resize_size)),
        v2.Lambda(lambda img: TF.crop(img, top=0, left=0, height=img_size, width=img_size)),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    tta_2 = v2.Compose([
        v2.Resize((resize_size, resize_size)),
        v2.Lambda(
            lambda img: TF.crop(
                img,
                top=0,
                left=resize_size - img_size,
                height=img_size,
                width=img_size,
            )
        ),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    tta_3 = v2.Compose([
        v2.Resize((resize_size, resize_size)),
        v2.Lambda(
            lambda img: TF.crop(
                img,
                top=resize_size - img_size,
                left=0,
                height=img_size,
                width=img_size,
            )
        ),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    tta_4 = v2.Compose([
        v2.Resize((resize_size, resize_size)),
        v2.Lambda(
            lambda img: TF.crop(
                img,
                top=resize_size - img_size,
                left=resize_size - img_size,
                height=img_size,
                width=img_size,
            )
        ),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    tta_5 = v2.Compose([
        v2.Resize(resize_size),
        v2.CenterCrop(img_size),
        v2.Lambda(lambda img: TF.hflip(img)),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    tta_6 = v2.Compose([
        v2.Resize(resize_size),
        v2.CenterCrop(img_size),
        v2.Lambda(lambda img: TF.rotate(img, angle=10.0)),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    return [base, tta_1, tta_2, tta_3, tta_4, tta_5, tta_6]


def load_model(checkpoint_path, backbone, num_classes, dropout, use_deeper_fc, device):
    model = get_model(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=False,
        dropout=dropout,
        use_deeper_fc=use_deeper_fc,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if len(state_dict) > 0 and next(iter(state_dict)).startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.use_tta:
        transforms_list = get_tta_transforms(img_size=args.img_size)
    else:
        transforms_list = [get_val_transforms(img_size=args.img_size)]

    test_dataset = TestDataset(args.test_dir, transform=None)
    image_names = test_dataset.image_names

    print(f"Found {len(test_dataset)} test images")
    model = load_model(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        num_classes=args.num_classes,
        dropout=args.dropout,
        use_deeper_fc=args.use_deeper_fc,
        device=device,
    )

    avg_probs = None
    for tta_idx, transform in enumerate(transforms_list):
        print(f"Running view {tta_idx + 1}/{len(transforms_list)}")
        test_dataset_tta = TestDataset(args.test_dir, transform=transform)
        test_loader = DataLoader(
            test_dataset_tta,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        all_probs = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1).cpu()
                all_probs.append(probs)

        probs_this_view = torch.cat(all_probs, dim=0)
        if avg_probs is None:
            avg_probs = probs_this_view
        else:
            avg_probs += probs_this_view

    avg_probs /= len(transforms_list)
    pred_labels = torch.argmax(avg_probs, dim=1).tolist()

    predictions = []
    for image_name, pred_label in zip(image_names, pred_labels):
        image_stem = os.path.splitext(image_name)[0]
        predictions.append((image_stem, int(pred_label)))

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        writer.writerows(predictions)

    print(f"Saved {len(predictions)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
