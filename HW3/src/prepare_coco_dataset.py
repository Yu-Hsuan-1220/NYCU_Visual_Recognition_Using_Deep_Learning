"""
Convert raw dataset to COCO format with 5-fold cross-validation splits.

Usage:
    python prepare_coco_dataset.py --data_root ../dataset --output_dir ../dataset/annotations
"""
import argparse
import json
import os

import cv2
import numpy as np
import tifffile
from pycocotools import mask as mask_utils
from sklearn.model_selection import StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw data to COCO format")
    parser.add_argument("--data_root", type=str, default="../dataset",
                        help="Root directory of the dataset")
    parser.add_argument("--output_dir", type=str, default="../dataset/annotations",
                        help="Output directory for COCO annotation JSONs")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for fold splitting")
    parser.add_argument("--min_area", type=int, default=10,
                        help="Minimum mask area to keep an instance")
    return parser.parse_args()


CATEGORIES = [
    {"id": 1, "name": "class1", "supercategory": "cell"},
    {"id": 2, "name": "class2", "supercategory": "cell"},
    {"id": 3, "name": "class3", "supercategory": "cell"},
    {"id": 4, "name": "class4", "supercategory": "cell"},
]


def binary_mask_to_rle(binary_mask):
    """Encode a binary mask as COCO RLE."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def binary_mask_to_polygon(binary_mask):
    """Convert a binary mask to polygon segmentation format."""
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        # Need at least 3 points for a valid polygon
        if contour.shape[0] >= 3:
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:
                polygons.append(polygon)
    return polygons


def process_dataset(data_root, min_area=10):
    """Process all training data and return COCO-format structures."""
    train_dir = os.path.join(data_root, "train")
    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    # Track number of classes per image for stratification
    class_counts = []
    image_dirs = sorted(os.listdir(train_dir))
    valid_dirs = []

    for dirname in image_dirs:
        img_dir = os.path.join(train_dir, dirname)
        if not os.path.isdir(img_dir):
            continue

        img_path = os.path.join(img_dir, "image.tif")
        if not os.path.exists(img_path):
            continue

        # Read image to get dimensions
        img = tifffile.imread(img_path)
        h, w = img.shape[:2]

        images.append({
            "id": img_id,
            "file_name": os.path.join("train", dirname, "image.tif"),
            "height": h,
            "width": w,
        })

        # Find class mask files
        class_files = sorted(
            [f for f in os.listdir(img_dir)
             if f.startswith("class") and f.endswith(".tif")]
        )
        class_counts.append(len(class_files))
        valid_dirs.append(dirname)

        for class_file in class_files:
            # Extract class id: class1.tif -> 1
            class_id = int(class_file.replace("class", "").replace(".tif", ""))

            mask_path = os.path.join(img_dir, class_file)
            mask = tifffile.imread(mask_path)

            # Extract unique instance IDs (skip background=0)
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids > 0]

            for inst_id in instance_ids:
                binary_mask = (mask == inst_id).astype(np.uint8)
                area = int(binary_mask.sum())

                if area < min_area:
                    continue

                # Bounding box
                ys, xs = np.where(binary_mask)
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
                bbox_w = x2 - x1 + 1
                bbox_h = y2 - y1 + 1

                if bbox_w <= 0 or bbox_h <= 0:
                    continue

                # Use polygon format for compatibility
                polygons = binary_mask_to_polygon(binary_mask)
                if len(polygons) == 0:
                    # Fallback to RLE if polygon extraction fails
                    seg = binary_mask_to_rle(binary_mask)
                else:
                    seg = polygons

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "segmentation": seg,
                    "area": area,
                    "bbox": [x1, y1, bbox_w, bbox_h],
                    "iscrowd": 0,
                })
                ann_id += 1

        img_id += 1

    return images, annotations, class_counts, valid_dirs


def create_coco_json(images, annotations, image_ids_set):
    """Create a COCO JSON dict for a subset of images."""
    sub_images = [img for img in images if img["id"] in image_ids_set]
    sub_annotations = [ann for ann in annotations if ann["image_id"] in image_ids_set]
    return {
        "images": sub_images,
        "annotations": sub_annotations,
        "categories": CATEGORIES,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Processing dataset...")
    images, annotations, class_counts, valid_dirs = process_dataset(
        args.data_root, args.min_area
    )
    print(f"  Total images: {len(images)}")
    print(f"  Total annotations: {len(annotations)}")

    # Save full training set annotation
    full_coco = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    full_path = os.path.join(args.output_dir, "train_all.json")
    with open(full_path, "w") as f:
        json.dump(full_coco, f)
    print(f"  Saved full annotations to {full_path}")

    # Create 5-fold stratified splits
    class_counts = np.array(class_counts)
    image_ids = np.array([img["id"] for img in images])

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    for fold_idx, (train_indices, val_indices) in enumerate(
        skf.split(image_ids, class_counts)
    ):
        train_ids = set(image_ids[train_indices].tolist())
        val_ids = set(image_ids[val_indices].tolist())

        train_json = create_coco_json(images, annotations, train_ids)
        val_json = create_coco_json(images, annotations, val_ids)

        train_path = os.path.join(args.output_dir, f"fold{fold_idx}_train.json")
        val_path = os.path.join(args.output_dir, f"fold{fold_idx}_val.json")

        with open(train_path, "w") as f:
            json.dump(train_json, f)
        with open(val_path, "w") as f:
            json.dump(val_json, f)

        print(
            f"  Fold {fold_idx}: {len(train_json['images'])} train, "
            f"{len(val_json['images'])} val images"
        )

    print("Done!")


if __name__ == "__main__":
    main()
