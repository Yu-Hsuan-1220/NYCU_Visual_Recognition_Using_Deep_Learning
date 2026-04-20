"""Analyze bounding box size distribution for train and validation sets."""

import json
import sys
import numpy as np


def analyze(ann_file, label):
    with open(ann_file) as f:
        data = json.load(f)
    widths = [a["bbox"][2] for a in data["annotations"]]
    heights = [a["bbox"][3] for a in data["annotations"]]
    areas = [w * h for w, h in zip(widths, heights)]

    print(f"\n=== {label} ({len(widths)} boxes) ===")
    for name, vals in [("Width", widths), ("Height", heights), ("Area", areas)]:
        v = np.array(vals)
        print(f"  {name:6s}: min={v.min():.1f}  max={v.max():.1f}  "
              f"mean={v.mean():.1f}  median={np.median(v):.1f}  std={v.std():.1f}")
        pcts = np.percentile(v, [10, 25, 50, 75, 90])
        print(f"          percentiles [10,25,50,75,90]: "
              f"{pcts[0]:.1f}, {pcts[1]:.1f}, {pcts[2]:.1f}, {pcts[3]:.1f}, {pcts[4]:.1f}")

    # Size buckets
    areas = np.array(areas)
    small = (areas < 32**2).sum()
    medium = ((areas >= 32**2) & (areas < 96**2)).sum()
    large = (areas >= 96**2).sum()
    total = len(areas)
    print(f"  COCO size split:  small(<32²)={small} ({100*small/total:.1f}%)  "
          f"medium(32²-96²)={medium} ({100*medium/total:.1f}%)  "
          f"large(>96²)={large} ({100*large/total:.1f}%)")

    return widths, heights, areas


def main():
    train_ann = sys.argv[1] if len(sys.argv) > 1 else "../dataset/train.json"
    val_ann = sys.argv[2] if len(sys.argv) > 2 else "../dataset/valid.json"

    tw, th, ta = analyze(train_ann, "Train")
    vw, vh, va = analyze(val_ann, "Valid")

    # Compare
    print("\n=== Train vs Valid comparison ===")
    for name, t, v in [("Width", tw, vw), ("Height", th, vh), ("Area", ta, va)]:
        t, v = np.array(t), np.array(v)
        print(f"  {name:6s}: train_mean={t.mean():.1f} vs valid_mean={v.mean():.1f}  "
              f"train_median={np.median(t):.1f} vs valid_median={np.median(v):.1f}")


if __name__ == "__main__":
    main()
