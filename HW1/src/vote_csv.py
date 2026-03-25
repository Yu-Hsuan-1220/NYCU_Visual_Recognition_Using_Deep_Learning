import os
import csv
import argparse
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Majority vote for an odd number of prediction CSV files"
    )
    parser.add_argument(
        "--input_csvs",
        nargs="+",
        required=True,
        help="Input prediction CSV files (odd count required)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="vote_prediction.csv",
        help="Output voted CSV path",
    )
    return parser.parse_args()


def read_prediction_csv(csv_path):
    preds = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if ("image_name" not in reader.fieldnames or
                "pred_label" not in reader.fieldnames):
            raise ValueError(
                f"Invalid header in {csv_path}, expect image_name, pred_label"
            )
        for row in reader:
            image_name = row["image_name"].strip()
            pred_label = int(row["pred_label"])
            preds[image_name] = pred_label
    return preds


def majority_vote(labels):
    counter = Counter(labels)
    max_count = max(counter.values())
    candidates = [k for k, v in counter.items() if v == max_count]
    return min(candidates)


def main():
    args = parse_args()

    if len(args.input_csvs) % 2 == 0:
        raise ValueError("Please provide an odd number of input CSV files.")

    all_preds = []
    for csv_path in args.input_csvs:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Input file not found: {csv_path}")
        all_preds.append(read_prediction_csv(csv_path))

    reference_names = sorted(all_preds[0].keys())
    for i, preds in enumerate(all_preds[1:], start=2):
        if set(preds.keys()) != set(reference_names):
            raise ValueError(
                f"Image name set mismatch between file #1 and file #{i}."
            )

    voted_rows = []
    for image_name in reference_names:
        labels = [preds[image_name] for preds in all_preds]
        voted_label = majority_vote(labels)
        voted_rows.append((image_name, voted_label))

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        writer.writerows(voted_rows)

    print(f"Saved voted predictions to {args.output_csv}")
    print(f"Used {len(args.input_csvs)} input files")


if __name__ == "__main__":
    main()
