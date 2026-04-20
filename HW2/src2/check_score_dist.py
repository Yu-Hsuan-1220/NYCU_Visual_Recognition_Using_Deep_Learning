import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "pred.json"
data = json.load(open(path))

bins = [0] * 10  # [0.0-0.1), [0.1-0.2), ..., [0.9-1.0]
for item in data:
    s = item["score"]
    idx = min(int(s * 10), 9)
    bins[idx] += 1

total = len(data)
print(f"Total predictions: {total}\n")
print(f"{'Range':<12} {'Count':>8} {'Percent':>8}")
print("-" * 30)
for i, count in enumerate(bins):
    lo, hi = i * 0.1, (i + 1) * 0.1
    pct = count / total * 100 if total else 0
    print(f"[{lo:.1f}, {hi:.1f})  {count:>8}  {pct:>7.2f}%")
