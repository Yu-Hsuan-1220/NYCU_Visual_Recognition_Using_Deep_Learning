"""HW4 dataset loaders.

Train layout:
    data_root/train/degraded/{rain,snow}-N.png
    data_root/train/clean/{rain,snow}_clean-N.png

Test layout:
    data_root/test/degraded/{i}.png   (i in 0..99, all 256x256 RGB)
"""
from __future__ import annotations

import os
import random
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DE_TYPES = {"rain": 0, "snow": 1}


def _to_tensor(arr):
    # HWC uint8 -> CHW float in [0, 1]
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def _pair_paths(data_root):
    degraded_dir = os.path.join(data_root, "train", "degraded")
    clean_dir = os.path.join(data_root, "train", "clean")
    pairs = []
    for p in sorted(glob(os.path.join(degraded_dir, "*.png"))):
        name = os.path.basename(p)
        if name.startswith("rain-"):
            de = "rain"
            clean = os.path.join(clean_dir, name.replace("rain-", "rain_clean-"))
        elif name.startswith("snow-"):
            de = "snow"
            clean = os.path.join(clean_dir, name.replace("snow-", "snow_clean-"))
        else:
            continue
        if os.path.isfile(clean):
            pairs.append((p, clean, DE_TYPES[de]))
    return pairs


def _split_pairs(pairs, val_ratio, seed):
    rng = random.Random(seed)
    by_type = {0: [], 1: []}
    for x in pairs:
        by_type[x[2]].append(x)
    train, val = [], []
    for de_id, lst in by_type.items():
        lst = sorted(lst)
        rng.shuffle(lst)
        n_val = max(1, int(round(len(lst) * val_ratio))) if val_ratio > 0 else 0
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
    return train, val


class HW4TrainDataset(Dataset):
    def __init__(
        self,
        data_root,
        patch_size=128,
        is_train=True,
        val_ratio=0.05,
        seed=42,
        aug_flip=True,
        aug_rot90=True,
        aug_rgb_shuffle=False,
        aug_mixup_p=0.0,
    ):
        super().__init__()
        all_pairs = _pair_paths(data_root)
        if len(all_pairs) == 0:
            raise RuntimeError(f"No paired images found under {data_root}/train")
        train_pairs, val_pairs = _split_pairs(all_pairs, val_ratio, seed)
        self.pairs = train_pairs if is_train else val_pairs
        self.patch_size = patch_size
        self.is_train = is_train
        self.aug_flip = aug_flip
        self.aug_rot90 = aug_rot90
        self.aug_rgb_shuffle = aug_rgb_shuffle
        self.aug_mixup_p = aug_mixup_p

    def __len__(self):
        return len(self.pairs)

    def _load(self, idx):
        deg_path, clean_path, de_id = self.pairs[idx]
        deg = np.array(Image.open(deg_path).convert("RGB"))
        clean = np.array(Image.open(clean_path).convert("RGB"))
        return deg, clean, de_id

    def _crop(self, deg, clean):
        H, W = deg.shape[:2]
        ps = self.patch_size
        if ps <= 0 or (H == ps and W == ps):
            return deg, clean
        if H < ps or W < ps:
            pad_h = max(0, ps - H)
            pad_w = max(0, ps - W)
            deg = np.pad(deg, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            clean = np.pad(clean, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            H, W = deg.shape[:2]
        if self.is_train:
            y = random.randint(0, H - ps)
            x = random.randint(0, W - ps)
        else:
            y = (H - ps) // 2
            x = (W - ps) // 2
        return deg[y:y + ps, x:x + ps], clean[y:y + ps, x:x + ps]

    def _aug(self, deg, clean):
        if not self.is_train:
            return deg, clean
        if self.aug_flip:
            if random.random() < 0.5:
                deg, clean = deg[:, ::-1], clean[:, ::-1]
            if random.random() < 0.5:
                deg, clean = deg[::-1, :], clean[::-1, :]
        if self.aug_rot90:
            k = random.randint(0, 3)
            if k:
                deg = np.rot90(deg, k=k, axes=(0, 1))
                clean = np.rot90(clean, k=k, axes=(0, 1))
        if self.aug_rgb_shuffle and random.random() < 0.5:
            perm = np.random.permutation(3)
            deg = deg[..., perm]
            clean = clean[..., perm]
        return deg, clean

    def __getitem__(self, idx):
        deg, clean, de_id = self._load(idx)
        deg, clean = self._crop(deg, clean)
        deg, clean = self._aug(deg, clean)
        deg_t = _to_tensor(deg)
        clean_t = _to_tensor(clean)

        if self.is_train and self.aug_mixup_p > 0 and random.random() < self.aug_mixup_p:
            # mixup with another sample of the SAME degradation type
            same = [i for i, (_, _, d) in enumerate(self.pairs) if d == de_id and i != idx]
            j = random.choice(same)
            deg2, clean2, _ = self._load(j)
            deg2, clean2 = self._crop(deg2, clean2)
            deg2, clean2 = self._aug(deg2, clean2)
            deg2_t = _to_tensor(deg2)
            clean2_t = _to_tensor(clean2)
            lam = float(np.random.beta(0.2, 0.2))
            deg_t = lam * deg_t + (1 - lam) * deg2_t
            clean_t = lam * clean_t + (1 - lam) * clean2_t

        return deg_t, clean_t, de_id


class HW4TestDataset(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.paths = sorted(
            glob(os.path.join(data_root, "test", "degraded", "*.png"))
        )
        if not self.paths:
            raise RuntimeError(f"No test images under {data_root}/test/degraded")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        name = os.path.basename(p)
        img = np.array(Image.open(p).convert("RGB"))
        return name, _to_tensor(img)
