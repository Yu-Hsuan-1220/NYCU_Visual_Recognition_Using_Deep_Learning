import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class ImageClassificationDataset(Dataset):
    """
    Custom Dataset for image classification with 100 classes.
    """

    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Args:
            root_dir (str): Directory with images organized by class folders.
            transform: Optional transform to be applied on a sample.
            is_train (bool): Whether this is training set.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        self.samples = []
        self.class_to_idx = {}

        class_folders = sorted(
            [d for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d))],
            key=lambda x: int(x)
        )

        for class_folder in class_folders:
            class_idx = int(class_folder)
            self.class_to_idx[class_folder] = class_idx
            class_path = os.path.join(root_dir, class_folder)

            for img_name in os.listdir(class_path):
                exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
                if img_name.lower().endswith(exts):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append((img_path, class_idx))

        self.num_classes = len(class_folders)
        msg = f"Loaded {len(self.samples)} images from {self.num_classes} cls"
        print(msg)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transforms(img_size=224, use_RandAugment=False):
    """Get training transforms with augmentation."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if use_RandAugment:
        return v2.Compose([
            v2.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandAugment(num_ops=2, magnitude=9),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.RandomErasing(p=0.2)
        ])
    return v2.Compose([
        v2.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])


def get_val_transforms(img_size=224):
    """Get validation transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return v2.Compose([
        v2.Resize(int(img_size * 1.14)),
        v2.CenterCrop(img_size),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])


def get_dataloaders(
    train_dir, val_dir, batch_size=32, num_workers=4,
    img_size=224, use_RandAugment=False
):
    """
    Create train and validation dataloaders.
    """
    from torch.utils.data import DataLoader

    train_transform = get_train_transforms(img_size, use_RandAugment)
    val_transform = get_val_transforms(img_size)

    train_dataset = ImageClassificationDataset(
        root_dir=train_dir,
        transform=train_transform,
        is_train=True
    )

    val_dataset = ImageClassificationDataset(
        root_dir=val_dir,
        transform=val_transform,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
