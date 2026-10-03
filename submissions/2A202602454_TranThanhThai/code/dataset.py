"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
Giao diện chuẩn:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_path = Path(labels_dir)
    train_file = labels_path / f"train_subset{fold}.csv"
    val_file = labels_path / f"val_subset{fold}.csv"
    test_file = labels_path / f"test_subset{fold}.csv"

    if not train_file.exists() or not val_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Không tìm thấy đủ file split fold {fold} trong {labels_dir}")

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        required_cols = {"Filename", "Label"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"{name} split thiếu các cột bắt buộc: {required_cols - set(df.columns)}")
        if "Species" not in df.columns:
            df["Species"] = df["Label"].map(lambda l: CLASS_NAMES[l] if 0 <= l < len(CLASS_NAMES) else str(l))

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path | None = None) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
    2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
    3. hợp ba tập phải bằng đúng 17.509 ảnh
    4. mọi Filename đều tồn tại trong `images_dir` (nếu images_dir tồn tại)
    """
    train_files = set(train_df["Filename"].astype(str))
    val_files = set(val_df["Filename"].astype(str))
    test_files = set(test_df["Filename"].astype(str))

    # 1. Giao giữa các cặp tập
    overlap_train_val = train_files.intersection(val_files)
    overlap_train_test = train_files.intersection(test_files)
    overlap_val_test = val_files.intersection(test_files)

    if overlap_train_val:
        raise AssertionError(f"Rò rỉ dữ liệu: {len(overlap_train_val)} ảnh trùng giữa train và val")
    if overlap_train_test:
        raise AssertionError(f"Rò rỉ dữ liệu: {len(overlap_train_test)} ảnh trùng giữa train và test")
    if overlap_val_test:
        raise AssertionError(f"Rò rỉ dữ liệu: {len(overlap_val_test)} ảnh trùng giữa val và test")

    # 2. Hợp ba tập
    union_files = train_files | val_files | test_files
    total_count = len(union_files)
    if total_count != 17509:
        raise AssertionError(f"Tổng số ảnh của 3 tập là {total_count} (yêu cầu đúng 17.509 ảnh)")

    # 3. Thống kê số lượng
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)

    def class_distribution(df: pd.DataFrame) -> dict[int, int]:
        counts = df["Label"].value_counts().to_dict()
        return {c: int(counts.get(c, 0)) for c in range(NUM_CLASSES)}

    per_class_train = class_distribution(train_df)
    per_class_val = class_distribution(val_df)
    per_class_test = class_distribution(test_df)

    # 4. Kiểm tra file tồn tại trên đĩa (nếu thư mục ảnh đã sẵn sàng)
    missing_files = []
    if images_dir is not None:
        img_path = Path(images_dir)
        if img_path.exists():
            for f in union_files:
                if not (img_path / f).exists():
                    missing_files.append(f)
            if missing_files:
                raise FileNotFoundError(f"Có {len(missing_files)} ảnh trong CSV không tìm thấy trong {images_dir}")

    stats = {
        "n": {"train": n_train, "val": n_val, "test": n_test, "total": total_count},
        "ratio": {"train": n_train / total_count, "val": n_val / total_count, "test": n_test / total_count},
        "per_class": {
            "train": per_class_train,
            "val": per_class_val,
            "test": per_class_test,
        },
        "overlap": {
            "train_val": len(overlap_train_val),
            "train_test": len(overlap_train_test),
            "val_test": len(overlap_val_test),
        },
        "missing_files": len(missing_files),
    }
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic") -> transforms.Compose:
    """Tạo transform. `aug` chọn mức augmentation; định nghĩa các giá trị:
    'basic', 'color', 'trivial', 'randaug'.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: Resize(256) -> CenterCrop(img_size) + ToTensor + Normalize.
    """
    if train:
        t_list: list = []
        if aug == "basic":
            t_list = [
                transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        elif aug == "color":
            t_list = [
                transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
        elif aug == "trivial":
            t_list = [
                transforms.TrivialAugmentWide(),
                transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        elif aug == "randaug":
            t_list = [
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        elif aug == "flip_vh":
            # Thử nghiệm lật ngang + lật dọc cho ảnh chụp từ robot
            t_list = [
                transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        else:
            t_list = [
                transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ]

        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transforms.Compose(t_list)

    # Đánh giá (Val / Test): không dùng augmentation ngẫu nhiên
    if img_size >= 256:
        val_t = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    else:
        val_t = [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    return transforms.Compose(val_t)


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (image_tensor, label: int, filename: str).
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.copy().reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].to_numpy(dtype=str)
        self.labels = self.df["Label"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str]:
        fname = self.filenames[i]
        label = int(self.labels[i])
        img_path = self.images_dir / fname

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            raise IOError(f"Lỗi khi mở ảnh {img_path}: {e}")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, fname


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader theo quy chuẩn thực nghiệm.

    - train=True: shuffle hoặc balanced sampler, drop_last=True
    - train=False: shuffle=False, drop_last=False, giữ nguyên thứ tự file
    """
    dataset = DeepWeedsDataset(df, images_dir, transform=transform)

    if train:
        if sampler == "balanced":
            labels = df["Label"].to_numpy(dtype=int)
            counts = np.bincount(labels, minlength=NUM_CLASSES)
            # Trọng số mẫu tỉ lệ nghịch với tần suất lớp
            class_weights = 1.0 / np.maximum(counts, 1)
            sample_weights = class_weights[labels]
            sample_weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
            data_sampler = WeightedRandomSampler(
                weights=sample_weights_tensor,
                num_samples=len(sample_weights_tensor),
                replacement=True,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=data_sampler,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=True,
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Đánh giá: không shuffle, không drop_last
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
