"""
Dataset + transforms for the selected CelebA identity subset.

Builds a per-identity train/val/test split (stratified so every identity is
represented in all three splits), and applies consistent resizing/normalization
plus a light augmentation policy on the training split only.
"""
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SHARED_POOL_DIR = DATA_DIR / "shared_pool"
SPLITS_DIR = DATA_DIR / "splits"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_IMAGES_PER_IDENTITY = 23
MAX_IMAGES_PER_IDENTITY = 25

IMAGE_SIZE = 128  # keep small for CPU-friendly training (Explorer's older nodes)

# ImageNet stats — used for both models so the pretrained ResNet gets inputs in the
# distribution it was trained on, and the custom CNN gets the same preprocessing for
# a fair comparison.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def build_splits(identity_ids: List[int], val_frac=0.15, test_frac=0.15, seed=42) -> pd.DataFrame:
    """Split only verified images from data/shared_pool/<identity_id>/ folders."""
    if len(identity_ids) < 4 or len(identity_ids) > 6:
        raise ValueError("Milestone data must use 4-6 distinct shared-pool identities")
    if len(set(identity_ids)) != len(identity_ids):
        raise ValueError("Identity IDs must be distinct")

    id_to_label = {iid: i for i, iid in enumerate(sorted(identity_ids))}
    splits = []
    for iid in sorted(identity_ids):
        identity_dir = SHARED_POOL_DIR / str(iid)
        if not identity_dir.is_dir():
            raise FileNotFoundError(f"Shared-pool folder is missing for identity {iid}: {identity_dir}")
        image_paths = sorted(
            path.name for path in identity_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not MIN_IMAGES_PER_IDENTITY <= len(image_paths) <= MAX_IMAGES_PER_IDENTITY:
            raise ValueError(
                f"Identity {iid} has {len(image_paths)} shared-pool images; "
                f"the required range is {MIN_IMAGES_PER_IDENTITY}-{MAX_IMAGES_PER_IDENTITY}"
            )

        group = pd.DataFrame({"image_id": image_paths})
        group["identity"] = iid
        group["label"] = id_to_label[iid]
        g = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(g)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError(f"Identity {iid} has too few images ({n}) for a 3-way split")
        g.loc[: n_train - 1, "split"] = "train"
        g.loc[n_train : n_train + n_val - 1, "split"] = "val"
        g.loc[n_train + n_val :, "split"] = "test"
        splits.append(g)

    out = pd.concat(splits, ignore_index=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(SPLITS_DIR / "splits.csv", index=False)
    (SPLITS_DIR / "label_map.json").write_text(
        json.dumps({str(k): v for k, v in id_to_label.items()}, indent=2)
    )
    return out


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        # Lighter augmentation than a first pass: with only ~22 training images per
        # identity, heavy color jitter + rotation added more variance than the custom
        # CNN could learn through, contributing to it collapsing to one predicted class.
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.ToTensor(),
                transforms.Normalize(NORM_MEAN, NORM_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


class CelebASubset(Dataset):
    def __init__(self, split: str, splits_df: pd.DataFrame = None):
        if splits_df is None:
            splits_df = pd.read_csv(SPLITS_DIR / "splits.csv")
        self.df = splits_df[splits_df["split"] == split].reset_index(drop=True)
        self.transform = get_transforms(train=(split == "train"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = SHARED_POOL_DIR / str(row["identity"]) / row["image_id"]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, int(row["label"])

    @property
    def num_classes(self) -> int:
        return self.df["label"].nunique()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--identity-ids", type=int, nargs="+", required=True)
    args = ap.parse_args()
    df = build_splits(args.identity_ids)
    print(df.groupby(["identity", "split"]).size())
