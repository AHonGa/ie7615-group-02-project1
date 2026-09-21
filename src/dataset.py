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
IMG_DIR = DATA_DIR / "img_align_celeba"
SPLITS_DIR = DATA_DIR / "splits"

IMAGE_SIZE = 128  # keep small for CPU-friendly training (Explorer's older nodes)

# ImageNet stats — used for both models so the pretrained ResNet gets inputs in the
# distribution it was trained on, and the custom CNN gets the same preprocessing for
# a fair comparison.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def build_splits(identity_ids: List[int], val_frac=0.15, test_frac=0.15, seed=42) -> pd.DataFrame:
    """Stratified per-identity split. Returns a DataFrame with columns
    [image_id, identity, label, split] where `label` is a 0..K-1 remapped class id."""
    identity_path = DATA_DIR / "identity_CelebA.txt"
    df = pd.read_csv(identity_path, sep=r"\s+", header=None, names=["image_id", "identity"])
    df = df[df["identity"].isin(identity_ids)].reset_index(drop=True)

    id_to_label = {iid: i for i, iid in enumerate(sorted(identity_ids))}
    df["label"] = df["identity"].map(id_to_label)

    rng = pd.Series(range(len(df)))
    splits = []
    for iid, group in df.groupby("identity"):
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
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.RandomRotation(degrees=8),
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
        img_path = IMG_DIR / row["image_id"]
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
