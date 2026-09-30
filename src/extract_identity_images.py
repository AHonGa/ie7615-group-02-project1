"""
Extracts just the images belonging to one CelebA identity ID into their own folder,
ready to upload to the shared class Google Drive.

Usage:
    python src/extract_identity_images.py <identity_id>

Example:
    python src/extract_identity_images.py 4321

Output: copies that identity's images into extracted_identities/<identity_id>/
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
IMG_DIR = DATA_DIR / "img_align_celeba"
OUT_ROOT = REPO_ROOT / "extracted_identities"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/extract_identity_images.py <identity_id>")

    identity_id = int(sys.argv[1])
    identity_path = DATA_DIR / "identity_CelebA.txt"
    df = pd.read_csv(identity_path, sep=r"\s+", header=None, names=["image_id", "identity"])

    rows = df[df["identity"] == identity_id]
    if rows.empty:
        raise SystemExit(f"No images found for identity_id={identity_id}. Check the ID is correct.")

    out_dir = OUT_ROOT / str(identity_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_id in rows["image_id"]:
        src = IMG_DIR / image_id
        dst = out_dir / image_id
        if not src.exists():
            print(f"  [warn] missing source image: {src}")
            continue
        shutil.copy2(src, dst)
        copied += 1

    print(f"Copied {copied} images for identity_id={identity_id} into: {out_dir}")
    print("Next: upload this whole folder to the shared Google Drive, named with the identity ID.")


if __name__ == "__main__":
    main()
