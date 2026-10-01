"""
Milestone 2 — extract face crops for a list of CelebA identity IDs drawn from the
class-wide pool (not just your own team's 4 identities), so you can build the
synthetic multi-celebrity grid images required for detection.

This reuses the same logic as extract_identity_images.py but accepts many IDs at
once and writes each into its own subfolder, ready for the grid-composition step.

Usage:
    python src/extract_class_pool_images.py <id1> <id2> <id3> ...

Example (the 4 team IDs + 5 more from the class pool spreadsheet):
    python src/extract_class_pool_images.py 3 7 1212 8335 2619 797 10002 5695 4422

Output: copies each identity's images into extracted_identities/<identity_id>/
        (same output layout extract_identity_images.py already uses, so anything
        already extracted for your own 4 IDs is left alone / simply confirmed).
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
IMG_DIR = DATA_DIR / "img_align_celeba"
OUT_ROOT = REPO_ROOT / "extracted_identities"


def extract_one(df: pd.DataFrame, identity_id: int) -> int:
    rows = df[df["identity"] == identity_id]
    if rows.empty:
        print(f"  [warn] no images found for identity_id={identity_id} — check the ID is correct")
        return 0

    out_dir = OUT_ROOT / str(identity_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_id in rows["image_id"]:
        src = IMG_DIR / image_id
        dst = out_dir / image_id
        if dst.exists():
            copied += 1  # already extracted previously, count it as available
            continue
        if not src.exists():
            print(f"    [warn] missing source image: {src}")
            continue
        shutil.copy2(src, dst)
        copied += 1

    return copied


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python src/extract_class_pool_images.py <id1> <id2> ...\n"
            "Pass every identity ID you want available for building the "
            "3x3 multi-celebrity grids (your team's 4 plus enough others "
            "from the class pool spreadsheet to reach 9+ unique identities)."
        )

    identity_ids = [int(x) for x in sys.argv[1:]]
    identity_path = DATA_DIR / "identity_CelebA.txt"
    if not identity_path.exists():
        raise SystemExit(f"Missing {identity_path} — make sure data/identity_CelebA.txt exists.")

    df = pd.read_csv(identity_path, sep=r"\s+", header=None, names=["image_id", "identity"])

    print(f"Extracting {len(identity_ids)} identities: {identity_ids}\n")
    summary = []
    for iid in identity_ids:
        n = extract_one(df, iid)
        summary.append((iid, n))
        print(f"  id={iid:>6}  images_available={n}")

    missing = [iid for iid, n in summary if n == 0]
    if missing:
        print(f"\n[!] These IDs have NO images available: {missing} — double-check them "
              "against the class pool spreadsheet before building grids.")
    else:
        print(f"\nAll {len(identity_ids)} identities are ready in extracted_identities/")
        print("Next: run src/build_synthetic_detection_dataset.py to compose the 3x3 grids.")


if __name__ == "__main__":
    main()
