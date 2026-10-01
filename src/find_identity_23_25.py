"""
Individual task (before group Milestone 1 work): find CelebA identity IDs that have
between 23 and 25 images, so you can claim one unique ID on the class Excel sheet and
upload just that identity's images to the shared Google Drive.

Usage:
    python src/find_identity_23_25.py

Requires data/identity_CelebA.txt (already present from the earlier group setup).

Output: prints eligible identity IDs sorted by count, and writes
logs/eligible_identities_23_25.csv so you can browse the full list.
"""
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
LOGS_DIR = REPO_ROOT / "logs"

identity_path = DATA_DIR / "identity_CelebA.txt"
if not identity_path.exists():
    raise SystemExit(f"Missing {identity_path} — make sure data/identity_CelebA.txt exists.")

df = pd.read_csv(identity_path, sep=r"\s+", header=None, names=["image_id", "identity"])
counts = Counter(df["identity"])

eligible = sorted([(iid, c) for iid, c in counts.items() if 23 <= c <= 25], key=lambda x: x[0])

LOGS_DIR.mkdir(exist_ok=True)
out_df = pd.DataFrame(eligible, columns=["identity_id", "num_images"])
out_df.to_csv(LOGS_DIR / "eligible_identities_23_25.csv", index=False)

print(f"Found {len(eligible)} identities with 23-25 images (full list in "
      f"logs/eligible_identities_23_25.csv).\n")
print("First 30 candidates:")
for iid, c in eligible[:30]:
    print(f"  identity_id={iid:>6}  num_images={c}")

print(
    "\nNext steps:\n"
    "1. Open the class Excel sheet and check which IDs are already claimed.\n"
    "2. Pick an unclaimed ID from the list above (or from "
    "logs/eligible_identities_23_25.csv).\n"
    "3. Enter it on the Excel sheet to claim it.\n"
    "4. Run: python src/extract_identity_images.py <your_identity_id>\n"
    "   to copy just that identity's images into their own folder, ready to upload."
)
