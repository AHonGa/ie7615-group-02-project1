"""Count CelebA images per identity before claiming an identity for the shared pool.

Counts are verified against files present in data/img_align_celeba, not just metadata.
The group's final selection is verified separately from data/shared_pool.
"""
import csv
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
LOGS_DIR = REPO_ROOT / "logs"

identity_path = DATA_DIR / "identity_CelebA.txt"
if not identity_path.exists():
    raise SystemExit(f"Missing {identity_path} — make sure data/identity_CelebA.txt exists.")
image_dir = DATA_DIR / "img_align_celeba"
if not image_dir.is_dir():
    raise SystemExit(f"Missing image directory {image_dir} — extract img_align_celeba.zip first.")

identity_images = []
with identity_path.open(encoding="utf-8") as identity_file:
    for line in identity_file:
        image_id, identity = line.split()
        identity_images.append((image_id, int(identity)))

metadata_counts = Counter(identity for _, identity in identity_images)
candidate_ids = {identity for identity, count in metadata_counts.items() if 23 <= count <= 25}
available_counts = Counter(
    identity
    for image_id, identity in identity_images
    if identity in candidate_ids and (image_dir / image_id).is_file()
)
eligible = sorted(
    (identity, count)
    for identity, count in available_counts.items()
    if 23 <= count <= 25 and count == metadata_counts[identity]
)

LOGS_DIR.mkdir(exist_ok=True)
with (LOGS_DIR / "eligible_identities_23_25.csv").open("w", newline="", encoding="utf-8") as output_file:
    writer = csv.writer(output_file)
    writer.writerow(["identity_id", "num_images"])
    writer.writerows(eligible)

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
