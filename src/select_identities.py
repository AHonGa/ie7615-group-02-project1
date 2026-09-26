"""Verify a 4-6 identity selection against the completed shared image pool.

Shared-pool layout: data/shared_pool/<identity_id>/<image files>

Example:
    python src/select_identities.py --identity-ids 3 7 8335 1212 \\
        --diversity-review "Briefly describe the visible variation reviewed."
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_POOL_DIR = REPO_ROOT / "data" / "shared_pool"
LOGS_DIR = REPO_ROOT / "logs"
DOCS_DIR = REPO_ROOT / "docs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_IMAGES = 23
MAX_IMAGES = 25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-ids", type=int, nargs="+", required=True)
    parser.add_argument(
        "--diversity-review",
        required=True,
        help="Brief notes on the visual diversity reviewed in the shared pool",
    )
    args = parser.parse_args()

    if not 4 <= len(args.identity_ids) <= 6:
        raise SystemExit("Select 4-6 distinct identities from the completed shared pool.")
    if len(set(args.identity_ids)) != len(args.identity_ids):
        raise SystemExit("Identity IDs must be distinct.")
    if not SHARED_POOL_DIR.is_dir():
        raise SystemExit(
            f"Shared pool is missing: {SHARED_POOL_DIR}\n"
            "Place each identity's contributed images in data/shared_pool/<identity_id>/."
        )

    report_rows = []
    errors = []
    for identity_id in args.identity_ids:
        identity_dir = SHARED_POOL_DIR / str(identity_id)
        if not identity_dir.is_dir():
            errors.append(f"identity {identity_id}: no shared-pool folder")
            continue

        images = [
            path for path in identity_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        image_count = len(images)
        if not MIN_IMAGES <= image_count <= MAX_IMAGES:
            errors.append(
                f"identity {identity_id}: found {image_count} images; "
                f"required range is {MIN_IMAGES}-{MAX_IMAGES}"
            )
        report_rows.append({"identity_id": identity_id, "num_images": image_count})

    if errors:
        raise SystemExit("Shared-pool verification failed:\n- " + "\n- ".join(errors))

    result = {
        "source": "data/shared_pool",
        "counts_verified": True,
        "required_image_count": {"min": MIN_IMAGES, "max": MAX_IMAGES},
        "diversity_review": args.diversity_review,
        "selected": report_rows,
    }
    LOGS_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    (LOGS_DIR / "selected_identities.json").write_text(json.dumps(result, indent=2))

    lines = [
        "# Shared-pool identity selection",
        "",
        "Counts below were verified by counting image files in the completed shared pool.",
        "",
        "| Identity ID | Shared-pool images |",
        "|---|---:|",
    ]
    lines.extend(f"| {row['identity_id']} | {row['num_images']} |" for row in report_rows)
    lines.extend(["", "## Visual diversity review", "", args.diversity_review, ""])
    (DOCS_DIR / "identity_selection_report.md").write_text("\n".join(lines))

    print("Verified shared-pool identities:")
    for row in report_rows:
        print(f"  identity_id={row['identity_id']:>6}  shared_pool_images={row['num_images']:>2}")
    print("Counts are within the required range (23-25); selection report written.")


if __name__ == "__main__":
    main()