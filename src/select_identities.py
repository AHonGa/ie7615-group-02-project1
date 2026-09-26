"""Selection helpers for the project’s celebrity identity pool.

The repo uses this module both as a validation step for the shared pool and as a
lightweight CLI for selecting a few candidate identities that satisfy the rubric's
count constraints. The test suite validates the shared-pool contract directly, so
this module exposes the same constants and CLI flags the earlier notebook workflow
expected.
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SHARED_POOL_DIR = DATA_DIR / "shared_pool"
LOGS_DIR = REPO_ROOT / "logs"
DOCS_DIR = REPO_ROOT / "docs"


def _resolve_identity_ids(args):
    if getattr(args, "identity_ids", None):
        return list(args.identity_ids)
    if getattr(args, "fixed_ids", None):
        return list(args.fixed_ids)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity-ids", type=int, nargs="+", default=None)
    ap.add_argument("--fixed-ids", type=int, nargs="+", default=None)
    ap.add_argument("--diversity-review", type=str, default="")
    args = ap.parse_args()

    identity_ids = _resolve_identity_ids(args)
    if identity_ids is None:
        raise SystemExit("Usage: python src/select_identities.py --identity-ids 3 7 1212 8335")
    if len(identity_ids) < 4:
        raise SystemExit("At least four identities must be selected.")

    LOGS_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    selected = []
    for identity_id in identity_ids:
        identity_dir = SHARED_POOL_DIR / str(identity_id)
        if not identity_dir.is_dir():
            raise SystemExit(f"Missing shared-pool directory for identity_id={identity_id}: {identity_dir}")

        image_files = [p for p in identity_dir.iterdir() if p.is_file()]
        num_images = len(image_files)
        if not 23 <= num_images <= 25:
            raise SystemExit(
                f"identity_id={identity_id} has {num_images} images in the shared pool; "
                "expected 23-25 images."
            )
        selected.append({"identity_id": int(identity_id), "num_images": int(num_images)})

    result = {
        "counts_verified": True,
        "n_identities": len(identity_ids),
        "selected": selected,
        "diversity_review": args.diversity_review,
    }
    (LOGS_DIR / "selected_identities.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()


    print("\nSelected identities:")
    for r in report_rows:
        print(f"  id={r['identity_id']:>6}  n_images={r['num_images']:>3}  attrs={r['dominant_attributes']}")
    print(f"\nWrote logs/selected_identities.json and docs/identity_selection_report.md")


if __name__ == "__main__":
    main()
