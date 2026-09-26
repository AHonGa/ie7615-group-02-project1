"""
Step 1 — Select 4-6 distinct CelebA identities for the classification baseline.

Why data-driven selection: CelebA identity IDs are anonymized integers with no
public name mapping and highly uneven image counts per identity (most identities
have only ~20-30 images; a few outliers have 40+). Rather than hand-picking IDs
blind, this script reads the real metadata (identity_CelebA.txt, list_attr_celeba.txt)
and selects a subset that satisfies the rubric's stated criteria:
  - sufficient, roughly balanced image counts per identity (>= MIN_IMAGES each)
  - visual diversity across attributes (gender, hair color, eyewear, facial hair)

Run this after downloading CelebA into data/ (see README "Data setup").

Usage:
    python src/select_identities.py --n-identities 5 --min-images 30

Output:
    - prints the chosen identity IDs + rationale
    - writes logs/selected_identities.json for the other notebooks to consume
    - writes docs/identity_selection_report.md (paste into docs/proposal.md)
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
LOGS_DIR = REPO_ROOT / "logs"
DOCS_DIR = REPO_ROOT / "docs"

# Attributes (from list_attr_celeba.txt's 40 columns) used to score visual diversity.
# Chosen because they are visually salient and roughly bimodal in CelebA.
DIVERSITY_ATTRS = [
    "Male",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Gray_Hair",
    "Eyeglasses",
    "Mustache",
    "Bald",
    "Wearing_Hat",
    "Pale_Skin",
]


def load_identity_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["image_id", "identity"])
    return df


def load_attrs(path: Path) -> pd.DataFrame:
    """Handles both the original CelebA format (line 1 = image count, line 2 =
    space-separated attribute names, then data) and the comma-separated format
    used by some Kaggle mirrors (header on line 1, no count line)."""
    with open(path) as f:
        first_line = f.readline().strip()

    if first_line.isdigit():
        # Original format: skip the count line, header is line 2, whitespace-separated
        df = pd.read_csv(path, sep=r"\s+", header=1, index_col=0)
    else:
        # Kaggle CSV format: header is line 1, comma-separated
        df = pd.read_csv(path, sep=",", header=0, index_col=0)

    df = df.replace(-1, 0)  # CelebA encodes attributes as {-1, 1}; use {0, 1}
    df.index.name = "image_id"
    return df


def greedy_diverse_subset(candidate_ids, per_id_attr_mean: pd.DataFrame, n: int, seed: int):
    """Greedily pick identities that maximize attribute-space spread (max-min distance)."""
    rng = random.Random(seed)
    candidate_ids = list(candidate_ids)
    rng.shuffle(candidate_ids)
    chosen = [candidate_ids.pop()]
    while len(chosen) < n and candidate_ids:
        best_id, best_score = None, -1.0
        for cid in candidate_ids:
            v = per_id_attr_mean.loc[cid].values
            min_dist = min(
                ((v - per_id_attr_mean.loc[c].values) ** 2).sum() ** 0.5 for c in chosen
            )
            if min_dist > best_score:
                best_score, best_id = min_dist, cid
        chosen.append(best_id)
        candidate_ids.remove(best_id)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-identities", type=int, default=5, help="4-6 per rubric")
    ap.add_argument("--min-images", type=int, default=30, help="min images per chosen identity")
    ap.add_argument("--max-images", type=int, default=45, help="cap, to keep classes balanced")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    identity_path = DATA_DIR / "identity_CelebA.txt"
    attr_path = DATA_DIR / "list_attr_celeba.txt"
    if not identity_path.exists() or not attr_path.exists():
        raise SystemExit(
            f"Missing CelebA metadata. Expected:\n  {identity_path}\n  {attr_path}\n"
            "See README.md 'Data setup' to download them."
        )

    identities = load_identity_map(identity_path)
    attrs = load_attrs(attr_path)

    counts = Counter(identities["identity"])
    eligible = [i for i, c in counts.items() if args.min_images <= c <= args.max_images]
    if len(eligible) < args.n_identities:
        # relax the cap if too few identities qualify
        eligible = [i for i, c in counts.items() if c >= args.min_images]
    print(f"Identities with >= {args.min_images} images: {len(eligible)}")

    merged = identities.merge(attrs, on="image_id")
    per_id_attr_mean = (
        merged[merged["identity"].isin(eligible)]
        .groupby("identity")[DIVERSITY_ATTRS]
        .mean()
    )

    chosen = greedy_diverse_subset(eligible, per_id_attr_mean, args.n_identities, args.seed)

    LOGS_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    report_rows = []
    for cid in chosen:
        n_imgs = counts[cid]
        attr_profile = per_id_attr_mean.loc[cid]
        dominant = attr_profile[attr_profile > 0.5].index.tolist()
        report_rows.append(
            {
                "identity_id": int(cid),
                "num_images": int(n_imgs),
                "dominant_attributes": dominant,
            }
        )

    result = {
        "n_identities": args.n_identities,
        "min_images": args.min_images,
        "max_images": args.max_images,
        "seed": args.seed,
        "selected": report_rows,
    }
    (LOGS_DIR / "selected_identities.json").write_text(json.dumps(result, indent=2))

    lines = [
        "# Celebrity subset selection\n",
        f"Selected **{args.n_identities}** identities from CelebA identities with "
        f"between {args.min_images} and {args.max_images} images each, chosen to "
        "maximize spread across gender, hair color, eyewear, and facial-hair attributes "
        "(greedy max-min diversity over the attribute vectors in `list_attr_celeba.txt`).\n",
        "| Identity ID | # Images | Dominant attributes |",
        "|---|---|---|",
    ]
    for r in report_rows:
        lines.append(f"| {r['identity_id']} | {r['num_images']} | {', '.join(r['dominant_attributes']) or '(none dominant)'} |")
    lines.append(
        "\n**Why these criteria:** sufficient per-identity images (>= "
        f"{args.min_images}) keeps train/val/test splits viable at a small scale; "
        "capping at the upper bound keeps classes roughly balanced so accuracy isn't "
        "dominated by the largest class; maximizing attribute-space distance ensures "
        "the model must learn genuinely discriminative facial features rather than "
        "relying on a single correlated cue (e.g., only hair color)."
    )
    (DOCS_DIR / "identity_selection_report.md").write_text("\n".join(lines) + "\n")

    print("\nSelected identities:")
    for r in report_rows:
        print(f"  id={r['identity_id']:>6}  n_images={r['num_images']:>3}  attrs={r['dominant_attributes']}")
    print(f"\nWrote logs/selected_identities.json and docs/identity_selection_report.md")


if __name__ == "__main__":
    main()
