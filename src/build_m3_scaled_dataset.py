"""Milestone 3 — scaled-up detection dataset built with the Milestone 2 grid method.

Why: the Milestone 2 dataset (6 base grids, 30 images) shows each celebrity in only
about 4 of the ~25 available face photos, so YOLOv8 memorizes those photos and cannot
name the same person in a different photo. This script keeps everything about the
Milestone 2 design (same 13 class-pool identities, same class ids, same 3x3 / 640 px
composition, scale and placement jitter, backgrounds) and changes only the amount of
data and how it is split:

  1. Each identity's face photos are split 70/15/15 into train / val / test photos
     BEFORE any grid is built, so no photo appears in two splits (no leakage).
  2. Grids for a split draw only from that split's photos, cycling through every
     photo so all of them are used.
  3. Identities are balanced per split: each grid takes the 9 least-used identities
     (random tie-break), so every identity, including Group 02's 3/7/1212/8335,
     appears in train, val and test about equally often.

Offline augmentation is not applied here; YOLOv8's online augmentation (flip, HSV,
scale, translate, mosaic) varies every training image each epoch, and its intensity
is one of the swept hyperparameters.

Usage (from the repo root):
    python src/build_m3_scaled_dataset.py                 # 120 / 12 / 12 grids
    python src/build_m3_scaled_dataset.py --train 60 --val 12 --test 12
Output: detection_dataset_m3/{train,val,test}/{images,labels}/, classes.json,
        photo_split.csv, manifest.csv, previews/ (a few grids with boxes drawn)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_synthetic_detection_dataset import (  # noqa: E402  (reuse the Milestone 2 design)
    BACKGROUND_COLORS, CANVAS_SIZE, CELL_SIZE, GRID_N, draw_preview,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = REPO_ROOT / "extracted_identities"
M2_DIR = REPO_ROOT / "detection_dataset"
OUT_DIR = REPO_ROOT / "detection_dataset_m3"
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def split_photos(class_map: dict[int, int], rng: random.Random) -> dict[str, dict[int, list[Path]]]:
    """Per identity, shuffle its photos and cut them 70/15/15 (at least 2 val and 2 test photos)."""
    pools = {s: {} for s in SPLIT_FRACTIONS}
    for iid in class_map:
        photos = sorted((EXTRACTED_DIR / str(iid)).glob("*.jpg")) + sorted((EXTRACTED_DIR / str(iid)).glob("*.png"))
        if len(photos) < 6:
            raise SystemExit(f"Identity {iid} has only {len(photos)} photos; need at least 6 to split.")
        rng.shuffle(photos)
        n_test = max(2, round(len(photos) * SPLIT_FRACTIONS["test"]))
        n_val = max(2, round(len(photos) * SPLIT_FRACTIONS["val"]))
        pools["test"][iid] = photos[:n_test]
        pools["val"][iid] = photos[n_test:n_test + n_val]
        pools["train"][iid] = photos[n_test + n_val:]
    return pools


class PhotoCycler:
    """Hands out an identity's photos in shuffled order, reshuffling once all are used."""

    def __init__(self, pool: dict[int, list[Path]], rng: random.Random):
        self.pool, self.rng, self.queues = pool, rng, {}

    def next(self, iid: int) -> Path:
        if not self.queues.get(iid):
            self.queues[iid] = self.pool[iid][:]
            self.rng.shuffle(self.queues[iid])
        return self.queues[iid].pop()


def compose_grid(identity_ids, cycler: PhotoCycler, class_map, rng: random.Random):
    """Same composition as Milestone 2's compose_grid, but faces come from one split's photo pool."""
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), rng.choice(BACKGROUND_COLORS))
    cells = list(range(GRID_N * GRID_N))
    rng.shuffle(cells)
    rows = []
    for slot, iid in zip(cells, identity_ids):
        r, c = divmod(slot, GRID_N)
        photo = cycler.next(iid)
        size = int(CELL_SIZE * rng.uniform(0.65, 0.92))
        face = Image.open(photo).convert("RGB").resize((size, size))
        x = c * CELL_SIZE + rng.randint(0, max(CELL_SIZE - size, 0))
        y = r * CELL_SIZE + rng.randint(0, max(CELL_SIZE - size, 0))
        canvas.paste(face, (x, y))
        rows.append((class_map[iid], (x + size / 2) / CANVAS_SIZE, (y + size / 2) / CANVAS_SIZE,
                     size / CANVAS_SIZE, size / CANVAS_SIZE, iid, photo.name))
    return canvas, rows


def build(n_grids: dict[str, int], seed: int = 42, out_dir: Path = OUT_DIR, n_previews: int = 3) -> Path:
    raw = json.loads((M2_DIR / "classes.json").read_text())
    class_map = {int(k): v for k, v in raw.items()}  # same ids as Milestone 2
    rng = random.Random(seed)
    pools = split_photos(class_map, rng)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "previews").mkdir(parents=True)
    shutil.copy2(M2_DIR / "classes.json", out_dir / "classes.json")
    pd.DataFrame([{"identity_id": iid, "split": s, "photo": p.name}
                  for s, pool in pools.items() for iid, photos in pool.items() for p in photos]
                 ).to_csv(out_dir / "photo_split.csv", index=False)

    manifest = []
    ids = sorted(class_map)
    for split, n in n_grids.items():
        (out_dir / split / "images").mkdir(parents=True)
        (out_dir / split / "labels").mkdir(parents=True)
        # each split gets its own RNG, so changing the number of train grids leaves val/test unchanged
        rng = random.Random(f"{seed}-{split}")
        cycler, usage = PhotoCycler(pools[split], rng), Counter({i: 0 for i in ids})
        for g in range(1, n + 1):
            # the 9 least-used identities so far (random tie-break) -> balanced appearances
            chosen = sorted(ids, key=lambda i: (usage[i], rng.random()))[:GRID_N * GRID_N]
            usage.update(chosen)
            canvas, rows = compose_grid(chosen, cycler, class_map, rng)
            name = f"{split}_{g:04d}"
            canvas.save(out_dir / split / "images" / f"{name}.jpg", quality=95)
            (out_dir / split / "labels" / f"{name}.txt").write_text(
                "".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for c, xc, yc, w, h, _, _ in rows))
            if g <= n_previews:
                draw_preview(canvas, rows).save(out_dir / "previews" / f"{name}_preview.jpg", quality=90)
            manifest += [{"split": split, "grid": name, "identity_id": iid, "class_id": c, "source_image": src,
                          "x_center": round(xc, 4), "y_center": round(yc, 4), "width": round(w, 4)}
                         for c, xc, yc, w, h, iid, src in rows]
    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=120)
    ap.add_argument("--val", type=int, default=12)
    ap.add_argument("--test", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = build({"train": args.train, "val": args.val, "test": args.test}, args.seed)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
