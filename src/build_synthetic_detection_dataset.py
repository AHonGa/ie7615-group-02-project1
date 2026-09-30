"""
Milestone 2 — compose synthetic multi-celebrity 3x3 grid images for object
detection, with YOLO-format bounding box annotations.

Each grid image places 9 unique celebrity face crops (drawn from the class-wide
identity pool, not just your team's own 4 IDs) into a 3x3 layout, then resizes
the final composite to 640x640 (YOLOv8's default input size). Placement, scale,
and background are randomized per grid so no two grids look identical.

Requires: extracted_identities/<identity_id>/*.jpg for every ID passed in
(run src/extract_class_pool_images.py first).

Usage:
    python src/build_synthetic_detection_dataset.py --ids 3 7 1212 8335 2619 797 10002 5695 4422 --n-grids 6

    --ids       every identity ID available to draw from (needs >= 9 for a full grid)
    --n-grids   how many 3x3 grid images to produce (e.g. 6: 2 for the report + 4 for Priyanka)
    --seed      base random seed (each grid gets seed + grid_index, so runs are reproducible)

Output (under detection_dataset/):
    images/grid_0001.jpg ... images/grid_000N.jpg   — 640x640 composite images
    labels/grid_0001.txt ... labels/grid_000N.txt   — YOLO-format labels, one row per face:
                                                        class_id x_center y_center width height
                                                        (all normalized 0-1)
    classes.json                                     — identity_id -> class_id mapping used
                                                        (share this mapping with the group /
                                                        Priyanka if grids from multiple teams
                                                        need to be combined for Milestone 3,
                                                        so class ids line up across everyone's data)
    manifest.csv                                      — one row per grid: which 9 identities +
                                                         which image file were used, for the
                                                         data-preparation notebook to narrate
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = REPO_ROOT / "extracted_identities"
OUT_DIR = REPO_ROOT / "detection_dataset"
IMAGES_DIR = OUT_DIR / "images"
LABELS_DIR = OUT_DIR / "labels"
PREVIEW_DIR = OUT_DIR / "previews"  # same images with boxes drawn on top, for sanity-checking

CANVAS_SIZE = 640
GRID_N = 3  # 3x3
CELL_SIZE = CANVAS_SIZE // GRID_N  # ~213

# A handful of plain background colors to vary "background" across grids/cells,
# standing in for the "shared backgrounds" the assignment describes. Swap in real
# photo backgrounds here later if you want something more realistic.
BACKGROUND_COLORS = [
    (235, 235, 235), (210, 220, 230), (230, 215, 200),
    (220, 230, 220), (225, 210, 225), (200, 200, 210),
]


def load_identity_images(identity_id: int) -> list[Path]:
    folder = EXTRACTED_DIR / str(identity_id)
    if not folder.exists():
        raise SystemExit(
            f"No extracted images for identity {identity_id} — run "
            f"src/extract_class_pool_images.py {identity_id} first."
        )
    imgs = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    if not imgs:
        raise SystemExit(f"extracted_identities/{identity_id}/ exists but has no images.")
    return imgs


def compose_grid(identity_ids: list[int], class_map: dict[int, int], rng: random.Random):
    """Build one 640x640 3x3 grid image + its YOLO label rows."""
    assert len(identity_ids) == GRID_N * GRID_N, "need exactly 9 identities for a 3x3 grid"

    bg_color = rng.choice(BACKGROUND_COLORS)
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), bg_color)
    label_rows = []

    # shuffle which identity goes in which cell, so the arrangement varies per grid
    cell_order = list(range(GRID_N * GRID_N))
    rng.shuffle(cell_order)

    for slot, identity_id in zip(cell_order, identity_ids):
        row, col = divmod(slot, GRID_N)
        cell_x0, cell_y0 = col * CELL_SIZE, row * CELL_SIZE

        img_path = rng.choice(load_identity_images(identity_id))
        face = Image.open(img_path).convert("RGB")

        # random scale: face fills 65-92% of the cell, so faces vary in size across grids
        scale = rng.uniform(0.65, 0.92)
        face_size = int(CELL_SIZE * scale)
        face = face.resize((face_size, face_size))

        # random placement jitter within the cell (face doesn't sit dead-center every time)
        max_offset = CELL_SIZE - face_size
        off_x = rng.randint(0, max(max_offset, 0))
        off_y = rng.randint(0, max(max_offset, 0))

        paste_x = cell_x0 + off_x
        paste_y = cell_y0 + off_y
        canvas.paste(face, (paste_x, paste_y))

        # YOLO format: class_id x_center y_center width height, normalized to canvas size
        x_center = (paste_x + face_size / 2) / CANVAS_SIZE
        y_center = (paste_y + face_size / 2) / CANVAS_SIZE
        w_norm = face_size / CANVAS_SIZE
        h_norm = face_size / CANVAS_SIZE
        class_id = class_map[identity_id]
        label_rows.append((class_id, x_center, y_center, w_norm, h_norm, identity_id, img_path.name))

    return canvas, label_rows


def draw_preview(canvas: Image.Image, label_rows) -> Image.Image:
    """Return a copy of the grid with bounding boxes drawn on top, for visual verification."""
    preview = canvas.copy()
    draw = ImageDraw.Draw(preview)
    for class_id, xc, yc, w, h, identity_id, _ in label_rows:
        x0 = (xc - w / 2) * CANVAS_SIZE
        y0 = (yc - h / 2) * CANVAS_SIZE
        x1 = (xc + w / 2) * CANVAS_SIZE
        y1 = (yc + h / 2) * CANVAS_SIZE
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        draw.text((x0 + 2, max(y0 - 12, 0)), f"id{identity_id}", fill=(255, 0, 0))
    return preview


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", required=True,
                     help="all identity IDs available to draw from (need >= 9)")
    ap.add_argument("--n-grids", type=int, default=6,
                     help="how many 3x3 grid images to produce (default 6: 2 report + 4 extra)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if len(set(args.ids)) < GRID_N * GRID_N:
        raise SystemExit(
            f"Need at least {GRID_N * GRID_N} unique identity IDs, got {len(set(args.ids))}. "
            "Pull in more IDs from the class pool spreadsheet."
        )

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # fixed identity_id -> class_id mapping, sorted for determinism; share classes.json
    # with the group / Priyanka if grids need to line up with other teams' class ids
    unique_ids = sorted(set(args.ids))
    class_map = {iid: idx for idx, iid in enumerate(unique_ids)}
    (OUT_DIR / "classes.json").write_text(json.dumps(class_map, indent=2))

    manifest_rows = []
    used_arrangements = set()

    for grid_idx in range(1, args.n_grids + 1):
        rng = random.Random(args.seed + grid_idx)

        # pick 9 unique identities for this grid; retry if we land on an arrangement
        # (identity set + order) identical to a previous grid, so none repeat
        for _attempt in range(50):
            chosen = rng.sample(unique_ids, GRID_N * GRID_N)
            arrangement_key = tuple(chosen)
            if arrangement_key not in used_arrangements:
                used_arrangements.add(arrangement_key)
                break

        canvas, label_rows = compose_grid(chosen, class_map, rng)

        name = f"grid_{grid_idx:04d}"
        canvas.save(IMAGES_DIR / f"{name}.jpg", quality=95)

        with open(LABELS_DIR / f"{name}.txt", "w") as f:
            for class_id, xc, yc, w, h, _, _ in label_rows:
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        preview = draw_preview(canvas, label_rows)
        preview.save(PREVIEW_DIR / f"{name}_preview.jpg", quality=90)

        for class_id, xc, yc, w, h, identity_id, src_name in label_rows:
            manifest_rows.append({
                "grid": name, "identity_id": identity_id, "class_id": class_id,
                "source_image": src_name, "x_center": round(xc, 4), "y_center": round(yc, 4),
                "width": round(w, 4), "height": round(h, 4),
            })

        print(f"Wrote {name}.jpg + {name}.txt — identities: {chosen}")

    pd.DataFrame(manifest_rows).to_csv(OUT_DIR / "manifest.csv", index=False)
    print(f"\nDone. {args.n_grids} grids written to {IMAGES_DIR}/, labels to {LABELS_DIR}/, "
          f"preview images (boxes drawn) to {PREVIEW_DIR}/.")
    print(f"Class mapping saved to {OUT_DIR / 'classes.json'} — share this with the group "
          "if combining grids across teams for Milestone 3.")


if __name__ == "__main__":
    main()
