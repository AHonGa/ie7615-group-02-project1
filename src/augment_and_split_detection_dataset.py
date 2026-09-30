"""
Milestone 2 — apply a documented augmentation strategy to the synthetic grid
images and organize the result into a leak-free train/val/test split.

Why these augmentations (documented per the assignment's rubric):
  - Horizontal flip (p=0.5): faces are roughly left-right symmetric, so flipping
    is a free, label-safe way to double effective viewpoint diversity.
  - Bounded rotation (+/-8 degrees): simulates a slightly tilted camera/photo,
    without rotating so far that faces become unrealistic within a "grid photo"
    context; bounding boxes are recomputed from the rotated face corners.
  - Brightness/contrast jitter (0.85x-1.15x each): simulates different lighting
    conditions across "shared backgrounds", without altering bounding boxes.
  - Slight zoom (0.95x-1.05x, centered): simulates minor camera distance/framing
    differences; bounding boxes are rescaled around the image center to match.
  All transforms are bounded specifically so that every face stays inside the
  640x640 frame (boxes are clipped/dropped only if less than 10% of the box
  would remain visible, which does not happen at these bounded parameters in
  practice for a centered 3x3 grid).

Usage:
    python src/augment_and_split_detection_dataset.py --n-aug 4 --seed 42

Reads:   detection_dataset/images/*.jpg + detection_dataset/labels/*.txt
         (produced by build_synthetic_detection_dataset.py)
Writes:  detection_dataset/train/images, /labels
         detection_dataset/val/images,   /labels
         detection_dataset/test/images,  /labels
         detection_dataset/split_manifest.csv  (which base grid + augmentation
                                                 went into which split, for the
                                                 data-preparation notebook)

Split is done at the BASE GRID level (not per augmented copy), so every
augmented version of a given grid lands in the same split — this is what
prevents image leakage across train/val/test.
"""
import argparse
import csv
import math
import random
from pathlib import Path

from PIL import Image, ImageEnhance

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "detection_dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

CANVAS_SIZE = 640
BG_FALLBACK = (222, 222, 222)  # used to fill corners exposed by rotation/zoom-out

# ---- documented augmentation parameters ----
FLIP_PROB = 0.5
ROTATION_RANGE_DEG = 8.0        # +/- degrees
BRIGHTNESS_RANGE = (0.85, 1.15)
CONTRAST_RANGE = (0.85, 1.15)
ZOOM_RANGE = (0.95, 1.05)
MIN_VISIBLE_FRACTION = 0.10      # drop a box if less than this fraction of its area survives


def read_yolo_labels(path: Path):
    rows = []
    for line in path.read_text().strip().splitlines():
        cls, xc, yc, w, h = line.split()
        rows.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return rows


def write_yolo_labels(path: Path, rows):
    with open(path, "w") as f:
        for cls, xc, yc, w, h in rows:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def flip_boxes(rows):
    return [(cls, 1.0 - xc, yc, w, h) for cls, xc, yc, w, h in rows]


def zoom_boxes(rows, zoom):
    # centered zoom: point'_norm = (point_norm - 0.5) * zoom + 0.5
    out = []
    for cls, xc, yc, w, h in rows:
        xc2 = (xc - 0.5) * zoom + 0.5
        yc2 = (yc - 0.5) * zoom + 0.5
        w2 = w * zoom
        h2 = h * zoom
        out.append((cls, xc2, yc2, w2, h2))
    return out


def rotate_boxes(rows, angle_deg):
    """Recompute axis-aligned boxes after rotating the canvas by angle_deg about its center."""
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    out = []
    for cls, xc, yc, w, h in rows:
        # 4 corners of the box, centered at (0,0) for rotation, in normalized units
        corners = [
            (xc - w / 2 - 0.5, yc - h / 2 - 0.5),
            (xc + w / 2 - 0.5, yc - h / 2 - 0.5),
            (xc - w / 2 - 0.5, yc + h / 2 - 0.5),
            (xc + w / 2 - 0.5, yc + h / 2 - 0.5),
        ]
        rotated = []
        for x, y in corners:
            xr = x * cos_t - y * sin_t
            yr = x * sin_t + y * cos_t
            rotated.append((xr + 0.5, yr + 0.5))
        xs = [p[0] for p in rotated]
        ys = [p[1] for p in rotated]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        out.append((cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return out


def clip_and_filter(rows):
    """Clip boxes to the [0,1] frame; drop any box that loses too much area off-frame."""
    kept = []
    for cls, xc, yc, w, h in rows:
        x0, x1 = xc - w / 2, xc + w / 2
        y0, y1 = yc - h / 2, yc + h / 2
        orig_area = max(w, 1e-9) * max(h, 1e-9)

        cx0, cx1 = max(x0, 0.0), min(x1, 1.0)
        cy0, cy1 = max(y0, 0.0), min(y1, 1.0)
        if cx1 <= cx0 or cy1 <= cy0:
            continue  # fully off-frame

        clipped_area = (cx1 - cx0) * (cy1 - cy0)
        if clipped_area / orig_area < MIN_VISIBLE_FRACTION:
            continue

        kept.append((cls, (cx0 + cx1) / 2, (cy0 + cy1) / 2, cx1 - cx0, cy1 - cy0))
    return kept


def augment_one(image: Image.Image, rows, rng: random.Random):
    rows = list(rows)

    if rng.random() < FLIP_PROB:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        rows = flip_boxes(rows)

    angle = rng.uniform(-ROTATION_RANGE_DEG, ROTATION_RANGE_DEG)
    image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=BG_FALLBACK)
    rows = rotate_boxes(rows, angle)

    zoom = rng.uniform(*ZOOM_RANGE)
    new_size = max(int(round(CANVAS_SIZE * zoom)), 1)
    resized = image.resize((new_size, new_size), Image.BICUBIC)
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BG_FALLBACK)
    offset = (CANVAS_SIZE - new_size) // 2
    canvas.paste(resized, (offset, offset))
    image = canvas
    rows = zoom_boxes(rows, zoom)

    brightness = rng.uniform(*BRIGHTNESS_RANGE)
    contrast = rng.uniform(*CONTRAST_RANGE)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)

    rows = clip_and_filter(rows)
    return image, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-aug", type=int, default=4,
                     help="augmented copies to generate per base grid (default 4)")
    ap.add_argument("--val-grids", type=int, default=1,
                     help="how many base grids go to validation (default 1)")
    ap.add_argument("--test-grids", type=int, default=1,
                     help="how many base grids go to test (default 1)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base_images = sorted(IMAGES_DIR.glob("*.jpg"))
    if not base_images:
        raise SystemExit(f"No base grid images found in {IMAGES_DIR} — run "
                          "build_synthetic_detection_dataset.py first.")

    rng = random.Random(args.seed)
    shuffled = base_images[:]
    rng.shuffle(shuffled)

    n_val, n_test = args.val_grids, args.test_grids
    n_train = len(shuffled) - n_val - n_test
    if n_train <= 0:
        raise SystemExit(f"Not enough base grids ({len(shuffled)}) for the requested "
                          f"val={n_val} + test={n_test} split.")

    split_assignment = {}
    for img in shuffled[:n_train]:
        split_assignment[img.stem] = "train"
    for img in shuffled[n_train:n_train + n_val]:
        split_assignment[img.stem] = "val"
    for img in shuffled[n_train + n_val:]:
        split_assignment[img.stem] = "test"

    for split in ("train", "val", "test"):
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for img_path in base_images:
        stem = img_path.stem
        split = split_assignment[stem]
        label_path = LABELS_DIR / f"{stem}.txt"
        base_rows = read_yolo_labels(label_path)
        base_image = Image.open(img_path).convert("RGB")

        # keep the un-augmented original as one sample in the split too
        out_name = f"{stem}_orig"
        base_image.save(DATASET_DIR / split / "images" / f"{out_name}.jpg", quality=95)
        write_yolo_labels(DATASET_DIR / split / "labels" / f"{out_name}.txt", base_rows)
        manifest_rows.append([stem, out_name, split, "none", len(base_rows)])

        for aug_idx in range(1, args.n_aug + 1):
            aug_rng = random.Random(args.seed * 1000 + hash(stem) % 1000 + aug_idx)
            aug_image, aug_rows = augment_one(base_image, base_rows, aug_rng)
            out_name = f"{stem}_aug{aug_idx}"
            aug_image.save(DATASET_DIR / split / "images" / f"{out_name}.jpg", quality=95)
            write_yolo_labels(DATASET_DIR / split / "labels" / f"{out_name}.txt", aug_rows)
            manifest_rows.append([stem, out_name, split, f"aug{aug_idx}", len(aug_rows)])

    with open(DATASET_DIR / "split_manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["base_grid", "output_name", "split", "augmentation", "n_boxes"])
        writer.writerows(manifest_rows)

    counts = {"train": 0, "val": 0, "test": 0}
    for row in manifest_rows:
        counts[row[2]] += 1

    print("Split (by base grid, no leakage):")
    for split, imgs in [("train", shuffled[:n_train]), ("val", shuffled[n_train:n_train+n_val]),
                         ("test", shuffled[n_train+n_val:])]:
        print(f"  {split}: base grids = {[p.stem for p in imgs]}")
    print(f"\nTotal images written (original + augmented copies): "
          f"train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print(f"Manifest written to {DATASET_DIR / 'split_manifest.csv'}")


if __name__ == "__main__":
    main()
