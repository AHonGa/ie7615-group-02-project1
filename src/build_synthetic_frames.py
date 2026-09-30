"""Build synthetic multi-identity frames with face-level YOLO labels.

Images are selected from the main CelebA archive via identity_CelebA.txt.
Source photographs are split before composition, and YuNet face boxes are
transformed through resizing, mirroring, and placement into each composite.
"""
import argparse
import csv
import hashlib
import random
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "img_align_celeba"
DEFAULT_IDENTITY_MAP = REPO_ROOT / "data" / "identity_CelebA.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "synthetic_face_frames_70_15_15"
DEFAULT_FACE_MODEL = Path.home() / ".cache" / "ie7615" / "face_detection_yunet_2023mar.onnx"
FACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
FACE_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DEFAULT_IDENTITIES = [3, 7, 1212, 8335]
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def split_sources(source_files, rng):
    shuffled = list(source_files)
    rng.shuffle(shuffled)
    split_names = list(SPLIT_RATIOS)
    exact_counts = {name: len(shuffled) * SPLIT_RATIOS[name] for name in split_names}
    counts = {name: int(exact_counts[name]) for name in split_names}
    remainder_order = sorted(
        split_names,
        key=lambda name: exact_counts[name] - counts[name],
        reverse=True,
    )
    for name in remainder_order[:len(shuffled) - sum(counts.values())]:
        counts[name] += 1

    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def make_frame_identity_sets(identity_ids, frame_count, rng):
    if not 2 <= len(identity_ids) <= 4:
        raise ValueError("Synthetic frames support 2-4 unique identities")
    if frame_count < 1:
        raise ValueError("Each split needs at least one frame")

    frame_identity_sets = [set() for _ in range(frame_count)]
    shuffled_ids = list(identity_ids)
    rng.shuffle(shuffled_ids)
    for identity_index, identity_id in enumerate(shuffled_ids):
        frame_identity_sets[identity_index % frame_count].add(identity_id)

    for frame_identities in frame_identity_sets:
        while len(frame_identities) < 2:
            available_ids = [identity_id for identity_id in identity_ids if identity_id not in frame_identities]
            frame_identities.add(rng.choice(available_ids))
        target_count = rng.randint(len(frame_identities), len(identity_ids))
        while len(frame_identities) < target_count:
            available_ids = [identity_id for identity_id in identity_ids if identity_id not in frame_identities]
            frame_identities.add(rng.choice(available_ids))
    return [list(frame_identities) for frame_identities in frame_identity_sets]


def make_background(size, rng):
    palettes = [
        ((226, 222, 204), (194, 204, 193), (147, 133, 116)),
        ((220, 229, 230), (183, 205, 207), (131, 140, 137)),
        ((234, 222, 207), (206, 196, 181), (151, 132, 113)),
        ((222, 225, 211), (190, 203, 184), (135, 139, 119)),
    ]
    wall_top, wall_bottom, floor_color = rng.choice(palettes)
    horizon = rng.randint(round(size * 0.66), round(size * 0.75))
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)
    for y in range(size):
        if y <= horizon:
            ratio = y / horizon
            start, end = wall_top, wall_bottom
        else:
            ratio = (y - horizon) / (size - horizon)
            start, end = wall_bottom, floor_color
        color = tuple(round(a * (1 - ratio) + b * ratio) for a, b in zip(start, end))
        draw.line((0, y, size, y), fill=color)

    unit = size / 640
    line_width = max(2, round(5 * unit))
    draw.line((0, horizon, size, horizon), fill=tuple(max(0, c - 24) for c in wall_bottom), width=line_width)

    board_left = rng.randint(round(size * 0.06), round(size * 0.28))
    board_top = rng.randint(round(size * 0.15), round(size * 0.31))
    board_width = rng.randint(round(size * 0.43), round(size * 0.60))
    board_height = rng.randint(round(size * 0.25), round(size * 0.34))
    board_right = min(size - round(size * 0.08), board_left + board_width)
    board_bottom = min(round(size * 0.62), board_top + board_height)
    board_fill = rng.choice(((49, 86, 77), (56, 82, 91), (218, 222, 214)))
    draw.rectangle(
        (board_left - line_width, board_top - line_width, board_right + line_width, board_bottom + line_width),
        fill=(111, 91, 70),
    )
    draw.rectangle((board_left, board_top, board_right, board_bottom), fill=board_fill)
    for _ in range(rng.randint(2, 5)):
        x1 = rng.randint(board_left + line_width * 3, max(board_left + line_width * 3, board_right - round(size * 0.12)))
        y1 = rng.randint(board_top + line_width * 3, max(board_top + line_width * 3, board_bottom - line_width * 3))
        x2 = min(board_right - line_width * 2, x1 + rng.randint(round(size * 0.06), round(size * 0.19)))
        draw.line((x1, y1, x2, y1), fill=rng.choice(((224, 222, 195), (174, 202, 191), (198, 202, 193))), width=line_width)

    window_left = rng.choice((round(size * 0.68), round(size * 0.04)))
    window_top = rng.randint(round(size * 0.12), round(size * 0.24))
    window_width = rng.randint(round(size * 0.19), round(size * 0.27))
    window_height = rng.randint(round(size * 0.28), round(size * 0.38))
    window_right = min(size - round(size * 0.025), window_left + window_width)
    window_bottom = min(round(size * 0.63), window_top + window_height)
    draw.rectangle((window_left - line_width, window_top - line_width, window_right + line_width, window_bottom + line_width), fill=(116, 103, 84))
    draw.rectangle((window_left, window_top, window_right, window_bottom), fill=rng.choice(((176, 203, 209), (191, 211, 211), (180, 199, 215))))
    draw.line((window_left + window_width // 2, window_top, window_left + window_width // 2, window_bottom), fill=(225, 222, 207), width=line_width)
    draw.line((window_left, window_top + window_height // 2, window_right, window_top + window_height // 2), fill=(225, 222, 207), width=line_width)

    for _ in range(rng.randint(1, 3)):
        poster_left = rng.randint(round(size * 0.68), round(size * 0.82))
        poster_top = rng.randint(round(size * 0.18), round(size * 0.42))
        poster_width = rng.randint(round(size * 0.07), round(size * 0.11))
        poster_height = rng.randint(round(size * 0.11), round(size * 0.17))
        draw.rectangle(
            (poster_left, poster_top, min(size, poster_left + poster_width), min(horizon, poster_top + poster_height)),
            fill=rng.choice(((190, 155, 119), (151, 174, 160), (180, 170, 142))),
            outline=(239, 231, 211),
            width=max(1, line_width // 2),
        )

    for desk_x in (rng.uniform(-0.1, 0.18), rng.uniform(0.73, 0.92)):
        x = round(size * desk_x)
        y = rng.randint(round(size * 0.78), round(size * 0.86))
        desk_width = round(size * rng.uniform(0.24, 0.34))
        draw.polygon(
            ((x, y), (x + desk_width, y - round(size * 0.025)),
             (x + desk_width + round(size * 0.04), y + round(size * 0.035)),
             (x + round(size * 0.035), y + round(size * 0.065))),
            fill=rng.choice(((122, 94, 69), (143, 111, 78), (105, 91, 74))),
        )
        leg_y = min(size, y + round(size * 0.15))
        draw.line((x + round(size * 0.05), y + round(size * 0.05), x + round(size * 0.04), leg_y), fill=(75, 72, 65), width=line_width)
        draw.line((x + desk_width, y + round(size * 0.02), x + desk_width + round(size * 0.02), leg_y), fill=(75, 72, 65), width=line_width)

    return image.filter(ImageFilter.GaussianBlur(rng.uniform(2.0, 5.5) * unit))


def portrait_crop(image):
    width, height = image.size
    return image.crop((round(width * 0.06), round(height * 0.04), round(width * 0.94), round(height * 0.94)))


def load_face_detector(model_path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        temporary_path = model_path.with_suffix(model_path.suffix + ".download")
        try:
            urllib.request.urlretrieve(FACE_MODEL_URL, temporary_path)
            temporary_path.replace(model_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != FACE_MODEL_SHA256:
        raise ValueError(f"Unexpected YuNet model checksum for {model_path}: {digest}")
    return cv2.FaceDetectorYN.create(str(model_path), "", (178, 218), 0.5, 0.3, 5000)


def collect_source_images(source_dir, identity_map_path, identity_ids):
    selected_ids = set(identity_ids)
    sources_by_identity = {identity_id: [] for identity_id in identity_ids}
    with identity_map_path.open(encoding="utf-8") as identity_file:
        for line in identity_file:
            image_name, identity_text = line.split()
            identity_id = int(identity_text)
            if identity_id in selected_ids:
                image_path = source_dir / image_name
                if image_path.is_file():
                    sources_by_identity[identity_id].append(image_path)

    for identity_id, image_paths in sources_by_identity.items():
        if len(image_paths) < 10:
            raise ValueError(f"Expected at least 10 source images for identity {identity_id}; found {len(image_paths)}")
    return sources_by_identity


def detect_face_box(portrait, detector):
    rgb = np.asarray(portrait)
    detector.setInputSize((portrait.width, portrait.height))
    _, detections = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if detections is None or len(detections) == 0:
        raise ValueError(f"YuNet could not find a face in an aligned source crop ({portrait.size})")

    detection = max(
        detections,
        key=lambda row: (float(row[-1]), float(row[2] * row[3])),
    )
    x, y, width, height = map(float, detection[:4])
    pad_x, pad_y = width * 0.03, height * 0.04
    left = max(0, int(np.floor(x - pad_x)))
    top = max(0, int(np.floor(y - pad_y)))
    right = min(portrait.width, int(np.ceil(x + width + pad_x)))
    bottom = min(portrait.height, int(np.ceil(y + height + pad_y)))
    return left, top, right - left, bottom - top


def transform_box(box, frame_size, flip_horizontal, scale, rotation_deg):
    left, top, right, bottom = map(float, box)
    if flip_horizontal:
        left, right = frame_size - right, frame_size - left

    center = frame_size / 2
    corners = ((left, top), (right, top), (right, bottom), (left, bottom))
    radians = np.deg2rad(rotation_deg)
    cosine, sine = np.cos(radians), np.sin(radians)
    transformed = []
    for x, y in corners:
        x = center + (x - center) * scale
        y = center + (y - center) * scale
        relative_x, relative_y = x - center, y - center
        transformed.append((
            center + cosine * relative_x + sine * relative_y,
            center - sine * relative_x + cosine * relative_y,
        ))

    left = max(0, int(np.floor(min(point[0] for point in transformed) + 1e-8)))
    top = max(0, int(np.floor(min(point[1] for point in transformed) + 1e-8)))
    right = min(frame_size, int(np.ceil(max(point[0] for point in transformed) - 1e-8)))
    bottom = min(frame_size, int(np.ceil(max(point[1] for point in transformed) - 1e-8)))
    if right <= left or bottom <= top:
        raise ValueError(f"Augmentation removed a face box: {box}")
    return left, top, right, bottom


def augment_frame_and_boxes(frame, annotations, rng):
    frame_size = frame.width
    if frame.width != frame.height:
        raise ValueError("Frame augmentation expects square images")

    flip_horizontal = rng.random() < 0.5
    scale = rng.uniform(0.92, 1.08)
    scaled_size = round(frame_size * scale)
    scale = scaled_size / frame_size
    rotation_deg = rng.uniform(-6.0, 6.0)
    brightness = rng.uniform(0.90, 1.10)
    contrast = rng.uniform(0.90, 1.10)

    if flip_horizontal:
        frame = ImageOps.mirror(frame)
    if scale != 1.0:
        scaled = frame.resize((scaled_size, scaled_size), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (frame_size, frame_size), frame.getpixel((0, 0)))
        offset = (frame_size - scaled_size) // 2
        canvas.paste(scaled, (offset, offset))
        frame = canvas
    frame = frame.rotate(
        rotation_deg,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=frame.getpixel((0, 0)),
    )
    frame = ImageEnhance.Brightness(frame).enhance(brightness)
    frame = ImageEnhance.Contrast(frame).enhance(contrast)

    transformed_annotations = []
    for annotation in annotations:
        transformed = annotation.copy()
        transformed["bbox"] = transform_box(
            annotation["bbox"], frame_size, flip_horizontal, scale, rotation_deg
        )
        transformed_annotations.append(transformed)

    parameters = {
        "horizontal_flip": flip_horizontal,
        "scale": scale,
        "rotation_deg": rotation_deg,
        "brightness": brightness,
        "contrast": contrast,
    }
    return frame, transformed_annotations, parameters


def compose_frame(identity_ids, sources_by_identity, class_ids, frame_size, rng, detector, face_box_cache):
    frame = make_background(frame_size, rng)
    layouts = {
        2: [(0.35, 0.58), (0.65, 0.58)],
        3: [(0.24, 0.59), (0.50, 0.54), (0.76, 0.59)],
        4: [(0.18, 0.57), (0.39, 0.53), (0.61, 0.53), (0.82, 0.57)],
    }
    anchors = layouts[len(identity_ids)]
    rng.shuffle(anchors)
    base_heights = {2: 0.52, 3: 0.47, 4: 0.41}
    annotations = []

    for identity_id, (center_x, center_y) in sorted(zip(identity_ids, anchors), key=lambda item: item[1][1]):
        source_path = rng.choice(sources_by_identity[identity_id])
        with Image.open(source_path) as source:
            portrait = portrait_crop(source.convert("RGB"))
        if source_path not in face_box_cache:
            face_box_cache[source_path] = detect_face_box(portrait, detector)
        face_x, face_y, face_width, face_height = face_box_cache[source_path]
        target_height = round(frame_size * base_heights[len(identity_ids)] * rng.uniform(0.88, 1.10))
        target_width = round(target_height * portrait.width / portrait.height)
        portrait = portrait.resize((target_width, target_height), Image.Resampling.LANCZOS)
        portrait = ImageEnhance.Brightness(portrait).enhance(rng.uniform(0.92, 1.08))
        portrait = ImageEnhance.Contrast(portrait).enhance(rng.uniform(0.95, 1.05))
        portrait = ImageEnhance.Color(portrait).enhance(rng.uniform(0.94, 1.06))
        mirrored = rng.random() < 0.5
        if mirrored:
            portrait = ImageOps.mirror(portrait)

        x = round(frame_size * (center_x + rng.uniform(-0.025, 0.025)) - target_width / 2)
        y = round(frame_size * (center_y + rng.uniform(-0.035, 0.035)) - target_height / 2)
        x = max(0, min(frame_size - target_width, x))
        y = max(0, min(frame_size - target_height, y))
        edge = max(5, round(frame_size * 0.018))
        mask = Image.new("L", portrait.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle(
            (edge, edge, target_width - edge, target_height - edge),
            radius=edge * 2,
            fill=255,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(edge * 0.65))
        frame.paste(portrait, (x, y), mask)

        scale_x = target_width / portrait.width
        scale_y = target_height / portrait.height
        face_x *= scale_x
        face_y *= scale_y
        face_width *= scale_x
        face_height *= scale_y
        if mirrored:
            face_x = target_width - face_x - face_width
        face_left = max(0, round(x + face_x))
        face_top = max(0, round(y + face_y))
        face_right = min(frame_size, round(x + face_x + face_width))
        face_bottom = min(frame_size, round(y + face_y + face_height))
        annotations.append({
            "class_id": class_ids[identity_id],
            "identity_id": identity_id,
            "source_image": source_path.name,
            "bbox": (face_left, face_top, face_right, face_bottom),
        })

    return frame, annotations


def write_visual_checks(output_dir, identity_ids):
    visual_dir = output_dir / "visual_checks"
    visual_dir.mkdir(exist_ok=True)
    for split in SPLIT_RATIOS:
        image_paths = sorted((output_dir / "images" / split).glob("*.jpg"))
        if not image_paths:
            continue
        preview_indices = sorted({0, len(image_paths) // 2})
        for index in preview_indices:
            image_path = image_paths[index]
            label_path = output_dir / "labels" / split / f"{image_path.stem}.txt"
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")
            draw = ImageDraw.Draw(image)
            for line in label_path.read_text(encoding="utf-8").splitlines():
                class_id, center_x, center_y, width, height = line.split()
                center_x, center_y, width, height = map(float, (center_x, center_y, width, height))
                left = round((center_x - width / 2) * image.width)
                top = round((center_y - height / 2) * image.height)
                right = round((center_x + width / 2) * image.width)
                bottom = round((center_y + height / 2) * image.height)
                draw.rectangle((left, top, right, bottom), outline=(255, 35, 45), width=3)
                draw.text((left, max(0, top - 15)), str(identity_ids[int(class_id)]), fill=(255, 35, 45))
            image.save(visual_dir / f"{split}_box_review_{index:05d}.jpg", quality=95)


def build_dataset(source_dir, identity_map_path, output_dir, identity_ids, frame_size, counts, seed, face_model_path):
    rng = random.Random(seed)
    class_ids = {identity_id: index for index, identity_id in enumerate(identity_ids)}
    split_sources_by_identity = {split: {} for split in SPLIT_RATIOS}
    sources_by_identity = collect_source_images(source_dir, identity_map_path, identity_ids)

    for identity_id in identity_ids:
        partitions = split_sources(sorted(sources_by_identity[identity_id]), rng)
        for split, files in partitions.items():
            if not files:
                raise ValueError(f"No {split} source images available for identity {identity_id}")
            split_sources_by_identity[split][identity_id] = files

    output_dir.mkdir(parents=True, exist_ok=False)
    detector = load_face_detector(face_model_path)
    face_box_cache = {}
    source_split_path = output_dir / "source_splits.csv"
    with source_split_path.open("w", newline="", encoding="utf-8") as source_split_file:
        source_writer = csv.writer(source_split_file)
        source_writer.writerow(["split", "identity_id", "source_image"])
        for split, identity_sources in split_sources_by_identity.items():
            for identity_id, image_paths in identity_sources.items():
                source_writer.writerows(
                    (split, identity_id, image_path.name) for image_path in image_paths
                )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow([
            "split", "image", "identity_id", "source_image", "face_x", "face_y",
            "face_width", "face_height", "horizontal_flip", "scale", "rotation_deg",
            "brightness", "contrast",
        ])

        for split, frame_count in counts.items():
            image_dir = output_dir / "images" / split
            label_dir = output_dir / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            available_identities = list(identity_ids)
            frame_identity_sets = make_frame_identity_sets(identity_ids, frame_count, rng)

            for frame_index, frame_identities in enumerate(frame_identity_sets):
                frame, annotations = compose_frame(
                    frame_identities, split_sources_by_identity[split], class_ids, frame_size,
                    rng, detector, face_box_cache,
                )
                frame, annotations, parameters = augment_frame_and_boxes(frame, annotations, rng)
                labels = []
                for annotation in annotations:
                    left, top, right, bottom = annotation["bbox"]
                    width, height = right - left, bottom - top
                    center_x = (left + width / 2) / frame_size
                    center_y = (top + height / 2) / frame_size
                    labels.append(
                        f"{annotation['class_id']} {center_x:.8f} {center_y:.8f} "
                        f"{width / frame_size:.8f} {height / frame_size:.8f}"
                    )
                stem = f"{split}_{frame_index:05d}"
                frame.save(image_dir / f"{stem}.jpg", quality=92)
                (label_dir / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
                for annotation in annotations:
                    left, top, right, bottom = annotation["bbox"]
                    writer.writerow([
                        split, f"images/{split}/{stem}.jpg", annotation["identity_id"],
                        annotation["source_image"], left, top, right - left, bottom - top,
                        parameters["horizontal_flip"], f"{parameters['scale']:.6f}",
                        f"{parameters['rotation_deg']:.6f}", f"{parameters['brightness']:.6f}",
                        f"{parameters['contrast']:.6f}",
                    ])

    yaml_lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(identity_ids)}",
        "names:",
    ]
    yaml_lines.extend(f"  {class_ids[identity_id]}: '{identity_id}'" for identity_id in identity_ids)
    (output_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "Synthetic multi-identity classroom-style frames for the selected CelebA identities.\n"
        "YOLO boxes enclose YuNet-detected face regions; classes map to identity IDs\n"
        "in dataset.yaml. Source photographs are partitioned before frame composition,\n"
        "so source files do not cross train/validation/test splits. Background scenes,\n"
        "group layouts, portrait scales, positions, and lighting vary by frame. These are\n"
        "synthetic composites, not natural-scene face-detection annotations.\n\n"
        "Frame augmentations: horizontal flip p=0.5; scale 0.92-1.08; rotation -6 to +6 degrees;\n"
        "brightness and contrast 0.90-1.10. Geometric transforms are applied to both pixels\n"
        "and face boxes; manifest.csv records the sampled parameters for every frame.\n",
        encoding="utf-8",
    )
    write_visual_checks(output_dir, identity_ids)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--identity-map", type=Path, default=DEFAULT_IDENTITY_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--face-model-path", type=Path, default=DEFAULT_FACE_MODEL)
    parser.add_argument("--identities", type=int, nargs="+", default=DEFAULT_IDENTITIES)
    parser.add_argument("--frame-size", type=int, default=640)
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--val-count", type=int, default=40)
    parser.add_argument("--test-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if len(args.identities) < 2 or len(set(args.identities)) != len(args.identities):
        parser.error("Provide at least two unique identity IDs.")
    if args.frame_size < 256:
        parser.error("Frame size must be at least 256 pixels.")
    if min(args.train_count, args.val_count, args.test_count) < 1:
        parser.error("Each split must contain at least one frame.")
    if not args.source_dir.is_dir():
        parser.error(f"CelebA image directory does not exist: {args.source_dir}")
    if not args.identity_map.is_file():
        parser.error(f"CelebA identity map does not exist: {args.identity_map}")

    counts = {"train": args.train_count, "val": args.val_count, "test": args.test_count}
    manifest = build_dataset(
        args.source_dir, args.identity_map, args.output_dir, args.identities,
        args.frame_size, counts, args.seed, args.face_model_path,
    )
    print(f"Generated {sum(counts.values())} frames at {args.output_dir.resolve()}")
    print(f"YOLO dataset config: {args.output_dir / 'dataset.yaml'}")
    print(f"Source manifest: {manifest}")


if __name__ == "__main__":
    main()
