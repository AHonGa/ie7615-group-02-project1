"""Render curated test examples with YOLO predictions and ground-truth boxes."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "detection_dataset"
DEFAULT_MODEL = (
    REPO_ROOT / "runs" / "detect" / "celeba_yolov8n_milestone2" / "weights" / "best.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "detect" / "celeba_yolov8n_test_gallery"
GALLERY_CASES = (
    {
        "stem": "grid_0006_orig",
        "confidence": 0.03,
        "name": "success_and_identity_confusion",
        "title": "Correct localization and identity confusions",
        "caption": (
            "Identity 7 is localized correctly, while identity 3 is predicted over "
            "faces belonging to other identities."
        ),
    },
    {
        "stem": "grid_0006_aug1",
        "confidence": 0.03,
        "name": "missed_detections",
        "title": "Missed detections at the selected confidence floor",
        "caption": (
            "No prediction survives confidence 0.03; cyan outlines show the nine "
            "ground-truth faces that were missed at this threshold."
        ),
    },
    {
        "stem": "grid_0006_aug4",
        "confidence": 0.01,
        "name": "overlapping_predictions",
        "title": "Overlapping class predictions",
        "caption": (
            "Two different identity predictions overlap the same face region "
            "(prediction-box IoU 0.95), illustrating a duplicate/confusion error."
        ),
    },
)

COLORS = {
    "ground_truth": (28, 180, 220),
    "correct": (35, 190, 95),
    "confusion": (255, 174, 45),
    "false_positive": (235, 65, 75),
    "duplicate": (205, 92, 230),
}
HEADER_HEIGHT = 42
FOOTER_HEIGHT = 30
MATCH_IOU = 0.50
NMS_IOU = 0.50


def load_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def read_labels(path, width, height):
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        class_id, center_x, center_y, box_width, box_height = line.split()
        center_x, box_width = float(center_x) * width, float(box_width) * width
        center_y, box_height = float(center_y) * height, float(box_height) * height
        labels.append(
            (
                int(class_id),
                (
                    center_x - box_width / 2,
                    center_y - box_height / 2,
                    center_x + box_width / 2,
                    center_y + box_height / 2,
                ),
            )
        )
    return labels


def intersection_over_union(first, second):
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def classify_predictions(detections, ground_truth):
    matched = set()
    outcomes = []
    for class_id, confidence, box in sorted(detections, key=lambda item: item[1], reverse=True):
        overlaps = [
            (intersection_over_union(box, truth_box), index, truth_class)
            for index, (truth_class, truth_box) in enumerate(ground_truth)
        ]
        best_iou, best_index, best_class = max(overlaps, default=(0.0, None, None))
        if best_iou >= MATCH_IOU and class_id == best_class and best_index not in matched:
            outcome = "correct"
            matched.add(best_index)
        elif best_iou >= MATCH_IOU and class_id != best_class:
            outcome = "confusion"
        elif best_iou >= MATCH_IOU:
            outcome = "duplicate"
        else:
            outcome = "false_positive"
        outcomes.append((class_id, confidence, box, outcome))
    return outcomes


def draw_dashed_box(draw, box, color, width=2, dash=8, gap=5):
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    for start in range(x0, x1, dash + gap):
        draw.line((start, y0, min(start + dash, x1), y0), fill=color, width=width)
        draw.line((start, y1, min(start + dash, x1), y1), fill=color, width=width)
    for start in range(y0, y1, dash + gap):
        draw.line((x0, start, x0, min(start + dash, y1)), fill=color, width=width)
        draw.line((x1, start, x1, min(start + dash, y1)), fill=color, width=width)


def draw_tag(draw, position, text, color, font):
    x, y = position
    bounds = draw.textbbox((x, y), text, font=font)
    draw.rectangle(bounds, fill=(16, 22, 27))
    draw.text((x, y), text, fill=color, font=font)


def render_case(model, dataset_dir, output_dir, case):
    image_path = dataset_dir / "test" / "images" / f"{case['stem']}.jpg"
    label_path = dataset_dir / "test" / "labels" / f"{case['stem']}.txt"
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing curated test image or labels for {case['stem']}")

    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=case["confidence"],
        iou=NMS_IOU,
        max_det=300,
        verbose=False,
    )[0]
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    ground_truth = read_labels(label_path, width, height)
    detections = []
    if result.boxes is not None:
        detections = [
            (int(class_id), float(confidence), box.tolist())
            for class_id, confidence, box in zip(
                result.boxes.cls.cpu(), result.boxes.conf.cpu(), result.boxes.xyxy.cpu()
            )
        ]
    outcomes = classify_predictions(detections, ground_truth)

    canvas = Image.new("RGB", (width, HEADER_HEIGHT + height + FOOTER_HEIGHT), (20, 27, 32))
    canvas.paste(image, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(17)
    label_font = load_font(13)
    small_font = load_font(11)
    draw.text((12, 11), case["title"], fill=(245, 247, 248), font=title_font)

    for _, box in ground_truth:
        shifted = (box[0], box[1] + HEADER_HEIGHT, box[2], box[3] + HEADER_HEIGHT)
        draw_dashed_box(draw, shifted, COLORS["ground_truth"])

    for class_id, confidence, box, outcome in outcomes:
        shifted = (box[0], box[1] + HEADER_HEIGHT, box[2], box[3] + HEADER_HEIGHT)
        color = COLORS[outcome]
        draw.rectangle(shifted, outline=color, width=3)
        draw_tag(
            draw,
            (int(box[0]), min(int(box[3] + HEADER_HEIGHT) + 2, HEADER_HEIGHT + height - 16)),
            f"{model.names[class_id]} {confidence:.3f}",
            color,
            label_font,
        )

    footer = (
        f"GT: cyan dashed  |  Correct: green  |  Identity confusion: amber  |  "
        f"False positive: red  |  Duplicate: purple  |  conf >= {case['confidence']:.3f}"
    )
    draw.text((10, HEADER_HEIGHT + height + 8), footer, fill=(222, 230, 234), font=small_font)
    output_path = output_dir / f"{case['name']}.png"
    canvas.save(output_path)
    return output_path, len(detections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.data.is_dir() or not args.model.is_file():
        parser.error("The detection dataset directory and trained checkpoint must exist")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        parser.error("Install dependencies with `pip install -r requirements.txt` (ultralytics)")
        raise error

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    for case in GALLERY_CASES:
        output_path, detection_count = render_case(model, args.data, args.output, case)
        print(f"{output_path}: {detection_count} predictions; {case['caption']}")


if __name__ == "__main__":
    main()