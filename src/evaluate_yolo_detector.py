"""Evaluate a trained YOLO detector and report per-class test metrics."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "detection_dataset" / "dataset.yaml"
DEFAULT_MODEL = (
    REPO_ROOT / "runs" / "detect" / "celeba_yolov8n_milestone2" / "weights" / "best.pt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "detect" / "celeba_yolov8n_milestone2_test"


def box_iou(first, second):
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def read_ground_truth(label_path, image_width, image_height):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id, center_x, center_y, width, height = line.split()
        center_x, width = float(center_x) * image_width, float(width) * image_width
        center_y, height = float(center_y) * image_height, float(height) * image_height
        boxes.append(
            (
                int(class_id),
                (
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                ),
            )
        )
    return boxes


def count_fixed_threshold_matches(
    predictions, dataset_root, split, confidence, match_iou
):
    totals = defaultdict(lambda: {"support": 0, "tp": 0, "fp": 0, "iou_sum": 0.0})
    ground_truth_count = 0

    for prediction in predictions:
        image_path = Path(prediction.path)
        height, width = prediction.orig_shape
        ground_truth = read_ground_truth(
            dataset_root / split / "labels" / f"{image_path.stem}.txt", width, height
        )
        ground_truth_count += len(ground_truth)
        matched_ground_truth = set()
        for class_id, _ in ground_truth:
            totals[class_id]["support"] += 1

        if prediction.boxes is None:
            continue

        detections = sorted(
            zip(
                prediction.boxes.cls.cpu().tolist(),
                prediction.boxes.conf.cpu().tolist(),
                prediction.boxes.xyxy.cpu().tolist(),
            ),
            key=lambda row: row[1],
            reverse=True,
        )
        for class_value, score, predicted_box in detections:
            if score < confidence:
                continue
            class_id = int(class_value)
            candidates = [
                (box_iou(predicted_box, target_box), target_index)
                for target_index, (target_class, target_box) in enumerate(ground_truth)
                if target_class == class_id and target_index not in matched_ground_truth
            ]
            best_iou, best_index = max(candidates, default=(0.0, None))
            if best_index is not None and best_iou >= match_iou:
                matched_ground_truth.add(best_index)
                totals[class_id]["tp"] += 1
                totals[class_id]["iou_sum"] += best_iou
            else:
                totals[class_id]["fp"] += 1

    return totals, ground_truth_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--conf",
        type=float,
        help="Override the confidence threshold selected by validation F1",
    )
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not args.data.is_file() or not args.model.is_file():
        parser.error("Both the dataset YAML and trained model checkpoint must exist")
    if args.imgsz < 1 or args.batch < 1:
        parser.error("imgsz and batch must be positive")
    if args.conf is not None and not 0 <= args.conf <= 1:
        parser.error("conf must be between 0 and 1")
    if not 0 < args.match_iou <= 1:
        parser.error("match-iou must be greater than 0 and at most 1")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        parser.error("Install dependencies with `pip install -r requirements.txt` (ultralytics)")
        raise error

    dataset_root = args.data.resolve().parent
    split_images = dataset_root / args.split / "images"
    image_paths = sorted(
        path
        for path in split_images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        parser.error(f"No images found in {split_images}")

    model = YOLO(str(args.model))
    val_options = {
        "data": str(args.data.resolve()),
        "split": "val",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "plots": False,
        "verbose": False,
    }
    if args.device:
        val_options["device"] = args.device
    validation_metrics = model.val(**val_options)
    if args.conf is None:
        from ultralytics.utils.metrics import smooth

        best_f1_index = smooth(validation_metrics.box.f1_curve.mean(0), 0.1).argmax()
        confidence = float(validation_metrics.box.px[best_f1_index])
    else:
        confidence = args.conf

    test_options = dict(val_options)
    test_options["split"] = args.split
    metrics = model.val(**test_options)

    predict_options = {
        "source": [str(path) for path in image_paths],
        "imgsz": args.imgsz,
        "conf": confidence,
        "verbose": False,
    }
    if args.device:
        predict_options["device"] = args.device
    predictions = model.predict(**predict_options)
    totals, ground_truth_count = count_fixed_threshold_matches(
        predictions, dataset_root, args.split, confidence, args.match_iou
    )

    identity_by_class = {
        class_id: int(identity_id)
        for identity_id, class_id in json.loads(
            (dataset_root / "classes.json").read_text(encoding="utf-8")
        ).items()
    }
    ap_by_class = {
        int(class_id): {
            "ap50": float(metrics.box.ap50[index]),
            "ap50_95": float(metrics.box.ap[index]),
        }
        for index, class_id in enumerate(metrics.box.ap_class_index)
    }
    metric_index_by_class = {
        int(class_id): index for index, class_id in enumerate(metrics.box.ap_class_index)
    }

    rows = []
    for class_id in sorted(totals):
        counts = totals[class_id]
        if counts["support"] == 0:
            continue
        tp, fp, support = counts["tp"], counts["fp"], counts["support"]
        metric_index = metric_index_by_class[class_id]
        class_precision, class_recall, _, _ = metrics.box.class_result(metric_index)
        rows.append(
            {
                "identity_id": identity_by_class[class_id],
                "class_id": class_id,
                "ground_truth_faces": support,
                "precision": float(class_precision),
                "recall": float(class_recall),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": support - tp,
                "precision_at_validation_confidence": tp / (tp + fp) if tp + fp else 0.0,
                "recall_at_validation_confidence": tp / support,
                "mean_iou_of_matches": counts["iou_sum"] / tp if tp else None,
                "ap50": ap_by_class.get(class_id, {}).get("ap50"),
                "ap50_95": ap_by_class.get(class_id, {}).get("ap50_95"),
            }
        )

    true_positives = sum(row["true_positives"] for row in rows)
    false_positives = sum(row["false_positives"] for row in rows)
    matched_iou_sum = sum(totals[row["class_id"]]["iou_sum"] for row in rows)
    summary = {
        "split": args.split,
        "images": len(image_paths),
        "ground_truth_faces": ground_truth_count,
        "confidence_threshold": confidence,
        "iou_match_threshold": args.match_iou,
        "precision": float(metrics.results_dict["metrics/precision(B)"]),
        "recall": float(metrics.results_dict["metrics/recall(B)"]),
        "precision_at_validation_confidence": true_positives
        / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0,
        "recall_at_validation_confidence": true_positives / ground_truth_count
        if ground_truth_count
        else 0.0,
        "mean_iou_of_matches": matched_iou_sum / true_positives if true_positives else None,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.split}_per_class_metrics.csv"
    json_path = args.output_dir / f"{args.split}_detection_metrics.json"
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({key: value for key, value in summary.items() if key != "per_class"}, indent=2))
    print(f"Per-class table: {csv_path}")
    print(f"Full report: {json_path}")


if __name__ == "__main__":
    main()