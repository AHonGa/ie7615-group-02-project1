"""Run a small validation-only sweep of YOLO confidence and NMS IoU thresholds."""
import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "detection_dataset" / "dataset.yaml"
DEFAULT_MODEL = (
    REPO_ROOT / "runs" / "detect" / "celeba_yolov8n_milestone2" / "weights" / "best.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "detect" / "celeba_yolov8n_milestone2_sweep.csv"
DEFAULT_CONFIDENCE_THRESHOLDS = [0.001, 0.01, 0.05, 0.25]
DEFAULT_NMS_IOU_THRESHOLDS = [0.50, 0.70]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument(
        "--conf",
        type=float,
        nargs="+",
        default=DEFAULT_CONFIDENCE_THRESHOLDS,
        help="Confidence thresholds to compare",
    )
    parser.add_argument(
        "--iou",
        type=float,
        nargs="+",
        default=DEFAULT_NMS_IOU_THRESHOLDS,
        help="NMS IoU thresholds to compare",
    )
    args = parser.parse_args()

    if not args.data.is_file() or not args.model.is_file():
        parser.error("Both the dataset YAML and trained model checkpoint must exist")
    if args.imgsz < 1 or args.batch < 1:
        parser.error("imgsz and batch must be positive")
    if any(not 0 <= value <= 1 for value in args.conf + args.iou):
        parser.error("confidence and NMS IoU thresholds must be between 0 and 1")

    validation_images = args.data.resolve().parent / "val" / "images"
    image_count = sum(
        path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for path in validation_images.iterdir()
        if path.is_file()
    )
    if image_count == 0:
        parser.error(f"No validation images found in {validation_images}")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        parser.error("Install dependencies with `pip install -r requirements.txt` (ultralytics)")
        raise error

    model = YOLO(str(args.model))
    rows = []
    for confidence in args.conf:
        for nms_iou in args.iou:
            options = {
                "data": str(args.data.resolve()),
                "split": "val",
                "imgsz": args.imgsz,
                "batch": args.batch,
                "workers": args.workers,
                "conf": confidence,
                "iou": nms_iou,
                "plots": False,
                "verbose": False,
                "project": str(args.output.parent / "sweep_runs"),
                "name": "validation",
                "exist_ok": True,
            }
            if args.device:
                options["device"] = args.device
            metrics = model.val(**options)
            precision = float(metrics.results_dict["metrics/precision(B)"])
            recall = float(metrics.results_dict["metrics/recall(B)"])
            rows.append(
                {
                    "split": "val",
                    "images": image_count,
                    "confidence": confidence,
                    "nms_iou": nms_iou,
                    "precision": precision,
                    "recall": recall,
                    "f1": 2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0,
                    "map50": float(metrics.box.map50),
                    "map50_95": float(metrics.box.map),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("conf  nms_iou  precision  recall  F1     mAP50  mAP50-95")
    for row in rows:
        print(
            f"{row['confidence']:<5g} {row['nms_iou']:<8g} "
            f"{row['precision']:<10.3f} {row['recall']:<7.3f} "
            f"{row['f1']:<6.3f} {row['map50']:<6.3f} {row['map50_95']:.3f}"
        )
    print(f"Sweep results: {args.output}")


if __name__ == "__main__":
    main()