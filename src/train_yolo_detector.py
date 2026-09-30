"""Fine-tune a pretrained YOLOv8 detector on the synthetic CelebA frames."""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data" / "synthetic_face_frames_70_15_15" / "dataset.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained YOLOv8 checkpoint")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", help="Ultralytics device, for example 0 or cpu")
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "detect")
    parser.add_argument("--name", default="celeba_yolov8n")
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"Dataset YAML does not exist: {args.data}")
    if args.epochs < 1 or args.patience < 1 or args.imgsz < 1 or args.batch < 1:
        parser.error("epochs, patience, imgsz, and batch must be positive")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        parser.error("Install dependencies with `pip install -r requirements.txt` (ultralytics)")
        raise error

    model = YOLO(args.model)
    train_options = {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "plots": True,
        "seed": 42,
    }
    if args.device:
        train_options["device"] = args.device

    model.train(**train_options)
    run_dir = Path(model.trainer.save_dir)
    best_checkpoint = run_dir / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Training did not produce the best checkpoint: {best_checkpoint}")

    test_model = YOLO(str(best_checkpoint))
    test_metrics = test_model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        plots=True,
        project=str(args.project),
        name=f"{args.name}_test",
    )
    metrics_path = run_dir / "test_metrics.json"
    metrics_path.write_text(
        json.dumps({key: float(value) for key, value in test_metrics.results_dict.items()}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Training curves and metrics: {run_dir / 'results.png'} and {run_dir / 'results.csv'}")
    print(f"Held-out test metrics: {metrics_path}")


if __name__ == "__main__":
    main()