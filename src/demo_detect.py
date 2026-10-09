"""Milestone 3 live demo: run the fine-tuned YOLOv8 celebrity detector on images.

Command line (annotated copies saved to results/milestone3/demo/):
    python src/demo_detect.py                                # all held-out test images
    python src/demo_detect.py --source path/to/photo.jpg     # your own image(s)

Web app (upload an image, adjust thresholds):
    python src/demo_detect.py --app            # opens http://127.0.0.1:7860
    python src/demo_detect.py --app --share    # public link, needed on Colab
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_detection as m3  # noqa: E402


def load_model(weights: Path):
    from ultralytics import YOLO

    if not weights.is_file():
        sys.exit(f"Model not found: {weights}\nTrain it first: python src/run_milestone3.py")
    return YOLO(str(weights))


def detect(model, image, conf: float, iou: float, imgsz: int = 640):
    """Return (annotated RGB image, list of detection rows) for a path or an RGB numpy array."""
    # Ultralytics expects BGR arrays; copy so the reversed view is contiguous.
    source = image if isinstance(image, (str, Path)) else np.ascontiguousarray(image[:, :, ::-1])
    r = model.predict(source=str(source) if isinstance(source, Path) else source, conf=conf, iou=iou,
                      imgsz=imgsz, verbose=False)[0]
    rows = []
    for cls, score, box in zip(m3._np(r.boxes.cls), m3._np(r.boxes.conf), m3._np(r.boxes.xyxy)):
        name = model.names[int(cls)]
        rows.append([name + (" (Group 02)" if m3.is_our_group(name) else ""), round(float(score), 3),
                     *[int(v) for v in box]])
    rows.sort(key=lambda row: (row[3], row[2]))  # reading order: top-to-bottom, left-to-right
    return r.plot()[:, :, ::-1], rows


def run_cli(model, sources: list[Path], conf: float, iou: float, out_dir: Path) -> None:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sources:
        annotated, rows = detect(model, src, conf, iou)
        out = out_dir / f"{src.stem}_detected.jpg"
        Image.fromarray(annotated).save(out)
        print(f"\n{src.name}: {len(rows)} face(s) -> {out}")
        for name, score, *_ in rows:
            print(f"  {name:<24} {score:.2f}")


def run_app(model, conf: float, iou: float, share: bool) -> None:
    import gradio as gr

    headers = ["identity", "confidence", "x1", "y1", "x2", "y2"]

    def fn(image, conf_thr, iou_thr):
        if image is None:
            return None, []
        return detect(model, image, conf_thr, iou_thr)

    import inspect

    examples = [[str(p), conf, iou] for p in m3.list_images("test")[:3]]
    params = inspect.signature(gr.Interface.__init__).parameters
    no_flag = {"flagging_mode": "never"} if "flagging_mode" in params else {"allow_flagging": "never"}
    gr.Interface(
        fn=fn,
        inputs=[gr.Image(type="numpy", label="Image with faces"),
                gr.Slider(0.05, 0.95, value=conf, step=0.05, label="Confidence threshold"),
                gr.Slider(0.1, 0.9, value=iou, step=0.05, label="NMS IoU threshold")],
        outputs=[gr.Image(label="Detections"), gr.Dataframe(headers=headers, label="Detected identities")],
        examples=examples,
        title="IE7615 Group 02 - YOLOv8 celebrity detector",
        description="Fine-tuned YOLOv8n that locates faces and names one of 13 CelebA identities. "
                    "Try a held-out test grid below or upload your own image.",
        **no_flag,
    ).launch(share=share)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=m3.FINAL_WEIGHTS)
    p.add_argument("--source", type=Path, nargs="*", help="image file(s); default: the test split")
    p.add_argument("--conf", type=float, default=m3.DEFAULT_CONF)
    p.add_argument("--iou", type=float, default=m3.DEFAULT_NMS_IOU)
    p.add_argument("--out", type=Path, default=m3.RESULTS_DIR / "demo")
    p.add_argument("--app", action="store_true", help="launch the Gradio web app")
    p.add_argument("--share", action="store_true", help="public Gradio link (use on Colab)")
    args = p.parse_args()

    model = load_model(args.weights)
    if args.app:
        run_app(model, args.conf, args.iou, args.share)
    else:
        run_cli(model, args.source or m3.list_images("test"), args.conf, args.iou, args.out)


if __name__ == "__main__":
    main()
