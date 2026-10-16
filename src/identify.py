"""Milestone 4 live demo: detect and identify celebrities in any image.

YOLOv8 (Milestone 3) finds each face, the 13-identity ResNet18 (Milestone 1 architecture) names it,
and the two votes are combined. Names below 0.5 confidence are marked with '?'.

    python src/identify.py --source photo.jpg [more.jpg ...]   # annotated copies -> results/milestone4/demo/
    python src/identify.py --source some_folder/               # every image in a folder
    python src/identify.py --app                               # web app at http://127.0.0.1:7860
    python src/identify.py --app --share                       # public link (e.g. on Colab)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_detection as m3  # noqa: E402
import m4_pipeline as m4  # noqa: E402

COLUMNS = ["identity", "confidence", "detector says", "classifier says", "found by", "x1", "y1", "x2", "y2"]


def default_mode() -> str:
    s = m4.RESULTS_DIR / "summary.json"
    try:
        return json.loads(s.read_text())["pipeline"]["chosen_mode"]
    except (OSError, KeyError, ValueError):
        return "fusion"


def run(pipe, bgr: np.ndarray, mode: str, conf: float):
    dets = pipe.detect(bgr, mode=mode, conf=conf)
    dets.sort(key=lambda d: (round(d["box"][1] / 60), d["box"][0]))  # reading order
    rows = []
    for d in dets:
        ours = " (Group 02)" if m3.is_our_group(d["name"]) else ""
        rows.append([d["name"] + ours + ("?" if d["score"] < m4.LOW_CONFIDENCE else ""), round(d["score"], 2),
                     f"{d['det_name']} {d['det_conf']:.2f}" if d["det_label"] >= 0 else "-",
                     f"{d['clf_name']} {d['clf_conf']:.2f}", d["source"], *[int(v) for v in d["box"]]])
    return m4.draw_detections(bgr, dets), rows


def images_from(sources: list[Path]) -> list[Path]:
    out = []
    for s in sources:
        if s.is_dir():
            out += sorted(p for p in s.iterdir() if p.suffix.lower() in m3.IMG_EXTS)
        else:
            out.append(s)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, nargs="*", help="image files or folders; default: test grids")
    p.add_argument("--mode", choices=m4.MODES, default=None, help="default: the setting chosen on validation")
    p.add_argument("--conf", type=float, default=m3.DEFAULT_CONF)
    p.add_argument("--out", type=Path, default=m4.RESULTS_DIR / "demo")
    p.add_argument("--app", action="store_true")
    p.add_argument("--share", action="store_true")
    a = p.parse_args()
    mode = a.mode or default_mode()

    for w in (m4.DETECTOR_WEIGHTS, m4.CLASSIFIER_WEIGHTS):
        if not w.is_file():
            sys.exit(f"Missing model: {w}\nTrain it first: python src/run_milestone4.py")
    pipe = m4.Pipeline()
    import cv2

    if a.app:
        import inspect

        import gradio as gr

        def fn(image, conf_thr, mode_choice):
            if image is None:
                return None, []
            annotated, rows = run(pipe, np.ascontiguousarray(image[:, :, ::-1]), mode_choice, conf_thr)
            return annotated[:, :, ::-1], rows

        params = inspect.signature(gr.Interface.__init__).parameters
        no_flag = {"flagging_mode": "never"} if "flagging_mode" in params else {"allow_flagging": "never"}
        examples = [[str(x), a.conf, mode] for x in m3.list_images("test")[:3]]
        gr.Interface(
            fn=fn,
            inputs=[gr.Image(type="numpy", label="Image"),
                    gr.Slider(0.05, 0.95, value=a.conf, step=0.05, label="Detector confidence threshold"),
                    gr.Radio(list(m4.MODES), value=mode, label="Naming mode")],
            outputs=[gr.Image(label="Identified celebrities"), gr.Dataframe(headers=COLUMNS, label="Faces")],
            examples=examples, title="IE7615 Group 02 - celebrity detection + identification",
            description="YOLOv8 finds the faces, a ResNet18 classifier names them (13 CelebA identities). "
                        "Names marked '?' have confidence below 0.5.",
            **no_flag,
        ).launch(share=a.share)
        return

    a.out.mkdir(parents=True, exist_ok=True)
    for src in images_from(a.source or m3.list_images("test")):
        bgr = cv2.imread(str(src))
        if bgr is None:
            print(f"\n{src}: could not read image, skipped")
            continue
        annotated, rows = run(pipe, bgr, mode, a.conf)
        out = a.out / f"{src.stem}_identified.jpg"
        cv2.imwrite(str(out), annotated)
        print(f"\n{src.name}: {len(rows)} face(s), mode={mode} -> {out}")
        for r in rows:
            print(f"  {r[0]:<26} {r[1]:.2f}   detector: {r[2]:<18} classifier: {r[3]:<18} [{r[4]}]")


if __name__ == "__main__":
    main()
