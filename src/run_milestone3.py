"""Milestone 3 end to end: scaled dataset, Milestone 2 reference run, YOLOv8
fine-tuning sweep, test evaluation, threshold sweep, gallery and results document.

One command (from the repo root, GPU strongly recommended):
    python src/run_milestone3.py
Quick smoke test (2 epochs per run, finishes on CPU in a few minutes):
    python src/run_milestone3.py --epochs 2

notebooks/06_yolov8_training_evaluation.ipynb runs the same steps cell by cell.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_detection as m3  # noqa: E402


def step_build_dataset(rebuild: bool = False) -> Path:
    """Build detection_dataset_m3/ (deterministic, seed 42) unless it already exists; return its data.yaml."""
    from build_m3_scaled_dataset import build

    if rebuild or not (m3.SCALED_DATASET_DIR / "test" / "images").is_dir():
        build({"train": 120, "val": 12, "test": 12}, seed=42, out_dir=m3.SCALED_DATASET_DIR)
    m3.DATASET_DIR = m3.SCALED_DATASET_DIR
    return m3.write_data_yaml(m3.SCALED_DATASET_DIR)


def step_m2_reference(epochs: int, patience: int, batch: int, device=None) -> dict:
    """Baseline configuration trained and tested on the original Milestone 2 dataset, for comparison."""
    m2_yaml = m3.write_data_yaml(m3.M2_DATASET_DIR)
    cfg = m3.RunConfig("m2_original_baseline", "Baseline config on the original Milestone 2 dataset")
    run_dir = m3.RUNS_DIR / cfg.name
    if not (run_dir / "weights" / "best.pt").is_file():
        print("\n=== Training baseline on the original Milestone 2 dataset (reference) ===")
        run_dir = m3.train_run(cfg, m2_yaml, epochs=epochs, patience=patience, batch=batch, device=device)
    m3.save_run_logs(run_dir, m3.RESULTS_DIR / "train_logs" / cfg.name)
    best = run_dir / "weights" / "best.pt"
    val = m3.ultralytics_metrics(best, m2_yaml, "val", name=f"{cfg.name}_val")
    test = m3.ultralytics_metrics(best, m2_yaml, "test", name=f"{cfg.name}_test")
    from ultralytics import YOLO

    op = m3.summarize(m3.predict_and_match(YOLO(str(best)), m3.list_images("test", m3.M2_DATASET_DIR)))
    row = {"dataset": "Milestone 2 original (20/5/5 images)",
           "epochs_run": int(m3.read_results_csv(run_dir)["epoch"].max()),
           "val mAP50": val["mAP50"], "val mAP50-95": val["mAP50-95"],
           "test mAP50": test["mAP50"], "test mAP50-95": test["mAP50-95"],
           "test precision": test["precision"], "test recall": test["recall"],
           "test mean IoU": op["mean_iou"], "faces correct": f"{op['correct']}/{op['faces']}"}
    m3.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (m3.RESULTS_DIR / "m2_original_reference.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def step_train_sweep(data_yaml: Path, epochs: int, patience: int, batch: int, device=None,
                     only: list[str] | None = None, skip_existing: bool = True) -> dict[str, Path]:
    """Train every sweep configuration; return {run name: run folder}.

    With skip_existing, a run whose folder already holds weights/best.pt is reused,
    so an interrupted sweep can be resumed without retraining finished runs.
    """
    run_dirs = {}
    for cfg in m3.SWEEP_CONFIGS:
        if only and cfg.name not in only:
            continue
        existing = m3.RUNS_DIR / cfg.name
        if skip_existing and (existing / "weights" / "best.pt").is_file() and (existing / "results.csv").is_file():
            print(f"=== {cfg.name}: already trained, reusing {existing} ===")
            run_dirs[cfg.name] = existing
            m3.save_run_logs(existing, m3.RESULTS_DIR / "train_logs" / cfg.name)
            continue
        print(f"\n=== Training {cfg.name}: {cfg.description} ===")
        run_dirs[cfg.name] = m3.train_run(cfg, data_yaml, epochs=epochs, patience=patience, batch=batch,
                                          device=device)
        m3.save_run_logs(run_dirs[cfg.name], m3.RESULTS_DIR / "train_logs" / cfg.name)
    return run_dirs


def step_compare_runs(run_dirs: dict[str, Path], data_yaml: Path):
    """Score each run on val (used for selection) and test (reported); pick the best on val."""
    import pandas as pd

    cfgs = {c.name: c for c in m3.SWEEP_CONFIGS}
    rows = []
    for name, run_dir in run_dirs.items():
        best = run_dir / "weights" / "best.pt"
        imgsz = cfgs[name].imgsz
        val = m3.ultralytics_metrics(best, data_yaml, "val", imgsz=imgsz, name=f"{name}_val")
        test = m3.ultralytics_metrics(best, data_yaml, "test", imgsz=imgsz, name=f"{name}_test")
        log = m3.read_results_csv(run_dir)
        rows.append({**cfgs[name].as_row(), "epochs_run": int(log["epoch"].max()),
                     "val mAP50": val["mAP50"], "val mAP50-95": val["mAP50-95"],
                     "test mAP50": test["mAP50"], "test mAP50-95": test["mAP50-95"],
                     "test precision": test["precision"], "test recall": test["recall"]})
    df = pd.DataFrame(rows)
    # Select on validation only so the test split stays a clean held-out estimate.
    best_name = df.sort_values(["val mAP50-95", "val mAP50"], ascending=False).iloc[0]["run"]
    m3.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(m3.RESULTS_DIR / "training_sweep.csv", index=False)
    m3.plot_sweep_curves(run_dirs, m3.RESULTS_DIR / "training_sweep_curves.png")
    return df, best_name


def step_final_model(run_dirs: dict[str, Path], best_name: str, data_yaml: Path) -> dict:
    """Copy the selected checkpoint to models/, plot its curves, and compute the full test report."""
    cfg = {c.name: c for c in m3.SWEEP_CONFIGS}[best_name]
    m3.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dirs[best_name] / "weights" / "best.pt", m3.FINAL_WEIGHTS)
    m3.plot_loss_curves(run_dirs[best_name], m3.RESULTS_DIR / "loss_curves.png",
                        f"Training / validation curves - selected run '{best_name}'")

    names = m3.load_class_names()
    test = m3.ultralytics_metrics(m3.FINAL_WEIGHTS, data_yaml, "test", imgsz=cfg.imgsz, name="final_test")
    from ultralytics import YOLO

    matched = m3.predict_and_match(YOLO(str(m3.FINAL_WEIGHTS)), m3.list_images("test"), imgsz=cfg.imgsz)
    op = m3.summarize(matched)
    iou_by_class = m3.per_class_iou(matched, names)
    per_class = []
    for cid, name in names.items():
        u = test["per_class"].get(cid)
        per_class.append({
            "class": cid, "identity": name + (" (ours)" if m3.is_our_group(name) else ""),
            "test faces": iou_by_class[cid]["faces"],
            "precision": u["precision"] if u else float("nan"),
            "recall": u["recall"] if u else float("nan"),
            "mAP50": u["mAP50"] if u else float("nan"),
            "mAP50-95": u["mAP50-95"] if u else float("nan"),
            "mean IoU": iou_by_class[cid]["mean_iou"],
        })
    import pandas as pd

    pd.DataFrame(per_class).to_csv(m3.RESULTS_DIR / "test_metrics_per_class.csv", index=False)
    report = {
        "selected_run": best_name, "config": cfg.as_row(), "weights": str(m3.FINAL_WEIGHTS.relative_to(m3.REPO_ROOT)),
        "test_overall": {k: test[k] for k in ("precision", "recall", "mAP50", "mAP50-95")},
        "test_operating_point": {"conf": m3.DEFAULT_CONF, "nms_iou": m3.DEFAULT_NMS_IOU, **op},
    }
    (m3.RESULTS_DIR / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2(run_dirs[best_name] / "args.yaml", m3.RESULTS_DIR / "final_model_hyperparameters.yaml")
    return report


def step_threshold_sweep(imgsz: int):
    conf_df, nms_df = m3.threshold_sweep(m3.FINAL_WEIGHTS, m3.list_images("test"), imgsz=imgsz)
    conf_df.to_csv(m3.RESULTS_DIR / "threshold_sweep_conf.csv", index=False)
    nms_df.to_csv(m3.RESULTS_DIR / "threshold_sweep_nms.csv", index=False)
    m3.plot_threshold_sweep(conf_df, nms_df, m3.RESULTS_DIR / "threshold_sweep.png")
    return conf_df, nms_df


def step_gallery(imgsz: int):
    return m3.build_gallery(m3.FINAL_WEIGHTS, m3.load_class_names(), m3.RESULTS_DIR / "gallery", imgsz=imgsz)


def step_write_doc(report: dict, sweep_df, conf_df, nms_df, gallery_df, epochs: int, m2_ref: dict | None = None) -> Path:
    """Write docs/milestone3_detection_results.md: ~4-page main report plus an appendix of supporting tables."""
    import pandas as pd

    rel = "../results/milestone3"
    t, op = report["test_overall"], report["test_operating_point"]
    sel = report["selected_run"]
    per_class = pd.read_csv(m3.RESULTS_DIR / "test_metrics_per_class.csv")
    splits = m3.split_summary(m3.SCALED_DATASET_DIR)
    n = {r["split"]: int(r["images"]) for _, r in splits.iterrows()}
    grids = f"{n['train']}/{n['val']}/{n['test']} grids"
    sv = sweep_df.set_index("run")
    pc = per_class.dropna(subset=["mAP50"])

    # Comparison rows: original M2 data vs scaled data with the same baseline config, plus the selected run.
    compare_rows = []
    if m2_ref:
        compare_rows.append({"model / data": "baseline, Milestone 2 data (20/5/5 images)",
                             "test mAP50": m2_ref["test mAP50"], "test mAP50-95": m2_ref["test mAP50-95"],
                             "precision": m2_ref["test precision"], "recall": m2_ref["test recall"]})
    if "baseline" in sv.index:
        compare_rows.append({"model / data": f"baseline, scaled data ({grids})",
                             "test mAP50": sv.loc["baseline", "test mAP50"], "test mAP50-95": sv.loc["baseline", "test mAP50-95"],
                             "precision": sv.loc["baseline", "test precision"], "recall": sv.loc["baseline", "test recall"]})
    if sel != "baseline":
        compare_rows.append({"model / data": f"{sel} (selected), scaled data", "test mAP50": t["mAP50"],
                             "test mAP50-95": t["mAP50-95"], "precision": t["precision"], "recall": t["recall"]})
    compare = pd.DataFrame(compare_rows)
    sweep_view = sweep_df[["run", "model", "lr0", "imgsz", "epochs_run", "val mAP50", "val mAP50-95",
                           "test mAP50", "test mAP50-95"]] if "model" in sweep_df else sweep_df
    conf_view = conf_df[["conf", "predictions", "correct", "wrong_identity", "false_positive", "missed",
                         "precision", "recall", "f1"]]
    nms_view = nms_df[["nms_iou", "predictions", "correct", "false_positive", "precision", "recall", "f1"]]
    aug_view = sweep_df[["run", "augmentation", "what changes"]]
    best_conf = conf_df.loc[conf_df["f1"].idxmax()]

    # Gallery and stress-test summaries.
    grades = gallery_df["caption"].str.extract(r"^\*\*(\w+)")[0]
    test_grades = grades[gallery_df["source"] == "Test split"].value_counts()
    n_test_grids = int((gallery_df["source"] == "Test split").sum())
    stress = gallery_df[gallery_df["source"] != "Test split"]
    kind = stress["source"].str.extract(r"\((.*)\)")[0]
    st = {k: (int(g["correct"].sum()), int(g["faces"].sum())) for k, g in stress.groupby(kind)}
    stress_txt = "; ".join(f"{k}: {c}/{f} faces correct" for k, (c, f) in sorted(st.items(), key=lambda kv: kv[1][0] / kv[1][1]))

    # Training-curve facts for the selected run.
    log = m3.read_results_csv(m3.RESULTS_DIR / "train_logs" / sel)
    cls_train, cls_val = log["train/cls_loss"].iloc[-1], log["val/cls_loss"].iloc[-1]
    photos_per_id = round(pd.read_csv(m3.SCALED_DATASET_DIR / "photo_split.csv").query("split == 'train'")
                          .groupby("identity_id").size().mean())

    # Sentences that depend on what the sweep actually showed.
    iou_txt = ("That is why mAP@0.5 and mAP@0.5:0.95 are almost identical: nearly every correct detection also "
               "passes the strictest IoU threshold. This is partly a property of our synthetic grids, where each "
               "face is a sharp-edged square crop on a plain background; real group photos would be harder to localize."
               if abs(t["mAP50"] - t["mAP50-95"]) < 0.02 else
               f"The gap between mAP@0.5 ({m3.fmt(t['mAP50'])}) and mAP@0.5:0.95 ({m3.fmt(t['mAP50-95'])}) shows "
               "the remaining error in box tightness.")
    hp = []
    if {"lr_high", "baseline"} <= set(sv.index):
        if sv.loc["lr_high", "val mAP50"] < 0.5 * sv.loc["baseline", "val mAP50"]:
            hp.append(f"The learning rate had the largest effect: a 5x higher rate ({sv.loc['lr_high', 'lr0']}) diverged "
                      f"and stopped at epoch {int(sv.loc['lr_high', 'epochs_run'])} with val mAP@0.5 = "
                      f"{m3.fmt(sv.loc['lr_high', 'val mAP50'])}.")
        else:
            hp.append(f"A 5x higher learning rate reached val mAP@0.5 = {m3.fmt(sv.loc['lr_high', 'val mAP50'])} "
                      f"(baseline {m3.fmt(sv.loc['baseline', 'val mAP50'])}).")
    if "lr_low" in sv.index:
        hp.append(f"A 5x lower rate ({sv.loc['lr_low', 'lr0']}) learned more slowly "
                  f"(val mAP@0.5 = {m3.fmt(sv.loc['lr_low', 'val mAP50'])}).")
    if "imgsz_416" in sv.index:
        hp.append(f"Shrinking the input to 416 px gave val mAP@0.5 = {m3.fmt(sv.loc['imgsz_416', 'val mAP50'])}, "
                  "since faces become smaller.")
    if {"aug_light", "baseline"} <= set(sv.index):
        hp.append(f"Lighter online augmentation reached val mAP@0.5:0.95 = {m3.fmt(sv.loc['aug_light', 'val mAP50-95'])} "
                  f"vs {m3.fmt(sv.loc['baseline', 'val mAP50-95'])} for the baseline, consistent with the grids "
                  "already being varied by construction.")
    if {"yolov8s_aug_light", "aug_light"} <= set(sv.index):
        d = sv.loc["yolov8s_aug_light", "val mAP50-95"] - sv.loc["aug_light", "val mAP50-95"]
        hp.append(f"The larger YOLOv8s model {'improved' if d > 0.01 else 'did not clearly improve'} on the same "
                  f"settings (val mAP@0.5:0.95 {m3.fmt(sv.loc['yolov8s_aug_light', 'val mAP50-95'])} vs "
                  f"{m3.fmt(sv.loc['aug_light', 'val mAP50-95'])} for YOLOv8n)"
                  + (", suggesting capacity was part of the identity bottleneck." if d > 0.01 else
                     ", suggesting the bottleneck is the number of distinct photos rather than model size."))
    test_best = sweep_df.loc[sweep_df["test mAP50-95"].idxmax(), "run"]
    if test_best != sel:
        gap = sv.loc[test_best, "test mAP50"] - sv.loc[sel, "test mAP50"]
        hp.append(f"On the test split `{test_best}` scored "
                  + (f"clearly higher (mAP@0.5 = {m3.fmt(sv.loc[test_best, 'test mAP50'])} vs {m3.fmt(sv.loc[sel, 'test mAP50'])}). "
                     "We keep the model chosen on validation, because picking the test winner would make the test score "
                     f"an optimistic estimate; `{test_best}` is the first candidate to re-evaluate in Milestone 4."
                     if gap > 0.03 else
                     f"slightly higher than the selected `{sel}`; with only {n['test']} test grids, differences this "
                     "small are within noise, so we keep the validation choice."))
    nms_flat = nms_df["f1"].max() - nms_df["f1"].min() < 0.02
    worst = pc.loc[pc["mAP50"].idxmin()]

    # Wrong names at the operating point, counted from the test-split gallery captions.
    import re
    from collections import Counter
    wrong_by_id, pairs, given = Counter(), Counter(), {}
    for cap in gallery_df.loc[gallery_df["source"] == "Test split", "caption"]:
        for true_id, pred_id in re.findall(r"(celeb_\d+) predicted as (celeb_\d+)", cap):
            wrong_by_id[true_id] += 1
            pairs[tuple(sorted((true_id, pred_id)))] += 1
            given.setdefault(true_id, set()).add(pred_id)
    pc_plain = pc.assign(name=pc["identity"].str.replace(" (ours)", "", regex=False)).set_index("name")
    faces_of = pc_plain["test faces"].astype(int).to_dict()
    hardest = [k for k, _ in wrong_by_id.most_common(3)]
    hardest_txt = ", ".join(f"{k} ({wrong_by_id[k]} of {faces_of[k]} test faces misnamed)" for k in hardest)
    pair_txt = ", ".join(f"{a} vs {b}" for (a, b), _ in pairs.most_common(3))
    # An identity with near-perfect mAP but many wrong names illustrates mAP vs top-1 accuracy.
    high_map = [k for k in pc_plain.index if pc_plain.loc[k, "mAP50"] >= 0.95 and wrong_by_id[k] > 0]
    example = max(high_map, key=lambda k: wrong_by_id[k]) if high_map else None
    example_txt = ""
    if example:
        others = sorted(given[example], key=lambda k: int(k.split("_")[1]))
        example_txt = (f" For example, {example} has mAP@0.5 = {m3.fmt(pc_plain.loc[example, 'mAP50'])}, yet at our "
                       f"operating point {wrong_by_id[example]} of its {faces_of[example]} test faces are given another "
                       f"name ({' or '.join(others)}; see the gallery).")
    ours_ids = [k for k in pc_plain.index if m3.is_our_group(k)]
    ours_txt = "; ".join(f"{k} named correctly on {faces_of[k] - wrong_by_id[k]} of {faces_of[k]} test faces"
                         for k in ours_ids)
    op_f1 = 2 * op["precision"] * op["recall"] / (op["precision"] + op["recall"])
    acc_pct = 100 * op["correct"] / op["faces"]

    doc = f"""# Project 1 - Milestone 3: YOLOv8 Transfer Learning and Evaluation

IE7615 - Group 02

## 1. Setup

We fine-tuned COCO-pretrained YOLOv8 models (Ultralytics) to find every face in a synthetic 3x3 grid of
CelebA face crops and name it as one of the 13 class-pool identities (one YOLO class per identity).

**Data.** A first run on our submitted Milestone 2 dataset (6 base grids, 30 images) learned where faces
are but not who they are: each identity appeared in only about 4 of its ~25 photos, and the test split
missed 4 identities. We rebuilt the dataset with the **same Milestone 2 method** (same identities and class
ids, 640 px 3x3 grids, same scale/placement jitter; `src/build_m3_scaled_dataset.py`), changing only its
size and split: each identity's photos are divided 70/15/15 into train/val/test **before** any grid is
built, so no photo appears in two splits, and every identity appears in every split. The result has
{n['train']} train, {n['val']} val and {n['test']} test grids ({n['test'] * 9} test faces), about
{photos_per_id} distinct training photos per identity.

**Training.** AdamW set explicitly (Ultralytics' default `optimizer=auto` ignores `lr0`, so the
learning-rate sweep would otherwise have no effect), up to {epochs} epochs with early stopping (patience 30),
batch 16, seed 42. {len(sweep_df)} configurations were trained (section 3) and the final model was **selected on
validation mAP@0.5:0.95 only**; the test split was used once for reporting.
Selected run: **`{sel}`**: {report['config']['what changes'][0].lower() + report['config']['what changes'][1:]}; weights in
`{report['weights']}`.

## 2. Test-set results

| Metric | Value |
|---|---|
| mAP@0.5 | {m3.fmt(t['mAP50'])} |
| mAP@0.5:0.95 | {m3.fmt(t['mAP50-95'])} |
| Precision (Ultralytics) | {m3.fmt(t['precision'])} |
| Recall (Ultralytics) | {m3.fmt(t['recall'])} |
| Mean IoU (correct detections) | {m3.fmt(op['mean_iou'])} |
| Faces located (any identity) | {op['correct'] + op['wrong_identity']} / {op['faces']} |
| **Faces correctly identified** | **{op['correct']} / {op['faces']} ({acc_pct:.1f}%)** |

mAP, precision and recall are Ultralytics' validator metrics. IoU comes from our matcher at confidence
{op['conf']} and NMS IoU {op['nms_iou']}: each prediction claims the unclaimed ground-truth face it overlaps
most (IoU >= 0.5), and IoU is averaged over correctly identified faces. Per-identity results are in
Appendix A.

**How to read these numbers.** The table holds two kinds of metric, and they answer different questions.

- *mAP* is a ranking metric. Ultralytics computes it over all predictions down to a very low confidence
  (0.001), so an identity can score a high AP even when the correct name is not the model's top answer
  at a usable threshold.{example_txt}
- *Precision/recall* differ between the two sources. Ultralytics reports them at the single confidence
  that maximizes mean F1 across classes, giving {m3.fmt(t['precision'])} / {m3.fmt(t['recall'])}. Our matcher, at a fixed confidence of {op['conf']},
  counts {op['correct']} correct, {op['wrong_identity']} wrong-identity and {op['false_positive']} false-positive boxes: precision {m3.fmt(op['precision'])}, recall {m3.fmt(op['recall'])}, F1 {m3.fmt(op_f1)}.

We therefore treat **{op['correct']} of {op['faces']} faces correctly identified** as the headline accuracy and use mAP mainly
to compare runs.

**Effect of the dataset** (each row tested on its own dataset's held-out split):

{m3.md_table(compare)}

## 3. Hyperparameter sweep

Training-time settings (one change at a time from the baseline; augmentation details in Appendix B):

{m3.md_table(sweep_view)}

![Validation curves across sweep runs]({rel}/training_sweep_curves.png)

Inference-time settings (confidence threshold and NMS IoU threshold, test split; tables in Appendix C):

![Threshold sweep]({rel}/threshold_sweep.png)

## 4. Training and validation curves (selected run)

![Loss curves]({rel}/loss_curves.png)

## 5. Interpretation

**Localization is solved; identification is the bottleneck.** The detector places a box on
{m3.fmt(op['face_localization_rate'] * 100, 0)}% of test faces, and those boxes are almost exact (mean IoU
{m3.fmt(op['mean_iou'])}). {iou_txt} The remaining errors are mostly identity errors: {op['wrong_identity']}
faces were found but given the wrong name, against {op['missed']} missed faces and {op['false_positive']}
false positive(s). The curves show the same pattern: box loss converges for training and validation alike,
while validation classification loss levels off at {m3.fmt(cls_val, 2)} against {m3.fmt(cls_train, 2)} for
training, a sign of overfitting to about {photos_per_id} photos per identity. Accuracy varies by identity,
from mAP@0.5 = {m3.fmt(worst['mAP50'])} ({worst['identity']}) up to {m3.fmt(pc['mAP50'].max())}. Counting wrong names directly at
the operating point gives a clearer picture: the hardest identities are {hardest_txt}, and the
errors cluster in a few look-alike pairs ({pair_txt}). Group 02's identities: {ours_txt}.

**Data mattered most.** With the same baseline configuration, the original 30-image Milestone 2 dataset
reaches test mAP@0.5 = {m3.fmt(compare['test mAP50'].iloc[0])}, while the scaled dataset reaches
{m3.fmt(compare['test mAP50'].iloc[1]) if len(compare) > 1 else 'n/a'}. Because test photos never appear in
training, these scores measure recognizing known celebrities in new photos, which is what the final system
must do.

**Hyperparameters.** {' '.join(hp)} At inference, raising the confidence threshold trades recall for
precision ({m3.fmt(conf_df['precision'].iloc[0])}/{m3.fmt(conf_df['recall'].iloc[0])} at
{m3.fmt(conf_df['conf'].iloc[0], 2)} to {m3.fmt(conf_df['precision'].iloc[-1])}/{m3.fmt(conf_df['recall'].iloc[-1])}
at {m3.fmt(conf_df['conf'].iloc[-1], 2)}), with the best F1 at {m3.fmt(best_conf['conf'], 2)}.
{'The NMS IoU threshold had no effect, because faces in a grid never overlap.' if nms_flat else 'Low NMS thresholds suppressed neighbouring faces, high ones kept duplicate boxes.'}

**Limits and next steps.** Of the {n_test_grids} test grids, {int(test_grades.get('Success', 0))} are
near-perfect (at least 8 of 9 faces correct), {int(test_grades.get('Partial', 0))} partial and
{int(test_grades.get('Failure', 0))} failures. Stress tests show where the model breaks ({stress_txt}):
training faces always fill 65-92% of a grid cell, so much smaller faces are outside what it learned. For
Milestone 4 we plan to add smaller-scale and crowded compositions to training and to compare this
single-stage detector with a two-stage pipeline (face detection followed by the Milestone 1 classifier),
which targets the identity errors directly.

## 6. Gallery and demo

- Gallery: `results/milestone3/gallery/README.md` ({len(gallery_df)} annotated multi-face images graded
  success / partial / failure, with captions).
- Live demo: `python src/demo_detect.py --app` (upload app) or `python src/demo_detect.py --source <image>`;
  instructions in the README.

---

## Appendix A. Per-identity test results

"(ours)" marks Group 02's identities.

{m3.md_table(per_class)}

## Appendix B. Sweep configurations

{m3.md_table(aug_view)}

## Appendix C. Inference-threshold tables (test split)

{m3.md_table(conf_view)}

{m3.md_table(nms_view)}

Per-run logs (`results.csv`, `args.yaml`, Ultralytics plots) are in `results/milestone3/train_logs/`.
"""
    m3.DOC_PATH.write_text(doc, encoding="utf-8")
    return m3.DOC_PATH


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="e.g. 0 for the first GPU, or cpu")
    p.add_argument("--only", nargs="*", help="train only these sweep runs, e.g. --only baseline")
    p.add_argument("--retrain", action="store_true", help="retrain runs that already have weights")
    args = p.parse_args()

    data_yaml = step_build_dataset()
    print(m3.split_summary().to_string(index=False))
    m2_ref = step_m2_reference(args.epochs, args.patience, args.batch, args.device)
    run_dirs = step_train_sweep(data_yaml, args.epochs, args.patience, args.batch, args.device, args.only,
                                skip_existing=not args.retrain)
    sweep_df, best_name = step_compare_runs(run_dirs, data_yaml)
    print(sweep_df.to_string(index=False))
    report = step_final_model(run_dirs, best_name, data_yaml)
    print(json.dumps(report, indent=2))
    imgsz = report["config"]["imgsz"]
    conf_df, nms_df = step_threshold_sweep(imgsz)
    gallery_df = step_gallery(imgsz)
    doc = step_write_doc(report, sweep_df, conf_df, nms_df, gallery_df, args.epochs, m2_ref)
    print(f"\nDone. Model: {m3.FINAL_WEIGHTS}\nResults: {m3.RESULTS_DIR}\nDocument: {doc}")


if __name__ == "__main__":
    main()
