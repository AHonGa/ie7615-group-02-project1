# Project 1 — Celebrity Identification and Detection

IE/EI 7615 — Deep Learning Architectures, Group 02. A discriminative computer-vision
pipeline built in milestones:

- **Milestone 1** — single-face classification on a CelebA identity subset (custom CNN vs. fine-tuned ResNet18).
- **Milestone 2** — synthetic multi-celebrity detection dataset (3x3 grids, YOLO labels, augmentation, splits).
- **Milestone 3** — YOLOv8 transfer learning, detection metrics, hyperparameter sweep, gallery and live demo.

## Team

| Name | Milestone 1 | Milestone 2 | Milestone 3 |
|---|---|---|---|
| Abdellah Faleh | Data pipeline, model training (Custom CNN + ResNet18), results write-up | Detection dataset pipeline (grids, YOLO labels, augmentation, split) | YOLOv8 training, evaluation, sweep, gallery, demo |
| Jin-Woo Hong | GitHub repo setup & maintenance, reproducibility (README, environment) | Synthetic face-frame data pipeline | YOLOv8n baseline on the original Milestone 2 dataset, evaluation/threshold-sweep/gallery scripts, live-demo notebook |
| Samuel Tong | Evaluation & review (results, confusion matrix/metrics sanity-check, proposal review) | Dataset QA, YOLO annotation/split review, documentation review | Reviewed final YOLOv8 metrics, hyperparameter sweep, error patterns, visualization gallery, live demo, and rubric completeness |
| Tristan Lyons | Canvas submission coordination | Code/results review, Canvas submission | Review, Canvas submission |

## Repository layout

```
.
├── README.md
├── requirements.txt
├── data/                      # CelebA files go here (not committed — see Data setup)
├── extracted_identities/      # M2: face crops for the 13 class-pool identities
├── detection_dataset/         # M2: synthetic grids + YOLO labels in train/ val/ test/ (submitted dataset)
├── detection_dataset_m3/      # M3: scaled dataset, same method, photo-level split (120/12/12 grids)
├── models/                    # M3: fine-tuned YOLOv8 checkpoint
├── results/milestone3/        # M3: metrics, sweeps, loss curves, training logs, gallery
├── runs/detect/               # M3: YOLOv8n baseline run on the original M2 dataset (weights, metrics, gallery)
├── src/
│   ├── select_identities.py   # M1: pick identities from CelebA metadata
│   ├── dataset.py             # M1: PyTorch Dataset + transforms for the chosen subset
│   ├── models.py              # M1: CustomCNN architecture
│   ├── utils.py               # M1: seeding, training loop, metrics, plotting helpers
│   ├── build_synthetic_detection_dataset.py, augment_and_split_detection_dataset.py  # M2
│   ├── train_yolo_detector.py # M3: YOLOv8 fine-tuning on the original M2 dataset (baseline run)
│   ├── evaluate_yolo_detector.py, sweep_yolo_inference.py, render_yolo_test_gallery.py  # M3: baseline evaluation
│   ├── build_m3_scaled_dataset.py  # M3: rebuild the dataset at scale with the M2 method
│   ├── m3_detection.py        # M3: YOLOv8 training, evaluation, IoU matching, gallery
│   ├── run_milestone3.py      # M3: full pipeline in one command + results document
│   └── demo_detect.py         # M3: live demo (command line or Gradio app)
├── notebooks/
│   ├── 01_data_preparation.ipynb        # M1: subset selection + split + preprocessing
│   ├── 02_custom_cnn_training.ipynb     # M1: train the from-scratch CNN
│   ├── 03_transfer_learning_resnet.ipynb# M1: fine-tune pretrained ResNet18
│   ├── 04_model_evaluation_comparison.ipynb # M1: test metrics, confusion matrix, comparison
│   ├── 05_detection_dataset_preparation.ipynb # M2: CelebA crops -> YOLO detection dataset
│   ├── 06_yolov8_live_demo.ipynb            # M3: load a fine-tuned detector, try a test/uploaded image
│   └── 06_yolov8_training_evaluation.ipynb  # M3: full pipeline (scaled dataset, sweep, results, gallery)
├── logs/                      # M1: loss/accuracy curves, comparison_table.csv, run logs
└── docs/
    ├── proposal.md, training_results.md      # M1 write-ups
    ├── milestone2_detection_dataset.md       # M2 dataset documentation
    ├── test_detection_gallery.md             # M3 baseline-run gallery (original M2 dataset)
    ├── milestone3_detection_results.md       # M3 detection-results document (main report)
    └── Milestone3_Detection_Results.pdf      # M3 report, PDF export
```

## Data setup

This repo does not commit the CelebA dataset. Download **Align&Cropped Images**
(`img_align_celeba.zip`), `identity_CelebA.txt`, and `list_attr_celeba.txt` from the
[official CelebA page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (or your course's
mirror) and place them as:

```
data/
├── img_align_celeba/           # ~202,599 .jpg files
├── identity_CelebA.txt         # "<filename> <identity_id>" per line
└── list_attr_celeba.txt        # 40 binary attributes per image
```

## Environment

```bash
pip install -r requirements.txt
```

Two supported compute setups:
- **Northeastern Research Computing Explorer (Open OnDemand, CPU-only)** — some
  compute nodes are older Xeons without AVX2, which can silently kill PyTorch/TF kernels
  during model-building ops. If a kernel dies with no traceback while building the model,
  try a newer partition/node, or reduce batch size and disable AVX-dependent ops.
- **Local machine (Windows/Mac/Linux)**, CPU or GPU (GPU strongly recommended — CelebA
  face crops at 128–224px make CPU-only ResNet fine-tuning slow; keep epochs/batch small
  on CPU, see notebook comments).

## Milestone 1 — how to reproduce

1. `python src/select_identities.py` — prints the chosen identity IDs, per-identity image
   counts, and diversity rationale (also written to `logs/selected_identities.json`).
2. Run `notebooks/01_data_preparation.ipynb` — builds the train/val/test split and
   preprocessing pipeline for the selected identities; writes split manifests to `data/splits/`.
3. Run `notebooks/02_custom_cnn_training.ipynb` and `notebooks/03_transfer_learning_resnet.ipynb`
   — each trains one architecture, saves the best checkpoint to `logs/`, and saves its
   loss/accuracy curve plot to `logs/`.
4. Run `notebooks/04_model_evaluation_comparison.ipynb` — loads both checkpoints, evaluates
   on the held-out test split, produces the confusion matrix and the model-comparison table
   (`logs/comparison_table.csv`), and prints the best-model justification.

## Milestone 1 — celebrity subset

4 identities — the CelebA IDs each team member individually claimed for the
class-wide identity-pooling task (23-25 images each). Selected with:

```bash
python src/select_identities.py --fixed-ids 3 7 1212 8335
```

see `logs/selected_identities.json` and `docs/proposal.md` for the full rationale.

| Identity ID | # Images | Claimed by | Dominant attributes |
|---|---|---|---|
| 3 | 23-25 | Abdellah Faleh | _fill in after re-running select_identities.py_ |
| 7 | 23-25 | _fill in_ | _fill in after re-running select_identities.py_ |
| 1212 | 23-25 | Samuel Tong | _fill in after re-running select_identities.py_ |
| 8335 | 23-25 | _fill in_ | _fill in after re-running select_identities.py_ |

## Milestone 1 — best model

**ResNet18 (fine-tuned)** — 87.5% test accuracy, vs. 50% for the custom CNN, on the
updated 4-identity subset (3, 7, 1212, 8335 — the IDs individually claimed by team
members). This subset is harder than the original one (3 of 4 identities are male,
several share overlapping hair-color attributes), which is why both models scored
lower than on the original 5-identity set (was 95%/85%) — see
`docs/training_results.md` for the full comparison, confusion matrices, and
justification. Carried forward checkpoint: `logs/resnet18_finetune_best.pt`.

## Milestone 2 — detection dataset

`detection_dataset/` holds 30 synthetic 640x640 frames built from 6 base 3x3 grids
(9 unique faces each, drawn from 13 class-pool identities), each with an original and
4 augmented copies, plus one YOLO label file per image. The split is by base grid, so
no arrangement leaks between splits: **20 train / 5 val / 5 test** images
(180 / 45 / 45 faces). Identity-to-class mapping: `detection_dataset/classes.json`.
Pipeline: `notebooks/05_detection_dataset_preparation.ipynb`; augmentation parameters
and design choices: `docs/milestone2_detection_dataset.md`.

## Status

- **Milestone 1** — complete and submitted.
- **Milestone 2** — complete and submitted.
- **Milestone 3** — complete: model trained, evaluated and documented (below). Due Oct 11.

## Milestone 3 — YOLOv8 transfer learning and evaluation

Fine-tunes a COCO-pretrained YOLOv8n to detect and name the 13 class-pool identities
(one YOLO class each), runs a hyperparameter sweep, and produces the test metrics,
detection gallery, live demo and results document.

**Dataset.** A first run on the submitted Milestone 2 dataset (30 images) learned face
locations but not identities: each celebrity appeared in only ~4 of its ~25 photos.
`src/build_m3_scaled_dataset.py` rebuilds it with the same Milestone 2 method, splitting
each identity's photos 70/15/15 *before* building grids (no photo in two splits) and
making 120 train / 12 val / 12 test grids in which every identity appears in every split.
The build is deterministic (seed 42); the notebook builds it automatically. The original
Milestone 2 dataset is still trained on once as a reference row in the results.

**Baseline on the original dataset.** `runs/detect/celeba_yolov8n_milestone2/` holds an
independent YOLOv8n run on the submitted 30-image Milestone 2 dataset (`src/train_yolo_detector.py`,
evaluated with `src/evaluate_yolo_detector.py`, `src/sweep_yolo_inference.py` and
`src/render_yolo_test_gallery.py`; gallery in `docs/test_detection_gallery.md`). It reaches test
mAP@0.5 = 0.249, consistent with the 0.232 reference row in the main report, and confirms that the
original dataset was too small to learn identities.

**Results** (held-out test split, 12 grids / 108 faces; full report in
`docs/milestone3_detection_results.md`):

| Metric | Value |
|---|---|
| mAP@0.5 / mAP@0.5:0.95 | 0.879 / 0.876 |
| Precision / Recall (Ultralytics) | 0.849 / 0.759 |
| Mean IoU (correct detections) | 0.993 |
| Faces located / correctly identified | 108 of 108 / 84 of 108 (77.8%) |
| Baseline config on the original Milestone 2 data (test mAP@0.5) | 0.232 |

The final model (`aug_light`: YOLOv8n, AdamW, lr0 = 0.001, 640 px, lighter online augmentation)
was selected on validation mAP from a 6-run sweep over learning rate, image size,
augmentation intensity and model size (YOLOv8n vs YOLOv8s), plus an inference-time sweep over
confidence and NMS IoU thresholds. Every test face is found with a near-exact box; the
remaining errors are wrong identities, concentrated in a few look-alike pairs.

**Reproduce everything (one notebook run):** open
`notebooks/06_yolov8_training_evaluation.ipynb` in Google Colab, set *Runtime → Change runtime
type → T4 GPU*, then *Runtime → Run all* (about 45–60 min). Or from the repo root:

```bash
pip install -r requirements.txt
python src/run_milestone3.py              # full sweep, GPU recommended
python src/run_milestone3.py --epochs 2   # quick smoke test on CPU
```

**Live demo** (uses the committed checkpoint, no training needed):

```bash
python src/demo_detect.py                          # annotate the held-out test images
python src/demo_detect.py --source my_photo.jpg    # any image
python src/demo_detect.py --app                    # Gradio upload app at http://127.0.0.1:7860
python src/demo_detect.py --app --share            # public link (needed on Colab)
```

| Output | Location |
|---|---|
| Fine-tuned checkpoint | `models/celeba_yolov8_best.pt` |
| Final hyperparameters | `results/milestone3/final_model_hyperparameters.yaml` |
| Test metrics (overall, per identity) | `results/milestone3/test_metrics.json`, `test_metrics_per_class.csv` |
| Training sweep (lr, image size, augmentation, model size) | `results/milestone3/training_sweep.csv`, `training_sweep_curves.png` |
| Threshold sweep (confidence, NMS IoU) | `results/milestone3/threshold_sweep_*.csv`, `threshold_sweep.png` |
| Loss curves and per-run training logs | `results/milestone3/loss_curves.png`, `train_logs/` |
| Detection gallery (captioned) | `results/milestone3/gallery/README.md` |
| Results document | `docs/milestone3_detection_results.md` |

Code: `src/m3_detection.py` (dataset config, training, evaluation, IoU matching, gallery),
`src/run_milestone3.py` (pipeline steps, results document), `src/demo_detect.py` (demo).
