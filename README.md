# Project 1, Milestone 1 — Celebrity Classification Baseline

IE/EI 7615 — Deep Learning Architectures. Discriminative computer-vision baseline for
celebrity identification: a small CelebA identity subset, a custom CNN, and a
transfer-learning (ResNet18) baseline, compared and carried forward into Milestone 2/3.

## Team

| Name           | Role                                                                                                       |
| -------------- | ---------------------------------------------------------------------------------------------------------- |
| Abdellah Faleh | Data pipeline, model training (Custom CNN + ResNet18), results write-up                                    |
| Jin-Woo Hong   | GitHub repo setup & maintenance, reproducibility (README, environment)                                     |
| Samuel Tong    | Evaluation & review (results, confusion matrix/metrics sanity-check, proposal review)                      |
| Tristan Lyons  | Milestone 2 planning (synthetic multi-celebrity dataset, YOLO annotations), Canvas submission coordination |

## Repository layout

```
.
├── README.md
├── requirements.txt
├── data/                      # CelebA files go here (not committed — see Data setup)
├── src/
│   ├── select_identities.py   # Step 1: pick 4-6 identities from CelebA metadata
│   ├── dataset.py             # PyTorch Dataset + transforms for the chosen subset
│   ├── models.py              # CustomCNN architecture
│   └── utils.py               # seeding, training loop, metrics, plotting helpers
├── notebooks/
│   ├── 01_data_preparation.ipynb        # subset selection + split + preprocessing
│   ├── 02_custom_cnn_training.ipynb     # train the from-scratch CNN
│   ├── 03_transfer_learning_resnet.ipynb# fine-tune pretrained ResNet18
│   └── 04_model_evaluation_comparison.ipynb # test metrics, confusion matrix, comparison table
├── logs/                      # saved loss/accuracy curves, comparison_table.csv, run logs
└── docs/
    ├── proposal.md            # refreshed 1-page team proposal (export to PDF for submission)
    └── training_results.md    # training-results write-up (export to PDF/Markdown for submission)
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

## How to reproduce

1. `python src/select_identities.py --fixed-ids 3 7 1212 8335` — confirms the chosen
   identity IDs (the ones individually claimed by team members), prints their real
   image counts and dominant attributes (also written to `logs/selected_identities.json`).
2. Run `notebooks/01_data_preparation.ipynb` — builds the train/val/test split and
   preprocessing pipeline for the selected identities; writes split manifests to `data/splits/`.
3. Run `notebooks/02_custom_cnn_training.ipynb` and `notebooks/03_transfer_learning_resnet.ipynb`
   — each trains one architecture, saves the best checkpoint to `logs/`, and saves its
   loss/accuracy curve plot to `logs/`.
4. Run `notebooks/04_model_evaluation_comparison.ipynb` — loads both checkpoints, evaluates
   on the held-out test split, produces the confusion matrix and the model-comparison table
   (`logs/comparison_table.csv`), and prints the best-model justification.

## Celebrity subset

4 identities — the CelebA IDs each team member individually claimed for the
class-wide identity-pooling task (23-25 images each). Selected with:

```bash
python src/select_identities.py --fixed-ids 3 7 1212 8335
```

see `logs/selected_identities.json` and `docs/proposal.md` for the full rationale.

| Identity ID | # Images | Claimed by     | Dominant attributes |
| ----------- | -------- | -------------- | ------------------- |
| 3           | 25       | Abdellah Faleh | Male, Wearing_Hat   |
| 7           | 24       | Jin-Woo Hong   | Male, Black_Hair    |
| 1212        | 25       | Samuel Tong    | Male, Black_Hair    |
| 8335        | 25       | Tristan Lyons  | Male, Brown_Hair    |

## Synthetic multi-identity frames

Build a reproducible 640x640 composite dataset with 2-4 different identities per frame:

```bash
python src/build_synthetic_frames.py
```

The generator reads `data/img_align_celeba/` and `data/identity_CelebA.txt`. On its first run,
it downloads the YuNet face detector into the user cache (`~/.cache/ie7615/`). The default
output is `data/synthetic_face_frames_70_15_15/` with 200 train, 40 validation, and 40 test
frames, one YOLO face-box label file per image, `dataset.yaml`, a source manifest, and visual
samples in `visual_checks/`. Generated data stays local because `data/` is git-ignored. Source
photos are split by identity at 70/15/15 before composition to prevent image-level leakage.
Largest-remainder allocation gives 17/4/4 source images for identities 3, 1212, and 8335,
and 17/4/3 for identity 7. Each split's frame schedule includes every identity whenever
the split has at least one frame. Frames use varied
classroom-style backgrounds, group layouts, portrait scales, positions, and lighting. Geometric
augmentations transform the face boxes along with the pixels; `manifest.csv` records the sampled
frame-level parameters. These are synthetic composites, not natural-scene face annotations.
Override counts, paths, or seed with CLI options.

For Requirement 2, OpenCV YuNet is used as an automatic face pre-annotation tool. It detects
face regions on the source portrait crops; the boxes are transformed with each crop into the
composite and updated again during geometric augmentation before being written as normalized
YOLO labels. YuNet localizes faces but does not identify them: class IDs come from the selected
CelebA identity mapping. Review the generated overlays in `visual_checks/` to visually verify
sample box placement.

### Synthetic-frame augmentations

| Augmentation                        | Parameters                                                     | Why                                                                                                                   |
| ----------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Per-portrait horizontal mirror      | Probability 0.5 per portrait                                   | Adds left/right pose variation without changing identity. Face boxes are mirrored with each crop.                     |
| Per-portrait appearance jitter      | Brightness 0.92–1.08, contrast 0.95–1.05, saturation 0.94–1.06 | Adds modest subject-level lighting and color variation without washing out facial detail.                             |
| Whole-frame horizontal flip         | Probability 0.5 per frame                                      | Adds composition-level left/right variation; all boxes are flipped with the image.                                    |
| Whole-frame scale                   | 0.92–1.08, centered                                            | Simulates small framing-distance changes while keeping the full canvas; the same scale is applied to box corners.     |
| Whole-frame rotation                | -6° to +6°                                                     | Simulates mild camera tilt. The transformed four corners form a valid axis-aligned YOLO box, clipped to image bounds. |
| Whole-frame brightness and contrast | Each factor 0.90–1.10                                          | Simulates exposure and contrast variation without changing class or geometry.                                         |

Every frame-level sample is recorded in `manifest.csv` so the exact applied transforms can be reviewed.

## YOLOv8 transfer learning and evaluation

Fine-tune a pretrained YOLOv8 nano checkpoint on the synthetic multi-identity dataset:

```bash
python src/train_yolo_detector.py
```

The trainer starts from `yolov8n.pt` (downloaded by Ultralytics if needed), trains for up to
300 epochs, and stops early after 30 epochs without validation improvement. It selects the
best validation checkpoint and evaluates it on the held-out `test` split. Ultralytics writes
the training loss/metric curves to `runs/detect/celeba_yolov8n/results.png`, epoch-level
values to `results.csv`, and test evaluation artifacts to a sibling `celeba_yolov8n_test`
run. The script also saves test metrics as `test_metrics.json` beside the best checkpoint.
Use `--epochs`, `--patience`, `--batch`, `--imgsz`, `--device`, or `--data` to adjust the run.

The ResNet18 result below is the Milestone 1 image-classification baseline; YOLOv8 is the
transfer-learning model for the Milestone 2 multi-face detection task.

## Best model

**ResNet18 (fine-tuned)** — 93.75% test accuracy, vs. 68.75% for the custom CNN, on the
updated 4-identity subset (3, 7, 1212, 8335 — the IDs individually claimed by team
members). This is the current verified Milestone 1 run, and it supersedes older
historical numbers in the repository. Carried forward checkpoint:
`logs/resnet18_finetune_best.pt`.

## Status

Milestone 1 complete on the updated identity subset (3, 7, 1212, 8335): data
pipeline rebuilt, both models retrained and evaluated, and the current results are
recorded in `docs/proposal.md` and `docs/training_results.md`. Remaining: GitHub
repo setup with all collaborators and final Canvas submission.
