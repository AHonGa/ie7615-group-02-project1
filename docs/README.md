# Project 1 — Celebrity Classification & Detection

IE/EI 7615 — Deep Learning Architectures. Discriminative computer-vision pipeline for
celebrity identification and detection. Milestone 1 builds a single-face classification
baseline (a small CelebA identity subset, a custom CNN, and a transfer-learning ResNet18
baseline, compared). Milestone 2 builds the synthetic multi-celebrity detection dataset
(YOLO-format annotations) that Milestone 3 fine-tunes YOLOv8 on.

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
│   ├── select_identities.py             # Milestone 1: pick 4-6 identities from CelebA metadata
│   ├── extract_identity_images.py       # Milestone 1: export one identity's crops (individual task)
│   ├── extract_class_pool_images.py     # Milestone 2: export crops for many identities (class pool)
│   ├── build_synthetic_detection_dataset.py # Milestone 2: compose 3x3 grids + YOLO annotations
│   ├── augment_and_split_detection_dataset.py # Milestone 2: augmentation + leak-free train/val/test split
│   ├── dataset.py             # PyTorch Dataset + transforms for the chosen subset
│   ├── models.py              # CustomCNN architecture
│   └── utils.py               # seeding, training loop, metrics, plotting helpers
├── notebooks/
│   ├── 01_data_preparation.ipynb        # subset selection + split + preprocessing
│   ├── 02_custom_cnn_training.ipynb     # train the from-scratch CNN
│   ├── 03_transfer_learning_resnet.ipynb# fine-tune pretrained ResNet18
│   ├── 04_model_evaluation_comparison.ipynb # test metrics, confusion matrix, comparison table
│   ├── 05_detection_dataset_preparation.ipynb # Milestone 2: grid composition, augmentation, split
│   └── 06_yolov8_live_demo.ipynb         # load the fine-tuned detector; choose a test or uploaded image
├── logs/                      # saved loss/accuracy curves, comparison_table.csv, run logs
├── detection_dataset/         # Milestone 2 output (not committed if large — see Milestone 2 section)
│   ├── images/, labels/       # 6 base 3x3 grid images + YOLO label files
│   ├── previews/              # same grids with bounding boxes drawn on top, for verification
│   ├── train/, val/, test/    # augmented + split dataset (images/ + labels/ each)
│   ├── classes.json           # identity_id -> class_id mapping
│   ├── manifest.csv           # per-face provenance for the 6 base grids
│   └── split_manifest.csv     # per-image provenance for the train/val/test split
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

## Best model

**ResNet18 (fine-tuned)** — 87.5% test accuracy, vs. 50% for the custom CNN, on the
updated 4-identity subset (3, 7, 1212, 8335 — the IDs individually claimed by team
members). This subset is harder than the original one (3 of 4 identities are male,
several share overlapping hair-color attributes), which is why both models scored
lower than on the original 5-identity set (was 95%/85%) — see
`docs/training_results.md` for the full comparison, confusion matrices, and
justification. Carried forward checkpoint: `logs/resnet18_finetune_best.pt`.

## Milestone 2 — Detection dataset construction

Synthetic multi-celebrity 3x3 grid images with YOLO-format bounding box
annotations, built from a **class-wide identity pool** (not just our own 4
IDs) per the assignment's specific instructions. 6 grids total: 2 for this
report, 4 additional ones sent separately to the TA for cross-group
aggregation ahead of Milestone 3.

**Class pool used** (13 identities from the shared class spreadsheet):

| Group    | Names                                                       | Celeb IDs              |
| -------- | ----------------------------------------------------------- | ---------------------- |
| 1        | Dario Garza, Yosephine Tong, Murat Akca                     | 797, 10002, 5695       |
| 2 (ours) | Abdellah Faleh, Samuel Tong, Jin-woo Hong, Tristan Lyons    | 3, 1212, 7, 8335       |
| 3        | Julia Rasmussen, Rhea Paul, Mus Ab Irfan Yilmaz, Masato Kan | 4422, 2970, 7007, 2336 |
| 4        | David Fung                                                  | 4428                   |
| 5        | Jiasong Zhang                                               | 2619                   |

**Grid composition:** each of the 6 grids places 9 unique identities (drawn
from the 13 above) into a 3x3 layout, resized to 640x640 (YOLOv8's default
input size). Placement, scale, and background are randomized per grid so no
two grids repeat the same arrangement.

**Augmentation strategy** (applied per grid to build the training set; boxes
are mathematically recomputed for every transform, not just copied):

| Augmentation      | Parameters  | Why                                                                                 |
| ----------------- | ----------- | ----------------------------------------------------------------------------------- |
| Horizontal flip   | p = 0.5     | Faces are roughly left-right symmetric — free extra viewpoint diversity, label-safe |
| Bounded rotation  | ±8°         | Simulates a slightly tilted photo; boxes recomputed from the rotated face corners   |
| Brightness jitter | 0.85x-1.15x | Simulates different lighting; pixel-only, doesn't affect boxes                      |
| Contrast jitter   | 0.85x-1.15x | Same rationale as brightness                                                        |
| Centered zoom     | 0.95x-1.05x | Simulates minor camera distance/framing differences; boxes rescaled to match        |

**Train/val/test split:** done at the _base grid_ level (not per augmented
copy) so every augmented version of a grid stays in the same split — this is
what prevents image leakage. Default: 4 base grids → train, 1 → val, 1 → test;
each grid contributes 1 original + 4 augmented copies:

| Split | Base grids | Total images |
| ----- | ---------- | ------------ |
| train | 4          | 20           |
| val   | 1          | 5            |
| test  | 1          | 5            |

**How to reproduce:**

```bash
python src/extract_class_pool_images.py 3 7 1212 8335 2619 797 10002 5695 4422 2970 7007 2336 4428
python src/build_synthetic_detection_dataset.py --ids 3 7 1212 8335 2619 797 10002 5695 4422 2970 7007 2336 4428 --n-grids 6
python src/augment_and_split_detection_dataset.py --n-aug 4
```

or run `notebooks/05_detection_dataset_preparation.ipynb` end-to-end, which
performs all three steps with inline visualizations of sample annotated grids
and the split's leak-free verification.

**Dataset location:** `detection_dataset/` (see repository layout above);
`classes.json` holds the identity_id → class_id mapping — share this with the
group if grids from multiple teams need to be combined for Milestone 3.

## Status

Milestone 1 complete on the updated identity subset (3, 7, 1212, 8335): data
pipeline rebuilt, both models retrained and evaluated, results documented in
`docs/proposal.pdf` and `docs/training_results.pdf`.

Milestone 2 complete: 6 synthetic multi-celebrity grids built from the
class-wide identity pool, YOLO annotations generated, augmentation strategy
applied and documented, leak-free train/val/test split produced. Remaining:
GitHub repo kept up to date with all collaborators, and final Canvas
submission (link + report).
