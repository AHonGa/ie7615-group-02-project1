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
│   ├── 04_model_evaluation_comparison.ipynb # test metrics, confusion matrix, comparison table
│   ├── 05_detection_dataset_preparation.ipynb # synthetic detection dataset preparation
│   └── 06_yolov8_live_demo.ipynb         # load the fine-tuned detector and try a test/uploaded image
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

## Celebrity subset

4 identities — the CelebA IDs each team member individually claimed for the
class-wide identity-pooling task (23-25 images each). Selected with:

```bash
python src/select_identities.py --fixed-ids 3 7 1212 8335
```

see `logs/selected_identities.json` and `docs/proposal.md` for the full rationale.

| Identity ID | # Images | Claimed by     | Dominant attributes                             |
| ----------- | -------- | -------------- | ----------------------------------------------- |
| 3           | 23-25    | Abdellah Faleh | _fill in after re-running select_identities.py_ |
| 7           | 23-25    | _fill in_      | _fill in after re-running select_identities.py_ |
| 1212        | 23-25    | Samuel Tong    | _fill in after re-running select_identities.py_ |
| 8335        | 23-25    | _fill in_      | _fill in after re-running select_identities.py_ |

## Best model

**ResNet18 (fine-tuned)** — 87.5% test accuracy, vs. 50% for the custom CNN, on the
updated 4-identity subset (3, 7, 1212, 8335 — the IDs individually claimed by team
members). This subset is harder than the original one (3 of 4 identities are male,
several share overlapping hair-color attributes), which is why both models scored
lower than on the original 5-identity set (was 95%/85%) — see
`docs/training_results.md` for the full comparison, confusion matrices, and
justification. Carried forward checkpoint: `logs/resnet18_finetune_best.pt`.

## Status

Milestone 1 complete on the updated identity subset (3, 7, 1212, 8335): data
pipeline rebuilt, both models retrained and evaluated, results documented in
`docs/proposal.pdf` and `docs/training_results.pdf`. Remaining: fill in who claimed
IDs 7 and 8335, GitHub repo setup with all collaborators, and final Canvas
submission.
