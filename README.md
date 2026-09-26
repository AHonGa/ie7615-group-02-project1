# Project 1, Milestone 1 — Celebrity Classification Baseline

IE/EI 7615 — Deep Learning Architectures. Discriminative computer-vision baseline for
celebrity identification: a small CelebA identity subset, a custom CNN, and a
transfer-learning (ResNet18) baseline, compared and carried forward into Milestone 2/3.

## Team

| Name | Role |
|---|---|
| Abdellah Faleh | Data pipeline, model training (Custom CNN + ResNet18), results write-up |
| Jin-Woo Hong | GitHub repo setup & maintenance, reproducibility (README, environment) |
| Samuel Tong | Evaluation & review (results, confusion matrix/metrics sanity-check, proposal review) |

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

5 identities selected from CelebA (30 images each), chosen for balanced counts and
attribute diversity — see `logs/selected_identities.json` and `docs/proposal.md` for
the full rationale.

| Identity ID | # Images | Dominant attributes |
|---|---|---|
| 6439 | 30 | Blond Hair |
| 5640 | 30 | Male, Gray Hair, Eyeglasses, Bald |
| 6195 | 30 | Male, Black Hair, Mustache |
| 5485 | 30 | Male, Eyeglasses, Wearing Hat |
| 7087 | 30 | Black Hair, Eyeglasses |

## Best model

**ResNet18 (fine-tuned)** — 95% test accuracy, vs. 85% for the tuned custom CNN
(from-scratch training initially collapsed to 20% with heavier augmentation/higher
learning rate; fixed by lightening augmentation and lowering the learning rate — see
`docs/training_results.md` for the full comparison, confusion matrices, and
justification). Carried forward checkpoint: `logs/resnet18_finetune_best.pt`.

## Status

Milestone 1 complete: identities selected, data pipeline built, both models trained
and evaluated on real CelebA data, results documented in `docs/proposal.pdf` and
`docs/training_results.pdf`. Remaining: team roster and division of labor (pending
instructor group assignment) and final Canvas submission.
