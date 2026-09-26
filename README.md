# Project 1, Milestone 1 — Celebrity Classification Baseline

IE/EI 7615 — Deep Learning Architectures. Discriminative computer-vision baseline for
celebrity identification: a small CelebA identity subset, a custom CNN, and a
transfer-learning (ResNet18) baseline, compared and carried forward into Milestone 2/3.

## Team

| Name           | Role                                                                                  |
| -------------- | ------------------------------------------------------------------------------------- |
| Abdellah Faleh | Data pipeline, model training (Custom CNN + ResNet18), results write-up               |
| Jin-Woo Hong   | GitHub repo setup & maintenance, reproducibility (README, environment)                |
| Samuel Tong    | Evaluation & review (results, confusion matrix/metrics sanity-check, proposal review) |

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

## Milestone identity requirements

The 23–25 image requirement is a course requirement. Before claiming an identity,
run `python src/find_identity_23_25.py`; it reports identities whose metadata count is
23–25 and verifies that every corresponding image file is present in
`data/img_align_celeba/`. Do not claim an identity based on an unverified count.

After the class shared pool is complete, place its contributed images in
`data/shared_pool/<identity_id>/`. Each group selects 4–6 distinct identities from
that pool. Check the images for visual diversity, then run the pool verifier. For the
currently proposed IDs:

```bash
python src/select_identities.py --identity-ids 3 7 8335 1212 --diversity-review "Summarize the visual variation reviewed across the selected pool images."
```

The command counts image files in each shared-pool folder and writes
`logs/selected_identities.json` only if every selected identity has 23–25 images.
The data-preparation notebook creates splits from those folders only; it does not
read training images from the full CelebA archive. If a count fails, complete or
correct the shared pool and select again before training.

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

1. Verify individual candidate counts, complete the class shared pool, and run the
   shared-pool selection command above. The report records the verified counts and
   visual-diversity review.
2. Run `notebooks/01_data_preparation.ipynb` — builds the train/val/test split and
   preprocessing pipeline for the selected identities; writes split manifests to `data/splits/`.
3. Run `notebooks/02_custom_cnn_training.ipynb` and `notebooks/03_transfer_learning_resnet.ipynb`
   — each trains one architecture, saves the best checkpoint to `logs/`, and saves its
   loss/accuracy curve plot to `logs/`.
4. Run `notebooks/04_model_evaluation_comparison.ipynb` — loads both checkpoints, evaluates
   on the held-out test split, produces the confusion matrix and the model-comparison table
   (`logs/comparison_table.csv`), and prints the best-model justification.

## Previous experiment results

The existing plots, comparison table, and PDFs document an earlier run using five
CelebA identities with 30 images each. Those results are retained as historical
artifacts only; they do not meet the revised 23–25 image and shared-pool requirements.
Rerun preparation, training, and evaluation after the shared pool passes verification
before reporting milestone results.

## Status

Milestone 1 complete: identities selected, data pipeline built, both models trained
and evaluated on real CelebA data, and the results are documented in
`docs/proposal.pdf` and `docs/training_results.pdf`. Remaining: team roster and
final division of labor, pending instructor group assignment and final Canvas
submission.
