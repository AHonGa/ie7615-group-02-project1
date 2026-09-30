# Project 1 — Team Proposal (Refreshed, Milestone 1)

_Export this file to PDF for submission (e.g. `pandoc proposal.md -o proposal.pdf`, or open in a
Markdown editor and print to PDF). Keep it to 1 page._

**Team:** Abdellah Faleh, Jin-Woo Hong, Samuel Tong, Tristan Lyons
**Course:** EI 7615 — Deep Learning Architectures
**Date:** September 26, 2026

## Project

Discriminative computer-vision pipeline for celebrity identification and detection
(Project 1). Milestone 1 establishes a single-face classification baseline over a
small CelebA identity subset; Milestones 2-3 extend it to multi-face detection with
YOLOv8.

## Celebrity identities (final list)

Selected 4 identities — the CelebA IDs each team member individually claimed for
the class-wide identity-pooling task (23-25 images each), rather than an
independently-diversity-selected set:

| Identity ID | # Images | Claimed by | Dominant attributes |
|---|---|---|---|
| 3 | 25 | Abdellah Faleh | Male, Wearing_Hat |
| 7 | 24 | Jin-Woo Hong | Male, Black_Hair |
| 1212 | 25 | Samuel Tong | Male, Black_Hair |
| 8335 | 25 | Tristan Lyons | Male, Brown_Hair |

## Framework choice

**PyTorch** (torchvision for the pretrained ResNet18 backbone). Chosen for:
- Flexible custom `nn.Module` definitions for the from-scratch CNN.
- `torchvision.models` gives ImageNet-pretrained weights for the transfer-learning
  baseline with feature-extraction and fine-tuning modes built in.
- Runs on both CPU (Northeastern Explorer) and any team member's local GPU without
  code changes.

_Adjust this section if your team ultimately used TensorFlow/Keras instead._

## Division of labor

| Task | Owner |
|---|---|
| Data pipeline, model training (Custom CNN + ResNet18) | Abdellah Faleh |
| Results write-up (proposal, training-results docs) | Abdellah Faleh |
| GitHub repo setup & maintenance, reproducibility (README, environment) | Jin-Woo Hong |
| Evaluation & review (results, confusion matrix/metrics sanity-check, proposal review) | Samuel Tong |
| Milestone 2 planning (synthetic multi-celebrity dataset, YOLO annotations), Canvas submission coordination | Tristan Lyons |

## Anticipated risks

- **Compute constraints**: Explorer's CPU-only nodes (some without AVX2) can silently
  kill TensorFlow/PyTorch kernels during model-building ops; mitigated by keeping
  image size (128x128) and batch size small, and by testing on a different
  partition/node if a kernel dies with no traceback.
- **Class imbalance**: CelebA identities have uneven image counts; mitigated by
  selecting identities within a bounded image-count range and stratifying the
  train/val/test split per identity.
- **Small per-identity sample size**: only 23-25 images per identity limits how much a
  from-scratch CNN can learn. This risk materialized during training: an initial custom
  CNN configuration (heavier augmentation, higher learning rate) collapsed to predicting
  a single class (20% accuracy, equivalent to random guessing). We mitigated it by
  lightening augmentation, lowering the learning rate, and reducing dropout, which
  brought the custom CNN to a genuine (if lower, on this harder identity subset) 50%
  test accuracy — below the fine-tuned ResNet18's 87.5%, but well above the 25%
  random-guessing baseline. See `docs/training_results.md` for the full comparison.
- **Milestone 2 dependency**: the model carried forward from Milestone 1 becomes part
  of the Milestone 2/3 detection pipeline, so architecture/checkpoint format decisions
  made here should stay compatible with a YOLOv8 integration later.
