# Project 1 — Team Proposal (Refreshed, Milestone 1)

_Export this file to PDF for submission if needed, or keep it as a Markdown brief for the project record._

**Team:** Abdellah Faleh, Jin-Woo Hong, Samuel Tong, Tristan Lyons  
**Course:** EI 7615 — Deep Learning Architectures  
**Date:** 2026-09-26

## Project

This project builds a discriminative computer-vision baseline for celebrity identification
using a compact CelebA subset. The Milestone 1 task is to compare a custom CNN trained from
scratch against a transfer-learning baseline built on a pretrained ResNet18. The best model
is then carried forward into the later detection-stage milestones.

## Celebrity identities (final list)

We selected the four identities used by the class-wide identity-pooling task, with each
identity verified to have 23–25 shared-pool images before split generation:

| Identity ID | # Images | Claimed by     | Dominant attributes |
| ----------- | -------: | -------------- | ------------------- |
| 3           |       25 | Abdellah Faleh | Male, Wearing_Hat   |
| 7           |       24 | Jin-Woo Hong   | Male, Black_Hair    |
| 1212        |       25 | Samuel Tong    | Male, Black_Hair    |
| 8335        |       25 | Tristan Lyons  | Male, Brown_Hair    |

## Framework choice

**PyTorch** and `torchvision` were chosen because they provide:

- flexible custom model definitions for the from-scratch CNN
- ImageNet-pretrained ResNet18 weights for transfer learning
- simple CPU/GPU portability for local and cluster execution

## Division of labor

| Task                                                  | Owner          |
| ----------------------------------------------------- | -------------- |
| Data pipeline, model training (Custom CNN + ResNet18) | Abdellah Faleh |
| Results write-up and milestone summary                | Abdellah Faleh |
| GitHub repo setup and reproducibility                 | Jin-Woo Hong   |
| Evaluation and review                                 | Samuel Tong    |
| Milestone 2 planning and submission coordination      | Tristan Lyons  |

## Current results

The final verified Milestone 1 results from the latest run are:

| Model                 | Test accuracy | Validation accuracy | Parameters |
| --------------------- | ------------: | ------------------: | ---------: |
| Custom CNN            |        0.6875 |              0.8750 |  1,207,588 |
| ResNet18 (fine-tuned) |        0.9375 |              1.0000 | 11,178,564 |

The pretrained ResNet18 is the best model and is the one carried forward for the next milestone.

## Anticipated risks and mitigation

- **Small dataset size**: only ~16–17 train images per identity. This is mitigated by using a
  lightweight split and a strong pretrained backbone.
- **Compute constraints**: some older CPU nodes can fail during PyTorch model build steps; the project
  keeps the image size and batch size modest to remain portable.
- **Class overlap**: several identities share hair-color patterns, so the custom CNN is naturally
  harder to optimize than the pretrained ResNet baseline.

The latest values above are the current source-of-truth metrics and supersede earlier historical
numbers in the repository.
