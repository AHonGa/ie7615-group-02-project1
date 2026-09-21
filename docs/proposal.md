# Project 1 — Team Proposal (Refreshed, Milestone 1)

_Export this file to PDF for submission (e.g. `pandoc proposal.md -o proposal.pdf`, or open in a
Markdown editor and print to PDF). Keep it to 1 page._

**Team:** _names_
**Course:** EI 7615 — Deep Learning Architectures
**Date:** _fill in_

## Project

Discriminative computer-vision pipeline for celebrity identification and detection
(Project 1). Milestone 1 establishes a single-face classification baseline over a
small CelebA identity subset; Milestones 2-3 extend it to multi-face detection with
YOLOv8.

## Celebrity identities (final list)

_Paste the table from `docs/identity_selection_report.md` here once
`src/select_identities.py` has been run against the real dataset. Include identity
IDs, image counts, and the diversity rationale (gender, hair color, eyewear, etc.)._

## Framework choice

**PyTorch** (torchvision for the pretrained ResNet18 backbone). Chosen for:
- Flexible custom `nn.Module` definitions for the from-scratch CNN.
- `torchvision.models` gives ImageNet-pretrained weights for the transfer-learning
  baseline with feature-extraction and fine-tuning modes built in.
- Runs on both CPU (Northeastern Explorer) and any team member's local GPU without
  code changes.

_Adjust this section if your team ultimately used TensorFlow/Keras instead._

## Division of labor (Milestone 2 outlook)

| Task | Owner |
|---|---|
| Custom CNN architecture + training | _name_ |
| Transfer-learning baseline + training | _name_ |
| Evaluation, comparison table, confusion matrix | _name_ |
| GitHub repo setup, README, reproducibility | _name_ |
| Proposal + training-results write-up | _name_ |
| Milestone 2 planning (synthetic multi-celebrity dataset, YOLO annotations) | _name_ |

## Anticipated risks

- **Compute constraints**: Explorer's CPU-only nodes (some without AVX2) can silently
  kill TensorFlow/PyTorch kernels during model-building ops; mitigated by keeping
  image size (128x128) and batch size small, and by testing on a different
  partition/node if a kernel dies with no traceback.
- **Class imbalance**: CelebA identities have uneven image counts; mitigated by
  selecting identities within a bounded image-count range and stratifying the
  train/val/test split per identity.
- **Small per-identity sample size**: only ~20-45 images per identity limits how much
  a from-scratch CNN can learn; mitigated by heavier augmentation on the custom CNN
  and by including a pretrained baseline that doesn't depend solely on this dataset's
  size.
- **Milestone 2 dependency**: the model carried forward from Milestone 1 becomes part
  of the Milestone 2/3 detection pipeline, so architecture/checkpoint format decisions
  made here should stay compatible with a YOLOv8 integration later.
