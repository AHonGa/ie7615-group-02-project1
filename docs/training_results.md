# Training Results — Milestone 1 Classification Baseline

## 1. Setup recap

- **Identities:** 4 CelebA identities from the class-wide identity-pooling task:

  | Identity ID | # Images | Claimed by     | Dominant attributes |
  | ----------- | -------: | -------------- | ------------------- |
  | 3           |       25 | Abdellah Faleh | Male, Wearing_Hat   |
  | 7           |       24 | Jin-Woo Hong   | Male, Black_Hair    |
  | 1212        |       25 | Samuel Tong    | Male, Black_Hair    |
  | 8335        |       25 | Tristan Lyons  | Male, Brown_Hair    |

- **Split:** per-identity 70/15/15 split, generating train/validation/test counts such as 17/16/17 and 4/4/4 per identity.
- **Preprocessing:** 128x128 resize, ImageNet normalization, light train-only augmentation.
- **Models compared:**
  1. Custom CNN (from scratch)
  2. ResNet18 (fine-tuned, ImageNet-pretrained)

## 2. Model comparison table

| Model                     | Test Accuracy | Val Accuracy |     Params | Size (MB) | Train Time (s) |
| ------------------------- | ------------: | -----------: | ---------: | --------: | -------------: |
| **ResNet18 (fine-tuned)** |    **0.9375** |   **1.0000** | 11,178,564 |     42.64 |          177.5 |
| Custom CNN (from scratch) |        0.6875 |       0.8750 |  1,207,588 |      4.61 |          372.2 |

## 3. Best-model justification

The **ResNet18 fine-tuned model** is the strongest baseline for this task, reaching 93.75% test
accuracy and perfect validation accuracy. The custom CNN remains a valid lightweight benchmark,
but it trails substantially at 68.75% test accuracy on the same held-out split.

This is the current verified Milestone 1 result set and supersedes any older historical values in
this repository. The best checkpoint is the one saved at `logs/resnet18_finetune_best.pt` and is the
model carried forward into the next milestone.
