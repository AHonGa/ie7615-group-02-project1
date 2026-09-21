# Training Results — Milestone 1 Classification Baseline

_Export to PDF or submit as Markdown, 2-4 pages. Fill in every `_TODO_` after running
the four notebooks in `notebooks/` on real data — do not submit this template with
placeholder numbers._

## 1. Setup recap

- **Identities:** _TODO — paste from `docs/identity_selection_report.md`_
- **Split:** stratified 70/15/15 (train/val/test) per identity
- **Preprocessing:** 128x128 resize, ImageNet normalization, train-only augmentation
  (horizontal flip, color jitter, ±8° rotation)
- **Models compared:**
  1. Custom CNN (from scratch) — `src/models.py::CustomCNN`
  2. ResNet18 (fine-tuned, ImageNet-pretrained) — `src/models.py::build_resnet18`
  3. _(optional third variant, e.g. ResNet18 feature-extraction)_

## 2. Model comparison table

_TODO — paste the contents of `logs/comparison_table.csv` here as a Markdown table
(produced by `04_model_evaluation_comparison.ipynb`)._

| Model | Test Accuracy | Val Accuracy | Params | Size (MB) | Train Time (s) |
|---|---|---|---|---|---|
| custom_cnn | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| resnet18_finetune | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

## 3. Per-class performance (best model)

_TODO — paste `logs/<best_run>_per_class_report.csv`, and embed
`logs/<best_run>_confusion_matrix.png`._

![confusion matrix](../logs/PLACEHOLDER_confusion_matrix.png)

## 4. Training curves

_TODO — embed `logs/custom_cnn_curves.png` and `logs/resnet18_finetune_curves.png`._

![custom CNN curves](../logs/custom_cnn_curves.png)
![resnet18 curves](../logs/resnet18_finetune_curves.png)

## 5. Best-model justification

_TODO — 1-2 paragraphs. Address, per the rubric: accuracy, parameter count/model
size, training time, and how much labeled data each approach needed. State which
model is carried forward into Milestone 2/3 and why._

_Example structure (replace with your actual numbers and conclusion):_

> The [ResNet18 fine-tuned / custom CNN] model achieved the highest test accuracy
> (X.XX vs Y.YY), at the cost of [Z]x more parameters and [W]x longer training time.
> Given [N] images per identity, the [pretrained backbone's transferred features /
> from-scratch model's smaller footprint] made it the stronger choice because
> ___. We carry this checkpoint (`logs/<run_name>_best.pt`) forward into
> Milestone 2's detection pipeline.
