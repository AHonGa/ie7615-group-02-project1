# Milestone 4 result tables (generated)

## Data split (face photos, shared by classifier and detector)

| split | photos | identities | photos per identity |
|---|---|---|---|
| train | 219 | 13 | 16.800 |
| val | 50 | 13 | 3.800 |
| test | 50 | 13 | 3.800 |

## Classifier (single face photos)

Validation accuracy 0.840, test accuracy 0.780 (39/50); best epoch 5.

| identity | photos | correct | accuracy |
|---|---|---|---|
| celeb_3 (ours) | 4 | 4 | 1 |
| celeb_7 (ours) | 4 | 4 | 1 |
| celeb_797 | 4 | 4 | 1 |
| celeb_1212 (ours) | 4 | 2 | 0.500 |
| celeb_2336 | 4 | 4 | 1 |
| celeb_2619 | 4 | 4 | 1 |
| celeb_2970 | 4 | 3 | 0.750 |
| celeb_4422 | 4 | 1 | 0.250 |
| celeb_4428 | 3 | 1 | 0.333 |
| celeb_5695 | 4 | 3 | 0.750 |
| celeb_7007 | 4 | 2 | 0.500 |
| celeb_8335 (ours) | 4 | 4 | 1 |
| celeb_10002 | 3 | 3 | 1 |

## Integrated pipeline (multi-face grids)

Validation (used to choose the setting):

| setting | faces | correct | wrong_identity | missed | extra_boxes | accuracy | precision | recall | f1 | mean_iou |
|---|---|---|---|---|---|---|---|---|---|---|
| Milestone 3 detector alone (per-class NMS) | 108 | 95 | 12 | 1 | 3 | 0.880 | 0.864 | 0.880 | 0.872 | 0.994 |
| Detector alone, one box per face | 108 | 95 | 12 | 1 | 0 | 0.880 | 0.888 | 0.880 | 0.884 | 0.994 |
| Detector boxes + classifier names | 108 | 89 | 18 | 1 | 0 | 0.824 | 0.832 | 0.824 | 0.828 | 0.994 |
| Detector boxes + detector and classifier votes combined | 108 | 98 | 9 | 1 | 0 | 0.907 | 0.916 | 0.907 | 0.912 | 0.994 |

Test:

| setting | faces | correct | wrong_identity | missed | extra_boxes | accuracy | precision | recall | f1 | mean_iou |
|---|---|---|---|---|---|---|---|---|---|---|
| Milestone 3 detector alone (per-class NMS) | 108 | 84 | 24 | 0 | 8 | 0.778 | 0.724 | 0.778 | 0.750 | 0.993 |
| Detector alone, one box per face | 108 | 84 | 24 | 0 | 0 | 0.778 | 0.778 | 0.778 | 0.778 | 0.993 |
| Detector boxes + classifier names | 108 | 80 | 28 | 0 | 0 | 0.741 | 0.741 | 0.741 | 0.741 | 0.991 |
| Detector boxes + detector and classifier votes combined | 108 | 89 | 19 | 0 | 0 | 0.824 | 0.824 | 0.824 | 0.824 | 0.991 |

Chosen on validation: `fusion`.
