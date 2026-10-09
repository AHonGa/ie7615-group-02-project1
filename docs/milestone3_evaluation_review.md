# Milestone 3 Evaluation Review

Reviewer: Samuel Tong

## What I reviewed

I reviewed the final Milestone 3 YOLOv8 results to make sure the numbers,
testing, and final conclusions all made sense and matched the milestone
requirements. I mainly focused on the detection metrics, hyperparameter sweep, error patterns,
gallery, demo, and overall completeness of the submission.

## Results review

The final selected YOLOv8n model reported:
- mAP@0.5: 0.879
- mAP@0.5:0.95: 0.876
- Precision: 0.849
- Recall: 0.759
- Mean IoU: 0.993
- Faces detected: 108/108
- Faces correctly identified: 84/108

The main takeaway I saw is that the model is very good at finding the faces and
drawing the boxes in the correct location, but the harder part is identifying
the correct celebrity.

The 108/108 face detection result and 0.993 mean IoU show that localization is
working very well. Most of the remaining errors are identity mix-ups rather
than missed faces.

## Dataset comparison

I also reviewed the comparison between the original Milestone 2 dataset and the
larger Milestone 3 dataset.

The original dataset did not have enough different photos per celebrity, so the
model learned where the faces were but had more trouble learning the identity.

After increasing the number of photos per celebrity and making sure the same
photo was not shared between train, validation, and test, the model performance
improved a lot.

This supports the idea that the amount and variety of training data had a bigger
impact than just using a larger model.

## Hyperparameter sweep

I reviewed the different configurations that were tested, including:

- learning rate
- image size
- augmentation
- YOLOv8n vs YOLOv8s
- confidence threshold
- NMS IoU threshold

The final YOLOv8n model with lighter augmentation was selected based on
validation performance instead of choosing whichever model had the highest test
score. I believe this was the right approach because the test set should mainly be used
for the final evaluation and not for choosing the model. The confidence threshold results also show the expected tradeoff. As confidence
goes up, precision improves but recall starts to drop.

## Error review

The biggest issue I noticed is still celebrity identity confusion. The model usually finds the correct face location, but some faces are assigned
the wrong identity. This is more noticeable for celebrities that look similar
or still have a limited number of different training photos.

## Final check

I checked that the project includes the main Milestone 3 requirements:

- trained YOLOv8 model
- overall detection metrics
- per-class results
- hyperparameter testing
- training and validation curves
- detection gallery
- live demo
- instructions to reproduce the results
