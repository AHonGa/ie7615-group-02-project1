# Celebrity subset selection

Selected **4** identities from CelebA identities with between 30 and 45 images each, chosen to maximize spread across gender, hair color, eyewear, and facial-hair attributes (greedy max-min diversity over the attribute vectors in `list_attr_celeba.txt`).

| Identity ID | # Images | Dominant attributes |
|---|---|---|
| 3 | 25 | Male, Wearing_Hat |
| 7 | 24 | Male, Black_Hair |
| 1212 | 25 | Male, Black_Hair |
| 8335 | 25 | Male, Brown_Hair |

**Why these criteria:** sufficient per-identity images (>= 30) keeps train/val/test splits viable at a small scale; capping at the upper bound keeps classes roughly balanced so accuracy isn't dominated by the largest class; maximizing attribute-space distance ensures the model must learn genuinely discriminative facial features rather than relying on a single correlated cue (e.g., only hair color).
