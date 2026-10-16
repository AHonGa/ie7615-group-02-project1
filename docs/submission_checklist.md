**Repo:** https://github.com/AHonGa/ie7615-group-02-project1

# Milestone 2 — Canvas Submission Checklist

- [x] Synthetic detection dataset built and committed (`detection_dataset/`)
- [x] Data-preparation notebook in repo (`notebooks/05_detection_dataset_preparation.ipynb`)
- [x] Dataset documentation written (`docs/milestone2_detection_dataset.md`)
- [x] Repo set to public per direct confirmation from Priyanka/TA
- [X] All team members have an individual commit in `main`
- [X] Final commit hash recorded for submission
- [X] Submitted via Canvas: repo link + commit hash

---

# Milestone 3 — Canvas Submission Checklist

- [x] Trained YOLOv8 checkpoint in repo (`models/celeba_yolov8_best.pt`)
- [x] Detection-results document (`docs/milestone3_detection_results.md` + PDF): metrics table, sweep summary, loss curves, interpretation
- [x] Hyperparameter sweep: 6 training runs + confidence/NMS threshold sweep (`results/milestone3/`)
- [x] Detection gallery, 19 captioned examples with successes and failures (`results/milestone3/gallery/`)
- [x] Live demo with run instructions in README (`src/demo_detect.py`, `notebooks/06_yolov8_live_demo.ipynb`)
- [x] Reproduction notebook (`notebooks/06_yolov8_training_evaluation.ipynb`)
- [x] Original-dataset baseline kept (`runs/detect/celeba_yolov8n_milestone2/`) and evaluation review (`docs/milestone3_evaluation_review.md`)
- [x] Larger dataset approved by the TA (Oct 6)
- [x] All team members have an individual Milestone 3 commit in `main`
- [x] Final commit hash recorded for submission (added as comment in Canvas submission)
- [x] Submitted via Canvas: repo link + commit hash

---

# Milestone 4 — Final Submission Checklist

**Deadlines:** presentation **Mon Oct 19, 8:00 PM EST** (all 4 members attend) · Canvas submission **Wed Oct 21, 11:00 PM EST**

### Integrated pipeline
- [x] One command runs detection + identification: `python src/identify.py --source <image or folder>`
- [x] Live demo web app: `python src/identify.py --app` (`--share` for a public link)
- [x] Full Milestone 4 run in one command: `python src/run_milestone4.py` (or `notebooks/07_integrated_pipeline.ipynb`)
- [x] Pipeline code: `src/m4_pipeline.py` (YOLOv8 detector + ResNet18 classifier + vote)
- [x] Retrained 13-identity classifier committed: `models/celeba_resnet18_13id.pt`
- [x] Results: `results/milestone4/results_tables.md`, `results/milestone4/summary.json`

### Final gallery (`results/milestone4/gallery/`, captions in `README.md`)
- [x] ≥ 20 correctly identified single faces — **20**
- [x] ≥ 5 misclassified single faces with analysis — **11**
- [x] ≥ 10 multi-face images with boxes and names — **19**

### Final report
- [x] `docs/Final_Report_Group02.pdf` — 10 pages, with front page
- [x] Ethics section (bias, privacy/consent, misuse), at least one page
- [x] Ethics wording reviewed (GDPR/BIPA softened after review)
- [x] Front page matches the TA's required format (format requested from the TA)

### Presentation
- [x] Slides ready (15 slides), every member presents a part
- [x] Speaker scripts and Q&A study sheet shared with the team
- [ ] Demo link created before 8:00 PM on Oct 19 and shared with the presenter
- [ ] All 4 members join the meeting on time

### Repository
- [x] README has the Milestone 4 section and presentation-readiness notes
- [x] PR #5 reviewed and merged into `main`
- [x] All team members have an individual Milestone 4 commit or review on GitHub
- [x] Release created: tag `v1.0-project1`, target `main` (after this checklist is merged)

### Canvas submission
- [x] Final report PDF uploaded
- [x] Repository link at the release: https://github.com/AHonGa/ie7615-group-02-project1/releases/tag/v1.0-project1
- [x] Presentation slides uploaded
- [x] Submitted before Wed Oct 21, 11:00 PM EST