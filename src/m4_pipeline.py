"""Milestone 4: integrated identification pipeline.

    image --> YOLOv8 detector (Milestone 3)  --> one box per face
          --> crop each box
          --> ResNet18 classifier (Milestone 1 architecture, retrained on all 13 identities)
          --> combine detector and classifier votes --> labelled boxes

The Milestone 1 classifier only knew Group 02's 4 identities, so it is retrained on the 13
class-pool identities with the SAME photo-level train/val/test split as the Milestone 3
detection dataset (detection_dataset_m3/photo_split.csv). A test photo is therefore never seen
by either model during training, and classifier class ids equal detector class ids.

Used by src/run_milestone4.py, src/identify.py and notebooks/07_integrated_pipeline.ipynb.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_detection as m3  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FACES_DIR = REPO_ROOT / "extracted_identities"
PHOTO_SPLIT = REPO_ROOT / "detection_dataset_m3" / "photo_split.csv"
CLASSES_JSON = REPO_ROOT / "detection_dataset_m3" / "classes.json"
DETECTOR_WEIGHTS = REPO_ROOT / "models" / "celeba_yolov8_best.pt"
CLASSIFIER_WEIGHTS = REPO_ROOT / "models" / "celeba_resnet18_13id.pt"
RESULTS_DIR = REPO_ROOT / "results" / "milestone4"

IMAGE_SIZE = 128                       # same as Milestone 1 (src/dataset.py)
NORM_MEAN = [0.485, 0.456, 0.406]      # ImageNet statistics, as in Milestone 1
NORM_STD = [0.229, 0.224, 0.225]
MODES = ("detector", "classifier", "fusion")
LOW_CONFIDENCE = 0.5                   # below this the demo marks a name as uncertain


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def class_map() -> dict[str, int]:
    """CelebA identity id (str) -> class id, shared by detector and classifier."""
    return {str(k): int(v) for k, v in json.loads(CLASSES_JSON.read_text(encoding="utf-8")).items()}


def class_names() -> dict[int, str]:
    return {cid: f"celeb_{celeb}" for celeb, cid in sorted(class_map().items(), key=lambda kv: kv[1])}


def photo_table() -> "pd.DataFrame":
    """One row per face photo: identity, class id, split, path."""
    import pandas as pd

    df = pd.read_csv(PHOTO_SPLIT, dtype={"identity_id": str})
    cmap = class_map()
    df["label"] = df["identity_id"].map(cmap).astype(int)
    df["path"] = [str(FACES_DIR / i / p) for i, p in zip(df["identity_id"], df["photo"])]
    return df


def split_counts() -> "pd.DataFrame":
    df = photo_table()
    out = df.groupby("split").agg(photos=("photo", "size"), identities=("label", "nunique")).reset_index()
    out["photos per identity"] = (out["photos"] / out["identities"]).round(1)
    order = {"train": 0, "val": 1, "test": 2}
    return out.sort_values("split", key=lambda s: s.map(order)).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Classifier (PyTorch)
# --------------------------------------------------------------------------- #
def _transforms(train: bool):
    from torchvision import transforms as T

    base = [T.Resize((IMAGE_SIZE, IMAGE_SIZE))]  # square resize, exactly like a detector crop
    if train:
        # Mimic what crops from the detector look like: small shifts/zoom (box jitter),
        # flips and lighting changes. Rotation stays small so faces remain upright.
        base += [T.RandomHorizontalFlip(),
                 T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                 T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2)]
    return T.Compose(base + [T.ToTensor(), T.Normalize(NORM_MEAN, NORM_STD)])


class _FacePhotos:
    """torch Dataset over photo_table() rows."""

    def __init__(self, rows, train: bool):
        from PIL import Image

        self.Image = Image
        self.paths = list(rows["path"])
        self.labels = [int(v) for v in rows["label"]]
        self.tf = _transforms(train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(self.Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def _m1_models():
    """Load src/models.py (Milestone 1 architectures) by path. The repo also has a models/ folder
    holding weights, so a plain `import models` can resolve to the wrong thing in a notebook."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("m1_models", Path(__file__).resolve().parent / "models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resnet18(num_classes: int, pretrained: bool):
    """Milestone 1 ResNet18 (finetune mode). Without `pretrained` no ImageNet download is needed,
    which is what loading a trained checkpoint wants (the live demo then works offline)."""
    if pretrained:
        return _m1_models().build_resnet18(num_classes, mode="finetune")
    import torch.nn as nn
    from torchvision import models as tvm

    model = tvm.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _device(device=None):
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_classifier(epochs: int = 30, lr: float = 3e-4, batch: int = 32, seed: int = 42,
                     limit: int | None = None, device=None, out: Path = CLASSIFIER_WEIGHTS) -> "pd.DataFrame":
    """Fine-tune the Milestone 1 ResNet18 (src/models.build_resnet18) on all 13 identities.

    The checkpoint with the best validation accuracy is saved. `limit` keeps only a few photos
    per split (smoke test). Returns the per-epoch history.
    """
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = _device(device)

    df = photo_table()
    if limit:
        df = df.groupby("split", group_keys=False).head(limit)
    train_rows = df[df.split == "train"]
    # a final batch of exactly one image would break BatchNorm in training mode
    train_dl = DataLoader(_FacePhotos(train_rows, True), batch_size=batch, shuffle=True, num_workers=0,
                          drop_last=len(train_rows) % batch == 1, generator=torch.Generator().manual_seed(seed))
    val_dl = DataLoader(_FacePhotos(df[df.split == "val"], False), batch_size=64, num_workers=0)

    n_classes = len(class_map())
    model = _resnet18(n_classes, pretrained=True).to(dev)  # ImageNet-pretrained, as in Milestone 1
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    history, best_acc = [], -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = tr_correct = tr_n = 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(y)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_n += len(y)
        sched.step()

        model.eval()
        va_loss = va_correct = va_n = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                va_loss += loss_fn(logits, y).item() * len(y)
                va_correct += (logits.argmax(1) == y).sum().item()
                va_n += len(y)
        row = {"epoch": epoch, "train_loss": tr_loss / tr_n, "train_acc": tr_correct / tr_n,
               "val_loss": va_loss / max(va_n, 1), "val_acc": va_correct / max(va_n, 1)}
        history.append(row)
        print(f"epoch {epoch:3d}  train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f}  "
              f"val loss {row['val_loss']:.3f} acc {row['val_acc']:.3f}", flush=True)
        if row["val_acc"] > best_acc:
            best_acc = row["val_acc"]
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "num_classes": n_classes,
                        "class_names": [class_names()[i] for i in range(n_classes)],
                        "image_size": IMAGE_SIZE, "best_epoch": epoch, "best_val_acc": best_acc}, out)
    hist = pd.DataFrame(history)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hist.to_csv(RESULTS_DIR / "classifier_history.csv", index=False)
    return hist


class Classifier:
    """Loads the 13-identity ResNet18 and returns class probabilities for RGB face crops."""

    def __init__(self, weights: Path = CLASSIFIER_WEIGHTS, device=None):
        import torch

        self.torch = torch
        self.dev = _device(device)
        ckpt = torch.load(weights, map_location=self.dev)
        self.model = _resnet18(ckpt["num_classes"], pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.dev).eval()
        self.tf = _transforms(train=False)
        self.best_epoch = ckpt.get("best_epoch")

    def probs(self, images) -> np.ndarray:
        """images: list of PIL RGB images -> (N, 13) softmax probabilities."""
        if not images:
            return np.zeros((0, len(class_map())))
        torch = self.torch
        x = torch.stack([self.tf(im.convert("RGB")) for im in images]).to(self.dev)
        with torch.no_grad():
            return torch.softmax(self.model(x), dim=1).cpu().numpy()


def evaluate_classifier(clf, split: str) -> "pd.DataFrame":
    """Classify every face photo of a split; one row per photo with top-1/top-2 and correctness."""
    import pandas as pd
    from PIL import Image

    names = class_names()
    df = photo_table().query("split == @split").reset_index(drop=True)
    probs = np.concatenate([clf.probs([Image.open(p) for p in df["path"][i:i + 64]])
                            for i in range(0, len(df), 64)]) if len(df) else np.zeros((0, len(names)))
    order = np.argsort(-probs, axis=1)
    out = df[["identity_id", "label", "photo", "path"]].copy()
    out["true"] = [names[c] for c in out["label"]]
    out["pred_label"] = order[:, 0]
    out["pred"] = [names[c] for c in order[:, 0]]
    out["conf"] = probs[np.arange(len(df)), order[:, 0]]
    out["second"] = [names[c] for c in order[:, 1]]
    out["second_conf"] = probs[np.arange(len(df)), order[:, 1]]
    out["true_prob"] = probs[np.arange(len(df)), out["label"]]
    out["correct"] = out["pred_label"] == out["label"]
    return out


def per_identity_accuracy(evals: "pd.DataFrame") -> "pd.DataFrame":
    g = evals.groupby("true")
    out = g.agg(photos=("correct", "size"), correct=("correct", "sum")).reset_index()
    out["accuracy"] = out["correct"] / out["photos"]
    out["identity"] = [t + (" (ours)" if m3.is_our_group(t) else "") for t in out["true"]]
    key = out["true"].str.replace("celeb_", "").astype(int)
    return out.assign(_k=key).sort_values("_k")[["identity", "photos", "correct", "accuracy"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Integrated pipeline
# --------------------------------------------------------------------------- #
class Pipeline:
    """Detector + classifier. `detect(image)` returns one dict per face."""

    def __init__(self, detector_weights: Path = DETECTOR_WEIGHTS, classifier=None,
                 classifier_weights: Path = CLASSIFIER_WEIGHTS, device=None):
        from ultralytics import YOLO

        self.yolo = YOLO(str(detector_weights))
        self.clf = classifier if classifier is not None else Classifier(classifier_weights, device=device)
        self.names = class_names()
        self.n = len(self.names)

    # -- detector ---------------------------------------------------------- #
    def _yolo(self, bgr: np.ndarray, conf: float, iou: float, agnostic: bool):
        r = self.yolo.predict(source=bgr, conf=conf, iou=iou, imgsz=640, agnostic_nms=agnostic, verbose=False)[0]
        return (m3._np(r.boxes.xyxy).reshape(-1, 4).astype(float), m3._np(r.boxes.cls).astype(int).reshape(-1),
                m3._np(r.boxes.conf).reshape(-1).astype(float))

    def find_faces(self, bgr: np.ndarray, conf: float = m3.DEFAULT_CONF, iou: float = m3.DEFAULT_NMS_IOU,
                   agnostic: bool = True, fallback: bool = True):
        """Boxes from the detector. The detector was trained on faces that fill one cell of a 3x3
        grid; if it finds nothing (e.g. a single full-frame portrait), the image is shrunk into one
        cell-sized patch of a blank canvas and searched again, and as a last resort the whole image
        is treated as one face. Returns (boxes, det_cls, det_conf, source)."""
        boxes, cls, cf = self._yolo(bgr, conf, iou, agnostic)
        if len(boxes) or not fallback:
            return boxes, cls, cf, ["detector"] * len(boxes)
        import cv2

        h, w = bgr.shape[:2]
        s = 200 / max(h, w)
        small = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))))
        canvas = np.full((640, 640, 3), 128, dtype=bgr.dtype)
        oy, ox = (640 - small.shape[0]) // 2, (640 - small.shape[1]) // 2
        canvas[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
        boxes, cls, cf = self._yolo(canvas, conf, iou, agnostic)
        if len(boxes):
            boxes = (boxes - [ox, oy, ox, oy]) / s
            boxes = np.clip(boxes, 0, [w, h, w, h])
            return boxes, cls, cf, ["detector (rescaled)"] * len(boxes)
        return np.array([[0, 0, w, h]], dtype=float), np.array([-1]), np.array([0.0]), ["whole image"]

    # -- full pipeline ----------------------------------------------------- #
    def detect(self, bgr: np.ndarray, mode: str = "fusion", conf: float = m3.DEFAULT_CONF,
               iou: float = m3.DEFAULT_NMS_IOU, agnostic: bool = True, fallback: bool = True) -> list[dict]:
        from PIL import Image

        boxes, det_cls, det_conf, source = self.find_faces(bgr, conf, iou, agnostic, fallback)
        h, w = bgr.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes:
            x1, y1 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
            x2, y2 = int(min(w, np.ceil(x2))), int(min(h, np.ceil(y2)))
            crop = bgr[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1), ::-1]
            crops.append(Image.fromarray(np.ascontiguousarray(crop)))
        probs = self.clf.probs(crops)
        out = []
        for i in range(len(boxes)):
            p = probs[i]
            clf_cls = int(np.argmax(p))
            d_cls, d_conf = int(det_cls[i]), float(det_conf[i])
            onehot = np.zeros(self.n)
            if d_cls >= 0:
                onehot[d_cls] = d_conf
            fused = 0.5 * (p + onehot)
            if mode == "detector" and d_cls >= 0:
                label, score = d_cls, d_conf
            elif mode == "classifier" or d_cls < 0:
                label, score = clf_cls, float(p[clf_cls])
            else:  # fusion: average of the classifier distribution and the detector's vote
                label = int(np.argmax(fused))
                score = float(fused[label]) / (0.5 * (1 + d_conf))
            out.append({"box": boxes[i], "label": label, "name": self.names[label], "score": score,
                        "det_label": d_cls, "det_name": self.names.get(d_cls, "-"), "det_conf": d_conf,
                        "clf_label": clf_cls, "clf_name": self.names[clf_cls], "clf_conf": float(p[clf_cls]),
                        "source": source[i]})
        return out

    def evaluate(self, images: list[Path], mode: str, agnostic: bool = True, **kw) -> list:
        """Run on labelled grid images and match to ground truth with the Milestone 3 matcher."""
        import cv2

        results = []
        for path in images:
            bgr = cv2.imread(str(path))
            h, w = bgr.shape[:2]
            gt_cls, gt_boxes = m3.read_yolo_labels(m3.label_path_for(path), w, h)
            dets = self.detect(bgr, mode=mode, agnostic=agnostic, fallback=False, **kw)
            pr_boxes = np.array([d["box"] for d in dets]).reshape(-1, 4)
            pr_cls = np.array([d["label"] for d in dets], dtype=int)
            pr_conf = np.array([d["score"] for d in dets], dtype=float)
            st, pg, pi, gs = m3.match_predictions(gt_cls, gt_boxes, pr_cls, pr_boxes, pr_conf)
            results.append(m3.ImageResult(path, gt_cls, gt_boxes, pr_cls, pr_boxes, pr_conf, st, pg, pi, gs))
        return results


def compare_modes(pipe: Pipeline, split: str, dataset_dir: Path = m3.SCALED_DATASET_DIR) -> "pd.DataFrame":
    """Detector-only (Milestone 3 setting and with class-agnostic NMS), classifier-only and fusion."""
    import pandas as pd

    images = m3.list_images(split, dataset_dir)
    rows = []
    for mode, agnostic, desc in [("detector", False, "Milestone 3 detector alone (per-class NMS)"),
                                 ("detector", True, "Detector alone, one box per face"),
                                 ("classifier", True, "Detector boxes + classifier names"),
                                 ("fusion", True, "Detector boxes + detector and classifier votes combined")]:
        s = m3.summarize(pipe.evaluate(images, mode, agnostic=agnostic))
        rows.append({"setting": desc, "mode": mode, "agnostic_nms": agnostic, "faces": s["faces"],
                     "correct": s["correct"], "wrong_identity": s["wrong_identity"], "missed": s["missed"],
                     "extra_boxes": s["false_positive"], "accuracy": s["correct"] / s["faces"],
                     "precision": s["precision"], "recall": s["recall"], "f1": s["f1"], "mean_iou": s["mean_iou"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_classifier_history(hist, out: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="train")
    ax[0].plot(hist["epoch"], hist["val_loss"], label="validation")
    ax[0].set_title("classifier loss")
    ax[1].plot(hist["epoch"], hist["train_acc"], label="train")
    ax[1].plot(hist["epoch"], hist["val_acc"], label="validation")
    ax[1].set_title("classifier accuracy")
    ax[1].set_ylim(0, 1.02)
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_confusion(evals, out: Path, title: str) -> Path:
    import matplotlib.pyplot as plt

    names = class_names()
    n = len(names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(evals["label"], evals["pred_label"]):
        cm[int(t), int(p)] += 1
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    labels = [names[i].replace("celeb_", "") for i in range(n)]
    ax.set_xticks(range(n), labels, rotation=90)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("predicted identity")
    ax.set_ylabel("true identity")
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def draw_detections(bgr: np.ndarray, dets: list[dict]) -> np.ndarray:
    """Boxes with the final name; uncertain names get a '?' and an orange box."""
    import cv2

    img = bgr.copy()
    thick = max(2, img.shape[0] // 320)
    fs = max(0.42, img.shape[0] / 1500)
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        unsure = d["score"] < LOW_CONFIDENCE
        color = (0, 140, 255) if unsure else (40, 170, 40)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        label = f"{d['name']}{'?' if unsure else ''} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        lx, ty = max(0, min(x1, img.shape[1] - tw - 4)), max(y1, th + 4)
        cv2.rectangle(img, (lx, ty - th - 4), (lx + tw + 4, ty), color, -1)
        cv2.putText(img, label, (lx + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _thumb(path_or_img, size=(160, 196)):
    import cv2

    img = cv2.imread(str(path_or_img)) if isinstance(path_or_img, (str, Path)) else path_or_img
    return cv2.resize(img, size)


def _banner(img, text, color):
    import cv2

    bar = np.full((26, img.shape[1], 3), color, dtype=np.uint8)
    cv2.putText(bar, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def single_face_card(row, correct: bool) -> np.ndarray:
    """Test photo with true/predicted names; misclassified cards also show a training photo of
    the predicted identity so the look-alike can be judged by eye."""
    tile = _banner(_thumb(row["path"]), f"true: {row['true']}", (60, 60, 60))
    pred = _banner(np.zeros((196, 160, 3), np.uint8), f"pred: {row['pred']} {row['conf']:.2f}",
                   (40, 150, 40) if correct else (30, 30, 200))[:26]
    card = np.vstack([tile, pred])
    if not correct:
        df = photo_table()
        ref = df[(df.label == row["pred_label"]) & (df.split == "train")]["path"]
        if len(ref):
            ref_tile = _banner(_thumb(ref.iloc[0]), f"a train photo of {row['pred']}", (90, 60, 30))
            card = np.hstack([card, np.vstack([ref_tile, np.full((26, 160, 3), 255, np.uint8)])])
    return card


def misclassification_note(row) -> str:
    """Short automatic analysis of one wrong single-face prediction."""
    if row["conf"] < LOW_CONFIDENCE:
        kind = (f"Unsure mistake: top score only {row['conf']:.2f}; the true identity was the second guess."
                if row["second"] == row["true"] else
                f"Unsure mistake: top score only {row['conf']:.2f}, true identity scored {row['true_prob']:.2f}.")
    else:
        kind = (f"Confident mistake ({row['conf']:.2f}); the true identity scored only {row['true_prob']:.2f}.")
    return (f"**{row['true']}** predicted as **{row['pred']}**. {kind} Compare with the training photo of "
            f"{row['pred']} on the right: similar pose, hair or lighting is the usual cause with ~17 training "
            f"photos per identity.")


def make_stress_photos(rows, out_dir: Path) -> "pd.DataFrame":
    """Harder copies of test photos (dark, blurred, low resolution) for the misclassification gallery."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for _, r in rows.iterrows():
        img = cv2.imread(r["path"])
        h, w = img.shape[:2]
        for tag, im in (("dark", np.clip(img.astype(np.float32) * 0.3, 0, 255).astype(np.uint8)),
                        ("blur", cv2.GaussianBlur(img, (0, 0), 3.5)),
                        ("lowres", cv2.resize(cv2.resize(img, (w // 6, h // 6)), (w, h)))):
            p = out_dir / f"{r['identity_id']}_{Path(r['photo']).stem}_{tag}.jpg"
            cv2.imwrite(str(p), im)
            made.append({**r.to_dict(), "path": str(p), "photo": p.name, "variant": tag})
    import pandas as pd

    return pd.DataFrame(made)


def evaluate_photo_rows(clf, rows) -> "pd.DataFrame":
    """Like evaluate_classifier, but for an arbitrary table of photos (e.g. stress copies)."""
    from PIL import Image

    names = class_names()
    probs = clf.probs([Image.open(p) for p in rows["path"]])
    order = np.argsort(-probs, axis=1)
    out = rows.copy().reset_index(drop=True)
    idx = np.arange(len(out))
    out["true"] = [names[c] for c in out["label"]]
    out["pred_label"] = order[:, 0]
    out["pred"] = [names[c] for c in order[:, 0]]
    out["conf"] = probs[idx, order[:, 0]]
    out["second"] = [names[c] for c in order[:, 1]]
    out["second_conf"] = probs[idx, order[:, 1]]
    out["true_prob"] = probs[idx, out["label"].to_numpy()]
    out["correct"] = out["pred_label"] == out["label"]
    return out
