#!/usr/bin/env python3
"""
DEP Level 3 -- Scene Classifier.

Builds two ResNet18 classifiers from aria_hard_cutout frames:
  - Pod classifier: 3 classes (gold, red, green) from seconds 5-20 of each episode
  - Cup classifier: 3 classes (blue, red, green) from seconds 0-15 of each episode

Workflow:
  1. Extract aria_hard_cutout frames from all_aria datasets at specific time windows
  2. Save extracted frames as image folders and upload to HuggingFace
  3. Train ResNet18 classifiers (ImageNet pretrained backbone + 3-class head)
  4. Validate with train/val split confusion matrix

Usage:
  source /opt/lerobot_venv/bin/activate

  # Step 1: Extract frames and build classifier datasets:
  python3 files/train_scene_classifier.py extract

  # Step 2: Train both classifiers:
  python3 files/train_scene_classifier.py train

  # Step 3: Evaluate:
  python3 files/train_scene_classifier.py eval

  # Or do everything at once:
  python3 files/train_scene_classifier.py all

  # Upload trained models + frame datasets to HF:
  python3 files/train_scene_classifier.py upload
"""

import argparse
import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FPS = 30
ORG = "RAPOB"

# ============================================================================
# Dataset definitions
# ============================================================================

POD_CLASSES = ["gold", "red", "green"]
CUP_CLASSES = ["blue", "red", "green"]

# Pod datasets: extract seconds 5-20 (frames 150-600 relative to episode start)
POD_SOURCES = {
    "gold": f"{ORG}/dep_coffee_pod_gold_all_aria",
    "red": f"{ORG}/dep_coffee_pod_red_all_aria",
    "green": f"{ORG}/dep_coffee_pod_green_all_aria",
}
POD_FRAME_START = 150   # second 5 at 30fps
POD_FRAME_END = 600     # second 20 at 30fps

# Cup datasets: skip first 50 frames (~1.7 s startup), then frames 50-450 (~13 s)
CUP_SOURCES = {
    "blue": f"{ORG}/dep_coffee_cup_blue_all_aria",
    "red": f"{ORG}/dep_coffee_cup_red_all_aria",
    "green": f"{ORG}/dep_coffee_cup_green_all_aria",
}
CUP_FRAME_START = 50    # skip first 50 frames (~1.7 s wasted startup)
CUP_FRAME_END = 450     # second 15 at 30fps

ARIA_CAMERA_KEY = "observation.images.aria_hard_cutout"

# HF repos for classifier datasets
POD_CLASSIFIER_DATASET_REPO = f"{ORG}/dep_classifier_pod_hard_cutout"
CUP_CLASSIFIER_DATASET_REPO = f"{ORG}/dep_classifier_cup_hard_cutout"

# HF repos for trained models
POD_CLASSIFIER_MODEL_REPO = f"{ORG}/dep_classifier_pod_model"
CUP_CLASSIFIER_MODEL_REPO = f"{ORG}/dep_classifier_cup_model"

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Output directories
OUTPUT_BASE = Path("outputs/scene_classifier")
POD_OUTPUT = OUTPUT_BASE / "pod"
CUP_OUTPUT = OUTPUT_BASE / "cup"
POD_FRAMES_DIR = OUTPUT_BASE / "pod_frames"
CUP_FRAMES_DIR = OUTPUT_BASE / "cup_frames"


# ============================================================================
# Frame extraction from LeRobot video datasets
# ============================================================================

def get_episode_metadata(dataset_root: Path):
    """Load episode metadata to get frame ranges and video file mappings."""
    import pandas as pd

    ep_dir = dataset_root / "meta" / "episodes"
    parquet_files = []
    for chunk_dir in sorted(ep_dir.iterdir()):
        if chunk_dir.is_dir():
            parquet_files.extend(sorted(chunk_dir.glob("*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(f"No episode metadata found in {ep_dir}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True)


def extract_frames_from_video(
    video_path: Path,
    from_timestamp: float,
    to_timestamp: float,
    frame_start: int,
    frame_end: int,
) -> list[np.ndarray]:
    """Extract specific frames from a video file using pyav.

    Args:
        video_path: Path to mp4 file
        from_timestamp: Start timestamp of this episode within the video
        to_timestamp: End timestamp of this episode within the video
        frame_start: First frame to extract (relative to episode start)
        frame_end: Last frame to extract (relative to episode start)

    Returns:
        List of numpy arrays (H, W, 3) uint8
    """
    frames = []

    # Calculate the actual timestamps we want within the video
    episode_duration = to_timestamp - from_timestamp
    episode_frames = int(episode_duration * FPS)

    # Clamp frame range to actual episode length
    actual_end = min(frame_end, episode_frames)
    if frame_start >= actual_end:
        return frames

    # Target timestamps relative to video start
    target_start_ts = from_timestamp + (frame_start / FPS)
    target_end_ts = from_timestamp + (actual_end / FPS)

    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]

        # Seek to just before our target start
        seek_ts = max(0, target_start_ts - 0.5)
        container.seek(int(seek_ts * av.time_base), any_frame=False)

        for frame in container.decode(video=0):
            ts = float(frame.pts * stream.time_base)

            if ts < target_start_ts - (0.5 / FPS):
                continue
            if ts > target_end_ts + (0.5 / FPS):
                break

            img = frame.to_ndarray(format="rgb24")
            frames.append(img)

        container.close()
    except Exception as e:
        log.warning(f"Error reading {video_path}: {e}")

    return frames


def extract_classifier_frames(
    sources: dict[str, str],
    classes: list[str],
    frame_start: int,
    frame_end: int,
    output_dir: Path,
    subsample: int = 3,
) -> dict:
    """Extract labeled frames from all_aria datasets for classifier training.

    Args:
        sources: Mapping of class_name -> repo_id
        classes: Ordered class names
        frame_start: Start frame within each episode
        frame_end: End frame within each episode
        output_dir: Where to save extracted frames as class_name/*.jpg
        subsample: Take every Nth frame to reduce dataset size (default: every 3rd)

    Returns:
        Dict with extraction stats per class
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    for class_name, repo_id in sources.items():
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)

        log.info(f"Extracting '{class_name}' from {repo_id}...")
        log.info(f"  Frame window: {frame_start}-{frame_end} "
                 f"({frame_start/FPS:.1f}s - {frame_end/FPS:.1f}s)")

        meta = LeRobotDatasetMetadata(repo_id)
        ep_df = get_episode_metadata(meta.root)

        # Video metadata columns for aria_hard_cutout
        video_chunk_col = f"videos/{ARIA_CAMERA_KEY}/chunk_index"
        video_file_col = f"videos/{ARIA_CAMERA_KEY}/file_index"
        video_from_col = f"videos/{ARIA_CAMERA_KEY}/from_timestamp"
        video_to_col = f"videos/{ARIA_CAMERA_KEY}/to_timestamp"

        if video_from_col not in ep_df.columns:
            log.warning(f"  No aria_hard_cutout video metadata in {repo_id}. Skipping.")
            continue

        total_frames_saved = 0

        for _, row in ep_df.iterrows():
            ep_idx = row["episode_index"]
            chunk_idx = row[video_chunk_col]
            file_idx = row[video_file_col]
            from_ts = row[video_from_col]
            to_ts = row[video_to_col]

            # Find the video file
            video_dir = meta.root / "videos" / ARIA_CAMERA_KEY / f"chunk-{chunk_idx:03d}"
            video_file = video_dir / f"file-{file_idx:03d}.mp4"

            if not video_file.exists():
                # Try legacy naming
                video_file = video_dir / f"episode_{ep_idx:06d}.mp4"
                if not video_file.exists():
                    log.warning(f"  Video not found for ep {ep_idx}")
                    continue

            # Extract frames from the target time window
            raw_frames = extract_frames_from_video(
                video_file, from_ts, to_ts, frame_start, frame_end
            )

            # Subsample and save as JPEG
            for i, img in enumerate(raw_frames):
                if i % subsample != 0:
                    continue
                frame_path = class_dir / f"ep{ep_idx:03d}_f{frame_start + i:05d}.jpg"
                Image.fromarray(img).save(frame_path, quality=90)
                total_frames_saved += 1

        log.info(f"  Saved {total_frames_saved} frames for '{class_name}'")
        stats[class_name] = total_frames_saved

    # Summary
    total = sum(stats.values())
    log.info(f"Total frames extracted: {total}")
    return stats


# ============================================================================
# Dataset and model classes
# ============================================================================

class ImageFolderDataset(Dataset):
    """Simple image folder dataset. Expects: root/class_name/*.jpg"""

    def __init__(self, root: Path, classes: list[str], transform=None):
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = []

        for class_name in classes:
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = transforms.ToTensor()(img)  # -> (3, H, W) float [0,1]
        if self.transform:
            img = self.transform(img)
        return img, label


class TransformSubset(Dataset):
    """Wraps a Subset with a specific transform applied after loading."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class SceneClassifier(nn.Module):
    """ResNet18-based scene classifier."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict(self, image: torch.Tensor, classes: list[str]) -> tuple[str, float]:
        """Predict class from a single image tensor."""
        self.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        with torch.no_grad():
            logits = self(image)
            probs = torch.softmax(logits, dim=-1)
            confidence, idx = probs.max(dim=-1)
        return classes[idx.item()], confidence.item()


# ============================================================================
# Transforms
# ============================================================================

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_cup_train_transform():
    """Colour-preserving augmentation for cup colour classification.

    Uses minimal hue jitter (0.02) to preserve red / green / blue identity
    while still varying brightness and saturation to handle real lighting
    differences.  Stronger saturation range (0.4) forces the backbone to
    generalise colour rather than texture.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.4, hue=0.02),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================================
# Training
# ============================================================================

def train_classifier(
    frames_dir: Path,
    classes: list[str],
    output_dir: Path,
    classifier_name: str,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    val_split: float = 0.15,
    device: str = "cuda",
    class_weights: list[float] | None = None,
    train_transform_fn=None,
):
    """Train a scene classifier on extracted frames.

    Args:
        class_weights: Optional per-class loss weights, e.g. [1.0, 2.0, 1.0] to
            upweight an under-predicted class.  Must match len(classes).
        train_transform_fn: Callable that returns a transforms.Compose when
            called with no args.  Defaults to get_train_transform.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load full dataset (no transforms -- applied per-split)
    full_dataset = ImageFolderDataset(frames_dir, classes)
    log.info(f"\n[{classifier_name}] Total samples: {len(full_dataset)}")

    # Class distribution
    label_counts = Counter(label for _, label in full_dataset.samples)
    for cls in classes:
        idx = full_dataset.class_to_idx[cls]
        log.info(f"  {cls}: {label_counts.get(idx, 0)} frames")

    if len(full_dataset) == 0:
        log.error(f"No frames found in {frames_dir}. Run 'extract' first.")
        return None

    # Balance classes by undersampling to the smallest class size
    samples_by_class: dict[int, list] = defaultdict(list)
    for i, (_, label) in enumerate(full_dataset.samples):
        samples_by_class[label].append(i)

    min_class_size = min(len(v) for v in samples_by_class.values())
    log.info(f"  Balancing: undersampling all classes to {min_class_size} samples")

    balanced_indices = []
    rng = random.Random(42)
    for label_idx in sorted(samples_by_class.keys()):
        indices = samples_by_class[label_idx]
        rng.shuffle(indices)
        balanced_indices.extend(indices[:min_class_size])

    rng.shuffle(balanced_indices)
    balanced_subset = torch.utils.data.Subset(full_dataset, balanced_indices)
    log.info(f"  Balanced total: {len(balanced_subset)} ({min_class_size} x {len(classes)} classes)")

    # Split train/val from balanced subset
    n_val = max(1, int(len(balanced_subset) * val_split))
    n_train = len(balanced_subset) - n_val
    train_subset, val_subset = random_split(
        balanced_subset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    _train_tfm = train_transform_fn if train_transform_fn is not None else get_train_transform
    train_dataset = TransformSubset(train_subset, _train_tfm())
    val_dataset = TransformSubset(val_subset, get_eval_transform())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    log.info(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = SceneClassifier(num_classes=len(classes)).to(device_obj)
    if class_weights is not None:
        _cw = torch.tensor(class_weights, dtype=torch.float).to(device_obj)
        criterion = nn.CrossEntropyLoss(weight=_cw)
        log.info(f"  Using class weights: { {c: w for c, w in zip(classes, class_weights)} }")
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device_obj), labels.to(device_obj)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device_obj), labels.to(device_obj)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 4),
        })

        log.info(
            f"  Epoch {epoch+1:>3d}/{epochs} -- "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} -- "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "classifier_name": classifier_name,
                "val_acc": val_acc,
                "epoch": epoch + 1,
            }, output_dir / "best_model.pth")
            log.info(f"    ** New best: {val_acc:.4f}")

    # Save final + history
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "classifier_name": classifier_name,
        "val_acc": val_acc,
        "epoch": epochs,
    }, output_dir / "final_model.pth")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info(f"\n[{classifier_name}] Best val accuracy: {best_val_acc:.4f}")
    return model


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_classifier(
    checkpoint_path: Path,
    frames_dir: Path,
    classes: list[str],
    classifier_name: str,
    device: str = "cuda",
):
    """Evaluate a trained classifier with confusion matrix."""
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)

    model = SceneClassifier(num_classes=len(classes)).to(device_obj)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    log.info(f"\n[{classifier_name}] Evaluating {checkpoint_path}")
    log.info(f"  Trained {ckpt['epoch']} epochs, val_acc={ckpt['val_acc']:.4f}")

    dataset = ImageFolderDataset(frames_dir, classes, transform=get_eval_transform())
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    correct, total = 0, 0
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device_obj), labels.to(device_obj)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            for t, p in zip(labels, predicted):
                confusion[t.item()][p.item()] += 1

    acc = correct / total if total > 0 else 0
    log.info(f"\n  Overall accuracy: {acc:.4f} ({correct}/{total})")
    log.info(f"\n  Confusion matrix:")
    header = "  " + f"{'':>10s} " + " ".join(f"{c:>8s}" for c in classes)
    log.info(header)
    for i, cls in enumerate(classes):
        row = " ".join(f"{confusion[i][j].item():>8d}" for j in range(len(classes)))
        log.info(f"  {cls:>10s} {row}")

    log.info(f"\n  Per-class accuracy:")
    for i, cls in enumerate(classes):
        cls_total = confusion[i].sum().item()
        cls_correct = confusion[i][i].item()
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0
        log.info(f"    {cls}: {cls_acc:.4f} ({cls_correct}/{cls_total})")


# ============================================================================
# Upload to HuggingFace
# ============================================================================

def upload_frames_as_mp4(frames_dir: Path, repo_id: str, classes: list[str]):
    """Encode per-class JPEG frames into MP4 videos and upload to HF.

    Uploading a handful of MP4 files is orders of magnitude faster than
    uploading thousands of individual JPEGs.  One H.264 MP4 per class plus a
    metadata.json are committed to the dataset repo.
    """
    import tempfile
    from huggingface_hub import HfApi, create_repo

    log.info(f"Encoding frames as MP4 for {repo_id}...")
    create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api = HfApi()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        metadata: dict = {"classes": classes, "fps": FPS, "class_stats": {}}

        for class_name in classes:
            class_dir = frames_dir / class_name
            if not class_dir.exists():
                log.warning(f"  Class dir not found: {class_dir}")
                continue

            jpg_files = sorted(class_dir.glob("*.jpg"))
            if not jpg_files:
                log.warning(f"  No frames for class '{class_name}'")
                continue

            log.info(f"  Encoding {len(jpg_files)} frames → {class_name}.mp4 ...")

            # Dimensions from first frame
            first = Image.open(jpg_files[0]).convert("RGB")
            w, h = first.size

            out_mp4 = tmp_path / f"{class_name}.mp4"
            container = av.open(str(out_mp4), mode="w")
            stream = container.add_stream("libx264", rate=FPS)
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "fast"}

            for jpg_path in jpg_files:
                img_arr = np.array(Image.open(jpg_path).convert("RGB"))
                frame = av.VideoFrame.from_ndarray(img_arr, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()

            size_mb = out_mp4.stat().st_size / 1e6
            log.info(f"    {out_mp4.name}: {size_mb:.1f} MB")
            metadata["class_stats"][class_name] = {
                "frames": len(jpg_files),
                "video_file": f"{class_name}.mp4",
            }

        with open(tmp_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        log.info(f"  Uploading {len(classes)} MP4(s) + metadata to {repo_id} ...")
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Classifier frames as MP4 ({', '.join(classes)})",
        )

    log.info(f"  Uploaded: https://huggingface.co/datasets/{repo_id}")


def upload_frames_dataset(frames_dir: Path, repo_id: str, classes: list[str]):
    """Upload classifier frames to HF as MP4 videos (kept for back-compat)."""
    upload_frames_as_mp4(frames_dir, repo_id, classes)


def upload_model(output_dir: Path, repo_id: str):
    """Upload trained classifier model to HF."""
    from huggingface_hub import HfApi, create_repo

    log.info(f"Uploading model to {repo_id}...")
    create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

    api = HfApi()
    for fname in ["best_model.pth", "training_history.json"]:
        fpath = output_dir / fname
        if fpath.exists():
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="model",
            )
    log.info(f"  Uploaded: https://huggingface.co/{repo_id}")


# ============================================================================
# Inference helpers (for orchestrator integration)
# ============================================================================

def load_classifier(checkpoint_path: str, device: str = "cuda") -> tuple[SceneClassifier, list[str]]:
    """Load a trained classifier for inference.

    Returns:
        (model, classes) tuple
    """
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    model = SceneClassifier(num_classes=len(ckpt["classes"])).to(device_obj)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["classes"]


def classify_frame(
    model: SceneClassifier,
    frame: torch.Tensor,
    classes: list[str],
    device: str = "cuda",
) -> tuple[str, float]:
    """Classify a single aria frame.

    Args:
        model: Trained SceneClassifier
        frame: (C, H, W) or (H, W, C) tensor from aria_hard_cutout
        classes: Class names list
        device: Device for inference

    Returns:
        (class_name, confidence)
    """
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    transform = get_eval_transform()

    if frame.dtype == torch.uint8:
        frame = frame.float() / 255.0
    if frame.dim() == 3 and frame.shape[-1] == 3:
        frame = frame.permute(2, 0, 1)

    frame = transform(frame).to(device_obj)
    return model.predict(frame, classes)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser_arg = argparse.ArgumentParser(
        description="DEP Scene Classifier -- extract, train, eval, upload"
    )
    parser_arg.add_argument(
        "command",
        choices=["extract", "train", "eval", "upload", "all"],
        help="Command to run",
    )
    parser_arg.add_argument("--epochs", type=int, default=30)
    parser_arg.add_argument("--batch-size", type=int, default=32)
    parser_arg.add_argument("--lr", type=float, default=1e-4)
    parser_arg.add_argument("--device", type=str, default="cuda")
    parser_arg.add_argument(
        "--subsample", type=int, default=3,
        help="Take every Nth frame during extraction (default: 3)"
    )
    parser_arg.add_argument(
        "--which", choices=["pod", "cup", "both"], default="both",
        help="Which classifier to operate on"
    )
    args = parser_arg.parse_args()

    do_extract = args.command in ("extract", "all")
    do_train = args.command in ("train", "all")
    do_eval = args.command in ("eval", "all")
    do_upload = args.command == "upload"

    # ---- EXTRACT ----
    if do_extract:
        log.info("=" * 70)
        log.info("EXTRACTING CLASSIFIER FRAMES")
        log.info("=" * 70)

        if args.which in ("pod", "both"):
            log.info("\n--- POD CLASSIFIER (seconds 5-20) ---")
            if POD_FRAMES_DIR.exists():
                shutil.rmtree(POD_FRAMES_DIR)
            pod_stats = extract_classifier_frames(
                POD_SOURCES, POD_CLASSES, POD_FRAME_START, POD_FRAME_END,
                POD_FRAMES_DIR, subsample=args.subsample,
            )
            log.info(f"Pod stats: {pod_stats}")

        if args.which in ("cup", "both"):
            log.info("\n--- CUP CLASSIFIER (seconds 0-15) ---")
            if CUP_FRAMES_DIR.exists():
                shutil.rmtree(CUP_FRAMES_DIR)
            cup_stats = extract_classifier_frames(
                CUP_SOURCES, CUP_CLASSES, CUP_FRAME_START, CUP_FRAME_END,
                CUP_FRAMES_DIR, subsample=args.subsample,
            )
            log.info(f"Cup stats: {cup_stats}")

    # ---- TRAIN ----
    if do_train:
        log.info("\n" + "=" * 70)
        log.info("TRAINING CLASSIFIERS")
        log.info("=" * 70)

        if args.which in ("pod", "both"):
            log.info("\n--- POD CLASSIFIER ---")
            train_classifier(
                POD_FRAMES_DIR, POD_CLASSES, POD_OUTPUT, "pod_classifier",
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, device=args.device,
            )

        if args.which in ("cup", "both"):
            log.info("\n--- CUP CLASSIFIER ---")
            train_classifier(
                CUP_FRAMES_DIR, CUP_CLASSES, CUP_OUTPUT, "cup_classifier",
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, device=args.device,
                # CUP_CLASSES = ["blue", "red", "green"] — upweight red (idx 1)
                # to fix chronic red→green misclassification
                class_weights=[1.0, 2.0, 1.0],
                train_transform_fn=get_cup_train_transform,
            )

    # ---- EVAL ----
    if do_eval:
        log.info("\n" + "=" * 70)
        log.info("EVALUATING CLASSIFIERS")
        log.info("=" * 70)

        if args.which in ("pod", "both"):
            ckpt = POD_OUTPUT / "best_model.pth"
            if ckpt.exists():
                evaluate_classifier(ckpt, POD_FRAMES_DIR, POD_CLASSES, "pod_classifier", args.device)
            else:
                log.warning(f"No pod checkpoint at {ckpt}")

        if args.which in ("cup", "both"):
            ckpt = CUP_OUTPUT / "best_model.pth"
            if ckpt.exists():
                evaluate_classifier(ckpt, CUP_FRAMES_DIR, CUP_CLASSES, "cup_classifier", args.device)
            else:
                log.warning(f"No cup checkpoint at {ckpt}")

    # ---- UPLOAD ----
    if do_upload:
        log.info("\n" + "=" * 70)
        log.info("UPLOADING TO HUGGINGFACE")
        log.info("=" * 70)

        if args.which in ("pod", "both"):
            if POD_FRAMES_DIR.exists():
                upload_frames_dataset(POD_FRAMES_DIR, POD_CLASSIFIER_DATASET_REPO, POD_CLASSES)
            if POD_OUTPUT.exists():
                upload_model(POD_OUTPUT, POD_CLASSIFIER_MODEL_REPO)

        if args.which in ("cup", "both"):
            if CUP_FRAMES_DIR.exists():
                upload_frames_dataset(CUP_FRAMES_DIR, CUP_CLASSIFIER_DATASET_REPO, CUP_CLASSES)
            if CUP_OUTPUT.exists():
                upload_model(CUP_OUTPUT, CUP_CLASSIFIER_MODEL_REPO)


if __name__ == "__main__":
    main()
