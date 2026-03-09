#!/usr/bin/env python3
"""
DEP Policy Selector — Three strategies for choosing which policy to run.

The DEP (Decoupled Evaluation Pipeline) uses a two-stage coffee task:
  Stage 1 (POD): pick up a colored pod  (gold / red / green)
  Stage 2 (CUP): pick up a colored cup  (blue / red / green)

The scene classifier identifies which color variant is present.  This module
decides *what to do* with that classification, providing three strategies:

  1. SingleEpisodeSelector
     Classifier picks the variant → replay episode 0 from that variant's
     dataset as an open-loop trajectory.  No learned policy needed.

  2. TrainedModelSelector
     Classifier picks the variant → load the corresponding specialist ACT
     policy and run closed-loop inference.  (Full DEP Level 3.)

  3. MonolithicSelector
     No classifier.  Two fixed policies (one for pod, one for cup) are
     loaded at startup.  Uses whatever was configured statically.

All three implement the same interface so the orchestrator doesn't care
which strategy is active.

Usage (from sync_orchestrator):
    selector = TrainedModelSelector(config, ...)
    spec = selector.select_policy("pod", classifier_result)
    loaded = selector.get_loaded_policy(spec)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger("policy_selector")


# ── Classifier helpers ─────────────────────────────────────────────────────

_VENV_SITE = "/opt/lerobot_venv/lib"
if os.path.isdir(_VENV_SITE):
    for _d in os.listdir(_VENV_SITE):
        _sp = os.path.join(_VENV_SITE, _d, "site-packages")
        if os.path.isdir(_sp) and _sp not in sys.path:
            sys.path.insert(0, _sp)
            break


class Stage(Enum):
    """Two-stage coffee task."""
    POD = "pod"
    CUP = "cup"


@dataclass
class ClassifierResult:
    """Output from the scene classifier."""
    stage: Stage
    predicted_class: str   # e.g. "gold", "blue"
    confidence: float
    raw_probs: dict[str, float] = field(default_factory=dict)


@dataclass
class SelectedPolicy:
    """What the selector decided to use."""
    stage: Stage
    variant: str                       # e.g. "gold", "blue"
    pretrained_path: str               # HF repo id or local path
    task: str                          # natural-language task description
    source: str                        # "classifier" | "static"
    # For SingleEpisodeSelector only:
    dataset_repo_id: str | None = None
    episode_index: int = 0


# ── Scene Classifier (standalone, no ROS2) ─────────────────────────────────


class SceneClassifier:
    """Runs a trained ResNet18 classifier on a single frame.

    This is a pure-inference wrapper — no ROS2, no timers.  The orchestrator
    feeds frames in; this returns predictions.
    """

    def __init__(
        self,
        checkpoint_path: str,
        classes: list[str],
        device: str = "cuda",
        min_confidence: float = 0.7,
        num_frames_to_average: int = 5,
    ):
        from torchvision import models, transforms
        import torch.nn as nn

        self.classes = classes
        self.min_confidence = min_confidence
        self.num_frames_to_average = num_frames_to_average
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(classes))
        state_dict = ckpt["model_state_dict"]
        # train_scene_classifier.py wraps ResNet as self.backbone, so keys are
        # prefixed with "backbone." — strip that prefix if present.
        if any(k.startswith("backbone.") for k in state_dict):
            state_dict = {k[len("backbone."):]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Rolling buffer for frame averaging
        self._prob_buffer: list[np.ndarray] = []

        logger.info(
            f"SceneClassifier loaded: {classes} from {checkpoint_path} "
            f"(device={self.device}, min_conf={min_confidence})"
        )

    def reset(self):
        """Clear the rolling probability buffer."""
        self._prob_buffer.clear()

    def classify_frame(self, frame_bgr: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """Classify a single BGR frame.

        Returns:
            (predicted_class, confidence, {class: prob})
        """
        rgb = frame_bgr[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        tensor = self.transform(tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Accumulate and average
        self._prob_buffer.append(probs)
        if len(self._prob_buffer) > self.num_frames_to_average:
            self._prob_buffer.pop(0)

        avg_probs = np.mean(self._prob_buffer, axis=0)
        idx = int(np.argmax(avg_probs))
        conf = float(avg_probs[idx])
        class_name = self.classes[idx]

        prob_dict = {c: float(avg_probs[i]) for i, c in enumerate(self.classes)}
        return class_name, conf, prob_dict

    def classify_until_confident(
        self, frame_source, timeout_s: float = 5.0, poll_hz: float = 10.0,
    ) -> ClassifierResult | None:
        """Keep classifying frames until confidence exceeds threshold.

        Args:
            frame_source: callable that returns (frame_bgr, Stage) or None.
            timeout_s: give up after this many seconds.
            poll_hz: how often to poll for new frames.

        Returns:
            ClassifierResult if confident, None if timed out.
        """
        self.reset()
        t0 = time.time()

        while time.time() - t0 < timeout_s:
            result = frame_source()
            if result is None:
                time.sleep(1.0 / poll_hz)
                continue

            frame_bgr, stage = result
            class_name, conf, prob_dict = self.classify_frame(frame_bgr)

            if conf >= self.min_confidence:
                logger.info(
                    f"Classifier confident: {stage.value}/{class_name} "
                    f"({conf:.2f} >= {self.min_confidence})"
                )
                return ClassifierResult(
                    stage=stage,
                    predicted_class=class_name,
                    confidence=conf,
                    raw_probs=prob_dict,
                )

            time.sleep(1.0 / poll_hz)

        logger.warning(f"Classifier timed out after {timeout_s}s")
        return None


# ── Abstract Base ──────────────────────────────────────────────────────────


class PolicySelectorBase(ABC):
    """Interface for policy selection strategies."""

    @abstractmethod
    def select_policy(self, stage: Stage, classifier_result: ClassifierResult | None) -> SelectedPolicy:
        """Choose which policy/episode to use for the given stage.

        Args:
            stage: POD or CUP.
            classifier_result: output from SceneClassifier (None for monolithic).

        Returns:
            SelectedPolicy describing what to load/run.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Human-readable strategy name."""
        ...


# ── Strategy 1: Single Episode ────────────────────────────────────────────


@dataclass
class SingleEpisodeConfig:
    """Config for single-episode replay."""
    pod_datasets: dict[str, str] = field(default_factory=lambda: {
        "gold": "RAPOB/dep_coffee_pod_gold_no_aria",
        "red":  "RAPOB/dep_coffee_pod_red_no_aria",
        "green": "RAPOB/dep_coffee_pod_green_no_aria",
    })
    cup_datasets: dict[str, str] = field(default_factory=lambda: {
        "blue":  "RAPOB/dep_coffee_cup_blue_no_aria",
        "red":   "RAPOB/dep_coffee_cup_red_no_aria",
        "green": "RAPOB/dep_coffee_cup_green_no_aria",
    })
    # Per-variant episode indices; fallback default_episode_index used if variant not listed
    pod_episode_indices: dict[str, int] = field(default_factory=lambda: {})
    cup_episode_indices: dict[str, int] = field(default_factory=lambda: {})
    default_episode_index: int = 0


class SingleEpisodeSelector(PolicySelectorBase):
    """Classifier picks variant → replay a single dataset episode.

    No trained policy needed.  The orchestrator will extract the action
    sequence from the dataset episode and replay open-loop.
    """

    def __init__(self, config: SingleEpisodeConfig):
        self.config = config

    def get_name(self) -> str:
        return "single_episode"

    def select_policy(self, stage: Stage, classifier_result: ClassifierResult | None) -> SelectedPolicy:
        if classifier_result is None:
            raise ValueError("SingleEpisodeSelector requires a classifier result")

        variant = classifier_result.predicted_class
        datasets = (
            self.config.pod_datasets if stage == Stage.POD
            else self.config.cup_datasets
        )

        if variant not in datasets:
            raise ValueError(
                f"Unknown variant '{variant}' for stage {stage.value}. "
                f"Available: {list(datasets.keys())}"
            )

        dataset_id = datasets[variant]
        ep_indices = (
            self.config.pod_episode_indices if stage == Stage.POD
            else self.config.cup_episode_indices
        )
        episode_index = ep_indices.get(variant, self.config.default_episode_index)
        task = (
            f"Replay episode {episode_index} from {variant} {stage.value} dataset"
        )

        logger.info(f"[SingleEpisode] {stage.value}/{variant} → {dataset_id} ep{episode_index}")

        return SelectedPolicy(
            stage=stage,
            variant=variant,
            pretrained_path="",  # no trained model
            task=task,
            source="classifier",
            dataset_repo_id=dataset_id,
            episode_index=episode_index,
        )


# ── Strategy 2: Trained Model (DEP) ──────────────────────────────────────


@dataclass
class TrainedModelConfig:
    """Config for classifier-driven trained model selection."""
    pod_models: dict[str, dict[str, str]] = field(default_factory=lambda: {
        "gold":  {"pretrained_path": "RAPOB/dep_act_pod_gold_no_aria_40ep",
                   "task": "Pick up the gold coffee pod"},
        "red":   {"pretrained_path": "RAPOB/dep_act_pod_red_no_aria_30ep",
                   "task": "Pick up the red coffee pod"},
        "green": {"pretrained_path": "RAPOB/dep_act_pod_green_no_aria_25ep",
                   "task": "Pick up the green coffee pod"},
    })
    cup_models: dict[str, dict[str, str]] = field(default_factory=lambda: {
        "blue":  {"pretrained_path": "RAPOB/dep_act_cup_blue_no_aria_30ep",
                   "task": "Pick up the blue cup and make coffee"},
        "red":   {"pretrained_path": "RAPOB/dep_act_cup_red_no_aria_20ep",
                   "task": "Pick up the red cup and make coffee"},
        "green": {"pretrained_path": "RAPOB/dep_act_cup_green_no_aria_25ep",
                   "task": "Pick up the green cup and make coffee"},
    })


class TrainedModelSelector(PolicySelectorBase):
    """Classifier picks variant → load that variant's trained ACT policy.

    This is the full DEP Level 3 pipeline.
    """

    def __init__(self, config: TrainedModelConfig):
        self.config = config

    def get_name(self) -> str:
        return "trained_model"

    def select_policy(self, stage: Stage, classifier_result: ClassifierResult | None) -> SelectedPolicy:
        if classifier_result is None:
            raise ValueError("TrainedModelSelector requires a classifier result")

        variant = classifier_result.predicted_class
        models = (
            self.config.pod_models if stage == Stage.POD
            else self.config.cup_models
        )

        if variant not in models:
            raise ValueError(
                f"Unknown variant '{variant}' for stage {stage.value}. "
                f"Available: {list(models.keys())}"
            )

        entry = models[variant]
        pretrained_path = entry["pretrained_path"]
        task = entry.get("task", f"{stage.value} {variant}")

        logger.info(f"[TrainedModel] {stage.value}/{variant} → {pretrained_path}")

        return SelectedPolicy(
            stage=stage,
            variant=variant,
            pretrained_path=pretrained_path,
            task=task,
            source="classifier",
        )


# ── Strategy 3: Monolithic ───────────────────────────────────────────────


@dataclass
class MonolithicConfig:
    """Config for monolithic (no-classifier) policy pair."""
    pod_pretrained_path: str = "RAPOB/dep_act_pod_gold_no_aria_40ep"
    pod_task: str = "Pick up the coffee pod"
    cup_pretrained_path: str = "RAPOB/dep_act_cup_blue_no_aria_30ep"
    cup_task: str = "Pick up the cup and make coffee"


class MonolithicSelector(PolicySelectorBase):
    """No classifier.  Fixed pod + cup policies configured at startup."""

    def __init__(self, config: MonolithicConfig):
        self.config = config

    def get_name(self) -> str:
        return "monolithic"

    def select_policy(self, stage: Stage, classifier_result: ClassifierResult | None = None) -> SelectedPolicy:
        if stage == Stage.POD:
            path = self.config.pod_pretrained_path
            task = self.config.pod_task
        else:
            path = self.config.cup_pretrained_path
            task = self.config.cup_task

        logger.info(f"[Monolithic] {stage.value} → {path}")

        return SelectedPolicy(
            stage=stage,
            variant="default",
            pretrained_path=path,
            task=task,
            source="static",
        )


# ── Episode Replay Helper ────────────────────────────────────────────────


class EpisodeReplayer:
    """Extract and replay an action trajectory from a LeRobot dataset episode.

    Used by SingleEpisodeSelector.  Loads the dataset, reads all action
    frames for the given episode, then provides them as an iterator.
    """

    def __init__(self, dataset_repo_id: str, episode_index: int = 0):
        self.dataset_repo_id = dataset_repo_id
        self.episode_index = episode_index
        self._actions: list[dict[str, float]] = []
        self._cursor = 0

    def load(self):
        """Load the episode action trajectory from the dataset.

        Reads action data directly from parquet files (LeRobot v3 format)
        instead of instantiating the full LeRobotDataset, which is very slow
        because it indexes all video files.
        """
        import pyarrow.parquet as pq

        logger.info(
            f"Loading episode {self.episode_index} from {self.dataset_repo_id}..."
        )

        # Resolve local dataset path (same as LeRobotDataset)
        lerobot_home = os.environ.get(
            "HF_LEROBOT_HOME",
            os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "lerobot"),
        )
        ds_root = Path(os.path.expanduser(lerobot_home)) / self.dataset_repo_id
        if not ds_root.exists():
            raise FileNotFoundError(
                f"Dataset not found at {ds_root}. Download it first."
            )

        # Read episodes metadata to get from/to indices
        ep_meta_dir = ds_root / "meta" / "episodes"
        ep_table = None
        for chunk_dir in sorted(ep_meta_dir.iterdir()):
            for pf in sorted(chunk_dir.glob("*.parquet")):
                t = pq.read_table(pf, columns=["episode_index", "dataset_from_index", "dataset_to_index"])
                ep_table = t if ep_table is None else __import__("pyarrow").concat_tables([ep_table, t])

        if ep_table is None:
            raise FileNotFoundError(f"No episode metadata found in {ep_meta_dir}")

        ep_df = ep_table.to_pandas()
        ep_row = ep_df[ep_df["episode_index"] == self.episode_index]
        if len(ep_row) == 0:
            raise ValueError(
                f"Episode {self.episode_index} not found in {self.dataset_repo_id}"
            )
        from_idx = int(ep_row.iloc[0]["dataset_from_index"])
        to_idx = int(ep_row.iloc[0]["dataset_to_index"])

        # Read action data from parquet (only action + index columns)
        data_dir = ds_root / "data"
        data_table = None
        for chunk_dir in sorted(data_dir.iterdir()):
            for pf in sorted(chunk_dir.glob("*.parquet")):
                t = pq.read_table(pf, columns=["action", "index"])
                data_table = t if data_table is None else __import__("pyarrow").concat_tables([data_table, t])

        if data_table is None:
            raise FileNotFoundError(f"No data files found in {data_dir}")

        # Read action feature names from info.json (e.g. left_shoulder_pan.pos)
        import json
        info = json.loads((ds_root / "meta" / "info.json").read_text())
        action_names = info.get("features", {}).get("action", {}).get("names", [])

        data_df = data_table.to_pandas()
        ep_data = data_df[(data_df["index"] >= from_idx) & (data_df["index"] < to_idx)]
        ep_data = ep_data.sort_values("index")

        self._actions = []
        for _, row in ep_data.iterrows():
            action_val = row["action"]
            action_dict = {}
            if hasattr(action_val, '__len__'):
                for j, v in enumerate(action_val):
                    key = action_names[j] if j < len(action_names) else f"action_{j}"
                    action_dict[key] = float(v)
            else:
                key = action_names[0] if action_names else "action_0"
                action_dict[key] = float(action_val)
            self._actions.append(action_dict)

        self._cursor = 0
        logger.info(
            f"Loaded {len(self._actions)} action frames from "
            f"{self.dataset_repo_id} episode {self.episode_index}"
        )

    def reset(self):
        """Reset cursor to the beginning."""
        self._cursor = 0

    @property
    def done(self) -> bool:
        return self._cursor >= len(self._actions)

    @property
    def num_frames(self) -> int:
        return len(self._actions)

    @property
    def progress(self) -> float:
        if not self._actions:
            return 0.0
        return self._cursor / len(self._actions)

    def next_action(self) -> dict[str, float] | None:
        """Get the next action frame, or None if episode is complete."""
        if self._cursor >= len(self._actions):
            return None
        action = self._actions[self._cursor]
        self._cursor += 1
        return action
