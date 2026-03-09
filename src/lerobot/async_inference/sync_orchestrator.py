#!/usr/bin/env python3
"""
Synchronous Multi-Policy Orchestrator

Unlike the async orchestrator (orchestrator_v2.py) which uses separate gRPC
policy servers and has inherent network-induced timing jitter, this module
runs ALL policies in the same process with direct `policy.select_action()`
calls.

Trade-offs vs. async:
  + Zero network latency - actions are perfectly in time with observations
  + Deterministic timing - the control loop has no async delays
  + Simpler deployment - single process, no gRPC servers to manage
  - Policy switching is slower (~1-3s to swap GPU memory) vs. async (instant)
  - Higher peak GPU memory if multiple models are loaded simultaneously

Architecture:
  - All policies are loaded at startup (one actively on GPU, rest on CPU)
  - State machine: IDLE → RUNNING → PAUSED → RESETTING → COMPLETE
  - Same ROS2 services: /orchestrator/{start,pause,resume,reset,stop,switch_policy}
  - Same optional integrations: SARM, RND, Aria glasses
  - Movement buffer for rewind/reset trajectories

Run:
    python -m lerobot.async_inference.sync_orchestrator \\
        --config_path=launch/sync_orchestrator.yaml

Control via ROS2 services:
    ros2 service call /orchestrator/start std_srvs/srv/Trigger
    ros2 service call /orchestrator/switch_policy std_srvs/srv/Trigger
"""

import collections
import json
import logging
import subprocess
import threading
import time
import warnings

import numpy as np

# torchvision video I/O is deprecated in favour of TorchCodec, but TorchCodec
# crashes on this system (std::bad_alloc). Suppress the noisy warning.
warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated",
    category=UserWarning,
    module="torchvision",
)
from contextlib import nullcontext
from copy import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from pprint import pformat
from typing import Any, Optional

import draccus
import torch

# Import camera configs to register them with draccus
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
try:
    from lerobot.cameras.ros2.configuration_ros2 import ROS2CameraConfig  # noqa: F401
except ImportError:
    pass  # ROS2 not available (e.g. running outside container without cv_bridge)
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.bi_so101_follower import BiSO101FollowerConfig  # noqa: F401

from .helpers import (
    map_robot_keys_to_lerobot_features,
    raw_observation_to_observation,
)
from .movement_buffer import MovementBuffer
from .policy_selector import (
    ClassifierResult,
    EpisodeReplayer,
    MonolithicConfig,
    MonolithicSelector,
    PolicySelectorBase,
    SceneClassifier,
    SelectedPolicy,
    SingleEpisodeConfig,
    SingleEpisodeSelector,
    Stage,
    TrainedModelConfig,
    TrainedModelSelector,
)

# Try to import ROS2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32, Int32, String
    from std_srvs.srv import Trigger

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object

logger = logging.getLogger("sync_orchestrator")


# ========== States ==========


class State(Enum):
    """Orchestrator states - identical to async version."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    RESETTING = "resetting"
    COMPLETE = "complete"


# ========== Configuration ==========


@dataclass
class PolicySpec:
    """Specification for a locally-loaded policy."""

    name: str = "policy1"
    pretrained_path: str = ""
    policy_type: str = "act"
    actions_per_chunk: int = 100
    device: str = "cuda"
    task: str = "pick up the object"
    # Keep on GPU when not active? (uses more VRAM but switching is instant)
    keep_on_gpu: bool = False
    # Override saved config fields at load time (e.g. noise_scheduler_type, num_inference_steps)
    config_overrides: dict = field(default_factory=dict)


@dataclass
class SARMConfig:
    """SARM progress monitoring configuration (optional)."""

    enabled: bool = False
    progress_topic: str = "/sarm/progress"
    stage_topic: str = "/sarm/stage"
    stage_name_topic: str = "/sarm/stage_name"
    policy_switch_threshold: float = 0.95


@dataclass
class RNDConfig:
    """RND uncertainty monitoring configuration (optional)."""

    enabled: bool = False
    pause_topic: str = "/robot/pause"
    uncertainty_topic: str = "/uncertainty/rolling"


@dataclass
class AriaConfig:
    """Aria gesture control configuration (optional)."""

    enabled: bool = False
    gaze_topic: str = "/aria/gaze_gesture/detected"
    eyes_closed_topic: str = "/aria/blink/eyes_closed_detected"
    double_blink_topic: str = "/aria/blink/double_detected"


@dataclass
class TaskEndConfig:
    """Dual RND monitoring (optional).

    Two policy-specific RND models with distinct roles:

      TOP CAMERA → Gradient completion (task end detection)
        Monitors rate-of-change of uncertainty on the top camera.
        When gradient stays near zero → task is done → auto-switch policy.
        Only runs during policy 1 (disabled after switch).

      ARIA CAMERA → Raw uncertainty (attention monitoring)
        Monitors whether the operator's gaze is in-distribution.
        High uncertainty → operator not paying attention → auto-pause.
        Runs during ALL policies (always active).
    """

    enabled: bool = False
    # Model paths (policy-specific checkpoints from train_rnd.py)
    top_rnd_model_path: str = ""
    aria_rnd_model_path: str = ""
    # Camera observation keys
    top_camera_key: str = "observation.images.top"
    aria_camera_key: str = "observation.images.aria_gaussian_attention"
    # Top camera: gradient completion (RNDCompletionDetector)
    gradient_window: int = 30       # ~1s at 30fps
    stability_window: int = 45      # ~1.5s sustained low-gradient
    min_frames: int = 150           # ~5s minimum before detection
    auto_switch_policy: bool = True
    # Aria camera: attention monitoring (raw uncertainty)
    aria_uncertainty_threshold: float = 2.0  # normalized; above this → pause
    aria_sustained_frames: int = 15          # consecutive high-unc frames to trigger
    aria_resume_threshold: float = 1.0       # below this → auto-resume
    # ROS2 topics
    task_end_topic: str = "/orchestrator/task_end"
    completion_topic: str = "/orchestrator/completion"   # top gradient diagnostics
    attention_topic: str = "/orchestrator/attention"     # aria uncertainty diagnostics


@dataclass
class RosbagConfig:
    """Configuration for rosbag recording & optional HF Hub upload.

    Records all specified ROS2 topics to a rosbag via `ros2 bag record`.
    This runs as a subprocess — no in-process video encoding, no segfaults.
    """

    enabled: bool = False
    output_dir: str = "/home/acumino/vigil_ws/outputs/rosbags"
    # HuggingFace upload
    upload_to_hub: bool = False
    repo_id: str = ""
    private: bool = True
    # Topics to record (all robot + extra topics)
    topics: list = field(default_factory=lambda: [
        "/camera/top/image_raw",
        "/camera/wrist_0/image_raw",
        "/camera/wrist_1/image_raw",
        "/aria/eye_gaze/gaussian_attention",
        "/aria/eye_tracking/image_raw",
        "/joint_states",
    ])
    # Storage type: 'sqlite3' (default) or 'mcap'
    storage: str = "mcap"
    # Compression settings
    compression: bool = True
    compression_mode: str = "file"       # 'file' or 'message'
    compression_format: str = "zstd"     # 'zstd', 'lz4', or other supported formats


@dataclass
class OrchestratorConfig:
    """Orchestrator-specific settings."""

    num_episodes: int = 1
    max_episode_time_s: float = 300.0
    transition_delay_s: float = 2.0
    reset_playback_speed: float = 0.5
    rewind_seconds: float = 5.0  # 0 = full rewind
    use_amp: bool = False

    # Task descriptions (for data recording)
    policy_1_task: str = "pick up the object"
    policy_2_task: str = "place the object"

    # Optional integrations
    sarm: SARMConfig = field(default_factory=SARMConfig)
    rnd: RNDConfig = field(default_factory=RNDConfig)
    aria: AriaConfig = field(default_factory=AriaConfig)
    task_end: TaskEndConfig = field(default_factory=TaskEndConfig)
    rosbag: RosbagConfig = field(default_factory=RosbagConfig)


@dataclass
class ClassifierConfig:
    """Scene classifier settings for DEP two-stage selection."""

    enabled: bool = False
    pod_checkpoint: str = "/home/acumino/vigil_ws/outputs/scene_classifier/pod/best_model.pth"
    cup_checkpoint: str = "/home/acumino/vigil_ws/outputs/scene_classifier/cup/best_model.pth"
    pod_classes: list = field(default_factory=lambda: ["gold", "red", "green"])
    cup_classes: list = field(default_factory=lambda: ["blue", "red", "green"])
    camera_key: str = "observation.images.aria_hard_cutout"
    device: str = "cuda"
    min_confidence: float = 0.7
    num_frames_to_average: int = 5
    classification_timeout_s: float = 5.0


@dataclass
class SyncMultiPolicyConfig:
    """Complete configuration for synchronous multi-policy orchestrator."""

    # Robot
    robot: RobotConfig = field(default_factory=lambda: BiSO101FollowerConfig())

    # Policies (loaded locally, no gRPC) — used by monolithic mode
    policies: list = field(default_factory=list)

    # Selection mode: "monolithic" | "trained_model" | "single_episode"
    selection_mode: str = "monolithic"

    # Classifier settings (used by trained_model and single_episode modes)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

    # Model maps for trained_model mode (variant → pretrained_path + task)
    pod_models: dict = field(default_factory=lambda: {
        "gold":  {"pretrained_path": "RAPOB/dep_act_pod_gold_no_aria_40ep",
                   "task": "Pick up the gold coffee pod"},
        "red":   {"pretrained_path": "RAPOB/dep_act_pod_red_no_aria_30ep",
                   "task": "Pick up the red coffee pod"},
        "green": {"pretrained_path": "RAPOB/dep_act_pod_green_no_aria_25ep",
                   "task": "Pick up the green coffee pod"},
    })
    cup_models: dict = field(default_factory=lambda: {
        "blue":  {"pretrained_path": "RAPOB/dep_act_cup_blue_no_aria_30ep",
                   "task": "Pick up the blue cup and make coffee"},
        "red":   {"pretrained_path": "RAPOB/dep_act_cup_red_no_aria_20ep",
                   "task": "Pick up the red cup and make coffee"},
        "green": {"pretrained_path": "RAPOB/dep_act_cup_green_no_aria_25ep",
                   "task": "Pick up the green cup and make coffee"},
    })

    # Dataset maps for single_episode mode (variant → dataset repo_id)
    pod_datasets: dict = field(default_factory=lambda: {
        "gold":  "RAPOB/dep_coffee_pod_gold_no_aria",
        "red":   "RAPOB/dep_coffee_pod_red_no_aria",
        "green": "RAPOB/dep_coffee_pod_green_no_aria",
    })
    cup_datasets: dict = field(default_factory=lambda: {
        "blue":  "RAPOB/dep_coffee_cup_blue_no_aria",
        "red":   "RAPOB/dep_coffee_cup_red_no_aria",
        "green": "RAPOB/dep_coffee_cup_green_no_aria",
    })
    pod_episode_indices: dict = field(default_factory=lambda: {})  # variant → episode index
    cup_episode_indices: dict = field(default_factory=lambda: {})  # variant → episode index
    default_episode_index: int = 0  # fallback if variant not in above dicts

    # Control loop
    fps: float = 30.0
    buffer_max_frames: int = 3000

    # Orchestrator settings
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    def get_policy_specs(self) -> list[PolicySpec]:
        """Convert policies dict list to PolicySpec objects."""
        return [
            PolicySpec(
                name=ps.get("name", f"policy{i}"),
                pretrained_path=ps["pretrained_path"],
                policy_type=ps.get("policy_type", "act"),
                actions_per_chunk=ps.get("actions_per_chunk", 100),
                device=ps.get("device", "cuda"),
                task=ps.get("task", "pick up the object"),
                keep_on_gpu=ps.get("keep_on_gpu", False),
                config_overrides=ps.get("config_overrides", {}),
            )
            for i, ps in enumerate(self.policies)
        ]


# ========== Loaded Policy Container ==========


class LoadedPolicy:
    """Container for a fully loaded policy + its processors."""

    def __init__(
        self,
        spec: PolicySpec,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
    ):
        self.spec = spec
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self._on_gpu = False

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def task(self) -> str:
        return self.spec.task

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def to_gpu(self):
        """Move policy to GPU for inference."""
        if not self._on_gpu:
            target = torch.device(self.spec.device)
            logger.info(f"  Moving {self.name} → {target}")
            t0 = time.perf_counter()
            self.policy.to(target)
            self._on_gpu = True
            logger.info(f"  Moved in {time.perf_counter() - t0:.2f}s")

    def to_cpu(self):
        """Move policy to CPU to free GPU memory."""
        if self._on_gpu and not self.spec.keep_on_gpu:
            logger.info(f"  Moving {self.name} → CPU")
            self.policy.to("cpu")
            torch.cuda.empty_cache()
            self._on_gpu = False

    def reset(self):
        """Reset policy internal state (action queue, etc.)."""
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def select_action(
        self, observation: dict[str, Any], use_amp: bool = False
    ) -> torch.Tensor:
        """
        Run synchronous inference: observation → action.

        This is the core advantage over async: direct call, zero latency.
        """
        observation = copy(observation)
        device = self.device

        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type)
            if device.type == "cuda" and use_amp
            else nullcontext(),
        ):
            # Normalize and preprocess
            observation = self.preprocessor(observation)

            # Direct policy call - no gRPC, no serialization, no network
            action = self.policy.select_action(observation)

            # Denormalize
            action = self.postprocessor(action)

        return action


# ========== Synchronous Multi-Policy Orchestrator ==========


class SyncMultiPolicyOrchestrator:
    """
    Synchronous multi-policy orchestrator.

    All policies run in the same process. The active policy runs on GPU,
    inactive ones sit on CPU. Switching moves models between devices.

    Same state machine, same ROS2 services, same SARM/RND/Aria support
    as the async orchestrator - but with deterministic timing.
    """

    def __init__(self, config: SyncMultiPolicyConfig, ros_node: Optional["Node"] = None):
        self.config = config
        self.orch_config = config.orchestrator
        self.ros_node = ros_node

        # State machine
        self._state = State.IDLE
        self._state_lock = threading.Lock()
        self._previous_state = State.IDLE

        # Control flags
        self._running = False
        self._reset_triggered = threading.Event()

        # Timing
        self._state_entry_time = time.time()
        self._episode_start_time = 0.0
        self._dt = 1.0 / config.fps

        # Episode tracking
        self._current_episode = 0
        self._current_policy_idx = 0

        # SARM state (optional)
        self._sarm_progress = 0.0
        self._sarm_stage = 0
        self._sarm_stage_name = ""

        # RND state (optional)
        self._rnd_uncertainty = 0.0
        self._rnd_paused = False

        # Task end detection (optional, dual RND)
        self._task_end_detected = False

        # Robot
        logger.info("Connecting robot...")
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()
        logger.info("Robot connected")

        # Movement buffer
        self.buffer = MovementBuffer(
            max_frames=config.buffer_max_frames, validate_positions=False
        )

        # LeRobot feature mapping (needed for observation conversion)
        self.lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Load all policies (monolithic mode loads from config.policies;
        # classifier modes start empty and load on-demand)
        self.loaded_policies: list[LoadedPolicy] = []
        self._selection_mode = config.selection_mode
        self._policy_selector: PolicySelectorBase | None = None
        self._pod_classifier: SceneClassifier | None = None
        self._cup_classifier: SceneClassifier | None = None
        self._current_stage: Stage = Stage.POD
        self._episode_replayer: EpisodeReplayer | None = None

        # Rolling frame buffer for pre-gesture classification.
        # Stores (capture_time, frame_bgr) for the last FRAME_BUFFER_SECS seconds.
        self._FRAME_BUFFER_SECS = 8.0   # keep frames for 8s
        self._GAZE_HOLD_SECS    = 1.5   # how long the user must hold gaze to trigger
        self._CLASSIFY_WARMUP_SECS = 5.0  # wait this long for frames before classifying
        self._frame_buffer: collections.deque = collections.deque()
        self._frame_buffer_lock = threading.Lock()

        # Classifier predictions from external dep_classifier_node
        # (subscribes to /dep/classifier/status JSON topic)
        self._pod_pred_history: list[tuple[float, str, float]] = []  # (time, class, conf)
        self._cup_pred_history: list[tuple[float, str, float]] = []
        self._classifier_pred_lock = threading.Lock()

        self._init_policy_selector()

        # For monolithic mode, load policies from config at startup
        if self._selection_mode == "monolithic":
            self._load_all_policies()

        # Rosbag recording (optional)
        self._rosbag_cfg = self.orch_config.rosbag
        self._rosbag_proc: subprocess.Popen | None = None
        self._rosbag_path: Path | None = None
        self._stop_service_called = False  # Only save rosbag if /stop is called
        self._episode_succeeded = False  # Set True when COMPLETE is reached

        # Setup task end detector if configured
        if self.orch_config.task_end.enabled:
            self._init_task_end_detector()

        # External classifier node provides predictions via /dep/classifier/status
        if self._selection_mode in ("trained_model", "single_episode") and self.config.classifier.enabled:
            logger.info("Using external dep_classifier_node (subscribing to /dep/classifier/status)")

        # Setup ROS2 if available
        if ros_node is not None and ROS2_AVAILABLE:
            self._setup_ros2(ros_node)

        logger.info("SyncMultiPolicyOrchestrator initialized")
        logger.info(f"  Selection: {self._selection_mode}")
        logger.info(f"  Policies: {[p.name for p in self.loaded_policies] or '(loaded on-demand)'}")
        logger.info(
            f"  SARM: {'enabled' if self.orch_config.sarm.enabled else 'disabled'}"
        )
        logger.info(
            f"  RND:  {'enabled' if self.orch_config.rnd.enabled else 'disabled'}"
        )
        logger.info(
            f"  Aria: {'enabled' if self.orch_config.aria.enabled else 'disabled'}"
        )
        logger.info(
            f"  Task End: {'enabled' if self.orch_config.task_end.enabled else 'disabled'}"
        )
        logger.info(
            f"  Rosbag: {'enabled' if self._rosbag_cfg.enabled else 'disabled'}"
            + (f" → {self._rosbag_cfg.output_dir}" if self._rosbag_cfg.enabled else "")
        )

    # ========== Policy Selection ==========

    def _init_policy_selector(self):
        """Initialize the policy selector strategy and classifiers."""
        mode = self._selection_mode
        logger.info(f"Selection mode: {mode}")

        if mode == "monolithic":
            cfg = MonolithicConfig()
            # Monolithic uses the first two entries from config.policies
            specs = self.config.get_policy_specs()
            if len(specs) >= 1:
                cfg.pod_pretrained_path = specs[0].pretrained_path
                cfg.pod_task = specs[0].task
            if len(specs) >= 2:
                cfg.cup_pretrained_path = specs[1].pretrained_path
                cfg.cup_task = specs[1].task
            self._policy_selector = MonolithicSelector(cfg)

        elif mode == "trained_model":
            tm_cfg = TrainedModelConfig(
                pod_models=self.config.pod_models,
                cup_models=self.config.cup_models,
            )
            self._policy_selector = TrainedModelSelector(tm_cfg)

        elif mode == "single_episode":
            se_cfg = SingleEpisodeConfig(
                pod_datasets=self.config.pod_datasets,
                cup_datasets=self.config.cup_datasets,
                pod_episode_indices=self.config.pod_episode_indices,
                cup_episode_indices=self.config.cup_episode_indices,
                default_episode_index=self.config.default_episode_index,
            )
            self._policy_selector = SingleEpisodeSelector(se_cfg)

        else:
            raise ValueError(f"Unknown selection_mode: {mode!r}")

        logger.info(f"Policy selector: {self._policy_selector.get_name()}")

    def _init_classifiers(self):
        """Load pod and cup scene classifiers."""
        cc = self.config.classifier
        if not cc.enabled:
            logger.warning(
                "Classifier not enabled but selection_mode requires it. "
                "Enable classifier in config."
            )
            return

        try:
            self._pod_classifier = SceneClassifier(
                checkpoint_path=cc.pod_checkpoint,
                classes=cc.pod_classes,
                device=cc.device,
                min_confidence=cc.min_confidence,
                num_frames_to_average=cc.num_frames_to_average,
            )
            self._cup_classifier = SceneClassifier(
                checkpoint_path=cc.cup_checkpoint,
                classes=cc.cup_classes,
                device=cc.device,
                min_confidence=cc.min_confidence,
                num_frames_to_average=cc.num_frames_to_average,
            )
            logger.info("Scene classifiers loaded (pod + cup)")
        except Exception as e:
            logger.error(f"Failed to load classifiers: {e}")

    def _start_frame_capture_thread(self):
        """Deprecated — frame capture is now done via ROS2 subscription."""
        logger.warning("_start_frame_capture_thread called but is no longer used")

    def _frame_capture_loop(self):
        """Deprecated — frame capture is now done via ROS2 subscription."""
        pass

    def _classifier_img_cb(self, msg: "RosImage"):
        """ROS2 callback: cache incoming classifier camera frames into the rolling buffer.

        Converts sensor_msgs/Image → BGR uint8 numpy array WITHOUT touching the
        robot serial port (avoids contention with the main control loop).
        """
        try:
            import numpy as np_local
            enc = msg.encoding.lower()
            arr = np_local.frombuffer(msg.data, dtype=np_local.uint8).reshape(
                msg.height, msg.width, -1
            )
            if enc in ("rgb8", "rgb"):
                frame = arr[:, :, ::-1].copy()  # RGB → BGR
            elif enc in ("bgr8", "bgr"):
                frame = arr
            elif enc in ("mono8",):
                frame = np_local.stack([arr[:, :, 0]] * 3, axis=-1)
            else:
                # Unknown encoding — try to use as-is
                frame = arr if arr.shape[2] == 3 else arr[:, :, :3]

            now = time.time()
            cutoff = now - self._FRAME_BUFFER_SECS
            with self._frame_buffer_lock:
                self._frame_buffer.append((now, frame))
                while self._frame_buffer and self._frame_buffer[0][0] < cutoff:
                    self._frame_buffer.popleft()
                buf_len = len(self._frame_buffer)

            if buf_len == 1:
                logger.info(f"Classifier frame buffer: FIRST frame received (shape={frame.shape}, enc={enc})")
        except Exception as e:
            logger.warning(f"_classifier_img_cb: failed to process image: {e}")

    def _dep_classifier_status_cb(self, msg: "String"):
        """Callback for /dep/classifier/status JSON topic (from dep_classifier_node).

        Stores pod and cup predictions with timestamps for majority voting.
        """
        try:
            data = json.loads(msg.data)
            now = time.time()
            pod = data.get("pod", {})
            cup = data.get("cup", {})
            with self._classifier_pred_lock:
                n_pod = len(self._pod_pred_history)
                if pod.get("prediction") and pod["prediction"] not in ("unknown", "unavailable", "error"):
                    self._pod_pred_history.append(
                        (now, pod["prediction"], pod.get("confidence", 0.0))
                    )
                if cup.get("prediction") and cup["prediction"] not in ("unknown", "unavailable", "error"):
                    self._cup_pred_history.append(
                        (now, cup["prediction"], cup.get("confidence", 0.0))
                    )
                # Trim to last 30s to avoid unbounded growth
                cutoff = now - 30.0
                self._pod_pred_history = [
                    x for x in self._pod_pred_history if x[0] >= cutoff
                ]
                self._cup_pred_history = [
                    x for x in self._cup_pred_history if x[0] >= cutoff
                ]
                if n_pod == 0 and len(self._pod_pred_history) > 0:
                    logger.info(f"First classifier prediction received: pod={pod}, cup={cup}")
        except Exception as e:
            logger.warning(f"Classifier status parse error: {e}")

    @staticmethod
    def _extract_frame_from_obs(raw_obs: dict, camera_key_short: str) -> np.ndarray | None:
        """Pull the BGR uint8 frame for *camera_key_short* out of a raw observation dict."""
        for key in raw_obs:
            if camera_key_short in key:
                val = raw_obs[key]
                if isinstance(val, np.ndarray):
                    return val
                elif isinstance(val, torch.Tensor):
                    frame = val.cpu().numpy()
                    if frame.ndim == 3 and frame.shape[0] in (1, 3):
                        frame = frame.transpose(1, 2, 0)
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).clip(0, 255).astype(np.uint8)
                    return frame
        return None

    def _get_classifier_frame(self, before_time: float | None = None) -> tuple[np.ndarray, Stage] | None:
        """Return a classifier frame from the rolling buffer.

        Falls back to the robot's ROS2 bridge camera cache when the rolling
        buffer is empty (safe — no serial-port access).

        Args:
            before_time: if given, return the most recent buffered frame whose
                capture time is <= before_time (i.e. a frame taken *before* the
                gaze gesture was detected).  If None, return the latest frame.
        """
        with self._frame_buffer_lock:
            buf_empty = not self._frame_buffer
            if not buf_empty:
                if before_time is not None:
                    # Walk newest→oldest, pick first frame before the cutoff
                    for ts, frame in reversed(self._frame_buffer):
                        if ts <= before_time:
                            return frame, self._current_stage
                    # All buffered frames are newer than before_time — use oldest available
                    logger.debug(
                        f"No pre-gesture frames before t={before_time:.2f}, "
                        f"using oldest buffered frame (t={self._frame_buffer[0][0]:.2f})"
                    )
                    return self._frame_buffer[0][1], self._current_stage
                else:
                    return self._frame_buffer[-1][1], self._current_stage

        # Buffer is empty — try the robot's ROS2 bridge camera cache
        # (this reads from a dict — no serial port access)
        cam_key = self.config.classifier.camera_key.replace(
            "observation.images.", ""
        )
        bridge = getattr(self.robot, "ros2_bridge", None)
        if bridge is not None:
            frame = bridge.get_camera_frame(cam_key)
            if frame is not None:
                logger.info(
                    f"_get_classifier_frame: got frame from robot bridge "
                    f"(shape={frame.shape})"
                )
                return frame, self._current_stage

        logger.warning(
            f"_get_classifier_frame: no frames available — "
            f"is the classifier camera topic publishing? "
            f"(expected topic for '{cam_key}')"
        )
        return None

    def classify_and_select(self, stage: Stage, before_time: float | None = None) -> SelectedPolicy | None:
        """Run the classifier for the given stage, then select a policy.

        Args:
            before_time: if set, only use buffered frames captured before this
                timestamp (see _get_classifier_frame).

        Returns None if classification fails.
        """
        classifier = (
            self._pod_classifier if stage == Stage.POD
            else self._cup_classifier
        )

        if classifier is None:
            logger.error(f"No classifier for stage {stage.value}")
            return None

        self._current_stage = stage
        cc = self.config.classifier

        frame_source = (lambda: self._get_classifier_frame(before_time=before_time))

        # Quick sanity-check: can we get a frame right now?
        test_frame = frame_source()
        if test_frame is None:
            logger.warning(
                f"classify_and_select: frame_source() returned None immediately "
                f"(before_time={before_time}, buffer has {len(self._frame_buffer)} frames) — "
                "classification will likely time out"
            )
        else:
            logger.info(f"classify_and_select: got test frame shape={test_frame[0].shape}")

        result = classifier.classify_until_confident(
            frame_source=frame_source,
            timeout_s=cc.classification_timeout_s,
        )

        if result is None:
            # Timed out — take the best guess from the most recent buffered frame
            classifier.reset()
            logger.warning(f"Classification timed out for {stage.value}, using best guess")
            frame_result = self._get_classifier_frame(before_time=before_time)
            if frame_result:
                frame, _ = frame_result
                class_name, conf, probs = classifier.classify_frame(frame)
                result = ClassifierResult(
                    stage=stage,
                    predicted_class=class_name,
                    confidence=conf,
                    raw_probs=probs,
                )
            else:
                return None

        return self._policy_selector.select_policy(stage, result)

    def load_selected_policy(self, selection: SelectedPolicy) -> bool:
        """Load or prepare the policy chosen by the selector.

        For trained_model: loads the ACT checkpoint and puts it on GPU.
        For single_episode: loads the dataset episode into EpisodeReplayer.
        For monolithic: the policies are already loaded at startup.

        Returns True on success.
        """
        mode = self._selection_mode

        if mode == "single_episode":
            logger.info(
                f"load_selected_policy | dataset={selection.dataset_repo_id} "
                f"episode={selection.episode_index}"
            )
            try:
                self._episode_replayer = EpisodeReplayer(
                    dataset_repo_id=selection.dataset_repo_id,
                    episode_index=selection.episode_index,
                )
                self._episode_replayer.load()
                logger.info(f"Episode loaded: {self._episode_replayer.num_frames} frames")
                return True
            except Exception as e:
                import traceback
                logger.error(f"Failed to load episode: {e}\n{traceback.format_exc()}")
                return False

        elif mode == "trained_model":
            # Unload current policy, load the selected one
            if self.loaded_policies:
                self.loaded_policies[0].to_cpu()
                self.loaded_policies.clear()

            spec = PolicySpec(
                name=f"{selection.stage.value}_{selection.variant}",
                pretrained_path=selection.pretrained_path,
                task=selection.task,
            )
            try:
                loaded = self._load_single_policy(spec)
                loaded.to_gpu()
                self.loaded_policies = [loaded]
                self._current_policy_idx = 0
                logger.info(
                    f"Loaded {spec.name} "
                    f"({sum(p.numel() for p in loaded.policy.parameters()) / 1e6:.1f}M params)"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to load policy {spec.pretrained_path}: {e}")
                return False

        elif mode == "monolithic":
            # Policies already loaded, just switch to correct index
            idx = 0 if selection.stage == Stage.POD else 1
            if idx < len(self.loaded_policies):
                self.handle_switch_policy(policy_idx=idx)
            return True

        return False

    # ========== Policy Loading ==========

    def _load_all_policies(self):
        """Load all policies at startup."""
        specs = self.config.get_policy_specs()
        if not specs:
            raise ValueError("No policies configured")

        for i, spec in enumerate(specs):
            logger.info(f"Loading policy {i + 1}/{len(specs)}: {spec.name}")
            logger.info(f"  Path: {spec.pretrained_path}")

            loaded = self._load_single_policy(spec)
            self.loaded_policies.append(loaded)

            logger.info(
                f"  Loaded: {spec.name} ({sum(p.numel() for p in loaded.policy.parameters()) / 1e6:.1f}M params)"
            )

        # Move the first policy to GPU, keep others on CPU
        self.loaded_policies[0].to_gpu()
        logger.info(f"Active policy: {self.loaded_policies[0].name} (on GPU)")

    def _load_single_policy(self, spec: PolicySpec) -> LoadedPolicy:
        """Load a single policy from pretrained path.

        Uses the same approach as the async policy_server: load directly via
        from_pretrained() which already has input_features/output_features
        baked into the saved config. This avoids the need for ds_meta.
        """
        from lerobot.policies.factory import get_policy_class

        # Get the correct policy class (ACTPolicy, DiffusionPolicy, etc.)
        policy_class = get_policy_class(spec.policy_type)

        # Build cli_overrides from config_overrides dict so they take effect
        # even when loading from a pretrained checkpoint whose config.json
        # would otherwise override any runtime settings (e.g. DDIM, num_inference_steps).
        cli_overrides = []
        for k, v in spec.config_overrides.items():
            cli_overrides.extend([f"--{k}", str(v)])

        # Load pretrained model directly - config already has features
        # Load on CPU first to avoid loading all models onto GPU
        policy = policy_class.from_pretrained(spec.pretrained_path, cli_overrides=cli_overrides)
        policy.to("cpu")
        policy.eval()

        # Load pre/post processors from the pretrained directory
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=spec.pretrained_path,
        )

        loaded = LoadedPolicy(
            spec=spec,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )

        return loaded

    def _init_task_end_detector(self):
        """Initialize dual RND detectors.

        Top camera  → RNDCompletionDetector (gradient → task end, policy 1 only)
        Aria camera → raw uncertainty monitor (attention, always active)
        """
        task_end_cfg = self.orch_config.task_end

        # --- Top camera: gradient-based task completion ---
        self._top_rnd = None
        self._completion_detector = None
        if task_end_cfg.top_rnd_model_path:
            top_path = Path(task_end_cfg.top_rnd_model_path)
            if top_path.is_dir():
                top_path = top_path / "rnd_model.pth"
            if top_path.exists():
                try:
                    from lerobot.common.uncertainty import RNDModule
                    from lerobot.common.uncertainty.rnd_completion_detector import RNDCompletionDetector

                    logger.info(f"Loading TOP RND (completion) from {top_path}")
                    top_rnd = RNDModule.load_from_checkpoint(top_path, device="cuda")
                    self._completion_detector = RNDCompletionDetector(
                        rnd_module=top_rnd,
                        gradient_window=task_end_cfg.gradient_window,
                        stability_window=task_end_cfg.stability_window,
                        min_frames=task_end_cfg.min_frames,
                    )
                    self._top_rnd = top_rnd
                    logger.info("✅ Top camera RND (gradient completion) initialized")
                except Exception as e:
                    logger.error(f"Failed to load top RND: {e}")
            else:
                logger.warning(f"Top RND model not found: {top_path}")

        # --- Aria camera: raw uncertainty for attention monitoring ---
        self._aria_rnd = None
        self._aria_high_unc_count = 0
        self._aria_paused_by_attention = False
        if task_end_cfg.aria_rnd_model_path:
            aria_path = Path(task_end_cfg.aria_rnd_model_path)
            if aria_path.is_dir():
                aria_path = aria_path / "rnd_model.pth"
            if aria_path.exists():
                try:
                    from lerobot.common.uncertainty import RNDModule

                    logger.info(f"Loading ARIA RND (attention) from {aria_path}")
                    self._aria_rnd = RNDModule.load_from_checkpoint(aria_path, device="cuda")
                    logger.info("✅ Aria camera RND (attention monitoring) initialized")
                except Exception as e:
                    logger.error(f"Failed to load aria RND: {e}")
            else:
                logger.warning(f"Aria RND model not found: {aria_path}")

    # ========== Rosbag Recording ==========

    def _start_rosbag(self):
        """Start ros2 bag record as a subprocess."""
        if not self._rosbag_cfg.enabled:
            return
        if self._rosbag_proc is not None:
            logger.warning("Rosbag already recording, stopping previous")
            self._stop_rosbag()

        cfg = self._rosbag_cfg
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique bag name with timestamp
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        bag_name = f"episode_{self._current_episode}_{timestamp}"
        self._rosbag_path = output_dir / bag_name

        cmd = [
            "ros2", "bag", "record",
            "--output", str(self._rosbag_path),
            "--storage", cfg.storage,
        ]
        # Add compression if enabled
        if cfg.compression:
            cmd.extend(["--compression-mode", cfg.compression_mode])
            cmd.extend(["--compression-format", cfg.compression_format])
        # Add all topics
        cmd.extend(cfg.topics)

        logger.info(f"Starting rosbag: {bag_name}")
        logger.info(f"  Storage: {cfg.storage}")
        if cfg.compression:
            logger.info(f"  Compression: {cfg.compression_format} ({cfg.compression_mode})")
        else:
            logger.info(f"  Compression: disabled")
        logger.info(f"  Topics: {cfg.topics}")

        try:
            self._rosbag_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            logger.info(f"Rosbag recording started (PID {self._rosbag_proc.pid})")
        except Exception as e:
            logger.error(f"Failed to start rosbag: {e}")
            self._rosbag_proc = None

    def _stop_rosbag(self):
        """Stop the rosbag recording subprocess gracefully."""
        if self._rosbag_proc is None:
            return

        import signal

        # Check if already dead
        if self._rosbag_proc.poll() is not None:
            logger.info(f"Rosbag process already exited (rc={self._rosbag_proc.returncode})")
            self._rosbag_proc = None
            return

        comp_msg = f" (waiting for {self._rosbag_cfg.compression_format} compression)" if self._rosbag_cfg.compression else ""
        logger.info(f"Stopping rosbag recording{comp_msg}...")
        try:
            # SIGINT for graceful shutdown
            self._rosbag_proc.send_signal(signal.SIGINT)

            # Poll with progress logging instead of blocking wait
            for i in range(60):  # Up to 60 seconds
                try:
                    self._rosbag_proc.wait(timeout=1)
                    break  # Exited cleanly
                except subprocess.TimeoutExpired:
                    if i > 0 and i % 5 == 0:
                        comp_note = "compressed " if self._rosbag_cfg.compression else ""
                        logger.info(f"  Still flushing {comp_note}rosbag... ({i}s)")
            else:
                # 60s elapsed — force kill
                logger.warning("Rosbag didn't stop after 60s, killing")
                self._rosbag_proc.kill()
                self._rosbag_proc.wait(timeout=5)

            logger.info(f"Rosbag saved: {self._rosbag_path}")
        except Exception as e:
            logger.error(f"Error stopping rosbag: {e}")
        finally:
            self._rosbag_proc = None

    def _upload_rosbag(self):
        """Upload rosbag directory to HuggingFace Hub.

        Deletes uncompressed .mcap files before upload so only the
        zstd-compressed versions are uploaded (and viewable in Foxglove).
        """
        cfg = self._rosbag_cfg
        if not cfg.upload_to_hub or not cfg.repo_id:
            return
        if self._rosbag_path is None or not self._rosbag_path.exists():
            logger.warning("No rosbag to upload")
            return

        try:
            # Delete uncompressed .mcap files (keep only .mcap.zstd)
            import glob
            bag_dir = self._rosbag_path
            mcap_files = list(bag_dir.glob("*.mcap"))
            zstd_files = list(bag_dir.glob("*.mcap.zstd"))
            if zstd_files:
                for f in mcap_files:
                    # Don't delete if it's actually a .mcap.zstd
                    if not str(f).endswith(".zstd"):
                        size_mb = f.stat().st_size / (1024 * 1024)
                        logger.info(f"  Removing uncompressed {f.name} ({size_mb:.0f}MB)")
                        f.unlink()

            # Log what we're uploading
            remaining = list(bag_dir.iterdir())
            total_mb = sum(f.stat().st_size for f in remaining if f.is_file()) / (1024 * 1024)
            logger.info(f"Uploading rosbag to {cfg.repo_id} ({total_mb:.0f}MB, {len(remaining)} files)")

            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(cfg.repo_id, exist_ok=True, private=cfg.private,
                            repo_type="dataset")
            api.upload_folder(
                folder_path=str(self._rosbag_path),
                repo_id=cfg.repo_id,
                path_in_repo=self._rosbag_path.name,
                repo_type="dataset",
                commit_message=f"Add rosbag: {self._rosbag_path.name}",
            )
            logger.info(f"Rosbag uploaded to {cfg.repo_id}")
        except Exception as e:
            logger.error(f"Error uploading rosbag: {e}", exc_info=True)

    # ========== Properties ==========

    @property
    def state(self) -> State:
        with self._state_lock:
            return self._state

    @property
    def running(self) -> bool:
        return self._running

    @property
    def active_policy(self) -> LoadedPolicy | None:
        if not self.loaded_policies:
            return None
        return self.loaded_policies[self._current_policy_idx]

    # ========== State Machine ==========

    def _transition_to(self, new_state: State):
        """Transition to a new state."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            self._state_entry_time = time.time()

        logger.info(f"State: {old_state.value} → {new_state.value}")
        self._on_state_enter(new_state, old_state)

    def _on_state_enter(self, new_state: State, old_state: State):
        """Handle actions when entering a new state."""
        if new_state == State.IDLE:
            pass

        elif new_state == State.RUNNING:
            if not self.buffer.is_recording:
                self.buffer.start_recording()
            # Start rosbag on first transition to RUNNING
            if self._rosbag_cfg.enabled and self._rosbag_proc is None:
                self._start_rosbag()

        elif new_state == State.PAUSED:
            self._previous_state = old_state

        elif new_state == State.RESETTING:
            if self.buffer.is_recording:
                self.buffer.stop_recording()

        elif new_state == State.COMPLETE:
            if self.buffer.is_recording:
                self.buffer.stop_recording()
            # COMPLETE is a successful finish — always save & upload rosbag
            self._episode_succeeded = True
            if self._rosbag_proc is not None:
                import threading as _th
                _th.Thread(
                    target=self._stop_and_upload_rosbag,
                    name="rosbag_complete",
                    daemon=True,
                ).start()

    # ========== ROS2 Setup ==========

    def _setup_ros2(self, node: "Node"):
        """Set up ROS2 services and subscriptions."""
        # Control services (always enabled)
        node.create_service(Trigger, "/orchestrator/start", self._srv_start)
        node.create_service(Trigger, "/orchestrator/pause", self._srv_pause)
        node.create_service(Trigger, "/orchestrator/resume", self._srv_resume)
        node.create_service(Trigger, "/orchestrator/reset", self._srv_reset)
        node.create_service(Trigger, "/orchestrator/stop", self._srv_stop)
        node.create_service(
            Trigger, "/orchestrator/switch_policy", self._srv_switch_policy
        )

        # State publisher
        self._state_pub = node.create_publisher(String, "/orchestrator/state", 10)
        node.create_timer(0.5, self._publish_state)

        logger.info(
            "ROS2 services: /orchestrator/{start,pause,resume,reset,stop,switch_policy}"
        )

        # SARM subscriptions (optional)
        if self.orch_config.sarm.enabled:
            node.create_subscription(
                Float32,
                self.orch_config.sarm.progress_topic,
                self._sarm_progress_cb,
                10,
            )
            node.create_subscription(
                Int32,
                self.orch_config.sarm.stage_topic,
                self._sarm_stage_cb,
                10,
            )
            node.create_subscription(
                String,
                self.orch_config.sarm.stage_name_topic,
                self._sarm_stage_name_cb,
                10,
            )
            logger.info(
                f"SARM subscriptions: {self.orch_config.sarm.progress_topic}"
            )

        # RND subscription (optional)
        if self.orch_config.rnd.enabled:
            node.create_subscription(
                Bool,
                self.orch_config.rnd.pause_topic,
                self._rnd_pause_cb,
                10,
            )
            node.create_subscription(
                Float32,
                self.orch_config.rnd.uncertainty_topic,
                self._rnd_uncertainty_cb,
                10,
            )
            logger.info(
                f"RND subscriptions: {self.orch_config.rnd.pause_topic}"
            )

        # Aria subscriptions (optional)
        if self.orch_config.aria.enabled:
            node.create_subscription(
                String,
                self.orch_config.aria.gaze_topic,
                self._aria_gaze_cb,
                10,
            )
            node.create_subscription(
                Bool,
                self.orch_config.aria.eyes_closed_topic,
                self._aria_eyes_closed_cb,
                10,
            )
            node.create_subscription(
                Bool,
                self.orch_config.aria.double_blink_topic,
                self._aria_double_blink_cb,
                10,
            )
            logger.info(
                f"Aria subscriptions: {self.orch_config.aria.gaze_topic}"
            )

        # Classifier topic subscription (external dep_classifier_node)
        if (self._selection_mode in ("trained_model", "single_episode")
                and self.config.classifier.enabled):
            node.create_subscription(
                String,
                "/dep/classifier/status",
                self._dep_classifier_status_cb,
                10,
            )
            logger.info("Classifier subscription: /dep/classifier/status")
            topic_info = node.get_publishers_info_by_topic("/dep/classifier/status")
            if not topic_info:
                logger.warning(
                    "⚠️  No publishers on /dep/classifier/status! "
                    "Is dep_classifier_node running? "
                    "Classification will fail until this topic is published."
                )
            else:
                logger.info(f"  /dep/classifier/status has {len(topic_info)} publisher(s) — OK")

        # Task end publishers (optional - dual RND models)
        if self.orch_config.task_end.enabled:
            from std_msgs.msg import Float32MultiArray

            self._task_end_pub = node.create_publisher(
                Float32MultiArray,
                self.orch_config.task_end.task_end_topic,
                10,
            )
            if self._completion_detector is not None:
                self._completion_pub = node.create_publisher(
                    Float32MultiArray,
                    self.orch_config.task_end.completion_topic,
                    10,
                )
                logger.info(
                    f"Top RND completion → {self.orch_config.task_end.completion_topic}"
                )
            if self._aria_rnd is not None:
                self._attention_pub = node.create_publisher(
                    Float32MultiArray,
                    self.orch_config.task_end.attention_topic,
                    10,
                )
                logger.info(
                    f"Aria RND attention → {self.orch_config.task_end.attention_topic}"
                )

    # ========== ROS2 Service Handlers ==========

    def _srv_start(self, req, res):
        success, msg = self.handle_start()
        res.success = success
        res.message = msg
        return res

    def _srv_pause(self, req, res):
        success, msg = self.handle_pause()
        res.success = success
        res.message = msg
        return res

    def _srv_resume(self, req, res):
        success, msg = self.handle_resume()
        res.success = success
        res.message = msg
        return res

    def _srv_reset(self, req, res):
        success, msg = self.handle_reset()
        res.success = success
        res.message = msg
        return res

    def _srv_stop(self, req, res):
        success, msg = self.handle_stop()
        res.success = success
        res.message = msg
        return res

    def _srv_switch_policy(self, req, res):
        success, msg = self.handle_switch_policy()
        res.success = success
        res.message = msg
        return res

    def _publish_state(self):
        """Publish current state."""
        if not hasattr(self, "_state_pub"):
            return

        stats = self.buffer.get_stats()
        policy_name = self.active_policy.name if self.active_policy else "none"
        replayer_info = ""
        if self._episode_replayer is not None:
            replayer_info = f" | replay: {self._episode_replayer.progress:.0%}"
        msg = String()
        msg.data = (
            f"{self.state.value} | "
            f"stage: {self._current_stage.value} | "
            f"mode: {self._selection_mode} | "
            f"policy: {policy_name}{replayer_info} | "
            f"buffer: {stats['buffer_size']}"
        )
        self._state_pub.publish(msg)

    # ========== Action Handlers ==========

    def handle_start(self) -> tuple[bool, str]:
        """Start or resume execution."""
        if self.state == State.IDLE:
            if self.active_policy is not None:
                self.active_policy.reset()
            self._episode_start_time = time.time()
            self._recording_frame_count = 0
            self._transition_to(State.RUNNING)
            policy_name = self.active_policy.name if self.active_policy else "episode_replay"
            return True, f"Started with policy '{policy_name}'"
        elif self.state == State.PAUSED:
            return self.handle_resume()
        return False, f"Cannot start from {self.state.value}"

    def handle_pause(self) -> tuple[bool, str]:
        """Pause execution."""
        if self.state == State.RUNNING:
            self._transition_to(State.PAUSED)
            return True, "Paused"
        return False, f"Cannot pause from {self.state.value}"

    def handle_resume(self) -> tuple[bool, str]:
        """Resume execution."""
        if self.state == State.PAUSED:
            self._transition_to(State.RUNNING)
            return True, "Resumed"
        return False, f"Cannot resume from {self.state.value}"

    def handle_reset(self) -> tuple[bool, str]:
        """Trigger reset trajectory (rewind by rewind_seconds or full)."""
        if self.state in [State.RUNNING, State.PAUSED]:
            self._transition_to(State.RESETTING)

            # Calculate how many frames to rewind
            max_frames = None  # None = full rewind
            if self.orch_config.rewind_seconds > 0:
                max_frames = int(self.orch_config.rewind_seconds * self.config.fps)
                logger.info(
                    f"Rewinding {self.orch_config.rewind_seconds}s "
                    f"({max_frames} frames at {self.config.fps} fps)"
                )

            # Execute partial or full reset (blocking)
            success = self._execute_reset_trajectory(
                playback_speed=self.orch_config.reset_playback_speed,
                max_frames=max_frames,
            )

            # Only clear rewound portion from buffer (or all if full rewind)
            if max_frames is not None:
                self.buffer.trim_last_n(max_frames)
            else:
                self.buffer.clear()

            if max_frames is not None and success:
                # Partial rewind: pause briefly so robot settles, reset policy
                # action chunks to avoid stale queued actions, then resume
                self._transition_to(State.PAUSED)
                self.active_policy.reset()  # flush stale action chunks
                logger.info("Settling after rewind (1s)...")
                time.sleep(1.0)
                self._transition_to(State.RUNNING)
                return True, f"Rewound {self.orch_config.rewind_seconds}s - resumed"
            else:
                # Full rewind: go back to IDLE
                self.active_policy.reset()
                self._transition_to(State.IDLE)
                return success, "Reset complete" if success else "Reset failed"
        return False, f"Cannot reset from {self.state.value}"

    def handle_stop(self) -> tuple[bool, str]:
        """Stop the orchestrator. Rosbag flush + upload runs in background."""
        self._stop_service_called = True  # Mark explicit stop (keeps rosbag)
        self._running = False  # Signal control loop to exit immediately

        if self._rosbag_cfg.enabled and self._rosbag_proc is not None:
            # Do the slow rosbag stop + upload in a background thread
            # so the service call returns instantly
            import threading
            threading.Thread(
                target=self._stop_and_upload_rosbag,
                name="rosbag_cleanup",
                daemon=True,
            ).start()
            return True, "Stopping (rosbag flushing in background)"

        return True, "Stopping"

    def _stop_and_upload_rosbag(self):
        """Background thread: stop rosbag, upload, log completion."""
        try:
            self._stop_rosbag()
            self._upload_rosbag()
        except Exception as e:
            logger.error(f"Rosbag cleanup error: {e}")

    def _stop_and_delete_rosbag(self):
        """Stop rosbag recording and delete the bag (stop service was NOT called)."""
        try:
            self._stop_rosbag()
            self._delete_rosbag()
        except Exception as e:
            logger.error(f"Rosbag delete error: {e}")

    def _delete_rosbag(self):
        """Delete the rosbag directory (recording discarded)."""
        if self._rosbag_path is None or not self._rosbag_path.exists():
            return
        import shutil
        bag_size_mb = sum(
            f.stat().st_size for f in self._rosbag_path.rglob("*") if f.is_file()
        ) / (1024 * 1024)
        logger.info(
            f"Deleting rosbag (stop service not called): "
            f"{self._rosbag_path.name} ({bag_size_mb:.0f}MB)"
        )
        shutil.rmtree(self._rosbag_path, ignore_errors=True)
        self._rosbag_path = None
        logger.info("Rosbag deleted")

    def handle_switch_policy(self, policy_idx: Optional[int] = None) -> tuple[bool, str]:
        """
        Switch to a different policy.

        This is the key difference from async: we move models between GPU/CPU.
        It takes ~1-3s but produces zero timing jitter during execution.
        """
        was_running = self.state == State.RUNNING

        if was_running:
            self._transition_to(State.PAUSED)

        old_name = self.active_policy.name

        # Move current policy off GPU
        logger.info(f"Switching policy: {old_name} → ...")
        t0 = time.perf_counter()
        self.active_policy.to_cpu()

        # Switch index
        if policy_idx is not None:
            self._current_policy_idx = policy_idx % len(self.loaded_policies)
        else:
            self._current_policy_idx = (self._current_policy_idx + 1) % len(
                self.loaded_policies
            )

        new_policy = self.active_policy
        new_policy.to_gpu()
        new_policy.reset()

        switch_time = time.perf_counter() - t0
        logger.info(
            f"Switched: {old_name} → {new_policy.name} in {switch_time:.2f}s"
        )

        # Optional transition delay
        if self.orch_config.transition_delay_s > 0:
            logger.info(
                f"Transition delay: {self.orch_config.transition_delay_s}s"
            )
            time.sleep(self.orch_config.transition_delay_s)

        if was_running:
            self._transition_to(State.RUNNING)

        return True, f"Switched: {old_name} → {new_policy.name} ({switch_time:.2f}s)"

    # ========== Reset Execution ==========

    def _execute_reset_trajectory(
        self, playback_speed: float = 0.5, max_frames: int | None = None
    ) -> bool:
        """Execute reverse trajectory from buffer.

        Args:
            playback_speed: Speed multiplier (0.5 = half speed).
            max_frames: If set, only rewind the last N frames (partial rewind).
        """
        trajectory = self.buffer.get_reverse_trajectory(
            playback_speed=playback_speed,
            smooth_window=1,
            max_frames=max_frames,
        )

        if not trajectory:
            logger.warning("Empty buffer - cannot reset")
            return False

        logger.info(
            f"Executing reset: {len(trajectory)} frames at {playback_speed}x speed"
        )

        for i, positions in enumerate(trajectory):
            if not self._running:
                logger.info("Reset aborted - stopping")
                return False

            try:
                self.robot.send_action(positions)
            except Exception as e:
                logger.error(f"Reset error: {e}")
                return False

            time.sleep(self._dt)

            if (i + 1) % 100 == 0:
                logger.info(f"Reset progress: {i + 1}/{len(trajectory)}")

        logger.info("Reset complete")
        return True

    # ========== Observation Pipeline ==========

    def _prepare_observation(self, raw_obs: dict) -> dict[str, Any]:
        """
        Prepare a raw robot observation for policy inference.

        Args:
            raw_obs: Raw observation dict from robot.get_observation()

        Returns:
            Dict ready for policy.select_action() - tensors on correct
            device, images as float32 CHW, state as [1, D].
        """
        # Add task string
        raw_obs["task"] = self.active_policy.task

        # Convert to policy format using the same helpers as async
        policy_obs = raw_observation_to_observation(
            raw_obs,
            self.lerobot_features,
            self.active_policy.policy.config.image_features,
        )

        # Move tensors to policy device
        device = self.active_policy.device
        for k, v in policy_obs.items():
            if isinstance(v, torch.Tensor):
                policy_obs[k] = v.to(device)

        return policy_obs

    # ========== SARM Callbacks ==========

    def _sarm_progress_cb(self, msg: "Float32"):
        self._sarm_progress = msg.data
        if self.state == State.RUNNING:
            if self._sarm_progress >= self.orch_config.sarm.policy_switch_threshold:
                logger.info(
                    f"SARM progress {self._sarm_progress:.3f} >= threshold, switching policy"
                )
                self.handle_switch_policy()

    def _sarm_stage_cb(self, msg: "Int32"):
        self._sarm_stage = msg.data

    def _sarm_stage_name_cb(self, msg: "String"):
        self._sarm_stage_name = msg.data

    # ========== RND Callbacks ==========

    def _rnd_pause_cb(self, msg: "Bool"):
        should_pause = msg.data
        if should_pause and not self._rnd_paused:
            if self.state == State.RUNNING:
                logger.info("RND uncertainty high - pausing")
                self.handle_pause()
        elif not should_pause and self._rnd_paused:
            if self.state == State.PAUSED:
                logger.info("RND uncertainty cleared - resuming")
                self.handle_resume()
        self._rnd_paused = should_pause

    def _rnd_uncertainty_cb(self, msg: "Float32"):
        self._rnd_uncertainty = msg.data

    # ========== Aria Callbacks ==========

    def _aria_gaze_cb(self, msg: "String"):
        """Handle gaze gesture triggers (1.5s hold).

        LEFT  → Pause (if running)
        RIGHT → Start (if idle) / Resume (if paused)

        In classifier modes (trained_model / single_episode):
          RIGHT + IDLE → wait for frames → classify → load → start
          RIGHT + PAUSED → resume
        """
        raw_val = msg.data
        direction = raw_val.lower().strip()
        logger.info(f"👁️ GAZE MSG received: repr={repr(raw_val)} → direction='{direction}' | state={self.state}")
        if direction == "left":
            if self.state == State.RUNNING:
                logger.info("👁️ LEFT GAZE - Pausing")
                self.handle_pause()
        elif direction == "right":
            if self.state == State.IDLE:
                logger.info("👁️ RIGHT GAZE - Starting")
                if self._selection_mode in ("trained_model", "single_episode"):
                    t = threading.Thread(
                        target=self._classify_and_start,
                        daemon=True,
                    )
                    t.start()
                elif self._selection_mode == "monolithic":
                    # Switch to correct pre-loaded policy for current stage then start
                    stage = self._current_stage
                    idx = 0 if stage == Stage.POD else 1
                    policy_name = self.loaded_policies[idx].name if idx < len(self.loaded_policies) else None
                    logger.info(f"👁️ RIGHT GAZE - Monolithic stage={stage.value}, using policy {idx} ({policy_name})")
                    selection = self._policy_selector.select_policy(stage)
                    if self.load_selected_policy(selection):
                        self.handle_start()
                    else:
                        logger.error("Failed to switch policy for monolithic start")
                else:
                    self.handle_start()
            elif self.state == State.PAUSED:
                logger.info("👁️ RIGHT GAZE - Resuming")
                self.handle_resume()

    def _classify_and_start(self):
        """Wait for classifier predictions from dep_classifier_node, then start.

        Clears stale predictions, waits _CLASSIFY_WARMUP_SECS for fresh
        predictions to accumulate on /dep/classifier/status, then picks
        the most confident class via majority vote.
        """
        stage = self._current_stage
        warmup = self._CLASSIFY_WARMUP_SECS

        logger.info(
            f"🔍 _classify_and_start | stage={stage.value} | "
            f"waiting {warmup}s for classifier predictions..."
        )

        # Clear old predictions so we only use fresh ones
        with self._classifier_pred_lock:
            if stage == Stage.POD:
                self._pod_pred_history.clear()
            else:
                self._cup_pred_history.clear()

        # Wait for predictions to accumulate
        t_start = time.time()
        time.sleep(warmup)

        # Collect predictions received during the warmup window
        with self._classifier_pred_lock:
            history = (self._pod_pred_history if stage == Stage.POD
                       else self._cup_pred_history)
            recent = [(t, cls, conf) for t, cls, conf in history if t >= t_start]

        if not recent:
            logger.error(
                f"No classifier predictions received during {warmup}s — "
                f"is dep_classifier_node running? Staying IDLE."
            )
            return

        # Majority vote weighted by confidence
        vote_counts: dict[str, int] = {}
        conf_sums: dict[str, float] = {}
        for _, cls, conf in recent:
            vote_counts[cls] = vote_counts.get(cls, 0) + 1
            conf_sums[cls] = conf_sums.get(cls, 0.0) + conf

        best_class = max(conf_sums, key=conf_sums.get)
        avg_conf = conf_sums[best_class] / vote_counts[best_class]

        logger.info(
            f"🎯 Classification: {best_class} "
            f"(conf={avg_conf:.2f}, votes={vote_counts[best_class]}/{len(recent)})"
        )

        # Build ClassifierResult for the policy selector
        total_conf = sum(conf_sums.values())
        probs = {cls: conf_sums[cls] / total_conf for cls in conf_sums} if total_conf > 0 else {}
        result = ClassifierResult(
            stage=stage,
            predicted_class=best_class,
            confidence=avg_conf,
            raw_probs=probs,
        )

        selection = self._policy_selector.select_policy(stage, result)
        if selection is None:
            logger.error("Policy selection failed — staying IDLE")
            return

        logger.info(
            f"Selected: {selection.variant} ({selection.source}) "
            f"for {stage.value}"
        )

        if not self.load_selected_policy(selection):
            logger.error("Failed to load selected policy — staying IDLE")
            return

        self.handle_start()

    def _aria_eyes_closed_cb(self, msg: "Bool"):
        """Eyes closed → advance stage / complete.

        In classifier modes:
          POD stage → pause, advance to CUP, wait 5s, classify CUP, start
          CUP stage → COMPLETE
        In monolithic mode:
          RUNNING/PAUSED → rewind as before
        """
        if not msg.data:
            return
        if self.state not in [State.RUNNING, State.PAUSED]:
            return

        if self._selection_mode in ("trained_model", "single_episode", "monolithic"):
            if self._current_stage == Stage.POD:
                logger.info("👁️ EYES CLOSED - Pod stage done, advancing to CUP")
                logger.info("   Look RIGHT at the cup to start CUP stage")
                # Stop current execution, go IDLE, wait for RIGHT gaze
                self._transition_to(State.IDLE)
                self._episode_replayer = None
                self._current_stage = Stage.CUP
            else:
                logger.info("👁️ EYES CLOSED - Cup stage done, task complete")
                self._transition_to(State.COMPLETE)
        else:
            logger.info("👁️ EYES CLOSED - Resetting")
            self._reset_triggered.set()

    def _aria_double_blink_cb(self, msg: "Bool"):
        if not msg.data:
            return
        if self.state == State.IDLE:
            logger.info("👁️ DOUBLE BLINK - Starting")
            self.handle_start()
        elif self.state == State.RUNNING:
            logger.info("👁️ DOUBLE BLINK - Pausing")
            self.handle_pause()
        elif self.state == State.PAUSED:
            logger.info("👁️ DOUBLE BLINK - Resuming")
            self.handle_resume()

    # ========== Task End Detection ==========

    def _check_task_end(self, observation: dict, action: torch.Tensor):
        """Dual RND monitoring each control loop step.

        Top camera  → gradient completion (task end, policy 1 only)
        Aria camera → raw uncertainty (attention, always active)

        Wrapped in try/except so RND errors never crash the control loop.
        """
        task_end_cfg = self.orch_config.task_end
        action_b = action.unsqueeze(0) if action.dim() == 1 else action

        # Helper: extract state tensor from observation
        state_keys = [k for k in observation if "state" in k and isinstance(observation[k], torch.Tensor)]
        obs_state = observation[state_keys[0]].unsqueeze(0) if state_keys else None

        # ---- TOP CAMERA: gradient completion (only during policy 1) ----
        if self._completion_detector is not None and self._current_policy_idx == 0:
            top_key = task_end_cfg.top_camera_key
            if top_key in observation and isinstance(observation[top_key], torch.Tensor):
                try:
                    obs_img = observation[top_key]
                    if obs_img.dim() == 3:
                        obs_img = obs_img.unsqueeze(0)

                    # Match state/action dims to what top RND expects
                    _state = self._match_dims(obs_state, self._top_rnd.state_dim, obs_img.device)
                    _action = self._match_dims(action_b, self._top_rnd.action_dim, obs_img.device)

                    is_complete, confidence = self._completion_detector.update(
                        obs_img, _state, _action
                    )

                    # Publish diagnostics
                    if hasattr(self, "_completion_pub"):
                        from std_msgs.msg import Float32MultiArray
                        status = self._completion_detector.get_status()
                        msg = Float32MultiArray()
                        msg.data = [
                            float(is_complete),
                            confidence,
                            status["current_uncertainty"],
                            status["smooth_gradient"],
                        ]
                        self._completion_pub.publish(msg)

                    if is_complete and not self._task_end_detected:
                        self._task_end_detected = True
                        logger.info(
                            f"🎯 TOP GRADIENT COMPLETION - confidence: {confidence:.2f}, "
                            f"gradient: {self._completion_detector.get_status()['smooth_gradient']:.6f}"
                        )
                        self._handle_task_end_trigger()
                        return
                except Exception as e:
                    logger.error(f"Top RND error (non-fatal): {e}", exc_info=False)

        # ---- ARIA CAMERA: attention monitoring (always active) ----
        if self._aria_rnd is not None:
            aria_key = task_end_cfg.aria_camera_key
            if aria_key in observation and isinstance(observation[aria_key], torch.Tensor):
                try:
                    obs_img = observation[aria_key]
                    if obs_img.dim() == 3:
                        obs_img = obs_img.unsqueeze(0)

                    # Match state/action dims to what aria RND expects
                    _state = self._match_dims(obs_state, self._aria_rnd.state_dim, obs_img.device)
                    _action = self._match_dims(action_b, self._aria_rnd.action_dim, obs_img.device)

                    step_unc, rolling_unc = self._aria_rnd.compute_uncertainty(
                        obs_img, _state, _action, normalize=True,
                    )

                    # Publish diagnostics
                    if hasattr(self, "_attention_pub"):
                        from std_msgs.msg import Float32MultiArray
                        msg = Float32MultiArray()
                        msg.data = [step_unc, rolling_unc, float(self._aria_high_unc_count)]
                        self._attention_pub.publish(msg)

                    # Attention check: sustained high uncertainty → pause
                    if rolling_unc > task_end_cfg.aria_uncertainty_threshold:
                        self._aria_high_unc_count += 1
                        if (
                            self._aria_high_unc_count >= task_end_cfg.aria_sustained_frames
                            and not self._aria_paused_by_attention
                            and self.state == State.RUNNING
                        ):
                            self._aria_paused_by_attention = True
                            logger.warning(
                                f"⚠️  ATTENTION LOST - aria uncertainty {rolling_unc:.2f} > "
                                f"{task_end_cfg.aria_uncertainty_threshold} for "
                                f"{self._aria_high_unc_count} frames — auto-pausing"
                            )
                            self.handle_pause()
                    else:
                        self._aria_high_unc_count = 0
                        # Auto-resume if we paused due to attention and it recovered
                        if (
                            self._aria_paused_by_attention
                            and rolling_unc < task_end_cfg.aria_resume_threshold
                            and self.state == State.PAUSED
                        ):
                            self._aria_paused_by_attention = False
                            logger.info(
                                f"✅ ATTENTION RECOVERED - aria uncertainty {rolling_unc:.2f} — "
                                f"resetting policy and resuming"
                            )
                            self.active_policy.reset()  # flush stale action queue
                            self.handle_resume()
                except Exception as e:
                    logger.error(f"Aria RND error (non-fatal): {e}", exc_info=False)

    @staticmethod
    def _match_dims(tensor, target_dim: int, device) -> torch.Tensor:
        """Pad or truncate a [B, D] tensor to [B, target_dim].

        Handles the case where the policy's observation state/action dim
        doesn't match what the RND model was trained with.
        """
        if tensor is None:
            return torch.zeros(1, target_dim, device=device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        current_dim = tensor.shape[-1]
        if current_dim == target_dim:
            return tensor
        elif current_dim < target_dim:
            return torch.nn.functional.pad(tensor, (0, target_dim - current_dim))
        else:
            return tensor[..., :target_dim]

    def _handle_task_end_trigger(self):
        """Handle task end: auto-switch policy and reset completion detector."""
        if self.orch_config.task_end.auto_switch_policy:
            logger.info("Auto-switching to next policy")
            self.handle_switch_policy()
        # Reset completion detector (top camera) — it won't run during policy 2
        if self._completion_detector is not None:
            self._completion_detector.reset()
        self._task_end_detected = False

    # ========== Recording ==========



    # ========== Main Control Loop ==========

    def _control_loop(self):
        """
        Synchronous control loop.

        This is the heart of the system: observe → infer → act in one
        tight loop with no network calls. Every action is in perfect
        temporal alignment with its observation.
        """
        logger.info("Control loop started")
        step = 0
        fps_window: list[float] = []
        _consecutive_comm_errors = 0
        _MAX_COMM_ERRORS = 3  # pause after this many consecutive failures
        _MAX_RECOVERY_WAIT = 30.0  # seconds to wait before giving up

        while self._running:
            loop_start = time.perf_counter()

            # Only run inference when in RUNNING state
            if self.state == State.RUNNING:
                try:
                    if self._selection_mode == "single_episode" and self._episode_replayer is not None:
                        # ── Single Episode: open-loop replay ──
                        action_dict = self._episode_replayer.next_action()
                        if action_dict is None:
                            logger.info(
                                f"Episode replay complete "
                                f"({self._episode_replayer.num_frames} frames)"
                            )
                            self._transition_to(State.PAUSED)
                            continue

                        performed = self.robot.send_action(action_dict)

                        if self.buffer.is_recording:
                            record = performed if performed else action_dict
                            self.buffer.record_frame(record)

                        if step % 300 == 0:
                            logger.info(
                                f"Replay progress: "
                                f"{self._episode_replayer.progress:.0%} "
                                f"({self._episode_replayer._cursor}/"
                                f"{self._episode_replayer.num_frames})"
                            )
                    else:
                        # ── Trained Model / Monolithic: closed-loop inference ──
                        # 1. OBSERVE - capture from robot hardware
                        raw_obs = self.robot.get_observation()
                        observation = self._prepare_observation(raw_obs)

                        # 2. INFER - direct policy call, zero latency
                        action = self.active_policy.select_action(
                            observation, use_amp=self.orch_config.use_amp
                        )

                        # 3. ACT - send to robot
                        action_dict = self._action_to_dict(action)
                        performed = self.robot.send_action(action_dict)

                        # 4. RECORD to buffer for rewind
                        if self.buffer.is_recording:
                            record = performed if performed else action_dict
                            self.buffer.record_frame(record)

                        # 5. (rosbag records topics automatically via subprocess)

                        # 6. CHECK task end (optional)
                        if self.orch_config.task_end.enabled:
                            self._check_task_end(observation, action)

                    step += 1
                    _consecutive_comm_errors = 0  # reset on success

                except Exception as e:
                    _consecutive_comm_errors += 1
                    is_comm_error = "sync read" in str(e).lower() or "status packet" in str(e).lower()

                    if is_comm_error and _consecutive_comm_errors >= _MAX_COMM_ERRORS:
                        logger.warning(
                            f"⚠️  SERVO COMM LOST ({_consecutive_comm_errors} consecutive failures) — auto-pausing"
                        )
                        self._transition_to(State.PAUSED)

                        # Wait for comms to recover
                        t0 = time.time()
                        recovered = False
                        while time.time() - t0 < _MAX_RECOVERY_WAIT and self._running:
                            time.sleep(0.5)
                            try:
                                self.robot.get_observation()
                                recovered = True
                                break
                            except Exception:
                                elapsed = time.time() - t0
                                if int(elapsed) % 5 == 0:
                                    logger.info(f"  Waiting for servo comms... ({elapsed:.0f}s)")

                        if recovered:
                            logger.info("✅ Servo comms recovered — resetting policy and resuming")
                            self.active_policy.reset()  # flush stale action queue
                            _consecutive_comm_errors = 0
                            self._transition_to(State.RUNNING)
                        else:
                            logger.error(
                                f"❌ Servo comms not recovered after {_MAX_RECOVERY_WAIT}s — staying paused"
                            )
                    elif not is_comm_error:
                        logger.error(f"Control loop error: {e}", exc_info=True)

            # Check for reset trigger (from Aria eyes closed)
            if self._reset_triggered.is_set():
                self._reset_triggered.clear()
                self.handle_reset()

            # Check episode timeout
            if (
                self.state == State.RUNNING
                and self._episode_start_time > 0
                and time.time() - self._episode_start_time
                > self.orch_config.max_episode_time_s
            ):
                logger.info("Episode timeout reached")
                self._transition_to(State.COMPLETE)

            # Rate limiting
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, self._dt - elapsed)
            time.sleep(sleep_time)

            # FPS tracking
            actual_dt = time.perf_counter() - loop_start
            fps_window.append(1.0 / actual_dt if actual_dt > 0 else 0)
            if len(fps_window) > 100:
                fps_window.pop(0)
            if step > 0 and step % 300 == 0:
                avg_fps = sum(fps_window) / len(fps_window)
                policy_name = self.active_policy.name if self.active_policy else "episode_replay"
                logger.info(
                    f"Step {step} | FPS: {avg_fps:.1f} | "
                    f"Buffer: {self.buffer.get_stats()['buffer_size']} | "
                    f"Policy: {policy_name}"
                )

        logger.info(f"Control loop stopped after {step} steps")

    def _action_to_dict(self, action: torch.Tensor) -> dict[str, float]:
        """Convert action tensor to named dict for robot.send_action()."""
        action = action.squeeze(0).cpu()  # Remove batch dim
        return {
            key: action[i].item()
            for i, key in enumerate(self.robot.action_features)
        }

    # ========== Main Entry ==========

    def run(self):
        """Run the synchronous orchestrator."""
        self._running = True

        # Treat SIGTERM (pkill) the same as Ctrl+C so the finally cleanup runs
        import signal as _signal
        _signal.signal(_signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

        logger.info("=" * 60)
        logger.info("SYNC MULTI-POLICY ORCHESTRATOR")
        logger.info("=" * 60)
        logger.info(
            "ROS2 services: /orchestrator/{start,pause,resume,reset,stop,switch_policy}"
        )
        logger.info(
            f"Policies: {[p.name for p in self.loaded_policies]}"
        )
        active_name = self.active_policy.name if self.active_policy else "(awaiting classification)"
        logger.info(f"Selection: {self._selection_mode} | Stage: {self._current_stage.value}")
        logger.info(f"Active: {active_name}")
        logger.info(f"FPS: {self.config.fps}")
        logger.info(
            f"SARM: {'enabled' if self.orch_config.sarm.enabled else 'disabled'}"
        )
        logger.info(
            f"RND: {'enabled' if self.orch_config.rnd.enabled else 'disabled'}"
        )
        logger.info(
            f"Aria: {'enabled' if self.orch_config.aria.enabled else 'disabled'}"
        )
        logger.info(
            f"Task End: {'enabled' if self.orch_config.task_end.enabled else 'disabled'}"
        )
        logger.info(
            f"Rosbag: {'enabled' if self._rosbag_cfg.enabled else 'disabled'}"
            + (f" → {self._rosbag_cfg.output_dir}" if self._rosbag_cfg.enabled else "")
        )
        logger.info("=" * 60)

        try:
            self._control_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self._running = False
            # 1. Stop rosbag recording (skip if handle_stop/COMPLETE already handled it)
            if self._rosbag_proc is not None:
                if self._stop_service_called or self._episode_succeeded:
                    self._stop_rosbag()
                    self._upload_rosbag()
                else:
                    self._stop_and_delete_rosbag()
            # 2. Disconnect robot
            self.robot.disconnect()
            logger.info("Orchestrator stopped")

    def stop(self):
        """Stop the orchestrator."""
        self._running = False


# ========== ROS2 Node Wrapper ==========


class SyncOrchestratorNode(Node):
    """ROS2 node wrapper for the sync orchestrator."""

    def __init__(self, config: SyncMultiPolicyConfig):
        super().__init__("sync_multi_policy_orchestrator")
        self.orchestrator = SyncMultiPolicyOrchestrator(config, ros_node=self)
        self.get_logger().info("Sync orchestrator node initialized")

    def run(self):
        """Run orchestrator with ROS2 spinning."""
        self._spin_thread = threading.Thread(
            target=lambda: rclpy.spin(self), daemon=True
        )
        self._spin_thread.start()
        try:
            self.orchestrator.run()
        finally:
            # Stop ROS2 spin before shutdown to avoid segfault
            self.destroy_node()


# ========== Entry Point ==========


@draccus.wrap()
def run_sync_orchestrator(cfg: SyncMultiPolicyConfig):
    """Entry point for the synchronous multi-policy orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # override any existing logging config
    )
    logger.info(pformat(asdict(cfg)))

    if ROS2_AVAILABLE:
        rclpy.init()
        try:
            node = SyncOrchestratorNode(cfg)
            node.run()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
    else:
        logger.warning("ROS2 not available - running without ROS2 services")
        orchestrator = SyncMultiPolicyOrchestrator(cfg)
        orchestrator.run()


if __name__ == "__main__":
    run_sync_orchestrator()
