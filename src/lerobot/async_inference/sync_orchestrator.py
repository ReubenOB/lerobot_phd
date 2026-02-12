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

import logging
import subprocess
import threading
import time
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
from lerobot.cameras.ros2.configuration_ros2 import ROS2CameraConfig  # noqa: F401
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
    """Task end detection configuration (optional)."""

    enabled: bool = False
    rnd_model_path: str = ""
    uncertainty_threshold: float = 2.0
    action_variance_threshold: float = 0.01
    min_sustained_frames: int = 10
    task_end_topic: str = "/orchestrator/task_end"
    auto_switch_policy: bool = True


@dataclass
class RosbagConfig:
    """Configuration for rosbag recording & optional HF Hub upload.

    Records all specified ROS2 topics to a rosbag via `ros2 bag record`.
    This runs as a subprocess — no in-process video encoding, no segfaults.
    """

    enabled: bool = False
    output_dir: str = "/home/rapob/vigil_ws/outputs/rosbags"
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
class SyncMultiPolicyConfig:
    """Complete configuration for synchronous multi-policy orchestrator."""

    # Robot
    robot: RobotConfig = field(default_factory=lambda: BiSO101FollowerConfig())

    # Policies (loaded locally, no gRPC)
    policies: list = field(default_factory=list)

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

        # Task end detector (optional)
        self._task_end_detector = None
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

        # Load all policies
        self.loaded_policies: list[LoadedPolicy] = []
        self._load_all_policies()

        # Rosbag recording (optional)
        self._rosbag_cfg = self.orch_config.rosbag
        self._rosbag_proc: subprocess.Popen | None = None
        self._rosbag_path: Path | None = None

        # Setup task end detector if configured
        if self.orch_config.task_end.enabled:
            self._init_task_end_detector()

        # Setup ROS2 if available
        if ros_node is not None and ROS2_AVAILABLE:
            self._setup_ros2(ros_node)

        logger.info("SyncMultiPolicyOrchestrator initialized")
        logger.info(f"  Policies: {[p.name for p in self.loaded_policies]}")
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

        # Load pretrained model directly - config already has features
        # Load on CPU first to avoid loading all models onto GPU
        policy = policy_class.from_pretrained(spec.pretrained_path)
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
        """Initialize task end detector with RND model."""
        try:
            from lerobot.common.uncertainty import RNDModuleUniversal, TaskEndDetector

            task_end_cfg = self.orch_config.task_end
            rnd_path = Path(task_end_cfg.rnd_model_path)

            if not rnd_path.exists():
                logger.warning(f"Task end RND model not found: {rnd_path}")
                return

            logger.info(f"Loading task end RND model from {rnd_path}")
            rnd_module = RNDModuleUniversal.load(rnd_path, device="cuda")

            self._task_end_detector = TaskEndDetector(
                rnd_module=rnd_module,
                uncertainty_threshold=task_end_cfg.uncertainty_threshold,
                action_variance_threshold=task_end_cfg.action_variance_threshold,
                min_sustained_frames=task_end_cfg.min_sustained_frames,
                device="cuda",
            )
            logger.info("Task end detector initialized")

        except Exception as e:
            logger.error(f"Failed to initialize task end detector: {e}")
            self._task_end_detector = None

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
            "--compression-mode", "file",
            "--compression-format", "zstd",
        ]
        # Add all topics
        cmd.extend(cfg.topics)

        logger.info(f"Starting rosbag: {bag_name}")
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

        logger.info("Stopping rosbag recording (waiting for zstd compression)...")
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
                        logger.info(f"  Still flushing compressed rosbag... ({i}s)")
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
    def active_policy(self) -> LoadedPolicy:
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
            # Stop rosbag and upload on completion
            self._stop_rosbag()
            self._upload_rosbag()

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

        # Task end publisher (optional)
        if (
            self.orch_config.task_end.enabled
            and self._task_end_detector is not None
        ):
            from std_msgs.msg import Float32MultiArray

            self._task_end_pub = node.create_publisher(
                Float32MultiArray,
                self.orch_config.task_end.task_end_topic,
                10,
            )
            logger.info(
                f"Task end detection enabled, publishing to {self.orch_config.task_end.task_end_topic}"
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
        msg = String()
        msg.data = (
            f"{self.state.value} | "
            f"policy: {self.active_policy.name} | "
            f"buffer: {stats['buffer_size']} | "
            f"sarm: {self._sarm_progress:.2f}"
        )
        self._state_pub.publish(msg)

    # ========== Action Handlers ==========

    def handle_start(self) -> tuple[bool, str]:
        """Start or resume execution."""
        if self.state == State.IDLE:
            self.active_policy.reset()
            self._episode_start_time = time.time()
            self._recording_frame_count = 0
            self._transition_to(State.RUNNING)
            return True, f"Started with policy '{self.active_policy.name}'"
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
        """Stop the orchestrator. Stops rosbag and uploads if configured."""
        self._stop_rosbag()
        self._upload_rosbag()
        self._running = False
        return True, "Stopping"

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
        """
        direction = msg.data.lower()
        if direction == "left":
            if self.state == State.RUNNING:
                logger.info("👁️ LEFT GAZE - Pausing")
                self.handle_pause()
        elif direction == "right":
            if self.state == State.IDLE:
                logger.info("👁️ RIGHT GAZE - Starting")
                self.handle_start()
            elif self.state == State.PAUSED:
                logger.info("👁️ RIGHT GAZE - Resuming")
                self.handle_resume()

    def _aria_eyes_closed_cb(self, msg: "Bool"):
        if msg.data and self.state in [State.RUNNING, State.PAUSED]:
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
        """Check if the current task has ended (via RND + action variance)."""
        if self._task_end_detector is None:
            return

        import torch as th

        # Extract image for RND
        img_keys = [
            k for k in observation if "image" in k and isinstance(observation[k], th.Tensor)
        ]
        if not img_keys:
            return

        obs_img = observation[img_keys[0]]
        if obs_img.dim() == 3:
            obs_img = obs_img.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        result = self._task_end_detector.update(obs_img, action)

        # Publish via ROS2
        if hasattr(self, "_task_end_pub"):
            from std_msgs.msg import Float32MultiArray

            msg = Float32MultiArray()
            msg.data = [
                float(result["task_end_detected"]),
                result["uncertainty"],
                result["action_variance"],
                float(result["consecutive_frames"]),
            ]
            self._task_end_pub.publish(msg)

        # Auto-switch on task end
        if result["task_end_detected"] and not self._task_end_detected:
            self._task_end_detected = True
            logger.info(
                f"🎯 TASK END DETECTED - uncertainty: {result['uncertainty']:.2f}, "
                f"action_variance: {result['action_variance']:.4f}"
            )
            if self.orch_config.task_end.auto_switch_policy:
                logger.info("Auto-switching to next policy")
                self.handle_switch_policy()
                if self._task_end_detector is not None:
                    self._task_end_detector.reset()
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

        while self._running:
            loop_start = time.perf_counter()

            # Only run inference when in RUNNING state
            if self.state == State.RUNNING:
                try:
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

                except Exception as e:
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
                logger.info(
                    f"Step {step} | FPS: {avg_fps:.1f} | "
                    f"Buffer: {self.buffer.get_stats()['buffer_size']} | "
                    f"Policy: {self.active_policy.name}"
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

        logger.info("=" * 60)
        logger.info("SYNC MULTI-POLICY ORCHESTRATOR")
        logger.info("=" * 60)
        logger.info(
            "ROS2 services: /orchestrator/{start,pause,resume,reset,stop,switch_policy}"
        )
        logger.info(
            f"Policies: {[p.name for p in self.loaded_policies]}"
        )
        logger.info(f"Active: {self.active_policy.name}")
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
            # 1. Stop rosbag recording
            self._stop_rosbag()
            self._upload_rosbag()
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
