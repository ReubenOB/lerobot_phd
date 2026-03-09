#!/usr/bin/env python3
"""
Multi-Policy Orchestrator - The brain for robot control.

This module provides the main state machine and coordination for
multi-policy execution including:
- State transitions: IDLE -> RUNNING -> COMPLETE
- Policy switching (manual or SARM-triggered)
- Pause/resume control (manual, RND-triggered, or Aria gestures)
- Data recording coordination
- Movement buffer for reset/rewind

Optional integrations (enabled via config):
- SARM progress monitoring for automatic policy switching
- RND uncertainty for automatic pause/resume
- Aria gestures for hands-free control

Run:
    python -m lerobot.async_inference.orchestrator --config_path=launch/orchestrator.yaml
"""

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pprint import pformat
from typing import Optional, Callable

import draccus

# Import camera configs to register them with draccus
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
try:
    from lerobot.cameras.ros2.configuration_ros2 import ROS2CameraConfig  # noqa: F401
except ImportError:
    pass  # ROS2 not available (e.g. running outside container without cv_bridge)
from lerobot.robots import RobotConfig
from lerobot.robots.bi_so101_follower import BiSO101FollowerConfig  # noqa: F401

from .multi_policy_client import PolicyClient, PolicyServerSpec

# Try to import ROS2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32, Int32, String, Bool
    from std_srvs.srv import Trigger
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object

logger = logging.getLogger("orchestrator")


# ========== States ==========

class State(Enum):
    """Orchestrator states."""
    IDLE = "idle"           # Waiting to start
    RUNNING = "running"     # Executing policy
    PAUSED = "paused"       # Paused (manual, RND, or Aria)
    RESETTING = "resetting" # Executing reset trajectory
    COMPLETE = "complete"   # Episode complete


# ========== Configuration ==========

@dataclass
class SARMConfig:
    """SARM progress monitoring configuration (optional)."""
    enabled: bool = False
    progress_topic: str = "/sarm/progress"
    stage_topic: str = "/sarm/stage"
    stage_name_topic: str = "/sarm/stage_name"
    # Threshold for policy switch (0-1, progress at which to switch to next policy)
    policy_switch_threshold: float = 0.95


@dataclass
class RNDConfig:
    """RND uncertainty monitoring configuration (optional)."""
    enabled: bool = False
    pause_topic: str = "/robot/pause"  # Now uses ros2_bridge's pause topic
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
    """Task end detection configuration (optional).
    
    Uses RND uncertainty and action variance to detect when a policy's
    task is complete, enabling automatic policy switching.
    """
    enabled: bool = False
    # Path to trained RND model (universal or policy-specific)
    rnd_model_path: str = ""
    # Normalized uncertainty threshold (2.0 = 2 std above mean)
    uncertainty_threshold: float = 2.0
    # Action variance threshold (below this = stationary)
    action_variance_threshold: float = 0.01
    # Consecutive frames needed to trigger detection
    min_sustained_frames: int = 10
    # Topic to publish task end detection results
    task_end_topic: str = "/orchestrator/task_end"
    # Whether to auto-switch policy on task end
    auto_switch_policy: bool = True


@dataclass
class OrchestratorConfig:
    """Main orchestrator configuration."""
    # Episode settings
    num_episodes: int = 1
    max_episode_time_s: float = 300.0
    transition_delay_s: float = 2.0
    
    # Reset settings
    reset_playback_speed: float = 0.5
    
    # Task descriptions
    policy_1_task: str = "pick up the object"
    policy_2_task: str = "place the object"
    
    # Optional integrations
    sarm: SARMConfig = field(default_factory=SARMConfig)
    rnd: RNDConfig = field(default_factory=RNDConfig)
    aria: AriaConfig = field(default_factory=AriaConfig)
    task_end: TaskEndConfig = field(default_factory=TaskEndConfig)


@dataclass
class MultiPolicyConfig:
    """Complete configuration for multi-policy orchestrator."""
    # Robot
    robot: RobotConfig = field(default_factory=lambda: BiSO101FollowerConfig())
    
    # Policy servers (list of dicts, converted to PolicyServerSpec)
    policy_servers: list = field(default_factory=list)
    
    # Client settings
    fps: float = 30.0
    environment_dt: float = 0.033
    buffer_max_frames: int = 3000
    chunk_size_threshold: float = 0.5
    
    # Orchestrator settings
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    
    def get_policy_servers(self) -> list[PolicyServerSpec]:
        """Convert policy_servers dict list to PolicyServerSpec objects."""
        return [
            PolicyServerSpec(
                name=ps["name"],
                address=ps["address"],
                pretrained_path=ps["pretrained_path"],
                policy_type=ps.get("policy_type", "act"),
                actions_per_chunk=ps.get("actions_per_chunk", 15),
                device=ps.get("device", "cuda"),
                task=ps.get("task", "pick up the object")
            )
            for ps in self.policy_servers
        ]


# ========== Orchestrator ==========

class MultiPolicyOrchestrator:
    """
    Main orchestrator for multi-policy robot execution.
    
    This is THE brain. Responsibilities:
    - State machine (IDLE/RUNNING/PAUSED/RESETTING/COMPLETE)
    - Policy switching decisions
    - ROS2 service handlers for manual control
    - Optional: SARM progress monitoring
    - Optional: RND uncertainty pause/resume
    - Optional: Aria gesture control
    
    Uses PolicyClient for actual robot control.
    """
    
    def __init__(self, config: MultiPolicyConfig, ros_node: Optional["Node"] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config: Multi-policy configuration
            ros_node: Optional ROS2 node for topic subscriptions
        """
        self.config = config
        self.orch_config = config.orchestrator
        self.ros_node = ros_node
        
        # State machine
        self._state = State.IDLE
        self._state_lock = threading.Lock()
        self._previous_state = State.IDLE  # For pause/resume
        
        # Control flags
        self._running = False
        self._reset_triggered = threading.Event()
        
        # Timing
        self._state_entry_time = time.time()
        self._episode_start_time = 0.0
        
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
        
        # Initialize policy client
        self._init_client()
        
        # Setup task end detector if configured
        if self.orch_config.task_end.enabled:
            self._init_task_end_detector()
        
        # Setup ROS2 if available
        if ros_node is not None and ROS2_AVAILABLE:
            self._setup_ros2(ros_node)
        
        logger.info("MultiPolicyOrchestrator initialized")
        logger.info(f"  SARM: {'enabled' if self.orch_config.sarm.enabled else 'disabled'}")
        logger.info(f"  RND:  {'enabled' if self.orch_config.rnd.enabled else 'disabled'}")
        logger.info(f"  Aria: {'enabled' if self.orch_config.aria.enabled else 'disabled'}")
        logger.info(f"  Task End: {'enabled' if self.orch_config.task_end.enabled else 'disabled'}")
    
    def _init_client(self):
        """Initialize the policy client."""
        self.client = PolicyClient(
            robot_config=self.config.robot,
            policy_servers=self.config.get_policy_servers(),
            buffer_max_frames=self.config.buffer_max_frames,
            fps=self.config.fps,
            chunk_size_threshold=self.config.chunk_size_threshold,
            environment_dt=self.config.environment_dt,
        )
    
    def _init_task_end_detector(self):
        """Initialize task end detector with RND model."""
        try:
            from lerobot.common.uncertainty import RNDModuleUniversal, TaskEndDetector
            from pathlib import Path
            
            task_end_cfg = self.orch_config.task_end
            rnd_path = Path(task_end_cfg.rnd_model_path)
            
            if not rnd_path.exists():
                logger.warning(f"Task end RND model not found: {rnd_path}")
                return
            
            # Load RND model
            logger.info(f"Loading task end RND model from {rnd_path}")
            rnd_module = RNDModuleUniversal.load(rnd_path, device="cuda")
            
            # Create detector
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
    
    # ========== Properties ==========
    
    @property
    def state(self) -> State:
        with self._state_lock:
            return self._state
    
    @property
    def running(self) -> bool:
        return self._running
    
    @property
    def active_policy(self) -> PolicyServerSpec:
        return self.client.active_policy
    
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
            self.client.pause()
            self.client.stop_buffer_recording()
            
        elif new_state == State.RUNNING:
            self.client.resume()
            self.client.start_buffer_recording()
            
        elif new_state == State.PAUSED:
            self._previous_state = old_state
            self.client.pause()
            
        elif new_state == State.RESETTING:
            self.client.pause()
            self.client.stop_buffer_recording()
            
        elif new_state == State.COMPLETE:
            self.client.pause()
            self.client.stop_buffer_recording()
    
    # ========== ROS2 Setup ==========
    
    def _setup_ros2(self, node: "Node"):
        """Set up ROS2 services and subscriptions."""
        
        # Control services (always enabled)
        node.create_service(Trigger, "/orchestrator/start", self._srv_start)
        node.create_service(Trigger, "/orchestrator/pause", self._srv_pause)
        node.create_service(Trigger, "/orchestrator/resume", self._srv_resume)
        node.create_service(Trigger, "/orchestrator/reset", self._srv_reset)
        node.create_service(Trigger, "/orchestrator/stop", self._srv_stop)
        node.create_service(Trigger, "/orchestrator/switch_policy", self._srv_switch_policy)
        
        # State publisher
        self._state_pub = node.create_publisher(String, "/orchestrator/state", 10)
        node.create_timer(0.5, self._publish_state)
        
        logger.info("ROS2 services: /orchestrator/{start,pause,resume,reset,stop,switch_policy}")
        
        # SARM subscriptions (optional)
        if self.orch_config.sarm.enabled:
            node.create_subscription(
                Float32, self.orch_config.sarm.progress_topic,
                self._sarm_progress_cb, 10
            )
            node.create_subscription(
                Int32, self.orch_config.sarm.stage_topic,
                self._sarm_stage_cb, 10
            )
            node.create_subscription(
                String, self.orch_config.sarm.stage_name_topic,
                self._sarm_stage_name_cb, 10
            )
            logger.info(f"SARM subscriptions: {self.orch_config.sarm.progress_topic}")
        
        # RND subscription (optional) - uses ros2_bridge's pause topic
        if self.orch_config.rnd.enabled:
            node.create_subscription(
                Bool, self.orch_config.rnd.pause_topic,
                self._rnd_pause_cb, 10
            )
            node.create_subscription(
                Float32, self.orch_config.rnd.uncertainty_topic,
                self._rnd_uncertainty_cb, 10
            )
            logger.info(f"RND subscriptions: {self.orch_config.rnd.pause_topic}")
        
        # Aria subscriptions (optional)
        if self.orch_config.aria.enabled:
            node.create_subscription(
                String, self.orch_config.aria.gaze_topic,
                self._aria_gaze_cb, 10
            )
            node.create_subscription(
                Bool, self.orch_config.aria.eyes_closed_topic,
                self._aria_eyes_closed_cb, 10
            )
            node.create_subscription(
                Bool, self.orch_config.aria.double_blink_topic,
                self._aria_double_blink_cb, 10
            )
            logger.info(f"Aria subscriptions: {self.orch_config.aria.gaze_topic}")
        
        # Task end detection publisher (optional)
        if self.orch_config.task_end.enabled and self._task_end_detector is not None:
            from std_msgs.msg import Float32MultiArray
            self._task_end_pub = node.create_publisher(
                Float32MultiArray, 
                self.orch_config.task_end.task_end_topic, 
                10
            )
            logger.info(f"Task end detection enabled, publishing to {self.orch_config.task_end.task_end_topic}")
    
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
        if not hasattr(self, '_state_pub'):
            return
        
        stats = self.client.get_buffer_stats()
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
            # Send policy setup
            self.client.send_policy_setup()
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
        """Trigger reset trajectory."""
        if self.state in [State.RUNNING, State.PAUSED]:
            self._transition_to(State.RESETTING)
            
            # Execute reset (blocking)
            success = self.client.execute_reset_trajectory(
                playback_speed=self.orch_config.reset_playback_speed
            )
            
            self.client.clear_buffer()
            self._transition_to(State.IDLE)
            
            return success, "Reset complete" if success else "Reset failed"
        return False, f"Cannot reset from {self.state.value}"
    
    def handle_stop(self) -> tuple[bool, str]:
        """Stop the orchestrator."""
        self._running = False
        return True, "Stopping"
    
    def handle_switch_policy(self) -> tuple[bool, str]:
        """Switch to the next policy."""
        was_running = self.state == State.RUNNING
        
        if was_running:
            self._transition_to(State.PAUSED)
        
        old_name = self.active_policy.name
        new_name = self.client.switch_policy()
        
        if was_running:
            self._transition_to(State.RUNNING)
        
        return True, f"Switched: {old_name} → {new_name}"
    
    # ========== SARM Callbacks (Optional) ==========
    
    def _sarm_progress_cb(self, msg: "Float32"):
        """Handle SARM progress update."""
        self._sarm_progress = msg.data
        
        # Auto-switch policy at threshold
        if self.state == State.RUNNING:
            if self._sarm_progress >= self.orch_config.sarm.policy_switch_threshold:
                logger.info(f"SARM progress {self._sarm_progress:.3f} >= threshold, switching policy")
                self.handle_switch_policy()
    
    def _sarm_stage_cb(self, msg: "Int32"):
        """Handle SARM stage update."""
        self._sarm_stage = msg.data
    
    def _sarm_stage_name_cb(self, msg: "String"):
        """Handle SARM stage name update."""
        self._sarm_stage_name = msg.data
    
    # ========== RND Callbacks (Optional) ==========
    
    def _rnd_pause_cb(self, msg: "Bool"):
        """Handle RND pause signal."""
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
        """Handle RND uncertainty update."""
        self._rnd_uncertainty = msg.data
    
    # ========== Aria Callbacks (Optional) ==========
    
    def _aria_gaze_cb(self, msg: "String"):
        """Handle Aria gaze gesture."""
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
        """Handle Aria eyes closed for reset."""
        if msg.data and self.state in [State.RUNNING, State.PAUSED]:
            logger.info("👁️ EYES CLOSED - Resetting")
            self._reset_triggered.set()
    
    def _aria_double_blink_cb(self, msg: "Bool"):
        """Handle Aria double blink for toggle."""
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
    
    # ========== Task End Detection (Optional) ==========
    
    def update_task_end_detection(self, obs_img, action, obs_state=None):
        """
        Update task end detector with current observation and action.
        
        Call this from the control loop when task_end is enabled.
        
        Args:
            obs_img: Current image observation [B, 3, H, W] or [3, H, W]
            action: Current/predicted action [B, action_dim] or [action_dim]
            obs_state: Optional state observation
        
        Returns:
            Dict with detection results or None if detector not available
        """
        if self._task_end_detector is None:
            return None
        
        import torch
        
        # Ensure proper dimensions
        if obs_img.dim() == 3:
            obs_img = obs_img.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if obs_state is not None and obs_state.dim() == 1:
            obs_state = obs_state.unsqueeze(0)
        
        # Update detector
        result = self._task_end_detector.update(obs_img, action, obs_state)
        
        # Publish results if ROS2 available
        if hasattr(self, '_task_end_pub'):
            from std_msgs.msg import Float32MultiArray
            msg = Float32MultiArray()
            msg.data = [
                float(result['task_end_detected']),
                result['uncertainty'],
                result['action_variance'],
                float(result['consecutive_frames']),
            ]
            self._task_end_pub.publish(msg)
        
        # Handle task end if detected
        if result['task_end_detected'] and not self._task_end_detected:
            self._task_end_detected = True
            logger.info(f"🎯 TASK END DETECTED - uncertainty: {result['uncertainty']:.2f}, "
                       f"action_variance: {result['action_variance']:.4f}")
            
            if self.orch_config.task_end.auto_switch_policy:
                logger.info("Auto-switching to next policy")
                self.handle_switch_policy()
                # Reset detector for next task
                self._task_end_detector.reset()
                self._task_end_detected = False
        
        return result
    
    def reset_task_end_detector(self):
        """Reset task end detector for new episode/task."""
        if self._task_end_detector is not None:
            self._task_end_detector.reset()
            self._task_end_detected = False
    
    # ========== Main Loop ==========
    
    def run(self):
        """Run the orchestrator."""
        self._running = True
        
        # Connect to servers
        self.client.connect()
        
        # Start client threads
        action_thread = threading.Thread(target=self.client.action_receiver_loop, daemon=True)
        control_thread = threading.Thread(target=self.client.control_loop, daemon=True)
        
        action_thread.start()
        control_thread.start()
        
        logger.info("=" * 60)
        logger.info("ORCHESTRATOR RUNNING")
        logger.info("=" * 60)
        logger.info("ROS2 services: /orchestrator/{start,pause,resume,reset,stop,switch_policy}")
        logger.info(f"Available policies: {[p.name for p in self.client.policy_servers]}")
        logger.info(f"SARM: {'enabled' if self.orch_config.sarm.enabled else 'disabled'}")
        logger.info(f"RND: {'enabled' if self.orch_config.rnd.enabled else 'disabled'}")
        logger.info(f"Aria: {'enabled' if self.orch_config.aria.enabled else 'disabled'}")
        logger.info(f"Task End: {'enabled' if self.orch_config.task_end.enabled else 'disabled'}")
        logger.info("=" * 60)
        
        try:
            while self._running:
                # Check for reset trigger (from Aria eyes closed)
                if self._reset_triggered.is_set():
                    self._reset_triggered.clear()
                    self.handle_reset()
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            logger.info("Interrupted")
        
        finally:
            self.client.disconnect()
            action_thread.join(timeout=2)
            control_thread.join(timeout=2)
            logger.info("Orchestrator stopped")
    
    def stop(self):
        """Stop the orchestrator."""
        self._running = False


# ========== ROS2 Node Wrapper ==========

class OrchestratorNode(Node):
    """ROS2 node wrapper for the orchestrator."""
    
    def __init__(self, config: MultiPolicyConfig):
        super().__init__('multi_policy_orchestrator')
        self.orchestrator = MultiPolicyOrchestrator(config, ros_node=self)
        self.get_logger().info("Orchestrator node initialized")
    
    def run(self):
        """Run orchestrator with ROS2 spinning."""
        # Spin ROS2 in background
        spin_thread = threading.Thread(target=lambda: rclpy.spin(self), daemon=True)
        spin_thread.start()
        
        # Run orchestrator
        self.orchestrator.run()


# ========== Entry Point ==========

@draccus.wrap()
def run_orchestrator(cfg: MultiPolicyConfig):
    """Entry point for the multi-policy orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    logger.info(pformat(asdict(cfg)))
    
    if ROS2_AVAILABLE:
        rclpy.init()
        try:
            node = OrchestratorNode(cfg)
            node.run()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
    else:
        logger.warning("ROS2 not available - running without ROS2 services")
        orchestrator = MultiPolicyOrchestrator(cfg)
        orchestrator.run()


if __name__ == "__main__":
    run_orchestrator()
