#!/usr/bin/env python3
"""
Multi-Policy Test with ROS2 Control

Extended version of single_policy_test.py that supports:
- Multiple policy servers with automatic cycling
- Pause/Start/Reset via ROS2 services and Aria gaze
- Movement buffer for reset trajectory
- Policy switching via service calls (cycles to next policy)

Run:
    1. Policy Servers: 
       python -m lerobot.async_inference.policy_server --host=localhost --port=8080 --fps=30
       python -m lerobot.async_inference.policy_server --host=localhost --port=8081 --fps=30
    2. Robot Client:  
       python -m lerobot.async_inference.multi_policy_client --config_path=launch/multi_policy_test.yaml
    3. Control:       
       ros2 service call /robot/start std_srvs/srv/Trigger
       ros2 service call /robot/switch_policy std_srvs/srv/Trigger  # Cycles to next policy
"""

import argparse
import logging
import math
import pickle
import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
from typing import Optional

import draccus
import grpc
import torch

# LeRobot imports
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.ros2.configuration_ros2 import ROS2CameraConfig  # noqa: F401
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.bi_so101_follower import BiSO101FollowerConfig  # noqa: F401
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .movement_buffer import MovementBuffer
from .helpers import RemotePolicyConfig, TimedAction, TimedObservation, map_robot_keys_to_lerobot_features

# ROS2
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from std_srvs.srv import Trigger
    from std_msgs.msg import Bool, String
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("multi_policy_client")


class RobotState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    RESETTING = "resetting"


@dataclass
class PolicyServerSpec:
    """Specification for a single policy server."""
    name: str = field(metadata={"help": "Policy name identifier"})
    address: str = field(metadata={"help": "Server address (host:port)"})
    pretrained_path: str = field(metadata={"help": "Policy checkpoint path"})
    policy_type: str = field(default="act", metadata={"help": "Policy type"})
    actions_per_chunk: int = field(default=15, metadata={"help": "Actions per chunk"})
    device: str = field(default="cuda", metadata={"help": "Policy device"})
    task: str = field(default="", metadata={"help": "Task description"})


@dataclass
class MultiPolicyConfig:
    """Configuration for multi-policy test."""
    
    # Required fields (no defaults)
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})
    policy_servers: list[dict] = field(default_factory=list, metadata={"help": "List of policy server specs"})
    
    # Timing
    fps: int = field(default=30, metadata={"help": "Control loop FPS"})
    chunk_size_threshold: float = field(default=0.25, metadata={"help": "Chunk refill threshold"})
    
    # Buffer
    buffer_max_frames: int = field(default=3000, metadata={"help": "Max buffer frames"})
    reset_playback_speed: float = field(default=0.5, metadata={"help": "Reset speed multiplier"})
    
    @property
    def environment_dt(self) -> float:
        return 1.0 / self.fps
    
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


class MultiPolicyClient:
    """Multi-policy robot client with ROS2 control and policy switching."""
    
    def __init__(self, config: MultiPolicyConfig):
        self.config = config
        
        # State
        self._state = RobotState.IDLE
        self._state_lock = threading.RLock()
        self._running = True
        
        # Robot
        logger.info("Connecting robot...")
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()
        logger.info("Robot connected")
        
        # Policy servers
        self.policy_servers = config.get_policy_servers()
        if not self.policy_servers:
            raise ValueError("No policy servers configured")
        
        self.active_policy_idx = 0
        logger.info(f"Configured {len(self.policy_servers)} policy servers")
        
        # gRPC connections
        self.channels: dict[str, grpc.Channel] = {}
        self.stubs: dict[str, services_pb2_grpc.AsyncInferenceStub] = {}
        self._connect_to_servers()
        
        # Movement buffer
        self.buffer = MovementBuffer(max_frames=config.buffer_max_frames, validate_positions=False)
        logger.info(f"Movement buffer: {config.buffer_max_frames} frames")
        
        # Action queues (one per policy)
        self.action_queues: dict[str, Queue] = {spec.name: Queue() for spec in self.policy_servers}
        self.action_queue_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1
        
        # Threading
        self.start_barrier = threading.Barrier(2)
        self.must_go = threading.Event()
        self.must_go.set()
        
        # Pause mechanism (like RobotClientMulti)
        self.paused = threading.Event()  # Set = paused, clear = running
        
        logger.info("MultiPolicyClient initialized")
        
        # ROS2 publisher for robot pause control (if available)
        self.pause_publisher = None
        if ROS2_AVAILABLE:
            try:
                import rclpy
                from std_msgs.msg import Bool
                # Create a simple publisher node for pause control
                if not rclpy.ok():
                    rclpy.init()
                self._pause_node = rclpy.create_node('pause_publisher')
                self.pause_publisher = self._pause_node.create_publisher(Bool, '/robot/pause_control', 10)
                logger.info("✅ Robot pause publisher initialized")
            except Exception as e:
                logger.warning(f"⚠️  Could not initialize robot pause publisher: {e}")
    
    def _publish_robot_pause_command(self, paused: bool):
        """Send pause/resume command to robot (works with any pause source: RND, Aria glasses, manual, etc.)"""
        try:
            # Direct robot pause if available
            if hasattr(self.robot, 'ros2_bridge') and self.robot.ros2_bridge:
                self.robot.ros2_bridge.uncertainty_paused = paused
                logger.info(f"🤖 Robot pause set directly: {paused}")
            
            # Publish to pause topic for other systems
            if self.pause_publisher:
                pause_msg = Bool()
                pause_msg.data = paused
                self.pause_publisher.publish(pause_msg)
                action = "PAUSED" if paused else "RESUMED"
                logger.info(f"📢 Published robot {action} command")
                
        except Exception as e:
            logger.warning(f"⚠️  Could not send robot pause command: {e}")
    
    @property
    def state(self) -> RobotState:
        with self._state_lock:
            return self._state
    
    @state.setter
    def state(self, new_state: RobotState):
        with self._state_lock:
            old = self._state
            self._state = new_state
            logger.info(f"State: {old.value} → {new_state.value}")
            
            # Update pause mechanism based on state
            if new_state == RobotState.RUNNING:
                self.paused.clear()  # Unpause
                if not self.buffer.is_recording:
                    self.buffer.start_recording()
            elif new_state == RobotState.PAUSED:
                self.paused.set()  # Pause
            elif new_state == RobotState.IDLE:
                self.paused.set()  # Pause
                if self.buffer.is_recording:
                    self.buffer.stop_recording()
            elif new_state == RobotState.RESETTING:
                self.paused.set()  # Pause
                if self.buffer.is_recording:
                    self.buffer.stop_recording()
    
    @property
    def is_paused(self) -> bool:
        """Check if execution is paused (either custom pause OR robot uncertainty pause)."""
        custom_paused = self.paused.is_set()
        robot_paused = (hasattr(self.robot, 'ros2_bridge') and 
                       self.robot.ros2_bridge and 
                       self.robot.ros2_bridge.is_paused())
        
        # Debug logging when pause states differ
        if custom_paused != robot_paused:
            logger.debug(f"Pause state mismatch: custom={custom_paused}, robot={robot_paused}")
        
        return custom_paused or robot_paused
    
    @property
    def running(self) -> bool:
        return self._running
    
    @property
    def active_policy(self) -> PolicyServerSpec:
        """Get the currently active policy server spec."""
        return self.policy_servers[self.active_policy_idx]
    
    @property
    def active_stub(self) -> services_pb2_grpc.AsyncInferenceStub:
        """Get the stub for the currently active policy."""
        return self.stubs[self.active_policy.name]
    
    @property
    def active_queue(self) -> Queue:
        """Get the action queue for the currently active policy."""
        return self.action_queues[self.active_policy.name]
    
    def _connect_to_servers(self):
        """Connect to all policy servers and send initial configs."""
        for spec in self.policy_servers:
            logger.info(f"Connecting to {spec.name} at {spec.address}...")
            
            channel = grpc.insecure_channel(
                spec.address,
                grpc_channel_options(initial_backoff=f"{self.config.environment_dt:.4f}s")
            )
            self.channels[spec.name] = channel
            stub = services_pb2_grpc.AsyncInferenceStub(channel)
            self.stubs[spec.name] = stub
            
            # Wait for server
            while self._running:
                try:
                    stub.Ready(services_pb2.Empty())
                    break
                except grpc.RpcError:
                    time.sleep(0.5)
            
            logger.info(f"Connected to {spec.name}")
    
    def _send_policy_setup(self, spec: PolicyServerSpec):
        """Send policy configuration to a specific server."""
        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        policy_config = RemotePolicyConfig(
            spec.policy_type,
            spec.pretrained_path,
            lerobot_features,
            spec.actions_per_chunk,
            spec.device,
        )
        
        policy_bytes = pickle.dumps(policy_config)
        stub = self.stubs[spec.name]
        stub.SendPolicyInstructions(services_pb2.PolicySetup(data=policy_bytes))
        logger.info(f"Policy setup sent to {spec.name}")
    
    # ========== Service Handlers ==========
    
    def handle_start(self) -> tuple[bool, str]:
        if self.state in [RobotState.IDLE, RobotState.PAUSED]:
            # Send policy setup to active server
            self._send_policy_setup(self.active_policy)
            
            self.state = RobotState.RUNNING
            # Unpause robot for any control source (RND, Aria, manual, etc.)
            self._publish_robot_pause_command(False)
            return True, f"Started with policy '{self.active_policy.name}' (buffer: {self.buffer.frame_count} frames)"
        return False, f"Cannot start from {self.state.value}"
    
    def handle_pause(self) -> tuple[bool, str]:
        if self.state == RobotState.RUNNING:
            self.state = RobotState.PAUSED
            # Pause robot for manual control
            self._publish_robot_pause_command(True)
            return True, f"Paused (buffer: {self.buffer.frame_count} frames)"
        return False, f"Cannot pause from {self.state.value}"
    
    def handle_resume(self) -> tuple[bool, str]:
        if self.state == RobotState.PAUSED:
            self.state = RobotState.RUNNING
            # Resume robot after manual pause
            self._publish_robot_pause_command(False)
            return True, "Resumed"
        return False, f"Cannot resume from {self.state.value}"
    
    def handle_reset(self) -> tuple[bool, str]:
        if self.state in [RobotState.RUNNING, RobotState.PAUSED]:
            previous_state = self.state  # Remember state before reset
            self.state = RobotState.RESETTING
            
            # CRITICAL: Unpause robot BEFORE executing reset so it can move
            self._publish_robot_pause_command(False)
            
            success = self._execute_reset()
            self.buffer.clear()
            # Return to IDLE state after reset - requires explicit start to continue
            self.state = RobotState.IDLE
            return success, "Reset complete - call /robot/start to begin next attempt" if success else "Reset failed"
        return False, f"Cannot reset from {self.state.value}"
    
    def handle_stop(self) -> tuple[bool, str]:
        self._running = False
        return True, "Stopping"
    
    def handle_switch_policy(self) -> tuple[bool, str]:
        """Cycle to the next policy server."""
        # Pause if running
        was_running = False
        if self.state == RobotState.RUNNING:
            was_running = True
            self.state = RobotState.PAUSED
            logger.info("⏸️  Pausing before policy switch")
        
        old_policy = self.active_policy.name
        
        # Clear active policy's action queue
        with self.action_queue_lock:
            self.active_queue.queue.clear()
        
        # Switch to next policy
        self.active_policy_idx = (self.active_policy_idx + 1) % len(self.policy_servers)
        new_policy = self.active_policy.name
        
        # Send policy setup to new server
        self._send_policy_setup(self.active_policy)
        
        logger.info(f"🔀 Switched policy: {old_policy} → {new_policy}")
        
        # Resume if was running
        if was_running:
            self.state = RobotState.RUNNING
            logger.info("▶️  Resuming with new policy")
            return True, f"Switched to policy: {new_policy}"
        
        return True, f"Switched to policy: {new_policy}"
            # Switch active policy
        self.active_policy_idx = policy_idx
        logger.info(f"🔀 Switched policy: {old_policy} → {policy_name}")
        
        # Send policy setup to new server
        self._send_policy_setup(self.active_policy)
        
        # Set must_go to trigger replanning
        self.must_go.set()
        
        # Resume if was running
        if was_running:
            self.state = RobotState.RUNNING
            self._publish_robot_pause_command(False)
            logger.info(f"▶️  Resumed with new policy")
        
        return True, f"Switched to policy: {policy_name}"
    
    def _execute_reset(self) -> bool:
        """Replay buffer in reverse."""
        trajectory = self.buffer.get_reverse_trajectory(
            playback_speed=self.config.reset_playback_speed,
            smooth_window=1,  # No smoothing - preserve exact trajectory
        )
        
        if not trajectory:
            logger.warning("Empty buffer - cannot reset")
            return False
        
        logger.info(f"Resetting: {len(trajectory)} frames")
        dt = 1.0 / self.config.fps
        
        for i, positions in enumerate(trajectory):
            if not self._running or self.state != RobotState.RESETTING:
                return False
            
            try:
                self.robot.send_action(positions)
            except Exception as e:
                logger.error(f"Reset error: {e}")
                return False
            
            time.sleep(dt)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Reset progress: {i+1}/{len(trajectory)}")
        
        # CRITICAL: Clear old action chunks and force replanning
        self._clear_action_chunks_and_replan()
        
        return True
    
    def _clear_action_chunks_and_replan(self):
        """Clear action queue after reset - next start will trigger fresh planning."""
        logger.info("🧹 Clearing action chunks after reset...")
        
        # Clear active policy's action queue
        with self.action_queue_lock:
            while not self.active_queue.empty():
                try:
                    self.active_queue.get_nowait()
                except:
                    break
        
        # Set must_go flag so next start triggers immediate replanning
        self.must_go.set()
        logger.info("🎯 Action chunks cleared - next start will trigger fresh planning")
    
    # ========== Main Loops ==========
    
    def action_receiver_loop(self):
        """Thread: receive actions from active policy server."""
        self.start_barrier.wait()
        logger.info("Action receiver started")
        
        empty_count = 0
        while self._running:
            try:
                # Get actions from active policy server
                actions_chunk = self.active_stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    empty_count += 1
                    if empty_count % 100 == 0:
                        logger.debug(f"Empty action chunks: {empty_count}")
                    continue
                
                empty_count = 0
                timed_actions = pickle.loads(actions_chunk.data)
                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                
                logger.debug(f"Received {len(timed_actions)} actions from {self.active_policy.name}")
                
                # Add to active policy's queue
                with self.action_queue_lock:
                    for action in timed_actions:
                        if action.get_timestep() > self.latest_action:
                            self.active_queue.put(action)
                
                self.must_go.set()
                
            except grpc.RpcError as e:
                if self._running:
                    logger.error(f"Action receive error: {e}")
                    time.sleep(0.1)
        
        logger.info("Action receiver stopped")
    
    def control_loop(self):
        """Main control loop - follows RobotClientMulti pattern."""
        self.start_barrier.wait()
        logger.info("Control loop started - waiting for /robot/start service call")
        
        while self._running:
            loop_start = time.perf_counter()
            
            # Execute actions only when RUNNING (robot.send_action also handles pause checking)
            if self._has_actions() and self.state == RobotState.RUNNING:
                self._execute_action()
            
            # Always send observations (policy server needs them to generate actions)
            if self._ready_for_observation():
                self._send_observation_to_server()
            
            # Maintain loop rate
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, self.config.environment_dt - elapsed))
        
        logger.info("Control loop stopped")
    
    def _execute_action(self):
        """Execute an action from the queue and record to buffer."""
        try:
            with self.action_queue_lock:
                timed_action = self.active_queue.get_nowait()
            
            action_dict = self._action_to_dict(timed_action.get_action())
            logger.debug(f"Executing action at timestep {timed_action.get_timestep()}")
            
            # Send to robot (robot handles uncertainty pause internally)
            performed = self.robot.send_action(action_dict)
            self.latest_action = timed_action.get_timestep()
            
            # Always record to buffer (even if robot skipped due to uncertainty)
            # This allows trajectory recording during pause for later reset
            if self.buffer.is_recording:
                # Use performed if robot executed, otherwise use intended action
                record_action = performed if performed else action_dict
                self.buffer.record_frame(record_action)
                
        except Empty:
            pass
        except Exception as e:
            logger.error(f"Action error: {e}")
    
    def _send_observation_to_server(self):
        """Capture and send observation to active policy server."""
        try:
            raw_obs = self.robot.get_observation()
            raw_obs["task"] = self.active_policy.task
            
            obs = TimedObservation(
                timestamp=time.time(),
                observation=raw_obs,
                timestep=max(self.latest_action, 0),
            )
            
            with self.action_queue_lock:
                obs.must_go = self.must_go.is_set() and self.active_queue.empty()
            
            self._send_observation(obs)
            
            if obs.must_go:
                self.must_go.clear()
                
        except Exception as e:
            logger.error(f"Observation error: {e}")
    
    def _has_actions(self) -> bool:
        with self.action_queue_lock:
            return not self.active_queue.empty()
    
    def _ready_for_observation(self) -> bool:
        with self.action_queue_lock:
            if self.action_chunk_size <= 0:
                return True
            return self.active_queue.qsize() / self.action_chunk_size <= self.config.chunk_size_threshold
    
    def _action_to_dict(self, action_tensor: torch.Tensor) -> dict:
        return {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
    
    def _send_observation(self, obs: TimedObservation):
        try:
            obs_bytes = pickle.dumps(obs)
            obs_iter = send_bytes_in_chunks(obs_bytes, services_pb2.Observation, log_prefix="Obs", silent=True)
            self.active_stub.SendObservations(obs_iter)
        except grpc.RpcError as e:
            logger.error(f"Send observation error: {e}")
    
    def shutdown(self):
        """Clean shutdown."""
        self._running = False
        self.robot.disconnect()
        for channel in self.channels.values():
            channel.close()
        logger.info("Shutdown complete")


# ========== ROS2 Node ==========

if ROS2_AVAILABLE:
    class ROS2ControlNode(Node):
        def __init__(self, client: MultiPolicyClient):
            super().__init__("multi_policy_control")
            self.client = client
            
            # ROS2 Services for manual control
            self.create_service(Trigger, "/robot/start", self._start)
            self.create_service(Trigger, "/robot/pause", self._pause)
            self.create_service(Trigger, "/robot/resume", self._resume)
            self.create_service(Trigger, "/robot/reset", self._reset)
            self.create_service(Trigger, "/robot/stop", self._stop)
            self.create_service(Trigger, "/robot/switch_policy", self._switch_policy)
            
            # Subscribe to pause commands (from RND, Aria, or other systems)
            self.pause_sub = self.create_subscription(
                Bool, "/robot/pause_control", self._pause_control_callback, 10
            )
            
            # Subscribe to Aria gaze control
            # Left gaze = pause/stop, Right gaze = start/resume
            self.gaze_sub = self.create_subscription(
                String, "/aria/gaze_gesture/detected", self._gaze_callback, 10
            )
            
            # Subscribe to Aria eyes closed for reset
            self.eyes_closed_sub = self.create_subscription(
                Bool, "/aria/blink/eyes_closed_detected", self._eyes_closed_callback, 10
            )
            
            # Publisher to control robot's built-in pause system
            self.pause_control_pub = self.create_publisher(Bool, "/robot/pause_control", 10)
            
            # State publisher
            self.state_pub = self.create_publisher(String, "/robot/state", 10)
            self.create_timer(0.5, self._publish_state)
            
            self.get_logger().info("ROS2 services: /robot/{start,pause,resume,reset,stop}")
            self.get_logger().info("ROS2 subscriber: /robot/pause_control (multi-source pause)")
            self.get_logger().info("ROS2 subscriber: /aria/gaze_gesture/detected (left=pause, right=start)")
            self.get_logger().info("ROS2 subscriber: /aria/blink/eyes_closed_detected (reset)")
            self.get_logger().info("ROS2 publisher: /robot/pause_control (robot control)")
        
        def _pause_control_callback(self, msg):
            """Handle pause signals from any source (RND uncertainty, Aria glasses, manual, etc.)"""
            if msg.data:
                self.get_logger().warn("🔴 ROBOT PAUSED - Pause signal received")
            else:
                self.get_logger().info("✅ ROBOT RESUMED - Resume signal received")
        
        def _gaze_callback(self, msg):
            """Handle Aria gaze gesture detection.
            Left gaze = pause/stop, Right gaze = start/resume
            """
            gaze_direction = msg.data.lower()
            
            if gaze_direction == "left":
                # Left gaze = pause if running, stop if idle
                if self.client.state == RobotState.RUNNING:
                    self.get_logger().info("👁️ LEFT GAZE - Pausing robot")
                    success, message = self.client.handle_pause()
                    if success:
                        self._publish_pause_control(True)
                elif self.client.state == RobotState.IDLE:
                    self.get_logger().info("👁️ LEFT GAZE - Already idle, stopping")
                    success, message = self.client.handle_stop()
                    if success:
                        self._publish_pause_control(True)
                        
            elif gaze_direction == "right":
                # Right gaze = start if idle, resume if paused
                if self.client.state == RobotState.IDLE:
                    self.get_logger().info("👁️ RIGHT GAZE - Starting robot")
                    success, message = self.client.handle_start()
                    if success:
                        self._publish_pause_control(False)
                elif self.client.state == RobotState.PAUSED:
                    self.get_logger().info("👁️ RIGHT GAZE - Resuming robot")
                    success, message = self.client.handle_resume()
                    if success:
                        self._publish_pause_control(False)
        
        def _eyes_closed_callback(self, msg):
            """Handle Aria eyes closed detection for reset."""
            if msg.data:  # Eyes are closed
                if self.client.state in [RobotState.RUNNING, RobotState.PAUSED]:
                    self.get_logger().info("👁️ EYES CLOSED - Triggering reset")
                    success, message = self.client.handle_reset()
                    if success:
                        self._publish_pause_control(False)  # Unpause for reset movement
        
        def _publish_pause_control(self, paused: bool):
            """Publish pause command to robot's control system"""
            if hasattr(self, 'pause_control_pub'):
                pause_msg = Bool()
                pause_msg.data = paused
                self.pause_control_pub.publish(pause_msg)
                action = "PAUSE" if paused else "RESUME"
                self.get_logger().info(f"📢 Published {action} to /robot/pause_control")
        
        def _start(self, req, res):
            res.success, res.message = self.client.handle_start()
            if res.success:
                self._publish_pause_control(False)  # Resume robot
            return res
        
        def _pause(self, req, res):
            res.success, res.message = self.client.handle_pause()
            if res.success:
                self._publish_pause_control(True)  # Pause robot
            return res
        
        def _resume(self, req, res):
            res.success, res.message = self.client.handle_resume()
            if res.success:
                self._publish_pause_control(False)  # Resume robot
            return res
        
        def _reset(self, req, res):
            res.success, res.message = self.client.handle_reset()
            if res.success:
                self._publish_pause_control(False)  # Resume robot after reset
            return res
        
        def _stop(self, req, res):
            res.success, res.message = self.client.handle_stop()
            if res.success:
                self._publish_pause_control(True)  # Pause robot
            return res
        
        def _switch_policy(self, req, res):
            """Cycle to next policy server."""
            self.get_logger().info("🔀 Policy switch requested")
            res.success, res.message = self.client.handle_switch_policy()
            return res
        
        def _publish_state(self):
            msg = String()
            stats = self.client.buffer.get_stats()
            msg.data = f"{self.client.state.value} | policy: {self.client.active_policy.name} | buffer: {stats['buffer_size']} frames"
            self.state_pub.publish(msg)


# ========== Main ==========

@draccus.wrap()
def main(cfg: MultiPolicyConfig):
    """Main entry point."""
    # Setup logging with debug level
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    logger.info("=" * 60)
    logger.info("MULTI-POLICY TEST WITH ROS2 CONTROL")
    logger.info("=" * 60)
    
    client = MultiPolicyClient(cfg)
    
    # Start threads
    action_thread = threading.Thread(target=client.action_receiver_loop, daemon=True)
    control_thread = threading.Thread(target=client.control_loop, daemon=True)
    
    action_thread.start()
    control_thread.start()
    
    try:
        if ROS2_AVAILABLE:
            # Only init if not already initialized
            if not rclpy.ok():
                rclpy.init()
            
            node = ROS2ControlNode(client)
            
            logger.info("=" * 60)
            logger.info("ROS2 SERVICES READY:")
            logger.info("  ros2 service call /robot/start std_srvs/srv/Trigger")
            logger.info("  ros2 service call /robot/pause std_srvs/srv/Trigger")
            logger.info("  ros2 service call /robot/resume std_srvs/srv/Trigger")
            logger.info("  ros2 service call /robot/reset std_srvs/srv/Trigger")
            logger.info("  ros2 service call /robot/stop std_srvs/srv/Trigger")
            logger.info("  ros2 service call /robot/switch_policy std_srvs/srv/Trigger  # Cycles to next policy")
            logger.info("=" * 60)
            logger.info(f"Available policies: {[p.name for p in client.policy_servers]}")
            logger.info(f"Active policy: {client.active_policy.name}")
            logger.info("=" * 60)
            
            executor = MultiThreadedExecutor()
            executor.add_node(node)
            
            try:
                while client.running:
                    executor.spin_once(timeout_sec=0.1)
            finally:
                executor.shutdown()
                node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
        else:
            logger.warning("ROS2 not available - press Ctrl+C to stop")
            while client.running:
                time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Interrupted")
    
    finally:
        client.shutdown()
        action_thread.join(timeout=2)
        control_thread.join(timeout=2)


if __name__ == "__main__":
    main()
