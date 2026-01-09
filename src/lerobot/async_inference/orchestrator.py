# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi-policy orchestrator with finite state machine.

This module provides the main state machine and coordination for
multi-policy execution including:
- State transitions: IDLE -> POLICY_1 -> POLICY_2 -> COMPLETE
- SARM progress monitoring for policy completion
- RND uncertainty handling for pause/resume
- Data recording coordination

NOTE: Gaze-based object selection is trained INTO the ACT policies.
This orchestrator does NOT handle gaze selection separately.
The Aria camera with gaze visualization is an input to the trained policies.
"""

import logging
import threading
import time
from dataclasses import asdict
from enum import Enum
from pprint import pformat
from typing import Any

import draccus

from .configs_multi import MultiPolicyConfig, OrchestratorConfig
from .data_recorder import MultiPolicyDataRecorder
from .movement_buffer import MovementBuffer
from .multi_policy_client import RobotClientMulti

# Try to import ROS2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32, Int32, String, Bool
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object


logger = logging.getLogger("orchestrator")


class State(Enum):
    """Orchestrator states for multi-policy execution.
    
    Simplified state machine (no separate gaze waiting states):
    - IDLE: System initialized, waiting to start
    - POLICY_1: Executing first policy (pick pod - gaze selection trained in)
    - POLICY_2: Executing second policy (make coffee - gaze selection trained in)
    - PAUSED: Paused due to RND uncertainty
    - RESETTING: Replaying buffer in reverse to return to start position
    - COMPLETE: Episode complete
    """
    
    IDLE = "idle"
    POLICY_1 = "policy_1"      # Pick pod policy (gaze is in the policy)
    POLICY_2 = "policy_2"      # Make coffee policy (gaze is in the policy)
    PAUSED = "paused"          # Paused due to uncertainty
    RESETTING = "resetting"    # Replaying buffer in reverse
    COMPLETE = "complete"      # Episode complete


class MultiPolicyOrchestrator:
    """
    Main orchestrator for multi-policy execution with bimanual robot.
    
    Coordinates:
    - RobotClientMulti for policy execution on separate servers
    - MultiPolicyDataRecorder for recording
    - SARM progress monitoring (via ROS2 topics)
    - RND uncertainty handling for pause/resume
    
    NOTE: Gaze-based object selection is NOT handled here.
    Gaze is trained into the ACT policies via Aria camera input.
    """
    
    def __init__(
        self,
        config: MultiPolicyConfig,
        ros_node: "Node | None" = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: Multi-policy configuration
            ros_node: Optional ROS2 node for topic subscriptions
        """
        self.config = config
        self.ros_node = ros_node
        self.orch_config = config.orchestrator
        
        # State machine
        self.state = State.IDLE
        self.state_lock = threading.Lock()
        self.previous_state = State.IDLE  # For pause/resume
        
        # Timing
        self.state_entry_time = time.time()
        self.episode_start_time = 0.0
        
        # Episode tracking
        self.current_episode = 0
        
        # SARM progress
        self.sarm_progress = 0.0
        self.sarm_stage = 0
        self.sarm_stage_name = ""
        
        # RND uncertainty
        self.rnd_uncertainty = 0.0
        self.rnd_paused = False
        
        # Movement buffer for reset/rewind
        self.movement_buffer = MovementBuffer(max_frames=3000, validate_positions=False)
        self.reset_triggered = threading.Event()
        
        # Blink-based start/stop control
        self.blink_toggle_enabled = True  # Enable double-blink start/stop
        
        # Initialize components
        self._init_components()
        
        # ROS2 subscriptions
        if ros_node is not None and ROS2_AVAILABLE:
            self._setup_ros2(ros_node)
        
        # Control thread
        self.running = False
        self.control_thread: threading.Thread | None = None
        
        logger.info("MultiPolicyOrchestrator initialized")
    
    def _init_components(self):
        """Initialize sub-components."""
        # Robot client (multi-server)
        self.robot_client = RobotClientMulti(self.config)
        
        # Data recorder
        robot_features = {
            **self.robot_client.robot.observation_features,
            **self.robot_client.robot.action_features,
        }
        self.data_recorder = MultiPolicyDataRecorder(
            config=self.config.dataset,
            robot_features=robot_features,
            robot_type=self.config.robot.type,
        )
        
        logger.info("Components initialized")
    
    def _setup_ros2(self, node: "Node"):
        """Set up ROS2 subscriptions for SARM and RND."""
        # SARM progress
        self.sarm_progress_sub = node.create_subscription(
            Float32,
            self.config.sarm.progress_topic,
            self._sarm_progress_callback,
            10
        )
        self.sarm_stage_sub = node.create_subscription(
            Int32,
            self.config.sarm.stage_topic,
            self._sarm_stage_callback,
            10
        )
        self.sarm_stage_name_sub = node.create_subscription(
            String,
            self.config.sarm.stage_name_topic,
            self._sarm_stage_name_callback,
            10
        )
        
        # RND uncertainty (pause signal)
        self.rnd_pause_sub = node.create_subscription(
            Bool,
            self.config.rnd.pause_topic,
            self._rnd_pause_callback,
            10
        )
        self.rnd_uncertainty_sub = node.create_subscription(
            Float32,
            self.config.rnd.uncertainty_topic,
            self._rnd_uncertainty_callback,
            10
        )
        
        # Blink detection for start/stop toggle
        self.blink_double_sub = node.create_subscription(
            Bool,
            '/aria/blink/double_detected',
            self._double_blink_callback,
            10
        )
        
        # Gaze gesture for reset/rewind
        self.gaze_reset_sub = node.create_subscription(
            Bool,
            '/aria/gaze_gesture/reset_triggered',
            self._reset_gesture_callback,
            10
        )
        
        logger.info("ROS2 subscriptions set up (including blink and gaze gesture)")
    
    def _sarm_progress_callback(self, msg: "Float32"):
        """Callback for SARM progress updates."""
        self.sarm_progress = msg.data
        self._update_recorder_metadata()
        
        # Check if we should switch policies
        self._check_and_handle_policy_switch()
    
    def _sarm_stage_callback(self, msg: "Int32"):
        """Callback for SARM stage updates."""
        self.sarm_stage = msg.data
        self._update_recorder_metadata()
    
    def _sarm_stage_name_callback(self, msg: "String"):
        """Callback for SARM stage name updates."""
        self.sarm_stage_name = msg.data
        self._update_recorder_metadata()
    
    def _rnd_pause_callback(self, msg: "Bool"):
        """Callback for RND pause signal."""
        should_pause = msg.data
        
        if should_pause and not self.rnd_paused:
            self._handle_pause()
        elif not should_pause and self.rnd_paused:
            self._handle_resume()
        
        self.rnd_paused = should_pause
    
    def _double_blink_callback(self, msg: "Bool"):
        """Callback for double blink detection - toggle start/stop."""
        if not msg.data or not self.blink_toggle_enabled:
            return
        
        with self.state_lock:
            current_state = self.state
        
        logger.info(f"Double blink detected in state: {current_state.value}")
        
        if current_state == State.IDLE:
            # Start execution - transition to first policy
            self.start_episode()
        elif current_state in [State.POLICY_1, State.POLICY_2]:
            # Stop execution - pause the robot
            self._handle_pause()
        elif current_state == State.PAUSED:
            # Resume execution
            self._handle_resume()
    
    def _reset_gesture_callback(self, msg: "Bool"):
        """Callback for gaze gesture reset - trigger buffer reverse playback."""
        if not msg.data:
            return
        
        with self.state_lock:
            current_state = self.state
            # Only allow reset from active or paused states
            if current_state not in [State.POLICY_1, State.POLICY_2, State.PAUSED]:
                logger.warning(f"Reset gesture ignored in state: {current_state.value}")
                return
        
        logger.info("Reset gesture detected - triggering buffer rewind")
        self.reset_triggered.set()
    
    def _rnd_uncertainty_callback(self, msg: "Float32"):
        """Callback for RND uncertainty updates."""
        self.rnd_uncertainty = msg.data
        self._update_recorder_metadata()
    
    def _update_recorder_metadata(self):
        """Update data recorder with current metadata."""
        active_policy = ""
        with self.state_lock:
            if self.state == State.POLICY_1:
                active_policy = "pick_pod"
            elif self.state == State.POLICY_2:
                active_policy = "make_coffee"
        
        self.data_recorder.update_metadata(
            active_policy=active_policy,
            fsm_state=self.state.value,
            sarm_progress=self.sarm_progress,
            sarm_stage=self.sarm_stage,
            sarm_stage_name=self.sarm_stage_name,
            rnd_uncertainty=self.rnd_uncertainty,
            rnd_paused=self.rnd_paused,
        )
    
    def _check_and_handle_policy_switch(self):
        """Check if SARM progress indicates policy switch needed."""
        with self.state_lock:
            if self.state == State.POLICY_1:
                if self.sarm_progress >= self.config.sarm.policy_switch_threshold:
                    logger.info(f"SARM progress {self.sarm_progress:.3f} >= threshold, switching to POLICY_2")
                    self._do_transition(State.POLICY_2)
            elif self.state == State.POLICY_2:
                if self.sarm_progress >= self.config.sarm.policy_switch_threshold:
                    logger.info(f"SARM progress {self.sarm_progress:.3f} >= threshold, episode complete")
                    self._do_transition(State.COMPLETE)
    
    def _do_transition(self, new_state: State):
        """Internal transition (already holding lock)."""
        old_state = self.state
        self.state = new_state
        self.state_entry_time = time.time()
        logger.info(f"Transition: {old_state.value} -> {new_state.value}")
        self._on_state_enter(new_state, old_state)
    
    def transition_to(self, new_state: State):
        """
        Transition to a new state (public, acquires lock).
        
        Args:
            new_state: The state to transition to
        """
        with self.state_lock:
            self._do_transition(new_state)
    
    def _on_state_enter(self, new_state: State, old_state: State):
        """Handle actions on entering a new state."""
        
        if new_state == State.IDLE:
            # Ensure robot is paused and buffer recording is stopped when entering IDLE
            self.robot_client.pause()
            if self.movement_buffer.is_recording:
                self.movement_buffer.stop_recording()
            logger.info("Entered IDLE state")
        
        elif new_state == State.POLICY_1:
            # Switch to pick-pod server
            servers = self.config.get_policy_servers()
            if servers:
                self.robot_client.switch_server(servers[0].name)
                logger.info(f"Switched to server: {servers[0].name}")
            self.robot_client.resume()
            
            # Reset SARM progress for new policy
            self.sarm_progress = 0.0
            
            # Start recording joint positions for potential reset
            self.movement_buffer.start_recording()
            logger.info("Movement buffer recording started")
            
        elif new_state == State.POLICY_2:
            # Switch to make-coffee server
            servers = self.config.get_policy_servers()
            if len(servers) > 1:
                self.robot_client.switch_server(servers[1].name)
                logger.info(f"Switched to server: {servers[1].name}")
            self.robot_client.resume()
            
            # Reset SARM progress for new policy
            self.sarm_progress = 0.0
            
        elif new_state == State.PAUSED:
            self.previous_state = old_state
            self.robot_client.pause()
            logger.info("Robot paused due to RND uncertainty")
            
        elif new_state == State.RESETTING:
            # Stop policy execution and begin reset
            self.robot_client.pause()
            self.movement_buffer.stop_recording()
            logger.info("Entering RESETTING state - buffer recording stopped")
            
        elif new_state == State.COMPLETE:
            self.robot_client.pause()
            self.movement_buffer.stop_recording()
            logger.info("Episode complete - buffer recording stopped")
    
    def _handle_pause(self):
        """Handle pause request from RND or blink toggle."""
        with self.state_lock:
            if self.state not in [State.PAUSED, State.IDLE, State.COMPLETE, State.RESETTING]:
                self.previous_state = self.state
                self.state = State.PAUSED
                self.robot_client.pause()
                logger.info(f"Paused from {self.previous_state.value} due to high uncertainty")

    
    def _handle_resume(self):
        """Handle resume after uncertainty clears."""
        with self.state_lock:
            if self.state == State.PAUSED:
                # Ensure we have a valid previous state to return to
                if self.previous_state in [State.POLICY_1, State.POLICY_2]:
                    self.state = self.previous_state
                    self.robot_client.resume()
                    logger.info(f"Resumed to {self.state.value} - uncertainty cleared")
                else:
                    # Defensive: if previous_state is invalid, go to IDLE
                    logger.warning(f"Cannot resume to {self.previous_state.value}, transitioning to IDLE")
                    self.state = State.IDLE
    
    def _check_timeout(self) -> bool:
        """Check if episode has timed out."""
        elapsed = time.time() - self.episode_start_time
        return elapsed >= self.orch_config.max_episode_time_s
    
    def start_episode(self):
        """Start a new episode."""
        self.current_episode += 1
        self.episode_start_time = time.time()
        self.sarm_progress = 0.0
        self.sarm_stage = 0
        
        # Start data recording
        self.data_recorder.start_episode()
        
        # Transition to first policy
        self.transition_to(State.POLICY_1)
        
        logger.info(f"Episode {self.current_episode} started")
    
    def end_episode(self, success: bool = True):
        """End the current episode."""
        # Save data
        task_description = f"{self.orch_config.policy_1_task} then {self.orch_config.policy_2_task}"
        self.data_recorder.end_episode(
            task=task_description,
            success=success,
        )
        
        logger.info(f"Episode {self.current_episode} ended, success={success}")
    
    def _record_current_position(self):
        """Record current joint position to movement buffer."""
        try:
            observation = self.robot_client.get_current_observation()
            # Extract joint positions from observation
            # The observation contains state features like position.left_*, position.right_*
            positions = {}
            for key, value in observation.items():
                if key.startswith("observation.state"):
                    # Extract just the joint name from observation.state.left_shoulder_pan, etc.
                    joint_name = key.replace("observation.state.", "")
                    positions[joint_name] = float(value) if not hasattr(value, 'item') else value.item()
            
            if positions:
                self.movement_buffer.record_frame(positions)
        except Exception as e:
            logger.debug(f"Error recording position: {e}")
    
    def _execute_reset_trajectory(self):
        """Execute the reverse trajectory to return robot to start position."""
        logger.info("Starting reset trajectory execution")
        
        # Get reverse trajectory
        trajectory = self.movement_buffer.get_reverse_trajectory(
            playback_speed=0.5,  # Slower for safety
            smooth_window=5,
        )
        
        if not trajectory:
            logger.warning("No trajectory to execute - buffer was empty")
            return
        
        logger.info(f"Executing {len(trajectory)} frames for reset")
        
        # Execute each position in the trajectory
        for i, positions in enumerate(trajectory):
            # Check if we should abort
            if not self.running:
                logger.info("Reset aborted - orchestrator stopping")
                break
            
            with self.state_lock:
                if self.state != State.RESETTING:
                    logger.info("Reset aborted - state changed")
                    break
            
            try:
                # Send positions directly to robot
                self.robot_client.robot.send_action(positions)
            except Exception as e:
                logger.error(f"Error sending reset position: {e}")
                break
            
            # Control frequency for playback
            time.sleep(self.config.environment_dt)
            
            # Periodic logging
            if (i + 1) % 100 == 0:
                logger.debug(f"Reset progress: {i + 1}/{len(trajectory)} frames")
        
        logger.info("Reset trajectory execution complete")
        
        # Clear buffer after reset
        self.movement_buffer.clear()
    
    def run_episode(self):
        """Run a single episode."""
        self.start_episode()
        
        # Clear any pending reset trigger
        self.reset_triggered.clear()
        
        try:
            while self.running:
                with self.state_lock:
                    current_state = self.state
                
                if current_state == State.COMPLETE:
                    break
                
                # Check for reset trigger (from gaze gesture callback)
                if self.reset_triggered.is_set():
                    self.reset_triggered.clear()
                    logger.info("Reset triggered - transitioning to RESETTING")
                    self.transition_to(State.RESETTING)
                    self._execute_reset_trajectory()
                    self.transition_to(State.IDLE)
                    break  # End this episode
                
                # Check for timeout
                if self._check_timeout():
                    logger.warning("Episode timeout")
                    break
                
                # Control loop step
                if current_state in [State.POLICY_1, State.POLICY_2]:
                    self.robot_client.step()
                    # Record current position for potential reset
                    self._record_current_position()
                
                # Small sleep to prevent busy loop
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            logger.info("Episode interrupted")
        
        # Determine success based on final state
        success = (self.state == State.COMPLETE)
        
        # Stop recording if still active
        if self.movement_buffer.get_stats()["is_recording"]:
            self.movement_buffer.stop_recording()
        
        self.end_episode(success=success)
        self.transition_to(State.IDLE)
    
    def run(self):
        """Run the orchestrator for all episodes."""
        self.running = True
        
        logger.info(f"Starting orchestrator for {self.orch_config.num_episodes} episodes")
        
        try:
            # Connect to servers
            self.robot_client.connect()
            
            for episode in range(self.orch_config.num_episodes):
                logger.info(f"=== Episode {episode + 1}/{self.orch_config.num_episodes} ===")
                self.run_episode()
                
                # Delay between episodes
                if episode < self.orch_config.num_episodes - 1:
                    time.sleep(self.orch_config.transition_delay_s)
            
            # Push to hub if configured
            if self.config.dataset.push_to_hub:
                logger.info("Pushing dataset to HuggingFace Hub...")
                self.data_recorder.push_to_hub()
                logger.info("Dataset pushed to Hub")
                
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            raise
        finally:
            self.running = False
            self.robot_client.disconnect()
            logger.info("Orchestrator stopped")
    
    def stop(self):
        """Stop the orchestrator."""
        self.running = False
        self.robot_client.pause()


class OrchestratorNode(Node):
    """ROS2 node wrapper for the orchestrator."""
    
    def __init__(self, config: MultiPolicyConfig):
        super().__init__('multi_policy_orchestrator')
        self.orchestrator = MultiPolicyOrchestrator(config, ros_node=self)
        self.get_logger().info("Orchestrator node initialized")
    
    def run(self):
        """Run the orchestrator."""
        # Create a thread for ROS2 spinning
        spin_thread = threading.Thread(target=self._spin_ros2, daemon=True)
        spin_thread.start()
        
        # Run the orchestrator
        self.orchestrator.run()
    
    def _spin_ros2(self):
        """Spin ROS2 in background thread."""
        rclpy.spin(self)


@draccus.wrap()
def run_orchestrator(cfg: MultiPolicyConfig):
    """Entry point for the multi-policy orchestrator."""
    logging.basicConfig(level=logging.INFO)
    logging.info(pformat(asdict(cfg)))
    
    if ROS2_AVAILABLE:
        rclpy.init()
        try:
            node = OrchestratorNode(cfg)
            node.run()
        finally:
            rclpy.shutdown()
    else:
        # Run without ROS2
        logger.warning("Running without ROS2 - SARM and RND monitoring disabled")
        orchestrator = MultiPolicyOrchestrator(cfg)
        orchestrator.run()


if __name__ == "__main__":
    run_orchestrator()
