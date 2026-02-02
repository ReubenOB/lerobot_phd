#!/usr/bin/env python3
"""
Policy Client - Simple robot + policy server interface.

This module provides a clean interface for:
- Robot connection and control
- Policy server communication (gRPC)
- Action execution and observation streaming
- Movement buffer for trajectory recording

NOTE: This is intentionally simple - NO state machine.
State management belongs in the orchestrator.

Run:
    # Use via orchestrator, not directly
    python -m lerobot.async_inference.orchestrator --config_path=...
"""

import logging
import pickle
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional, Callable

import grpc
import torch

from lerobot.robots import make_robot_from_config

from .movement_buffer import MovementBuffer
from .helpers import RemotePolicyConfig, TimedAction, TimedObservation, map_robot_keys_to_lerobot_features

try:
    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

logger = logging.getLogger("policy_client")


@dataclass
class PolicyServerSpec:
    """Specification for a policy server."""
    name: str
    address: str
    pretrained_path: str
    policy_type: str = "act"
    actions_per_chunk: int = 15
    device: str = "cuda"
    task: str = "pick up the object"


class PolicyClient:
    """
    Simple robot + policy server client.
    
    Responsibilities:
    - Robot connection and low-level control
    - Policy server communication (gRPC)
    - Action execution from queue
    - Observation streaming to server
    - Movement buffer recording
    
    NOT responsible for:
    - State machine (IDLE/RUNNING/PAUSED) - that's the orchestrator
    - SARM/RND monitoring - that's the orchestrator
    - Aria gesture handling - that's the orchestrator
    """
    
    def __init__(
        self,
        robot_config,
        policy_servers: list[PolicyServerSpec],
        buffer_max_frames: int = 3000,
        fps: float = 30.0,
        chunk_size_threshold: float = 0.5,
        environment_dt: float = 0.033,
    ):
        """
        Initialize the policy client.
        
        Args:
            robot_config: Robot configuration
            policy_servers: List of policy server specs
            buffer_max_frames: Max frames for movement buffer
            fps: Control loop frequency
            chunk_size_threshold: Threshold for requesting new observations
            environment_dt: Control loop timestep
        """
        self.fps = fps
        self.chunk_size_threshold = chunk_size_threshold
        self.environment_dt = environment_dt
        
        # Robot
        logger.info("Connecting robot...")
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()
        logger.info("Robot connected")
        
        # Policy servers
        self.policy_servers = policy_servers
        if not self.policy_servers:
            raise ValueError("No policy servers configured")
        
        self.active_policy_idx = 0
        logger.info(f"Configured {len(self.policy_servers)} policy servers")
        
        # gRPC connections
        self.channels: dict[str, grpc.Channel] = {}
        self.stubs: dict[str, services_pb2_grpc.AsyncInferenceStub] = {}
        
        # Movement buffer
        self.buffer = MovementBuffer(max_frames=buffer_max_frames, validate_positions=False)
        logger.info(f"Movement buffer: {buffer_max_frames} frames")
        
        # Action queues (one per policy)
        self.action_queues: dict[str, Queue] = {spec.name: Queue() for spec in self.policy_servers}
        self.action_queue_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1
        
        # Threading
        self.start_barrier = threading.Barrier(2)
        self.must_go = threading.Event()
        self.must_go.set()
        
        # Control flags
        self._running = True
        self._paused = True  # Start paused, orchestrator will unpause
        self._pause_lock = threading.Lock()
        
        logger.info("PolicyClient initialized")
    
    # ========== Properties ==========
    
    @property
    def running(self) -> bool:
        return self._running
    
    @property
    def is_paused(self) -> bool:
        """Check if client is paused (local or via ros2_bridge)."""
        with self._pause_lock:
            local_paused = self._paused
        
        # Also check ros2_bridge pause state
        robot_paused = (
            hasattr(self.robot, 'ros2_bridge') and 
            self.robot.ros2_bridge and 
            self.robot.ros2_bridge.is_paused()
        )
        
        return local_paused or robot_paused
    
    @property
    def active_policy(self) -> PolicyServerSpec:
        """Get the currently active policy server spec."""
        return self.policy_servers[self.active_policy_idx]
    
    @property
    def active_stub(self):
        """Get the stub for the currently active policy."""
        return self.stubs.get(self.active_policy.name)
    
    @property
    def active_queue(self) -> Queue:
        """Get the action queue for the currently active policy."""
        return self.action_queues[self.active_policy.name]
    
    # ========== Connection ==========
    
    def connect(self):
        """Connect to all policy servers."""
        if not GRPC_AVAILABLE:
            raise RuntimeError("gRPC not available")
        
        for spec in self.policy_servers:
            logger.info(f"Connecting to {spec.name} at {spec.address}...")
            
            channel = grpc.insecure_channel(
                spec.address,
                grpc_channel_options(initial_backoff=f"{self.environment_dt:.4f}s")
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
    
    def disconnect(self):
        """Disconnect from robot and servers."""
        self._running = False
        self.robot.disconnect()
        for channel in self.channels.values():
            channel.close()
        logger.info("Disconnected")
    
    # ========== Control Methods (called by orchestrator) ==========
    
    def pause(self):
        """Pause action execution."""
        with self._pause_lock:
            self._paused = True
        
        # Also pause via ros2_bridge if available
        if hasattr(self.robot, 'ros2_bridge') and self.robot.ros2_bridge:
            self.robot.ros2_bridge.set_pause(True)
        
        logger.info("Paused")
    
    def resume(self):
        """Resume action execution."""
        with self._pause_lock:
            self._paused = False
        
        # Also unpause via ros2_bridge if available
        if hasattr(self.robot, 'ros2_bridge') and self.robot.ros2_bridge:
            self.robot.ros2_bridge.set_pause(False)
        
        logger.info("Resumed")
    
    def stop(self):
        """Stop the client."""
        self._running = False
    
    def start_buffer_recording(self):
        """Start recording to movement buffer."""
        if not self.buffer.is_recording:
            self.buffer.start_recording()
            logger.info("Buffer recording started")
    
    def stop_buffer_recording(self):
        """Stop recording to movement buffer."""
        if self.buffer.is_recording:
            self.buffer.stop_recording()
            logger.info("Buffer recording stopped")
    
    def clear_buffer(self):
        """Clear the movement buffer."""
        self.buffer.clear()
        logger.info("Buffer cleared")
    
    def get_buffer_stats(self) -> dict:
        """Get movement buffer statistics."""
        return self.buffer.get_stats()
    
    # ========== Policy Server Control ==========
    
    def send_policy_setup(self, spec: Optional[PolicyServerSpec] = None):
        """Send policy configuration to a server (defaults to active)."""
        spec = spec or self.active_policy
        
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
    
    def switch_policy(self, policy_idx: Optional[int] = None) -> str:
        """
        Switch to a different policy server.
        
        Args:
            policy_idx: Index of policy to switch to. If None, cycles to next.
            
        Returns:
            Name of the new active policy
        """
        # Clear current action queue
        with self.action_queue_lock:
            self.active_queue.queue.clear()
        
        # Switch to specified or next policy
        if policy_idx is not None:
            self.active_policy_idx = policy_idx % len(self.policy_servers)
        else:
            self.active_policy_idx = (self.active_policy_idx + 1) % len(self.policy_servers)
        
        new_policy = self.active_policy.name
        logger.info(f"Switched to policy: {new_policy}")
        
        # Send policy setup to new server
        self.send_policy_setup(self.active_policy)
        
        return new_policy
    
    def clear_action_queue(self):
        """Clear action queue and signal for fresh planning."""
        with self.action_queue_lock:
            while not self.active_queue.empty():
                try:
                    self.active_queue.get_nowait()
                except:
                    break
        
        self.must_go.set()
        logger.info("Action queue cleared")
    
    # ========== Reset Execution ==========
    
    def execute_reset_trajectory(self, playback_speed: float = 0.5) -> bool:
        """
        Execute reverse trajectory from buffer.
        
        Args:
            playback_speed: Speed multiplier for playback (0.5 = half speed)
            
        Returns:
            True if reset completed successfully
        """
        trajectory = self.buffer.get_reverse_trajectory(
            playback_speed=playback_speed,
            smooth_window=1,
        )
        
        if not trajectory:
            logger.warning("Empty buffer - cannot reset")
            return False
        
        logger.info(f"Executing reset: {len(trajectory)} frames at {playback_speed}x speed")
        dt = 1.0 / self.fps
        
        for i, positions in enumerate(trajectory):
            if not self._running:
                logger.info("Reset aborted - client stopping")
                return False
            
            try:
                self.robot.send_action(positions)
            except Exception as e:
                logger.error(f"Reset error: {e}")
                return False
            
            time.sleep(dt)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Reset progress: {i+1}/{len(trajectory)}")
        
        # Clear action queue after reset
        self.clear_action_queue()
        
        logger.info("Reset complete")
        return True
    
    # ========== Main Loops ==========
    
    def action_receiver_loop(self):
        """Thread: receive actions from active policy server."""
        self.start_barrier.wait()
        logger.info("Action receiver started")
        
        empty_count = 0
        while self._running:
            try:
                if not self.active_stub:
                    time.sleep(0.1)
                    continue
                
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
        """Main control loop."""
        self.start_barrier.wait()
        logger.info("Control loop started")
        
        while self._running:
            loop_start = time.perf_counter()
            
            # Execute actions only when not paused
            if self._has_actions() and not self.is_paused:
                self._execute_action()
            
            # Always send observations (policy server needs them)
            if self._ready_for_observation():
                self._send_observation_to_server()
            
            # Maintain loop rate
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, self.environment_dt - elapsed))
        
        logger.info("Control loop stopped")
    
    def _execute_action(self):
        """Execute an action from the queue and record to buffer."""
        try:
            with self.action_queue_lock:
                timed_action = self.active_queue.get_nowait()
            
            action_dict = self._action_to_dict(timed_action.get_action())
            logger.debug(f"Executing action at timestep {timed_action.get_timestep()}")
            
            # Send to robot
            performed = self.robot.send_action(action_dict)
            self.latest_action = timed_action.get_timestep()
            
            # Record to buffer
            if self.buffer.is_recording:
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
            return self.active_queue.qsize() / self.action_chunk_size <= self.chunk_size_threshold
    
    def _action_to_dict(self, action_tensor: torch.Tensor) -> dict:
        return {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
    
    def _send_observation(self, obs: TimedObservation):
        try:
            obs_bytes = pickle.dumps(obs)
            obs_iter = send_bytes_in_chunks(obs_bytes, services_pb2.Observation, log_prefix="Obs", silent=True)
            self.active_stub.SendObservations(obs_iter)
        except grpc.RpcError as e:
            logger.error(f"Send observation error: {e}")
    
    # ========== Utility ==========
    
    def get_current_observation(self) -> dict:
        """Get the current robot observation."""
        return self.robot.get_observation()
