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
Extended RobotClient with multi-server support.

This module provides RobotClientMulti, which extends the existing RobotClient
patterns to support:
- Maintaining gRPC connections to multiple PolicyServers
- Clean server switching with pause/resume
- Integration with DataRecorder for multi-policy recording
"""

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from queue import Queue
from typing import Any

import grpc
import torch

from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .configs import AGGREGATE_FUNCTIONS, get_aggregate_function
from .configs_multi import MultiPolicyConfig, PolicyServerSpec
from .helpers import (
    Action,
    FPSTracker,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
)


class RobotClientMulti:
    """
    Multi-server robot client for policy orchestration.
    
    Maintains gRPC connections to multiple PolicyServers and provides
    methods for switching between them during execution.
    """
    
    prefix = "robot_client_multi"
    logger = get_logger(prefix)
    
    def __init__(self, config: MultiPolicyConfig):
        """
        Initialize RobotClientMulti with multi-policy configuration.
        
        Args:
            config: MultiPolicyConfig containing robot and policy server specs
        """
        self.config = config
        
        # Initialize robot
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()
        self.logger.info("Robot connected")
        
        # Get LeRobot features mapping
        self.lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        
        # Initialize connections to all policy servers
        self.policy_servers = config.get_policy_servers()
        self._init_connections()
        
        # Aggregate function
        self.aggregate_fn = get_aggregate_function(config.aggregate_fn_name)
        
        # State management
        self.shutdown_event = threading.Event()
        self.paused = threading.Event()  # Set when paused
        
        # Action queue management (per-server)
        self.action_queues: dict[str, Queue] = {
            spec.name: Queue() for spec in self.policy_servers
        }
        self.action_queue_locks: dict[str, threading.Lock] = {
            spec.name: threading.Lock() for spec in self.policy_servers
        }
        
        # Current active server
        self.active_server: str | None = None
        self.active_server_lock = threading.Lock()
        
        # Latest action tracking
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1
        
        # FPS tracking
        self.fps_tracker = FPSTracker(target_fps=config.fps)
        
        # Chunk size threshold for observation sending
        self._chunk_size_threshold = config.chunk_size_threshold
        
        # Thread synchronization
        self.start_barrier = threading.Barrier(2)
        self.must_go = threading.Event()
        self.must_go.set()
        
        self.logger.info(f"Initialized with {len(self.policy_servers)} policy servers")
    
    def _init_connections(self):
        """Initialize gRPC connections to all policy servers."""
        self.channels: dict[str, grpc.Channel] = {}
        self.stubs: dict[str, services_pb2_grpc.AsyncInferenceStub] = {}
        self.policy_configs: dict[str, RemotePolicyConfig] = {}
        
        for spec in self.policy_servers:
            # Create channel
            channel = grpc.insecure_channel(
                spec.address,
                grpc_channel_options(initial_backoff=f"{self.config.environment_dt:.4f}s")
            )
            self.channels[spec.name] = channel
            self.stubs[spec.name] = services_pb2_grpc.AsyncInferenceStub(channel)
            
            # Create policy config for this server
            self.policy_configs[spec.name] = RemotePolicyConfig(
                policy_type=spec.policy_type,
                pretrained_name_or_path=spec.pretrained_path,
                lerobot_features=self.lerobot_features,
                actions_per_chunk=spec.actions_per_chunk,
                device=spec.device,
            )
            
            self.logger.info(f"Created connection to {spec.name} at {spec.address}")
    
    @property
    def running(self) -> bool:
        """Check if client is running."""
        return not self.shutdown_event.is_set()
    
    @property
    def is_paused(self) -> bool:
        """Check if client is paused."""
        return self.paused.is_set()
    
    def start(self) -> bool:
        """Start the robot client and verify connections to all servers."""
        try:
            for name, stub in self.stubs.items():
                # Verify connection
                start_time = time.perf_counter()
                stub.Ready(services_pb2.Empty())
                end_time = time.perf_counter()
                self.logger.info(f"Connected to {name} in {end_time - start_time:.4f}s")
                
                # Send policy instructions
                policy_config = self.policy_configs[name]
                policy_config_bytes = pickle.dumps(policy_config)
                policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)
                
                self.logger.info(
                    f"Sending policy instructions to {name}: "
                    f"{policy_config.policy_type} @ {policy_config.pretrained_name_or_path}"
                )
                stub.SendPolicyInstructions(policy_setup)
            
            self.shutdown_event.clear()
            return True
            
        except grpc.RpcError as e:
            self.logger.error(f"Failed to start: {e}")
            return False
    
    def stop(self):
        """Stop the robot client and close all connections."""
        self.shutdown_event.set()
        
        # Close all channels
        for name, channel in self.channels.items():
            channel.close()
            self.logger.debug(f"Closed channel to {name}")
        
        # Disconnect robot
        self.robot.disconnect()
        self.logger.info("Robot disconnected, client stopped")
    
    def switch_server(self, server_name: str) -> bool:
        """
        Switch to a different policy server.
        
        Args:
            server_name: Name of the server to switch to
            
        Returns:
            True if switch was successful
        """
        if server_name not in self.stubs:
            self.logger.error(f"Unknown server: {server_name}")
            return False
        
        with self.active_server_lock:
            old_server = self.active_server
            self.active_server = server_name
        
        # Clear action queue for new server
        with self.action_queue_locks[server_name]:
            self.action_queues[server_name] = Queue()
        
        # Reset latest action tracking
        with self.latest_action_lock:
            self.latest_action = -1
        
        self.logger.info(f"Switched server: {old_server} -> {server_name}")
        return True
    
    def pause(self):
        """Pause action execution (robot holds position)."""
        self.paused.set()
        self.logger.info("Robot paused")
    
    def resume(self):
        """Resume action execution."""
        self.paused.clear()
        self.logger.info("Robot resumed")
    
    def get_active_stub(self) -> services_pb2_grpc.AsyncInferenceStub | None:
        """Get the stub for the currently active server."""
        with self.active_server_lock:
            if self.active_server is None:
                return None
            return self.stubs.get(self.active_server)
    
    def get_active_action_queue(self) -> tuple[Queue, threading.Lock] | None:
        """Get the action queue for the currently active server."""
        with self.active_server_lock:
            if self.active_server is None:
                return None
            return (
                self.action_queues[self.active_server],
                self.action_queue_locks[self.active_server]
            )
    
    def send_observation(self, obs: TimedObservation) -> bool:
        """
        Send observation to the currently active policy server.
        
        Args:
            obs: Timed observation to send
            
        Returns:
            True if sent successfully
        """
        if not self.running:
            raise RuntimeError("Client not running")
        
        stub = self.get_active_stub()
        if stub is None:
            self.logger.warning("No active server, cannot send observation")
            return False
        
        try:
            observation_bytes = pickle.dumps(obs)
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT_MULTI] Observation",
                silent=True,
            )
            _ = stub.SendObservations(observation_iterator)
            self.logger.debug(f"Sent observation #{obs.get_timestep()}")
            return True
            
        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation: {e}")
            return False
    
    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        queue: Queue,
        lock: threading.Lock,
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Aggregate incoming actions with existing queue."""
        if aggregate_fn is None:
            aggregate_fn = lambda x1, x2: x2
        
        future_action_queue = Queue()
        
        with lock:
            current_action_queue = {
                action.get_timestep(): action.get_action() 
                for action in queue.queue
            }
        
        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action
            
            if new_action.get_timestep() <= latest_action:
                continue
            
            if new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
            else:
                future_action_queue.put(
                    TimedAction(
                        timestamp=new_action.get_timestamp(),
                        timestep=new_action.get_timestep(),
                        action=aggregate_fn(
                            current_action_queue[new_action.get_timestep()],
                            new_action.get_action()
                        ),
                    )
                )
        
        with lock:
            # Replace queue contents
            with self.active_server_lock:
                if self.active_server:
                    self.action_queues[self.active_server] = future_action_queue
    
    def receive_actions(self, verbose: bool = False):
        """Receive actions from the active policy server."""
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")
        
        while self.running:
            stub = self.get_active_stub()
            if stub is None:
                time.sleep(0.01)
                continue
            
            try:
                actions_chunk = stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue
                
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                
                # Get current queue
                queue_info = self.get_active_action_queue()
                if queue_info is not None:
                    queue, lock = queue_info
                    self._aggregate_action_queues(
                        timed_actions, queue, lock, self.aggregate_fn
                    )
                
                self.must_go.set()
                
                if verbose and timed_actions:
                    self.logger.info(
                        f"Received {len(timed_actions)} actions for step "
                        f"#{timed_actions[0].get_timestep()}"
                    )
                
            except grpc.RpcError as e:
                self.logger.debug(f"RPC error (may be expected during switch): {e}")
                time.sleep(0.01)
    
    def actions_available(self) -> bool:
        """Check if actions are available in the active queue."""
        queue_info = self.get_active_action_queue()
        if queue_info is None:
            return False
        queue, lock = queue_info
        with lock:
            return not queue.empty()
    
    def _action_tensor_to_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        """Convert action tensor to dictionary."""
        return {
            key: action_tensor[i].item() 
            for i, key in enumerate(self.robot.action_features)
        }
    
    def control_loop_action(self, verbose: bool = False) -> dict[str, Any] | None:
        """Execute action from queue."""
        if self.is_paused:
            return None
        
        queue_info = self.get_active_action_queue()
        if queue_info is None:
            return None
        
        queue, lock = queue_info
        
        try:
            with lock:
                if queue.empty():
                    return None
                timed_action = queue.get_nowait()
            
            performed_action = self.robot.send_action(
                self._action_tensor_to_dict(timed_action.get_action())
            )
            
            with self.latest_action_lock:
                self.latest_action = timed_action.get_timestep()
            
            if verbose:
                self.logger.debug(f"Action #{timed_action.get_timestep()} performed")
            
            return performed_action
            
        except Exception as e:
            self.logger.error(f"Error in control_loop_action: {e}")
            return None
    
    def _ready_to_send_observation(self) -> bool:
        """Check if ready to send observation based on queue size."""
        queue_info = self.get_active_action_queue()
        if queue_info is None:
            return True
        
        queue, lock = queue_info
        with lock:
            queue_size = queue.qsize()
        
        if self.action_chunk_size <= 0:
            return True
        
        return queue_size / self.action_chunk_size <= self._chunk_size_threshold
    
    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation | None:
        """Capture and send observation."""
        try:
            start_time = time.perf_counter()
            
            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task
            
            with self.latest_action_lock:
                latest_action = self.latest_action
            
            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )
            
            # Check must_go flag
            queue_info = self.get_active_action_queue()
            if queue_info is not None:
                queue, lock = queue_info
                with lock:
                    observation.must_go = self.must_go.is_set() and queue.empty()
            
            self.send_observation(observation)
            
            if observation.must_go:
                self.must_go.clear()
            
            if verbose:
                obs_time = time.perf_counter() - start_time
                self.logger.debug(f"Observation capture took {obs_time:.4f}s")
            
            return raw_observation
            
        except Exception as e:
            self.logger.error(f"Error in observation: {e}")
            return None
    
    def control_loop(
        self,
        task: str,
        verbose: bool = False,
        on_observation: Callable[[RawObservation], None] | None = None,
        on_action: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[RawObservation | None, dict[str, Any] | None]:
        """
        Main control loop with callbacks for recording.
        
        Args:
            task: Task description string
            verbose: Enable verbose logging
            on_observation: Callback for each observation
            on_action: Callback for each action
        """
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")
        
        performed_action = None
        captured_observation = None
        
        while self.running:
            loop_start = time.perf_counter()
            
            # Perform actions if available and not paused
            if self.actions_available() and not self.is_paused:
                performed_action = self.control_loop_action(verbose)
                if performed_action is not None and on_action is not None:
                    on_action(performed_action)
            
            # Send observations
            if self._ready_to_send_observation():
                captured_observation = self.control_loop_observation(task, verbose)
                if captured_observation is not None and on_observation is not None:
                    on_observation(captured_observation)
            
            # Maintain control frequency
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, self.config.environment_dt - elapsed))
        
        return captured_observation, performed_action
    
    def get_current_observation(self) -> RawObservation:
        """Get current robot observation without sending to server."""
        return self.robot.get_observation()
