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
Configuration dataclasses for multi-policy orchestration system.

This module provides draccus-compatible configuration classes for:
- Multiple policy server specifications
- Gaze-based object selection
- SARM progress monitoring
- RND uncertainty detection
- Multi-policy orchestration
"""

from dataclasses import dataclass, field
from typing import Any

from lerobot.robots.config import RobotConfig

from .configs import AGGREGATE_FUNCTIONS, DEFAULT_FPS


@dataclass
class PolicyServerSpec:
    """Specification for a single policy server connection."""

    name: str = field(metadata={"help": "Unique name for this policy server (e.g., 'pick_pod', 'make_coffee')"})
    host: str = field(default="localhost", metadata={"help": "Host address of the policy server"})
    port: int = field(default=8080, metadata={"help": "Port number of the policy server"})
    policy_type: str = field(default="act", metadata={"help": "Type of policy (act, diffusion, etc.)"})
    pretrained_path: str = field(default="", metadata={"help": "HuggingFace model path or local path"})
    actions_per_chunk: int = field(default=50, metadata={"help": "Number of actions per inference chunk"})
    device: str = field(default="cuda", metadata={"help": "Device for policy inference"})
    task: str = field(default="", metadata={"help": "Task description for this policy (used for SARM)"})

    def __post_init__(self):
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")
        if not self.name:
            raise ValueError("Policy server name cannot be empty")

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


# NOTE: GazeRegion and GazeConfig are NOT needed.
# Gaze-based object selection is trained directly into the ACT policies.
# The Aria camera with gaze visualization is fed as input to the policy.


@dataclass
class SARMConfig:
    """Configuration for SARM progress monitoring."""

    # Model path
    model_path: str = field(
        default="",
        metadata={"help": "Path to pretrained SARM model (HuggingFace or local)"}
    )
    
    # Device
    device: str = field(default="cuda", metadata={"help": "Device for SARM inference"})
    
    # Progress threshold to switch between policies
    # When SARM progress >= this threshold, the orchestrator switches to the next policy
    policy_switch_threshold: float = field(
        default=0.95,
        metadata={"help": "Progress threshold to trigger policy switch (0-1)"}
    )
    
    # Annotation scheme
    scheme: str = field(
        default="sparse",
        metadata={"help": "SARM annotation scheme ('sparse' or 'dense')"}
    )
    
    # ROS2 topics
    progress_topic: str = field(
        default="/sarm/progress",
        metadata={"help": "Topic to publish overall progress (Float32)"}
    )
    stage_topic: str = field(
        default="/sarm/stage",
        metadata={"help": "Topic to publish current stage index (Int32)"}
    )
    stage_name_topic: str = field(
        default="/sarm/stage_name",
        metadata={"help": "Topic to publish current stage name (String)"}
    )
    
    # Update rate
    rate_hz: float = field(default=10.0, metadata={"help": "Update rate in Hz"})
    
    # Camera topics for observation
    camera_topics: list[str] = field(
        default_factory=lambda: ["/camera/top/image_raw"],
        metadata={"help": "Camera topics for SARM observation"}
    )


@dataclass
class RNDConfig:
    """Configuration for RND uncertainty monitoring (uses Aria camera ONLY)."""

    # Model path
    model_path: str = field(
        default="",
        metadata={"help": "Path to pretrained RND model"}
    )
    
    # Uncertainty threshold
    threshold: float = field(
        default=10.5,
        metadata={"help": "Uncertainty threshold for pause trigger"}
    )
    
    # Enable pause on high uncertainty
    enable_pause: bool = field(
        default=True,
        metadata={"help": "Whether to pause on high uncertainty"}
    )
    
    # ROS2 topics
    pause_topic: str = field(
        default="/uncertainty/pause",
        metadata={"help": "Topic to subscribe for pause signal (Bool)"}
    )
    uncertainty_topic: str = field(
        default="/uncertainty/rolling",
        metadata={"help": "Topic to subscribe for rolling uncertainty (Float32)"}
    )
    
    # Camera topic for RND (uses Aria glasses only for gaze-based safety)
    camera_topic: str = field(
        default="/aria/eye_gaze/visualization_small",
        metadata={"help": "Camera topic for RND observation (Aria glasses only)"}
    )


@dataclass
class DatasetConfig:
    """Configuration for dataset recording."""

    repo_id: str = field(
        default="",
        metadata={"help": "HuggingFace dataset repository ID"}
    )
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Recording FPS"})
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push dataset to HuggingFace Hub after recording"}
    )
    use_videos: bool = field(
        default=True,
        metadata={"help": "Whether to save camera data as videos"}
    )
    private: bool = field(
        default=True,
        metadata={"help": "Whether the dataset should be private on Hub"}
    )
    image_writer_threads: int = field(
        default=4,
        metadata={"help": "Number of threads for image writing"}
    )


@dataclass
class OrchestratorConfig:
    """Configuration for the multi-policy orchestrator state machine."""

    # State machine timing
    transition_delay_s: float = field(
        default=0.5,
        metadata={"help": "Delay in seconds between state transitions"}
    )
    
    # Max episode time
    max_episode_time_s: float = field(
        default=300.0,
        metadata={"help": "Maximum episode time in seconds"}
    )
    
    # Number of episodes to record
    num_episodes: int = field(
        default=10,
        metadata={"help": "Number of episodes to record"}
    )
    
    # Task descriptions for each policy (used for SARM text embeddings)
    policy_1_task: str = field(
        default="Pick up the coffee pod and insert it into the machine",
        metadata={"help": "Task description for policy 1 (used for SARM)"}
    )
    policy_2_task: str = field(
        default="Pick up the cup and make coffee",
        metadata={"help": "Task description for policy 2 (used for SARM)"}
    )


@dataclass
class MultiPolicyConfig:
    """Complete configuration for multi-policy orchestration system.
    
    NOTE: Gaze-based object selection is NOT handled by this system.
    Gaze selection is trained directly into the ACT policies via the Aria camera input.
    """

    # Robot configuration (bimanual)
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})
    
    # Policy servers
    policy_servers: list[dict[str, Any]] = field(
        default_factory=list,
        metadata={"help": "List of policy server specifications"}
    )
    
    # Sub-configurations (no gaze - gaze is in the policy)
    sarm: SARMConfig = field(default_factory=SARMConfig)
    rnd: RNDConfig = field(default_factory=RNDConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    
    # Control behavior
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Control loop FPS"})
    chunk_size_threshold: float = field(
        default=0.5,
        metadata={"help": "Threshold for action chunk size control"}
    )
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"Aggregate function name. Options: {list(AGGREGATE_FUNCTIONS.keys())}"}
    )
    
    # Debug options
    debug_visualize: bool = field(
        default=False,
        metadata={"help": "Enable debug visualization"}
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step in seconds."""
        return 1 / self.fps

    def get_policy_servers(self) -> list[PolicyServerSpec]:
        """Convert policy server dicts to PolicyServerSpec objects."""
        return [PolicyServerSpec(**spec) for spec in self.policy_servers]

    def __post_init__(self):
        """Validate configuration."""
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(
                f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}"
            )
        if self.aggregate_fn_name not in AGGREGATE_FUNCTIONS:
            raise ValueError(
                f"Unknown aggregate function '{self.aggregate_fn_name}'. "
                f"Available: {list(AGGREGATE_FUNCTIONS.keys())}"
            )
