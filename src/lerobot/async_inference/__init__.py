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
Async inference module for multi-policy orchestration.

This module provides components for:
- Multi-policy execution with state machine orchestration
- Movement buffer for trajectory recording and reverse playback
- Data recording with rich metadata
- Policy server communication via gRPC
"""

from .movement_buffer import (
    MovementBuffer,
    TrajectoryFrame,
    TrajectoryInterpolator,
    create_movement_buffer,
    DEFAULT_JOINT_LIMITS,
)
from .orchestrator import (
    MultiPolicyOrchestrator,
    OrchestratorNode,
    State,
    run_orchestrator,
)
from .data_recorder import (
    MultiPolicyDataRecorder,
    RecordingMetadata,
)
from .multi_policy_client import RobotClientMulti
from .configs_multi import (
    MultiPolicyConfig,
    OrchestratorConfig,
    PolicyServerSpec,
    SARMConfig,
)

__all__ = [
    # Movement buffer
    "MovementBuffer",
    "TrajectoryFrame",
    "TrajectoryInterpolator",
    "create_movement_buffer",
    "DEFAULT_JOINT_LIMITS",
    # Orchestrator
    "MultiPolicyOrchestrator",
    "OrchestratorNode",
    "State",
    "run_orchestrator",
    # Data recorder
    "MultiPolicyDataRecorder",
    "RecordingMetadata",
    # Client
    "RobotClientMulti",
    # Configs
    "MultiPolicyConfig",
    "OrchestratorConfig",
    "PolicyServerSpec",
    "SARMConfig",
]
