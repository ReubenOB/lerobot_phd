#!/usr/bin/env python

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

from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("ros2")
@dataclass
class ROS2CameraConfig(CameraConfig):
    """Configuration for ROS2 camera subscriber.

    Attributes:
        topic_name: ROS2 topic to subscribe to (e.g., '/camera/top/image_raw')
        fps: Expected frame rate (for compatibility, not enforced)
        width: Expected image width (for validation)
        height: Expected image height (for validation)
    """
    topic_name: str = "/camera/image_raw"
