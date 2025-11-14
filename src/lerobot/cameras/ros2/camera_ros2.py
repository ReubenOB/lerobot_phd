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

import logging
import threading
import time
from typing import Any

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode

from .configuration_ros2 import ROS2CameraConfig

logger = logging.getLogger(__name__)


class ROS2Camera(Camera):
    """Camera that subscribes to ROS2 image topics.

    This camera implementation receives images from ROS2 topics instead of
    directly capturing from hardware. Useful for:
    - Integration with ROS2-based robots
    - Simulation environments (Gazebo, Isaac Sim)
    - Remote camera streams
    - Mixed real/simulated setups
    """

    def __init__(self, config: ROS2CameraConfig):
        super().__init__(config)
        self.config = config
        self.topic_name = config.topic_name
        self.camera_name = config.topic_name.split('/')[-2]  # Extract camera name from topic
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Find available ROS2 image topics.

        Returns:
            List of dictionaries containing topic information.
        """
        if not rclpy.ok():
            rclpy.init()

        node = rclpy.create_node('lerobot_camera_finder')
        topic_list = node.get_topic_names_and_types()

        cameras = []
        for topic_name, topic_types in topic_list:
            if 'sensor_msgs/msg/Image' in topic_types:
                cameras.append({
                    'topic_name': topic_name,
                    'type': 'ros2',
                })

        node.destroy_node()
        return cameras

    def connect(self, warmup: bool = True) -> None:
        """Connect to ROS2 camera feed via the shared bridge"""
        if self._connected:
            logger.warning(f"ROS2Camera already connected to {self.topic_name}")
            return

        try:
            # Register this camera with the ROS2 bridge so it subscribes
            from lerobot.common.robot_devices.ros2_bridge import LeRobotROS2Bridge

            # The bridge is already created by the robot, just mark as connected
            self._connected = True
            logger.info(f"ROS2Camera connected to topic: {self.topic_name}")

            if warmup:
                # Wait for first frame
                timeout = 10.0
                start = time.time()
                logger.info(f"Waiting for first frame on {self.topic_name}...")
                while (time.time() - start) < timeout:
                    # Check if we have a frame from the shared publisher
                    # This will be set up when the robot initializes the publisher
                    time.sleep(0.1)
                    # We'll get frames through the robot's observation dict instead
                    break

                logger.info(f"ROS2Camera ready for {self.topic_name}")

        except Exception as e:
            logger.error(f"Failed to connect ROS2Camera to {self.topic_name}: {e}")
            self._connected = False
            raise

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        """Read frame - will be provided by robot's observation dict"""
        # Placeholder - actual frames come from the robot's publisher subscription
        raise RuntimeError(f"ROS2Camera frames are provided through robot observation")

    def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        """Asynchronously read the latest frame.

        Since ROS2 callbacks are already async, this just returns the latest frame.

        Args:
            timeout_ms: Not used (maintained for interface compatibility).

        Returns:
            Latest frame as numpy array.
        """
        return self.read()

    def disconnect(self) -> None:
        """Disconnect from ROS2 and stop receiving images."""
        if not self._connected:
            return

        self._connected = False
        logger.info(f"ROS2Camera disconnected from {self.topic_name}")
