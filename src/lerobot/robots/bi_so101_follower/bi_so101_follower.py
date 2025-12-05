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
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

from ..robot import Robot
from .config_bi_so101_follower import BiSO101FollowerConfig

logger = logging.getLogger(__name__)

# Check if ROS2 is available
try:
    from lerobot.common.robot_devices.ros2_bridge import LeRobotROS2Bridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logger.info("[LeRobot] ROS2 not available")


class BiSO101Follower(Robot):
    """
    [Bimanual SO-101 Follower Arms](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    """

    config_class = BiSO101FollowerConfig
    name = "bi_so101_follower"

    def __init__(self, config: BiSO101FollowerConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SO101FollowerConfig(
            id=config.left_arm_id,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
            enable_ros2_bridge=False,  # Bimanual parent creates shared bridge
        )

        right_arm_config = SO101FollowerConfig(
            id=config.right_arm_id,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            cameras={},
            enable_ros2_bridge=False,  # Bimanual parent creates shared bridge
        )

        self.left_arm = SO101Follower(left_arm_config)
        self.right_arm = SO101Follower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # Initialize ROS2 bridge for bimanual robot if ROS2 is available
        self.ros2_bridge = None
        if ROS2_AVAILABLE:
            try:
                # Build joint names with left_/right_ prefixes
                joint_names = (
                    [f"left_{motor}.pos" for motor in self.left_arm.bus.motors] +
                    [f"right_{motor}.pos" for motor in self.right_arm.bus.motors]
                )
                logger.info(f"[BiSO101] Creating ROS2 bridge with joint names: {joint_names}")
                
                # Get camera names from config
                camera_names = list(config.cameras.keys()) if config.cameras else []

                # Check which cameras are ROS2 subscribers
                subscribe_to_cameras = {}
                for cam_name, cam_config in config.cameras.items():
                    if cam_config.type == 'ros2':
                        subscribe_to_cameras[cam_name] = cam_config.topic_name

                # Always create bridge for joint state publishing (even without ROS2 cameras)
                self.ros2_bridge = LeRobotROS2Bridge(
                    node_name='lerobot_bi_so101_follower',
                    joint_names=joint_names,
                    camera_names=camera_names,
                    subscribe_to_cameras=subscribe_to_cameras,
                )
                logger.info(f"[LeRobot] ROS2 bridge enabled for BiSO101Follower, bridge.enabled={self.ros2_bridge.enabled}")
            except Exception as e:
                logger.warning(f"[LeRobot] ROS2 bridge failed to initialize: {e}")
                import traceback
                traceback.print_exc()

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        for cam in self.cameras.values():
            cam.connect()
            time.sleep(0.5)  # Small delay between camera connections to avoid USB bandwidth issues

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Add "left_" prefix - but filter out camera keys (we handle cameras separately)
        left_obs = self.left_arm.get_observation()
        for key, value in left_obs.items():
            # Skip camera observations from child arms (we handle cameras at this level)
            if key not in self.left_arm.cameras:
                obs_dict[f"left_{key}"] = value

        # Add "right_" prefix - but filter out camera keys
        right_obs = self.right_arm.get_observation()
        for key, value in right_obs.items():
            if key not in self.right_arm.cameras:
                obs_dict[f"right_{key}"] = value

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()

            # Check if this is a ROS2 camera - get frame from bridge instead
            if hasattr(cam, 'config') and cam.config.type == 'ros2':
                if self.ros2_bridge:
                    frame = self.ros2_bridge.get_camera_frame(cam_key)
                    if frame is not None:
                        obs_dict[cam_key] = frame
                    else:
                        logger.warning(f"No frame available for ROS2 camera {cam_key}")
                else:
                    logger.warning(f"ROS2 camera {cam_key} but no ros2_bridge set")
            else:
                obs_dict[cam_key] = cam.async_read()

            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        # Publish to ROS2 if enabled
        if self.ros2_bridge:
            self.ros2_bridge.publish_joint_states(obs_dict)
            self.ros2_bridge.publish_images(obs_dict)

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command both arms to move to target joint configurations.
        
        If ROS2 bridge is enabled and uncertainty is high, will skip sending action
        (non-blocking) to allow faster resume when uncertainty normalizes.
        
        Returns:
            the action sent to the motors. Returns the input action unchanged if
            paused due to high uncertainty.
        """
        # Check for uncertainty pause (if ROS2 bridge is enabled)
        # Non-blocking: just skip this action if paused
        if self.ros2_bridge and self.ros2_bridge.is_paused():
            # Return input action without sending (robot holds position)
            return action
        
        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        # Add prefixes back
        prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()
        
        # Shutdown ROS2 bridge
        if self.ros2_bridge:
            self.ros2_bridge.shutdown()
