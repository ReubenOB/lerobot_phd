#!/usr/bin/env python3
"""
ROS2 Publisher for LeRobot
Integrates ROS2 publishing directly into LeRobot robot classes
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header
import threading
from typing import Dict, Optional
import numpy as np
from cv_bridge import CvBridge


class LeRobotROS2Publisher:
    """
    ROS2 publisher that can be integrated into LeRobot robot classes.
    Publishes joint states and camera feeds directly when robot observations are available.
    """

    def __init__(self, node_name: str = 'lerobot_publisher', joint_names: list = None, camera_names: list = None):
        """
        Initialize ROS2 publisher

        Args:
            node_name: Name for the ROS2 node
            joint_names: List of joint names for the robot
            camera_names: List of camera names (e.g., ['top', 'wrist'])
        """
        self.joint_names = joint_names or []
        self.camera_names = camera_names or []
        self.node = None
        self.joint_state_publisher = None
        self.image_publishers = {}
        self.enabled = False
        self.bridge = CvBridge()

        # Try to initialize ROS2
        try:
            if not rclpy.ok():
                rclpy.init()

            self.node = rclpy.create_node(node_name)
            self.joint_state_publisher = self.node.create_publisher(JointState, 'joint_states', 10)

            # Create image publishers for each camera
            for camera_name in self.camera_names:
                topic_name = f'camera/{camera_name}/image_raw'
                self.image_publishers[camera_name] = self.node.create_publisher(
                    Image, topic_name, 10)

            self.enabled = True

            # Spin in background thread
            self.spin_thread = threading.Thread(target=self._spin, daemon=True)
            self.spin_thread.start()

            print(f"[LeRobot] ROS2 publisher initialized: {node_name}")
            if self.camera_names:
                print(f"[LeRobot] Publishing camera feeds: {self.camera_names}")

        except Exception as e:
            print(f"[LeRobot] ROS2 not available: {e}")
            self.enabled = False

    def _spin(self):
        """Background thread to spin ROS2"""
        while rclpy.ok() and self.enabled:
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def publish_joint_states(self, observation: Dict[str, float]):
        """
        Publish joint states from robot observation

        Args:
            observation: Dictionary of joint_name -> position
        """
        if not self.enabled or not self.joint_state_publisher:
            return

        try:
            msg = JointState()
            msg.header = Header()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'

            msg.name = []
            msg.position = []
            msg.velocity = []
            msg.effort = []

            for joint_name in self.joint_names:
                if joint_name in observation:
                    # Strip .pos suffix from joint names for ROS2 compatibility
                    clean_joint_name = joint_name.replace('.pos', '')
                    msg.name.append(clean_joint_name)

                    # Convert from degrees to radians for ROS2
                    position_rad = float(observation[joint_name]) * (3.14159265359 / 180.0)
                    msg.position.append(position_rad)
                    msg.velocity.append(0.0)
                    msg.effort.append(0.0)

            self.joint_state_publisher.publish(msg)

        except Exception as e:
            print(f"[LeRobot] Failed to publish joint states: {e}")

    def publish_images(self, observation: dict):
        """Publish camera images from observation dictionary."""
        if not self.image_publishers:
            return

        for camera_name in self.camera_names:
            if camera_name in observation:
                img_data = observation[camera_name]

                # Convert numpy array to ROS2 Image message
                if img_data is not None and hasattr(img_data, 'shape'):
                    img_msg = Image()
                    img_msg.header.stamp = self.node.get_clock().now().to_msg()
                    img_msg.header.frame_id = camera_name
                    img_msg.height = img_data.shape[0]
                    img_msg.width = img_data.shape[1]
                    img_msg.encoding = 'rgb8' if img_data.shape[2] == 3 else 'rgba8'
                    img_msg.is_bigendian = 0
                    img_msg.step = img_data.shape[1] * img_data.shape[2]
                    img_msg.data = img_data.tobytes()

                    self.image_publishers[camera_name].publish(img_msg)

    def shutdown(self):
        """Shutdown ROS2 publisher"""
        self.enabled = False
        if self.node:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
