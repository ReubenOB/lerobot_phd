#!/usr/bin/env python3
"""
ROS2 Publisher for LeRobot
Integrates ROS2 publishing directly into LeRobot robot classes
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import threading
from typing import Dict, Optional


class LeRobotROS2Publisher:
    """
    ROS2 publisher that can be integrated into LeRobot robot classes.
    Publishes joint states directly when robot observations are available.
    """

    def __init__(self, node_name: str = 'lerobot_publisher', joint_names: list = None):
        """
        Initialize ROS2 publisher

        Args:
            node_name: Name for the ROS2 node
            joint_names: List of joint names for the robot
        """
        self.joint_names = joint_names or []
        self.node = None
        self.publisher = None
        self.enabled = False

        # Try to initialize ROS2
        try:
            if not rclpy.ok():
                rclpy.init()

            self.node = rclpy.create_node(node_name)
            self.publisher = self.node.create_publisher(JointState, 'joint_states', 10)
            self.enabled = True

            # Spin in background thread
            self.spin_thread = threading.Thread(target=self._spin, daemon=True)
            self.spin_thread.start()

            print(f"[LeRobot] ROS2 publisher initialized: {node_name}")

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
        if not self.enabled or not self.publisher:
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
                    msg.name.append(joint_name)
                    msg.position.append(float(observation[joint_name]))
                    msg.velocity.append(0.0)
                    msg.effort.append(0.0)

            self.publisher.publish(msg)

        except Exception as e:
            print(f"[LeRobot] Failed to publish joint states: {e}")

    def shutdown(self):
        """Shutdown ROS2 publisher"""
        self.enabled = False
        if self.node:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
