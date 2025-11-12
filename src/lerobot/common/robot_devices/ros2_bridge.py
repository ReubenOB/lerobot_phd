#!/usr/bin/env python3
"""
ROS2 Bridge for LeRobot
Integrates ROS2 publishing and subscribing directly into LeRobot robot classes
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header
import threading
from typing import Dict, Optional
import numpy as np


class LeRobotROS2Bridge:
    """
    ROS2 bridge that can be integrated into LeRobot robot classes.
    Handles both publishing robot state and subscribing to camera feeds.
    """

    # Class-level shared executor
    _shared_executor = None
    _executor_thread = None
    _executor_lock = threading.Lock()

    def __init__(self, node_name: str = 'lerobot_bridge', joint_names: list = None, camera_names: list = None, subscribe_to_cameras: dict = None):
        """
        Initialize ROS2 bridge

        Args:
            node_name: Name for the ROS2 node
            joint_names: List of joint names for the robot
            camera_names: List of camera names to publish (e.g., ['top', 'wrist'])
            subscribe_to_cameras: Dict of camera_name -> topic_name to subscribe to
        """
        self.joint_names = joint_names or []
        self.camera_names = camera_names or []
        self.subscribe_to_cameras = subscribe_to_cameras or {}
        self.node = None
        self.joint_state_publisher = None
        self.image_publishers = {}
        self.enabled = False
        self.camera_subscribers = {}
        self.camera_frames = {}  # Store latest frames from subscribed cameras
        self.camera_locks = {}  # Thread locks for camera frames

        # Try to initialize ROS2
        try:
            if not rclpy.ok():
                rclpy.init()

            self.node = rclpy.create_node(node_name)
            self.joint_state_publisher = self.node.create_publisher(JointState, 'joint_states', 10)

            # Create image publishers for each camera
            for camera_name in self.camera_names:
                topic_name = f'/camera/{camera_name}/image_raw'
                self.image_publishers[camera_name] = self.node.create_publisher(
                    Image, topic_name, 10)

            # Create camera subscribers
            for camera_name, topic_name in self.subscribe_to_cameras.items():
                self.camera_frames[camera_name] = None
                self.camera_locks[camera_name] = threading.Lock()

                def make_callback(cam_name):
                    def callback(msg):
                        self._camera_callback(cam_name, msg)
                    return callback

                self.camera_subscribers[camera_name] = self.node.create_subscription(
                    Image,
                    topic_name,
                    make_callback(camera_name),
                    10
                )
                print(f"[LeRobot] Subscribed to camera: {camera_name} -> {topic_name}")

            self.enabled = True

            # Spin in background thread
            self.spin_thread = threading.Thread(target=self._spin, daemon=True)
            self.spin_thread.start()

            print(f"[LeRobot] ROS2 bridge initialized: {node_name}")
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
            # Skip publishing if we're subscribing to this camera from ROS2
            if camera_name in self.subscribe_to_cameras:
                continue

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

    def _camera_callback(self, camera_name: str, msg: Image):
        """Callback for receiving camera images"""
        try:
            # Directly convert Image message to numpy without cv_bridge to avoid NumPy issues
            import numpy as np

            # Decode the image data
            if msg.encoding == 'rgb8':
                dtype = np.uint8
                channels = 3
            elif msg.encoding == 'bgr8':
                dtype = np.uint8
                channels = 3
            else:
                print(f"[LeRobot] Unsupported encoding: {msg.encoding}")
                return

            # Reshape the flat data into an image array
            img_array = np.frombuffer(msg.data, dtype=dtype).reshape(
                msg.height, msg.width, channels)

            # Convert BGR to RGB if needed
            if msg.encoding == 'bgr8':
                img_array = img_array[:, :, ::-1]  # Reverse channel order

            with self.camera_locks[camera_name]:
                self.camera_frames[camera_name] = img_array.copy()

        except Exception as e:
            print(f"[LeRobot] Failed to convert camera image for {camera_name}: {e}")

    def get_camera_frame(self, camera_name: str) -> Optional[np.ndarray]:
        """Get the latest frame from a subscribed camera"""
        if camera_name not in self.camera_frames:
            return None
        with self.camera_locks[camera_name]:
            if self.camera_frames[camera_name] is not None:
                return self.camera_frames[camera_name].copy()
        return None

    def shutdown(self):
        """Shutdown ROS2 publisher"""
        self.enabled = False
        if self.node:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
