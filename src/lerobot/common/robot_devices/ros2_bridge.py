#!/usr/bin/env python3
"""
ROS2 Bridge for LeRobot
Integrates ROS2 publishing, subscribing, and MoveIt control directly into LeRobot robot classes.
Provides FollowJointTrajectory action server for MoveIt integration.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import Header, Bool
import threading
import time
from typing import Dict, Optional, Callable
import numpy as np


class LeRobotROS2Bridge:
    """
    ROS2 bridge that can be integrated into LeRobot robot classes.
    Handles publishing robot state, subscribing to camera feeds, and MoveIt trajectory control.
    """

    def __init__(self, node_name: str = 'lerobot_bridge', joint_names: list = None, 
                 camera_names: list = None, subscribe_to_cameras: dict = None,
                 send_action_callback: Callable = None, enable_uncertainty_pause: bool = True,
                 on_resume_callback: Callable = None):
        """
        Initialize ROS2 bridge

        Args:
            node_name: Name for the ROS2 node
            joint_names: List of joint names for the robot (e.g., ['shoulder_pan.pos', ...])
            camera_names: List of camera names to publish (e.g., ['top', 'wrist'])
            subscribe_to_cameras: Dict of camera_name -> topic_name to subscribe to
            send_action_callback: Callback function to send actions to robot (for MoveIt control)
            enable_uncertainty_pause: Enable listening to /uncertainty/pause topic
            on_resume_callback: Callback to call when resuming from pause (e.g., to reset policy)
        """
        self.joint_names = joint_names or []
        self.camera_names = camera_names or []
        self.subscribe_to_cameras = subscribe_to_cameras or {}
        self.send_action_callback = send_action_callback
        self.enable_uncertainty_pause = enable_uncertainty_pause
        self.on_resume_callback = on_resume_callback
        self.node = None
        self.joint_state_publisher = None
        self.image_publishers = {}
        self.enabled = False
        self.camera_subscribers = {}
        self.camera_frames = {}
        self.camera_locks = {}
        self.executor = None
        self.spin_thread = None
        
        # Lock to prevent concurrent bus access during trajectory execution
        self.bus_lock = threading.Lock()
        self.trajectory_executing = False
        
        # Uncertainty-based pause state
        self.uncertainty_paused = False
        self.uncertainty_pause_lock = threading.Lock()
        self._just_resumed = False  # Flag to indicate we just resumed from pause
        
        # Clean joint names (without .pos suffix) for ROS2
        self.clean_joint_names = [j.replace('.pos', '') for j in self.joint_names]

        # Try to initialize ROS2
        try:
            if not rclpy.ok():
                rclpy.init()

            self.node = rclpy.create_node(node_name)
            
            # Callback group for concurrent callbacks
            self.callback_group = ReentrantCallbackGroup()
            
            # Joint state publisher
            self.joint_state_publisher = self.node.create_publisher(
                JointState, '/joint_states', 10)

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
                    Image, topic_name, make_callback(camera_name), 10)
                print(f"[LeRobot] Subscribed to camera: {camera_name} -> {topic_name}")

            # FollowJointTrajectory action server for MoveIt
            if self.send_action_callback:
                self.trajectory_action_server = ActionServer(
                    self.node,
                    FollowJointTrajectory,
                    '/arm_controller/follow_joint_trajectory',
                    self._execute_trajectory_callback,
                    callback_group=self.callback_group
                )
                print(f"[LeRobot] MoveIt trajectory action server enabled")
            
            # Trajectory topic subscriber (alternative to action)
            self.trajectory_sub = self.node.create_subscription(
                JointTrajectory,
                '/arm_controller/joint_trajectory',
                self._trajectory_callback,
                10,
                callback_group=self.callback_group
            )

            # Subscribe to uncertainty pause topic
            if self.enable_uncertainty_pause:
                self.pause_subscriber = self.node.create_subscription(
                    Bool,
                    '/uncertainty/pause',
                    self._uncertainty_pause_callback,
                    10
                )
                print(f"[LeRobot] Uncertainty pause subscriber enabled")

            self.enabled = True

            # Use MultiThreadedExecutor for action server
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.node)
            
            # Spin in background thread
            self.spin_thread = threading.Thread(target=self._spin, daemon=True)
            self.spin_thread.start()

            print(f"[LeRobot] ROS2 bridge initialized: {node_name}")
            print(f"[LeRobot] Joint names: {self.clean_joint_names}")

        except Exception as e:
            print(f"[LeRobot] ROS2 not available: {e}")
            import traceback
            traceback.print_exc()
            self.enabled = False

    def _spin(self):
        """Background thread to spin ROS2"""
        try:
            while rclpy.ok() and self.enabled:
                self.executor.spin_once(timeout_sec=0.01)
        except Exception as e:
            print(f"[LeRobot] Spin error: {e}")

    def _uncertainty_pause_callback(self, msg: Bool):
        """Callback for uncertainty pause signal."""
        with self.uncertainty_pause_lock:
            was_paused = self.uncertainty_paused
            self.uncertainty_paused = msg.data
            
            if msg.data and not was_paused:
                print("[LeRobot] ⚠️  HIGH UNCERTAINTY - Robot paused")
            elif not msg.data and was_paused:
                print("[LeRobot] ✓ Uncertainty normalized - Robot resumed")
                self._just_resumed = True
                # Call resume callback (e.g., to reset policy action queue)
                if self.on_resume_callback:
                    try:
                        self.on_resume_callback()
                    except Exception as e:
                        print(f"[LeRobot] Resume callback error: {e}")

    def is_paused(self) -> bool:
        """Check if robot should be paused due to high uncertainty."""
        with self.uncertainty_pause_lock:
            return self.uncertainty_paused

    def check_and_clear_resumed(self) -> bool:
        """Check if we just resumed from pause and clear the flag.
        
        Returns:
            True if we just resumed (policy should be reset), False otherwise.
        """
        with self.uncertainty_pause_lock:
            if self._just_resumed:
                self._just_resumed = False
                return True
            return False

    def wait_for_resume(self, timeout: float = None) -> bool:
        """
        Block until uncertainty drops and pause is released.
        
        Args:
            timeout: Maximum time to wait in seconds. None = wait forever.
            
        Returns:
            True if resumed, False if timeout occurred.
        """
        start_time = time.time()
        while self.is_paused():
            time.sleep(0.05)  # Check every 50ms
            if timeout and (time.time() - start_time) > timeout:
                return False
        return True

    def publish_joint_states(self, observation: Dict[str, float]):
        """
        Publish joint states from robot observation

        Args:
            observation: Dictionary of joint_name -> position (e.g., {'shoulder_pan.pos': 45.0})
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
                    position_rad = float(observation[joint_name]) * (np.pi / 180.0)
                    msg.position.append(position_rad)
                    msg.velocity.append(0.0)
                    msg.effort.append(0.0)

            self.joint_state_publisher.publish(msg)

        except Exception as e:
            print(f"[LeRobot] Failed to publish joint states: {e}")

    def _trajectory_callback(self, msg: JointTrajectory):
        """Handle incoming trajectory commands (topic-based)."""
        if not self.send_action_callback or len(msg.points) == 0:
            return
            
        try:
            # Get the final target position
            target_point = msg.points[-1]
            
            # Build action dict for LeRobot
            action = {}
            for joint_name, pos in zip(msg.joint_names, target_point.positions):
                # Convert from radians to degrees
                pos_deg = float(pos) * (180.0 / np.pi)
                action[f"{joint_name}.pos"] = pos_deg
            
            print(f"[LeRobot] Trajectory target: {action}")
            self.send_action_callback(action)
            
        except Exception as e:
            print(f"[LeRobot] Trajectory callback error: {e}")

    def _execute_trajectory_callback(self, goal_handle):
        """Execute a FollowJointTrajectory action (for MoveIt integration)."""
        print("[LeRobot] Executing MoveIt trajectory...")
        
        trajectory = goal_handle.request.trajectory
        
        if len(trajectory.points) == 0:
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            return result
        
        print(f"[LeRobot] Trajectory: {len(trajectory.points)} points, joints: {list(trajectory.joint_names)}")
        
        # Signal that trajectory is executing (to pause observation loop)
        self.trajectory_executing = True
        
        start_time = time.time()
        
        try:
            # Acquire lock to prevent concurrent bus access
            with self.bus_lock:
                for idx, point in enumerate(trajectory.points):
                    target_time = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
                    
                    # Wait until target time
                    elapsed = time.time() - start_time
                    if target_time > elapsed:
                        time.sleep(target_time - elapsed)
                    
                    # Build action dict for LeRobot
                    action = {}
                    for joint_name, pos in zip(trajectory.joint_names, point.positions):
                        # Convert from radians to degrees
                        pos_deg = float(pos) * (180.0 / np.pi)
                        action[f"{joint_name}.pos"] = pos_deg
                    
                    # Send to robot
                    if self.send_action_callback:
                        self.send_action_callback(action)
                    
                    # Publish feedback
                    feedback = FollowJointTrajectory.Feedback()
                    feedback.header.stamp = self.node.get_clock().now().to_msg()
                    feedback.joint_names = list(trajectory.joint_names)
                    feedback.desired.positions = list(point.positions)
                    feedback.actual.positions = list(point.positions)  # Approximate
                    feedback.error.positions = [0.0] * len(trajectory.joint_names)
                    goal_handle.publish_feedback(feedback)
                    
                    if idx % 10 == 0:
                        print(f"[LeRobot] Point {idx+1}/{len(trajectory.points)}")
            
            goal_handle.succeed()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            print("[LeRobot] Trajectory complete")
            return result
            
        except Exception as e:
            print(f"[LeRobot] Trajectory failed: {e}")
            import traceback
            traceback.print_exc()
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            return result
        finally:
            self.trajectory_executing = False

    def publish_images(self, observation: dict):
        """Publish camera images from observation dictionary."""
        if not self.image_publishers:
            return

        for camera_name in self.camera_names:
            if camera_name in self.subscribe_to_cameras:
                continue

            if camera_name in observation:
                img_data = observation[camera_name]

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
            if msg.encoding == 'rgb8':
                dtype = np.uint8
                channels = 3
            elif msg.encoding == 'bgr8':
                dtype = np.uint8
                channels = 3
            else:
                print(f"[LeRobot] Unsupported encoding: {msg.encoding}")
                return

            img_array = np.frombuffer(msg.data, dtype=dtype).reshape(
                msg.height, msg.width, channels)

            if msg.encoding == 'bgr8':
                img_array = img_array[:, :, ::-1]

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
        """Shutdown ROS2 bridge"""
        self.enabled = False
        if self.node:
            self.node.destroy_node()
        # Don't shutdown rclpy as other nodes may be using it
