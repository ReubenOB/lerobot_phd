#!/usr/bin/env python3
"""
ROS2 Bridge for LeRobot
Integrates ROS2 publishing, subscribing, and MoveIt control directly into LeRobot robot classes.
Provides FollowJointTrajectory action server for MoveIt integration.
Supports optional control services and Aria gesture subscriptions.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import Header, Bool, String
import threading
import time
from typing import Dict, Optional, Callable
import numpy as np

# Optional service imports
try:
    from std_srvs.srv import Trigger
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False


class LeRobotROS2Bridge:
    """
    ROS2 bridge that can be integrated into LeRobot robot classes.
    Handles publishing robot state, subscribing to camera feeds, and MoveIt trajectory control.
    """
    
    # SO-101 URDF joint limits (in degrees) - symmetric around 0
    # These define the physical range for each joint type
    # Format: (lower_deg, upper_deg) 
    SO101_JOINT_LIMITS = {
        'shoulder_pan': (-110.0, 110.0),    # ±110°
        'shoulder_lift': (-100.0, 100.0),   # ±100°
        'elbow_flex': (-100.0, 90.0),       # -100° to +90°
        'wrist_flex': (-95.0, 95.0),        # ±95°
        'wrist_roll': (-160.0, 160.0),      # ±160°
    }

    def __init__(self, node_name: str = 'lerobot_bridge', joint_names: list = None, 
                 camera_names: list = None, subscribe_to_cameras: dict = None,
                 send_action_callback: Callable = None, enable_pause: bool = True,
                 on_resume_callback: Callable = None, prismatic_gripper: bool = False,
                 joint_offsets: dict = None,
                 # Control service callbacks (optional)
                 on_start: Callable = None,
                 on_pause: Callable = None,
                 on_resume: Callable = None,
                 on_reset: Callable = None,
                 on_stop: Callable = None,
                 on_switch_policy: Callable = None,
                 # Aria gesture callbacks (optional)
                 on_gaze_left: Callable = None,
                 on_gaze_right: Callable = None,
                 on_eyes_closed: Callable = None,
                 # State publisher callback (optional)
                 get_state_string: Callable = None):
        """
        Initialize ROS2 bridge

        Args:
            node_name: Name for the ROS2 node
            joint_names: List of joint names for the robot (e.g., ['shoulder_pan.pos', ...])
            camera_names: List of camera names to publish (e.g., ['top', 'wrist'])
            subscribe_to_cameras: Dict of camera_name -> topic_name to subscribe to
            send_action_callback: Callback function to send actions to robot (for MoveIt control)
            enable_pause: Enable listening to /robot/pause topic
            on_resume_callback: Callback to call when resuming from pause (e.g., to reset policy)
            prismatic_gripper: If True, convert gripper values from 0-100 range to meters for prismatic joints
            joint_offsets: Dict of joint_name -> offset in degrees to correct calibration mismatch
            
            Control service callbacks (all optional, return tuple[bool, str]):
                on_start: Called when /robot/start service is triggered
                on_pause: Called when /robot/pause service is triggered
                on_resume: Called when /robot/resume service is triggered
                on_reset: Called when /robot/reset service is triggered
                on_stop: Called when /robot/stop service is triggered
                on_switch_policy: Called when /robot/switch_policy service is triggered
            
            Aria gesture callbacks (all optional):
                on_gaze_left: Called when left gaze gesture detected
                on_gaze_right: Called when right gaze gesture detected
                on_eyes_closed: Called when eyes closed detected
            
            State publisher (optional):
                get_state_string: Callback that returns current state string for /robot/state topic
        """
        self.joint_names = joint_names or []
        self.camera_names = camera_names or []
        self.subscribe_to_cameras = subscribe_to_cameras or {}
        self.send_action_callback = send_action_callback
        self.enable_pause = enable_pause
        self.on_resume_callback = on_resume_callback
        self.prismatic_gripper = prismatic_gripper
        self.joint_offsets = joint_offsets or {}
        
        # Control service callbacks
        self.on_start = on_start
        self.on_pause = on_pause
        self.on_resume = on_resume
        self.on_reset = on_reset
        self.on_stop = on_stop
        self.on_switch_policy = on_switch_policy
        
        # Aria gesture callbacks
        self.on_gaze_left = on_gaze_left
        self.on_gaze_right = on_gaze_right
        self.on_eyes_closed = on_eyes_closed
        
        # State publisher callback
        self.get_state_string = get_state_string
        
        # Prismatic gripper conversion constants
        # LeRobot gripper: 0 (closed) to 100 (open)
        # Parallel gripper: -0.035 (open) to -0.0075 (closed) meters for left finger
        self.gripper_min_meters = -0.035  # fully open
        self.gripper_max_meters = -0.0075  # fully closed
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
        
        # Generic pause state (can be triggered by uncertainty, RND, Aria, manual, etc.)
        self.pause_requested = False
        self.pause_lock = threading.Lock()
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

            # Subscribe to pause control topic (generic - works with uncertainty, RND, Aria, manual, etc.)
            if self.enable_pause:
                self.pause_subscriber = self.node.create_subscription(
                    Bool,
                    '/robot/pause',
                    self._pause_callback,
                    10
                )
                print(f"[LeRobot] Pause control subscriber enabled on /robot/pause")
            
            # Create control services if callbacks are provided
            if SERVICES_AVAILABLE and any([self.on_start, self.on_pause, self.on_resume, 
                                           self.on_reset, self.on_stop, self.on_switch_policy]):
                self._create_control_services()
            
            # Create Aria gesture subscriptions if callbacks are provided
            if any([self.on_gaze_left, self.on_gaze_right]):
                self.gaze_sub = self.node.create_subscription(
                    String, "/aria/gaze_gesture/detected", self._gaze_callback, 10
                )
                print("[LeRobot] Aria gaze gesture subscriber enabled")
            
            if self.on_eyes_closed:
                self.eyes_closed_sub = self.node.create_subscription(
                    Bool, "/aria/blink/eyes_closed_detected", self._eyes_closed_callback, 10
                )
                print("[LeRobot] Aria eyes closed subscriber enabled")
            
            # Create state publisher if callback is provided
            if self.get_state_string:
                self.state_pub = self.node.create_publisher(String, "/robot/state", 10)
                self.node.create_timer(0.5, self._publish_state)
                print("[LeRobot] State publisher enabled on /robot/state")

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

    def _pause_callback(self, msg: Bool):
        """Callback for pause control signal (from uncertainty, RND, Aria, manual, etc.)."""
        with self.pause_lock:
            was_paused = self.pause_requested
            self.pause_requested = msg.data
            
            if msg.data and not was_paused:
                print("[LeRobot] ⏸️  PAUSE REQUESTED - Robot paused")
            elif not msg.data and was_paused:
                print("[LeRobot] ▶️  RESUME REQUESTED - Robot resumed")
                self._just_resumed = True
                # Call resume callback (e.g., to reset policy action queue)
                if self.on_resume_callback:
                    try:
                        self.on_resume_callback()
                    except Exception as e:
                        print(f"[LeRobot] Resume callback error: {e}")

    def is_paused(self) -> bool:
        """Check if robot should be paused (from any source: uncertainty, RND, Aria, manual, etc.)."""
        with self.pause_lock:
            return self.pause_requested

    def check_and_clear_resumed(self) -> bool:
        """Check if we just resumed from pause and clear the flag.
        
        Returns:
            True if we just resumed (policy should be reset), False otherwise.
        """
        with self.pause_lock:
            if self._just_resumed:
                self._just_resumed = False
                return True
            return False

    def set_pause(self, paused: bool):
        """Set pause state directly (alternative to topic-based control).
        
        Args:
            paused: True to pause, False to resume
        """
        with self.pause_lock:
            was_paused = self.pause_requested
            self.pause_requested = paused
            
            if paused and not was_paused:
                print("[LeRobot] ⏸️  PAUSE SET - Robot paused")
            elif not paused and was_paused:
                print("[LeRobot] ▶️  RESUME SET - Robot resumed")
                self._just_resumed = True
                if self.on_resume_callback:
                    try:
                        self.on_resume_callback()
                    except Exception as e:
                        print(f"[LeRobot] Resume callback error: {e}")
    
    def wait_for_resume(self, timeout: float = None) -> bool:
        """
        Block until pause is released.
        
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

    # ========== Control Services ==========
    
    def _create_control_services(self):
        """Create ROS2 services for robot control."""
        services = []
        if self.on_start:
            self.node.create_service(Trigger, "/robot/start", self._handle_start)
            services.append("start")
        if self.on_pause:
            self.node.create_service(Trigger, "/robot/pause", self._handle_pause)
            services.append("pause")
        if self.on_resume:
            self.node.create_service(Trigger, "/robot/resume", self._handle_resume)
            services.append("resume")
        if self.on_reset:
            self.node.create_service(Trigger, "/robot/reset", self._handle_reset)
            services.append("reset")
        if self.on_stop:
            self.node.create_service(Trigger, "/robot/stop", self._handle_stop)
            services.append("stop")
        if self.on_switch_policy:
            self.node.create_service(Trigger, "/robot/switch_policy", self._handle_switch_policy)
            services.append("switch_policy")
        
        print(f"[LeRobot] Control services enabled: /robot/{{{','.join(services)}}}")
    
    def _handle_start(self, req, res):
        res.success, res.message = self.on_start()
        return res
    
    def _handle_pause(self, req, res):
        res.success, res.message = self.on_pause()
        return res
    
    def _handle_resume(self, req, res):
        res.success, res.message = self.on_resume()
        return res
    
    def _handle_reset(self, req, res):
        res.success, res.message = self.on_reset()
        return res
    
    def _handle_stop(self, req, res):
        res.success, res.message = self.on_stop()
        return res
    
    def _handle_switch_policy(self, req, res):
        res.success, res.message = self.on_switch_policy()
        return res
    
    # ========== Aria Gesture Callbacks ==========
    
    def _gaze_callback(self, msg):
        """Handle Aria gaze gesture detection."""
        gaze_direction = msg.data.lower()
        
        if gaze_direction == "left" and self.on_gaze_left:
            print("[LeRobot] 👁️ LEFT GAZE detected")
            try:
                self.on_gaze_left()
            except Exception as e:
                print(f"[LeRobot] Gaze left callback error: {e}")
                
        elif gaze_direction == "right" and self.on_gaze_right:
            print("[LeRobot] 👁️ RIGHT GAZE detected")
            try:
                self.on_gaze_right()
            except Exception as e:
                print(f"[LeRobot] Gaze right callback error: {e}")
    
    def _eyes_closed_callback(self, msg):
        """Handle Aria eyes closed detection."""
        if msg.data and self.on_eyes_closed:
            print("[LeRobot] 👁️ EYES CLOSED detected")
            try:
                self.on_eyes_closed()
            except Exception as e:
                print(f"[LeRobot] Eyes closed callback error: {e}")
    
    # ========== State Publisher ==========
    
    def _publish_state(self):
        """Publish robot state to /robot/state topic."""
        if self.get_state_string and hasattr(self, 'state_pub'):
            try:
                msg = String()
                msg.data = self.get_state_string()
                self.state_pub.publish(msg)
            except Exception as e:
                print(f"[LeRobot] State publish error: {e}")

    def publish_joint_states(self, observation: Dict[str, float]):
        """
        Publish joint states from robot observation

        Args:
            observation: Dictionary of joint_name -> position (e.g., {'shoulder_pan.pos': 45.0})
                        Values are normalized ±100 (RANGE_M100_100 mode) or 0-100 for gripper
                        
        Conversion:
            LeRobot RANGE_M100_100 mode:
                - -100 = calibration range_min position
                - +100 = calibration range_max position  
                - 0 = midpoint of calibration range
                
            We convert to URDF radians where 0 = physical center of joint.
            If calibration was symmetric, LeRobot 0 = URDF 0.
            If calibration was asymmetric, use joint_offsets to correct.
        """
        if not self.enabled or not self.joint_state_publisher:
            return

        try:
            msg = JointState()
            msg.header = Header()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = 'world'

            msg.name = []
            msg.position = []
            msg.velocity = []
            msg.effort = []

            for joint_name in self.joint_names:
                if joint_name in observation:
                    # Strip .pos suffix from joint names for ROS2 compatibility
                    clean_joint_name = joint_name.replace('.pos', '')
                    msg.name.append(clean_joint_name)
                    
                    # Get the raw value from observation (normalized ±100)
                    raw_value = float(observation[joint_name])

                    # Handle gripper differently if using prismatic gripper
                    if 'gripper' in joint_name.lower() and self.prismatic_gripper:
                        # Convert from 0-100 range to meters for prismatic gripper
                        # 0 = closed (max_meters), 100 = open (min_meters)
                        gripper_value = max(0, min(100, raw_value))
                        # Linear interpolation: 0->closed, 100->open
                        position = self.gripper_max_meters + (self.gripper_min_meters - self.gripper_max_meters) * (gripper_value / 100.0)
                        msg.position.append(position)
                    else:
                        # Convert normalized ±100 to physical degrees using URDF limits
                        position_deg = self._normalized_to_degrees(clean_joint_name, raw_value)
                        
                        # Apply calibration offset if configured (in degrees)
                        # Use this when calibration midpoint ≠ physical center
                        offset = self.joint_offsets.get(clean_joint_name, 0.0)
                        position_deg += offset
                        
                        # Convert degrees to radians for ROS2
                        position_rad = position_deg * (np.pi / 180.0)
                        msg.position.append(position_rad)
                    
                    msg.velocity.append(0.0)
                    msg.effort.append(0.0)

            if len(msg.name) > 0:
                self.joint_state_publisher.publish(msg)

        except Exception as e:
            print(f"[LeRobot] Failed to publish joint states: {e}")
    
    def _normalized_to_degrees(self, joint_name: str, normalized_value: float) -> float:
        """
        Convert normalized ±100 value to physical degrees based on URDF joint limits.
        
        LeRobot normalization (RANGE_M100_100):
            -100 → calibration range_min position (physical minimum during calibration)
            +100 → calibration range_max position (physical maximum during calibration)
            0 → midpoint of calibration range
            
        We assume calibration covered the full physical range, so:
            -100 → URDF lower limit
            +100 → URDF upper limit
            0 → URDF center (which is physical 0° for symmetric limits)
            
        For asymmetric limits (like elbow_flex: -100° to +90°):
            -100 → -100° physical
            +100 → +90° physical
            0 → -5° physical (midpoint)
            
        Args:
            joint_name: Joint name (e.g., 'left_shoulder_pan' or 'shoulder_pan')
            normalized_value: Value in range ±100
            
        Returns:
            Position in degrees relative to URDF limits
        """
        # Extract base joint name (remove left_/right_ prefix if present)
        base_joint = joint_name
        for prefix in ['left_', 'right_']:
            if joint_name.startswith(prefix):
                base_joint = joint_name[len(prefix):]
                break
        
        # Get joint limits from the table
        limits = self.SO101_JOINT_LIMITS.get(base_joint)
        
        if limits is None:
            # Unknown joint - treat ±100 as ±100 degrees (original fallback behavior)
            return normalized_value
        
        lower_deg, upper_deg = limits
        
        # Map normalized ±100 to URDF limits
        # normalized -100 → lower_deg
        # normalized +100 → upper_deg
        # Linear interpolation: position = lower + ((normalized + 100) / 200) * range
        range_deg = upper_deg - lower_deg
        position_deg = lower_deg + ((normalized_value + 100.0) / 200.0) * range_deg
        
        return position_deg

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

    def configure_control(self, 
                          on_start: Callable = None,
                          on_pause: Callable = None,
                          on_resume: Callable = None,
                          on_reset: Callable = None,
                          on_stop: Callable = None,
                          on_switch_policy: Callable = None,
                          on_gaze_left: Callable = None,
                          on_gaze_right: Callable = None,
                          on_eyes_closed: Callable = None,
                          get_state_string: Callable = None):
        """
        Configure control callbacks after initialization.
        
        Use this to add control services and Aria subscriptions after the bridge
        has been created (e.g., from multi_policy_client).
        
        Args:
            on_start: Callback for /robot/start service (returns tuple[bool, str])
            on_pause: Callback for /robot/pause service
            on_resume: Callback for /robot/resume service
            on_reset: Callback for /robot/reset service
            on_stop: Callback for /robot/stop service
            on_switch_policy: Callback for /robot/switch_policy service
            on_gaze_left: Callback for left gaze gesture
            on_gaze_right: Callback for right gaze gesture
            on_eyes_closed: Callback for eyes closed detection
            get_state_string: Callback that returns state string for /robot/state topic
        """
        if not self.enabled or not self.node:
            print("[LeRobot] Cannot configure control - ROS2 bridge not enabled")
            return
        
        # Store callbacks
        self.on_start = on_start
        self.on_pause = on_pause
        self.on_resume = on_resume
        self.on_reset = on_reset
        self.on_stop = on_stop
        self.on_switch_policy = on_switch_policy
        self.on_gaze_left = on_gaze_left
        self.on_gaze_right = on_gaze_right
        self.on_eyes_closed = on_eyes_closed
        self.get_state_string = get_state_string
        
        # Create control services
        if SERVICES_AVAILABLE and any([on_start, on_pause, on_resume, on_reset, on_stop, on_switch_policy]):
            self._create_control_services()
        
        # Create Aria gesture subscriptions
        if any([on_gaze_left, on_gaze_right]):
            if not hasattr(self, 'gaze_sub'):
                self.gaze_sub = self.node.create_subscription(
                    String, "/aria/gaze_gesture/detected", self._gaze_callback, 10
                )
                print("[LeRobot] Aria gaze gesture subscriber enabled")
        
        if on_eyes_closed:
            if not hasattr(self, 'eyes_closed_sub'):
                self.eyes_closed_sub = self.node.create_subscription(
                    Bool, "/aria/blink/eyes_closed_detected", self._eyes_closed_callback, 10
                )
                print("[LeRobot] Aria eyes closed subscriber enabled")
        
        # Create state publisher
        if get_state_string:
            if not hasattr(self, 'state_pub'):
                self.state_pub = self.node.create_publisher(String, "/robot/state", 10)
                self.node.create_timer(0.5, self._publish_state)
                print("[LeRobot] State publisher enabled on /robot/state")

    def shutdown(self):
        """Shutdown ROS2 bridge"""
        self.enabled = False
        if self.node:
            self.node.destroy_node()
        # Don't shutdown rclpy as other nodes may be using it
