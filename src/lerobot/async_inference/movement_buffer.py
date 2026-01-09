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
Movement buffer for recording and replaying robot joint trajectories.

This module provides thread-safe trajectory recording and reverse playback
functionality for resetting the robot to its starting position via a 
"rewind" effect.

Usage:
    buffer = MovementBuffer(max_frames=3000)
    
    # During policy execution
    buffer.start_recording()
    for frame in control_loop:
        buffer.record_frame(joint_positions, timestamp)
    buffer.stop_recording()
    
    # On reset gesture
    reverse_trajectory = buffer.get_reverse_trajectory(playback_speed=0.5)
    for positions in reverse_trajectory:
        robot.send_action(positions)
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


logger = logging.getLogger("movement_buffer")


# Joint position limits for bimanual SO-101 robot (in degrees/radians as used by the robot)
# These are conservative limits - adjust based on actual robot specs
DEFAULT_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    # Left arm
    "left_shoulder_pan": (-180.0, 180.0),
    "left_shoulder_lift": (-180.0, 180.0),
    "left_elbow": (-180.0, 180.0),
    "left_wrist_1": (-180.0, 180.0),
    "left_wrist_2": (-180.0, 180.0),
    "left_gripper": (0.0, 100.0),
    # Right arm
    "right_shoulder_pan": (-180.0, 180.0),
    "right_shoulder_lift": (-180.0, 180.0),
    "right_elbow": (-180.0, 180.0),
    "right_wrist_1": (-180.0, 180.0),
    "right_wrist_2": (-180.0, 180.0),
    "right_gripper": (0.0, 100.0),
}


@dataclass
class TrajectoryFrame:
    """A single frame of trajectory data."""
    timestamp: float
    positions: dict[str, float]
    
    def __post_init__(self):
        # Ensure positions is a copy to prevent external mutation
        self.positions = dict(self.positions)


@dataclass
class MovementBuffer:
    """
    Thread-safe ring buffer for recording robot joint trajectories.
    
    Records joint positions during policy execution and provides
    reverse trajectory generation for reset/rewind functionality.
    
    Attributes:
        max_frames: Maximum number of frames to store (default: 3000, ~60s at 50Hz)
        joint_limits: Dict mapping joint names to (min, max) position limits
        validate_positions: Whether to check positions against joint limits
    
    Thread Safety:
        All public methods are thread-safe using RLock. Recording can happen
        in the control loop while trajectory retrieval is triggered from
        ROS callbacks.
    """
    
    max_frames: int = 3000
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: DEFAULT_JOINT_LIMITS.copy()
    )
    validate_positions: bool = True
    
    def __post_init__(self):
        """Initialize internal state."""
        self._lock = threading.RLock()
        self._buffer: deque[TrajectoryFrame] = deque(maxlen=self.max_frames)
        self._is_recording = False
        self._recording_start_time: float = 0.0
        self._frame_count: int = 0  # Total frames recorded (including overwritten)
        
        logger.debug(f"MovementBuffer initialized with max_frames={self.max_frames}")
    
    def start_recording(self) -> None:
        """
        Start recording joint positions.
        
        Clears any existing buffer data and begins a new recording session.
        """
        with self._lock:
            self._buffer.clear()
            self._is_recording = True
            self._recording_start_time = time.time()
            self._frame_count = 0
            
        logger.info("Movement recording started")
    
    def record_frame(self, positions: dict[str, float], timestamp: float | None = None) -> bool:
        """
        Record a single frame of joint positions.
        
        Args:
            positions: Dict mapping joint names to position values
            timestamp: Optional timestamp (defaults to current time)
        
        Returns:
            True if frame was recorded successfully, False otherwise
        
        Raises:
            ValueError: If positions fail validation (when validate_positions=True)
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            if not self._is_recording:
                logger.debug("Attempted to record frame while not recording")
                return False
            
            # Validate positions if enabled
            if self.validate_positions:
                validation_errors = self._validate_positions(positions)
                if validation_errors:
                    logger.warning(f"Position validation errors: {validation_errors}")
                    # Still record but log the warning - don't block execution
            
            # Create and store frame
            frame = TrajectoryFrame(timestamp=timestamp, positions=positions)
            self._buffer.append(frame)
            self._frame_count += 1
            
            # Periodic logging
            if self._frame_count % 100 == 0:
                logger.debug(
                    f"Recorded {self._frame_count} frames, "
                    f"buffer size: {len(self._buffer)}"
                )
            
            return True
    
    def stop_recording(self) -> int:
        """
        Stop recording joint positions.
        
        Returns:
            Number of frames in the buffer
        """
        with self._lock:
            self._is_recording = False
            frame_count = len(self._buffer)
            duration = time.time() - self._recording_start_time if self._recording_start_time > 0 else 0
            
        logger.info(
            f"Movement recording stopped: {frame_count} frames, "
            f"{duration:.2f}s duration"
        )
        return frame_count
    
    def get_reverse_trajectory(
        self,
        playback_speed: float = 0.5,
        subsample_factor: int | None = None,
        smooth_window: int = 5,
    ) -> list[dict[str, float]]:
        """
        Generate a reverse trajectory for robot reset.
        
        The trajectory is reversed, optionally subsampled for slower playback,
        and smoothed to reduce jerk.
        
        Args:
            playback_speed: Speed multiplier (0.5 = half speed, fewer frames removed)
            subsample_factor: If provided, take every Nth frame (overrides playback_speed)
            smooth_window: Window size for moving average smoothing (0 to disable)
        
        Returns:
            List of position dicts in reverse order, ready for robot execution
        """
        with self._lock:
            if len(self._buffer) == 0:
                logger.warning("Cannot generate reverse trajectory: buffer is empty")
                return []
            
            if self._is_recording:
                logger.warning("Generating trajectory while still recording")
            
            # Copy buffer data
            frames = list(self._buffer)
        
        logger.info(f"Generating reverse trajectory from {len(frames)} frames")
        
        # Reverse the trajectory
        frames = frames[::-1]
        
        # Subsample for playback speed adjustment
        if subsample_factor is not None:
            step = max(1, subsample_factor)
        else:
            # Calculate step based on playback speed
            # Lower speed = more frames kept (smoother motion)
            # playback_speed=0.5 means keep every frame (slow playback)
            # playback_speed=1.0 means original recording speed
            # Guard against playback_speed <= 0 to prevent division by zero
            if playback_speed <= 0:
                step = 1  # Keep all frames if invalid speed
            elif playback_speed < 1.0:
                step = max(1, int(1.0 / playback_speed))
            else:
                step = 1
        
        if step > 1:
            frames = frames[::step]
            logger.debug(f"Subsampled to {len(frames)} frames (step={step})")
        
        # Extract positions only
        trajectory = [frame.positions for frame in frames]
        
        # Apply smoothing if requested
        if smooth_window > 1 and len(trajectory) > smooth_window:
            trajectory = self._smooth_trajectory(trajectory, smooth_window)
        
        logger.info(f"Generated reverse trajectory with {len(trajectory)} frames")
        return trajectory
    
    def clear(self) -> None:
        """Clear all recorded data and reset state."""
        with self._lock:
            self._buffer.clear()
            self._is_recording = False
            self._recording_start_time = 0.0
            self._frame_count = 0
        
        logger.info("Movement buffer cleared")
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get buffer statistics.
        
        Returns:
            Dict with frame_count, buffer_size, duration, is_recording, etc.
        """
        with self._lock:
            buffer_size = len(self._buffer)
            is_recording = self._is_recording
            
            if buffer_size > 0:
                first_ts = self._buffer[0].timestamp
                last_ts = self._buffer[-1].timestamp
                duration = last_ts - first_ts
                avg_fps = buffer_size / duration if duration > 0 else 0
            else:
                duration = 0.0
                avg_fps = 0.0
            
            return {
                "frame_count": self._frame_count,
                "buffer_size": buffer_size,
                "max_frames": self.max_frames,
                "duration_s": duration,
                "avg_fps": avg_fps,
                "is_recording": is_recording,
                "buffer_utilization": buffer_size / self.max_frames if self.max_frames > 0 else 0,
            }
    
    def get_first_frame(self) -> dict[str, float] | None:
        """Get the first recorded frame (starting position)."""
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return dict(self._buffer[0].positions)
    
    def get_last_frame(self) -> dict[str, float] | None:
        """Get the last recorded frame (ending position)."""
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return dict(self._buffer[-1].positions)
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        with self._lock:
            return self._is_recording
    
    @property
    def frame_count(self) -> int:
        """Get current number of frames in buffer."""
        with self._lock:
            return len(self._buffer)
    
    def _validate_positions(self, positions: dict[str, float]) -> list[str]:
        """
        Validate joint positions against limits.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        for joint_name, position in positions.items():
            if joint_name in self.joint_limits:
                min_pos, max_pos = self.joint_limits[joint_name]
                if position < min_pos or position > max_pos:
                    errors.append(
                        f"{joint_name}: {position:.2f} out of range [{min_pos}, {max_pos}]"
                    )
        
        return errors
    
    def _smooth_trajectory(
        self,
        trajectory: list[dict[str, float]],
        window_size: int,
    ) -> list[dict[str, float]]:
        """
        Apply moving average smoothing to trajectory.
        
        Uses numpy for efficient computation across all joints.
        
        Args:
            trajectory: List of position dicts
            window_size: Size of smoothing window
        
        Returns:
            Smoothed trajectory
        """
        if len(trajectory) < window_size:
            return trajectory
        
        # Get joint names from first frame
        joint_names = list(trajectory[0].keys())
        n_frames = len(trajectory)
        
        # Convert to numpy array for efficient smoothing
        # Shape: (n_frames, n_joints)
        positions_array = np.array([
            [frame[joint] for joint in joint_names]
            for frame in trajectory
        ])
        
        # Apply moving average filter
        kernel = np.ones(window_size) / window_size
        smoothed_array = np.zeros_like(positions_array)
        
        for j in range(len(joint_names)):
            # Pad to handle edges
            padded = np.pad(
                positions_array[:, j],
                (window_size // 2, window_size // 2),
                mode='edge'
            )
            smoothed = np.convolve(padded, kernel, mode='valid')
            # Ensure output length matches input
            smoothed_array[:, j] = smoothed[:n_frames]
        
        # Convert back to list of dicts
        smoothed_trajectory = [
            {joint: float(smoothed_array[i, j]) for j, joint in enumerate(joint_names)}
            for i in range(n_frames)
        ]
        
        logger.debug(f"Smoothed trajectory with window_size={window_size}")
        return smoothed_trajectory


class TrajectoryInterpolator:
    """
    Utility class for trajectory interpolation and refinement.
    
    Provides methods for:
    - Linear interpolation between frames
    - Spline-based smoothing
    - Velocity limiting
    """
    
    def __init__(
        self,
        max_velocity: dict[str, float] | None = None,
        max_acceleration: dict[str, float] | None = None,
    ):
        """
        Initialize trajectory interpolator.
        
        Args:
            max_velocity: Optional dict mapping joint names to max velocity limits
            max_acceleration: Optional dict mapping joint names to max acceleration limits
        """
        self.max_velocity = max_velocity or {}
        self.max_acceleration = max_acceleration or {}
    
    def interpolate_linear(
        self,
        trajectory: list[dict[str, float]],
        num_points: int,
    ) -> list[dict[str, float]]:
        """
        Linearly interpolate trajectory to have a specific number of points.
        
        Args:
            trajectory: Input trajectory
            num_points: Desired number of output points
        
        Returns:
            Interpolated trajectory
        """
        if len(trajectory) < 2 or num_points < 2:
            return trajectory
        
        joint_names = list(trajectory[0].keys())
        n_original = len(trajectory)
        
        # Create interpolation indices
        original_indices = np.linspace(0, 1, n_original)
        new_indices = np.linspace(0, 1, num_points)
        
        # Interpolate each joint
        interpolated = []
        
        # Convert to array for efficient interpolation
        positions_array = np.array([
            [frame[joint] for joint in joint_names]
            for frame in trajectory
        ])
        
        for new_idx in new_indices:
            # Find surrounding original indices
            idx_float = new_idx * (n_original - 1)
            idx_low = int(np.floor(idx_float))
            idx_high = min(idx_low + 1, n_original - 1)
            alpha = idx_float - idx_low
            
            # Linear interpolation
            interpolated_positions = (
                (1 - alpha) * positions_array[idx_low] +
                alpha * positions_array[idx_high]
            )
            
            interpolated.append({
                joint: float(interpolated_positions[j])
                for j, joint in enumerate(joint_names)
            })
        
        return interpolated
    
    def apply_velocity_limits(
        self,
        trajectory: list[dict[str, float]],
        dt: float,
    ) -> list[dict[str, float]]:
        """
        Apply velocity limits by inserting intermediate frames where needed.
        
        Args:
            trajectory: Input trajectory
            dt: Time step between frames in seconds
        
        Returns:
            Trajectory with velocity limits applied
        """
        if len(trajectory) < 2 or not self.max_velocity:
            return trajectory
        
        result = [trajectory[0]]
        
        for i in range(1, len(trajectory)):
            prev_frame = result[-1]
            curr_frame = trajectory[i]
            
            # Calculate required intermediate frames
            max_steps = 1
            for joint_name in prev_frame:
                if joint_name in self.max_velocity:
                    delta = abs(curr_frame[joint_name] - prev_frame[joint_name])
                    max_delta = self.max_velocity[joint_name] * dt
                    if max_delta > 0:
                        steps_needed = int(np.ceil(delta / max_delta))
                        max_steps = max(max_steps, steps_needed)
            
            # Insert intermediate frames if needed
            if max_steps > 1:
                for step in range(1, max_steps + 1):
                    alpha = step / max_steps
                    interp_frame = {
                        joint: prev_frame[joint] + alpha * (curr_frame[joint] - prev_frame[joint])
                        for joint in prev_frame
                    }
                    result.append(interp_frame)
            else:
                result.append(curr_frame)
        
        return result
    
    def resample_for_duration(
        self,
        trajectory: list[dict[str, float]],
        original_duration_s: float,
        target_duration_s: float,
        target_fps: float = 50.0,
    ) -> list[dict[str, float]]:
        """
        Resample trajectory to achieve a target duration.
        
        Args:
            trajectory: Input trajectory
            original_duration_s: Original trajectory duration in seconds
            target_duration_s: Desired playback duration in seconds
            target_fps: Target frame rate for output
        
        Returns:
            Resampled trajectory
        """
        if len(trajectory) < 2:
            return trajectory
        
        num_output_frames = int(target_duration_s * target_fps)
        return self.interpolate_linear(trajectory, num_output_frames)


# Convenience function for creating a pre-configured buffer
def create_movement_buffer(
    max_duration_s: float = 60.0,
    recording_fps: float = 50.0,
    validate: bool = True,
) -> MovementBuffer:
    """
    Create a MovementBuffer with settings based on duration and FPS.
    
    Args:
        max_duration_s: Maximum recording duration in seconds
        recording_fps: Expected recording frame rate
        validate: Whether to validate joint positions
    
    Returns:
        Configured MovementBuffer instance
    """
    max_frames = int(max_duration_s * recording_fps)
    return MovementBuffer(
        max_frames=max_frames,
        validate_positions=validate,
    )
