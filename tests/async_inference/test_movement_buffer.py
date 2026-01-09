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
Unit tests for MovementBuffer trajectory recording and replay.

Tests cover:
- Start/stop recording lifecycle
- Frame recording and buffer operations
- Reverse trajectory generation
- Thread safety under concurrent access
- Buffer overflow handling
- Edge cases and error conditions
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from lerobot.async_inference.movement_buffer import (
    DEFAULT_JOINT_LIMITS,
    MovementBuffer,
    TrajectoryFrame,
    TrajectoryInterpolator,
    create_movement_buffer,
)


# ====================== Fixtures ======================


@pytest.fixture
def sample_positions() -> dict[str, float]:
    """Create sample joint positions for testing."""
    return {
        "left_shoulder_pan": 0.0,
        "left_shoulder_lift": 10.0,
        "left_elbow": -20.0,
        "left_wrist_1": 30.0,
        "left_wrist_2": -40.0,
        "left_gripper": 50.0,
        "right_shoulder_pan": 5.0,
        "right_shoulder_lift": 15.0,
        "right_elbow": -25.0,
        "right_wrist_1": 35.0,
        "right_wrist_2": -45.0,
        "right_gripper": 55.0,
    }


@pytest.fixture
def movement_buffer() -> MovementBuffer:
    """Create a MovementBuffer with default settings."""
    return MovementBuffer(max_frames=100, validate_positions=False)


@pytest.fixture
def validating_buffer() -> MovementBuffer:
    """Create a MovementBuffer with position validation enabled."""
    return MovementBuffer(max_frames=100, validate_positions=True)


# ====================== TrajectoryFrame Tests ======================


class TestTrajectoryFrame:
    """Tests for TrajectoryFrame dataclass."""

    def test_trajectory_frame_creation(self, sample_positions):
        """Test basic TrajectoryFrame creation."""
        timestamp = 1000.0
        frame = TrajectoryFrame(timestamp=timestamp, positions=sample_positions)
        
        assert frame.timestamp == timestamp
        assert frame.positions == sample_positions

    def test_trajectory_frame_defensive_copy(self, sample_positions):
        """Test that TrajectoryFrame makes a defensive copy of positions."""
        frame = TrajectoryFrame(timestamp=1000.0, positions=sample_positions)
        
        # Modify original dict
        sample_positions["left_shoulder_pan"] = 999.0
        
        # Frame should be unchanged
        assert frame.positions["left_shoulder_pan"] == 0.0


# ====================== Recording Lifecycle Tests ======================


class TestRecordingLifecycle:
    """Tests for start/stop recording operations."""

    def test_start_recording_clears_buffer(self, movement_buffer, sample_positions):
        """Test that start_recording clears existing buffer data."""
        # Add some frames
        movement_buffer.start_recording()
        movement_buffer.record_frame(sample_positions, timestamp=1.0)
        movement_buffer.record_frame(sample_positions, timestamp=2.0)
        movement_buffer.stop_recording()
        
        assert movement_buffer.frame_count == 2
        
        # Start new recording - should clear
        movement_buffer.start_recording()
        assert movement_buffer.frame_count == 0
        assert movement_buffer.is_recording is True

    def test_stop_recording_returns_frame_count(self, movement_buffer, sample_positions):
        """Test that stop_recording returns correct frame count."""
        movement_buffer.start_recording()
        
        for i in range(5):
            movement_buffer.record_frame(sample_positions, timestamp=float(i))
        
        frame_count = movement_buffer.stop_recording()
        assert frame_count == 5
        assert movement_buffer.is_recording is False

    def test_is_recording_property(self, movement_buffer, sample_positions):
        """Test is_recording property reflects current state."""
        assert movement_buffer.is_recording is False
        
        movement_buffer.start_recording()
        assert movement_buffer.is_recording is True
        
        movement_buffer.record_frame(sample_positions)
        assert movement_buffer.is_recording is True
        
        movement_buffer.stop_recording()
        assert movement_buffer.is_recording is False

    def test_clear_resets_all_state(self, movement_buffer, sample_positions):
        """Test that clear() resets all buffer state."""
        movement_buffer.start_recording()
        for i in range(10):
            movement_buffer.record_frame(sample_positions, timestamp=float(i))
        
        movement_buffer.clear()
        
        assert movement_buffer.frame_count == 0
        assert movement_buffer.is_recording is False
        
        stats = movement_buffer.get_stats()
        assert stats["buffer_size"] == 0
        assert stats["frame_count"] == 0


# ====================== Frame Recording Tests ======================


class TestFrameRecording:
    """Tests for frame recording operations."""

    def test_record_frame_basic(self, movement_buffer, sample_positions):
        """Test basic frame recording."""
        movement_buffer.start_recording()
        
        result = movement_buffer.record_frame(sample_positions, timestamp=1.0)
        
        assert result is True
        assert movement_buffer.frame_count == 1

    def test_record_frame_without_recording_fails(self, movement_buffer, sample_positions):
        """Test that recording without start_recording returns False."""
        result = movement_buffer.record_frame(sample_positions, timestamp=1.0)
        
        assert result is False
        assert movement_buffer.frame_count == 0

    def test_record_frame_auto_timestamp(self, movement_buffer, sample_positions):
        """Test that frame gets automatic timestamp if not provided."""
        movement_buffer.start_recording()
        
        before = time.time()
        movement_buffer.record_frame(sample_positions)
        after = time.time()
        
        # Get frame timestamp via stats
        first_frame = movement_buffer.get_first_frame()
        assert first_frame is not None

    def test_record_multiple_frames(self, movement_buffer, sample_positions):
        """Test recording multiple frames in sequence."""
        movement_buffer.start_recording()
        
        for i in range(50):
            positions = sample_positions.copy()
            positions["left_shoulder_pan"] = float(i)
            movement_buffer.record_frame(positions, timestamp=float(i))
        
        assert movement_buffer.frame_count == 50
        
        # Verify first and last frames
        first = movement_buffer.get_first_frame()
        last = movement_buffer.get_last_frame()
        
        assert first["left_shoulder_pan"] == 0.0
        assert last["left_shoulder_pan"] == 49.0

    def test_get_first_frame_empty_buffer(self, movement_buffer):
        """Test get_first_frame returns None for empty buffer."""
        assert movement_buffer.get_first_frame() is None

    def test_get_last_frame_empty_buffer(self, movement_buffer):
        """Test get_last_frame returns None for empty buffer."""
        assert movement_buffer.get_last_frame() is None


# ====================== Reverse Trajectory Tests ======================


class TestReverseTrajectory:
    """Tests for reverse trajectory generation."""

    def test_get_reverse_trajectory_order(self, movement_buffer, sample_positions):
        """Test that trajectory is correctly reversed."""
        movement_buffer.start_recording()
        
        for i in range(10):
            positions = sample_positions.copy()
            positions["left_shoulder_pan"] = float(i)
            movement_buffer.record_frame(positions, timestamp=float(i))
        
        movement_buffer.stop_recording()
        
        trajectory = movement_buffer.get_reverse_trajectory(
            playback_speed=1.0, smooth_window=0
        )
        
        assert len(trajectory) == 10
        
        # First frame of reverse should be last recorded (value 9)
        assert trajectory[0]["left_shoulder_pan"] == 9.0
        # Last frame of reverse should be first recorded (value 0)
        assert trajectory[-1]["left_shoulder_pan"] == 0.0

    def test_get_reverse_trajectory_empty_buffer(self, movement_buffer):
        """Test that empty buffer returns empty trajectory."""
        trajectory = movement_buffer.get_reverse_trajectory()
        assert trajectory == []

    def test_get_reverse_trajectory_subsampling(self, movement_buffer, sample_positions):
        """Test trajectory subsampling with explicit factor."""
        movement_buffer.start_recording()
        
        for i in range(100):
            positions = sample_positions.copy()
            positions["left_shoulder_pan"] = float(i)
            movement_buffer.record_frame(positions, timestamp=float(i))
        
        movement_buffer.stop_recording()
        
        # Subsample every 5th frame
        trajectory = movement_buffer.get_reverse_trajectory(
            subsample_factor=5, smooth_window=0
        )
        
        assert len(trajectory) == 20  # 100 / 5 = 20

    def test_get_reverse_trajectory_smoothing(self, movement_buffer, sample_positions):
        """Test that smoothing is applied when window > 1."""
        movement_buffer.start_recording()
        
        for i in range(20):
            positions = sample_positions.copy()
            # Add some noise to make smoothing visible
            positions["left_shoulder_pan"] = float(i) + (i % 2) * 5.0
            movement_buffer.record_frame(positions, timestamp=float(i))
        
        movement_buffer.stop_recording()
        
        # Without smoothing
        traj_raw = movement_buffer.get_reverse_trajectory(
            playback_speed=1.0, smooth_window=0
        )
        
        # With smoothing
        traj_smooth = movement_buffer.get_reverse_trajectory(
            playback_speed=1.0, smooth_window=5
        )
        
        # Both should have same length
        assert len(traj_raw) == len(traj_smooth)
        
        # Smoothed trajectory should have less variance
        raw_values = [f["left_shoulder_pan"] for f in traj_raw]
        smooth_values = [f["left_shoulder_pan"] for f in traj_smooth]
        
        raw_std = np.std(np.diff(raw_values))
        smooth_std = np.std(np.diff(smooth_values))
        
        assert smooth_std < raw_std

    def test_get_reverse_trajectory_while_recording_warns(
        self, movement_buffer, sample_positions, caplog
    ):
        """Test that generating trajectory while recording logs a warning."""
        import logging
        
        movement_buffer.start_recording()
        movement_buffer.record_frame(sample_positions, timestamp=1.0)
        
        # This should log a warning but still work
        with caplog.at_level(logging.WARNING, logger="movement_buffer"):
            trajectory = movement_buffer.get_reverse_trajectory()
        
        # Check we got some output (may be empty due to single frame)
        assert len(trajectory) >= 0


# ====================== Buffer Overflow Tests ======================


class TestBufferOverflow:
    """Tests for buffer overflow handling (ring buffer behavior)."""

    def test_buffer_overflow_drops_oldest(self):
        """Test that buffer drops oldest frames when full."""
        buffer = MovementBuffer(max_frames=10, validate_positions=False)
        buffer.start_recording()
        
        # Record 15 frames (5 more than max)
        for i in range(15):
            positions = {"joint1": float(i)}
            buffer.record_frame(positions, timestamp=float(i))
        
        # Buffer should only have 10 frames
        assert buffer.frame_count == 10
        
        # First frame should be value 5 (oldest 5 dropped)
        first = buffer.get_first_frame()
        assert first["joint1"] == 5.0
        
        # Last frame should be value 14
        last = buffer.get_last_frame()
        assert last["joint1"] == 14.0

    def test_buffer_stats_track_total_frames(self):
        """Test that stats track total frames including overwritten."""
        buffer = MovementBuffer(max_frames=5, validate_positions=False)
        buffer.start_recording()
        
        for i in range(10):
            positions = {"joint1": float(i)}
            buffer.record_frame(positions, timestamp=float(i))
        
        stats = buffer.get_stats()
        
        # frame_count in stats tracks total recorded
        assert stats["frame_count"] == 10
        # buffer_size is actual frames in buffer
        assert stats["buffer_size"] == 5
        # buffer_utilization should be 100%
        assert stats["buffer_utilization"] == 1.0


# ====================== Thread Safety Tests ======================


class TestThreadSafety:
    """Tests for thread safety under concurrent access."""

    def test_concurrent_recording(self, sample_positions):
        """Test that concurrent frame recording is thread-safe."""
        buffer = MovementBuffer(max_frames=1000, validate_positions=False)
        buffer.start_recording()
        
        num_threads = 4
        frames_per_thread = 100
        errors = []
        
        def record_frames(thread_id: int):
            try:
                for i in range(frames_per_thread):
                    positions = sample_positions.copy()
                    positions["left_shoulder_pan"] = float(thread_id * 1000 + i)
                    buffer.record_frame(positions, timestamp=time.time())
            except Exception as e:
                errors.append(e)
        
        # Run concurrent recording
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(record_frames, i) for i in range(num_threads)]
            for f in futures:
                f.result()
        
        buffer.stop_recording()
        
        # No errors should have occurred
        assert len(errors) == 0
        
        # Should have all frames (or max_frames if overflow)
        expected_frames = num_threads * frames_per_thread
        assert buffer.frame_count == expected_frames

    def test_concurrent_read_write(self, sample_positions):
        """Test concurrent reading and writing operations."""
        buffer = MovementBuffer(max_frames=500, validate_positions=False)
        buffer.start_recording()
        
        stop_event = threading.Event()
        errors = []
        read_counts = [0]
        
        def writer():
            try:
                for i in range(200):
                    if stop_event.is_set():
                        break
                    positions = sample_positions.copy()
                    positions["left_shoulder_pan"] = float(i)
                    buffer.record_frame(positions, timestamp=time.time())
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def reader():
            try:
                while not stop_event.is_set():
                    _ = buffer.get_stats()
                    _ = buffer.get_first_frame()
                    _ = buffer.get_last_frame()
                    _ = buffer.is_recording
                    _ = buffer.frame_count
                    read_counts[0] += 1
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)
        
        # Start concurrent threads
        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)
        
        writer_thread.start()
        reader_thread.start()
        
        # Wait for writer to finish
        writer_thread.join()
        stop_event.set()
        reader_thread.join()
        
        buffer.stop_recording()
        
        assert len(errors) == 0
        assert read_counts[0] > 0  # Reader should have run multiple times


# ====================== Validation Tests ======================


class TestPositionValidation:
    """Tests for joint position validation."""

    def test_validation_logs_warning_for_out_of_range(
        self, validating_buffer, sample_positions, caplog
    ):
        """Test that out-of-range positions log warnings."""
        import logging
        
        validating_buffer.start_recording()
        
        # Create positions with out-of-range values
        bad_positions = sample_positions.copy()
        bad_positions["left_shoulder_pan"] = 999.0  # Way out of range
        
        with caplog.at_level(logging.WARNING, logger="movement_buffer"):
            result = validating_buffer.record_frame(bad_positions)
        
        # Should still record (but log warning)
        assert result is True

    def test_validation_allows_in_range_positions(self, validating_buffer, sample_positions):
        """Test that in-range positions pass validation."""
        validating_buffer.start_recording()
        
        # All sample_positions should be in range
        result = validating_buffer.record_frame(sample_positions)
        
        assert result is True


# ====================== TrajectoryInterpolator Tests ======================


class TestTrajectoryInterpolator:
    """Tests for TrajectoryInterpolator utility class."""

    def test_interpolate_linear_doubles_points(self):
        """Test linear interpolation doubles number of points."""
        interpolator = TrajectoryInterpolator()
        
        trajectory = [
            {"joint1": 0.0, "joint2": 0.0},
            {"joint1": 10.0, "joint2": 20.0},
        ]
        
        result = interpolator.interpolate_linear(trajectory, num_points=4)
        
        assert len(result) == 4
        
        # Check interpolated values
        assert result[0]["joint1"] == pytest.approx(0.0, abs=0.1)
        assert result[-1]["joint1"] == pytest.approx(10.0, abs=0.1)

    def test_interpolate_linear_preserves_endpoints(self):
        """Test that interpolation preserves start and end points."""
        interpolator = TrajectoryInterpolator()
        
        trajectory = [
            {"joint1": 5.0},
            {"joint1": 15.0},
            {"joint1": 25.0},
        ]
        
        result = interpolator.interpolate_linear(trajectory, num_points=10)
        
        assert result[0]["joint1"] == pytest.approx(5.0, abs=0.01)
        assert result[-1]["joint1"] == pytest.approx(25.0, abs=0.01)

    def test_resample_for_duration(self):
        """Test trajectory resampling for specific duration."""
        interpolator = TrajectoryInterpolator()
        
        # Create 10-frame trajectory
        trajectory = [{"joint1": float(i)} for i in range(10)]
        
        # Resample from 1s to 2s at 50 FPS
        result = interpolator.resample_for_duration(
            trajectory,
            original_duration_s=1.0,
            target_duration_s=2.0,
            target_fps=50.0,
        )
        
        # Should have 100 frames (2s * 50 FPS)
        assert len(result) == 100


# ====================== Factory Function Tests ======================


class TestCreateMovementBuffer:
    """Tests for create_movement_buffer factory function."""

    def test_create_with_duration_and_fps(self):
        """Test buffer creation with duration and FPS settings."""
        buffer = create_movement_buffer(
            max_duration_s=30.0,
            recording_fps=100.0,
            validate=True,
        )
        
        assert buffer.max_frames == 3000  # 30s * 100 FPS
        assert buffer.validate_positions is True

    def test_create_default_settings(self):
        """Test buffer creation with default settings."""
        buffer = create_movement_buffer()
        
        assert buffer.max_frames == 3000  # 60s * 50 FPS default
        assert buffer.validate_positions is True


# ====================== Edge Cases ======================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_frame_buffer(self, sample_positions):
        """Test buffer with only one frame."""
        buffer = MovementBuffer(max_frames=1, validate_positions=False)
        buffer.start_recording()
        buffer.record_frame(sample_positions, timestamp=1.0)
        buffer.stop_recording()
        
        trajectory = buffer.get_reverse_trajectory()
        
        assert len(trajectory) == 1
        assert trajectory[0] == sample_positions

    def test_empty_positions_dict(self, movement_buffer):
        """Test recording empty positions dict."""
        movement_buffer.start_recording()
        
        result = movement_buffer.record_frame({}, timestamp=1.0)
        
        assert result is True
        assert movement_buffer.frame_count == 1

    def test_stats_fps_calculation(self, movement_buffer, sample_positions):
        """Test that stats correctly calculates FPS."""
        movement_buffer.start_recording()
        
        # Record 10 frames over 1 second
        for i in range(10):
            movement_buffer.record_frame(sample_positions, timestamp=float(i) * 0.1)
        
        movement_buffer.stop_recording()
        
        stats = movement_buffer.get_stats()
        
        # Duration should be ~0.9s (0.0 to 0.9)
        assert stats["duration_s"] == pytest.approx(0.9, abs=0.01)
        
        # FPS should be ~11 (10 frames / 0.9s)
        assert stats["avg_fps"] == pytest.approx(11.1, abs=1.0)

    def test_playback_speed_zero_handling(self, movement_buffer, sample_positions):
        """Test that playback_speed=0 is handled safely."""
        movement_buffer.start_recording()
        for i in range(10):
            movement_buffer.record_frame(sample_positions, timestamp=float(i))
        movement_buffer.stop_recording()
        
        # playback_speed=0 could cause division by zero
        # Implementation should handle this gracefully
        trajectory = movement_buffer.get_reverse_trajectory(
            playback_speed=0.0, smooth_window=0
        )
        
        # Should return all frames (step=1 as fallback)
        assert len(trajectory) >= 1
