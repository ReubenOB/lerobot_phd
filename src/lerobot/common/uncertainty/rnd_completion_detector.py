#!/usr/bin/env python3
"""
Real-time RND-based Task Completion Detector

Detects when a task is complete by monitoring the gradient (rate of change)
of RND uncertainty. When the gradient stays near zero for a sustained period,
the task is considered complete.

Usage:
    detector = RNDCompletionDetector(
        rnd_module=rnd,
        gradient_window=30,      # 1 second at 30fps
        stability_window=45,     # 1.5 seconds to confirm
        min_frames=150,          # 5 seconds minimum before detection
    )
    
    # In your control loop:
    is_done, confidence = detector.update(image, state, action)
    if is_done:
        print("Task complete!")
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional
import torch


class RNDCompletionDetector:
    """
    Real-time task completion detector using RND uncertainty gradient.
    
    The key insight is that when a task is complete:
    1. The robot stops moving → actions become stable
    2. The scene stabilizes → uncertainty stops changing
    3. The gradient of uncertainty approaches zero
    
    We detect completion when the smoothed gradient stays below a threshold
    for a sustained number of frames.
    """
    
    def __init__(
        self,
        rnd_module,
        gradient_window: int = 30,      # Window for smoothing gradient (1 sec @ 30fps)
        stability_window: int = 45,      # Consecutive low-gradient frames needed (1.5 sec)
        min_frames: int = 150,           # Minimum frames before detection allowed (5 sec)
        gradient_percentile: float = 25, # Threshold = this percentile of observed gradients
        warmup_frames: int = 60,         # Frames to collect before computing threshold
    ):
        """
        Args:
            rnd_module: Trained RND module for computing uncertainty
            gradient_window: Size of window for smoothing gradient
            stability_window: Number of consecutive low-gradient frames to trigger completion
            min_frames: Minimum frames into task before completion can be detected
            gradient_percentile: Percentile of gradient values to use as threshold
            warmup_frames: Number of frames to collect for threshold calibration
        """
        self.rnd = rnd_module
        self.gradient_window = gradient_window
        self.stability_window = stability_window
        self.min_frames = min_frames
        self.gradient_percentile = gradient_percentile
        self.warmup_frames = warmup_frames
        
        # Buffers
        self.uncertainty_buffer = deque(maxlen=gradient_window + 10)
        self.gradient_buffer = deque(maxlen=stability_window + 10)
        self.all_gradients = []  # For computing adaptive threshold
        
        # State
        self.frame_count = 0
        self.consecutive_stable = 0
        self.is_complete = False
        self.completion_frame = None
        self.gradient_threshold = None
        
        # Running statistics
        self.smooth_gradient = 0.0
        self.current_uncertainty = 0.0
        
    def reset(self):
        """Reset detector for a new episode."""
        self.uncertainty_buffer.clear()
        self.gradient_buffer.clear()
        self.all_gradients.clear()
        self.frame_count = 0
        self.consecutive_stable = 0
        self.is_complete = False
        self.completion_frame = None
        self.gradient_threshold = None
        self.smooth_gradient = 0.0
        self.current_uncertainty = 0.0
        
        # Also reset RND rolling buffer if available
        if hasattr(self.rnd, 'reset_rolling_buffer'):
            self.rnd.reset_rolling_buffer()
    
    def update(
        self,
        image: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, float]:
        """
        Update detector with new observation.
        
        Args:
            image: Current camera image [C, H, W] or [1, C, H, W]
            state: Current robot state (optional)
            action: Current/predicted action (optional)
            
        Returns:
            Tuple of (is_complete, confidence)
            - is_complete: True if task completion detected
            - confidence: Confidence of completion (0-1), based on stability
        """
        self.frame_count += 1
        
        # Already detected - return cached result
        if self.is_complete:
            return True, 1.0
        
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if state is not None and state.dim() == 1:
            state = state.unsqueeze(0)
        if action is not None and action.dim() == 1:
            action = action.unsqueeze(0)
        
        # Compute uncertainty
        with torch.no_grad():
            step_unc, rolling_unc = self.rnd.compute_uncertainty(
                image, state, action, normalize=True
            )
        
        self.current_uncertainty = rolling_unc
        self.uncertainty_buffer.append(rolling_unc)
        
        # Need enough history for gradient
        if len(self.uncertainty_buffer) < 3:
            return False, 0.0
        
        # Compute instantaneous gradient
        recent = list(self.uncertainty_buffer)
        instant_gradient = abs(recent[-1] - recent[-2])
        
        # Smooth gradient over window
        self.gradient_buffer.append(instant_gradient)
        self.all_gradients.append(instant_gradient)
        
        if len(self.gradient_buffer) >= self.gradient_window:
            self.smooth_gradient = np.mean(list(self.gradient_buffer)[-self.gradient_window:])
        else:
            self.smooth_gradient = np.mean(list(self.gradient_buffer))
        
        # Compute adaptive threshold during warmup
        if self.frame_count == self.warmup_frames:
            self.gradient_threshold = np.percentile(
                self.all_gradients, self.gradient_percentile
            )
        
        # Can't detect before min_frames or threshold computed
        if self.frame_count < self.min_frames or self.gradient_threshold is None:
            return False, 0.0
        
        # Check if gradient is below threshold
        if self.smooth_gradient < self.gradient_threshold:
            self.consecutive_stable += 1
        else:
            self.consecutive_stable = 0
        
        # Compute confidence (0-1 based on how close to triggering)
        confidence = min(1.0, self.consecutive_stable / self.stability_window)
        
        # Check if stability threshold met
        if self.consecutive_stable >= self.stability_window:
            self.is_complete = True
            self.completion_frame = self.frame_count - self.stability_window
            return True, 1.0
        
        return False, confidence
    
    def get_status(self) -> dict:
        """Get current detector status for debugging/visualization."""
        return {
            'frame_count': self.frame_count,
            'is_complete': self.is_complete,
            'completion_frame': self.completion_frame,
            'current_uncertainty': self.current_uncertainty,
            'smooth_gradient': self.smooth_gradient,
            'gradient_threshold': self.gradient_threshold,
            'consecutive_stable': self.consecutive_stable,
            'stability_progress': self.consecutive_stable / self.stability_window if self.stability_window > 0 else 0,
        }
    
    def get_progress(self) -> float:
        """
        Get estimated task progress (0-1).
        
        This is a heuristic based on:
        - Time into task
        - Stability (approaching completion)
        
        Returns value from 0 (just started) to 1 (complete).
        """
        if self.is_complete:
            return 1.0
        
        if self.frame_count < self.min_frames:
            # Early in task - use time-based estimate
            return self.frame_count / self.min_frames * 0.5  # Max 50% during warmup
        
        # After min_frames, add stability contribution
        time_progress = min(0.7, self.frame_count / 700)  # Assume ~700 frames typical
        stability_progress = (self.consecutive_stable / self.stability_window) * 0.3
        
        return min(0.99, time_progress + stability_progress)


class RNDCompletionDetectorROS:
    """
    ROS2 wrapper for RND Completion Detector.
    
    Subscribes to camera topics and publishes completion status.
    Can be used as a drop-in replacement for SARM progress tracking.
    """
    
    def __init__(
        self,
        rnd_path: str,
        policy_path: str = None,
        camera_topic: str = "/camera/top/image_raw",
        device: str = "cuda",
        **detector_kwargs
    ):
        """
        Initialize ROS2 node for completion detection.
        
        Args:
            rnd_path: Path to trained RND model
            policy_path: Path to policy (for policy-specific RND)
            camera_topic: ROS topic for camera images
            device: Device to run inference on
            **detector_kwargs: Additional args for RNDCompletionDetector
        """
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image
            from std_msgs.msg import Float32, Bool
            from cv_bridge import CvBridge
        except ImportError:
            raise ImportError("ROS2 packages not available. Install rclpy, sensor_msgs, cv_bridge.")
        
        self.device = device
        self.bridge = CvBridge()
        
        # Load RND model
        from lerobot.scripts.eval_rnd import load_rnd_model
        from pathlib import Path
        
        self.rnd, _ = load_rnd_model(Path(rnd_path), policy_path, device)
        self.detector = RNDCompletionDetector(self.rnd, **detector_kwargs)
        
        # Image transforms
        from torchvision.transforms import v2 as transforms
        self.transforms = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((96, 96), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        
        # ROS2 setup would go here
        # This is a template - actual implementation depends on your ROS2 setup
        
    def image_callback(self, msg):
        """Process incoming camera image."""
        # Convert ROS image to tensor
        cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        image = self.transforms(cv_image).to(self.device)
        
        # Update detector
        is_complete, confidence = self.detector.update(image)
        
        # Publish results
        # self.complete_pub.publish(Bool(data=is_complete))
        # self.confidence_pub.publish(Float32(data=confidence))
        # self.progress_pub.publish(Float32(data=self.detector.get_progress()))
        
        return is_complete, confidence
    
    def reset(self):
        """Reset for new episode."""
        self.detector.reset()


# Convenience function for quick testing
def create_detector_from_checkpoint(
    rnd_path: str,
    policy_path: str = None,
    device: str = "cuda",
    **kwargs
) -> RNDCompletionDetector:
    """
    Create a completion detector from a saved RND checkpoint.
    
    Args:
        rnd_path: Path to RND model directory or file
        policy_path: Path to policy (required for policy-specific RND)
        device: Device to run on
        **kwargs: Additional arguments for RNDCompletionDetector
        
    Returns:
        Configured RNDCompletionDetector ready for use
    """
    from lerobot.scripts.eval_rnd import load_rnd_model
    from pathlib import Path
    
    rnd, _ = load_rnd_model(Path(rnd_path), policy_path, device)
    return RNDCompletionDetector(rnd, **kwargs)
