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
Enhanced data recording with multi-policy metadata.

This module provides MultiPolicyDataRecorder which extends LeRobotDataset
to capture rich metadata for multi-policy episodes including:
- Active policy per frame
- SARM progress and stage
- RND uncertainty
- Gaze selection state
- FSM state
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.constants import ACTION, OBS_STR

from .configs_multi import DatasetConfig, MultiPolicyConfig


logger = logging.getLogger("data_recorder")


@dataclass
class RecordingMetadata:
    """Metadata captured per frame for multi-policy recording."""
    
    # Timing
    timestamp: float = 0.0
    frame_index: int = 0
    
    # Policy state
    active_policy: str = ""
    fsm_state: str = ""
    
    # SARM progress
    sarm_progress: float = 0.0
    sarm_stage: int = 0
    sarm_stage_name: str = ""
    
    # RND uncertainty
    rnd_uncertainty: float = 0.0
    rnd_paused: bool = False
    
    # Gaze selection
    selected_pod: str = ""
    selected_cup: str = ""
    gaze_yaw: float = 0.0
    gaze_pitch: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "active_policy": self.active_policy,
            "fsm_state": self.fsm_state,
            "sarm_progress": self.sarm_progress,
            "sarm_stage": self.sarm_stage,
            "sarm_stage_name": self.sarm_stage_name,
            "rnd_uncertainty": self.rnd_uncertainty,
            "rnd_paused": self.rnd_paused,
            "selected_pod": self.selected_pod,
            "selected_cup": self.selected_cup,
            "gaze_yaw": self.gaze_yaw,
            "gaze_pitch": self.gaze_pitch,
        }


@dataclass
class EpisodeMetadata:
    """Metadata for a complete episode."""
    
    episode_index: int = 0
    task: str = ""
    selected_pod: str = ""
    selected_cup: str = ""
    success: bool = False
    start_time: float = 0.0
    end_time: float = 0.0
    num_frames: int = 0
    policies_used: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "episode_index": self.episode_index,
            "task": self.task,
            "selected_pod": self.selected_pod,
            "selected_cup": self.selected_cup,
            "success": self.success,
            "duration_s": self.end_time - self.start_time,
            "num_frames": self.num_frames,
            "policies_used": self.policies_used,
        }


class MultiPolicyDataRecorder:
    """
    Data recorder for multi-policy orchestration with rich metadata.
    
    Wraps LeRobotDataset to add per-frame metadata tracking for
    multi-policy episodes.
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        robot_features: dict[str, Any],
        robot_type: str = "so101",
    ):
        """
        Initialize the data recorder.
        
        Args:
            config: Dataset configuration
            robot_features: Robot observation/action features from robot.observation_features
            robot_type: Type of robot for dataset metadata
        """
        self.config = config
        self.robot_type = robot_type
        
        # Build dataset features from robot features
        obs_features = hw_to_dataset_features(robot_features, OBS_STR, use_video=config.use_videos)
        action_features = hw_to_dataset_features(robot_features, ACTION)
        self.dataset_features = {**obs_features, **action_features}
        
        # Dataset (created lazily)
        self.dataset: LeRobotDataset | None = None
        
        # Recording state
        self.is_recording = False
        self.frame_count = 0
        self.episode_metadata: EpisodeMetadata | None = None
        self.frame_metadata_buffer: list[RecordingMetadata] = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Current metadata state
        self.current_metadata = RecordingMetadata()
        
        logger.info(f"DataRecorder initialized for {config.repo_id}")
    
    def create_dataset(self) -> LeRobotDataset:
        """Create or get the LeRobotDataset."""
        if self.dataset is None:
            self.dataset = LeRobotDataset.create(
                repo_id=self.config.repo_id,
                fps=self.config.fps,
                features=self.dataset_features,
                robot_type=self.robot_type,
                use_videos=self.config.use_videos,
                image_writer_threads=self.config.image_writer_threads,
            )
            logger.info(f"Created dataset: {self.config.repo_id}")
        return self.dataset
    
    def start_episode(
        self,
        task: str,
        selected_pod: str = "",
        selected_cup: str = "",
    ):
        """
        Start recording a new episode.
        
        Args:
            task: Task description string
            selected_pod: Selected pod for this episode
            selected_cup: Selected cup for this episode
        """
        with self.lock:
            # Ensure dataset exists
            self.create_dataset()
            
            # Initialize episode metadata
            self.episode_metadata = EpisodeMetadata(
                episode_index=len(self.dataset.episodes) if self.dataset else 0,
                task=task,
                selected_pod=selected_pod,
                selected_cup=selected_cup,
                start_time=time.time(),
            )
            
            # Reset state
            self.frame_count = 0
            self.frame_metadata_buffer = []
            self.is_recording = True
            
            # Reset current metadata
            self.current_metadata = RecordingMetadata(
                selected_pod=selected_pod,
                selected_cup=selected_cup,
            )
            
            logger.info(
                f"Started episode {self.episode_metadata.episode_index}: "
                f"pod={selected_pod}, cup={selected_cup}"
            )
    
    def update_metadata(
        self,
        active_policy: str | None = None,
        fsm_state: str | None = None,
        sarm_progress: float | None = None,
        sarm_stage: int | None = None,
        sarm_stage_name: str | None = None,
        rnd_uncertainty: float | None = None,
        rnd_paused: bool | None = None,
        gaze_yaw: float | None = None,
        gaze_pitch: float | None = None,
    ):
        """
        Update current metadata state.
        
        Only non-None values are updated.
        """
        with self.lock:
            if active_policy is not None:
                self.current_metadata.active_policy = active_policy
            if fsm_state is not None:
                self.current_metadata.fsm_state = fsm_state
            if sarm_progress is not None:
                self.current_metadata.sarm_progress = sarm_progress
            if sarm_stage is not None:
                self.current_metadata.sarm_stage = sarm_stage
            if sarm_stage_name is not None:
                self.current_metadata.sarm_stage_name = sarm_stage_name
            if rnd_uncertainty is not None:
                self.current_metadata.rnd_uncertainty = rnd_uncertainty
            if rnd_paused is not None:
                self.current_metadata.rnd_paused = rnd_paused
            if gaze_yaw is not None:
                self.current_metadata.gaze_yaw = gaze_yaw
            if gaze_pitch is not None:
                self.current_metadata.gaze_pitch = gaze_pitch
    
    def record_frame(
        self,
        observation: dict[str, Any],
        action: dict[str, Any] | None = None,
    ) -> bool:
        """
        Record a single frame with current metadata.
        
        Args:
            observation: Robot observation dictionary
            action: Optional action dictionary
            
        Returns:
            True if frame was recorded successfully
        """
        if not self.is_recording:
            logger.warning("Not recording, ignoring frame")
            return False
        
        if self.dataset is None:
            logger.error("Dataset not initialized")
            return False
        
        with self.lock:
            try:
                # Build frame dictionary
                frame_dict = build_dataset_frame(
                    self.dataset_features,
                    observation,
                    prefix=OBS_STR,
                )
                
                # Add action if provided
                if action is not None:
                    action_frame = build_dataset_frame(
                        self.dataset_features,
                        action,
                        prefix=ACTION,
                    )
                    frame_dict.update(action_frame)
                
                # Add frame to dataset
                self.dataset.add_frame(frame_dict)
                
                # Record metadata
                metadata = RecordingMetadata(
                    timestamp=time.time(),
                    frame_index=self.frame_count,
                    active_policy=self.current_metadata.active_policy,
                    fsm_state=self.current_metadata.fsm_state,
                    sarm_progress=self.current_metadata.sarm_progress,
                    sarm_stage=self.current_metadata.sarm_stage,
                    sarm_stage_name=self.current_metadata.sarm_stage_name,
                    rnd_uncertainty=self.current_metadata.rnd_uncertainty,
                    rnd_paused=self.current_metadata.rnd_paused,
                    selected_pod=self.current_metadata.selected_pod,
                    selected_cup=self.current_metadata.selected_cup,
                    gaze_yaw=self.current_metadata.gaze_yaw,
                    gaze_pitch=self.current_metadata.gaze_pitch,
                )
                self.frame_metadata_buffer.append(metadata)
                
                # Track policies used
                if self.episode_metadata is not None:
                    if (metadata.active_policy and 
                        metadata.active_policy not in self.episode_metadata.policies_used):
                        self.episode_metadata.policies_used.append(metadata.active_policy)
                
                self.frame_count += 1
                return True
                
            except Exception as e:
                logger.error(f"Error recording frame: {e}")
                return False
    
    def save_episode(self, success: bool = True) -> bool:
        """
        Save the current episode.
        
        Args:
            success: Whether the episode was successful
            
        Returns:
            True if saved successfully
        """
        if not self.is_recording:
            logger.warning("Not recording, nothing to save")
            return False
        
        if self.dataset is None:
            logger.error("Dataset not initialized")
            return False
        
        with self.lock:
            try:
                # Update episode metadata
                if self.episode_metadata is not None:
                    self.episode_metadata.success = success
                    self.episode_metadata.end_time = time.time()
                    self.episode_metadata.num_frames = self.frame_count
                
                # Build task string with metadata
                task = self.episode_metadata.task if self.episode_metadata else ""
                
                # Save episode
                self.dataset.save_episode(task=task)
                
                # Save frame metadata to JSON
                self._save_frame_metadata()
                
                # Reset state
                self.is_recording = False
                
                logger.info(
                    f"Saved episode {self.episode_metadata.episode_index}: "
                    f"{self.frame_count} frames, success={success}"
                )
                return True
                
            except Exception as e:
                logger.error(f"Error saving episode: {e}")
                return False
    
    def _save_frame_metadata(self):
        """Save frame metadata buffer to JSON file."""
        if self.dataset is None or self.episode_metadata is None:
            return
        
        import json
        
        # Create metadata directory
        meta_dir = Path(self.dataset.root) / "meta" / "multi_policy"
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        # Save episode metadata
        episode_file = meta_dir / f"episode_{self.episode_metadata.episode_index:06d}.json"
        with open(episode_file, "w") as f:
            json.dump({
                "episode": self.episode_metadata.to_dict(),
                "frames": [m.to_dict() for m in self.frame_metadata_buffer],
            }, f, indent=2)
        
        logger.debug(f"Saved metadata to {episode_file}")
    
    def discard_episode(self):
        """Discard the current episode without saving."""
        with self.lock:
            if self.dataset is not None:
                self.dataset.clear_episode_buffer()
            
            self.is_recording = False
            self.frame_count = 0
            self.frame_metadata_buffer = []
            
            logger.info("Discarded episode")
    
    def push_to_hub(self) -> bool:
        """
        Push the dataset to HuggingFace Hub.
        
        Returns:
            True if push was successful
        """
        if self.dataset is None:
            logger.error("No dataset to push")
            return False
        
        if not self.config.push_to_hub:
            logger.info("Push to hub disabled in config")
            return False
        
        try:
            self.dataset.push_to_hub(private=self.config.private)
            logger.info(f"Pushed dataset to hub: {self.config.repo_id}")
            return True
        except Exception as e:
            logger.error(f"Error pushing to hub: {e}")
            return False
    
    def finalize(self):
        """Finalize the dataset (consolidate, etc.)."""
        if self.dataset is not None:
            try:
                self.dataset.finalize()
                logger.info("Dataset finalized")
            except Exception as e:
                logger.error(f"Error finalizing dataset: {e}")
    
    def get_episode_count(self) -> int:
        """Get the number of episodes recorded."""
        if self.dataset is None:
            return 0
        return len(self.dataset.episodes)
    
    def get_frame_count(self) -> int:
        """Get the number of frames in current episode."""
        return self.frame_count
    
    def get_current_metadata(self) -> RecordingMetadata:
        """Get a copy of current metadata state."""
        with self.lock:
            return RecordingMetadata(
                timestamp=self.current_metadata.timestamp,
                frame_index=self.current_metadata.frame_index,
                active_policy=self.current_metadata.active_policy,
                fsm_state=self.current_metadata.fsm_state,
                sarm_progress=self.current_metadata.sarm_progress,
                sarm_stage=self.current_metadata.sarm_stage,
                sarm_stage_name=self.current_metadata.sarm_stage_name,
                rnd_uncertainty=self.current_metadata.rnd_uncertainty,
                rnd_paused=self.current_metadata.rnd_paused,
                selected_pod=self.current_metadata.selected_pod,
                selected_cup=self.current_metadata.selected_cup,
                gaze_yaw=self.current_metadata.gaze_yaw,
                gaze_pitch=self.current_metadata.gaze_pitch,
            )
