#!/usr/bin/env python3
"""
SARM Progress Publisher - Runs in LeRobot environment, publishes to ROS2.

This script properly uses the SARM preprocessor pipeline to encode images
and compute progress, matching how SARM is used in training/evaluation.

IMPORTANT: The task_instruction MUST match the task description used during training.
Check your dataset's meta/tasks.jsonl file to find the correct task string.
For example, if tasks.jsonl contains {"task": "bimanual_pick_place"}, use:
    --task_instruction="bimanual_pick_place"

Using a different task instruction will cause the model to output near-zero values
because SARM was trained to detect irrelevant/perturbed task descriptions.

Key implementation details:
- Uses SARMEncodingProcessorStep for proper CLIP encoding
- Passes state_features (zeros if no robot state available)  
- Passes lengths tensor for sequence masking
- Uses the correct frame structure expected by SARM

Run from LeRobot environment:
    python -m lerobot.async_inference.sarm_progress_publisher \
        --model_path=RAPOB/sarm_3 \
        --camera_topics /camera/top/image_raw \
        --task_instruction="bimanual_pick_place"
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import draccus
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from lerobot.policies.sarm.modeling_sarm import SARMRewardModel
from lerobot.policies.sarm.sarm_utils import pad_state_to_max_dim

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage
    from std_msgs.msg import Float32, Int32, String
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("Warning: ROS2 not available - progress will only be shown in console")


@dataclass
class SARMConfig:
    """Configuration for SARM progress monitoring."""
    model_path: str = field(metadata={"help": "Path to SARM model (HuggingFace or local)"})
    device: str = field(default="cuda", metadata={"help": "Device for inference"})
    rate_hz: float = field(default=10.0, metadata={"help": "Update rate"})
    scheme: str = field(default="sparse", metadata={"help": "Annotation scheme: sparse or dense"})
    camera_topics: list[str] = field(
        default_factory=lambda: ["/camera/top/image_raw"], 
        metadata={"help": "Camera topics to subscribe to"}
    )
    task_instruction: str = field(
        default="bimanual_pick_place", 
        metadata={"help": "Task description for CLIP text encoding. MUST match the task string from training dataset's meta/tasks.jsonl"}
    )
    smoothing_alpha: float = field(
        default=0.3,
        metadata={"help": "Exponential smoothing factor (0-1). Lower = smoother, higher = more responsive"}
    )
    use_delayed_mode: bool = field(
        default=True,
        metadata={"help": "If True, add delay to match SARM's bidirectional training context (more accurate but adds latency). If False, use only past frames (lower latency but less accurate)."}
    )
    debug: bool = field(
        default=False,
        metadata={"help": "If True, print debug info for EVERY inference (verbose output)"}
    )

class SARMProgressPublisher:
    """
    SARM progress computation with ROS2 publishing.
    
    Uses the same encoding pipeline as SARM training/evaluation:
    1. CLIP image encoding with proper normalization
    2. CLIP text encoding for task instruction
    3. State features (zeros for now, could integrate robot state)
    4. Proper sequence lengths for transformer attention masking
    """
    
    def __init__(self, config: SARMConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Load SARM model
        print(f"[SARM] Loading model from: {config.model_path}")
        self.sarm_model = SARMRewardModel.from_pretrained(config.model_path)
        self.sarm_model.to(self.device)
        self.sarm_model.eval()
        
        # Extract model configuration
        self.n_obs_steps = self.sarm_model.config.n_obs_steps  # Number of observation frames (typically 8)
        self.frame_gap = self.sarm_model.config.frame_gap  # Gap between frames (typically 30)
        self.max_state_dim = self.sarm_model.config.max_state_dim
        self.num_frames = 1 + self.n_obs_steps  # Total observation frames (9 for n_obs_steps=8)
        
        # Get stage names for display
        self.stage_names = getattr(
            self.sarm_model.config,
            f'{config.scheme}_subtask_names',
            [f'stage_{i}' for i in range(self.sarm_model.config.num_sparse_stages)]
        )
        self.num_stages = len(self.stage_names)
        
        print(f"[SARM] Model config:")
        print(f"[SARM]   n_obs_steps: {self.n_obs_steps}")
        print(f"[SARM]   frame_gap: {self.frame_gap}")
        print(f"[SARM]   num_frames needed: {self.num_frames}")
        print(f"[SARM]   num_stages: {self.num_stages}")
        print(f"[SARM]   stage_names: {self.stage_names}")
        print(f"[SARM]   num_sparse_stages: {self.sarm_model.config.num_sparse_stages}")
        print(f"[SARM]   sparse_temporal_proportions: {self.sarm_model.config.sparse_temporal_proportions}")
        print(f"[SARM]   annotation_mode: {self.sarm_model.config.annotation_mode}")
        
        # Load CLIP model (same as used in SARM training)
        print(f"[SARM] Loading CLIP model...")
        clip_model_name = "openai/clip-vit-base-patch32"
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=True)
        self.clip_model.to(self.device)
        self.clip_model.eval()
        
        # Pre-compute text embeddings for task instruction
        self._encode_task_instruction(config.task_instruction)
        
        # Warn about common task instruction mistakes
        if config.task_instruction in ["perform the task", "task", "do the task"]:
            print(f"[SARM] ⚠️  WARNING: Using generic task instruction '{config.task_instruction}'")
            print(f"[SARM] ⚠️  For correct progress estimation, use the EXACT task string from training.")
            print(f"[SARM] ⚠️  Check your dataset's meta/tasks.jsonl file for the correct task string.")
        
        print(f"[SARM] ✓ Models loaded successfully!")
        
        # Frame buffer for temporal sequence
        # SARM uses bidirectional sampling with frames spaced by frame_gap
        # For 30fps camera, frame_gap=30 means 1 second between samples
        self.frame_buffer = []
        self.frame_timestamps = []
        self.data_lock = threading.Lock()
        
        # Calculate expected temporal spacing (in seconds)
        # frame_gap is in frames at the dataset's fps (typically 30)
        self.expected_frame_spacing = self.frame_gap / 30.0  # seconds between samples
        
        # Delayed mode parameters
        self.use_delayed_mode = config.use_delayed_mode
        self.half_window = self.n_obs_steps // 2  # Number of frames before/after center
        
        if self.use_delayed_mode:
            # In delayed mode, we wait for "future" frames to arrive
            # The delay equals half the temporal window
            self.inference_delay = self.half_window * self.expected_frame_spacing
            print(f"[SARM] Delayed mode: inference delay = {self.inference_delay:.1f}s (bidirectional context)")
        else:
            self.inference_delay = 0
            print(f"[SARM] Causal mode: no delay (past-only context, less accurate)")
        
        print(f"[SARM] Expected temporal spacing: {self.expected_frame_spacing:.2f}s between samples")
        
        # Smoothing for stable progress output
        self.smoothed_progress = None
        self.smoothing_alpha = config.smoothing_alpha
        
        # ROS2 node
        self.ros_node = None
        if ROS2_AVAILABLE:
            self._init_ros2()
    
    def _encode_task_instruction(self, task: str):
        """Encode task instruction with CLIP text encoder."""
        print(f"[SARM] Encoding task: '{task}'")
        inputs = self.clip_processor(text=[task], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            self.text_embeddings = self.clip_model.get_text_features(**inputs).float()
        print(f"[SARM] Text embeddings shape: {self.text_embeddings.shape}")
    
    def _init_ros2(self):
        """Initialize ROS2 publishers and subscribers."""
        rclpy.init()
        self.ros_node = Node("sarm_progress_publisher")
        
        # Subscribe to camera topics
        for topic in self.config.camera_topics:
            self.ros_node.create_subscription(
                RosImage, topic,
                self._camera_callback,
                10
            )
            print(f"[SARM] Subscribed to: {topic}")
        
        # Publishers
        self.progress_pub = self.ros_node.create_publisher(Float32, '/sarm/progress', 10)
        self.stage_pub = self.ros_node.create_publisher(Int32, '/sarm/stage', 10)
        self.stage_name_pub = self.ros_node.create_publisher(String, '/sarm/stage_name', 10)
        
        print(f"[SARM] Publishing to: /sarm/progress, /sarm/stage, /sarm/stage_name")
    
    def _camera_callback(self, msg):
        """Handle incoming camera images."""
        try:
            # Convert ROS image to numpy
            if msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding == 'bgr8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                print(f"[SARM] Unsupported encoding: {msg.encoding}")
                return
            
            # Use ROS message timestamp instead of wall time for better temporal accuracy
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            with self.data_lock:
                self.frame_buffer.append(img)
                self.frame_timestamps.append(timestamp)
                
                # Keep buffer size for the required temporal window + some margin
                # We need (num_frames - 1) * frame_spacing seconds of history
                max_age = (self.num_frames + 2) * self.expected_frame_spacing
                while len(self.frame_timestamps) > 2 and (timestamp - self.frame_timestamps[0]) > max_age:
                    self.frame_buffer.pop(0)
                    self.frame_timestamps.pop(0)
                    
        except Exception as e:
            print(f"[SARM] Error processing image: {e}")
    
    @torch.no_grad()
    def _encode_images(self, images: list[np.ndarray]) -> torch.Tensor:
        """
        Encode images using CLIP, matching SARM preprocessor behavior.
        
        Args:
            images: List of RGB numpy images (H, W, 3) uint8
            
        Returns:
            Encoded features tensor (1, T, 512)
        """
        # Convert to PIL images (as done in SARMEncodingProcessorStep)
        pil_images = []
        for img in images:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            pil_images.append(Image.fromarray(img))
        
        # Process through CLIP
        inputs = self.clip_processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get image embeddings
        embeddings = self.clip_model.get_image_features(**inputs)
        
        # Reshape to (1, T, 512) - batch size 1
        embeddings = embeddings.unsqueeze(0)  # (T, 512) -> (1, T, 512)
        
        return embeddings.float()
    
    def _sample_temporal_frames(self) -> Optional[tuple[list[np.ndarray], float]]:
        """
        Sample frames with temporal spacing matching SARM's observation_delta_indices.
        
        SARM uses bidirectional sampling: past + current + future frames.
        The observation_delta_indices are: [-120, -90, -60, -30, 0, 30, 60, 90, 120]
        which means:
        - Index 0: 4 seconds in past (oldest)
        - Index 4: current frame (center)
        - Index 8: 4 seconds in future (newest)
        
        Two modes:
        1. Delayed mode (use_delayed_mode=True): 
           - Wait for "future" frames to arrive
           - Center frame = frame from `inference_delay` seconds ago
           - Matches training distribution exactly
           
        2. Causal mode (use_delayed_mode=False):
           - Use only past frames (no delay)
           - Center frame = current frame
           - Less accurate but no latency
        
        Returns:
            Tuple of (list of num_frames images, center_time) or None if not enough data
        """
        with self.data_lock:
            if len(self.frame_buffer) < 2:
                return None
            
            now = self.frame_timestamps[-1]
            
            if self.use_delayed_mode:
                # Delayed mode: center frame is from `inference_delay` seconds ago
                # This allows us to have both past and future context
                center_time = now - self.inference_delay
                
                # Check if we have enough data (need frames from center-4s to center+4s)
                min_time_needed = center_time - self.half_window * self.expected_frame_spacing
                max_time_needed = center_time + self.half_window * self.expected_frame_spacing
                
                oldest_time = self.frame_timestamps[0]
                if oldest_time > min_time_needed:
                    return None  # Not enough past data
                if now < max_time_needed:
                    return None  # Not enough future data
                    
            else:
                # Causal mode: center frame is current, only use past
                center_time = now
                total_time_needed = (self.num_frames - 1) * self.expected_frame_spacing
                oldest_time = self.frame_timestamps[0]
                if (now - oldest_time) < total_time_needed:
                    return None
            
            # Sample frames at correct temporal positions
            sampled_frames = []
            for i in range(self.num_frames):
                if self.use_delayed_mode:
                    # Bidirectional: sample relative to center frame
                    delta = (i - self.half_window) * self.expected_frame_spacing
                    target_time = center_time + delta
                else:
                    # Causal: sample from past only, oldest at index 0
                    target_time = now - (self.num_frames - 1 - i) * self.expected_frame_spacing
                
                # Find closest frame in buffer
                best_idx = 0
                best_diff = float('inf')
                for j, ts in enumerate(self.frame_timestamps):
                    diff = abs(ts - target_time)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = j
                
                sampled_frames.append(self.frame_buffer[best_idx].copy())
            
            return sampled_frames, center_time
    
    def compute_progress(self, debug: bool = False) -> tuple[Optional[float], Optional[int], Optional[str]]:
        """
        Compute SARM progress from buffered frames.
        
        Returns:
            (progress, stage_idx, stage_name) or (None, None, None) if not ready
        """
        # Get temporally sampled frames
        result = self._sample_temporal_frames()
        if result is None:
            return None, None, None
        
        frames, center_time = result
        
        try:
            # Encode images with CLIP
            video_features = self._encode_images(frames)  # (1, T, 512)
            
            if debug:
                mode_str = "delayed (bidirectional)" if self.use_delayed_mode else "causal (past-only)"
                print(f"\n[DEBUG] Mode: {mode_str}")
                print(f"[DEBUG] video_features shape: {video_features.shape}")
                print(f"[DEBUG] video_features norm per frame: {[f'{n:.2f}' for n in video_features[0].norm(dim=-1).tolist()]}")
                print(f"[DEBUG] text_embeddings norm: {self.text_embeddings.norm(dim=-1).mean().item():.4f}")
                # Check cosine similarity between text and each frame
                text_norm = self.text_embeddings / self.text_embeddings.norm(dim=-1, keepdim=True)
                video_norm = video_features / video_features.norm(dim=-1, keepdim=True)
                cosine_sim = (video_norm @ text_norm.T).squeeze()
                avg_sim = cosine_sim.mean().item()
                print(f"[DEBUG] text-video cosine similarity: {[f'{s:.3f}' for s in cosine_sim.tolist()]}")
                print(f"[DEBUG] Average cosine similarity: {avg_sim:.3f}")
                if avg_sim < 0.25:
                    print(f"[WARNING] LOW cosine similarity ({avg_sim:.3f})! Expected >0.30 for good matches.")
                    print(f"[WARNING] This indicates weak text-video alignment. Check:")
                    print(f"[WARNING]   1. Task instruction matches training data exactly")
                    print(f"[WARNING]   2. Camera view shows task-relevant features")
                    print(f"[WARNING]   3. Visual scene matches training conditions")
            
            # Create state features (zeros - we could integrate robot state here)
            batch_size = 1
            seq_len = video_features.shape[1]
            state_features = torch.zeros(
                batch_size, seq_len, self.max_state_dim, 
                device=self.device, dtype=torch.float32
            )
            
            # Create lengths tensor (all frames are valid)
            lengths = torch.tensor([seq_len], dtype=torch.int32, device=self.device)
            
            if debug:
                print(f"[DEBUG] state_features shape: {state_features.shape}")
                print(f"[DEBUG] lengths: {lengths}")
            
            # Determine which frame to extract the prediction from
            # - Delayed mode: use center frame (index = half_window = 4), matches training
            # - Causal mode: use last frame (index = num_frames - 1 = 8), since that's "now"
            if self.use_delayed_mode:
                target_idx = self.half_window  # Center frame (index 4)
            else:
                target_idx = self.num_frames - 1  # Last frame (index 8)
            
            # Call SARM model - only get prediction for the target frame
            rewards, stage_probs, raw_values = self.sarm_model.calculate_rewards(
                text_embeddings=self.text_embeddings,
                video_embeddings=video_features,
                state_features=state_features,
                lengths=lengths,
                return_all_frames=False,  # Only get single frame prediction
                return_stages=True,
                return_raw=True,  # Get raw stage_idx and tau_pred for debugging
                head_mode=self.config.scheme,
                frame_index=target_idx,  # Specify which frame to predict for
            )
            raw_stage_idx, raw_tau = raw_values
            
            if debug:
                print(f"[DEBUG] target_idx: {target_idx}")
                # Convert numpy arrays to scalars for printing
                stage_val = float(raw_stage_idx) if isinstance(raw_stage_idx, np.ndarray) else raw_stage_idx
                tau_val = float(raw_tau) if isinstance(raw_tau, np.ndarray) else raw_tau
                print(f"[DEBUG] RAW stage_idx: {stage_val}, tau_pred: {tau_val:.4f}")
                print(f"[DEBUG] progress (normalized): {rewards}")
                print(f"[DEBUG] stage_probs: {stage_probs}")
            
            # Extract progress value
            if isinstance(rewards, np.ndarray):
                raw_progress = float(rewards[0] if rewards.ndim > 0 else rewards)
            else:
                raw_progress = float(rewards)
            
            raw_progress = max(0.0, min(1.0, raw_progress))
            
            # Apply exponential smoothing to reduce jitter
            if self.smoothed_progress is None:
                self.smoothed_progress = raw_progress
            else:
                self.smoothed_progress = (
                    self.smoothing_alpha * raw_progress + 
                    (1 - self.smoothing_alpha) * self.smoothed_progress
                )
            progress = self.smoothed_progress
            
            # Extract stage prediction (single value when return_all_frames=False)
            if isinstance(stage_probs, np.ndarray):
                if stage_probs.ndim >= 2:
                    stage_idx = int(np.argmax(stage_probs[0]))
                else:
                    stage_idx = int(np.argmax(stage_probs))
            else:
                stage_idx = 0
            
            stage_name = self.stage_names[stage_idx] if stage_idx < len(self.stage_names) else "unknown"
            
            if debug:
                print(f"[DEBUG] raw_progress={raw_progress:.4f}, smoothed={progress:.4f}")
            
            return progress, stage_idx, stage_name
            
        except Exception as e:
            print(f"\n[SARM] Error computing progress: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def publish_progress(self, progress: float, stage_idx: int, stage_name: str):
        """Publish progress to ROS2 topics."""
        if self.ros_node is None:
            return
        
        self.progress_pub.publish(Float32(data=progress))
        self.stage_pub.publish(Int32(data=stage_idx))
        self.stage_name_pub.publish(String(data=stage_name))
    
    def run(self):
        """Main loop."""
        # Calculate how much buffered data we need
        if self.use_delayed_mode:
            # Delayed mode needs:
            # - Past frames: half_window * spacing = 4 * 1.0 = 4s
            # - Future frames (delay): half_window * spacing = 4 * 1.0 = 4s
            # Total = 8 seconds of buffer
            total_time_needed = (self.half_window * 2) * self.expected_frame_spacing
        else:
            # Causal mode just needs all 9 frames in chronological order
            # = (num_frames - 1) * spacing = 8 * 1.0 = 8 seconds
            total_time_needed = (self.num_frames - 1) * self.expected_frame_spacing
        
        print(f"\n[SARM] 🎬 Starting progress monitoring")
        print(f"[SARM] Rate: {self.config.rate_hz} Hz")
        print(f"[SARM] Task: '{self.config.task_instruction}'")
        print(f"[SARM] Temporal window: {total_time_needed:.1f}s ({self.num_frames} frames)")
        print(f"[SARM] Waiting for data from: {self.config.camera_topics}")
        print()
        
        rate = 1.0 / self.config.rate_hz
        log_counter = 0
        last_progress = None
        
        debug_counter = 0
        
        try:
            while True:
                loop_start = time.time()
                
                # Spin ROS2
                if self.ros_node:
                    rclpy.spin_once(self.ros_node, timeout_sec=0.001)
                
                # Check buffer status
                with self.data_lock:
                    buffer_size = len(self.frame_buffer)
                    if buffer_size > 1:
                        time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
                    else:
                        time_span = 0
                
                if buffer_size == 0:
                    if log_counter % 50 == 0:
                        print(f"\r[SARM] Waiting for camera data...", end='', flush=True)
                    log_counter += 1
                    time.sleep(max(0, rate - (time.time() - loop_start)))
                    continue
                    
                if time_span < total_time_needed:
                    remaining = total_time_needed - time_span
                    if log_counter % 10 == 0:
                        print(f"\r[SARM] Buffering: {time_span:.1f}s / {total_time_needed:.1f}s ({remaining:.1f}s remaining)", 
                              end='', flush=True)
                    log_counter += 1
                    time.sleep(max(0, rate - (time.time() - loop_start)))
                    continue
                
                # Compute progress (debug every 5 seconds or always if config.debug)
                do_debug = self.config.debug or debug_counter == 0
                progress, stage_idx, stage_name = self.compute_progress(debug=do_debug)
                debug_counter = (debug_counter + 1) % int(self.config.rate_hz * 5)
                
                if progress is not None:
                    # Publish
                    self.publish_progress(progress, stage_idx, stage_name)
                    
                    # Console output
                    progress_bar = '█' * int(progress * 20) + '░' * (20 - int(progress * 20))
                    print(f"\r[SARM] [{progress_bar}] {progress*100:5.1f}% | Stage {stage_idx}/{self.num_stages-1}: {stage_name:15s}", 
                          end='', flush=True)
                    
                    # Periodic detailed log
                    log_counter += 1
                    if log_counter >= int(self.config.rate_hz * 2):
                        print()  # New line for detailed log
                        if last_progress is not None:
                            delta = progress - last_progress
                            print(f"[SARM] Progress={progress:.4f} (Δ={delta:+.4f}), Stage={stage_idx}, Name='{stage_name}'")
                        else:
                            print(f"[SARM] Progress={progress:.4f}, Stage={stage_idx}, Name='{stage_name}'")
                        last_progress = progress
                        log_counter = 0
                
                # Maintain rate
                elapsed = time.time() - loop_start
                time.sleep(max(0, rate - elapsed))
                
        except KeyboardInterrupt:
            print("\n[SARM] Stopped by user")
        finally:
            if self.ros_node:
                self.ros_node.destroy_node()
                rclpy.shutdown()


@draccus.wrap()
def main(config: SARMConfig):
    """Main entry point."""
    publisher = SARMProgressPublisher(config)
    publisher.run()


if __name__ == "__main__":
    main()
