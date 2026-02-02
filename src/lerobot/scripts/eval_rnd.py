#!/usr/bin/env python3
"""
Evaluate RND Uncertainty on Recorded Dataset

This script evaluates trained RND models on recorded demonstration data,
visualizing uncertainty predictions over time - similar to SARM visualization
but using RND for progress/completion detection.

Features:
- Runs RND on all frames of test episodes
- Plots uncertainty over time with episode boundaries
- Detects task start/end based on uncertainty patterns
- Outputs statistics and visualizations

Usage:
    # Evaluate on recorded data
    python -m lerobot.scripts.eval_rnd \
        --rnd-path outputs/rnd_universal/task_end/rnd_universal.pth \
        --dataset-repo-id RAPOB/bimanual_transfer_pen_holder_50_no_aria \
        --camera-key observation.images.top \
        --output-dir outputs/rnd_eval \
        --episodes 0 1 2 3 4

    # Compare with policy-specific RND
    python -m lerobot.scripts.eval_rnd \
        --rnd-path outputs/rnd/policy_specific/rnd_model.pth \
        --policy-path AcuBrain/act_bimanual_pen_pick_place_augmented \
        --dataset-repo-id RAPOB/bimanual_transfer_pen_holder_50_no_aria \
        --camera-key observation.images.top \
        --output-dir outputs/rnd_eval_policy_specific
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

# Set RMW before any ROS imports
os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_image_transforms(image_size: tuple[int, int]) -> transforms.Compose:
    """Create image transforms matching training."""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.Resize(image_size, antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
    ])


def load_rnd_model(rnd_path: Path, policy_path: Optional[str], device: str):
    """Load RND model - either universal or policy-specific."""
    rnd_path = Path(rnd_path)
    
    # Check if it's a directory or file
    if rnd_path.is_dir():
        # Check for universal RND first
        if (rnd_path / "rnd_universal.pth").exists():
            rnd_path = rnd_path / "rnd_universal.pth"
        elif (rnd_path / "rnd_model.pth").exists():
            rnd_path = rnd_path / "rnd_model.pth"
        else:
            raise FileNotFoundError(f"No RND model found in {rnd_path}")
    
    # Load checkpoint to check type
    checkpoint = torch.load(rnd_path, map_location=device, weights_only=False)
    
    # Determine if this is universal or policy-specific
    if 'backbone_type' in checkpoint:
        # Universal RND
        logger.info("Loading Universal RND model")
        from lerobot.common.uncertainty.rnd_module_universal import RNDModuleUniversal
        rnd = RNDModuleUniversal.load(rnd_path, device=device)
        is_universal = True
    elif 'resnet_model' in checkpoint:
        # Policy-specific RND with saved backbone
        logger.info("Loading Policy-Specific RND model (with saved backbone)")
        from lerobot.common.uncertainty import RNDModule
        
        # Load saved ResNet backbone
        resnet = checkpoint['resnet_model']
        resnet = resnet.to(device)
        resnet.eval()
        
        # Create RND module
        rnd = RNDModule(
            resnet_backbone=resnet,
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim'],
            image_size=tuple(checkpoint['image_size']),
            device=device,
        )
        rnd.load(rnd_path)
        is_universal = False
    else:
        # Policy-specific RND - need to reload policy
        logger.info("Loading Policy-Specific RND model (needs policy reload)")
        if policy_path is None:
            raise ValueError("Policy path required for policy-specific RND without saved backbone")
        
        from lerobot.common.uncertainty import RNDModule
        from lerobot.policies.act.configuration_act import ACTConfig  # Register
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        # Load policy config and backbone
        policy_config = PreTrainedConfig.from_pretrained(policy_path)
        
        # We need dataset metadata - use a minimal dataset
        temp_dataset = LeRobotDataset(checkpoint.get('dataset_repo_id', 'lerobot/pusht'))
        
        policy = make_policy(policy_config, ds_meta=temp_dataset.meta)
        policy = policy.to(device)
        policy.eval()
        
        # Extract backbone
        if hasattr(policy, "model") and hasattr(policy.model, "backbone"):
            resnet = policy.model.backbone
        elif hasattr(policy, "backbone"):
            resnet = policy.backbone
        else:
            raise AttributeError("Could not find ResNet backbone in policy")
        
        # Create RND module
        rnd = RNDModule(
            resnet_backbone=resnet,
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim'],
            image_size=tuple(checkpoint['image_size']),
            device=device,
        )
        rnd.load(rnd_path)
        is_universal = False
    
    return rnd, is_universal


def evaluate_on_dataset(
    rnd_path: Path,
    dataset_repo_id: str,
    camera_key: str,
    output_dir: Path,
    episodes: Optional[list[int]] = None,
    policy_path: Optional[str] = None,
    device: str = "cuda",
    revision: str = "v3.0",
):
    """Evaluate RND on recorded dataset episodes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load RND model
    logger.info(f"Loading RND from {rnd_path}")
    rnd, is_universal = load_rnd_model(rnd_path, policy_path, device)
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_repo_id} (revision: {revision})")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    if episodes:
        dataset = LeRobotDataset(dataset_repo_id, episodes=episodes, revision=revision)
    else:
        dataset = LeRobotDataset(dataset_repo_id, revision=revision)
    
    logger.info(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    logger.info(f"Camera keys: {dataset.meta.camera_keys}")
    
    # Validate camera key
    if camera_key not in dataset.meta.camera_keys:
        available = ", ".join(dataset.meta.camera_keys)
        raise ValueError(f"Camera '{camera_key}' not found. Available: {available}")
    
    # Get image size from RND model
    if hasattr(rnd, 'image_size'):
        image_size = rnd.image_size
    else:
        image_size = (96, 96)
    
    image_transforms = make_image_transforms(image_size)
    
    # Find state and action keys
    sample = dataset[0]
    state_keys = [k for k in sample.keys() if "state" in k and "action" not in k]
    action_keys = [k for k in sample.keys() if "action" in k]
    
    if not state_keys:
        state_keys = [k for k in sample.keys() if "pos" in k or "position" in k]
    
    logger.info(f"State keys: {state_keys}")
    logger.info(f"Action keys: {action_keys}")
    
    # Process each episode
    results = {}
    all_uncertainties = []
    episode_boundaries = []
    frame_count = 0
    
    # Get episode indices - convert tensors to integers for proper set operations
    episode_indices_raw = dataset.hf_dataset["episode_index"]
    episode_indices = [int(e) if hasattr(e, 'item') else int(e) for e in episode_indices_raw]
    unique_episodes = sorted(set(episode_indices))
    
    # Build episode boundaries efficiently (O(n) instead of O(n²))
    ep_frame_map = {ep: [] for ep in unique_episodes}
    for frame_idx, ep_idx in enumerate(episode_indices):
        ep_frame_map[ep_idx].append(frame_idx)
    
    logger.info(f"Processing {len(unique_episodes)} episodes ({len(episode_indices)} total frames)...")
    
    rnd.eval()
    if hasattr(rnd, 'reset_rolling_buffer'):
        rnd.reset_rolling_buffer()
    
    for ep_idx in tqdm(unique_episodes, desc="Evaluating episodes"):
        # Get frames for this episode (already computed)
        ep_frame_indices = ep_frame_map[ep_idx]
        
        ep_uncertainties = []
        ep_rolling = []
        
        # Reset rolling buffer for each episode
        if hasattr(rnd, 'reset_rolling_buffer'):
            rnd.reset_rolling_buffer()
        
        for frame_idx in ep_frame_indices:
            item = dataset[frame_idx]
            
            # Get image
            image = item[camera_key]
            image = image_transforms(image).unsqueeze(0).to(device)
            
            # Get state
            if state_keys and rnd.state_dim > 0:
                state_parts = [item[k] for k in state_keys if k in item]
                state = torch.cat(state_parts) if len(state_parts) > 1 else state_parts[0]
                state = state.unsqueeze(0).to(device)
            else:
                state = None
            
            # Get action
            if action_keys and rnd.action_dim > 0:
                action = item[action_keys[0]].unsqueeze(0).to(device)
            else:
                action = None
            
            # Compute uncertainty
            with torch.no_grad():
                step_unc, rolling_unc = rnd.compute_uncertainty(
                    image, state, action, normalize=True
                )
            
            ep_uncertainties.append(step_unc)
            ep_rolling.append(rolling_unc)
            all_uncertainties.append(step_unc)
        
        # Compute gradient (rate of change) of uncertainty
        ep_unc_arr = np.array(ep_rolling)
        gradient = np.gradient(ep_unc_arr)
        abs_gradient = np.abs(gradient)
        
        # Compute rolling variance (stability measure) - window of 10 frames
        window = 10
        rolling_var = []
        for i in range(len(ep_unc_arr)):
            start = max(0, i - window // 2)
            end = min(len(ep_unc_arr), i + window // 2 + 1)
            rolling_var.append(np.var(ep_unc_arr[start:end]))
        rolling_var = np.array(rolling_var)
        
        # Smooth the gradient with a larger window for stability detection
        smooth_window = 30  # 1 second at 30fps
        smooth_gradient = np.convolve(abs_gradient, np.ones(smooth_window)/smooth_window, mode='same')
        
        # Detect "stable" regions - low gradient AND low variance
        # Stability score = inverse of (abs_gradient + rolling_var)
        stability = 1.0 / (abs_gradient + rolling_var + 1e-6)
        stability_normalized = (stability - stability.min()) / (stability.max() - stability.min() + 1e-6)
        
        # === TASK COMPLETION DETECTION ===
        # Find when gradient stays near zero for sustained period
        # Threshold: gradient < X percentile of all gradients for Y consecutive frames
        gradient_threshold = np.percentile(abs_gradient, 25)  # 25th percentile = low activity
        consecutive_threshold = 45  # 1.5 seconds at 30fps
        
        # Find first point where gradient is low for consecutive frames
        low_gradient_mask = smooth_gradient < gradient_threshold
        completion_frame = None
        consecutive_count = 0
        
        # Start looking after first 150 frames (5 seconds) to avoid detecting start state
        min_start_frame = min(150, len(low_gradient_mask) // 5)
        
        for i in range(min_start_frame, len(low_gradient_mask)):
            if low_gradient_mask[i]:
                consecutive_count += 1
                if consecutive_count >= consecutive_threshold:
                    completion_frame = i - consecutive_threshold + 1
                    break
            else:
                consecutive_count = 0
        
        # If no completion detected, use end of episode
        if completion_frame is None:
            completion_frame = len(ep_frame_indices) - 1
        
        # Compute completion statistics
        completion_pct = completion_frame / len(ep_frame_indices) * 100
        frames_after_completion = len(ep_frame_indices) - completion_frame
        
        # Find end detection point: last N frames where stability is high
        end_window = 15  # Last 15 frames
        end_stability = np.mean(stability_normalized[-end_window:])
        
        # Store episode results
        results[ep_idx] = {
            'step_uncertainties': ep_uncertainties,
            'rolling_uncertainties': ep_rolling,
            'gradient': gradient.tolist(),
            'abs_gradient': abs_gradient.tolist(),
            'smooth_gradient': smooth_gradient.tolist(),
            'rolling_variance': rolling_var.tolist(),
            'stability': stability_normalized.tolist(),
            'end_stability': float(end_stability),
            'num_frames': len(ep_frame_indices),
            'completion_frame': int(completion_frame),
            'completion_pct': float(completion_pct),
            'frames_after_completion': int(frames_after_completion),
            'gradient_threshold': float(gradient_threshold),
            'mean_uncertainty': float(np.mean(ep_uncertainties)),
            'std_uncertainty': float(np.std(ep_uncertainties)),
            'max_uncertainty': float(np.max(ep_uncertainties)),
            'min_uncertainty': float(np.min(ep_uncertainties)),
            'mean_gradient': float(np.mean(abs_gradient)),
            'end_gradient': float(np.mean(abs_gradient[-end_window:])),
            'end_variance': float(np.mean(rolling_var[-end_window:])),
        }
        
        # Track episode boundaries
        episode_boundaries.append(frame_count)
        frame_count += len(ep_frame_indices)
    
    episode_boundaries.append(frame_count)  # End of last episode
    
    # Compute overall statistics including completion info
    completion_frames = [results[ep]['completion_frame'] for ep in results]
    total_frames = [results[ep]['num_frames'] for ep in results]
    completion_pcts = [results[ep]['completion_pct'] for ep in results]
    frames_wasted = [results[ep]['frames_after_completion'] for ep in results]
    
    overall_stats = {
        'mean': float(np.mean(all_uncertainties)),
        'std': float(np.std(all_uncertainties)),
        'max': float(np.max(all_uncertainties)),
        'min': float(np.min(all_uncertainties)),
        'threshold_2std': float(np.mean(all_uncertainties) + 2 * np.std(all_uncertainties)),
        # Completion detection stats
        'avg_completion_frame': float(np.mean(completion_frames)),
        'avg_total_frames': float(np.mean(total_frames)),
        'avg_completion_pct': float(np.mean(completion_pcts)),
        'avg_frames_wasted': float(np.mean(frames_wasted)),
        'wasted_pct': float(np.mean(frames_wasted) / np.mean(total_frames) * 100),
    }
    
    logger.info(f"Overall uncertainty - Mean: {overall_stats['mean']:.4f}, Std: {overall_stats['std']:.4f}")
    logger.info(f"Task completion detected at {overall_stats['avg_completion_pct']:.1f}% of episode on average")
    logger.info(f"Average wasted frames: {overall_stats['avg_frames_wasted']:.0f} ({overall_stats['wasted_pct']:.1f}%)")
    
    # Save results
    output_data = {
        'rnd_path': str(rnd_path),
        'dataset_repo_id': dataset_repo_id,
        'camera_key': camera_key,
        'is_universal': is_universal,
        'overall_stats': overall_stats,
        'detection_params': {
            'gradient_smooth_window': 30,
            'consecutive_threshold': 45,
            'min_start_frame': 150,
            'gradient_percentile': 25,
        },
        'episode_results': {int(k): {
            'num_frames': v['num_frames'],
            'completion_frame': v['completion_frame'],
            'completion_pct': v['completion_pct'],
            'frames_after_completion': v['frames_after_completion'],
            'mean_uncertainty': v['mean_uncertainty'],
            'std_uncertainty': v['std_uncertainty'],
            'max_uncertainty': v['max_uncertainty'],
            'min_uncertainty': v['min_uncertainty'],
            'gradient_threshold': v['gradient_threshold'],
        } for k, v in results.items()},
    }
    
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Create visualizations
    create_visualizations(results, episode_boundaries, overall_stats, output_dir)
    
    logger.info(f"Results saved to {output_dir}")
    return results, overall_stats


def create_visualizations(results, episode_boundaries, stats, output_dir):
    """Create visualization plots."""
    
    # 1. All episodes uncertainty over time
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Concatenate all uncertainties
    all_step = []
    all_rolling = []
    for ep_idx in sorted(results.keys()):
        all_step.extend(results[ep_idx]['step_uncertainties'])
        all_rolling.extend(results[ep_idx]['rolling_uncertainties'])
    
    frames = np.arange(len(all_step))
    
    # Plot step uncertainty
    ax1 = axes[0]
    ax1.plot(frames, all_step, 'b-', alpha=0.5, linewidth=0.5, label='Step Uncertainty')
    ax1.plot(frames, all_rolling, 'r-', linewidth=1.5, label='Rolling Uncertainty')
    ax1.axhline(y=stats['threshold_2std'], color='g', linestyle='--', label=f'Threshold ({stats["threshold_2std"]:.2f})')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add episode boundaries
    for i, boundary in enumerate(episode_boundaries[:-1]):
        ax1.axvline(x=boundary, color='orange', linestyle=':', alpha=0.5)
        if i < 10:  # Label first 10 episodes
            ax1.text(boundary + 5, ax1.get_ylim()[1] * 0.9, f'Ep {i}', fontsize=8)
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Normalized Uncertainty')
    ax1.set_title('RND Uncertainty Over Time (All Episodes)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot histogram
    ax2 = axes[1]
    ax2.hist(all_step, bins=100, alpha=0.7, color='blue', label='Step Uncertainty')
    ax2.axvline(x=stats['mean'], color='r', linestyle='-', linewidth=2, label=f'Mean ({stats["mean"]:.2f})')
    ax2.axvline(x=stats['threshold_2std'], color='g', linestyle='--', linewidth=2, label=f'Threshold ({stats["threshold_2std"]:.2f})')
    ax2.set_xlabel('Uncertainty')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Uncertainty Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "rnd_uncertainty_all_episodes.png", dpi=150)
    plt.close()
    
# 2. Individual episode plots with gradient analysis and completion detection (first 5)
    num_plots = min(5, len(results))
    fig, axes = plt.subplots(num_plots, 3, figsize=(18, 4 * num_plots))
    if num_plots == 1:
        axes = axes.reshape(1, -1)

    for i, ep_idx in enumerate(sorted(results.keys())[:num_plots]):
        ep_data = results[ep_idx]
        frames = np.arange(ep_data['num_frames'])
        completion_frame = ep_data.get('completion_frame', ep_data['num_frames'] - 1)
        completion_pct = ep_data.get('completion_pct', 100.0)
        
        # Column 1: Uncertainty with completion marker
        ax1 = axes[i, 0]
        ax1.plot(frames, ep_data['step_uncertainties'], 'b-', alpha=0.5, label='Step')
        ax1.plot(frames, ep_data['rolling_uncertainties'], 'r-', linewidth=2, label='Rolling')
        ax1.axhline(y=stats['threshold_2std'], color='g', linestyle='--', label='Threshold')
        ax1.axvline(x=completion_frame, color='red', linestyle='-', linewidth=3, label=f'Done @ {completion_frame}')
        ax1.axvspan(completion_frame, frames[-1], alpha=0.2, color='red', label='After completion')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Uncertainty')
        ax1.set_title(f'Ep {ep_idx} - Task Complete @ frame {completion_frame} ({completion_pct:.0f}%)')
        ax1.legend(loc='upper right', fontsize=7)
        ax1.grid(True, alpha=0.3)
        
        # Column 2: Gradient (rate of change) with smooth and completion marker
        ax2 = axes[i, 1]
        ax2.plot(frames, ep_data['abs_gradient'], 'purple', alpha=0.4, linewidth=0.5, label='|Gradient|')
        if 'smooth_gradient' in ep_data:
            ax2.plot(frames, ep_data['smooth_gradient'], 'blue', linewidth=2, label='Smooth gradient')
            ax2.axhline(y=ep_data.get('gradient_threshold', 0), color='orange', linestyle='--', label='Threshold')
        ax2.axvline(x=completion_frame, color='red', linestyle='-', linewidth=3, label='Completion')
        ax2.axvspan(completion_frame, frames[-1], alpha=0.2, color='green', label='Stable')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('|dU/dt|')
        ax2.set_title(f'Gradient → 0 = Task Done')
        ax2.legend(loc='upper right', fontsize=7)
        ax2.grid(True, alpha=0.3)
        
        # Column 3: Combined view - Uncertainty + Gradient overlay
        ax3 = axes[i, 2]
        ax3_twin = ax3.twinx()
        
        ln1 = ax3.plot(frames, ep_data['rolling_uncertainties'], 'blue', linewidth=2, label='Uncertainty')
        ax3.set_ylabel('Uncertainty', color='blue')
        ax3.tick_params(axis='y', labelcolor='blue')
        
        if 'smooth_gradient' in ep_data:
            ln2 = ax3_twin.plot(frames, ep_data['smooth_gradient'], 'orange', linewidth=2, label='Gradient')
        else:
            ln2 = ax3_twin.plot(frames, ep_data['abs_gradient'], 'orange', linewidth=2, label='Gradient')
        ax3_twin.set_ylabel('Gradient', color='orange')
        ax3_twin.tick_params(axis='y', labelcolor='orange')
        
        ax3.axvline(x=completion_frame, color='red', linestyle='-', linewidth=3)
        ax3.axvspan(completion_frame, frames[-1], alpha=0.15, color='green')
        ax3.set_xlabel('Frame')
        ax3.set_title(f'Combined: {ep_data["num_frames"] - completion_frame} frames after completion')
        
        # Combine legends
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax3.legend(lns, labs, loc='upper right', fontsize=7)
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "rnd_uncertainty_individual_episodes.png", dpi=150)
    plt.close()
    
    # 3. Progress-like visualization (normalized per episode)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for ep_idx in sorted(results.keys())[:10]:  # First 10 episodes
        ep_data = results[ep_idx]
        # Normalize time to 0-1
        normalized_time = np.linspace(0, 1, ep_data['num_frames'])
        # Invert uncertainty to get "progress" (low uncertainty = high progress)
        progress = 1 - np.clip(np.array(ep_data['rolling_uncertainties']) / stats['threshold_2std'], 0, 1)
        ax.plot(normalized_time, progress, alpha=0.5, label=f'Ep {ep_idx}')
    
    ax.set_xlabel('Normalized Episode Time')
    ax.set_ylabel('Estimated Progress (1 - normalized uncertainty)')
    ax.set_title('RND-based Progress Estimation (similar to SARM)')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "rnd_progress_estimation.png", dpi=150)
    plt.close()
    
    # 4. Task Completion Detection Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Collect completion metrics from all episodes
    completion_frames = [results[ep].get('completion_frame', results[ep]['num_frames']-1) for ep in sorted(results.keys())]
    completion_pcts = [results[ep].get('completion_pct', 100.0) for ep in sorted(results.keys())]
    total_frames = [results[ep]['num_frames'] for ep in sorted(results.keys())]
    frames_after = [results[ep].get('frames_after_completion', 0) for ep in sorted(results.keys())]
    ep_indices = list(sorted(results.keys()))
    
    # Plot 1: Completion frame vs total frames
    ax1 = axes[0, 0]
    x = np.arange(len(ep_indices))
    width = 0.35
    bars1 = ax1.bar(x - width/2, completion_frames, width, label='Completion Frame', color='green', alpha=0.8)
    bars2 = ax1.bar(x + width/2, total_frames, width, label='Total Frames', color='gray', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Frame Number')
    ax1.set_title('Task Completion vs Episode Length')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ep_indices)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (cf, tf) in enumerate(zip(completion_frames, total_frames)):
        wasted = tf - cf
        ax1.annotate(f'{wasted}\nwasted', (i, tf), ha='center', va='bottom', fontsize=8, color='red')
    
    # Plot 2: Completion percentage distribution
    ax2 = axes[0, 1]
    ax2.bar(ep_indices, completion_pcts, color='blue', alpha=0.7)
    ax2.axhline(y=np.mean(completion_pcts), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(completion_pcts):.1f}%')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Completion %')
    ax2.set_title(f'Task Completes at {np.mean(completion_pcts):.0f}% of Episode (avg)')
    ax2.set_ylim(0, 110)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add text annotations
    for i, pct in enumerate(completion_pcts):
        ax2.text(ep_indices[i], pct + 2, f'{pct:.0f}%', ha='center', fontsize=8)
    
    # Plot 3: Gradient over time showing completion point
    ax3 = axes[1, 0]
    for ep_idx in sorted(results.keys())[:8]:  # First 8
        ep_data = results[ep_idx]
        frames = np.arange(ep_data['num_frames'])
        comp_frame = ep_data.get('completion_frame', ep_data['num_frames']-1)
        
        if 'smooth_gradient' in ep_data:
            ax3.plot(frames, ep_data['smooth_gradient'], alpha=0.6, label=f'Ep {ep_idx}')
            ax3.scatter([comp_frame], [ep_data['smooth_gradient'][comp_frame]], 
                       s=100, marker='o', zorder=5)  # Mark completion
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Smooth Gradient (rate of change)')
    ax3.set_title('Gradient → 0 = Task Complete (circles mark detection)')
    ax3.legend(loc='upper right', fontsize=7)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Compute summary statistics
    avg_completion_frame = np.mean(completion_frames)
    avg_total_frames = np.mean(total_frames)
    avg_wasted_frames = np.mean(frames_after)
    avg_wasted_pct = avg_wasted_frames / avg_total_frames * 100
    avg_completion_pct = np.mean(completion_pcts)
    
    summary_text = f"""
    TASK COMPLETION DETECTION SUMMARY
    ═══════════════════════════════════════
    
    Episodes Analyzed: {len(ep_indices)}
    
    Average Completion Frame: {avg_completion_frame:.0f}
    Average Episode Length: {avg_total_frames:.0f}
    
    Average Completion: {avg_completion_pct:.1f}% of episode
    Wasted Frames (avg): {avg_wasted_frames:.0f} ({avg_wasted_pct:.1f}%)
    
    ═══════════════════════════════════════
    
    REAL-TIME DETECTION PARAMETERS:
    • Gradient window: 30 frames (1 sec @ 30fps)
    • Stability threshold: 25th percentile of gradient
    • Consecutive frames needed: 45 (1.5 sec)
    • Minimum start frame: 150 (5 sec into task)
    
    ═══════════════════════════════════════
    
    USAGE IN REAL-TIME:
    1. Compute rolling gradient over 30-frame window
    2. When gradient < threshold for 45+ frames
       AND we're past 150 frames → TASK COMPLETE
    3. Can terminate episode early, saving {avg_wasted_pct:.0f}% time
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "rnd_end_detection_analysis.png", dpi=150)
    plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RND uncertainty on recorded dataset"
    )
    
    parser.add_argument(
        "--rnd-path",
        type=Path,
        required=True,
        help="Path to trained RND model (directory or .pth file)",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="Dataset repo ID (e.g., 'RAPOB/bimanual_transfer_pen_holder_50_no_aria')",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        required=True,
        help="Camera observation key (e.g., 'observation.images.top')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rnd_eval"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Specific episodes to evaluate (default: all)",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default=None,
        help="Policy path (required for policy-specific RND)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="v3.0",
        help="Dataset revision/tag",
    )
    
    args = parser.parse_args()
    
    evaluate_on_dataset(
        rnd_path=args.rnd_path,
        dataset_repo_id=args.dataset_repo_id,
        camera_key=args.camera_key,
        output_dir=args.output_dir,
        episodes=args.episodes,
        policy_path=args.policy_path,
        device=args.device,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
