#!/usr/bin/env python3

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
Train Universal RND Module - Policy-Agnostic Uncertainty Estimation

This script trains an RND module using a pretrained ImageNet backbone,
making it reusable across different policies for the same task/environment.

Benefits over policy-specific RND:
- No dependency on specific policy weights
- Train once, use for all policies
- Better for environment-level novelty detection
- Simpler deployment

Optionally, you can train on "mid-task" frames only (excluding first/last N%)
to make the RND better at detecting task start/end states.

Usage:
    # Basic training (all frames)
    python -m lerobot.scripts.train_rnd_universal \
        --dataset-repo-id RAPOB/my_dataset \
        --camera-key observation.images.top \
        --output-dir outputs/rnd_universal/my_task

    # Train on mid-task frames only (better for task end detection)
    python -m lerobot.scripts.train_rnd_universal \
        --dataset-repo-id RAPOB/my_dataset \
        --camera-key observation.images.top \
        --output-dir outputs/rnd_universal/my_task \
        --exclude-start-pct 0.1 \
        --exclude-end-pct 0.1

    # Image-only mode (no state/action features)
    python -m lerobot.scripts.train_rnd_universal \
        --dataset-repo-id RAPOB/my_dataset \
        --camera-key observation.images.top \
        --output-dir outputs/rnd_universal/my_task \
        --image-only
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms

from lerobot.common.uncertainty.rnd_module_universal import RNDModuleUniversal
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_image_transforms(image_size: tuple[int, int]) -> transforms.Compose:
    """Create image transforms for RND training."""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.Resize(image_size, antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
    ])


class UniversalRNDDataset(Dataset):
    """
    Dataset wrapper for training Universal RND.
    
    Supports:
    - Image-only mode (for maximum generality)
    - Image + state + action mode (for better uncertainty estimation)
    - Mid-task frame filtering (exclude start/end of episodes)
    """

    def __init__(
        self,
        lerobot_dataset: LeRobotDataset,
        camera_key: str,
        include_state: bool = True,
        include_action: bool = True,
        exclude_start_pct: float = 0.0,
        exclude_end_pct: float = 0.0,
        image_size: tuple[int, int] = (96, 96),
    ):
        """
        Initialize dataset.
        
        Args:
            lerobot_dataset: Source LeRobot dataset
            camera_key: Camera observation key (e.g., "observation.images.top")
            include_state: Whether to include state features
            include_action: Whether to include action features
            exclude_start_pct: Fraction of episode start to exclude (0-0.5)
            exclude_end_pct: Fraction of episode end to exclude (0-0.5)
            image_size: Target image size for transforms
        """
        self.dataset = lerobot_dataset
        self.camera_key = camera_key
        self.include_state = include_state
        self.include_action = include_action
        self.transforms = make_image_transforms(image_size)
        
        # Validate camera key
        if camera_key not in lerobot_dataset.meta.camera_keys:
            available = ", ".join(lerobot_dataset.meta.camera_keys)
            raise ValueError(f"Camera key '{camera_key}' not found. Available: {available}")
        
        # Find state and action keys
        sample = lerobot_dataset[0]
        self.state_keys = [k for k in sample.keys() if "state" in k and "action" not in k]
        self.action_keys = [k for k in sample.keys() if "action" in k]
        
        if not self.state_keys and include_state:
            # Fallback to position keys
            self.state_keys = [k for k in sample.keys() if "pos" in k or "position" in k]
        
        logger.info(f"Camera key: {camera_key}")
        logger.info(f"State keys: {self.state_keys}")
        logger.info(f"Action keys: {self.action_keys}")
        
        # Compute dimensions
        if include_state and self.state_keys:
            self.state_dim = sum(sample[k].shape[0] for k in self.state_keys if k in sample)
        else:
            self.state_dim = 0
            
        if include_action and self.action_keys:
            self.action_dim = sample[self.action_keys[0]].shape[0]
        else:
            self.action_dim = 0
        
        logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        
        # Build valid indices (filtered by episode position)
        self.valid_indices = self._build_valid_indices(exclude_start_pct, exclude_end_pct)
        logger.info(f"Total frames: {len(lerobot_dataset)}, Valid frames: {len(self.valid_indices)}")

    def _build_valid_indices(self, exclude_start_pct: float, exclude_end_pct: float) -> list[int]:
        """Build list of valid frame indices, excluding episode start/end."""
        if exclude_start_pct == 0 and exclude_end_pct == 0:
            return list(range(len(self.dataset)))
        
        valid = []
        
        # Get episode boundaries - optimized O(n) approach
        episode_indices = list(self.dataset.hf_dataset["episode_index"])
        n_frames = len(episode_indices)
        
        # Build episode start/end boundaries in one pass
        ep_boundaries = {}  # ep_idx -> (start_frame, end_frame)
        current_ep = episode_indices[0] if n_frames > 0 else None
        ep_start = 0
        
        for i in range(n_frames):
            if episode_indices[i] != current_ep:
                # End of previous episode
                ep_boundaries[current_ep] = (ep_start, i)
                current_ep = episode_indices[i]
                ep_start = i
        
        # Don't forget the last episode
        if current_ep is not None:
            ep_boundaries[current_ep] = (ep_start, n_frames)
        
        logger.info(f"Found {len(ep_boundaries)} episodes")
        
        # Now filter frames for each episode
        for ep_idx, (ep_start, ep_end) in ep_boundaries.items():
            ep_len = ep_end - ep_start
            start_exclude = int(ep_len * exclude_start_pct)
            end_exclude = int(ep_len * exclude_end_pct)
            
            # Keep middle portion
            keep_start = ep_start + start_exclude
            keep_end = ep_end - end_exclude
            
            if keep_start < keep_end:
                valid.extend(range(keep_start, keep_end))
        
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        item = self.dataset[actual_idx]
        
        # Get image
        image = item[self.camera_key]
        if self.transforms:
            image = self.transforms(image)
        
        # Get state (if requested)
        if self.include_state and self.state_keys:
            state_parts = [item[k] for k in self.state_keys if k in item]
            state = torch.cat(state_parts) if len(state_parts) > 1 else state_parts[0]
        else:
            state = torch.zeros(0)
        
        # Get action (if requested)
        if self.include_action and self.action_keys:
            action = item[self.action_keys[0]]
        else:
            action = torch.zeros(0)
        
        return image, state, action


def train_rnd_universal(
    dataset_repo_id: str,
    camera_key: str,
    output_dir: Path,
    backbone_type: str = "resnet18",
    num_epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    image_size: tuple[int, int] = (96, 96),
    device: str = "cuda",
    num_workers: int = 4,
    image_only: bool = False,
    exclude_start_pct: float = 0.0,
    exclude_end_pct: float = 0.0,
):
    """
    Train Universal RND module on demonstrations.
    
    Args:
        dataset_repo_id: HuggingFace dataset ID
        camera_key: Camera observation key
        output_dir: Directory to save trained model
        backbone_type: ResNet backbone ("resnet18", "resnet34", "resnet50")
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        image_size: Target image size
        device: Device to use
        num_workers: DataLoader workers
        image_only: If True, only use image features (no state/action)
        exclude_start_pct: Fraction of episode start to exclude
        exclude_end_pct: Fraction of episode end to exclude
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_repo_id}")
    lerobot_dataset = LeRobotDataset(dataset_repo_id)
    
    # Create RND dataset
    rnd_dataset = UniversalRNDDataset(
        lerobot_dataset=lerobot_dataset,
        camera_key=camera_key,
        include_state=not image_only,
        include_action=not image_only,
        exclude_start_pct=exclude_start_pct,
        exclude_end_pct=exclude_end_pct,
        image_size=image_size,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        rnd_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Create RND module
    logger.info(f"Creating Universal RND with {backbone_type} backbone")
    rnd = RNDModuleUniversal(
        backbone_type=backbone_type,
        state_dim=rnd_dataset.state_dim,
        action_dim=rnd_dataset.action_dim,
        image_size=image_size,
        device=device,
    )
    
    # Update optimizer learning rate
    rnd.optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=learning_rate)
    
    # Train
    logger.info(f"Starting training for {num_epochs} epochs")
    rnd.train_on_dataset(dataloader, num_epochs=num_epochs)
    
    # Compute threshold statistics
    logger.info("Computing uncertainty threshold...")
    all_uncertainties = []
    rnd.eval()
    
    with torch.no_grad():
        for batch_idx, (obs_img, obs_state, action) in enumerate(dataloader):
            if batch_idx >= 100:  # Sample 100 batches
                break
            obs_img = obs_img.to(device)
            obs_state = obs_state.to(device) if rnd_dataset.state_dim > 0 else None
            action = action.to(device) if rnd_dataset.action_dim > 0 else None
            
            step_unc, _ = rnd.compute_uncertainty(obs_img, obs_state, action, normalize=False)
            all_uncertainties.append(step_unc)
    
    uncertainty_mean = float(np.mean(all_uncertainties))
    uncertainty_std = float(np.std(all_uncertainties))
    uncertainty_threshold = uncertainty_mean + 2.0 * uncertainty_std
    
    logger.info(f"Uncertainty - Mean: {uncertainty_mean:.4f}, Std: {uncertainty_std:.4f}")
    logger.info(f"Threshold (mean + 2*std): {uncertainty_threshold:.4f}")
    
    # Save model
    output_path = output_dir / "rnd_universal.pth"
    rnd.save(output_path)
    
    # Save training info
    info = {
        "dataset_repo_id": dataset_repo_id,
        "camera_key": camera_key,
        "backbone_type": backbone_type,
        "image_only": image_only,
        "state_dim": rnd_dataset.state_dim,
        "action_dim": rnd_dataset.action_dim,
        "image_size": list(image_size),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "exclude_start_pct": exclude_start_pct,
        "exclude_end_pct": exclude_end_pct,
        "total_frames": len(lerobot_dataset),
        "valid_frames": len(rnd_dataset),
        "uncertainty_mean": uncertainty_mean,
        "uncertainty_std": uncertainty_std,
        "uncertainty_threshold": uncertainty_threshold,
    }
    
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Model saved to {output_path}")
    logger.info("Training complete!")
    
    return rnd


def main():
    parser = argparse.ArgumentParser(
        description="Train Universal RND module for policy-agnostic uncertainty estimation"
    )
    
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="Dataset repo ID (e.g., 'RAPOB/my_dataset')",
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
        default=Path("outputs/rnd_universal"),
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--backbone-type",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="ResNet backbone type",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[96, 96],
        help="Target image size (height width)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Use image features only (no state/action)",
    )
    parser.add_argument(
        "--exclude-start-pct",
        type=float,
        default=0.0,
        help="Fraction of episode start to exclude (0-0.5). Use for task-end detection training.",
    )
    parser.add_argument(
        "--exclude-end-pct",
        type=float,
        default=0.0,
        help="Fraction of episode end to exclude (0-0.5). Use for task-end detection training.",
    )
    
    args = parser.parse_args()
    
    train_rnd_universal(
        dataset_repo_id=args.dataset_repo_id,
        camera_key=args.camera_key,
        output_dir=args.output_dir,
        backbone_type=args.backbone_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=tuple(args.image_size),
        device=args.device,
        num_workers=args.num_workers,
        image_only=args.image_only,
        exclude_start_pct=args.exclude_start_pct,
        exclude_end_pct=args.exclude_end_pct,
    )


if __name__ == "__main__":
    main()
