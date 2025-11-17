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
Train RND (Random Network Distillation) module for uncertainty estimation.

This script trains an RND predictor network on successful demonstrations from a LeRobot dataset.
The RND module can then be used during deployment to detect out-of-distribution states.

Usage with local policy:
    python lerobot/scripts/train_rnd.py \
        --policy-path outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model \
        --dataset-repo-id lerobot/so101_pick_place \
        --output-dir outputs/rnd/so101_pick_place \
        --num-epochs 200 \
        --batch-size 32

Usage with HuggingFace policy:
    python lerobot/scripts/train_rnd.py \
        --policy-path your_username/act_so101_pick_place \
        --dataset-repo-id lerobot/so101_pick_place \
        --output-dir outputs/rnd/so101_pick_place \
        --num-epochs 200 \
        --batch-size 32
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.uncertainty import RNDModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_image_transforms(image_size: tuple[int, int]) -> transforms.Compose:
    """Create image transforms matching those used during policy training."""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.Resize(image_size, antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
    ])


def load_policy_backbone(policy_path: str | Path, device: str) -> tuple[torch.nn.Module, PreTrainedConfig]:
    """Load the ResNet backbone from a trained policy (local or HuggingFace)."""
    logger.info(f"Loading policy from {policy_path}")

    # Load policy config (handles both local and HF repos)
    policy_config = PreTrainedConfig.from_pretrained(policy_path)

    # Create a minimal dataset metadata for policy initialization
    # This is needed because make_policy expects dataset metadata
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    # We'll load the actual dataset later, but need minimal meta for policy creation
    temp_dataset = LeRobotDataset(policy_config.dataset_repo_id if hasattr(
        policy_config, 'dataset_repo_id') else "lerobot/pusht")

    # Load the full policy using make_policy (same as record.py)
    policy = make_policy(policy_config, ds_meta=temp_dataset.meta)
    policy = policy.to(device)
    policy.eval()

    # Extract ResNet encoder - ACT policies have it in different places
    if hasattr(policy, 'model') and hasattr(policy.model, 'backbone'):
        resnet = policy.model.backbone
    elif hasattr(policy, 'backbone'):
        resnet = policy.backbone
    else:
        raise AttributeError("Could not find ResNet backbone in policy")

    logger.info("Successfully loaded policy backbone")
    return resnet, policy_config


class RNDDataset(torch.utils.data.Dataset):
    """Wrapper around LeRobotDataset for RND training."""

    def __init__(self, lerobot_dataset: LeRobotDataset):
        self.dataset = lerobot_dataset
        self.camera_keys = lerobot_dataset.meta.camera_keys
        # Use first camera if multiple available
        self.camera_key = self.camera_keys[0] if self.camera_keys else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Extract image (use first camera)
        if self.camera_key:
            image = item[self.camera_key]
        else:
            raise ValueError("No camera keys found in dataset")

        # Extract state (joint positions)
        state_keys = [k for k in item.keys() if 'state' in k and 'action' not in k]
        if state_keys:
            state = item[state_keys[0]]
        else:
            # Fallback: try to find position-related keys
            state_keys = [k for k in item.keys() if 'pos' in k or 'position' in k]
            if state_keys:
                state = torch.cat([item[k] for k in state_keys])
            else:
                raise ValueError("Could not find state in dataset")

        # Extract action
        action_keys = [k for k in item.keys() if 'action' in k]
        if action_keys:
            action = item[action_keys[0]]
        else:
            raise ValueError("Could not find action in dataset")

        return image, state, action


def train_rnd(
    policy_path: str | Path,
    dataset_repo_id: str,
    output_dir: Path,
    num_epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    image_size: tuple[int, int] = (96, 96),
    device: str = "cuda",
    num_workers: int = 4,
):
    """Train RND module on successful demonstrations."""

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load policy backbone (same way as record.py)
    resnet, policy_config = load_policy_backbone(policy_path, device)

    # Load dataset
    logger.info(f"Loading dataset: {dataset_repo_id}")
    image_transforms = make_image_transforms(image_size)
    lerobot_dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        image_transforms=image_transforms,
    )

    # Determine dimensions from dataset
    sample = lerobot_dataset[0]
    state_keys = [k for k in sample.keys() if 'state' in k and 'action' not in k]
    action_keys = [k for k in sample.keys() if 'action' in k]

    if state_keys:
        state_dim = sample[state_keys[0]].shape[0]
    else:
        # Fallback
        state_keys = [k for k in sample.keys() if 'pos' in k or 'position' in k]
        state_dim = sum(sample[k].shape[0] for k in state_keys)

    action_dim = sample[action_keys[0]].shape[0]

    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create RND module
    logger.info("Initializing RND module")
    rnd = RNDModule(
        resnet_backbone=resnet,
        state_dim=state_dim,
        action_dim=action_dim,
        image_size=image_size,
        device=str(device),
    )

    # Create dataloader
    rnd_dataset = RNDDataset(lerobot_dataset)
    dataloader = DataLoader(
        rnd_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Train RND
    logger.info(f"Starting RND training for {num_epochs} epochs")
    rnd.train_on_dataset(dataloader, num_epochs=num_epochs)

    # Save trained RND model
    output_path = output_dir / 'rnd_model.pth'
    torch.save({
        'predictor_state_dict': rnd.predictor.state_dict(),
        'target_state_dict': rnd.target.state_dict(),
        'resnet_state_dict': rnd.resnet.state_dict(),  # Save ResNet backbone
        'state_dim': state_dim,
        'action_dim': action_dim,
        'image_size': tuple(image_size),
    }, output_path)
    logger.info(f"RND model saved to {output_path}")

    # Save training info
    info = {
        "dataset_repo_id": dataset_repo_id,
        "policy_path": str(policy_path),
        "policy_type": policy_config.type if hasattr(policy_config, 'type') else "unknown",
        "state_dim": state_dim,
        "action_dim": action_dim,
        "image_size": image_size,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    import json
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info("RND training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train RND module for uncertainty estimation")
    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Path to trained policy (local dir or HuggingFace repo like 'username/model_name')",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="Dataset repo ID (e.g., 'username/dataset_name' or 'lerobot/pusht')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rnd"),
        help="Directory to save trained RND model",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=200,
        help="Number of training epochs (200 for sim, 2000 for real)",
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
        help="Learning rate for predictor network",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[96, 96],
        help="Image size (height width)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )

    args = parser.parse_args()

    train_rnd(
        policy_path=args.policy_path,
        dataset_repo_id=args.dataset_repo_id,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=tuple(args.image_size),
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
