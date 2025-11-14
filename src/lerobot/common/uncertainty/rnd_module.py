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
RND (Random Network Distillation) Module for Uncertainty Estimation

Implements Random Network Distillation for uncertainty estimation in robot policies.
Based on Burda et al., 2018 and adapted for SO-101 arms with ACT policy.

Reference:
    - Burda et al., 2018. "Exploration by Random Network Distillation."
    - "Uncertainty-Aware Failure Detection for Imitation Learning Robot Policies"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from pathlib import Path
from typing import Tuple, Optional


class RNDNetwork(nn.Module):
    """Simple MLP used for both the predictor and target networks."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class RNDModule(nn.Module):
    """
    Random Network Distillation (RND) for Uncertainty Estimation.

    - Uses ACT's trained ResNet backbone for image feature extraction.
    - Trains a predictor network to match outputs of a frozen, randomly
      initialized target network on successful (in-distribution) data.
    - Computes a per-timestep novelty score (L2 distance) and
      a rolling average over the last 10 timesteps.
    """

    def __init__(
        self,
        resnet_backbone: nn.Module,
        state_dim: int,
        action_dim: int,
        rnd_hidden_dim: int = 512,
        rnd_out_dim: int = 256,
        image_size: Tuple[int, int] = (96, 96),
        device: str = "cuda",
        rolling_window: int = 10
    ):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.image_size = image_size
        self.rolling_window = rolling_window

        # Use the same ResNet as ACT for image embedding
        self.resnet = resnet_backbone
        self.resnet.eval()  # freeze backbone
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Determine input dimensionality for RND nets
        resnet_feat_dim = self._get_resnet_output_dim()
        in_dim = resnet_feat_dim + state_dim + action_dim

        # Build target and predictor networks
        self.target = RNDNetwork(in_dim, rnd_hidden_dim, rnd_out_dim).to(device)
        self.predictor = RNDNetwork(in_dim, rnd_hidden_dim, rnd_out_dim).to(device)

        # Freeze target network
        for p in self.target.parameters():
            p.requires_grad = False

        # Rolling buffer for smoothing recent scores
        self.recent_scores = deque(maxlen=rolling_window)

        # Optimizer for training the predictor
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-4)

        # Statistics for normalization
        self.register_buffer('uncertainty_mean', torch.tensor(0.0))
        self.register_buffer('uncertainty_std', torch.tensor(1.0))

    def _get_resnet_output_dim(self) -> int:
        """Determine ResNet output dimension."""
        example_input = torch.zeros(1, 3, *self.image_size).to(self.device)
        with torch.no_grad():
            resnet_feat = self.resnet(example_input)
            # Handle different output types
            if isinstance(resnet_feat, dict):
                # If it's a dict, use the last feature or 'out' key
                resnet_feat = resnet_feat.get('out', list(resnet_feat.values())[-1])
            elif isinstance(resnet_feat, (list, tuple)):
                resnet_feat = resnet_feat[0]
        return resnet_feat.flatten(1).shape[1]

    def encode_inputs(
        self,
        obs_img: torch.Tensor,
        obs_state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Encodes (image, state, action) into flat feature vector."""
        with torch.no_grad():
            img_feat = self.resnet(obs_img)
            # Handle different output types
            if isinstance(img_feat, dict):
                img_feat = img_feat.get('out', list(img_feat.values())[-1])
            elif isinstance(img_feat, (list, tuple)):
                img_feat = img_feat[0]
        img_feat = img_feat.flatten(start_dim=1)
        x = torch.cat([img_feat, obs_state, action], dim=1)
        return x

    def train_on_dataset(self, dataloader, num_epochs: int = 200):
        """
        Trains the predictor network to match the frozen target on ID data.

        Args:
            dataloader: iterable yielding (obs_img, obs_state, action)
            num_epochs: number of training epochs (200 sim / 2000 real)
        """
        self.train()
        print(f"[RND] Training for {num_epochs} epochs on {len(dataloader)} batches...")

        all_uncertainties = []

        for epoch in range(num_epochs):
            total_loss = 0.0
            batch_count = 0

            for obs_img, obs_state, act in dataloader:
                obs_img = obs_img.to(self.device)
                obs_state = obs_state.to(self.device)
                act = act.to(self.device)

                # Prepare input
                x = self.encode_inputs(obs_img, obs_state, act)

                # Forward target (frozen)
                with torch.no_grad():
                    y_target = self.target(x)

                # Forward predictor
                y_pred = self.predictor(x)

                # Compute L2 loss
                loss = F.mse_loss(y_pred, y_target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # Collect uncertainties for normalization
                with torch.no_grad():
                    err = torch.sum((y_target - y_pred) ** 2, dim=1)
                    all_uncertainties.extend(err.cpu().numpy())

                # Print progress every 100 batches
                if batch_count % 100 == 0:
                    avg_loss_so_far = total_loss / batch_count
                    print(
                        f"  Epoch [{epoch+1}/{num_epochs}] - Batch [{batch_count}/{len(dataloader)}] - Loss: {avg_loss_so_far:.6f}")

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] COMPLETE - Avg Loss: {avg_loss:.6f}")

        # Compute normalization statistics
        if all_uncertainties:
            import numpy as np
            all_uncertainties = np.array(all_uncertainties)
            self.uncertainty_mean = torch.tensor(np.mean(all_uncertainties)).to(self.device)
            self.uncertainty_std = torch.tensor(np.std(all_uncertainties) + 1e-8).to(self.device)
            print(
                f"[RND] Uncertainty stats - Mean: {self.uncertainty_mean:.4f}, Std: {self.uncertainty_std:.4f}")

        print("[RND] Training complete.")

    @torch.no_grad()
    def compute_uncertainty(
        self,
        obs_img: torch.Tensor,
        obs_state: torch.Tensor,
        action: torch.Tensor,
        normalize: bool = True
    ) -> Tuple[float, float]:
        """
        Computes per-timestep RND novelty score and rolling average.

        Args:
            obs_img: Tensor [B, 3, H, W]
            obs_state: Tensor [B, state_dim]
            action: Tensor [B, action_dim]
            normalize: Whether to normalize uncertainty using training stats

        Returns:
            - step_uncertainty: per-timestep scalar novelty (mean across batch)
            - rolling_uncertainty: mean of last N novelty scores
        """
        self.eval()
        x = self.encode_inputs(
            obs_img.to(self.device),
            obs_state.to(self.device),
            action.to(self.device)
        )

        y_target = self.target(x)
        y_pred = self.predictor(x)

        # L2 error per-sample
        err = torch.sum((y_target - y_pred) ** 2, dim=1)
        step_uncertainty = err.mean().item()

        # Normalize if requested
        if normalize and self.uncertainty_std > 0:
            step_uncertainty = (step_uncertainty - self.uncertainty_mean.item()
                                ) / self.uncertainty_std.item()

        # Update rolling buffer
        self.recent_scores.append(step_uncertainty)
        rolling_uncertainty = sum(self.recent_scores) / len(self.recent_scores)

        return step_uncertainty, rolling_uncertainty

    def reset_rolling_buffer(self):
        """Reset the rolling uncertainty buffer."""
        self.recent_scores.clear()

    def save(self, path: Path):
        """Save RND module state."""
        torch.save({
            'predictor_state_dict': self.predictor.state_dict(),
            'target_state_dict': self.target.state_dict(),
            'uncertainty_mean': self.uncertainty_mean,
            'uncertainty_std': self.uncertainty_std,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'image_size': self.image_size,
        }, path)
        print(f"[RND] Saved to {path}")

    def load(self, path: Path):
        """Load RND module state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        self.target.load_state_dict(checkpoint['target_state_dict'])
        self.uncertainty_mean = checkpoint['uncertainty_mean'].to(self.device)
        self.uncertainty_std = checkpoint['uncertainty_std'].to(self.device)
        print(f"[RND] Loaded from {path}")
