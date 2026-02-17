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
Universal RND (Random Network Distillation) Module - Policy-Agnostic

This module uses a pretrained ImageNet backbone instead of extracting features
from a specific trained ACT policy. Benefits:

1. Train once, use for all policies on the same task/environment
2. No dependency on specific policy weights
3. Better for detecting environment-level novelty
4. Simpler deployment (one RND model per task type)

The tradeoff is less sensitivity to policy-specific uncertainties, but this
is often desirable for:
- Task completion detection (scene returns to known terminal state)
- Environment novelty (objects moved, new objects, lighting)
- Multi-policy switching (policy-agnostic progress estimation)

Reference:
    - Burda et al., 2018. "Exploration by Random Network Distillation."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from pathlib import Path
from typing import Tuple, Optional, Literal
import numpy as np


class RNDNetwork(nn.Module):
    """Simple MLP used for both the predictor and target networks."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class RNDModuleUniversal(nn.Module):
    """
    Universal RND Module - Policy-Agnostic Uncertainty Estimation.
    
    Uses a pretrained backbone (ResNet18 by default) instead of the policy's
    ResNet, making it reusable across different policies for the same task.
    
    Key differences from policy-specific RND:
    - Backbone is a standard pretrained model (ImageNet)
    - Can optionally include/exclude state and action in features
    - Better for detecting scene-level novelty vs action-level uncertainty
    
    Use cases:
    - Task completion detection
    - Environment change detection  
    - Multi-policy switching (policy-agnostic)
    - User distraction detection (same model works during any policy)
    """

    def __init__(
        self,
        backbone_type: Literal["resnet18", "resnet34", "resnet50"] = "resnet18",
        state_dim: int = 0,  # 0 = don't use state features
        action_dim: int = 0,  # 0 = don't use action features
        rnd_hidden_dim: int = 512,
        rnd_out_dim: int = 256,
        image_size: Tuple[int, int] = (96, 96),
        device: str = "cuda",
        rolling_window: int = 20,
        pretrained: bool = True,
    ):
        """
        Initialize Universal RND Module.
        
        Args:
            backbone_type: Which ResNet to use ("resnet18", "resnet34", "resnet50")
            state_dim: Dimension of state vector (0 = image only)
            action_dim: Dimension of action vector (0 = no actions)
            rnd_hidden_dim: Hidden dimension for RND networks
            rnd_out_dim: Output dimension for RND networks
            image_size: Expected input image size (H, W)
            device: Device to run on
            rolling_window: Window size for rolling uncertainty average
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.image_size = image_size
        self.rolling_window = rolling_window
        self.backbone_type = backbone_type

        # Load pretrained backbone
        self.backbone = self._create_backbone(backbone_type, pretrained)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Get backbone output dimension
        backbone_feat_dim = self._get_backbone_output_dim()
        in_dim = backbone_feat_dim + state_dim + action_dim
        
        print(f"[RND Universal] Backbone: {backbone_type}, features: {backbone_feat_dim}")
        print(f"[RND Universal] State dim: {state_dim}, Action dim: {action_dim}")
        print(f"[RND Universal] Total input dim: {in_dim}")

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
        
        # Move to device
        self.to(device)

    def _create_backbone(self, backbone_type: str, pretrained: bool) -> nn.Module:
        """Create pretrained backbone network."""
        from torchvision import models
        from torchvision.models.feature_extraction import create_feature_extractor
        
        if backbone_type == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet18(weights=weights)
        elif backbone_type == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet34(weights=weights)
        elif backbone_type == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
        
        # Extract features before final FC layer
        backbone = create_feature_extractor(base_model, return_nodes={"avgpool": "features"})
        return backbone.to(self.device)

    def _get_backbone_output_dim(self) -> int:
        """Determine backbone output dimension."""
        example_input = torch.zeros(1, 3, *self.image_size).to(self.device)
        with torch.no_grad():
            output = self.backbone(example_input)
            features = output["features"]
        return features.flatten(1).shape[1]

    def encode_inputs(
        self,
        obs_img: torch.Tensor,
        obs_state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode (image, optional state, optional action) into flat feature vector.
        
        Args:
            obs_img: Image tensor [B, 3, H, W]
            obs_state: Optional state tensor [B, state_dim]
            action: Optional action tensor [B, action_dim]
        
        Returns:
            Concatenated feature vector
        """
        with torch.no_grad():
            output = self.backbone(obs_img)
            img_feat = output["features"]
        
        img_feat = img_feat.flatten(start_dim=1)
        
        # Concatenate optional features
        features = [img_feat]
        if obs_state is not None and self.state_dim > 0:
            features.append(obs_state)
        if action is not None and self.action_dim > 0:
            features.append(action)
        
        return torch.cat(features, dim=1)

    def train_on_dataset(self, dataloader, num_epochs: int = 200):
        """
        Train the predictor network to match the frozen target on ID data.

        Args:
            dataloader: Iterable yielding (obs_img,) or (obs_img, obs_state, action)
            num_epochs: Number of training epochs
        """
        self.train()
        print(f"[RND Universal] Training for {num_epochs} epochs on {len(dataloader)} batches...")

        for epoch in range(num_epochs):
            total_loss = 0.0
            batch_count = 0

            for batch in dataloader:
                # Handle variable batch contents
                if len(batch) == 1:
                    obs_img = batch[0].to(self.device)
                    obs_state, action = None, None
                elif len(batch) == 3:
                    obs_img, obs_state, action = batch
                    obs_img = obs_img.to(self.device)
                    obs_state = obs_state.to(self.device) if self.state_dim > 0 else None
                    action = action.to(self.device) if self.action_dim > 0 else None
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")

                # Encode inputs
                x = self.encode_inputs(obs_img, obs_state, action)

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

                if batch_count % 100 == 0:
                    avg_loss_so_far = total_loss / batch_count
                    print(f"  Epoch [{epoch+1}/{num_epochs}] - Batch [{batch_count}] - Loss: {avg_loss_so_far:.6f}")

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] COMPLETE - Avg Loss: {avg_loss:.6f}")

        # Compute normalization statistics
        self._compute_uncertainty_stats(dataloader)
        print("[RND Universal] Training complete.")

    def _compute_uncertainty_stats(self, dataloader):
        """Compute uncertainty statistics for normalization."""
        print("[RND Universal] Computing uncertainty statistics...")
        all_uncertainties = []

        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 1:
                    obs_img = batch[0].to(self.device)
                    obs_state, action = None, None
                else:
                    obs_img, obs_state, action = batch
                    obs_img = obs_img.to(self.device)
                    obs_state = obs_state.to(self.device) if self.state_dim > 0 else None
                    action = action.to(self.device) if self.action_dim > 0 else None

                x = self.encode_inputs(obs_img, obs_state, action)
                y_target = self.target(x)
                y_pred = self.predictor(x)

                err = torch.sum((y_target - y_pred) ** 2, dim=1)
                all_uncertainties.extend(err.cpu().numpy())

        if all_uncertainties:
            all_uncertainties = np.array(all_uncertainties)
            self.uncertainty_mean = torch.tensor(np.mean(all_uncertainties)).to(self.device)
            self.uncertainty_std = torch.tensor(np.std(all_uncertainties) + 1e-8).to(self.device)
            print(f"[RND Universal] Stats - Mean: {self.uncertainty_mean:.4f}, Std: {self.uncertainty_std:.4f}")

    @torch.no_grad()
    def compute_uncertainty(
        self,
        obs_img: torch.Tensor,
        obs_state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        normalize: bool = False
    ) -> Tuple[float, float]:
        """
        Compute per-timestep RND novelty score and rolling average.

        Args:
            obs_img: Image tensor [B, 3, H, W]
            obs_state: Optional state tensor [B, state_dim]
            action: Optional action tensor [B, action_dim]
            normalize: Whether to normalize using training stats

        Returns:
            - step_uncertainty: Per-timestep novelty score
            - rolling_uncertainty: Rolling average over window
        """
        self.eval()
        x = self.encode_inputs(
            obs_img.to(self.device),
            obs_state.to(self.device) if obs_state is not None else None,
            action.to(self.device) if action is not None else None
        )

        y_target = self.target(x)
        y_pred = self.predictor(x)

        err = torch.sum((y_target - y_pred) ** 2, dim=1)
        step_uncertainty = err.mean().item()

        if normalize and self.uncertainty_std > 1e-6:
            step_uncertainty = (step_uncertainty - self.uncertainty_mean.item()) / self.uncertainty_std.item()

        self.recent_scores.append(step_uncertainty)
        rolling_uncertainty = sum(self.recent_scores) / len(self.recent_scores)

        return step_uncertainty, rolling_uncertainty

    def reset_rolling_buffer(self):
        """Reset the rolling uncertainty buffer."""
        self.recent_scores.clear()

    def save(self, path: Path):
        """Save RND module state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'predictor_state_dict': self.predictor.state_dict(),
            'target_state_dict': self.target.state_dict(),
            'backbone_state_dict': self.backbone.state_dict(),
            'backbone_type': self.backbone_type,
            'uncertainty_mean': self.uncertainty_mean,
            'uncertainty_std': self.uncertainty_std,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'image_size': self.image_size,
            'rolling_window': self.rolling_window,
        }, path)
        print(f"[RND Universal] Saved to {path}")

    @classmethod
    def load(cls, path: Path, device: str = "cuda") -> "RNDModuleUniversal":
        """Load RND module from checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Create module with saved config
        module = cls(
            backbone_type=checkpoint.get('backbone_type', 'resnet18'),
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim'],
            image_size=checkpoint['image_size'],
            device=device,
            rolling_window=checkpoint.get('rolling_window', 20),
            pretrained=False,  # We'll load the weights
        )
        
        # Load weights
        module.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        module.target.load_state_dict(checkpoint['target_state_dict'])
        if 'backbone_state_dict' in checkpoint:
            module.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        module.uncertainty_mean = checkpoint['uncertainty_mean'].to(device)
        module.uncertainty_std = checkpoint['uncertainty_std'].to(device)
        
        print(f"[RND Universal] Loaded from {path}")
        return module


class TaskEndDetector(nn.Module):
    """
    Combines RND uncertainty with action variance to detect task/episode completion.
    
    Detection heuristics:
    1. High RND uncertainty (state not seen during mid-task training)
    2. Low action variance (robot hovering/holding position)
    3. Sustained pattern (not just momentary spike)
    
    This is particularly useful for:
    - Automatic policy switching when current task is complete
    - Detecting when to stop recording an episode
    - Triggering human handover at task boundaries
    """

    def __init__(
        self,
        rnd_module: RNDModuleUniversal,
        uncertainty_threshold: float = 2.0,
        action_variance_threshold: float = 0.01,
        min_sustained_frames: int = 10,
        device: str = "cuda",
    ):
        """
        Initialize Task End Detector.
        
        Args:
            rnd_module: Trained RND module (universal or policy-specific)
            uncertainty_threshold: Normalized uncertainty above this = OOD
            action_variance_threshold: Action variance below this = stationary
            min_sustained_frames: Consecutive frames needed to trigger detection
            device: Device to run on
        """
        super().__init__()
        self.rnd = rnd_module
        self.uncertainty_threshold = uncertainty_threshold
        self.action_variance_threshold = action_variance_threshold
        self.min_sustained_frames = min_sustained_frames
        self.device = device
        
        # State tracking
        self.consecutive_end_frames = 0
        self.action_history = deque(maxlen=10)
        self._task_end_detected = False

    def reset(self):
        """Reset detector state for new episode."""
        self.consecutive_end_frames = 0
        self.action_history.clear()
        self._task_end_detected = False
        self.rnd.reset_rolling_buffer()

    @torch.no_grad()
    def update(
        self,
        obs_img: torch.Tensor,
        action: torch.Tensor,
        obs_state: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Update detector with new observation and action.
        
        Args:
            obs_img: Current image observation
            action: Current/predicted action
            obs_state: Optional state observation
        
        Returns:
            Dict with detection results:
                - task_end_detected: bool
                - uncertainty: float (normalized)
                - action_variance: float
                - consecutive_frames: int
        """
        # Compute RND uncertainty
        step_unc, rolling_unc = self.rnd.compute_uncertainty(
            obs_img, obs_state, action, normalize=True
        )
        
        # Track action variance
        self.action_history.append(action.cpu().numpy().flatten())
        if len(self.action_history) >= 3:
            actions = np.array(list(self.action_history))
            action_variance = np.mean(np.var(actions, axis=0))
        else:
            action_variance = float('inf')
        
        # Check end conditions
        high_uncertainty = rolling_unc > self.uncertainty_threshold
        low_action_variance = action_variance < self.action_variance_threshold
        
        if high_uncertainty or low_action_variance:
            self.consecutive_end_frames += 1
        else:
            self.consecutive_end_frames = 0
        
        # Detect task end
        if self.consecutive_end_frames >= self.min_sustained_frames:
            self._task_end_detected = True
        
        return {
            'task_end_detected': self._task_end_detected,
            'uncertainty': rolling_unc,
            'action_variance': action_variance,
            'consecutive_frames': self.consecutive_end_frames,
            'high_uncertainty': high_uncertainty,
            'low_action_variance': low_action_variance,
        }

    @property
    def task_end_detected(self) -> bool:
        return self._task_end_detected
