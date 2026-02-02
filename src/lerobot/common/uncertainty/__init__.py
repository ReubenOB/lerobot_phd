#!/usr/bin/env python3

"""
__init__.py for lerobot.common.uncertainty module

This module provides uncertainty estimation for robot policies using
Random Network Distillation (RND).

Two implementations are available:

1. RNDModule (Policy-Specific):
   - Uses the ResNet backbone from a trained ACT policy
   - More sensitive to policy-specific uncertainties
   - Requires re-training for each policy
   - Better for detecting when a specific policy is uncertain

2. RNDModuleUniversal (Policy-Agnostic):
   - Uses a pretrained ImageNet backbone
   - Train once, use for all policies on same task
   - Better for environment-level novelty detection
   - Better for task completion detection

Additionally, TaskEndDetector combines RND with action variance
to detect task/episode completion.
"""

from .rnd_module import RNDModule, RNDNetwork
from .rnd_module_universal import RNDModuleUniversal, TaskEndDetector

__all__ = [
    'RNDModule',
    'RNDNetwork',
    'RNDModuleUniversal',
    'TaskEndDetector',
]
