#!/usr/bin/env python3
"""
Build DEP (Decoupled Evaluation Pipeline) datasets.

Creates 12 datasets from source recordings:
  - 6 "all_aria" variants (all 4 aria streams + robot cameras)
  - 6 "no_aria" variants (robot cameras only, no aria streams)

Source datasets are combined using aggregate_datasets, then stripped
of unwanted camera streams for the no_aria variants.

Usage:
  # Inside Docker container with lerobot venv:
  source /opt/lerobot_venv/bin/activate
  python3 files/build_dep_datasets.py

  # Build only specific groups:
  python3 files/build_dep_datasets.py --groups pod_gold cup_blue

  # Dry run (just show what would be built):
  python3 files/build_dep_datasets.py --dry-run
"""

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_tools import modify_features
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Aria streams to REMOVE when creating no_aria variants
ARIA_STREAMS_TO_REMOVE = [
    "observation.images.aria_gaussian_blur",
    "observation.images.aria_brightness_boost",
    "observation.images.aria_gaussian_attention",
    "observation.images.aria_hard_cutout",
]

ORG = "RAPOB"


@dataclass
class DatasetGroup:
    name: str
    sources: list[str]
    expected_episodes: int
    all_aria_repo: str
    no_aria_repo: str


GROUPS = [
    DatasetGroup(
        name="pod_gold",
        sources=[
            "RAPOB/coffee_bimanual_aria_all_gold_2",
            "RAPOB/coffee_bimanual_aria_all_gold_v2",
        ],
        expected_episodes=40,
        all_aria_repo=f"{ORG}/dep_coffee_pod_gold_all_aria",
        no_aria_repo=f"{ORG}/dep_coffee_pod_gold_no_aria",
    ),
    DatasetGroup(
        name="pod_red",
        sources=[
            "RAPOB/coffee_bimanual_aria_all_red_3",
            "RAPOB/coffee_bimanual_aria_all_red_v2_10_a",
        ],
        expected_episodes=30,
        all_aria_repo=f"{ORG}/dep_coffee_pod_red_all_aria",
        no_aria_repo=f"{ORG}/dep_coffee_pod_red_no_aria",
    ),
    DatasetGroup(
        name="pod_green",
        sources=[
            "RAPOB/coffee_bimanual_aria_all_green_1_reconstructed",
            "RAPOB/coffee_bimanual_aria_all_green_v2_5",
        ],
        expected_episodes=25,
        all_aria_repo=f"{ORG}/dep_coffee_pod_green_all_aria",
        no_aria_repo=f"{ORG}/dep_coffee_pod_green_no_aria",
    ),
    DatasetGroup(
        name="cup_blue",
        sources=[
            "RAPOB/coffee_bimanual_aria_stage_2_all_blue_b",
            "RAPOB/coffee_bimanual_aria_stage_2_all_blue_v2_10",
        ],
        expected_episodes=30,
        all_aria_repo=f"{ORG}/dep_coffee_cup_blue_all_aria",
        no_aria_repo=f"{ORG}/dep_coffee_cup_blue_no_aria",
    ),
    DatasetGroup(
        name="cup_red",
        sources=[
            "RAPOB/coffee_bimanual_aria_stage_2_all_red",
        ],
        expected_episodes=20,
        all_aria_repo=f"{ORG}/dep_coffee_cup_red_all_aria",
        no_aria_repo=f"{ORG}/dep_coffee_cup_red_no_aria",
    ),
    DatasetGroup(
        name="cup_green",
        sources=[
            "RAPOB/coffee_bimanual_aria_stage_2_all_green_a_combine",
            "RAPOB/coffee_bimanual_aria_stage_2_all_green_v2_5",
        ],
        expected_episodes=25,
        all_aria_repo=f"{ORG}/dep_coffee_cup_green_all_aria",
        no_aria_repo=f"{ORG}/dep_coffee_cup_green_no_aria",
    ),
]


def get_root() -> Path:
    """Get the dataset cache root directory."""
    return Path(HF_LEROBOT_HOME)


def build_all_aria(group: DatasetGroup, root: Path) -> LeRobotDataset:
    """Aggregate source datasets into a single all_aria dataset."""
    output_dir = root / group.all_aria_repo
    if output_dir.exists():
        log.info(f"Cleaning previous build: {output_dir}")
        shutil.rmtree(output_dir)

    log.info(f"[{group.name}] Aggregating {len(group.sources)} sources → {group.all_aria_repo}")
    for src in group.sources:
        log.info(f"  Source: {src}")

    if len(group.sources) == 1:
        # Single source — just load it directly (no aggregation needed)
        log.info(f"  Single source, loading directly as {group.all_aria_repo}")
        ds = LeRobotDataset(group.sources[0], root=root)
        # Copy to output location
        src_dir = root / group.sources[0]
        shutil.copytree(src_dir, output_dir)
        return LeRobotDataset(group.all_aria_repo, root=root)
    else:
        aggregate_datasets(
            repo_ids=group.sources,
            aggr_repo_id=group.all_aria_repo,
            roots=[root] * len(group.sources),
            aggr_root=root,
        )
        return LeRobotDataset(group.all_aria_repo, root=root)


def build_no_aria(group: DatasetGroup, root: Path) -> LeRobotDataset:
    """Create no_aria variant by removing all aria streams from the all_aria dataset."""
    output_dir = root / group.no_aria_repo
    if output_dir.exists():
        log.info(f"Cleaning previous build: {output_dir}")
        shutil.rmtree(output_dir)

    log.info(f"[{group.name}] Creating no_aria variant: {group.no_aria_repo}")

    # Load the all_aria dataset
    all_aria_ds = LeRobotDataset(group.all_aria_repo, root=root)

    # Figure out which streams actually exist in the dataset
    existing_camera_keys = set(all_aria_ds.meta.camera_keys)
    streams_to_remove = [s for s in ARIA_STREAMS_TO_REMOVE if s in existing_camera_keys]

    if not streams_to_remove:
        log.warning(f"  No aria streams found to remove. Keys: {existing_camera_keys}")
        # Just copy the all_aria dataset
        shutil.copytree(root / group.all_aria_repo, output_dir)
        return LeRobotDataset(group.no_aria_repo, root=root)

    log.info(f"  Removing streams: {streams_to_remove}")

    result_ds = modify_features(
        dataset=all_aria_ds,
        remove_features=streams_to_remove,
        output_dir=output_dir,
        repo_id=group.no_aria_repo,
    )

    return result_ds


def upload_dataset(repo_id: str, root: Path) -> None:
    """Upload dataset to HuggingFace Hub."""
    log.info(f"Uploading {repo_id} to HuggingFace Hub...")
    create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    ds = LeRobotDataset(repo_id, root=root)
    ds.push_to_hub(private=True)
    log.info(f"  Uploaded: https://huggingface.co/datasets/{repo_id}")


def main():
    parser_arg = argparse.ArgumentParser(description="Build DEP datasets")
    parser_arg.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Specific groups to build (default: all). Options: pod_gold pod_red pod_green cup_blue cup_red cup_green",
    )
    parser_arg.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be built without doing anything",
    )
    parser_arg.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading to HuggingFace Hub",
    )
    parser_arg.add_argument(
        "--only",
        choices=["all_aria", "no_aria"],
        default=None,
        help="Only build one variant type",
    )
    args = parser_arg.parse_args()

    root = get_root()
    groups_to_build = GROUPS
    if args.groups:
        valid_names = {g.name for g in GROUPS}
        for name in args.groups:
            if name not in valid_names:
                raise ValueError(f"Unknown group: {name}. Valid: {valid_names}")
        groups_to_build = [g for g in GROUPS if g.name in args.groups]

    log.info("=" * 80)
    log.info("DEP DATASET BUILD PLAN")
    log.info("=" * 80)
    for g in groups_to_build:
        log.info(f"\n--- {g.name.upper()} ({g.expected_episodes} episodes) ---")
        for src in g.sources:
            log.info(f"  Source: {src}")
        if args.only != "no_aria":
            log.info(f"  -> {g.all_aria_repo}")
        if args.only != "all_aria":
            log.info(f"  -> {g.no_aria_repo}")
    log.info(f"\nTotal datasets to build: {len(groups_to_build) * (1 if args.only else 2)}")
    log.info("=" * 80)

    if args.dry_run:
        log.info("Dry run — no changes made.")
        return

    for group in groups_to_build:
        log.info(f"\n{'='*60}")
        log.info(f"Building: {group.name.upper()}")
        log.info(f"{'='*60}")

        # Step 1: Build all_aria variant
        if args.only != "no_aria":
            all_aria_ds = build_all_aria(group, root)
            log.info(f"  all_aria: {all_aria_ds.num_episodes} episodes, {all_aria_ds.num_frames} frames")
            if not args.skip_upload:
                upload_dataset(group.all_aria_repo, root)

        # Step 2: Build no_aria variant (from all_aria)
        if args.only != "all_aria":
            no_aria_ds = build_no_aria(group, root)
            log.info(f"  no_aria: {no_aria_ds.num_episodes} episodes, {no_aria_ds.num_frames} frames")
            if not args.skip_upload:
                upload_dataset(group.no_aria_repo, root)

    log.info("\n" + "=" * 80)
    log.info("BUILD COMPLETE")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
