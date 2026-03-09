#!/usr/bin/env python3
"""
Build DEP hard-cutout-only datasets from existing all_aria DEP datasets.

Takes the 6 all_aria DEP datasets (which have 4 aria streams + robot cameras)
and strips 3 aria streams, keeping only observation.images.aria_hard_cutout
plus the robot cameras (top, wrist_0, wrist_1).

Creates 6 datasets:
  RAPOB/dep_coffee_pod_gold_hard_cutout   (40 episodes)
  RAPOB/dep_coffee_pod_red_hard_cutout    (30 episodes)
  RAPOB/dep_coffee_pod_green_hard_cutout  (25 episodes)
  RAPOB/dep_coffee_cup_blue_hard_cutout   (30 episodes)
  RAPOB/dep_coffee_cup_red_hard_cutout    (20 episodes)
  RAPOB/dep_coffee_cup_green_hard_cutout  (25 episodes)

Usage (inside Docker container with lerobot venv):
  source /opt/lerobot_venv/bin/activate
  python3 files/build_dep_hard_cutout_datasets.py

  # Build only specific groups:
  python3 files/build_dep_hard_cutout_datasets.py --groups pod_gold cup_blue

  # Dry run:
  python3 files/build_dep_hard_cutout_datasets.py --dry-run

  # Skip upload:
  python3 files/build_dep_hard_cutout_datasets.py --no-upload

  # Validate existing datasets (check video/timestamp consistency):
  python3 files/build_dep_hard_cutout_datasets.py --validate
"""

import argparse
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lerobot.datasets.dataset_tools import remove_feature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Aria streams to REMOVE (keep only aria_hard_cutout)
ARIA_STREAMS_TO_REMOVE = [
    "observation.images.aria_gaussian_blur",
    "observation.images.aria_brightness_boost",
    "observation.images.aria_gaussian_attention",
]

ORG = "RAPOB"


@dataclass
class DatasetGroup:
    name: str
    all_aria_repo: str
    hard_cutout_repo: str
    expected_episodes: int


GROUPS = [
    DatasetGroup(
        name="pod_gold",
        all_aria_repo=f"{ORG}/dep_coffee_pod_gold_all_aria",
        hard_cutout_repo=f"{ORG}/dep_coffee_pod_gold_hard_cutout",
        expected_episodes=40,
    ),
    DatasetGroup(
        name="pod_red",
        all_aria_repo=f"{ORG}/dep_coffee_pod_red_all_aria",
        hard_cutout_repo=f"{ORG}/dep_coffee_pod_red_hard_cutout",
        expected_episodes=30,
    ),
    DatasetGroup(
        name="pod_green",
        all_aria_repo=f"{ORG}/dep_coffee_pod_green_all_aria",
        hard_cutout_repo=f"{ORG}/dep_coffee_pod_green_hard_cutout",
        expected_episodes=25,
    ),
    DatasetGroup(
        name="cup_blue",
        all_aria_repo=f"{ORG}/dep_coffee_cup_blue_all_aria",
        hard_cutout_repo=f"{ORG}/dep_coffee_cup_blue_hard_cutout",
        expected_episodes=30,
    ),
    DatasetGroup(
        name="cup_red",
        all_aria_repo=f"{ORG}/dep_coffee_cup_red_all_aria",
        hard_cutout_repo=f"{ORG}/dep_coffee_cup_red_hard_cutout",
        expected_episodes=20,
    ),
    DatasetGroup(
        name="cup_green",
        all_aria_repo=f"{ORG}/dep_coffee_cup_green_all_aria",
        hard_cutout_repo=f"{ORG}/dep_coffee_cup_green_hard_cutout",
        expected_episodes=25,
    ),
]


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def validate_dataset(repo_id: str) -> bool:
    """Validate a dataset: check that all video files cover their episode timestamps.

    Returns True if all ok, False if issues found.
    """
    import pandas as pd

    root = HF_LEROBOT_HOME / repo_id
    if not root.exists():
        log.error(f"  Dataset not found locally: {root}")
        return False

    ds = LeRobotDataset(repo_id=repo_id)
    log.info(f"  Episodes: {ds.meta.total_episodes}, Frames: {ds.meta.total_frames}")
    log.info(f"  Cameras: {ds.meta.camera_keys}")

    issues = []
    for ep_idx in range(ds.meta.total_episodes):
        # Get max timestamp for this episode from parquet data
        ep_data = ds.hf_dataset.filter(lambda x: x["episode_index"] == ep_idx)
        max_ts = max(ep_data["timestamp"])

        # Check each video key
        for vkey in ds.meta.video_keys:
            video_path = ds.root / ds.meta.get_video_file_path(ep_idx, vkey)
            if not video_path.exists():
                issues.append(f"  MISSING: ep{ep_idx:03d} {vkey} -> {video_path}")
                continue
            try:
                duration = get_video_duration(video_path)
            except Exception as e:
                issues.append(f"  FFPROBE FAIL: ep{ep_idx:03d} {vkey} -> {e}")
                continue

            gap = max_ts - duration
            if gap > 0.5:  # more than 0.5s gap = truncated
                issues.append(
                    f"  TRUNCATED: ep{ep_idx:03d} {vkey} "
                    f"video={duration:.1f}s, data={max_ts:.1f}s, gap={gap:.1f}s "
                    f"-> {video_path.name}"
                )

    if issues:
        log.error(f"  Found {len(issues)} issue(s):")
        for issue in issues:
            log.error(issue)
        return False
    else:
        log.info("  All videos OK")
        return True


def build_hard_cutout(group: DatasetGroup, push: bool = True) -> None:
    """Create hard-cutout-only variant by removing 3 aria streams from all_aria dataset."""
    output_dir = HF_LEROBOT_HOME / group.hard_cutout_repo

    log.info(f"\n{'='*70}")
    log.info(f"[{group.name}] {group.all_aria_repo} -> {group.hard_cutout_repo}")
    log.info(f"{'='*70}")

    # Clean previous output
    if output_dir.exists():
        log.info(f"  Removing previous build: {output_dir}")
        shutil.rmtree(output_dir)

    # Load source — let LeRobotDataset resolve paths via HF_LEROBOT_HOME
    log.info(f"  Loading source: {group.all_aria_repo}")
    source = LeRobotDataset(repo_id=group.all_aria_repo)
    log.info(f"    Episodes: {source.meta.total_episodes} (expected {group.expected_episodes})")
    log.info(f"    Frames:   {source.meta.total_frames}")
    log.info(f"    Cameras:  {source.meta.camera_keys}")

    # Figure out which streams to remove
    existing_keys = set(source.meta.camera_keys)
    streams_to_remove = [s for s in ARIA_STREAMS_TO_REMOVE if s in existing_keys]

    if not streams_to_remove:
        log.warning(f"  No matching aria streams found to remove! Keys: {existing_keys}")
        return

    log.info(f"  Removing: {streams_to_remove}")
    log.info(f"  Keeping:  {[k for k in source.meta.camera_keys if k not in streams_to_remove]}")

    # Strip the 3 aria streams, keeping hard_cutout + robot cameras
    result = remove_feature(
        dataset=source,
        feature_names=streams_to_remove,
        repo_id=group.hard_cutout_repo,
    )

    log.info(f"  Result cameras: {result.meta.camera_keys}")
    log.info(f"  Result episodes: {result.meta.total_episodes}")

    # Validate the new dataset
    log.info(f"  Validating...")
    ok = validate_dataset(group.hard_cutout_repo)
    if not ok:
        log.error(f"  VALIDATION FAILED for {group.hard_cutout_repo}")
        return

    # Upload
    if push:
        log.info(f"  Pushing to Hub: {group.hard_cutout_repo}...")
        result.push_to_hub(repo_id=group.hard_cutout_repo, private=True)
        log.info(f"  Done: https://huggingface.co/datasets/{group.hard_cutout_repo}")
    else:
        log.info(f"  Skipping upload (--no-upload)")


def main():
    parser = argparse.ArgumentParser(description="Build DEP hard-cutout-only datasets")
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Specific groups to build (default: all). "
        "Options: pod_gold pod_red pod_green cup_blue cup_red cup_green",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be built")
    parser.add_argument("--no-upload", action="store_true", help="Skip uploading to HuggingFace")
    parser.add_argument("--validate", action="store_true",
                        help="Only validate existing hard_cutout datasets, don't rebuild")
    args = parser.parse_args()

    groups = GROUPS
    if args.groups:
        valid = {g.name for g in GROUPS}
        for g in args.groups:
            if g not in valid:
                parser.error(f"Unknown group: {g}. Valid: {sorted(valid)}")
        groups = [g for g in GROUPS if g.name in args.groups]

    # Validate-only mode
    if args.validate:
        log.info("=" * 70)
        log.info("VALIDATING HARD-CUTOUT DATASETS")
        log.info("=" * 70)
        all_ok = True
        for g in groups:
            log.info(f"\n--- {g.hard_cutout_repo} ---")
            ok = validate_dataset(g.hard_cutout_repo)
            if not ok:
                all_ok = False
        if all_ok:
            log.info("\nAll datasets passed validation.")
        else:
            log.error("\nSome datasets have issues. Rebuild with:")
            log.error("  python3 files/build_dep_hard_cutout_datasets.py")
        return

    log.info("=" * 70)
    log.info("BUILD DEP HARD-CUTOUT DATASETS")
    log.info(f"HF_LEROBOT_HOME: {HF_LEROBOT_HOME}")
    log.info(f"Groups: {[g.name for g in groups]}")
    log.info(f"Removing: {ARIA_STREAMS_TO_REMOVE}")
    log.info(f"Keeping:  observation.images.aria_hard_cutout + robot cameras")
    log.info("=" * 70)

    if args.dry_run:
        log.info("\n--- DRY RUN ---")
        for g in groups:
            source = HF_LEROBOT_HOME / g.all_aria_repo
            exists = source.exists()
            log.info(f"  {g.all_aria_repo} -> {g.hard_cutout_repo}  "
                     f"({'exists' if exists else 'MISSING'})")
        return

    for g in groups:
        build_hard_cutout(g, push=not args.no_upload)

    log.info("\n" + "=" * 70)
    log.info("ALL DATASETS PROCESSED")
    log.info("=" * 70)
    log.info("\nNew datasets:")
    for g in groups:
        log.info(f"  {g.hard_cutout_repo}")
    log.info("\nTo train, run from inside the container:")
    log.info("  source /opt/lerobot_venv/bin/activate")
    log.info("  cd /home/acumino/vigil_ws/src/deps/lerobot_phd")
    log.info("  python scripts/queue_training.py --config launch/train_dep_queue.yaml")


if __name__ == "__main__":
    main()
