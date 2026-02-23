#!/usr/bin/env python3
"""
Build all training datasets for the 6-model coffee pipeline.

Creates 12 datasets total:
  - 6 merged datasets with all 4 aria streams (for future splitting)
  - 6 datasets with NO aria streams, only robot cameras (for training)

For replay baseline, just use any dataset with:
    lerobot-replay --dataset.repo_id=AcuBrain/dep_coffee_pod_gold_no_aria --dataset.episode=0

Naming convention:
    AcuBrain/dep_coffee_{object}_{color}_{variant}

    Objects: pod (stage 1), cup (stage 2)
    Colors:  gold, red, green (pods) / blue, red, green (cups)
    Variants:
        _all_aria           -> all 4 aria streams + robot cameras
        _no_aria            -> robot cameras only (no aria glasses, for training)

Usage:
    # Dry run - show what would be created
    python build_training_datasets.py --dry-run

    # Build all datasets (downloads, merges, splits, pushes)
    python build_training_datasets.py

    # Build only the all-aria merged sets
    python build_training_datasets.py --stage all-aria

    # Build only no-aria training sets
    python build_training_datasets.py --stage no-aria

    # Don't push to Hub (local only)
    python build_training_datasets.py --no-push
"""

import argparse
import logging
import sys
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_tools import remove_feature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME


OUTPUT_NAMESPACE = "AcuBrain"

# The 4 aria feature names
ARIA_FEATURES = [
    "observation.images.aria_gaussian_blur",
    "observation.images.aria_brightness_boost",
    "observation.images.aria_gaussian_attention",
    "observation.images.aria_hard_cutout",
]

# All aria features are removed for the no_aria variant
REMOVE_ALL_ARIA = ARIA_FEATURES


# === DATASET DEFINITIONS ===
# Each entry: (output_name, source_repos_with_all_4_aria)
# These are the datasets that have all 4 aria streams intact

DATASET_DEFS = {
    # --- PODS (Stage 1) ---
    "pod_gold": {
        "sources_all": [
            "RAPOB/coffee_bimanual_aria_all_gold_2",       # 20 eps, v1
            "RAPOB/coffee_bimanual_aria_all_gold_v2",      # 20 eps, v2
        ],
        "expected_eps": 40,
    },
    "pod_red": {
        "sources_all": [
            "RAPOB/coffee_bimanual_aria_all_red_3",        # 20 eps, v1
            "RAPOB/coffee_bimanual_aria_all_red_v2_10_a",  # 10 eps, v2
        ],
        "expected_eps": 30,
    },
    "pod_green": {
        # Green v1 _all was deleted. Use reconstructed version.
        # Run reconstruct_all_aria.py first to create this:
        "sources_all": [
            "RAPOB/coffee_bimanual_aria_all_green_1_reconstructed",  # 20 eps, v1 reconstructed
            "RAPOB/coffee_bimanual_aria_all_green_v2_5",             # 5 eps, v2
        ],
        "expected_eps": 25,
    },

    # --- CUPS (Stage 2) ---
    "cup_blue": {
        "sources_all": [
            "RAPOB/coffee_bimanual_aria_stage_2_all_blue_b",    # 20 eps, v1
            "RAPOB/coffee_bimanual_aria_stage_2_all_blue_v2_10", # 10 eps, v2
        ],
        "expected_eps": 30,
    },
    "cup_red": {
        "sources_all": [
            "RAPOB/coffee_bimanual_aria_stage_2_all_red",  # 20 eps, only version
        ],
        "expected_eps": 20,
    },
    "cup_green": {
        "sources_all": [
            "RAPOB/coffee_bimanual_aria_stage_2_all_green_a_combine",  # 20 eps, v1
            "RAPOB/coffee_bimanual_aria_stage_2_all_green_v2_5",       # 5 eps, v2
        ],
        "expected_eps": 25,
    },
}


def make_repo_id(name: str, variant: str) -> str:
    """Generate standardized repo ID."""
    return f"{OUTPUT_NAMESPACE}/dep_coffee_{name}_{variant}"


def print_plan():
    """Print what datasets will be created."""
    print("\n" + "=" * 80)
    print("DATASET BUILD PLAN")
    print("=" * 80)

    for name, defn in DATASET_DEFS.items():
        print(f"\n--- {name.upper()} ---")
        print(f"  Sources ({defn['expected_eps']} eps total):")
        for src in defn["sources_all"]:
            print(f"    - {src}")
        print(f"  Will create:")
        print(f"    1. {make_repo_id(name, 'all_aria'):60s}  (all 4 aria + cameras, {defn['expected_eps']} eps)")
        print(f"    2. {make_repo_id(name, 'no_aria'):60s}  (robot cameras only, {defn['expected_eps']} eps)")

    print(f"\n{'=' * 80}")
    print(f"TOTAL: {len(DATASET_DEFS) * 2} datasets")
    print(f"  Replay baseline: use lerobot-replay --dataset.episode=0 on any no_aria dataset")
    print(f"{'=' * 80}\n")


def build_all_aria(name: str, defn: dict, push: bool, logger: logging.Logger) -> str:
    """Stage 1: Merge source datasets into a single all-aria dataset."""
    repo_id = make_repo_id(name, "all_aria")
    sources = defn["sources_all"]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Building ALL-ARIA: {repo_id}")
    logger.info(f"  Sources: {sources}")
    logger.info(f"{'=' * 60}")

    # Pre-download all source datasets (metadata + videos) so aggregate_datasets
    # can find the video files on disk
    for src_repo in sources:
        logger.info(f"  Downloading: {src_repo}")
        LeRobotDataset(repo_id=src_repo)

    if len(sources) == 1:
        logger.info("  Single source — copying dataset with new repo ID")
    else:
        logger.info(f"  Merging {len(sources)} datasets")

    aggregate_datasets(
        repo_ids=sources,
        aggr_repo_id=repo_id,
    )

    # Verify
    merged = LeRobotDataset(repo_id=repo_id)
    logger.info(f"  ✓ Created: {merged.meta.total_episodes} episodes, {merged.meta.total_frames} frames")
    logger.info(f"    Features: {list(merged.meta.features.keys())}")

    if push:
        logger.info(f"  Pushing {repo_id} to Hub...")
        merged.push_to_hub(repo_id=repo_id, private=True)
        logger.info(f"  ✓ Pushed")

    return repo_id


def build_no_aria(name: str, all_aria_repo_id: str, push: bool, logger: logging.Logger) -> str:
    """Stage 2: Remove ALL aria streams, keep only robot cameras."""
    repo_id = make_repo_id(name, "no_aria")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Building NO-ARIA: {repo_id}")
    logger.info(f"  Source: {all_aria_repo_id}")
    logger.info(f"  Removing: {REMOVE_ALL_ARIA}")
    logger.info(f"{'=' * 60}")

    source = LeRobotDataset(repo_id=all_aria_repo_id)

    result = remove_feature(
        dataset=source,
        feature_names=REMOVE_ALL_ARIA,
        repo_id=repo_id,
    )

    logger.info(f"  ✓ Created: {result.meta.total_episodes} episodes")
    logger.info(f"    Features: {list(result.meta.features.keys())}")

    if push:
        logger.info(f"  Pushing {repo_id} to Hub...")
        result.push_to_hub(repo_id=repo_id, private=True)
        logger.info(f"  ✓ Pushed")

    return repo_id


def main():
    global OUTPUT_NAMESPACE

    parser = argparse.ArgumentParser(
        description="Build all 12 training datasets for the 6-model coffee pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["all-aria", "no-aria", "all"],
        default="all",
        help="Which stage to build (default: all)",
    )
    parser.add_argument("--no-push", action="store_true", help="Don't push to Hub")
    parser.add_argument("--dry-run", action="store_true", help="Just print the plan")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(DATASET_DEFS.keys()),
        default=None,
        help="Only build specific objects (e.g., --only pod_gold cup_blue)",
    )
    parser.add_argument(
        "--namespace",
        default=OUTPUT_NAMESPACE,
        help=f"HuggingFace namespace for output datasets (default: {OUTPUT_NAMESPACE})",
    )

    args = parser.parse_args()
    OUTPUT_NAMESPACE = args.namespace

    if args.dry_run:
        print_plan()
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    push = not args.no_push
    targets = args.only or list(DATASET_DEFS.keys())

    print_plan()

    results = {"all_aria": {}, "no_aria": {}}

    for name in targets:
        defn = DATASET_DEFS[name]
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# Processing: {name.upper()}")
        logger.info(f"{'#' * 60}")

        try:
            # Stage 1: All-aria merged
            if args.stage in ("all-aria", "all"):
                all_aria_id = build_all_aria(name, defn, push, logger)
                results["all_aria"][name] = all_aria_id
            else:
                # Need the all-aria repo to exist for other stages
                all_aria_id = make_repo_id(name, "all_aria")

            # Stage 2: No-aria (robot cameras only)
            if args.stage in ("no-aria", "all"):
                na_id = build_no_aria(name, all_aria_id, push, logger)
                results["no_aria"][name] = na_id

        except Exception as e:
            logger.error(f"✗ Failed processing {name}: {e}")
            logger.exception(e)
            continue

    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("BUILD SUMMARY")
    logger.info(f"{'=' * 80}")
    for stage, datasets in results.items():
        if datasets:
            logger.info(f"\n  {stage.upper()}:")
            for name, repo_id in datasets.items():
                logger.info(f"    ✓ {repo_id}")


if __name__ == "__main__":
    main()
