#!/usr/bin/env python3
"""
Merge multiple LeRobot datasets into a single aggregated dataset.

This script uses the aggregate_datasets function from lerobot to combine
multiple datasets with compatible schemas (same fps, robot_type, features).

Usage:
    # Merge 3 datasets into a new one
    python merge_datasets.py \
        --datasets RAPOB/dataset_1 RAPOB/dataset_2 RAPOB/dataset_3 \
        --output-repo-id RAPOB/merged_dataset
    
    # Merge with custom output directory
    python merge_datasets.py \
        --datasets RAPOB/d1 RAPOB/d2 \
        --output-repo-id RAPOB/merged \
        --output-dir /path/to/output
    
    # Merge without pushing to Hub
    python merge_datasets.py \
        --datasets RAPOB/d1 RAPOB/d2 \
        --output-repo-id RAPOB/merged \
        --no-push
    
    # Merge with local dataset paths
    python merge_datasets.py \
        --datasets RAPOB/d1 RAPOB/d2 \
        --dataset-roots /path/to/d1 /path/to/d2 \
        --output-repo-id RAPOB/merged
"""

import argparse
import logging
import os
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, HF_LEROBOT_HOME


def merge_datasets(
    repo_ids: list[str],
    output_repo_id: str,
    dataset_roots: list[Path] | None = None,
    output_dir: Path | None = None,
    push_to_hub: bool = True,
    private: bool = True,
):
    """Merge multiple LeRobot datasets into a single aggregated dataset.
    
    Args:
        repo_ids: List of HuggingFace repo IDs to merge (e.g., ["RAPOB/d1", "RAPOB/d2"])
        output_repo_id: HuggingFace repo ID for the merged output dataset
        dataset_roots: Optional list of local root paths for the source datasets
        output_dir: Optional output directory for the merged dataset. If None, uses default.
        push_to_hub: Whether to push the merged dataset to HuggingFace Hub.
        private: Whether to make the pushed dataset private.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    if len(repo_ids) < 2:
        raise ValueError("Need at least 2 datasets to merge")
    
    logger.info(f"Merging {len(repo_ids)} datasets into: {output_repo_id}")
    logger.info(f"Source datasets: {repo_ids}")
    
    # Ensure datasets are available locally (download from Hub if needed)
    logger.info("\n" + "="*60)
    logger.info("Ensuring source datasets are available locally...")
    logger.info("="*60)
    
    for i, repo_id in enumerate(repo_ids):
        root = dataset_roots[i] if dataset_roots else None
        local_root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        data_dir = local_root / "data"
        
        if not data_dir.exists() or not any(data_dir.rglob("*.parquet")):
            logger.info(f"  [{i+1}] {repo_id}: Not found locally, downloading from Hub...")
            try:
                # LeRobotDataset auto-downloads data + videos from Hub if missing
                ds = LeRobotDataset(repo_id=repo_id, root=root)
                logger.info(f"        ✓ Downloaded: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")
                del ds  # Free memory
            except Exception as e:
                logger.error(f"  ✗ Failed to download {repo_id}: {e}")
                raise
        else:
            logger.info(f"  [{i+1}] {repo_id}: Found locally at {local_root}")
    
    # Validate datasets before merging
    logger.info("\n" + "="*60)
    logger.info("Validating source datasets...")
    logger.info("="*60)
    
    all_metadata = []
    for i, repo_id in enumerate(repo_ids):
        root = dataset_roots[i] if dataset_roots else None
        try:
            if root:
                meta = LeRobotDatasetMetadata(repo_id, root=root)
            else:
                meta = LeRobotDatasetMetadata(repo_id)
            all_metadata.append(meta)
            logger.info(f"  [{i+1}] {repo_id}:")
            logger.info(f"      Episodes: {meta.total_episodes}")
            logger.info(f"      Frames: {meta.total_frames}")
            logger.info(f"      FPS: {meta.fps}")
            logger.info(f"      Robot: {meta.robot_type}")
            logger.info(f"      Features: {len(meta.features)} ({', '.join(list(meta.features.keys())[:5])}...)")
        except Exception as e:
            logger.error(f"  ✗ Failed to load {repo_id}: {e}")
            raise
    
    # Verify compatibility
    logger.info("\n" + "="*60)
    logger.info("Checking compatibility...")
    logger.info("="*60)
    
    ref_meta = all_metadata[0]
    ref_fps = ref_meta.fps
    ref_robot = ref_meta.robot_type
    ref_features = set(ref_meta.features.keys())
    
    compatible = True
    for i, meta in enumerate(all_metadata[1:], start=2):
        if meta.fps != ref_fps:
            logger.error(f"  ✗ FPS mismatch: {repo_ids[0]} has {ref_fps}, {repo_ids[i-1]} has {meta.fps}")
            compatible = False
        if meta.robot_type != ref_robot:
            logger.error(f"  ✗ Robot type mismatch: {repo_ids[0]} has {ref_robot}, {repo_ids[i-1]} has {meta.robot_type}")
            compatible = False
        if set(meta.features.keys()) != ref_features:
            diff_missing = ref_features - set(meta.features.keys())
            diff_extra = set(meta.features.keys()) - ref_features
            if diff_missing:
                logger.error(f"  ✗ {repo_ids[i-1]} missing features: {diff_missing}")
            if diff_extra:
                logger.error(f"  ✗ {repo_ids[i-1]} has extra features: {diff_extra}")
            compatible = False
    
    if not compatible:
        raise ValueError("Datasets are not compatible for merging. See errors above.")
    
    logger.info("  ✓ All datasets compatible!")
    
    # Calculate totals
    total_episodes = sum(m.total_episodes for m in all_metadata)
    total_frames = sum(m.total_frames for m in all_metadata)
    logger.info(f"  Total episodes after merge: {total_episodes}")
    logger.info(f"  Total frames after merge: {total_frames}")
    
    # Perform the merge
    logger.info("\n" + "="*60)
    logger.info("Merging datasets...")
    logger.info("="*60)
    
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=output_repo_id,
        roots=dataset_roots,
        aggr_root=output_dir,
    )
    
    logger.info("  ✓ Merge complete!")
    
    # Load the merged dataset to verify and optionally push
    logger.info("\n" + "="*60)
    logger.info("Verifying merged dataset...")
    logger.info("="*60)
    
    merged_dataset = LeRobotDataset(repo_id=output_repo_id, root=output_dir)
    logger.info(f"  Episodes: {merged_dataset.meta.total_episodes}")
    logger.info(f"  Frames: {merged_dataset.meta.total_frames}")
    logger.info(f"  Features: {list(merged_dataset.meta.features.keys())}")
    
    # Push to HuggingFace Hub
    if push_to_hub:
        logger.info("\n" + "="*60)
        logger.info(f"Pushing {output_repo_id} to HuggingFace Hub...")
        logger.info("="*60)
        try:
            merged_dataset.push_to_hub(
                repo_id=output_repo_id,
                private=private,
            )
            logger.info(f"  ✓ Successfully pushed {output_repo_id} to Hub")
        except Exception as e:
            logger.error(f"  ✗ Failed to push {output_repo_id}: {e}")
            raise
    
    logger.info("\n" + "="*60)
    logger.info(f"✓ Successfully merged {len(repo_ids)} datasets into {output_repo_id}")
    logger.info("="*60)
    
    return merged_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple LeRobot datasets into a single aggregated dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge 3 datasets
  python merge_datasets.py \\
      --datasets RAPOB/d1 RAPOB/d2 RAPOB/d3 \\
      --output-repo-id RAPOB/merged

  # Merge with local paths
  python merge_datasets.py \\
      --datasets RAPOB/d1 RAPOB/d2 \\
      --dataset-roots /data/d1 /data/d2 \\
      --output-repo-id RAPOB/merged

  # Merge without pushing to Hub
  python merge_datasets.py \\
      --datasets RAPOB/d1 RAPOB/d2 \\
      --output-repo-id RAPOB/merged \\
      --no-push
        """
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset repo IDs to merge (e.g., RAPOB/d1 RAPOB/d2 RAPOB/d3)"
    )
    parser.add_argument(
        "--output-repo-id",
        required=True,
        help="Output repo ID for the merged dataset (e.g., RAPOB/merged_dataset)"
    )
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        default=None,
        help="Optional local root paths for source datasets (must match --datasets order)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for merged dataset. If not specified, uses default HF cache."
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push the merged dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the pushed dataset public (default is private)"
    )
    
    args = parser.parse_args()
    
    # Validate dataset-roots if provided
    dataset_roots = None
    if args.dataset_roots:
        if len(args.dataset_roots) != len(args.datasets):
            parser.error(f"--dataset-roots must have same number of paths as --datasets "
                        f"({len(args.dataset_roots)} != {len(args.datasets)})")
        dataset_roots = [Path(r) for r in args.dataset_roots]
    
    # Check HF_TOKEN for pushing
    if not args.no_push and not os.environ.get("HF_TOKEN"):
        logging.warning("HF_TOKEN not set. You may need it to push to HuggingFace Hub.")
    
    merge_datasets(
        repo_ids=args.datasets,
        output_repo_id=args.output_repo_id,
        dataset_roots=dataset_roots,
        output_dir=args.output_dir,
        push_to_hub=not args.no_push,
        private=not args.public,
    )


if __name__ == "__main__":
    main()
