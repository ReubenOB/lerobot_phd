#!/usr/bin/env python3
"""
Split a bimanual dataset with all 4 Aria eye gaze streams into separate datasets,
each containing only one Aria stream along with the regular cameras (top, wrist_0, wrist_1).

Usage:
    python split_aria_streams.py --dataset-id RAPOB/coffee_bimanual_aria_all_gold_2
"""

import argparse
import logging
from pathlib import Path

from lerobot.datasets.dataset_tools import remove_feature
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# Define the 4 Aria eye gaze streams
ARIA_STREAMS = {
    "gaussian_blur": "observation.images.aria_gaussian_blur",
    "brightness_boost": "observation.images.aria_brightness_boost",
    "gaussian_attention": "observation.images.aria_gaussian_attention",
    "hard_cutout": "observation.images.aria_hard_cutout",
}


def split_aria_dataset(
    source_repo_id: str,
    output_base_dir: Path | None = None,
    push_to_hub: bool = True,
):
    """Split dataset with all Aria streams into 4 separate datasets, one per stream.
    
    Args:
        source_repo_id: HuggingFace repo ID of the source dataset (e.g., "RAPOB/coffee_bimanual_aria_all_gold_2")
        output_base_dir: Base directory for output datasets. If None, uses default HF cache location.
        push_to_hub: Whether to push the split datasets to HuggingFace Hub.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading source dataset: {source_repo_id}")
    
    # Load the original dataset
    source_dataset = LeRobotDataset(repo_id=source_repo_id)
    
    logger.info(f"Source dataset has {source_dataset.meta.total_episodes} episodes")
    logger.info(f"Features: {list(source_dataset.meta.features.keys())}")
    
    # Verify all 4 Aria streams exist
    for stream_name, feature_name in ARIA_STREAMS.items():
        if feature_name not in source_dataset.meta.features:
            raise ValueError(f"Missing expected Aria stream: {feature_name}")
    
    logger.info("All 4 Aria streams found in source dataset")
    
    # Create 4 separate datasets, each with only one Aria stream
    for keep_stream, keep_feature in ARIA_STREAMS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Creating dataset for: {keep_stream}")
        logger.info(f"{'='*60}")
        
        # Features to remove (the other 3 Aria streams)
        features_to_remove = [
            feature for stream, feature in ARIA_STREAMS.items() 
            if stream != keep_stream
        ]
        
        logger.info(f"Keeping: {keep_feature}")
        logger.info(f"Removing: {features_to_remove}")
        
        # New repo ID: replace "_all" with the specific stream name
        new_repo_id = source_repo_id.replace("_all", f"_{keep_stream}")
        
        logger.info(f"Creating new dataset: {new_repo_id}")
        
        # Remove the other 3 Aria streams
        new_dataset = remove_feature(
            dataset=source_dataset,
            feature_names=features_to_remove,
            output_dir=output_base_dir / keep_stream if output_base_dir else None,
            repo_id=new_repo_id,
        )
        
        logger.info(f"✓ Created dataset with {new_dataset.meta.total_episodes} episodes")
        logger.info(f"  Features: {list(new_dataset.meta.features.keys())}")
        
        # Push to HuggingFace Hub
        if push_to_hub:
            logger.info(f"Pushing {new_repo_id} to HuggingFace Hub...")
            try:
                new_dataset.push_to_hub(
                    repo_id=new_repo_id,
                    private=True,  # Match the original dataset's privacy setting
                )
                logger.info(f"✓ Successfully pushed {new_repo_id} to Hub")
            except Exception as e:
                logger.error(f"✗ Failed to push {new_repo_id}: {e}")
        
        logger.info(f"{'='*60}\n")
    
    logger.info("All datasets created successfully!")
    logger.info("\nCreated datasets:")
    for stream_name in ARIA_STREAMS.keys():
        new_repo_id = source_repo_id.replace("_all", f"_{stream_name}")
        logger.info(f"  - {new_repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Split Aria eye gaze dataset into separate datasets per stream"
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Source dataset repo ID (e.g., RAPOB/coffee_bimanual_aria_all_gold_2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory for split datasets (default: HF cache)",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push datasets to HuggingFace Hub (only create locally)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    split_aria_dataset(
        source_repo_id=args.dataset_id,
        output_base_dir=output_dir,
        push_to_hub=not args.no_push,
    )


if __name__ == "__main__":
    main()
