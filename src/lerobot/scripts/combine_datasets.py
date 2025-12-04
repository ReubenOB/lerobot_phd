#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Combine multiple LeRobot datasets into a single dataset.

This script downloads datasets from HuggingFace Hub, combines them into a single
dataset, and optionally re-uploads to the Hub.

Inspired by phosphobot's dataset merging functionality.

Usage examples:

1. Combine two local datasets:
    python -m lerobot.scripts.combine_datasets \
        --repo-ids user/dataset1 user/dataset2 \
        --output-repo-id user/combined_dataset

2. Combine from HuggingFace Hub and upload result:
    python -m lerobot.scripts.combine_datasets \
        --repo-ids lerobot/aloha_sim_insertion_human lerobot/aloha_sim_transfer_cube_human \
        --output-repo-id myuser/combined_aloha \
        --push-to-hub

3. Combine with custom task name:
    python -m lerobot.scripts.combine_datasets \
        --repo-ids user/dataset1 user/dataset2 \
        --output-repo-id user/combined \
        --tasks "pick and place object" "stack blocks"
"""

import argparse
import logging
import shutil
from pathlib import Path
from pprint import pformat

import pandas as pd
from huggingface_hub import HfApi, create_repo

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    write_json,
    write_jsonlines,
    write_stats,
    create_lerobot_dataset_card,
    serialize_dict,
    append_jsonlines,
    EPISODES_STATS_PATH,
)
from lerobot.datasets.compute_stats import aggregate_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_datasets_compatible(datasets: list[LeRobotDatasetMetadata]) -> dict:
    """Validate that all datasets are compatible for merging."""
    if len(datasets) < 2:
        raise ValueError("Need at least 2 datasets to combine")
    
    reference = datasets[0]
    
    # Check FPS consistency
    fps_values = [ds.fps for ds in datasets]
    if len(set(fps_values)) > 1:
        raise ValueError(f"Datasets have different FPS values: {fps_values}")
    
    # Check robot type consistency  
    robot_types = [ds.robot_type for ds in datasets]
    if len(set(robot_types)) > 1:
        logger.warning(f"Datasets have different robot types: {robot_types}")
    
    # Check feature compatibility - find common features
    all_features = [set(ds.features.keys()) for ds in datasets]
    common_features = all_features[0].intersection(*all_features[1:])
    
    if len(common_features) == 0:
        raise ValueError("Datasets have no common features")
    
    # Check camera keys
    camera_keys_list = [set(ds.camera_keys) for ds in datasets]
    common_cameras = camera_keys_list[0].intersection(*camera_keys_list[1:])
    
    if len(common_cameras) == 0:
        logger.warning("Datasets have no common camera keys - videos may need remapping")
    
    return {
        "fps": reference.fps,
        "robot_type": reference.robot_type,
        "common_features": common_features,
        "common_cameras": common_cameras,
        "all_features": [ds.features for ds in datasets],
    }


def combine_datasets(
    repo_ids: list[str],
    output_repo_id: str,
    root: Path | None = None,
    tasks: list[str] | None = None,
    camera_mapping: dict[str, dict[str, str]] | None = None,
    push_to_hub: bool = False,
    hub_token: str | None = None,
    private: bool = False,
) -> LeRobotDataset:
    """
    Combine multiple LeRobot datasets into a single dataset.
    
    Args:
        repo_ids: List of HuggingFace repo IDs to combine
        output_repo_id: Repo ID for the combined dataset
        root: Local root directory for datasets
        tasks: Optional list of task descriptions for each dataset
        camera_mapping: Optional mapping of camera keys between datasets
            e.g., {"dataset2": {"observation.images.wrist": "observation.images.wrist_0"}}
        push_to_hub: Whether to push the combined dataset to HuggingFace Hub
        hub_token: HuggingFace API token
        private: Whether to make the Hub repo private
        
    Returns:
        The combined LeRobotDataset
    """
    if root is None:
        root = HF_LEROBOT_HOME
    root = Path(root)
    
    logger.info(f"Combining {len(repo_ids)} datasets into {output_repo_id}")
    logger.info(f"Using root directory: {root}")
    
    # Load metadata for all datasets first
    logger.info("Loading dataset metadata...")
    metas = []
    for repo_id in repo_ids:
        dataset_path = root / repo_id
        logger.info(f"  Loading metadata for {repo_id} from {dataset_path}")
        meta = LeRobotDatasetMetadata(repo_id, root=dataset_path)
        metas.append(meta)
    
    # Validate compatibility
    logger.info("Validating dataset compatibility...")
    compat_info = validate_datasets_compatible(metas)
    logger.info(f"Common features: {compat_info['common_features']}")
    logger.info(f"Common cameras: {compat_info['common_cameras']}")
    
    # Load full datasets
    logger.info("Loading full datasets...")
    datasets = []
    for repo_id in repo_ids:
        dataset_path = root / repo_id
        logger.info(f"  Loading {repo_id} from {dataset_path}")
        ds = LeRobotDataset(repo_id, root=dataset_path)
        datasets.append(ds)
    
    # Prepare output directory
    output_dir = root / output_repo_id
    logger.info(f"Output directory: {output_dir}")
    if output_dir.exists():
        logger.warning(f"Output directory {output_dir} exists, removing...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    # Create subdirectories
    (output_dir / "meta").mkdir()
    (output_dir / "data" / "chunk-000").mkdir(parents=True)
    (output_dir / "videos" / "chunk-000").mkdir(parents=True)
    
    # Initialize tracking variables
    total_episodes = 0
    total_frames = 0
    total_videos = 0
    all_tasks = []
    all_episodes = []
    all_episode_stats = {}  # Dict mapping new episode index to stats
    task_to_task_index = {}
    
    # Process each dataset
    for ds_idx, (repo_id, dataset) in enumerate(zip(repo_ids, datasets)):
        logger.info(f"Processing dataset {ds_idx + 1}/{len(datasets)}: {repo_id}")
        
        ds_episodes = dataset.meta.total_episodes
        ds_frames = dataset.meta.total_frames
        
        # Get task for this dataset
        if tasks and ds_idx < len(tasks):
            task_str = tasks[ds_idx]
        else:
            # Try to get task from dataset
            ds_tasks = dataset.meta.tasks
            if ds_tasks:
                task_str = list(ds_tasks.values())[0] if isinstance(ds_tasks, dict) else ds_tasks[0].get("task", f"task_{ds_idx}")
            else:
                task_str = f"task_{ds_idx}"
        
        # Assign task index
        if task_str not in task_to_task_index:
            task_to_task_index[task_str] = len(task_to_task_index)
        task_index = task_to_task_index[task_str]
        
        # Determine camera mapping for this dataset
        ds_camera_mapping = {}
        if camera_mapping and repo_id in camera_mapping:
            ds_camera_mapping = camera_mapping[repo_id]
        
        # Process episodes
        for ep_idx in range(ds_episodes):
            new_ep_idx = total_episodes + ep_idx
            
            # Copy parquet file - get_data_file_path returns relative path, need to prepend root
            src_parquet = dataset.meta.root / dataset.meta.get_data_file_path(ep_idx)
            dst_parquet = output_dir / "data" / "chunk-000" / f"episode_{new_ep_idx:06d}.parquet"
            
            # Load, modify indices, and save parquet
            df = pd.read_parquet(src_parquet)
            df["episode_index"] = new_ep_idx
            df["index"] = df["index"] + total_frames
            df["task_index"] = task_index
            df.to_parquet(dst_parquet)
            
            # Copy video files
            for cam_key in dataset.meta.video_keys:
                # get_video_file_path returns relative path, need to prepend root
                src_video = dataset.meta.root / dataset.meta.get_video_file_path(ep_idx, cam_key)
                
                # Apply camera mapping if specified
                dst_cam_key = ds_camera_mapping.get(cam_key, cam_key)
                
                dst_video_dir = output_dir / "videos" / "chunk-000" / dst_cam_key
                dst_video_dir.mkdir(parents=True, exist_ok=True)
                dst_video = dst_video_dir / f"episode_{new_ep_idx:06d}.mp4"
                
                if src_video.exists():
                    shutil.copy2(src_video, dst_video)
                    total_videos += 1
            
            # Track episode metadata
            ep_length = len(df)
            all_episodes.append({
                "episode_index": new_ep_idx,
                "tasks": [task_str],
                "length": ep_length,
            })
            
            # Track episode stats if available
            if hasattr(dataset.meta, 'episodes_stats') and dataset.meta.episodes_stats:
                if ep_idx in dataset.meta.episodes_stats:
                    all_episode_stats[new_ep_idx] = dataset.meta.episodes_stats[ep_idx]
        
        total_episodes += ds_episodes
        total_frames += ds_frames
        
        logger.info(f"  Added {ds_episodes} episodes, {ds_frames} frames")
    
    # Build tasks list
    for task_str, task_idx in sorted(task_to_task_index.items(), key=lambda x: x[1]):
        all_tasks.append({
            "task_index": task_idx,
            "task": task_str,
        })
    
    # Aggregate stats from episode stats
    logger.info("Aggregating statistics...")
    if all_episode_stats:
        combined_stats = aggregate_stats(list(all_episode_stats.values()))
    else:
        # Fallback to using dataset-level stats if episode stats not available
        all_stats = [dataset.meta.stats for dataset in datasets]
        combined_stats = aggregate_stats(all_stats)
    
    # Use first dataset as reference for features
    ref_meta = metas[0]
    
    # Build info.json
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": compat_info["robot_type"],
        "fps": compat_info["fps"],
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(all_tasks),
        "total_videos": total_videos,
        "total_chunks": 1,
        "chunks_size": total_episodes,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": ref_meta.features,
        "splits": {"train": f"0:{total_episodes}"},
    }
    
    # Write metadata files
    logger.info("Writing metadata files...")
    write_json(info, output_dir / "meta" / "info.json")
    write_jsonlines(all_tasks, output_dir / "meta" / "tasks.jsonl")
    write_jsonlines(all_episodes, output_dir / "meta" / "episodes.jsonl")
    
    # Write stats.json for backward compatibility (write_stats prepends "meta/")
    write_stats(combined_stats, output_dir)
    
    # Write episodes_stats.jsonl for v2.1 format
    if all_episode_stats:
        for ep_idx, ep_stats in sorted(all_episode_stats.items()):
            episode_stats_entry = {"episode_index": ep_idx, "stats": serialize_dict(ep_stats)}
            append_jsonlines(episode_stats_entry, output_dir / EPISODES_STATS_PATH)
    
    # Create README
    logger.info("Creating README...")
    card = create_lerobot_dataset_card(
        tags=["LeRobot", "combined", f"lerobot-version:{CODEBASE_VERSION}"],
        dataset_info=info,
        text=f"Combined dataset from: {', '.join(repo_ids)}",
    )
    card.save(output_dir / "README.md")
    
    logger.info(f"Combined dataset created at {output_dir}")
    logger.info(f"  Total episodes: {total_episodes}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Total tasks: {len(all_tasks)}")
    logger.info(f"  Total videos: {total_videos}")
    
    # Push to hub if requested
    if push_to_hub:
        logger.info(f"Pushing to HuggingFace Hub: {output_repo_id}")
        api = HfApi(token=hub_token)
        
        # Create repo
        create_repo(
            repo_id=output_repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=private,
            token=hub_token,
        )
        
        # Upload
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=output_repo_id,
            repo_type="dataset",
        )
        
        # Create version tag
        api.create_tag(
            repo_id=output_repo_id,
            repo_type="dataset",
            tag=CODEBASE_VERSION,
            tag_message=f"LeRobot dataset version {CODEBASE_VERSION}",
        )
        logger.info(f"Created tag {CODEBASE_VERSION}")
        logger.info(f"Pushed to https://huggingface.co/datasets/{output_repo_id}")
    
    # Return the combined dataset
    return LeRobotDataset(output_repo_id, root=output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple LeRobot datasets into one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine two datasets locally
  python -m lerobot.scripts.combine_datasets \\
      --repo-ids user/dataset1 user/dataset2 \\
      --output-repo-id user/combined

  # Combine and push to Hub
  python -m lerobot.scripts.combine_datasets \\
      --repo-ids lerobot/aloha_sim_insertion_human lerobot/aloha_sim_transfer_cube_human \\
      --output-repo-id myuser/combined_aloha \\
      --push-to-hub

  # Combine with custom tasks
  python -m lerobot.scripts.combine_datasets \\
      --repo-ids user/pick_dataset user/place_dataset \\
      --output-repo-id user/pick_and_place \\
      --tasks "pick up the cube" "place the cube"
        """,
    )
    
    parser.add_argument(
        "--repo-ids",
        type=str,
        nargs="+",
        required=True,
        help="List of HuggingFace dataset repo IDs to combine",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        required=True,
        help="Output repo ID for the combined dataset",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Local root directory for datasets (default: ~/.cache/huggingface/lerobot)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Task descriptions for each dataset (in order)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the combined dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-token",
        type=str,
        default=None,
        help="HuggingFace API token (uses cached token if not provided)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Hub repo private",
    )
    
    args = parser.parse_args()
    
    logger.info(f"Arguments:\n{pformat(vars(args))}")
    
    combined_dataset = combine_datasets(
        repo_ids=args.repo_ids,
        output_repo_id=args.output_repo_id,
        root=args.root,
        tasks=args.tasks,
        push_to_hub=args.push_to_hub,
        hub_token=args.hub_token,
        private=args.private,
    )
    
    logger.info(f"\nCombined dataset:\n{combined_dataset}")


if __name__ == "__main__":
    main()
