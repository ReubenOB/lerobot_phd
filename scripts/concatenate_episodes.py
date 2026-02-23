#!/usr/bin/env python3
"""
Concatenate episodes from two sequential datasets into single continuous trajectories.

This is for when you have two datasets that represent two stages of a task:
  - Dataset A: episodes for stage 1 (e.g., "pick up coffee pod")
  - Dataset B: episodes for stage 2 (e.g., "place pod in machine")

Where the end of each A episode ~ the start of the corresponding B episode.
This script stitches them together: A_ep0 + B_ep0 -> combined_ep0, etc.

This is DIFFERENT from combine_datasets.py (which keeps episodes separate) or
merge_datasets.py (which just appends). This creates genuinely continuous
trajectories that train the policy to do the full sequence.

If one dataset has more episodes than the other, the extras are ignored.

Usage:
    python scripts/concatenate_episodes.py \
        --dataset-a RAPOB/stage1_dataset \
        --dataset-b RAPOB/stage2_dataset \
        --output-repo-id RAPOB/full_task_dataset \
        --task "pick up coffee pod and place in machine"
"""

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import datasets
import pandas as pd
from huggingface_hub import HfApi, create_repo

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    write_json,
    write_stats,
    write_tasks,
    write_episodes,
    create_lerobot_dataset_card,
)
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.video_utils import get_video_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_pairs(pairs_str: str) -> list[tuple[int, int]]:
    """Parse episode pair specification like '0:0,1:1,2:3,4:5'."""
    pairs = []
    for pair in pairs_str.split(","):
        a, b = pair.strip().split(":")
        pairs.append((int(a), int(b)))
    return pairs


def extract_video_segment(
    src_video: Path,
    dst_video: Path,
    from_ts: float,
    to_ts: float,
) -> None:
    """Extract a time segment from a video file using ffmpeg stream copy (no re-encoding)."""
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    duration = to_ts - from_ts
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{from_ts:.6f}",
        "-i", str(src_video),
        "-t", f"{duration:.6f}",
        "-c", "copy",
        "-movflags", "+faststart",
        str(dst_video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")


def concatenate_two_videos(vid_a: Path, vid_b: Path, dst: Path) -> None:
    """Concatenate two video files using ffmpeg concat demuxer (no re-encoding)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(f"file '{vid_a.resolve()}'\n")
        f.write(f"file '{vid_b.resolve()}'\n")
        concat_list = f.name
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            "-movflags", "+faststart",
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr[-500:]}")
    finally:
        Path(concat_list).unlink(missing_ok=True)


def get_video_source_path(meta_root: Path, video_path_template: str, vk: str, chunk_idx: int, file_idx: int) -> Path:
    """Resolve the source video file path from the dataset's video_path template."""
    return meta_root / video_path_template.format(
        video_key=vk, chunk_index=chunk_idx, file_index=file_idx
    )


def concatenate_episodes(
    dataset_a_id: str,
    dataset_b_id: str,
    output_repo_id: str,
    root: Path | None = None,
    task: str | None = None,
    overlap_frames: int = 0,
    pairs: list[tuple[int, int]] | None = None,
    push_to_hub: bool = False,
    hub_token: str | None = None,
    private: bool = False,
) -> LeRobotDataset:
    if root is None:
        root = HF_LEROBOT_HOME
    root = Path(root)

    logger.info("=" * 60)
    logger.info("Episode Concatenation: Stitching Sequential Datasets")
    logger.info("=" * 60)
    logger.info(f"  Dataset A (stage 1): {dataset_a_id}")
    logger.info(f"  Dataset B (stage 2): {dataset_b_id}")
    logger.info(f"  Output:              {output_repo_id}")
    if overlap_frames > 0:
        logger.info(f"  Overlap trim:        {overlap_frames} frames from B start")

    # Ensure datasets are available locally
    logger.info("\nEnsuring datasets are available locally...")
    for repo_id in [dataset_a_id, dataset_b_id]:
        local_root = root / repo_id
        data_dir = local_root / "data"
        if not data_dir.exists() or not any(data_dir.rglob("*.parquet")):
            logger.info(f"  {repo_id}: Not found locally, downloading from Hub...")
            try:
                ds = LeRobotDataset(repo_id=repo_id, root=local_root)
                logger.info(f"    Downloaded: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")
                del ds
            except Exception as e:
                logger.error(f"  Failed to download {repo_id}: {e}")
                raise
        else:
            logger.info(f"  {repo_id}: Found locally at {local_root}")

    # Load both datasets
    logger.info("\nLoading datasets...")
    ds_a = LeRobotDataset(dataset_a_id, root=root / dataset_a_id)
    ds_b = LeRobotDataset(dataset_b_id, root=root / dataset_b_id)

    logger.info(f"  A: {ds_a.meta.total_episodes} episodes, {ds_a.meta.total_frames} frames")
    logger.info(f"  B: {ds_b.meta.total_episodes} episodes, {ds_b.meta.total_frames} frames")

    # Validate compatibility
    if ds_a.meta.fps != ds_b.meta.fps:
        raise ValueError(f"FPS mismatch: A={ds_a.meta.fps}, B={ds_b.meta.fps}")

    fps = ds_a.meta.fps

    # Use common video keys
    common_video_keys = [k for k in ds_a.meta.video_keys if k in ds_b.meta.video_keys]
    logger.info(f"  Common video keys: {common_video_keys}")

    # Determine episode pairs
    if pairs is None:
        n_pairs = min(ds_a.meta.total_episodes, ds_b.meta.total_episodes)
        pairs = [(i, i) for i in range(n_pairs)]
        if ds_a.meta.total_episodes != ds_b.meta.total_episodes:
            logger.warning(
                f"Datasets have different episode counts "
                f"(A={ds_a.meta.total_episodes}, B={ds_b.meta.total_episodes}). "
                f"Using first {n_pairs} pairs, ignoring extras."
            )

    logger.info(f"\nStitching {len(pairs)} episode pairs...")

    # Load episode metadata for both datasets (to get video timestamps)
    ep_meta_a = pd.read_parquet(root / dataset_a_id / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    ep_meta_b = pd.read_parquet(root / dataset_b_id / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    # Load ALL data from both datasets (single bulk parquet files)
    data_path_a = root / dataset_a_id / ds_a.meta.get_data_file_path(0)
    data_path_b = root / dataset_b_id / ds_b.meta.get_data_file_path(0)
    logger.info(f"  Loading data from {data_path_a}...")
    all_data_a = pd.read_parquet(data_path_a)
    logger.info(f"  Loading data from {data_path_b}...")
    all_data_b = pd.read_parquet(data_path_b)
    logger.info(f"  Loaded {len(all_data_a)} + {len(all_data_b)} total rows")

    # Prepare output directory
    output_dir = root / output_repo_id
    if output_dir.exists():
        logger.warning(f"Output directory exists, removing: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "meta").mkdir()
    (output_dir / "data" / "chunk-000").mkdir(parents=True)

    task_str = task or "concatenated task"
    total_frames = 0
    total_videos = 0

    # Build episode metadata (v3.0 format)
    episodes_rows = []

    # Temp dir for extracted video segments
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for out_ep_idx, (a_ep, b_ep) in enumerate(pairs):
            logger.info(f"  [{out_ep_idx + 1}/{len(pairs)}] A_ep{a_ep} + B_ep{b_ep} -> ep{out_ep_idx}")

            # -- DATA: Filter per-episode rows from bulk parquet --
            df_a = all_data_a[all_data_a["episode_index"] == a_ep].copy()
            df_b = all_data_b[all_data_b["episode_index"] == b_ep].copy()

            if len(df_a) == 0:
                raise ValueError(f"No data found for A episode {a_ep}")
            if len(df_b) == 0:
                raise ValueError(f"No data found for B episode {b_ep}")

            # Trim overlap from the start of B
            if overlap_frames > 0 and overlap_frames < len(df_b):
                df_b = df_b.iloc[overlap_frames:]

            len_a = len(df_a)
            len_b = len(df_b)
            ep_length = len_a + len_b

            # Reset indices
            df_a = df_a.reset_index(drop=True)
            df_b = df_b.reset_index(drop=True)

            # Recompute frame_index, timestamp, episode_index, index, task_index
            df_a["frame_index"] = range(len_a)
            df_b["frame_index"] = range(len_a, ep_length)
            df_a["timestamp"] = [i / fps for i in range(len_a)]
            df_b["timestamp"] = [(len_a + i) / fps for i in range(len_b)]
            df_a["episode_index"] = out_ep_idx
            df_b["episode_index"] = out_ep_idx
            df_a["index"] = range(total_frames, total_frames + len_a)
            df_b["index"] = range(total_frames + len_a, total_frames + ep_length)
            df_a["task_index"] = 0
            df_b["task_index"] = 0

            # Combine and write
            common_cols = sorted(set(df_a.columns) & set(df_b.columns))
            df_combined = pd.concat([df_a[common_cols], df_b[common_cols]], ignore_index=True)
            dst_parquet = output_dir / "data" / "chunk-000" / f"episode_{out_ep_idx:06d}.parquet"
            df_combined.to_parquet(dst_parquet)

            # -- VIDEOS: Extract segments and concatenate --
            ep_row = {"episode_index": out_ep_idx}
            ep_row["tasks"] = [task_str]
            ep_row["length"] = ep_length
            ep_row["data/chunk_index"] = 0
            ep_row["data/file_index"] = out_ep_idx
            ep_row["dataset_from_index"] = total_frames
            ep_row["dataset_to_index"] = total_frames + ep_length

            for vk in common_video_keys:
                # Get source video file and timestamps from episode metadata
                a_vid_chunk = int(ep_meta_a.iloc[a_ep][f"videos/{vk}/chunk_index"])
                a_vid_file = int(ep_meta_a.iloc[a_ep][f"videos/{vk}/file_index"])
                a_from_ts = float(ep_meta_a.iloc[a_ep][f"videos/{vk}/from_timestamp"])
                a_to_ts = float(ep_meta_a.iloc[a_ep][f"videos/{vk}/to_timestamp"])

                b_vid_chunk = int(ep_meta_b.iloc[b_ep][f"videos/{vk}/chunk_index"])
                b_vid_file = int(ep_meta_b.iloc[b_ep][f"videos/{vk}/file_index"])
                b_from_ts = float(ep_meta_b.iloc[b_ep][f"videos/{vk}/from_timestamp"])
                b_to_ts = float(ep_meta_b.iloc[b_ep][f"videos/{vk}/to_timestamp"])

                src_vid_a = get_video_source_path(
                    root / dataset_a_id, ds_a.meta.info["video_path"], vk, a_vid_chunk, a_vid_file
                )
                src_vid_b = get_video_source_path(
                    root / dataset_b_id, ds_b.meta.info["video_path"], vk, b_vid_chunk, b_vid_file
                )

                # Output video (one file per episode per camera)
                dst_video_dir = output_dir / "videos" / vk / "chunk-000"
                dst_video_dir.mkdir(parents=True, exist_ok=True)
                dst_video = dst_video_dir / f"episode_{out_ep_idx:06d}.mp4"

                try:
                    # Extract segments from source videos
                    seg_a = tmp_path / f"seg_a_{out_ep_idx}_{vk.replace('.','_')}.mp4"
                    seg_b = tmp_path / f"seg_b_{out_ep_idx}_{vk.replace('.','_')}.mp4"

                    extract_video_segment(src_vid_a, seg_a, a_from_ts, a_to_ts)
                    extract_video_segment(src_vid_b, seg_b, b_from_ts, b_to_ts)

                    # Trim overlap from video B if needed
                    if overlap_frames > 0:
                        overlap_secs = overlap_frames / fps
                        seg_b_trimmed = tmp_path / f"seg_b_trim_{out_ep_idx}_{vk.replace('.','_')}.mp4"
                        extract_video_segment(seg_b, seg_b_trimmed, overlap_secs, b_to_ts - b_from_ts)
                        seg_b = seg_b_trimmed

                    # Concatenate
                    concatenate_two_videos(seg_a, seg_b, dst_video)
                    total_videos += 1

                    # Get actual video duration
                    try:
                        vinfo = get_video_info(dst_video)
                        vid_duration = vinfo["video.duration_s"]
                    except Exception:
                        vid_duration = ep_length / fps

                except Exception as e:
                    logger.warning(f"    {vk}: Video processing failed: {e}")
                    vid_duration = ep_length / fps

                # Episode metadata for this video key (timestamps start at 0 for per-episode files)
                ep_row[f"videos/{vk}/chunk_index"] = 0
                ep_row[f"videos/{vk}/file_index"] = out_ep_idx
                ep_row[f"videos/{vk}/from_timestamp"] = 0.0
                ep_row[f"videos/{vk}/to_timestamp"] = vid_duration

                # Clean up temp segments for this episode/key
                for f in tmp_path.glob(f"seg_*_{out_ep_idx}_{vk.replace('.','_')}*"):
                    f.unlink(missing_ok=True)

            ep_row["meta/episodes/chunk_index"] = 0
            ep_row["meta/episodes/file_index"] = 0
            episodes_rows.append(ep_row)

            total_frames += ep_length
            logger.info(f"    {len_a} + {len_b} = {ep_length} frames")

    # Build tasks
    tasks_df = pd.DataFrame({"task_index": [0], "task": [task_str]})

    # Build info.json
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": ds_a.meta.robot_type,
        "fps": fps,
        "total_episodes": len(pairs),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_videos,
        "total_chunks": 1,
        "chunks_size": len(pairs),
        "data_path": "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/episode_{file_index:06d}.mp4",
        "features": {k: v for k, v in ds_a.meta.features.items() if k in ds_b.meta.features},
        "splits": {"train": f"0:{len(pairs)}"},
    }

    # Write metadata
    logger.info("\nWriting metadata...")
    write_json(info, output_dir / "meta" / "info.json")
    write_tasks(tasks_df, output_dir)

    # Build episodes dataset from rows
    episodes_dict = {}
    if episodes_rows:
        for key in episodes_rows[0]:
            episodes_dict[key] = [row[key] for row in episodes_rows]
    episodes_dataset = datasets.Dataset.from_dict(episodes_dict)
    write_episodes(episodes_dataset, output_dir)

    # Write stats
    logger.info("Writing statistics...")
    try:
        stats_list = []
        if ds_a.meta.stats:
            stats_list.append(ds_a.meta.stats)
        if ds_b.meta.stats:
            stats_list.append(ds_b.meta.stats)
        if stats_list:
            combined_stats = aggregate_stats(stats_list)
            write_stats(combined_stats, output_dir)
    except Exception as e:
        logger.warning(f"Could not write stats (non-fatal): {e}")

    # Create README
    card = create_lerobot_dataset_card(
        tags=["LeRobot", "concatenated", f"lerobot-version:{CODEBASE_VERSION}"],
        dataset_info=info,
        text=(
            f"Concatenated sequential dataset.\n\n"
            f"Stage 1: {dataset_a_id} ({ds_a.meta.total_episodes} episodes)\n"
            f"Stage 2: {dataset_b_id} ({ds_b.meta.total_episodes} episodes)\n\n"
            f"Each episode is a continuous trajectory from stage 1 -> stage 2.\n"
            f"Total: {len(pairs)} episode pairs, {total_frames} frames.\n"
        ),
    )
    card.save(output_dir / "README.md")

    logger.info("\n" + "=" * 60)
    logger.info("Concatenation complete!")
    logger.info(f"  Episodes:    {len(pairs)}")
    logger.info(f"  Frames:      {total_frames}")
    logger.info(f"  Videos:      {total_videos}")
    logger.info(f"  Output:      {output_dir}")
    logger.info("=" * 60)

    # Push to hub
    if push_to_hub:
        logger.info(f"\nPushing to Hub: {output_repo_id}")
        api = HfApi(token=hub_token)
        create_repo(
            repo_id=output_repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=private,
            token=hub_token,
        )
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=output_repo_id,
            repo_type="dataset",
        )
        api.create_tag(
            repo_id=output_repo_id,
            repo_type="dataset",
            tag=CODEBASE_VERSION,
            tag_message=f"LeRobot dataset version {CODEBASE_VERSION}",
        )
        logger.info(f"Pushed: https://huggingface.co/datasets/{output_repo_id}")

    # Validate by loading the output
    logger.info("\nValidating output dataset...")
    try:
        out_ds = LeRobotDataset(output_repo_id, root=output_dir)
        logger.info(f"  Loaded OK: {out_ds.meta.total_episodes} episodes, {out_ds.meta.total_frames} frames")
        return out_ds
    except Exception as e:
        logger.warning(f"  Validation load failed (dataset still on disk): {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate episodes from two sequential datasets into continuous trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/concatenate_episodes.py \\
      --dataset-a RAPOB/pick_stage --dataset-b RAPOB/place_stage \\
      --output-repo-id RAPOB/pick_and_place \\
      --task "pick up pod and place in machine"

  python scripts/concatenate_episodes.py \\
      --dataset-a RAPOB/stage1 --dataset-b RAPOB/stage2 \\
      --output-repo-id RAPOB/combined \\
      --overlap-frames 15

  python scripts/concatenate_episodes.py \\
      --dataset-a RAPOB/stage1 --dataset-b RAPOB/stage2 \\
      --output-repo-id RAPOB/combined \\
      --pairs "0:0,1:1,3:2,4:3"
        """,
    )
    parser.add_argument("--dataset-a", required=True, help="Stage 1 dataset repo ID")
    parser.add_argument("--dataset-b", required=True, help="Stage 2 dataset repo ID")
    parser.add_argument("--output-repo-id", required=True, help="Output repo ID")
    parser.add_argument("--root", type=Path, default=None, help="Local root for datasets")
    parser.add_argument("--task", type=str, default=None, help="Task description for combined dataset")
    parser.add_argument("--overlap-frames", type=int, default=0,
                        help="Frames to trim from start of B (if datasets overlap at boundary)")
    parser.add_argument("--pairs", type=str, default=None,
                        help='Explicit episode pairs as "a:b,a:b,..." (default: pair by index)')
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub-token", type=str, default=None, help="HF API token")
    parser.add_argument("--private", action="store_true", help="Make Hub repo private")

    args = parser.parse_args()
    episode_pairs = parse_pairs(args.pairs) if args.pairs else None

    concatenate_episodes(
        dataset_a_id=args.dataset_a,
        dataset_b_id=args.dataset_b,
        output_repo_id=args.output_repo_id,
        root=args.root,
        task=args.task,
        overlap_frames=args.overlap_frames,
        pairs=episode_pairs,
        push_to_hub=args.push_to_hub,
        hub_token=args.hub_token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
