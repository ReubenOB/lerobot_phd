#!/usr/bin/env python3
"""
Standalone episode replayer — no Aria, no orchestrator, no ROS2.

Replays a single episode from any DEP dataset directly on the robot.
Use this to browse episodes and find a good one, then set it in dep_single_episode.yaml.

Usage (inside vigil_ws container):
    source /opt/lerobot_venv/bin/activate
    export HF_LEROBOT_HOME=/home/acumino/vigil_ws/datasets

    # List all episodes with frame counts (no robot needed):
    python3 -m lerobot.async_inference.replay_episode \\
        --dataset RAPOB/dep_coffee_pod_gold_no_aria --list

    # Replay episode 3:
    python3 -m lerobot.async_inference.replay_episode \\
        --dataset RAPOB/dep_coffee_pod_gold_no_aria --episode 3

Robot ports default to the values in dep_single_episode.yaml. Override with --left-port / --right-port.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pyarrow
import pyarrow.parquet as pq


def resolve_dataset_path(repo_id: str) -> Path:
    lerobot_home = os.environ.get(
        "HF_LEROBOT_HOME",
        os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "lerobot"),
    )
    return Path(os.path.expanduser(lerobot_home)) / repo_id


def list_episodes(dataset_repo_id: str):
    ds_root = resolve_dataset_path(dataset_repo_id)
    ep_meta_dir = ds_root / "meta" / "episodes"

    ep_table = None
    for chunk_dir in sorted(ep_meta_dir.iterdir()):
        for pf in sorted(chunk_dir.glob("*.parquet")):
            t = pq.read_table(pf, columns=["episode_index", "length", "stats/timestamp/max"])
            ep_table = t if ep_table is None else pyarrow.concat_tables([ep_table, t])

    rows = ep_table.to_pydict()
    print(f"\nDataset: {dataset_repo_id}")
    print(f"{'Ep':>4}  {'Frames':>7}  {'Duration':>9}")
    print("-" * 26)
    for i in range(len(rows["episode_index"])):
        ep = rows["episode_index"][i]
        length = rows["length"][i]
        dur = rows["stats/timestamp/max"][i][0] if rows["stats/timestamp/max"][i] else 0.0
        print(f"{ep:>4}  {length:>7}  {dur:>8.1f}s")
    print()


def replay_episode(
    dataset_repo_id: str,
    episode_index: int,
    fps: float = 30.0,
    left_port: str = "/dev/ttyACM_left_follower",
    right_port: str = "/dev/ttyACM_right_follower",
    left_arm_id: str = "ARM_0",
    right_arm_id: str = "ARM_1",
):
    from lerobot.robots.bi_so101_follower import BiSO101FollowerConfig
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.async_inference.policy_selector import EpisodeReplayer

    # No cameras needed for open-loop replay
    robot_config = BiSO101FollowerConfig(
        left_arm_port=left_port,
        right_arm_port=right_port,
        left_arm_id=left_arm_id,
        right_arm_id=right_arm_id,
        prismatic_gripper=True,
        cameras={},
    )

    print(f"Connecting to robot ({left_port}, {right_port})...")
    robot = make_robot_from_config(robot_config)
    robot.connect()
    print("Robot connected.")

    try:
        print(f"Loading {dataset_repo_id} episode {episode_index}...")
        replayer = EpisodeReplayer(dataset_repo_id, episode_index)
        replayer.load()
        print(f"Loaded {replayer.num_frames} frames (~{replayer.num_frames / fps:.1f}s at {fps} fps)")
        print("Starting in 2s... Ctrl+C to stop.")
        time.sleep(2.0)

        dt = 1.0 / fps
        step = 0
        while not replayer.done:
            t0 = time.perf_counter()
            action = replayer.next_action()
            if action is None:
                break
            robot.send_action(action)
            step += 1
            if step % 150 == 0:
                print(f"  {replayer.progress:.0%} ({step}/{replayer.num_frames})")
            sleep_time = dt - (time.perf_counter() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"Done ({replayer.num_frames} frames).")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.disconnect()
        print("Robot disconnected.")


def main():
    parser = argparse.ArgumentParser(
        description="Replay a dataset episode on the robot. No Aria or ROS2 needed."
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset repo ID, e.g. RAPOB/dep_coffee_pod_gold_no_aria")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to replay (default: 0)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Replay fps (default: 30)")
    parser.add_argument("--list", action="store_true",
                        help="List all episodes with frame counts, then exit (no robot needed)")
    parser.add_argument("--left-port", default="/dev/ttyACM_left_follower",
                        help="Left arm serial port")
    parser.add_argument("--right-port", default="/dev/ttyACM_right_follower",
                        help="Right arm serial port")
    parser.add_argument("--left-arm-id", default="ARM_0")
    parser.add_argument("--right-arm-id", default="ARM_1")
    args = parser.parse_args()

    if args.list:
        list_episodes(args.dataset)
        return

    replay_episode(
        args.dataset,
        args.episode,
        fps=args.fps,
        left_port=args.left_port,
        right_port=args.right_port,
        left_arm_id=args.left_arm_id,
        right_arm_id=args.right_arm_id,
    )


if __name__ == "__main__":
    main()
