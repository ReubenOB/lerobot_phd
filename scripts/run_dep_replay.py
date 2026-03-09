#!/usr/bin/env python3
"""
DEP Level 1 — Replay baseline.

Replays recorded joint positions from each episode of each hard_cutout dataset
onto the robot. This is the simplest baseline: can the robot reproduce
the demonstration perfectly?

Usage:
  # Inside Docker container with lerobot venv:
  source /opt/lerobot_venv/bin/activate

  # Replay episode 0 from a single dataset:
  python3 files/run_dep_replay.py --dataset RAPOB/dep_coffee_pod_gold_no_aria --episode 0

  # Replay all episodes from all no_aria datasets:
  python3 files/run_dep_replay.py --all

  # Replay all episodes from a specific group:
  python3 files/run_dep_replay.py --groups pod_gold cup_blue

  # List what would be replayed:
  python3 files/run_dep_replay.py --all --dry-run

  # Or use lerobot-replay directly:
  lerobot-replay \
    --robot.type=bi_so101_follower \
    --robot.prismatic_gripper=true \
    --robot.left_arm_id=ARM_0 \
    --robot.right_arm_id=ARM_1 \
    --robot.left_arm_port=/dev/ttyACM_left_follower \
    --robot.right_arm_port=/dev/ttyACM_right_follower \
    --dataset.repo_id=RAPOB/dep_coffee_pod_gold_no_aria \
    --dataset.episode=0
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ORG = "RAPOB"

# No-aria datasets for replay testing
NO_ARIA_DATASETS = {
    "pod_gold": {"repo_id": f"{ORG}/dep_coffee_pod_gold_no_aria", "episodes": 40},
    "pod_red": {"repo_id": f"{ORG}/dep_coffee_pod_red_no_aria", "episodes": 30},
    "pod_green": {"repo_id": f"{ORG}/dep_coffee_pod_green_no_aria", "episodes": 25},
    "cup_blue": {"repo_id": f"{ORG}/dep_coffee_cup_blue_no_aria", "episodes": 30},
    "cup_red": {"repo_id": f"{ORG}/dep_coffee_cup_red_no_aria", "episodes": 20},
    "cup_green": {"repo_id": f"{ORG}/dep_coffee_cup_green_no_aria", "episodes": 25},
}

# Robot config for bimanual SO101 follower
ROBOT_ARGS = [
    "--robot.type=bi_so101_follower",
    "--robot.prismatic_gripper=true",
    "--robot.left_arm_id=ARM_0",
    "--robot.right_arm_id=ARM_1",
    "--robot.left_arm_port=/dev/ttyACM_left_follower",
    "--robot.right_arm_port=/dev/ttyACM_right_follower",
]


def run_replay(repo_id: str, episode: int, dry_run: bool = False) -> dict:
    """Run lerobot-replay for a single episode."""
    cmd = [
        "lerobot-replay",
        *ROBOT_ARGS,
        f"--dataset.repo_id={repo_id}",
        f"--dataset.episode={episode}",
    ]

    result = {
        "repo_id": repo_id,
        "episode": episode,
        "command": " ".join(cmd),
        "status": "skipped" if dry_run else "pending",
    }

    if dry_run:
        log.info(f"  [DRY RUN] Would replay: {repo_id} episode {episode}")
        return result

    log.info(f"  Replaying: {repo_id} episode {episode}")
    log.info(f"  Command: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        result["status"] = "success" if proc.returncode == 0 else "failed"
        result["returncode"] = proc.returncode
        if proc.returncode != 0:
            result["stderr"] = proc.stderr[-500:] if proc.stderr else ""
            log.error(f"  FAILED (rc={proc.returncode}): {proc.stderr[-200:]}")
        else:
            log.info(f"  SUCCESS")
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        log.error(f"  TIMEOUT after 300s")
    except FileNotFoundError:
        result["status"] = "command_not_found"
        log.error("  lerobot-replay not found. Is lerobot installed?")

    return result


def main():
    parser_arg = argparse.ArgumentParser(description="DEP Level 1 — Replay baseline")
    parser_arg.add_argument(
        "--dataset",
        type=str,
        help="Single dataset repo_id to replay",
    )
    parser_arg.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode to replay (default: 0). Use -1 for all episodes.",
    )
    parser_arg.add_argument(
        "--groups",
        nargs="*",
        help="Specific groups to replay (e.g., pod_gold cup_blue)",
    )
    parser_arg.add_argument(
        "--all",
        action="store_true",
        help="Replay episode 0 from all hard_cutout datasets",
    )
    parser_arg.add_argument(
        "--all-episodes",
        action="store_true",
        help="Replay ALL episodes (not just episode 0)",
    )
    parser_arg.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be replayed without doing anything",
    )
    parser_arg.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    args = parser_arg.parse_args()

    if not args.dataset and not args.all and not args.groups:
        parser_arg.error("Specify --dataset, --groups, or --all")

    results = []

    if args.dataset:
        # Single dataset mode
        if args.episode == -1 or args.all_episodes:
            # Need to know how many episodes
            log.info(f"Loading dataset metadata for {args.dataset}...")
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            ds = LeRobotDataset(args.dataset)
            for ep in range(ds.num_episodes):
                results.append(run_replay(args.dataset, ep, args.dry_run))
        else:
            results.append(run_replay(args.dataset, args.episode, args.dry_run))
    else:
        # Multi-dataset mode
        datasets = NO_ARIA_DATASETS
        if args.groups:
            datasets = {k: v for k, v in datasets.items() if k in args.groups}

        for group_name, info in datasets.items():
            log.info(f"\n--- {group_name.upper()} ---")
            if args.all_episodes:
                for ep in range(info["episodes"]):
                    results.append(run_replay(info["repo_id"], ep, args.dry_run))
            else:
                results.append(run_replay(info["repo_id"], args.episode, args.dry_run))

    # Summary
    log.info(f"\n{'='*60}")
    log.info("REPLAY SUMMARY")
    log.info(f"{'='*60}")
    for r in results:
        status_icon = {"success": "OK", "failed": "FAIL", "timeout": "TIMEOUT", "skipped": "SKIP"}.get(
            r["status"], "?"
        )
        log.info(f"  [{status_icon}] {r['repo_id']} ep={r['episode']}")

    success = sum(1 for r in results if r["status"] == "success")
    total = len(results)
    log.info(f"\nResults: {success}/{total} succeeded")

    # Save results
    output_path = args.output or f"outputs/dep_replay_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
