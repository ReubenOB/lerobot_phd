#!/usr/bin/env python3
"""
Queue multiple training jobs to run sequentially.

This script allows you to queue multiple training configurations that will run
one after another, useful for overnight training sessions.

Usage:
    # Define jobs in a YAML config file
    python queue_training.py --config training_queue.yaml
    
    # Or pass jobs directly via command line
    python queue_training.py \
        --jobs \
            RAPOB/dataset1:output1 \
            RAPOB/dataset2:output2 \
            RAPOB/dataset3:output3
    
    # With custom train config
    python queue_training.py \
        --train-config launch/train_v3.yaml \
        --jobs RAPOB/d1:out1 RAPOB/d2:out2

Example training_queue.yaml:
    train_config: launch/train_v3.yaml
    jobs:
      - dataset: RAPOB/coffee_bimanual_aria_gaussian_blur_gold_2
        output: aria_blur_train
        policy_repo: RAPOB/act_blur_v1
      - dataset: RAPOB/coffee_bimanual_aria_brightness_boost_gold_2
        output: aria_brightness_train
        policy_repo: RAPOB/act_brightness_v1
      - dataset: RAPOB/coffee_bimanual_aria_gaussian_attention_gold_2
        output: aria_attention_train
        policy_repo: RAPOB/act_attention_v1
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class TrainingJob:
    """Represents a single training job."""
    
    def __init__(
        self,
        dataset: str,
        output: str,
        policy_repo: Optional[str] = None,
        train_config: str = "launch/train_v3.yaml",
        extra_args: Optional[Dict[str, str]] = None,
    ):
        self.dataset = dataset
        self.output = output
        self.policy_repo = policy_repo
        self.train_config = train_config
        self.extra_args = extra_args or {}
        self.start_time = None
        self.end_time = None
        self.return_code = None
        self.log_file = None
    
    def __repr__(self):
        return f"TrainingJob(dataset={self.dataset}, output={self.output})"
    
    def duration_str(self) -> str:
        """Get formatted duration string."""
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            return f"{int(hours)}h {int(minutes)}m"
        return "N/A"


class TrainingQueue:
    """Manages a queue of training jobs."""
    
    def __init__(
        self,
        jobs: List[TrainingJob],
        work_dir: Path,
        log_dir: Optional[Path] = None,
    ):
        self.jobs = jobs
        self.work_dir = Path(work_dir)
        self.log_dir = Path(log_dir) if log_dir else self.work_dir / "logs" / "queue"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.queue_start_time = None
        self.queue_end_time = None
        
        # Setup queue logging
        self.queue_log_file = self.log_dir / f"queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logging for the queue."""
        logger = logging.getLogger("training_queue")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler
        file_handler = logging.FileHandler(self.queue_log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _build_train_command(self, job: TrainingJob) -> List[str]:
        """Build the lerobot-train command for a job."""
        cmd = [
            "lerobot-train",
            f"--config={job.train_config}",
            f"--dataset.repo_id={job.dataset}",
        ]
        
        # Add policy repo if specified
        if job.policy_repo:
            cmd.append(f"--policy.repo_id={job.policy_repo}")
        
        # Add extra arguments
        for key, value in job.extra_args.items():
            cmd.append(f"--{key}={value}")
        
        return cmd
    
    def run_job(self, job: TrainingJob, job_idx: int) -> bool:
        """Run a single training job.
        
        Returns:
            bool: True if job completed successfully, False otherwise.
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting Job {job_idx + 1}/{len(self.jobs)}")
        self.logger.info(f"Dataset: {job.dataset}")
        self.logger.info(f"Output: {job.output}")
        self.logger.info(f"Policy Repo: {job.policy_repo or 'auto-generated'}")
        self.logger.info("=" * 80)
        
        # Setup job log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        job.log_file = self.log_dir / f"job_{job_idx + 1}_{job.output}_{timestamp}.log"
        
        # Build command
        cmd = self._build_train_command(job)
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        # Run training
        job.start_time = time.time()
        
        try:
            with open(job.log_file, 'w') as log_f:
                log_f.write(f"Training Job Log\n")
                log_f.write(f"Command: {' '.join(cmd)}\n")
                log_f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_f.write("=" * 80 + "\n\n")
                
                # Run the training command
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=self.work_dir,
                )
                
                # Stream output to both console and log file
                for line in process.stdout:
                    print(line, end='')  # Print to console
                    log_f.write(line)    # Write to log file
                    log_f.flush()
                
                # Wait for completion
                job.return_code = process.wait()
            
            job.end_time = time.time()
            
            if job.return_code == 0:
                self.logger.info("✓ Job completed successfully")
                self.logger.info(f"Duration: {job.duration_str()}")
                return True
            else:
                self.logger.error(f"✗ Job failed with return code {job.return_code}")
                self.logger.error(f"Duration: {job.duration_str()}")
                self.logger.error(f"Check log: {job.log_file}")
                return False
        
        except KeyboardInterrupt:
            self.logger.warning("Job interrupted by user")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            job.end_time = time.time()
            job.return_code = -1
            raise
        
        except Exception as e:
            self.logger.error(f"✗ Job failed with exception: {e}")
            job.end_time = time.time()
            job.return_code = -1
            return False
    
    def run(self, stop_on_failure: bool = False) -> Dict[str, any]:
        """Run all jobs in the queue.
        
        Args:
            stop_on_failure: If True, stop the queue if any job fails.
        
        Returns:
            Dict with summary statistics.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING QUEUE STARTED")
        self.logger.info(f"Total jobs: {len(self.jobs)}")
        self.logger.info(f"Work directory: {self.work_dir}")
        self.logger.info(f"Log directory: {self.log_dir}")
        self.logger.info("=" * 80 + "\n")
        
        self.queue_start_time = time.time()
        
        successful_jobs = []
        failed_jobs = []
        
        try:
            for idx, job in enumerate(self.jobs):
                success = self.run_job(job, idx)
                
                if success:
                    successful_jobs.append(job)
                else:
                    failed_jobs.append(job)
                    if stop_on_failure:
                        self.logger.warning("Stopping queue due to job failure")
                        break
                
                # Brief pause between jobs
                if idx < len(self.jobs) - 1:
                    self.logger.info("\nWaiting 5 seconds before next job...\n")
                    time.sleep(5)
        
        except KeyboardInterrupt:
            self.logger.warning("\n\nQueue interrupted by user")
        
        finally:
            self.queue_end_time = time.time()
            self._print_summary(successful_jobs, failed_jobs)
        
        return {
            "total": len(self.jobs),
            "successful": len(successful_jobs),
            "failed": len(failed_jobs),
            "duration": self.queue_end_time - self.queue_start_time,
        }
    
    def _print_summary(self, successful_jobs: List[TrainingJob], failed_jobs: List[TrainingJob]):
        """Print queue summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING QUEUE SUMMARY")
        self.logger.info("=" * 80)
        
        total_duration = self.queue_end_time - self.queue_start_time
        hours = total_duration // 3600
        minutes = (total_duration % 3600) // 60
        
        self.logger.info(f"Total jobs: {len(self.jobs)}")
        self.logger.info(f"Successful: {len(successful_jobs)}")
        self.logger.info(f"Failed: {len(failed_jobs)}")
        self.logger.info(f"Total duration: {int(hours)}h {int(minutes)}m")
        
        if successful_jobs:
            self.logger.info("\n✓ Successful jobs:")
            for job in successful_jobs:
                self.logger.info(f"  - {job.dataset} → {job.output} ({job.duration_str()})")
        
        if failed_jobs:
            self.logger.info("\n✗ Failed jobs:")
            for job in failed_jobs:
                self.logger.info(f"  - {job.dataset} → {job.output} (log: {job.log_file})")
        
        self.logger.info(f"\nQueue log: {self.queue_log_file}")
        self.logger.info("=" * 80)


def load_queue_from_yaml(config_path: Path) -> TrainingQueue:
    """Load training queue from YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    train_config = config.get("train_config", "launch/train_v3.yaml")
    work_dir = Path(config.get("work_dir", os.getcwd()))
    log_dir = Path(config["log_dir"]) if "log_dir" in config else None
    
    jobs = []
    for job_config in config["jobs"]:
        job = TrainingJob(
            dataset=job_config["dataset"],
            output=job_config["output"],
            policy_repo=job_config.get("policy_repo"),
            train_config=job_config.get("train_config", train_config),
            extra_args=job_config.get("extra_args", {}),
        )
        jobs.append(job)
    
    return TrainingQueue(jobs, work_dir=work_dir, log_dir=log_dir)


def create_jobs_from_args(jobs_args: List[str], train_config: str) -> List[TrainingJob]:
    """Create TrainingJob objects from command-line arguments.
    
    Format: dataset_id:output_name[:policy_repo]
    """
    jobs = []
    for job_str in jobs_args:
        parts = job_str.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid job format: {job_str}. Expected 'dataset:output' or 'dataset:output:policy_repo'")
        
        dataset = parts[0]
        output = parts[1]
        policy_repo = parts[2] if len(parts) > 2 else None
        
        jobs.append(TrainingJob(
            dataset=dataset,
            output=output,
            policy_repo=policy_repo,
            train_config=train_config,
        ))
    
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Queue multiple training jobs to run sequentially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From YAML config
  python queue_training.py --config my_training_queue.yaml

  # From command line (dataset:output format)
  python queue_training.py \\
      --jobs \\
          RAPOB/coffee_aria_blur:blur_train \\
          RAPOB/coffee_aria_attention:attention_train \\
          RAPOB/coffee_aria_cutout:cutout_train

  # With custom train config
  python queue_training.py \\
      --train-config launch/train_v3.yaml \\
      --jobs RAPOB/dataset1:output1 RAPOB/dataset2:output2
        """
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file with job definitions"
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        help="Training jobs in format 'dataset:output' or 'dataset:output:policy_repo'"
    )
    parser.add_argument(
        "--train-config",
        default="launch/train_v3.yaml",
        help="Path to train config YAML (default: launch/train_v3.yaml)"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.cwd(),
        help="Working directory for training (default: current directory)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for queue logs (default: work_dir/logs/queue)"
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop the queue if any job fails"
    )
    
    args = parser.parse_args()
    
    # Load jobs
    if args.config:
        queue = load_queue_from_yaml(args.config)
    elif args.jobs:
        jobs = create_jobs_from_args(args.jobs, args.train_config)
        queue = TrainingQueue(
            jobs,
            work_dir=args.work_dir,
            log_dir=args.log_dir,
        )
    else:
        parser.error("Must provide either --config or --jobs")
    
    # Run the queue
    try:
        summary = queue.run(stop_on_failure=args.stop_on_failure)
        
        # Exit with appropriate code
        if summary["failed"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n\nQueue interrupted by user. Exiting.")
        sys.exit(130)


if __name__ == "__main__":
    main()
