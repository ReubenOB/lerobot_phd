#!/usr/bin/env python3
"""
Standalone movement buffer tester with keyboard controls.

Tests the movement buffer with simulated joint positions.
Press 's' to start recording, 'q' to stop, 'r' for reverse trajectory, 'c' to clear.
"""

import time
import threading
import sys
import os

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from movement_buffer import MovementBuffer
import math


def simulate_joint_positions(t: float) -> dict[str, float]:
    """Generate realistic joint positions for bimanual robot."""
    # Simulate smooth sinusoidal movements
    return {
        "left_shoulder_pan": 0.5 * math.sin(0.5 * t),
        "left_shoulder_lift": -1.0 + 0.3 * math.cos(0.3 * t),
        "left_elbow": 1.5 + 0.4 * math.sin(0.7 * t),
        "left_wrist_1": 0.2 * math.cos(0.9 * t),
        "left_wrist_2": 1.0 + 0.1 * math.sin(1.1 * t),
        "left_gripper": 0.0,
        "right_shoulder_pan": -0.5 * math.sin(0.5 * t),
        "right_shoulder_lift": -1.0 + 0.3 * math.cos(0.3 * t + 0.5),
        "right_elbow": 1.5 + 0.4 * math.sin(0.7 * t + 0.3),
        "right_wrist_1": 0.2 * math.cos(0.9 * t + 0.2),
        "right_wrist_2": 1.0 + 0.1 * math.sin(1.1 * t + 0.4),
        "right_gripper": 0.0,
    }


class MovementBufferTester:
    def __init__(self):
        self.buffer = MovementBuffer(max_frames=500, validate_positions=True)
        self.recording = False
        self.start_time = None
        
        print("🤖 Movement Buffer Tester")
        print("Commands:")
        print("  s - Start recording")
        print("  q - Stop recording")
        print("  r - Get reverse trajectory")
        print("  c - Clear buffer")
        print("  i - Show info")
        print("  x - Exit")
        print()
    
    def start_recording(self):
        if not self.recording:
            self.buffer.start_recording()
            self.recording = True
            self.start_time = time.time()
            print("🔴 Recording started...")
        else:
            print("⚠️  Already recording")
    
    def stop_recording(self):
        if self.recording:
            frame_count = self.buffer.stop_recording()
            self.recording = False
            duration = time.time() - self.start_time if self.start_time else 0
            print(f"⏹️  Recording stopped - {frame_count} frames in {duration:.1f}s")
        else:
            print("⚠️  Not recording")
    
    def get_reverse_trajectory(self):
        if self.buffer.frame_count == 0:
            print("⚠️  No data to reverse")
            return
        
        print(f"🔄 Generating reverse trajectory from {self.buffer.frame_count} frames...")
        start_time = time.time()
        
        trajectory = self.buffer.get_reverse_trajectory(
            playback_speed=0.5,  # Half speed
            subsample_factor=2,  # Every other frame
            smooth_window=5      # 5-frame smoothing
        )
        
        duration = time.time() - start_time
        print(f"✅ Generated {len(trajectory)} trajectory points in {duration*1000:.1f}ms")
        
        # Show first and last points
        if trajectory:
            print(f"   First point: {list(trajectory[0].keys())[:3]}...")
            print(f"   Last point: {list(trajectory[-1].keys())[:3]}...")
    
    def show_info(self):
        stats = self.buffer.get_stats()
        print(f"📊 Buffer Info:")
        print(f"   Frame count: {stats['frame_count']}")
        print(f"   Max frames: {stats['max_frames']}")
        print(f"   Is recording: {stats['is_recording']}")
        print(f"   Duration: {stats['duration_s']:.1f}s")
        if stats['frame_count'] > 0:
            print(f"   Recording FPS: {stats['recording_fps']:.1f}")
    
    def clear_buffer(self):
        self.buffer.clear()
        self.recording = False
        print("🗑️  Buffer cleared")
    
    def run_simulation(self):
        """Background thread simulating robot movement."""
        while True:
            if self.recording:
                t = time.time() - self.start_time if self.start_time else 0
                positions = simulate_joint_positions(t)
                self.buffer.record_frame(positions)
            
            time.sleep(0.02)  # 50 Hz
    
    def run(self):
        # Start simulation thread
        sim_thread = threading.Thread(target=self.run_simulation, daemon=True)
        sim_thread.start()
        
        try:
            while True:
                cmd = input("Command (s/q/r/c/i/x): ").strip().lower()
                
                if cmd == 's':
                    self.start_recording()
                elif cmd == 'q':
                    self.stop_recording()
                elif cmd == 'r':
                    self.get_reverse_trajectory()
                elif cmd == 'c':
                    self.clear_buffer()
                elif cmd == 'i':
                    self.show_info()
                elif cmd == 'x':
                    break
                else:
                    print("Invalid command")
                
        except KeyboardInterrupt:
            pass
        
        print("👋 Goodbye!")


if __name__ == '__main__':
    tester = MovementBufferTester()
    tester.run()