#!/usr/bin/env python3
"""
Policy reset tester with keyboard spacebar control.

This script runs a single policy and allows you to test the reset functionality
using the spacebar instead of gaze gestures.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import threading
import time
import sys
import select
import tty
import termios


class PolicyResetTester(Node):
    def __init__(self):
        super().__init__('policy_reset_tester')
        
        # Publisher to trigger reset (simulates gaze gesture)
        self.reset_pub = self.create_publisher(
            Bool,
            '/aria/gaze_gesture/reset_triggered',
            10
        )
        
        # Publisher to start/stop policy (simulates double blink)
        self.blink_pub = self.create_publisher(
            Bool,
            '/aria/blink/double_detected',
            10
        )
        
        self.get_logger().info("🎮 Policy Reset Tester")
        self.get_logger().info("Controls:")
        self.get_logger().info("  SPACEBAR - Trigger reset")
        self.get_logger().info("  ENTER - Double blink (start/stop policy)")
        self.get_logger().info("  ESC - Exit")
        self.get_logger().info("")
        self.get_logger().info("Start your policy in another terminal, then use these controls")
        
        # Setup non-blocking keyboard input
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
        # Start keyboard listener
        self.running = True
        self.key_thread = threading.Thread(target=self.key_listener, daemon=True)
        self.key_thread.start()
    
    def key_listener(self):
        """Listen for keyboard input in a separate thread."""
        while self.running:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                
                if key == ' ':  # Spacebar
                    self.trigger_reset()
                elif key == '\r' or key == '\n':  # Enter
                    self.trigger_double_blink()
                elif key == '\x1b':  # ESC
                    self.get_logger().info("ESC pressed - exiting...")
                    self.running = False
                    break
    
    def trigger_reset(self):
        """Publish reset trigger message."""
        msg = Bool()
        msg.data = True
        self.reset_pub.publish(msg)
        self.get_logger().info("🔄 RESET TRIGGERED! (Spacebar pressed)")
    
    def trigger_double_blink(self):
        """Publish double blink trigger message."""
        msg = Bool()
        msg.data = True
        self.blink_pub.publish(msg)
        self.get_logger().info("👁️👁️ DOUBLE BLINK TRIGGERED! (Enter pressed)")
    
    def cleanup(self):
        """Restore terminal settings."""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


def main():
    rclpy.init()
    node = PolicyResetTester()
    
    try:
        # Keep the node alive while keyboard thread runs
        while node.running:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.get_logger().info("👋 Policy reset tester stopped")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()