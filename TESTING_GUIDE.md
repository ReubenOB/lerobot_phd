# Individual Component Testing Guide

This guide provides step-by-step instructions for testing each component individually before running the full system.

## Prerequisites

Build the vigil_aria package:
```bash
cd /home/rapob/vigil_ws
colcon build --packages-select vigil_aria
source install/setup.bash
```

## 1. Test Blink Detection Only

### Start Aria Streaming + Eye Gaze
```bash
# Terminal 1: Aria streaming
ros2 launch vigil_aria aria_streaming.launch.py

# Terminal 2: Eye gaze inference
ros2 launch vigil_aria aria_eye_gaze.launch.py

# Terminal 3: Blink detection
ros2 launch vigil_aria blink_detector.launch.py \
    ear_threshold:=0.21 \
    double_blink_min_interval:=0.3 \
    double_blink_max_interval:=0.6
```

### Test Blink Detection
```bash
# Terminal 4: Blink tester
python3 /home/rapob/vigil_ws/src/vigil_robot/src/vigil_aria/vigil_aria/test_blink_standalone.py
```

**What to test:**
- Single blinks should be detected and counted
- Double blinks (2 blinks within 0.3-0.6s) should trigger the "DOUBLE BLINK TRIGGER!"
- Try different timing patterns
- Check that triple blinks don't trigger double blink

## 2. Test Gaze Gestures Only

### Start Aria Streaming + Eye Gaze + Gaze Gestures
```bash
# Terminal 1: Aria streaming (if not already running)
ros2 launch vigil_aria aria_streaming.launch.py

# Terminal 2: Eye gaze inference (if not already running) 
ros2 launch vigil_aria aria_eye_gaze.launch.py

# Terminal 3: Gaze gesture detection
ros2 launch vigil_aria gaze_gesture.launch.py \
    stare_threshold_yaw:=0.5 \
    stare_threshold_pitch:=0.4 \
    stare_duration:=1.5
```

### Test Gaze Gestures
```bash
# Terminal 4: Gaze tester
python3 /home/rapob/vigil_ws/src/vigil_robot/src/vigil_aria/vigil_aria/test_gaze_standalone.py
```

**What to test:**
- Look straight ahead - should show "center"
- Look up for 2+ seconds - should trigger "RESET TRIGGERED!"
- Look down for 2+ seconds - should trigger "RESET TRIGGERED!"
- Look left/right for 2+ seconds - should trigger "RESET TRIGGERED!"
- Quick glances should NOT trigger reset

## 3. Test Movement Buffer Only (No Robot Required)

```bash
# Test the movement buffer with simulated robot movements
cd /home/rapob/vigil_ws/src/deps/lerobot_phd
python3 src/lerobot/async_inference/test_movement_buffer_standalone.py
```

**Commands to test:**
- `s` - Start recording (watch frame count increase)
- Wait 5-10 seconds to record some movement
- `q` - Stop recording
- `i` - Show buffer info
- `r` - Generate reverse trajectory (should be quick)
- `c` - Clear buffer
- `x` - Exit

**What to verify:**
- Buffer records frames at ~50Hz when recording
- Reverse trajectory generation is fast (<50ms)
- Buffer respects max_frames limit
- Stats show reasonable FPS values

## 4. Test Policy Reset with Keyboard (Single Policy + Real Robot)

This tests the reset functionality with a real robot running a single policy.

### Start Robot Connection
```bash
# Terminal 1: Connect to robot
cd /home/rapob/vigil_ws/src/deps/lerobot_phd
python -m lerobot.scripts.control_robot \
    --robot_type=bi_so101_follower \
    --robot_port_left_leader=/dev/ttyACM_LeftLeader \
    --robot_port_left_follower=/dev/ttyACM_LeftFollower \
    --robot_port_right_leader=/dev/ttyACM_RightLeader \
    --robot_port_right_follower=/dev/ttyACM_RightFollower
```

### Start Policy Server
```bash
# Terminal 2: Start a single policy server for testing
python -m lerobot.async_inference.server \
    --host=localhost \
    --port=8080 \
    --pretrained_path=RAPOB/coffee_pick_pod_act \
    --device=cuda
```

### Start Simple Orchestrator (Modified for Testing)
Create a minimal test orchestrator or use keyboard reset tester:

```bash
# Terminal 3: Policy reset tester (publishes ROS2 messages)
cd /home/rapob/vigil_ws/src/deps/lerobot_phd
python3 src/lerobot/async_inference/test_policy_reset.py
```

### Test Reset Functionality
**Controls:**
- `ENTER` - Simulate double blink (start/stop policy)
- `SPACEBAR` - Simulate reset gesture (trigger reverse trajectory)
- `ESC` - Exit

**What to test:**
1. Press ENTER to start policy execution
2. Let robot move for 5-10 seconds
3. Press SPACEBAR - robot should stop and move in reverse
4. Verify robot returns close to starting position

## 5. Test SARM Progress Monitoring

### Start SARM Node
```bash
# Make sure camera is running first
ros2 run vigil_uncertainty sarm_progress_node \
    --model_path RAPOB/coffee_pick_pod_sarm \
    --camera_topics /camera/top/image_raw \
    --task_description "Pick up the coffee pod that the user is looking at"
```

### Monitor SARM Output
```bash
# Watch SARM progress values
ros2 topic echo /sarm/progress

# Should see Float32 values between 0.0 and 1.0
# Progress should increase as task is completed
```

**What to test:**
- SARM publishes progress values regularly
- Progress increases during task execution
- Progress reaches ~0.95+ when task is nearly complete

## 6. Full System Integration Test

Once individual components work, test everything together:

```bash
# Terminal 1: All Aria nodes
ros2 launch vigil_aria aria_streaming.launch.py
ros2 launch vigil_aria aria_eye_gaze.launch.py  
ros2 launch vigil_aria blink_detector.launch.py
ros2 launch vigil_aria gaze_gesture.launch.py

# Terminal 2: Policy server 1
python -m lerobot.async_inference.server --port=8080 --pretrained_path=RAPOB/coffee_pick_pod_act

# Terminal 3: Policy server 2  
python -m lerobot.async_inference.server --port=8081 --pretrained_path=RAPOB/coffee_make_coffee_act

# Terminal 4: SARM nodes (one for each policy)
ros2 run vigil_uncertainty sarm_progress_node --policy_name pick_pod
ros2 run vigil_uncertainty sarm_progress_node --policy_name make_coffee

# Terminal 5: RND node
ros2 run vigil_uncertainty rnd_uncertainty_node

# Terminal 6: Full orchestrator
cd /home/rapob/vigil_ws/src/deps/lerobot_phd
python -m lerobot.async_inference.orchestrator --config_path=launch/multi_policy_coffee.yaml
```

## Troubleshooting Tips

### Blink Detection Issues
- Check Aria eye tracking camera is publishing: `ros2 topic hz /aria/et_camera/image_raw`
- Verify MediaPipe is installed: `pip install mediapipe`
- Adjust EAR threshold if blinks aren't detected
- Ensure good lighting on your eyes

### Gaze Gesture Issues  
- Check eye gaze direction topic: `ros2 topic echo /aria/eye_gaze/direction`
- Verify gaze angles are in reasonable range (±1 radian)
- Adjust stare thresholds if gestures don't trigger
- Make sure you hold the gaze direction for full duration

### Movement Buffer Issues
- Check joint_states topic exists: `ros2 topic list | grep joint`
- Verify buffer records reasonable joint values
- Test with simulated movement first before real robot

### Robot Connection Issues
- Check device permissions: `ls -la /dev/ttyACM*`
- Verify robot ports match your configuration
- Test basic robot control before adding orchestrator

## Expected SARM Models

You'll need to train these SARM models:
- `RAPOB/coffee_pick_pod_sarm` - For detecting pod picking completion
- `RAPOB/coffee_make_coffee_sarm` - For detecting coffee making completion

Each should be trained on the respective task with appropriate stage annotations.