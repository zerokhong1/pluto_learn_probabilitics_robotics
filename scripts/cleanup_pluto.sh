#!/bin/bash
# Kills all Gazebo + ROS2 Pluto processes for a clean slate.
# See TROUBLESHOOTING.md Issue #3 — zombie processes from previous runs.

set -e

echo "🧹 Cleaning up Pluto processes..."

pkill -9 -f "gz sim"                || true
pkill -9 -f "gzserver"              || true
pkill -9 -f "gzclient"              || true
pkill -9 -f "ros2"                  || true
pkill -9 -f "robot_state_publisher" || true
pkill -9 -f "joint_state_publisher" || true
pkill -9 -f "ros_gz_bridge"         || true
pkill -9 -f "parameter_bridge"      || true
pkill -9 -f "rviz2"                 || true
pkill -9 -f "hallway_simulator"     || true
pkill -9 -f "auto_drive"            || true
pkill -9 -f "bayes_filter"          || true
pkill -9 -f "chest_panel"           || true
pkill -9 -f "eye_state"             || true

# Clean up DDS and Gazebo shared state
rm -rf /dev/shm/fastrtps_* 2>/dev/null || true
rm -rf /tmp/gz-*            2>/dev/null || true

sleep 1

REMAINING=$(ps aux | grep -E "gz sim|ros2|robot_state|rviz2|hallway_sim|auto_drive" | grep -v grep | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo "⚠️  $REMAINING process(es) still alive (may need sudo):"
    ps aux | grep -E "gz sim|ros2|robot_state|rviz2" | grep -v grep
else
    echo "✅ Clean. Ready to launch."
fi
