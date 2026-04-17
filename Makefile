.PHONY: clean run stop verify teleop build

SHELL := /bin/bash
SOURCE := source /opt/ros/jazzy/setup.bash && source ~/pluto_robot/install/setup.bash
DISPLAY_ENV := export DISPLAY=:1 __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0

clean:
	@./scripts/cleanup_pluto.sh

build:
	@cd ~/pluto_robot && colcon build --symlink-install 2>&1 | tail -5

run: clean
	@echo "🚀 Launching Pluto (Gazebo + RViz2)..."
	@( $(DISPLAY_ENV) && $(SOURCE) && ros2 launch pluto_gazebo gazebo_demo.launch.py ) &
	@sleep 8
	@$(MAKE) verify

run-standalone: clean
	@echo "🚀 Launching Pluto standalone (no Gazebo)..."
	@( export DISPLAY=:1 && $(SOURCE) && ros2 launch pluto_gazebo standalone_demo.launch.py ) &

stop: clean

verify:
	@echo "🔍 Verifying Pluto is alive..."
	@echo "--- Clock (should tick) ---"
	@( $(SOURCE) && timeout 3 ros2 topic hz /clock 2>&1 | head -3 ) || echo "⚠️  /clock not ticking — Gazebo paused?"
	@echo "--- cmd_vel publishers ---"
	@( $(SOURCE) && ros2 topic info /cmd_vel 2>/dev/null | grep "Publisher count" ) || echo "⚠️  /cmd_vel not found"
	@echo "--- Active nodes ---"
	@( $(SOURCE) && ros2 node list )
	@echo "--- Odom ---"
	@( $(SOURCE) && timeout 3 ros2 topic echo /odom --once 2>/dev/null | grep "x:" | head -2 ) || echo "⚠️  /odom not publishing"

teleop:
	@( $(SOURCE) && ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel )
