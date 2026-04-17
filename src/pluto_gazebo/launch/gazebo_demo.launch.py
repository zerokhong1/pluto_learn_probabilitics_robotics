"""
Gazebo Harmonic demo launch.
Starts: gz sim (with pluto_hallway.sdf) + ROS-Gz bridge + robot_state_publisher
        + Bayes filter node + RViz2 + auto cmd_vel driver.

Fixes applied per TROUBLESHOOTING.md:
  - -r flag: Gazebo starts unpaused (no manual click needed)
  - auto_drive node: publishes /cmd_vel so Pluto moves immediately
  - NVIDIA EGL + LD_PRELOAD env fixes for Ubuntu 24.04 + NVIDIA GPU
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch.substitutions import Command
from launch_ros.actions import Node


def generate_launch_description():
    gz_pkg   = get_package_share_directory('pluto_gazebo')
    desc_pkg = get_package_share_directory('pluto_description')

    world_file = os.path.join(gz_pkg, 'worlds', 'pluto_hallway.sdf')
    urdf_file  = os.path.join(desc_pkg, 'urdf', 'pluto.urdf.xacro')
    rviz_cfg   = os.path.join(gz_pkg, 'config', 'pluto_demo.rviz')

    robot_description = Command(['xacro ', urdf_file])

    return LaunchDescription([

        # ── Gazebo Sim ─────────────────────────────────────────────────────────
        # -r   : run immediately, DO NOT start paused (fixes Issue #1 from TROUBLESHOOTING)
        # -v4  : verbose level 4 for debug
        # __EGL_VENDOR_LIBRARY_FILENAMES : force NVIDIA EGL, avoids Mesa segfault
        # LD_PRELOAD : fixes snap/libpthread symbol conflict on Ubuntu 24.04
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', '-v4', world_file],
            output='screen',
            additional_env={
                'GZ_SIM_RESOURCE_PATH': gz_pkg,
                'LD_PRELOAD': '/lib/x86_64-linux-gnu/libpthread.so.0',
                '__EGL_VENDOR_LIBRARY_FILENAMES': '/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
            },
        ),

        # ── Robot State Publisher ──────────────────────────────────────────────
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'use_sim_time': True,
                'robot_description': robot_description,
            }],
            output='screen',
        ),

        # ── ROS ↔ Gazebo Bridge ────────────────────────────────────────────────
        # 3s delay for Gazebo to finish initializing before bridging
        TimerAction(period=3.0, actions=[
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name='gz_bridge',
                arguments=[
                    '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                    '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                    '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                    '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
                    '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
                    '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                    '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
                ],
                remappings=[('/tf', 'tf')],
                output='screen',
            ),
        ]),

        # ── Auto-drive + Bayes Filter + Visualization ──────────────────────────
        # 5s delay: bridge must be up before cmd_vel has a path to Gazebo
        TimerAction(period=5.0, actions=[

            # Publishes /cmd_vel at 10 Hz so Pluto drives forward automatically
            # (fixes Issue #2: no cmd_vel publisher → robot sits still)
            Node(
                package='pluto_gazebo',
                executable='auto_drive',
                name='auto_drive',
                output='screen',
            ),

            Node(
                package='pluto_filters',
                executable='bayes_filter_1d',
                name='bayes_filter_1d',
                output='screen',
            ),
            Node(
                package='pluto_visualization',
                executable='chest_panel_publisher',
                name='chest_panel',
                output='screen',
            ),
            Node(
                package='pluto_visualization',
                executable='eye_state_publisher',
                name='eye_state',
                output='screen',
            ),
        ]),

        # ── RViz2 ──────────────────────────────────────────────────────────────
        TimerAction(period=5.5, actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_cfg],
                parameters=[{'use_sim_time': True}],
                additional_env={
                    'LD_PRELOAD': '/lib/x86_64-linux-gnu/libpthread.so.0',
                },
                output='screen',
            ),
        ]),
    ])
