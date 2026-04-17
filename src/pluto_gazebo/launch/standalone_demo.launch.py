"""
Standalone Demo Launch — NO Gazebo required.
Runs: Python hallway simulator + robot_state_publisher + RViz2.
Immediately shows Pluto moving + Bayes filter belief histogram updating.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch.substitutions import Command
from launch_ros.actions import Node


def generate_launch_description():
    gz_pkg   = get_package_share_directory('pluto_gazebo')
    desc_pkg = get_package_share_directory('pluto_description')

    urdf_file = os.path.join(desc_pkg, 'urdf', 'pluto.urdf.xacro')
    rviz_cfg  = os.path.join(gz_pkg,   'config', 'pluto_demo.rviz')

    robot_description = Command(['xacro ', urdf_file])

    return LaunchDescription([

        # ── Robot State Publisher ──────────────────────────────────────────────
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'use_sim_time': False,
                'robot_description': robot_description,
            }],
            output='screen',
        ),

        # ── Hallway Simulator (simulation + Bayes filter in one node) ──────────
        Node(
            package='pluto_gazebo',
            executable='hallway_simulator',
            name='hallway_simulator',
            output='screen',
        ),

        # ── Eye State Publisher ────────────────────────────────────────────────
        Node(
            package='pluto_visualization',
            executable='eye_state_publisher',
            name='eye_state',
            output='screen',
        ),

        # ── Chest Panel Publisher ──────────────────────────────────────────────
        Node(
            package='pluto_visualization',
            executable='chest_panel_publisher',
            name='chest_panel',
            output='screen',
        ),

        # ── RViz2 (slight delay so robot_description is ready) ─────────────────
        # LD_PRELOAD fixes snap/libpthread conflict on Ubuntu 24.04
        TimerAction(period=1.5, actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_cfg],
                additional_env={
                    'LD_PRELOAD': '/lib/x86_64-linux-gnu/libpthread.so.0',
                },
                output='screen',
            ),
        ]),
    ])
