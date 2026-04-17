import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    gazebo_pkg = get_package_share_directory('pluto_gazebo')
    desc_pkg = get_package_share_directory('pluto_description')

    world_file = os.path.join(gazebo_pkg, 'worlds', 'hallway_with_doors.sdf')
    urdf_file = os.path.join(desc_pkg, 'urdf', 'pluto.urdf.xacro')

    robot_description = Command(['xacro ', urdf_file])

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        # Gazebo
        Node(
            package='ros_gz_sim',
            executable='gzserver',
            arguments=['--verbose', '-s', 'libgz_sim_physics_system.so',
                       world_file],
            output='screen',
        ),

        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'use_sim_time': True,
                'robot_description': robot_description,
            }],
        ),

        # Spawn robot
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=['-name', 'pluto',
                       '-topic', 'robot_description',
                       '-x', '1.0', '-y', '0.0', '-z', '0.1'],
            output='screen',
        ),

        # Bridge ROS-Gazebo topics
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
                '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
                '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                '/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
                '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
            ],
            output='screen',
        ),
    ])
