from setuptools import find_packages, setup

package_name = 'pluto_gazebo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/worlds', [
            'worlds/hallway_with_doors.sdf',
            'worlds/symmetric_room.sdf',
            'worlds/landmark_world.sdf',
            'worlds/pluto_hallway.sdf',
        ]),
        ('share/' + package_name + '/launch', [
            'launch/hallway.launch.py',
            'launch/standalone_demo.launch.py',
            'launch/gazebo_demo.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/pluto_demo.rviz',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thailuu',
    maintainer_email='zerokhong1@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hallway_simulator = pluto_gazebo.hallway_simulator:main',
            'auto_drive        = pluto_gazebo.auto_drive:main',
        ],
    },
)
