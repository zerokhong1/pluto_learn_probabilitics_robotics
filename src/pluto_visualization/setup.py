from setuptools import find_packages, setup

package_name = 'pluto_visualization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'eye_state_publisher = pluto_visualization.eye_state_publisher.eye_state_publisher:main',
            'chest_panel_publisher = pluto_visualization.belief_display.chest_panel_publisher:main',
        ],
    },
)
