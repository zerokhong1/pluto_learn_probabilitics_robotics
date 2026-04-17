from setuptools import find_packages, setup

package_name = 'pluto_description'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', ['urdf/pluto.urdf.xacro']),
        ('share/' + package_name + '/launch', ['launch/display.launch.py']),
        ('share/' + package_name + '/config', ['config/pluto.rviz']),
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
        ],
    },
)
