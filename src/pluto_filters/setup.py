from setuptools import find_packages, setup

package_name = 'pluto_filters'

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
            'bayes_filter_1d = pluto_filters.bayes_filter.discrete_bayes_filter:main',
            'kalman_filter_1d = pluto_filters.kalman_filters.kalman_filter:main',
            'ekf_node = pluto_filters.kalman_filters.ekf:main',
            'ukf_node = pluto_filters.kalman_filters.ukf:main',
            'information_filter = pluto_filters.kalman_filters.information_filter:main',
            'particle_filter = pluto_filters.particle_filters.particle_filter:main',
            'motion_model_node = pluto_filters.motion_models.velocity_motion_model:main',
            'beam_model_node = pluto_filters.measurement_models.beam_model:main',
            'likelihood_field_node = pluto_filters.measurement_models.likelihood_field:main',
            'lio_2d_node = pluto_filters.ieskf_lio.lio_2d:main',
        ],
    },
)
