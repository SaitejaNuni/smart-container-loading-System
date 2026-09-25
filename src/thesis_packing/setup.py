from setuptools import find_packages, setup

package_name = 'thesis_packing'

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
    maintainer='saiteja',
    maintainer_email='saiteja.nuni@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'placement_planner_node = thesis_packing.placement_planner_node:main',
            'random_package_generator_node = thesis_packing.random_package_generator_node:main',
            'container_state_node = thesis_packing.container_state_node:main',
            'visualization_node = thesis_packing.visualization_node:main',
        ],
    },
)
