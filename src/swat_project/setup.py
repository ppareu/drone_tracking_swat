from setuptools import find_packages, setup
import os
import glob

package_name = 'swat_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/yolov8_model', glob.glob(os.path.join('yolov8_model', '*.pt'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='phb',
    maintainer_email='bin000120@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'minimap = swat_project.minimap:main',
            'cam_capture = swat_project.cam_capture:main',
            'enemy_tracking = swat_project.enemy_tracking:main',
        ],
    },
)
