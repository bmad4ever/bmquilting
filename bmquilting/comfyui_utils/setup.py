from setuptools import setup, find_packages

setup(
    name='comfyui-utils',
    version='1.0.0',
    packages=find_packages(where='bmquilting', include=['bmquilting.comfy_utils', 'bmquilting.comfy_utils.*']),
    package_dir={'': 'bmquilting'},
    install_requires=[
        'numpy>=1.23.5',
        'joblib~=1.3.2',
        'opencv-python~=4.8.1.78',
        'torch'
    ],
)
