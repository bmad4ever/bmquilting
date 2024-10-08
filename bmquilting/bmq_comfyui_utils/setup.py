from setuptools import setup, find_packages

setup(
    name='bmq-comfyui-utils',
    version='1.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.23.5',
        'joblib~=1.3.2',
        'opencv-python~=4.8.1.78',
        'torch'
    ],
)
