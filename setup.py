from setuptools import setup, find_packages

setup(
    name='bmquilting',
    author="Bruno Madeira",
    author_email="bmad.works@gmail.com",
    version='1.0.0',
    packages=find_packages(exclude=['bmquilting.comfyui_utils', 'bmquilting.comfyui_utils.*']),
    install_requires=[
        'numpy>=1.23.5',
        'joblib~=1.3.2',
        'opencv-python~=4.8.1.78',
        'scikit-learn~=1.3.1',
    ],
    extras_require={
        'comfy': [
            'bmquilting.comfyui-utils @ git+https://github.com/bmad4ever/bmquilting.git#egg=bmquilting.comfyui-utils&subdirectory=bmquilting/comfyui_utils'
        ],
    }
)
