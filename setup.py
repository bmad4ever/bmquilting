from setuptools import setup, find_packages

setup(
    name='bmquilting',
    author="Bruno Madeira",
    author_email="bmad.works@gmail.com",
    version='1.0.0',
    packages=["", "jena2020", "misc"],
    package_dir={
        "": ".",
        "jena2020": "./jena2020",
        "misc": "./misc",
    },
    install_requires=[
        'numpy>=1.23.5',
        'joblib~=1.3.2',
        'opencv-python~=4.8.1.78',
        'scikit-learn~=1.3.1',
    ],
    extras_require={
        'comfy': [
            'comfyui-utils @ TODO'
        ],
    }
)
