from setuptools import setup, find_packages

setup(
    name='bmquilting',
    author="Bruno Madeira",
    author_email="bmad.works@gmail.com",
    version='2.0.0a1',   # TODO remove suffix on release
    packages=find_packages(),
    install_requires=[
        'numpy>=1.23.5',
        'joblib>=1.3.2',
        'opencv-python>=4.0.1.24',
        'pyastar2d>=1.0.6',
        'scikit-learn>=1.3.1', 
    ],
    extras_require={
        'fast': ['numba>=0.62.1'],
        'extras': [
            'bmq-extras @ git+https://github.com/bmad4ever/bmquilting.git#egg=bmq-extras&subdirectory=extras'
        ],
    }
)


