from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow>=2.3']

setup(
    name='ddos_detection',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='DDoS detection using Bi-LSTM CNN on CICIDS-2017 dataset.'
)