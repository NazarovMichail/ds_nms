from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name='my_ml_package',
    version='0.1',
    description='Package for data processing and machine learning model optimization',
    author='Your Name',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt')
)
