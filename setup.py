from setuptools import setup, find_packages
import chardet


def parse_requirements(filename):
    with open(filename, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        print(f"Detected encoding: {result['encoding']}")

setup(
    name='ds_nms',
    version='0.1',
    description='Пакет для анализа данных и машинного обучения',
    author='Назаров Михаил',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt')
)
