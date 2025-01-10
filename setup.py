from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r', encoding='windows-1251') as file:
        return file.read().splitlines()

setup(
    name='ds_nms',
    version='0.1',
    description='Пакет для анализа данных и машинного обучения',
    author='Назаров Михаил',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt')
)
