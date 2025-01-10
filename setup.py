from setuptools import setup, find_packages
import chardet


def detect_encoding(filename):
    with open(filename, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def parse_requirements(filename):
    ENCODING = detect_encoding(filename)
    with open(filename, 'r', encoding=ENCODING) as file:
        return file.read().splitlines()

filename = "requirements.txt"


setup(
    name='ds_nms',
    version='0.1',
    description='Пакет для анализа данных и машинного обучения',
    author='Назаров Михаил',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt')
)
