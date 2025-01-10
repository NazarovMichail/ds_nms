from setuptools import setup, find_packages
import chardet


ENCODING = None
def parse_requirements(filename, encoding=None):
    if encoding is None:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read().splitlines()
    with open(filename, 'r', encoding=ENCODING) as file:
        return file.read().splitlines()

def detect_encoding(filename):
    with open(filename, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

filename = "requirements.txt"
ENCODING = detect_encoding(filename)

setup(
    name='ds_nms',
    version='0.1',
    description='Пакет для анализа данных и машинного обучения',
    author='Назаров Михаил',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt')
)
