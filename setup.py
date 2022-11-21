#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O., MashaPo

from setuptools import setup
from typing import List


def readme() -> str:
    with open('README.md', 'r', encoding='utf-8') as f_readme:
        return f_readme.read()


def requirements() -> List[str]:
    with open('requirements.txt', 'r', encoding='utf-8') as f_requirements:
        return f_requirements.read()


setup(name='telegram_notifier',
      version='0.0.1',
      description='Package that helps you to log your training processes via telegram bot.',
      long_description=readme(),
      url='https://github.com/shpq/telegram-notifier',
      author='Denis Manichkin',
      keywords='telegram bot machine learning logging',
      author_email='denis.manichkin@gmail.com',
      packages=['telegram_notifier'],
      install_requires=requirements(),
      include_package_data=True)

# python3 setup.py bdist_wheel
