from setuptools import setup, find_packages

setup(
    name='10K-emerging-risk-detection',
    version='0.1.0',
    packages=find_packages(include=['preprocessing']),
    setup_requires=['pytest-runner', 'flake8'],
    include_package_date=True,
)
