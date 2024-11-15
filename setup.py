from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="data_management",
    version="1.0",
    description="A data management repository for PAGER project",
    author="Dr. Ruggero Vasile",
    author_email=["ruggero.vasile@gfz-potsdam.de", "ruleva1983@gmail.com"],
    url="",
    packages=find_packages(),
    install_requires=required
)