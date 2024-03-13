from setuptools import setup, find_packages
import os


def read_requirements():
    with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), encoding="utf-8") as f:
        return f.read().splitlines()


setup(
    name="feafea",
    version="0.1.0",
    author="Mathspace Pty. Ltd.",
    author_email="mbehabadi@mathspace.com.au",
    description="A simple feature flag system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mathspace/feafea",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
