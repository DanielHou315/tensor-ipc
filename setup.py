#!/usr/bin/env python3
"""Setup script for Victor Python IPC package."""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="victor-python-ipc",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Inter-Process Communication for Victor robots with shared memory support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Victor-Python-IPC",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware :: Hardware Drivers",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=5.0",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
        "test": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-mock>=3.10",
        ],
    },
    entry_points={
        "console_scripts": [
            "victor-ipc=victor_ipc.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "victor_ipc": ["schemas/*.json", "config/*.json"],
    },
)
