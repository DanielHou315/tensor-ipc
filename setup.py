#!/usr/bin/env python3
"""Setup script for tensor-ipc package."""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core requirements - only essential dependencies
core_requirements = [
    "numpy>=1.21.0",
    "cyclonedds==0.10.5",  # Use specific version for compatibility
    "posix-ipc>=1.0.0",
]

setup(
    name="tensor-ipc",
    version="0.1.0",
    author="Daniel Hou",
    author_email="houhd@umich.edu",
    description="Flexible IPC for tensor data sharing with seamless ROS integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danielhou315/tensor-ipc",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Networking",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
        ],
        "test": [
            "pytest>=7.0",
        ],
    },
    include_package_data=True,
)
