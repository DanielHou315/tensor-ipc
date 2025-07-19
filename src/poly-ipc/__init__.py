"""Victor Python IPC - Inter-Process Communication for Victor robots.

This package provides shared memory IPC capabilities for Victor robots,
allowing efficient communication between processes without ROS dependencies.

The new adapter system supports mixed ROS, numpy, and torch tensor communication
with flexible transport modes (SHM for same-machine, ROS for cross-machine).
"""

__version__ = "0.1.0"
__author__ = "Daniel Hou"
__email__ = "houhd@umich.edu"

# Core shared memory classes
from .producer import TensorPublisher
from .consumer import TensorConsumer
from .registry import PoolMetadata, TensorPoolRegistry
from .backends import NumpyBackend, TorchBackend, TorchCUDABackend, create_backend, get_available_backends, is_backend_available

# The adapter system is now deprecated in favor of the unified TensorStream.

__all__ = [
    # Core classes
    "TensorPublisher",
    "TensorConsumer", 
    "PoolMetadata",
    "TensorPoolRegistry",
    # Backend classes
    "NumpyBackend",
    "TorchBackend",
    "TorchCUDABackend",
    "create_backend",
    "get_available_backends",
    "is_backend_available",
]
