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
from .core.consumer import TensorConsumer
from .core.producer import TensorProducer

from .core.metadata import (
    PoolMetadata,
    MetadataCreator
)

__all__ = [
    "TensorConsumer",
    "TensorProducer",
    "PoolMetadata",
    "MetadataCreator"
]

from .rosext import ROS_AVAILABLE
if ROS_AVAILABLE:
    from .rosext import (
        ROSTensorProducer,
        ROSTensorConsumer
    )
    __all__.extend([
        "ROSTensorProducer",
        "ROSTensorConsumer"
    ])