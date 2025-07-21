"""
Backend initialization and factory functions for Victor Python IPC.
"""
from __future__ import annotations

# Import base classes and types
from .base_backend import (
    HistoryPadStrategy,
    TensorConsumerBackend,
    TensorProducerBackend,
)
from ..core.metadata import PoolMetadata

class DependencyError(ImportError):
    """Raised when a required dependency is not available."""
    pass

_available_backends = {}

# 1. Import NumPy backend
import numpy as np
from .numpy_backend import (
    NumpyProducerBackend,
    NumpyConsumerBackend,
)
_available_backends["numpy"] = (NumpyProducerBackend, NumpyConsumerBackend)

# 2. Import PyTorch backend if available
try:
    import torch
except ImportError:
    torch = None
    mp = None

# Import pytorch cpu backend
if torch is not None:
    from .torch_backend import (
        TorchProducerBackend,
        TorchConsumerBackend
    )
    _available_backends["torch"] = (TorchProducerBackend, TorchConsumerBackend)

# Import pytorch cuda backend
if torch is not None and  torch.cuda.is_available():
    from .torch_cuda_backend import (
        TorchCUDAProducerBackend,
        TorchCUDAConsumerBackend,
    )
    _available_backends["torch_cuda"] = (TorchCUDAProducerBackend, TorchCUDAConsumerBackend)

def create_producer_backend(
    pool_metadata: PoolMetadata,
    history_pad_strategy: HistoryPadStrategy = "zero",
    force: bool = False
) -> (TensorProducerBackend, PoolMetadata):
    """Factory function to create the appropriate backend."""
    backend_type = pool_metadata.backend_type
    if backend_type not in _available_backends.keys():
        raise ValueError(f"Unsupported backend type: {backend_type}. Available backends: {list(_available_backends.keys())}")
    return _available_backends[backend_type][0](
        pool_metadata,
        history_pad_strategy=history_pad_strategy,
        force=force,
    )

def create_consumer_backend(
    pool_metadata: PoolMetadata,
) -> TensorConsumerBackend:
    """Factory function to create a consumer backend based on pool metadata."""
    backend_type = pool_metadata.backend_type
    if backend_type not in _available_backends.keys():
        raise ValueError(f"Unsupported backend type: {backend_type}. Available backends: {list(_available_backends.keys())}")
    return _available_backends[backend_type][1](pool_metadata)

def get_available_backends() -> list[str]:
    """Get list of available backend types."""
    return list(_available_backends.keys())

def is_backend_available(backend_type: str) -> bool:
    """Check if a specific backend is available."""
    return backend_type in _available_backends

def detect_backend_from_data(data) -> str:
    """Detect the backend type based on the data type."""
    if isinstance(data, np.ndarray):
        return "numpy"
    elif torch is not None and isinstance(data, torch.Tensor):
        if data.is_cuda:
            return "torch_cuda"
        else:
            return "torch"
    else:
        raise ValueError(f"Unsupported data type for backend detection: {type(data)}")