"""
Backend initialization and factory functions for Victor Python IPC.
"""
from __future__ import annotations
from typing import Any

import numpy as np
from ..utils import get_torch, DependencyError

# Import base classes and types
from .base_backend import (
    HistoryPadStrategy,
    TensorConsumerBackend,
    TensorProducerBackend,
)
from .numpy_backend import (
    NumpyProducerBackend,
    NumpyConsumerBackend,
)

# Handle PyTorch imports with availability checking
torch = get_torch()
TORCH_AVAILABLE = torch is not None
CUDA_AVAILABLE = False

# Only import torch backends if PyTorch is available
if TORCH_AVAILABLE:
    try:
        from .torch_backend import TorchBackend
        # Check CUDA availability
        if torch is not None and torch.cuda.is_available():
            CUDA_AVAILABLE = True
    except ImportError:
        # Fallback if torch backends fail to import
        TORCH_AVAILABLE = False
        TorchBackend = None
else:
    TorchBackend = None

# TorchCUDABackend is not available (commented out due to PyTorch compatibility issues)
TorchCUDABackend = None


def create_backend(backend_type: str, pool_name: str, shape: tuple, dtype: Any,
                  history_len: int = 1, is_producer: bool = False, 
                  history_pad_strategy: HistoryPadStrategy = "zero",
                  device: str = "cpu") -> TensorBackend:
    """Factory function to create the appropriate backend."""
    if backend_type == "numpy":
        return NumpyBackend(pool_name, shape, dtype, history_len, is_producer, history_pad_strategy, device)
    
    elif backend_type == "torch":
        if not TORCH_AVAILABLE or TorchBackend is None:
            raise DependencyError(
                "PyTorch backend requested but PyTorch is not available. "
                "Install PyTorch with: pip install torch"
            )
        
        # TorchBackend now handles all devices by converting to CPU for sharing
        # and back to target device when retrieving
        return TorchBackend(pool_name, shape, dtype, history_len, is_producer, history_pad_strategy, device)
    
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


def detect_backend_from_data(data: Any) -> str:
    """Detect the appropriate backend type from sample data."""
    if isinstance(data, np.ndarray):
        return "numpy"
    elif TORCH_AVAILABLE and torch and isinstance(data, torch.Tensor):
        return "torch"
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def get_available_backends() -> list[str]:
    """Get list of available backend types."""
    backends = ["numpy"]
    if TORCH_AVAILABLE:
        backends.append("torch")
    return backends


def is_backend_available(backend_type: str) -> bool:
    """Check if a specific backend is available."""
    if backend_type == "numpy":
        return True
    elif backend_type == "torch":
        return TORCH_AVAILABLE
    else:
        return False


def is_cuda_available() -> bool:
    """Check if CUDA is available for PyTorch backends."""
    return CUDA_AVAILABLE


def get_available_devices() -> list[str]:
    """Get list of available devices."""
    devices = ["cpu"]
    if CUDA_AVAILABLE and torch is not None:
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    return devices


# Export commonly used classes and functions
__all__ = [
    'TensorBackend', 'HistoryPadStrategy',
    'NumpyBackend', 
    'TorchBackend',  # Handles all devices with CPU sharing + device conversion
    'create_backend', 'detect_backend_from_data',
    'get_available_backends', 'is_backend_available',
    'TORCH_AVAILABLE', 'CUDA_AVAILABLE', 'is_cuda_available', 'get_available_devices'
]
