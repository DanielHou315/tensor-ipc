from __future__ import annotations
from typing import Any
import numpy as np
import atexit
import weakref

from .registry import TensorPoolRegistry, PoolMetadata
from .backends import create_backend, TensorBackend
from .utils import get_torch

# Get optional dependencies
torch = get_torch()
TORCH_AVAILABLE = torch is not None

# Global registry for cleanup on exit
_active_publishers = weakref.WeakSet()

def _cleanup_all_publishers():
    """Cleanup all active publishers on program exit."""
    for publisher in list(_active_publishers):
        try:
            publisher.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during exit

# Register exit handler
atexit.register(_cleanup_all_publishers)

class TensorPublisher:
    """Client for writing tensors to a shared pool with notification support."""

    def __init__(self, metadata: PoolMetadata):
        self.metadata = metadata
        self._cleaned_up = False
        
        # Create the appropriate backend as producer
        self.backend: TensorBackend = create_backend(
            backend_type=metadata.backend_type,
            pool_name=metadata.name,
            shape=metadata.shape,
            dtype=metadata.dtype_str,
            history_len=metadata.history_len,
            is_producer=True,
            device=metadata.device
        )
        
        # Register for cleanup on exit
        _active_publishers.add(self)

    def __del__(self):
        """Automatic cleanup on object deletion."""
        self.cleanup()

    @classmethod
    def from_sample(cls,
                    pool_name: str,
                    sample: Any,
                    history_len: int = 1) -> "TensorPublisher":
        """Create a producer and its underlying pool from a sample tensor."""
        if TensorPoolRegistry.pool_exists(pool_name):
            raise ValueError(f"Pool '{pool_name}' already exists.")
        
        metadata = PoolMetadata.from_sample(pool_name, sample, history_len)
        TensorPoolRegistry.register_pool(metadata)
        
        return cls(metadata)

    def publish(self, data: Any) -> None:
        """Publish data to the shared pool."""
        # Validate data matches expected shape and type
        self._validate_input(data)

        # Use the backend's publish method directly
        self.backend.publish(data)

    def _validate_input(self, data: Any) -> None:
        """Validate input data matches pool metadata exactly."""
        # Strict type checking first - must match the original sample type exactly
        if self.metadata.backend_type == "numpy":
            if not isinstance(data, np.ndarray):
                raise TypeError(f"Pool is for numpy.ndarray, but got {type(data)}")
        elif self.metadata.backend_type == "torch":
            if not TORCH_AVAILABLE or torch is None:
                raise TypeError("Pool is for torch.Tensor, but torch is not available")
            if not isinstance(data, torch.Tensor):
                raise TypeError(f"Pool is for torch.Tensor, but got {type(data)}")
        
        if not hasattr(data, 'shape'):
            raise TypeError(f"Data must have a 'shape' attribute, got {type(data)}")
            
        if data.shape != self.metadata.shape:
            raise ValueError(f"Input shape {data.shape} does not match pool shape {self.metadata.shape}")
        
        # Additional validation for numpy and torch
        if self.metadata.backend_type == "numpy":
            if str(data.dtype) != self.metadata.dtype_str:
                raise TypeError(f"Incorrect dtype: expected {self.metadata.dtype_str}, got {data.dtype}")
        elif self.metadata.backend_type == "torch":
            if str(data.dtype) != self.metadata.dtype_str:
                raise TypeError(f"Incorrect dtype: expected {self.metadata.dtype_str}, got {data.dtype}")
            # Only check device for torch tensors - we already validated this is a torch tensor above
            if TORCH_AVAILABLE and torch and isinstance(data, torch.Tensor):
                device_str = str(getattr(data, 'device', 'cpu'))  # Safe access to device attribute
                if device_str != self.metadata.device:
                    raise TypeError(f"Incorrect device: expected {self.metadata.device}, got {device_str}")

    def cleanup(self):
        """Clean up producer resources."""
        if self._cleaned_up:
            return
        
        try:
            # Clean up the backend
            if hasattr(self, 'backend') and self.backend is not None:
                self.backend.cleanup()
                self.backend = None
        except Exception:
            pass  # Ignore cleanup errors
        finally:
            self._cleaned_up = True


