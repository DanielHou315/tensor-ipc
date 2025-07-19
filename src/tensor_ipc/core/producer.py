from __future__ import annotations
from typing import Any
import numpy as np

from backends import (
    create_producer_backend,
    create_consumer_backend,
    TensorProducer,
    TensorConsumer
)
from .metadata import PoolMetadata, PoolProgressMessage
from .dds import DDSProducer

class TensorProducer:
    """Client for writing tensors to a shared pool with notification support."""

    def __init__(self, 
        pool_metadata: PoolMetadata,
        dds_participant: Any = None,  # Optional DDS participant for notifications
    ):
        self._pool_metadata = pool_metadata

        # # DDS producer for notifications
        self._dds_producer = DDSProducer(
            dds_participant, 
            pool_metadata.name, 
            pool_metadata
        )
        
        
        # Create the appropriate backend as producer
        self.backend: TensorProducer = create_producer_backend(
            pool_metadata=pool_metadata,
        )

    @classmethod
    def from_sample(cls,
                    pool_name: str,
                    sample: Any,
                    history_len: int = 1) -> "Tensorproducer":
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

    def write(self, data: Any) -> int:
        """Write data to the tensor pool and return the current frame index."""
        # Validate data matches expected shape and type
        self._validate_input(data)

        # Write data using the backend's write method
        idx = self.backend.write(data)

        # Publish progress notification
        message = PoolProgressMessage(
            pool_name=self._pool_metadata.name,
            latest_frame=idx
        )
        self._dds_producer.publish_progress(message)

        return idx

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

    def __del__(self):
        """Automatic cleanup on object deletion."""
        self.cleanup()

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


