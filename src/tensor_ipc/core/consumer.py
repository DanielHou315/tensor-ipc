"""
The TensorConsumer class provides a unified interface for subscribing to tensor data
from shared memory pools with optional callback support.
"""
from __future__ import annotations
from typing import Optional, Callable, Any
from ..metadata import PoolMetadata
from ..backends import (
    create_backend,
    TensorConsumerBackend,
    detect_backend_from_data
)

# DDS imports
from cyclonedds.domain import DomainParticipant
from .dds import DDSConsumer

class TensorConsumer:
    """A simplified consumer for tensor data streams from shared memory pools."""

    def __init__(self, 
                 pool_metadata: PoolMetadata,
                 dds_participant: DomainParticipant,
                 keep_last: int = 10,
                 on_new_data_callback = None):
        """Initialize consumer with user-specified parameters (pool may not exist yet)."""
        self.pool_name = pool_name
        self._cleaned_up = False
        
        # Store user-specified parameters for backend creation
        self._metadata = pool_metadata

        # DDS consumer for notifications
        self._dds_consumer = DDSConsumer(
            dds_participant, 
            pool_name,
            type(pool_metadata), 
            keep_last=keep_last,
            new_data_callback=self._on_new_data,
            connection_lost_callback=self._on_connection_lost
        )
        
        # Backend will be created and handle all connection logic
        self.backend: TensorBackend = create_consumer_backend(
            pool_metadata=pool_metadata,
        )
        
        # Register callback with backend if provided
        if callback:
            self.set_callback(callback)

    @classmethod
    def from_sample(cls, pool_name: str, sample: Any, 
                   history_len: int = 1,
                   callback: Optional[Callable[[Any], None]] = None) -> "TensorConsumer":
        """Create a consumer from a sample tensor/array to infer metadata."""
        # Detect backend type from sample
        backend_type = detect_backend_from_data(sample)
        
        # Extract metadata from sample
        shape = sample.shape if hasattr(sample, 'shape') else (1,)
        dtype = str(sample.dtype) if hasattr(sample, 'dtype') else "float64"
        
        # Extract device for torch tensors
        device = "cpu"
        if hasattr(sample, 'device'):
            device = str(sample.device)
        
        return cls(
            pool_name=pool_name,
            backend_type=backend_type,
            shape=shape,
            dtype=dtype,
            history_len=history_len,
            device=device,
            callback=callback
        )

    def get(self, history_len: int = 1, block: bool = True, 
            as_numpy: bool = False, timeout: float = 1.0) -> Optional[Any]:
        """Get tensor data from the pool. Returns None if backend not connected yet."""
        return self.backend.get(history_len, block, as_numpy, timeout)

    def set_callback(self, callback: Callable[[Any], None]) -> None:
        """Set or change the callback for receiving new data. Backend will handle the notification thread in future."""
        # TODO: Backend should handle callbacks with efficient notification thread
        # For now, store callback but don't use it since backend doesn't support it yet
        pass

    def __del__(self):
        """Automatic cleanup on object deletion."""
        self.cleanup()

    def cleanup(self):
        """Clean up all resources used by the consumer."""
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
