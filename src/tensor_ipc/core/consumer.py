"""
The TensorConsumer class provides a unified interface for subscribing to tensor data
from shared memory pools with optional callback support.
"""
from __future__ import annotations
from typing import Optional, Callable, Any
from multiprocessing import Lock
import numpy as np

from .metadata import PoolMetadata, MetadataCreator
from ..backends import (
    create_consumer_backend,
    detect_backend_from_data
)

# DDS imports
from cyclonedds.domain import DomainParticipant
from .dds import DDSConsumer

class TensorConsumer:
    """A simplified consumer for tensor data streams from shared memory pools."""

    def __init__(self, 
        pool_metadata: PoolMetadata,
        keep_last: int = 10,
        dds_participant: DomainParticipant|None = None,
        on_new_data_callback = None
    ):
        """Initialize consumer with user-specified parameters (pool may not exist yet)."""
        # Store user-specified parameters for backend creation
        self._pool_metadata = pool_metadata

        # Backend will be created and handle all connection logic
        self.backend = create_consumer_backend(
            pool_metadata=self._pool_metadata,
        )

        # DDS consumer for notifications
        self._dds_consumer = DDSConsumer(
            self._pool_metadata.name,
            type(self._pool_metadata), 
            dds_participant=dds_participant,
            keep_last=keep_last,
            new_data_callback=self._on_new_progress,
            connection_lost_callback=self._on_connection_lost
        )
        self._connection_lock = Lock()
        self._connected = False

        # Register callback with backend if provided
        self._on_new_data_callback = on_new_data_callback

        # Progress tracking
        self._last_read_index = -1
        self._update_latest_index_lock = Lock()

        # Cleanup flag
        self._cleaned_up = False

    @classmethod
    def from_sample(cls, 
        pool_name: str, 
        sample: Any, 
        dds_participant: DomainParticipant,
        history_len: int = 1,
        keep_last: int = 10,
        callback: Optional[Callable[[Any], None]] = None
    ) -> "TensorConsumer":
        """Create a consumer from a sample tensor/array to infer metadata."""
        # Detect backend type from sample
        backend_type = detect_backend_from_data(sample)
        pool_metadata = MetadataCreator.from_sample(
            name=pool_name,
            data=sample,
            history_len=history_len,
            backend=backend_type
        )
        return cls(
            pool_metadata=pool_metadata,
            dds_participant=dds_participant,
            keep_last=keep_last,
            on_new_data_callback=callback
        )

    def get(self, 
        history_len: int = 1,
        block: bool = False, 
        as_numpy: bool = False,
        latest_first: bool = True,
        timeout: float = 0.1
    ) -> Optional[Any]:
        """
        Get latest tensor data from the pool. Returns None if backend not connected yet.
        
        args:
            history_len: Number of frames to read from the pool.
            - set history_len=1 to read latest frame only, which would return tensor with no history dimension
            block: Whether to block until data is available.
            as_numpy: Convert to NumPy array if True.
            latest_first: If True, return tensor with latest frame first (index 0)
            timeout: Maximum time to wait for data if blocking in seconds
        """
        # Check connection
        if not self._connected:
            self._connect()
            if not self._connected:
                return None
        
        # If blocking, wait for next progress message
        if block:
            progress = self._dds_consumer.read_next_progress(timeout=int(timeout * 1000))
            if progress is None:
                return None
        # Read latest data from backend
        indices = np.arange(
            self.backend.current_latest_index, 
            self.backend.current_latest_index - history_len, -1
        ) % self.backend.max_history_len
        data = self.backend.read(indices, as_numpy=as_numpy)
        if data is None:
            return None
        
        # reverse data if latest_first is False
        if not latest_first:
            # Reverse the order if latest_first is False
            data = data[::-1]
        return data

    def _connect(self):
        """Connect to the tensor pool and initialize the backend."""
        if self._connected:
            return
        
        # Call connection
        res = self._dds_consumer.connect()
        if not res:
            print(f"Failed to connect to DDS for pool '{self._pool_metadata.name}'")
            return
        
        # Otherwise, backend is connected
        recv_pool_metadata = self._dds_consumer.metadata
        if recv_pool_metadata is None \
            or not (recv_pool_metadata == self._pool_metadata):
            print(f"Received metadata does not match expected for pool '{self._pool_metadata.name}'")
            return
        
        # Still update since IPC handle may be different for CUDA
        self._pool_metadata = recv_pool_metadata
        self.backend.connect(self._pool_metadata)

        # Update progress
        with self._connection_lock:
            self._connected = True

    def _on_new_progress(self, data_reader):
        """Handle new progress notification from DDS."""
        if not self._connected:
            self._connect()
            if not self._connected:
                return
            
        # Update latest index
        with self._update_latest_index_lock:
            self.backend.update_frame_index(data_reader.read(N=1)[-1].latest_index)
        
        # Call user callback if registered
        if callable(self._on_new_data_callback):
            data = self.get(
                history_len=1, 
                block=False, 
                as_numpy=False, 
                latest_first=True
            )
            self._on_new_data_callback(data)

    def _on_connection_lost(self) -> None:
        """Handle connection loss notification from DDS."""
        if not self._connected:
            return
        print(f"Connection to tensor pool '{self._pool_metadata.name}' lost.")
        with self._connection_lock:
            self.backend.cleanup()
            self._connected = False

    def __del__(self):
        """Automatic cleanup on object deletion."""
        self.cleanup()

    def cleanup(self):
        """Clean up all resources used by the consumer."""
        if self._cleaned_up or self.backend is None:
            return
        try:
            self.backend.cleanup()
        except Exception as e:
            pass
        finally:
            self._cleaned_up = True