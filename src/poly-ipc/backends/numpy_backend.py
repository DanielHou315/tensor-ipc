"""
NumPy backend for Victor Python IPC.

Provides NumpyBackend for CPU-based numpy arrays with zero-copy shared memory communication.
"""
from __future__ import annotations
from typing import Optional, Any, Union, Type
import numpy as np
import mmap

from cyclonedds.domain import DomainParticipant
from .base_backend import (
    TensorProducerBackend,
    TensorConsumerBackend,
    HistoryPadStrategy
)
from ..metadata import PoolMetadata, TorchCUDAPoolMetadata
from ..utils import require_posix_ipc

# Get POSIX IPC
posix_ipc = require_posix_ipc()

class NumpyProducerBackend(TensorProducerBackend):
    """Native NumPy backend with single tensor pool and history padding."""
    
    def __init__(self, 
        dds_participant: DomainParticipant,
        pool_metadata: Union[PoolMetadata, TorchCUDAPoolMetadata],
        history_pad_strategy: HistoryPadStrategy = "zero",
    ):
        self._metadata = pool_metadata
        # Set shared memory name before parent init
        self._shape_tuple = tuple(pool_metadata.shape)
        self._pool_shape = (pool_metadata.history_len,) + self._shape_tuple
        self._shm_name = f"polyipc_{self._metadata.name}"

        super().__init__(pool_metadata, dds_participant, history_pad_strategy)
        self._shared_memory: Optional[Any] = None  # posix_ipc.SharedMemory
        self._shared_mmap: Optional[mmap.mmap] = None  # Memory map for tensor data
    
    def _init_tensor_pool(self) -> None:
        """Create NumPy tensor pool with shape (history_len, *shape)."""
        # Create pool shape: (history_len, *sample_shape)

        # Producer creates the shared memory using POSIX IPC
        self._shared_memory = posix_ipc.SharedMemory(
            self._shm_name, 
            flags=posix_ipc.O_CREX, 
            size=self._metadata.total_size
        )
        
        # Create memory map and numpy array
        self._shared_mmap = mmap.mmap(self._shared_memory.fd, self._metadata.total_size)
        self._shared_memory.close_fd()  # Close fd, keep shared memory object
        self._tensor_pool = np.ndarray(self._pool_shape, dtype=self._metadata.dtype_str, buffer=self._shared_mmap)
        
        self._initialize_history_padding()
        self._is_initialized = True
        self._pool_connected = True

    def _write_data(self, data: np.ndarray, frame_index: int) -> None:
        """Write data to the current tensor slot."""
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(data)}")
        if self._tensor_pool is None or self._shape_tuple is None:
            raise RuntimeError("Tensors not created yet")
        
        if data.shape != self._shape_tuple:
            raise ValueError(f"Shape mismatch: expected {self._shape_tuple}, got {data.shape}")
        if data.dtype != self._tensor_pool.dtype:
            raise TypeError(f"Dtype mismatch: expected {self._tensor_pool.dtype}, got {data.dtype}")
        
        # Copy data into current slot
        self._tensor_pool[frame_index] = data

    def cleanup(self) -> None:
        """Clean up NumPy backend resources."""
        super().cleanup()
        
        # Clean up shared memory map
        if self._shared_mmap is not None:
            try:
                self._shared_mmap.close()
            except Exception:
                pass
            self._shared_mmap = None
        # Clean up shared memory
        if self._shared_memory is not None:
            try:
                self._shared_memory.close()
            except Exception:
                pass
            try:
                self._shared_memory.unlink()
            except (posix_ipc.ExistentialError, AttributeError):
                pass
            self._shared_memory = None
        self._tensor_pool = None


class NumpyConsumerBackend(TensorConsumerBackend):
    """Native NumPy backend for consuming shared memory pools."""
    def __init__(self,
        pool_name: str,
        # dds_participant: DomainParticipant, 
        # metadata_type: Type[Union[PoolMetadata, TorchCUDAPoolMetadata]], 
        # keep_last: int = 10
    ):
        # self._metadata_type = pool_metadata_type
        # Set shared memory name before parent init
        self._shm_name = f"polyipc_{pool_name}"

        super().__init__(pool_name)
        self._shared_memory: Optional[Any] = None

    def _connect_tensor_pool(self, pool_metadata) -> None:
        """Connect to existing NumPy tensor pool."""
        # Consumer connects to existing shared memory
        self._metadata = pool_metadata
        assert isinstance(pool_metadata, (PoolMetadata, TorchCUDAPoolMetadata)), \
            f"Expected PoolMetadata or TorchCUDAPoolMetadata, got {type(pool_metadata)}"
        
        # Actually connect to shared memory
        shm_name = f"/polyipc_{self._shm_name.lstrip('/')}"
        self._shared_memory = posix_ipc.SharedMemory(shm_name)
        
        # Create memory map and numpy array
        self._shared_mmap = mmap.mmap(self._shared_memory.fd, pool_metadata.total_size)
        self._shared_memory.close_fd()  # Close fd, keep shared memory object
        self._pool_shape = (pool_metadata.history_len,) + tuple(pool_metadata.shape)
        self._tensor_pool = np.ndarray(self._pool_shape, dtype=pool_metadata.dtype_str, buffer=self._shared_mmap)
        
        self._pool_connected = True

    def read(self, indices):
        """Read data from the tensor pool at specified indices."""
        if self._tensor_pool is None:
            return None
        tensor_slice = self._tensor_pool[indices]
        tensor_slice.flags.writeable = False  # Ensure read-only access
        return tensor_slice
    
    def read_history(self, latest_index, history_len):
        """Read a history of data from the tensor pool."""
        if self._tensor_pool is None or self._metadata is None:
            return None
        assert history_len > 0 and history_len <= self._metadata.history_len, \
            f"History length must be in range [1, {self._metadata.history_len}]"
        
        # Actually generate indices
        indices = [(latest_index - i) % self._metadata.history_len for i in range(history_len)]
        return self.read(indices)

    # def _get_single(self, history_index: int) -> Optional[np.ndarray]:
    #     """Get single numpy array at specified history index."""
    #     if not (0 <= history_index < self.history_len):
    #         raise ValueError(f"History index {history_index} out of range [0, {self.history_len})")
        
    #     if self._tensor_pool is None:
    #         return None
        
    #     # Get current state but don't update last_seen_sequence yet
    #     current_sequence, current_index = self._get_shared_state()

    #     # Check if there's actually new data
    #     if current_sequence <= self._last_seen_sequence and self._last_seen_sequence != -1:
    #         return None  # No new data available
        
    #     # Update last seen sequence now that we're consuming data
    #     self._last_seen_sequence = current_sequence
        
    #     # Calculate actual index (accounting for circular buffer)
    #     # Most recent data is at (current_index - 1) % history_len
    #     actual_index = (current_index - 1 - history_index) % self.history_len
        
    #     data = self._tensor_pool[actual_index]
    #     result = data.copy()  # Always return copy for safety

    #     return result
    
    # def _get_history(self, count: int) -> Optional[Any]:
    #     """Get multiple history entries efficiently."""
    #     if self._tensor_pool is None:
    #         return None
            
    #     count = min(count, self.history_len)
    #     if count <= 0:
    #         return None
        
    #     # Get current state but don't update last_seen_sequence yet
    #     current_sequence, current_index = self._get_shared_state()
        
    #     # Check if there's actually new data
    #     if current_sequence <= self._last_seen_sequence and self._last_seen_sequence != -1:
    #         return None  # No new data available
        
    #     # Update last seen sequence now that we're consuming data
    #     self._last_seen_sequence = current_sequence
        
    #     # Create indices for history (most recent first)
    #     indices = [(current_index - 1 - i) % self.history_len for i in range(count)]
        
    #     # Always return copy for safety
    #     data = self._tensor_pool[indices]
    #     return data.copy()
    
    def to_numpy(self, data: Any) -> np.ndarray:
        """Convert the input data to a NumPy array."""
        return np.asarray(data)
    
    def cleanup(self) -> None:
        """Clean up NumPy backend resources."""
        super().cleanup()
        
        # Clean up shared memory map
        if self._shared_mmap is not None:
            try:
                self._shared_mmap.close()
            except Exception:
                pass
            self._shared_mmap = None
        
        # Clean up shared memory
        if self._shared_memory is not None:
            try:
                self._shared_memory.close()
            except Exception:
                pass
            self._shared_memory = None
        self._tensor_pool = None
