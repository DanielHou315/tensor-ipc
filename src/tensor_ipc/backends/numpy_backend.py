"""
NumPy backend for Victor Python IPC.

Provides NumpyBackend for CPU-based numpy arrays with zero-copy shared memory communication.
"""
from __future__ import annotations
from typing import Union
import numpy as np
import mmap

from .base_backend import (
    TensorProducerBackend,
    TensorConsumerBackend,
    HistoryPadStrategy
)
from ..metadata import PoolMetadata, TorchCUDAPoolMetadata
from ..utils import require_posix_ipc

# Get POSIX IPC
posix_ipc = require_posix_ipc()


NUMPY_TYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "int8": np.int8,
    "uint16": np.uint16,
    "int16": np.int16,
    "bool": np.bool_,
}

class NumpyProducerBackend(TensorProducerBackend):
    """Native NumPy backend with single tensor pool and history padding."""
    
    def __init__(self, 
        pool_metadata: Union[PoolMetadata, TorchCUDAPoolMetadata],
        history_pad_strategy: HistoryPadStrategy = "zero",
    ):
        super().__init__(
            pool_metadata=pool_metadata,
            history_pad_strategy=history_pad_strategy
        )

    def _init_tensor_pool(self) -> None:
        """Create NumPy tensor pool with shape (history_len, *shape)."""
        # Create pool shape: (history_len, *sample_shape)
        pool_metadata = self._pool_metadata
        
        self._element_shape = tuple(pool_metadata.shape)
        self._pool_shape = (pool_metadata.history_len,) + self._element_shape

        # Producer creates the shared memory using POSIX IPC
        self._shared_memory = posix_ipc.SharedMemory(
            self._pool_metadata.shm_name, 
            flags=posix_ipc.O_CREX, 
            size=self._pool_metadata.total_size
        )
        
        # Create memory map and numpy array
        self._shared_mmap = mmap.mmap(self._shared_memory.fd, self._pool_metadata.total_size)
        self._shared_memory.close_fd()  # Close fd, keep shared memory object

        # Check if numpy dtype is supported
        assert self._pool_metadata.dtype_str in NUMPY_TYPE_MAP, \
            f"Unsupported dtype: {self._pool_metadata.dtype_str}. Supported types: {list(NUMPY_TYPE_MAP.keys())}"
        self._tensor_pool = np.ndarray(
            self._pool_shape,
            dtype=NUMPY_TYPE_MAP[self._pool_metadata.dtype_str],
            buffer=self._shared_mmap
        )

    def _write_data(self, data: np.ndarray, frame_index: int) -> None:
        """Write data to the current tensor slot."""
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(data)}")
        if self._tensor_pool is None or self._element_shape is None:
            raise RuntimeError("Tensors not created yet")
        
        if data.shape != self._element_shape:
            raise ValueError(f"Shape mismatch: expected {self._element_shape}, got {data.shape}")
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
        pool_metadata: Union[PoolMetadata, TorchCUDAPoolMetadata],
    ):
        super().__init__(pool_metadata)

    def _connect_tensor_pool(self, pool_metadata) -> None:
        """Connect to existing NumPy tensor pool."""
        # Consumer connects to existing shared memory
        self._metadata = pool_metadata
        assert isinstance(pool_metadata, (PoolMetadata, TorchCUDAPoolMetadata)), \
            f"Expected PoolMetadata or TorchCUDAPoolMetadata, got {type(pool_metadata)}"
        self._pool_shape = (pool_metadata.history_len,) + tuple(pool_metadata.shape)

        # Actually connect to shared memory
        self._shared_memory = posix_ipc.SharedMemory(self._pool_metadata.shm_name)
        # Create memory map and numpy array
        self._shared_mmap = mmap.mmap(self._shared_memory.fd, pool_metadata.total_size)
        self._shared_memory.close_fd()  # Close fd, keep shared memory object

        # Fix: Use NUMPY_TYPE_MAP instead of string dtype
        assert pool_metadata.dtype_str in NUMPY_TYPE_MAP, \
            f"Unsupported dtype: {pool_metadata.dtype_str}. Supported types: {list(NUMPY_TYPE_MAP.keys())}"
        
        self._tensor_pool = np.ndarray(
            self._pool_shape,
            dtype=NUMPY_TYPE_MAP[pool_metadata.dtype_str],
            buffer=self._shared_mmap
        )
        self._pool_connected = True
        self._connected = True

    def _read_indices(self, indices):
        """Read data from the tensor pool at specified indices."""
        if self._tensor_pool is None:
            return None
        tensor_slice = self._tensor_pool[indices]
        tensor_slice.flags.writeable = False  # Ensure read-only access
        return tensor_slice

    def _to_numpy(self, data):
        """Convert data to NumPy array if necessary."""
        # Data is already numpy array
        return data

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
