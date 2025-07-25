"""
PyTorch backends for Victor Python IPC.

Provides TorchBackend that layers on top of NumPy backend for shared memory,
with zero-copy tensor views and device conversion support.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import torch

from ..core.metadata import PoolMetadata
from .base_backend import (
    TensorBackendMixin, 
    HistoryPadStrategy
)
from .numpy_backend import (
    NumpyProducerBackend,
    NumpyConsumerBackend
)

# Type mappings between PyTorch and NumPy
TORCH_TO_NP_DICT = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32:   np.int32,
    torch.int64:   np.int64,
    torch.int16:   np.int16,
    torch.int8:    np.int8,
    torch.uint8:   np.uint8,
    torch.bool:    np.bool_,
}

TORCH_TYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

class TorchBackendMixin(TensorBackendMixin):
    @classmethod
    def to_numpy(cls, data):
        """Convert data to NumPy array if necessary."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.array(data)

    @classmethod
    def from_numpy(cls, data: np.ndarray) -> torch.Tensor:
        """Convert NumPy array to PyTorch tensor."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        raise TypeError(f"Expected np.ndarray, got {type(data)}")

class TorchProducerBackend(NumpyProducerBackend):
    """
    PyTorch producer backend that inherits from NumPy backend.
    
    This backend uses NumPy's POSIX shared memory for the underlying storage
    and provides zero-copy tensor views via torch.from_numpy().
    """
    mixin = TorchBackendMixin
    def __init__(self,
        pool_metadata: PoolMetadata,
        history_pad_strategy: HistoryPadStrategy = "zero",
        force: bool = False
    ):
        # Initialize NumPy backend - it handles all the shared memory setup
        super().__init__(
            pool_metadata=pool_metadata,
            history_pad_strategy=history_pad_strategy,
            force=force
        )

    def _init_tensor_pool(self, force=False) -> None:
        """Initialize the shared tensor pool with the specified metadata."""
        # Call parent to initialize the numpy pool
        super()._init_tensor_pool(force=force)
        # Convert numpy to torch tensor using shared memory
        if self._tensor_pool is not None:
            self._tensor_pool = torch.from_numpy(self._tensor_pool)

    def _initialize_history_padding(self, fill=0) -> None:
        """Initialize history padding based on the specified strategy."""
        assert self._tensor_pool is not None, "Tensor pool must be initialized before padding."
        if self._history_pad_strategy == "zero":
            self._tensor_pool.fill_(0)
        elif self._history_pad_strategy == "fill" and not self._history_initialized:
            # Fill with a specific value for fill-padding
            self._tensor_pool.fill_(fill)
        else:
            raise Exception(f"Unknown history padding strategy: {self._history_pad_strategy}")
        self._history_initialized = True

    def _write_data(self, data: Any, frame_index: int) -> None:
        """Write torch tensor to the current tensor slot."""
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(data)}")
        if self._tensor_pool is None or self._element_shape is None:
            raise RuntimeError("Tensors not created yet")
        if data.shape != self._element_shape:
            raise ValueError(f"Shape mismatch: expected {self._element_shape}, got {data.shape}")
        if data.dtype != self._tensor_pool.dtype:
            raise TypeError(f"Dtype mismatch: expected {self._tensor_pool.dtype}, got {data.dtype}")
        
        # Convert to CPU or assert same device
        if str(self._pool_metadata.device) == "cpu":
            data = data.cpu()  # Ensure data is on CPU for shared memory
        elif str(data.device).startswith('cuda') and str(self._pool_metadata.device).startswith('cuda'):
            # For CUDA devices, extract device indices and compare
            data_device_str = str(data.device)
            pool_device_str = str(self._pool_metadata.device)
            
            # Extract device index (default to 0 if not specified)
            data_idx = data_device_str.split(':')[1] if ':' in data_device_str else '0'
            pool_idx = pool_device_str.split(':')[1] if ':' in pool_device_str else '0'
            
            if data_idx != pool_idx:
                raise ValueError(f"CUDA device index mismatch: pool device {self._pool_metadata.device}, got {data.device}")
        elif str(data.device) != str(self._pool_metadata.device):
            raise ValueError(f"Data must be on the same device as the pool: {self._pool_metadata.device}, got {data.device}")
        
        # Write to array as torch tensor
        self._tensor_pool[frame_index] = data

class TorchConsumerBackend(NumpyConsumerBackend):
    """
    PyTorch consumer backend that inherits from NumPy backend.
    
    This backend provides zero-copy tensor views via torch.from_numpy().
    """
    mixin = TorchBackendMixin
    def __init__(self,
                 pool_metadata: PoolMetadata):
        # Store target device and dtype for tensor conversion
        self._target_device = torch.device(pool_metadata.device if hasattr(pool_metadata, 'device') else 'cpu')
        # Initialize NumPy backend - it handles all the shared memory setup
        super().__init__(pool_metadata)

    def connect(self, pool_metadata) -> bool:
        """Connect to the shared tensor pool using NumPy's mmap."""
        # Call parent to connect to the numpy pool
        if self._connected:
            return True
        
        result = super().connect(pool_metadata)
        if result is False:
            return False
        # Convert numpy array to torch tensor view
        if self._tensor_pool is not None:
            self._tensor_pool = torch.from_numpy(self._tensor_pool)
        return True

    def _read_indices(self, indices):
        """Read data from the tensor pool at specified indices and convert to torch tensor."""
        # Get numpy data from parent (zero-copy)
        if not self._connected or self._tensor_pool is None:
            return None
        
        indices = torch.tensor(indices, device=self._target_device)
        tensor_slice = self._tensor_pool[indices]
        return tensor_slice

    def cleanup(self) -> None:
        super().cleanup()