"""
PyTorch backends for Victor Python IPC.

Provides TorchBackend that layers on top of NumPy backend for shared memory,
with zero-copy tensor views and device conversion support.
"""
from __future__ import annotations
from typing import Optional, Any, Union
import numpy as np

from .numpy_backend import NumpyBackend
from .base_backend import HistoryPadStrategy
from ..utils import get_torch, DependencyError

# Runtime imports - check availability first
torch_avail = get_torch()
TORCH_AVAILABLE = torch_avail is not None

if not TORCH_AVAILABLE:
    raise DependencyError(
        "PyTorch is not available. Install it with: pip install torch"
    )

# Import torch at module level
import torch
TorchTensor = torch.Tensor
TorchDtype = torch.dtype

class TorchBackend(NumpyBackend):
    """
    PyTorch backend that layers on top of NumPy backend.
    
    This backend uses NumPy's POSIX shared memory for the underlying storage
    and provides zero-copy tensor views via torch.from_numpy().
    
    * Producer/Consumer: Use NumpyBackend for shared memory management
    * Tensors: Convert to/from CPU numpy arrays for sharing
    * Device support: Automatically converts tensors to target device
    """

    # Dtype mapping between PyTorch and NumPy
    _NP_FROM_TORCH = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int32:   np.int32,
        torch.int64:   np.int64,
        torch.int16:   np.int16,
        torch.int8:    np.int8,
        torch.uint8:   np.uint8,
        torch.bool:    np.bool_,
    }
    _TORCH_FROM_NP = {v: k for k, v in _NP_FROM_TORCH.items()}

    def __init__(self,
                 pool_name: str,
                 shape: tuple,
                 dtype: Any,
                 history_len: int = 1,
                 is_producer: bool = False,
                 history_pad_strategy: HistoryPadStrategy = "zero",
                 device: str = "cpu"):

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")

        # Normalize dtype
        if isinstance(dtype, str):
            if dtype.startswith("torch."):
                dtype = getattr(torch, dtype.split(".")[-1])
            else:  # numpy style string
                dtype = self._TORCH_FROM_NP[np.dtype(dtype).type]

        # Store target device for tensor conversion
        self._target_device = torch.device(device)
        self._torch_dtype = dtype
        
        # Convert torch dtype to numpy for underlying storage
        if dtype in self._NP_FROM_TORCH:
            np_dtype = self._NP_FROM_TORCH[dtype]
        else:
            raise ValueError(f"Unsupported torch dtype: {dtype}")

        # Initialize NumPy backend with CPU storage
        super().__init__(
            pool_name=pool_name,
            shape=shape,
            dtype=np_dtype,  # Use numpy dtype for storage
            history_len=history_len,
            is_producer=is_producer,
            history_pad_strategy=history_pad_strategy,
            device="cpu"  # Always use CPU for shared memory
        )

    def _get_single(self, history_index: int) -> Optional[torch.Tensor]:
        """Get single tensor at specified history index, converted to target device."""
        # Get numpy array from parent
        np_data = super()._get_single(history_index)
        if np_data is None:
            return None
        
        # Convert to torch tensor and move to target device
        tensor = torch.from_numpy(np_data).to(dtype=self._torch_dtype)
        if self._target_device.type != "cpu":
            tensor = tensor.to(self._target_device)
        return tensor
    
    def _get_history(self, count: int) -> Optional[torch.Tensor]:
        """Get multiple history entries as a single tensor."""
        # Get numpy array from parent
        np_data = super()._get_history(count)
        if np_data is None:
            return None
        
        # Convert to torch tensor and move to target device
        tensor = torch.from_numpy(np_data).to(dtype=self._torch_dtype)
        if self._target_device.type != "cpu":
            tensor = tensor.to(self._target_device)
        return tensor
    
    def _to_numpy(self, data: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array."""
        # Move to CPU if needed and convert to numpy
        if data.device.type != "cpu":
            data = data.cpu()
        return data.detach().numpy()
    
    def publish(self, data: torch.Tensor) -> None:
        """Publish torch tensor to shared pool."""
        if not self.is_producer:
            raise RuntimeError("Only producers can publish data")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(data)}")
        
        # Convert to CPU numpy array for storage
        if data.device.type != "cpu":
            data = data.cpu()
        
        # Ensure correct dtype
        if data.dtype != self._torch_dtype:
            data = data.to(dtype=self._torch_dtype)
        
        # Convert to numpy and publish via parent
        np_data = data.detach().numpy()
        super().publish(np_data)
