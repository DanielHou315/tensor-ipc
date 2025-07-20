"""
Multi-process registry for tensor pools using multiprocessing.managers.
Provides a centralized registry that can be shared across processes.
"""
from dataclasses import dataclass
import os
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence
from typing import ByteString

@dataclass
class PoolProgressMessage(IdlStruct):
    """Message structure for pool progress updates."""
    pool_name: str
    latest_frame: int = 0

@dataclass
class PoolMetadata(IdlStruct):
    """Metadata for a tensor pool that can be serialized across processes."""
    name: str
    shape: sequence[int]

    dtype_str: str  # String representation of dtype for serialization
    backend_type: str = "Virtual"  # "numpy" or "torch"
    history_len: int = 1
    device: str = "cpu"
    
    element_size: int = 0
    total_size: int = 0
    shm_name: str = ""  # Shared memory identifier
    creator_pid: int = 0

    def __eq__(self, other):
        if not isinstance(other, PoolMetadata):
            return NotImplemented
        return (
            self.name == other.name and
            self.shape == other.shape and
            self.dtype_str == other.dtype_str and
            self.backend_type == other.backend_type and
            self.history_len == other.history_len and
            self.device == other.device and
            self.element_size == other.element_size and
            self.total_size == other.total_size and
            self.shm_name == other.shm_name and
            self.creator_pid == other.creator_pid
        )

@dataclass
class TorchCUDAPoolMetadata(PoolMetadata):
    """Specialized metadata for CUDA tensors."""
    device: str = "cuda"
    cuda_ipc_handle: ByteString = b''  # Handle for CUDA IPC

    def __eq__(self, other):
        if not isinstance(other, TorchCUDAPoolMetadata):
            return NotImplemented
        # We don't care if cuda_ipc handle matches since it's a runtime handle
        return super().__eq__(other) and self.cuda_ipc_handle == other.cuda_ipc_handle

class MetadataCreator:
    @staticmethod
    def from_numpy_sample(name: str, sample_data, history_len: int = 1) -> 'PoolMetadata':
        """Create PoolMetadata from a sample tensor/array."""
        import numpy as np
    
        if not isinstance(sample_data, np.ndarray):
            raise TypeError(f"Sample data must be a numpy array, got {type(sample_data)}")

        # Compute metadata from sample data
        shape = list(sample_data.shape)
        element_size = sample_data.dtype.itemsize
        total_size = int(np.prod(shape)) * element_size * history_len
        
        return PoolMetadata(
            name=name,
            shape=shape,
            dtype_str=str(sample_data.dtype),
            backend_type='numpy',
            device='cpu',
            history_len=history_len,
            element_size=element_size,
            total_size=total_size,
            shm_name=f"tensoripc_{name}",
            creator_pid=os.getpid()
        )
    
    @staticmethod
    def from_torch_sample(name: str, sample_data, history_len: int = 1) -> 'PoolMetadata':
        """Create PoolMetadata from a sample tensor/array."""
        import torch
        import os

        if not isinstance(sample_data, torch.Tensor):
            raise TypeError(f"Sample data must be a torch tensor, got {type(sample_data)}")

        if str(sample_data.device) != 'cpu':
            raise ValueError(f"Sample data must be on CPU for metadata creation, detected device: {sample_data.device}")

        shape = list(sample_data.shape)
        dtype_str = str(sample_data.dtype).split('.')[-1]  # Get dtype string like 'float32'
        element_size = sample_data.element_size()

        # Calculate total size for shared memory
        total_size = int(torch.prod(torch.tensor(shape))) * element_size * history_len
        
        return PoolMetadata(
            name=name,
            shape=shape,
            dtype_str=dtype_str,
            backend_type='torch',
            device='cpu',
            history_len=history_len,
            element_size=element_size,
            total_size=total_size,
            shm_name=f"tensoripc_{name}",
            creator_pid=os.getpid()
        )
    
    @classmethod
    def from_torch_cuda_sample(cls,
        name: str,
        sample_data,
        cuda_ipc_handle: ByteString,
        history_len: int = 1
    ) -> 'TorchCUDAPoolMetadata':
        """Create PoolMetadata from a sample CUDA tensor."""
        import torch
        import os
        
        if not (isinstance(sample_data, torch.Tensor) and sample_data.is_cuda):
            raise TypeError("Sample data must be a CUDA tensor.")

        if not str(sample_data.device).startswith('cuda'):
            raise ValueError("Sample data must be on CUDA for metadata creation.")
        
        shape = list(sample_data.shape)
        dtype_str = str(sample_data.dtype)
        element_size = sample_data.element_size()

        # Calculate total size for shared memory
        total_size = int(torch.prod(torch.tensor(shape))) * element_size * history_len
        
        return TorchCUDAPoolMetadata(
            name=name,
            shape=shape,
            dtype_str=dtype_str,
            backend_type='torch',
            device='cuda',
            history_len=history_len,
            element_size=element_size,
            total_size=total_size,
            shm_name=f"pool_{name}_{os.getpid()}",
            creator_pid=os.getpid(),
            cuda_ipc_handle=cuda_ipc_handle
        )
    
    @staticmethod
    def from_sample(name, data, history_len, backend):
        if backend == "numpy":
            return MetadataCreator.from_numpy_sample(
                name=name,
                sample_data=data,
                history_len=history_len
            )
        elif backend == "torch":
            return MetadataCreator.from_torch_sample(
                name=name,
                sample_data=data,
                history_len=history_len
            )
        elif backend == "torch_cuda":
            return MetadataCreator.from_torch_cuda_sample(
                name=name,
                sample_data=data,
                cuda_ipc_handle=b'',
                history_len=history_len
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")