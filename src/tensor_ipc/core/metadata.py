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
    
    # Tensor reconstruction fields
    tensor_size: sequence[int] = ()
    tensor_stride: sequence[int] = ()
    tensor_offset: int = 0
    
    # CUDA IPC fields from _share_cuda_()
    storage_device: int = 0
    storage_handle: ByteString = b''
    storage_size_bytes: int = 0
    storage_offset_bytes: int = 0
    ref_counter_handle: ByteString = b''
    ref_counter_offset: int = 0
    event_handle: ByteString = b''
    event_sync_required: bool = True

    def __eq__(self, other):
        if not isinstance(other, TorchCUDAPoolMetadata):
            return NotImplemented
        # Compare base fields but exclude runtime-specific handles
        base_equal = (
            self.name == other.name and
            self.shape == other.shape and
            self.dtype_str == other.dtype_str and
            self.backend_type == other.backend_type and
            self.history_len == other.history_len and
            self.device == other.device and
            self.element_size == other.element_size and
            self.total_size == other.total_size and
            self.tensor_size == other.tensor_size and
            self.tensor_stride == other.tensor_stride and
            self.tensor_offset == other.tensor_offset and
            self.storage_device == other.storage_device and
            self.storage_size_bytes == other.storage_size_bytes and
            self.storage_offset_bytes == other.storage_offset_bytes and
            self.ref_counter_offset == other.ref_counter_offset and
            self.event_sync_required == other.event_sync_required
        )
        # Note: We don't compare handles as they are runtime-specific
        return base_equal

    def is_valid(self) -> bool:
        """
        Check if metadata is valid for reconstruction. 
        """
        return len(self.storage_handle) > 0 \
            and len(self.ref_counter_handle) > 0 \
            and len(self.event_handle) > 0 \

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
        history_len: int = 1,
        untyped_storage_inst = None,
    ) -> 'TorchCUDAPoolMetadata':
        """Create TorchCUDAPoolMetadata from a sample CUDA tensor."""
        import torch
        import os
        
        if not (isinstance(sample_data, torch.Tensor) and sample_data.is_cuda):
            raise TypeError("Sample data must be a CUDA tensor.")

        if not str(sample_data.device).startswith('cuda'):
            raise ValueError("Sample data must be on CUDA for metadata creation.")
        
        shape = list(sample_data.shape)
        dtype_str = str(sample_data.dtype).split('.')[-1]
        element_size = sample_data.element_size()

        # Calculate total size for shared memory
        total_size = int(torch.prod(torch.tensor(shape))) * element_size * history_len
        
        if untyped_storage_inst is None:
            return TorchCUDAPoolMetadata(
                name=name,
                shape=shape,
                dtype_str=dtype_str,
                backend_type='torch',
                device=f'cuda',
                history_len=history_len,
                element_size=element_size,
                total_size=total_size,
                shm_name=f"pool_{name}_{os.getpid()}",
                creator_pid=os.getpid()
            )

        assert isinstance(untyped_storage_inst, torch.UntypedStorage), \
            "untyped_storage_inst must be an instance of torch.UntypedStorage"
        # Get CUDA IPC information
        (ipc_dev, ipc_handle, ipc_size, ipc_off_bytes,
        ref_handle, ref_off, evt_handle, evt_sync) = untyped_storage_inst._share_cuda_()
        return TorchCUDAPoolMetadata(
            name=name,
            shape=shape,
            dtype_str=dtype_str,
            backend_type='torch',
            device=f'cuda:{ipc_dev}',
            history_len=history_len,
            element_size=element_size,
            total_size=total_size,
            shm_name=f"pool_{name}_{os.getpid()}",
            creator_pid=os.getpid(),
            
            # Tensor reconstruction fields
            tensor_size=tuple(sample_data.size()),
            tensor_stride=tuple(sample_data.stride()),
            tensor_offset=sample_data.storage_offset(),
            
            # CUDA IPC fields
            storage_device=ipc_dev,
            storage_handle=ipc_handle,
            storage_size_bytes=ipc_size,
            storage_offset_bytes=ipc_off_bytes,
            ref_counter_handle=ref_handle,
            ref_counter_offset=ref_off,
            event_handle=evt_handle,
            event_sync_required=evt_sync
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
                history_len=history_len
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")