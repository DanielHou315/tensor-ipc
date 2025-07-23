"""
Multi-process registry for tensor pools using multiprocessing.managers.
Provides a centralized registry that can be shared across processes.
"""
from dataclasses import dataclass
import os
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence
import json

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

    # Extra payload json
    payload_json: str = ""

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
            self.shm_name == other.shm_name
            # self.creator_pid == other.creator_pid     # Skip PID comparison for equality
        )


"""
Acts as a namespace for metadata creation utilities.
"""
class MetadataCreator:
    @staticmethod
    def payload_from_torch_cuda_storage(tensor) -> str:
        import torch
        assert isinstance(tensor, torch.Tensor), \
            "tensor must be an instance of torch.Tensor"
        untyped_storage_inst = tensor.untyped_storage()
        # Get CUDA IPC information
        (ipc_dev, ipc_handle, ipc_size, ipc_off_bytes,
        ref_handle, ref_off, evt_handle, evt_sync) = untyped_storage_inst._share_cuda_()

        payload_json = json.dumps({
            'tensor_size': tensor.size(),
            'tensor_stride': tensor.stride(),
            'tensor_offset': tensor.storage_offset(),
            "storage_device": ipc_dev,
            "storage_handle": ipc_handle.hex(),
            "storage_size_bytes": ipc_size,
            "storage_offset_bytes": ipc_off_bytes,
            "ref_counter_handle": ref_handle.hex(),
            "ref_counter_offset": ref_off,
            "event_handle": evt_handle.hex(),
            "event_sync_required": evt_sync
        })
        return payload_json

    @staticmethod
    def verify_torch_cuda_payload(metadata: PoolMetadata) -> bool:
        """Verify that the payload JSON in metadata contains all required keys."""
        if not metadata.payload_json:
            return False
        try:
            payload = json.loads(metadata.payload_json)
            assert_payload_keys = [
                'tensor_size', 
                'tensor_stride', 
                'tensor_offset', 
                'storage_device', 
                'storage_handle', 
                'storage_size_bytes', 
                'storage_offset_bytes', 
                'ref_counter_handle', 
                'ref_counter_offset', 
                'event_handle', 
                'event_sync_required'
            ]
            for key in assert_payload_keys:
                assert key in payload, f"Missing key in payload: {key}"
            
            # Check each type   
            assert len(payload['tensor_size']) > 0, "tensor_size must be a non-empty list"
            assert len(payload['tensor_stride']) > 0, "tensor_stride must be a non-empty list"
            assert isinstance(payload['tensor_offset'], int), "tensor_offset must be an integer"
            assert isinstance(payload['storage_device'], int), "storage_device must be an integer"
            assert isinstance(payload['storage_handle'], str), "storage_handle must be a string"
            bytes.fromhex(payload['storage_handle'])  # Verify it's valid hex
            assert isinstance(payload['storage_size_bytes'], int), "storage_size_bytes must be an integer"
            assert isinstance(payload['storage_offset_bytes'], int), "storage_offset_bytes must be an integer"
            assert isinstance(payload['ref_counter_handle'], str), "ref_counter_handle must be a string"
            bytes.fromhex(payload['ref_counter_handle'])  # Verify it's valid hex
            assert isinstance(payload['ref_counter_offset'], int), "ref_counter_offset must be an integer"
            assert isinstance(payload['event_handle'], str), "event_handle must be a string"
            bytes.fromhex(payload['event_handle'])  # Verify it's valid hex
            assert isinstance(payload['event_sync_required'], bool), "event_sync_required must be a boolean"
            return True
        except json.JSONDecodeError:
            print("Invalid JSON in payload:", metadata.payload_json)
            return False
        except (AssertionError, TypeError) as e:
            print("Invalid payload element:", e)
            return False
        except Exception as e:
            print("Unexpected error during payload verification:", e)
        return False

    """
    From sample functions
    """
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
        tensor_pool = None,
    ) -> 'PoolMetadata':
        """Create PoolMetadata from a sample CUDA tensor."""
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
        
        if tensor_pool is None:
            return PoolMetadata(
                name=name,
                shape=shape,
                dtype_str=dtype_str,
                backend_type='torch_cuda',
                device='cuda',
                history_len=history_len,
                element_size=element_size,
                total_size=total_size,
                shm_name=f"pool_{name}",
                creator_pid=os.getpid()
            )

        payload_json = cls.payload_from_torch_cuda_storage(tensor_pool)
        ipc_dev = json.loads(payload_json).get("ipc_dev", 0)

        return PoolMetadata(
            name=name,
            shape=shape,
            dtype_str=dtype_str,
            backend_type='torch',
            device=f'cuda:{ipc_dev}',
            history_len=history_len,
            element_size=element_size,
            total_size=total_size,
            shm_name=f"pool_{name}",
            creator_pid=os.getpid(),
            payload_json=payload_json,
        )

    @classmethod
    def from_sample(cls, name, data, history_len, backend):
        if backend == "numpy":
            return cls.from_numpy_sample(
                name=name,
                sample_data=data,
                history_len=history_len
            )
        elif backend == "torch":
            return cls.from_torch_sample(
                name=name,
                sample_data=data,
                history_len=history_len
            )
        elif backend == "torch_cuda":
            return cls.from_torch_cuda_sample(
                name=name,
                sample_data=data,
                history_len=history_len
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")