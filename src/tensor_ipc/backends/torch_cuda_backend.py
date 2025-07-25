"""
PyTorch backends for Victor Python IPC.

Provides TorchBackend that layers on top of NumPy backend for shared memory,
with zero-copy tensor views and device conversion support.
"""
from __future__ import annotations
import json

from .base_backend import HistoryPadStrategy
from ..core.metadata import MetadataCreator, PoolMetadata

# Import torch at module level
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from .torch_backend import TorchProducerBackend, TorchConsumerBackend
from .torch_backend import TORCH_TYPE_MAP

class TorchCUDAProducerBackend(TorchProducerBackend):
    """
    Zero-copy CUDA backend: one producer, many readers, all on the
    same GPU (or peer-to-peer enabled GPUs).  Uses the exact IPC path
    that torch.multiprocessing uses internally, exposed via
      storage()._share_cuda_()  and
    torch.cuda.CUDAStorage._new_shared_cuda().
    """

    def __init__(self,
        pool_metadata: PoolMetadata,
        history_pad_strategy: HistoryPadStrategy = "zero",
        force=False,        # Not used but kept for compatibility
    ):
        assert isinstance(pool_metadata, PoolMetadata), \
            "pool_metadata must be PoolMetadata for CUDA backend"
        super().__init__(pool_metadata, history_pad_strategy)

    # ---------- pool life-cycle -----------------------------------
    def _init_tensor_pool(self, force=False) -> None:
        """Producer: allocate GPU ring-buffer and write IPC handle to metadata."""
        self._element_shape = tuple(self._pool_metadata.shape)
        self._pool_shape = (self._pool_metadata.history_len,) + self._element_shape

        assert self._pool_metadata.dtype_str in TORCH_TYPE_MAP, \
            f"Unsupported dtype: {self._pool_metadata.dtype_str}"
        assert isinstance(self._pool_metadata, PoolMetadata), \
            "pool_metadata must be PoolMetadata for CUDA backend"
        
        self._tensor_pool = torch.zeros(
            self._pool_shape,
            dtype=TORCH_TYPE_MAP[self._pool_metadata.dtype_str],
            device=self._pool_metadata.device
        )

        # Ask PyTorch for an IPC handle to its underlying storage
        self.ut_storage = self._tensor_pool.untyped_storage()

        # Extract IPC handle
        metadata_payload_json = MetadataCreator.payload_from_torch_cuda_storage(
            self._tensor_pool
        )
        self._pool_metadata.payload_json = metadata_payload_json

    # ---------- cleanup -------------------------------------------
    def cleanup(self):
        self._lock.close()
        self._lock.unlink()
        self._tensor_pool = None   # let CUDAStorage ref-count free itself


class TorchCUDAConsumerBackend(TorchConsumerBackend):
    """
    Consumer backend for CUDA tensors. Inherits from TorchCConsumerBackend
    to reuse the shared memory and IPC logic.
    """
    def __init__(self, pool_metadata: PoolMetadata):

        assert isinstance(pool_metadata, PoolMetadata), \
            "pool_metadata must be PoolMetadata for CUDA backend"
        super().__init__(
            pool_metadata=pool_metadata,
        )
        
    def connect(self, pool_metadata) -> bool:
        """Consumer: read IPC handle from metadata and attach."""
        if self._connected:
            return True
        
        if not isinstance(pool_metadata, PoolMetadata) or \
            not MetadataCreator.verify_torch_cuda_payload(pool_metadata):
            print("Invalid pool metadata for CUDA backend:", pool_metadata)
            return False
        # Cache used values
        self._target_device = torch.device(self._pool_metadata.device)

        # Get metadata from payload, at this point it should be verified
        metadata_dict = json.loads(pool_metadata.payload_json)

        # Fix: Use correct storage class for PyTorch
        # torch.UntypedStorage is correct for PyTorch >= 1.10
        storage_cls = getattr(torch, "UntypedStorage", None)
        if storage_cls is None:
            # Fallback for older PyTorch versions
            storage_cls = getattr(torch.storage, "UntypedStorage", None)
        if storage_cls is None:
            raise RuntimeError("Could not find UntypedStorage class in torch")
        try:
            self._tensor_pool = rebuild_cuda_tensor(
                torch.Tensor,
                tensor_size      = metadata_dict['tensor_size'],
                tensor_stride    = metadata_dict['tensor_stride'],
                tensor_offset    = metadata_dict['tensor_offset'],
                storage_cls      = storage_cls,
                dtype            = TORCH_TYPE_MAP[pool_metadata.dtype_str],
                requires_grad    = False,
                storage_device   = metadata_dict['storage_device'],
                storage_handle   = bytes.fromhex(metadata_dict['storage_handle']),
                storage_size_bytes   = metadata_dict['storage_size_bytes'],
                storage_offset_bytes = metadata_dict['storage_offset_bytes'],
                ref_counter_handle   = bytes.fromhex(metadata_dict['ref_counter_handle']),
                ref_counter_offset   = metadata_dict['ref_counter_offset'],
                event_handle         = bytes.fromhex(metadata_dict['event_handle']),
                event_sync_required  = metadata_dict['event_sync_required'],
            )
            self._connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to CUDA tensor pool with error: {e}")
            return False

    # ---------- cleanup -------------------------------------------
    def cleanup(self):
        self._lock.close()
        self._lock.unlink()
        # Fix: Only delete _tensor_pool if it exists
        if hasattr(self, '_tensor_pool'):
            self._tensor_pool = None   # let CUDAStorage ref-count free itself