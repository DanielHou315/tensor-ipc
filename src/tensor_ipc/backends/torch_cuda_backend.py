"""
PyTorch backends for Victor Python IPC.

Provides TorchBackend that layers on top of NumPy backend for shared memory,
with zero-copy tensor views and device conversion support.
"""
from __future__ import annotations

from .base_backend import HistoryPadStrategy
from ..core.metadata import TorchCUDAPoolMetadata

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
        pool_metadata: TorchCUDAPoolMetadata,
        history_pad_strategy: HistoryPadStrategy = "zero",
        force=False,        # Not used but kept for compatibility
    ):
        assert isinstance(pool_metadata, TorchCUDAPoolMetadata), \
            "pool_metadata must be TorchCUDAPoolMetadata for CUDA backend"
        super().__init__(pool_metadata, history_pad_strategy)

    # ---------- pool life-cycle -----------------------------------
    def _init_tensor_pool(self, force=False) -> None:
        """Producer: allocate GPU ring-buffer and write IPC handle to metadata."""
        self._element_shape = tuple(self._pool_metadata.shape)
        self._pool_shape = (self._pool_metadata.history_len,) + self._element_shape
        

        assert self._pool_metadata.dtype_str in TORCH_TYPE_MAP, \
            f"Unsupported dtype: {self._pool_metadata.dtype_str}"
        assert isinstance(self._pool_metadata, TorchCUDAPoolMetadata), \
            "pool_metadata must be TorchCUDAPoolMetadata for CUDA backend"
        
        self._tensor_pool = torch.zeros(
            self._pool_shape,
            dtype=TORCH_TYPE_MAP[self._pool_metadata.dtype_str],
            device=self._pool_metadata.device
        )

        # Ask PyTorch for an IPC handle to its underlying storage
        self.ut_storage = self._tensor_pool.untyped_storage()._share_cuda_()

        # Extract IPC handle
        (ipc_dev, ipc_handle, ipc_size, ipc_off_bytes,
         ref_handle, ref_off, evt_handle, evt_sync) = self.ut_storage
        
        # Update metadata with IPC information
        self._pool_metadata.tensor_size = self._tensor_pool.size()
        self._pool_metadata.tensor_stride = self._tensor_pool.stride()
        self._pool_metadata.tensor_offset = self._tensor_pool.storage_offset()

        # CUDA IPC fields from _share_cuda_()
        self._pool_metadata.storage_device = ipc_dev
        self._pool_metadata.storage_handle = ipc_handle
        self._pool_metadata.storage_size_bytes = ipc_size
        self._pool_metadata.storage_offset_bytes = ipc_off_bytes
        self._pool_metadata.ref_counter_handle = ref_handle
        self._pool_metadata.ref_counter_offset = ref_off
        self._pool_metadata.event_handle = evt_handle
        self._pool_metadata.event_sync_required = evt_sync

    # ---------- cleanup -------------------------------------------
    def cleanup(self):
        del self._tensor_pool
        self._tensor_pool = None   # let CUDAStorage ref-count free itself


class TorchCUDAConsumerBackend(TorchConsumerBackend):
    """
    Consumer backend for CUDA tensors. Inherits from TorchCConsumerBackend
    to reuse the shared memory and IPC logic.
    """
    
    def __init__(self, pool_metadata: TorchCUDAPoolMetadata):

        assert isinstance(pool_metadata, TorchCUDAPoolMetadata), \
            "pool_metadata must be TorchCUDAPoolMetadata for CUDA backend"
        super().__init__(
            pool_metadata=pool_metadata,
        )
        
    def connect(self, pool_metadata) -> bool:
        """Consumer: read IPC handle from metadata and attach."""
        if self._connected:
            return True
        if not isinstance(pool_metadata, TorchCUDAPoolMetadata) or not pool_metadata.is_valid():
            print("Invalid pool metadata for CUDA backend")
            print(pool_metadata)
            return False
        # Cache used values
        self._target_device = torch.device(self._pool_metadata.device)
        self._tensor_pool = rebuild_cuda_tensor(
            torch.Tensor,
            tensor_size      = pool_metadata.tensor_size,
            tensor_stride    = pool_metadata.tensor_stride,
            tensor_offset    = pool_metadata.tensor_offset,
            storage_cls      = torch.UntypedStorage,
            dtype            = TORCH_TYPE_MAP[pool_metadata.dtype_str],
            requires_grad    = False,
            storage_device   = pool_metadata.storage_device,
            storage_handle   = pool_metadata.storage_handle,
            storage_size_bytes   = pool_metadata.storage_size_bytes,
            storage_offset_bytes = pool_metadata.storage_offset_bytes,
            ref_counter_handle   = pool_metadata.ref_counter_handle,
            ref_counter_offset   = pool_metadata.ref_counter_offset,
            event_handle         = pool_metadata.event_handle,
            event_sync_required  = pool_metadata.event_sync_required,
        )
        self._connected = True
        return True

    # ---------- cleanup -------------------------------------------
    def cleanup(self):
        self._tensor_pool = None   # let CUDAStorage ref-count free itself