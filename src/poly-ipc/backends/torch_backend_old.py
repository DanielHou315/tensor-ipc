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

    # ---------- ctor -------------------------------------------------
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

        # normalise dtype --------------------------------------------------
        if isinstance(dtype, str):
            if dtype.startswith("torch."):
                dtype = getattr(torch, dtype.split(".")[-1])
            else:                       # numpy style string
                dtype = self._TORCH_FROM_NP[np.dtype(dtype).type]

        # Store target device for tensor conversion - shared memory always uses CPU
        self._target_device = torch.device(device)
        self._np_dtype = self._NP_FROM_TORCH[dtype]
        self._torch_dtype = dtype
        self._shm_name = f"{pool_name}_torch_pool"

        super().__init__(pool_name, shape, dtype, history_len,
                         is_producer, history_pad_strategy, device)

        self._shared_memory: Optional[cpusm] = None
        if self.is_producer:
            self._create_pool()
        else:
            try:
                self._connect_pool()
            except Exception:
                pass     # consumer retries later
    # -----------------------------------------------------------------

    # ---------- Shared-memory helpers --------------------------------
    def _bytes_required(self) -> int:
        return int(np.prod((self.history_len, *self._sample_shape))) * \
               np.dtype(self._np_dtype).itemsize

    def _wrap_shm_as_tensor(self, shm_buf) -> torch.Tensor:
        np_view = np.ndarray((self.history_len, *self._sample_shape),
                             dtype=self._np_dtype,
                             buffer=shm_buf)
        t = torch.from_numpy(np_view)   # zero-copy view
        return t

    # ---------- pool lifecycle ---------------------------------------
    def _create_pool(self) -> None:
        nbytes = self._bytes_required()
        try:
            self._shared_memory = cpusm(self._shm_name, create=True, size=nbytes)
        except FileExistsError:
            raise RuntimeError(f"Shared memory {self._shm_name} already exists")

        self._tensor_pool = self._wrap_shm_as_tensor(self._shared_memory.buf)
        self._initialize_history_padding()
        self._is_initialized = True
        self._pool_connected = True

    def _connect_pool(self) -> None:
        try:
            self._shared_memory = cpusm(self._shm_name, create=False)
        except FileNotFoundError:
            raise RuntimeError("Producer not started yet – SHM block missing")

        self._tensor_pool = self._wrap_shm_as_tensor(self._shared_memory.buf)
        self._pool_connected = True

    def _initialize_history_padding(self) -> None:
        """Initialize history padding based on the strategy."""
        if self._tensor_pool is None:
            return
            
        if self.history_pad_strategy == "zero":
            self._tensor_pool.fill_(0)
        elif self.history_pad_strategy == "fill":
            # Will be filled with first published value
            self._tensor_pool.fill_(0)

    # ---------- publishing / reading ---------------------------------
    def publish(self, data: Union['torch.Tensor', np.ndarray]) -> None:
        if not self.is_producer:
            raise RuntimeError("Only producers may publish")

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        if not isinstance(data, torch.Tensor):
            raise TypeError("publish() expects torch.Tensor or np.ndarray")

        if tuple(data.shape) != self._sample_shape:
            raise ValueError("Shape mismatch")
        if data.dtype != self._torch_dtype:
            raise TypeError("dtype mismatch")

        data_cpu = data.cpu()
        seq, idx = self._get_shared_state()

        # first-publish “fill” strategy
        if not self._is_initialized and self.history_pad_strategy == "fill":
            self._tensor_pool[:] = data_cpu
            self._is_initialized = True
        else:
            self._tensor_pool[idx].copy_(data_cpu)

        new_idx = (idx + 1) % self.history_len
        self._update_shared_state(seq + 1, new_idx)
        self.notify_consumers()

    # identical get / _get_single / _get_history logic to NumPy backend
    # but without .copy() so readers see live buffer -------------------
    def _get_single(self, history_index: int):
        if self._tensor_pool is None:
            return None
        seq, cur_idx = self._get_shared_state()
        self._last_seen_sequence = seq
        real_idx = (cur_idx - 1 - history_index) % self.history_len
        cpu_tensor = self._tensor_pool[real_idx]
        
        # Convert back to target device if not CPU
        if self._target_device.type != 'cpu':
            return cpu_tensor.to(self._target_device)
        return cpu_tensor

    def _get_history(self, count: int):
        if self._tensor_pool is None:
            return None
        count = min(count, self.history_len)
        seq, cur_idx = self._get_shared_state()
        self._last_seen_sequence = seq
        idxs = [(cur_idx - 1 - i) % self.history_len for i in range(count)]
        cpu_tensors = self._tensor_pool[idxs]
        
        # Convert back to target device if not CPU
        if self._target_device.type != 'cpu':
            return cpu_tensors.to(self._target_device)
        return cpu_tensors

    def _to_numpy(self, data):
        # Always convert to CPU first for numpy conversion
        return data.cpu().numpy()

    # ---------- dtype helpers ----------------------------------------
    @staticmethod
    def _numpy_to_torch_dtype(np_dtype: np.dtype):
        return TorchBackend._TORCH_FROM_NP[np_dtype.type]

    # ---------- cleanup ----------------------------------------------
    def cleanup(self):
        super().cleanup()
        if self._shared_memory is not None:
            self._shared_memory.close()
            if self.is_producer:
                try:
                    self._shared_memory.unlink()
                except FileNotFoundError:
                    pass
            self._shared_memory = None

# # ------------------------------------------------------------------
# #  TorchCUDABackend   (drop into the same module as NumpyBackend)
# # ------------------------------------------------------------------
# NOT Working due to PyTorch internal CUDA errors
# class TorchCUDABackend(TensorBackend):
#     """
#     Zero-copy CUDA backend: one producer, many readers, all on the
#     same GPU (or peer-to-peer enabled GPUs).  Uses the exact IPC path
#     that torch.multiprocessing uses internally, exposed via
#       storage()._share_cuda_()  and
#     torch.cuda.CUDAStorage._new_shared_cuda().
#     """

#     IPC_STRUCT_FMT = "<iQq"                      # device, handle(8), size
#     IPC_STRUCT_SIZE = struct.calcsize(IPC_STRUCT_FMT)

#     def __init__(self,
#                  pool_name: str,
#                  shape: tuple,
#                  dtype: Any,
#                  history_len: int = 1,
#                  is_producer: bool = False,
#                  history_pad_strategy: HistoryPadStrategy = "zero",
#                  device: str = "cuda:0"):

#         if not TORCH_AVAILABLE:
#             raise RuntimeError("PyTorch not available")

#         # normalise dtype -------------------------------------------
#         if isinstance(dtype, str):
#             if dtype.startswith("torch."):
#                 dtype = getattr(torch, dtype.split(".")[-1])
#             else:
#                 raise ValueError("Use torch.<dtype> strings for CUDA backend")

#         if not str(device).startswith("cuda"):
#             raise ValueError("TorchCUDABackend supports only CUDA tensors")

#         self._torch_dtype = dtype
#         self._device      = torch.device(device)
#         self._pool_shape  = (history_len, *shape)

#         super().__init__(pool_name, shape, dtype, history_len,
#                          is_producer, history_pad_strategy,
#                          device=str(self._device))

#         # producer creates pool now; consumer will retry connect ----
#         if self.is_producer:
#             self._create_pool()
#         else:
#             try:
#                 self._connect_pool()
#             except RuntimeError:
#                 pass

#     # ---------- IPC handle helpers --------------------------------
#     @staticmethod
#     def _encode_ipc(device: int, handle: bytes, size: int) -> bytes:
#         """Pack device, handle(8 bytes) and size into a fixed struct."""
#         handle_int = int.from_bytes(handle[:8], "little", signed=False)
#         return struct.pack(TorchCUDABackend.IPC_STRUCT_FMT,
#                            device, handle_int, size)

#     @staticmethod
#     def _decode_ipc(buf: memoryview):
#         device, handle_int, size = struct.unpack(
#             TorchCUDABackend.IPC_STRUCT_FMT, buf[:TorchCUDABackend.IPC_STRUCT_SIZE]
#         )
#         handle = handle_int.to_bytes(8, "little")
#         return device, handle, size

#     # ---------- pool life-cycle -----------------------------------
#     def _create_pool(self):
#         """Producer: allocate GPU ring-buffer and write IPC handle to metadata."""
#         self._tensor_pool = torch.empty(self._pool_shape,
#                                         dtype=self._torch_dtype,
#                                         device=self._device)

#         # Ask PyTorch for an IPC handle to its underlying storage
#         # In newer PyTorch versions, _share_cuda_() returns a tuple with 8 elements:
#         # (device_idx, handle_bytes, size_bytes, offset, pool_id, offset_bytes, view_info, requires_grad)
#         cuda_ipc_data = self._tensor_pool.storage()._share_cuda_()
        
#         # Extract the required elements for IPC
#         device_idx = cuda_ipc_data[0]  # Device index
#         handle_bytes = cuda_ipc_data[1]  # IPC handle
#         size_bytes = cuda_ipc_data[2]  # Size in bytes
        
#         ipc_blob = self._encode_ipc(device_idx, handle_bytes, size_bytes)

#         # write IPC blob right after JSON header in metadata shm ----
#         json_len = struct.unpack_from('I', self._metadata_shm.buf, 0)[0]
#         dst_off  = METADATA_HEADER_SIZE + json_len          # skip JSON
#         self._metadata_shm.buf[dst_off:
#                                dst_off + self.IPC_STRUCT_SIZE] = ipc_blob

#         self._initialize_history_padding()
#         self._is_initialized  = True
#         self._pool_connected  = True

#     def _connect_pool(self):
#         """Consumer: read IPC handle from metadata and attach."""
#         if not self._connect_shared_metadata():
#             raise RuntimeError("CUDA pool metadata not ready yet")

#         # read IPC blob --------------------------------------------
#         json_len = struct.unpack_from('I', self._metadata_shm.buf, 0)[0]
#         src_off  = METADATA_HEADER_SIZE + json_len
#         ipc_view = self._metadata_shm.buf[src_off:
#                                           src_off + self.IPC_STRUCT_SIZE]

#         device_idx, handle, size_bytes = self._decode_ipc(ipc_view)

#         # Use the new PyTorch CUDA IPC API for consumers
#         # Try to reconstruct tensor using torch.cuda.from_dlpack or alternative methods
#         try:
#             # Modern approach: use the torch.storage API
#             storage = torch.UntypedStorage.from_buffer(handle, size_bytes, device=f"cuda:{device_idx}")
#             # Create typed storage
#             typed_storage = torch.storage.TypedStorage(
#                 wrap_storage=storage,
#                 dtype=self._torch_dtype,
#                 _internal=True
#             )
#             # Create tensor from storage
#             self._tensor_pool = torch.tensor([], dtype=self._torch_dtype, device=self._device).set_(
#                 typed_storage, 0, self._pool_shape
#             )
#         except Exception:
#             # Fallback: try alternative reconstruction method
#             # This is a simplified approach that may work for basic cases
#             self._tensor_pool = torch.empty(self._pool_shape, dtype=self._torch_dtype, device=self._device)
#             # Note: This fallback doesn't provide true IPC, just creates a new tensor
#             # In a real implementation, we'd need to use the proper PyTorch IPC reconstruction

#         self._pool_connected = True

#     def _initialize_history_padding(self) -> None:
#         """Initialize history padding based on the strategy."""
#         if self._tensor_pool is None:
#             return
            
#         if self.history_pad_strategy == "zero":
#             self._tensor_pool.fill_(0)
#         elif self.history_pad_strategy == "fill":
#             # Will be filled with first published value
#             self._tensor_pool.fill_(0)

#     # ---------- publish / read (same as CPU torch backend) ---------
#     def publish(self, data: torch.Tensor):
#         if not self.is_producer:
#             raise RuntimeError("Only producer may publish")

#         if data.device != self._device:
#             data = data.to(self._device)

#         if tuple(data.shape) != self._sample_shape:
#             raise ValueError("Shape mismatch")

#         seq, idx = self._get_shared_state()

#         if not self._is_initialized and self.history_pad_strategy == "fill":
#             self._tensor_pool.copy_(data.expand_as(self._tensor_pool))
#             self._is_initialized = True
#         else:
#             self._tensor_pool[idx].copy_(data)

#         self._update_shared_state(seq + 1, (idx + 1) % self.history_len)
#         self.notify_consumers()

#     def _get_single(self, history_index: int):
#         if self._tensor_pool is None:
#             return None
#         seq, cur = self._get_shared_state()
#         self._last_seen_sequence = seq
#         idx = (cur - 1 - history_index) % self.history_len
#         return self._tensor_pool[idx]

#     def _get_history(self, count: int):
#         if self._tensor_pool is None:
#             return None
#         count = min(count, self.history_len)
#         seq, cur = self._get_shared_state()
#         self._last_seen_sequence = seq
#         idxs = [(cur - 1 - i) % self.history_len for i in range(count)]
#         return self._tensor_pool[idxs]

#     def _to_numpy(self, data):               # convenience
#         return data.cpu().numpy()

#     # ---------- cleanup -------------------------------------------
#     def cleanup(self):
#         super().cleanup()        # semaphores & metadata
#         self._tensor_pool = None   # let CUDAStorage ref-count free itself
