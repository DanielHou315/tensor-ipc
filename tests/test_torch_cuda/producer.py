#!/usr/bin/env python3
import torch, textwrap, os

a = torch.arange(4, device="cuda")           # tensor to share
storage = a.untyped_storage()

# Grab the IPC tuple
(ipc_dev, ipc_handle, ipc_size, ipc_off_bytes,
 ref_handle, ref_off, evt_handle, evt_sync) = storage._share_cuda_()

META = dict(
    size=tuple(a.size()),
    stride=tuple(a.stride()),
    offset=a.storage_offset(),
    dtype=str(a.dtype).split('.')[-1],
    storage_device=ipc_dev,
    storage_handle=ipc_handle.hex(),
    storage_size_bytes=ipc_size,
    storage_offset_bytes=ipc_off_bytes,
    ref_counter_handle=ref_handle.hex(),
    ref_counter_offset=ref_off,
    event_handle=evt_handle.hex(),
    event_sync_required=evt_sync,
)

print("\n=== COPY THIS BLOCK INTO consumer.py ===")
print("META =", textwrap.indent(repr(META), "       "))
print("=== END BLOCK ===\n")
print("Producer PID:", os.getpid(), "tensor:", a)
input("Run consumer *now*, then press ENTER to exit → ")