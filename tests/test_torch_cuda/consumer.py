#!/usr/bin/env python3
import torch, os
from torch.multiprocessing.reductions import rebuild_cuda_tensor

# ── paste producer block below ─────────────────────────────────────
META =        {'size': (4,), 'stride': (1,), 'offset': 0, 'dtype': 'int64', 'storage_device': 0, 'storage_handle': '0163d0c7914948560000266f3d0000000000000020000000000000020000000000000001000000ff00002a000000000000009103d0c17c00005c0000000000000000', 'storage_size_bytes': 32, 'storage_offset_bytes': 0, 'ref_counter_handle': '2f746f7263685f343032363135305f323330393338353936365f30', 'ref_counter_offset': 0, 'event_handle': '266f3d00000000000100000000000000ff0f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', 'event_sync_required': True}
# ───────────────────────────────────────────────────────────────────

DTYPE = { k.split('.')[-1]: getattr(torch, k) for k in dir(torch)
          if k.startswith(('float','int','uint','bool')) }

def _b(hexstr):    # hex → bytes
    return bytes.fromhex(hexstr)

torch.cuda.set_device(META["storage_device"])
torch.empty(1, device=f"cuda:{META['storage_device']}")   # init context

t = rebuild_cuda_tensor(
    torch.Tensor,
    tensor_size      = META["size"],
    tensor_stride    = META["stride"],
    tensor_offset    = META["offset"],
    storage_cls      = torch.UntypedStorage,
    dtype            = DTYPE[META["dtype"]],
    requires_grad    = False,
    storage_device   = META["storage_device"],
    storage_handle   = _b(META["storage_handle"]),
    storage_size_bytes   = META["storage_size_bytes"],
    storage_offset_bytes = META["storage_offset_bytes"],
    ref_counter_handle   = _b(META["ref_counter_handle"]),
    ref_counter_offset   = META["ref_counter_offset"],
    event_handle         = _b(META["event_handle"]),
    event_sync_required  = META["event_sync_required"],
)

print(f"[consumer {os.getpid()}] got tensor:", t)
t[0] += 1000
torch.cuda.synchronize()
print("[consumer] modified first element →", t)