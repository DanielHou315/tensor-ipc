# producer_consumer.py
import torch, torch.multiprocessing as mp
from torch.multiprocessing.reductions import rebuild_cuda_tensor

def consumer(meta):
    torch.cuda.set_device(meta["storage_device"])        # create a local context
    # a dummy op to make sure the context is initialised
    torch.empty(1, device=f'cuda:{meta["storage_device"]}')  

    t = rebuild_cuda_tensor(
        torch.Tensor,
        tensor_size      = meta["size"],
        tensor_stride    = meta["stride"],
        tensor_offset    = meta["offset"],
        storage_cls      = meta["storage_cls"],
        dtype            = meta["dtype"],
        requires_grad    = False,
        storage_device   = meta["storage_device"],
        storage_handle   = meta["storage_handle"],
        storage_size_bytes   = meta["storage_size_bytes"],
        storage_offset_bytes = meta["storage_offset_bytes"],
        ref_counter_handle   = meta["ref_counter_handle"],
        ref_counter_offset   = meta["ref_counter_offset"],
        event_handle         = meta["event_handle"],
        event_sync_required  = meta["event_sync_required"],
    )
    print("Consumer received:", t)        # ✅ works
    # do work…
    del t                                 # release asap

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    a = torch.arange(5, device="cuda")    # producer tensor
    storage = a.untyped_storage()

    meta = dict(
        size      = a.size(),
        stride    = a.stride(),
        offset    = a.storage_offset(),
        storage_cls = type(storage),
        dtype       = a.dtype,
    )
    (meta["storage_device"],
     meta["storage_handle"],
     meta["storage_size_bytes"],
     meta["storage_offset_bytes"],
     meta["ref_counter_handle"],
     meta["ref_counter_offset"],
     meta["event_handle"],
     meta["event_sync_required"]) = storage._share_cuda_()

    p = mp.Process(target=consumer, args=(meta,))
    p.start()
    p.join()                               # keep producer alive