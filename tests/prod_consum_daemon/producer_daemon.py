import sys
sys.path.append("src/")
import numpy as np
import time
import torch
import json
import argparse
from cyclonedds.domain import DomainParticipant
from tensor_ipc.core.producer import TensorProducer

def create_sample_data(shape, dtype_info, backend):
    dtype_name = dtype_info
    if backend == "numpy":
        return np.ones(shape, dtype=eval(f"np.{dtype_name}"))
    elif backend == "torch":
        torch_dtype = getattr(torch, dtype_name)
        return torch.ones(shape, dtype=torch_dtype)
    elif backend == "torch_cuda":
        torch_dtype = getattr(torch, dtype_name)
        return torch.ones(shape, dtype=torch_dtype, device='cuda')
    else:
        raise ValueError(f"Unknown backend: {backend}")

def create_frame_data_with_timestamp(shape, dtype_info, backend, frame_value, timestamp):
    dtype_name = dtype_info
    np_dtype = np.dtype(dtype_name)
    if dtype_name == "uint8":
        frame_value = frame_value % 256
    
    if backend == "numpy":
        data = np.full(shape, frame_value, dtype=np.dtype(dtype_name))
        # Embed timestamp in first few elements if possible
        if data.size >= 2:
            # Store timestamp as two 32-bit values for precision
            ts_high = int(timestamp) & 0xFFFFFFFF
            ts_low = int((timestamp % 1) * 1e9) & 0xFFFFFFFF
            data.flat[0] = ts_high % (256 if np_dtype == np.uint8 else 2**31)
            if data.size > 1:
                data.flat[1] = ts_low % (256 if np_dtype == np.uint8 else 2**31)
        return data
    elif backend in ["torch", "torch_cuda"]:
        torch_dtype = getattr(torch, dtype_name)
        device = 'cuda' if backend == "torch_cuda" else 'cpu'
        data = torch.full(shape, frame_value, dtype=torch_dtype, device=device)
        if data.numel() >= 2:
            ts_high = int(timestamp) & 0xFFFFFFFF
            ts_low = int((timestamp % 1) * 1e9) & 0xFFFFFFFF
            data.view(-1)[0] = ts_high % (256 if torch_dtype == torch.uint8 else 2**31)
            if data.numel() > 1:
                data.view(-1)[1] = ts_low % (256 if torch_dtype == torch.uint8 else 2**31)
        return data
    else:
        raise ValueError(f"Unknown backend: {backend}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config file path")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    pool_name = config["pool_name"]
    shape = tuple(config["shape"])
    dtype_info = config["dtype_info"]
    backend = config["backend"]
    history_len = config["history_len"]
    num_frames = config["num_frames"]
    
    # Create sample data
    sample_data = create_sample_data(shape, dtype_info, backend)
    
    # Create DDS participant
    participant = DomainParticipant()
    
    # Create producer
    producer = TensorProducer.from_sample(
        pool_name=pool_name,
        sample=sample_data,
        history_len=history_len,
        dds_participant=participant,
        keep_last=10
    )

    print(f"Producer ready for {num_frames} frames")

    # Write frames with timestamps
    for i in range(num_frames):
        timestamp = time.time()
        frame_value = i % 256
        frame_data = create_frame_data_with_timestamp(shape, dtype_info, backend, frame_value, timestamp)
        frame_idx = producer.put(frame_data)
        print(f"Written frame {i}, idx={frame_idx}, timestamp={timestamp}")
        time.sleep(0.05)  # 50ms between frames
    
    # Keep producer alive for a bit
    time.sleep(2)
    producer.cleanup()
    print("Producer finished")