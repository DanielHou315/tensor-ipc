import sys
sys.path.append("src/")
import numpy as np
import time
import torch
import json
import argparse
from cyclonedds.domain import DomainParticipant
from tensor_ipc.core.consumer import TensorConsumer

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

def extract_timestamp_from_data(data, backend):
    """Extract embedded timestamp from tensor data."""
    try:
        if backend == "numpy":
            if data.size >= 2:
                ts_high = int(data.flat[0])
                ts_low = int(data.flat[1]) if data.size > 1 else 0
                return ts_high + (ts_low / 1e9)
        elif backend in ["torch", "torch_cuda"]:
            if data.numel() >= 2:
                flat_data = data.view(-1)
                ts_high = int(flat_data[0].item())
                ts_low = int(flat_data[1].item()) if data.numel() > 1 else 0
                return ts_high + (ts_low / 1e9)
    except:
        pass
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--results", required=True, help="Results file path")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    pool_name = config["pool_name"]
    shape = tuple(config["shape"])
    dtype_info = config["dtype_info"]
    backend = config["backend"]
    history_len = config["history_len"]
    read_history = config["read_history"]
    num_frames = config["num_frames"]
    
    # Create sample data
    sample_data = create_sample_data(shape, dtype_info, backend)
    
    # Create DDS participant
    participant = DomainParticipant()
    
    # Create consumer
    consumer = TensorConsumer.from_sample(
        pool_name=pool_name,
        sample=sample_data,
        dds_participant=participant,
        history_len=history_len,
        keep_last=10
    )
    
    print("Consumer ready")
    
    # Wait for producer to start
    time.sleep(1)
    
    latencies = []
    successful_reads = 0
    
    # Read frames and measure latency
    for i in range(num_frames + 5):  # Read a few extra to catch all frames
        read_time = time.time()
        data = consumer.get(
            history_len=read_history,
            as_numpy=False,
            latest_first=True
        )
        
        if data is not None:
            successful_reads += 1
            
            if read_history == 1:
                # Single frame
                write_timestamp = extract_timestamp_from_data(data, backend)
                if write_timestamp:
                    latency = (read_time - write_timestamp) * 1000
                    latencies.append(latency)
                    print(f"Read frame, latency: {latency:.2f}ms")
            else:
                # Multiple frames - measure latest frame latency
                latest_data = data[0] if hasattr(data, 'shape') and len(data.shape) > len(shape) else data
                write_timestamp = extract_timestamp_from_data(latest_data, backend)
                if write_timestamp:
                    latency = (read_time - write_timestamp) * 1000
                    latencies.append(latency)
                    print(f"Read {read_history} frames, latest latency: {latency:.2f}ms")

        time.sleep(0.05)  # Read every 50ms

    # Save results
    results = {
        "successful_reads": successful_reads,
        "latencies": latencies
    }
    
    with open(args.results, 'w') as f:
        json.dump(results, f)
    
    consumer.cleanup()
    print(f"Consumer finished. Successful reads: {successful_reads}, Latencies: {len(latencies)}")