"""
Test multi-process producer/consumer with different frequencies and time consistency validation.
"""
import numpy as np
import time
import sys
import subprocess
import os
from pathlib import Path
from typing import Optional

import torch
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from poly_ipc import TensorPublisher

def _producer_worker(
    backend_type, 
    pool_name: str, 
    shape: tuple, 
    dtype: str, 
    pub_rate: float, 
    duration: float, 
    delay: float, 
    shared_stats_path: str
):
    import json
    import time
    shared_stats = {}
    timestamp_log = {}  # counter -> publish_timestamp
    loop_timings = []  # Track loop performance
    try:
        if delay > 0:
            time.sleep(delay)
        if backend_type == "torch":
            sample = torch.zeros(shape, dtype=getattr(torch, dtype))
        elif backend_type == "torch_cuda":
            sample = torch.zeros(shape, dtype=getattr(torch, dtype), device='cuda')
        elif backend_type == "numpy":
            sample = np.zeros(shape, dtype=dtype)
        publisher = TensorPublisher.from_sample(pool_name, sample, history_len=10)
        #  ⬇️  producer/_producer_worker  – add right after `publisher = …`
        shared_stats["shape"]  = list(shape)  # Ensure JSON serializable
        shared_stats["dtype"]  = str(sample.dtype)
        shared_stats["device"] = sample.device.type if hasattr(sample, "device") else "cpu"

        shared_stats['producer_started'] = True
        print(f"Producer {pool_name} started, publishing at {pub_rate} Hz for {duration}s")
        start_time = time.perf_counter()
        counter = 0
        target_interval = 1.0 / pub_rate
        
        while time.perf_counter() - start_time < duration:
            loop_start = time.perf_counter()
            
            publish_time = time.perf_counter()
            
            # Create data tensor of the correct type based on backend
            if backend_type == "torch":
                data = torch.zeros(shape, dtype=getattr(torch, dtype))
            elif backend_type == "torch_cuda":
                data = torch.zeros(shape, dtype=getattr(torch, dtype), device='cuda')
            elif backend_type == "numpy":
                data = np.zeros(shape, dtype=dtype)
            
            # Store counter in tensor - handle different tensor types
            if dtype == "uint8":
                # For uint8, store counter in first two elements
                if backend_type in ["torch", "torch_cuda"]:
                    # PyTorch tensor - use view(-1) instead of .flat
                    flat_view = data.view(-1)
                    flat_view[0] = counter % 256
                    if len(flat_view) > 1:
                        flat_view[1] = (counter // 256) % 256  # High byte of counter
                else:
                    # NumPy array - use .flat
                    data.flat[0] = counter % 256
                    if len(data.flat) > 1:
                        data.flat[1] = (counter // 256) % 256  # High byte of counter
            else:
                # For other types, store counter in first element, and another copy in second
                if backend_type in ["torch", "torch_cuda"]:
                    # PyTorch tensor - use view(-1) instead of .flat
                    flat_view = data.view(-1)
                    flat_view[0] = counter
                    if len(flat_view) > 1:
                        flat_view[1] = counter
                else:
                    # NumPy array - use .flat
                    data.flat[0] = counter
                    if len(data.flat) > 1:
                        data.flat[1] = counter
            
            # Time data preparation
            data_prep_time = time.perf_counter()
                    
            # Log publish timestamp
            timestamp_log[counter] = publish_time
            
            publisher.publish(data)
            publish_complete_time = time.perf_counter()
            
            # if counter % 20 == 0:  # Log every 20th publish
            #     print(f"Producer {pool_name} published counter {counter}")
            counter += 1
            
            # Calculate timing metrics
            elapsed = time.perf_counter() - loop_start
            loop_duration = publish_complete_time - loop_start
            data_prep_duration = data_prep_time - loop_start
            publish_duration = publish_complete_time - data_prep_time
            
            # Store timing info
            loop_timings.append({
                'counter': counter - 1,
                'loop_duration': loop_duration,
                'data_prep_duration': data_prep_duration,
                'publish_duration': publish_duration,
                'target_interval': target_interval,
                'actual_interval': elapsed
            })
            
            sleep_time = max(0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        print(f"Producer {pool_name} finished, published {counter} samples")
        
        # Calculate performance statistics
        if loop_timings:
            avg_loop = np.mean([t['loop_duration'] for t in loop_timings])
            max_loop = np.max([t['loop_duration'] for t in loop_timings])
            avg_prep = np.mean([t['data_prep_duration'] for t in loop_timings])
            avg_pub = np.mean([t['publish_duration'] for t in loop_timings])
            
            shared_stats['performance'] = {
                'avg_loop_duration_ms': avg_loop * 1000,
                'max_loop_duration_ms': max_loop * 1000,
                'avg_data_prep_ms': avg_prep * 1000,
                'avg_publish_ms': avg_pub * 1000,
                'target_interval_ms': target_interval * 1000,
                'actual_rate_hz': counter / duration
            }
            print(f"Producer {pool_name} performance: avg_loop={avg_loop*1000:.2f}ms, max_loop={max_loop*1000:.2f}ms")
        
        shared_stats['producer_counter'] = counter
        shared_stats['producer_finished'] = True
        shared_stats['timestamp_log'] = timestamp_log
        shared_stats['loop_timings'] = loop_timings[-50:]  # Keep last 50 for debugging
        
        try:
            publisher.cleanup()  # Explicit cleanup (also happens automatically)
        except Exception as e:
            print(f"Producer {pool_name} cleanup warning: {e}")
    except Exception as e:
        print(f"Producer {pool_name} error: {e}")
        shared_stats['producer_error'] = str(e)
        # Cleanup will happen automatically in __del__
    with open(shared_stats_path, "w") as f:
        json.dump(shared_stats, f)
    exit(0)

class ProducerProcess:
    def __init__(self, 
        backend_type, 
        pool_name: str, 
        shape: tuple, 
        dtype: str, 
        pub_rate: float, 
        duration: float, 
        delay_start: float = 0.0
    ):
        self.backend_type = backend_type
        self.pool_name = pool_name
        self.shape = shape
        self.dtype = dtype
        self.pub_rate = pub_rate
        self.duration = duration
        self.delay = delay_start
        self.proc: Optional[subprocess.Popen] = None
        self.stats_file = f"/tmp/producer_stats_{os.getpid()}_{pool_name}.json"

    def start(self):
        args = [
            sys.executable, __file__, "run",
            self.backend_type, 
            self.pool_name, 
            str(self.shape), 
            self.dtype, 
            str(self.pub_rate), 
            str(self.duration), 
            str(self.delay), 
            self.stats_file
        ]
        self.proc = subprocess.Popen(args)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()

    def get_stats(self):
        import json
        if os.path.exists(self.stats_file):
            with open(self.stats_file, "r") as f:
                return json.load(f)
        return {}

if __name__ == "__main__":
    # Entrypoint for subprocess
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        backend_type = sys.argv[2]
        pool_name = sys.argv[3]
        shape = eval(sys.argv[4])
        dtype = sys.argv[5]
        pub_rate = float(sys.argv[6])
        duration = float(sys.argv[7])
        delay = float(sys.argv[8])
        stats_path = sys.argv[9]
        _producer_worker(backend_type, pool_name, shape, dtype, pub_rate, duration, delay, stats_path)