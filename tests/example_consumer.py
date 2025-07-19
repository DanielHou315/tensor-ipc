"""
Test multi-process producer/consumer with different frequencies and time consistency validation.
"""
import numpy as np
import time
import sys
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from poly_ipc import TensorConsumer

def _consumer_worker(backend_type, pool_name: str, shape: tuple, dtype: str, poll_rate: float, duration: float, shared_stats_path: str):
    import json
    import time
    shared_stats = {}
    received_data = []
    timestamp_log = {}  # counter -> receive_timestamp
    loop_timings = []  # Track loop performance
    try:
        if backend_type == "torch":
            sample = torch.zeros(shape, dtype=getattr(torch, dtype))
        elif backend_type == "torch_cuda":
            sample = torch.zeros(shape, dtype=getattr(torch, dtype), device='cuda')
        elif backend_type == "numpy":
            sample = np.zeros(shape, dtype=dtype)
        consumer = TensorConsumer.from_sample(pool_name, sample, history_len=10)
        #  ⬇️  consumer/_consumer_worker  – add right after `consumer = …`
        shared_stats["shape"]  = shape
        shared_stats["dtype"]  = str(sample.dtype)
        shared_stats["device"] = sample.device.type if hasattr(sample, "device") else "cpu"

        start_time = time.perf_counter()
        counter = 0
        last_data_counter = -1
        shared_stats['consumer_started'] = True
        
        # Give producer some time to start
        time.sleep(0.5)

        while time.perf_counter() - start_time < duration:
            loop_start = time.perf_counter()
            
            try:
                # Use blocking without timeout first, then add timeout if needed
                get_start = time.perf_counter()
                data = consumer.get(history_len=1, block=True, timeout=1.0)
                get_complete = time.perf_counter()
                
                if data is not None:
                    receive_time = time.perf_counter()
                    process_start = time.perf_counter()

                    # Time data extraction
                    extract_start = time.perf_counter()
                    if dtype == "uint8":
                        # For uint8, extract counter from first two bytes - handle different tensor types
                        if backend_type in ["torch", "torch_cuda"]:
                            # PyTorch tensor - use view(-1) instead of .flat
                            flat_view = data.view(-1)
                            counter_low = int(flat_view[0])
                            counter_high = int(flat_view[1]) if len(flat_view) > 1 else 0
                        else:
                            # NumPy array - use .flat
                            counter_low = int(data.flat[0])
                            counter_high = int(data.flat[1]) if len(data.flat) > 1 else 0
                        data_counter = counter_low + (counter_high * 256)
                    else:
                        # For other types, counter is in first element - handle different tensor types
                        if backend_type in ["torch", "torch_cuda"]:
                            # PyTorch tensor - use view(-1) instead of .flat
                            flat_view = data.view(-1)
                            data_counter = int(flat_view[0])
                        else:
                            # NumPy array - use .flat
                            data_counter = int(data.flat[0])
                    extract_complete = time.perf_counter()
                    
                    # Time timestamp logging
                    timestamp_start = time.perf_counter()
                    timestamp_log[data_counter] = receive_time
                    timestamp_complete = time.perf_counter()
                    
                    # Time data appending
                    append_start = time.perf_counter()
                    if data_counter != last_data_counter:
                        received_data.append({
                            'receive_time': receive_time,
                            'data_counter': data_counter,
                            'consumer_counter': counter
                        })
                        last_data_counter = data_counter
                    append_complete = time.perf_counter()
                    
                    process_complete = time.perf_counter()
                    counter += 1
                    
                    # Store detailed timing info
                    loop_duration = process_complete - loop_start
                    get_duration = get_complete - get_start
                    process_duration = process_complete - process_start
                    extract_duration = extract_complete - extract_start
                    timestamp_duration = timestamp_complete - timestamp_start
                    append_duration = append_complete - append_start
                    
                    loop_timings.append({
                        'counter': counter - 1,
                        'data_counter': data_counter,
                        'loop_duration': loop_duration,
                        'get_duration': get_duration,
                        'process_duration': process_duration,
                        'extract_duration': extract_duration,
                        'timestamp_duration': timestamp_duration,
                        'append_duration': append_duration,
                        'had_data': True
                    })
                    
                    # Log detailed timing for slow operations
                    # if loop_duration > 0.020:  # Log if loop takes >20ms
                    #     print(f"Consumer {pool_name} slow loop {counter-1}: "
                    #           f"total={loop_duration*1000:.2f}ms, "
                    #           f"get={get_duration*1000:.2f}ms, "
                    #           f"extract={extract_duration*1000:.2f}ms, "
                    #           f"timestamp={timestamp_duration*1000:.2f}ms, "
                    #           f"append={append_duration*1000:.2f}ms")
                else:
                    # Track timing even when no data
                    loop_duration = get_complete - loop_start
                    get_duration = get_complete - get_start
                    
                    loop_timings.append({
                        'counter': -1,
                        'data_counter': -1,
                        'loop_duration': loop_duration,
                        'get_duration': get_duration,
                        'process_duration': 0,
                        'extract_duration': 0,
                        'timestamp_duration': 0,
                        'append_duration': 0,
                        'had_data': False
                    })
                    
                    # Log when we get no data after waiting
                    if get_duration > 0.500:  # Log if we waited >500ms for nothing
                        print(f"Consumer {pool_name} waited {get_duration*1000:.2f}ms but got no data")

            except Exception as e:
                print(f"Consumer {pool_name} exception in loop: {e}")
                break
        
        # Calculate performance statistics with detailed breakdown
        if loop_timings:
            valid_timings = [t for t in loop_timings if t['had_data']]
            if valid_timings:
                avg_loop = np.mean([t['loop_duration'] for t in valid_timings])
                max_loop = np.max([t['loop_duration'] for t in valid_timings])
                avg_get = np.mean([t['get_duration'] for t in valid_timings])
                avg_process = np.mean([t['process_duration'] for t in valid_timings])
                avg_extract = np.mean([t['extract_duration'] for t in valid_timings])
                avg_timestamp = np.mean([t['timestamp_duration'] for t in valid_timings])
                avg_append = np.mean([t['append_duration'] for t in valid_timings])
                
                # Find slowest operations
                slowest_loop = max(valid_timings, key=lambda x: x['loop_duration'])
                slowest_get = max(valid_timings, key=lambda x: x['get_duration'])
                
                shared_stats['performance'] = {
                    'avg_loop_duration_ms': avg_loop * 1000,
                    'max_loop_duration_ms': max_loop * 1000,
                    'avg_get_ms': avg_get * 1000,
                    'avg_process_ms': avg_process * 1000,
                    'avg_extract_ms': avg_extract * 1000,
                    'avg_timestamp_ms': avg_timestamp * 1000,
                    'avg_append_ms': avg_append * 1000,
                    'valid_loops': len(valid_timings),
                    'total_loops': len(loop_timings),
                    'slowest_loop_counter': slowest_loop['counter'],
                    'slowest_loop_duration_ms': slowest_loop['loop_duration'] * 1000,
                    'slowest_get_counter': slowest_get['counter'],
                    'slowest_get_duration_ms': slowest_get['get_duration'] * 1000
                }
                
                print(f"Consumer {pool_name} performance breakdown:")
                print(f"  avg_loop={avg_loop*1000:.2f}ms, max_loop={max_loop*1000:.2f}ms")
                print(f"  avg_get={avg_get*1000:.2f}ms, avg_extract={avg_extract*1000:.3f}ms")
                print(f"  avg_timestamp={avg_timestamp*1000:.3f}ms, avg_append={avg_append*1000:.3f}ms")
                print(f"  slowest loop: counter {slowest_loop['counter']} ({slowest_loop['loop_duration']*1000:.2f}ms)")
                print(f"  slowest get: counter {slowest_get['counter']} ({slowest_get['get_duration']*1000:.2f}ms)")
                
        shared_stats['consumer_counter'] = counter
        shared_stats['consumer_finished'] = True
        shared_stats['received_data'] = received_data
        shared_stats['timestamp_log'] = timestamp_log
        shared_stats['loop_timings'] = loop_timings[-50:]  # Keep last 50 for debugging
        
        try:
            consumer.cleanup()  # Explicit cleanup (also happens automatically)
        except Exception as e:
            print(f"Consumer {pool_name} cleanup warning: {e}")
    except Exception as e:
        print(f"Consumer {pool_name} fatal error: {e}")
        shared_stats['consumer_error'] = str(e)
        # Cleanup will happen automatically in __del__
    with open(shared_stats_path, "w") as f:
        json.dump(shared_stats, f)
    exit(0)

class ConsumerProcess:
    def __init__(self, 
        backend_type, 
        pool_name: str, 
        shape: tuple, 
        dtype: str, 
        poll_rate: float, 
        duration: float
    ):
        self.backend_type = backend_type
        self.pool_name = pool_name
        self.shape = shape
        self.dtype = dtype
        self.poll_rate = poll_rate
        self.duration = duration
        self.proc: Optional[subprocess.Popen] = None
        self.stats_file = f"/tmp/consumer_stats_{os.getpid()}_{pool_name}.json"

    def start(self):
        args = [
            sys.executable, __file__, "run",
            self.backend_type,
            self.pool_name,
            str(self.shape),
            self.dtype,
            str(self.poll_rate),
            str(self.duration),
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
        poll_rate = float(sys.argv[6])
        duration = float(sys.argv[7])
        stats_path = sys.argv[8]
        _consumer_worker(backend_type, pool_name, shape, dtype, poll_rate, duration, stats_path)
