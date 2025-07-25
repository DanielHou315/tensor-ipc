import sys
sys.path.append("src/")

import time
import subprocess
import json
import os
from statistics import mean, stdev

# Import functions from prod_consum_daemon

import pytest
import itertools

matrix_test_params = {
    "backend": ["numpy", "torch", "torch_cuda"],
    "dtype_info": ["float32","uint8","int32"],
    "shape": [(7,), (1920, 1080, 3)],
    "history_len": [5, 20],
    "read_history": [2,],
}

def template_test_producer_consumer_subprocess(
    backend="numpy",
    shape=(4, 3),
    dtype_info="float32",
    history_len=5,
    read_history=2,
    num_frames=50
):
    """Test function using subprocess.Popen for truly independent processes."""
    print(f"Testing {backend} Producer-Consumer with Subprocess")
    print("=" * 50)

    # Create temporary directory for scripts and config
    daemon_dir = os.path.join(os.path.dirname(__file__), "prod_consum_daemon")
    producer_script_path = os.path.join(daemon_dir, "producer_daemon.py")
    consumer_script_path = os.path.join(daemon_dir, "consumer_daemon.py")
    # Create config file
    pool_name = f"test_pool_{backend}_{int(time.time() * 1000)}"
    config = {
        "pool_name": pool_name,
        "shape": list(shape),
        "dtype_info": dtype_info,
        "backend": backend,
        "history_len": history_len,
        "read_history": read_history,
        "num_frames": num_frames
    }
    
    config_path = os.path.join(daemon_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)

    results_path = os.path.join(daemon_dir, "results.json")

    # Start consumer process first
    consumer_proc = subprocess.Popen([
        sys.executable, consumer_script_path,
        "--config", config_path,
        "--results", results_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait a moment then start producer
    time.sleep(0.1)
    
    producer_proc = subprocess.Popen([
        sys.executable, producer_script_path,
        "--config", config_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    print("✓ Producer and Consumer processes started")
    
    # Wait for both processes to complete
    producer_stdout, producer_stderr = producer_proc.communicate(timeout=30)
    consumer_stdout, consumer_stderr = consumer_proc.communicate(timeout=30)
    
    print("Producer output:", producer_stdout)
    if producer_stderr:
        print("Producer errors:", producer_stderr)
        
    print("Consumer output:", consumer_stdout)
    if consumer_stderr:
        print("Consumer errors:", consumer_stderr)
    
    # Check exit codes
    if producer_proc.returncode != 0:
        raise RuntimeError(f"Producer process failed with exit code {producer_proc.returncode}")
    
    if consumer_proc.returncode != 0:
        raise RuntimeError(f"Consumer process failed with exit code {consumer_proc.returncode}")
    
    # Read results
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        successful_reads = results["successful_reads"]
        latencies = results["latencies"]
        
        print(f"Successful reads: {successful_reads}")
        
        if len(latencies) > 1:
            latencies = latencies[1:]  # Skip first measurement
            avg_latency = mean(latencies)
            std_latency = stdev(latencies) if len(latencies) > 1 else 0
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            print(f"\nLatency Statistics ({len(latencies)} samples):")
            print(f"  Average: {avg_latency:.2f} ms")
            print(f"  Std Dev: {std_latency:.2f} ms")
            print(f"  Min: {min_latency:.2f} ms")
            print(f"  Max: {max_latency:.2f} ms")
        
        # Basic validation
        if successful_reads < num_frames * 0.5:  # At least 50% success rate
            raise RuntimeError(f"Too few successful reads: {successful_reads}/{num_frames}")
            
    else:
        raise RuntimeError("Results file not found")

    print("✓ Test completed successfully")

params = list(itertools.product(
    matrix_test_params["backend"],
    matrix_test_params["dtype_info"],
    matrix_test_params["shape"],
    matrix_test_params["history_len"],
    matrix_test_params["read_history"]
))

@pytest.mark.parametrize(
    "param_tuple",
    params,
    ids=[f"case_{i:03d}" for i in range(len(params))]
)
def test_producer_consumer_matrix(param_tuple):
    """Pytest parameterized test for all combinations."""
    backend, dtype_info, shape, history_len, read_history = param_tuple
    read_history = min(read_history, history_len)

    template_test_producer_consumer_subprocess(
        backend=backend,
        shape=shape,
        dtype_info=dtype_info,
        history_len=history_len,
        read_history=read_history,
        num_frames=50
    )

if __name__ == "__main__":
    pytest.main([__file__])