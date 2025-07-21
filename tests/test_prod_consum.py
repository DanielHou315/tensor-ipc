import sys
sys.path.append("src/")

import numpy as np
import time
import multiprocessing as mp
from statistics import mean, stdev
from cyclonedds.domain import DomainParticipant

from tensor_ipc.core.producer import TensorProducer
from tensor_ipc.core.consumer import TensorConsumer
import pytest
import itertools

# Set multiprocessing start method to spawn
mp.set_start_method('spawn', force=True)

# Import torch only if available
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

matrix_test_params = {
    "backend": ["numpy"] + (["torch"] if TORCH_AVAILABLE else []) + (["torch_cuda"] if CUDA_AVAILABLE else []),
    "dtype_info": [
        ("float32", np.float32),
        ("uint8", np.uint8), 
        ("int32", np.int32)
    ],
    "shape": [(7,), (4,3), (256, 256, 3)],
    "history_len": [5, 10],
    "read_history": [2, 3],
}

def create_sample_data(shape, dtype_info, backend):
    """Create sample data for the specified backend and dtype."""
    dtype_name, np_dtype = dtype_info
    
    if backend == "numpy":
        return np.ones(shape, dtype=np_dtype)
    elif backend == "torch":
        torch_dtype = getattr(torch, dtype_name)
        return torch.ones(shape, dtype=torch_dtype)
    elif backend == "torch_cuda":
        torch_dtype = getattr(torch, dtype_name)
        return torch.ones(shape, dtype=torch_dtype, device='cuda')
    else:
        raise ValueError(f"Unknown backend: {backend}")

def create_frame_data(shape, dtype_info, backend, frame_value):
    """Create frame data with specific value for the backend."""
    dtype_name, np_dtype = dtype_info
    
    # Ensure frame_value fits in dtype range
    if np_dtype == np.uint8:
        frame_value = frame_value % 256
    
    if backend == "numpy":
        return np.full(shape, frame_value, dtype=np_dtype)
    elif backend == "torch":
        torch_dtype = getattr(torch, dtype_name)
        return torch.full(shape, frame_value, dtype=torch_dtype)
    elif backend == "torch_cuda":
        torch_dtype = getattr(torch, dtype_name)
        return torch.full(shape, frame_value, dtype=torch_dtype, device='cuda')
    else:
        raise ValueError(f"Unknown backend: {backend}")

def extract_frame_value(data, backend):
    """Extract the frame value from data for verification."""
    if backend == "numpy":
        return int(data.flatten()[0])
    elif backend in ["torch", "torch_cuda"]:
        return int(data.flatten()[0].item())
    else:
        raise ValueError(f"Unknown backend: {backend}")

def producer_process(
    pool_name,
    sample_data,
    shape,
    dtype_info,
    backend,
    history_len,
    command_queue,
    result_queue
):
    """Independent producer process."""
    try:
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
        
        print("Producer started and waiting for commands")
        result_queue.put(("READY", None))
        
        frame_counter = 0
        while True:
            command = command_queue.get(timeout=10)
            if command == "STOP":
                break
            elif command == "WRITE":
                timestamp = time.time()
                frame_value = frame_counter % 256
                frame_data = create_frame_data(shape, dtype_info, backend, frame_value)
                frame_idx = producer.put(frame_data)
                frame_counter += 1
                result_queue.put(("WRITTEN", frame_idx, timestamp, frame_value))
        
        producer.cleanup()
        print("Producer finished")
    except Exception as e:
        print(f"Producer error: {e}")
        result_queue.put(("ERROR", str(e)))
        raise

def consumer_process(
    pool_name,
    sample_data,
    backend,
    history_len,
    command_queue,
    result_queue
):
    """Independent consumer process."""
    try:
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
        
        print("Consumer started and waiting for commands")
        result_queue.put(("READY", None))
        
        while True:
            command = command_queue.get(timeout=10)
            if command == "STOP":
                break
            elif isinstance(command, tuple) and command[0] == "READ":
                _, read_history = command
                read_time = time.time()
                
                # Get data with history
                data = consumer.get(
                    history_len=read_history,
                    block=False,
                    as_numpy=False,
                    latest_first=True
                )
                
                if data is not None:
                    if read_history == 1:
                        # Single frame, no history dimension
                        frame_values = [extract_frame_value(data, backend)]
                    else:
                        # Multiple frames with history dimension
                        frame_values = [extract_frame_value(data[i], backend) for i in range(data.shape[0] if hasattr(data, 'shape') else len(data))]
                    result_queue.put(("READ_SUCCESS", frame_values, read_time))
                else:
                    result_queue.put(("READ_EMPTY", None, read_time))
        
        consumer.cleanup()
        print("Consumer finished")
        
    except Exception as e:
        print(f"Consumer error: {e}")
        result_queue.put(("ERROR", str(e)))
        raise

def template_test_producer_consumer(
    backend="numpy",
    shape=(4, 3),
    dtype_info=("float32", np.float32),
    history_len=5,
    read_history=2,
    num_frames=15
):
    """Test function that coordinates producer and consumer processes via queues."""
    print(f"Testing {backend} Producer-Consumer with Queue Coordination")
    print("=" * 50)

    # Create sample data
    sample_data = create_sample_data(shape, dtype_info, backend)
    pool_name = f"test_pool_{backend}_{int(time.time() * 1000)}"  # Unique name

    dtype_name, np_dtype = dtype_info
    
    print(f"Backend: {backend}")
    print(f"Pool name: {pool_name}")
    print(f"Pool shape: {shape}")
    print(f"History length: {history_len}")
    print(f"Data type: {dtype_name}")
    print(f"Frames: {num_frames}")
    print()

    # Create queues for communication
    producer_command_queue = mp.Queue()
    producer_result_queue = mp.Queue()
    consumer_command_queue = mp.Queue()
    consumer_result_queue = mp.Queue()

    # Start producer process
    producer_proc = mp.Process(
        target=producer_process,
        args=(pool_name, sample_data, shape, dtype_info, backend, history_len, 
              producer_command_queue, producer_result_queue)
    )
    producer_proc.start()
    
    # Wait for producer to be ready
    result = producer_result_queue.get(timeout=10)
    if result[0] != "READY":
        assert False, f"Producer failed to initialize: {result}"
    print("✓ Producer ready")
        
    # Start consumer process
    consumer_proc = mp.Process(
        target=consumer_process,
        args=(pool_name, sample_data, backend, history_len,
              consumer_command_queue, consumer_result_queue)
    )
    consumer_proc.start()
    
    # Wait for consumer to be ready
    result = consumer_result_queue.get(timeout=10)
    if result[0] != "READY":
        assert False, f"Consumer failed to initialize: {result}"
    print("✓ Consumer ready")
    
    # Wait a moment for DDS discovery
    time.sleep(0.5)
    
    latencies = []
    written_data = []
    frame_mismatches = 0
    
    for i in range(num_frames):
        # Command producer to write
        producer_command_queue.put("WRITE")
        
        # Wait for producer result
        result = producer_result_queue.get(timeout=20)
        if result[0] == "WRITTEN":
            _, frame_idx, write_timestamp, frame_value = result
            written_data.append((write_timestamp, frame_value))
        elif result[0] == "ERROR":
            raise RuntimeError(f"Producer error: {result[1]}")
        
        # Skip reading until we have enough frames
        if i < read_history - 1:
            continue

        # Command consumer to read
        consumer_command_queue.put(("READ", read_history))
        
        # Wait for consumer result
        result = consumer_result_queue.get(timeout=20)
        if result[0] == "READ_SUCCESS":
            _, read_frame_values, read_time = result

            # Get expected frame values (latest first)
            expected_indices = [i - j for j in range(read_history)]
            expected_frame_values = [written_data[idx][1] for idx in expected_indices if 0 <= idx < len(written_data)]
            expected_timestamps = [written_data[idx][0] for idx in expected_indices if 0 <= idx < len(written_data)]
            
            # Compare frame values
            if len(expected_frame_values) > 0 and read_frame_values[:len(expected_frame_values)] == expected_frame_values:
                if len(expected_timestamps) > 0:
                    latest_latency = (read_time - expected_timestamps[0]) * 1000
                    latencies.append(latest_latency)
            else:
                frame_mismatches += 1
                print(f"Frame mismatch! Expected {expected_frame_values}, got {read_frame_values[:len(expected_frame_values)]}")
        elif result[0] == "ERROR":
            raise RuntimeError(f"Consumer error: {result[1]}")
        
        time.sleep(0.01)
    
    # Check frame mismatches
    if frame_mismatches > num_frames * 0.1:  # Allow some tolerance for DDS timing
        raise RuntimeError(f"Too many frame mismatches: {frame_mismatches}")
    
    # Calculate statistics
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

    # Test single frame read
    print("\nTesting single frame read...")
    consumer_command_queue.put(("READ", 1))
    result = consumer_result_queue.get(timeout=5)
    if result[0] == "READ_SUCCESS":
        print("✓ Single frame read successful")
    else:
        print("✗ Single frame read failed")

    # Stop processes
    print("\nStopping processes...")
    producer_command_queue.put("STOP")
    consumer_command_queue.put("STOP")

    producer_proc.join(timeout=5)
    consumer_proc.join(timeout=5)

    if producer_proc.is_alive():
        producer_proc.terminate()
        producer_proc.join()
        
    if consumer_proc.is_alive():
        consumer_proc.terminate()
        consumer_proc.join()

    if producer_proc.exitcode != 0:
        raise RuntimeError(f"Producer process failed with exit code {producer_proc.exitcode}")
        
    if consumer_proc.exitcode != 0:
        raise RuntimeError(f"Consumer process failed with exit code {consumer_proc.exitcode}")

    print("✓ Test completed successfully")

@pytest.mark.parametrize("backend", matrix_test_params["backend"])
@pytest.mark.parametrize("dtype_info", matrix_test_params["dtype_info"])
@pytest.mark.parametrize("shape", matrix_test_params["shape"])
@pytest.mark.parametrize("history_len", matrix_test_params["history_len"])
@pytest.mark.parametrize("read_history", matrix_test_params["read_history"])
def test_producer_consumer_matrix(backend, dtype_info, shape, history_len, read_history):
    """Pytest parameterized test for all combinations."""
    read_history = min(read_history, history_len)
    
    template_test_producer_consumer(
        backend=backend,
        shape=shape,
        dtype_info=dtype_info,
        history_len=history_len,
        read_history=read_history,
        num_frames=10
    )

if __name__ == "__main__":
    # Parse command line arguments for specific test numbers
    test_numbers = []
    if len(sys.argv) > 1:
        try:
            test_numbers = [int(arg) for arg in sys.argv[1:]]
        except ValueError:
            print("Error: All arguments must be valid test numbers (integers)")
            sys.exit(1)
    
    print("Running comprehensive Producer-Consumer test matrix...")
    print("=" * 60)
    
    # Get all combinations
    param_combinations = list(itertools.product(
        matrix_test_params["backend"],
        matrix_test_params["dtype_info"],
        matrix_test_params["shape"], 
        matrix_test_params["history_len"],
        matrix_test_params["read_history"]
    ))
    
    total_tests = len(param_combinations)
    
    # Filter tests if specific numbers provided
    if test_numbers:
        invalid_numbers = [num for num in test_numbers if num < 1 or num > total_tests]
        if invalid_numbers:
            print(f"Error: Invalid test numbers {invalid_numbers}. Valid range: 1-{total_tests}")
            sys.exit(1)
        
        filtered_combinations = [(i, param_combinations[i-1]) for i in test_numbers]
        print(f"Running specific tests: {test_numbers}")
    else:
        filtered_combinations = [(i, combo) for i, combo in enumerate(param_combinations, 1)]
        print(f"Running all {total_tests} tests")
    
    passed_tests = 0
    failed_tests = []
    
    for test_num, (backend, dtype_info, shape, history_len, read_history) in filtered_combinations:
        read_history = min(read_history, history_len)
        
        print(f"\nTest {test_num}/{total_tests}:")
        print(f"  backend: {backend}")
        print(f"  dtype: {dtype_info[0]}")
        print(f"  shape: {shape}")
        print(f"  history_len: {history_len}")
        print(f"  read_history: {read_history}")
        
        try:
            template_test_producer_consumer(
                backend=backend,
                shape=shape,
                dtype_info=dtype_info,
                history_len=history_len,
                read_history=read_history,
                num_frames=8
            )
            passed_tests += 1
            print("  ✓ PASSED")
        except Exception as e:
            failed_tests.append((test_num, backend, dtype_info[0], shape, history_len, read_history, str(e)))
            print(f"  ✗ FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    if test_numbers:
        print(f"Selected tests: {len(filtered_combinations)}")
    else:
        print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFAILED TESTS:")
        for test_num, backend, dtype, shape, history_len, read_history, error in failed_tests:
            print(f"  Test {test_num}: backend={backend}, dtype={dtype}, shape={shape}, "
                  f"history_len={history_len}, read_history={read_history}")
            print(f"    Error: {error}")
    else:
        print("\n🎉 All tests passed!")
