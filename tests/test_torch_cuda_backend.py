import sys
sys.path.append("src/")


import torch
import time
import multiprocessing as mp
from statistics import mean, stdev

from tensor_ipc.core.metadata import MetadataCreator
from tensor_ipc.backends.torch_cuda_backend import (
    TorchCUDAProducerBackend, 
    TorchCUDAConsumerBackend
)
import pytest
import itertools

# Set multiprocessing start method to spawn
mp.set_start_method('spawn', force=True)

matrix_test_params = {
    "dtype": [torch.float32, torch.uint8, torch.int32],
    "shape": [(7,), (4,3), (1024, 768, 3), (1920, 1080, 3)],
    "history_len": [5, 20],
    "read_history": [2, 5],
}

# matrix_test_params = {
#     "dtype": [torch.uint8],
#     "shape": [(1920, 1080, 3)],
#     "history_len": [10],
#     "read_history": [2],
# }


def producer_process(
    pool_metadata,
    shape,
    dtype,
    command_queue,
    result_queue
):
    """Independent producer process that writes frame indices when commanded."""
    try:
        producer = TorchCUDAProducerBackend(
            pool_metadata=pool_metadata,
            history_pad_strategy="zero",
            force=False
        )
        
        print("Producer started and waiting for commands")
        
        # Send the updated metadata with IPC handles to main process
        result_queue.put(("METADATA_READY", producer.metadata))
        
        frame_counter = 0
        while True:
            # Wait for command from main process
            command = command_queue.get(timeout=10)
            if command == "STOP":
                break
            elif command == "WRITE":
                # Write frame index data (modulo 256 for uint8 compatibility)
                timestamp = time.time()
                frame_value = frame_counter % 256
                frame_data = torch.full(shape, frame_value, dtype=dtype, device='cuda')
                frame_idx = producer.write(frame_data)
                frame_counter += 1
                # Send result back to main process with timestamp
                result_queue.put(("WRITTEN", frame_idx, timestamp, frame_value))
        producer.cleanup()
        print("Producer finished")
    except Exception as e:
        print(f"Producer error: {e}")
        raise


def consumer_process(
    pool_metadata,
    command_queue,
    result_queue
):
    """Independent consumer process that reads when commanded."""
    try:
        consumer = TorchCUDAConsumerBackend(
            pool_metadata=pool_metadata,
        )
        
        print("Consumer started and waiting for commands")
        
        while True:
            # Wait for command from main process
            command = command_queue.get(timeout=10)
            if command == "STOP":
                print("Consumer stopping")
                break
            elif isinstance(command, tuple) and command[0] == "READ":
                # Extract read indices from command
                if not consumer.is_connected:
                    consumer.connect(pool_metadata)

                _, indices = command
                # Read data
                read_time = time.time()
                data = consumer.read(indices, as_numpy=False)
                if data is not None and data.shape[0] > 0:
                    # Extract frame values from data
                    frame_values = [int(data[i].flatten()[0]) for i in range(data.shape[0])]
                    result_queue.put(("READ_SUCCESS", frame_values, read_time))
                else:
                    result_queue.put(("READ_EMPTY", None, read_time))
        consumer.cleanup()
        print("Consumer finished")
        
    except Exception as e:
        print(f"Consumer error: {e}")
        raise


def template_test_torch_backend(
    shape=(1, 3),
    dtype=torch.float32,
    history_len=5,
    read_history=2,
    num_frames=15,
    hz=10
):
    """Test function that coordinates producer and consumer processes via queues."""
    print("Testing TorchBackend Producer-Consumer with Queue Coordination")
    print("=" * 50)

    # Create sample data for metadata
    sample_data = torch.ones(shape, dtype=dtype, device="cuda")

    # Create metadata
    pool_metadata = MetadataCreator.from_torch_cuda_sample(
        name="test_pool",
        sample_data=sample_data,
        history_len=history_len
    )

    print(f"Pool shape: {shape}")
    print(f"History length: {history_len}")
    print(f"Data type: {dtype}")
    print(f"Frames: {num_frames}")
    print(f"Shared memory size: {pool_metadata.total_size} bytes")
    print()

    # Create queues for communication
    producer_command_queue = mp.Queue()
    producer_result_queue = mp.Queue()
    consumer_command_queue = mp.Queue()
    consumer_result_queue = mp.Queue()

    # Start producer process first
    producer_proc = mp.Process(
        target=producer_process,
        args=(pool_metadata, shape, dtype, producer_command_queue, producer_result_queue)
    )
    producer_proc.start()
    
    # Wait for producer to initialize and get updated metadata with IPC handles
    print("Waiting for producer metadata with IPC handles...")
    result = producer_result_queue.get(timeout=10)
    if result[0] == "METADATA_READY":
        producer_metadata = result[1]
        print("✓ Producer metadata with IPC handles received")
    else:
        assert False, "Producer failed to initialize properly"
        
    # Now start consumer process with the updated metadata
    consumer_proc = mp.Process(
        target=consumer_process,
        args=(producer_metadata, consumer_command_queue, consumer_result_queue)
    )
    consumer_proc.start()
    
    # Wait a moment for consumer to initialize
    time.sleep(0.2)
    
    # print("Coordinating producer and consumer...")
    latencies = []
    written_data = []  # Store (timestamp, frame_value) tuples
    frame_mismatches = 0
    
    for i in range(num_frames):
        # Command producer to write
        producer_command_queue.put("WRITE")
        
        # Wait for producer result
        result = producer_result_queue.get(timeout=20)
        if result[0] == "WRITTEN":
            _, frame_idx, write_timestamp, frame_value = result
            written_data.append((write_timestamp, frame_value))
            # print(f"  Producer wrote frame {frame_value} to slot {frame_idx}, timestamp: {write_timestamp}")
        elif result[0] == "ERROR":
            raise RuntimeError(f"Producer error: {result[1]}")
        
        # If we have enough frames for reading
        if i < read_history - 1:
            # print("Skipping read")
            continue

        # Calculate read indices (most recent frames)
        end_idx = frame_idx % history_len
        read_indices = [(end_idx - j) % history_len for j in range(read_history)]
        # print(f"  Reading indices: {read_indices}")
        
        # Command consumer to read
        consumer_command_queue.put(("READ", read_indices))
        
        # Wait for consumer result
        result = consumer_result_queue.get(timeout=20)
        if result[0] == "READ_SUCCESS":
            _, read_frame_values, read_time = result

            # Get expected frame values (time-backwards, latest frame first)
            expected_indices = [i - j for j in range(read_history)]
            expected_frame_values = [written_data[idx][1] for idx in expected_indices if 0 <= idx < len(written_data)]
            expected_timestamps = [written_data[idx][0] for idx in expected_indices if 0 <= idx < len(written_data)]
            
            # Compare frame values
            if len(expected_frame_values) > 0 and read_frame_values[:len(expected_frame_values)] == expected_frame_values:
                # Calculate latency using the timestamp of the first (most recent) frame
                if len(expected_timestamps) > 0 and expected_timestamps[0] > 0:
                    latest_latency = (read_time - expected_timestamps[0]) * 1000
                    latencies.append(latest_latency)
            else:
                frame_mismatches += 1
                assert False, f"Frame value mismatch! Expected {expected_frame_values}, got {read_frame_values[:len(expected_frame_values)]}"
        else:
            assert False, f"Unexpected consumer result: {result}"
        
        # Small delay between frames
        time.sleep(0.01)
    
    # Check if we had frame mismatches - this indicates a failure in IPC
    if frame_mismatches > 0:
        raise RuntimeError(f"Test failed: {frame_mismatches} frame value mismatches detected. "
                            "Consumer is not reading fresh data from producer's shared memory. "
                            "This indicates CUDA IPC is not working properly.")
    
    # Calculate and print final statistics
    if len(latencies) > 1:
        latencies = latencies[1:]
        avg_latency = mean(latencies)
        std_latency = stdev(latencies) if len(latencies) > 1 else 0
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"\nFinal Latency Statistics ({len(latencies)} samples):")
        print(f"  Average: {avg_latency:.2f} ms")
        print(f"  Std Dev: {std_latency:.2f} ms")
        print(f"  Min: {min_latency:.2f} ms")
        print(f"  Max: {max_latency:.2f} ms")
    else:
        raise RuntimeError("No valid latency measurements captured - this indicates the test failed")
    
    # Test reading full history
    print("\nTesting full history read...")
    full_indices = list(range(history_len))
    consumer_command_queue.put(("READ", full_indices))
    
    try:
        result = consumer_result_queue.get(timeout=5)
        if result[0] == "READ_SUCCESS":
            _, frame_values, _ = result
            print(f"✓ Full history read successful, got {len(frame_values)} frame values")
        else:
            print("✗ Full history read failed")
    except Exception as e:
        print(f"✗ Full history read timed out: {e}")

    # Stop both processes
    print("\nStopping processes...")
    producer_command_queue.put("STOP")
    consumer_command_queue.put("STOP")

    # Wait for processes to complete
    producer_proc.join(timeout=5)
    consumer_proc.join(timeout=5)

    # Check if processes completed successfully
    if producer_proc.is_alive():
        producer_proc.terminate()
        producer_proc.join()
        raise RuntimeError("Producer process failed to stop")
        
    if consumer_proc.is_alive():
        consumer_proc.terminate()
        consumer_proc.join()
        raise RuntimeError("Consumer process failed to stop")

    if producer_proc.exitcode != 0:
        raise RuntimeError(f"Producer process failed with exit code {producer_proc.exitcode}")
        
    if consumer_proc.exitcode != 0:
        raise RuntimeError(f"Consumer process failed with exit code {consumer_proc.exitcode}")

    print("✓ Both processes completed successfully")

    print("\n" + "=" * 50)
    print("Queue-coordinated test completed successfully!")


@pytest.mark.parametrize("dtype", matrix_test_params["dtype"])
@pytest.mark.parametrize("shape", matrix_test_params["shape"])
@pytest.mark.parametrize("history_len", matrix_test_params["history_len"])
@pytest.mark.parametrize("read_history", matrix_test_params["read_history"])
def test_torch_backend_matrix(dtype, shape, history_len, read_history):
    """Pytest parameterized test for all combinations of test parameters."""
    # Ensure read_history doesn't exceed history_len
    read_history = min(read_history, history_len)

    template_test_torch_backend(
        shape=shape,
        dtype=dtype,
        history_len=history_len,
        read_history=read_history,
        num_frames=15
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
    
    print("Running comprehensive test matrix...")
    print("=" * 60)
    
    # Get all combinations of test parameters
    param_combinations = list(itertools.product(
        matrix_test_params["dtype"],
        matrix_test_params["shape"], 
        matrix_test_params["history_len"],
        matrix_test_params["read_history"]
    ))
    
    total_tests = len(param_combinations)
    
    # Filter tests if specific numbers provided
    if test_numbers:
        # Validate test numbers
        invalid_numbers = [num for num in test_numbers if num < 1 or num > total_tests]
        if invalid_numbers:
            print(f"Error: Invalid test numbers {invalid_numbers}. Valid range: 1-{total_tests}")
            sys.exit(1)
        
        # Filter combinations to only run specified tests
        filtered_combinations = [(i, param_combinations[i-1]) for i in test_numbers]
        print(f"Running specific tests: {test_numbers}")
    else:
        filtered_combinations = [(i, combo) for i, combo in enumerate(param_combinations, 1)]
        print(f"Running all {total_tests} tests")
    
    passed_tests = 0
    failed_tests = []
    
    for test_num, (dtype, shape, history_len, read_history) in filtered_combinations:
        # Ensure read_history doesn't exceed history_len
        read_history = min(read_history, history_len)
        
        print(f"\nTest {test_num}/{total_tests}:")
        print(f"  dtype: {dtype}")
        print(f"  shape: {shape}")
        print(f"  history_len: {history_len}")
        print(f"  read_history: {read_history}")
        
        try:
            template_test_torch_backend(
                shape=shape,
                dtype=dtype,
                history_len=history_len,
                read_history=read_history,
                num_frames=10  # Reduced for faster execution
            )
            passed_tests += 1
            print("  ✓ PASSED")
        except Exception as e:
            failed_tests.append((test_num, dtype, shape, history_len, read_history, str(e)))
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
        for test_num, dtype, shape, history_len, read_history, error in failed_tests:
            print(f"  Test {test_num}: dtype={dtype}, shape={shape}, "
                  f"history_len={history_len}, read_history={read_history}")
            print(f"    Error: {error}")
    else:
        print("\n🎉 All tests passed!")