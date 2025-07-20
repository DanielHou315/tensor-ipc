import sys
sys.path.append("src/")

import numpy as np
import torch

from tensor_ipc.core.metadata import MetadataCreator
from tensor_ipc.backends.torch_backend import (
    TorchProducerBackend, 
    TorchConsumerBackend
)
import pytest
import itertools

import time

matrix_test_params = {
    "dtype": [torch.float32, torch.uint8, torch.int32],
    "shape": [(7,), (4,3), (1024, 768, 3), (1920, 1080, 3)],
    "history_len": [5, 20],
    "read_history": [2, 5],
}

# matrix_test_params = {
#     "dtype": [torch.float32],
#     "shape": [(7,)],
#     "history_len": [5],
#     "read_history": [2],
# }


def template_test_torch_backend(
    shape=(1, 3),
    dtype=torch.float32,
    history_len=5,
    read_history=2,
    num_frames=15
):
    print("Testing TorchBackend Producer-Consumer")
    print("=" * 50)

    # Create sample data for metadata
    sample_data = torch.ones(shape, dtype=dtype)

    # Create metadata
    pool_metadata = MetadataCreator.from_torch_sample(
        name="test_pool",
        sample_data=sample_data,
        history_len=history_len
    )

    print(f"Pool shape: {shape}")
    print(f"History length: {history_len}")
    print(f"Total pool shape: ({history_len}, {shape})")
    print(f"Data type: {dtype}")
    print(f"Shared memory size: {pool_metadata.total_size} bytes")
    print(f"Shared memory name: {pool_metadata.shm_name}")
    print()

    # Create producer
    print("Creating producer...")
    producer = TorchProducerBackend(
        pool_metadata=pool_metadata,
        history_pad_strategy="zero",
        force=True
    )

    # Give producer time to initialize
    time.sleep(0.1)

    # Create consumer
    consumer = TorchConsumerBackend(
        pool_metadata=pool_metadata,
    )

    # Verify consumer is connected
    assert consumer.is_connected, "Consumer failed to connect to producer"
    print("✓ Consumer connected successfully")

    # Test initial read (should be zeros due to zero padding)
    read_indices = np.arange(read_history, 0, -1) % history_len
    initial_data = consumer.read(read_indices)
    assert initial_data is not None, "Failed to read initial data"
    assert initial_data.shape == (read_history,) + shape, f"Wrong shape: {initial_data.shape}"
    assert np.allclose(initial_data, 0), "Initial data should be zeros"
    print("✓ Initial zero-padded data verified")

    # Produce and consume data
    print(f"\nProducing and consuming {num_frames} frames...")
    published_data = []

    for i in range(num_frames):
        # Create test data: ones array multiplied by frame index
        test_data = torch.ones(shape, dtype=dtype) * (i + 1)
        published_data.append(test_data.clone())

        # Write data
        frame_idx = producer.write(test_data)
        # print(f"Frame {i+1}: wrote to slot {frame_idx} at {producer._pool_metadata.shm_name}")
        
        # Read latest data
        if i >= read_history - 1:  # Can only read when we have enough history
            # Read the last 'read_history' frames
            end_idx = (frame_idx) % history_len
            # print("End index for reading:", end_idx)

            # Generate read indices
            read_indices = np.arange(end_idx, end_idx-read_history, -1) % history_len
            read_data = consumer.read(read_indices)
            # print(f"  Read from {consumer._pool_metadata.shm_name} at indices {read_indices.tolist()}")

            # Compute reverse of published data for verification
            expected_data = torch.stack(published_data[-read_history:][::-1])
            assert torch.allclose(read_data, expected_data), \
                f"Mismatch in read data at frame {i+1}: expected {expected_data}, got {read_data}"
            # print(f"  ✓ Read verification passed for frames ending at {i+1}")

    print(f"\n✓ All {num_frames} frames produced and verified successfully!")

    # Test reading full history
    print("\nTesting full history read...")
    full_history = consumer.read(indices=np.arange(history_len))  # Read all history
    if full_history is None:
        raise RuntimeError("Failed to read full history from consumer")

    assert full_history.shape == (history_len,) + shape, f"Wrong full history shape: {full_history.shape}"
    print(f"✓ Full history read successful, shape: {full_history.shape}")

    # Test as_numpy parameter
    numpy_data = consumer.read([0,1], as_numpy=True)
    assert isinstance(numpy_data, np.ndarray), "as_numpy=True should return numpy array"
    print("✓ as_numpy parameter works correctly")

    # Cleanup
    print("\nCleaning up...")
    consumer.cleanup()
    producer.cleanup()
    print("✓ Cleanup completed")

    print("\n" + "=" * 50)
    print("All tests passed! TorchBackend is working correctly.")

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
            print(f"  ✓ PASSED")
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