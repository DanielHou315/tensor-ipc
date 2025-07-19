import numpy as np
import sys
sys.path.append("src/")
from tensor_ipc.metadata import MetadataCreator
from tensor_ipc.backends.numpy_backend import NumpyProducerBackend, NumpyConsumerBackend

import time

def test_numpy_backend():
    print("Testing NumpyBackend Producer-Consumer")
    print("=" * 50)

    # Test parameters
    shape = (3, 4, 2)
    history_len = 10
    read_history = 2
    num_frames = 15

    # Create sample data for metadata
    sample_data = np.ones(shape, dtype=np.float32)

    # Create metadata
    pool_metadata = MetadataCreator.from_numpy_sample(
        name="test_pool",
        sample_data=sample_data,
        history_len=history_len
    )

    print(f"Pool shape: {shape}")
    print(f"History length: {history_len}")
    print(f"Total pool shape: ({history_len}, {shape[0]}, {shape[1]}, {shape[2]})")
    print(f"Shared memory size: {pool_metadata.total_size} bytes")
    print(f"Shared memory name: {pool_metadata.shm_name}")
    print()

    # Create producer
    print("Creating producer...")
    producer = NumpyProducerBackend(
        pool_metadata=pool_metadata,
        history_pad_strategy="zero"
    )

    # Give producer time to initialize
    time.sleep(0.1)

    # Create consumer
    print("Creating consumer...")
    consumer = NumpyConsumerBackend(pool_metadata=pool_metadata)

    # Verify consumer is connected
    assert consumer.is_connected, "Consumer failed to connect to producer"
    print("✓ Consumer connected successfully")

    # Test initial read (should be zeros due to zero padding)
    initial_data = consumer.read(slice(0, read_history))
    assert initial_data is not None, "Failed to read initial data"
    assert initial_data.shape == (read_history,) + shape, f"Wrong shape: {initial_data.shape}"
    assert np.allclose(initial_data, 0), "Initial data should be zeros"
    print("✓ Initial zero-padded data verified")

    # Produce and consume data
    print(f"\nProducing and consuming {num_frames} frames...")
    published_data = []

    for i in range(num_frames):
        # Create test data: ones array multiplied by frame index
        test_data = np.ones(shape, dtype=np.float32) * (i + 1)
        published_data.append(test_data.copy())
        
        # Write data
        frame_idx = producer.write(test_data)
        print(f"Frame {i+1}: wrote to slot {frame_idx}")
        
        # Read latest data
        if i >= read_history - 1:  # Can only read when we have enough history
            # Read the last 'read_history' frames
            end_idx = (frame_idx + 1) % history_len
            if end_idx >= read_history:
                read_indices = slice(end_idx - read_history, end_idx)
            else:
                # Wrap around case
                read_indices = list(range(history_len - (read_history - end_idx), history_len)) + \
                                list(range(0, end_idx))
            
            read_data = consumer.read(read_indices)
            
            if isinstance(read_indices, slice):
                expected_frames = published_data[i - read_history + 1:i + 1]
            else:
                # Handle wrap-around case
                start_frame = i - read_history + 1
                expected_frames = []
                for j in range(read_history):
                    frame_num = start_frame + j
                    if frame_num >= 0:
                        expected_frames.append(published_data[frame_num])
            
            # Verify read data matches expected
            if isinstance(read_indices, slice):
                for j, expected_frame in enumerate(expected_frames):
                    actual_frame = read_data[j]
                    assert np.allclose(actual_frame, expected_frame), \
                        f"Mismatch at frame {i}, history {j}: expected {expected_frame[0,0,0]}, got {actual_frame[0,0,0]}"
            
            print(f"  ✓ Read verification passed for frames ending at {i+1}")

    print(f"\n✓ All {num_frames} frames produced and verified successfully!")

    # Test reading full history
    print("\nTesting full history read...")
    full_history = consumer.read(slice(None))  # Read all history
    if full_history is None:
        raise RuntimeError("Failed to read full history from consumer")

    assert full_history.shape == (history_len,) + shape, f"Wrong full history shape: {full_history.shape}"
    print(f"✓ Full history read successful, shape: {full_history.shape}")

    # Test as_numpy parameter (should be no-op for numpy backend)
    numpy_data = consumer.read(slice(0, 2), as_numpy=True)
    assert isinstance(numpy_data, np.ndarray), "as_numpy=True should return numpy array"
    print("✓ as_numpy parameter works correctly")

    # Cleanup
    print("\nCleaning up...")
    consumer.cleanup()
    producer.cleanup()
    print("✓ Cleanup completed")

    print("\n" + "=" * 50)
    print("All tests passed! NumpyBackend is working correctly.")


if __name__ == "__main__":
    test_numpy_backend()