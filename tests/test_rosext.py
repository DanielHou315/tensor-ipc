import sys
sys.path.append("src/")

import numpy as np
import time
import multiprocessing as mp
from statistics import mean, stdev
import pytest
import itertools
from typing import List, Tuple
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cyclonedds.domain import DomainParticipant

from tensor_ipc.core.metadata import PoolMetadata, MetadataCreator
from tensor_ipc.rosext.ros2pool_consumer import ROSTensorConsumer
from tensor_ipc.rosext.pool2ros_producer import ROSTensorProducer

# Set multiprocessing start method to spawn
mp.set_start_method('spawn', force=True)

# Test parameters for different image sizes
image_test_params = {
    "image_size": [
        (3, 3, 3),      # Small test image
        (640, 480, 3),  # VGA
        (1280, 720, 3), # 720p HD
    ],
    "encoding": ["rgb8", "bgr8"],
    "history_len": [5, 10],
    "read_history": [1, 3],
}

def create_test_image(height: int, width: int, channels: int, frame_value: int) -> np.ndarray:
    """Create a test image with specific dimensions and frame value."""
    # Create a pattern that's easy to verify
    image = np.full((height, width, channels), frame_value % 256, dtype=np.uint8)
    
    # Add a unique pattern to make frame identification easier
    if height >= 3 and width >= 3:
        # Top-left corner pattern
        image[0, 0, :] = frame_value % 256
        image[0, 1, :] = (frame_value + 1) % 256
        image[1, 0, :] = (frame_value + 2) % 256
    
    return image

def extract_frame_identifier(image: np.ndarray) -> int:
    """Extract frame identifier from image pattern."""
    # print("Extracting frame identifier from image shape:", image.shape)
    if image.shape == 3:
        return int(image[0, 0, 0])
    elif image.shape == 4:
        return int(image[0, 0, 0, 0])
    return int(image.flatten()[0])

def create_image_message(image: np.ndarray, encoding: str, frame_id: str = "camera") -> Image:
    """Create a ROS Image message from numpy array."""
    msg = Image()
    msg.header.stamp.sec = int(time.time())
    msg.header.stamp.nanosec = int((time.time() % 1) * 1e9)
    msg.header.frame_id = frame_id
    msg.height = image.shape[0]
    msg.width = image.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = False
    msg.step = image.shape[1] * image.shape[2]
    msg.data = image.tobytes()
    return msg

def ros_producer_process(
    pool_name: str,
    image_size: Tuple[int, int, int],
    encoding: str,
    history_len: int,
    command_queue: mp.Queue,
    result_queue: mp.Queue
):
    """ROS producer process that publishes Image messages."""
    try:
        if not rclpy.ok():
            rclpy.init()

        node = Node(f'test_producer_{int(time.time() * 1000)}')
        
        height, width, channels = image_size
        sample_image = create_test_image(height, width, channels, 0)
        
        # Create pool metadata
        pool_metadata = MetadataCreator.from_sample(
            name=pool_name,
            data=sample_image,
            history_len=history_len,
            backend="numpy"
        )
        
        # Create DDS participant
        participant = DomainParticipant()
        
        # Create ROS producer
        ros_producer = ROSTensorProducer(
            pool_metadata=pool_metadata,
            node=node,
            ros_topic=f"/test_images_{pool_name}",
            ros_msg_type=Image,
            qos=10,
            keep_last=10,
            dds_participant=participant
        )
        
        result_queue.put(("READY", None))
        
        frame_counter = 0
        while True:
            command = command_queue.get(timeout=10)
            if command == "STOP":
                break
            elif command == "PUBLISH":
                timestamp = time.time()
                
                # Create test image with frame identifier
                test_image = create_test_image(height, width, channels, frame_counter)
                
                # Convert to ROS message and publish
                ros_producer.put(test_image, encoding="rgb8")
                
                frame_counter += 1
                result_queue.put(("PUBLISHED", frame_counter - 1, timestamp))
        
        node.destroy_node()
        rclpy.shutdown()
        
    except Exception as e:
        result_queue.put(("ERROR", str(e)))
        raise

def ros_consumer_process(
    pool_name: str,
    image_size: Tuple[int, int, int],
    encoding: str,
    history_len: int,
    command_queue: mp.Queue,
    result_queue: mp.Queue
):
    """ROS consumer process that subscribes to Image messages and reads from tensor pool."""
    try:
        rclpy.init()
        node = Node(f'test_consumer_{int(time.time() * 1000)}')
        
        height, width, channels = image_size
        sample_image = create_test_image(height, width, channels, 0)
        
        # Create pool metadata
        pool_metadata = MetadataCreator.from_sample(
            name=pool_name,
            data=sample_image,
            history_len=history_len,
            backend="numpy"
        )
        
        # Create DDS participant
        participant = DomainParticipant()
        
        # Create ROS consumer
        ros_consumer = ROSTensorConsumer(
            pool_metadata=pool_metadata,
            node=node,
            ros_topic=f"/test_images_{pool_name}",
            ros_msg_type=Image,
            qos=10,
            keep_last=10,
            dds_participant=participant
        )
        
        result_queue.put(("READY", None))
        
        while True:
            command = command_queue.get(timeout=10)
            if command == "STOP":
                break
            elif isinstance(command, tuple) and command[0] == "READ":
                _, read_history = command
                read_time = time.time()
                
                # Spin once to process any pending ROS messages
                rclpy.spin_once(node, timeout_sec=0.001)
                
                # Read from tensor pool
                data = ros_consumer.get(
                    history_len=read_history,
                    as_numpy=True,
                    latest_first=True
                )
                
                if data is None:
                    result_queue.put(("READ_EMPTY", None, read_time))
                    continue
                
                if read_history == 1:
                    # Single frame
                    frame_ids = [extract_frame_identifier(data)]
                else:
                    # Multiple frames with history dimension
                    frame_ids = [extract_frame_identifier(data[i]) for i in range(data.shape[0])]
                
                result_queue.put(("READ_SUCCESS", frame_ids, read_time))
        
        node.destroy_node()
        rclpy.shutdown()
        
    except Exception as e:
        result_queue.put(("ERROR", str(e)))
        raise

def template_test_ros_image_latency(
    image_size: Tuple[int, int, int] = (64, 64, 3),
    encoding: str = "rgb8",
    history_len: int = 5,
    read_history: int = 1,
    num_frames: int = 20
) -> dict:
    """Test ROS Image message latency with statistical analysis."""
    
    height, width, channels = image_size
    pool_name = f"test_ros_pool_{height}x{width}_{int(time.time() * 1000)}"
    
    # Create queues for communication
    producer_command_queue = mp.Queue()
    producer_result_queue = mp.Queue()
    consumer_command_queue = mp.Queue()
    consumer_result_queue = mp.Queue()
    
    # Start producer process
    producer_proc = mp.Process(
        target=ros_producer_process,
        args=(pool_name, image_size, encoding, history_len,
              producer_command_queue, producer_result_queue)
    )
    producer_proc.start()
    
    # Wait for producer to be ready
    result = producer_result_queue.get(timeout=15)
    assert result[0] == "READY", f"Producer failed to initialize: {result}"
    
    # Start consumer process
    consumer_proc = mp.Process(
        target=ros_consumer_process,
        args=(pool_name, image_size, encoding, history_len,
              consumer_command_queue, consumer_result_queue)
    )
    consumer_proc.start()
    
    # Wait for consumer to be ready
    result = consumer_result_queue.get(timeout=15)
    assert result[0] == "READY", f"Consumer failed to initialize: {result}"
    
    # Wait for ROS and DDS discovery
    time.sleep(1.0)
    
    latencies = []
    publish_times = []
    frame_mismatches = 0
    successful_reads = 0
    
    for i in range(num_frames):
        # Command producer to publish
        producer_command_queue.put("PUBLISH")
        
        # Wait for producer result
        result = producer_result_queue.get(timeout=10)
        if result[0] == "PUBLISHED":
            _, frame_idx, publish_timestamp = result
            publish_times.append((frame_idx, publish_timestamp))
        elif result[0] == "ERROR":
            raise RuntimeError(f"Producer error: {result[1]}")
        
        # Allow some time for message propagation
        time.sleep(0.05)
        
        # Command consumer to read
        consumer_command_queue.put(("READ", read_history))
        
        # Wait for consumer result
        result = consumer_result_queue.get(timeout=10)
        if result[0] == "READ_SUCCESS":
            _, read_frame_ids, read_time = result
            successful_reads += 1
            
            # Calculate latency based on most recent frame
            if len(read_frame_ids) > 0 and len(publish_times) > 0:
                latest_frame_id = read_frame_ids[0]
                
                # Find corresponding publish time
                for pub_frame_idx, pub_time in reversed(publish_times):
                    if pub_frame_idx == latest_frame_id:
                        latency_ms = (read_time - pub_time) * 1000
                        latencies.append(latency_ms)
                        break
                else:
                    frame_mismatches += 1
                    
        elif result[0] == "READ_EMPTY":
            continue
        elif result[0] == "ERROR":
            raise RuntimeError("Consumer error:", result[1])

    # Stop processes
    producer_command_queue.put("STOP")
    consumer_command_queue.put("STOP")
    
    producer_proc.join(timeout=10)
    consumer_proc.join(timeout=10)
    
    if producer_proc.is_alive():
        producer_proc.terminate()
        producer_proc.join()
    
    if consumer_proc.is_alive():
        consumer_proc.terminate()
        consumer_proc.join()
    
    # Statistical analysis
    stats = {
        "image_size": image_size,
        "encoding": encoding,
        "total_frames": num_frames,
        "successful_reads": successful_reads,
        "frame_mismatches": frame_mismatches,
        "latencies": latencies,
    }
    
    if latencies:
        stats.update({
            "mean_latency_ms": mean(latencies),
            "std_latency_ms": stdev(latencies) if len(latencies) > 1 else 0.0,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "median_latency_ms": sorted(latencies)[len(latencies)//2],
            "p95_latency_ms": sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) > 20 else max(latencies),
            "p99_latency_ms": sorted(latencies)[int(len(latencies)*0.99)] if len(latencies) > 100 else max(latencies),
        })
    
    return stats

# Generate test parameters
params = list(itertools.product(
    image_test_params["image_size"],
    image_test_params["encoding"],
    image_test_params["history_len"],
    image_test_params["read_history"]
))

@pytest.mark.parametrize(
    "param_tuple",
    params,
    ids=[f"img_{h}x{w}x{c}_{enc}_h{hl}_r{rh}" 
         for (h,w,c), enc, hl, rh in params]
)
def test_ros_image_latency_matrix(param_tuple):
    """Pytest parameterized test for ROS Image latency across different configurations."""
    image_size, encoding, history_len, read_history = param_tuple
    read_history = min(read_history, history_len)
    
    stats = template_test_ros_image_latency(
        image_size=image_size,
        encoding=encoding,
        history_len=history_len,
        read_history=read_history,
        num_frames=15
    )
    
    # Assert basic functionality
    assert stats["successful_reads"] > 0, "No successful reads occurred"
    assert stats["frame_mismatches"] < stats["total_frames"] * 0.3, "Too many frame mismatches"
    
    # Assert latency bounds (adjust based on your requirements)
    # if stats["latencies"]:
    #     assert stats["mean_latency_ms"] < 100, f"Mean latency too high: {stats['mean_latency_ms']:.2f}ms"
    #     assert stats["max_latency_ms"] < 200, f"Max latency too high: {stats['max_latency_ms']:.2f}ms"
    
    # Print statistics for analysis
    height, width, channels = image_size
    print(f"\n=== ROS Image Test Results ===")
    print(f"Image Size: {height}x{width}x{channels} ({encoding})")
    print(f"History: {history_len}, Read: {read_history}")
    print(f"Successful Reads: {stats['successful_reads']}/{stats['total_frames']}")
    
    if stats["latencies"]:
        print(f"Latency Stats (ms):")
        print(f"  Mean: {stats['mean_latency_ms']:.2f} ± {stats['std_latency_ms']:.2f}")
        print(f"  Min/Max: {stats['min_latency_ms']:.2f} / {stats['max_latency_ms']:.2f}")
        print(f"  Median: {stats['median_latency_ms']:.2f}")
        if 'p95_latency_ms' in stats:
            print(f"  P95: {stats['p95_latency_ms']:.2f}")

def test_ros_image_size_scaling():
    """Test latency scaling across different image sizes."""
    results = []
    
    test_sizes = [(3, 3, 3), (64, 64, 3), (320, 240, 3), (640, 480, 3)]
    
    for size in test_sizes:
        stats = template_test_ros_image_latency(
            image_size=size,
            encoding="rgb8",
            history_len=5,
            read_history=1,
            num_frames=20
        )
        results.append(stats)
    
    # Print comparative analysis
    print(f"\n=== Image Size Scaling Analysis ===")
    for stats in results:
        h, w, c = stats["image_size"]
        pixels = h * w * c
        if stats["latencies"]:
            print(f"{h}x{w}x{c} ({pixels:,} bytes): "
                  f"Mean={stats['mean_latency_ms']:.2f}ms, "
                  f"Std={stats['std_latency_ms']:.2f}ms")

if __name__ == "__main__":
    # Run simplest test case
    print("Running simplest ROS Image test case...")
    stats = template_test_ros_image_latency(
        image_size=(3, 3, 3),      # Smallest image
        encoding="rgb8",
        history_len=5,
        read_history=1,
        num_frames=10              # Fewer frames for quick test
    )
    
    print(f"\n=== Simple Test Results ===")
    print(f"Image Size: 3x3x3 (rgb8)")
    print(f"Successful Reads: {stats['successful_reads']}/{stats['total_frames']}")
    print(f"Frame Mismatches: {stats['frame_mismatches']}")
    
    if stats["latencies"]:
        print(f"Latency Stats (ms):")
        print(f"  Mean: {stats['mean_latency_ms']:.2f} ± {stats['std_latency_ms']:.2f}")
        print(f"  Min/Max: {stats['min_latency_ms']:.2f} / {stats['max_latency_ms']:.2f}")
        print(f"Test PASSED - Basic functionality working")
    else:
        print("Test FAILED - No latency measurements obtained")
