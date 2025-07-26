# API Documentation

This document provides detailed API reference for the tensor-ipc package, covering core classes, metadata creation utilities, and ROS extensions.

## Core Classes

### TensorProducer

A client for writing tensors to a shared pool with notification support.

#### Constructor

```python
TensorProducer(
    pool_metadata: PoolMetadata,
    dds_participant: Optional[DomainParticipant] = None,
    keep_last: int = 10
)
```

**Parameters:**
- `pool_metadata`: Metadata describing the shared memory pool structure
- `dds_participant`: Optional DDS participant for notifications (default: None)
- `keep_last`: Number of latest frames to keep in DDS history (default: 10)

**Raises:**
- `ValueError`: If the topic name is already published by another producer

#### Class Methods

##### `from_sample(pool_name, sample, history_len=1, dds_participant=None, keep_last=10)`

Create a producer and its underlying pool from a sample tensor.

**Parameters:**
- `pool_name` (str): Name of the shared memory pool
- `sample` (Any): Sample tensor or array to infer metadata
- `history_len` (int): Number of frames to keep in the pool (default: 1)
- `dds_participant`: Optional DDS participant for notifications
- `keep_last` (int): Number of latest frames to keep in DDS history (default: 10)

**Returns:**
- `TensorProducer`: New producer instance

**Example:**
```python
import numpy as np
from tensor_ipc import TensorProducer

# Create sample data
sample_data = np.random.rand(640, 480, 3).astype(np.float32)

# Create producer from sample
producer = TensorProducer.from_sample(
    pool_name="camera_feed",
    sample=sample_data,
    history_len=5
)
```

#### Instance Methods

##### `put(data, *args, **kwargs)`

Write data to the tensor pool and return the current frame index.

**Parameters:**
- `data` (Any): Tensor or array data to write
- `*args`, `**kwargs`: Additional arguments (for ROS compatibility)

**Returns:**
- `int`: Current frame index

**Raises:**
- `TypeError`: If data backend type doesn't match pool backend type

**Example:**
```python
# Write frame to pool
frame_idx = producer.put(image_data)
print(f"Written frame at index: {frame_idx}")
```

##### `cleanup()`

Clean up producer resources. Called automatically on object deletion.

---

### TensorConsumer

A simplified consumer for tensor data streams from shared memory pools.

#### Constructor

```python
TensorConsumer(
    pool_metadata: PoolMetadata,
    keep_last: int = 10,
    dds_participant: Optional[DomainParticipant] = None,
    on_new_data_callback: Optional[Callable] = None
)
```

**Parameters:**
- `pool_metadata`: Metadata for the shared memory pool
- `keep_last` (int): Number of latest frames to keep in DDS history (default: 10)
- `dds_participant`: Optional DDS participant for notifications
- `on_new_data_callback`: Optional callback called when new data is available

#### Class Methods

##### `from_sample(pool_name, sample, dds_participant=None, history_len=1, keep_last=10, callback=None)`

Create a consumer from a sample tensor/array to infer metadata.

**Parameters:**
- `pool_name` (str): Name of the shared memory pool
- `sample` (Any): Sample tensor or array to infer metadata
- `dds_participant`: Optional DDS participant for notifications
- `history_len` (int): Number of frames to keep in the pool (default: 1)
- `keep_last` (int): Number of latest frames to keep in DDS history (default: 10)
- `callback`: Optional callback for new data notifications

**Returns:**
- `TensorConsumer`: New consumer instance

#### Instance Methods

##### `get(history_len=1, as_numpy=False, latest_first=True)`

Get latest tensor data from the pool.

**Parameters:**
- `history_len` (int): Number of frames to read from the pool (default: 1)
- `as_numpy` (bool): Convert to NumPy array if True (default: False)
- `latest_first` (bool): If True, return tensor with latest frame first (default: True)

**Returns:**
- `Optional[Any]`: Tensor data or None if not available

**Example:**
```python
from tensor_ipc import TensorConsumer

# Create consumer from sample
consumer = TensorConsumer.from_sample(
    pool_name="camera_feed",
    sample=sample_data,
    history_len=5
)

# Read latest frame
latest_frame = consumer.get(history_len=1, as_numpy=True)

# Read last 3 frames
history_data = consumer.get(history_len=3, latest_first=True)
```

##### `cleanup()`

Clean up all resources used by the consumer. Called automatically on object deletion.

---

## Metadata

### PoolMetadata

Metadata for a tensor pool that can be serialized across processes.

#### Fields

- `name` (str): Pool name
- `shape` (List[int]): Tensor shape
- `dtype_str` (str): String representation of data type
- `backend_type` (str): Backend type ("numpy", "torch", "torch_cuda")
- `history_len` (int): Number of frames to keep in history (default: 1)
- `device` (str): Device location ("cpu", "cuda:0", etc.) (default: "cpu")
- `element_size` (int): Size of each element in bytes
- `total_size` (int): Total memory size required
- `shm_name` (str): Shared memory identifier
- `creator_pid` (int): Process ID of creator
- `payload_json` (str): Extra payload for backend-specific data

#### Methods

##### `__eq__(other)`

Compare two PoolMetadata instances for equality. Excludes creator_pid from comparison.

---

### MetadataCreator

Acts as a namespace for metadata creation utilities.

#### Static Methods

##### `from_numpy_sample(name, sample_data, history_len=1)`

Create PoolMetadata from a sample numpy array.

**Parameters:**
- `name` (str): Name of the shared memory pool
- `sample_data` (np.ndarray): Sample numpy array
- `history_len` (int): Number of frames to keep in the pool (default: 1)

**Returns:**
- `PoolMetadata`: Metadata for the shared memory pool

**Raises:**
- `TypeError`: If sample_data is not a numpy array

##### `from_torch_sample(name, sample_data, history_len=1)`

Create PoolMetadata from a sample torch tensor (CPU only).

**Parameters:**
- `name` (str): Name of the shared memory pool
- `sample_data` (torch.Tensor): Sample torch tensor (must be on CPU)
- `history_len` (int): Number of frames to keep in the pool (default: 1)

**Returns:**
- `PoolMetadata`: Metadata for the shared memory pool

**Raises:**
- `TypeError`: If sample_data is not a torch tensor
- `ValueError`: If tensor is not on CPU

##### `from_torch_cuda_sample(name, sample_data, history_len=1, tensor_pool=None)`

Create PoolMetadata from a sample CUDA tensor.

**Parameters:**
- `name` (str): Name of the shared memory pool
- `sample_data` (torch.Tensor): Sample CUDA tensor
- `history_len` (int): Number of frames to keep in the pool (default: 1)
- `tensor_pool`: Optional existing tensor pool for metadata (default: None)

**Returns:**
- `PoolMetadata`: Metadata for the shared memory pool

**Raises:**
- `TypeError`: If sample_data is not a CUDA tensor
- `ValueError`: If tensor is not on CUDA device

##### `from_sample(name, data, history_len, backend)`

Unified method to create PoolMetadata from a sample tensor/array.

**Parameters:**
- `name` (str): Name of the shared memory pool
- `data` (Any): Sample tensor or array
- `history_len` (int): Number of frames to keep in the pool
- `backend` (str): Backend type ("numpy", "torch", "torch_cuda")

**Returns:**
- `PoolMetadata`: Metadata for the shared memory pool

**Raises:**
- `ValueError`: If backend type is not supported

**Example:**
```python
import numpy as np
import torch
from tensor_ipc import MetadataCreator

# NumPy example
np_data = np.random.rand(224, 224, 3).astype(np.float32)
metadata = MetadataCreator.from_numpy_sample(
    name="image_pool",
    sample_data=np_data,
    history_len=10
)

# PyTorch CPU example
torch_data = torch.randn(224, 224, 3)
metadata = MetadataCreator.from_torch_sample(
    name="torch_pool",
    sample_data=torch_data,
    history_len=5
)

# CUDA example (if available)
if torch.cuda.is_available():
    cuda_data = torch.randn(224, 224, 3, device='cuda')
    metadata = MetadataCreator.from_torch_cuda_sample(
        name="cuda_pool",
        sample_data=cuda_data,
        history_len=3
    )

# Unified interface
metadata = MetadataCreator.from_sample(
    name="unified_pool",
    data=np_data,
    history_len=5,
    backend="numpy"
)
```

#### Utility Methods

##### `payload_from_torch_cuda_storage(tensor)`

Extract CUDA IPC information from a CUDA tensor.

**Parameters:**
- `tensor` (torch.Tensor): CUDA tensor

**Returns:**
- `str`: JSON string containing CUDA IPC handle information

##### `verify_torch_cuda_payload(metadata)`

Verify that the payload JSON in metadata contains all required CUDA keys.

**Parameters:**
- `metadata` (PoolMetadata): Metadata to verify

**Returns:**
- `bool`: True if payload is valid, False otherwise

---

## ROS Extensions

The ROS extensions provide seamless integration between tensor-ipc pools and ROS2 topics. These classes require `rclpy` and `ros2_numpy` to be installed.

### ROSTensorProducer

A producer that connects to a ROS topic and publishes tensor data.

#### Constructor

```python
ROSTensorProducer(
    pool_metadata: PoolMetadata,
    node: Node,
    ros_topic: str,
    ros_msg_type: Any,
    qos: int = 10,
    keep_last: int = 10,
    dds_participant: Optional[DomainParticipant] = None
)
```

**Parameters:**
- `pool_metadata`: Metadata for the shared memory pool
- `node` (rclpy.Node): ROS2 node instance
- `ros_topic` (str): ROS topic name to publish to
- `ros_msg_type`: ROS message type (e.g., sensor_msgs.msg.Image)
- `qos` (int): ROS QoS profile setting (default: 10)
- `keep_last` (int): Number of latest frames to keep in DDS history (default: 10)
- `dds_participant`: Optional DDS participant for notifications

#### Methods

##### `put(data, *args, **kwargs)`

Publish tensor data to both the ROS topic AND update the tensor pool.

**Parameters:**
- `data` (np.array | torch.Tensor): The tensor data to publish
- `*args`, `**kwargs`: Additional arguments passed to `ros2_numpy.msgify()`

**Example:**
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from tensor_ipc import ROSTensorProducer, MetadataCreator
import numpy as np

# Initialize ROS
rclpy.init()
node = Node('image_publisher')

# Create sample image data
image_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Create metadata
metadata = MetadataCreator.from_sample(
    name="camera_topic",
    data=image_data,
    history_len=5,
    backend="numpy"
)

# Create ROS producer
producer = ROSTensorProducer(
    pool_metadata=metadata,
    node=node,
    ros_topic="/camera/image_raw",
    ros_msg_type=Image,
    qos=10
)

# Publish data
producer.put(image_data, encoding='rgb8')
```

##### `cleanup()`

Clean up producer resources and destroy ROS publisher.

---

### ROSTensorConsumer

A consumer that subscribes to ROS topics and creates shared memory pools.

#### Constructor

```python
ROSTensorConsumer(
    pool_metadata: PoolMetadata,
    node: Node,
    ros_topic: str,
    ros_msg_type: Any,
    qos: int = 10,
    keep_last: int = 10,
    dds_participant: Optional[DomainParticipant] = None,
    on_new_data_callback: Optional[Callable] = None
)
```

**Parameters:**
- `pool_metadata`: Metadata for the shared memory pool
- `node` (rclpy.Node): ROS2 node instance
- `ros_topic` (str): ROS topic name to subscribe to
- `ros_msg_type`: ROS message type to expect
- `qos` (int): ROS QoS profile setting (default: 10)
- `keep_last` (int): Number of latest frames to keep in DDS history (default: 10)
- `dds_participant`: Optional DDS participant for notifications
- `on_new_data_callback`: Optional callback for new data notifications

#### Methods

##### `get(history_len=1, as_numpy=False, latest_first=True)`

Get tensor data from the pool (wraps TensorConsumer's get method).

**Parameters:**
- `history_len` (int): Number of frames to retrieve (default: 1)
- `as_numpy` (bool): If True, return data as numpy arrays (default: False)
- `latest_first` (bool): If True, return the latest frames first (default: True)

**Returns:**
- `Optional[Any]`: The latest tensor data from the pool, or None if not available

**Example:**
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from tensor_ipc import ROSTensorConsumer, MetadataCreator
import numpy as np

# Initialize ROS
rclpy.init()
node = Node('image_subscriber')

# Create sample for metadata
sample_data = np.zeros((480, 640, 3), dtype=np.uint8)

# Create metadata
metadata = MetadataCreator.from_sample(
    name="camera_topic",
    data=sample_data,
    history_len=5,
    backend="numpy"
)

# Create ROS consumer
consumer = ROSTensorConsumer(
    pool_metadata=metadata,
    node=node,
    ros_topic="/camera/image_raw",
    ros_msg_type=Image,
    qos=10
)

# Read data in a loop
while rclpy.ok():
    rclpy.spin_once(node, timeout_sec=0.1)
    
    # Get latest image
    image_data = consumer.get(as_numpy=True)
    if image_data is not None:
        print(f"Received image: {image_data.shape}")
```

##### `cleanup()`

Cleanup resources and destroy ROS subscription.

---

## Backend Support

The tensor-ipc package supports multiple backends for data storage:

### NumPy Backend
- **Backend Type**: `"numpy"`
- **Device**: Always `"cpu"`
- **Data Types**: All NumPy dtypes
- **Implementation**: Uses `posix_ipc.SharedMemory` for shared memory

### PyTorch CPU Backend
- **Backend Type**: `"torch"`
- **Device**: `"cpu"`
- **Data Types**: All PyTorch dtypes
- **Implementation**: Uses `posix_ipc.SharedMemory` with PyTorch tensor views

### PyTorch CUDA Backend
- **Backend Type**: `"torch_cuda"`
- **Device**: `"cuda:N"` (where N is device index)
- **Data Types**: All PyTorch dtypes supported on CUDA
- **Implementation**: Uses CUDA IPC handles for inter-process GPU memory sharing

**Note**: CUDA backend requires careful setup and may not work across all CUDA versions. Run tests to verify compatibility in your environment.

---

## Error Handling

### Common Exceptions

- **`ValueError`**: Invalid parameters, unsupported backend, topic already published
- **`TypeError`**: Wrong data type, mismatched backend types
- **`ConnectionError`**: Failed to connect to producer, DDS communication issues
- **`AssertionError`**: Metadata validation failures

### Best Practices

1. **Always call cleanup()**: Ensures proper resource cleanup
2. **Handle None returns**: `consumer.get()` returns None when no data is available
3. **Validate data types**: Ensure data matches the expected backend and shape
4. **Set appropriate history_len**: Use the formula: `max_reading_history + ceil(pub_rate/min_sub_rate) + 1`
5. **Test CUDA compatibility**: Run tests before deploying CUDA backend in production

---

## Thread Safety

- **Producer**: Thread-safe for writing operations
- **Consumer**: Thread-safe for reading operations
- **Locking**: Frame-level locking prevents read/write conflicts
- **DDS**: Thread-safe for notifications and metadata exchange

Multiple consumers can read simultaneously from the same pool, but write operations are exclusive per frame.