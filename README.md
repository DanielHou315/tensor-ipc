# Tensor IPC

High-performance inter-process communication for tensor data with seamless ROS integration.

## Overview

`tensor-ipc` provides efficient shared memory communication for tensor data between processes, with built-in support for ROS topics. It enables zero-copy data sharing using POSIX shared memory and integrates with ROS for distributed communication.

## Key Features

- 🚀 **Zero-Copy Shared Memory**: POSIX shared memory with per-frame locking for safe concurrent access
- 🤖 **ROS Integration**: Built-in ROS producers and consumers with automatic type conversion
- 🧠 **Multi-Backend Support**: Native support for NumPy arrays and PyTorch tensors (CPU/CUDA)
- 📦 **DDS Notifications**: Real-time notifications using CycloneDDS for efficient polling
- 🛡️ **Type Safety**: Automatic validation of tensor shapes, dtypes, and devices
- 🔄 **History Management**: Configurable history buffers with circular indexing

## Installation

```bash
git clone https://github.com/your-org/tensor-ipc.git
cd tensor-ipc
pip install -e .
```

For ROS support:
```bash
pip install -e ".[ros]"
```

## Quick Start

### 1. Basic Shared Memory Communication

**Producer:**
```python
import numpy as np
import time
from tensor_ipc.core.producer import TensorProducer

# Create producer from sample data
producer = TensorProducer.from_sample(
    pool_name="camera_feed",
    sample=np.zeros((480, 640, 3), dtype=np.uint8),
    history_len=10
)

# Publish data
for i in range(100):
    image_data = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    frame_idx = producer.put(image_data)
    print(f"Published frame {frame_idx}")
    time.sleep(0.1)

producer.cleanup()
```

**Consumer:**
```python
from tensor_ipc.core.consumer import TensorConsumer
from tensor_ipc.core.metadata import MetadataCreator
import numpy as np

# Create consumer with expected metadata
sample = np.zeros((480, 640, 3), dtype=np.uint8)
metadata = MetadataCreator.from_numpy_sample("camera_feed", sample, history_len=10)
consumer = TensorConsumer(metadata)

# Get latest frame
latest_frame = consumer.get(history_len=1, as_numpy=True)
if latest_frame is not None:
    print(f"Received frame with shape: {latest_frame.shape}")

# Get multiple frames with history
history = consumer.get(history_len=5, latest_first=True)
if history is not None:
    print(f"Received {len(history)} frames")

consumer.cleanup()
```

### 2. PyTorch Support

```python
import torch
from tensor_ipc.core.producer import TensorProducer
from tensor_ipc.core.consumer import TensorConsumer

# Producer with PyTorch tensors
sample_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
producer = TensorProducer.from_sample("torch_pool", sample_tensor)

# Publish tensor data
data = torch.randn(3, 224, 224, dtype=torch.float32)
producer.put(data)

# Consumer automatically handles PyTorch tensors
from tensor_ipc.core.metadata import MetadataCreator
metadata = MetadataCreator.from_torch_sample("torch_pool", sample_tensor)
consumer = TensorConsumer(metadata)

result = consumer.get(as_numpy=False)  # Returns PyTorch tensor
```

### 3. ROS Integration

**ROS to Shared Memory Bridge:**
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
from tensor_ipc.rosext import ROSTensorConsumer
from tensor_ipc.core.metadata import MetadataCreator

rclpy.init()
node = Node('tensor_bridge')

# Create metadata for expected image format
sample_image = np.zeros((480, 640, 3), dtype=np.uint8)
metadata = MetadataCreator.from_numpy_sample("ros_camera", sample_image)

# Bridge ROS topic to shared memory
ros_consumer = ROSTensorConsumer(
    pool_metadata=metadata,
    node=node,
    ros_topic="/camera/image_raw",
    ros_msg_type=Image,
    on_new_data_callback=lambda data: print(f"New image: {data.shape}")
)

rclpy.spin(node)
```

**Shared Memory to ROS Bridge:**
```python
from tensor_ipc.rosext import ROSTensorProducer
from sensor_msgs.msg import Image

# Publish shared memory data to ROS
ros_producer = ROSTensorProducer(
    pool_metadata=metadata,
    node=node,
    ros_topic="/processed_images",
    ros_msg_type=Image
)

# Data from shared memory gets published to ROS
data = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
ros_producer.put(data)
```

## Advanced Features

### CUDA Support

```python
import torch
from tensor_ipc.core.producer import TensorProducer

# CUDA tensors with IPC sharing
if torch.cuda.is_available():
    cuda_tensor = torch.zeros(3, 224, 224, device='cuda:0')
    producer = TensorProducer.from_sample("cuda_pool", cuda_tensor)
    
    # Publish CUDA tensor directly
    gpu_data = torch.randn(3, 224, 224, device='cuda:0')
    producer.put(gpu_data)
```

### Callbacks and Notifications

```python
def on_new_data(data):
    print(f"Callback triggered with data shape: {data.shape}")

consumer = TensorConsumer(
    metadata,
    on_new_data_callback=on_new_data
)

# Callback will be triggered when new data arrives
```

### History Management

```python
# Get last 5 frames in chronological order
history = consumer.get(history_len=5, latest_first=False)

# Get last 3 frames with latest first
recent = consumer.get(history_len=3, latest_first=True)
```

## Architecture

- **Backends**: Pluggable backends for NumPy, PyTorch CPU, and PyTorch CUDA
- **Shared Memory**: POSIX shared memory with memory-mapped arrays
- **Locking**: Per-frame reader-writer locks for safe concurrent access
- **Notifications**: CycloneDDS for real-time progress updates
- **ROS Bridge**: Automatic conversion between ROS messages and tensor data

## API Reference

### Core Classes

- `TensorProducer`: Creates and publishes to shared memory pools
- `TensorConsumer`: Subscribes to and reads from shared memory pools
- `PoolMetadata`: Describes pool structure and properties

### ROS Extensions

- `ROSTensorProducer`: Publishes shared memory data to ROS topics
- `ROSTensorConsumer`: Subscribes to ROS topics and creates shared memory pools

### Metadata Creation

- `MetadataCreator.from_numpy_sample()`: Create metadata from NumPy arrays
- `MetadataCreator.from_torch_sample()`: Create metadata from PyTorch tensors
- `MetadataCreator.from_torch_cuda_sample()`: Create metadata for CUDA tensors

## Requirements

- Python 3.8+
- NumPy
- CycloneDX (for DDS notifications)
- Optional: PyTorch (for tensor support)
- Optional: ROS 2 + ros2_numpy (for ROS integration)

## License

MIT License
