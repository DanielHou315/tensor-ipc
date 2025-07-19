# Victor-Python-IPC

Efficient, flexible, and high-performance inter-process communication (IPC) for robotics applications, with seamless integration between shared memory and ROS.

## Overview

`Victor-Python-IPC` is a library designed to provide a high-performance communication layer for robotics systems. It enables zero-copy shared memory (SHM) communication for processes running on the same machine and integrates transparently with ROS for distributed network communication.

The core philosophy is to abstract the transport layer, allowing developers to work with `numpy` arrays and `torch` tensors directly, whether the data is coming from a local process or a remote ROS node.

## Key Features

- 🚀 **High-Performance SHM**: Zero-copy data sharing between processes using POSIX shared memory for maximum efficiency.
- 🤖 **Seamless ROS Integration**: A built-in ROS bridge allows streams to subscribe to ROS topics and make the data available locally as if it were produced in shared memory.
- 🧠 **Unified Tensor API**: Work with `numpy.ndarray` and `torch.Tensor` directly, regardless of the data source.
- ⚙️ **Producer/Stream Model**: A simple and powerful producer/consumer pattern (`TensorProducer`/`TensorStream`).
- 📦 **Multi-Process Registry**: A robust, file-system-backed registry allows processes to discover data pools by name.
- 🛡️ **Type and Shape Safety**: Automatically validates data schemas to prevent mismatches between processes.

## Installation

```bash
git clone https://github.com/UM-ARM-Lab/Victor-Python-IPC.git
cd Victor-Python-IPC
pip install -e .
```

For development, which includes testing dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

The library revolves around two core classes: `TensorProducer` to create and publish data, and `TensorStream` to subscribe to it.

### 1. Local Shared Memory Communication

This is the most performant mode for processes running on the same machine.

**Producer Process:**
```python
import numpy as np
import time
from polymorph_ipc import TensorProducer

# 1. Create a producer for a pool named 'my_numpy_pool'
# The pool's shape, dtype, and history length are defined by the sample array.
producer = TensorProducer.from_sample(
    pool_name="my_numpy_pool",
    sample=np.zeros((480, 640, 3), dtype=np.uint8),
    history_len=10
)

# 2. Publish data
for i in range(100):
    # Create a new numpy array with updated data
    image_data = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    producer.publish(image_data)
    print(f"Published frame {i}")
    time.sleep(0.1)

# 3. Clean up resources
producer.cleanup()
```

**Consumer Process:**
```python
import time
from polymorph_ipc import TensorStream

# 1. Create a stream to subscribe to the pool
# It will automatically find the pool registered by the producer.
stream = TensorStream(pool_name="my_numpy_pool")

# 2. Get data
# Get the most recently published tensor (non-blocking)
latest_image = stream.get_last()
if latest_image is not None:
    print(f"Got last image with shape: {latest_image.shape}")

# Wait for the next new tensor to be published (blocking)
print("Waiting for new image...")
new_image = stream.get_new(timeout=1.0)
if new_image is not None:
    print(f"Got new image with shape: {new_image.shape}")

# 3. Clean up resources
stream.cleanup()
```

### 2. Subscribing to a ROS Topic

The `TensorStream` can directly subscribe to a ROS topic. It handles the conversion and creates a local shared memory pool, allowing other local processes to access the ROS data with zero-copy reads.

```python
import rclpy
import numpy as np
from sensor_msgs.msg import Image
from polymorph_ipc import TensorStream

# Standard ROS setup
rclpy.init()
node = rclpy.create_node('my_ros_subscriber')

# 1. Create a stream that bridges a ROS topic to a local SHM pool
# Provide ROS details and a sample tensor to define the pool's structure.
ros_stream = TensorStream(
    pool_name="camera_stream_from_ros",
    ros_node=node,
    ros_topic="/camera/image_raw",
    ros_msg_type=Image,
    ros_sample_data=np.zeros((480, 640, 3), dtype=np.uint8)
)

print("Subscribed to ROS topic. Spinning...")
# The stream handles the subscription and data conversion in the background.
# We just need to spin the ROS node.
rclpy.spin(node)

# 2. In another local process, you can now access this data via SHM:
# local_stream = TensorStream(pool_name="camera_stream_from_ros")
# image = local_stream.get_last()

# 3. Clean up
ros_stream.cleanup()
node.destroy_node()
rclpy.shutdown()
```
