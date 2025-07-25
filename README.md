# Tensor IPC

High-performance and Flexible IPC for tensor data with seamless ROS integration for robotics research.

## Overview

`tensor-ipc` provides efficient shared memory communication for tensor data between processes, with built-in support for ROS topics. It enables zero-copy data sharing using POSIX shared memory and integrates with ROS for distributed communication and sim/real transfer.

## Key Features

- 🚀 **Zero-Copy Shared Memory**: POSIX shared memory with per-frame locking for safe concurrent access
- 🤖 **ROS Integration**: Built-in ROS producers and consumers with automatic type conversion (custom types are supported through ros2_numpy)
- 🧠 **Multi-Backend Support**: Native support for NumPy arrays and PyTorch tensors (CPU/CUDA)
- 📦 **DDS Notifications**: Real-time notifications and synchronization using CycloneDDS for efficient polling
- 🛡️ **Type Safety**: Automatic validation of tensor shapes, dtypes, and devices
- 🔄 **History Management**: Configurable history buffers with circular indexing

## Installation

```bash
git clone https://github.com/danielhou315/tensor-ipc.git
cd tensor-ipc
pip install -e .
```
- For torch/torch CUDA support, you must install `torch` in the same Python environment. 
- For ROS support, you must install [ros2_numpy](https://github.com/Box-Robotics/ros2_numpy) in the same Python environment.
- Otherwise, only `numpy` backend will be available. 

## Quick Start

Refer to `examples/` to see basic usage. Documentation is coming soon (hopefully). 

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
- **Shared Memory**: Numpy/PyTorch backend uses POSIX shared memory with memory-mapped arrays. Torch CUDA backend uses CUDA API through PyTorch. 
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
- `MetadataCreator.from_sample()`: Unifies creation of metadata from sample

## Requirements

- Python 3.7+
- NumPy
- CycloneDDS (for DDS notifications)
- Optional: PyTorch (for tensor support)
- Optional: ROS 2 + ros2_numpy (for ROS integration)

## License

MIT License

## GenAI 
This library (especially documentation) is partly written by various LLMs. 