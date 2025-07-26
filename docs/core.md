# Tensor IPC Core Documentation

## Features & Design

The `tensor-ipc` package is designed for data ingestion scenarios where one producer produces data at some rate and one or multiple consumers consume the data. A specific use case in robotics research is the transfer of information between an arbitrary simulator and a policy, or between multiple sensor polling processes and the policy. While these programs often run in different environments and may be imposssible to combine, using `tensor-ipc` enables their communications with zero-copy of data and with an easy to use API. 

The core (`producer` and `consumer`) APIs are designed like the Python `mp.Queue` interface, with `put()` method for the producer and `get()` method for the consumer. Unlike the queue, the `get()` method has a few parameters to allow you to get data in different fashion. 

### Metadata

A data pool is described by a `PoolMetadata` object, which includes necessary information that holds the necessary information to create/connect to a pool. A `PoolMetadata` object can either be manually created or created with the `MetadataCreator.from_sample()` method, which creates a `PoolMetadata` object and populate the fields automatically. 

### Backends

Each producer creates a data **pool**, which is an array of memory that it will share with other consumer processes. The producer owns the **pool** and is the only one that can write to the pool. Note that by design, the pool holds tensors of identical shapes, and only tensors of identical shapes can be written to the pool by the `producer.put()` method. 
- A feature required by many robotic applications is the storage of observation history. Instead of implementing this in a custom queue and stack them time and time again, this pool supports configuring a `history_len` of data that is kept in the pool, and can be easily read by the `consumer.get()` method without repeated stacking. In other words, the producer writes to a single **frame** in a multi-frame buffer, whereas the consumer can read one or multiple frames at once. 

Each consumer connects to a pool with read access. It will attempt to connect to the producer pool if it is created. If the producer cannot be connected, the consumer will lazy-intialize and retry when you start reading data. If the pool still cannot be connected at data reading time, `consumer.get()` returns `None`. 

This package supports multiple backends for holding the data pool. Currently, NumPy, Torch (CPU), and Torch (CUDA) is supported. Note that to ensure efficienty of operation and not allow user error by negligence, pools cannot be connected by producer/consumers using different data types. The matching of pool is enforced by the metadata. 

- **numpy**: To use the NumPy backend, specify `backend='numpy'` when required. NumPy pool is implemented with `posix_ipc.SharedMemory`, which allocates a shared memory chunk in RAM. 

- **torch**: To use the Torch (CPU) backend, specify `backend='torch'` when required. PyTorch CPU pool is created in an identical fashion as the NumPy pool. The PyTorch backend is optional, meaning the package will function without a PyTorch install in the current environment, but neither the `torch` nor `torch_cuda` backends can be created. 

- **torch_cuda**: To use the Torch (CUDA) backend, specify `backend='torch_cuda'` when required. PyTorch CUDA pool is created with PyTorch internal API that interacts with CUDA API. Note that unlike the PyTorch official documentation of CUDA tensor sharing, which can only be done with multiprocessing created with `forkserver` or `spawn` method, our method leverages the underlying CUDA IPC handle to achieve data sharing across independent system processes. However, the CUDA IPC handle bytes may differ from system to system and from CUDA version to CUDA version. We do NOT check for the versioning since enforcing CUDA versions often do not make sense. You should validate the data you receive and verify that the CUDA backend indeed works in your particular setup. Running `tests/test_torch_cuda_backend.py` may be helpful. 

### Communication

The communciation between producer and consumers is implemented with `cyclonedds` (DDS for short) for a balance between flexibility and performance. This is inspired by ROS2 systems as my team works on robotic projects. 

**Metadata Matching** Pools are connected by acquiring and validing `PoolMetadata`. A `PoolMetadata` is required to create both a producer and a consumer. When the producer starts up, it regularly sends its metadata information over DDS on topic `tensoripc_{pool_name}_meta`. When the consumer is connecting, it receives this metadata and compares with its own. Once the metadata is validated, the metadata can be used to connect the pools. The DDS system serves two purposes here: both for ensuring the rebuild of a pool with correct metadata, and for notifying the consumer process of a started producer. In some backends (Torch CUDA), the PoolMetadata on the consumer side is not complete, since it requires the CUDA IPC handle information created by the producer. In these cases, the metadata validation does not contain this `payload` section. 

**Progress Update** Once pools are connected, the consumer can receive producer updates through the `tensoripc_{pool_name}_progress` topic and update its internal state. Therefore, whenever users query the consumer with the `get()` method, it will be using the most up to date data. The latency of this DDS system working is below 0.5 ms on a modern Linux-x64 system for any sized tensors. 

**Destroy** Once the producer shuts down, the consumer receives a disconnection message, and the pool is automatically disconnected on the consumer side. This means that the consumer can no longer access any of the pool, including any history, once the producer shuts down. 

**Locking** To ensure read/write safety across threads, Locks are implemented through `posix_ipc.Semaphore`. This lock works on a per-frame level, so that the writer only locks the specific frame that it writes to, and the reader only locks the frames that it reads. Multiple readers can simultaneously read a frame, but when the writer is writing a frame, no read operation can be done on that frame. 

- **Best Practice**: to minimize the impact of locking on reading speeds, it is recommended to set the `history_len` to be slightly larger than the maximum history length that you want to read across all consumers. This is to leave enough caching frames for the producer to write to without blocking or being blocked by another consumer process. A simple formulat for setting `history_len` is 
$$
    \text{history len} = \text{max reading history len} + ceil(\text{pub rate} / \text{min sub rate}) + 1
$$

### ROS

Besides sharing tensors, we provide an **optional** ROS extension that can convert pool tensors to/from ROS messages. This provides a (more or less) seemless integration of robotic policies trained/validated with simulators and physical robotic systems using ROS. This extension is particularly useful when you want to deploy a policy trained with PyTorch on a physical robot. The ROS extended classes follow identical API structure as the core producer and consumer classes, so you don't have to switch radically between APIs for your code interacting with different systems. 

To use the ROS2 extension part, you must have `rclpy` and `ros2_numpy` installed in your python environment. Usually this means installing this package with system pip. We use the `ros2_numpy.numpify` and `ros2_numpy.msgify` functions extensively to convert tensors to/from ROS messages. You may define your own convesion functions for your custom types using the `ros2_numpy` registration features. See API documentation for more details. 
