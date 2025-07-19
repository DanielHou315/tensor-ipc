"""
Native tensor backends for NumPy and PyTorch with DDS-based notifications.
Each backend creates a shared tensor pool with shape (history_len, *sample_shape) and shared metadata.
"""
from __future__ import annotations
from typing import Optional, Any, Union, Literal, Type
from abc import ABC, abstractmethod
import numpy as np

# from cyclonedds.domain import DomainParticipant
# from .dds import DDSProducer, DDSConsumer
from ..metadata import PoolMetadata, PoolProgressMessage, TorchCUDAPoolMetadata

# History padding strategies
HistoryPadStrategy = Literal["zero", "fill"]


class TensorProducerBackend(ABC):
    """Base class for tensor producer backends that publish data via DDS notifications."""
    
    def __init__(self,
        pool_metadata: Union[PoolMetadata, TorchCUDAPoolMetadata],
        # dds_participant: DomainParticipant,
        history_pad_strategy: HistoryPadStrategy = "zero",
    ):
        self._metadata = pool_metadata
        self.history_pad_strategy = history_pad_strategy

        # Storage for the single tensor pool with shape (history, *sample_shape)
        self._tensor_pool: Optional[Any] = None
        
        # Current frame tracking
        self._current_frame_index = 0
        
        # # DDS producer for notifications
        # self._dds_producer = DDSProducer(
        #     dds_participant, 
        #     pool_metadata.name, 
        #     pool_metadata
        # )
        
        # Initialize the tensor pool
        self._history_intialized = False
        self._init_tensor_pool()

    @abstractmethod
    def _init_tensor_pool(self) -> None:
        """Initialize the shared tensor pool. Must be implemented by subclasses."""
        pass

    def _initialize_history_padding(self, fill=0) -> None:
        """Initialize history padding based on the specified strategy."""
        assert self._tensor_pool is not None, "Tensor pool must be initialized before padding."
        if self.history_pad_strategy == "zero":
            # Fill with zeros for zero-padding
            self._tensor_pool.fill(0)
        elif self.history_pad_strategy == "fill" and not self._history_initialized:
            # Fill with a specific value for fill-padding
            self._tensor_pool.fill(fill)
        else:
            raise Exception(f"Unknown history padding strategy: {self.history_pad_strategy}")
        self._history_initialized = True

    def write(self, data: Any) -> None:
        """Publish data to the current tensor slot and notify consumers."""
        if not self._history_initialized:
            self._initialize_history_padding()
            
        # Write data to the current slot
        self._write_data(data, self._current_frame_index)

        # # Create progress message
        # progress_message = PoolProgressMessage(
        #     pool_name=self._metadata.name,
        #     latest_frame=self._current_frame_index
        # )
        
        # # Publish progress notification
        # self._dds_producer.publish_progress(progress_message)
        
        # Update frame index
        self._current_frame_index = (self._current_frame_index + 1) % self._metadata.history_len

        return self._current_frame_index

    @abstractmethod
    def _write_data(self, data: Any, frame_index: int) -> None:
        """Write data to the tensor pool at specified frame index."""
        pass

    @property
    def metadata(self) -> Union[PoolMetadata, TorchCUDAPoolMetadata]:
        """Get the pool metadata."""
        return self._metadata

    @property
    def current_frame_index(self) -> int:
        """Get the current frame index."""
        return self._current_frame_index

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up backend resources."""
        pass


class TensorConsumerBackend(ABC):
    """Base class for tensor consumer backends that receive DDS notifications."""
    
    def __init__(self,
        pool_name: str, 
        # dds_participant: DomainParticipant, 
        # metadata_type: Type[Union[PoolMetadata, TorchCUDAPoolMetadata]], 
    ):
        self.pool_name = pool_name
       
        # Storage for the single tensor pool with shape (history, *sample_shape)
        self._tensor_pool: Optional[Any] = None
        self._metadata: Optional[Union[PoolMetadata, TorchCUDAPoolMetadata]] = None
        
        # Connection state
        # self._connected = False
        self._last_frame_index = -1
        
        # # DDS consumer for notifications
        # self._dds_consumer = DDSConsumer(
        #     dds_participant, 
        #     pool_name,
        #     metadata_type, 
        #     keep_last=keep_last,
        #     new_data_callback=self._on_new_data,
        #     connection_lost_callback=self._on_connection_lost
        # )
        
        # # Try initial connection
        # self._try_connect()

    # def _try_connect(self) -> bool:
    #     """Attempt to connect and initialize tensor pool if metadata is available."""
    #     if self._connected:
    #         return True
            
    #     if self._dds_consumer.is_connected and self._dds_consumer.metadata:
    #         self._metadata = self._dds_consumer.metadata
    #         self._init_tensor_pool()
    #         self._connected = True
    #         return True
        
    #     return False

    # @abstractmethod
    # def _connect_tensor_pool(self, *args, **kwargs) -> None:
    #     """Connect to the tensor pool and initialize it."""
    #     # This method should be implemented by subclasses to handle specific backend logic
    #     raise NotImplementedError("Subclasses must implement _connect_tensor_pool")

    # @abstractmethod
    # def _on_new_data(self, data: Any) -> None:
    #     """Handle new data received from DDS."""
    #     # This will be called when new progress messages arrive
    #     # Subclasses can override for custom handling
    #     pass

    # @abstractmethod
    # def _on_connection_lost(self) -> None:
    #     """Handle connection loss to the producer."""
    #     self._connected = False
    #     self._metadata = None
    #     self._tensor_pool = None

    @abstractmethod
    def read(self, indices):
        """Read data from the tensor pool at specified indices."""
        # This method should be implemented by subclasses to handle specific read logic
        raise NotImplementedError("Subclasses must implement read method")
    
    @abstractmethod
    def to_numpy(self, data: Any) -> np.ndarray:
        """
        Convert the input data to a NumPy array.
        """
        pass
    
    # def get(self, history_len: int = 1, block: bool = True,
    #         as_numpy: bool = False, timeout: float = 1.0) -> Optional[Any]:
    #     """Unified interface to get tensor data."""
    #     if not isinstance(history_len, int) or history_len <= 0:
    #         raise ValueError(f"history_len must be a positive integer, got {history_len}")

    #     # Try to connect if not connected
    #     if not self._try_connect():
    #         if not block:
    #             return None
    #         # Wait for connection with timeout
    #         import time
    #         start_time = time.time()
    #         while time.time() - start_time < timeout:
    #             if self._try_connect():
    #                 break
    #             time.sleep(0.01)
    #         else:
    #             return None

    #     # If blocking, wait for new data
    #     if block:
    #         progress_msg = self._dds_consumer.read_next_progress(timeout=int(timeout * 1000))
    #         if progress_msg is None:
    #             return None
    #         self._last_frame_index = progress_msg.latest_frame
    #     else:
    #         # Non-blocking: check for latest progress
    #         latest_progress = self._dds_consumer.read_latest_progress(max_n=1)
    #         if latest_progress:
    #             self._last_frame_index = latest_progress[0].latest_frame

    #     # Get the data
    #     if history_len == 1:
    #         result = self._get_single(self._last_frame_index)
    #     else:
    #         result = self._get_history(history_len, self._last_frame_index)
        
    #     # Convert to numpy if requested
    #     if as_numpy and result is not None:
    #         if history_len == 1:
    #             result = self._to_numpy(result)
    #         else:
    #             result = [self._to_numpy(item) for item in result]
        
    #     return result
    
    # @abstractmethod
    # def _get_single(self, frame_index: int) -> Optional[Any]:
    #     """Get single tensor at specified frame index."""
    #     pass
    
    # @abstractmethod
    # def _get_history(self, count: int, latest_frame_index: int) -> Optional[Any]:
    #     """Get multiple history entries efficiently."""
    #     pass

    @property
    def metadata(self) -> Optional[Union[PoolMetadata, TorchCUDAPoolMetadata]]:
        """Get the pool metadata."""
        return self._metadata

    # @property
    # def is_connected(self) -> bool:
    #     """Check if connected to producer."""
    #     return self._connected
    
    def cleanup(self) -> None:
        """Clean up backend resources."""
        pass