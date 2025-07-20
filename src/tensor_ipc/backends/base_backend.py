"""
Native tensor backends for NumPy and PyTorch with DDS-based notifications.
Each backend creates a shared tensor pool with shape (history_len, *sample_shape) and shared metadata.
"""
from __future__ import annotations
from typing import Optional, Any, Union, Literal
from abc import ABC, abstractmethod
from ..core.metadata import PoolMetadata, TorchCUDAPoolMetadata

# History padding strategies
HistoryPadStrategy = Literal["zero", "fill"]

class TensorProducerBackend(ABC):
    """Base class for tensor producer backends that publish data via DDS notifications."""
    def __init__(self,
        pool_metadata: Union[PoolMetadata, TorchCUDAPoolMetadata],
        history_pad_strategy: HistoryPadStrategy = "zero",
        force: bool = False,
    ):
        self._pool_metadata = pool_metadata

        # Check and set history strategy
        assert history_pad_strategy in ["zero", "fill"], \
            f"Invalid history_pad_strategy: {history_pad_strategy}. Must be 'zero' or 'fill'."
        self._history_pad_strategy = history_pad_strategy

        # Storage for the single tensor pool with shape (history, *sample_shape)
        self._tensor_pool: Optional[Any] = None
        
        # Current frame tracking
        self._current_frame_index = -1
        
        # Initialize the tensor pool
        self._history_initialized = False
        self._init_tensor_pool(force=force)
        if self._history_pad_strategy == "zero":
            self._initialize_history_padding(fill=0)

    @abstractmethod
    def _init_tensor_pool(self, force=False) -> None:
        """Initialize the shared tensor pool. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _init_tensor_pool")

    @abstractmethod
    def _initialize_history_padding(self, fill=0) -> None:
        """Initialize history padding based on the specified strategy."""
        raise NotImplementedError("Subclasses must implement _initialize_history_padding")

    def write(self, data: Any) -> int:
        """Publish data to the current tensor slot and notify consumers."""
        # Update frame index
        self._current_frame_index = (self._current_frame_index + 1) % self._pool_metadata.history_len

        # Write data to the current slot
        self._write_data(data, self._current_frame_index)

        # Pad history if strategy is "fill"
        if not self._history_initialized and self._history_pad_strategy == "fill":
            self._initialize_history_padding()

        return self._current_frame_index

    @abstractmethod
    def _write_data(self, data: Any, frame_index: int) -> None:
        """Write data to the tensor pool at specified frame index."""
        pass

    @property
    def metadata(self) -> Union[PoolMetadata, TorchCUDAPoolMetadata]:
        """Get the pool metadata."""
        return self._pool_metadata

    @property
    def current_frame_index(self) -> int:
        """Get the current frame index."""
        return self._current_frame_index
    
    @property
    def max_history_len(self) -> int:
        """Get the maximum history length."""
        return self._pool_metadata.history_len

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up backend resources."""
        pass

class TensorConsumerBackend(ABC):
    """Base class for tensor consumer backends that receive DDS notifications."""
    
    def __init__(self,
        metadata: Union[PoolMetadata, TorchCUDAPoolMetadata], 
    ):
        self._pool_metadata = metadata

        # Storage for the single tensor pool with shape (history, *sample_shape)
        self._tensor_pool: Optional[Any] = None

        # Connection state
        self._connected = False
        self._latest_data_frame_index = -1

        # Try initial connection
        try:
            self._connect_tensor_pool(self._pool_metadata)
        except Exception as e:
            print(f"Waiting for producer '{self._pool_metadata.name}' to startup: {e}")
            self._connected = False

    @abstractmethod
    def _connect_tensor_pool(self, pool_metadata) -> None:
        """Connect to the tensor pool and initialize it."""
        # This method should be implemented by subclasses to handle specific backend logic
        raise NotImplementedError("Subclasses must implement _connect_tensor_pool")

    def read(self, indices, as_numpy=False):
        """
        Read data from the tensor pool at specified indices.
        
        - Return None if not connected.
        - Convert to NumPy array if as_numpy is True.
        """
        if not self._connected:
            return None
        data = self._read_indices(indices)
        if as_numpy:
           return self._to_numpy(data)
        return data
    
    @abstractmethod
    def _read_indices(self, indices):
        """Read data from the tensor pool at specified indices."""
        # This method should be implemented by subclasses to handle specific read logic
        raise NotImplementedError("Subclasses must implement _read_indices method")

    @abstractmethod
    def _to_numpy(self, data):
        """Convert data to NumPy array if necessary."""
        # This method should be implemented by subclasses to handle specific conversion logic
        raise NotImplementedError("Subclasses must implement _to_numpy method")
    
    def update_frame_index(self, latest_index: int) -> None:
        """
        Update the last frame index to the latest received index.
        Called by Consumer class
        """
        self._latest_data_frame_index = latest_index
    
    @property
    def metadata(self) -> Optional[Union[PoolMetadata, TorchCUDAPoolMetadata]]:
        """Get the pool metadata."""
        return self._pool_metadata

    @property
    def is_connected(self) -> bool:
        """Check if connected to producer."""
        return self._connected
    
    @property
    def max_history_len(self) -> int:
        """Get the maximum history length."""
        return self._pool_metadata.history_len
    
    @property
    def current_latest_index(self) -> int:
        """Get the latest frame index."""
        return self._latest_data_frame_index
    
    def cleanup(self) -> None:
        """Clean up backend resources."""
        pass