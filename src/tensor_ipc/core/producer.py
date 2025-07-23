from __future__ import annotations
from typing import Any

from ..backends import (
    create_producer_backend,
    detect_backend_from_data
)
from .metadata import (
    PoolMetadata,
    PoolProgressMessage,
    MetadataCreator
)
from .dds import (
    DDSProducer,
    is_topic_published
)

class TensorProducer:
    """Client for writing tensors to a shared pool with notification support."""
    def __init__(self, 
        pool_metadata: PoolMetadata,
        dds_participant:Any = None,  # Optional DDS participant for notifications
        keep_last: int = 10
    ):
        # Ensure unique pub
        if is_topic_published(pool_metadata.name):
            raise ValueError(f"Topic '{pool_metadata.name}' is already published. Use a unique name.")

        # Create the appropriate backend as producer
        self.backend = create_producer_backend(
            pool_metadata=pool_metadata,
            force=True,
        )

        # self._pool_metadata = self.backend.metadata
        self._pool_metadata = self.backend.metadata

        # DDS producer for notifications
        self._dds_producer = DDSProducer(
            self._pool_metadata.name, 
            self._pool_metadata,
            dds_participant=dds_participant,
            keep_last=keep_last
        )
        
        # Initialize cleanup flag
        self._cleaned_up = False

    @classmethod
    def from_sample(cls,
        pool_name: str,
        sample: Any,
        history_len: int = 1,
        dds_participant: Any = None,
        keep_last: int = 10
    ) -> "TensorProducer":
        """Create a producer and its underlying pool from a sample tensor."""
        # Detect backend type from sample
        backend_type = detect_backend_from_data(sample)
        pool_metadata = MetadataCreator.from_sample(
            name=pool_name,
            data=sample,
            history_len=history_len,
            backend=backend_type
        )
        return cls(
            pool_metadata,
            dds_participant=dds_participant,
            keep_last=keep_last
        )

    def put(self, data: Any) -> int:
        """
        Write data to the tensor pool and return the current frame index.
        """
        # Validate data matches expected shape and type
        if not detect_backend_from_data(data) == self._pool_metadata.backend_type:
            raise TypeError(f"Data backend type {detect_backend_from_data(data)} does not match pool backend type {self._pool_metadata.backend_type}")
        
        # Write data using the backend's write method
        idx = self.backend.write(data)
        
        # Publish progress notification
        message = PoolProgressMessage(
            pool_name=self._pool_metadata.name,
            latest_frame=idx
        )
        self._dds_producer.publish_progress(message)
        return idx

    def __del__(self):
        """Automatic cleanup on object deletion."""
        self.cleanup()

    def cleanup(self):
        """Clean up producer resources."""
        if self._cleaned_up:
            return
        try:
            self.backend.cleanup()
        except Exception:
            pass  # Ignore cleanup errors
        finally:
            self._cleaned_up = True


