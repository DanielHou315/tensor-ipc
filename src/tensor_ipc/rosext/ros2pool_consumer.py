from typing import Any, Optional

from rclpy.node import Node
import ros2_numpy

from ..core.metadata import PoolMetadata
from  ..core.producer import TensorProducer
from ..core.consumer import TensorConsumer
from cyclonedds.domain import DomainParticipant

class ROSTensorConsumer:
    """A simplified consumer for tensor data streams from shared memory pools."""
    def __init__(self,
        pool_metadata: PoolMetadata,
        node: Node, 
        ros_topic: str,
        ros_msg_type: Any,
        qos: int = 10,      # ROS qos settings
        keep_last: int = 10,
        dds_participant: Optional[DomainParticipant] = None,
        on_new_data_callback = None
    ):
        assert isinstance(pool_metadata, PoolMetadata), "pool_metadata must be an instance of PoolMetadata"
        assert isinstance(node, Node), "node must be an instance of rclpy.Node"
        self.node = node
        self._sub = self.node.create_subscription(
            ros_msg_type,
            ros_topic,
            self._on_new_data,
            qos_profile=qos
        )
        self.tensor_producer = TensorProducer(
            pool_metadata=pool_metadata,
            keep_last=keep_last,
            dds_participant=dds_participant,
        )
        self.tensor_consumer = TensorConsumer(
            pool_metadata=pool_metadata,
            keep_last=keep_last,
            dds_participant=dds_participant,
            on_new_data_callback=on_new_data_callback
        )

    def _on_new_data(self, msg: Any) -> None:
        """
        Callback for new ROS messages.
        Converts the ROS message to a numpy array and publishes it to the tensor pool.
        """
        # Convert ROS message to numpy array
        data = ros2_numpy.numpify(msg)
        data = self.tensor_producer.backend.mixin.from_numpy(data)
        # Publish to tensor pool
        self.tensor_producer.put(data)

    def get(self, 
        history_len: int = 1,
        as_numpy: bool = False,
        latest_first: bool = True
    ): 
        """
        Get tensor data from the pool, which just wraps around the TensorConsumer's get method.
        """
        return self.tensor_consumer.get(
            history_len=history_len,
            as_numpy=as_numpy,
            latest_first=latest_first
        )