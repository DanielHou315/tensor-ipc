from typing import Any, Optional

from rclpy.node import Node
import ros2_numpy

from ..core.metadata import PoolMetadata
from  ..core.producer import TensorProducer
from cyclonedds.domain import DomainParticipant

class ROSTensorProducer:
    """
    A producer that connects to a ROS topic and publishes tensor data.
    
    This class extends the TensorProducer to handle ROS-specific logic.
    """
    def __init__(self, 
        pool_metadata: PoolMetadata,
        node: Node,
        ros_topic: str,
        ros_msg_type: Any,
        qos: int = 10,      # ROS qos settings
        keep_last: int = 10,
        dds_participant: Optional[DomainParticipant] = None
    ):
        self.node = node
        self.ros_msg_type = ros_msg_type
        self._pub = self.node.create_publisher(
            ros_msg_type,
            ros_topic,
            qos_profile=qos
        )
        self.tensor_producer = TensorProducer(
            pool_metadata=pool_metadata,
            keep_last=keep_last,
            dds_participant=dds_participant  # ROS does not use DDS directly
        )

    def connect_tensor_pool(self, pool_metadata) -> None:
        """Connect to the ROS topic and initialize it."""
        # Implement ROS-specific connection logic here
        self._connected = True  # Set to True after successful connection

    def put(self, data, *args, **kwargs) -> None:
        # Convert the data to a ROS message
        data = self.tensor_producer.backend.mixin.to_numpy(data)
        msg = ros2_numpy.msgify(self.ros_msg_type, data, *args, **kwargs)
        self._pub.publish(msg)