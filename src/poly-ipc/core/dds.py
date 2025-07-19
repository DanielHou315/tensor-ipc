from typing import Optional, Union, Type
from time import perf_counter
from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.core import Listener

from cyclonedds.util import duration
from cyclonedds.qos import Qos, Policy

from metadata import (
    PoolProgressMessage,
    PoolMetadata,
    TorchCUDAPoolMetadata,
    MetadataCreator
)
import time

metadata_qos = Qos(
    Policy.Durability.Transient,
    Policy.History.KeepLast(1),
)

class DDSProducer:
    def __init__(self, 
        participant: DomainParticipant, 
        topic_name: str, 
        metadata_msg: Union[
            PoolMetadata,
            TorchCUDAPoolMetadata,
        ],
        keep_last: int = 10
    ):
        self._dp = participant

        # Topic setup
        # We setup the main topic name to produce metadata once on startup
        # and a progress topic to publish pool progress messages at higher rate
        self._topic_name = topic_name

        _metadata_type = type(metadata_msg)

        self._metadata = metadata_msg
        self._metadata_topic = Topic(participant, f"polyipc_{topic_name}", _metadata_type)
        self._metadata_heartbeat = -1
        self._metadata_writer = DataWriter(participant, self._metadata_topic, metadata_qos)
        # Publish initial metadata
        self._publish_metadata()

        # Define progress topic and worker
        progress_qos = Qos(
            Policy.Durability.TransientLocal,
            Policy.History.KeepLast(keep_last),
        )
        self._progress_topic = Topic(participant, f"polyipc_{topic_name}_progress", PoolProgressMessage)
        self._progress_writer = DataWriter(participant, self._progress_topic, progress_qos)

    def _publish_metadata(self):
        """Publish metadata to the DDS."""
        if perf_counter() - self._metadata_heartbeat > 1.0:
            self._metadata_writer.write(self._metadata)
            self._metadata_heartbeat = perf_counter()

    def publish_progress(self, message: PoolProgressMessage):
        self._publish_metadata()
        self._progress_writer.write(message)

    @property
    def metadata(self):
        return self._metadata

class DDSConsumer:
    def __init__(self, 
        participant: DomainParticipant, 
        topic_name: str, 
        metadata_type: Type[Union[PoolMetadata, TorchCUDAPoolMetadata]],
        keep_last: int = 10, 
        new_data_callback=None,
        connection_lost_callback=None
    ):
        self._dp = participant

        # Topic setup
        # We setup the main topic name to acquire metadata on startup, and repeat on all read attempts until success
        # and a progress topic to read pool progress messages at high rate
        
        self._topic_name = topic_name
        self._metadata_topic = Topic(participant, f"polyipc_{topic_name}", metadata_type)
        self._metadata_reader = DataReader(participant, self._metadata_topic)

        progress_qos = Qos(
            Policy.Durability.TransientLocal,
            Policy.History.KeepLast(keep_last),
        )
        self._progress_listener = Listener(
            on_data_available=new_data_callback,
            on_liveliness_lost=connection_lost_callback
        )
        self._progress_topic = Topic(participant, f"polyipc_{topic_name}_progress", PoolProgressMessage)
        self._progress_reader = DataReader(
            participant,
            self._progress_topic,
            progress_qos,
            listener=self._progress_listener
        )

        # Attempt initial connection, and if fails, subsequent reads will retry
        self._connected = False
        self._metadata = None
        self._connect()

    def _connect(self):
        if self._connected:
            return
        try:
            metadata = self._metadata_reader.take_next()
            if metadata is None:
                return
            self._metadata = metadata
            self._connected = True
        except Exception as e:
            print(f"Error connecting to DDS: {e}")

    def read_next_progress(self, timeout=100):
        # Attempt re-connection if not connected
        if not self._connected:
            self._connect()
            if not self._connected:
                return None
            
        # Try reading from pool, and if fails, attempt to reconnect
        return self._progress_reader.take_one(timeout=duration(milliseconds=timeout))

    def read_latest_progress(self, max_n: int = 1) -> Optional[PoolProgressMessage]:
        """Read progress messages from the DDS.

        Args:
            blocking (bool, optional): Whether to block until a message is available. Defaults to True.
            timeout (float, optional): The maximum time to wait for a message. Defaults to 0.1.

        Returns:
            Optional[PoolProgressMessage]: new progress message if available, otherwise None.
        """
        # Attempt re-connection if not connected
        samples = self.read_all_progress()
        if samples is None or len(samples) == 0:
            return None
        return samples[:max_n]

    def read_all_progress(self, latest_first=True):
        # Attempt re-connection if not connected
        if not self._connected:
            self._connect()
            if not self._connected:
                return None
            
        # Try reading from pool, and if fails, attempt to reconnect
        takes = []
        while True:
            take = self._progress_reader.take(N=10)
            if take is None or len(take) == 0:
                break
            takes.extend(take)
        if latest_first:
            takes.reverse()
        return takes

    @property
    def is_connected(self) -> bool:
        """Check if the consumer is connected to the DDS."""
        return self._connected
    
    @property
    def metadata(self):
        return self._metadata

if __name__ == "__main__":
    # Example usage
    from cyclonedds.domain import DomainParticipant
    import numpy as np
    
    participant = DomainParticipant()

    produce_sample = np.array(np.random.rand(10, 8), dtype=np.float32)
    producer = DDSProducer(participant, "example_pool", MetadataCreator.from_numpy_sample("example_pool", produce_sample, history_len=5))

    def dummy_print(msg):
        print(f"Progress update: ", msg)
    consumer = DDSConsumer(participant, "example_pool", PoolMetadata, callback=dummy_print)

    while True:
        # Publish progress message
        for f in range(5):
            time.sleep(0.1)
            push_sample = PoolProgressMessage(pool_name="example_pool", latest_frame=f)
            producer.publish_progress(push_sample)

        progress_msg = consumer.read_latest_progress(max_n=2)
        print("Read progress message")

        if progress_msg:
            print(f"Received progress: {progress_msg}")
            break
        time.sleep(1)