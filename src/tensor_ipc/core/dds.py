from typing import Optional, Union, Type, Callable, List

from time import perf_counter
from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.core import Listener

from cyclonedds.util import duration
from cyclonedds.qos import Qos, Policy

from .metadata import (
    PoolProgressMessage,
    PoolMetadata,
)

metadata_qos = Qos(
    Policy.Durability.Transient,
    Policy.History.KeepLast(1)
)
PROC_DDS_PARTICIPANT = DomainParticipant()

def is_topic_published(topic_name, domain_id: int = 0) -> bool:
    """
    Check if a topic is already published in the DDS
    This is done by creating a subscription and checking if there are any matched publications.

    Note: This is a SLOW operation, so it should not be used in performance-critical paths.
    """
    # topic = Topic(PROC_DDS_PARTICIPANT, topic_name, PoolMetadata)
    # sub = DataReader(PROC_DDS_PARTICIPANT, topic)
    # print("Getting publishers for topic:", topic_name)
    # pubs = sub.get_publisher()
    # print(pubs)

    # The Python API is broken and subject to change, so we use the terminal as a workaround
    import subprocess
    cmd = f"ddsls --topic dcpspublication | grep -c 'topic_name: {topic_name}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    num_pubs = int(result.stdout.strip())
    return num_pubs > 0

class DDSProducer:
    def __init__(self, 
        topic_name: str, 
        metadata_msg: PoolMetadata,
        dds_participant: DomainParticipant|None=None,
        keep_last: int = 10
    ):
        if dds_participant is None:
            dds_participant = PROC_DDS_PARTICIPANT
        self._dp = dds_participant

        # Topic setup
        # We setup the main topic name to produce metadata once on startup
        # and a progress topic to publish pool progress messages at higher rate
        self._topic_name = topic_name

        _metadata_type = type(metadata_msg)

        self._metadata = metadata_msg
        self._metadata_topic = Topic(self._dp, f"tensoripc_{topic_name}_meta", _metadata_type)
        self._metadata_heartbeat = -1
        self._metadata_writer = DataWriter(self._dp, self._metadata_topic, metadata_qos)
        # Publish initial metadata
        self._publish_metadata()

        # Define progress topic and worker
        progress_qos = Qos(
            Policy.Durability.TransientLocal,
            Policy.History.KeepLast(keep_last),
        )
        self._progress_topic = Topic(self._dp, f"tensoripc_{topic_name}_progress", PoolProgressMessage)
        self._progress_writer = DataWriter(self._dp, self._progress_topic, progress_qos)

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
        topic_name: str, 
        metadata_type: Type[PoolMetadata],
        keep_last: int = 10, 
        dds_participant: DomainParticipant|None = None,
        new_data_callback=None,
        connection_lost_callback=None
    ):
        if dds_participant is None:
            dds_participant = PROC_DDS_PARTICIPANT
        self._dp = dds_participant

        # Topic setup
        # We setup the main topic name to acquire metadata on startup, and repeat on all read attempts until success
        # and a progress topic to read pool progress messages at high rate
        
        self._topic_name = topic_name
        self._metadata_topic = Topic(self._dp, f"tensoripc_{topic_name}_meta", metadata_type)
        self._metadata_reader = DataReader(self._dp, self._metadata_topic)

        progress_qos = Qos(
            Policy.Durability.TransientLocal,
            Policy.History.KeepLast(keep_last),
        )

        callback_kwargs = {}
        if isinstance(new_data_callback, Callable):
            callback_kwargs['on_data_available'] = new_data_callback
        if isinstance(connection_lost_callback, Callable):
            callback_kwargs['on_liveliness_lost'] = connection_lost_callback

        self._progress_listener = Listener(**callback_kwargs)
        self._progress_topic = Topic(self._dp, f"tensoripc_{topic_name}_progress", PoolProgressMessage)
        self._progress_reader = DataReader(
            self._dp,
            self._progress_topic,
            progress_qos,
            listener=self._progress_listener
        )

        # Attempt initial connection, and if fails, subsequent reads will retry
        self._connected = False
        self._metadata = None
        self.connect()

    def connect(self):
        if self._connected:
            return True
        try:
            metadata = self._metadata_reader.take_one(timeout=duration(milliseconds=50))
            if metadata is None:
                print(f"No metadata available for topic '{self._topic_name}'")
                return True
            self._metadata = metadata
            self._connected = True
        except Exception as e:
            print(f"Error connecting to DDS: {e}")
        return self._connected

    def read_next_progress(self, timeout=100):
        # Attempt re-connection if not connected
        if not self._connected:
            self.connect()
            if not self._connected:
                return None
            
        # Try reading from pool, and if fails, attempt to reconnect
        return self._progress_reader.take_one(timeout=duration(milliseconds=timeout))

    def read_latest_progress(self, max_n: int = 1) -> Optional[List[PoolProgressMessage]]:
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
            self.connect()
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