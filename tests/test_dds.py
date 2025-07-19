# Example usage
from cyclonedds.domain import DomainParticipant
import numpy as np
import time

# Import tensoripc source code
import sys
sys.path.append("src/")
from tensor_ipc.core.dds import DDSProducer, DDSConsumer
from tensor_ipc.metadata import PoolMetadata, PoolProgressMessage, MetadataCreator


def test_dds():
    participant = DomainParticipant()

    produce_sample = np.array(np.random.rand(10, 8), dtype=np.float32)
    producer = DDSProducer(participant, "example_pool", MetadataCreator.from_numpy_sample("example_pool", produce_sample, history_len=5))

    def dummy_print(reader):
        print(f"Progress update: ", reader.read(N=10)[-1])


    consumer = DDSConsumer(participant, "example_pool", PoolMetadata, new_data_callback=dummy_print)

    for iter in range(5):
        # Publish progress message
        for f in range(5):
            time.sleep(0.1)
            push_sample = PoolProgressMessage(pool_name="example_pool", latest_frame=f)
            producer.publish_progress(push_sample)

        progress_msg = consumer.read_latest_progress(max_n=2)
        if not progress_msg:
            continue
        
        assert len(progress_msg) == 2, "Expected 2 progress messages"
        assert isinstance(progress_msg, list)
        assert isinstance(progress_msg[0], PoolProgressMessage), "Expected PoolProgressMessage type"
        assert progress_msg[0].pool_name == "example_pool", "Pool name mismatch"
        assert progress_msg[0].latest_frame == 4, "Latest frame mismatch"
        assert progress_msg[1].latest_frame == 3, "2nd-latest frame mismatch"
        time.sleep(1)
    return