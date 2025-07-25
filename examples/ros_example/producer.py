import numpy as np
import time
import os
from tensor_ipc import ROSTensorProducer, MetadataCreator
from PIL import Image

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage

# Open the image file
current_dir = os.path.dirname(os.path.abspath(__file__))
img = Image.open(os.path.join(current_dir, "cat.png"))
image_data = np.array(img)[...,:3]  # 4-channel to 3-channel conversion
print(f"Image shape: {image_data.shape}, dtype: {image_data.dtype}")

# Create producer from sample data
metadata = MetadataCreator.from_sample(
    name="camera_example_ros",
    data=image_data,
    history_len=5,
    backend="numpy"  # Use numpy for simpler setup
)

if not rclpy.ok():
    rclpy.init()
node = Node("camera_example_ros_consumer")

producer = ROSTensorProducer(
    metadata,
    node=node,
    ros_topic="camera_example_ros",
    ros_msg_type=ROSImage,
    qos=10,  # ROS QoS settings
)

if __name__ == "__main__":
    print("Publishing images... Press Ctrl+C to stop.")
    try:
        while True:
            frame_idx = producer.put(image_data, encoding="rgb8")
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.1) # Simulate camera frame rate
    except KeyboardInterrupt:
        pass
    finally:
        producer.cleanup()          # Remember to cleanup! Otherwise there could be a memory leak. 
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Producer stopped.")