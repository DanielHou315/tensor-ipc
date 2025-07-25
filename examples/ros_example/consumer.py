import os
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage

from tensor_ipc import ROSTensorConsumer, MetadataCreator
from PIL import Image
import matplotlib.pyplot as plt

# Open the image file
current_dir = os.path.dirname(os.path.abspath(__file__))
img = Image.open(os.path.join(current_dir, "cat.png"))
image_data = np.array(img)[...,:3]
sample = np.zeros_like(image_data)  # Example sample data
print(f"Image shape: {sample.shape}, dtype: {sample.dtype}")

# Create metadata and node
metadata = MetadataCreator.from_sample(
    name="camera_example_ros",
    data=sample,
    history_len=5,
    backend="numpy"  # Use numpy for simpler setup
)

if not rclpy.ok():
    rclpy.init()
node = Node("camera_example_ros_consumer")

consumer = ROSTensorConsumer(
    pool_metadata=metadata,
    node=node,
    ros_topic="camera_example_ros",
    ros_msg_type=ROSImage,
)

if __name__ == "__main__":
    try:
        while True:
            # Spin once to process ROS callbacks
            rclpy.spin_once(node, timeout_sec=0.1)
            
            # Try to get image data
            image_frame = consumer.get(as_numpy=True)
            if image_frame is not None:
                print(f"Received frame with shape: {image_frame.shape}, dtype: {image_frame.dtype}")
                if len(image_frame.shape) == 4:
                    image_frame = image_frame[0]
                plt.imshow(image_frame)
                plt.title("Received Image from ROS Topic")
                plt.show()
                image_displayed = True
                print("Image displayed successfully!")
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        consumer.cleanup()          # Remember to cleanup! Otherwise there could be a memory leak. 
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Consumer stopped.")