import os
import numpy as np
import time
from tensor_ipc import TensorConsumer
from PIL import Image
import matplotlib.pyplot as plt

# Open the image file
current_dir = os.path.dirname(os.path.abspath(__file__))
img = Image.open(os.path.join(current_dir, "ghost.jpg"))
image_data = np.array(img)
sample = np.zeros_like(image_data)  # Example sample data
print(f"Image shape: {sample.shape}, dtype: {sample.dtype}")

# Create producer from sample data
consumer = TensorConsumer.from_sample(
    pool_name="camera_example",
    sample=sample,
    history_len=5
)

if __name__ == "__main__":
    try:
        print("Waiting for image feed...")
        while True:
            image_frame = consumer.get(as_numpy=True)
            if image_frame is not None:
                print(f"Received frame with shape: {image_frame.shape}, dtype: {image_frame.dtype}")
                if len(image_frame.shape) == 4:
                    image_frame = image_frame[0]
                plt.imshow(image_frame)
                plt.show()
                time.sleep(1)
                break
            time.sleep(0.1) # Simulate camera frame rate
    except KeyboardInterrupt:
        pass
    finally:
        consumer.cleanup()
        print("Consumer stopped.")