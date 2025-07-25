import numpy as np
import time
import os
from tensor_ipc import TensorProducer
from PIL import Image

# Open the image file
current_dir = os.path.dirname(os.path.abspath(__file__))
img = Image.open(os.path.join(current_dir, "ghost.jpg"))
image_data = np.array(img) 
print(f"Image shape: {image_data.shape}, dtype: {image_data.dtype}")

# Create producer from sample data
producer = TensorProducer.from_sample(
    pool_name="camera_example",
    sample=image_data,
    history_len=5
)

if __name__ == "__main__":
    try:
        print("Publishing images... Press Ctrl+C to stop.")
        while True:
            frame_idx = producer.put(image_data)
            time.sleep(0.1) # Simulate camera frame rate
    except KeyboardInterrupt:
        pass
    finally:
        producer.cleanup()
        print("Producer stopped.")