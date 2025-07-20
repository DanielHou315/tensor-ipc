# 1. Check if ROS and ros2_numpy are available
ROS_AVAILABLE = True
try:
    import rclpy
    import ros2_numpy
    from ros2_numpy import numpify, msgify
except ImportError:
    ROS_AVAILABLE = False

# 2. If they are available, import pool2ros_producer and pool2ros_consumer
if ROS_AVAILABLE:
    from .pool2ros_producer import Pool2ROSProducer
    from .pool2ros_consumer import Pool2ROSConsumer