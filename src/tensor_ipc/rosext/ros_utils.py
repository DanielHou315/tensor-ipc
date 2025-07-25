"""
Built-in typing conversions and helpers for ROS messages using ros2_numpy.
Provides validation for ROS message conversions to ensure they work correctly.
"""
from typing import Any, Type
import numpy as np

# Get ros2_numpy if available
import ros2_numpy

class ROSConversionError(Exception):
    """Raised when ROS message conversion fails."""
    pass

def validate_ros_msgify(ros_msg_type: Type[Any], sample_data: np.ndarray) -> bool:
    """
    Validate that a ROS message type can be converted to numpy arrays.
    """
    try:
        ros_msg = ros2_numpy.msgify(ros_msg_type, sample_data)
        return True
    except Exception as e:
        return False
    
def validate_ros_numpify(ros_msg: Any) -> bool:
    """
    Validate that a ROS message type can be converted from numpy arrays.
    """
    try:
        converted_back = ros2_numpy.numpify(ros_msg)
        return True
    except Exception as e:
        return False
