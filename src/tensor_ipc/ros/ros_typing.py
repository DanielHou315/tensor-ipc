"""
Built-in typing conversions and helpers for ROS messages using ros2_numpy.
Provides validation for ROS message conversions to ensure they work correctly.
"""
from typing import Any, Type
import numpy as np
from ..utils import get_ros2_numpy

# Get ros2_numpy if available
ros2_numpy_info = get_ros2_numpy()
ROS_AVAILABLE = ros2_numpy_info is not None

if ROS_AVAILABLE:
    ros2_numpy = ros2_numpy_info['module']
    numpify = ros2_numpy_info['numpify']
    msgify = ros2_numpy_info['msgify']
else:
    ros2_numpy = None
    numpify = None
    msgify = None

class ROSConversionError(Exception):
    """Raised when ROS message conversion fails."""
    pass

def validate_ros_conversion(ros_msg_type: Type[Any], sample_data: np.ndarray) -> None:
    """
    Validate that a ROS message type can be converted to/from numpy arrays.
    
    This function tests both directions:
    1. numpy -> ROS message (msgify)
    2. ROS message -> numpy (numpify)
    
    Args:
        ros_msg_type: The ROS message class to test
        sample_data: A sample numpy array to test conversion with
        
    Raises:
        ROSConversionError: If conversion is not supported for this message type
        ImportError: If ros2_numpy is not available
    """
    if not ROS_AVAILABLE or msgify is None or numpify is None:
        raise ImportError("ros2_numpy is required for ROS message conversion. Please install it with: pip install ros2-numpy")
    
    try:
        # Test numpy -> ROS message conversion
        ros_msg = msgify(ros_msg_type, sample_data)
        
        # Test ROS message -> numpy conversion  
        converted_back = numpify(ros_msg)
        
        # Verify the round-trip conversion preserves basic properties
        if converted_back.shape != sample_data.shape:
            raise ROSConversionError(
                f"Round-trip conversion failed: shape changed from {sample_data.shape} to {converted_back.shape}"
            )
            
        if converted_back.dtype != sample_data.dtype:
            # Allow some dtype flexibility, but warn about significant changes
            if not np.can_cast(converted_back.dtype, sample_data.dtype, casting='safe'):
                raise ROSConversionError(
                    f"Round-trip conversion failed: dtype changed from {sample_data.dtype} to {converted_back.dtype} "
                    f"and conversion is not safe"
                )
        
    except Exception as e:
        if isinstance(e, ROSConversionError):
            raise
        
        # Convert other exceptions to ROSConversionError with helpful context
        error_msg = (
            f"ROS message type '{ros_msg_type.__name__}' does not support conversion with ros2_numpy. "
            f"You may need to register a custom converter using @ros2_numpy.converts_to_numpy "
            f"and @ros2_numpy.converts_from_numpy decorators. "
            f"See victor_hardware_interfaces.py for an example. "
            f"Original error: {str(e)}"
        )
        raise ROSConversionError(error_msg) from e

def test_ros_to_numpy_conversion(ros_msg: Any) -> np.ndarray:
    """
    Test converting a ROS message to numpy array.
    
    Args:
        ros_msg: A ROS message instance
        
    Returns:
        The converted numpy array
        
    Raises:
        ROSConversionError: If conversion fails
        ImportError: If ros2_numpy is not available
    """
    if not ROS_AVAILABLE or numpify is None:
        raise ImportError("ros2_numpy is required for ROS message conversion. Please install it with: pip install ros2-numpy")
    
    try:
        return numpify(ros_msg)
    except Exception as e:
        error_msg = (
            f"Failed to convert ROS message of type '{type(ros_msg).__name__}' to numpy array. "
            f"You may need to register a custom converter using @ros2_numpy.converts_to_numpy. "
            f"Original error: {str(e)}"
        )
        raise ROSConversionError(error_msg) from e

def test_numpy_to_ros_conversion(ros_msg_type: Type[Any], numpy_data: np.ndarray) -> Any:
    """
    Test converting a numpy array to ROS message.
    
    Args:
        ros_msg_type: The ROS message class to convert to
        numpy_data: The numpy array to convert
        
    Returns:
        The converted ROS message
        
    Raises:
        ROSConversionError: If conversion fails
        ImportError: If ros2_numpy is not available
    """
    if not ROS_AVAILABLE or msgify is None:
        raise ImportError("ros2_numpy is required for ROS message conversion. Please install it with: pip install ros2-numpy")
    
    try:
        return msgify(ros_msg_type, numpy_data)
    except Exception as e:
        error_msg = (
            f"Failed to convert numpy array to ROS message of type '{ros_msg_type.__name__}'. "
            f"You may need to register a custom converter using @ros2_numpy.converts_from_numpy. "
            f"Original error: {str(e)}"
        )
        raise ROSConversionError(error_msg) from e

def get_ros_msg_type(name_or_type: Any) -> Type[Any]:
    """
    Resolves a string name or a direct type into a ROS message type.
    
    Args:
        name_or_type: A string module path like "sensor_msgs.msg.Image" or a ROS message class.

    Returns:
        The resolved ROS message class.
        
    Raises:
        TypeError: If the input is not a string or a type.
        ImportError: If the string name cannot be imported.
    """
    if isinstance(name_or_type, str):
        # Try to import the message type from its module path
        try:
            module_parts = name_or_type.split('.')
            if len(module_parts) < 3:  # Need at least package.msg.Type
                raise ValueError(f"Invalid ROS message type string: '{name_or_type}'. Expected format: 'package.msg.MessageType'")
            
            module_path = '.'.join(module_parts[:-1])
            class_name = module_parts[-1]
            
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
            
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Could not import ROS message type '{name_or_type}'. "
                f"Make sure the package is installed and the path is correct. "
                f"Expected format: 'package.msg.MessageType' (e.g., 'sensor_msgs.msg.Image'). "
                f"Original error: {str(e)}"
            ) from e
            
    elif isinstance(name_or_type, type):
        return name_or_type
    else:
        raise TypeError(f"Expected a string or a type for ros_msg_type, got {type(name_or_type)}")
