"""
Utility functions and dependency checking for Victor Python IPC.
"""
import sys
import platform
import warnings
from typing import Optional, Dict, Any


class SchemaMismatchError(RuntimeError):
    """Raised when a channel's shape or dtype does not match expectations."""
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class DependencyError(ImportError):
    """Raised when a required dependency is not available."""
    pass


class AvailabilityChecker:
    """
    Centralized availability checking for all dependencies and platform features.
    
    This class performs one-time checks during module import and caches the results
    to avoid repeated import attempts.
    """
    
    def __init__(self):
        self._checks_performed = False
        self._availability = {}
        self._warnings_issued = set()
        
    def check_all(self) -> Dict[str, Any]:
        """Perform all availability checks and return results."""
        if self._checks_performed:
            return self._availability.copy()
            
        self._availability = {
            'numpy': self._check_numpy(),
            'posix_ipc': self._check_posix_ipc(),
            'torch': self._check_torch(),
            'ros2_numpy': self._check_ros2_numpy(),
            'platform_support': self._check_platform_support(),
        }
        
        self._checks_performed = True
        return self._availability.copy()
    
    def _check_numpy(self) -> Dict[str, Any]:
        """Check NumPy availability (required)."""
        try:
            import numpy as np
            return {
                'available': True,
                'version': np.__version__,
                'module': np,
                'error': None
            }
        except ImportError as e:
            # NumPy is absolutely required - this is a fatal error
            raise DependencyError(
                "NumPy is required for Victor Python IPC but is not available. "
                "Please install it with: pip install numpy"
            ) from e
    
    def _check_posix_ipc(self) -> Dict[str, Any]:
        """Check POSIX IPC availability (required)."""
        try:
            import posix_ipc
            return {
                'available': True,
                'version': getattr(posix_ipc, '__version__', 'unknown'),
                'module': posix_ipc,
                'error': None
            }
        except ImportError as e:
            # POSIX IPC is now required - this is a fatal error
            raise DependencyError(
                "POSIX IPC is required for Victor Python IPC but is not available. "
                "Please install it with: pip install posix-ipc"
            ) from e
    
    def _check_torch(self) -> Dict[str, Any]:
        """Check PyTorch availability (optional)."""
        try:
            import torch
            import torch.multiprocessing as mp
            
            # Test if CUDA is available
            cuda_available = torch.cuda.is_available()
            cuda_device_count = torch.cuda.device_count() if cuda_available else 0
            
            return {
                'available': True,
                'version': torch.__version__,
                'module': torch,
                'multiprocessing': mp,
                'cuda_available': cuda_available,
                'cuda_device_count': cuda_device_count,
                'error': None
            }
        except ImportError as e:
            info_msg = (
                "PyTorch is not available. Torch tensor support will be disabled. "
                "To enable PyTorch support, install with: pip install torch"
            )
            # Don't warn about torch since it's truly optional
            return {
                'available': False,
                'version': None,
                'module': None,
                'multiprocessing': None,
                'cuda_available': False,
                'cuda_device_count': 0,
                'error': str(e),
                'info': info_msg
            }
    
    def _check_ros2_numpy(self) -> Dict[str, Any]:
        """Check ros2_numpy availability (optional)."""
        try:
            import ros2_numpy
            from ros2_numpy import numpify, msgify
            
            return {
                'available': True,
                'version': getattr(ros2_numpy, '__version__', 'unknown'),
                'module': ros2_numpy,
                'numpify': numpify,
                'msgify': msgify,
                'error': None
            }
        except ImportError as e:
            info_msg = (
                "ros2_numpy is not available. ROS message conversion will be disabled. "
                "To enable ROS support, install with: pip install ros2-numpy"
            )
            # Don't warn about ros2_numpy since it's truly optional
            return {
                'available': False,
                'version': None,
                'module': None,
                'numpify': None,
                'msgify': None,
                'error': str(e),
                'info': info_msg
            }
    
    def _check_platform_support(self) -> Dict[str, Any]:
        """Check platform-specific features."""
        system = platform.system()
        
        # Check for POSIX-like system (required for posix_ipc)
        posix_like = system in ('Linux', 'Darwin', 'FreeBSD', 'OpenBSD', 'NetBSD')
        
        # Check for shared memory support
        shared_memory_available = True
        try:
            import importlib.util
            shared_memory_available = importlib.util.find_spec('multiprocessing.shared_memory') is not None
        except (ImportError, AttributeError):
            shared_memory_available = False
        
        result = {
            'system': system,
            'posix_like': posix_like,
            'shared_memory_available': shared_memory_available,
            'python_version': sys.version,
        }
        
        # Issue warnings for unsupported platforms
        if not posix_like and 'platform_posix' not in self._warnings_issued:
            warnings.warn(
                f"Platform '{system}' may not fully support POSIX IPC features. "
                "Some functionality may be limited.",
                UserWarning,
                stacklevel=3
            )
            self._warnings_issued.add('platform_posix')
        
        if not shared_memory_available and 'platform_shm' not in self._warnings_issued:
            warnings.warn(
                "multiprocessing.shared_memory is not available. "
                "This may limit some IPC functionality.",
                UserWarning,
                stacklevel=3
            )
            self._warnings_issued.add('platform_shm')
        
        return result
    
    def require_numpy(self):
        """Ensure NumPy is available (raises if not)."""
        result = self.check_all()
        if not result['numpy']['available']:
            raise DependencyError("NumPy is required but not available")
        return result['numpy']['module']
    
    def require_posix_ipc(self):
        """Ensure POSIX IPC is available (raises if not)."""
        result = self.check_all()
        if not result['posix_ipc']['available']:
            raise DependencyError(
                "POSIX IPC is required for this operation but not available. "
                "Install with: pip install posix-ipc"
            )
        return result['posix_ipc']['module']
    
    def get_torch(self) -> Optional[Any]:
        """Get torch module if available, None otherwise."""
        result = self.check_all()
        return result['torch']['module'] if result['torch']['available'] else None
    
    def get_torch_multiprocessing(self) -> Optional[Any]:
        """Get torch.multiprocessing if available, None otherwise."""
        result = self.check_all()
        return result['torch']['multiprocessing'] if result['torch']['available'] else None
    
    def get_ros2_numpy(self) -> Optional[Dict[str, Any]]:
        """Get ros2_numpy functions if available, None otherwise."""
        result = self.check_all()
        if result['ros2_numpy']['available']:
            return {
                'module': result['ros2_numpy']['module'],
                'numpify': result['ros2_numpy']['numpify'],
                'msgify': result['ros2_numpy']['msgify']
            }
        return None
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        result = self.check_all()
        return result['torch']['available'] and result['torch']['cuda_available']
    
    def print_availability_report(self):
        """Print a detailed availability report."""
        result = self.check_all()
        
        print("Victor Python IPC - Dependency Availability Report")
        print("=" * 50)
        
        # Required dependencies
        print("\nRequired Dependencies:")
        numpy_info = result['numpy']
        print(f"  ✓ NumPy: {numpy_info['version']}")
        
        # Platform support
        platform_info = result['platform_support']
        print("\nPlatform Support:")
        print(f"  System: {platform_info['system']}")
        print(f"  POSIX-like: {'✓' if platform_info['posix_like'] else '✗'}")
        print(f"  Shared Memory: {'✓' if platform_info['shared_memory_available'] else '✗'}")
        
        # IPC support
        print("\nIPC Support:")
        posix_info = result['posix_ipc']
        if posix_info['available']:
            print(f"  ✓ POSIX IPC: {posix_info['version']}")
        else:
            print("  ✗ POSIX IPC: Not available")
        
        # Optional dependencies
        print("\nOptional Dependencies:")
        torch_info = result['torch']
        if torch_info['available']:
            print(f"  ✓ PyTorch: {torch_info['version']}")
            if torch_info['cuda_available']:
                print(f"    ✓ CUDA: {torch_info['cuda_device_count']} device(s)")
            else:
                print("    ✗ CUDA: Not available")
        else:
            print("  ✗ PyTorch: Not available")
        
        ros_info = result['ros2_numpy']
        if ros_info['available']:
            print(f"  ✓ ros2_numpy: {ros_info['version']}")
        else:
            print("  ✗ ros2_numpy: Not available")


# Global availability checker instance
_availability_checker = AvailabilityChecker()


# Convenience functions for common checks
def check_availability() -> Dict[str, Any]:
    """Check availability of all dependencies."""
    return _availability_checker.check_all()


def require_numpy():
    """Ensure NumPy is available."""
    return _availability_checker.require_numpy()


def require_posix_ipc():
    """Ensure POSIX IPC is available."""
    return _availability_checker.require_posix_ipc()


def get_torch():
    """Get torch module if available."""
    return _availability_checker.get_torch()


def get_torch_multiprocessing():
    """Get torch.multiprocessing if available."""
    return _availability_checker.get_torch_multiprocessing()


def get_ros2_numpy():
    """Get ros2_numpy functions if available."""
    return _availability_checker.get_ros2_numpy()


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return _availability_checker.is_cuda_available()


def print_availability_report():
    """Print a detailed availability report."""
    _availability_checker.print_availability_report()


class EnsureROSNumpy:
    """Context manager that ensures ros2_numpy is available."""
    
    def __init__(self):
        pass

    def __enter__(self):
        ros2_numpy_info = get_ros2_numpy()
        if ros2_numpy_info is None:
            raise ImportError(
                "ros2_numpy is required for ROS message conversion but is not available. "
                "Please install it with: pip install ros2-numpy"
            )
        return ros2_numpy_info

    def __exit__(self, exc_type, exc_value, traceback):
        pass