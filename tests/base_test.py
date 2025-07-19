"""
Test multi-process producer/consumer with different frequencies and time consistency validation.
"""
from pathlib import Path
from abc import ABC
import subprocess
import os
import signal
import time

class BaseTest(ABC):
    """Test suite for multi-process frequency and time consistency."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Clean up any leftover shared memory objects
        self._cleanup_shared_memory()
        
        self._cleanup_pools()
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Clean up shared memory objects
        self._cleanup_shared_memory()
        
        # Clean up any leftover processes
        self._cleanup_processes()
        
        self._cleanup_pools()
    
    def _cleanup_shared_memory(self):
        """Clean up POSIX shared memory objects."""
        try:
            # List and remove shared memory objects starting with /pmipc
            result = subprocess.run(['ls', '/dev/shm/'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('pmipc'):
                        try:
                            os.unlink(f'/dev/shm/{line}')
                            print(f"Cleaned up stale shared memory: {line}")
                        except FileNotFoundError:
                            pass
                        except Exception as e:
                            print(f"Warning: Could not clean up {line}: {e}")
        except Exception as e:
            print(f"Warning: Could not list shared memory objects: {e}")
            
    def _cleanup_processes(self):
        """Kill any leftover test processes."""
        try:
            # Kill any Python processes that might be running our test scripts
            result = subprocess.run(['pgrep', '-f', 'example_producer.py'], capture_output=True, text=True)
            if result.returncode == 0:
                for pid in result.stdout.strip().split('\n'):
                    if pid:
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            print(f"Killed leftover producer process: {pid}")
                        except (ProcessLookupError, ValueError):
                            pass
                            
            result = subprocess.run(['pgrep', '-f', 'example_consumer.py'], capture_output=True, text=True)
            if result.returncode == 0:
                for pid in result.stdout.strip().split('\n'):
                    if pid:
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            print(f"Killed leftover consumer process: {pid}")
                        except (ProcessLookupError, ValueError):
                            pass
        except Exception as e:
            print(f"Warning: Could not clean up processes: {e}")
            
        # Give processes time to clean up
        time.sleep(0.2)
    
    def _cleanup_pools(self):
        """Clean up all pools and IPC objects."""
        try:
            from poly_ipc import TensorPoolRegistry
            
            test_pool_names = [
                "test_multiprocess_freq", "test_consumer_first", "test_numpy_7", 
                "test_numpy_1_3_2", "test_numpy_7_delayed", "test_numpy_large_1080p_history30",
                "test_numpy_large_480p_history30"
            ]
            
            for pool_name in test_pool_names:
                try:
                    TensorPoolRegistry.unregister_pool(pool_name)
                except Exception:
                    pass
                        
        except Exception:
            pass
        
        # Clean up temporary files
        fallback_dir = Path("/tmp/tensorpool-fs")
        if fallback_dir.exists():
            for file in fallback_dir.glob("*.json"):
                try:
                    file.unlink()
                except Exception:
                    pass

        # Clean up shared memory objects more thoroughly
        import glob
        for shm_path in glob.glob("/dev/shm/*numpy_pool*"):
            try:
                Path(shm_path).unlink()
            except Exception:
                pass
        for shm_path in glob.glob("/dev/shm/*metadata*"):
            try:
                Path(shm_path).unlink()
            except Exception:
                pass
