"""
Base IPC class and native tensor backends with optional IPC binding.
"""
from __future__ import annotations
from typing import Optional, Any, Dict
from .utils import get_torch, get_torch_multiprocessing, require_numpy, require_posix_ipc

# Get optional dependencies
torch = get_torch()
mp = get_torch_multiprocessing()
TORCH_AVAILABLE = torch is not None

# Ensure numpy is available (required)
np = require_numpy()

# Ensure POSIX IPC is available (required)
posix_ipc = require_posix_ipc()

class PosixIPC:
    """POSIX IPC implementation using semaphores."""
    
    def __init__(self, pool_name: str, is_producer: bool = False):
        self.pool_name = pool_name
        self.is_producer = is_producer
    
        self._notif_sem: Optional[Any] = None
        self._mtx_sem: Optional[Any] = None
        
        if is_producer:
            self._create_ipc_objects()
    
    def _notif_name(self) -> str:
        """Generate notification semaphore name with pmipc prefix."""
        return f"/pmipc_{self.pool_name.lstrip('/')}_notif"
    
    def _mtx_name(self) -> str:
        """Generate mutex semaphore name with pmipc prefix."""
        return f"/pmipc_{self.pool_name.lstrip('/')}_mtx"
    
    def _create_ipc_objects(self):
        """Create POSIX IPC objects with robust handling (producer only)."""
        notif_name = self._notif_name()
        mtx_name = self._mtx_name()
        
        try:
            # Try to create new semaphores
            self._notif_sem = posix_ipc.Semaphore(
                notif_name, 
                flags=posix_ipc.O_CREX, 
                initial_value=0
            )
            self._mtx_sem = posix_ipc.Semaphore(
                mtx_name, 
                flags=posix_ipc.O_CREX, 
                initial_value=1
            )
        except posix_ipc.ExistentialError:
            # Semaphores already exist - try to handle robustly
            try:
                # Check if existing semaphores are from dead processes
                existing_notif = posix_ipc.Semaphore(notif_name)
                existing_mtx = posix_ipc.Semaphore(mtx_name)
                
                # Try to acquire mutex with timeout to detect if owner is alive
                try:
                    existing_mtx.acquire(timeout=0.1)  # 100ms timeout
                    existing_mtx.release()
                    # If we can acquire/release, the semaphores are healthy but in use
                    existing_notif.close()
                    existing_mtx.close()
                    raise RuntimeError(f"IPC objects for pool '{self.pool_name}' are actively in use")
                except posix_ipc.BusyError:
                    # Mutex is locked - check if we can recover
                    existing_notif.close()
                    existing_mtx.close()
                    
                    # Clean up potentially stale semaphores
                    try:
                        posix_ipc.unlink_semaphore(notif_name)
                        posix_ipc.unlink_semaphore(mtx_name)
                        print(f"Cleaned up stale semaphores for pool '{self.pool_name}'")
                        
                        # Now create new ones
                        self._notif_sem = posix_ipc.Semaphore(
                            notif_name, 
                            flags=posix_ipc.O_CREX, 
                            initial_value=0
                        )
                        self._mtx_sem = posix_ipc.Semaphore(
                            mtx_name, 
                            flags=posix_ipc.O_CREX, 
                            initial_value=1
                        )
                    except Exception as cleanup_error:
                        raise RuntimeError(f"Failed to clean up and recreate semaphores: {cleanup_error}")
                except Exception:
                    # Some other error during testing
                    existing_notif.close()
                    existing_mtx.close()
                    raise RuntimeError(f"IPC objects for pool '{self.pool_name}' exist but are in unknown state")
            
            except posix_ipc.ExistentialError:
                # Partial cleanup case - some semaphores exist, others don't
                try:
                    posix_ipc.unlink_semaphore(notif_name)
                except Exception:
                    pass
                try:
                    posix_ipc.unlink_semaphore(mtx_name)
                except Exception:
                    pass
                # Try creating again
                self._notif_sem = posix_ipc.Semaphore(
                    notif_name, 
                    flags=posix_ipc.O_CREX, 
                    initial_value=0
                )
                self._mtx_sem = posix_ipc.Semaphore(
                    mtx_name, 
                    flags=posix_ipc.O_CREX, 
                    initial_value=1
                )
        except Exception as e:
            raise RuntimeError(f"Failed to create IPC objects for pool '{self.pool_name}': {e}")
    
    def connect_to_ipc_objects(self) -> bool:
        """Connect to existing IPC objects (consumer only)."""
        if self._notif_sem is not None and self._mtx_sem is not None:
            return True
            
        try:
            self._notif_sem = posix_ipc.Semaphore(self._notif_name())
            self._mtx_sem = posix_ipc.Semaphore(self._mtx_name())
            return True
        except posix_ipc.ExistentialError:
            return False
    
    def notify_consumers(self) -> None:
        """Send notification to consumers."""
        if self.is_producer and self._notif_sem is not None:
            self._notif_sem.release()
    
    def wait_for_notification(self, timeout: Optional[float] = None) -> bool:
        """Wait for notification from producer."""
        if not self.is_producer and self._notif_sem is not None:
            try:
                if timeout is None:
                    self._notif_sem.acquire()
                    return True
                else:
                    self._notif_sem.acquire(timeout=timeout)
                    return True
            except posix_ipc.BusyError:
                return False
        return False
    
    def cleanup(self) -> None:
        """Clean up IPC resources."""
        if self._notif_sem:
            self._notif_sem.close()
            if self.is_producer:
                try:
                    self._notif_sem.unlink()
                except posix_ipc.ExistentialError:
                    pass
        
        if self._mtx_sem:
            self._mtx_sem.close()
            if self.is_producer:
                try:
                    self._mtx_sem.unlink()
                except posix_ipc.ExistentialError:
                    pass