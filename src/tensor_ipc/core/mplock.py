"""
Writer‑priority readers–writer lock implemented with POSIX named semaphores +
a tiny shared‑memory counter.

The interface is purposely minimal:

    lock = MPLock("/tensor_pool_A")          # same name used by all processes

    # Producer
    with lock.write_lock():
        ...  # mutate the tensor pool

    # Consumers
    with lock.read_lock():
        data = pool[idx]

Call `lock.close()` in long‑running daemons and `lock.unlink()` when the
tensor_pool itself is permanently destroyed.
"""
from __future__ import annotations
import mmap
import struct
from contextlib import contextmanager
from typing import Final

import posix_ipc


class MPLock:
    _COUNT_FMT: Final[str] = "Q"          # unsigned 8‑byte counter
    _COUNT_SIZE: Final[int] = struct.calcsize(_COUNT_FMT)

    def __init__(self, name: str, *, create: bool = False):
        """
        `name` MUST start with a slash and be unique per tensor_pool, e.g.
          "/tensor_pool_<uuid>"

        The first process (the producer) passes `create=True`; every
        subsequent reader/writer uses the default (`create=False`).
        """
        flags = posix_ipc.O_CREAT if create else 0
        self._mutex   = posix_ipc.Semaphore(name + "_mutex",   flags=flags, initial_value=1)
        self._wrt     = posix_ipc.Semaphore(name + "_wrt",     flags=flags, initial_value=1)
        self._readtry = posix_ipc.Semaphore(name + "_readtry", flags=flags, initial_value=1)

        # shared counter for the number of *active* readers
        self._shm = posix_ipc.SharedMemory(name + "_cnt", flags=flags, size=self._COUNT_SIZE)
        self._mm  = mmap.mmap(self._shm.fd, self._COUNT_SIZE)
        self._shm.close_fd()      # mmap now owns the fd

    # --------------------------------------------------------------------- #
    # low‑level helpers                                                     #
    # --------------------------------------------------------------------- #
    def _get_rcount(self) -> int:
        self._mm.seek(0)
        return struct.unpack(self._COUNT_FMT, self._mm.read(self._COUNT_SIZE))[0]

    def _set_rcount(self, val: int) -> None:
        self._mm.seek(0)
        self._mm.write(struct.pack(self._COUNT_FMT, val))

    # --------------------------------------------------------------------- #
    # public API                                                            #
    # --------------------------------------------------------------------- #
    @contextmanager
    def read_lock(self):
        """
        Multiple readers may hold the lock simultaneously *unless* a writer
        is waiting, in which case new readers block until the writer finishes.
        """
        # Writer‑priority solution (Tanembaum variant):
        self._readtry.acquire()       # ① do *not* starve waiting writers
        self._mutex.acquire()
        rcount = self._get_rcount() + 1
        self._set_rcount(rcount)
        if rcount == 1:               # first reader blocks writers
            self._wrt.acquire()
        self._mutex.release()
        self._readtry.release()

        try:
            yield
        finally:
            self._mutex.acquire()
            rcount = self._get_rcount() - 1
            self._set_rcount(rcount)
            if rcount == 0:           # last reader releases writers
                self._wrt.release()
            self._mutex.release()

    @contextmanager
    def write_lock(self):
        """
        The single producer obtains exclusive access.  Once it starts waiting,
        new readers are blocked at step ① above, ensuring writer priority.
        """
        self._readtry.acquire()       # block *new* readers
        self._wrt.acquire()           # wait for *existing* readers
        try:
            yield
        finally:
            self._wrt.release()
            self._readtry.release()

    # --------------------------------------------------------------------- #
    # cleanup                                                               #
    # --------------------------------------------------------------------- #
    def close(self):
        """Close memory map & semaphores in this *process* only."""
        for sem in (self._mutex, self._wrt, self._readtry):
            try:
                sem.close()
            except Exception as e:
                pass

    def unlink(self):
        """
        Permanently remove the IPC objects *system‑wide*.
        Call this exactly once when the tensor_pool is irrevocably gone.
        """
        for sem in (self._mutex, self._wrt, self._readtry, self._shm):
            try:
                sem.unlink()
            except Exception as e:
                pass
