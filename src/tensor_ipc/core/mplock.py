# per_frame_lock.py
import mmap, struct, posix_ipc
from contextlib import contextmanager
from typing import List

class _SingleFrameLock:
    """Writer‑priority RW lock for one frame (lazy creation)."""
    _FMT = "Q"; _SIZE = struct.calcsize(_FMT)

    def __init__(self, base_name: str, index: int):
        self._name = f"{base_name}_f{index}"
        self._mutex = self._wrt = self._readtry = None
        self._shm = self._mm = None
        self._ready = False

    # ----- lazy open / create -------------------------------------------------
    def _open_sem(self, suffix: str, init: int):
        full = self._name + suffix
        try:
            sem = posix_ipc.Semaphore(full,
                                      flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL,
                                      initial_value=init)
        except posix_ipc.ExistentialError:
            sem = posix_ipc.Semaphore(full)
        return sem

    def _ensure(self) -> bool:
        if self._ready:
            return True
        try:
            self._mutex   = self._open_sem("_mutex", 1)
            self._wrt     = self._open_sem("_wrt", 1)
            self._readtry = self._open_sem("_readtry", 1)
            # reader counter
            try:
                shm = posix_ipc.SharedMemory(self._name + "_cnt",
                                             flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL,
                                             size=self._SIZE)
                created = True
            except posix_ipc.ExistentialError:
                shm = posix_ipc.SharedMemory(self._name + "_cnt"); created = False
            mm = mmap.mmap(shm.fd, self._SIZE); shm.close_fd()
            if created:
                mm.seek(0); mm.write(b"\x00" * self._SIZE); mm.flush()
            self._shm, self._mm = shm, mm
            self._ready = True
        except Exception:
            self.close()
        return self._ready

    # ----- internal counter helpers ------------------------------------------
    def _get_rcount(self):
        self._mm.seek(0); return struct.unpack(self._FMT, self._mm.read(self._SIZE))[0]
    def _set_rcount(self, v: int):
        self._mm.seek(0); self._mm.write(struct.pack(self._FMT, v))

    # ----- public context managers --------------------------------------------
    @contextmanager
    def read_lock(self):
        if not self._ensure():
            yield; return
        self._readtry.acquire()
        self._mutex.acquire()
        rc = self._get_rcount() + 1
        self._set_rcount(rc)
        if rc == 1:
            self._wrt.acquire()
        self._mutex.release()
        self._readtry.release()
        try:
            yield
        finally:
            self._mutex.acquire()
            rc = self._get_rcount() - 1
            self._set_rcount(rc)
            if rc == 0:
                self._wrt.release()
            self._mutex.release()

    @contextmanager
    def write_lock(self):
        if not self._ensure():
            yield; return
        self._readtry.acquire()
        self._wrt.acquire()
        try:
            yield
        finally:
            self._wrt.release()
            self._readtry.release()

    # ----- cleanup ------------------------------------------------------------
    def close(self):
        for obj in (self._mm, self._mutex, self._wrt, self._readtry, self._shm):
            try:
                if obj is not None and hasattr(obj, "close"):
                    obj.close()
            except Exception:
                pass
        self._mm = self._mutex = self._wrt = self._readtry = self._shm = None
        self._ready = False

    def unlink(self):
        for obj in (self._mutex, self._wrt, self._readtry, self._shm):
            try:
                if obj is not None:
                    obj.unlink()
            except Exception:
                pass


class PoolFrameLocks:
    """
    Collection of per‑frame locks.
    base_name must start with '/'
    """
    def __init__(self, base_name: str, history_len: int):
        assert base_name.startswith("/"), "Semaphore base name must start with '/'"
        self.history_len = history_len
        self._locks = [_SingleFrameLock(base_name, i) for i in range(history_len)]

    # Writer API
    @contextmanager
    def write_frame(self, frame_index: int):
        with self._locks[frame_index].write_lock():
            yield

    # Reader API: acquire multiple frames (sorted to avoid deadlock)
    @contextmanager
    def read_frames(self, frame_indices: List[int]):
        if isinstance(frame_indices, slice):
            frame_indices = list(range(*frame_indices.indices(self.history_len)))
        indices = sorted(set(frame_indices))
        # acquire all
        ctxs = [self._locks[i].read_lock() for i in indices]
        for c in ctxs: c.__enter__()
        try:
            yield
        finally:
            # release in reverse order
            for c in reversed(ctxs):
                c.__exit__(None, None, None)

    def close(self):
        for l in self._locks: l.close()

    def unlink(self):
        for l in self._locks: l.unlink()
