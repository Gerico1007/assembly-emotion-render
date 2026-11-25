"""One process at a time on the GPU (D11).

Turbo on a 4 GB card leaves about 370 MB free. Without this lock, a second process
loading a model on CUDA runs out of memory after its load is paid. With it, the
second process learns at once that the GPU is taken: "auto" goes to CPU, an
explicit "cuda" is refused with `gpu_busy`.

The lock is `flock` on one file. The kernel releases it when the holder exits,
however it exits, so a crash never leaves the GPU marked as taken.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

DEFAULT_PATH = Path.home() / ".cache" / "assembly-emotion-render" / "gpu.lock"


def default_path() -> Path:
    override = os.environ.get("EMOTION_RENDER_GPU_LOCK")
    return Path(override) if override else DEFAULT_PATH


class GpuLock:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else default_path()
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def _open(self) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)

    def acquire(self) -> bool:
        """Take the GPU. False when another process holds it."""
        if self.held:
            return True
        fd = self._open()
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            os.ftruncate(self._fd, 0)
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def holder(self) -> int | None:
        """The pid of another process holding the GPU, -1 if held but unreadable, None if free."""
        if self.held:
            return None
        fd = self._open()
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raw = os.pread(fd, 32, 0).decode(errors="replace").strip()
                return int(raw) if raw.isdigit() else -1
            fcntl.flock(fd, fcntl.LOCK_UN)
            return None
        finally:
            os.close(fd)
