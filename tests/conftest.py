import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _private_gpu_lock(tmp_path, monkeypatch):
    # A test must never take, or wait for, this host's real GPU lock.
    monkeypatch.setenv("EMOTION_RENDER_GPU_LOCK", str(tmp_path / "gpu.lock"))
