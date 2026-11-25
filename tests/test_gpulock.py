"""B2: the GPU lock, shared by every process that puts a model on the GPU (D11)."""
import subprocess
import sys
import textwrap

import pytest

from emotion_render import Refused, core
from emotion_render.engines import ENGINES
from emotion_render.gpulock import GpuLock
from emotion_render.script import parse

from fakes import FakeBackend


def _plan(engine="turbo"):
    return core.Plan(voice=None, spec=ENGINES[engine], script=parse("Hello."), params={})


@pytest.fixture
def holder(tmp_path):
    """Another process holding the GPU lock until the test ends."""
    path = tmp_path / "gpu.lock"
    code = textwrap.dedent(f"""
        import sys
        from emotion_render.gpulock import GpuLock
        lock = GpuLock({str(path)!r})
        assert lock.acquire()
        print("held", flush=True)
        sys.stdin.read()
    """)
    proc = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, cwd=str(tmp_path.parent), env=None)
    assert proc.stdout.readline().strip() == "held"
    yield path, proc
    proc.stdin.close()
    proc.wait(10)


def test_a_free_gpu_has_no_holder(tmp_path):
    lock = GpuLock(tmp_path / "gpu.lock")
    assert lock.holder() is None
    assert lock.acquire() and lock.held
    lock.release()
    assert not lock.held and lock.holder() is None


def test_the_second_process_gets_the_cpu(holder):
    path, proc = holder
    lock = GpuLock(path)
    assert lock.holder() == proc.pid
    assert not lock.acquire()
    assert core.claim_device("auto", _plan(), FakeBackend(), lock) == "cpu"
    with pytest.raises(Refused) as e:
        core.claim_device("cuda", _plan(), FakeBackend(), lock)
    assert e.value.code == "gpu_busy" and e.value.detail["holder_pid"] == proc.pid


def test_the_gpu_is_free_again_when_the_holder_exits(holder):
    path, proc = holder
    proc.stdin.close()
    proc.wait(10)
    lock = GpuLock(path)
    assert core.claim_device("auto", _plan(), FakeBackend(), lock) == "cuda"
    assert lock.held
    lock.release()


def test_a_process_that_holds_the_lock_keeps_the_gpu(tmp_path):
    lock = GpuLock(tmp_path / "gpu.lock")
    assert lock.acquire()
    assert core.claim_device("cuda", _plan(), FakeBackend(), lock) == "cuda"
    lock.release()


def test_multilingual_never_takes_the_lock(tmp_path):
    lock = GpuLock(tmp_path / "gpu.lock")
    assert core.claim_device("auto", _plan("multilingual"), FakeBackend(), lock) == "cpu"
    assert not lock.held
