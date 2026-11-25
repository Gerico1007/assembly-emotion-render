"""R17: a render's peak RSS is that render's, not the process's lifetime peak."""
from emotion_render import core
from emotion_render.backend import TorchBackend, peak_rss_mb
from emotion_render.engines import ENGINES
from emotion_render.script import parse
from emotion_render.voices import Voice

from fakes import FakeBackend, make_wav


def _plan(tmp_path):
    ref = make_wav(tmp_path / "ref.wav", 10)
    voice = Voice(id="t-en", person="T", lang="en", engine="turbo", reference=ref,
                  consent={"person": "T", "given": "2026-11-25", "scope": "tests"})
    return core.Plan(voice, ENGINES["turbo"], parse("Hello."), {})


def test_a_render_right_after_its_load_keeps_the_load_in_its_peaks(tmp_path):
    backend = FakeBackend()
    model, _ = core.load_model(ENGINES["turbo"], "cpu", backend)
    core.render(_plan(tmp_path), model, device="cpu", out=tmp_path / "a.wav", backend=backend, load_seconds=12.7)
    assert backend.resets == ["cpu"]                  # once, before the load


def test_a_render_on_a_warm_model_measures_only_itself(tmp_path):
    backend = FakeBackend()
    core.render(_plan(tmp_path), {"engine": "turbo", "device": "cuda"}, device="cuda",
                out=tmp_path / "a.wav", backend=backend, load_seconds=0.0)
    assert backend.resets == ["cuda"]                 # before the generation


def test_resetting_the_peak_drops_what_came_before():
    block = bytearray(200 * 2**20)
    block[:: 4096] = b"x" * len(block[:: 4096])
    before = peak_rss_mb()
    del block
    TorchBackend().reset_peaks("cpu")                 # no torch import on cpu
    after = peak_rss_mb()
    assert before - after > 150
