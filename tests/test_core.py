"""B1: the CLI renders through core with its phase-2 behavior, on a fake backend."""
import json

import pytest

from emotion_render import Refused, cli, core, voices

from fakes import FakeBackend
from store import make_store

# The record's keys, in order, as phase 2 published them (README, A4 sidecars).
RECORD_KEYS = [
    "ok", "voice", "person", "lang", "engine", "engine_text", "clean_text", "tags", "params", "script",
    "chatterbox_version", "emotion_render_version", "device", "gpu", "load_seconds", "render_seconds",
    "audio_seconds", "seconds_per_audio_second", "peak_rss_mb", "peak_vram_reserved_mb", "out", "reference",
    "consent", "watermark", "created_at",
]


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = make_store(tmp_path / "store")
    monkeypatch.setattr(voices, "REGISTRY_DIR", path)
    return path


def test_the_cli_renders_through_core(store, tmp_path, monkeypatch, capsys):
    backend = FakeBackend()
    monkeypatch.setattr(cli, "make_backend", lambda: backend)
    out = tmp_path / "out" / "x.wav"
    assert cli.main(["--voice", "ada-en", "--text", "[sigh] Hello.", "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert list(printed) == RECORD_KEYS and printed == sidecar
    assert printed["device"] == "cuda" and printed["tags"] == ["sigh"] and printed["audio_seconds"] == 2.0
    assert backend.loads == [("turbo", "cuda")] and out.is_file()


def test_the_cli_memory_guard_refuses_before_loading(store, tmp_path, monkeypatch, capsys):
    backend = FakeBackend(available_mb=100)
    monkeypatch.setattr(cli, "make_backend", lambda: backend)
    assert cli.main(["--voice", "ada-fr", "--text", "Bonjour.", "--out", str(tmp_path / "x.wav")]) == 2
    assert "100 MB of RAM available" in capsys.readouterr().err
    assert backend.loads == []
    backend.available = 100
    assert cli.main(["--voice", "ada-fr", "--text", "Bonjour.", "--no-memory-guard",
                     "--out", str(tmp_path / "x.wav")]) == 0


def test_plan_is_what_check_prints(store, capsys):
    assert cli.main(["--voice", "ada-fr", "--text", "Bonjour.", "--exaggeration", "0.7", "--check"]) == 0
    printed = json.loads(capsys.readouterr().out)
    p = core.plan("ada-fr", "Bonjour.", 0.7)
    assert printed == {"ok": True, "check": True, **p.as_dict()}
    assert p.params == {"exaggeration": 0.7}


def test_every_refusal_carries_a_code(store):
    with pytest.raises(Refused) as e:
        core.plan("ada-en", "Hi [whisper].")
    assert e.value.to_dict()["code"] == "unknown_tag" and e.value.to_dict()["message"] == str(e.value)
