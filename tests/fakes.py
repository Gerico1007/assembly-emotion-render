"""A backend that behaves like TorchBackend without torch or a model."""
from __future__ import annotations

import threading
import wave
from pathlib import Path


def make_wav(path: Path, seconds: float, rate: int = 24000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(seconds * rate))
    return path


class FakeOOM(RuntimeError):
    pass


class FakeBackend:
    def __init__(self, cuda_mb: int | None = 3717, available_mb: int | None = 20000, audio_seconds: float = 2.0):
        self.cuda_mb = cuda_mb
        self.available = available_mb
        self.audio_seconds = audio_seconds
        self.loads: list[tuple[str, str]] = []
        self.unloads: list[tuple[str, str]] = []
        self.generated: list[dict] = []
        self.gate: threading.Event | None = None   # set: generate waits on it
        self.started = threading.Event()            # set when a generate begins
        self.oom = False

    def cuda_total_mb(self):
        return self.cuda_mb

    def available_mb(self):
        return self.available

    def load(self, engine, device):
        self.loads.append((engine, device))
        return {"engine": engine, "device": device}

    def generate(self, model, engine, text, reference, lang, params):
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(20), "test gate never opened"
        if self.oom:
            raise FakeOOM("CUDA out of memory")
        self.generated.append({"engine": engine, "device": model["device"], "text": text,
                               "reference": reference, "lang": lang, "params": params})
        return b"\0\0" * int(self.audio_seconds * 24000), 24000

    def save_wav(self, audio, sr, path):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(audio)
        return len(audio) / 2 / sr

    def reset_peaks(self, device):
        pass

    def peaks(self, device):
        cuda = device == "cuda"
        return {"gpu": "Fake GPU" if cuda else None, "peak_rss_mb": 100,
                "peak_vram_reserved_mb": 50 if cuda else None}

    def is_out_of_memory(self, exc):
        return isinstance(exc, FakeOOM)

    def free_cache(self, device):
        pass

    def unload(self, model, device):
        self.unloads.append((model["engine"], device))

    def engine_version(self):
        return "fake"
