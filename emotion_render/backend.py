"""What a render needs from torch and Chatterbox, behind one small seam.

The CLI and the MCP server both render through `core.render`, which only talks to
a backend. `TorchBackend` is the real one and imports torch only when a method is
called. Tests use a fake with the same methods, so no test needs the model.
"""
from __future__ import annotations

import resource
from importlib.metadata import PackageNotFoundError, version

from .engines import adapter, available_mb


class TorchBackend:
    def cuda_total_mb(self) -> int | None:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory // 2**20

    def available_mb(self) -> int | None:
        return available_mb()

    def load(self, engine: str, device: str):
        return adapter(engine).load(device)

    def generate(self, model, engine: str, text: str, reference: str, lang: str, params: dict):
        """Returns (audio, sample_rate). `audio` is only ever handed back to `save_wav`."""
        wav = adapter(engine).generate(model, text, reference, lang, params)
        wav = wav.detach().cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return wav, model.sr

    def save_wav(self, audio, sample_rate: int, path) -> float:
        """Writes the wav, returns its duration in seconds."""
        import torchaudio

        torchaudio.save(str(path), audio, sample_rate)
        return audio.shape[-1] / sample_rate

    def reset_peaks(self, device: str) -> None:
        if device == "cuda":
            import torch

            torch.cuda.reset_peak_memory_stats()

    def peaks(self, device: str) -> dict:
        gpu = vram = None
        if device == "cuda":
            import torch

            gpu = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.max_memory_reserved() / 2**20)
        return {
            "gpu": gpu,
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            "peak_vram_reserved_mb": vram,
        }

    def is_out_of_memory(self, exc: BaseException) -> bool:
        import torch

        return isinstance(exc, torch.OutOfMemoryError)

    def free_cache(self, device: str) -> None:
        if device == "cuda":
            import torch

            torch.cuda.empty_cache()

    def unload(self, model, device: str) -> None:
        del model
        self.free_cache(device)

    def engine_version(self) -> str | None:
        try:
            return version("chatterbox-tts")
        except PackageNotFoundError:
            return None
