"""The one code path from a request to a rendered wav, shared by the CLI and the MCP server (D9).

`plan` holds every rule that can be checked without a model: the voice and its
consent (I2), the tags (I1), the settings, the length. `render` runs one
generation on a model that is already loaded and writes the wav. Between them,
`claim_device` and `guard_memory` decide where the model goes and whether this
host can afford it.

Because both surfaces call these functions, a refusal is the same refusal with the
same message whether it reaches a person at the terminal or an agent over MCP.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import Refused, __version__, voices
from .engines import ENGINES, EngineSpec, choose_device
from .script import Script, parse, validate

# Turbo is made for short clips (Sugi, CPU: about 2 minutes per 10 s of audio).
# Longer text needs chunking at sentence ends, which is not built yet.
MAX_CHARS = 400


@dataclass(frozen=True)
class Plan:
    voice: voices.Voice
    spec: EngineSpec
    script: Script
    params: dict

    def as_dict(self) -> dict:
        return {
            "voice": self.voice.id,
            "person": self.voice.person,
            "lang": self.voice.lang,
            "engine": self.spec.name,
            "engine_text": self.script.engine_text,
            "clean_text": self.script.clean_text,
            "tags": list(self.script.tags),
            "params": self.params,
        }


def _params(spec: EngineSpec, defaults: dict, exaggeration, cfg_weight) -> dict:
    given = {
        k: v
        for k, v in (("exaggeration", exaggeration), ("cfg_weight", cfg_weight))
        if v is not None
    }
    ignored = sorted(set(given) - spec.params)
    if ignored:
        raise Refused(
            f"{spec.name} does not take {', '.join(ignored)}. "
            f"A setting the engine would ignore is refused, the same as a tag (I1)",
            "setting_not_taken",
            accepted=sorted(spec.params),
        )
    return {**defaults, **given}


def plan(
    voice_id: str,
    text: str,
    exaggeration: float | None = None,
    cfg_weight: float | None = None,
    registry_dir: Path | None = None,
    voice: voices.Voice | None = None,
) -> Plan:
    """Everything `--check` does. No model. Raises Refused with a stable code."""
    voice = voice if voice is not None else voices.load(voice_id, registry_dir)
    spec = ENGINES[voice.engine]
    script = parse(text)
    validate(script, spec.tags, spec.name)
    if len(script.clean_text) > MAX_CHARS:
        raise Refused(
            f"{len(script.clean_text)} characters. Up to {MAX_CHARS} per clip until chunking is built",
            "text_too_long",
            limit=MAX_CHARS,
        )
    return Plan(voice, spec, script, _params(spec, voice.defaults, exaggeration, cfg_weight))


def claim_device(requested: str, p: Plan, backend, lock) -> str:
    """Choose the device and, for CUDA, take the GPU lock (D11) before any load.

    A process that already holds the lock keeps using the GPU. Otherwise the lock
    is probed, then taken; if another process wins the race in between, the
    choice is made again with that process as the holder.
    """
    cuda_total = backend.cuda_total_mb()
    chars = len(p.script.clean_text)
    holder = None if lock.held else lock.holder()
    device = choose_device(requested, p.spec, cuda_total, holder, chars)
    if device == "cuda" and not lock.acquire():
        device = choose_device(requested, p.spec, cuda_total, lock.holder() or -1, chars)
    return device


def guard_memory(spec: EngineSpec, device: str, backend, beside: bool = False) -> None:
    """Refuse a load this host cannot afford. `beside`: the process already holds a model."""
    need = spec.min_ram_mb_beside.get(device, spec.min_ram_mb[device]) if beside else spec.min_ram_mb[device]
    free = backend.available_mb()
    if free is not None and free < need:
        where = " beside the model already loaded" if beside else ""
        raise Refused(
            f"{free} MB of RAM available, {spec.name} on {device} needs {need} MB{where}. "
            f"Loading now could push this host into swap. Retry when it is quieter",
            "low_memory",
            available_mb=free,
            needed_mb=need,
        )


def load_model(spec: EngineSpec, device: str, backend):
    """(model, load_seconds)"""
    backend.reset_peaks(device)
    t0 = time.perf_counter()
    model = backend.load(spec.name, device)
    return model, time.perf_counter() - t0


def render(p: Plan, model, *, device: str, out: Path, backend, load_seconds: float) -> dict:
    """One generation on a loaded model. Writes `out` (wav) and returns the record.

    The record's keys and order are the CLI's since phase 2; the sidecar JSON and
    the line printed by `emotion-render` are this dict.
    """
    backend.reset_peaks(device)
    t1 = time.perf_counter()
    audio, sr = backend.generate(
        model, p.spec.name, p.script.engine_text, str(p.voice.reference), p.voice.lang, p.params
    )
    render_s = time.perf_counter() - t1
    out.parent.mkdir(parents=True, exist_ok=True)
    audio_s = backend.save_wav(audio, sr, out)
    peaks = backend.peaks(device)
    return {
        "ok": True,
        **p.as_dict(),
        "script": p.script.raw,
        "chatterbox_version": backend.engine_version(),
        "emotion_render_version": __version__,
        "device": device,
        "gpu": peaks["gpu"],
        "load_seconds": round(load_seconds, 1),
        "render_seconds": round(render_s, 1),
        "audio_seconds": round(audio_s, 2),
        "seconds_per_audio_second": round(render_s / audio_s, 2) if audio_s else None,
        "peak_rss_mb": peaks["peak_rss_mb"],
        "peak_vram_reserved_mb": peaks["peak_vram_reserved_mb"],
        "out": str(out.resolve()),
        "reference": str(p.voice.reference),
        "consent": p.voice.consent,
        # chatterbox-tts 0.1.7 applies Perth in both tts_turbo.py and mtl_tts.py generate().
        "watermark": "perth",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
