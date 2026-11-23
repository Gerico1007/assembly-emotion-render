"""The engines this package drives, and what each one consumes and needs.

Model code is imported only when a render starts, so validation and --check
never load torch.

The RAM and VRAM figures come from A4 on eury, 2026-11-23 (GTX 1650, 3.63 GiB
usable, torch 2.6.0+cu124, chatterbox-tts 0.1.7). The sidecar JSON of each run is
in out/ and the summary is in the phase-2 brief.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

from .. import Refused


@dataclass(frozen=True)
class EngineSpec:
    name: str
    module: str
    languages: frozenset[str]
    tags: frozenset[str]          # in-text tags the model acts out (I1)
    params: frozenset[str]        # generation settings the model takes
    min_ram_mb: dict[str, int]    # host RAM that must be free before loading, per device
    min_vram_mb: int | None       # GPU memory it needs. None: it has fit on no GPU measured


# Turbo's paralinguistic tags. The Chatterbox README names [cough] [laugh]
# [chuckle] "and more". [chuckle] and [sigh] were consumed in Sugi's runs and in
# A4. The rest come from the list Sugi relayed. The model's tokenizer files,
# where the full list lives, have not been read yet.
TURBO_TAGS = frozenset({
    "laugh", "chuckle", "sigh", "gasp", "cough",
    "clear throat", "sniff", "groan", "shush",
})

# Same 23 ids as SUPPORTED_LANGUAGES in chatterbox-tts 0.1.7 mtl_tts.py.
MULTILINGUAL_LANGUAGES = frozenset({
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
})

ENGINES: dict[str, EngineSpec] = {
    # A4 peak RSS: 6427 MB on CPU, 4854 MB on CUDA. Peak VRAM reserved: 3350 MB.
    # Turbo's generate() takes exaggeration and cfg_weight but ignores them with
    # a warning, so params is empty and the CLI refuses them (I1).
    "turbo": EngineSpec(
        name="turbo",
        module="emotion_render.engines.turbo",
        languages=frozenset({"en"}),
        tags=TURBO_TAGS,
        params=frozenset(),
        min_ram_mb={"cpu": 7000, "cuda": 5500},
        min_vram_mb=3500,
    ),
    # A4 peak RSS: 6938 MB on CPU. On CUDA it loaded, then ran out of GPU memory
    # during generation, both runs (3.63 GiB card, 26 MiB left free).
    # Emotion is one exaggeration level per call. It takes no tags.
    "multilingual": EngineSpec(
        name="multilingual",
        module="emotion_render.engines.multilingual",
        languages=MULTILINGUAL_LANGUAGES,
        tags=frozenset(),
        params=frozenset({"exaggeration", "cfg_weight"}),
        min_ram_mb={"cpu": 7500},
        min_vram_mb=None,
    ),
}


def choose_device(requested: str, spec: EngineSpec, cuda_total_mb: int | None) -> str:
    """The device to render on. `cuda_total_mb` is None when torch sees no CUDA device.

    "auto" takes the GPU only where the engine is known to fit. An explicit
    "cuda" that is known not to fit is refused before a model load is spent on it.
    """
    fits = (
        cuda_total_mb is not None
        and spec.min_vram_mb is not None
        and cuda_total_mb >= spec.min_vram_mb
    )
    if requested == "auto":
        return "cuda" if fits else "cpu"
    if requested == "cuda" and not fits:
        if cuda_total_mb is None:
            raise Refused("--device cuda was asked, but torch sees no CUDA device")
        if spec.min_vram_mb is None:
            raise Refused(
                f"{spec.name} has not fit on any GPU measured so far (it ran out of "
                f"memory on a 3.63 GiB GTX 1650). This card has {cuda_total_mb} MB. Use --device cpu"
            )
        raise Refused(
            f"{spec.name} needs {spec.min_vram_mb} MB of GPU memory, this card has "
            f"{cuda_total_mb} MB. Use --device cpu"
        )
    return requested


def adapter(name: str):
    """The module that loads and runs engine `name`. Importing it imports torch."""
    _require_watermarker()
    return importlib.import_module(ENGINES[name].module)


def _require_watermarker() -> None:
    # Both engines watermark every output with Perth. When Perth's import fails,
    # perth/__init__.py sets PerthImplicitWatermarker to None without a word, and
    # Chatterbox dies later on "'NoneType' object is not callable". Name the real
    # cause here, before a model load is spent on it.
    try:
        from perth.perth_net.perth_net_implicit.perth_watermarker import (  # noqa: F401
            PerthImplicitWatermarker,
        )
    except ImportError as e:
        raise RuntimeError(
            f"the Perth watermarker cannot load: {e}. Chatterbox watermarks every "
            f"output, so nothing can render. Reinstall this package's dependencies: "
            f"uv pip install --python .venv/bin/python -e '.[test]'"
        ) from e


def available_mb() -> int | None:
    """MemAvailable from /proc/meminfo, in MB. None where it cannot be read."""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None
