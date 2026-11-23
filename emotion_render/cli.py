"""emotion-render: a tagged script in, an acted wav out, with a record of how it was made.

Exit codes: 0 rendered (or checked), 2 refused before any model loaded, 1 failed.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import Refused, __version__, voices
from .engines import ENGINES, adapter, available_mb, choose_device
from .script import parse, validate

# Turbo is made for short clips (Sugi, CPU: about 2 minutes per 10 s of audio).
# Longer text needs chunking at sentence ends, which is not built yet.
MAX_CHARS = 400


def _args(argv):
    p = argparse.ArgumentParser(prog="emotion-render", description=__doc__.splitlines()[0])
    p.add_argument("--voice", required=True, help="voice id: the name of a .json in ~/.local/share/assembly-emotion-render/voices/")
    p.add_argument("--text", required=True, help="the script, tags in square brackets")
    p.add_argument("--out", type=Path, help="wav to write. A .json record is written beside it")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--exaggeration", type=float,
                   help="multilingual only: emotion intensity (Chatterbox default 0.5)")
    p.add_argument("--cfg-weight", type=float,
                   help="multilingual only: guidance weight, lower is slower and more deliberate (default 0.5)")
    p.add_argument("--check", action="store_true",
                   help="validate voice, script and settings, then exit without loading a model")
    p.add_argument("--no-memory-guard", action="store_true",
                   help="load even when this host has less free RAM than the engine needs")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = p.parse_args(argv)
    if not args.check and args.out is None:
        p.error("--out is required unless --check")
    return args


def _params(spec, defaults: dict, args) -> dict:
    given = {
        k: v
        for k, v in (("exaggeration", args.exaggeration), ("cfg_weight", args.cfg_weight))
        if v is not None
    }
    ignored = sorted(set(given) - spec.params)
    if ignored:
        raise Refused(
            f"{spec.name} does not take {', '.join(ignored)}. "
            f"A setting the engine would ignore is refused, the same as a tag (I1)"
        )
    return {**defaults, **given}


def _refuse(reason) -> int:
    print(f"REFUSED  {reason}", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    args = _args(argv)
    try:
        voice = voices.load(args.voice)
        spec = ENGINES[voice.engine]
        script = parse(args.text)
        validate(script, spec.tags, spec.name)
        if len(script.clean_text) > MAX_CHARS:
            raise Refused(
                f"{len(script.clean_text)} characters. Up to {MAX_CHARS} per clip until chunking is built"
            )
        params = _params(spec, voice.defaults, args)
    except Refused as e:
        return _refuse(e)

    plan = {
        "voice": voice.id,
        "person": voice.person,
        "lang": voice.lang,
        "engine": spec.name,
        "engine_text": script.engine_text,
        "clean_text": script.clean_text,
        "tags": list(script.tags),
        "params": params,
    }
    if args.check:
        print(json.dumps({"ok": True, "check": True, **plan}, ensure_ascii=False))
        return 0

    import torch
    import torchaudio

    cuda_total_mb = (
        torch.cuda.get_device_properties(0).total_memory // 2**20
        if torch.cuda.is_available()
        else None
    )
    try:
        device = choose_device(args.device, spec, cuda_total_mb)
    except Refused as e:
        return _refuse(e)

    if not args.no_memory_guard:
        need = spec.min_ram_mb[device]
        free = available_mb()
        if free is not None and free < need:
            return _refuse(
                f"{free} MB of RAM available, {spec.name} on {device} needs {need} MB. "
                f"Loading now could push this host into swap. Retry when it is quieter"
            )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    engine = adapter(spec.name)
    t0 = time.perf_counter()
    model = engine.load(device)
    t1 = time.perf_counter()
    wav = engine.generate(model, script.engine_text, str(voice.reference), voice.lang, params)
    t2 = time.perf_counter()

    wav = wav.detach().cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    sr = model.sr
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.out), wav, sr)

    audio_s = wav.shape[-1] / sr
    render_s = t2 - t1
    try:
        chatterbox_version = version("chatterbox-tts")
    except PackageNotFoundError:
        chatterbox_version = None

    record = {
        "ok": True,
        **plan,
        "script": script.raw,
        "chatterbox_version": chatterbox_version,
        "emotion_render_version": __version__,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "load_seconds": round(t1 - t0, 1),
        "render_seconds": round(render_s, 1),
        "audio_seconds": round(audio_s, 2),
        "seconds_per_audio_second": round(render_s / audio_s, 2) if audio_s else None,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "peak_vram_reserved_mb": (
            round(torch.cuda.max_memory_reserved() / 2**20) if device == "cuda" else None
        ),
        "out": str(args.out.resolve()),
        "reference": str(voice.reference),
        "consent": voice.consent,
        # chatterbox-tts 0.1.7 applies Perth in both tts_turbo.py and mtl_tts.py generate().
        "watermark": "perth",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
