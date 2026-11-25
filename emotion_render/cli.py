"""emotion-render: a tagged script in, an acted wav out, with a record of how it was made.

Exit codes: 0 rendered (or checked), 2 refused before any model loaded, 1 failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import Refused, __version__, core
from .gpulock import GpuLock

MAX_CHARS = core.MAX_CHARS


def make_backend():
    """Replaced in tests by a fake with the same methods."""
    from .backend import TorchBackend

    return TorchBackend()


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


def _refuse(reason) -> int:
    print(f"REFUSED  {reason}", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    args = _args(argv)
    try:
        plan = core.plan(args.voice, args.text, args.exaggeration, args.cfg_weight)
    except Refused as e:
        return _refuse(e)

    if args.check:
        print(json.dumps({"ok": True, "check": True, **plan.as_dict()}, ensure_ascii=False))
        return 0

    backend = make_backend()
    # Held until this process exits. The kernel drops it then, however the exit happens.
    lock = GpuLock()
    try:
        device = core.claim_device(args.device, plan, backend, lock)
        if not args.no_memory_guard:
            core.guard_memory(plan.spec, device, backend)
    except Refused as e:
        lock.release()
        return _refuse(e)

    model, load_s = core.load_model(plan.spec, device, backend)
    record = core.render(plan, model, device=device, out=args.out, backend=backend, load_seconds=load_s)
    args.out.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
