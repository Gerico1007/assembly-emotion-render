"""The voice registry: whose voice it is, which engine speaks it, and their consent (I2).

Each person keeps their own voices in their own store, outside this repo:
~/.local/share/assembly-emotion-render/voices/<id>.json beside the reference clip.
The repo ships only a template, voices/example.json, which is refused as it stands.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import Refused
from .engines import ENGINES

REGISTRY_DIR = Path.home() / ".local" / "share" / "assembly-emotion-render" / "voices"
CONSENT_FIELDS = ("person", "given", "scope")
_VOICE_ID = re.compile(r"[a-z0-9][a-z0-9-]*")


@dataclass(frozen=True)
class Voice:
    id: str
    person: str
    lang: str
    engine: str
    reference: Path
    consent: dict
    defaults: dict = field(default_factory=dict)


def load(voice_id: str, registry_dir: Path | None = None) -> Voice:
    registry = registry_dir if registry_dir is not None else REGISTRY_DIR
    if not _VOICE_ID.fullmatch(voice_id):
        raise Refused(f"'{voice_id}' is not a voice id (lowercase letters, digits, dashes)")

    path = registry / f"{voice_id}.json"
    if not path.is_file():
        known = ", ".join(sorted(p.stem for p in registry.glob("*.json"))) or "none"
        raise Refused(
            f"no voice '{voice_id}' in {registry}. Known voices: {known}. "
            f"To add yours, see 'Add your own voice' in the README"
        )
    data = json.loads(path.read_text(encoding="utf-8"))

    person = data.get("person")
    engine = data.get("engine")
    lang = data.get("lang")
    if not person:
        raise Refused(f"{voice_id}: the entry does not name whose voice it is")
    if engine not in ENGINES:
        raise Refused(f"{voice_id}: unknown engine '{engine}'. Engines: {', '.join(sorted(ENGINES))}")
    spec = ENGINES[engine]
    if lang not in spec.languages:
        raise Refused(
            f"{voice_id}: {engine} does not speak '{lang}'. "
            f"It speaks: {', '.join(sorted(spec.languages))}"
        )

    consent = data.get("consent")
    if not isinstance(consent, dict) or any(not consent.get(k) for k in CONSENT_FIELDS):
        raise Refused(
            f"{voice_id}: no consent recorded for {person}'s voice. Fill in consent "
            f"with {', '.join(CONSENT_FIELDS)}. Only {person} can give it, and a "
            f"recording passed on by someone else is not consent (I2)"
        )
    if consent["person"] != person:
        raise Refused(
            f"{voice_id}: the consent is given by {consent['person']}, "
            f"but the voice is {person}'s (I2)"
        )

    defaults = data.get("defaults") or {}
    extra = sorted(set(defaults) - spec.params)
    if extra:
        raise Refused(f"{voice_id}: {engine} does not take {', '.join(extra)}")

    # A bare file name is looked up beside the entry, in the same store.
    reference = Path(data.get("reference") or "").expanduser()
    if not reference.is_absolute():
        reference = registry / reference
    if not reference.is_file() or reference.stat().st_size == 0:
        raise Refused(f"{voice_id}: reference clip missing or empty: {reference}")

    return Voice(
        id=voice_id,
        person=person,
        lang=lang,
        engine=engine,
        reference=reference,
        consent=consent,
        defaults=dict(defaults),
    )
