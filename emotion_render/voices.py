"""The voice registry: whose voice it is, which engine speaks it, and their consent (I2).

Each person keeps their own voices in their own store, outside this repo:
~/.local/share/assembly-emotion-render/voices/<id>.json beside the reference clip.
The repo ships only a template, voices/example.json, which is refused as it stands.

Two optional fields decide who may use a voice through the MCP server (I3):
`owner`, a client id, and `allowed_clients`, a list of client ids. The CLI does
not read them: whoever runs the CLI already reads the store directly.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import Refused
from .engines import ENGINES

REGISTRY_DIR = Path.home() / ".local" / "share" / "assembly-emotion-render" / "voices"
CONSENT_FIELDS = ("person", "given", "scope")
_VOICE_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
ID_RULE = "lowercase letters, digits, dashes"


def is_id(value) -> bool:
    """Voice ids and client ids follow the same rule."""
    return isinstance(value, str) and _VOICE_ID.fullmatch(value) is not None


@dataclass(frozen=True)
class Voice:
    id: str
    person: str
    lang: str
    engine: str
    reference: Path
    consent: dict
    defaults: dict = field(default_factory=dict)
    owner: str | None = None
    allowed_clients: tuple[str, ...] = ()


def _registry(registry_dir: Path | None) -> Path:
    return registry_dir if registry_dir is not None else REGISTRY_DIR


def ids(registry_dir: Path | None = None) -> list[str]:
    registry = _registry(registry_dir)
    return sorted(p.stem for p in registry.glob("*.json") if is_id(p.stem))


def read_entry(voice_id: str, registry_dir: Path | None = None) -> dict:
    """The raw entry, unvalidated. Refuses a malformed id or a missing entry."""
    registry = _registry(registry_dir)
    if not is_id(voice_id):
        raise Refused(f"'{voice_id}' is not a voice id ({ID_RULE})", "invalid_voice_id")
    path = registry / f"{voice_id}.json"
    if not path.is_file():
        known = ", ".join(sorted(p.stem for p in registry.glob("*.json"))) or "none"
        raise Refused(
            f"no voice '{voice_id}' in {registry}. Known voices: {known}. "
            f"To add yours, see 'Add your own voice' in the README",
            "unknown_voice",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def access(data: dict) -> tuple[str | None, tuple[str, ...]]:
    """(owner, allowed_clients) of an entry, malformed values treated as absent."""
    owner = data.get("owner")
    allowed = data.get("allowed_clients") or []
    return (
        owner if is_id(owner) else None,
        tuple(c for c in allowed if is_id(c)) if isinstance(allowed, list) else (),
    )


def check_consent(voice_id: str, person, consent) -> None:
    """I2, the same rule for an entry on disk and for one being added."""
    if not isinstance(consent, dict) or any(not consent.get(k) for k in CONSENT_FIELDS):
        raise Refused(
            f"{voice_id}: no consent recorded for {person}'s voice. Fill in consent "
            f"with {', '.join(CONSENT_FIELDS)}. Only {person} can give it, and a "
            f"recording passed on by someone else is not consent (I2)",
            "no_consent",
            required=list(CONSENT_FIELDS),
        )
    if consent["person"] != person:
        raise Refused(
            f"{voice_id}: the consent is given by {consent['person']}, "
            f"but the voice is {person}'s (I2)",
            "consent_mismatch",
        )


def check_engine(voice_id: str, engine, lang) -> None:
    if engine not in ENGINES:
        raise Refused(
            f"{voice_id}: unknown engine '{engine}'. Engines: {', '.join(sorted(ENGINES))}",
            "unknown_engine",
            accepted=sorted(ENGINES),
        )
    spec = ENGINES[engine]
    if lang not in spec.languages:
        raise Refused(
            f"{voice_id}: {engine} does not speak '{lang}'. "
            f"It speaks: {', '.join(sorted(spec.languages))}",
            "engine_language",
            accepted=sorted(spec.languages),
        )


def check_defaults(voice_id: str, engine: str, defaults) -> None:
    spec = ENGINES[engine]
    extra = sorted(set(defaults or {}) - spec.params)
    if extra:
        raise Refused(
            f"{voice_id}: {engine} does not take {', '.join(extra)}",
            "setting_not_taken",
            accepted=sorted(spec.params),
        )


def validate(voice_id: str, data: dict, registry_dir: Path | None = None) -> Voice:
    registry = _registry(registry_dir)
    person = data.get("person")
    engine = data.get("engine")
    lang = data.get("lang")
    if not person:
        raise Refused(f"{voice_id}: the entry does not name whose voice it is", "invalid_entry")
    check_engine(voice_id, engine, lang)
    consent = data.get("consent")
    check_consent(voice_id, person, consent)
    defaults = data.get("defaults") or {}
    check_defaults(voice_id, engine, defaults)

    # A bare file name is looked up beside the entry, in the same store.
    reference = Path(data.get("reference") or "").expanduser()
    if not reference.is_absolute():
        reference = registry / reference
    if not reference.is_file() or reference.stat().st_size == 0:
        raise Refused(f"{voice_id}: reference clip missing or empty: {reference}", "missing_reference")

    owner, allowed = access(data)
    return Voice(
        id=voice_id,
        person=person,
        lang=lang,
        engine=engine,
        reference=reference,
        consent=consent,
        defaults=dict(defaults),
        owner=owner,
        allowed_clients=allowed,
    )


def load(voice_id: str, registry_dir: Path | None = None) -> Voice:
    return validate(voice_id, read_entry(voice_id, registry_dir), registry_dir)


def write_entry(voice_id: str, data: dict, registry_dir: Path | None = None, *, replace: bool = False) -> Path:
    """Write an entry atomically. Without `replace`, an existing id is refused."""
    registry = _registry(registry_dir)
    if not is_id(voice_id):
        raise Refused(f"'{voice_id}' is not a voice id ({ID_RULE})", "invalid_voice_id")
    registry.mkdir(parents=True, exist_ok=True)
    path = registry / f"{voice_id}.json"
    if path.exists() and not replace:
        raise Refused(
            f"a voice '{voice_id}' already exists. A voice is never overwritten: remove it, then add it again",
            "voice_exists",
        )
    fd, tmp = tempfile.mkstemp(dir=registry, prefix=f".{voice_id}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path
