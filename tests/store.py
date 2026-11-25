"""A voice store for tests: real entries and clips, no person's voice."""
from __future__ import annotations

import json
from pathlib import Path

from fakes import make_wav


def consent(person):
    return {"person": person, "given": "2026-11-21", "scope": "tests"}


# id: (entry fields, reference clip name or None for a missing clip)
VOICES = {
    "ada-en": ({"person": "Ada", "lang": "en", "engine": "turbo", "consent": consent("Ada"),
                  "owner": "ada", "allowed_clients": ["bea"]}, "ada-en.wav"),
    "ada-fr": ({"person": "Ada", "lang": "fr", "engine": "multilingual", "consent": consent("Ada"),
                  "defaults": {"exaggeration": 0.5}, "owner": "ada", "allowed_clients": ["bea"]},
                 "ada-fr.wav"),
    "ada-private": ({"person": "Ada", "lang": "en", "engine": "turbo", "consent": consent("Ada"),
                       "owner": "ada", "allowed_clients": []}, "ada-private.wav"),
    "noor-fr": ({"person": "Noor", "lang": "fr", "engine": "multilingual",
                      "consent": consent("Noor"), "owner": "noor", "allowed_clients": []},
                     "noor-fr.wav"),
    "unowned": ({"person": "Eli", "lang": "en", "engine": "turbo", "consent": consent("Eli")}, "unowned.wav"),
    # Broken entries, owned by ada, one per refusal voices.validate can raise.
    "ada-noconsent": ({"person": "Ada", "lang": "en", "engine": "turbo", "consent": None,
                         "owner": "ada"}, "x1.wav"),
    "ada-mismatch": ({"person": "Ada", "lang": "en", "engine": "turbo", "consent": consent("Bea"),
                        "owner": "ada"}, "x2.wav"),
    "ada-badengine": ({"person": "Ada", "lang": "en", "engine": "bark", "consent": consent("Ada"),
                         "owner": "ada"}, "x3.wav"),
    "ada-badlang": ({"person": "Ada", "lang": "fr", "engine": "turbo", "consent": consent("Ada"),
                       "owner": "ada"}, "x4.wav"),
    "ada-badparam": ({"person": "Ada", "lang": "en", "engine": "turbo", "consent": consent("Ada"),
                        "defaults": {"exaggeration": 0.5}, "owner": "ada"}, "x5.wav"),
    "ada-noref": ({"person": "Ada", "lang": "en", "engine": "turbo", "consent": consent("Ada"),
                     "owner": "ada"}, None),
    "ada-noperson": ({"person": "", "lang": "en", "engine": "turbo", "consent": consent("Ada"),
                        "owner": "ada"}, "x6.wav"),
}


def make_store(store: Path) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    for vid, (entry, clip) in VOICES.items():
        if clip is not None:
            make_wav(store / clip, 10)
        (store / f"{vid}.json").write_text(json.dumps(dict(entry, reference=clip or "gone.wav")), encoding="utf-8")
    return store
