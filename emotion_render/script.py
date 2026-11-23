"""Parse a tagged script and hold I1: a tag is consumed or refused."""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import Refused

TAG_RE = re.compile(r"\[([^\[\]]*)\]")


@dataclass(frozen=True)
class Script:
    raw: str
    engine_text: str          # tags normalized, handed to the engine as is
    clean_text: str           # tags removed, for transcripts and portal cards
    tags: tuple[str, ...]     # normalized, in order of appearance


def _normalize(tag: str) -> str:
    return " ".join(tag.lower().split())


def parse(raw: str) -> Script:
    if not raw or not raw.strip():
        raise Refused("the script is empty")

    tags: list[str] = []

    def keep(match: re.Match) -> str:
        tag = _normalize(match.group(1))
        if not tag:
            raise Refused("empty tag [] in the script")
        tags.append(tag)
        return f"[{tag}]"

    engine_text = TAG_RE.sub(keep, raw)

    leftover = TAG_RE.sub(" ", raw)
    if "[" in leftover or "]" in leftover:
        raise Refused(
            "a bracket in the script is not part of a whole tag. The engine would "
            "read it aloud, which is what this package exists to prevent (I1)"
        )

    return Script(
        raw=raw,
        engine_text=engine_text,
        clean_text=" ".join(leftover.split()),
        tags=tuple(tags),
    )


def validate(script: Script, allowed: frozenset[str], engine: str) -> None:
    unknown = sorted({t for t in script.tags if t not in allowed})
    if not unknown:
        return
    if allowed:
        accepted = " ".join(f"[{t}]" for t in sorted(allowed))
    else:
        accepted = "none, this engine takes no tags"
    raise Refused(
        f"{engine} does not act out {' '.join(f'[{t}]' for t in unknown)}. "
        f"Tags it acts out: {accepted}. A tag is consumed or refused, never spoken (I1)"
    )
