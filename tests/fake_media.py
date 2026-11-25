"""ffmpeg and ffprobe for tests: WAV in, WAV out, durations read with the wave module."""
from __future__ import annotations

import shutil
import wave
from pathlib import Path

from emotion_render.media import MediaError


class FakeMedia:
    """Accepts only WAV input, the way the tests write recordings."""

    def to_reference_wav(self, src: Path, dst: Path) -> None:
        try:
            with wave.open(str(src), "rb"):
                pass
        except (wave.Error, EOFError, OSError) as e:
            raise MediaError(f"not audio: {e}") from e
        shutil.copyfile(src, dst)

    def to_mp3(self, src: Path, dst: Path) -> None:
        shutil.copyfile(src, dst)

    def duration(self, path: Path) -> float:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()
