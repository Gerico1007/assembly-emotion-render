"""ffmpeg and ffprobe, the two outside programs the server uses. Tests replace this class."""
from __future__ import annotations

import subprocess
from pathlib import Path


class MediaError(RuntimeError):
    pass


def _run(args: list[str], timeout: int) -> str:
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise MediaError(f"{args[0]} could not run: {e}") from e
    if done.returncode != 0:
        raise MediaError(f"{args[0]} exit {done.returncode}: {done.stderr.strip()[:300]}")
    return done.stdout


class Media:
    def to_reference_wav(self, src: Path, dst: Path) -> None:
        """Any audio file in, mono 24 kHz WAV out (Chatterbox resamples to 24 kHz anyway)."""
        _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
              "-i", str(src), "-ac", "1", "-ar", "24000", str(dst)], timeout=120)

    def to_mp3(self, src: Path, dst: Path) -> None:
        _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
              "-i", str(src), "-codec:a", "libmp3lame", "-q:a", "4", str(dst)], timeout=120)

    def duration(self, path: Path) -> float:
        out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "csv=p=0", str(path)], timeout=30)
        try:
            return float(out.strip())
        except ValueError as e:
            raise MediaError(f"ffprobe gave no duration for {path.name}") from e
