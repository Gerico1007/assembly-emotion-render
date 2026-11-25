"""What the MCP server does, without the transport: voices, renders, the queue, retention.

Every method takes a `Caller`, the identity the transport established, and applies
I3 with it: a client sees, renders and reads only what it owns or was allowed.
Rules that need no transport (I1, I2, settings, devices) come from `core`, so they
are the CLI's rules.

Concurrency (D12): one render at a time, because a Chatterbox model keeps the
reference voice on the model object (`prepare_conditionals`) and two calls at once
would mix voices. Renders run in a worker thread so the server keeps answering
`check`, `list_voices` and progress while one is running.
"""
from __future__ import annotations

import base64
import binascii
import json
import math
import os
import re
import secrets
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import anyio

from . import Refused, core, voices
from .engines import ENGINES, choose_device
from .media import MediaError

SHARE = Path.home() / ".local" / "share" / "assembly-emotion-render"
STATE = Path.home() / ".local" / "state" / "assembly-emotion-render"

FOREVER = "forever"
MAX_KEEP_DAYS = 3650
DEFAULT_KEEP_DAYS = 7
MAX_WAIT_SECONDS = 120
MAX_WAITING = 3                       # renders waiting behind the one running
MODEL_IDLE_SECONDS = 600              # a model not used for this long is unloaded (not the warm one)
SWEEP_SECONDS = 3600
PROGRESS_SECONDS = 5
MIN_CLIP_SECONDS = 5.0                # Chatterbox asserts the reference is longer than 5 s
MAX_CLIP_SECONDS = 30.0               # past ~15 s Chatterbox uses nothing more (F13)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
FORMATS = {"mp3": "audio/mpeg", "wav": "audio/wav", "json": "application/json"}

RENDER_ID = re.compile(r"r-\d{8}-\d{6}-[0-9a-f]{6}")
INBOX_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True)
class Caller:
    client_id: str
    device: str | None = None
    local: bool = False     # stdio: the person's own process, which sees every voice in their store
    on_host: bool = False   # loopback or stdio: may be given file paths on this machine


@dataclass(frozen=True)
class Paths:
    store: Path
    renders: Path
    inbox: Path
    audit: Path
    tmp: Path

    @classmethod
    def default(cls) -> "Paths":
        return cls(
            store=voices.REGISTRY_DIR,
            renders=SHARE / "renders",
            inbox=SHARE / "inbox",
            audit=STATE / "audit.jsonl",
            tmp=STATE / "tmp",
        )


class AuditLog:
    """One JSON line per call: who, from which device, which tool and voice, what happened.

    Never the script text, never a token.
    """

    def __init__(self, path: Path, clock: Callable[[], float] = time.time):
        self.path = path
        self._clock = clock
        self._lock = threading.Lock()

    def write(self, **fields) -> None:
        line = {"time": datetime.fromtimestamp(self._clock(), timezone.utc).isoformat(timespec="seconds")}
        line.update({k: v for k, v in fields.items() if v is not None})
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def lines(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]


@dataclass
class Job:
    render_id: str
    caller: Caller
    voice_id: str
    text: str
    exaggeration: float | None
    cfg_weight: float | None
    device: str
    format: str
    keep_days: int | str
    status: str = "queued"           # queued, running, done, failed
    record: dict | None = None
    error: Refused | None = None
    done: anyio.Event = field(default_factory=anyio.Event)
    finished_at: float | None = None


@dataclass
class _Loaded:
    model: Any
    last_used: float


def parse_keep_days(value, *, allow_zero: bool, default=DEFAULT_KEEP_DAYS):
    """An int from 1 (0 with allow_zero) to 3650, or "forever". None means the default."""
    if value is None:
        return default
    if value == FOREVER:
        return FOREVER
    low = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= MAX_KEEP_DAYS:
        raise Refused(
            f"keep_days must be a whole number from {low} to {MAX_KEEP_DAYS}, or \"forever\"",
            "bad_retention",
            accepted={"min": low, "max": MAX_KEEP_DAYS, "or": FOREVER},
        )
    return value


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def _expires_at(created_iso: str, keep_days) -> str | None:
    if keep_days == FOREVER:
        return None
    created = datetime.fromisoformat(created_iso)
    return (created + timedelta(days=keep_days)).isoformat(timespec="seconds")


class Service:
    def __init__(
        self,
        paths: Paths,
        backend,
        media,
        lock,
        *,
        clock: Callable[[], float] = time.time,
        max_waiting: int = MAX_WAITING,
        model_idle_seconds: float = MODEL_IDLE_SECONDS,
        sweep_seconds: float = SWEEP_SECONDS,
        progress_seconds: float = PROGRESS_SECONDS,
    ):
        self.paths = paths
        self.backend = backend
        self.media = media
        self.lock = lock
        self.clock = clock
        self.audit = AuditLog(paths.audit, clock)
        self.max_waiting = max_waiting
        self.model_idle_seconds = model_idle_seconds
        self.sweep_seconds = sweep_seconds
        self.progress_seconds = progress_seconds
        self._jobs: dict[str, Job] = {}
        self._models: dict[tuple[str, str], _Loaded] = {}
        self._model_lock = threading.Lock()
        self._warm_key: tuple[str, str] | None = None
        self._cuda_total: int | None | bool = False   # False: not asked yet
        self._send = self._recv = None
        self._last_activity = clock()
        self._last_sweep = 0.0

    # -- lifecycle ---------------------------------------------------------

    @asynccontextmanager
    async def running(self, warm: str | None = None):
        """Start the render worker and housekeeping. `warm`: an engine to load at start, GPU only."""
        self._send, self._recv = anyio.create_memory_object_stream[Job](math.inf)
        await anyio.to_thread.run_sync(self.sweep)
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(self._worker, warm)
                tg.start_soon(self._housekeeping)
                try:
                    yield self
                finally:
                    tg.cancel_scope.cancel()
        finally:
            with self._model_lock:
                for (engine, device), loaded in list(self._models.items()):
                    self.backend.unload(loaded.model, device)
                self._models.clear()
            self.lock.release()

    def touch(self) -> None:
        self._last_activity = self.clock()

    def idle(self) -> bool:
        return not any(j.status in ("queued", "running") for j in self._jobs.values())

    def idle_for(self) -> float:
        return self.clock() - self._last_activity

    def loaded(self) -> list[tuple[str, str]]:
        return sorted(self._models)

    # -- access (I3) -------------------------------------------------------

    def _may_see(self, caller: Caller, data: dict) -> bool:
        if caller.local:
            return True
        owner, allowed = voices.access(data)
        return caller.client_id == owner or caller.client_id in allowed

    def _visible_ids(self, caller: Caller) -> list[str]:
        out = []
        for vid in voices.ids(self.paths.store):
            try:
                if self._may_see(caller, voices.read_entry(vid, self.paths.store)):
                    out.append(vid)
            except (Refused, ValueError, OSError):
                continue
        return out

    def _entry(self, caller: Caller, voice_id: str) -> dict:
        """The raw entry of a voice this caller may use. Any other voice is `unknown_voice`,
        so its existence is not revealed either."""
        if caller.local:
            return voices.read_entry(voice_id, self.paths.store)
        if not voices.is_id(voice_id):
            raise Refused(f"'{voice_id}' is not a voice id ({voices.ID_RULE})", "invalid_voice_id")
        try:
            data = voices.read_entry(voice_id, self.paths.store)
        except Refused as e:
            if e.code != "unknown_voice":
                raise
            data = None
        if data is None or not self._may_see(caller, data):
            known = ", ".join(self._visible_ids(caller)) or "none"
            raise Refused(f"no voice '{voice_id}' you may use. Your voices: {known}", "unknown_voice")
        return data

    def _voice(self, caller: Caller, voice_id: str) -> voices.Voice:
        data = self._entry(caller, voice_id)
        try:
            return voices.validate(voice_id, data, self.paths.store)
        except Refused as e:
            if e.code == "missing_reference" and not caller.on_host:
                raise Refused(f"{voice_id}: the reference clip is missing or empty on the server",
                              "missing_reference") from e
            raise

    # -- devices -----------------------------------------------------------

    def _cuda_total_mb(self) -> int | None:
        if self._cuda_total is False:
            self._cuda_total = self.backend.cuda_total_mb()
        return self._cuda_total

    def _preview_device(self, requested: str, p: core.Plan) -> str:
        """The device a render would use, refusals included, without taking the lock."""
        if requested not in ("auto", "cpu", "cuda"):
            raise Refused(f"device must be auto, cpu or cuda, not '{requested}'", "bad_device",
                          accepted=["auto", "cpu", "cuda"])
        holder = None if self.lock.held else self.lock.holder()
        return choose_device(requested, p.spec, self._cuda_total_mb(), holder, len(p.script.clean_text))

    # -- tools -------------------------------------------------------------

    def list_voices(self, caller: Caller) -> dict:
        out = []
        for vid in self._visible_ids(caller):
            data = voices.read_entry(vid, self.paths.store)
            owner, allowed = voices.access(data)
            spec = ENGINES.get(data.get("engine"))
            item = {
                "id": vid,
                "person": data.get("person"),
                "lang": data.get("lang"),
                "engine": data.get("engine"),
                "tags": sorted(spec.tags) if spec else [],
                "params": sorted(spec.params) if spec else [],
                "owner": owner,
            }
            if caller.local or caller.client_id == owner:
                item["allowed_clients"] = list(allowed)
            try:
                voices.validate(vid, data, self.paths.store)
                item["ready"] = True
            except Refused as e:
                item["ready"] = False
                item["refused"] = e.code
            out.append(item)
        return {"voices": out}

    def check(self, caller: Caller, voice: str, text: str, exaggeration=None, cfg_weight=None,
              device: str = "auto") -> dict:
        p = core.plan(voice, text, exaggeration, cfg_weight, voice=self._voice(caller, voice))
        return {"ok": True, "check": True, **p.as_dict(), "device": self._preview_device(device, p)}

    async def render(
        self,
        caller: Caller,
        voice: str,
        text: str,
        exaggeration=None,
        cfg_weight=None,
        device: str = "auto",
        format: str = "mp3",
        wait_seconds: float = 60,
        keep_days=None,
        on_progress: Callable[[float, float | None, str | None], Awaitable[None]] | None = None,
    ) -> dict:
        if format not in ("mp3", "wav"):
            raise Refused(f"format must be mp3 or wav, not '{format}'", "bad_format", accepted=["mp3", "wav"])
        wait = self._wait_seconds(wait_seconds)
        keep = parse_keep_days(keep_days, allow_zero=False)
        p = core.plan(voice, text, exaggeration, cfg_weight, voice=self._voice(caller, voice))
        self._preview_device(device, p)
        waiting = sum(1 for j in self._jobs.values() if j.status == "queued")
        if waiting >= self.max_waiting:
            raise Refused(
                f"{waiting} renders are already waiting behind the one running. Try again shortly",
                "busy",
                waiting=waiting,
            )
        now = self.clock()
        job = Job(
            render_id=f"r-{datetime.fromtimestamp(now, timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}",
            caller=caller, voice_id=voice, text=text, exaggeration=exaggeration, cfg_weight=cfg_weight,
            device=device, format=format, keep_days=keep,
        )
        self._jobs[job.render_id] = job
        self.touch()
        await self._send.send(job)
        return await self._await(job, wait, on_progress)

    async def get_render(self, caller: Caller, render_id: str, wait_seconds: float = 0, on_progress=None) -> dict:
        wait = self._wait_seconds(wait_seconds)
        job = self._own_job(caller, render_id)
        if job is not None:
            return await self._await(job, wait, on_progress)
        return self._done_payload(caller, self._record(caller, render_id))

    def keep_render(self, caller: Caller, render_id: str, keep_days) -> dict:
        keep = parse_keep_days(keep_days, allow_zero=True, default=None)
        if keep is None:
            raise Refused("keep_days is required", "bad_retention")
        job = self._own_job(caller, render_id)
        if job is not None and job.status in ("queued", "running"):
            if keep == 0:
                raise Refused("this render is still running. Delete it once it is done", "render_running")
            job.keep_days = keep
            return {"render_id": render_id, "status": "running", "keep_days": keep}
        record = self._record(caller, render_id)
        if keep == 0:
            self._delete_render(self._render_dir(caller) / render_id)
            self._jobs.pop(render_id, None)
            return {"render_id": render_id, "deleted": True}
        record["keep_days"] = keep
        record["expires_at"] = _expires_at(record["created_at"], keep)
        self._write_record(self._render_dir(caller) / f"{render_id}.json", record)
        return {"render_id": render_id, "keep_days": keep, "expires_at": record["expires_at"]}

    def read_render_file(self, caller: Caller, render_id: str, ext: str) -> tuple[bytes, str]:
        if ext not in FORMATS:
            raise Refused(f"no .{ext} file for a render", "unknown_render")
        record = self._record(caller, render_id)
        if ext == "json":
            return json.dumps(self._public_record(caller, record), ensure_ascii=False).encode(), FORMATS[ext]
        path = self._render_dir(caller) / f"{render_id}.{ext}"
        if not path.is_file():
            raise Refused(f"render {render_id} has no .{ext} file", "unknown_render")
        return path.read_bytes(), FORMATS[ext]

    def add_voice(self, caller: Caller, id: str, person: str, lang: str, engine: str, reference: dict,
                  consent: dict, defaults: dict | None = None, allowed_clients: list | None = None) -> dict:
        store = self.paths.store
        if not voices.is_id(id):
            raise Refused(f"'{id}' is not a voice id ({voices.ID_RULE})", "invalid_voice_id")
        if (store / f"{id}.json").exists() or (store / f"{id}.wav").exists():
            raise Refused(
                f"a voice '{id}' already exists. A voice is never overwritten: remove it, then add it again",
                "voice_exists",
            )
        if not isinstance(person, str) or not person.strip():
            raise Refused(f"{id}: the entry does not name whose voice it is", "invalid_entry")
        voices.check_engine(id, engine, lang)
        voices.check_defaults(id, engine, defaults or {})
        voices.check_consent(id, person, consent)
        allowed = allowed_clients or []
        if not isinstance(allowed, list) or not all(voices.is_id(c) for c in allowed):
            raise Refused(f"allowed_clients must be a list of client ids ({voices.ID_RULE})", "invalid_entry")

        src, from_inbox = self._reference_source(caller, reference)
        self.paths.tmp.mkdir(parents=True, exist_ok=True)
        store.mkdir(parents=True, exist_ok=True)
        tmp_wav = store / f".{id}.{secrets.token_hex(4)}.wav"
        try:
            try:
                self.media.to_reference_wav(src, tmp_wav)
                seconds = self.media.duration(tmp_wav)
            except MediaError as e:
                raise Refused(f"the reference recording could not be read as audio: {e}", "clip_unreadable") from e
            if seconds <= MIN_CLIP_SECONDS:
                raise Refused(
                    f"the reference recording is {seconds:.1f} s. Chatterbox needs more than "
                    f"{MIN_CLIP_SECONDS:.0f} s, about 10 s of clean speech is best",
                    "clip_too_short", seconds=round(seconds, 1), min_seconds=MIN_CLIP_SECONDS,
                )
            if seconds > MAX_CLIP_SECONDS:
                raise Refused(
                    f"the reference recording is {seconds:.1f} s. Up to {MAX_CLIP_SECONDS:.0f} s is "
                    f"accepted, and past about 15 s Chatterbox uses nothing more",
                    "clip_too_long", seconds=round(seconds, 1), max_seconds=MAX_CLIP_SECONDS,
                )
            entry = {
                "person": person,
                "lang": lang,
                "engine": engine,
                "reference": f"{id}.wav",
                "reference_seconds": round(seconds, 1),
                **({"defaults": dict(defaults)} if defaults else {}),
                "consent": {
                    **consent,
                    "asserted_by": caller.client_id,
                    "recorded_at": _iso(self.clock()),
                },
                "owner": caller.client_id,
                "allowed_clients": list(allowed),
            }
            os.replace(tmp_wav, store / f"{id}.wav")
            try:
                voices.write_entry(id, entry, store)
                voices.validate(id, entry, store)
            except BaseException:
                (store / f"{id}.wav").unlink(missing_ok=True)
                (store / f"{id}.json").unlink(missing_ok=True)
                raise
        finally:
            tmp_wav.unlink(missing_ok=True)
            if not from_inbox:
                src.unlink(missing_ok=True)
        if from_inbox:
            src.unlink(missing_ok=True)
        return {"id": id, **entry, "inbox_file_removed": from_inbox}

    def remove_voice(self, caller: Caller, voice_id: str) -> dict:
        data = self._entry(caller, voice_id)
        owner, _ = voices.access(data)
        if not caller.local and caller.client_id != owner:
            raise Refused(f"only {voice_id}'s owner can remove it", "not_allowed")
        store = self.paths.store
        reference = Path(data.get("reference") or "").expanduser()
        if not reference.is_absolute():
            reference = store / reference
        (store / f"{voice_id}.json").unlink(missing_ok=True)

        # The clip goes too, if it lives in the store and no other entry uses it.
        reference_deleted = False
        try:
            inside = reference.resolve().parent == store.resolve()
        except OSError:
            inside = False
        if inside and reference.is_file():
            shared = False
            for other in voices.ids(store):
                ref = Path(voices.read_entry(other, store).get("reference") or "").expanduser()
                ref = ref if ref.is_absolute() else store / ref
                shared = shared or ref.resolve() == reference.resolve()
            if not shared:
                reference.unlink()
                reference_deleted = True

        # Every render of the voice, for every client, pinned or not (D17, I2).
        deleted = 0
        if self.paths.renders.is_dir():
            for record_path in self.paths.renders.glob("*/*.json"):
                try:
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if record.get("voice") == voice_id:
                    self._delete_render(record_path.with_suffix(""))
                    self._jobs.pop(record_path.stem, None)
                    deleted += 1
        for job in self._jobs.values():
            if job.voice_id == voice_id and job.status == "queued":
                job.status = "failed"
                job.error = Refused(f"{voice_id} was removed before this render started", "unknown_voice")
                job.done.set()
        return {
            "removed": voice_id,
            "reference_deleted": reference_deleted,
            "renders_deleted": deleted,
            "note": "copies a client already downloaded are out of this server's reach",
        }

    # -- renders on disk ---------------------------------------------------

    def _wait_seconds(self, value) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= MAX_WAIT_SECONDS:
            raise Refused(f"wait_seconds must be from 0 to {MAX_WAIT_SECONDS}", "bad_wait",
                          accepted={"min": 0, "max": MAX_WAIT_SECONDS})
        return float(value)

    def _render_dir(self, caller: Caller) -> Path:
        return self.paths.renders / caller.client_id

    def _own_job(self, caller: Caller, render_id: str) -> Job | None:
        if not isinstance(render_id, str) or not RENDER_ID.fullmatch(render_id):
            raise Refused(f"'{render_id}' is not a render id", "unknown_render")
        job = self._jobs.get(render_id)
        if job is None or job.caller.client_id != caller.client_id:
            return None
        return job if job.status in ("queued", "running", "failed") else None

    def _record(self, caller: Caller, render_id: str) -> dict:
        """The record of one of this caller's unexpired renders."""
        if not isinstance(render_id, str) or not RENDER_ID.fullmatch(render_id):
            raise Refused(f"'{render_id}' is not a render id", "unknown_render")
        path = self._render_dir(caller) / f"{render_id}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = None
        if record is None or self._expired(record):
            raise Refused(f"no render {render_id} of yours (unknown, deleted or expired)", "unknown_render")
        return record

    def _expired(self, record: dict) -> bool:
        expires = record.get("expires_at")
        return expires is not None and datetime.fromisoformat(expires).timestamp() <= self.clock()

    @staticmethod
    def _delete_render(stem: Path) -> None:
        for ext in FORMATS:
            stem.with_suffix(f".{ext}").unlink(missing_ok=True)

    @staticmethod
    def _write_record(path: Path, record: dict) -> None:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def sweep(self) -> int:
        """Delete expired renders. Runs at start and hourly. Returns how many were deleted."""
        self._last_sweep = self.clock()
        deleted = 0
        if not self.paths.renders.is_dir():
            return 0
        for record_path in self.paths.renders.glob("*/*.json"):
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if self._expired(record):
                self._delete_render(record_path.with_suffix(""))
                self._jobs.pop(record_path.stem, None)
                deleted += 1
        return deleted

    def _public_record(self, caller: Caller, record: dict) -> dict:
        if caller.on_host:
            return dict(record)
        return {k: v for k, v in record.items() if k not in ("out", "reference")}

    def _done_payload(self, caller: Caller, record: dict) -> dict:
        fmt = record.get("format", "mp3")
        path = self._render_dir(caller) / f"{record['render_id']}.{fmt}"
        payload = {
            "status": "done",
            "render_id": record["render_id"],
            "record": self._public_record(caller, record),
            "audio": {
                "uri": f"emotion-render://renders/{record['render_id']}.{fmt}",
                "mime_type": FORMATS[fmt],
                "bytes": path.stat().st_size if path.is_file() else 0,
            },
        }
        if caller.on_host:
            payload["path"] = str(path)
        return payload

    async def _await(self, job: Job, wait: float, on_progress) -> dict:
        start = self.clock()
        with anyio.move_on_after(wait):
            while not job.done.is_set():
                with anyio.move_on_after(self.progress_seconds):
                    await job.done.wait()
                if on_progress is not None and not job.done.is_set():
                    await on_progress(self.clock() - start, None, self._status_message(job))
        if job.status == "failed":
            raise job.error
        if job.status == "done":
            return self._done_payload(job.caller, job.record)
        return {"status": "running", "render_id": job.render_id, "queue_position": self._position(job)}

    def _position(self, job: Job) -> int:
        if job.status != "queued":
            return 0
        queued = [j for j in self._jobs.values() if j.status == "queued"]
        return 1 + queued.index(job)

    def _status_message(self, job: Job) -> str:
        if job.status == "queued":
            return f"waiting, position {self._position(job)}"
        return "rendering"

    # -- the worker ----------------------------------------------------------

    async def _worker(self, warm: str | None) -> None:
        if warm:
            await anyio.to_thread.run_sync(self._warm, warm)
        async for job in self._recv:
            if job.status != "queued":
                continue
            job.status = "running"
            self.touch()
            try:
                job.record = await anyio.to_thread.run_sync(self._execute, job)
                job.status = "done"
            except Refused as e:
                job.status, job.error = "failed", e
            except Exception as e:  # a crash in one render must not stop the worker
                job.status = "failed"
                job.error = Refused(f"the render failed: {type(e).__name__}: {e}", "render_failed")
            job.finished_at = self.clock()
            job.done.set()
            self.touch()
            self.audit.write(client=job.caller.client_id, device=job.caller.device, tool="render.finished",
                             voice=job.voice_id, render_id=job.render_id,
                             outcome="done" if job.status == "done" else job.error.code)

    def _warm(self, engine: str) -> None:
        """Load `engine` at start, on the GPU only. Without the GPU it loads on first use instead."""
        spec = ENGINES[engine]
        cuda = self._cuda_total_mb()
        if cuda is None or spec.min_vram_mb is None or cuda < spec.min_vram_mb or not self.lock.acquire():
            return
        try:
            with self._model_lock:
                self._model(spec, "cuda")
            self._warm_key = (engine, "cuda")
        except Refused:
            self._release_gpu_if_unused()

    def _model(self, spec, device: str):
        """(model, load_seconds), loading it if needed. Call with _model_lock held."""
        key = (spec.name, device)
        loaded = self._models.get(key)
        if loaded is not None:
            loaded.last_used = self.clock()
            return loaded.model, 0.0
        core.guard_memory(spec, device, self.backend)
        model, seconds = core.load_model(spec, device, self.backend)
        self._models[key] = _Loaded(model, self.clock())
        return model, seconds

    def _release_gpu_if_unused(self) -> None:
        if not any(device == "cuda" for _, device in self._models):
            self.lock.release()

    def _execute(self, job: Job) -> dict:
        caller = job.caller
        voice = self._voice(caller, job.voice_id)        # again: the voice or its access may have changed
        p = core.plan(job.voice_id, job.text, job.exaggeration, job.cfg_weight, voice=voice)
        out_dir = self._render_dir(caller)
        out_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(out_dir, 0o700)
        wav = out_dir / f"{job.render_id}.wav"
        with self._model_lock:
            try:
                device = core.claim_device(job.device, p, self.backend, self.lock)
                model, load_s = self._model(p.spec, device)
            except Refused:
                self._release_gpu_if_unused()
                raise
            try:
                record = core.render(p, model, device=device, out=wav, backend=self.backend, load_seconds=load_s)
            except Exception as e:
                if self.backend.is_out_of_memory(e):
                    self.backend.free_cache(device)
                    wav.unlink(missing_ok=True)
                    raise Refused(
                        f"the GPU ran out of memory on this {len(p.script.clean_text)}-character script. "
                        f"Try device cpu or a shorter script",
                        "gpu_out_of_memory", chars=len(p.script.clean_text),
                    ) from e
                raise
        try:
            self.media.to_mp3(wav, wav.with_suffix(".mp3"))
        except MediaError as e:
            self._delete_render(out_dir / job.render_id)
            raise Refused(f"the mp3 could not be made: {e}", "render_failed") from e
        record.update({
            "render_id": job.render_id,
            "client": caller.client_id,
            "format": job.format,
            "keep_days": job.keep_days,
            "expires_at": _expires_at(record["created_at"], job.keep_days),
        })
        self._write_record(out_dir / f"{job.render_id}.json", record)
        return record

    async def _housekeeping(self) -> None:
        while True:
            await anyio.sleep(min(30.0, self.sweep_seconds, self.model_idle_seconds))
            await anyio.to_thread.run_sync(self.tidy)

    def tidy(self) -> None:
        """Unload idle models (not the warm one), sweep when due, forget old finished jobs."""
        now = self.clock()
        with self._model_lock:
            for key, loaded in list(self._models.items()):
                if key != self._warm_key and now - loaded.last_used >= self.model_idle_seconds:
                    self.backend.unload(loaded.model, key[1])
                    del self._models[key]
            self._release_gpu_if_unused()
        if now - self._last_sweep >= self.sweep_seconds:
            self.sweep()
        for rid, job in list(self._jobs.items()):
            if job.finished_at is not None and now - job.finished_at > 3600:
                del self._jobs[rid]

    # -- add_voice helpers -------------------------------------------------

    def _reference_source(self, caller: Caller, reference) -> tuple[Path, bool]:
        """(path to the recording, whether it came from the inbox)."""
        if not isinstance(reference, dict) or len(set(reference) & {"inbox_file", "data_base64"}) != 1:
            raise Refused(
                'reference must be {"inbox_file": "<name>"} or {"data_base64": "...", "mime_type": "audio/..."}',
                "invalid_entry",
            )
        if "inbox_file" in reference:
            name = reference["inbox_file"]
            if not isinstance(name, str) or not INBOX_NAME.fullmatch(name):
                raise Refused("inbox_file must be a bare file name in your inbox, no path", "invalid_entry")
            path = self.paths.inbox / caller.client_id / name
            if not path.is_file():
                raise Refused(f"no file {name} in your inbox", "missing_reference",
                              inbox=f"inbox/{caller.client_id}/")
            return path, True
        data = reference["data_base64"]
        if not isinstance(data, str) or len(data) > MAX_UPLOAD_BYTES * 4 // 3 + 4:
            raise Refused(f"the recording is over {MAX_UPLOAD_BYTES // 2**20} MB. Use the inbox for a larger file",
                          "clip_too_large", max_bytes=MAX_UPLOAD_BYTES)
        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as e:
            raise Refused("data_base64 is not valid base64", "clip_unreadable") from e
        if len(raw) > MAX_UPLOAD_BYTES:
            raise Refused(f"the recording is over {MAX_UPLOAD_BYTES // 2**20} MB. Use the inbox for a larger file",
                          "clip_too_large", max_bytes=MAX_UPLOAD_BYTES)
        self.paths.tmp.mkdir(parents=True, exist_ok=True)
        path = self.paths.tmp / f"upload-{secrets.token_hex(6)}"
        path.write_bytes(raw)
        os.chmod(path, 0o600)
        return path, False
