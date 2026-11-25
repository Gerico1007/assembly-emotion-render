"""The MCP surface through the SDK's in-memory client, on a fake engine (B4)."""
import base64
import dataclasses
import json
import threading

import anyio
import pytest

from emotion_render import cli, service as service_module, voices
from emotion_render.engines import ENGINES
from emotion_render.gpulock import GpuLock
from emotion_render.media import MediaError

from fake_media import FakeMedia
from fakes import FakeBackend, make_wav
from world import CYD, ADA, ADA_ON_HOST, LOCAL, BEA, World, call, refused

pytestmark = pytest.mark.anyio


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


# -- every refusal the CLI has, identical over MCP (D9) ---------------------

PARITY = [
    # (voice, text, extra MCP args, extra CLI args, code)
    ("ada-en", "Hello [whisper] there.", {}, [], "unknown_tag"),
    ("ada-en", "Hello [sigh there.", {}, [], "stray_bracket"),
    ("ada-en", "Hello [] there.", {}, [], "empty_tag"),
    ("ada-en", "   ", {}, [], "empty_script"),
    ("ada-en", "word " * 100, {}, [], "text_too_long"),
    ("ada-en", "Hello.", {"exaggeration": 0.7}, ["--exaggeration", "0.7"], "setting_not_taken"),
    ("ada-fr", "Bonjour [sigh].", {}, [], "unknown_tag"),
    ("ada-noconsent", "Hello.", {}, [], "no_consent"),
    ("ada-mismatch", "Hello.", {}, [], "consent_mismatch"),
    ("ada-badengine", "Hello.", {}, [], "unknown_engine"),
    ("ada-badlang", "Hello.", {}, [], "engine_language"),
    ("ada-badparam", "Hello.", {}, [], "setting_not_taken"),
    ("ada-noref", "Hello.", {}, [], "missing_reference"),
    ("ada-noperson", "Hello.", {}, [], "invalid_entry"),
    ("Ada", "Hello.", {}, [], "invalid_voice_id"),
]


@pytest.mark.parametrize("voice, text, mcp_args, cli_args, code", PARITY)
async def test_every_cli_refusal_is_the_same_over_mcp(world, monkeypatch, capsys, voice, text, mcp_args,
                                                      cli_args, code):
    monkeypatch.setattr(voices, "REGISTRY_DIR", world.paths.store)
    assert cli.main(["--voice", voice, "--text", text, *cli_args, "--check"]) == 2
    cli_message = capsys.readouterr().err.removeprefix("REFUSED  ").strip()

    async with world.serve(ADA_ON_HOST) as (ada,):
        content = await refused(ada, "check", voice=voice, text=text, **mcp_args)
        assert content["code"] == code
        assert content["message"] == cli_message
        render_refusal = await refused(ada, "render", voice=voice, text=text, **mcp_args)
        assert render_refusal == content
    assert world.backend.generated == []


async def test_multilingual_on_cuda_is_refused_like_the_cli(world, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(voices, "REGISTRY_DIR", world.paths.store)
    monkeypatch.setattr(cli, "make_backend", lambda: FakeBackend())
    assert cli.main(["--voice", "ada-fr", "--text", "Bonjour.", "--device", "cuda",
                     "--out", str(tmp_path / "x.wav")]) == 2
    cli_message = capsys.readouterr().err.removeprefix("REFUSED  ").strip()
    async with world.serve(ADA) as (ada,):
        content = await refused(ada, "render", voice="ada-fr", text="Bonjour.", device="cuda")
    assert content["code"] == "gpu_too_small"
    assert content["message"] == cli_message


async def test_unknown_tag_says_what_is_accepted(world):
    async with world.serve(ADA) as (ada,):
        content = await refused(ada, "check", voice="ada-en", text="Hi [whisper].")
    assert content["unknown"] == ["whisper"]
    assert "sigh" in content["accepted"] and "chuckle" in content["accepted"]


async def test_server_side_argument_refusals(world):
    async with world.serve(ADA) as (ada,):
        assert (await refused(ada, "render", voice="ada-en", text="Hi.", format="ogg"))["code"] == "bad_format"
        assert (await refused(ada, "render", voice="ada-en", text="Hi.", wait_seconds=500))["code"] == "bad_wait"
        assert (await refused(ada, "render", voice="ada-en", text="Hi.", device="tpu"))["code"] == "bad_device"
        for keep in (0, 4000, "soon"):
            content = await refused(ada, "render", voice="ada-en", text="Hi.", keep_days=keep)
            assert content["code"] == "bad_retention"
        assert (await refused(ada, "get_render", render_id="../etc"))["code"] == "unknown_render"


# -- I3 ----------------------------------------------------------------------

async def test_each_client_sees_only_its_own_and_granted_voices(world):
    async with world.serve(ADA, BEA, CYD, LOCAL) as (ada, bea, cyd, local):
        def ids(content):
            return {v["id"] for v in content["voices"]}

        _, mine = await call(ada, "list_voices")
        assert {"ada-en", "ada-fr", "ada-private"} <= ids(mine)
        assert "noor-fr" not in ids(mine) and "unowned" not in ids(mine)
        _, granted = await call(bea, "list_voices")
        assert ids(granted) == {"ada-en", "ada-fr"}
        assert all("allowed_clients" not in v for v in granted["voices"])  # only the owner sees the grants
        _, nothing = await call(cyd, "list_voices")
        assert nothing == {"voices": []}
        _, everything = await call(local, "list_voices")
        assert ids(everything) == set(voices.ids(world.paths.store))


async def test_a_voice_you_may_not_use_does_not_exist_for_you(world):
    async with world.serve(ADA, BEA, CYD) as (ada, bea, cyd):
        hidden = await refused(cyd, "check", voice="ada-en", text="Hi.")
        absent = await refused(cyd, "check", voice="nobody", text="Hi.")
        assert hidden["code"] == absent["code"] == "unknown_voice"
        # Word for word the answer for a voice that does not exist.
        assert hidden["message"].replace("ada-en", "?") == absent["message"].replace("nobody", "?")
        # Noor's voice is closed to everyone until Noor decides, Ada included (Q8).
        assert (await refused(ada, "render", voice="noor-fr", text="Bonjour."))["code"] == "unknown_voice"
        assert (await refused(bea, "render", voice="ada-private", text="Hi."))["code"] == "unknown_voice"
        assert (await refused(bea, "render", voice="unowned", text="Hi."))["code"] == "unknown_voice"


async def test_renders_are_readable_only_by_the_client_that_made_them(world):
    async with world.serve(BEA, ADA) as (bea, ada):
        ok, done = await call(bea, "render", voice="ada-en", text="[sigh] Hi.")
        assert ok and done["status"] == "done"
        uri = done["audio"]["uri"]
        assert (await bea.read_resource(uri)).contents[0].blob
        with pytest.raises(Exception, match="unknown_render"):
            await ada.read_resource(uri)
        assert (await refused(ada, "get_render", render_id=done["render_id"]))["code"] == "unknown_render"
        assert (await refused(ada, "keep_render", render_id=done["render_id"], keep_days=30))["code"] == \
            "unknown_render"


async def test_a_remote_caller_gets_no_server_paths(world):
    async with world.serve(BEA, ADA_ON_HOST) as (bea, ada_here):
        _, remote = await call(bea, "render", voice="ada-en", text="Hi.")
        _, here = await call(ada_here, "render", voice="ada-en", text="Hi.")
        noref = await refused(ada_here, "check", voice="ada-noref", text="Hi.")
    assert "path" not in remote and "out" not in remote["record"] and "reference" not in remote["record"]
    assert here["path"].endswith(".mp3") and here["record"]["reference"].endswith("ada-en.wav")
    assert str(world.paths.store) in noref["message"]


# -- rendering, devices, memory ----------------------------------------------

async def test_render_returns_a_link_record_and_optional_inline_audio(world):
    async with world.serve(BEA) as (bea,):
        result = await bea.call_tool("render", {"voice": "ada-en", "text": "[chuckle] Hi.",
                                                 "inline_audio": True})
        held_while_serving = world.lock.held
    assert [c.type for c in result.content] == ["text", "audio", "resource_link"]
    done = result.structured_content
    assert done["record"]["device"] == "cuda"               # Turbo warm on the GPU
    assert done["record"]["tags"] == ["chuckle"] and done["record"]["client"] == "bea"
    assert done["record"]["keep_days"] == 7 and done["record"]["expires_at"]
    assert world.backend.loads == [("turbo", "cuda")]       # loaded once, at start
    assert held_while_serving and not world.lock.held       # released when the server stops


async def test_multilingual_loads_on_cpu_and_unloads_when_idle(world):
    async with world.serve(BEA) as (bea,):
        ok, done = await call(bea, "render", voice="ada-fr", text="Bonjour.", exaggeration=0.7)
        assert ok and done["record"]["device"] == "cpu" and done["record"]["params"]["exaggeration"] == 0.7
        assert world.service.loaded() == [("multilingual", "cpu"), ("turbo", "cuda")]
        world.clock.advance(601)
        world.service.tidy()
        assert world.service.loaded() == [("turbo", "cuda")]   # the warm model stays
        assert world.lock.held


async def test_a_gpu_held_elsewhere_sends_auto_to_cpu_and_refuses_cuda(world):
    other = GpuLock(world.lock.path)
    assert other.acquire()
    try:
        async with world.serve(ADA) as (ada,):
            busy = await refused(ada, "render", voice="ada-en", text="Hi.", device="cuda")
            ok, done = await call(ada, "render", voice="ada-en", text="Hi.")
        assert busy["code"] == "gpu_busy"
        assert ok and done["record"]["device"] == "cpu"
    finally:
        other.release()


async def test_no_gpu(tmp_path):
    world = World(tmp_path, FakeBackend(cuda_mb=None))
    async with world.serve(ADA) as (ada,):
        assert (await refused(ada, "check", voice="ada-en", text="Hi.", device="cuda"))["code"] == "no_gpu"


async def test_a_script_too_long_for_the_gpu(world, monkeypatch):
    monkeypatch.setitem(ENGINES, "turbo", dataclasses.replace(ENGINES["turbo"], cuda_max_chars=10))
    async with world.serve(ADA) as (ada,):
        content = await refused(ada, "check", voice="ada-en", text="This is too long.", device="cuda")
        _, auto = await call(ada, "check", voice="ada-en", text="This is too long.")
    assert content["code"] == "text_too_long_for_gpu" and content["limit"] == 10
    assert auto["device"] == "cpu"


async def test_low_memory_refuses_the_load(tmp_path):
    world = World(tmp_path, FakeBackend(available_mb=100))
    async with world.serve(ADA, warm=None) as (ada,):
        content = await refused(ada, "render", voice="ada-en", text="Hi.")
    assert content["code"] == "low_memory" and content["available_mb"] == 100
    assert world.backend.loads == [] and not world.lock.held


async def test_gpu_out_of_memory_is_refused_and_the_model_stays(world):
    async with world.serve(ADA) as (ada,):
        world.backend.oom = True
        content = await refused(ada, "render", voice="ada-en", text="Hi.")
        world.backend.oom = False
        ok, _ = await call(ada, "render", voice="ada-en", text="Hi.")
    assert content["code"] == "gpu_out_of_memory" and content["chars"] == 3
    assert ok and world.backend.loads == [("turbo", "cuda")]


async def test_a_failed_mp3_is_render_failed(tmp_path):
    class BrokenMp3(FakeMedia):
        def to_mp3(self, src, dst):
            raise MediaError("no encoder")

    world = World(tmp_path, media=BrokenMp3())
    async with world.serve(ADA) as (ada,):
        content = await refused(ada, "render", voice="ada-en", text="Hi.")
    assert content["code"] == "render_failed"
    assert not list((world.paths.renders / "ada").glob("*"))


# -- queue, get_render, progress (D12) ---------------------------------------

async def _wait_started(backend):
    assert await anyio.to_thread.run_sync(lambda: backend.started.wait(10))


async def test_one_render_at_a_time_three_waiting_the_next_is_busy(world):
    world.backend.gate = threading.Event()
    async with world.serve(BEA) as (bea,):
        _, first = await call(bea, "render", voice="ada-en", text="One.", wait_seconds=0)
        await _wait_started(world.backend)
        positions = []
        for n in range(3):
            _, waiting = await call(bea, "render", voice="ada-en", text=f"Wait {n}.", wait_seconds=0)
            positions.append(waiting["queue_position"])
        busy = await refused(bea, "render", voice="ada-en", text="Too many.", wait_seconds=0)
        _, still = await call(bea, "check", voice="ada-en", text="Checks still answer.")
        world.backend.gate.set()
        ok, done = await call(bea, "get_render", render_id=waiting["render_id"], wait_seconds=10)
        _, first_done = await call(bea, "get_render", render_id=first["render_id"])
    # With wait_seconds=0 the answer can come back before the worker takes the job.
    assert first["status"] == "running" and first["queue_position"] in (0, 1)
    assert positions == [1, 2, 3]
    assert busy["code"] == "busy" and busy["waiting"] == 3
    assert still["ok"]
    assert ok and done["status"] == "done" and first_done["status"] == "done"
    assert [g["text"] for g in world.backend.generated] == ["One.", "Wait 0.", "Wait 1.", "Wait 2."]


async def test_progress_is_reported_while_waiting(world):
    world.backend.gate = threading.Event()
    seen = []

    async def on_progress(progress, total, message):
        seen.append(message)

    async with world.serve(BEA) as (bea,):
        async with anyio.create_task_group() as tg:
            async def open_gate():
                await anyio.sleep(0.4)
                world.backend.gate.set()
            tg.start_soon(open_gate)
            result = await bea.call_tool("render", {"voice": "ada-en", "text": "Hi.", "wait_seconds": 5},
                                          progress_callback=on_progress)
    assert result.structured_content["status"] == "done"
    assert "rendering" in seen


# -- retention (D17) ----------------------------------------------------------

async def test_keep_render_pins_shortens_and_deletes(world):
    async with world.serve(BEA) as (bea,):
        _, a = await call(bea, "render", voice="ada-en", text="A.", keep_days="forever")
        _, b = await call(bea, "render", voice="ada-en", text="B.")
        _, c = await call(bea, "render", voice="ada-en", text="C.")
        assert a["record"]["expires_at"] is None
        _, shorter = await call(bea, "keep_render", render_id=b["render_id"], keep_days=1)
        _, deleted = await call(bea, "keep_render", render_id=c["render_id"], keep_days=0)
        assert deleted == {"render_id": c["render_id"], "deleted": True}
        assert (await refused(bea, "get_render", render_id=c["render_id"]))["code"] == "unknown_render"
        assert (await refused(bea, "keep_render", render_id=b["render_id"], keep_days=-1))["code"] == \
            "bad_retention"

        world.clock.advance(2 * 86400)
        assert (await refused(bea, "get_render", render_id=b["render_id"]))["code"] == "unknown_render"
        with pytest.raises(Exception, match="unknown_render"):
            await bea.read_resource(b["audio"]["uri"])
        assert world.service.sweep() == 1
        world.clock.advance(3650 * 86400)
        ok, pinned = await call(bea, "get_render", render_id=a["render_id"])
    assert shorter["expires_at"] is not None
    assert ok and pinned["status"] == "done"
    left = sorted(p.name for p in (world.paths.renders / "bea").iterdir())
    assert left == sorted(f"{a['render_id']}.{ext}" for ext in ("json", "mp3", "wav"))


async def test_a_running_render_cannot_be_deleted_but_can_be_pinned(world):
    world.backend.gate = threading.Event()
    async with world.serve(BEA) as (bea,):
        _, running = await call(bea, "render", voice="ada-en", text="Hi.", wait_seconds=0)
        rid = running["render_id"]
        assert (await refused(bea, "keep_render", render_id=rid, keep_days=0))["code"] == "render_running"
        ok, pinned = await call(bea, "keep_render", render_id=rid, keep_days="forever")
        world.backend.gate.set()
        _, done = await call(bea, "get_render", render_id=rid, wait_seconds=10)
    assert ok and pinned["status"] == "running"
    assert done["record"]["keep_days"] == "forever" and done["record"]["expires_at"] is None


def test_keep_days_values():
    parse = service_module.parse_keep_days
    assert parse(None, allow_zero=False) == 7
    assert parse("forever", allow_zero=False) == "forever"
    assert parse(0, allow_zero=True) == 0
    for bad in (True, 0, 3651, 1.5, "7"):
        with pytest.raises(Exception):
            parse(bad, allow_zero=False)


# -- add_voice and remove_voice ---------------------------------------------

def _consent(person="Bea"):
    return {"person": person, "given": "2026-11-25", "scope": "tests"}


async def test_add_voice_from_the_inbox_owned_by_the_caller(world):
    make_wav(world.paths.inbox / "bea" / "take-1.wav", 10)
    async with world.serve(BEA, ADA) as (bea, ada):
        ok, entry = await call(bea, "add_voice", id="bea-en", person="Bea", lang="en", engine="turbo",
                               reference={"inbox_file": "take-1.wav"}, consent=_consent())
        _, bea_list = await call(bea, "list_voices")
        _, ada_list = await call(ada, "list_voices")
        ok2, done = await call(bea, "render", voice="bea-en", text="[laugh] Mine.")
    assert ok and entry["owner"] == "bea" and entry["allowed_clients"] == []
    assert entry["consent"]["asserted_by"] == "bea" and entry["consent"]["recorded_at"]
    assert entry["reference_seconds"] == 10.0 and entry["inbox_file_removed"]
    assert not (world.paths.inbox / "bea" / "take-1.wav").exists()
    assert (world.paths.store / "bea-en.wav").is_file()
    assert "bea-en" in {v["id"] for v in bea_list["voices"]}
    assert "bea-en" not in {v["id"] for v in ada_list["voices"]}
    assert ok2 and done["status"] == "done"


async def test_add_voice_from_base64(world, tmp_path):
    clip = make_wav(tmp_path / "upload.wav", 8)
    data = base64.b64encode(clip.read_bytes()).decode()
    async with world.serve(BEA) as (bea,):
        ok, entry = await call(bea, "add_voice", id="bea-fr", person="Bea", lang="fr", engine="multilingual",
                               reference={"data_base64": data, "mime_type": "audio/wav"}, consent=_consent(),
                               defaults={"exaggeration": 0.6}, allowed_clients=["ada"])
    assert ok and entry["defaults"] == {"exaggeration": 0.6} and entry["allowed_clients"] == ["ada"]
    assert not list(world.paths.tmp.glob("upload-*"))


@pytest.mark.parametrize("seconds, code", [(4, "clip_too_short"), (5, "clip_too_short"), (31, "clip_too_long")])
async def test_add_voice_refuses_a_clip_of_the_wrong_length(world, seconds, code):
    make_wav(world.paths.inbox / "bea" / "take.wav", seconds)
    async with world.serve(BEA) as (bea,):
        content = await refused(bea, "add_voice", id="bea-en", person="Bea", lang="en", engine="turbo",
                                reference={"inbox_file": "take.wav"}, consent=_consent())
    assert content["code"] == code
    assert not (world.paths.store / "bea-en.json").exists() and not (world.paths.store / "bea-en.wav").exists()
    assert (world.paths.inbox / "bea" / "take.wav").exists()   # kept, so the person can fix and retry


async def test_add_voice_refusals(world, monkeypatch):
    (world.paths.inbox / "bea").mkdir(parents=True)
    (world.paths.inbox / "bea" / "noise.m4a").write_bytes(b"not audio")
    make_wav(world.paths.inbox / "bea" / "take.wav", 10)
    monkeypatch.setattr(service_module, "MAX_UPLOAD_BYTES", 1000)

    def args(**over):
        base = dict(id="bea-en", person="Bea", lang="en", engine="turbo",
                    reference={"inbox_file": "take.wav"}, consent=_consent())
        base.update(over)
        return base

    cases = [
        (args(id="ada-en"), "voice_exists"),
        (args(id="Bea En"), "invalid_voice_id"),
        (args(person=""), "invalid_entry"),
        (args(consent={"person": "Bea"}), "no_consent"),
        (args(consent=_consent("Ada")), "consent_mismatch"),
        (args(lang="fr"), "engine_language"),
        (args(engine="bark"), "unknown_engine"),
        (args(defaults={"exaggeration": 0.5}), "setting_not_taken"),
        (args(allowed_clients=["Not An Id"]), "invalid_entry"),
        (args(reference={"inbox_file": "../ada/take.wav"}), "invalid_entry"),
        (args(reference={"inbox_file": "absent.wav"}), "missing_reference"),
        (args(reference={"inbox_file": "noise.m4a"}), "clip_unreadable"),
        (args(reference={"data_base64": "!!!"}), "clip_unreadable"),
        (args(reference={"data_base64": base64.b64encode(b"x" * 1001).decode()}), "clip_too_large"),
        (args(reference={"inbox_file": "take.wav", "data_base64": "AAAA"}), "invalid_entry"),
    ]
    async with world.serve(BEA) as (bea,):
        for kwargs, code in cases:
            content = await refused(bea, "add_voice", **kwargs)
            assert content["code"] == code, (kwargs, content)
    assert not (world.paths.store / "bea-en.json").exists()


async def test_remove_voice_deletes_every_render_of_it_pinned_or_not(world):
    async with world.serve(ADA, BEA) as (ada, bea):
        _, mine = await call(ada, "render", voice="ada-en", text="Mine.", keep_days="forever")
        _, theirs = await call(bea, "render", voice="ada-en", text="Theirs.", keep_days="forever")
        _, other = await call(bea, "render", voice="ada-fr", text="Bonjour.")
        not_owner = await refused(bea, "remove_voice", id="ada-en")
        hidden = await refused(bea, "remove_voice", id="noor-fr")
        ok, removed = await call(ada, "remove_voice", id="ada-en")
        gone = await refused(bea, "get_render", render_id=theirs["render_id"])
        ok_other, still = await call(bea, "get_render", render_id=other["render_id"])
        after = await refused(bea, "render", voice="ada-en", text="Hi.")
    assert not_owner["code"] == "not_allowed" and hidden["code"] == "unknown_voice"
    assert ok and removed["renders_deleted"] == 2 and removed["reference_deleted"]
    assert gone["code"] == "unknown_render"
    assert ok_other and still["status"] == "done"
    assert after["code"] == "unknown_voice"
    assert not (world.paths.store / "ada-en.json").exists() and not (world.paths.store / "ada-en.wav").exists()
    assert not list((world.paths.renders / "ada").glob(f"{mine['render_id']}.*"))


async def test_remove_voice_keeps_a_clip_another_entry_uses(world):
    entry = json.loads((world.paths.store / "ada-private.json").read_text())
    entry["reference"] = "ada-en.wav"
    (world.paths.store / "ada-private.json").write_text(json.dumps(entry))
    async with world.serve(ADA) as (ada,):
        _, removed = await call(ada, "remove_voice", id="ada-en")
    assert not removed["reference_deleted"] and (world.paths.store / "ada-en.wav").is_file()


# -- audit ----------------------------------------------------------------------

async def test_the_audit_log_names_who_and_never_the_script(world):
    async with world.serve(BEA, CYD) as (bea, cyd):
        await call(bea, "render", voice="ada-en", text="A secret sentence.")
        await call(cyd, "render", voice="ada-en", text="Another secret.")
    lines = world.service.audit.lines()
    raw = world.paths.audit.read_text()
    assert "secret" not in raw
    assert {"client": "bea", "device": "vm", "tool": "render", "voice": "ada-en", "outcome": "done"}.items() \
        <= next(x for x in lines if x["tool"] == "render" and x["client"] == "bea").items()
    assert any(x["client"] == "cyd" and x["outcome"] == "unknown_voice" for x in lines)


async def test_multilingual_beside_warm_turbo_uses_the_lower_guard(tmp_path):
    # 7000 MB free: refused for a fresh process (7500), enough beside Turbo (6500).
    world = World(tmp_path, FakeBackend(available_mb=7000))
    async with world.serve(BEA) as (bea,):
        ok, done = await call(bea, "render", voice="ada-fr", text="Bonjour.")
        world.backend.available = 6000
        world.clock.advance(601)
        world.service.tidy()                       # Multilingual unloaded, Turbo stays
        low = await refused(bea, "render", voice="ada-fr", text="Bonjour.")
    assert ok and done["record"]["device"] == "cpu"
    assert low["code"] == "low_memory" and low["needed_mb"] == 6500
    assert "beside the model already loaded" in low["message"]


async def test_a_script_over_300_characters_goes_to_cpu(world):
    text = ("word " * 70).strip() + "."
    async with world.serve(BEA) as (bea,):
        ok, done = await call(bea, "render", voice="ada-en", text=text)
        gpu = await refused(bea, "check", voice="ada-en", text=text, device="cuda")
    assert ok and len(done["record"]["clean_text"]) > 300 and done["record"]["device"] == "cpu"
    assert gpu["code"] == "text_too_long_for_gpu" and gpu["limit"] == 300
