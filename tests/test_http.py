"""B4 over streamable HTTP: binding (R13), systemd sockets (D16), the gate, tokens, idle exit."""
import json
import socket

import anyio
import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from emotion_render import clients as clients_module, mcp_server as M, voices
from emotion_render.clients import ClientsFile

from world import World

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "10.0.0.5", "8.8.8.8", "myhost", "localhost"])
def test_any_address_but_loopback_and_tailnet_is_refused(host):
    with pytest.raises(M.BindRefused):
        M.check_bind_host(host)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "100.64.0.7", "fd7a:115c:a1e0::1"])
def test_loopback_and_tailnet_are_accepted(host):
    M.check_bind_host(host)


def test_bind_refuses_before_opening_anything():
    with pytest.raises(M.BindRefused):
        M.bind_sockets(["127.0.0.1:0", "0.0.0.0:8772"])


def test_the_command_line_refuses_a_public_bind(capsys):
    assert M.main(["http", "--bind", "0.0.0.0:8772"]) == 2
    assert "only loopback" in capsys.readouterr().err


def test_systemd_sockets_are_taken_and_checked():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    fd = s.detach()
    env = {"LISTEN_PID": str(__import__("os").getpid()), "LISTEN_FDS": "1"}
    socks = M.systemd_sockets(env, start_fd=fd)
    assert socks[0].getsockname()[0] == "127.0.0.1" and "LISTEN_FDS" not in env
    socks[0].close()
    with pytest.raises(M.BindRefused):
        M.systemd_sockets({"LISTEN_PID": "1", "LISTEN_FDS": "1"})


def test_allowed_host_headers():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    assert M.allowed_hosts([s], ["host.tailnet.example"]) == sorted(
        {f"127.0.0.1:{port}", f"localhost:{port}", f"host.tailnet.example:{port}"})
    s.close()


async def test_idle_exit_waits_for_quiet_and_an_empty_queue(tmp_path):
    world = World(tmp_path)

    class FakeUvicorn:
        should_exit = False

    server = FakeUvicorn()
    world.service.touch()
    async with world.service.running(warm=None):
        with anyio.move_on_after(0.3):
            await M.watch_idle(server, world.service, idle_seconds=3600, tick=0.01)
        assert not server.should_exit
        world.clock.advance(3601)
        await M.watch_idle(server, world.service, idle_seconds=3600, tick=0.01)
    assert server.should_exit


@pytest.fixture
def served(tmp_path):
    world = World(tmp_path)
    clients = ClientsFile(tmp_path / "clients.json")
    tokens = {"bea": clients.add("bea", "vm"), "cyd": clients.add("cyd", "laptop")}
    socks = M.bind_sockets(["127.0.0.1:0"])
    url = f"http://127.0.0.1:{socks[0].getsockname()[1]}/mcp"
    return world, clients, tokens, socks, url


async def _with_server(world, clients, socks, body):
    stop = anyio.Event()
    async with anyio.create_task_group() as tg:
        tg.start_soon(lambda: M.serve_http(world.service, socks, clients=clients, host_names=[],
                                           idle_seconds=None, warm=None, stop=stop))
        await anyio.sleep(0.5)
        try:
            await body()
        finally:
            stop.set()


async def test_http_refuses_without_a_token_with_an_origin_or_a_foreign_host(served):
    world, clients, tokens, socks, url = served
    status = {}

    async def body():
        async with httpx2.AsyncClient() as h:
            status["none"] = (await h.post(url, json={})).status_code
            status["wrong"] = (await h.post(url, json={}, headers={"authorization": "Bearer er_wrong"})).status_code
            auth = {"authorization": f"Bearer {tokens['bea']}"}
            status["origin"] = (await h.post(url, json={}, headers={**auth, "origin": "https://evil.example"})).status_code
            status["host"] = (await h.post(url, json={}, headers={**auth, "host": "evil.example"})).status_code

    await _with_server(world, clients, socks, body)
    assert status == {"none": 401, "wrong": 401, "origin": 403, "host": 421}
    outcomes = [line["outcome"] for line in world.service.audit.lines()]
    assert outcomes.count("unauthenticated") == 2 and "refused_origin" in outcomes
    assert tokens["bea"] not in world.paths.audit.read_text()


async def test_the_token_decides_who_is_calling(served):
    world, clients, tokens, socks, url = served
    seen = {}

    async def as_client(name):
        async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {tokens[name]}"}) as h:
            async with Client(streamable_http_client(url, http_client=h)) as c:
                listed = await c.call_tool("list_voices", {})
                rendered = await c.call_tool("render", {"voice": "ada-en", "text": "[sigh] Hi."})
                seen[name] = (listed.structured_content, rendered)

    async def body():
        await as_client("bea")
        await as_client("cyd")

    await _with_server(world, clients, socks, body)
    bea_list, bea_render = seen["bea"]
    cyd_list, cyd_render = seen["cyd"]
    assert {v["id"] for v in bea_list["voices"]} == {"ada-en", "ada-fr"}
    assert bea_render.structured_content["record"]["client"] == "bea"
    assert bea_render.structured_content["path"].endswith(".mp3")      # loopback is on this host
    assert cyd_list == {"voices": []}
    assert cyd_render.is_error and cyd_render.structured_content["code"] == "unknown_voice"
    devices = {(x["client"], x["device"]) for x in world.service.audit.lines() if x.get("tool") == "render"}
    assert devices == {("bea", "vm"), ("cyd", "laptop")}


def test_add_client_prints_the_token_once_and_access_sets_grants(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(clients_module, "DEFAULT_PATH", tmp_path / "clients.json")
    assert M.main(["add-client", "bea", "--device", "vm"]) == 0
    token = capsys.readouterr().out.strip()
    assert ClientsFile(tmp_path / "clients.json").verify(token) == ("bea", "vm")
    assert M.main(["list-clients"]) == 0
    assert token not in capsys.readouterr().out

    world = World(tmp_path / "w")
    monkeypatch.setattr(voices, "REGISTRY_DIR", world.paths.store)
    assert M.main(["access", "noor-fr"]) == 0
    assert json.loads(capsys.readouterr().out) == {"voice": "noor-fr", "owner": "noor",
                                                   "allowed_clients": []}
    assert M.main(["access", "ada-en", "--owner", "ada", "--allow", "bea", "--allow", "ipad-shortcut"]) == 0
    assert json.loads(capsys.readouterr().out)["allowed_clients"] == ["bea", "ipad-shortcut"]
    assert M.main(["access", "ada-en", "--allow-none"]) == 0
    assert json.loads((world.paths.store / "ada-en.json").read_text())["allowed_clients"] == []
    assert M.main(["access", "ada-en", "--allow", "Not Valid"]) == 2


def test_the_sdk_verifier_maps_a_token_to_client_and_device(tmp_path):
    clients = ClientsFile(tmp_path / "clients.json")
    token = clients.add("bea", "vm")
    verifier = M.ClientTokenVerifier(clients)
    found = anyio.run(verifier.verify_token, token)
    assert found.client_id == "bea" and found.claims == {"device": "vm"}
    assert found.token == "redacted"
    assert anyio.run(verifier.verify_token, "er_nope") is None
