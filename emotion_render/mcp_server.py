"""emotion-render-mcp: the renderer as an MCP server, over stdio or streamable HTTP.

    emotion-render-mcp stdio                       one client, the person's own store, no token (D8)
    emotion-render-mcp http --systemd-sockets      on demand under systemd socket activation (D16)
    emotion-render-mcp http --bind 127.0.0.1:8772  a manual run
    emotion-render-mcp add-client bea --device vm prints a token once, stores its hash (D13)
    emotion-render-mcp remove-client bea [--device vm]
    emotion-render-mcp list-clients
    emotion-render-mcp access ada-en --owner ada --allow bea

Written against mcp 2.3.0 (`mcp.server.mcpserver.MCPServer`). Over HTTP the server
listens on loopback and tailnet addresses only and refuses to start on any other
(R13), refuses any request that carries an Origin header (a browser page), and
needs a bearer token from clients.json on every request.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

import anyio
import mcp_types as types
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError
from mcp.server.transport_security import TransportSecuritySettings
from starlette.datastructures import Headers

from . import Refused, __version__, voices
from .clients import ClientsError, ClientsFile
from .engines import ENGINES
from .gpulock import GpuLock
from .service import FORMATS, Caller, Paths, Service

log = logging.getLogger("emotion-render-mcp")

DEFAULT_PORT = 8772
TAILNET_V4 = ipaddress.ip_network("100.64.0.0/10")
TAILNET_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")
LOOPBACK = {"127.0.0.1", "::1"}
MAX_BODY = 8 * 1024 * 1024      # a 5 MB recording in base64 is about 6.7 MB of JSON
IDLE_MINUTES = 15

INSTRUCTIONS = """\
Acted speech in a cloned voice. Call list_voices to see the voices you may use and the
tags each engine acts out. turbo (English) acts out tags written in the text, such as
[sigh] or [chuckle]; multilingual (23 languages, French included) takes an exaggeration
level for the whole clip and no tags. A tag or setting an engine would ignore is refused,
never spoken. check validates without rendering. render returns a link to the audio,
read it with resources/read; a long render returns status "running" and a render_id to
pass to get_render. Renders are kept 7 days unless keep_days says otherwise."""


# -- identity ---------------------------------------------------------------

class ClientTokenVerifier:
    """The SDK's TokenVerifier protocol over clients.json. The raw token never leaves this call."""

    def __init__(self, clients: ClientsFile):
        self.clients = clients

    async def verify_token(self, token: str) -> AccessToken | None:
        found = self.clients.verify(token)
        if found is None:
            return None
        client_id, device = found
        return AccessToken(token="redacted", client_id=client_id, scopes=[], claims={"device": device})


def http_identify(ctx: Context) -> Caller:
    token = get_access_token()
    if token is None:  # RequireAuthMiddleware stops these before any tool runs
        raise Refused("this request carries no valid token", "not_allowed")
    request = ctx.request_context.request
    peer = getattr(getattr(request, "client", None), "host", None)
    return Caller(
        client_id=token.client_id,
        device=(token.claims or {}).get("device"),
        local=False,
        on_host=peer in LOOPBACK,
    )


def local_identify(client_id: str):
    caller = Caller(client_id=client_id, device="stdio", local=True, on_host=True)
    return lambda ctx: caller


# -- the MCP surface ----------------------------------------------------------

def _ok(payload: dict, extra: list | None = None) -> types.CallToolResult:
    text = types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))
    return types.CallToolResult(content=[text, *(extra or [])], structured_content=payload)


def _refused(e: Refused) -> types.CallToolResult:
    text = types.TextContent(type="text", text=f"REFUSED {e.code}: {e.message}")
    return types.CallToolResult(content=[text], structured_content=e.to_dict(), is_error=True)


def build_server(service: Service, identify, *, token_verifier: ClientTokenVerifier | None = None) -> MCPServer:
    auth = None
    if token_verifier is not None:
        # Bearer tokens checked by the SDK's middleware against our verifier. No OAuth
        # routes are served: resource_server_url is None and there is no auth provider.
        auth = AuthSettings(issuer_url=f"http://127.0.0.1:{DEFAULT_PORT}", resource_server_url=None)
    mcp = MCPServer(
        "assembly-emotion-render",
        version=__version__,
        instructions=INSTRUCTIONS,
        token_verifier=token_verifier,
        auth=auth,
    )

    async def call(ctx: Context, tool: str, voice, fn, extra=None) -> types.CallToolResult:
        caller = identify(ctx)
        service.touch()
        try:
            payload = await fn(caller)
        except Refused as e:
            service.audit.write(client=caller.client_id, device=caller.device, tool=tool, voice=voice,
                                outcome=e.code)
            return _refused(e)
        service.audit.write(client=caller.client_id, device=caller.device, tool=tool, voice=voice,
                            outcome=payload.get("status", "ok"), render_id=payload.get("render_id"))
        return _ok(payload, extra(caller, payload) if extra else None)

    def sync(fn):
        async def run(caller):
            return await anyio.to_thread.run_sync(lambda: fn(caller))
        return run

    def progress(ctx: Context):
        async def report(done: float, total, message):
            await ctx.report_progress(done, total, message)
        return report

    @mcp.tool(annotations=types.ToolAnnotations(read_only_hint=True))
    async def list_voices(ctx: Context) -> types.CallToolResult:
        """The voices you may use, each with its engine, language, and the tags and settings it takes."""
        return await call(ctx, "list_voices", None, sync(service.list_voices))

    @mcp.tool(annotations=types.ToolAnnotations(read_only_hint=True))
    async def check(ctx: Context, voice: str, text: str, exaggeration: float | None = None,
                    cfg_weight: float | None = None, device: str = "auto") -> types.CallToolResult:
        """Validate a render without making it: voice, consent, tags, settings, length, device. Under a second."""
        return await call(ctx, "check", voice, sync(
            lambda c: service.check(c, voice, text, exaggeration, cfg_weight, device)))

    @mcp.tool()
    async def render(ctx: Context, voice: str, text: str, exaggeration: float | None = None,
                     cfg_weight: float | None = None, device: str = "auto", format: str = "mp3",
                     inline_audio: bool = False, wait_seconds: float = 60,
                     keep_days: int | str | None = None) -> types.CallToolResult:
        """Render acted speech. Tags go in square brackets in the text (turbo only).

        Returns a link to the audio (read it with resources/read), or status "running"
        and a render_id for get_render if it takes longer than wait_seconds (at most 120).
        keep_days: 1 to 3650 or "forever", default 7. inline_audio puts the audio in the result.
        """
        def inline(caller, payload):
            if not inline_audio or payload.get("status") != "done":
                return None
            data, mime = service.read_render_file(caller, payload["render_id"], format)
            return [types.AudioContent(type="audio", data=_b64(data), mime_type=mime)]

        return await call(
            ctx, "render", voice,
            lambda c: service.render(c, voice, text, exaggeration, cfg_weight, device, format,
                                     wait_seconds, keep_days, progress(ctx)),
            extra=lambda c, p: (inline(c, p) or []) + _links(p),
        )

    @mcp.tool(annotations=types.ToolAnnotations(read_only_hint=True))
    async def get_render(ctx: Context, render_id: str, wait_seconds: float = 0) -> types.CallToolResult:
        """The result of one of your renders, waiting up to wait_seconds (at most 120) if it is still running."""
        return await call(ctx, "get_render", None,
                          lambda c: service.get_render(c, render_id, wait_seconds, progress(ctx)),
                          extra=lambda c, p: _links(p))

    @mcp.tool(annotations=types.ToolAnnotations(idempotent_hint=True))
    async def keep_render(ctx: Context, render_id: str, keep_days: int | str) -> types.CallToolResult:
        """Change how long one of your renders is kept: 1 to 3650 days, "forever", or 0 to delete it now."""
        return await call(ctx, "keep_render", None, sync(lambda c: service.keep_render(c, render_id, keep_days)))

    @mcp.tool()
    async def add_voice(ctx: Context, id: str, person: str, lang: str, engine: str, reference: dict,
                        consent: dict, defaults: dict | None = None,
                        allowed_clients: list[str] | None = None) -> types.CallToolResult:
        """Add a voice to the store, owned by you.

        reference: {"inbox_file": "<name>"} for a file already in your inbox, or
        {"data_base64": "...", "mime_type": "audio/..."} up to 5 MB. More than 5 s, at most 30 s.
        consent: {"person", "given", "scope"}, given by the person whose voice it is (I2).
        """
        return await call(ctx, "add_voice", id, sync(
            lambda c: service.add_voice(c, id, person, lang, engine, reference, consent, defaults,
                                        allowed_clients)))

    @mcp.tool(annotations=types.ToolAnnotations(destructive_hint=True))
    async def remove_voice(ctx: Context, id: str) -> types.CallToolResult:
        """Remove one of your voices, its reference clip, and every render of it, kept forever or not."""
        return await call(ctx, "remove_voice", id, sync(lambda c: service.remove_voice(c, id)))

    @mcp.resource("emotion-render://engines", mime_type="application/json")
    def engines() -> str:
        """The engines: languages, tags they act out, settings they take."""
        return json.dumps({
            name: {"languages": sorted(s.languages), "tags": sorted(s.tags), "params": sorted(s.params)}
            for name, s in ENGINES.items()
        })

    def read(ctx: Context, render_id: str, ext: str) -> bytes:
        caller = identify(ctx)
        try:
            data, _ = service.read_render_file(caller, render_id, ext)
        except Refused as e:
            service.audit.write(client=caller.client_id, device=caller.device, tool="read_render",
                                render_id=render_id, outcome=e.code)
            raise ResourceNotFoundError(f"{e.code}: {e.message}") from e
        service.audit.write(client=caller.client_id, device=caller.device, tool="read_render",
                            render_id=render_id, outcome="ok")
        return data

    @mcp.resource("emotion-render://renders/{render_id}.mp3", mime_type=FORMATS["mp3"])
    def render_mp3(render_id: str, ctx: Context) -> bytes:
        """One of your renders, as mp3."""
        return read(ctx, render_id, "mp3")

    @mcp.resource("emotion-render://renders/{render_id}.wav", mime_type=FORMATS["wav"])
    def render_wav(render_id: str, ctx: Context) -> bytes:
        """One of your renders, as wav."""
        return read(ctx, render_id, "wav")

    @mcp.resource("emotion-render://renders/{render_id}.json", mime_type=FORMATS["json"])
    def render_json(render_id: str, ctx: Context) -> bytes:
        """The record of one of your renders."""
        return read(ctx, render_id, "json")

    return mcp


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode()


def _links(payload: dict) -> list:
    audio = payload.get("audio")
    if payload.get("status") != "done" or not audio:
        return []
    return [types.ResourceLink(type="resource_link", uri=audio["uri"], name=f"{payload['render_id']}",
                               mime_type=audio["mime_type"], size=audio["bytes"])]


# -- HTTP: binding, the gate, idle exit ---------------------------------------

class BindRefused(ValueError):
    pass


def check_bind_host(host: str) -> None:
    """Loopback and tailnet addresses only (R13). Names are refused: they resolve to whatever DNS says."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError as e:
        raise BindRefused(f"'{host}' is not a literal IP address. Bind an address, not a name") from e
    if ip.is_loopback or ip in TAILNET_V4 or ip in TAILNET_V6:
        return
    raise BindRefused(
        f"refusing to listen on {host}: only loopback (127.0.0.1, ::1) and tailnet addresses "
        f"(100.64.0.0/10, fd7a:115c:a1e0::/48) are allowed. This host has no firewall, so any "
        f"other address would publish a voice-cloning tool"
    )


def parse_bind(value: str) -> tuple[str, int]:
    if value.startswith("["):
        host, _, port = value[1:].partition("]:")
    else:
        host, _, port = value.rpartition(":")
    if not host or not port.isdigit():
        raise BindRefused(f"'{value}' is not HOST:PORT")
    return host, int(port)


def bind_sockets(addresses: list[str]) -> list[socket.socket]:
    socks = []
    for value in addresses:
        host, port = parse_bind(value)
        check_bind_host(host)
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        s = socket.socket(family, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        s.bind((host, port))
        s.listen(128)
        socks.append(s)
    return socks


def systemd_sockets(environ=None, start_fd: int = 3) -> list[socket.socket]:
    """The listening sockets systemd passed (sd_listen_fds), checked like --bind."""
    env = os.environ if environ is None else environ
    if env.get("LISTEN_PID") != str(os.getpid()):
        raise BindRefused("--systemd-sockets: no sockets were passed to this process (LISTEN_PID)")
    count = int(env.get("LISTEN_FDS", "0"))
    if count < 1:
        raise BindRefused("--systemd-sockets: LISTEN_FDS is 0")
    socks = [socket.socket(fileno=start_fd + i) for i in range(count)]
    for s in socks:
        check_bind_host(s.getsockname()[0])
    for key in ("LISTEN_PID", "LISTEN_FDS", "LISTEN_FDNAMES"):
        env.pop(key, None)
    return socks


def allowed_hosts(socks: list[socket.socket], names: list[str]) -> list[str]:
    """Host header values the transport accepts (DNS-rebinding protection)."""
    hosts = []
    for s in socks:
        host, port = s.getsockname()[:2]
        hosts.append(f"[{host}]:{port}" if ":" in host else f"{host}:{port}")
        if host in LOOPBACK:
            hosts.append(f"localhost:{port}")
        hosts.extend(f"{n}:{port}" for n in names)
    return sorted(set(hosts))


class Gate:
    """Outermost ASGI layer: refuses browser requests, records activity and refused peers."""

    def __init__(self, app, service: Service):
        self.app = app
        self.service = service

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        self.service.touch()
        peer = (scope.get("client") or ("?",))[0]
        headers = Headers(scope=scope)
        if "origin" in headers:
            self.service.audit.write(tool="http", peer=peer, outcome="refused_origin")
            body = json.dumps({
                "error": "not_a_client",
                "message": "requests from a web page are refused. A page in a browser borrows the "
                           "trust of the machine it runs on; call through a server that holds a token",
            }).encode()
            await send({"type": "http.response.start", "status": 403,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})
            return

        status = {}

        async def watch(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        await self.app(scope, receive, watch)
        if status.get("code") == 401:
            self.service.audit.write(tool="http", peer=peer, outcome="unauthenticated")


async def watch_idle(server, service: Service, idle_seconds: float, tick: float = 15.0) -> None:
    """Ask uvicorn to exit after idle_seconds with no request and nothing rendering (D16)."""
    while not server.should_exit:
        await anyio.sleep(min(tick, idle_seconds))
        if service.idle() and service.idle_for() >= idle_seconds:
            log.warning("idle for %.0f s, exiting; the socket stays and the next request starts me again",
                        service.idle_for())
            server.should_exit = True
            return


def make_service(paths: Paths | None = None) -> Service:
    from .backend import TorchBackend
    from .media import Media

    return Service(paths or Paths.default(), TorchBackend(), Media(), GpuLock())


async def serve_http(service: Service, socks: list[socket.socket], *, clients: ClientsFile,
                     host_names: list[str], idle_seconds: float | None, warm: str | None,
                     stop: anyio.Event | None = None) -> None:
    """Serve until idle (D16), SIGTERM, or `stop` is set."""
    import uvicorn

    for s in socks:
        check_bind_host(s.getsockname()[0])
    mcp = build_server(service, http_identify, token_verifier=ClientTokenVerifier(clients))
    app = mcp.streamable_http_app(
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts(socks, host_names),
            allowed_origins=[],
        ),
        max_request_body_size=MAX_BODY,
    )
    config = uvicorn.Config(Gate(app, service), log_level="warning", access_log=False, lifespan="on",
                            timeout_graceful_shutdown=5)
    server = uvicorn.Server(config)
    async with service.running(warm=warm):
        async with anyio.create_task_group() as tg:
            if idle_seconds:
                tg.start_soon(watch_idle, server, service, idle_seconds)
            if stop is not None:
                async def on_stop():
                    await stop.wait()
                    server.should_exit = True
                tg.start_soon(on_stop)
            await server.serve(sockets=socks)
            tg.cancel_scope.cancel()


async def serve_stdio(service: Service, client_id: str) -> None:
    mcp = build_server(service, local_identify(client_id))
    async with service.running(warm=None):
        await mcp.run_stdio_async()


# -- command line -------------------------------------------------------------

def _args(argv):
    p = argparse.ArgumentParser(prog="emotion-render-mcp", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stdio", help="serve one client over stdio, from your own store")
    s.add_argument("--client", default="local", help="the owner recorded on voices added through it")

    h = sub.add_parser("http", help="serve streamable HTTP on loopback and tailnet addresses")
    where = h.add_mutually_exclusive_group(required=True)
    where.add_argument("--systemd-sockets", action="store_true", help="use the sockets systemd passed")
    where.add_argument("--bind", action="append", metavar="HOST:PORT",
                       help=f"listen here (repeatable), e.g. 127.0.0.1:{DEFAULT_PORT}")
    h.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                   help="a host name clients may use in the Host header, e.g. myhost.example.ts.net")
    h.add_argument("--idle-minutes", type=float, default=IDLE_MINUTES,
                   help=f"exit after this long without requests (default {IDLE_MINUTES}, 0 never)")
    h.add_argument("--warm", choices=["turbo", "none"], default="turbo",
                   help="load this engine on the GPU at start (default turbo)")

    a = sub.add_parser("add-client", help="create a token for one device of a client; printed once")
    a.add_argument("client")
    a.add_argument("--device", required=True)

    r = sub.add_parser("remove-client", help="revoke one device's token, or every token of a client")
    r.add_argument("client")
    r.add_argument("--device")

    sub.add_parser("list-clients", help="clients and devices, never tokens")

    v = sub.add_parser("access", help="show or set who may use a voice over MCP (I3)")
    v.add_argument("voice")
    v.add_argument("--owner")
    group = v.add_mutually_exclusive_group()
    group.add_argument("--allow", action="append", metavar="CLIENT", help="replaces the allowed list")
    group.add_argument("--allow-none", action="store_true", help="nobody but the owner")
    return p.parse_args(argv)


def _access(args) -> int:
    data = voices.read_entry(args.voice)
    if args.owner is not None or args.allow is not None or args.allow_none:
        if args.owner is not None:
            if not voices.is_id(args.owner):
                raise Refused(f"'{args.owner}' is not a client id ({voices.ID_RULE})", "invalid_entry")
            data["owner"] = args.owner
        if args.allow is not None or args.allow_none:
            allowed = [] if args.allow_none else args.allow
            if not all(voices.is_id(c) for c in allowed):
                raise Refused(f"allowed clients must be client ids ({voices.ID_RULE})", "invalid_entry")
            data["allowed_clients"] = allowed
        voices.write_entry(args.voice, data, replace=True)
    owner, allowed = voices.access(data)
    print(json.dumps({"voice": args.voice, "owner": owner, "allowed_clients": list(allowed)}))
    return 0


def main(argv=None) -> int:
    args = _args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(name)s: %(message)s")
    clients = ClientsFile()
    try:
        if args.cmd == "add-client":
            token = clients.add(args.client, args.device)
            print(token)
            print(f"token for {args.client} on {args.device}. Shown once; only its hash is kept in "
                  f"{clients.path}", file=sys.stderr)
            return 0
        if args.cmd == "remove-client":
            n = clients.remove(args.client, args.device)
            print(f"removed {n} token(s) of {args.client}", file=sys.stderr)
            return 0
        if args.cmd == "list-clients":
            print(json.dumps(clients.list(), indent=2))
            return 0
        if args.cmd == "access":
            return _access(args)
        if args.cmd == "stdio":
            if not voices.is_id(args.client):
                raise ClientsError(f"'{args.client}' is not a client id ({voices.ID_RULE})")
            anyio.run(serve_stdio, make_service(), args.client)
            return 0
        socks = systemd_sockets() if args.systemd_sockets else bind_sockets(args.bind)
        idle = args.idle_minutes * 60 if args.idle_minutes > 0 else None
        warm = None if args.warm == "none" else args.warm
        started = time.time()
        anyio.run(lambda: serve_http(make_service(), socks, clients=clients, host_names=args.allow_host,
                                     idle_seconds=idle, warm=warm), backend="asyncio")
        log.warning("stopped after %.0f s", time.time() - started)
        return 0
    except (ClientsError, BindRefused) as e:
        print(f"REFUSED  {e}", file=sys.stderr)
        return 2
    except Refused as e:
        print(f"REFUSED  {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
