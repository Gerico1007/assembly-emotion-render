"""A store, a service and in-memory MCP clients for the server tests. No model, no network."""
from __future__ import annotations

import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from mcp import Client

from emotion_render.gpulock import GpuLock
from emotion_render.mcp_server import build_server
from emotion_render.service import Caller, Paths, Service

from fake_media import FakeMedia
from fakes import FakeBackend
from store import make_store

ADA = Caller("ada", "phone")
ADA_ON_HOST = Caller("ada", "host", on_host=True)
BEA = Caller("bea", "vm")
CYD = Caller("cyd", "laptop")          # a client with a token and no voice granted
LOCAL = Caller("local", "stdio", local=True, on_host=True)


class Clock:
    def __init__(self):
        self.now = time.time()

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class World:
    def __init__(self, root: Path, backend: FakeBackend | None = None, media=None, **service_kw):
        self.root = root
        self.paths = Paths(root / "store", root / "renders", root / "inbox", root / "audit.jsonl", root / "tmp")
        make_store(self.paths.store)
        self.backend = backend or FakeBackend()
        self.media = media or FakeMedia()
        self.lock = GpuLock(root / "gpu.lock")
        self.clock = Clock()
        service_kw.setdefault("progress_seconds", 0.05)
        self.service = Service(self.paths, self.backend, self.media, self.lock, clock=self.clock, **service_kw)

    @asynccontextmanager
    async def serve(self, *callers: Caller, warm: str | None = "turbo"):
        """One in-memory MCP client per caller, all on the same running service."""
        async with self.service.running(warm=warm):
            async with AsyncExitStack() as stack:
                clients = []
                for caller in callers:
                    mcp = build_server(self.service, lambda ctx, c=caller: c)
                    clients.append(await stack.enter_async_context(Client(mcp)))
                yield clients


async def call(client: Client, tool: str, **args) -> tuple[bool, dict]:
    """(ok, structured content) of one tool call."""
    result = await client.call_tool(tool, args)
    return (not result.is_error), result.structured_content


async def refused(client: Client, tool: str, **args) -> dict:
    ok, content = await call(client, tool, **args)
    assert not ok, f"{tool} was expected to be refused, got {content}"
    return content
