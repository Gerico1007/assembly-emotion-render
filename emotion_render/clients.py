"""Who may call the MCP server over HTTP: one line per client device, the token's hash only (D13).

    ~/.config/assembly-emotion-render/clients.json
    {"clients": {"bea": {"tokens": [{"device": "vm", "sha256": "…", "created_at": "…"}]}}}

A raw token exists in exactly one place: the terminal of whoever ran `add-client`,
once. It is never written here, never logged, never returned by the server.
Revoking a device deletes its line; the server reads this file on each request,
so a revocation holds at once.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .voices import ID_RULE, is_id

DEFAULT_PATH = Path.home() / ".config" / "assembly-emotion-render" / "clients.json"
TOKEN_PREFIX = "er_"


class ClientsError(ValueError):
    pass


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class ClientsFile:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_PATH

    def _read(self) -> dict:
        if not self.path.is_file():
            return {"clients": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data.setdefault("clients", {})
        return data

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".clients.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(data, indent=2) + "\n")
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def add(self, client_id: str, device: str) -> str:
        """Create a token for one device of one client. Returns the raw token, the only time it exists."""
        if not is_id(client_id):
            raise ClientsError(f"'{client_id}' is not a client id ({ID_RULE})")
        if not is_id(device):
            raise ClientsError(f"'{device}' is not a device name ({ID_RULE})")
        data = self._read()
        tokens = data["clients"].setdefault(client_id, {"tokens": []})["tokens"]
        if any(t["device"] == device for t in tokens):
            raise ClientsError(f"{client_id} already has a token for device {device}. Remove it first")
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        tokens.append({
            "device": device,
            "sha256": _hash(token),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        self._write(data)
        return token

    def remove(self, client_id: str, device: str | None = None) -> int:
        """Revoke one device, or the whole client. Returns how many tokens were removed."""
        data = self._read()
        entry = data["clients"].get(client_id)
        if entry is None:
            raise ClientsError(f"no client '{client_id}'")
        before = len(entry["tokens"])
        if device is None:
            del data["clients"][client_id]
            removed = before
        else:
            entry["tokens"] = [t for t in entry["tokens"] if t["device"] != device]
            removed = before - len(entry["tokens"])
            if removed == 0:
                raise ClientsError(f"{client_id} has no token for device {device}")
            if not entry["tokens"]:
                del data["clients"][client_id]
        self._write(data)
        return removed

    def list(self) -> list[dict]:
        rows = []
        for client_id, entry in sorted(self._read()["clients"].items()):
            for t in entry["tokens"]:
                rows.append({"client": client_id, "device": t["device"], "created_at": t.get("created_at")})
        return rows

    def verify(self, token: str) -> tuple[str, str] | None:
        """(client_id, device) for a valid token, None otherwise. Every hash is compared."""
        if not token or not token.startswith(TOKEN_PREFIX):
            return None
        digest = _hash(token)
        found = None
        try:
            clients = self._read()["clients"]
        except (OSError, ValueError):
            return None
        for client_id, entry in clients.items():
            for t in entry.get("tokens", []):
                if hmac.compare_digest(digest, t.get("sha256", "")) and found is None:
                    found = (client_id, t["device"])
        return found
