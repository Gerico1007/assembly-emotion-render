"""B3: clients.json holds hashed per-device tokens, never a raw one."""
import stat

import pytest

from emotion_render.clients import ClientsError, ClientsFile


@pytest.fixture
def clients(tmp_path):
    return ClientsFile(tmp_path / "config" / "clients.json")


def test_a_token_is_returned_once_and_only_its_hash_is_kept(clients):
    token = clients.add("bea", "vm")
    raw = clients.path.read_text()
    assert token.startswith("er_") and len(token) > 40
    assert token not in raw and token[3:] not in raw
    assert stat.S_IMODE(clients.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(clients.path.parent.stat().st_mode) == 0o700
    assert clients.verify(token) == ("bea", "vm")


def test_wrong_or_malformed_tokens_are_rejected(clients):
    token = clients.add("bea", "vm")
    for bad in ("", "er_", token + "x", token[:-1], "Bearer " + token, token[3:]):
        assert clients.verify(bad) is None


def test_one_token_per_device_and_revoking_one_keeps_the_others(clients):
    phone = clients.add("ada", "phone")
    ipad = clients.add("ada", "ipad")
    with pytest.raises(ClientsError):
        clients.add("ada", "phone")
    assert clients.remove("ada", "phone") == 1
    assert clients.verify(phone) is None
    assert clients.verify(ipad) == ("ada", "ipad")
    assert clients.list() == [{"client": "ada", "device": "ipad", "created_at": clients.list()[0]["created_at"]}]
    assert clients.remove("ada") == 1
    assert clients.verify(ipad) is None and clients.list() == []


@pytest.mark.parametrize("client, device", [("Bea", "vm"), ("bea", "my vm"), ("../x", "vm")])
def test_ids_are_checked(clients, client, device):
    with pytest.raises(ClientsError):
        clients.add(client, device)
