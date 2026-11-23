import json

import pytest

from emotion_render import cli, voices

CONSENT = {"person": "Test", "given": "2026-11-21", "scope": "tests"}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF0000WAVE")
    entries = {
        "t-en": {"lang": "en", "engine": "turbo"},
        "t-fr": {"lang": "fr", "engine": "multilingual", "defaults": {"exaggeration": 0.5}},
    }
    for voice_id, entry in entries.items():
        entry.update(person="Test", reference=str(ref), consent=CONSENT)
        (tmp_path / f"{voice_id}.json").write_text(json.dumps(entry), encoding="utf-8")
    monkeypatch.setattr(voices, "REGISTRY_DIR", tmp_path)
    return tmp_path


def _check(capsys, *argv):
    code = cli.main([*argv, "--check"])
    return code, capsys.readouterr()


def test_check_accepts_turbo_tags(registry, capsys):
    code, out = _check(capsys, "--voice", "t-en", "--text", "[Sigh] Fine.")
    assert code == 0
    plan = json.loads(out.out)
    assert plan["tags"] == ["sigh"] and plan["clean_text"] == "Fine."


def test_check_refuses_an_unknown_tag(registry, capsys):
    code, out = _check(capsys, "--voice", "t-en", "--text", "Hello [whisper]")
    assert code == 2 and "REFUSED" in out.err


def test_check_refuses_a_tag_in_french(registry, capsys):
    code, _ = _check(capsys, "--voice", "t-fr", "--text", "Bonjour [sigh]")
    assert code == 2


def test_exaggeration_overrides_the_voice_default(registry, capsys):
    code, out = _check(capsys, "--voice", "t-fr", "--text", "Bonjour.", "--exaggeration", "0.9")
    assert code == 0
    assert json.loads(out.out)["params"] == {"exaggeration": 0.9}


def test_turbo_refuses_a_setting_it_would_ignore(registry, capsys):
    code, out = _check(capsys, "--voice", "t-en", "--text", "Hello.", "--exaggeration", "0.7")
    assert code == 2 and "exaggeration" in out.err


def test_text_over_the_limit_is_refused(registry, capsys):
    code, _ = _check(capsys, "--voice", "t-en", "--text", "word " * 100)
    assert code == 2


def test_out_is_required_without_check(registry):
    with pytest.raises(SystemExit):
        cli.main(["--voice", "t-en", "--text", "Hello."])
