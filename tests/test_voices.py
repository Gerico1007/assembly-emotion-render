import json
import shutil
from pathlib import Path

import pytest

from emotion_render import Refused, voices

REPO_VOICES = Path(__file__).resolve().parent.parent / "voices"
CONSENT = {"person": "Test", "given": "2026-11-21", "scope": "tests"}


def _entry(tmp_path, voice_id="t-en", **over):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF0000WAVE")
    entry = {
        "person": "Test",
        "lang": "en",
        "engine": "turbo",
        "reference": str(ref),
        "consent": dict(CONSENT),
    }
    entry.update(over)
    (tmp_path / f"{voice_id}.json").write_text(json.dumps(entry), encoding="utf-8")
    return tmp_path


def test_a_voice_with_its_own_consent_loads(tmp_path):
    v = voices.load("t-en", _entry(tmp_path))
    assert v.person == "Test" and v.engine == "turbo" and v.lang == "en"


def test_a_bare_reference_name_is_found_in_the_store(tmp_path):
    v = voices.load("t-en", _entry(tmp_path, reference="ref.wav"))
    assert v.reference == tmp_path / "ref.wav"


def test_no_consent_is_refused(tmp_path):
    with pytest.raises(Refused) as e:
        voices.load("t-en", _entry(tmp_path, consent=None))
    assert "I2" in str(e.value)


def test_incomplete_consent_is_refused(tmp_path):
    with pytest.raises(Refused):
        voices.load("t-en", _entry(tmp_path, consent={"person": "Test", "given": "2026-11-21"}))


def test_consent_given_by_someone_else_is_refused(tmp_path):
    other = dict(CONSENT, person="Someone Else")
    with pytest.raises(Refused) as e:
        voices.load("t-en", _entry(tmp_path, consent=other))
    assert "Someone Else" in str(e.value)


def test_turbo_refuses_french(tmp_path):
    with pytest.raises(Refused):
        voices.load("t-en", _entry(tmp_path, lang="fr"))


def test_multilingual_speaks_french(tmp_path):
    v = voices.load("t-fr", _entry(tmp_path, "t-fr", lang="fr", engine="multilingual",
                                   defaults={"exaggeration": 0.5}))
    assert v.defaults == {"exaggeration": 0.5}


def test_defaults_the_engine_does_not_take_are_refused(tmp_path):
    with pytest.raises(Refused):
        voices.load("t-en", _entry(tmp_path, defaults={"exaggeration": 0.5}))


def test_missing_reference_is_refused(tmp_path):
    with pytest.raises(Refused):
        voices.load("t-en", _entry(tmp_path, reference=str(tmp_path / "nowhere.wav")))


def test_unknown_voice_lists_the_known_ones_and_points_to_the_readme(tmp_path):
    with pytest.raises(Refused) as e:
        voices.load("nobody", _entry(tmp_path))
    assert "t-en" in str(e.value)
    assert "Add your own voice" in str(e.value)


def test_an_empty_store_says_so(tmp_path):
    with pytest.raises(Refused) as e:
        voices.load("me", tmp_path / "does-not-exist")
    assert "Known voices: none" in str(e.value)


@pytest.mark.parametrize("bad", ["../etc", "Ada", "a/b", ""])
def test_voice_ids_cannot_reach_outside_the_registry(tmp_path, bad):
    with pytest.raises(Refused):
        voices.load(bad, _entry(tmp_path))


def test_the_store_is_outside_the_repo():
    repo_root = REPO_VOICES.parent.resolve()
    assert repo_root not in voices.REGISTRY_DIR.resolve().parents


def test_the_repo_ships_only_the_template():
    assert sorted(p.name for p in REPO_VOICES.iterdir()) == ["example.json"]


def test_the_template_copied_as_is_is_refused_for_consent(tmp_path):
    shutil.copy(REPO_VOICES / "example.json", tmp_path / "me.json")
    with pytest.raises(Refused) as e:
        voices.load("me", tmp_path)
    assert "I2" in str(e.value)
