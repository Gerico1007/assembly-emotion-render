import pytest

from emotion_render import Refused
from emotion_render.engines import ENGINES
from emotion_render.script import parse, validate

TURBO = ENGINES["turbo"].tags
MULTILINGUAL = ENGINES["multilingual"].tags


def test_tags_are_normalized_and_kept_in_order():
    s = parse("[Chuckle] Oh, you caught me! [clear   throat] Well.")
    assert s.tags == ("chuckle", "clear throat")
    assert s.engine_text == "[chuckle] Oh, you caught me! [clear throat] Well."
    assert s.clean_text == "Oh, you caught me! Well."


def test_a_script_without_tags_passes_both_engines():
    s = parse("Bon, on continue.")
    validate(s, TURBO, "turbo")
    validate(s, MULTILINGUAL, "multilingual")
    assert s.tags == ()


def test_turbo_tags_pass_turbo():
    validate(parse("[sigh] Fine. [laugh]"), TURBO, "turbo")


def test_unknown_tag_is_refused_with_the_accepted_list():
    with pytest.raises(Refused) as e:
        validate(parse("Hello [whisper] there"), TURBO, "turbo")
    assert "[whisper]" in str(e.value)
    assert "[sigh]" in str(e.value)


def test_multilingual_refuses_every_tag():
    with pytest.raises(Refused) as e:
        validate(parse("Bonjour [sigh] toi"), MULTILINGUAL, "multilingual")
    assert "takes no tags" in str(e.value)


@pytest.mark.parametrize(
    "raw",
    ["Hello [sigh there", "Hello sigh] there", "[[laugh]] ha", "Hi [] there"],
)
def test_brackets_that_are_not_whole_tags_are_refused(raw):
    with pytest.raises(Refused):
        parse(raw)


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_script_is_refused(raw):
    with pytest.raises(Refused):
        parse(raw)
