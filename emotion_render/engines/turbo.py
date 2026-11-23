"""Chatterbox Turbo: English, paralinguistic tags inside the text.

API checked against the installed chatterbox-tts 0.1.7 source (A2, 2026-11-23).
"""


def load(device: str):
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    return ChatterboxTurboTTS.from_pretrained(device=device)


def generate(model, text: str, reference: str, lang: str, params: dict):
    # Turbo takes no generation settings here: EngineSpec.params is empty, so
    # the CLI has already refused any that were asked for.
    return model.generate(text, audio_prompt_path=reference)
