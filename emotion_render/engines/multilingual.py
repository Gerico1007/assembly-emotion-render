"""Chatterbox Multilingual: 23 languages, French included. Emotion is one
exaggeration level per call, not tags.

API checked against the installed chatterbox-tts 0.1.7 source (A2, 2026-11-23).
"""


def load(device: str):
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    return ChatterboxMultilingualTTS.from_pretrained(device=device)


def generate(model, text: str, reference: str, lang: str, params: dict):
    return model.generate(
        text,
        language_id=lang,
        audio_prompt_path=reference,
        **params,
    )
