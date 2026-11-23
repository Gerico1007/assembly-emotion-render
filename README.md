# assembly-emotion-render

Acted speech in a voice cloned from a short recording. A script with emotion tags
goes in. A wav comes out, with a JSON record beside it saying how it was made. The
engine is Resemble AI's [Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT).
Made for the G.Music Assembly, whose assembly-voice will call it as a command.

No voice ships with this repo. Each person records their own and keeps it in their
own folder.

## Two rules

- **I1** A tag is consumed or refused. It is never spoken and never dropped
  silently. The same holds for a setting the engine would ignore.
- **I2** No cloned voice without the consent of the person whose voice it is. A
  voice entry without that person's consent is refused.

## Engines

| engine | languages | emotion |
|---|---|---|
| `turbo` | English | tags in the text: `[laugh] [chuckle] [sigh] [gasp] [cough] [clear throat] [sniff] [groan] [shush]` |
| `multilingual` | 23, French included | `--exaggeration` and `--cfg-weight` for the whole clip, no tags |

## Install

Needs Python 3.11, [uv](https://docs.astral.sh/uv/) and ffmpeg. The CUDA build of
torch below also runs on a machine without a GPU.

```bash
git clone https://github.com/Gerico1007/assembly-emotion-render
cd assembly-emotion-render
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.6.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
```

The first render downloads the model weights from Hugging Face, about 4 GB for `turbo`.

## Add your own voice

Your voices live in your own folder, `~/.local/share/assembly-emotion-render/voices/`,
outside this repo. Pick an id for the voice: lowercase letters, digits and dashes,
for example `ada-en`. Below, replace `<your-id>` with it.

**1. Record** about 10 seconds of clean speech: one speaker, no music, little room
noise. Chatterbox refuses a clip of 5 seconds or less. Convert it to mono WAV in your folder:

```bash
mkdir -p ~/.local/share/assembly-emotion-render/voices
ffmpeg -i my-recording.m4a -ac 1 -ar 24000 ~/.local/share/assembly-emotion-render/voices/<your-id>.wav
```

**2. Copy the template** into the same folder:

```bash
cp voices/example.json ~/.local/share/assembly-emotion-render/voices/<your-id>.json
```

**3. Fill it in.** For a voice called `ada-en`, the filled file reads:

```json
{
  "person": "Ada",
  "lang": "en",
  "engine": "turbo",
  "reference": "ada-en.wav",
  "consent": {"person": "Ada", "given": "2026-11-23", "scope": "my own projects"}
}
```

- `lang` and `engine`: `en` with `turbo` to use tags. Any of the 23 language codes
  with `multilingual`, for example `fr` for French. A `multilingual` entry may add
  `"defaults": {"exaggeration": 0.5, "cfg_weight": 0.5}`.
- `reference`: a file name is looked up in the same folder. A full path also works.
- `consent`: who agreed, when, and for what. It must name the same person as
  `person`. Copied as is, the template has `"consent": null` and is refused. If
  the voice is not yours, the consent has to come from the person it belongs to.

**4. Check, then render:**

```bash
.venv/bin/emotion-render --voice <your-id> --text "[sigh] Hello." --check
.venv/bin/emotion-render --voice <your-id> --text "[chuckle] Oh, you caught me." --out out/first.wav
```

`--check` answers in a second without loading a model. If something is missing,
it says what.

## Use

```bash
.venv/bin/emotion-render --voice <voice-id> --text "[chuckle] Oh, you caught me." --out out/x.wav
.venv/bin/emotion-render --voice <voice-id> --text "Bon, on continue." --exaggeration 0.7 --out out/y.wav  # a multilingual voice
.venv/bin/emotion-render --voice <voice-id> --text "Hello [whisper]" --check  # refused, exit 2
```

`--device auto` (the default) uses the GPU when the engine fits on it, the CPU
otherwise. Exit codes: 0 rendered or checked, 2 refused, 1 failed.

## Cost

Measured on one machine (GTX 1650 with 4 GB, 2026-11-23), for a clip of about 10 s:

| engine | device | render time per second of audio | peak RAM |
|---|---|---|---|
| `turbo` | GPU | about 0.6 s | 4.9 GB |
| `turbo` | CPU | about 4 s | 6.4 GB |
| `multilingual` | CPU | about 10 s | 6.9 GB |

`multilingual` does not fit on a 4 GB GPU. Loading a model adds 12 to 18 s per
command. The record beside each wav holds the timings and peak memory of that render.
