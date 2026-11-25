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
The `test` extra includes the MCP SDK. Without the tests, `-e '.[mcp]'` installs the
MCP server and `-e .` the command line alone.

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

## MCP server

`emotion-render-mcp` serves the same renderer to any client that speaks the Model
Context Protocol, so an agent can render a voice or add one by calling a tool. It
refuses what the command line refuses, with the same codes and messages. The one
difference is `unknown_voice` over HTTP: it lists only the voices that client may use.

### Tools

| tool | what it does |
|---|---|
| `list_voices` | the voices you may use, with the tags and settings each engine takes |
| `check` | everything `--check` does, plus the device a render would use. No model, under a second |
| `render` | renders, returns the record and a link to the audio |
| `get_render` | the result of a render that took longer than `wait_seconds` |
| `keep_render` | how long a render is kept: 1 to 3650 days, `"forever"`, or 0 to delete it now |
| `add_voice` | adds a voice to the store, owned by you. Consent is required, as for a file (I2) |
| `remove_voice` | removes your voice, its clip, and every render of it, kept forever or not |

A render's audio is a resource, `emotion-render://renders/<render-id>.mp3` (also
`.wav`, `.json`), read with `resources/read`. `inline_audio: true` puts the audio in
the result instead. A render waits up to `wait_seconds` (default 60, at most 120),
then answers `"status": "running"` with a `render_id` for `get_render`. Renders are
kept 7 days unless `keep_days` says otherwise. A refusal is a tool error whose
content is `{"code", "message", ...}`, with what is accepted when that helps.

`add_voice` takes the recording as `{"data_base64": "...", "mime_type": "audio/..."}`
(up to 5 MB) or as `{"inbox_file": "<name>"}`, a file already in
`~/.local/share/assembly-emotion-render/inbox/<client>/`. More than 5 s and at most 30 s.

### On your own machine: stdio

No token and no port. It serves your own store, every voice in it.

```bash
claude mcp add emotion-render -- /path/to/assembly-emotion-render/.venv/bin/emotion-render-mcp stdio
```

Voices added through it are owned by `local`, or by the name given with `--client`.

### On a shared host: HTTP

The server listens on loopback and tailnet addresses only (`127.0.0.1`, `::1`,
`100.64.0.0/10`, `fd7a:115c:a1e0::/48`) and refuses to start on any other. Every
request needs a bearer token. Any request carrying an `Origin` header, which is a web
page in a browser, is refused.

Each client device gets its own token. It is printed once, and only its SHA-256 is
kept, in `~/.config/assembly-emotion-render/clients.json`:

```bash
emotion-render-mcp add-client <client> --device <device>    # prints the token, once
emotion-render-mcp remove-client <client> [--device <device>]
emotion-render-mcp list-clients                              # never shows a token
```

Over HTTP, a voice is visible only to its owner and the clients it allows. Both are
fields of the voice's entry, set with:

```bash
emotion-render-mcp access <voice-id> --owner <client> --allow <client> [--allow <client> ...]
emotion-render-mcp access <voice-id> --allow-none
```

A voice with no owner is not reachable over HTTP. To any other client, a voice it
may not use does not exist: it answers `unknown_voice`.

Run it by hand:

```bash
emotion-render-mcp http --bind 127.0.0.1:8772 --bind <tailnet-address>:8772
claude mcp add --transport http emotion-render http://127.0.0.1:8772/mcp --header "Authorization: Bearer <token>"
```

Or on demand, with systemd socket activation: the first connection starts the
server, and it exits after 15 minutes without a request. Only the socket stays, and
it holds no model, no GPU and no memory.

```ini
# ~/.config/systemd/user/emotion-render-mcp.socket
[Socket]
ListenStream=127.0.0.1:8772
ListenStream=<tailnet-address>:8772
FreeBind=yes

[Install]
WantedBy=sockets.target
```

```ini
# ~/.config/systemd/user/emotion-render-mcp.service (no [Install]: only a connection starts it)
[Service]
WorkingDirectory=%h/<path>/assembly-emotion-render
ExecStart=%h/<path>/assembly-emotion-render/.venv/bin/emotion-render-mcp http --systemd-sockets --idle-minutes 15
```

```bash
systemctl --user daemon-reload && systemctl --user enable --now emotion-render-mcp.socket
```

Each call is one line in `~/.local/state/assembly-emotion-render/audit.jsonl`: time,
client, device, tool, voice, outcome. Never the script, never a token.

### GPU and memory

One render at a time. Up to three wait, and a fourth is refused with `busy`. The
server warms `turbo` on the GPU at start. Only one process holds the GPU at a time
(`~/.cache/assembly-emotion-render/gpu.lock`): while the server holds it, a command
line render with `--device auto` runs on the CPU, and `--device cuda` is refused with
`gpu_busy`. `multilingual` loads on the CPU when asked, and unloads after 10 idle minutes.

Measured on a GTX 1650 with 4 GB, 2026-11-25:

| what | measured |
|---|---|
| server start to first `turbo` render | 19 to 27 s |
| server at rest, `turbo` warm | 2.3 GB RAM, 3.4 GB GPU |
| `turbo` on the GPU, 142 characters | 4.9 s for 10.3 s of audio |
| `turbo` on the GPU, 300 characters | 8.2 s for 18.5 s of audio |
| `turbo` on the GPU, 400 characters | out of GPU memory |
| `multilingual` on the CPU beside `turbo` | 6 GB more at peak, about 8 s per second of audio |

So on such a card, a script over 300 characters renders on the CPU (`--device cuda`
is refused with `text_too_long_for_gpu`), and loading `multilingual` beside `turbo`
needs 6.5 GB free.

## Cost

Measured on one machine (GTX 1650 with 4 GB, 2026-11-23), for a clip of about 10 s:

| engine | device | render time per second of audio | peak RAM |
|---|---|---|---|
| `turbo` | GPU | about 0.6 s | 4.9 GB |
| `turbo` | CPU | about 4 s | 6.4 GB |
| `multilingual` | CPU | about 10 s | 6.9 GB |

`multilingual` does not fit on a 4 GB GPU. Loading a model adds 12 to 18 s per
command. The record beside each wav holds the timings and peak memory of that render.
