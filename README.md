# GGUFllama

A lightweight, single-file Tkinter chat app for running local **GGUF** models directly on your GPU. No server, no network hop: models run in-process through [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python).

Built for Windows with an NVIDIA GPU. Includes a two-model **Discussion mode**, reasoning controls, an optional local HTTP API, a built-in **Block Editor**, and **Stable Diffusion 1.5** image generation via [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

## Features

- **Chat** with streaming replies, a live token counter, and a separate muted "Thinking" block for reasoning models.
- **Discussion mode**: load two models (A and B) at once and let them talk to each other. You can interject mid-discussion.
- **Models window**: browse a folder of `.gguf` files, load/unload/reload, set per-model context length, and assign Discussion A/B roles.
- **VRAM guard** that refuses loads that won't fit, with an opt-in **RAM spillover** mode that splits layers between GPU and system RAM.
- **Auto-max context** toggle to load every model at its maximum context length.
- **Reasoning controls**: on/off, Low/Medium level, and show/hide thinking (GPT-OSS and Qwen3/3.5 aware).
- **Output handling** for GPT-OSS (Harmony channels) and Qwen (implicit `<think>`), so reasoning and final answers are separated correctly.
- **Unload cancels generation**: unloading a model mid-reply stops it cleanly and frees VRAM.
- **Constraints Engine**: 22 toggleable output constraints (15 separate, 7 collective), embedded in the script.
- **Simple Mode** hides everything except the chat log and input box.
- **Notes** window, chat-log right-click menu (copy, character count, export to Notes), and a configurable Save Locations folder.
- **HTTP API** (off by default) so other local programs can use the loaded model.
- **Block Editor**: describe a small Tkinter app in plain English, get an editable block graph, and generate runnable Tkinter code.
- **Images (SD 1.5)**: generate images with stable-diffusion.cpp from inside the app.

## Requirements

- Windows, Python 3.10+
- An NVIDIA GPU (developed on an RTX 5060 Laptop, 8 GB VRAM)
- `llama-cpp-python` built with CUDA. The app opens without it, but Connect is refused.
- `flask` (optional, only for the Block Editor): `pip install flask`
- `.gguf` model files. The default models folder is `~/.lmstudio/models`, so existing LM Studio downloads are picked up automatically. Change it from the title-bar menu.

> **Note:** `llama-cpp-python` is pinned to a fixed version because upgrading has been unreliable. It can only load quantization formats that existed when that version was built, so newer formats (for example MXFP4) may fail to load.

## Run

```
python GGUFllama.py
```

1. Press **Connect** (checks the models folder).
2. Open **Models** and double-click a model to load it.
3. Type and press **Send**.

For Discussion mode, right-click two models and choose **Set as Discussion A / B**, load both, switch to Discussion, and enter a topic.

Right-click the **title bar** for Save Locations, Local Models Folder, Start/Stop HTTP API, Drawing, Music, Images (SD 1.5), and the Block Editor.

## HTTP API (optional)

Start it from the title-bar menu. It listens on `127.0.0.1:5178` only and never loads or unloads models itself.

| Endpoint | Purpose |
|---|---|
| `GET /api/status` | `{loaded, model, path, ctx, busy}` |
| `POST /api/chat` | Body `{messages, schema?, temperature?, max_tokens?}`, returns `{content, model}` |

If `schema` is supplied, the reply is constrained to that JSON schema. Requests queue behind any chat reply in progress, and unloading the model cancels them. There is no authentication, so keep it on localhost.

## Block Editor

Title-bar menu, **Open Block Editor** (runs at `http://127.0.0.1:5177`, needs Flask). It uses the model loaded in GGUFllama to turn a plain-English description into a validated block graph (Window, Button, Label, Entry, HttpGet, ReadFile, WriteFile, Timer, If, Print), then regenerates Tkinter code live with Copy, Save `.py` and Run.

## Image generation (Stable Diffusion 1.5)

Runs the prebuilt stable-diffusion.cpp executable as a subprocess, so it cannot clash with llama.cpp and frees its VRAM when done.

1. From the [stable-diffusion.cpp releases](https://github.com/leejet/stable-diffusion.cpp/releases), download the Windows **cuda12** zip and the **cudart** zip. Extract both, and put the cudart DLLs next to `sd-cli.exe`.
2. Download an SD 1.5 checkpoint (for example `v1-5-pruned-emaonly.safetensors` from Hugging Face).
3. Title-bar menu, **Images (SD 1.5)**, then browse to the exe and checkpoint, write a prompt, and press **Generate**.

Use 512x512 and about 20 steps for good results. Smaller sizes and very low step counts give abstract blurs. Unload any large LLM first, since 8 GB is tight for both at once.

`sd_setup_check.py` checks and fixes the setup, and `sd_test.py` generates a quick test image.

## Good to know

- Most settings are **session-only** and reset on each launch.
- Temperature is fixed at 0.7 in the chat UI. The HTTP API takes its own per-request value.
- The Harmony filter and reasoning controls are not applied to HTTP API requests.
- Not yet wired in: the role prompts in the Prompts window and Constraints Engine validation of replies.
