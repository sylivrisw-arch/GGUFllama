"""
check_requirements.py - checks what is installed and tells you what GGUFllama still needs.

Uses only the Python standard library, so it runs even on a fresh machine.
Run:  python check_requirements.py            (put it next to GGUFllama.py)
      python check_requirements.py --models "D:\\my\\gguf\\folder"   (check a different models folder)

Result levels:
  OK        found and working
  MISSING   REQUIRED - GGUFllama cannot run (or cannot load models) without it
  OPTIONAL  only needed for one feature; the rest of the app works without it
  INFO      just good to know
"""

import argparse
import importlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys

IS_WIN = os.name == "nt"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS = os.path.join(os.path.expanduser("~"), ".lmstudio", "models")
MIN_PY = (3, 9)
SD_VRAM_MB = 4000          # what GGUFllama's Images window assumes SD 1.5 needs

results = []               # (level, name, detail, fix)


def add(level, name, detail="", fix=""):
    results.append((level, name, detail, fix))


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def run(cmd, timeout=8):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           creationflags=0x08000000 if IS_WIN else 0)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as exc:                              # noqa: BLE001
        return -1, str(exc)


# ----------------------------------------------------------------------------- checks
def check_python():
    v = sys.version_info
    label = f"Python {v.major}.{v.minor}.{v.micro} ({platform.architecture()[0]})"
    if (v.major, v.minor) >= MIN_PY:
        add("OK", "Python", label)
    else:
        add("MISSING", "Python", label + f" - too old, need {MIN_PY[0]}.{MIN_PY[1]}+",
            "Install a newer Python from https://www.python.org/downloads/ "
            "(tick 'Add python.exe to PATH').")
    if not shutil.which("python") and not shutil.which("python3") and not shutil.which("py"):
        add("OPTIONAL", "Python on PATH",
            "not found on PATH - the Block Editor's Run / test-run launches 'python' and "
            "won't work (this matters mainly if GGUFllama is compiled to an .exe)",
            "Reinstall Python and tick 'Add python.exe to PATH', or add it in System > "
            "Environment Variables.")


def check_script():
    p = os.path.join(HERE, "GGUFllama.py")
    if os.path.isfile(p):
        add("OK", "GGUFllama.py", "found next to this script")
    else:
        add("INFO", "GGUFllama.py", "not found next to this script (fine if you run a compiled .exe)")


def check_tkinter():
    if has_module("tkinter"):
        try:
            import tkinter
            add("OK", "tkinter", f"Tk {tkinter.TkVersion}")
        except Exception as exc:                          # noqa: BLE001
            add("MISSING", "tkinter", f"present but won't import: {exc}",
                "Reinstall Python with the 'tcl/tk and IDLE' option ticked.")
    else:
        fix = ("Re-run the Python installer > Modify > tick 'tcl/tk and IDLE'." if IS_WIN else
               "Linux: sudo apt install python3-tk   |   macOS (Homebrew): brew install python-tk")
        add("MISSING", "tkinter", "not installed - the whole GUI needs it", fix)


def check_llama_cpp():
    if not has_module("llama_cpp"):
        add("MISSING", "llama-cpp-python",
            "not installed - the app opens but Authenticate is refused, so no model can load",
            "pip install llama-cpp-python   (for GPU use you need a CUDA build - see the "
            "llama-cpp-python docs for the prebuilt CUDA wheel index. GGUFllama is built "
            "against a pinned version, so install that one rather than the newest)")
        return
    try:
        llama_cpp = importlib.import_module("llama_cpp")
    except Exception as exc:                              # noqa: BLE001
        add("MISSING", "llama-cpp-python",
            f"installed but fails to import: {exc}",
            "Usually a missing CUDA/VC++ runtime or a wheel built for another Python. "
            "Reinstall the wheel that matches this Python version.")
        return
    ver = getattr(llama_cpp, "__version__", "unknown version")
    add("OK", "llama-cpp-python", f"version {ver}")
    fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    if fn is None:
        add("INFO", "GPU offload", "can't tell (this build doesn't expose llama_supports_gpu_offload)")
    else:
        try:
            if fn():
                add("OK", "GPU offload", "this llama-cpp-python build can use the GPU")
            else:
                add("MISSING", "GPU offload",
                    "this is a CPU-only build - models will run, but very slowly",
                    "Reinstall llama-cpp-python as a CUDA build (same version you pinned).")
        except Exception as exc:                          # noqa: BLE001
            add("INFO", "GPU offload", f"check failed: {exc}")
    add("INFO", "Model formats",
        "this build can only load quantizations that existed when it was built - "
        "newer ones (e.g. MXFP4) fail to load until llama-cpp-python is upgraded")


def check_gpu():
    exe = shutil.which("nvidia-smi")
    if not exe:
        add("OPTIONAL", "NVIDIA GPU", "nvidia-smi not found - no NVIDIA driver/GPU detected, "
            "so Local models would run on the CPU",
            "Install the NVIDIA driver from https://www.nvidia.com/drivers")
        return
    code, out = run([exe, "--query-gpu=name,memory.total,memory.free,driver_version",
                     "--format=csv,noheader,nounits"])
    if code != 0 or not out.strip():
        add("OPTIONAL", "NVIDIA GPU", "nvidia-smi ran but returned nothing useful", "")
        return
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            name, total, free, drv = parts[:4]
            add("OK", "NVIDIA GPU", f"{name} - {total} MB VRAM ({free} MB free), driver {drv}")
            try:
                if int(free) < SD_VRAM_MB:
                    add("INFO", "VRAM for Images",
                        f"only {free} MB free right now; the Images window wants ~{SD_VRAM_MB} MB "
                        "(unload the chat model first)")
            except ValueError:
                pass


def find_ggufs(root, limit=2000):
    found, total = [], 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".gguf"):
                total += 1
                if len(found) < 5:
                    found.append(os.path.join(dirpath, f))
                if total >= limit:
                    return found, total
    return found, total


def check_models(folder):
    if not os.path.isdir(folder):
        add("MISSING", "Models folder", f"{folder} does not exist",
            "Create it, or right-click GGUFllama's title bar > 'Local Models Folder' and "
            "point it at the folder that holds your .gguf files (or re-run this check with "
            "--models <folder>).")
        return
    found, total = find_ggufs(folder)
    if total == 0:
        add("MISSING", "Models folder", f"{folder} exists but has no .gguf files",
            "Download a .gguf model (e.g. from Hugging Face) into that folder.")
    else:
        add("OK", "Models folder", f"{folder} - {total}{'+' if total >= 2000 else ''} .gguf file(s)")


def check_flask():
    if has_module("flask"):
        try:
            import flask  # noqa: F401
            from importlib import metadata
            ver = metadata.version("flask")
        except Exception:                                 # noqa: BLE001
            ver = "?"
        add("OK", "Flask (Block Editor)", f"version {ver}")
    else:
        add("OPTIONAL", "Flask (Block Editor)",
            "not installed - only the Block Editor needs it, the rest of the app works",
            "pip install flask")


def check_speech():
    if not IS_WIN:
        add("OPTIONAL", "Speak (text-to-speech)",
            "the 'Speak' checkbox uses Windows System.Speech, so it only works on Windows", "")
        return
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        add("OPTIONAL", "Speak (text-to-speech)", "PowerShell not found on PATH",
            "Make sure C:\\Windows\\System32\\WindowsPowerShell\\v1.0 is on PATH.")
        return
    code, out = run([ps, "-NoProfile", "-Command",
                     "Add-Type -AssemblyName System.Speech; "
                     "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                     ".GetInstalledVoices().Count"], timeout=20)
    if code == 0 and out.strip().isdigit() and int(out.strip()) > 0:
        add("OK", "Speak (text-to-speech)", f"PowerShell + {out.strip()} installed voice(s)")
    else:
        add("OPTIONAL", "Speak (text-to-speech)",
            "PowerShell found but System.Speech or an installed voice is unavailable",
            "Windows Settings > Time & language > Speech > add a voice (a male voice suits "
            "the Terminator persona).")


def check_audio():
    if IS_WIN:
        add("OK" if has_module("winsound") else "OPTIONAL", "Music playback",
            "winsound available" if has_module("winsound") else "winsound missing", "")
    else:
        player = next((p for p in ("afplay", "aplay", "paplay") if shutil.which(p)), None)
        if player:
            add("OK", "Music playback", f"using {player}")
        else:
            add("OPTIONAL", "Music playback",
                "no audio player found (afplay / aplay / paplay) - the Music window can "
                "still 'Save WAV...'", "Linux: sudo apt install alsa-utils")


def check_sd():
    names = ("sd-cli.exe", "sd.exe") if IS_WIN else ("sd-cli", "sd")
    exe = next((shutil.which(n) for n in names if shutil.which(n)), None)
    if not exe:
        for n in names:
            for folder in (HERE, os.path.join(HERE, "sd"), os.path.join(HERE, "stable-diffusion.cpp")):
                p = os.path.join(folder, n)
                if os.path.isfile(p):
                    exe = p
                    break
            if exe:
                break
    if exe:
        add("OK", "stable-diffusion.cpp", exe)
    else:
        add("OPTIONAL", "stable-diffusion.cpp",
            "sd-cli.exe / sd.exe not found on PATH or next to this script - only the Images "
            "window needs it (you can also just browse to it inside that window)",
            "Download a prebuilt release from https://github.com/leejet/stable-diffusion.cpp/releases "
            "and choose it in Images (SD 1.5). You also need an SD 1.5 checkpoint (.safetensors/.ckpt).")
    ckpts = []
    for folder in (HERE, os.path.join(HERE, "models"), os.path.join(HERE, "sd")):
        if os.path.isdir(folder):
            ckpts += [f for f in os.listdir(folder)
                      if f.lower().endswith((".safetensors", ".ckpt"))]
    if ckpts:
        add("INFO", "SD 1.5 checkpoint", f"possible checkpoint(s) near this script: {', '.join(ckpts[:3])}")


def check_lore():
    for name in ("Terminator_Lore_Bible_v2.md", "Terminator.txt"):
        if os.path.isfile(os.path.join(HERE, name)):
            add("OK", "Terminator lore file", f"{name} found next to this script")
            return
    add("OPTIONAL", "Terminator lore file",
        "Terminator_Lore_Bible_v2.md / Terminator.txt not next to this script - the persona "
        "still works but has no lore to ground on (you can pick one with 'Lore file...')",
        "Put the lore bible in the same folder as GGUFllama.py.")


def check_ram():
    total = None
    try:
        if IS_WIN:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong),
                            ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                            ("ae", ctypes.c_ulonglong)]
            ms = MS()
            ms.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            total = ms.tp / 1024 ** 3
        elif os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                total = int(f.readline().split()[1]) / 1024 ** 2
    except Exception:                                     # noqa: BLE001
        pass
    if total:
        add("INFO", "System RAM", f"{total:.1f} GB (matters for 'Allow RAM spillover' with big models)")


# ----------------------------------------------------------------------------- report
COLORS = {"OK": "\033[32m", "MISSING": "\033[31m", "OPTIONAL": "\033[33m", "INFO": "\033[36m"}
RESET = "\033[0m"


def enable_colors():
    if not sys.stdout.isatty():
        return False
    if IS_WIN:
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:                                 # noqa: BLE001
            return False
    return True


def report(color):
    def c(level):
        return f"{COLORS[level]}{level:<8}{RESET}" if color else f"{level:<8}"

    print("=" * 72)
    print(" GGUFllama requirements check")
    print(f" {platform.system()} {platform.release()}  |  Python {platform.python_version()}")
    print("=" * 72)
    for level, name, detail, fix in results:
        print(f" {c(level)} {name}")
        if detail:
            print(f"          {detail}")
        if fix and level in ("MISSING", "OPTIONAL"):
            print(f"          -> {fix}")
    missing = [r for r in results if r[0] == "MISSING"]
    optional = [r for r in results if r[0] == "OPTIONAL"]
    print("-" * 72)
    if missing:
        print(f" NOT READY: {len(missing)} required item(s) missing:")
        for _l, name, _d, _f in missing:
            print(f"   - {name}")
    else:
        print(" READY: everything GGUFllama needs to run chat is in place.")
    if optional:
        print(f" {len(optional)} optional feature(s) not set up: "
              + ", ".join(r[1] for r in optional))
    print("=" * 72)
    return 1 if missing else 0


def main():
    ap = argparse.ArgumentParser(description="Check what GGUFllama needs.")
    ap.add_argument("--models", default=DEFAULT_MODELS, help="folder holding your .gguf files")
    args = ap.parse_args()

    check_python()
    check_script()
    check_tkinter()
    check_llama_cpp()
    check_gpu()
    check_ram()
    check_models(args.models)
    check_flask()
    check_speech()
    check_audio()
    check_sd()
    check_lore()

    code = report(enable_colors())
    # Keep the window open when double-clicked on Windows.
    if IS_WIN and sys.stdin and sys.stdin.isatty() and not os.environ.get("PROMPT"):
        try:
            input("\nPress Enter to close...")
        except EOFError:
            pass
    sys.exit(code)


if __name__ == "__main__":
    main()
