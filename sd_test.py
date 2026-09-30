"""
sd_test.py - makes one small test image with stable-diffusion.cpp.
Run:  python sd_test.py      (or double-click)
Shows everything the exe prints, then opens the picture if it worked.
"""
import os, subprocess, sys, time

EXE = r"C:\Users\wills\Downloads\sd-master-3f8527a-bin-win-cuda12-x64\sd-cli.exe"
MODEL = r"C:\Users\wills\OneDrive\Desktop\ggufllama\new\v1-5-pruned-emaonly.safetensors"
PROMPT = "a red apple on a wooden table, photo"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd_test.png")


def main():
    for label, path in (("exe", EXE), ("model", MODEL)):
        if not os.path.isfile(path):
            print(f"Can't find the {label}:\n  {path}\nEdit the paths at the top of this file.")
            return
    if os.path.exists(OUT):
        os.remove(OUT)
    cmd = [EXE, "-m", MODEL, "-p", PROMPT, "-W", "256", "-H", "256",
           "--steps", "4", "-o", OUT]
    print("Running (first run loads a 4 GB model, give it a minute)...\n")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", cwd=os.path.dirname(EXE))
    for line in proc.stdout:
        line = line.strip()
        if line:
            print("  " + line)
    code = proc.wait()
    print()
    if code == 0 and os.path.exists(OUT):
        print(f"SUCCESS in {time.time() - t0:.0f}s -> {OUT}")
        try:
            os.startfile(OUT)
        except (AttributeError, OSError):
            pass
    else:
        print(f"FAILED (exit code {code}). Copy the lines above and send them over.")


if __name__ == "__main__":
    try:
        main()
    finally:
        if os.name == "nt":
            input("\nPress Enter to close...")
