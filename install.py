"""
Geekatplay GenUpscale - Installer
by Geekatplay Studio / Vladimir Chopine
https://www.geekatplay.com

Downloads required models and installs missing custom node packs.
Run from anywhere:  python install.py
"""
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
COMFY_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODELS = os.path.join(COMFY_ROOT, "models")

# No custom node packs required - everything runs on ComfyUI core plus the
# nodes built into this pack.

MODEL_DOWNLOADS = [
    # (subdir, filename, url, approx size, needed by)
    ("controlnet", "Flux.1-dev-Controlnet-Upscaler.safetensors",
     "https://huggingface.co/f5aiteam/Controlnet/resolve/main/Flux.1-dev-Controlnet-Upscaler.safetensors",
     "3.6 GB", "Quality (FLUX) workflow"),
    ("upscale_models", "4xNomos8kDAT.pth",
     "https://huggingface.co/Maxivi/SDXLModels/resolve/main/4xNomos8kDAT.pth",
     "155 MB", "Quality (FLUX) workflow"),
    ("upscale_models", "4xClearRealityV1.pth",
     "https://huggingface.co/Kim2091/ClearRealityV1/resolve/main/4x-ClearRealityV1.pth",
     "9 MB", "Fast (SDXL) workflow"),
    ("controlnet", "controlnet_tile_sdxl_1_0.safetensors",
     "https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
     "2.5 GB", "Fast (SDXL) workflow"),
]


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url, dest):
    tmp = dest + ".part"

    def hook(blocks, bs, total):
        done = blocks * bs
        pct = f" {done * 100 // total}%" if total > 0 else ""
        sys.stdout.write(f"\r    {human(done)}{pct}   ")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    os.replace(tmp, dest)
    print()


def main():
    print("=" * 60)
    print(" Geekatplay GenUpscale installer")
    print(" Geekatplay Studio / Vladimir Chopine")
    print("=" * 60)
    print(f"ComfyUI root: {COMFY_ROOT}")
    if not os.path.isdir(MODELS):
        print("ERROR: models folder not found. Place this pack inside "
              "ComfyUI/custom_nodes/ and run again.")
        sys.exit(1)

    print("\n-- Models --")
    for sub, fname, url, size, used_by in MODEL_DOWNLOADS:
        folder = os.path.join(MODELS, sub)
        os.makedirs(folder, exist_ok=True)
        dest = os.path.join(folder, fname)
        if os.path.isfile(dest):
            print(f"  [ok] {sub}/{fname}")
            continue
        answer = input(f"  Download {fname} ({size}, for {used_by})? [Y/n] ").strip().lower()
        if answer in ("", "y", "yes"):
            print(f"  [..] {url}")
            download(url, dest)
            print(f"  [ok] saved to {sub}/{fname}")
        else:
            print(f"  [skip] {fname}")

    print("\nDone! Restart ComfyUI, then load a workflow from:")
    print(f"  {os.path.join(HERE, 'workflows')}")


if __name__ == "__main__":
    main()
