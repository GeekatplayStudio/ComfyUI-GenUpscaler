# Geekatplay GenUpscale

**Content-aware generative AI upscaler suite for ComfyUI**
by **Geekatplay Studio / Vladimir Chopine** — https://www.geekatplay.com

Upscale images to extreme sizes, split into tiles, and let a diffusion model
*regenerate* the fine detail that was never in the small image — fabric weave,
skin pores, foliage, material texture — while keeping structure locked to the
original. One knob controls how **creative vs. original** the result is.

## How it works

```
input image
   └─> GAN pre-upscale (4xNomos8kDAT / ClearReality SPAN)
        └─> extra scale (choose 2x / 4x / 8x total)
             └─> Auto Tile Planner (even 1024px tile grid, overlap)
                  └─> Tiled Generative Refine (built into this pack)
                      + tile ControlNet locks every tile to the source
                      + Detail Control preset drives denoise / CN strength
                       └─> Color Match GPU against original -> Save + Compare
```

## Included workflows

| Workflow | Engine | VRAM | Use for |
|---|---|---|---|
| `GAP-GenUpscale-Quality-FLUX.json` | FLUX.1-dev fp8 + jasperai Upscaler ControlNet | 16–24 GB | Best realism and faithfulness |
| `GAP-GenUpscale-Fast-SDXL.json` | SDXL + xinsir Tile ControlNet | 8–12 GB | Fast batches; swap in a photoreal checkpoint (Juggernaut XL, RealVisXL) for best results |

## Custom nodes

- **Detail Control (Creative vs Original)** — the main knob. Presets
  `archival → faithful → balanced → detailed → creative` (or a 0–1 slider)
  are mapped to research-tuned denoise, tile-ControlNet strength/end-percent,
  steps, CFG and Detail-Daemon amount for FLUX, SDXL or Qwen.
  - *archival* ≈ denoise 0.15, CN strong: pixel-faithful cleanup
  - *balanced* ≈ denoise 0.33: regenerates lost micro-texture on fabric/skin
  - *creative* ≈ denoise 0.5+, CN relaxed: bold reinterpretation of detail
- **Auto Tile Planner** — computes an even tile grid (width/height/padding/
  mask-blur) from the final image size. No more manual tile math.
- **Upscale Prompt Helper** — builds tile-safe prompts that describe *texture*,
  not composition (scene prompts make every tile hallucinate the subject).
  Content presets: photo, portrait/skin, landscape, architecture, fabric,
  artwork.
- **Tiled Generative Refine** — our own modern replacement for Ultimate SD
  Upscale: every tile is sampled from the *original* upscaled image and merged
  with smooth raised-cosine feathered blending, so there are no seams and no
  separate seam-fix pass, and no progressive tile-to-tile drift. Tiles can be
  sampled in true GPU batches (`batch_size`) for a large speedup, ControlNet
  hints (including chained ControlNets) are cropped per tile automatically,
  and the whole pipeline is pure tensor code — no PIL round trips.
- **Before/After Compare Slider** — our own image comparer: both images in
  one frame with a draggable divider (drag anywhere on the preview). Drawn
  directly on the node canvas, so it works at any zoom and has no widget
  side effects.
- **Color Match GPU** — our own replacement for the deprecated KJNodes
  ColorMatch. Pure PyTorch on GPU, zero pip dependencies. Classic methods
  (mkl, histogram, reinhard-lab, hm-mkl-hm compound) plus two new ones:
  **wavelet** (transfers only low-frequency color, preserving every bit of
  generated detail — the best default after upscaling) and **local**
  (per-region statistics smoothly interpolated across the image, fixes
  tile-local color drift a global transform can't). Optional
  preserve-luminance mode matches chroma only.

## Install

1. Clone this repository into `ComfyUI/custom_nodes/`:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/GeekatplayStudio/ComfyUI-GenUpscaler.git
   ```
2. Run `install.bat` (Windows) or `python install.py` — it offers the model
   downloads below (each confirmed before downloading, skipped if present).
3. Restart ComfyUI and load a workflow from the `workflows/` folder.

**No other custom node packs are required** — the pack runs on ComfyUI core
plus its own built-in nodes.

### Model downloads (also listed in a note inside each workflow)

| File | Folder | Link |
|---|---|---|
| `Flux.1-dev-Controlnet-Upscaler.safetensors` | `models/controlnet/` | https://huggingface.co/f5aiteam/Controlnet/resolve/main/Flux.1-dev-Controlnet-Upscaler.safetensors |
| `4xNomos8kDAT.pth` | `models/upscale_models/` | https://huggingface.co/Maxivi/SDXLModels/resolve/main/4xNomos8kDAT.pth |
| `4xClearRealityV1.pth` | `models/upscale_models/` | https://huggingface.co/Kim2091/ClearRealityV1/resolve/main/4x-ClearRealityV1.pth |
| `controlnet_tile_sdxl_1_0.safetensors` | `models/controlnet/` | https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors |

Also required (usually already present): `flux1-dev-fp8.safetensors` checkpoint
for the Quality workflow; any SDXL checkpoint for the Fast workflow.

## Tips

- **Total upscale factor** = 4x (GAN) x the *Extra Scale* node
  (0.5 → 2x, 1.0 → 4x, 2.0 → 8x).
- Keep the subject hint in the Prompt Helper about **materials** ("red silk,
  oak bark, brushed metal"), never "a woman in a forest".
- Seams visible? Raise overlap % in the Tile Planner — the refiner blends the
  entire overlap zone with a cosine ramp, so more overlap = softer transitions.
- Colors drift? Color Match GPU `wavelet` keeps all generated detail;
  `local` fixes region-by-region drift; `auto (hm-mkl-hm)` is the strongest
  global correction.
- Lots of VRAM? Raise `batch_size` on Tiled Generative Refine to sample
  several tiles per GPU pass.
- Faces in large scenes can degrade in tiled passes — run a FaceDetailer
  (Impact Pack) afterwards if needed.

## Credits

- Tile ControlNets by jasperai & xinsir · GAN models by Philip Hofmann
  (Nomos) & Kim2091 (ClearReality) · tiling approach inspired by Ultimate SD
  Upscale (ssitu) and Mixture of Diffusers; color transfer methods based on
  Reinhard et al. and Pitie et al. (MKL) — reimplemented from scratch in
  PyTorch for this pack.

© Geekatplay Studio / Vladimir Chopine. MIT license.
