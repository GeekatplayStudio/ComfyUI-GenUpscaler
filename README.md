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
| `GAP-GenUpscale-Fast-SDXL.json` | SDXL + xinsir Tile ControlNet | 8–12 GB | Fast batches; swap in a photoreal checkpoint for best results |
| `GAP-GenUpscale-360-HDR-FLUX.json` | FLUX.1-dev fp8 + 360 Circular Refine | 16–24 GB | **360° equirectangular panoramas** with zero seam artifacts, HDR tonemapping, Depth & Normal maps, and interactive 360 compare viewer |
| `GAP-GenUpscale-Quality-FLUX-DepthNormal.json` | FLUX.1-dev fp8 + Depth/Normal Generator | 16–24 GB | High-quality upscale output with additional 3D depth and surface normal map channels |
| `GAP-GenUpscale-Fast-SDXL-DepthNormal.json` | SDXL + Depth/Normal Generator | 8–12 GB | Fast SDXL upscale output with additional 3D depth and surface normal map channels |

## Custom nodes

- **Detail Control (Creative vs Original)** — the main knob. Presets
  `archival → faithful → balanced → detailed → creative` (or a 0–1 slider)
  are mapped to research-tuned denoise, tile-ControlNet strength/end-percent,
  steps, CFG and Detail-Daemon amount for FLUX, SDXL or Qwen.
  - *archival* ≈ denoise 0.15, CN strong: pixel-faithful cleanup
  - *balanced* ≈ denoise 0.33: regenerates lost micro-texture on fabric/skin
  - *creative* ≈ denoise 0.5+, CN relaxed: bold reinterpretation of detail
- **Auto Tile Planner & 360 Auto Tile Planner** — computes an even tile grid (width/height/padding/
  mask-blur) from the final image size. 360 version includes circular boundary padding.
- **Upscale Prompt Helper** — builds tile-safe prompts that describe *texture*,
  not composition (scene prompts make every tile hallucinate the subject).
- **Tiled Generative Refine & 360 Tiled Generative Refine** — modern replacement for Ultimate SD
  Upscale: every tile is sampled from the *original* upscaled image and merged with smooth raised-cosine
  feathered blending. **360 version** incorporates horizontal circular padding ($x=0 \leftrightarrow x=W$)
  to guarantee zero seam artifacts at the 360° boundary line.
- **HDR Tonemap & Contrast** — GPU-accelerated dynamic range enhancement, exposure correction,
  highlight compression, shadow lift, and tone mapping curves (Reinhard, ACES Filmic, Uncharted 2, Exponential, HDR Pop).
- **Depth & Surface Normal Generator** — extracts high-precision 3D depth maps and tangent-space surface
  normal maps ($R=N_x, G=N_y, B=N_z$) directly on GPU via Sobel/Scharr operators.
- **Before/After Compare Slider & 360 Interactive Compare Viewer** — canvas-rendered image comparers.
  **360 Compare Viewer** allows interactive click-and-drag camera panning/tilting (yaw & pitch) and FOV zoom inside
  the 360° equirectangular panorama while dragging a real-time side-by-side split comparison line.
- **Color Match GPU** — pure PyTorch GPU color transfer (wavelet, local per-region statistics, mkl, histogram).

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
