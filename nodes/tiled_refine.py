"""GAP Tiled Refine - modern tiled generative refiner (USDU replacement).

Improvements over Ultimate SD Upscale:
  * every tile is sampled from the ORIGINAL upscaled image and merged with
    smooth raised-cosine feathered blending in an accumulation buffer -
    seams are blended continuously, so no separate "seam fix" pass and no
    progressive drift (USDU pastes tiles one after another, so late tiles
    see already-modified neighbors)
  * tiles can be sampled in true GPU batches (batch_size) - big speedup
  * uniform tile grid, always exactly tile_width x tile_height, so every
    tile hits the resolution the model was trained at
  * pure tensor pipeline (no PIL round trips)
  * ControlNet hints are cropped per tile automatically (chained CNs too)
  * works with any model family (SD1.5 / SDXL / FLUX / Qwen)

by Geekatplay Studio / Vladimir Chopine
"""
import math
import torch

try:
    import comfy.samplers
    import comfy.utils
    from nodes import common_ksampler, VAEEncode, VAEDecode
    SAMPLERS = comfy.samplers.KSampler.SAMPLERS
    SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS
except ImportError:  # allows import outside ComfyUI (tests)
    SAMPLERS, SCHEDULERS = ["euler"], ["normal"]


def _tile_positions(size, tile, overlap):
    """1D positions so tiles of `tile` px cover `size` px with >= overlap."""
    if tile >= size:
        return [0]
    step = tile - overlap
    n = math.ceil((size - tile) / step) + 1
    if n == 1:
        return [0]
    # distribute evenly so the last tile ends exactly at the border
    positions = [round(i * (size - tile) / (n - 1)) for i in range(n)]
    return positions


def _feather_mask(tile_w, tile_h, overlap, x0, y0, img_w, img_h, device):
    """Raised-cosine 2D blend mask. Edges that touch the canvas border
    stay at full weight so border pixels are always fully covered."""
    ramp = overlap
    wx = torch.ones(tile_w, device=device)
    wy = torch.ones(tile_h, device=device)
    if ramp > 0:
        r = 0.5 - 0.5 * torch.cos(torch.linspace(0, math.pi, ramp, device=device))
        if x0 > 0:
            wx[:ramp] = torch.minimum(wx[:ramp], r)
        if x0 + tile_w < img_w:
            wx[-ramp:] = torch.minimum(wx[-ramp:], r.flip(0))
        if y0 > 0:
            wy[:ramp] = torch.minimum(wy[:ramp], r)
        if y0 + tile_h < img_h:
            wy[-ramp:] = torch.minimum(wy[-ramp:], r.flip(0))
    return (wy[:, None] * wx[None, :]).clamp_min(1e-4)  # (H, W)


def _crop_controlnet_chain(cond_dict, region, tile_size):
    """Clone the controlnet chain and crop each hint to the tile region."""
    if "control" not in cond_dict:
        return
    x1, y1, x2, y2 = region
    c = cond_dict["control"]
    cloned = c.copy()
    cond_dict["control"] = cloned
    node = cloned
    while node is not None:
        hint = node.cond_hint_original  # (B, C, H, W), same canvas as image
        hh, hw = hint.shape[2], hint.shape[3]
        # scale region if hint resolution differs from the working canvas
        sx, sy = hw / tile_size[2], hh / tile_size[3]
        cx1, cx2 = int(x1 * sx), int(x2 * sx)
        cy1, cy2 = int(y1 * sy), int(y2 * sy)
        cropped = hint[:, :, cy1:cy2, cx1:cx2]
        if cropped.shape[2] != tile_size[1] or cropped.shape[3] != tile_size[0]:
            cropped = torch.nn.functional.interpolate(
                cropped, size=(tile_size[1], tile_size[0]), mode="bilinear",
                align_corners=False)
        node.cond_hint_original = cropped
        prev = node.previous_controlnet
        node.set_previous_controlnet(prev.copy() if prev is not None else None)
        node = node.previous_controlnet


def _crop_cond(cond, region, tile_size, canvas_wh):
    """Crop conditioning extras (controlnet chain) for one tile region."""
    out = []
    for emb, d in cond:
        nd = d.copy()
        _crop_controlnet_chain(nd, region, (tile_size[0], tile_size[1],
                                            canvas_wh[0], canvas_wh[1]))
        out.append([emb, nd])
    return out


class GAPTiledRefine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The already-upscaled image to refine tile by tile."}),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (SAMPLERS,),
                "scheduler": (SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tile_width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "tile_height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "overlap": ("INT", {"default": 128, "min": 32, "max": 512, "step": 8,
                                    "tooltip": "Overlap between tiles in pixels. The full overlap is used as a smooth cosine blend zone."}),
            },
            "optional": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16,
                                       "tooltip": "Tiles sampled per GPU batch. Raise for speed if VRAM allows. Use 1 with ControlNet if you see artifacts."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "refine"
    CATEGORY = "Geekatplay/GenUpscale"
    DESCRIPTION = ("Tiled generative refiner with continuous feathered blending "
                   "(no seam-fix pass needed), batched tile sampling and per-tile "
                   "ControlNet cropping. by Geekatplay Studio / Vladimir Chopine")

    def refine(self, image, model, positive, negative, vae, seed, steps, cfg,
               sampler_name, scheduler, denoise, tile_width, tile_height,
               overlap, batch_size=1):
        if denoise <= 0:
            return (image, "denoise 0 - passthrough")

        b, img_h, img_w, _ = image.shape
        tile_width = min(tile_width, img_w + (-img_w) % 8)
        tile_height = min(tile_height, img_h + (-img_h) % 8)
        overlap = min(overlap, tile_width // 2, tile_height // 2)

        xs = _tile_positions(img_w, tile_width, overlap)
        ys = _tile_positions(img_h, tile_height, overlap)
        regions = [(x, y, min(x + tile_width, img_w), min(y + tile_height, img_h))
                   for y in ys for x in xs]

        encoder, decoder = VAEEncode(), VAEDecode()
        pbar = comfy.utils.ProgressBar(len(regions) * b)
        out_images = []

        for bi in range(b):
            src = image[bi:bi + 1]
            acc = torch.zeros(1, img_h, img_w, 3, dtype=torch.float32)
            weight = torch.zeros(1, img_h, img_w, 1, dtype=torch.float32)

            # group regions into batches
            for start in range(0, len(regions), batch_size):
                chunk = regions[start:start + batch_size]

                tiles = torch.cat([src[:, y1:y2, x1:x2, :] for x1, y1, x2, y2 in chunk], dim=0)
                (latent,) = encoder.encode(vae, tiles)

                if len(chunk) == 1:
                    pos = _crop_cond(positive, chunk[0], (tile_width, tile_height), (img_w, img_h))
                    neg = _crop_cond(negative, chunk[0], (tile_width, tile_height), (img_w, img_h))
                else:
                    # batch mode: concatenate per-tile hints along the batch dim
                    pos = self._crop_cond_batched(positive, chunk, (tile_width, tile_height), (img_w, img_h))
                    neg = self._crop_cond_batched(negative, chunk, (tile_width, tile_height), (img_w, img_h))

                (samples,) = common_ksampler(model, seed + start, steps, cfg,
                                             sampler_name, scheduler, pos, neg,
                                             latent, denoise=denoise)
                (decoded,) = decoder.decode(vae, samples)

                for ti, (x1, y1, x2, y2) in enumerate(chunk):
                    tile_out = decoded[ti:ti + 1, :y2 - y1, :x2 - x1, :].float().cpu()
                    m = _feather_mask(x2 - x1, y2 - y1, overlap, x1, y1,
                                      img_w, img_h, tile_out.device)
                    m = m[None, :, :, None]
                    acc[:, y1:y2, x1:x2, :] += tile_out * m
                    weight[:, y1:y2, x1:x2, :] += m
                    pbar.update(1)

            out_images.append(acc / weight)

        result = torch.cat(out_images, dim=0).clamp(0, 1)
        info = (f"{len(xs)}x{len(ys)} = {len(regions)} tiles of "
                f"{tile_width}x{tile_height}, overlap {overlap}px, "
                f"denoise {denoise}, batch {batch_size}")
        return (result, info)

    def _crop_cond_batched(self, cond, chunk, tile_size, canvas_wh):
        out = []
        for emb, d in cond:
            nd = d.copy()
            if "control" in nd:
                # crop hint for each region and concat on batch dim
                per_tile = []
                for region in chunk:
                    tmp = {"control": nd["control"]}
                    _crop_controlnet_chain(tmp, region, (tile_size[0], tile_size[1],
                                                         canvas_wh[0], canvas_wh[1]))
                    per_tile.append(tmp["control"])
                merged = per_tile[0]
                node_list = []
                n = merged
                while n is not None:
                    node_list.append(n)
                    n = n.previous_controlnet
                for depth, node in enumerate(node_list):
                    hints = []
                    for pt in per_tile:
                        nn = pt
                        for _ in range(depth):
                            nn = nn.previous_controlnet
                        hints.append(nn.cond_hint_original)
                    node.cond_hint_original = torch.cat(hints, dim=0)
                nd["control"] = merged
            out.append([emb, nd])
        return out
