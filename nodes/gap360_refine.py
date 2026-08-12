"""GAP 360 Tiled Refine & Planner - seamless 360 equirectangular upscaling.

Designed specifically for 360° panoramas:
  * Employs horizontal circular padding and boundary folding so tiles crossing
    the 360° seam (x=0 <-> x=W) sample continuously across the boundary with zero seam artifacts.
  * Preserves equirectangular geometry without destroying horizontal perspective stretching.
  * Seamless raised-cosine accumulation buffer with GPU batch sampling.

by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com
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
    """1D positions covering `size` with `tile` px and >= `overlap` px."""
    if tile >= size:
        return [0]
    step = tile - overlap
    n = math.ceil((size - tile) / step) + 1
    if n == 1:
        return [0]
    positions = [round(i * (size - tile) / (n - 1)) for i in range(n)]
    return positions


def _feather_mask(tile_w, tile_h, overlap, x0, y0, img_w, img_h, device, is_360=False):
    """Raised-cosine 2D blend mask. For 360 images, horizontal edges wrap,
    so left/right edges are always feathered unless at image boundaries."""
    ramp = overlap
    wx = torch.ones(tile_w, device=device)
    wy = torch.ones(tile_h, device=device)
    if ramp > 0:
        r = 0.5 - 0.5 * torch.cos(torch.linspace(0, math.pi, ramp, device=device))
        if is_360 or x0 > 0:
            wx[:ramp] = torch.minimum(wx[:ramp], r)
        if is_360 or (x0 + tile_w < img_w):
            wx[-ramp:] = torch.minimum(wx[-ramp:], r.flip(0))
        if y0 > 0:
            wy[:ramp] = torch.minimum(wy[:ramp], r)
        if y0 + tile_h < img_h:
            wy[-ramp:] = torch.minimum(wy[-ramp:], r.flip(0))
    return (wy[:, None] * wx[None, :]).clamp_min(1e-4)


def _crop_controlnet_chain(cond_dict, region, tile_size):
    if "control" not in cond_dict:
        return
    x1, y1, x2, y2 = region
    c = cond_dict["control"]
    cloned = c.copy()
    cond_dict["control"] = cloned
    node = cloned
    while node is not None:
        hint = node.cond_hint_original
        hh, hw = hint.shape[2], hint.shape[3]
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
    out = []
    for emb, d in cond:
        nd = d.copy()
        _crop_controlnet_chain(nd, region, (tile_size[0], tile_size[1],
                                            canvas_wh[0], canvas_wh[1]))
        out.append([emb, nd])
    return out


class GAP360TilePlanner:
    """Auto Tile Planner tailored for 360 equirectangular panoramas (2:1 aspect ratio)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_tile": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "overlap_pct": ("FLOAT", {"default": 15.0, "min": 8.0, "max": 33.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("tile_width", "tile_height", "padding", "mask_blur", "grid_tiles", "info")
    FUNCTION = "plan"
    CATEGORY = "Geekatplay/GenUpscale"

    @staticmethod
    def _snap(v, step=64):
        return max(step, int(round(v / step)) * step)

    def plan(self, image, target_tile, overlap_pct):
        _, h, w, _ = image.shape
        nx = max(1, round(w / target_tile))
        ny = max(1, round(h / target_tile))
        tile_w = self._snap(w / nx)
        tile_h = self._snap(h / ny)

        tile_w = min(tile_w, 2048)
        tile_h = min(tile_h, 2048)

        padding = self._snap(min(tile_w, tile_h) * overlap_pct / 100.0, 8)
        mask_blur = max(8, min(64, padding // 2))

        tiles_x = math.ceil(w / tile_w)
        tiles_y = math.ceil(h / tile_h)
        total = tiles_x * tiles_y

        info = (f"360 Panorama {w}x{h} -> {tiles_x}x{tiles_y} = {total} tiles of "
                f"{tile_w}x{tile_h}, padding {padding}px (seamless 360 wrap enabled)")

        return (tile_w, tile_h, padding, mask_blur, total, info)


class GAP360TiledRefine:
    """360° Equirectangular Tiled Generative Refiner with seamless circular wrapping."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The already-upscaled 360 equirectangular image to refine."}),
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
                "overlap": ("INT", {"default": 128, "min": 32, "max": 512, "step": 8}),
            },
            "optional": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "seamless_wrap": ("BOOLEAN", {"default": True, "tooltip": "Wrap left/right edges circularly to eliminate 360° seam."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "refine"
    CATEGORY = "Geekatplay/GenUpscale"
    DESCRIPTION = ("360 Equirectangular Tiled Generative Refiner. "
                   "Uses circular horizontal padding to guarantee a perfectly "
                   "seamless 360° wrap boundary. by Geekatplay Studio / Vladimir Chopine")

    def refine(self, image, model, positive, negative, vae, seed, steps, cfg,
               sampler_name, scheduler, denoise, tile_width, tile_height,
               overlap, batch_size=1, seamless_wrap=True):
        if denoise <= 0:
            return (image, "denoise 0 - passthrough")

        b, img_h, img_w, _ = image.shape
        tile_width = min(tile_width, img_w + (-img_w) % 8)
        tile_height = min(tile_height, img_h + (-img_h) % 8)
        overlap = min(overlap, tile_width // 2, tile_height // 2)

        # Pad image horizontally if seamless_wrap is enabled
        pad = overlap if seamless_wrap else 0
        if pad > 0:
            left_pad = image[:, :, -pad:, :]
            right_pad = image[:, :, :pad, :]
            padded_image = torch.cat([left_pad, image, right_pad], dim=2)
        else:
            padded_image = image

        pw_h, pw_w = padded_image.shape[1], padded_image.shape[2]

        xs = _tile_positions(pw_w, tile_width, overlap)
        ys = _tile_positions(pw_h, tile_height, overlap)
        regions = [(x, y, min(x + tile_width, pw_w), min(y + tile_height, pw_h))
                   for y in ys for x in xs]

        encoder, decoder = VAEEncode(), VAEDecode()
        pbar = comfy.utils.ProgressBar(len(regions) * b)
        out_images = []

        for bi in range(b):
            src = padded_image[bi:bi + 1]
            acc = torch.zeros(1, pw_h, pw_w, 3, dtype=torch.float32)
            weight = torch.zeros(1, pw_h, pw_w, 1, dtype=torch.float32)

            for start in range(0, len(regions), batch_size):
                chunk = regions[start:start + batch_size]
                tiles = torch.cat([src[:, y1:y2, x1:x2, :] for x1, y1, x2, y2 in chunk], dim=0)
                (latent,) = encoder.encode(vae, tiles)

                if len(chunk) == 1:
                    pos = _crop_cond(positive, chunk[0], (tile_width, tile_height), (pw_w, pw_h))
                    neg = _crop_cond(negative, chunk[0], (tile_width, tile_height), (pw_w, pw_h))
                else:
                    pos = self._crop_cond_batched(positive, chunk, (tile_width, tile_height), (pw_w, pw_h))
                    neg = self._crop_cond_batched(negative, chunk, (tile_width, tile_height), (pw_w, pw_h))

                (samples,) = common_ksampler(model, seed + start, steps, cfg,
                                             sampler_name, scheduler, pos, neg,
                                             latent, denoise=denoise)
                (decoded,) = decoder.decode(vae, samples)

                for ti, (x1, y1, x2, y2) in enumerate(chunk):
                    tile_out = decoded[ti:ti + 1, :y2 - y1, :x2 - x1, :].float().cpu()
                    m = _feather_mask(x2 - x1, y2 - y1, overlap, x1, y1,
                                      pw_w, pw_h, tile_out.device, is_360=seamless_wrap)
                    m = m[None, :, :, None]
                    acc[:, y1:y2, x1:x2, :] += tile_out * m
                    weight[:, y1:y2, x1:x2, :] += m
                    pbar.update(1)

            # Fold horizontal circular padding back into main canvas if seamless
            if pad > 0:
                # Add left pad area back into right of main image
                acc[:, :, pad:pad + pad, :] += acc[:, :, pw_w - pad:, :]
                weight[:, :, pad:pad + pad, :] += weight[:, :, pw_w - pad:, :]
                # Add right pad area back into left of main image
                acc[:, :, pw_w - 2 * pad:pw_w - pad, :] += acc[:, :, :pad, :]
                weight[:, :, pw_w - 2 * pad:pw_w - pad, :] += weight[:, :, :pad, :]
                # Crop to original image bounds
                acc = acc[:, :, pad:pad + img_w, :]
                weight = weight[:, :, pad:pad + img_w, :]

            out_images.append(acc / weight)

        result = torch.cat(out_images, dim=0).clamp(0, 1)
        info = (f"360 Refine: {len(xs)}x{len(ys)} = {len(regions)} tiles of "
                f"{tile_width}x{tile_height}, overlap {overlap}px, "
                f"denoise {denoise}, seamless_360={seamless_wrap}")
        return (result, info)

    def _crop_cond_batched(self, cond, chunk, tile_size, canvas_wh):
        out = []
        for emb, d in cond:
            nd = d.copy()
            if "control" in nd:
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
