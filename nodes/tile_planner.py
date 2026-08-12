"""Auto Tile Planner - computes optimal tile geometry for a tiled
generative upscale from the *final* image size, replacing the fragile
GetImageSize -> Multiply -> FloatToInt chains of older workflows.

by Geekatplay Studio / Vladimir Chopine
"""
import math


class GAPTilePlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_tile": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64,
                                        "tooltip": "Ideal tile edge in pixels. 1024 for SDXL/FLUX, 768 for SD1.5, up to 1536 for FLUX with lots of VRAM."}),
                "overlap_pct": ("FLOAT", {"default": 12.5, "min": 4.0, "max": 33.0, "step": 0.5,
                                          "tooltip": "Tile overlap as percent of tile size. More overlap = fewer seams, slower."}),
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

        # pick a tile count per axis so that tiles land close to target size,
        # then size tiles to divide the image evenly (fewer partial tiles)
        nx = max(1, round(w / target_tile))
        ny = max(1, round(h / target_tile))
        tile_w = self._snap(w / nx)
        tile_h = self._snap(h / ny)

        # clamp to sane diffusion sizes
        tile_w = min(tile_w, 2048)
        tile_h = min(tile_h, 2048)

        padding = self._snap(min(tile_w, tile_h) * overlap_pct / 100.0, 8)
        mask_blur = max(8, min(64, padding // 2))

        tiles_x = math.ceil(w / tile_w)
        tiles_y = math.ceil(h / tile_h)
        total = tiles_x * tiles_y

        info = (f"image {w}x{h} -> {tiles_x}x{tiles_y} = {total} tiles of "
                f"{tile_w}x{tile_h}, padding {padding}px, mask blur {mask_blur}px")

        return (tile_w, tile_h, padding, mask_blur, total, info)
