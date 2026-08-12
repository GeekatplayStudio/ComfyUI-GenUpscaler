"""GAP Compare Slider - before/after image comparer with a draggable
split slider, drawn directly on the node canvas (no DOM widgets, so it
is immune to frontend widget-serialization issues).

by Geekatplay Studio / Vladimir Chopine
"""
import random

import numpy as np
from PIL import Image

try:
    import folder_paths
except ImportError:  # allows import outside ComfyUI (tests)
    folder_paths = None


class GAPCompareSlider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "BEFORE image (left side of the slider)."}),
                "image_b": ("IMAGE", {"tooltip": "AFTER image (right side of the slider)."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/GenUpscale"
    DESCRIPTION = ("Before/after comparer with a draggable split slider. "
                   "Drag anywhere on the preview to move the divider. "
                   "by Geekatplay Studio / Vladimir Chopine")

    def _save_temp(self, image, tag):
        if folder_paths is None:
            return []
        out_dir = folder_paths.get_temp_directory()
        prefix = f"gap_compare_{tag}_" + "".join(
            random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        results = []
        full_out, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(prefix, out_dir,
                                             image.shape[2], image.shape[1])
        arr = np.clip(255.0 * image[0].cpu().numpy(), 0, 255).astype(np.uint8)
        file = f"{filename}_{counter:05}_.png"
        Image.fromarray(arr).save(f"{full_out}/{file}", compress_level=1)
        results.append({"filename": file, "subfolder": subfolder, "type": "temp"})
        return results

    def compare(self, image_a, image_b):
        return {"ui": {
            "a_images": self._save_temp(image_a, "a"),
            "b_images": self._save_temp(image_b, "b"),
        }}
