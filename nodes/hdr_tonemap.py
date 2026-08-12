"""GAP HDR Tonemap - modern dynamic range enhancement and state-of-the-art tonemapping.

Provides exposure adjustment, highlight compression, shadow lift, contrast, and
GPU-accelerated tone mapping curves (AgX Filmic, ACES Filmic 1.3, Reinhard, Uncharted 2, Exponential, HDR Pop).

by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com
"""
import torch


def _aces_filmic(x):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return (x * (a * x + b)) / (x * (c * x + d) + e)


def _agx_filmic(x):
    """AgX Filmic Color Transform - modern state-of-the-art highlight roll-off (Blender 4.x standard)."""
    # 1. Inset matrix to prevent hue skewing in high exposure highlights
    # AgX inset matrix
    r = 0.8424790 * x[..., 0] + 0.0784336 * x[..., 1] + 0.0792237 * x[..., 2]
    g = 0.0423282 * x[..., 0] + 0.8784680 * x[..., 1] + 0.0791616 * x[..., 2]
    b = 0.0423756 * x[..., 0] + 0.0784336 * x[..., 1] + 0.8791420 * x[..., 2]
    x_inset = torch.stack([r, g, b], dim=-1).clamp_min(1e-10)

    # 2. Log2 encoding (-10 to +6.5 EV normalization)
    min_ev = -10.0
    max_ev = 6.5
    log_x = (torch.log2(x_inset) - min_ev) / (max_ev - min_ev)
    log_x = log_x.clamp(0.0, 1.0)

    # 3. Sigmoidal S-curve contrast mapping
    s_curve = 0.5 - 0.5 * torch.cos(torch.pi * (log_x ** 1.35))
    return s_curve


def _uncharted2_curve(x):
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F


def _uncharted2(x):
    W = 11.2
    return _uncharted2_curve(x * 2.0) / _uncharted2_curve(torch.tensor(W, dtype=x.dtype, device=x.device))


class GAPHDRTonemap:
    TONE_OPERATORS = ["agx_filmic", "hdr_pop", "aces_filmic", "reinhard", "uncharted2", "exponential", "none"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to enhance / tonemap."}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.1,
                                       "tooltip": "EV exposure adjustment (-4 to +4 EV)."}),
                "highlights": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05,
                                         "tooltip": "Highlight compression / recovery factor."}),
                "shadows": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05,
                                      "tooltip": "Shadow lift factor."}),
                "contrast": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 2.0, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.05}),
                "tonemap_operator": (cls.TONE_OPERATORS, {"default": "agx_filmic"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "process"
    CATEGORY = "Geekatplay/GenUpscale"
    DESCRIPTION = ("HDR dynamic range enhancement & state-of-the-art AgX Filmic tonemapper. "
                   "Prevents highlight burning & hue skewing on GPU. "
                   "by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com")

    def process(self, image, exposure, highlights, shadows, contrast, saturation, tonemap_operator):
        x = image.clone().to(torch.float32)

        # 1. Apply EV Exposure adjustment
        if exposure != 0.0:
            x = x * (2.0 ** exposure)

        # 2. Shadows and Highlights adjustment
        if shadows != 1.0 or highlights != 1.0:
            lum = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
            lum = lum.unsqueeze(-1)
            shadow_mask = (1.0 - lum).clamp(0, 1) ** 2.0
            highlight_mask = lum.clamp(0, 1) ** 2.0
            x = x * (1.0 + shadow_mask * (shadows - 1.0) + highlight_mask * (highlights - 1.0))

        # 3. Tone Mapping Curve
        if tonemap_operator == "agx_filmic":
            x = _agx_filmic(x)
        elif tonemap_operator == "aces_filmic":
            x = _aces_filmic(x)
        elif tonemap_operator == "reinhard":
            x = x / (1.0 + x)
        elif tonemap_operator == "uncharted2":
            x = _uncharted2(x)
        elif tonemap_operator == "exponential":
            x = 1.0 - torch.exp(-x)
        elif tonemap_operator == "hdr_pop":
            base = x / (1.0 + x * 0.5)
            x = _aces_filmic(base * 1.1)

        # 4. Contrast
        if contrast != 1.0:
            x = (x - 0.5) * contrast + 0.5

        # 5. Saturation
        if saturation != 1.0:
            gray = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
            gray = gray.unsqueeze(-1)
            x = gray + (x - gray) * saturation

        result = x.clamp(0.0, 1.0).to(image.dtype)
        info = (f"HDR Tonemap ({tonemap_operator}): exp={exposure:+.1f}EV, "
                f"contrast={contrast:.2f}, sat={saturation:.2f}")
        return (result, info)
