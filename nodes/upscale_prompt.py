"""Upscale Prompt Helper - builds tile-safe positive/negative prompts.

For tiled upscaling the prompt must describe *texture qualities*, never
scene composition (each tile only sees a crop, so 'a woman on a swing'
makes every tile try to draw a woman). This node combines a curated
texture-detail base prompt with an optional short subject hint.

by Geekatplay Studio / Vladimir Chopine
"""

BASE_POSITIVE = (
    "masterpiece photograph, extremely detailed, sharp focus, "
    "intricate natural micro-texture, fine fabric weave, realistic skin pores, "
    "crisp material definition, high dynamic range, professional photography, "
    "8k uhd, film grain"
)

BASE_NEGATIVE = (
    "blurry, out of focus, jpeg artifacts, compression artifacts, "
    "smooth plastic skin, watercolor, painting, drawing, cartoon, "
    "oversaturated, deformed, mutated, duplicated features, "
    "text, watermark, signature, tiling seams, grid pattern"
)

STYLE_HINTS = {
    "photo / realistic": "",
    "portrait / skin": "detailed skin texture, individual hairs, natural skin tone, subsurface scattering",
    "landscape / nature": "detailed foliage, individual leaves and blades of grass, rock texture, natural lighting",
    "architecture / hard surface": "clean edges, detailed surface materials, brick and concrete texture, precise geometry",
    "fabric / product": "detailed textile weave, stitching, material fibers, studio product lighting",
    "artwork / painting": "detailed brush strokes, canvas texture, rich pigment, gallery quality reproduction",
}


class GAPUpscalePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_type": (list(STYLE_HINTS.keys()), {"default": "photo / realistic"}),
                "subject_hint": ("STRING", {"default": "", "multiline": True,
                                            "tooltip": "OPTIONAL short description of the overall image (e.g. 'red silk dress, forest'). Keep it about materials, not composition."}),
                "extra_negative": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "build"
    CATEGORY = "Geekatplay/GenUpscale"

    def build(self, content_type, subject_hint, extra_negative):
        parts = [BASE_POSITIVE]
        hint = STYLE_HINTS.get(content_type, "")
        if hint:
            parts.append(hint)
        if subject_hint.strip():
            parts.append(subject_hint.strip())
        positive = ", ".join(parts)

        negative = BASE_NEGATIVE
        if extra_negative.strip():
            negative = negative + ", " + extra_negative.strip()

        return (positive, negative)
