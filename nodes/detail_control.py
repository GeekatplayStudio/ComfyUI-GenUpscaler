"""Detail Control node - maps a single 'creativity' knob to all the
sampler/controlnet parameters that govern faithful-vs-creative behavior
in a tiled generative upscale.

by Geekatplay Studio / Vladimir Chopine
"""


class GAPDetailControl:
    """One knob to rule them all.

    creativity 0.0  = archival: reproduce the source almost exactly,
                      only sharpen and clean (denoise ~0.10, tile CN ~0.95)
    creativity 0.5  = balanced: regenerate lost micro-texture (fabric weave,
                      skin pores, foliage) while locking structure
    creativity 1.0  = creative: reinterpret each tile with strong generative
                      detail (denoise ~0.55, tile CN ~0.55)
    """

    PRESETS = {
        "archival (exact match)": 0.10,
        "faithful": 0.25,
        "balanced": 0.45,
        "detailed": 0.65,
        "creative": 0.85,
        "custom (use slider)": -1.0,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls.PRESETS.keys()), {"default": "balanced"}),
                "creativity": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Used when preset is 'custom'. 0 = keep original, 1 = maximum generated detail."}),
                "model_family": (["flux", "sdxl", "qwen"], {"default": "flux"}),
                "upscale_factor": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 16.0, "step": 0.5,
                                             "tooltip": "Total upscale factor. Higher factors automatically allow a bit more denoise because each tile sees less of the original."}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("denoise", "cn_strength", "detail_daemon", "steps", "cfg", "cn_end_percent", "info")
    FUNCTION = "compute"
    CATEGORY = "Geekatplay/GenUpscale"

    def compute(self, preset, creativity, model_family, upscale_factor):
        c = self.PRESETS.get(preset, -1.0)
        if c < 0:
            c = creativity
        c = max(0.0, min(1.0, c))

        # more upscale -> tiles contain less source information -> allow
        # slightly more denoise so the model can invent the missing texture
        factor_boost = min(0.08, max(0.0, (upscale_factor - 2.0) * 0.015))

        denoise = 0.15 + c * 0.40 + factor_boost           # 0.15 .. 0.63
        cn_end = 0.85 - c * 0.15                           # release CN early when creative
        detail_daemon = 0.05 + c * 0.25                    # extra micro detail

        if model_family == "flux":
            # jasperai Flux upscaler CN produces artifacts above ~0.7 strength
            cn_strength = 0.65 - c * 0.25                  # 0.65 .. 0.40
            steps = 20 + int(c * 8)
            cfg = 1.0          # flux-dev uses guidance embedding, keep CFG 1
            detail_daemon = 0.10 + c * 0.25
        elif model_family == "sdxl":
            # xinsir tile CN is happy at high strength
            cn_strength = 1.0 - c * 0.45                   # 1.0 .. 0.55
            steps = 18 + int(c * 12)
            cfg = 4.0 + c * 3.0
            detail_daemon = min(detail_daemon, 0.25)
        else:  # qwen
            cn_strength = 0.90 - c * 0.35
            steps = 16 + int(c * 8)
            cfg = 2.5 + c * 1.5
            detail_daemon = min(detail_daemon, 0.30)

        info = (f"creativity={c:.2f} | denoise={denoise:.2f} | "
                f"tile CN strength={cn_strength:.2f} end={cn_end:.2f} | "
                f"detail daemon={detail_daemon:.2f} | steps={steps} cfg={cfg:.1f}")

        return (round(denoise, 3), round(cn_strength, 3), round(detail_daemon, 3),
                steps, round(cfg, 2), round(cn_end, 3), info)
