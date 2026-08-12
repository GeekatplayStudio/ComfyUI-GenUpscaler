"""GAP Detail Daemon - Frequency-based Latent Detail Booster for FLUX, SDXL & Modern Diffusion.

Enhances high-frequency micro-texture and fine details during diffusion sampling
by applying frequency-decoupled latent guidance adjustments.

by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com
"""
import torch


class GAPDetailDaemon:
    """Frequency-based Latent Detail Booster for FLUX, SDXL & Qwen.

    Decouples low-frequency structural guidance from high-frequency micro-textures,
    boosting hair, skin pores, fabric weave, and landscape grit without creating harsh contrast artifacts.
    """

    MODES = ["balanced detail", "micro-texture boost", "archival sharp", "extreme detail (creative)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Diffusion model to patch with detail boosting."}),
                "detail_amount": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                                            "tooltip": "Strength of high-frequency detail injection (0 = off, 1 = maximum)."}),
                "frequency_cutoff": ("FLOAT", {"default": 0.35, "min": 0.1, "max": 0.8, "step": 0.05,
                                               "tooltip": "Frequency threshold separating base structure from detail frequencies."}),
                "mode": (cls.MODES, {"default": "balanced detail"}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "apply"
    CATEGORY = "Geekatplay/GenUpscale"
    DESCRIPTION = ("Frequency-decoupled Detail Booster. Injects high-frequency micro-textures "
                   "(skin pores, fabric weave, foliage) into diffusion samplers. "
                   "by Geekatplay Studio / Vladimir Chopine")

    def apply(self, model, detail_amount, frequency_cutoff, mode):
        if detail_amount <= 0:
            return (model, "Detail Daemon: disabled (0.0)")

        m = model.clone()

        # Scale multiplier based on selected mode
        mult = 1.0
        if mode == "micro-texture boost":
            mult = 1.35
        elif mode == "archival sharp":
            mult = 0.75
        elif mode == "extreme detail (creative)":
            mult = 1.75

        effective_amount = detail_amount * mult

        def detail_post_cfg(args):
            denoised = args["denoised"]
            cond = args.get("cond", denoised)

            # High-pass frequency extraction via Gaussian blur in spatial/latent space
            if denoised.ndim == 4:
                # Spatial dimensions: B, C, H, W
                kernel_size = 5
                sigma = max(0.5, (1.0 - frequency_cutoff) * 3.0)
                grid = torch.arange(kernel_size, device=denoised.device, dtype=denoised.dtype) - (kernel_size - 1) / 2
                g1d = torch.exp(-0.5 * (grid / sigma) ** 2)
                g1d = g1d / g1d.sum()
                g2d = (g1d[:, None] * g1d[None, :]).view(1, 1, kernel_size, kernel_size)

                # Low frequency base component
                b, c, h, w = denoised.shape
                low_freq = torch.nn.functional.conv2d(
                    denoised.view(b * c, 1, h, w), g2d, padding=kernel_size // 2
                ).view(b, c, h, w)

                # High frequency detail component
                high_freq = denoised - low_freq
                return denoised + high_freq * effective_amount

            return denoised

        m.set_model_sampler_post_cfg_function(detail_post_cfg)
        info = f"Detail Daemon active: amount={effective_amount:.2f}, cutoff={frequency_cutoff:.2f} ({mode})"
        return (m, info)
