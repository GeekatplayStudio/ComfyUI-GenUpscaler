"""GAP Color Match - pure-PyTorch GPU color transfer, no external deps.

Improvements over the (deprecated) KJNodes ColorMatch:
  * runs on GPU in torch (no color-matcher pip package, no numpy round trip)
  * 'wavelet' method: transfers only low-frequency color, keeping every bit
    of generated high-frequency detail - ideal after tiled upscaling
  * 'local' spatial matching: per-region statistics smoothly interpolated
    across the image, corrects tile-local color drift that a single global
    transform cannot fix
  * preserve_luminance option (match chroma only)

by Geekatplay Studio / Vladimir Chopine
"""
import torch
import torch.nn.functional as F

try:
    import comfy.model_management as mm
except ImportError:  # allows import outside ComfyUI (tests)
    mm = None


# ---------- color space ----------

def _srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


_RGB2XYZ = [[0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]]
_D65 = (0.95047, 1.0, 1.08883)


def rgb_to_lab(rgb):
    """rgb: (..., 3) in 0..1 -> Lab (L 0..100)"""
    m = torch.tensor(_RGB2XYZ, dtype=rgb.dtype, device=rgb.device)
    xyz = _srgb_to_linear(rgb) @ m.T
    xyz = xyz / torch.tensor(_D65, dtype=rgb.dtype, device=rgb.device)
    eps, kappa = 216 / 24389, 24389 / 27
    f = torch.where(xyz > eps, xyz ** (1 / 3), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


def lab_to_rgb(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    eps, kappa = 216 / 24389, 24389 / 27
    def finv(f):
        f3 = f ** 3
        return torch.where(f3 > eps, f3, (116 * f - 16) / kappa)
    xyz = torch.stack([finv(fx), finv(fy), finv(fz)], dim=-1)
    xyz = xyz * torch.tensor(_D65, dtype=lab.dtype, device=lab.device)
    m = torch.tensor(_RGB2XYZ, dtype=lab.dtype, device=lab.device)
    rgb = xyz @ torch.linalg.inv(m).T
    return _linear_to_srgb(rgb)


# ---------- transfer methods (operate on (N, 3) flat pixels) ----------

def _stats_transfer(t, r):
    """Reinhard: match mean/std per channel."""
    t_m, t_s = t.mean(0), t.std(0).clamp_min(1e-6)
    r_m, r_s = r.mean(0), r.std(0).clamp_min(1e-6)
    return (t - t_m) / t_s * r_s + r_m


def _mkl_transfer(t, r):
    """Monge-Kantorovich linear transfer (Pitie et al.)."""
    t64, r64 = t.double(), r.double()
    mu_t, mu_r = t64.mean(0), r64.mean(0)
    ct = torch.cov(t64.T) + torch.eye(3, dtype=torch.float64, device=t.device) * 1e-8
    cr = torch.cov(r64.T) + torch.eye(3, dtype=torch.float64, device=t.device) * 1e-8

    def sqrtm(m):
        vals, vecs = torch.linalg.eigh(m)
        return vecs @ torch.diag(vals.clamp_min(1e-12).sqrt()) @ vecs.T

    ct_s = sqrtm(ct)
    ct_si = torch.linalg.inv(ct_s)
    T = ct_si @ sqrtm(ct_s @ cr @ ct_s) @ ct_si
    return ((t64 - mu_t) @ T.T + mu_r).to(t.dtype)


def _hist_transfer(t, r):
    """Exact per-channel histogram matching via sorting."""
    out = torch.empty_like(t)
    n_t, n_r = t.shape[0], r.shape[0]
    pos = torch.linspace(0, n_r - 1, n_t, device=t.device)
    lo, frac = pos.floor().long(), pos - pos.floor()
    for ch in range(t.shape[1]):
        r_sorted, _ = torch.sort(r[:, ch])
        vals = r_sorted[lo] * (1 - frac) + r_sorted[(lo + 1).clamp_max(n_r - 1)] * frac
        idx = torch.argsort(t[:, ch])
        out[idx, ch] = vals
    return out


# ---------- spatial methods (operate on (1, H, W, 3) images) ----------

def _lowpass(img_bhwc, factor=16):
    """Cheap smooth low-pass: downscale + upscale."""
    x = img_bhwc.movedim(-1, 1)
    h, w = x.shape[2], x.shape[3]
    small = F.interpolate(x, size=(max(1, h // factor), max(1, w // factor)),
                          mode="area")
    return F.interpolate(small, size=(h, w), mode="bilinear",
                         align_corners=False).movedim(1, -1)


def _wavelet_transfer(t_img, r_img):
    """Keep target high frequencies, take reference low-frequency color."""
    return t_img - _lowpass(t_img) + _lowpass(r_img)


def _local_transfer(t_img, r_img, grid):
    """Per-region Reinhard in Lab with smoothly interpolated statistics."""
    t_lab = rgb_to_lab(t_img).movedim(-1, 1)   # (1,3,H,W)
    r_lab = rgb_to_lab(r_img).movedim(-1, 1)
    h, w = t_lab.shape[2], t_lab.shape[3]

    def stat_maps(x):
        mean = F.adaptive_avg_pool2d(x, (grid, grid))
        sq = F.adaptive_avg_pool2d(x * x, (grid, grid))
        std = (sq - mean * mean).clamp_min(1e-6).sqrt()
        up = lambda m: F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
        return up(mean), up(std)

    t_m, t_s = stat_maps(t_lab)
    r_m, r_s = stat_maps(r_lab)
    out_lab = (t_lab - t_m) / t_s * r_s + r_m
    return lab_to_rgb(out_lab.movedim(1, -1))


class GAPColorMatch:
    METHODS = ["auto (hm-mkl-hm)", "wavelet (keep detail)", "local (per-region)",
               "mkl", "histogram", "reinhard-lab"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE", {"tooltip": "Original image whose colors to restore."}),
                "image_target": ("IMAGE", {"tooltip": "Upscaled/generated image to correct."}),
                "method": (cls.METHODS, {"default": "auto (hm-mkl-hm)"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "preserve_luminance": ("BOOLEAN", {"default": False,
                                                   "tooltip": "Match chroma only; keep the target's own brightness/contrast."}),
                "local_regions": ("INT", {"default": 8, "min": 2, "max": 32,
                                          "tooltip": "Grid size for the 'local' method."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "match"
    CATEGORY = "Geekatplay/GenUpscale"
    DESCRIPTION = ("GPU color transfer with detail-preserving 'wavelet' and "
                   "spatially-aware 'local' modes. Replacement for the deprecated "
                   "KJNodes ColorMatch, no external dependencies. "
                   "by Geekatplay Studio / Vladimir Chopine")

    def match(self, image_ref, image_target, method, strength=1.0,
              preserve_luminance=False, local_regions=8):
        if strength == 0:
            return (image_target,)

        device = mm.get_torch_device() if mm else image_target.device
        out_batch = []
        for i in range(image_target.shape[0]):
            tgt = image_target[i:i + 1].to(device, torch.float32)
            ref_i = min(i, image_ref.shape[0] - 1)
            ref = image_ref[ref_i:ref_i + 1].to(device, torch.float32)

            # bring ref to target size for spatial methods
            if ref.shape[1:3] != tgt.shape[1:3]:
                ref_rs = F.interpolate(ref.movedim(-1, 1), size=tgt.shape[1:3],
                                       mode="bilinear", align_corners=False).movedim(1, -1)
            else:
                ref_rs = ref

            if method.startswith("wavelet"):
                res = _wavelet_transfer(tgt, ref_rs)
            elif method.startswith("local"):
                res = _local_transfer(tgt, ref_rs, local_regions)
            else:
                t_flat = tgt.reshape(-1, 3)
                r_flat = ref.reshape(-1, 3)
                if method == "mkl":
                    res = _mkl_transfer(t_flat, r_flat)
                elif method == "histogram":
                    res = _hist_transfer(t_flat, r_flat)
                elif method == "reinhard-lab":
                    res = lab_to_rgb(_stats_transfer(
                        rgb_to_lab(t_flat), rgb_to_lab(r_flat)))
                else:  # auto: hm -> mkl -> hm (best-scoring compound)
                    res = _hist_transfer(t_flat, r_flat)
                    res = _mkl_transfer(res, r_flat)
                    res = _hist_transfer(res, r_flat)
                res = res.reshape(tgt.shape)

            if preserve_luminance:
                res_lab = rgb_to_lab(res.clamp(0, 1))
                tgt_lab = rgb_to_lab(tgt)
                res_lab[..., 0] = tgt_lab[..., 0]
                res = lab_to_rgb(res_lab)

            if strength != 1.0:
                res = tgt + strength * (res - tgt)

            out_batch.append(res.clamp(0, 1).cpu())

        return (torch.cat(out_batch, dim=0).float(),)
