"""GAP Depth, Surface Normal & Curvature Generator - fast GPU 3D detail extraction.

Generates high-precision Depth Maps, Tangent-Space Surface Normal Maps (RGB),
and Surface Curvature Maps (Laplacian) using GPU multi-scale gradient operators (Sobel/Scharr).

by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com
"""
import torch
import torch.nn.functional as F


def _scharr_gradients(img_mono):
    """img_mono: (B, 1, H, W) -> gx, gy (Scharr operator has higher rotational symmetry than standard Sobel)."""
    scharr_x = torch.tensor([[-3., 0., 3.],
                             [-10., 0., 10.],
                             [-3., 0., 3.]], device=img_mono.device).view(1, 1, 3, 3) / 32.0
    scharr_y = torch.tensor([[-3., -10., -3.],
                             [ 0.,   0.,   0.],
                             [ 3.,  10.,   3.]], device=img_mono.device).view(1, 1, 3, 3) / 32.0

    gx = F.conv2d(img_mono, scharr_x, padding=1)
    gy = F.conv2d(img_mono, scharr_y, padding=1)
    return gx, gy


def _laplacian_curvature(img_mono):
    """Computes Surface Curvature map via 3x3 Laplacian filter."""
    laplacian_k = torch.tensor([[0.,  1., 0.],
                                [1., -4., 1.],
                                [0.,  1., 0.]], device=img_mono.device).view(1, 1, 3, 3)
    return F.conv2d(img_mono, laplacian_k, padding=1)


def _gaussian_blur(img_mono, kernel_size=5, sigma=1.0):
    if kernel_size <= 1:
        return img_mono
    k = kernel_size
    grid = torch.arange(k, device=img_mono.device, dtype=torch.float32) - (k - 1) / 2
    g1d = torch.exp(-0.5 * (grid / sigma) ** 2)
    g1d = g1d / g1d.sum()
    g2d = (g1d[:, None] * g1d[None, :]).view(1, 1, k, k)
    pad = k // 2
    return F.conv2d(img_mono, g2d, padding=pad)


class GAPDepthNormalGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to generate depth, normal, and curvature maps from."}),
                "normal_strength": ("FLOAT", {"default": 2.5, "min": 0.1, "max": 10.0, "step": 0.1,
                                              "tooltip": "Strength/scale of the surface normal detail."}),
                "depth_smoothness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.2,
                                               "tooltip": "Smoothness of the depth estimation map."}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "invert_y_normal": ("BOOLEAN", {"default": False, "tooltip": "Invert Green channel (DirectX vs OpenGL normal convention)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("depth", "normal", "curvature", "info")
    FUNCTION = "generate"
    CATEGORY = "Geekatplay/GenUpscale"
    DESCRIPTION = ("Extracts Depth Maps, Tangent-Space Surface Normal Maps, and Surface Curvature Maps "
                   "directly from images on GPU. by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com")

    def generate(self, image, normal_strength=2.5, depth_smoothness=1.0, invert_depth=False, invert_y_normal=False):
        x = image.movedim(-1, 1).to(torch.float32) # (B, 3, H, W)

        # 1. Convert to grayscale luminance
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]

        # 2. Smooth depth estimation
        if depth_smoothness > 0:
            k_size = int(depth_smoothness * 4) | 1
            gray_smooth = _gaussian_blur(gray, kernel_size=max(3, k_size), sigma=max(0.5, depth_smoothness))
        else:
            gray_smooth = gray

        depth = gray_smooth
        if invert_depth:
            depth = 1.0 - depth

        # Normalize depth to 0..1
        d_min = depth.view(depth.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1)
        d_max = depth.view(depth.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)

        # 3. Compute surface normals via Scharr spatial gradients
        gx, gy = _scharr_gradients(depth_norm)
        gx = gx * normal_strength
        gy = gy * normal_strength

        if invert_y_normal:
            gy = -gy

        # Tangent space normal vector: N = (-gx, -gy, 1.0)
        gz = torch.ones_like(gx)
        normals = torch.cat([-gx, -gy, gz], dim=1) # (B, 3, H, W)

        # Vector normalization ||N|| = 1
        norm_len = torch.sqrt(torch.sum(normals ** 2, dim=1, keepdim=True)).clamp_min(1e-6)
        normals_unit = normals / norm_len

        # Map vector range [-1..1] to RGB [0..1]
        normal_rgb = (normals_unit * 0.5 + 0.5).clamp(0, 1)

        # 4. Compute Surface Curvature map (Laplacian)
        curv = _laplacian_curvature(depth_norm) * 2.0 + 0.5
        curv_rgb = curv.clamp(0, 1).repeat(1, 3, 1, 1)

        # Convert back to (B, H, W, 3) format for ComfyUI
        out_depth = depth_norm.repeat(1, 3, 1, 1).movedim(1, -1).cpu()
        out_normal = normal_rgb.movedim(1, -1).cpu()
        out_curvature = curv_rgb.movedim(1, -1).cpu()

        info = f"Depth, Surface Normal & Curvature Maps generated (strength={normal_strength:.1f})"
        return (out_depth, out_normal, out_curvature, info)
