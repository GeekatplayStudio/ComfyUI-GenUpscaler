"""GAP Depth & Normal Map Generator - fast GPU surface depth and normal map extraction.

Generates high-precision depth maps and tangent-space surface normal maps (RGB)
using GPU spatial gradient operators (Sobel/Scharr), Gaussian smoothing, and vector normalization.

by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com
"""
import torch
import torch.nn.functional as F


def _sobel_gradients(img_mono):
    """img_mono: (B, 1, H, W) -> gx, gy"""
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=img_mono.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=img_mono.device).view(1, 1, 3, 3)

    gx = F.conv2d(img_mono, sobel_x, padding=1)
    gy = F.conv2d(img_mono, sobel_y, padding=1)
    return gx, gy


def _gaussian_blur(img_mono, kernel_size=5, sigma=1.0):
    if kernel_size <= 1:
        return img_mono
    k = kernel_size
    grid = torch.arange(k, device=img_mono.device, dtype=torch.float32) - (k - 1) / 2
    g1d = torch.exp(-0.5 * (grid / sigma) ** 2)
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]
    g2d = g2d.view(1, 1, k, k)
    pad = k // 2
    return F.conv2d(img_mono, g2d, padding=pad)


class GAPDepthNormalGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to generate depth and normal maps from."}),
                "normal_strength": ("FLOAT", {"default": 2.5, "min": 0.1, "max": 10.0, "step": 0.1,
                                              "tooltip": "Strength/scale of the surface normal detail."}),
                "depth_smoothness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.2,
                                               "tooltip": "Smoothness of the depth estimation map."}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "invert_y_normal": ("BOOLEAN", {"default": False, "tooltip": "Invert Green channel (DirectX vs OpenGL normal convention)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("depth", "normal", "info")
    FUNCTION = "generate"
    CATEGORY = "Geekatplay/GenUpscale"
    DESCRIPTION = ("Generates high-quality Depth Maps and Tangent-Space Surface Normal Maps "
                   "directly from input images on GPU. by Geekatplay Studio / Vladimir Chopine")

    def generate(self, image, normal_strength=2.5, depth_smoothness=1.0, invert_depth=False, invert_y_normal=False):
        # input image: (B, H, W, 3)
        device = image.device
        x = image.movedim(-1, 1).to(torch.float32) # (B, 3, H, W)

        # 1. Convert to grayscale luminance
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]

        # 2. Smooth depth estimation
        if depth_smoothness > 0:
            k_size = int(depth_smoothness * 4) | 1 # make odd
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

        # 3. Compute surface normals via Sobel spatial gradients
        gx, gy = _sobel_gradients(depth_norm)
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

        # Convert back to (B, H, W, 3) format for ComfyUI
        out_depth = depth_norm.repeat(1, 3, 1, 1).movedim(1, -1).cpu()
        out_normal = normal_rgb.movedim(1, -1).cpu()

        info = f"Depth & Normal Maps generated (strength={normal_strength:.1f})"
        return (out_depth, out_normal, info)
