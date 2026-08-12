"""GAP Depth, Surface Normal & Curvature Generator - AI Monocular Depth & 3D Surface Detail Extraction.

Uses Depth Anything v2 (SOTA AI Monocular Depth Estimation) to generate true 3D Depth Maps,
Tangent-Space Surface Normal Maps (RGB), and Surface Curvature Maps (Laplacian).

by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except ImportError:
    AutoImageProcessor = None
    AutoModelForDepthEstimation = None

# Global Model Cache: { (model_id, device_str): (processor, model) }
_DEPTH_MODEL_CACHE = {}


def _get_depth_anything_model(model_id, device):
    if AutoImageProcessor is None or AutoModelForDepthEstimation is None:
        return None, None

    cache_key = (model_id, str(device))
    if cache_key in _DEPTH_MODEL_CACHE:
        return _DEPTH_MODEL_CACHE[cache_key]

    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
        model.eval()
        _DEPTH_MODEL_CACHE[cache_key] = (processor, model)
        return processor, model
    except Exception as e:
        print(f"[GAP GenUpscale] Warning: Failed to load Depth Anything model '{model_id}': {e}")
        return None, None


def _scharr_gradients(img_mono):
    """img_mono: (B, 1, H, W) -> gx, gy (Scharr operator with high rotational symmetry)."""
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
    """True AI Monocular Depth & Tangent-Space Surface Normal Map Generator using Depth Anything v2."""

    DEPTH_MODELS = [
        "Depth Anything v2 (Small - Fast & Sharp)",
        "Depth Anything v2 (Base - High Quality)",
        "Depth Anything v2 (Large - Ultra Detail)",
        "Depth Anything v1 (Small)",
        "Heuristic (Fast Luminance Gradient)",
    ]

    MODEL_ID_MAP = {
        "Depth Anything v2 (Small - Fast & Sharp)": "depth-anything/Depth-Anything-V2-Small-hf",
        "Depth Anything v2 (Base - High Quality)": "depth-anything/Depth-Anything-V2-Base-hf",
        "Depth Anything v2 (Large - Ultra Detail)": "depth-anything/Depth-Anything-V2-Large-hf",
        "Depth Anything v1 (Small)": "LiheYoung/depth-anything-small-hf",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to generate 3D depth, normal, and curvature maps from."}),
                "normal_strength": ("FLOAT", {"default": 2.5, "min": 0.1, "max": 10.0, "step": 0.1,
                                              "tooltip": "Strength/scale of the surface normal detail."}),
                "depth_smoothness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1,
                                               "tooltip": "Gaussian smoothing filter applied to depth before normal computation."}),
                "invert_depth": ("BOOLEAN", {"default": False, "tooltip": "Invert depth map (Near=Black, Far=White vs Near=White, Far=Black)."}),
                "invert_y_normal": ("BOOLEAN", {"default": False, "tooltip": "Invert Green channel (DirectX vs OpenGL normal map convention)."}),
            },
            "optional": {
                "depth_model": (cls.DEPTH_MODELS, {"default": "Depth Anything v2 (Small - Fast & Sharp)",
                                                   "tooltip": "Select SOTA AI monocular depth estimation model."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("depth", "normal", "curvature", "info")
    FUNCTION = "generate"
    CATEGORY = "Geekatplay/GenUpscale"
    DESCRIPTION = ("Extracts true AI monocular 3D Depth Maps (Depth Anything v2), Tangent-Space Surface Normal Maps, "
                   "and Surface Curvature Maps directly on GPU. by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com")

    def generate(self, image, normal_strength=2.5, depth_smoothness=0.0, invert_depth=False, invert_y_normal=False,
                 depth_model="Depth Anything v2 (Small - Fast & Sharp)"):

        if not isinstance(depth_model, str) or depth_model not in self.DEPTH_MODELS:
            depth_model = "Depth Anything v2 (Small - Fast & Sharp)"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        b, h, w, c = image.shape
        info_str = f"Depth & Normal Generator [{depth_model}]"

        depth_tensor = None

        # 1. Run AI Depth Estimation (Depth Anything v2 / v1)
        if depth_model in self.MODEL_ID_MAP:
            model_id = self.MODEL_ID_MAP[depth_model]
            processor, model = _get_depth_anything_model(model_id, device)

            if processor is not None and model is not None:
                depth_list = []
                for i in range(b):
                    img_np = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)

                    inputs = processor(images=pil_img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        if device.type == "cuda":
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                raw_depth = model(**inputs).predicted_depth
                        else:
                            raw_depth = model(**inputs).predicted_depth

                    # Resize predicted depth back to original image resolution (1, 1, H, W)
                    depth_rescaled = F.interpolate(
                        raw_depth.unsqueeze(1).to(torch.float32),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False
                    )
                    depth_list.append(depth_rescaled)

                depth_tensor = torch.cat(depth_list, dim=0) # (B, 1, H, W)
                info_str += f" [AI Depth Anything v2, {w}x{h}]"

        # 2. Heuristic fallback if AI model unavailable or selected
        if depth_tensor is None:
            x = image.movedim(-1, 1).to(device, dtype=torch.float32)
            # Grayscale luminance
            gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
            depth_tensor = gray
            info_str += " [Heuristic Luminance Fallback]"

        # Apply optional smoothing
        if depth_smoothness > 0:
            k_size = int(depth_smoothness * 4) | 1
            depth_tensor = _gaussian_blur(depth_tensor, kernel_size=max(3, k_size), sigma=max(0.5, depth_smoothness))

        # Normalize depth per image in batch to range 0.0 .. 1.0
        d_min = depth_tensor.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        d_max = depth_tensor.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        depth_norm = (depth_tensor - d_min) / (d_max - d_min + 1e-6)

        if invert_depth:
            depth_norm = 1.0 - depth_norm

        # 3. Compute 3D Tangent-Space Surface Normal Map from AI Depth
        gx, gy = _scharr_gradients(depth_norm)
        gx = gx * normal_strength
        gy = gy * normal_strength

        if invert_y_normal:
            gy = -gy

        # 3D Normal Vector: N = (-gx, -gy, 1.0) normalized
        gz = torch.ones_like(gx)
        normals_raw = torch.cat([-gx, -gy, gz], dim=1) # (B, 3, H, W)
        norm_factor = torch.sqrt(torch.sum(normals_raw ** 2, dim=1, keepdim=True) + 1e-6)
        normals_unit = normals_raw / norm_factor # (B, 3, H, W) in [-1, 1]

        # Map [-1, 1] to RGB [0, 1]
        normals_rgb = 0.5 + 0.5 * normals_unit

        # 4. Compute Surface Curvature Map (Laplacian of depth)
        curvature_map = _laplacian_curvature(depth_norm)
        curv_min = curvature_map.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        curv_max = curvature_map.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        curvature_norm = (curvature_map - curv_min) / (curv_max - curv_min + 1e-6)

        # Convert tensors to ComfyUI (B, H, W, C) output format
        out_depth = depth_norm.repeat(1, 3, 1, 1).movedim(1, -1).cpu()
        out_normal = normals_rgb.movedim(1, -1).cpu()
        out_curvature = curvature_norm.repeat(1, 3, 1, 1).movedim(1, -1).cpu()

        return (out_depth, out_normal, out_curvature, info_str)
