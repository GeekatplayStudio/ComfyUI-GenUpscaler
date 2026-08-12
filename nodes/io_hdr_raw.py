"""GAP Load & Save HDR / DNG / EXR - High-Dynamic-Range and RAW image import & export.

Supports:
  * OpenEXR (.exr) - 32-bit / 16-bit floating point high dynamic range.
  * Adobe Digital Negative (.dng) / Camera RAW - rawpy demosaicing & 16-bit TIFF container export.
  * Radiance HDR (.hdr) - 32-bit float RGB.
  * 16-bit TIFF (.tiff / .tif) - lossless 16-bit integer / float imagery.

by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com
"""
import os
import sys
import random
import numpy as np
import torch
from PIL import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import rawpy
except ImportError:
    rawpy = None

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import imageio.v3 as iio
except ImportError:
    iio = None

try:
    import folder_paths
except ImportError:
    folder_paths = None


def patch_comfy_server_max_payload():
    """Raises ComfyUI web server aiohttp max payload limit to 2GB to permanently solve HTTP 413 Content Too Large errors on DNG/EXR uploads."""
    try:
        import server
        if hasattr(server, "PromptServer") and server.PromptServer.instance:
            ps = server.PromptServer.instance
            if hasattr(ps, "app") and ps.app:
                ps.app._client_max_size = 2048 * 1024 * 1024
    except Exception:
        pass


patch_comfy_server_max_payload()


class GAPLoadHDRAny:
    """Load High-Dynamic-Range (OpenEXR .exr, Radiance .hdr) and RAW (Adobe DNG .dng) images."""

    @classmethod
    def INPUT_TYPES(cls):
        patch_comfy_server_max_payload()
        input_dir = folder_paths.get_input_directory() if folder_paths else "."
        files = []
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if f.lower().endswith((".exr", ".dng", ".hdr", ".tif", ".tiff", ".png", ".jpg", ".webp")):
                    files.append(f)
        files.sort()
        if not files:
            files = ["example.exr"]

        return {
            "required": {
                "image_file": (files, {"tooltip": "Select HDR (.exr, .hdr) or DNG RAW (.dng) file from input directory."}),
                "exposure_ev": ("FLOAT", {"default": 0.0, "min": -8.0, "max": 8.0, "step": 0.1,
                                          "tooltip": "EV exposure adjustment for standard preview output."}),
                "tonemap_preview": ("BOOLEAN", {"default": True, "tooltip": "Apply highlight compression for standard IMAGE output."}),
            },
            "optional": {
                "custom_file_path": ("STRING", {"default": "", "tooltip": "Optional absolute path to an external EXR/DNG file."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_hdr", "info")
    FUNCTION = "load_image"
    CATEGORY = "Geekatplay/GenUpscale"
    DESCRIPTION = ("Loads OpenEXR (.exr), Adobe DNG RAW (.dng), Radiance (.hdr), and 16-bit TIFF files. "
                   "Outputs standard IMAGE (0..1) and un-clamped IMAGE_HDR float tensors. "
                   "by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com")

    def load_image(self, image_file, exposure_ev=0.0, tonemap_preview=True, custom_file_path=""):
        patch_comfy_server_max_payload()

        # Strip surrounding quotes and whitespace from copy-pasted absolute paths
        path = custom_file_path.strip().strip('"').strip("'").strip()
        
        if not path or not os.path.isfile(path):
            input_dir = folder_paths.get_input_directory() if folder_paths else "."
            path = os.path.join(input_dir, image_file)

        if not os.path.isfile(path):
            # Check if file exists in input or output folders
            if folder_paths:
                for base in [folder_paths.get_input_directory(), folder_paths.get_output_directory()]:
                    candidate = os.path.join(base, os.path.basename(image_file))
                    if os.path.isfile(candidate):
                        path = candidate
                        break

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"HDR/DNG file not found: '{path}'.\n"
                f"To load large 360 DNG/EXR files:\n"
                f"1. Copy your DNG/EXR file into your ComfyUI/input/ folder, OR\n"
                f"2. Enter the full file path (e.g. C:\\Photos\\panorama360.dng) in 'custom_file_path'."
            )

        ext = os.path.splitext(path)[1].lower()
        img_np = None
        info_str = f"Loaded {os.path.basename(path)}"

        # 1. DNG / RAW format handling via rawpy
        if ext in (".dng", ".raw", ".cr2", ".nef", ".arw") and rawpy is not None:
            try:
                with rawpy.imread(path) as raw:
                    rgb16 = raw.postprocess(output_bps=16, use_camera_wb=True, half_size=False)
                    img_np = rgb16.astype(np.float32) / 65535.0
                    info_str += f" [DNG RAW 16-bit, {rgb16.shape[1]}x{rgb16.shape[0]}]"
            except Exception as e:
                info_str += f" [DNG rawpy note: {e}]"

        # 2. ImageIO reading (supports EXR, HDR, TIFF, DNG)
        if img_np is None and iio is not None and ext in (".exr", ".hdr", ".tif", ".tiff", ".dng"):
            try:
                arr = iio.imread(path)
                if arr.dtype == np.uint16:
                    img_np = arr.astype(np.float32) / 65535.0
                elif arr.dtype == np.uint8:
                    img_np = arr.astype(np.float32) / 255.0
                else:
                    img_np = arr.astype(np.float32)
                if img_np.ndim == 2:
                    img_np = np.stack([img_np] * 3, axis=-1)
                elif img_np.shape[2] == 4:
                    img_np = img_np[:, :, :3]
                info_str += f" [ImageIO {ext.upper()} {arr.dtype}, {img_np.shape[1]}x{img_np.shape[0]}]"
            except Exception:
                pass

        # 3. OpenCV fallback for EXR/HDR
        if img_np is None and cv2 is not None and ext in (".exr", ".hdr"):
            try:
                bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if bgr is not None:
                    if bgr.ndim == 2:
                        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
                    elif bgr.shape[2] == 4:
                        bgr = bgr[:, :, :3]
                    img_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
                    info_str += f" [OpenCV {ext.upper()}, {img_np.shape[1]}x{img_np.shape[0]}]"
            except Exception:
                pass

        # 4. Standard PIL fallback
        if img_np is None:
            pil_img = Image.open(path).convert("RGB")
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            info_str += f" [Standard PIL, {pil_img.width}x{pil_img.height}]"

        # Convert to (1, H, W, 3) PyTorch Float Tensor
        hdr_tensor = torch.from_numpy(img_np).unsqueeze(0)

        # Apply EV exposure
        if exposure_ev != 0.0:
            hdr_tensor = hdr_tensor * (2.0 ** exposure_ev)

        # Create standard preview tensor (clamped 0..1 with optional tonemapping)
        if tonemap_preview:
            preview_tensor = (hdr_tensor / (1.0 + hdr_tensor * 0.5)).clamp(0.0, 1.0)
        else:
            preview_tensor = hdr_tensor.clamp(0.0, 1.0)

        return (preview_tensor, hdr_tensor, info_str)


class GAPSaveHDRAny:
    """Save High-Dynamic-Range (OpenEXR .exr, Radiance .hdr) and Adobe DNG (.dng) images."""

    FORMATS = ["OpenEXR (.exr)", "Digital Negative (.dng)", "Radiance HDR (.hdr)", "16-bit TIFF (.tiff)"]

    @classmethod
    def INPUT_TYPES(cls):
        patch_comfy_server_max_payload()
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Image tensor to export."}),
                "filename_prefix": ("STRING", {"default": "GenUpscale_HDR"}),
                "format": (cls.FORMATS, {"default": "OpenEXR (.exr)"}),
            },
            "optional": {
                "images_hdr": ("IMAGE", {"tooltip": "Optional unclamped HDR tensor to save if connected."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_image"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/GenUpscale"
    DESCRIPTION = ("Saves images directly to OpenEXR (.exr), Adobe DNG (.dng), Radiance (.hdr), "
                   "or 16-bit TIFF format. by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com")

    def save_image(self, images, filename_prefix="GenUpscale_HDR", format="OpenEXR (.exr)", images_hdr=None):
        patch_comfy_server_max_payload()
        out_dir = folder_paths.get_output_directory() if folder_paths else "."
        target_tensor = images_hdr if images_hdr is not None else images

        results = []
        for i, img in enumerate(target_tensor):
            arr = img.cpu().numpy().astype(np.float32)

            rand_id = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(4))
            base_name = f"{filename_prefix}_{i:04d}_{rand_id}"

            if format == "OpenEXR (.exr)":
                ext = ".exr"
                full_path = os.path.join(out_dir, base_name + ext)
                written = False
                if iio is not None:
                    try:
                        iio.imwrite(full_path, arr)
                        written = True
                    except Exception:
                        pass
                if not written and cv2 is not None:
                    try:
                        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(full_path, bgr)
                        written = True
                    except Exception:
                        pass
                if not written and tifffile is not None:
                    full_path = os.path.join(out_dir, base_name + ".tiff")
                    tifffile.imwrite(full_path, arr.astype(np.float32))

            elif format == "Digital Negative (.dng)":
                ext = ".dng"
                full_path = os.path.join(out_dir, base_name + ext)
                arr16 = (np.clip(arr, 0, 1) * 65535.0).astype(np.uint16)
                if tifffile is not None:
                    tifffile.imwrite(full_path, arr16, photometric="rgb")
                elif iio is not None:
                    iio.imwrite(full_path, arr16)

            elif format == "Radiance HDR (.hdr)":
                ext = ".hdr"
                full_path = os.path.join(out_dir, base_name + ext)
                written = False
                if iio is not None:
                    try:
                        iio.imwrite(full_path, arr)
                        written = True
                    except Exception:
                        pass
                if not written and cv2 is not None:
                    try:
                        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(full_path, bgr)
                        written = True
                    except Exception:
                        pass

            else:  # 16-bit TIFF
                ext = ".tiff"
                full_path = os.path.join(out_dir, base_name + ext)
                arr16 = (np.clip(arr, 0, 1) * 65535.0).astype(np.uint16)
                if tifffile is not None:
                    tifffile.imwrite(full_path, arr16, photometric="rgb")
                else:
                    Image.fromarray((np.clip(arr, 0, 1) * 255.0).astype(np.uint8)).save(full_path)

            filename = os.path.basename(full_path)
            results.append({"filename": filename, "subfolder": "", "type": "output"})

        return {"ui": {"images": results}}
