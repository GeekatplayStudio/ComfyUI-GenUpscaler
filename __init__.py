"""
Geekatplay GenUpscale - Generative AI Upscaler Suite
by Geekatplay Studio / Vladimir Chopine
https://www.geekatplay.com

Content-aware generative upscaling for ComfyUI:
extreme upscale -> tile -> generative detail refinement -> seamless blend.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    from .nodes import (
        GAPDetailControl,
        GAPTilePlanner,
        GAPUpscalePrompt,
        GAPColorMatch,
        GAPTiledRefine,
        GAPCompareSlider,
        GAP360TiledRefine,
        GAP360TilePlanner,
        GAPHDRTonemap,
        GAPDepthNormalGenerator,
        GAP360CompareViewer,
        GAPDetailDaemon,
    )
except ImportError:
    from nodes import (
        GAPDetailControl,
        GAPTilePlanner,
        GAPUpscalePrompt,
        GAPColorMatch,
        GAPTiledRefine,
        GAPCompareSlider,
        GAP360TiledRefine,
        GAP360TilePlanner,
        GAPHDRTonemap,
        GAPDepthNormalGenerator,
        GAP360CompareViewer,
        GAPDetailDaemon,
    )

NODE_CLASS_MAPPINGS = {
    "GAPDetailControl": GAPDetailControl,
    "GAPTilePlanner": GAPTilePlanner,
    "GAPUpscalePrompt": GAPUpscalePrompt,
    "GAPColorMatch": GAPColorMatch,
    "GAPTiledRefine": GAPTiledRefine,
    "GAPCompareSlider": GAPCompareSlider,
    "GAP360TiledRefine": GAP360TiledRefine,
    "GAP360TilePlanner": GAP360TilePlanner,
    "GAPHDRTonemap": GAPHDRTonemap,
    "GAPDepthNormalGenerator": GAPDepthNormalGenerator,
    "GAP360CompareViewer": GAP360CompareViewer,
    "GAPDetailDaemon": GAPDetailDaemon,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GAPDetailControl": "Detail Control (Creative vs Original) | GAP GenUpscale",
    "GAPTilePlanner": "Auto Tile Planner | GAP GenUpscale",
    "GAPUpscalePrompt": "Upscale Prompt Helper | GAP GenUpscale",
    "GAPColorMatch": "Color Match GPU | GAP GenUpscale",
    "GAPTiledRefine": "Tiled Generative Refine | GAP GenUpscale",
    "GAPCompareSlider": "Before/After Compare Slider | GAP GenUpscale",
    "GAP360TiledRefine": "360 Tiled Generative Refine | GAP GenUpscale",
    "GAP360TilePlanner": "360 Auto Tile Planner | GAP GenUpscale",
    "GAPHDRTonemap": "HDR Tonemap & Contrast | GAP GenUpscale",
    "GAPDepthNormalGenerator": "Depth & Surface Normal Generator | GAP GenUpscale",
    "GAP360CompareViewer": "360 Interactive Compare Viewer | GAP GenUpscale",
    "GAPDetailDaemon": "Detail Daemon (Micro-Texture Booster) | GAP GenUpscale",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
