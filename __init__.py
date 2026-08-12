"""
Geekatplay GenUpscale - Generative AI Upscaler Suite
by Geekatplay Studio / Vladimir Chopine
https://www.geekatplay.com

Content-aware generative upscaling for ComfyUI:
extreme upscale -> tile -> generative detail refinement -> seamless blend.
"""

from .nodes.detail_control import GAPDetailControl
from .nodes.tile_planner import GAPTilePlanner
from .nodes.upscale_prompt import GAPUpscalePrompt
from .nodes.color_match import GAPColorMatch
from .nodes.tiled_refine import GAPTiledRefine
from .nodes.compare_slider import GAPCompareSlider
from .nodes.gap360_refine import GAP360TiledRefine, GAP360TilePlanner
from .nodes.hdr_tonemap import GAPHDRTonemap
from .nodes.depth_normal import GAPDepthNormalGenerator
from .nodes.compare_360 import GAP360CompareViewer

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
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
