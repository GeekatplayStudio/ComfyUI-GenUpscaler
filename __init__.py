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

NODE_CLASS_MAPPINGS = {
    "GAPDetailControl": GAPDetailControl,
    "GAPTilePlanner": GAPTilePlanner,
    "GAPUpscalePrompt": GAPUpscalePrompt,
    "GAPColorMatch": GAPColorMatch,
    "GAPTiledRefine": GAPTiledRefine,
    "GAPCompareSlider": GAPCompareSlider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GAPDetailControl": "Detail Control (Creative vs Original) | GAP GenUpscale",
    "GAPTilePlanner": "Auto Tile Planner | GAP GenUpscale",
    "GAPUpscalePrompt": "Upscale Prompt Helper | GAP GenUpscale",
    "GAPColorMatch": "Color Match GPU | GAP GenUpscale",
    "GAPTiledRefine": "Tiled Generative Refine | GAP GenUpscale",
    "GAPCompareSlider": "Before/After Compare Slider | GAP GenUpscale",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
