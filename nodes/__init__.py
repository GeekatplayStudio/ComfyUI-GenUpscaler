"""Geekatplay GenUpscale - Node Definitions.

by Geekatplay Studio / Vladimir Chopine - https://www.geekatplay.com
"""
from .detail_control import GAPDetailControl
from .tile_planner import GAPTilePlanner
from .upscale_prompt import GAPUpscalePrompt
from .color_match import GAPColorMatch
from .tiled_refine import GAPTiledRefine
from .compare_slider import GAPCompareSlider
from .gap360_refine import GAP360TiledRefine, GAP360TilePlanner
from .hdr_tonemap import GAPHDRTonemap
from .depth_normal import GAPDepthNormalGenerator
from .compare_360 import GAP360CompareViewer
from .detail_daemon import GAPDetailDaemon

__all__ = [
    "GAPDetailControl",
    "GAPTilePlanner",
    "GAPUpscalePrompt",
    "GAPColorMatch",
    "GAPTiledRefine",
    "GAPCompareSlider",
    "GAP360TiledRefine",
    "GAP360TilePlanner",
    "GAPHDRTonemap",
    "GAPDepthNormalGenerator",
    "GAP360CompareViewer",
    "GAPDetailDaemon",
]
