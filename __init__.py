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
NODES_DIR = os.path.join(HERE, "nodes")

if HERE not in sys.path:
    sys.path.insert(0, HERE)
if NODES_DIR not in sys.path:
    sys.path.insert(0, NODES_DIR)

try:
    from .nodes.detail_control import GAPDetailControl  # type: ignore # pyright: ignore
    from .nodes.tile_planner import GAPTilePlanner  # type: ignore # pyright: ignore
    from .nodes.upscale_prompt import GAPUpscalePrompt  # type: ignore # pyright: ignore
    from .nodes.color_match import GAPColorMatch  # type: ignore # pyright: ignore
    from .nodes.tiled_refine import GAPTiledRefine  # type: ignore # pyright: ignore
    from .nodes.compare_slider import GAPCompareSlider  # type: ignore # pyright: ignore
    from .nodes.gap360_refine import GAP360TiledRefine, GAP360TilePlanner  # type: ignore # pyright: ignore
    from .nodes.hdr_tonemap import GAPHDRTonemap  # type: ignore # pyright: ignore
    from .nodes.depth_normal import GAPDepthNormalGenerator  # type: ignore # pyright: ignore
    from .nodes.compare_360 import GAP360CompareViewer  # type: ignore # pyright: ignore
    from .nodes.detail_daemon import GAPDetailDaemon  # type: ignore # pyright: ignore
    from .nodes.io_hdr_raw import GAPLoadHDRAny, GAPSaveHDRAny  # type: ignore # pyright: ignore
except ImportError:
    try:
        from nodes.detail_control import GAPDetailControl  # type: ignore # pyright: ignore
        from nodes.tile_planner import GAPTilePlanner  # type: ignore # pyright: ignore
        from nodes.upscale_prompt import GAPUpscalePrompt  # type: ignore # pyright: ignore
        from nodes.color_match import GAPColorMatch  # type: ignore # pyright: ignore
        from nodes.tiled_refine import GAPTiledRefine  # type: ignore # pyright: ignore
        from nodes.compare_slider import GAPCompareSlider  # type: ignore # pyright: ignore
        from nodes.gap360_refine import GAP360TiledRefine, GAP360TilePlanner  # type: ignore # pyright: ignore
        from nodes.hdr_tonemap import GAPHDRTonemap  # type: ignore # pyright: ignore
        from nodes.depth_normal import GAPDepthNormalGenerator  # type: ignore # pyright: ignore
        from nodes.compare_360 import GAP360CompareViewer  # type: ignore # pyright: ignore
        from nodes.detail_daemon import GAPDetailDaemon  # type: ignore # pyright: ignore
        from nodes.io_hdr_raw import GAPLoadHDRAny, GAPSaveHDRAny  # type: ignore # pyright: ignore
    except ImportError:
        from detail_control import GAPDetailControl  # type: ignore # pyright: ignore
        from tile_planner import GAPTilePlanner  # type: ignore # pyright: ignore
        from upscale_prompt import GAPUpscalePrompt  # type: ignore # pyright: ignore
        from color_match import GAPColorMatch  # type: ignore # pyright: ignore
        from tiled_refine import GAPTiledRefine  # type: ignore # pyright: ignore
        from compare_slider import GAPCompareSlider  # type: ignore # pyright: ignore
        from gap360_refine import GAP360TiledRefine, GAP360TilePlanner  # type: ignore # pyright: ignore
        from hdr_tonemap import GAPHDRTonemap  # type: ignore # pyright: ignore
        from depth_normal import GAPDepthNormalGenerator  # type: ignore # pyright: ignore
        from compare_360 import GAP360CompareViewer  # type: ignore # pyright: ignore
        from detail_daemon import GAPDetailDaemon  # type: ignore # pyright: ignore
        from io_hdr_raw import GAPLoadHDRAny, GAPSaveHDRAny  # type: ignore # pyright: ignore

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
    "GAPLoadHDRAny": GAPLoadHDRAny,
    "GAPSaveHDRAny": GAPSaveHDRAny,
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
    "GAPLoadHDRAny": "Load EXR / DNG / HDR Image | GAP GenUpscale",
    "GAPSaveHDRAny": "Save EXR / DNG / HDR Image | GAP GenUpscale",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
