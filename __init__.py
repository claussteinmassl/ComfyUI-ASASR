"""ComfyUI-ASASR: FLUX.1-dev dual-LoRA 4x super-resolution nodes."""
import os
import sys

# The node modules import the ``asasr`` package absolutely.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .asasr.native.nodes_native import (  # noqa: E402
    NATIVE_NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS,
    NATIVE_NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
