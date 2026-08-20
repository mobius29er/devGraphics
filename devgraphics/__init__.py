"""devGraphics -- production-ready graphics from a local diffusion install."""

from .backends.fooocus import Fooocus, FooocusError
from .iconset import generate, contact_sheet
from .postprocess import cutout, keep_subject, render, trim_square
from .vectorize import to_svg

__version__ = "0.1.0"
__all__ = [
    "Fooocus", "FooocusError", "generate", "contact_sheet",
    "cutout", "keep_subject", "render", "trim_square", "to_svg",
]
