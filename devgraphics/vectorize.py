"""
Raster -> SVG.

Diffusion models cannot emit vector output; they paint pixels. But flat,
hard-edged sticker art is close to the ideal input for a tracer, so vectorising
afterwards gets you most of the way to resolution independence.

Presets trade file size against fidelity. `flat` is the one you usually want for
icons -- roughly 18 paths and ~29 KB for a typical icon, versus 670 paths and
~385 KB at `fine`, with little visible difference below 64px.

Honest limitation: a traced icon is still an order of magnitude heavier than a
hand-authored one (~29 KB vs ~1 KB), and its many-coloured paths cannot inherit
`currentColor`. Ship them as files, not inlined, and do not expect CSS theming.
"""

import vtracer

PRESETS = {
    # detail-preserving; large. Use for hero art, not icons.
    "fine": dict(filter_speckle=4, color_precision=8, path_precision=8, corner_threshold=60),
    # balanced.
    "smooth": dict(filter_speckle=12, color_precision=6, path_precision=6, corner_threshold=70),
    # aggressive flattening; best size/quality for small icons.
    "flat": dict(filter_speckle=24, color_precision=4, path_precision=5, corner_threshold=80),
}


def to_svg(src, dest, preset="flat", mode="spline"):
    """Trace `src` (PNG, ideally full-resolution with alpha) into an SVG."""
    if preset not in PRESETS:
        raise ValueError("unknown preset %r; choose from %s" % (preset, sorted(PRESETS)))
    vtracer.convert_image_to_svg_py(src, dest, colormode="color", mode=mode, **PRESETS[preset])
    return dest
