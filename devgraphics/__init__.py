"""devGraphics -- a consistent asset set from the image generator of your choice.

Everything is imported lazily. That is not tidiness: with seven backends, an
eager package import would drag websocket-client in for someone who only wants
OpenAI, and would make `import devgraphics` fail outright on a machine without
the Rust wheel behind `to_svg`. PEP 562 keeps `from devgraphics import generate`
working while importing nothing until it is touched.
"""

import importlib

__version__ = "0.2.0"

#: attribute -> module it lives in. Kept explicit rather than scanned, so a typo
#: is an AttributeError here rather than a surprise at call time.
_EXPORTS = {
    "generate":       ".iconset",
    "contact_sheet":  ".iconset",
    "audit":          ".consistency",
    "cutout":         ".postprocess",
    "keep_subject":   ".postprocess",
    "render":         ".postprocess",
    "render_bytes":   ".postprocess",
    "trim_square":    ".postprocess",
    "to_svg":         ".vectorize",
    "Backend":        ".backends.base",
    "BackendError":   ".backends.base",
    "Capabilities":   ".backends.base",
    "Request":        ".backends.base",
    "load_backend":   ".backends.base",
    "Fooocus":        ".backends.fooocus",
    "FooocusError":   ".backends.fooocus",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    mod = importlib.import_module(module, __name__)
    value = getattr(mod, "load" if name == "load_backend" else name)
    globals()[name] = value          # import once, then it is a normal attribute
    return value


def __dir__():
    return sorted(list(globals()) + __all__)
