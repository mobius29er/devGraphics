"""
Turn a raw 1024x1024 SDXL render into a usable transparent icon.

Three decisions carry this module:

Flood-fill rather than threshold. An icon's own dark outline strokes sit close to
a dark backdrop, so a global colour-distance key eats them. Flood fill only
removes the *connected* outer region, so interior darks survive.

Seed from the whole border, not just the four corners. Diffusion backdrops are
never perfectly flat, so a patch touching the edge midway along a side is
unreachable from any corner.

Then drop stray blobs. Neither of the above removes speckle whose contrast against
the backdrop exceeds the fill threshold -- measured runs left 600-1600 such
fragments per image, showing up as grey smudges around the icon. Keeping only the
connected components that are a meaningful fraction of the largest one clears them
regardless of colour, which chasing the threshold never did.

This is also the layer that makes six different backends interchangeable. Only
one hosted API returns real alpha and no local SDXL install returns any; keying
the backdrop out here is what lets the rest of the pipeline stop caring which
generator produced the pixels.

PIL only, no numpy: keeps the tool dependency-light and OS-agnostic.
"""

import io
from collections import deque

from PIL import Image, ImageDraw, UnidentifiedImageError

from .backends.base import BackendError

SENTINEL = (255, 0, 255)
PNG_MAGIC = bytes((0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A))


def cutout(path, thresh=42, seeds=48):
    """Flood-fill the backdrop away; return (RGBA image, background share).

    `path` may be a filename or any file-like object, so a backend's bytes can go
    straight in without a temp file.
    """
    im = Image.open(path).convert("RGB")
    w, h = im.size

    points = []
    for i in range(seeds):
        t = i / float(max(seeds - 1, 1))
        x, y = int(t * (w - 1)), int(t * (h - 1))
        points += [(x, 0), (x, h - 1), (0, y), (w - 1, y)]

    px = im.load()
    for xy in points:
        if px[xy] != SENTINEL:  # already swallowed by an earlier fill
            ImageDraw.floodfill(im, xy, SENTINEL, thresh=thresh)

    rgba = im.convert("RGBA")
    px = rgba.load()
    removed = 0
    for y in range(h):
        for x in range(w):
            r, g, b, _ = px[x, y]
            if (r, g, b) == SENTINEL:
                px[x, y] = (0, 0, 0, 0)
                removed += 1
    return rgba, removed / float(w * h)


def keep_subject(rgba, work=256, keep_frac=0.15):
    """Drop opaque blobs too small to be part of the subject.

    Labels components on a downscaled mask -- a megapixel flood in pure Python is
    far too slow, and 256x256 is ample to separate an icon from speckle.
    """
    w, h = rgba.size
    small = rgba.resize((work, work), Image.NEAREST)
    alpha = small.split()[3].load()

    label = [[0] * work for _ in range(work)]
    sizes = {}
    current = 0
    for sy in range(work):
        for sx in range(work):
            if alpha[sx, sy] < 128 or label[sy][sx]:
                continue
            current += 1
            count = 0
            queue = deque([(sx, sy)])
            label[sy][sx] = current
            while queue:
                x, y = queue.popleft()
                count += 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < work and 0 <= ny < work:
                        if not label[ny][nx] and alpha[nx, ny] >= 128:
                            label[ny][nx] = current
                            queue.append((nx, ny))
            sizes[current] = count

    if not sizes:
        return rgba, 0

    biggest = max(sizes.values())
    keep = {k for k, v in sizes.items() if v >= biggest * keep_frac}

    mask = Image.new("L", (work, work), 0)
    mp = mask.load()
    for y in range(work):
        for x in range(work):
            if label[y][x] in keep:
                mp[x, y] = 255
    mask = mask.resize((w, h), Image.BILINEAR)

    out = rgba.copy()
    out.putalpha(Image.composite(rgba.split()[3], Image.new("L", (w, h), 0), mask))
    return out, len(sizes) - len(keep)


def trim_square(im, pad_ratio=0.06):
    """Crop to content, then re-centre on a square canvas with a little air."""
    bbox = im.getbbox()
    if not bbox:
        return im
    im = im.crop(bbox)
    side = int(max(im.size) * (1 + pad_ratio * 2))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(im, ((side - im.size[0]) // 2, (side - im.size[1]) // 2), im)
    return canvas


def render(src, dest, size=128, thresh=42, despeckle=True, keep_frac=0.15,
           pad_ratio=0.06):
    """Full path: key the backdrop, drop speckle, trim, square up, downscale."""
    im, ratio = cutout(src, thresh=thresh)
    if despeckle:
        im, _dropped = keep_subject(im, keep_frac=keep_frac)
    im = trim_square(im, pad_ratio=pad_ratio)
    im = im.resize((size, size), Image.LANCZOS)
    im.save(dest, "PNG", optimize=True)
    return ratio, im


# --- bytes in, bytes out ------------------------------------------------
#
# Backends hand back bytes rather than paths, because a hosted API has no
# filesystem to point at. These three are the seam between that contract and
# everything above, which was written path-first.

def to_png(data):
    """Normalise arbitrary image bytes to PNG.

    The backend contract is PNG specifically, and not for tidiness: cutout()
    flood-fills against the backdrop, and JPEG ringing around a hard outline is
    exactly the signal thresh=42 keys on. Gemini's response mime type enum offers
    JPEG only, so its backend has to come through here.
    """
    if data[:8] == PNG_MAGIC:
        return data
    buf = io.BytesIO()
    try:
        Image.open(io.BytesIO(data)).save(buf, "PNG")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        # Every backend's generate() funnels its response through here, and the
        # contract in backends/base.py is that a failure arrives as BackendError.
        # A truncated download or an HTML error page served as image bytes would
        # otherwise escape that hierarchy and kill a whole batch instead of one
        # icon.
        raise BackendError(
            "response is not a decodable image (%d bytes, starts %r): %s"
            % (len(data), data[:16], exc)) from exc
    return buf.getvalue()


def has_alpha(data, min_share=0.01):
    """True if these bytes carry a meaningful transparent region.

    A provider that accepts a transparent-background request and returns opaque
    white is a documented-but-unmeasured risk, so the caller verifies rather than
    trusts. `min_share` guards against an image that is merely antialiased at the
    edges.
    """
    im = Image.open(io.BytesIO(data))
    if im.mode not in ("RGBA", "LA", "PA") and "transparency" not in im.info:
        return False
    return _alpha_share(data) >= min_share


def render_bytes(data, dest, size=128, thresh=42, despeckle=True, keep_frac=0.15,
                 pad_ratio=0.06, cutout_bg=None):
    """render(), starting from bytes rather than a path.

    `cutout_bg` None means decide: bytes that already carry alpha skip the flood
    fill entirely, which is the whole point of a backend that can emit RGBA. True
    forces the fill, False forbids it.

    The returned ratio is the transparent share either way, so the drift audit
    and the "subject fills the frame" warning read the same number regardless of
    which backend produced the image.
    """
    png = to_png(data)
    keyed = has_alpha(png) if cutout_bg is None else not cutout_bg
    if keyed:
        im = Image.open(io.BytesIO(png)).convert("RGBA")
        ratio = _alpha_share(png)
    else:
        im, ratio = cutout(io.BytesIO(png), thresh=thresh)
    if despeckle:
        im, _dropped = keep_subject(im, keep_frac=keep_frac)
    im = trim_square(im, pad_ratio=pad_ratio)
    im = im.resize((size, size), Image.LANCZOS)
    im.save(dest, "PNG", optimize=True)
    return ratio, im


def _alpha_share(data):
    im = Image.open(io.BytesIO(data)).convert("RGBA")
    hist = im.split()[3].histogram()
    clear = sum(hist[:16])
    return clear / float(im.size[0] * im.size[1])
