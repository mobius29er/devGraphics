"""
Batch icon generation with a single shared style scaffold.

Style consistency across many subjects is the whole ballgame -- 88 independently
styled icons read as AI slop, one coherent set reads as a design system. Three
levers do the work: an identical prompt scaffold, an identical style list, and a
fixed seed (SDXL keeps a recognisable look across subjects at a shared seed).
"""

import os
import time

from .backends.fooocus import Fooocus
from .postprocess import render

# 90s surf/skate flair in the site's own palette, so the set belongs to the brand
# rather than looking like stock clipart.
SCAFFOLD = (
    "flat vector sticker icon of {subject}, bold thick cream-white outline, "
    "1990s surf skate sticker style, solid flat colour fill, warm orange and "
    "golden yellow and coral red palette, centered single object, dark charcoal "
    "background, minimal, clean geometry, no text, no letters, no words"
)

NEGATIVE = (
    "photo, realistic, 3d render, gradient mesh, drop shadow, text, letters, "
    "words, watermark, signature, busy background, multiple objects, frame, border"
)

STYLES = ["Fooocus V2", "Fooocus Sharp"]
SEED = 77_777


def generate(subjects, outdir, size=128, seed=SEED, host="127.0.0.1:7865", log=print):
    """subjects: {slug: prompt-fragment}. Returns {slug: png path}."""
    f = Fooocus(host=host)
    os.makedirs(os.path.join(outdir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "icons"), exist_ok=True)

    done = {}
    total = len(subjects)
    for n, (slug, subject) in enumerate(subjects.items(), 1):
        final = os.path.join(outdir, "icons", slug + ".png")
        if os.path.exists(final):
            log("  [%d/%d] %-20s cached" % (n, total, slug))
            done[slug] = final
            continue

        t0 = time.time()
        try:
            paths = f.generate(
                prompt=SCAFFOLD.format(subject=subject),
                negative=NEGATIVE,
                styles=STYLES,
                size="1024×1024",
                count=1,
                seed=seed,
            )
        except Exception as exc:
            log("  [%d/%d] %-20s FAILED %s" % (n, total, slug, exc))
            continue

        if not paths:
            log("  [%d/%d] %-20s no image returned" % (n, total, slug))
            continue

        raw = f.download(paths[0], os.path.join(outdir, "raw", slug + ".png"))
        ratio, _ = render(raw, final, size=size)
        done[slug] = final
        log("  [%d/%d] %-20s %.0fs  bg %.0f%%" % (n, total, slug, time.time() - t0, ratio * 100))

    return done


def contact_sheet(paths, dest, cell=96, cols=8, bg=(13, 13, 13, 255), label_h=0):
    """Lay icons out on the real hero background so drift is obvious."""
    from PIL import Image

    items = list(paths.items())
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * cell, rows * (cell + label_h)), bg)
    for i, (_slug, p) in enumerate(items):
        ic = Image.open(p).convert("RGBA").resize((cell - 20, cell - 20), Image.LANCZOS)
        x = (i % cols) * cell + 10
        y = (i // cols) * (cell + label_h) + 10
        sheet.paste(ic, (x, y), ic)
    sheet.save(dest)
    return dest
