"""
Make "consistent" a measurement rather than a claim.

This runs after generation, on the finished PNGs, which is exactly why it is
worth having: every consistency lever a backend offers is vendor-specific and
pre-generation, and half the backends have no seed at all. Six numbers per icon
work identically on all of them.

Four decisions carry the module.

**Six cheap PIL features, not an embedding.** CLIP or DINO distance would be more
perceptually faithful, and it costs torch plus a model download -- a five-megabyte
pure-Python install becomes a multi-gigabyte one, to catch drift that ink
fraction, stroke thickness, saturation, luminance, a 64-bin RGB histogram and the
outline colour already catch. The bundle was prototyped in the research at 7.2 ms
per 128px icon and measures 1.7 ms here, so an 88-icon set audits in a fifth of a
second against the ~25 minutes that generating it takes. That ratio is the whole
argument for auditing every batch instead of hiding it behind a flag nobody
remembers. Revisit the embedding only when a measured set proves these six miss
real drift, and write that measurement into docs/findings.md first.

**Median and MAD, not mean and stdev.** The gate is the Iglewicz-Hoaglin modified
z-score, z = 0.6745*(x - median)/MAD, flagged at |z| > 3.5 -- a cutoff their
simulations put at roughly the false-positive rate |z| > 3 has for the classical
score. The robust pair matters because a mean and a standard deviation are
computed *from* the icons they are supposed to expose: several drifted icons out
of ten inflate sigma enough to cover themselves, and a cluster of drift is the
normal failure, not a lone bad render. Median and MAD tolerate up to 50%
contamination, so the cluster still sticks out.

**One absolute threshold, and only one.** Palette, stroke weight and background
are declared by the user, so a constant lifted from one 90s-surf-sticker set is
wrong for the next project. Everything here is relative to the set except the
background-share floor, whose provenance is measured and stated at BG_FLOOR.

**It is blind to meaning, and says so out loud.** The features see colour and
morphology, never semantics, so report() ends with that limitation in the printed
output and not only in the docs -- otherwise the audit gets read as a correctness
check, which it is not.

PIL only, no numpy, matching the rest of the package.
"""

import os
import textwrap

from PIL import Image, ImageChops, ImageFilter, ImageStat

#: 4x4x4 = 64 RGB buckets. Coarse on purpose: finer bins turn a one-shade palette
#: shift into a total mismatch, which is drift the eye would never call drift.
BINS = 4

#: Iglewicz-Hoaglin. Not tunable per feature: one cutoff, applied to every column.
Z_CUTOFF = 3.5

#: The one absolute threshold. Measured on this repo's own runs: a centred icon
#: on a 1024x1024 render keys out to 73-87% background, and the single 34% render
#: (the bullseye) was a genuine failure where the model ignored "single centered
#: object" and filled the frame. Only valid for a square, centred, untrimmed
#: render -- postprocess.trim_square() crops away precisely the margin this
#: counts, so the number has to come from cutout time, not from the final icon.
BG_FLOOR = 0.60

#: Below this the median/MAD rule is noise: with six icons a MAD is estimated
#: from three deviations. audit() reports the numbers instead of gating on them.
MIN_SET = 8

#: Alpha at or above this counts as ink. Matches postprocess.keep_subject().
ALPHA_CUT = 128

#: The audited columns, in report order. "palette" and "outline" are the two
#: vector features reduced to a scalar against the set; the other four are read
#: straight off features().
KEYS = ("ink", "thick", "sat", "lum", "palette", "outline")

_RGB_SPAN = (3 * 255 ** 2) ** 0.5      # longest possible RGB distance, ~441.7


def features(src):
    """Six numbers describing one icon, over its opaque pixels only.

    `src` is a path, a file-like object, or a PIL image, so a caller can hand in
    whatever it already has open.

    Returns None when nothing is opaque. A blank render has no style to compare,
    and inventing zeros for it would drag every median in the set.

        ink      opaque share of the canvas
        thick    2*area/boundary -- see below
        sat      mean HSV saturation, 0-255
        lum      mean luma, 0-255
        hist     64-bin RGB histogram, normalised, as a tuple so it compares
        outline  mean colour of the one-pixel band just inside the edge

    `thick` is a stroke-thickness *estimate* and behaves differently on strokes
    and on solids: the research measured rings of nominal width 2/4/8/16 px at
    2.00/3.03/6.18/12.45 (reproduced here at 2.00/3.07/6.21/12.47), and a solid
    disc of radius 88 at 69.78 (reproduced exactly; 79.22 at r=100 here). So it
    recovers a stroke width to about 25%, and on a filled shape it reads about
    four fifths of the radius -- short of the radius because the discretised
    boundary band is longer than the ideal circumference. The useful part is that
    those two readings differ by an order of magnitude, and line art drifting
    into solid fill is a real failure. Treat it as a relative statistic across
    one set, never as a pixel measurement.

    `outline` is the sharpest single "does this belong" signal for an outlined
    sticker set: the outline colour is what a scaffold pins hardest and what a
    model drifts on quietly. The research recovered (255,245,220) exactly from a
    synthetic cream-outlined icon.
    """
    im = _open(src)
    w, h = im.size
    mask = im.split()[3].point(lambda v: 255 if v >= ALPHA_CUT else 0)
    area = sum(mask.histogram()[255:])
    if not area:
        return None

    ring = _boundary(mask)
    per = sum(ring.histogram()[255:])   # non-zero whenever area is

    rgb = im.convert("RGB")
    return {
        "ink": area / float(w * h),
        "thick": 2.0 * area / per,
        "sat": ImageStat.Stat(im.convert("HSV"), mask).mean[1],
        "lum": ImageStat.Stat(rgb.convert("L"), mask).mean[0],
        "hist": tuple(c / float(area) for c in _rgb_hist(rgb, mask)),
        "outline": tuple(int(round(v)) for v in ImageStat.Stat(rgb, ring).mean),
    }


def distance(a, b):
    """How far apart two feature dicts are. 0.0 identical, 6.0 maximally apart.

    Every term is mapped onto 0..1 before summing, because the raw features are
    on wildly different scales -- ink is a fraction, saturation and luma run to
    255, thickness is unbounded, and the histogram contributes 64 numbers against
    five scalars. Combine them unscaled and the comparison is decided by whichever
    feature happens to carry the largest units rather than by which one drifted.

    The six weights are equal, and that is a guess: no measurement in this repo
    says an outline shift matters as much as an ink shift. Change them only with
    a measured set to point at.
    """
    thick = abs(a["thick"] - b["thick"]) / max(a["thick"], b["thick"], 1e-6)
    return (abs(a["ink"] - b["ink"])
            + min(thick, 1.0)
            + abs(a["sat"] - b["sat"]) / 255.0
            + abs(a["lum"] - b["lum"]) / 255.0
            + _l1(a["hist"], b["hist"]) / 2.0
            + _rgb_dist(a["outline"], b["outline"]) / _RGB_SPAN)


def nearest(candidates, anchor):
    """Index of the candidate closest to `anchor` in this feature space.

    This is the best-of-n selector, and it is what stands in for a seed on a
    hosted backend that has none: generate n, keep the one that already looks
    like the set. Candidates and anchor may each be a path, an image, or an
    already-computed feature dict.

    Two honest caveats, because both cost the user money.

    It is a HYPOTHESIS. That centroid-nearest selection beats simply taking the
    first candidate is unmeasured in this repo -- it is plausible, cheap, and the
    only lever that scales with spend on a seedless API, but nobody here has run
    the A/B. Do not read a returned index as a quality guarantee.

    n > 1 multiplies hosted cost by n, exactly. Three candidates across 88 icons
    is 264 billed generations, so n must default to 1 wherever this is wired in.
    """
    ref = anchor if isinstance(anchor, dict) else features(anchor)
    if ref is None:
        raise ValueError("anchor has no opaque pixels; nothing to select against")

    best, best_d = None, None
    for i, cand in enumerate(candidates):
        feats = cand if isinstance(cand, dict) else features(cand)
        if feats is None:
            continue                    # a blank render never wins
        d = distance(feats, ref)
        if best_d is None or d < best_d:
            best, best_d = i, d
    if best is None:
        raise ValueError("no candidate has any opaque pixels")
    return best


def audit(paths, bg_shares=None, min_set=MIN_SET):
    """Score a whole set and flag the icons that drifted out of it.

    `paths` is {name: path} or a sequence of paths (named by filename stem); a
    PIL image works anywhere a path does. `bg_shares` is {name: share} as
    returned by postprocess.render(), and is the only way the BG_FLOOR check can
    run -- the share has to be measured on the untrimmed render, so this cannot
    recover it from the finished icon and does not pretend to.

    Returns a dict: per-icon `values` and `scores` (modified z), the `stats` each
    column was gated against, `flagged` outliers as (feature, value, z) with z
    None for the absolute checks, `gated`, and `notes` explaining anything that
    was skipped. Nothing here prints; report() does that.
    """
    items = _named(paths)
    feats, blank = {}, []
    for name, src in items:
        one = features(src)
        if one is None:
            blank.append(name)
        else:
            feats[name] = one

    names = list(feats)
    result = {"count": len(items), "scored": len(names), "min_set": min_set,
              "gated": len(names) >= min_set, "features": feats,
              "values": {}, "scores": {}, "stats": {}, "flagged": {},
              "centroid": (), "outline_median": (), "notes": []}

    if not names:
        result["gated"] = False
        result["notes"].append("nothing to audit: no icon had any opaque pixels")
        _flag_blank(result, blank)
        return result

    centroid = _centroid([feats[n]["hist"] for n in names])
    mid_outline = tuple(_median([feats[n]["outline"][c] for n in names])
                        for c in range(3))
    result["centroid"] = centroid
    result["outline_median"] = mid_outline

    for name in names:
        one = feats[name]
        result["values"][name] = {
            "ink": one["ink"], "thick": one["thick"], "sat": one["sat"],
            "lum": one["lum"],
            "palette": _l1(one["hist"], centroid),
            "outline": _rgb_dist(one["outline"], mid_outline),
        }
        result["scores"][name] = {}

    for key in KEYS:
        column = [result["values"][n][key] for n in names]
        scores, med, spread, kind = _modified_z(column)
        result["stats"][key] = {"median": med, "spread": spread, "kind": kind}
        for name, z in zip(names, scores):
            result["scores"][name][key] = z
        if kind == "MeanAD":
            result["notes"].append(
                "%s: MAD was 0 (over half the set shares a value), so the gate "
                "fell back to the mean absolute deviation" % key)
        elif kind == "none":
            result["notes"].append(
                "%s: identical across the whole set, so nothing can be flagged "
                "on it" % key)

    if result["gated"]:
        for name in names:
            for key in KEYS:
                z = result["scores"][name][key]
                if abs(z) > Z_CUTOFF:
                    result["flagged"].setdefault(name, []).append(
                        (key, result["values"][name][key], z))
    else:
        result["notes"].append(
            "%d icon(s) is below the %d the median/MAD rule needs to mean "
            "anything, so the relative gate is OFF and the per-icon numbers are "
            "printed for eyeballing instead" % (len(names), min_set))

    if bg_shares is None:
        result["notes"].append(
            "no background shares passed, so the %.0f%% floor was not applied; "
            "it is measured at cutout time on the untrimmed square render and "
            "trim_square() removes the margin it counts" % (BG_FLOOR * 100))
    else:
        for name, share in bg_shares.items():
            if share < BG_FLOOR:
                result["flagged"].setdefault(name, []).append(
                    ("bg_share", share, None))

    _flag_blank(result, blank)
    return result


def report(result):
    """The audit as one plain ASCII block, printed once after a batch.

    ASCII and no colour, like backends.base.report(): this lands in a Windows
    console at least as often as a UTF-8 terminal.
    """
    head = "drift audit: %d icon(s)" % result["count"]
    if not result["gated"]:
        head += "   [relative gate OFF]"
    lines = [head]

    for key in KEYS:
        stat = result["stats"].get(key)
        if stat is None:
            continue
        lines.append("  %-8s median %9.3f   %-7s %8.3f"
                     % (key, stat["median"], stat["kind"], stat["spread"]))

    # With the gate off the relative rule flags nothing, so the numbers are the
    # output. Printing them is the difference between "too small to gate" and
    # silently returning an empty result.
    if not result["gated"] and result["values"]:
        lines.append("")
        lines.append("  %-16s %8s %8s %8s %8s %8s %8s"
                     % ("icon", "ink", "thick", "sat", "lum", "palette",
                        "outline"))
        for name in sorted(result["values"]):
            val = result["values"][name]
            lines.append("  %-16s %8.3f %8.2f %8.1f %8.1f %8.3f %8.1f"
                         % (name, val["ink"], val["thick"], val["sat"],
                            val["lum"], val["palette"], val["outline"]))

    flagged = result["flagged"]
    if flagged:
        lines.append("")
        for name in sorted(flagged):
            for key, value, z in flagged[name]:
                lines.append("  DRIFT  %-16s %-9s %s"
                             % (name, key, _why(key, value, z)))
    lines.append("")
    if result["gated"]:
        lines.append("  %d/%d within tolerance"
                     % (result["count"] - len(flagged), result["count"]))
    else:
        # Not "n/n within tolerance": nothing was gated, so nothing passed.
        lines.append("  %d flagged; the relative gate did not run"
                     % len(flagged))

    for note in result["notes"]:
        wrapped = textwrap.wrap(note, 72) or [""]
        lines.append("  note  %s" % wrapped[0])
        for tail in wrapped[1:]:
            lines.append("        %s" % tail)

    # Not a footnote, and not only in the docs: without this sentence in the
    # output somebody will read a clean audit as "the icons are correct".
    lines += ["",
              "  This sees colour and morphology, not semantics. A perfectly",
              "  on-palette, on-stroke render of the WRONG OBJECT scores clean.",
              "  It catches style drift, which is what the product promises; it",
              "  does not catch subject failure, which docs/findings.md",
              "  identifies as the binding constraint."]
    return "\n".join(lines)


def snap_palette(image, palette_hex):
    """Force one icon onto the declared palette. Opt-in, and here is why.

    Hard quantisation with no dither bands every antialiased edge -- the icon was
    resized with LANCZOS, so its edge pixels are blends that have nowhere to go
    but one side or the other. On a set whose colours were already fine it also
    changes pixels the user never asked to change. Apply it after the final
    resize, and only when asked for.

    Its interaction with vectorize.to_svg() is untested: vtracer's `flat` preset
    already collapses colours, so snapping first may cut the path count or may
    stair-step the traced edges. One measured run would settle it; until then,
    do not recommend the combination.

    What it buys is a hard guarantee no vendor offers -- exact palette
    conformance -- on every backend, because it runs after generation.
    """
    rgba = _open(image)
    colours = [_hex_rgb(h) for h in palette_hex]
    if not colours:
        raise ValueError("snap_palette needs at least one colour")
    if len(colours) > 256:
        raise ValueError("a PIL palette holds 256 colours, got %d" % len(colours))

    # Fill all 256 slots by repeating the declared colours. Padding the tail with
    # zeros instead -- the obvious move -- quietly adds black as a palette entry:
    # measured, a (40,40,44) pixel then snaps to pure black rather than to the
    # nearest colour anybody declared. Repeat whole triples, never a flat byte
    # list, or a slice lands mid-colour and invents a blend of two entries.
    colours = (colours * (256 // len(colours) + 1))[:256]
    pal = Image.new("P", (1, 1))
    pal.putpalette([c for rgb in colours for c in rgb])

    alpha = rgba.split()[3]
    out = rgba.convert("RGB").quantize(
        palette=pal, dither=Image.Dither.NONE).convert("RGB")
    out.putalpha(alpha)                 # quantising RGB never touched the alpha
    return out


# --- internals ----------------------------------------------------------

def _open(src):
    if isinstance(src, Image.Image):
        return src.convert("RGBA")
    return Image.open(src).convert("RGBA")


def _boundary(mask):
    """The one-pixel band just inside the mask edge.

    Padded first: PIL's rank filters copy the outermost row and column instead of
    filtering them, so a subject running to the canvas edge reports no boundary
    there -- measured, an 8x8 solid mask erodes to itself and yields a boundary
    of 0 px, which would make `thick` an order of magnitude too high and divide
    by zero on the way. One transparent pixel of padding makes the canvas edge
    count as an edge, which is also what it is.
    """
    w, h = mask.size
    pad = Image.new("L", (w + 2, h + 2), 0)
    pad.paste(mask, (1, 1))
    inner = pad.filter(ImageFilter.MinFilter(3)).crop((1, 1, w + 1, h + 1))
    return ImageChops.difference(mask, inner)


def _rgb_hist(rgb, mask):
    """Counts per 4x4x4 bucket, over the masked pixels.

    The bucket index is packed into a single L band so the count is one C-level
    histogram rather than 16384 Python iterations per icon. Max index is
    3*16 + 3*4 + 3 = 63, so ImageChops.add never clips.
    """
    packed = None
    for band, weight in zip(rgb.split(), (BINS * BINS, BINS, 1)):
        term = band.point(lambda v, w=weight: ((v * BINS) // 256) * w)
        packed = term if packed is None else ImageChops.add(packed, term)
    return packed.histogram(mask)[:BINS ** 3]


def _named(paths):
    if hasattr(paths, "items"):
        return list(paths.items())
    return [(os.path.splitext(os.path.basename(str(p)))[0], p) for p in paths]


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _centroid(hists):
    n = float(len(hists))
    return tuple(sum(h[i] for h in hists) / n for i in range(len(hists[0])))


def _l1(a, b):
    """L1 over the palette histogram: 0.0 identical, 2.0 disjoint. The research
    measured 0.00 against itself and 1.40 for the same shape in another hue, so
    the scale has real dynamic range rather than crowding near zero."""
    return sum(abs(x - y) for x, y in zip(a, b))


def _rgb_dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _modified_z(values):
    """(scores, median, spread, kind) -- Iglewicz-Hoaglin modified z-scores.

    MAD hits 0 the moment more than half the set shares a value, which is not
    exotic: it is what a nine-icon set with one bad render looks like. Returning
    zeros there would go blind on exactly the case the audit exists for, so this
    uses the fallback from the same paper, z = (x - median)/(1.253314*MeanAD),
    and names it in the output so nobody reads a MeanAD spread as a MAD.
    """
    med = _median(values)
    dev = [abs(v - med) for v in values]
    mad = _median(dev)
    if mad:
        return ([0.6745 * (v - med) / mad for v in values], med, mad, "MAD")
    mean_ad = sum(dev) / float(len(dev))
    if mean_ad:
        return ([(v - med) / (1.253314 * mean_ad) for v in values],
                med, mean_ad, "MeanAD")
    return ([0.0] * len(values), med, 0.0, "none")


def _flag_blank(result, blank):
    for name in blank:
        result["flagged"].setdefault(name, []).append(("blank", 0.0, None))


def _why(key, value, z):
    if key == "bg_share":
        return ("%.0f%% (floor %.0f%%) -- subject fills the frame"
                % (value * 100, BG_FLOOR * 100))
    if key == "blank":
        return "no opaque pixels -- the render is empty"
    return "%9.3f  z=%+.1f" % (value, z)


def _hex_rgb(value):
    text = str(value).lstrip("#").strip()
    if len(text) != 6:
        raise ValueError("palette colour %r is not RRGGBB" % value)
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except ValueError:
        raise ValueError("palette colour %r is not hexadecimal" % value)
