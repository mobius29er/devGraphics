"""
What a run will cost, roughly, and where the number came from.

Every figure here rots. Two of the models priced on 2026-08-20 changed status
inside the same week of research -- Imagen 4 shut down on 2026-08-17 and xAI
retired image ids inside six months -- so the table ships with an as-of date per
row, the date is printed next to the estimate, and `[price."backend:model"]` in
the config overrides any row without a release. The wording is always "estimate",
never "quote".

Two honest distortions worth naming rather than hiding. First, none of these
providers actually bills per image: OpenAI and Gemini bill image output as
tokens, so a per-image figure is derived and depends on quality and size. The
rows below are the 1024x1024 low/standard case an icon set uses, and the note
says which. Second, the headline number hides multipliers -- best-of-n multiplies
billed generations by n, reference-conditioned calls add input-image tokens on
top, and Gemini's thinking pass is billed and cannot be disabled.

`None` means unknown here, and the caller must print "unknown" rather than zero.
That is the opposite of `Capabilities.cost_per_image`, where None means local and
free; the difference is deliberate, because a missing price and a free backend
must never be confused when a --max-spend gate is deciding whether to proceed.
Local backends return 0.0 from this module, which is a measured fact, not a gap.

Stability is absent on purpose: only its edit/control rows were first-party
verified, its per-generation credit prices are secondary-source only, and there
is no stability backend in base.BUILTIN to price. Add a [price."..."] row if you
wire one up.
"""

#: "backend:model" -> (usd_per_image, as_of, source note).
#: Backend names are the ones in backends.base.BUILTIN. xAI arrives through
#: `openai-compatible` (see COMPAT_PRESETS), so it is keyed under that name.
TABLE = {
    # OpenAI, from the image-generation guide's cost table, fetched 2026-08-20.
    # Square 1024x1024 at low quality, which is what an icon set uses; the
    # transparency-capable models offer no other square size.
    "openai:gpt-image-1-mini": (
        0.005, "2026-08-20",
        "1024x1024 low; medium $0.011, high $0.036. Cheapest transparent-capable "
        "path -- 88 icons is about $0.44."),
    "openai:gpt-image-1.5": (
        0.009, "2026-08-20",
        "1024x1024 low; medium $0.034, high $0.133. Billed per output token, so "
        "this is derived, not a list price."),
    "openai:gpt-image-1": (
        0.011, "2026-08-20",
        "1024x1024 low; medium $0.042, high $0.167. Scheduled for deprecation "
        "2026-10-23."),
    "openai:gpt-image-2": (
        0.006, "2026-08-20",
        "1024x1024 low; medium $0.053, high $0.211. Cheapest of the family and "
        "the one that rejects background=transparent."),

    # Gemini, from ai.google.dev/gemini-api/docs/pricing (page dated 2026-08-13,
    # retrieved 2026-08-20). Image output is billed as output tokens; these are
    # Google's own per-image equivalents at 1K.
    "gemini:gemini-3.1-flash-image": (
        0.067, "2026-08-20",
        "1K (1024x1024); $0.045 at 512, $0.101 at 2K. Thinking tokens are billed "
        "on top at $3/1M and cannot be disabled. Batch halves it."),
    "gemini:gemini-3.1-flash-lite-image": (
        0.0336, "2026-08-20",
        "1K, the only size Lite supports. Cheapest Gemini row, but Lite has no "
        "style-reference slot, so the consistency lever is gone."),
    "gemini:gemini-3-pro-image": (
        0.134, "2026-08-20",
        "1K or 2K; $0.24 at 4K, plus about $0.0011 per input image. Thinking "
        "billed at $12/1M."),
    "gemini:gemini-2.5-flash-image": (
        0.039, "2026-08-20", "legacy model; batch/flex $0.0195."),

    # xAI, from docs.x.ai models page, fetched 2026-08-20. Driven through the
    # openai-compatible backend. The response also reports actual spend in
    # usage.cost_in_usd_ticks, which is the number to trust after the fact.
    "openai-compatible:grok-imagine-image": (
        0.02, "2026-08-20", "xAI via openai-compatible; no seed, no negative "
                            "prompt, no alpha."),
    "openai-compatible:grok-imagine-image-2.0": (
        0.04, "2026-08-20", "xAI via openai-compatible."),
    "openai-compatible:grok-imagine-image-quality": (
        0.05, "2026-08-20", "xAI via openai-compatible."),
}

#: Free, and that is a measured fact rather than a missing row: an 88-icon set
#: is about 25 minutes of wall clock on a local box and costs nothing per image.
LOCAL = frozenset({"fooocus", "comfyui", "invokeai", "a1111"})

CAVEAT = "an estimate, not a quote"


def lookup(backend, model, overrides=None):
    """(usd_per_image, as_of, note, source) for one backend/model, or None.

    `overrides` is the config's [price.*] table; it wins over the shipped row,
    because the shipped one is the one that goes stale.
    """
    key = "%s:%s" % (backend, model)
    entry = (overrides or {}).get(key)
    if isinstance(entry, dict) and isinstance(entry.get("per_image"), (int, float)):
        return (float(entry["per_image"]),
                entry.get("as_of") or "undated",
                entry.get("note") or "",
                '[price."%s"] in your config' % key)
    if key in TABLE:
        usd, as_of, note = TABLE[key]
        return usd, as_of, note, "pricing.py"
    return None


def estimate(backend, model, count, n=1, overrides=None):
    """(total_usd, provenance) for `count` subjects at best-of-n.

    total_usd is None when no price is known -- report that as "unknown", never
    as zero, and never gate --max-spend on it.
    """
    n = max(1, int(n or 1))
    billed = max(0, int(count)) * n

    if backend in LOCAL:
        return 0.0, ("%s runs locally: no per-image charge. The cost is your GPU "
                     "and about 25 minutes of wall clock for 88 icons." % backend)

    found = lookup(backend, model, overrides)
    if not found:
        key = "%s:%s" % (backend, model)
        return None, ("no price on record for %s. Report the cost as unknown, "
                      "never as zero; pin one with:\n"
                      '    [price."%s"]\n'
                      "    per_image = 0.0\n"
                      '    as_of     = "YYYY-MM-DD"' % (key, key))

    usd, as_of, note, source = found
    detail = "%d image%s" % (billed, "" if billed == 1 else "s")
    if n > 1:
        detail = "%d subjects x n=%d = %s" % (count, n, detail)
    provenance = "%s at $%s each, from %s, as_of %s -- %s" % (
        detail, _money(usd), source, as_of, CAVEAT)
    if note:
        provenance += "\n  %s" % note
    return round(usd * billed, 4), provenance


def known_models(backend):
    """Models this table can price for one backend. For `backends --describe`."""
    prefix = backend + ":"
    return sorted(k[len(prefix):] for k in TABLE if k.startswith(prefix))


def money(usd):
    """A price with its currency, for anything a user reads."""
    return "$%s" % _money(usd)


def _money(usd):
    """Prices here run from $0.005 to $0.24, so two decimals would print half of
    them as $0.01 or $0.00."""
    text = "%.4f" % usd
    return text.rstrip("0").rstrip(".") if "." in text else text
