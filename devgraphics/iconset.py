"""
Batch generation with a single shared style scaffold.

Style consistency across many subjects is the whole ballgame -- 88 independently
styled icons read as AI slop, one coherent set reads as a design system. Which
levers are available depends on the backend, and that is the only reason this
module is more than a loop:

  A fixed seed on a pinned checkpoint. The measured, working strategy: identical
  prompt scaffold, identical style list, identical seed. Local backends only.

  An anchor image as a style reference. What replaces the seed when there isn't
  one. Render one hero icon first, then condition every other icon on it.

  The scaffold alone. The floor, and the only lever that works everywhere.

`preflight` in backends/base.py decides which of those a chosen backend can
actually honour, and this module refuses to quietly proceed without any of them.
A set generated with none of the three is a pile of unrelated pictures, and the
one thing worse than failing is producing that and calling it an icon set.

Two guards exist because the *skip* behaviour makes drift invisible. Finished
icons are skipped on re-run, which is what makes an interrupted batch free -- and
also what lets someone generate forty icons on Fooocus in January and the next
forty-eight on OpenAI in March. The lockfile catches that; nothing else can.
"""

import io
import os
import string
import time

from . import config, consistency, lockfile, pricing
from .backends import base
from .postprocess import render_bytes

# Kept as module constants because 0.1 exported them and the README shows them.
# The live values now come from config.DEFAULTS via a resolved profile.
SCAFFOLD = config.SCAFFOLD
NEGATIVE = config.NEGATIVE
STYLES = ["Fooocus V2", "Fooocus Sharp"]
SEED = 77_777


class SetError(RuntimeError):
    """The set cannot be generated as asked. Carries a human-readable reason."""


def build(profile):
    """Construct the backend a profile names, with its options and model.

    Separate from generate() so `devgraphics gen --dry-run` and
    `devgraphics backends --probe` can inspect a configured backend without
    generating anything -- which works only because backend constructors are
    forbidden from touching the network.
    """
    name = profile.get("backend") or "fooocus"
    options = dict(profile.get("options") or {})
    model = profile.get("model")
    key = base.MODEL_OPTION.get(name, "model") if model else None
    if key:
        options.setdefault(key, model)
    return base.load(name, **options)


def advisory_model(profile):
    """The profile's `model` when the backend cannot actually be told about it.

    Fooocus picks its checkpoint in its own UI; nothing in the queue protocol
    sets it. Recording the name is still worth doing -- the lockfile compares it
    on the next run, so swapping checkpoints halfway through a set is caught even
    though devgraphics could not have prevented it. What must not happen is the
    profile implying a pin that is not being applied, so the caller says so once.
    """
    name = profile.get("backend") or "fooocus"
    model = profile.get("model")
    if model and base.MODEL_OPTION.get(name, "model") is None:
        return model
    return None


def effective(profile):
    """A profile whose `model` reflects the model actually in play.

    A backend takes its model as an option, so `-O model=gpt-image-1-mini` and
    `-O checkpoint=sd_xl_base` are both legitimate ways to choose one -- and both
    used to leave profile["model"] empty. Two things read that field and neither
    complains when it is blank: pricing.estimate(), which then reports "unknown"
    for a model it has a published price for, and the lockfile, which then cannot
    tell that a set was half generated on one model and half on another. Lift it
    once, here, so both see the same answer however it was spelled.
    """
    if profile.get("model"):
        return profile
    key = base.MODEL_OPTION.get(profile.get("backend") or "fooocus", "model")
    named = (profile.get("options") or {}).get(key) if key else None
    return dict(profile, model=named) if named else profile


def plan(subjects, outdir, profile, force=False, only=()):
    """What a run would do, without doing any of it.

    Returns a dict with the four buckets that matter for both the dry run and the
    real one: what is already finished, what was hand-authored and must never be
    regenerated, what is left, and what that will cost.

    `only` narrows the run to named slugs and re-renders them whether or not they
    already exist. That is the iteration loop: a diffusion model gets some
    subjects wrong on the first pass, and the fix is a better prompt fragment for
    those subjects, not another twenty minutes for the whole set.
    """
    profile = effective(profile)
    only = tuple(only or ())
    unknown = [slug for slug in only if slug not in subjects]
    if unknown:
        raise SetError("no such subject in the manifest: %s; it has: %s"
                       % (", ".join(unknown), ", ".join(sorted(subjects))))

    lock = lockfile.read(outdir)
    hand = lockfile.hand_slugs(lock) if lock else set()
    cached, todo = [], []
    for slug in subjects:
        if slug in hand:
            continue
        if only:
            (todo if slug in only else cached).append(slug)
        elif not force and os.path.exists(_icon_path(outdir, slug)):
            cached.append(slug)
        else:
            todo.append(slug)

    anchor = profile.get("anchor")
    if anchor and anchor in todo:                # the anchor is rendered first
        todo.remove(anchor)
        todo.insert(0, anchor)

    n = max(1, int(profile.get("n") or 1))
    total, provenance = pricing.estimate(
        profile.get("backend"), profile.get("model"), len(todo), n=n,
        overrides=profile.get("_prices"))
    return {"cached": cached, "hand": sorted(hand), "todo": todo,
            "n": n, "anchor": anchor, "calls": len(todo) * n,
            "cost": total, "cost_note": provenance, "lock": lock}


def waivers(caps, profile, strict=True):
    """What a profile asks for that this backend cannot honour.

    Public because the dry run and the real run must agree exactly. A --dry-run
    that reports a fatal waiver the real run then forgives -- or the reverse --
    is worse than no dry run at all.
    """
    template = _request(profile, "placeholder")
    found = base.preflight(caps, template, strict=strict)
    return template, _forgive_anchor(found, caps, profile)


def generate(subjects, outdir, size=None, seed=None, host=None, profile=None,
             backend=None, force=False, allow_drift=False, write_lock=True,
             only=(), log=print):
    """subjects: {slug: prompt-fragment}. Returns {slug: png path}.

    The 0.1 signature still works -- generate(subjects, outdir, size=128,
    seed=N, host=H) drives Fooocus exactly as before -- because that call is in
    the README and in whatever scripts people already wrote. `profile` is the
    new road: a resolved dict from config.resolve().
    """
    profile = effective(_profile(profile, size=size, seed=seed, host=host,
                                 backend=backend))
    outdir = str(outdir)
    made = {}

    work = plan(subjects, outdir, profile, force=force, only=only)
    backend_obj = build(profile)
    caps = backend_obj.capabilities

    template, waived = waivers(caps, profile, strict=len(work["todo"]) > 1)
    log(base.report(caps, waived, template))
    advisory = advisory_model(profile)
    if advisory:
        log("  note  model %r is recorded but not applied: %s takes no model "
            "over the wire. Set it in the %s UI; the lockfile still compares it."
            % (advisory, caps.name, profile.get("backend")))
    if any(w.fatal for w in waived) and not allow_drift:
        raise SetError("refusing to generate a set with no consistency lever; "
                       "pass allow_drift=True (CLI: --allow-drift) to override")

    drift = lockfile.compare(work["lock"], profile) if work["lock"] else []
    if drift and not allow_drift:
        raise SetError(lockfile.drift_report(outdir, drift))
    if drift:
        log(lockfile.drift_report(outdir, drift))

    for directory in ("raw", "icons"):
        os.makedirs(os.path.join(outdir, directory), exist_ok=True)

    n = _candidates(work["n"], caps, log)
    anchor_ref = ()
    assets = dict((work["lock"] or {}).get("assets") or {})
    for slug in work["cached"]:
        made[slug] = _icon_path(outdir, slug)

    total = len(work["todo"])
    for index, slug in enumerate(work["todo"], 1):
        subject = subjects[slug]
        started = time.time()
        request = base.strip(
            _request(profile, subject, refs=anchor_ref, count=1), waived)
        try:
            data = _one(backend_obj, request, outdir, slug, n=n,
                        anchor=anchor_ref)
        except base.PaymentRequired as exc:
            raise SetError("out of credit after %d of %d icons: %s"
                           % (index - 1, total, exc))
        except Exception as exc:
            log("  [%d/%d] %-20s FAILED %s" % (index, total, slug, exc))
            continue

        final = _icon_path(outdir, slug)
        share, _image = render_bytes(data, final, size=profile["output"]["size"],
                                     **_post_kwargs(profile))
        made[slug] = final
        assets[slug] = lockfile.generated(
            subject, _seed_used(backend_obj, request), data,
            png=os.path.relpath(final, outdir), bg_share=share)
        if profile.get("anchor") == slug and caps.reference_images:
            anchor_ref = (data,)                 # every later icon references it
        log("  [%d/%d] %-20s %.0fs  bg %.0f%%"
            % (index, total, slug, time.time() - started, share * 100))

    if write_lock and made:
        lockfile.write(outdir, profile.get("_name"), profile, assets,
                       environment={"backend": caps.name},
                       previous=work["lock"])
    return made


def audit(made, outdir=None, log=print):
    """Run the numeric drift audit over a finished set and print the report.

    `outdir` is where the background shares come from, and passing it is not
    optional in practice. The 60% floor is the one absolute check the audit has,
    and it cannot be recovered from a finished icon: the share is measured on the
    untrimmed square render, and trim_square() has already cropped away the
    margin it counts. So it lives in the lockfile, and an audit that does not
    read the lockfile silently runs with its only absolute rule switched off.

    Measured, which is why this reads it: a real seven-icon set came back with
    two icons under the floor -- a bullseye at 0.34 and a subject the model got
    wrong at 0.47 -- and the run reported "0 flagged" because the shares never
    reached the audit.
    """
    result = consistency.audit(made, bg_shares=shares(outdir) if outdir else None)
    log(consistency.report(result))
    return result


def shares(outdir):
    """{slug: background share} from a set's lockfile, or None."""
    lock = lockfile.read(outdir)
    if not lock:
        return None
    found = dict((slug, entry["bg_share"])
                 for slug, entry in (lock.get("assets") or {}).items()
                 if entry.get("bg_share") is not None)
    return found or None


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


# --- internals ----------------------------------------------------------

def _one(backend_obj, request, outdir, slug, n=1, anchor=()):
    """Render one subject and return its PNG bytes.

    n > 1 is the best-of-n path: generate several and keep whichever already
    looks most like the anchor. It is the only lever that scales with spend on a
    backend with no seed, and it multiplies cost by exactly n.

    With no anchor there is nothing to select against, so the first candidate
    wins. That case is reachable: the anchor icon is itself the first thing
    rendered, and it has no earlier icon to resemble.
    """
    candidates = []
    for _attempt in range(n):
        images = backend_obj.generate(request)
        if not images:
            raise base.BackendError("backend returned no image")
        candidates.append(images[0])

    if len(candidates) == 1 or not anchor:
        chosen = candidates[0]
    else:
        chosen = candidates[consistency.nearest(
            [io.BytesIO(c) for c in candidates], io.BytesIO(anchor[0]))]

    raw = os.path.join(outdir, "raw", slug + ".png")
    with open(raw, "wb") as handle:
        handle.write(chosen)
    return chosen


def _request(profile, subject, refs=(), count=1):
    return base.Request(
        prompt=_fill(profile["scaffold"], subject=subject,
                     palette=", ".join(profile.get("palette") or []),
                     bg_hex=profile.get("bg_hex") or ""),
        negative=profile.get("negative") or "",
        seed=profile.get("seed"),
        size=tuple(profile["render"]),
        count=count,
        transparent=True,
        refs=tuple(refs),
        options={},
    )


def _fill(template, **values):
    """format() that leaves an unknown placeholder alone.

    The scaffold is hand-written in a config file. A stray {brace} in it should
    read oddly in the prompt, not abort a 25-minute batch with a KeyError.
    """
    class _Lenient(dict):
        def __missing__(self, key):
            return "{%s}" % key

    return string.Formatter().vformat(template, (), _Lenient(values))


def _profile(profile, size=None, seed=None, host=None, backend=None):
    """Fold the 0.1 keyword arguments into a resolved profile."""
    if profile is not None:
        return profile
    overrides = {}
    if size is not None:
        overrides["size"] = size
    if seed is not None:
        overrides["seed"] = seed
    if host is not None:
        overrides["host"] = host
    if backend is not None:
        overrides["backend"] = backend
    return config.resolve({}, None, overrides)


def _forgive_anchor(waivers, caps, profile):
    """A missing seed stops being fatal once an anchor can carry the style.

    Without this, every hosted-API user's first command fails and their second is
    --allow-drift, at which point the flag has stopped meaning anything.
    """
    if not (profile.get("anchor") and caps.reference_images):
        return waivers
    return tuple(
        base.Waiver(w.option, w.requested,
                    w.reason + "; the --anchor reference carries the style instead",
                    fatal=False)
        if w.option == "seed" else w
        for w in waivers)


def _candidates(n, caps, log):
    """Clamp best-of-n where it would only buy n identical images."""
    if n > 1 and caps.seed:
        log("  note  n=%d ignored: %s honours a fixed seed, so every candidate "
            "would be the same image" % (n, caps.name))
        return 1
    return n


def _post_kwargs(profile):
    """postprocess settings render_bytes() understands.

    snap_palette is dropped here rather than passed through: it is a separate,
    opt-in pass over the finished icon, not an argument to the cutout.
    """
    post = dict(profile.get("postprocess") or {})
    post.pop("snap_palette", None)
    return post


def _seed_used(backend_obj, request):
    """What the backend says it actually used, falling back to what we asked.

    A1111 echoes the real seed inside a JSON-encoded string when -1 was sent, and
    Stability echoes the one it chose. The lockfile wants the truth, not the ask.
    """
    return getattr(backend_obj, "last_seed", None) or request.seed


def _icon_path(outdir, slug):
    return os.path.join(outdir, "icons", slug + ".png")
