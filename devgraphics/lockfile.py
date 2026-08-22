"""
What produced the assets in this directory, and whether the current profile
still matches it.

The drift check is the reason this file exists, and it is worth more than it
sounds. `iconset` skips subjects whose PNG already exists, which makes an
interrupted 88-icon run cost nothing to resume -- and makes the realistic
failure mode not "the seed got dropped" but "forty icons were generated on
Fooocus in January and the next forty-eight on OpenAI in March". Nothing else in
the pipeline notices that: both runs succeed, every icon is individually fine,
and the set silently stops being a set. No capability check catches it, because
neither backend did anything wrong. `compare()` does.

This records INPUTS, not pixels, and nothing here may be described as a
reproducible build. Model weights, sampler, torch version and GPU all move
diffusion output, and hosted models are retired on the provider's schedule --
inside one week of research Imagen 4 shut down, two xAI image ids started 404ing
and gpt-image-2 removed the transparency support gpt-image-1 had. Re-running
this lock re-issues the same request; it does not promise the same image. The
`note` field says so inside the artifact itself, where somebody reading the file
in a year will find it.

JSON, not TOML: tomllib is read-only, nothing in the stdlib writes TOML, and no
human hand-edits this. `sort_keys=True, indent=2` makes it diff cleanly.

Hand-authored assets are recorded with source="hand" precisely so they are never
regenerated. Measured: SDXL has no reliable prior for abstract glyphs -- check
marks and lightning bolts failed six retries across three style combinations --
so those are hand-written SVG, and a later run must leave them alone.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

NAME = "devgraphics.lock.json"
LOCKFILE_VERSION = 1

NOTE = ("Advisory. Records the inputs that produced these files. It does NOT "
        "guarantee byte-identical regeneration: model weights, sampler, torch "
        "version and GPU all affect output, and hosted models are retired on "
        "the provider's schedule.")

HAND_NOTE = ("hand-authored; SDXL has no reliable prior for abstract glyphs "
             "(docs/findings.md). Never regenerated.")

#: Compared field by field, because these are the ones a human can act on. A
#: digest mismatch with none of these changed is reported separately rather than
#: silently passing -- options, output and postprocess move the result too.
WATCHED = ("backend", "model", "seed", "render", "scaffold", "negative",
           "bg_hex", "palette", "anchor", "n")


def path(outdir):
    return os.path.join(outdir, NAME)


def read(outdir):
    """The existing lock, or None. Corrupt is an error, not a shrug.

    Silently treating an unreadable lock as absent would disable the drift check
    at exactly the moment it matters, so the caller is told to look at it.
    """
    p = path(outdir)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        try:
            return json.load(f)
        except ValueError as exc:
            raise ValueError("%s is not readable JSON (%s). Delete it to start "
                             "a fresh record, but check what wrote it first."
                             % (p, exc))


def write(outdir, profile_name, profile, assets, environment=None, previous=None):
    """Write the lock and return its path.

    `previous` is the document read at the start of the run, if any: its asset
    entries are carried forward and overwritten by `assets`, so a partial run
    does not erase the record of the icons it skipped -- including the
    hand-authored ones, which no run ever regenerates.
    """
    from . import __version__
    from .config import digest, to_dict

    merged = dict((previous or {}).get("assets") or {})
    merged.update(assets or {})

    doc = {
        "lockfile_version": LOCKFILE_VERSION,
        "devgraphics": __version__,
        "generated": utcnow(),
        "profile_name": profile_name,
        "profile_digest": digest(profile),
        "note": NOTE,
        "profile": to_dict(profile),
        "environment": environment or {},
        "assets": merged,
    }
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)
    p = path(outdir)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    return p


def mark_hand(outdir, entries, profile=None, profile_name=None):
    """Record slugs as hand-authored, so no later run regenerates them.

    `entries` is {slug: record} from hand() or from glyphs.author().

    This is the only way into hand_slugs(), and without it the whole
    hand-authored path was unreachable: the lockfile could hold the flag,
    plan() honoured it, and nothing could set it. The practice this project
    documents -- generate what a model draws well, author the rest by hand, and
    never regenerate over the authored ones -- depended on a switch with no
    handle.

    A lockfile may not exist yet: someone can reasonably author a handful of
    glyphs before generating anything. The profile is then whatever the caller
    passes, or the defaults, and the next real run overwrites it.
    """
    existing = read(outdir)
    if profile is None:
        from .config import DEFAULTS
        profile = (existing or {}).get("profile") or dict(DEFAULTS)
    return write(outdir, profile_name or (existing or {}).get("profile_name"),
                 profile, entries, previous=existing)


def compare(existing, profile):
    """Human-readable lines describing how `profile` differs from the lock.

    Empty means the set is still being generated the way it was started.
    """
    from .config import digest, to_dict

    if not existing:
        return []
    was = existing.get("profile") or {}
    now = to_dict(profile)

    changes = []
    for key in WATCHED:
        old, new = was.get(key), now.get(key)
        if old != new:
            changes.append("%-9s %s -> %s" % (key, _clip(old), _clip(new)))
    if changes:
        return changes

    old_digest = existing.get("profile_digest")
    if old_digest and old_digest != digest(profile):
        # Nothing a human names changed, so say which table did rather than
        # printing two hashes and leaving them to diff it.
        moved = [k for k in ("options", "output", "postprocess")
                 if was.get(k) != now.get(k)]
        changes.append("profile   %s -> %s%s"
                       % (_short(old_digest), _short(digest(profile)),
                          "  (%s changed)" % ", ".join(moved) if moved else ""))
    return changes


def drift_report(outdir, changes):
    """The block printed when compare() found something. One paragraph, because
    the user is about to decide whether to override it."""
    lines = ["drift: %s was written with a different profile" % path(outdir)]
    lines += ["  " + c for c in changes]
    lines += ["adding assets under a changed profile is how a set stops matching.",
              "re-run with --allow-drift, or -o OTHERDIR for a new set."]
    return "\n".join(lines)


# --- asset entries ------------------------------------------------------

def generated(subject, seed_used, data, png=None, svg=None, bg_share=None):
    """One generated asset. `seed_used` is what the backend reports it used, not
    what was asked for: A1111 echoes the real seed in `info` when -1 was sent and
    Stability echoes `seed` back, and that is the number worth keeping."""
    entry = {"source": "generated",
             "subject": subject,
             "seed_used": seed_used,
             "png_sha256": sha256(data),
             "generated": utcnow()}
    if png:
        entry["png"] = _slash(png)
    if svg:
        entry["svg"] = _slash(svg)
    if bg_share is not None:
        # 73-87% is the measured range for a centred icon. Far below it means
        # the subject fills the frame, which usually means the model ignored
        # "single centered object" -- worth having on record per icon.
        entry["bg_share"] = round(float(bg_share), 4)
    return entry


def hand(data, png=None, note=HAND_NOTE):
    """One hand-authored asset. Recorded so no later run regenerates it."""
    entry = {"source": "hand", "png_sha256": sha256(data), "note": note}
    if png:
        entry["png"] = _slash(png)
    return entry


def hand_slugs(existing):
    """Slugs a previous run recorded as hand-authored."""
    return sorted(slug for slug, e in ((existing or {}).get("assets") or {}).items()
                  if e.get("source") == "hand")


def sha256(data):
    """Hex digest of bytes, or of the file at a path."""
    if isinstance(data, (bytes, bytearray)):
        return hashlib.sha256(data).hexdigest()
    with open(data, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def utcnow():
    """ISO-8601 UTC to the second. `datetime.UTC` is 3.11+, so timezone.utc."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slash(p):
    """Forward slashes in the record. The lock is committed and read on other
    machines; a Windows backslash in a JSON string is both escaped and wrong."""
    return str(p).replace("\\", "/")


def _clip(value, width=48):
    text = json.dumps(value) if not isinstance(value, str) else value
    return text if len(text) <= width else text[:width - 3] + "..."


def _short(digest_text):
    return digest_text[:len("sha256:") + 12] if digest_text else "?"
