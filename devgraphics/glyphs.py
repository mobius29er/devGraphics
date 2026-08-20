"""
Claude-authored SVG glyphs -- the half of an icon set diffusion cannot draw.

This is deliberately not a Backend and is deliberately absent from
`base.BUILTIN`, because Anthropic ships no image generation of any kind. The
Vision docs answer it verbatim: "No, Claude is an image understanding model only.
It can interpret and analyze images, but it cannot generate, produce, edit,
manipulate, or create images." There is no images path on /v1, and no image block
in the Messages response union -- text is the only thing that comes back. What
comes back here is SVG *source*, which is text.

That is exactly the right tool for one measured problem. docs/findings.md records
SDXL failing on check marks and lightning bolts across six retries and three
style combinations, every one of them the same striped rounded blob. Check, X,
arrow, bolt and plus/minus are the smaller half of a set by variety and the
larger half by usage -- the check mark alone was 43 of 470 emoji uses in the set
that prompted this work. A model that has read a million icon sets and writes
three coordinates fits that subset better than one painting pixels, and the
artefact is better too: a hand-authored glyph is ~1 KB and inherits
`currentColor`, against ~29 KB of 18-path traced SVG that cannot be themed at
all. Emitting currentColor is the whole reason to hand-author rather than trace,
so this module enforces it rather than hoping the model complied.

It is not reproducible, and no knob makes it so. The Messages API has no seed,
and temperature, top_p and top_k are all rejected with a 400 on current models.
So this is not a build step: author once, look at what came back, commit the SVG
to git, and treat regeneration as a human decision. `author()` refuses to
overwrite a glyph that already exists unless you pass force=True, so a re-run
cannot silently churn committed assets.

What comes back is untrusted model output being written into a build directory,
and it is treated that way. Every glyph is parsed as XML and rejected -- never
repaired, never written -- unless it has an svg root, carries the viewBox its
style declares, and contains no script, no <image>, no <foreignObject>, no event
handler attribute and no reference that leaves the document.

Whether Claude draws *good* glyphs is UNVERIFIED. No Anthropic document makes any
claim about SVG quality, because SVG is just text to the API, and this repo has
not measured it yet. The theory is sound; prove it the way everything else here
was proved -- put the output on iconset.contact_sheet() and write the result into
findings.md.
"""

import collections
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET

from . import _http
from .backends.base import (AuthError, BackendError, ModerationBlocked,
                            RateLimited, UnsupportedOption)

ENDPOINT = "https://api.anthropic.com/v1/messages"

#: Mandatory on every Messages request, and not a beta flag -- omitting it is a
#: 400, not a default.
API_VERSION = "2023-06-01"

#: Current as of the 2026-08-20 research pass. Model ids are not versioned by
#: date on this family; `claude-opus-5` is the id, with no suffix.
MODEL = "claude-opus-5"

API_KEY_ENV = "ANTHROPIC_API_KEY"

#: A hand-authored icon SVG measures 400-800 output tokens, so this is roughly
#: five times the room one glyph needs. Thinking tokens are billed against it too.
MAX_TOKENS = 4096

#: Also the server-side default on Opus 5, sent explicitly so that a change to
#: that default cannot quietly change glyph geometry under a committed set.
EFFORT = "high"

TIMEOUT = 300

#: Structured outputs, GA -- no beta header. This is the documented shape and it
#: is also the fix for the obvious trick: prefilling an assistant turn with
#: "<svg" to force raw SVG is a 400 on current models.
SCHEMA = {
    "type": "object",
    "properties": {"svg": {"type": "string"}, "viewBox": {"type": "string"}},
    "required": ["svg", "viewBox"],
    "additionalProperties": False,
}

#: A typo'd -O must fail loudly rather than silently change nothing.
ACCEPTED_OPTIONS = frozenset(("palette", "max_tokens", "effort", "timeout",
                              "base_url"))
PROBE_OPTIONS = frozenset(("timeout", "base_url"))

SVG_NS = "http://www.w3.org/2000/svg"

#: `script`, `image` and `foreignObject` are the classic SVG payload carriers.
#: `style` is here for a duller reason: a stylesheet can @import an external URL,
#: and a `fill:#f00` declaration silently defeats the currentColor rewrite that
#: is the entire point of authoring these by hand.
BANNED_TAGS = frozenset(("script", "image", "foreignObject", "style", "iframe",
                         "embed", "object", "audio", "video", "animate",
                         "set", "handler"))

_NUMBER = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
_LOCAL_URL = re.compile(r"url\(\s*['\"]?#")

#: Without this, ET.tostring writes <ns0:svg xmlns:ns0="..."> -- valid XML that
#: browsers will not render when the glyph is inlined into HTML. Registering the
#: SVG namespace as the default prefix mutates a global map inside ElementTree,
#: which is acceptable here only because every SVG tool on earth wants exactly
#: this mapping.
ET.register_namespace("", SVG_NS)

#: A square canvas, a uniform stroke, geometric construction: the same design
#: system vocabulary the raster scaffold carries, in the one place it can be
#: obeyed exactly. Material Design's 24 dp canvas with 2 dp strokes and 2 dp
#: padding is the default because 24 is what the icon ecosystem converged on;
#: IBM Carbon's 32x32 artboard with 2 px padding and 2 px strokes is the
#: alternative, and both numbers are cited in docs/backends.md.
Style = collections.namedtuple("Style", "name canvas stroke padding cap join")

MATERIAL_24 = Style("material-24", 24.0, 2.0, 2.0, "round", "round")
CARBON_32 = Style("carbon-32", 32.0, 2.0, 2.0, "butt", "miter")

STYLES = {MATERIAL_24.name: MATERIAL_24, CARBON_32.name: CARBON_32}
DEFAULT_STYLE = MATERIAL_24


class GlyphRejected(BackendError):
    """Model output that will not be written.

    A BackendError subclass so a caller catching the generation contract's error
    catches this too. It is never repaired: half-valid SVG written into a build
    directory is worse than no SVG.
    """


def author(subjects, style=None, api_key=None, model=MODEL, outdir=None,
           force=False, log=print, **options):
    """Author one SVG glyph per subject. Returns {slug: record}.

    `subjects` is {slug: description}, the same shape iconset.generate takes.
    `outdir` None authors without writing anything, which is how you look before
    you commit; with an outdir, an existing `slug.svg` is kept and no request is
    made for it at all unless force=True. That is the reproducibility guard: with
    no seed and no sampling controls, a re-run would otherwise replace reviewed,
    committed geometry with different geometry for free.

    Each record carries source="hand" plus the digest, model and style, which is
    what the lockfile needs to record a glyph next to generated PNGs and to
    notice a later hand-edit -- the only drift signal available when the
    generator itself cannot be pinned.

    Nothing is caught per glyph. A rejection or a refusal is a thing to look at,
    and glyphs already written stay on disk, so a re-run resumes where it stopped.
    """
    _reject_unknown(options, ACCEPTED_OPTIONS)
    st = resolve_style(style)
    key = _api_key(api_key)
    palette = options.get("palette")
    base_url = options.get("base_url", ENDPOINT)
    system = _system_prompt(st)

    records = collections.OrderedDict()
    total = len(subjects)
    for n, (slug, subject) in enumerate(subjects.items(), 1):
        path = os.path.join(outdir, slug + ".svg") if outdir else None
        if path and os.path.exists(path) and not force:
            with open(path, "r", encoding="utf-8") as fh:
                existing = fh.read()
            log("  [%d/%d] %-20s kept (force=True to re-author)" % (n, total, slug))
            records[slug] = _record(slug, path, existing, st, None, None, True)
            continue

        user = _user_prompt(slug, subject, st)
        payload = {
            "model": model,
            "max_tokens": options.get("max_tokens", MAX_TOKENS),
            "system": system,
            "messages": [{"role": "user", "content": user}],
            # No temperature/top_p/top_k anywhere: a non-default value on these
            # models is a 400, and there is no seed to ask for either.
            "output_config": {
                "effort": options.get("effort", EFFORT),
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
        }
        doc = _post(payload, key, base_url, options.get("timeout", TIMEOUT))
        source, claimed = _extract(doc, slug)
        root = validate(source, st, claimed=claimed)
        svg = recolour(root, palette)

        if path:
            _write(path, svg)
            log("  [%d/%d] %-20s %d bytes" % (n, total, slug, len(svg)))
        records[slug] = _record(slug, path, svg, st, model,
                                _digest(system + user), False)
    return records


def probe(api_key=None, model=MODEL, **options):
    """(ok, message) for credentials and model id. Never authors a glyph.

    Deliberately the cheapest call that still proves both halves: a wrong key is
    a 401 and a retired model id is a 404, and neither is distinguishable from a
    missing environment variable without asking the server. Sixteen output tokens
    on Opus 5 is under a tenth of a cent -- but it is not free, which is why it is
    a separate call you make once rather than something author() does per glyph.
    """
    _reject_unknown(options, PROBE_OPTIONS)
    try:
        key = _api_key(api_key)
    except AuthError as exc:
        return False, str(exc)

    payload = {"model": model, "max_tokens": 16,
               "messages": [{"role": "user", "content": "Reply with: ok"}]}
    try:
        doc = _post(payload, key, options.get("base_url", ENDPOINT),
                    options.get("timeout", 30))
    except BackendError as exc:
        return False, "%s: %s" % (model, exc)
    return True, "%s answered (id %s)" % (model, doc.get("id", "?"))


def resolve_style(style):
    """A Style, from a Style, a name in STYLES, or None for the default."""
    if style is None:
        return DEFAULT_STYLE
    if isinstance(style, Style):
        return style
    resolved = STYLES.get(style) if isinstance(style, str) else None
    if resolved is None:
        raise UnsupportedOption("unknown glyph style %r; known: %s"
                                % (style, ", ".join(sorted(STYLES))))
    return resolved


def validate(source, style=None, claimed=None):
    """Vet untrusted SVG text. Returns the parsed root, or raises GlyphRejected.

    Everything downstream takes the root this returns rather than a string, so
    there is no path that writes a glyph nobody checked.

    The checks are the ones that matter for text a model wrote that is about to
    land in a build directory and, later, in someone's HTML: well-formed XML, an
    svg root, the viewBox this style declares, no active content, and no
    reference that leaves the document.

    `claimed` is the viewBox the model reported alongside the source, which the
    structured-output schema asks for separately; a disagreement between what it
    said and what it drew is a rejection rather than a shrug.
    """
    st = resolve_style(style)
    # ElementTree does not expand external entities, but a DTD has no business in
    # a 1 KB glyph and internal entity expansion is a denial-of-service shape, so
    # this never reaches the parser.
    if "<!DOCTYPE" in source or "<!ENTITY" in source:
        raise GlyphRejected("rejected: a DOCTYPE or ENTITY declaration has no "
                            "place in an icon")
    try:
        root = ET.fromstring(source)
    except ET.ParseError as exc:
        raise GlyphRejected("rejected: not well-formed XML (%s)" % exc) from exc

    if _local(root.tag) != "svg":
        raise GlyphRejected("rejected: root element is %r, not svg"
                            % _local(root.tag))

    box = root.get("viewBox")
    if not box:
        raise GlyphRejected("rejected: no viewBox. Without one the glyph does "
                            "not scale, which is the only reason it is SVG")
    numbers = [float(n) for n in _NUMBER.findall(box)]
    if len(numbers) != 4:
        raise GlyphRejected("rejected: viewBox %r is not four numbers" % box)
    if numbers != [0.0, 0.0, st.canvas, st.canvas]:
        raise GlyphRejected("rejected: viewBox %r is not the %s canvas "
                            "0 0 %g %g" % (box, st.name, st.canvas, st.canvas))
    if claimed is not None and [float(n) for n in _NUMBER.findall(claimed)] != numbers:
        raise GlyphRejected("rejected: the model reported viewBox %r and drew "
                            "on %r" % (claimed, box))

    for el in root.iter():
        if not isinstance(el.tag, str):       # comment or processing instruction
            continue
        tag = _local(el.tag)
        if tag in BANNED_TAGS:
            raise GlyphRejected("rejected: <%s> is not allowed in a glyph" % tag)
        for name, value in el.attrib.items():
            attr = _local(name)
            if attr.startswith("on"):
                raise GlyphRejected("rejected: event handler attribute %r on <%s>"
                                    % (attr, tag))
            if attr == "href" and not value.startswith("#"):
                raise GlyphRejected("rejected: <%s %s=%r> leaves the document"
                                    % (tag, attr, value[:60]))
            if "url(" in value and not _LOCAL_URL.search(value):
                raise GlyphRejected("rejected: external url() in %r on <%s>"
                                    % (attr, tag))
    return root


def recolour(root, palette=None):
    """Serialise a validated root, painting it currentColor or a palette.

    currentColor by default, and not as a preference: it is the one property a
    traced SVG cannot have and the reason to author these by hand at all. The
    rewrite is applied rather than requested, so a model that emitted #FF6A00
    still produces a themeable glyph.

    `palette` may be a single colour for a fixed brand mark, or a sequence that
    is cycled over painted elements in document order -- useful for a two-tone
    set, arbitrary for anything more, since document order is not art direction.
    """
    colours = _colours(palette)
    index = 0
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if not el.tag.startswith("{"):
            # A glyph written without xmlns parses fine and then fails to render
            # as a standalone file; put every element in the SVG namespace.
            el.tag = "{%s}%s" % (SVG_NS, el.tag)
        el.attrib.pop("style", None)
        painted = [a for a in ("fill", "stroke")
                   if el.get(a) not in (None, "none")]
        if not painted:
            continue
        value = colours[index % len(colours)]
        index += 1
        for attr in painted:
            el.set(attr, value)
    return ET.tostring(root, encoding="unicode")


# --- prompts ------------------------------------------------------------

def _system_prompt(style):
    """The house style, stated as constraints rather than adjectives.

    Not cached: prompt caching needs a ~1024-token prefix to do anything, this is
    a few hundred, and a cache_control block that silently never hits is worse
    than none. A 10-15 glyph set is cents either way.
    """
    return (
        "You author SVG icon source by hand, the way a designer would.\n"
        "Canvas: viewBox \"0 0 %g %g\", square, no width or height attributes.\n"
        "Keep every drawn unit inside %g units of padding on all four sides.\n"
        "Construction is geometric: straight lines, arcs of constant radius, "
        "angles at 45 or 90 degrees wherever the symbol allows. Optical "
        "centring beats mathematical centring.\n"
        "Strokes are uniform: stroke-width %g, stroke-linecap %s, "
        "stroke-linejoin %s, fill none on any stroked shape.\n"
        "Colour: stroke and fill are the literal string currentColor. Never a "
        "hex value, never a named colour. The glyph inherits the CSS colour of "
        "whatever contains it.\n"
        "Forbidden: script, image, foreignObject, style elements, external "
        "references, href to anything but a same-document fragment, event "
        "handler attributes, gradients, filters, drop shadows, and text or "
        "letterforms of any kind.\n"
        "Prefer few paths. One path is ideal.\n"
        "Return the SVG document in the svg field and the canvas you actually "
        "used in the viewBox field."
        % (style.canvas, style.canvas, style.padding, style.stroke,
           style.cap, style.join)
    )


def _user_prompt(slug, subject, style):
    return ("Draw the %s glyph: %s.\n"
            "It sits beside other %s glyphs in one set, so it must read at "
            "16 px and share their stroke weight."
            % (slug, subject, style.name))


# --- transport ----------------------------------------------------------

def _post(payload, api_key, base_url, timeout):
    """One Messages round trip, with the provider cases _http cannot know about.

    _http already classifies 401/403, 402, 429 and the retryable 5xx range.
    Anthropic adds two on top: 529 overloaded_error, which is transient but sits
    outside _http's retry table, and the family of 400s that mean this code sent
    a parameter these models removed.
    """
    headers = {"x-api-key": api_key, "anthropic-version": API_VERSION}
    try:
        return _http.request_json(base_url, payload=payload, headers=headers,
                                  timeout=timeout)
    except BackendError as exc:
        refined = _refine(exc)
        if refined is exc:
            raise
        raise refined from exc


def _refine(exc):
    message = str(exc)
    if "HTTP 529" in message or "overloaded_error" in message:
        return RateLimited(message + "\n  overloaded_error is transient, but "
                                     "529 is outside _http's retry table")
    if "HTTP 400" in message:
        for removed in ("temperature", "top_p", "top_k", "budget_tokens"):
            if removed in message:
                return BackendError(
                    message + "\n  %s was removed on these models; this module "
                              "must not be sending it" % removed)
    return exc


def _extract(doc, slug):
    """(svg source, claimed viewBox) out of a Messages response."""
    stop = doc.get("stop_reason")
    if stop == "refusal":
        details = doc.get("stop_details") or {}
        raise ModerationBlocked(
            "Claude declined to author %r (%s): %s"
            % (slug, details.get("category"), details.get("explanation")))
    if stop == "max_tokens":
        raise GlyphRejected("%s: the response hit max_tokens, so the JSON is "
                            "truncated; raise max_tokens" % slug)

    # Thinking is on by default on this model, so the response carries thinking
    # blocks ahead of the answer. Only text blocks hold the structured output.
    text = "".join(block.get("text") or "" for block in doc.get("content") or []
                   if block.get("type") == "text")
    if not text.strip():
        raise GlyphRejected("%s: no text block in the Messages response" % slug)
    try:
        parsed = json.loads(text)
    except ValueError:
        raise GlyphRejected("%s: structured output was not JSON: %s"
                            % (slug, text[:200]))
    source = parsed.get("svg")
    if not source:
        raise GlyphRejected("%s: no svg field in %s" % (slug, sorted(parsed)))
    return source, parsed.get("viewBox")


# --- odds and ends ------------------------------------------------------

def _api_key(api_key):
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        raise AuthError(
            "%s is not set.\n  This module speaks raw HTTP and only understands "
            "x-api-key, so an `ant auth login` profile will not be picked up."
            % API_KEY_ENV)
    return key


def _reject_unknown(options, accepted):
    unknown = sorted(set(options) - accepted)
    if unknown:
        raise UnsupportedOption("glyphs does not accept %s; known options: %s"
                                % (", ".join(unknown), ", ".join(sorted(accepted))))


def _colours(palette):
    if palette is None:
        return ["currentColor"]
    if isinstance(palette, str):
        return [palette]
    colours = list(palette)
    if not colours:
        raise UnsupportedOption("palette is empty; omit it for currentColor")
    return colours


def _write(path, svg):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # These get committed, and .gitattributes normalises the repo to LF; writing
    # CRLF on Windows would show every glyph as modified on the next checkout.
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg.rstrip("\n") + "\n")


def _record(slug, path, svg, style, model, prompt_digest, kept):
    return {
        "slug": slug,
        "source": "hand",          # what the lockfile records next to generated PNGs
        "path": path,
        "svg": svg,
        "sha256": _digest(svg),    # the only drift signal available without a seed
        "bytes": len(svg.encode("utf-8")),
        "model": model,            # None when the committed file was kept
        "style": style.name,
        "view_box": "0 0 %g %g" % (style.canvas, style.canvas),
        "prompt_sha256": prompt_digest,
        "reproducible": False,
        "kept": kept,
    }


def _digest(text):
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _local(tag):
    return tag.rsplit("}", 1)[-1]
