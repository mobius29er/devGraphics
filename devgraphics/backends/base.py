"""
The contract every generator backend implements, and the loader that finds one.

Three decisions carry this module, and each is forced by something concrete.

**Backends return PNG bytes, not paths.** Fooocus hands back file paths on the
Fooocus host and fetches them over a second request; OpenAI hands back base64 in
the response body and has no filesystem to point at. Hoisting `download()` into
the contract would force every hosted backend to invent a path handle, so the
contract is bytes and path-fetching stays a local backend's private business.

**Capabilities are per instance, not per class.** This is not theoretical:
`gpt-image-1` honours a transparent background and `gpt-image-2` rejects it from
the same client class; ComfyUI's native alpha depends on whether a 444 MB matting
model is installed on *that* server. Capabilities must also be answerable with the
server switched off, so constructors must not touch the network -- otherwise
`--dry-run` needs the very thing it exists to avoid.

**Unsupported options are reported, never dropped silently.** A seed handed to a
backend that has none is the one failure that destroys this tool's only promise:
without a shared seed every icon is an independent draw and the set reads as AI
slop. Everything else genuinely degrades -- postprocess.cutout() already stands in
for the alpha channel SDXL never had.

Backends are named by string and imported on demand, so choosing `fooocus` never
imports an HTTP client for OpenAI and choosing `openai` never imports
websocket-client.

Typing note: the rest of devgraphics carries no annotations. This module does,
because it *is* the contract other people write against. The departure stops here.
"""

import importlib
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

ENTRY_POINT_GROUP = "devgraphics.backends"

#: name -> "module:Class". Values are strings so importing this module imports no
#: backend, and therefore no backend's transport or SDK.
BUILTIN: Dict[str, str] = {
    "fooocus":           "devgraphics.backends.fooocus:FooocusBackend",
    "comfyui":           "devgraphics.backends.comfyui:ComfyUIBackend",
    "invokeai":          "devgraphics.backends.invokeai:InvokeAIBackend",
    "a1111":             "devgraphics.backends.a1111:A1111Backend",
    "openai":            "devgraphics.backends.openai_images:OpenAIBackend",
    "gemini":            "devgraphics.backends.gemini:GeminiBackend",
    "openai-compatible": "devgraphics.backends.openai_compat:OpenAICompatBackend",
}

#: The profile says `model`; backends disagree about what to call it. This is the
#: one cross-cutting concept a user names in config and every backend spells
#: differently, so the mapping lives here rather than in seven `if` statements in
#: iconset. Unlisted backends -- including third-party ones -- get "model".
#: None means the backend has no such knob: a Fooocus checkpoint is chosen in the
#: Fooocus UI, not over the wire.
MODEL_OPTION: Dict[str, Optional[str]] = {
    "fooocus": None,
    "comfyui": "checkpoint",
    "a1111": "checkpoint",
}

#: Presets for `openai-compatible`, so nobody has to remember base URLs. These are
#: OpenAI-shaped /v1/images/generations endpoints, not separate modules; xAI in
#: particular retired two image model ids in six months, which is exactly why it
#: does not get a module of its own.
COMPAT_PRESETS: Dict[str, Dict[str, str]] = {
    "grok":      {"base_url": "https://api.x.ai/v1",
                  "api_key_env": "XAI_API_KEY"},
    "together":  {"base_url": "https://api.together.xyz/v1",
                  "api_key_env": "TOGETHER_API_KEY"},
    "deepinfra": {"base_url": "https://api.deepinfra.com/v1/openai",
                  "api_key_env": "DEEPINFRA_API_KEY"},
}


class BackendError(RuntimeError):
    """Generation failed. Backends raise this or a subclass."""


class AuthError(BackendError):
    """Missing or rejected credentials. Retrying will not help."""


class RateLimited(BackendError):
    """429. Carries the server's Retry-After when it gave one."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


class ModerationBlocked(BackendError):
    """The provider refused the prompt. Non-retryable -- change the request."""


class PaymentRequired(BackendError):
    """Out of credit. Fatal for the whole batch, not just this icon."""


class BackendNotFound(LookupError):
    """No backend is registered under that name."""


class UnsupportedOption(ValueError):
    """A backend was handed an option key it does not recognise.

    Backends raise this rather than ignoring unknown `Request.options` keys: a
    typo'd `-O` must fail loudly, not vanish and quietly change nothing.
    """


class MissingDependency(BackendError):
    """An optional dependency for this backend is not installed."""

    def __init__(self, backend, package, extra=None):
        hint = "pip install %s" % package
        if extra:
            hint = "pip install devgraphics[%s]   (or: %s)" % (extra, hint)
        super().__init__("the %s backend needs %s, which is not installed.\n  %s"
                         % (backend, package, hint))


@dataclass(frozen=True)
class Request:
    """One generation, in backend-neutral terms.

    A frozen dataclass rather than **kwargs, so `preflight` can see which knobs
    the caller actually set and so adding a field later does not break every
    backend's signature.

    `size` is (width, height) integers, never a string. Fooocus matches
    "1024x1024" with a U+00D7 multiplication sign buried in an HTML-laden choice
    label; OpenAI wants ASCII "1024x1024"; Gemini wants an aspect-ratio enum plus
    a size tier; ComfyUI wants two integers. Formatting is the backend's job.

    `refs` is bytes, symmetric with the return type: ComfyUI POSTs them to
    /upload/image, OpenAI base64s them into a data URL, Gemini inlines them.
    Nobody downstream wants a path.

    `options` carries backend-specific knobs -- Fooocus style names, a ComfyUI
    checkpoint filename. Backends must reject keys they do not recognise.
    """

    prompt: str
    negative: str = ""
    seed: Optional[int] = None
    size: Tuple[int, int] = (1024, 1024)
    count: int = 1
    transparent: bool = False
    refs: Tuple[bytes, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)

    #: The only fields a waiver may strip, and the value they revert to. `prompt`,
    #: `size` and `options` are deliberately absent: there is no neutral value for
    #: them, so no bug in the waiver path can ever blank them.
    _WAIVABLE = {"seed": None, "negative": "", "transparent": False,
                 "count": 1, "refs": ()}

    def without(self, *names):
        """A copy with each named option reverted to its neutral value."""
        return replace(self, **{n: self._WAIVABLE[n] for n in names
                                if n in self._WAIVABLE})


@dataclass(frozen=True)
class Capabilities:
    """What one *configured* backend instance can honour.

    `seed` and `deterministic` are separate promises. Stability accepts a seed and
    echoes back the one it used; Gemini exposes a seed field but no Google
    document claims a fixed seed reproduces an image, and its image models run a
    mandatory thinking pass. That is `seed=False` here rather than `seed=True`
    with a caveat, because a consistency strategy must not be built on it.

    `reference_images` is a count, not a flag: OpenAI's edits endpoint takes 16,
    one Gemini model takes 3 dedicated style references, xAI takes a source image
    and has no style-reference concept at all.

    `sizes` empty means "any (w, h)". Otherwise it lists the exact pairs the
    backend can produce; anything else is a waiver, the backend uses its nearest,
    and postprocess.render()'s LANCZOS downscale is the final normaliser.
    """

    name: str
    seed: bool = False
    deterministic: bool = False
    negative_prompt: bool = False
    transparent: bool = False
    reference_images: int = 0
    batch: bool = False                     # count > 1 honoured in one call
    sizes: Tuple[Tuple[int, int], ...] = ()
    cost_per_image: Optional[float] = None  # USD; None means local and free
    notes: Tuple[str, ...] = ()


class Backend(Protocol):
    """The structural contract. Nothing has to import devgraphics to satisfy it.

    Two conventions Protocol cannot express, so they are written here instead:

    Constructor: ``Backend(**options)``, where options come from a config
    profile's ``[options]`` table or from ``-O key=value``. Constructors must be
    cheap and must not touch the network, because ``capabilities`` has to be
    answerable with the server switched off or ``--dry-run`` is useless.

    Optional liveness check, discovered with ``hasattr`` so it stays optional::

        @classmethod
        def probe(cls, **options) -> Tuple[bool, str]

    It reports reachability and credential presence, and must never generate an
    image -- a paid backend whose probe costs money is a trap.
    """

    @property
    def capabilities(self) -> Capabilities:
        ...

    def generate(self, request: Request) -> Sequence[bytes]:
        """Render and return up to `request.count` images as PNG bytes.

        PNG specifically: postprocess.cutout() flood-fills against the backdrop,
        and JPEG ringing around a hard outline is exactly the signal thresh=42
        keys on. A provider that returns JPEG or WebP must transcode first --
        postprocess.to_png() is there for it.

        Raise BackendError (or a subclass) on failure, and UnsupportedOption on
        an option key the backend does not recognise.
        """
        ...


@dataclass(frozen=True)
class Waiver:
    """One requested thing the chosen backend cannot do."""

    option: str
    requested: Any
    reason: str
    fatal: bool = False

    def __str__(self):
        return "%s=%r: %s" % (self.option, self.requested, self.reason)


def nearest_size(caps, size):
    """The offered size closest to `size`, or `size` itself if any is allowed.

    Aspect ratio dominates area: a 1:1 icon squeezed into 1536x1024 is wrong in a
    way that a 1024x1024 icon downscaled to 128 is not.
    """
    if not caps.sizes:
        return tuple(size)
    want = size[0] * size[1]
    aspect = size[0] / float(size[1])
    return min(caps.sizes,
               key=lambda s: (round(abs(s[0] / float(s[1]) - aspect), 4),
                              abs(s[0] * s[1] - want)))


def preflight(caps, request, strict=True):
    """Diff a request against a backend's declared capabilities.

    Exactly one waiver is fatal: a seed the backend has no parameter for. That is
    the one case where the promise cannot be kept at all. Callers pass
    `strict=False` for a single one-off render, or when a reference anchor is
    carrying consistency instead -- see `iconset`.

    Nothing here mutates the request; the caller decides whether to stop or strip.
    """
    out = []
    if request.seed is not None and not caps.seed:
        out.append(Waiver(
            "seed", request.seed,
            "dropped -- %s has no seed parameter, so every icon becomes an "
            "independent draw" % caps.name, fatal=strict))
    if request.negative and not caps.negative_prompt:
        out.append(Waiver(
            "negative", _clip(request.negative),
            "dropped -- %s has no negative prompt; fold these exclusions into "
            "the prompt as affirmative phrasing" % caps.name))
    if request.transparent and not caps.transparent:
        out.append(Waiver(
            "transparent", True,
            "%s returns opaque images; postprocess.cutout will key the backdrop "
            "out instead" % caps.name))
    if request.refs and not caps.reference_images:
        out.append(Waiver(
            "refs", len(request.refs),
            "%s takes no reference image, so anchor-based style locking is "
            "unavailable; prompt scaffold only" % caps.name))
    elif len(request.refs) > caps.reference_images:
        out.append(Waiver(
            "refs", len(request.refs),
            "%s accepts at most %d reference image(s)"
            % (caps.name, caps.reference_images)))
    if request.count > 1 and not caps.batch:
        out.append(Waiver(
            "count", request.count,
            "%s generates one image per call; devgraphics will loop" % caps.name))
    if caps.sizes and tuple(request.size) not in caps.sizes:
        near = nearest_size(caps, request.size)
        out.append(Waiver(
            "size", tuple(request.size),
            "%s only offers %s; %dx%d is used and the icon is resized on the way "
            "out" % (caps.name, ", ".join("%dx%d" % s for s in caps.sizes),
                     near[0], near[1])))
    return tuple(out)


def strip(request, waivers):
    """Blank every waived option, so a backend never receives a value it has
    already declared it cannot honour.

    `count` and `size` are deliberately not blanked: looping for count and
    resizing for size are real fallbacks the caller performs, not losses.
    """
    return request.without(*[w.option for w in waivers
                             if w.option in ("seed", "negative", "transparent",
                                             "refs")])


def report(caps, waivers, request):
    """One block, printed once before the batch rather than 88 times during it.

    ASCII only and no colour: this lands in a Windows console as often as a
    UTF-8 terminal.
    """
    lines = ["backend: %s" % caps.name]
    for w in waivers:
        lines.append("  %-5s %s" % ("ERROR" if w.fatal else "warn", w))
    if request.seed is not None and caps.seed and not caps.deterministic:
        lines.append("  note  seed %d is sent, but %s does not guarantee "
                     "identical pixels across hosts or versions"
                     % (request.seed, caps.name))
    for n in caps.notes:
        lines.append("  note  %s" % n)
    if any(w.fatal for w in waivers):
        lines += ["",
                  "  A set generated without a fixed seed will not read as a set.",
                  "  Use --anchor SLUG (with -n) to select against a reference",
                  "  icon, or re-run with --allow-drift."]
    elif len(lines) == 1:
        lines.append("  everything requested is supported")
    return "\n".join(lines)


def _clip(text, width=40):
    return text if len(text) <= width else text[:width - 3] + "..."


# --- loading ------------------------------------------------------------

def load(spec, **options):
    """Resolve `spec`, construct it, and check it. Tried in order:

        "mypkg.thing:MyBackend"   any importable class, no packaging needed
        "fooocus"                 a name in BUILTIN
        "whatever"                a devgraphics.backends entry point

    Dotted paths are tried first because they are the case that actually turns up
    -- someone with a bespoke ComfyUI graph pointing at their own class. Entry
    points come last because resolving them walks every installed distribution's
    metadata.
    """
    if ":" in spec:
        target = spec
    elif spec in BUILTIN:
        target = BUILTIN[spec]
    else:
        target = _entry_point_target(spec)

    module_name, _, attr = target.partition(":")
    if not attr:
        raise ValueError("backend target %r must be module:Class" % target)
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, attr)
    except AttributeError as exc:
        raise BackendNotFound("%s has no attribute %r"
                              % (module_name, attr)) from exc
    try:
        backend = cls(**options)
    except TypeError as exc:
        raise TypeError("backend %r rejected options %s: %s"
                        % (spec, sorted(options), exc)) from exc
    check(backend, spec)
    return backend


def check(backend, spec="<backend>"):
    """Fail at load time rather than forty icons into a batch.

    Deliberately not an isinstance() test against a runtime_checkable Protocol:
    that verifies attribute *names* only, never signatures, so a class whose
    generate() took (prompt, negative) would pass and then explode on icon 1 of
    88. On 3.12+ such a check also uses inspect.getattr_static, which finds a
    `capabilities` property as a class descriptor and never evaluates it.
    """
    if not callable(getattr(backend, "generate", None)):
        raise TypeError("backend %r has no callable generate(request)" % spec)
    caps = getattr(backend, "capabilities", None)   # instance getattr, so a
    if not isinstance(caps, Capabilities):          # @property actually runs
        raise TypeError("backend %r must expose a Capabilities as .capabilities, "
                        "got %r" % (spec, type(caps).__name__))


def available():
    """Every backend name we know about: built in first, then third-party."""
    names = list(BUILTIN)
    for name, _ in _entry_points():
        if name not in names:
            names.append(name)
    return names


def _entry_points():
    """(name, value) pairs, across the 3.9 -> 3.10 entry_points() change.

    The selectable API (`entry_points(group=...)`) landed in 3.10; on 3.9 the call
    returns a dict keyed by group. Third-party backends are the whole point of the
    group, so this has to work on the floor version.
    """
    from importlib.metadata import entry_points
    try:
        if sys.version_info >= (3, 10):
            found = entry_points(group=ENTRY_POINT_GROUP)
        else:                                            # pragma: no cover
            found = entry_points().get(ENTRY_POINT_GROUP, [])
        return [(ep.name, ep.value) for ep in found]
    except Exception:                                    # pragma: no cover
        return []


def _entry_point_target(name):
    for ep_name, value in _entry_points():
        if ep_name == name:
            return value                                 # already "module:Class"
    raise BackendNotFound(
        "no backend named %r; known: %s. A dotted module:Class path also works."
        % (name, ", ".join(available())))
