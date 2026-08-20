"""
Project config: devgraphics.toml, or [tool.devgraphics] in pyproject.toml.

TOML rather than JSON, because this is the one file a human hand-edits and half
of what it holds is only safe to change if you know what was measured. `styles =
["Fooocus V2", "Fooocus Sharp"]` looks arbitrary until you read that `Sticker
Designs` and `Simple Vector Art` -- the two whose names promise flat vector
output -- lay down an asphalt texture that defeats the flood fill
(docs/findings.md). A comment on the line above the value is the only place that
warning can live, and JSON has no comments. It also has no multi-line strings,
and the prompt scaffold is forty words. The manifest stays JSON (flat, usually
machine-generated); the lockfile is written as JSON because tomllib is read-only
and nothing in the stdlib writes TOML.

Keys are never values. `validate` rejects anything key-shaped anywhere in the
document rather than trusting a schema to keep secrets out: a config that
*permits* `api_key = "sk-..."` eventually has one committed. The config names
environment variables; it never holds their contents.

`options` does not survive a backend change. A naive deep merge across `extends`
leaked Fooocus' `host` and `performance` into an OpenAI profile in the
prototype -- ComfyUI has no `performance`, OpenAI has no `host` -- so `_chain`
and `resolve` both drop the table whenever `backend` moves.

The selector key is `default_profile`, not `profile`: a top-level `profile =
"brand-icons"` collides with the [profile.brand-icons] table and tomllib rejects
the whole file with "Cannot overwrite a value".
"""

import copy
import difflib
import hashlib
import json
import os
import re

try:
    import tomllib
except ModuleNotFoundError:          # 3.9 / 3.10; pyproject declares the backport
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

#: Tried in this order in each directory on the way up, then pyproject.toml.
NAMES = (".devgraphics.toml", "devgraphics.toml")
PYPROJECT = "pyproject.toml"
TOOL_TABLE = "devgraphics"
CONFIG_VERSION = 1

# The measured defaults, same text as iconset's constants. One shared scaffold,
# one shared style list and one fixed seed across subjects is the entire
# consistency story: SDXL keeps a recognisable look across different subjects at
# a shared seed. Only {subject} is substituted -- bg_hex and palette are separate
# keys because postprocess and the contact sheet need them as values, not prose.
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

DEFAULTS = {
    "backend": "fooocus",
    "model": None,
    "seed": 77_777,
    "render": (1024, 1024),     # generation size; the tracer wants full res
    "bg_hex": "#0D0D0D",        # the background the finished set sits on
    "palette": [],
    "scaffold": SCAFFOLD,
    "negative": NEGATIVE,
    "anchor": None,             # slug rendered first, then used as a style ref
    "n": 1,                     # best-of-n; the only seed substitute on a
                                # seedless API, and it multiplies hosted cost
    "options": {},              # backend-specific; never inherited across backends
    "output": {"size": 128, "svg": None, "sheet": False},
    "postprocess": {"thresh": 42, "despeckle": True, "keep_frac": 0.15,
                    "pad_ratio": 0.06, "snap_palette": False},
}

TOP_KEYS = frozenset({"config_version", "manifest", "outdir", "default_profile",
                      "profile", "backend", "price"})
PROFILE_KEYS = frozenset(DEFAULTS) | {"extends"}
SUB_TABLES = ("options", "output", "postprocess")

#: Keys people reasonably write at profile level that belong in a sub-table.
#: Worth the dict: "unknown key 'styles'" is a far worse error than "it goes in
#: [profile.X.options]", and `styles` in particular is a Fooocus option with no
#: analogue on any other backend, so it must not look portable.
MOVED = {
    "styles": "options", "host": "options", "endpoint": "options",
    "performance": "options", "sharpness": "options", "guidance": "options",
    "quality": "options", "workflow": "options", "background": "options",
    "size": "output", "svg": "output", "sheet": "output",
    "thresh": "postprocess", "despeckle": "postprocess",
    "keep_frac": "postprocess", "pad_ratio": "postprocess",
    "snap_palette": "postprocess",
}

#: A key under any of these names is a leaked secret whatever its value is.
SECRET_NAMES = frozenset({"api_key", "apikey", "api-key", "key", "token",
                          "api_token", "access_token", "secret", "secret_key",
                          "password", "auth", "authorization", "bearer"})

#: (prefix, what it is). Prefixes only: a Together key is 64 hex characters with
#: no prefix and is indistinguishable from a checkpoint hash, so a value like
#: that is caught by the name rule above or not at all. Stated rather than
#: pretending the scan is exhaustive.
SECRET_SHAPES = (("sk-ant-", "an Anthropic key"),
                 ("sk-", "an OpenAI or Stability key"),
                 ("xai-", "an xAI key"),
                 ("AIza", "a Google API key"),
                 ("hf_", "a Hugging Face token"),
                 ("r8_", "a Replicate token"))
SECRET_MIN = 20

#: A multiplication sign is accepted alongside "x" because Fooocus' own
#: aspect-ratio labels spell the size with U+00D7 and people paste them straight
#: in. Spelled with chr() rather than inline, so this file stays ASCII.
SIZE_RE = re.compile(r"^\s*(\d+)\s*[xX" + chr(0x00D7) + r"]\s*(\d+)\s*$")
HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


class ConfigError(ValueError):
    """The config is wrong in a way that has to stop the run."""


# --- reading ------------------------------------------------------------

def load(path):
    """Parse one file. pyproject.toml means its [tool.devgraphics] table."""
    if tomllib is None:                                   # pragma: no cover
        raise ConfigError(
            "reading %s needs a TOML parser. Python 3.11+ has tomllib in the "
            "stdlib; on 3.9 and 3.10 run `pip install tomli` -- devgraphics "
            "declares it as a conditional dependency, so `pip install -U "
            "devgraphics` fixes this too." % path)
    with open(path, "rb") as f:
        try:
            doc = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError("%s is not valid TOML: %s" % (path, exc)) from exc
    if os.path.basename(path).lower() == PYPROJECT:
        doc = doc.get("tool", {}).get(TOOL_TABLE) or {}
    return doc


def find(start=None):
    """Nearest config walking up from `start`. Returns (path, table) or (None, {}).

    pyproject.toml is checked last in each directory, and only counts when it
    actually carries a [tool.devgraphics] table: most projects that consume an
    icon set are a JS site, and a pyproject.toml that happens to be lying around
    has nothing to do with this tool.
    """
    d = os.path.abspath(start or os.getcwd())
    while True:
        for name in NAMES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p, load(p)
        p = os.path.join(d, PYPROJECT)
        if os.path.isfile(p):
            table = load(p)
            if table:
                return p, table
        parent = os.path.dirname(d)
        if parent == d:
            return None, {}
        d = parent


def discover(explicit=None, start=None):
    """The front door: (path, validated config). `explicit` is --config PATH."""
    if explicit:
        if not os.path.isfile(explicit):
            raise ConfigError("no config file at %s" % explicit)
        path, cfg = explicit, load(explicit)
    else:
        path, cfg = find(start)
    validate(cfg, path)
    return path, cfg


def profile_names(cfg):
    return sorted(cfg.get("profile") or {})


def prices(cfg):
    """The [price."backend:model"] overrides, ready for pricing.estimate."""
    return cfg.get("price") or {}


# --- validation ---------------------------------------------------------

def validate(cfg, path=None):
    """Reject what would otherwise fail silently or leak. Returns `cfg`."""
    where = path or "config"
    if not isinstance(cfg, dict):
        raise ConfigError("%s: the top level must be a table" % where)

    for key in cfg:
        if key not in TOP_KEYS:
            raise ConfigError("%s: unknown top-level key %r%s"
                              % (where, key, _suggest(key, TOP_KEYS)))

    version = cfg.get("config_version")
    if version is not None and (not isinstance(version, int)
                                or version > CONFIG_VERSION):
        raise ConfigError("%s: config_version %r, but this devgraphics "
                          "understands %d" % (where, version, CONFIG_VERSION))

    _scan_secrets(cfg, where)

    for name, prof in (cfg.get("profile") or {}).items():
        if not isinstance(prof, dict):
            raise ConfigError("%s: [profile.%s] must be a table" % (where, name))
        _check_profile(prof, name, where)

    for key, table in (cfg.get("price") or {}).items():
        if ":" not in key:
            raise ConfigError('%s: [price."%s"] must be keyed "backend:model", '
                              'e.g. [price."openai:gpt-image-1.5"]' % (where, key))
        if not isinstance(table, dict) or not isinstance(table.get("per_image"),
                                                         (int, float)):
            raise ConfigError('%s: [price."%s"] needs a numeric per_image'
                              % (where, key))
    return cfg


def _check_profile(prof, name, where):
    for key in prof:
        if key in PROFILE_KEYS:
            continue
        if key in MOVED:
            raise ConfigError("%s: %r does not belong at profile level; move it "
                              "into [profile.%s.%s]"
                              % (where, key, name, MOVED[key]))
        raise ConfigError("%s: unknown key %r in [profile.%s]%s"
                          % (where, key, name, _suggest(key, PROFILE_KEYS)))
    for sub in SUB_TABLES:
        if sub in prof and not isinstance(prof[sub], dict):
            raise ConfigError("%s: [profile.%s.%s] must be a table"
                              % (where, name, sub))


def _scan_secrets(node, where, trail=""):
    """Walk the whole document. A key in the config is an error, never a warning:
    the file is committed, and a warning is something you scroll past once."""
    if isinstance(node, dict):
        for key, value in node.items():
            here = "%s.%s" % (trail, key) if trail else str(key)
            if str(key).lower() in SECRET_NAMES:
                _refuse(where, here, "a credential")
            _scan_secrets(value, where, here)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _scan_secrets(value, where, "%s[%d]" % (trail, i))
    elif isinstance(node, str) and len(node) >= SECRET_MIN:
        for prefix, what in SECRET_SHAPES:
            if node.startswith(prefix):
                _refuse(where, trail, what)


def _refuse(where, trail, what):
    # The value is never echoed back: this text lands in CI logs and pasted issues.
    raise ConfigError(
        "%s: %s looks like %s.\n"
        "  devgraphics never reads a key out of the config. The config names\n"
        "  environment variables; it does not hold their contents.\n"
        "  Delete it, rotate the key (assume it is already in git history), and\n"
        "  write instead:\n"
        "      [backend.openai]\n"
        '      api_key_env = "OPENAI_API_KEY"\n'
        "  then export that variable, or pass --env-file." % (where, trail, what))


def _suggest(name, known):
    close = difflib.get_close_matches(str(name), sorted(known), n=1)
    if close:
        return "; did you mean %r?" % close[0]
    return "; known: %s" % ", ".join(sorted(known))


# --- resolution ---------------------------------------------------------

def merge(base, over):
    """Recursive table merge. Lists replace wholesale -- a palette is one value,
    not a set of items to accumulate."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve(cfg, name=None, overrides=None):
    """Defaults < profile (following `extends`) < CLI overrides.

    `overrides` comes from argparse, where an unset flag is None, so None values
    are dropped rather than merged; otherwise every flag the user did not pass
    would blank a profile value. A flag whose home is a sub-table (--size,
    --svg, --sheet, --host) may be passed flat and is routed there. Anything
    else raises, because an override the resolver quietly ignored would also
    quietly change the digest and report drift on the next run.
    """
    name = name or cfg.get("default_profile")
    prof = _chain(cfg, name) if name else {}
    _check_profile(prof, name or "<default>", "config")

    resolved = merge(DEFAULTS, prof)
    over = _route(_prune(overrides or {}))
    if over.get("backend") and over["backend"] != resolved.get("backend"):
        # Options AND model: "juggernautXL_v8Rundiffusion" is as meaningless to
        # OpenAI as "host" is to it, so --backend drops both rather than handing
        # the new backend a checkpoint name it will reject or, worse, accept.
        resolved = dict(resolved, options={}, model=None)
    resolved = copy.deepcopy(merge(resolved, over))

    resolved["render"] = _size(resolved.get("render"))
    resolved["seed"] = _int_or_none(resolved.get("seed"), "seed")
    resolved["n"] = max(1, _int_or_none(resolved.get("n"), "n") or 1)
    if resolved.get("bg_hex"):
        resolved["bg_hex"] = _hex(resolved["bg_hex"], "bg_hex")
    resolved["palette"] = [_hex(c, "palette")
                           for c in (resolved.get("palette") or [])]
    return resolved


def _chain(cfg, name, seen=()):
    profiles = cfg.get("profile") or {}
    if name in seen:
        raise ConfigError("profile %r extends a cycle: %s -> %s"
                          % (seen[0], " -> ".join(seen), name))
    if name not in profiles:
        raise ConfigError("no [profile.%s] in the config%s"
                          % (name, _suggest(name, profiles) if profiles
                             else "; the config defines no profiles"))
    prof = dict(profiles[name])
    parent = prof.pop("extends", None)
    if parent is None:
        return prof
    base = _chain(cfg, parent, seen + (name,))
    if "backend" in prof and prof["backend"] != base.get("backend"):
        # A child that switches backends inherits the look, not the plumbing --
        # and a checkpoint name is plumbing, so `model` goes with `options`.
        base = dict(base, options={}, model=None)
    return merge(base, prof)


def _route(over):
    """Sort CLI overrides into profile keys and sub-table keys.

    `--size 128` is [output], `--host` is [options]: the CLI can hand the whole
    argparse namespace over flat instead of knowing which table each flag lives
    in. An unrecognised key is an error rather than a no-op, since it would
    otherwise ride into the profile, into the digest, and out again as spurious
    drift.
    """
    out, moved = {}, {}
    for key, value in over.items():
        if key in PROFILE_KEYS:
            out[key] = value
        elif key in MOVED:
            moved.setdefault(MOVED[key], {})[key] = value
        else:
            extra = ("; %s is a top-level config key, not a profile key" % key
                     if key in TOP_KEYS else _suggest(key, PROFILE_KEYS))
            raise ConfigError("unknown profile override %r%s" % (key, extra))
    for table, values in moved.items():
        out[table] = merge(out.get(table) or {}, values)
    return out


def _prune(d):
    """Drop None values, recursively: argparse hands us one per unset flag."""
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        out[k] = _prune(v) if isinstance(v, dict) else v
    return out


def _size(value):
    if isinstance(value, bool):
        raise ConfigError('render must be "WxH" or [W, H]; got %r' % (value,))
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, str):
        m = SIZE_RE.match(value)
        if not m:
            raise ConfigError('render must be "WxH", e.g. "1024x1024"; got %r'
                              % value)
        pair = (int(m.group(1)), int(m.group(2)))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            pair = (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            raise ConfigError("render must be two integers; got %r" % (value,))
    else:
        raise ConfigError('render must be "WxH" or [W, H]; got %r' % (value,))
    if pair[0] <= 0 or pair[1] <= 0:
        raise ConfigError("render must be positive; got %dx%d" % pair)
    return pair


def _int_or_none(value, what):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError("%s must be an integer; got %r" % (what, value))
    try:
        return int(value)
    except ValueError:
        raise ConfigError("%s must be an integer; got %r" % (what, value))


def _hex(value, what):
    text = str(value).strip()
    if not HEX_RE.match(text):
        raise ConfigError('%s must be a hex colour like "#0D0D0D"; got %r'
                          % (what, value))
    return text.upper()


# --- hashing ------------------------------------------------------------

def to_dict(profile):
    """A JSON-able copy for the lockfile. `render` goes back to "WxH", so the
    lock reads the way the config reads and diffs the same."""
    out = {}
    for key, value in profile.items():
        out[key] = "%dx%d" % tuple(value) if key == "render" else _plain(value)
    return out


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def digest(profile):
    """One value that moves whenever anything affecting the look moves.

    sort_keys, so two configs differing only in the order their keys were typed
    hash identically -- otherwise reformatting the TOML would read as drift.
    """
    blob = json.dumps(to_dict(profile), sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


# --- init ---------------------------------------------------------------

def starter_toml(backend="fooocus"):
    """The file `devgraphics init` writes.

    The comments are the point. This is the one file a human edits, and the
    measured traps -- which Fooocus styles defeat the flood fill, which colour
    the cutout keys out, which OpenAI model dropped transparency -- are only
    findable when they sit on the line above the value they explain. That is the
    reason the config is TOML and the manifest is not.
    """
    lines = [
        "# devgraphics.toml -- commit this. It contains no secrets, ever.",
        "config_version = 1",
        "",
        'manifest        = "icons/manifest.json"',
        'outdir          = "assets"',
        "# NOT `profile` -- that key collides with the [profile.*] tables below",
        "# and TOML rejects the whole file.",
        'default_profile = "brand-icons"',
        "",
        "# --- how the set looks. Once a set ships, anything changed here makes",
        "# the next icon stop matching the last one. The lockfile says so.",
        "[profile.brand-icons]",
        'backend  = "%s"' % backend,
    ]
    lines += _STARTER_MODEL.get(backend, ["seed     = 77777",
                                          'render   = "1024x1024"'])
    lines += [
        "",
        "# postprocess.cutout() keys the generated backdrop out; the contact",
        "# sheet then composites the icons onto this colour, so drift shows up",
        "# against the real background rather than against white.",
        'bg_hex   = "#0D0D0D"',
        'palette  = ["#FF7A28", "#FFC400", "#E8483A", "#FFF5DC"]',
        "",
        "# Only {subject} is substituted. Measured: SDXL has no reliable prior",
        "# for abstract glyphs -- check marks, arrows and bolts came back as the",
        "# same vague rounded rectangle across six retries and three style",
        "# combinations. Generate things, hand-author symbols (docs/findings.md).",
        # The closing quotes sit on the last content line: TOML trims a newline
        # straight after the opening """ but keeps the one before the closing
        # one, and a scaffold with a trailing newline goes to the model that way.
        'scaffold = """',
        _wrap_toml(SCAFFOLD) + '"""',
        'negative = """',
        _wrap_toml(NEGATIVE) + '"""',
        "",
    ]
    lines += _STARTER_OPTIONS.get(backend, [])
    lines += [
        "",
        "[profile.brand-icons.output]",
        "size  = 128",
        'svg   = "flat"    # 29 KB and 18 paths, against 385 KB and 670 for "fine"',
        "sheet = true",
        "",
        "[profile.brand-icons.postprocess]",
        "thresh    = 42     # higher eats the icon's own dark outline strokes",
        "despeckle = true   # measured 664-1627 stray fragments per image",
        "keep_frac = 0.15",
        "pad_ratio = 0.06",
        "snap_palette = false   # opt-in: it bands antialiased edges",
        "",
        "# --- how to reach a backend. Variable NAMES only, never values. ---",
    ]

    env = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}.get(backend)
    if env:
        model = "gpt-image-1.5" if backend == "openai" else "gemini-3.1-flash-image"
        per_image = "0.009" if backend == "openai" else "0.067"
        lines += [
            "[backend.%s]" % backend,
            'api_key_env = "%s"' % env,
            "",
            "# --- cost estimates. Defaults ship in pricing.py and rot fast:",
            "# two of the models priced on that date changed status inside the",
            "# same research window. Override here when they move.",
            '[price."%s:%s"]' % (backend, model),
            "per_image = %s" % per_image,
            'as_of     = "2026-08-20"',
        ]
    else:
        lines += [
            "# [backend.openai]",
            '# api_key_env = "OPENAI_API_KEY"   # a NAME. Never a key.',
            "",
            "# %s runs locally: no per-image cost. An 88-icon set is about 25"
            % backend,
            "# minutes of wall clock, not a day, and interrupted runs resume.",
        ]
    return "\n".join(lines) + "\n"


_STARTER_MODEL = {
    "fooocus": [
        'model    = "juggernautXL_v8Rundiffusion"',
        "seed     = 77777        # one seed across every subject is the lever",
        'render   = "1024x1024"  # generation size; the tracer wants full res',
    ],
    "comfyui": [
        'model    = "sd_xl_base_1.0.safetensors"',
        "seed     = 77777        # ComfyUI makes its noise on the CPU, so the",
        "                        # seed is GPU-independent -- pixels are not",
        'render   = "1024x1024"',
    ],
    "openai": [
        "# gpt-image-1.5, NOT gpt-image-2: the docs state plainly that 2 rejects",
        '# background="transparent", which 1, 1.5 and 1-mini support.',
        'model    = "gpt-image-1.5"',
        "# There is no seed parameter anywhere on this API. Consistency has to",
        "# come from the scaffold plus a reference image, so pin an anchor icon",
        "# and select best-of-n against it. n multiplies what you are billed.",
        'anchor   = "fire"',
        "n        = 3",
        'render   = "1024x1024"  # the only square size these models offer',
    ],
    "gemini": [
        'model    = "gemini-3.1-flash-image"',
        "# The seed field exists, but no Google document claims it reproduces an",
        "# image and a mandatory thinking pass runs on every call (and is",
        "# billed). Do not build a consistency strategy on it.",
        'anchor   = "fire"',
        "n        = 3",
        'render   = "1024x1024"',
    ],
}

_STARTER_OPTIONS = {
    "fooocus": [
        "[profile.brand-icons.options]",
        'host        = "127.0.0.1:7865"',
        'performance = "Speed"       # 13-15 s per 1024x1024 render',
        "sharpness   = 2.0",
        "# Measured: V2 + Sharp leaves a near-flat charcoal backdrop that keys",
        "# out cleanly. `Sticker Designs` and `Simple Vector Art` -- the two",
        "# whose names promise flat vector output -- lay down heavy asphalt",
        "# texture that defeats the flood fill, and negative-prompting",
        "# `texture, vignette` did not rescue them (docs/findings.md).",
        'styles      = ["Fooocus V2", "Fooocus Sharp"]',
    ],
    "comfyui": [
        "[profile.brand-icons.options]",
        'host     = "127.0.0.1:8188"',
        "# Leave `workflow` unset to use the SDXL graph that ships inside the",
        "# package. Point it at your own only in API format: File -> Export",
        "# Workflow (API), or the older Dev mode Options -> Save (API Format).",
        "# Converting a UI export by hand is not a field-stripping exercise --",
        "# it means rebuilding links and mapping positional widget values --",
        "# so export it properly rather than editing the JSON.",
        '# workflow = "workflows/my_sdxl.api.json"',
    ],
    "openai-compatible": [
        "[profile.brand-icons.options]",
        "# The generic path: anything that speaks POST {base_url}/images/",
        "# generations. `preset` fills in base_url and the key variable for the",
        "# endpoints we checked; set base_url yourself for anything else.",
        "# Note that Fireworks, LM Studio and vLLM do NOT serve an images",
        "# endpoint at all, whatever their chat API does.",
        'preset   = "grok"        # or together, deepinfra',
        'model    = "grok-imagine-image-2.0"',
        "# xAI has no seed and no negative prompt, so the seed above is dropped",
        "# with a warning. Pin an anchor icon instead, or accept the drift.",
        "# It has also retired two image model ids inside six months; a 404 here",
        "# usually means the id moved, not that the endpoint is down.",
        "# supports_seed = true   # only if you KNOW your endpoint honours it",
    ],
    "openai": [
        "[profile.brand-icons.options]",
        'quality       = "low"        # $0.009 per 1024x1024 image on 1.5',
        'output_format = "png"',
        "# `size` and the transparent background are NOT set here: size comes",
        "# from `render` above, and transparency is requested for every icon.",
        "# Both are a request rather than a guarantee -- the alpha channel is",
        "# verified after download and postprocess.cutout stays wired as the",
        "# fallback when a provider hands back an opaque image anyway.",
    ],
    "gemini": [
        "[profile.brand-icons.options]",
        "# No Gemini model emits an alpha channel, and the response mime enum",
        "# offers JPEG -- postprocess.to_png transcodes, but JPEG ringing around",
        "# a hard outline is exactly the signal thresh=42 keys on. Watch the",
        "# background share on the first few icons.",
        'aspect_ratio = "1:1"',
    ],
}


def _wrap_toml(text, width=74):
    r"""Wrap a long prompt into a TOML multi-line string with `\` continuations,
    so the file stays readable and the parsed value keeps its single spaces."""
    lines, line = [], ""
    for word in text.split(" "):
        if line and len(line) + 1 + len(word) > width:
            lines.append(line + " \\")
            line = word
        else:
            line = "%s %s" % (line, word) if line else word
    lines.append(line)
    return "\n".join(lines)
