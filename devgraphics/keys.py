"""
API keys come from the environment. The config names a variable; it never holds
one, and there is no schema key that would let it.

That is the whole design, and it is a deliberate refusal of the two obvious
alternatives. An optional `api_key = "sk-..."` would be convenient exactly once
and then live in git history forever -- hardcoded keys remain the leading way
keys leak, and a schema that permits one eventually gets one. And no `keyring`:
it is a native dependency, which breaks the pure-Python, identical-on-three-OSes
property this project sells, and the fallback people reach for instead -- a
0600 keys.json -- is a false promise on Windows, where os.chmod only toggles the
read-only bit and cannot restrict another account at all. A "secure" store that
is secure on two OSes out of three is worse than an honest environment variable,
because the user stops thinking about it.

`.env` is never auto-loaded. Implicit environment mutation from a file nobody
named makes `devgraphics gen` behave differently depending on which directory it
was run from. `load_env_file` is opt-in (--env-file), and uses setdefault
semantics -- the real environment always wins over the file, matching uv -- so
exporting a variable to override the file works the way everyone expects.
"""

import os

from .backends.base import COMPAT_PRESETS, AuthError
from .config import ConfigError

#: backend name -> the variables that provider's own SDK reads, in order.
#: gemini lists GEMINI_API_KEY first: google-genai prefers it when both are set,
#: and GOOGLE_API_KEY is shared with half of Google Cloud, so a stale one is the
#: likelier of the two to be sitting in a shell already.
CONVENTIONAL = {
    "openai":    ("OPENAI_API_KEY",),
    "gemini":    ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "xai":       ("XAI_API_KEY",),
    "together":  ("TOGETHER_API_KEY",),
    "deepinfra": ("DEEPINFRA_API_KEY",),
    "stability": ("STABILITY_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}

# The openai-compatible presets already carry a variable name in base.py. Folding
# them in here rather than retyping them keeps the two from ever disagreeing --
# xAI retired two model ids in six months, so that table will be edited again.
for _name, _preset in COMPAT_PRESETS.items():
    if _preset.get("api_key_env"):
        CONVENTIONAL.setdefault(_name, (_preset["api_key_env"],))

#: Backends that need no credential at all. Named so the error path can say
#: "fooocus takes no API key" instead of hunting for a variable that will never
#: exist. Local servers are unauthenticated by default, which is its own
#: problem -- see the README -- but not this module's.
LOCAL = frozenset({"fooocus", "comfyui", "invokeai", "a1111"})


def resolve(backend_name, explicit_env=None, config=None):
    """Return (key, variable_name) for `backend_name`.

    Order: --api-key-env NAME > [backend.NAME] api_key_env > the provider's
    conventional variable. Raises AuthError naming the exact variable to set.
    """
    entry = (config or {}).get("backend", {}).get(backend_name, {}) or {}
    for bad in ("api_key", "apikey", "key", "token", "secret"):
        if bad in entry:
            raise ConfigError(
                "[backend.%s] %s must not hold a value.\n"
                "  Write api_key_env = \"NAME\" and export NAME instead; then\n"
                "  rotate that key, because the config is a committed file."
                % (backend_name, bad))

    names = []
    for name in (explicit_env, entry.get("api_key_env")):
        if name and name not in names:
            names.append(name)
    for name in CONVENTIONAL.get(backend_name, ()):
        if name not in names:
            names.append(name)

    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip(), name

    raise AuthError(_missing(backend_name, names))


def _missing(backend_name, names):
    if not names:
        if backend_name in LOCAL:
            return ("%s is a local backend and takes no API key; nothing to "
                    "resolve." % backend_name)
        return ("no API key variable is known for backend %r. Name one:\n"
                "    [backend.%s]\n"
                "    api_key_env = \"YOUR_VARIABLE\"\n"
                "  or pass --api-key-env YOUR_VARIABLE."
                % (backend_name, backend_name))
    first = names[0]
    lines = ["no API key for backend %r." % backend_name,
             "  Looked at: %s (all unset or empty)." % ", ".join(names),
             "  Fix, in a shell:",
             "      set %s=...        (Windows cmd)" % first,
             "      $env:%s = '...'   (PowerShell)" % first,
             "      export %s=...     (bash, zsh)" % first,
             "  or put it in a file and pass --env-file PATH. The config names",
             "  the variable; it never holds the key."]
    return "\n".join(lines)


def load_env_file(path):
    """Load KEY=VALUE lines into os.environ, without overwriting anything.

    setdefault, not assignment: the real environment wins over the file, which
    is uv's rule and the only one that makes a temporary override possible
    without editing the file. Returns the names actually applied -- the caller
    can then say how many were already set, which is the confusing case.

    Deliberately not python-dotenv: no interpolation, no multi-line values, no
    inline-comment stripping (a `#` inside a key would be silently truncated,
    and truncating a credential produces a 401 that looks like a wrong key
    rather than a parsing bug). `export ` prefixes and matched surrounding
    quotes are tolerated because every .env in the wild has both.
    """
    applied = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            name, _, value = line.partition("=")
            name, value = name.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if not name:
                continue
            if name in os.environ:      # setdefault, spelled out so the caller
                continue                # can be told what the file did not do
            os.environ[name] = value
            applied.append(name)
    return applied
