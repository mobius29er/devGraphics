"""
Client for a stock AUTOMATIC1111 WebUI. The cheapest backend here to talk to.

One blocking POST to /sdapi/v1/txt2img is the whole round trip. The payload
takes real integer width and height -- no aspect-ratio enum with HTML buried
in the choice label, which is what backends/fooocus.py has to substring-match
-- a real `negative_prompt`, a real `seed`, and the image bytes come back
inline as base64 in the same response. No queue protocol, no websocket, no
second fetch for a file path on the server. That is why this module is a third
the size of the Fooocus one.

Three things about it are not obvious.

**The API is off by default.** Launched without `--api`, every /sdapi/v1/*
path 404s while the web UI itself keeps working, so the server looks healthy
and every call fails. It is the first thing anyone hits, so both `probe` and
`generate` name the flag instead of passing a bare "HTTP 404" upward.

**`info` is a JSON-encoded string inside the JSON response**, and it is the
only place the seed actually used is reported. That matters because the
lockfile records the seed: sending `seed=-1` and never reading `info` back
yields a set nobody can regenerate. A second json.loads recovers it onto
`self.last_seed`, with the whole parsed dict on `self.last_info`.

**The checkpoint is pinned per request through `override_settings`,** not by
POSTing /sdapi/v1/options. Both work. Only one of them avoids leaving a
different model loaded for whoever else has that instance open when a run dies
halfway through.

Three limits worth stating plainly. A1111 makes no cross-platform determinism
promise, so capabilities report `seed=True, deterministic=False`: a shared
seed holds a set's look together on one machine and is not a build artefact.
Stock A1111 has **no authentication at all** unless it was launched with
`--api-auth user:pass`, so a `--host` pointing anywhere but loopback is the
operator's problem, not this client's; where that flag is in use, put the same
"user:pass" in A1111_API_AUTH and it is sent as HTTP Basic -- environment
only, because credentials never belong in devgraphics.toml. And a batch is
trimmed to `count` from the front of the `images` list: whether a build ever
prefixes that list with a contact-sheet grid is UNVERIFIED here, and if one
does, the trim would keep the grid rather than an icon.

Payload defaults drift between WebUI versions (512x512, 20 steps, cfg 7 at the
time of the research), so nothing optional is sent unless the caller asked for
it. The running instance serves its own authoritative schema at /docs; that,
not this module, is the reference for anything below.
"""

import base64
import json
import os

from .._http import request_json
from ..postprocess import to_png
from .base import AuthError, BackendError, Capabilities, UnsupportedOption

DEFAULT_HOST = "127.0.0.1:7860"

#: HTTP Basic credentials for an instance launched with --api-auth, in the same
#: "user:pass" form that flag takes. Environment only, never a config value.
AUTH_ENV = "A1111_API_AUTH"

#: Every option key this backend accepts, in the constructor and in
#: Request.options alike. Anything else raises UnsupportedOption, because a
#: typo'd -O that silently changes nothing is worse than a crash.
OPTIONS = frozenset([
    "checkpoint",           # -> override_settings["sd_model_checkpoint"]
    "override_settings",    # the raw escape hatch, merged over `checkpoint`
    "sampler_name",
    "steps",
    "cfg_scale",
    "batch_size",
    "n_iter",
])

#: The type A1111's schema wants for each numeric knob. `-O steps=30` arrives as
#: the string "30", and how the server's pydantic version coerces that is its
#: business, not something to guess at.
TYPES = {"sampler_name": str, "steps": int, "cfg_scale": float,
         "batch_size": int, "n_iter": int}

NO_API = ("the WebUI is up but was launched without --api, so every /sdapi/v1/* "
          "path 404s. Restart it with --api (or --nowebui for a headless box).")

API_AUTH = ("this instance was launched with --api-auth; put the same "
            "\"user:pass\" in the %s environment variable." % AUTH_ENV)


class A1111Backend:
    """Text to image against /sdapi/v1/txt2img.

    Constructing this touches no network -- `capabilities` has to answer with the
    server switched off or --dry-run needs the very thing it exists to avoid.
    """

    def __init__(self, host=DEFAULT_HOST, timeout=900, **options):
        _reject_unknown(options)
        self.host = host
        self.base = _base_url(host)
        self.timeout = float(timeout)
        self.defaults = dict(options)
        #: The seed A1111 reports it actually used, after the most recent
        #: generate(). The lockfile records this, so it is the whole reason the
        #: `info` string gets a second json.loads.
        self.last_seed = None
        #: Everything else `info` carried. A batch is also said to report
        #: `all_seeds`, but only `seed` is confirmed by the research, so
        #: per-image seeds are read from here rather than promised.
        self.last_info = {}

    @property
    def capabilities(self):
        checkpoint = self.defaults.get("checkpoint")
        return Capabilities(
            name="a1111",
            seed=True,
            deterministic=False,        # no cross-host pixel promise
            negative_prompt=True,
            transparent=False,
            reference_images=0,         # img2img is a different endpoint
            batch=True,
            sizes=(),                   # arbitrary integer width/height
            cost_per_image=None,        # local GPU
            notes=(
                "the WebUI must have been launched with --api",
                "no native alpha; postprocess.cutout keys the backdrop out",
                "checkpoint pinned per request: %s"
                % (checkpoint or "whatever the server currently has loaded"),
            ),
        )

    def generate(self, request):
        payload = self._payload(request)
        doc = self._post("/sdapi/v1/txt2img", payload)

        images = doc.get("images") or []
        if not images:
            raise BackendError("A1111 accepted the request but returned no "
                               "images: %s" % json.dumps(doc)[:200])

        self.last_info = _info(doc)
        try:
            self.last_seed = int(self.last_info["seed"])
        except (KeyError, TypeError, ValueError):
            self.last_seed = None

        count = max(int(request.count), 1)
        return [to_png(_decode(s)) for s in images[:count]]

    @classmethod
    def probe(cls, **options):
        """Reachability, without generating anything.

        GET /sdapi/v1/options is the cheapest endpoint that proves the API router
        is actually mounted: it returns the live settings dict, including which
        checkpoint is loaded, and costs no GPU time.
        """
        options = dict(options)
        host = options.pop("host", DEFAULT_HOST)
        timeout = float(options.pop("timeout", 10))
        _reject_unknown(options)

        url = _base_url(host) + "/sdapi/v1/options"
        try:
            # retries=0: a box that is switched off should say so now, not after
            # four rounds of backoff.
            doc = request_json(url, headers=_auth_headers(), timeout=timeout,
                               retries=0)
        except AuthError as exc:
            return False, ("%s rejected the request: %s\n  %s"
                           % (host, exc, API_AUTH))
        except BackendError as exc:
            if _is_404(exc):
                return False, "%s: %s" % (host, NO_API)
            return False, "%s is not answering: %s" % (host, exc)

        loaded = ""
        if isinstance(doc, dict) and doc.get("sd_model_checkpoint"):
            loaded = ", checkpoint %s" % doc["sd_model_checkpoint"]
        return True, "A1111 API is up at %s%s" % (host, loaded)

    # --- internals ------------------------------------------------------

    def _payload(self, request):
        opts = dict(self.defaults)
        opts.update(request.options or {})
        _reject_unknown(opts)

        count = max(int(request.count), 1)
        batch = count
        if "batch_size" in opts:
            batch = max(_typed("batch_size", opts["batch_size"]), 1)
        # A1111's total is batch_size * n_iter. batch_size is VRAM-bound, so a
        # caller who pins it still has to end up with `count` images; pinning
        # n_iter as well means the caller owns that arithmetic.
        if "n_iter" in opts:
            iters = max(_typed("n_iter", opts["n_iter"]), 1)
        else:
            iters = (count + batch - 1) // batch

        payload = {
            "prompt": request.prompt,
            "negative_prompt": request.negative or "",
            # -1 is A1111's own "pick one"; what it picked comes back in info.
            "seed": -1 if request.seed is None else int(request.seed),
            "width": int(request.size[0]),
            "height": int(request.size[1]),
            "batch_size": batch,
            "n_iter": iters,
        }
        # Everything else is omitted unless asked for, so the server's own
        # version-appropriate defaults apply instead of a schema copied from a
        # wiki page that has since moved.
        for key in ("sampler_name", "steps", "cfg_scale"):
            if key in opts:
                payload[key] = _typed(key, opts[key])

        override = opts.get("override_settings") or {}
        if not isinstance(override, dict):
            raise UnsupportedOption(
                "override_settings must be a table of A1111 setting names, got %s"
                % type(override).__name__)
        override = dict(override)
        if opts.get("checkpoint"):
            override.setdefault("sd_model_checkpoint", opts["checkpoint"])
        if override:
            payload["override_settings"] = override
        return payload

    def _post(self, path, payload):
        try:
            return request_json(self.base + path, payload,
                                headers=_auth_headers(), timeout=self.timeout)
        except AuthError as exc:                    # subclass first
            raise AuthError("%s\n  %s" % (exc, API_AUTH)) from exc
        except BackendError as exc:
            if _is_404(exc):
                raise BackendError("%s\n  %s" % (exc, NO_API)) from exc
            raise


def _base_url(host):
    """--host is "host:port" everywhere else here; a full URL also works."""
    host = str(host).rstrip("/")
    return host if "://" in host else "http://%s" % host


def _auth_headers():
    creds = os.environ.get(AUTH_ENV, "").strip()
    if not creds:
        return {}
    token = base64.b64encode(creds.encode("utf-8")).decode("ascii")
    return {"Authorization": "Basic %s" % token}


def _is_404(exc):
    """_http collapses the status code into the message, and _retryable already
    matches on it the same way, so this is the available signal rather than a
    parallel one."""
    return "HTTP 404" in str(exc)


def _info(doc):
    """`info` is JSON *inside* the JSON body, so it needs a second json.loads."""
    raw = doc.get("info")
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _decode(text):
    # A1111 returns bare base64. Stripping a data: prefix costs one line and
    # turns a fork that adds one from an opaque binascii error into a non-event.
    if text.startswith("data:"):
        text = text.split(",", 1)[-1]
    return base64.b64decode(text)


def _typed(key, value):
    # UnsupportedOption rather than a bare ValueError: a bad -O value is the same
    # class of user mistake as a bad -O key, and callers already catch this one.
    try:
        return TYPES[key](value)
    except (TypeError, ValueError) as exc:
        raise UnsupportedOption("-O %s=%r is not a %s"
                                % (key, value, TYPES[key].__name__)) from exc


def _reject_unknown(options):
    unknown = sorted(k for k in options if k not in OPTIONS)
    if unknown:
        raise UnsupportedOption(
            "a1111 does not accept %s; it takes host, timeout and %s"
            % (", ".join(unknown), ", ".join(sorted(OPTIONS))))
