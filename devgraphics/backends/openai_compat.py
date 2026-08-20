"""
One backend for every server that speaks POST {base_url}/images/generations.

Three decisions carry this module.

**`response_format` is never sent.** The enum is the least portable thing in the
whole OpenAI image contract and every vendor breaks it differently: OpenAI spells
the value `b64_json`, Together spells the *request* value `base64` while still
naming the *response* field `b64_json`, DeepInfra accepts only `b64_json` and
cannot return a url at all, and OpenAI's GPT image models ignore the parameter
entirely and always answer with base64. Omitting it is the only choice that works
against all four, so `generate()` reads whichever of `url` / `b64_json` came back
and fetches the url when that is what it got. Put it in `extra_body` if you must,
and own the consequences.

**Capabilities are asserted, not detected.** A generic endpoint publishes nothing
about itself, so everything optional is reported False and stays False until you
say otherwise with `supports_seed`, `supports_negative_prompt` or
`supports_transparent`. That default is not pessimism: as of 2026-08-20 Together
is the only OpenAI-shaped host in this family with a seed, and xAI's REST field
list is exactly model, prompt, n, aspect_ratio, resolution, response_format, user,
storage_options -- no seed, no size, no negative_prompt. Asserting a capability
your server lacks is on you. The good outcome is a 400. The bad outcome is a
parameter accepted and ignored, and a set that drifts without telling you --
DeepInfra documents accepting `quality` and `style` "for compatibility" with no
effect on the output, so the bad outcome is real.

**One class rather than a module per vendor.** xAI has retired an image model id
twice in six months: `grok-2-image-1212` deprecated 2026-02-28 (its docs page now
404s) and `grok-imagine-image-pro` retired 2026-05-15 in favour of
`grok-imagine-image-quality`. A vendor whose ids move that fast earns a base_url
and a config line, not a module that has to be re-released to keep up.

Servers that do NOT serve this endpoint, recorded so nobody loses an afternoon
proving it again (researched 2026-08-20):

  Fireworks  images live at /inference/v1/workflows/accounts/fireworks/models/
             {model}/text_to_image; its OpenAI-compatibility page documents only
             /v1/completions and /v1/chat/completions.
  LM Studio  the published route list is /v1/models, /v1/responses,
             /v1/chat/completions, /v1/embeddings, /v1/completions. No images.
  vLLM       chat, completions, embeddings, score, audio, tokenize. It is a text
             and embedding server; there is no diffusion path.
  Ollama     UNVERIFIED, treat as unsupported. Image generation is experimental,
             macOS only, documented for the CLI, and absent from Ollama's own
             OpenAI-compatibility page. Community sources claim
             /v1/images/generations behind an x/imagegen runner; first-party docs
             do not.

Verified positive: OpenAI, xAI, Together, DeepInfra -- and DeepInfra's shim sits
under an extra path segment, https://api.deepinfra.com/v1/openai, which is why the
preset carries the whole base_url rather than a hostname.
"""

import base64
import json
import os
import urllib.parse

from .. import _http
from ..postprocess import to_png
from .base import (COMPAT_PRESETS, AuthError, BackendError, Capabilities,
                   UnsupportedOption)

#: Every option key this backend accepts, from a profile's [options] table or
#: from -O key=value. Anything else raises UnsupportedOption: a typo'd -O must
#: fail loudly rather than quietly change nothing.
OPTIONS = frozenset([
    "preset", "base_url", "model", "api_key_env", "extra_body", "size_param",
    "supports_seed", "supports_negative_prompt", "supports_transparent",
    "cost_per_image",
])

#: How this endpoint spells "how big". There is no portable answer: OpenAI and
#: DeepInfra take a WIDTHxHEIGHT string, Together takes width/height integers,
#: xAI has no size parameter at all and offers aspect_ratio plus a 1k/2k
#: resolution tier. Sending the wrong one is a 400, so it is configuration.
SIZE_PARAMS = ("size", "width_height", "aspect_ratio", "none")

PRESET_SIZE_PARAM = {
    "grok":      "aspect_ratio",
    "together":  "width_height",
    "deepinfra": "size",
}

#: xAI's aspect_ratio enum, numeric entries only -- "auto" is also accepted but
#: cannot be compared against a requested (w, h). Nearest ratio wins, and
#: postprocess.render()'s LANCZOS downscale is the final normaliser either way.
ASPECT_RATIOS = ("1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2", "9:19.5",
                 "19.5:9", "9:20", "20:9", "1:2", "2:1")

#: USD per image, from the live xAI models page (fetched 2026-08-20). Only ids
#: whose price is published per model are listed; for anything else --dry-run
#: needs -O cost_per_image=... before it can total a batch.
PRICES = {
    "grok-imagine-image":         0.02,
    "grok-imagine-image-2.0":     0.04,
    "grok-imagine-image-quality": 0.05,
}

PRESET_NOTES = {
    "grok": (
        "xAI has retired an image model id twice in six months: "
        "grok-2-image-1212 deprecated 2026-02-28 (its docs page 404s) and "
        "grok-imagine-image-pro retired 2026-05-15 -> grok-imagine-image-quality."
        " Live ids as of 2026-08-20: grok-imagine-image, grok-imagine-image-2.0, "
        "grok-imagine-image-quality. A 404 on generate means it moved again.",
        "xAI exposes no seed and no negative prompt, so the prompt scaffold is "
        "the only lever holding the set together.",
        "xAI has no size parameter: aspect_ratio plus resolution 1k|2k is the "
        "whole dimension story, and what '1k' returns at 1:1 is UNVERIFIED -- "
        "measure it before assuming 1024x1024.",
    ),
    "together": (
        "Together spells the response_format request value 'base64' while still "
        "naming the response field 'b64_json'. devgraphics never sends "
        "response_format, so this cannot bite -- unless you add one via "
        "extra_body, in which case 'b64_json' is the wrong word here.",
        "Together is the one host in this family with a real seed, and it takes "
        "seed and negative_prompt as ordinary body keys: -O supports_seed=true "
        "-O supports_negative_prompt=true turns both on.",
        "Together's current documentation gives the base_url as "
        "https://api.together.ai/v1 while this preset carries the older "
        "https://api.together.xyz/v1. If that 404s or fails to resolve, set "
        "base_url explicitly rather than assuming the model id moved.",
    ),
    "deepinfra": (
        "DeepInfra's OpenAI shim lives under /v1/openai, so the base_url carries "
        "an extra path segment. It returns b64_json only and cannot answer with "
        "a url.",
        "DeepInfra accepts quality and style 'for compatibility' and they have "
        "no effect on the output. Silently-ignored parameters are worse than "
        "rejected ones for a set that has to stay consistent.",
    ),
}


class OpenAICompatBackend:
    """Any OpenAI-shaped images endpoint, named by base_url + model.

    The constructor touches no network and reads no environment: `capabilities`
    has to be answerable with the server switched off and the key unset, or
    --dry-run needs the very thing it exists to avoid. The API key is resolved at
    request time, and only then.
    """

    def __init__(self, **options):
        unknown = sorted(set(options) - OPTIONS)
        if unknown:
            raise UnsupportedOption(
                "openai-compatible does not accept %s; accepted keys: %s"
                % (", ".join(unknown), ", ".join(sorted(OPTIONS))))

        self._options = dict(options)
        self.preset = options.get("preset") or None
        defaults = {}
        if self.preset:
            if self.preset not in COMPAT_PRESETS:
                raise ValueError("unknown preset %r; known: %s"
                                 % (self.preset, ", ".join(sorted(COMPAT_PRESETS))))
            defaults = COMPAT_PRESETS[self.preset]

        base_url = options.get("base_url") or defaults.get("base_url") or ""
        self.base_url = base_url.rstrip("/")
        self.api_key_env = (options.get("api_key_env")
                            or defaults.get("api_key_env") or "OPENAI_API_KEY")
        self.model = options.get("model") or ""
        if not self.base_url:
            raise ValueError("openai-compatible needs base_url (or a preset: %s)"
                             % ", ".join(sorted(COMPAT_PRESETS)))
        if not self.model:
            raise ValueError("openai-compatible needs model; a generic endpoint "
                             "has no default and vendors retire ids without "
                             "notice")

        self.size_param = (options.get("size_param")
                           or PRESET_SIZE_PARAM.get(self.preset, "size"))
        if self.size_param not in SIZE_PARAMS:
            raise ValueError("size_param must be one of %s, not %r"
                             % (", ".join(SIZE_PARAMS), self.size_param))

        self.extra_body = _mapping("extra_body", options.get("extra_body"))
        self.supports_seed = _flag("supports_seed", options.get("supports_seed"))
        self.supports_negative_prompt = _flag(
            "supports_negative_prompt", options.get("supports_negative_prompt"))
        self.supports_transparent = _flag(
            "supports_transparent", options.get("supports_transparent"))
        cost = options.get("cost_per_image")
        self.cost_per_image = (float(cost) if cost is not None
                               else PRICES.get(self.model))
        self._caps = None

    # --- contract -------------------------------------------------------

    @property
    def capabilities(self):
        if self._caps is None:
            self._caps = Capabilities(
                name="openai-compatible %s (%s)" % (self._label(), self.model),
                seed=self.supports_seed,
                # Even an honoured seed is only a promise about this host's
                # scheduler on this model version; nobody in this family
                # documents bit-identical output, so never deterministic.
                deterministic=False,
                negative_prompt=self.supports_negative_prompt,
                transparent=self.supports_transparent,
                # /images/generations carries no reference-image mechanism.
                # Editing is a different route (/images/edits) with a different
                # body, and not every host in this family serves it.
                reference_images=0,
                batch=True,
                sizes=(),
                cost_per_image=self.cost_per_image,
                notes=self._notes(),
            )
        return self._caps

    def generate(self, request):
        """POST the prompt, return one PNG per returned image."""
        cfg = self._for(request.options)
        body = {"model": cfg.model, "prompt": request.prompt}
        if request.count > 1:
            body["n"] = request.count
        body.update(cfg._size_fields(request.size))
        if cfg.supports_seed and request.seed is not None:
            body["seed"] = int(request.seed)
        if cfg.supports_negative_prompt and request.negative:
            body["negative_prompt"] = request.negative
        if cfg.supports_transparent and request.transparent:
            body["background"] = "transparent"
        # Last, deliberately: extra_body is the escape hatch, so it wins over
        # anything computed above. No response_format is ever added -- see the
        # module docstring for why sending it is a portability bug, not a
        # convenience.
        body.update(cfg.extra_body)

        doc = cfg._post(body)
        return cfg._images(doc)

    @classmethod
    def probe(cls, **options):
        """(reachable, message), listing models rather than making one.

        Never generates: a paid backend whose probe costs money is a trap.
        """
        try:
            self = cls(**options)
        except (UnsupportedOption, ValueError) as exc:
            return False, str(exc)
        if not os.environ.get(self.api_key_env):
            return False, ("%s is not set; export the API key into it or point "
                           "api_key_env at the variable that holds it"
                           % self.api_key_env)

        url = self.base_url + "/models"
        try:
            doc = _http.request_json(url, headers=self._headers(), retries=1)
        except BackendError as exc:
            return False, "%s: %s" % (url, exc)

        listed = doc.get("data") if isinstance(doc, dict) else doc
        ids = [m.get("id") for m in (listed or [])
               if isinstance(m, dict) and m.get("id")]
        # A missing id is reported, not failed: several hosts list only the
        # models a key is entitled to, and some list no image models at all on
        # /models even though /images/generations serves them.
        if ids and self.model not in ids:
            return True, ("%s reachable and %s is set, but %r is not among the "
                          "%d models it lists -- check whether the id was "
                          "retired" % (self.base_url, self.api_key_env,
                                       self.model, len(ids)))
        return True, "%s reachable, %s set, model %s" % (self.base_url,
                                                         self.api_key_env,
                                                         self.model)

    # --- request --------------------------------------------------------

    def _for(self, options):
        """This instance, or a copy carrying per-request option overrides.

        Rebuilding through the constructor means Request.options gets exactly
        the same validation as -O, including UnsupportedOption on a typo. An
        extra_body passed here replaces the configured one rather than merging
        with it.
        """
        if not options:
            return self
        merged = dict(self._options)
        merged.update(options)
        return OpenAICompatBackend(**merged)

    def _size_fields(self, size):
        width, height = int(size[0]), int(size[1])
        if self.size_param == "size":
            return {"size": "%dx%d" % (width, height)}
        if self.size_param == "width_height":
            return {"width": width, "height": height}
        if self.size_param == "aspect_ratio":
            # No resolution tier: xAI documents 1k|2k but states no default, and
            # guessing one for every host that spells size this way would be an
            # invention. Set it through extra_body if you need 2k.
            return {"aspect_ratio": _nearest_ratio(width, height)}
        return {}

    def _headers(self):
        key = os.environ.get(self.api_key_env)
        if not key:
            raise AuthError(
                "%s is not set in the environment. Keys are never read from the "
                "config file. A local server that ignores auth still needs the "
                "variable set to something -- the SDKs send a dummy value."
                % self.api_key_env)
        return {"Authorization": "Bearer %s" % key}

    def _post(self, body):
        url = self.base_url + "/images/generations"
        try:
            return _http.request_json(url, payload=body, headers=self._headers())
        except BackendError as exc:
            raise self._explain(exc, body)

    def _explain(self, exc, body):
        """Add the context a bare status line cannot carry.

        Only plain BackendError is rewrapped. AuthError, RateLimited,
        PaymentRequired and ModerationBlocked already say the useful thing and
        the caller switches on their class.
        """
        text = str(exc)
        if type(exc) is not BackendError:
            return exc
        if "HTTP 404" in text:
            hint = ("no such route or model on %s. Check the base_url keeps its "
                    "version segment (DeepInfra's shim lives under /v1/openai), "
                    "then check %r still exists -- xAI retired grok-2-image-1212 "
                    "on 2026-02-28 and grok-imagine-image-pro on 2026-05-15."
                    % (self.base_url, self.model))
            return BackendError(text + "\n  " + hint)
        if "HTTP 400" in text:
            hint = ("a 400 here is usually an unportable body key. This request "
                    "sent %s; size_param=%s chose the dimension keys, and "
                    "everything in extra_body goes to the server verbatim."
                    % (", ".join(sorted(body)), self.size_param))
            return BackendError(text + "\n  " + hint)
        return exc

    def _images(self, doc):
        items = doc.get("data") if isinstance(doc, dict) else None
        if not isinstance(items, list) or not items:
            raise BackendError("%s returned no images: %s"
                               % (self.base_url, json.dumps(doc)[:200]))
        out = []
        for item in items:
            if not isinstance(item, dict):
                raise BackendError("unexpected data entry from %s: %r"
                                   % (self.base_url, item))
            blob = item.get("b64_json")
            if blob:
                data = base64.b64decode(blob)
            elif item.get("url"):
                # No Authorization header: the url is a short-lived link that
                # often points at a CDN on another host, and the key has no
                # business travelling there.
                data = _http.request_bytes(item["url"])
            else:
                raise BackendError(
                    "response carried neither url nor b64_json (keys: %s). This "
                    "host may spell the payload differently; devgraphics does "
                    "not send response_format, so it cannot be a format mismatch."
                    % ", ".join(sorted(item)))
            out.append(to_png(data))
        return out

    # --- reporting ------------------------------------------------------

    def _label(self):
        return self.preset or (urllib.parse.urlsplit(self.base_url).netloc
                               or self.base_url)

    def _notes(self):
        notes = list(PRESET_NOTES.get(self.preset, ()))
        asserted = [n for n, on in (("seed", self.supports_seed),
                                    ("negative prompt",
                                     self.supports_negative_prompt),
                                    ("transparent background",
                                     self.supports_transparent)) if on]
        if asserted:
            notes.append(
                "you asserted %s for this endpoint; devgraphics cannot verify "
                "it. If the server ignores the parameter rather than rejecting "
                "it, the set drifts silently." % ", ".join(asserted))
        else:
            notes.append(
                "a generic endpoint cannot be introspected, so seed, negative "
                "prompt and alpha are reported unavailable. Turn on what your "
                "server really has with supports_seed, "
                "supports_negative_prompt, supports_transparent.")
        if self.cost_per_image is None:
            notes.append("no published per-image price for %s; set "
                         "cost_per_image to let --dry-run total a batch"
                         % self.model)
        return tuple(notes)


# --- option coercion --------------------------------------------------
#
# -O key=value hands everything over as a string while a TOML profile hands over
# real types, so both have to land on the same value.

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _flag(name, value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError("%s must be true or false, not %r" % (name, value))


def _mapping(name, value):
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise ValueError("%s must be a JSON object, e.g. "
                             '%s=\'{"steps": 30}\': %s'
                             % (name, name, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("%s must be a JSON object, not %r" % (name, value))
    return dict(value)


def _nearest_ratio(width, height):
    want = width / float(height)

    def distance(text):
        left, right = text.split(":")
        return abs(float(left) / float(right) - want)

    return min(ASPECT_RATIOS, key=distance)
