"""
Google Gemini image models on a plain AI Studio key (the Gemini Developer API).

Four decisions carry this module.

**Interactions, not generateContent.** Google runs two live surfaces over the
same models: POST /v1beta/interactions (snake_case, model id inside the body) and
the classic POST /v1/models/{id}:generateContent (camelCase, model id in the
path). generateContent is not deprecated -- the migration guide dated 2026-08-17
says it "remains fully supported" -- but every new capability lands on
Interactions, and Interactions is the only surface with the `store` flag, which is
the only way to stop Google retaining a client's brand prompts for 55 days.
Interactions is the default; `-O surface=generatecontent` is the other one. Do not
carry field names across: the casing flips wholesale, and the wrong one is ignored
rather than rejected.

**The response has to be walked, and two of its images are drafts.** Every SDK
example reads `interaction.output_image.data`. That property does not exist on the
wire; the API reference says of it verbatim "Note: this is added by the SDK". A
urllib client walks steps[] -> type == "model_output" -> content[] -> type ==
"image". Steps of type "thought" carry images too -- Gemini 3 image models emit up
to two interim renders to test composition -- so grabbing the first image block in
the document hands back a draft instead of the icon, silently and forever.

**No alpha, and the mime enum offers JPEG only.** ImageResponseFormat.mimeType has
exactly one non-unspecified value, IMAGE_JPEG, in the Interactions reference and in
both discovery documents, and the guide states flatly that the model does not
support generating a transparent background. So transparent=False, and every
buffer leaves through postprocess.to_png(). See the capabilities note for what
JPEG then does to the flood fill; that part is an inference, not a measurement.

**Imagen is absent because it is dead.** imagen-4.0-generate-001 and its ultra and
fast siblings reached their published shutdown date on 2026-08-17, and
models.predict never existed on v1 at all. An imagen id fails here with the date
rather than 404-ing eighty-eight times.

Raw stdlib HTTP throughout. google-genai 2.19.0 requires Python >=3.10, which
breaks this project's 3.9 floor outright, and pulls ten runtime dependencies
against a project that ships three -- to wrap one POST with two headers.
"""

import base64
import json
import os

from .._http import request_json
from ..postprocess import to_png
from .base import (AuthError, BackendError, Capabilities, PaymentRequired,
                   RateLimited, UnsupportedOption)

BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_MODEL = "gemini-3.1-flash-image"

#: Surface -> the API version its own documentation uses. Interactions exists
#: only under v1beta; the current generateContent guide moved to v1 (stable).
API_VERSION = {"interactions": "v1beta", "generatecontent": "v1"}

#: probe() pins v1beta rather than following api_version: models.list is
#: documented at /v1beta/models, and nothing checked says it answers on v1.
PROBE_VERSION = "v1beta"

#: Read in this order, which is the order the official SDKs read them in.
KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

#: Per-request option keys. Anything else is a typo'd -O and has to fail loudly.
#: Everything else about this backend -- key, model, surface, base URL -- is
#: instance configuration and lives on the constructor instead.
OPTIONS = frozenset(("aspect_ratio", "image_size", "thinking_level",
                     "system_instruction", "store", "seed"))

#: Published shutdown date for every Imagen 4 id on this API.
IMAGEN_SHUTDOWN = "2026-08-17"

_ASPECTS = ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9",
            "21:9")
#: Strip ratios, gemini-3.1-flash-image only.
_STRIPS = ("1:4", "4:1", "1:8", "8:1")

#: Tier -> the pixel edge it produces at 1:1. Uppercase K is mandatory; the guide
#: says lowercase "1k" is rejected. The 512 tier is spelled `512` in the
#: Interactions reference and in both discovery documents but `512px` in the
#: guide's prose, so `512` is what goes on the wire here and `-O image_size=512px`
#: is the escape hatch if a server disagrees.
_TIER_PX = {"512": 512, "1K": 1024, "2K": 2048, "4K": 4096}

#: What each model id can do, as documented on 2026-08-20. `refs` is the total
#: reference images accepted, `style_refs` how many of those may act as STYLE
#: references -- the one real consistency lever here, and it exists on exactly
#: one model. Prices are Google's own published per-image equivalents.
MODELS = {
    "gemini-3.1-flash-image": {
        "sizes": ((512, 512), (1024, 1024), (2048, 2048), (4096, 4096)),
        "tiers": ("512", "1K", "2K", "4K"),
        "aspects": _ASPECTS + _STRIPS,
        "refs": 14,           # 10 object + 4 character
        "style_refs": 3,
        "thinking": True,     # thinking_level is documented for this id only
        "cost": {"512": 0.045, "1K": 0.067, "2K": 0.101, "4K": 0.151},
    },
    "gemini-3.1-flash-lite-image": {
        "sizes": ((1024, 1024),),
        "tiers": ("1K",),
        "aspects": _ASPECTS,
        "refs": 14,           # object images only: no character, no style
        "style_refs": 0,
        "thinking": False,
        "cost": {"1K": 0.0336},
    },
    "gemini-3-pro-image": {
        "sizes": ((1024, 1024), (2048, 2048), (4096, 4096)),
        "tiers": ("1K", "2K", "4K"),
        "aspects": _ASPECTS,
        "refs": 11,           # 6 object + 5 character
        "style_refs": 0,
        "thinking": False,
        "cost": {"1K": 0.134, "2K": 0.134, "4K": 0.24},
    },
    "gemini-2.5-flash-image": {
        "sizes": ((1024, 1024),),
        "tiers": (),          # fixed 1024x1024 at 1:1; no size control at all
        "aspects": _ASPECTS,
        "refs": 3,            # guide: "works best with up to 3 images as input"
        "style_refs": 0,
        "thinking": False,
        "cost": {"1K": 0.039},
    },
}

#: An id nobody has heard of gets this. Refusing it would be worse than guessing:
#: the image line went through five generations in nine months, so an unknown id
#: is a new model far more often than it is a typo, and a hard list would strand
#: this backend the week a replacement ships. Every number is the most
#: conservative documented one, and capabilities says out loud that it is a guess.
UNKNOWN = {"sizes": (), "tiers": ("512", "1K", "2K", "4K"), "aspects": _ASPECTS,
           "refs": 3, "style_refs": 0, "thinking": False, "cost": {}}

_AUTH_HINT = ("no Gemini image model has a free tier, so the key's project must "
              "have billing enabled. Key: -O api_key=... or the environment")
_RATE_HINT = ("Gemini meters images per minute and also caps spend on a rolling "
              "10-minute window -- USD 10 per 10 min on Tier 1, roughly 149 "
              "images at the 1K price")


class GeminiBackend:
    """Gemini image generation behind the Backend contract.

    Nothing here touches the network before generate() or probe(): the model
    table is static, so capabilities answers with no key set and no connection.
    """

    def __init__(self, api_key=None, model=DEFAULT_MODEL,
                 surface="interactions", base_url=BASE_URL, api_version=None,
                 store=None, system_instruction=None, thinking_level=None,
                 aspect_ratio=None, image_size=None, seed=None, timeout=300):
        # aspect_ratio, image_size and seed are per-request knobs that also have
        # to be settable once, at construction: a config profile's [options]
        # table reaches the constructor, and a whole icon set wants one aspect
        # ratio, not one per call. Request.options still overrides these.
        self.defaults = _prune({"aspect_ratio": aspect_ratio,
                                "image_size": image_size, "seed": seed})
        self.model = str(model or DEFAULT_MODEL)
        _reject_imagen(self.model)
        self.surface = _surface(surface)
        self.base_url = str(base_url or BASE_URL).rstrip("/")
        self.api_version = str(api_version or API_VERSION[self.surface])
        self.api_key = api_key
        self.timeout = float(timeout)
        self.profile = MODELS.get(self.model, UNKNOWN)
        self.system_instruction = system_instruction

        # store and thinking_level default to None rather than to their values,
        # so "not asked for" stays distinguishable from "asked for" and the
        # generateContent surface can reject them instead of dropping them.
        _interactions_only(self.surface, {"store": store,
                                          "thinking_level": thinking_level})
        _check_thinking(self.model, self.profile, thinking_level)
        self.store = False if store is None else _as_bool(store)
        self.thinking_level = thinking_level

    @property
    def capabilities(self):
        tiers = self.profile["tiers"]
        tier = "1K" if "1K" in tiers else (tiers[0] if tiers else None)
        return Capabilities(
            name=self.model,
            # seed=False even though generation_config.seed is real and present on
            # both surfaces. It is documented in generic decoding language ("seed
            # used in decoding for reproducibility"); no Google page claims a fixed
            # seed reproduces an IMAGE, and none of the image guides mention seed
            # at all. Gemini 3 image models also run a thinking pass that "cannot
            # be disabled in the API" ahead of every render. seed=True with a
            # caveat would let iconset build its consistency strategy on an
            # unmeasured promise -- exactly the failure base.py names as the one
            # that destroys this tool's only promise. False is the honest answer
            # until somebody measures it; -O seed=N still sends the field, for
            # precisely that experiment, and still does not flip this flag.
            seed=False,
            deterministic=False,
            negative_prompt=False,
            transparent=False,
            reference_images=self.profile["refs"],
            batch=False,
            sizes=self.profile["sizes"],
            cost_per_image=_cost(self.profile, tier),
            notes=self._notes(tier),
        )

    def generate(self, request):
        """Render `request` and return PNG bytes, one entry per image returned.

        One HTTP call, and deliberately no loop over request.count: batch=False
        already tells the caller it has to loop, and looping here as well would
        turn a count of 4 into 16 renders at 6.7 cents each.

        request.negative and request.transparent have no destination on this
        provider at all. They are not dropped silently -- capabilities declares
        both False, base.preflight raises the waiver, and base.strip blanks them
        before anything reaches here.
        """
        unknown = sorted(set(request.options) - OPTIONS)
        if unknown:
            raise UnsupportedOption(
                "gemini does not accept %s; accepted keys: %s"
                % (", ".join(unknown), ", ".join(sorted(OPTIONS))))
        options = dict(self.defaults, **request.options)   # per call wins
        _interactions_only(self.surface,
                           {k: options[k] for k in ("store", "thinking_level")
                            if k in options})
        thinking = options.get("thinking_level", self.thinking_level)
        _check_thinking(self.model, self.profile, thinking)

        aspect = self._aspect(options.get("aspect_ratio"), request.size)
        tier = self._tier(options.get("image_size"), request.size)
        refs = self._refs(request.refs)
        seed = options.get("seed")
        system = options.get("system_instruction", self.system_instruction)
        store = _as_bool(options["store"]) if "store" in options else self.store

        if self.surface == "interactions":
            url = "%s/%s/interactions" % (self.base_url, self.api_version)
            body = _interaction_body(self.model, request.prompt, refs, aspect,
                                     tier, store, thinking, seed, system)
        else:
            url = "%s/%s/models/%s:generateContent" % (self.base_url,
                                                       self.api_version,
                                                       self.model)
            body = _content_body(request.prompt, refs, aspect, tier, seed, system)

        doc = _post(url, body, _api_key(self.api_key), self.timeout)
        images = (_from_interaction(doc) if self.surface == "interactions"
                  else _from_candidates(doc))
        if not images:
            raise BackendError("gemini returned no image block; body was %s"
                               % json.dumps(doc)[:400])
        # to_png unconditionally: the only documented response mime type is JPEG,
        # and cutout() downstream keys on exactly the ringing JPEG leaves behind.
        return [to_png(data) for data in images]

    @classmethod
    def probe(cls, api_key=None, model=DEFAULT_MODEL, surface="interactions",
              base_url=BASE_URL, api_version=None, store=None,
              system_instruction=None, thinking_level=None, timeout=60):
        """Credentials and model reach, from models.list. Renders nothing.

        A probe that generated would bill: every image model on this provider
        reads "Not available" under Free Tier, so the cheapest possible test
        render still costs 3.4 cents, and a paid backend whose probe costs money
        is a trap. Listing models is free and answers the question that actually
        bites, which is that model availability varies by tier.

        The other constructor options are accepted so probe() takes the same
        table, then ignored: nothing about them can be confirmed without spending
        a generation.
        """
        try:
            model = str(model or DEFAULT_MODEL)
            _reject_imagen(model)
            key = _api_key(api_key)
        except BackendError as exc:
            return False, str(exc)

        url = "%s/%s/models" % (str(base_url or BASE_URL).rstrip("/"),
                                PROBE_VERSION)
        try:
            doc = request_json(url, headers=_headers(key), timeout=timeout,
                               retries=1)
        except BackendError as exc:
            return False, "%s: %s" % (url, exc)

        # The models.list response shape was not part of the research this module
        # was written from, so it is read defensively: an unrecognised shape
        # degrades to "cannot confirm" rather than to a false negative.
        names = []
        for entry in doc.get("models") or []:
            name = entry.get("name") if isinstance(entry, dict) else None
            if isinstance(name, str):
                names.append(name.split("/")[-1])
        if names and model not in names:
            return False, ("key accepted, but %s is not among the %d models it "
                           "can reach -- image models need a billing-enabled "
                           "project" % (model, len(names)))
        if not names:
            return True, ("%s answered, but listed no models this client could "
                          "read" % url)
        return True, ("%s reachable, %d models visible, %s among them"
                      % (url, len(names), model))

    # --- request shaping ------------------------------------------------

    def _aspect(self, requested, size):
        """The aspect-ratio enum to send. There is no width/height parameter."""
        allowed = self.profile["aspects"]
        if requested is not None:
            if requested not in allowed:
                raise UnsupportedOption("aspect_ratio=%r: %s offers %s"
                                        % (requested, self.model,
                                           ", ".join(allowed)))
            return requested
        want = size[0] / float(size[1])
        return min(allowed, key=lambda a: abs(_ratio(a) - want))

    def _tier(self, requested, size):
        """The image_size tier to send, or None where the model has no control."""
        tiers = self.profile["tiers"]
        if requested is not None:
            if not tiers:
                raise UnsupportedOption("image_size=%r: %s has no size control "
                                        "at all" % (requested, self.model))
            value = str(requested)
            # "512px" is accepted and forwarded verbatim: Google's own guide
            # spells the smallest tier that way where the Interactions reference
            # and both discovery documents say "512". Only the server knows which
            # spelling it wants today, so a caller who hits the wrong one can send
            # the other without editing this file.
            canonical = value[:-2] if value.endswith("px") else value
            if canonical not in tiers:
                raise UnsupportedOption(
                    "image_size=%r: %s offers %s, and uppercase K is mandatory"
                    % (requested, self.model, ", ".join(tiers)))
            return value
        if not tiers:
            return None
        # Nearest tier, not smallest-that-fits, and an exact tie resolves down:
        # postprocess.render downscales to 128 px anyway, and 4K bills 2.3x what
        # 1K does for pixels nobody keeps.
        px = max(size)
        return min(tiers, key=lambda t: (abs(_TIER_PX[t] - px), _TIER_PX[t]))

    def _refs(self, refs):
        """(mime, base64) per reference image, capped at what the model takes."""
        cap = self.profile["refs"]
        if len(refs) > cap:
            raise BackendError(
                "%d reference images, but %s accepts at most %d%s"
                % (len(refs), self.model, cap,
                   "" if self.model in MODELS else
                   " (a conservative guess -- that model id is not in the table)"))
        return [(_mime(data), base64.b64encode(data).decode("ascii"))
                for data in refs]

    def _notes(self, tier):
        profile = self.profile
        notes = []
        if self.model not in MODELS:
            notes.append(
                "%s is not in the model table this backend was written against "
                "(2026-08-20), so the numbers above are a conservative guess "
                "rather than a claim. Unknown ids are allowed on purpose: this "
                "line shipped five model generations in nine months."
                % self.model)
        if profile["style_refs"]:
            notes.append(
                "up to %d of the %d reference images can act as STYLE references, "
                "which is the strongest consistency lever this provider has and "
                "the reason this backend needs no seed. Roles are inferred from "
                "the prompt text: there is no per-image role field, no weight, no "
                "strength, no ControlNet."
                % (profile["style_refs"], profile["refs"]))
        else:
            notes.append(
                "%s has NO style-reference slot. gemini-3.1-flash-image is the "
                "only model that does (3 of its 14). The %d reference images here "
                "still land, as object references -- so switching to the cheap "
                "Lite variant to save a few cents an icon silently discards the "
                "entire consistency mechanism this backend has instead of a seed."
                % (self.model, profile["refs"]))
        notes.append(
            "no Gemini image model generates a transparent background, and the "
            "response mime enum holds exactly one value, IMAGE_JPEG, so every "
            "buffer is transcoded by postprocess.to_png(). Inferred from that "
            "enum and NOT measured: JPEG ringing around a hard outline is the "
            "same signal cutout()'s thresh=42 keys on, so a dark charcoal "
            "backdrop may key out worse here than off a local backend. Measure "
            "it, and try a flat pure white or pure black backdrop, which survives "
            "JPEG better.")
        notes.append(
            "every image carries an invisible SynthID watermark, from every "
            "model, with no opt-out. Worth saying out loud before these ship as "
            "product icons.")
        if self.surface == "interactions":
            if self.store:
                notes.append(
                    "store=true: Google retains the prompt and the image for 55 "
                    "days on the paid tier, 1 day on free.")
            else:
                notes.append(
                    "store=false is sent, so Google retains neither prompt nor "
                    "image -- the wire default is true, for 55 days. The cost is "
                    "server-side continuation: previous_interaction_id requires "
                    "store=true, so turning storage off turns iterative "
                    "refinement off with it.")
        cost = _cost(profile, tier)
        if cost is not None:
            notes.append(
                "no image model here has a free tier, so this backend bills "
                "before it produces a pixel: about USD %.4f per image at %s. That "
                "is a floor, not a total -- thinking cannot be disabled on Gemini "
                "3 image models and thinking tokens are billed on top."
                % (cost, tier or "the default size"))
        return tuple(notes)


# --- bodies -------------------------------------------------------------

def _interaction_body(model, prompt, refs, aspect, tier, store, thinking, seed,
                      system):
    """Interactions API body: snake_case throughout, model id inside the JSON.

    response_format is a single object rather than a [text, image] array on
    purpose. One object suppresses the conversational text and returns the image
    alone, which is all this pipeline wants, and it sidesteps an unverified
    community report that ["IMAGE","TEXT"] pins output to 1K whatever image_size
    asks for.
    """
    fmt = {"type": "image", "aspect_ratio": aspect}
    if tier:
        fmt["image_size"] = tier
    body = {"model": model,
            "input": [{"type": "text", "text": prompt}],
            "response_format": fmt,
            "store": bool(store)}
    for mime, data in refs:
        body["input"].append({"type": "image", "mime_type": mime, "data": data})
    config = {}
    if thinking:
        config["thinking_level"] = thinking
    if seed is not None:
        config["seed"] = int(seed)
    if config:
        body["generation_config"] = config
    if system:
        # Documented as a plain string on both surfaces. Worth knowing for a
        # future continuation feature: previous_interaction_id does NOT carry
        # system_instruction, tools or generation_config forward, so every turn
        # has to resend all three.
        body["system_instruction"] = system
    return body


def _content_body(prompt, refs, aspect, tier, seed, system):
    """generateContent body: camelCase, except inside parts.

    inline_data and mime_type are snake_case here because that is how the guide's
    own REST examples spell them -- Google's proto-JSON layer accepts both inside
    parts, and nowhere else.

    responseFormat.image is the current spelling of aspect ratio and size.
    generationConfig.imageConfig is an older alias for the same two fields and is
    still live in both discovery documents; if some model ever rejects the new
    field, that is the fallback to try.
    """
    parts = [{"text": prompt}]
    for mime, data in refs:
        parts.append({"inline_data": {"mime_type": mime, "data": data}})
    image = {"aspectRatio": aspect}
    if tier:
        image["imageSize"] = tier
    config = {"responseModalities": ["IMAGE"], "responseFormat": {"image": image}}
    if seed is not None:
        config["seed"] = int(seed)
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": config}
    if system:
        body["systemInstruction"] = system
    return body


# --- responses ----------------------------------------------------------

def _from_interaction(doc):
    """steps[] -> model_output -> content[] -> image. See the module docstring."""
    status = doc.get("status")
    if status != "completed":
        text = ("interaction %s ended with status %r: %s"
                % (doc.get("id") or "?", status, json.dumps(doc)[:300]))
        if status == "budget_exceeded":
            raise PaymentRequired(text + "\n  fatal for the batch, not just "
                                         "this icon")
        raise BackendError(text)
    out = []
    for step in doc.get("steps") or []:
        # Anything that is not model_output is skipped, which is what keeps the
        # up-to-two interim "thought" images out of the icon set.
        if step.get("type") != "model_output":
            continue
        for block in step.get("content") or []:
            if block.get("type") == "image" and block.get("data"):
                out.append(base64.b64decode(block["data"]))
    return out


def _from_candidates(doc):
    """candidates[].content.parts[] -> inlineData. A text part usually leads."""
    out = []
    for candidate in doc.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if part.get("thought"):
                continue
            blob = part.get("inlineData") or part.get("inline_data")
            if isinstance(blob, dict) and blob.get("data"):
                out.append(base64.b64decode(blob["data"]))
    return out


# --- plumbing -----------------------------------------------------------

def _post(url, body, key, timeout):
    """request_json, plus the two provider facts _http cannot know.

    _http already maps 401/403 to AuthError and 429 to RateLimited. What it cannot
    say is why they happen here: on this provider a rejected key is more often a
    project without billing than a wrong key, and a 429 is as likely to be the
    rolling spend cap as it is images per minute.
    """
    try:
        return request_json(url, body, headers=_headers(key), timeout=timeout)
    except RateLimited as exc:
        raise RateLimited("%s\n  %s" % (exc, _RATE_HINT),
                          retry_after=exc.retry_after) from exc
    except AuthError as exc:
        raise AuthError("%s\n  %s" % (exc, _AUTH_HINT)) from exc


def _headers(key):
    # Header rather than the ?key= query parameter the older REST reference
    # examples still show. Both work; only one keeps the key out of proxy and
    # access logs.
    return {"x-goog-api-key": key}


def _api_key(explicit):
    if explicit:
        return str(explicit)
    for name in KEY_ENVS:
        value = os.environ.get(name)
        if value:
            return value
    raise AuthError("no Gemini API key: pass -O api_key=... or set %s"
                    % " or ".join(KEY_ENVS))


def _surface(value):
    """Normalise, so generate_content and generate-content also land."""
    name = str(value or "interactions").lower().replace("_", "").replace("-", "")
    if name not in API_VERSION:
        raise UnsupportedOption("surface=%r: use %s"
                                % (value, " or ".join(sorted(API_VERSION))))
    return name


def _reject_imagen(model):
    if model.lower().startswith("imagen"):
        raise BackendError(
            "%s is gone: every Imagen 4 id on the Gemini Developer API reached "
            "its published shutdown date on %s, and models.predict never existed "
            "on v1 (stable) at all. Google's deprecations table names "
            "gemini-3.1-flash-image as the replacement for all three."
            % (model, IMAGEN_SHUTDOWN))


def _interactions_only(surface, values):
    if surface == "interactions":
        return
    named = sorted(k for k, v in values.items() if v is not None)
    if named:
        raise UnsupportedOption(
            "%s: Interactions-only field(s), and this instance is on "
            "generatecontent. generateContent retains nothing, so it has no "
            "store flag, and thinking_level is documented for Interactions "
            "alone." % ", ".join(named))


def _check_thinking(model, profile, value):
    if value is None:
        return
    if not profile["thinking"]:
        raise UnsupportedOption(
            "thinking_level=%r: documented for gemini-3.1-flash-image only, and "
            "this instance is %s. Thinking still runs there -- it cannot be "
            "disabled on any Gemini 3 image model -- you just cannot set the "
            "level." % (value, model))
    if value not in ("minimal", "high"):
        raise UnsupportedOption(
            "thinking_level=%r: the image guide gives minimal or high for "
            "gemini-3.1-flash-image. The general Interactions reference also "
            "lists low and medium, unconfirmed for image models." % (value,))


def _cost(profile, tier):
    prices = profile["cost"]
    if not prices:
        return None
    return prices.get(tier) or prices.get("1K")


def _ratio(aspect):
    left, _, right = aspect.partition(":")
    return float(left) / float(right)


def _mime(data):
    """Sniff a reference image's type, because the wire field is not optional.

    Accepted input types run png, jpeg, webp, heic, heif, gif, bmp, tiff. Only the
    four worth sniffing are here; anything else is labelled png, which is what
    devgraphics itself always hands over.
    """
    if data[:8] == bytes((0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)):
        return "image/png"
    if data[:3] == bytes((0xFF, 0xD8, 0xFF)):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


def _prune(values):
    """Drop the keys that were not asked for.

    None has to stay distinguishable from a real value: `seed=None` means the
    caller said nothing, and putting it in the defaults would shadow a per-call
    seed with a null.
    """
    return dict((k, v) for k, v in values.items() if v is not None)


def _as_bool(value):
    """-O store=false arrives as a string, and bool("false") is True."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)
