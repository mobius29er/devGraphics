"""
OpenAI's GPT Image models over plain HTTP: /v1/images/generations and /edits.

Four facts about this API shape everything below.

**There is no seed.** Not on /v1/images/generations, not on /v1/images/edits, not
on the Responses image_generation tool -- the word does not occur in the schemas.
The fixed-seed style lock iconset.py leans on has no analogue here, so `seed=False`
in Capabilities is a measured fact rather than caution, and the substitute the docs
themselves name is the other one: feed an already-approved icon back in as a
reference through /v1/images/edits with `input_fidelity: "high"`. That anchor path
is implemented here.

**Capabilities are model-gated inside one client class.** `gpt-image-2` -- the
newest model, and the default on /edits -- rejects `background: "transparent"`
outright, while gpt-image-1, 1.5 and 1-mini honour it. `input_fidelity` is
accepted by 1 and 1.5, rejected by 1-mini, and must be omitted for 2 (which forces
high fidelity and bills for it). Arbitrary WIDTHxHEIGHT sizes are gpt-image-2 only,
so the flexible-size model and the transparent-capable models are disjoint sets.
All of that lives in MODELS, keyed by model id; an id that is not in the table gets
the conservative row and a note saying so, because a generous guess here is a batch
of 88 opaque squares.

**`response_format` is a hard 400, not a no-op.** gpt-image-* answers "Unknown
parameter", and LiteLLM, Genkit and the Azure SDK each shipped that exact bug. The
key is never sent -- not even as null -- which is also why both `b64_json` and
`url` are read on the way back: a legacy dall-e model left to its own default
answers with a url.

**`background: "transparent"` is a request, not a guarantee.** The parameter is
documented; the only evidence it produces real alpha headlessly is an OpenAI forum
thread where reporters needed the prompt to demand a hard cutout as well. Every
transparent render is checked with postprocess.has_alpha() and a miss is reported,
so render_bytes() falls back to the flood fill instead of writing an opaque square.

`model` is always sent explicitly: /v1/images/generations defaults to `dall-e-2`
"unless a parameter specific to the GPT image models is used", and /edits defaults
to `gpt-image-1.5`, so omitting it silently routes to a different model per
endpoint. Nothing here touches the network from __init__ or from `capabilities`:
the table is static, so --dry-run costs nothing and probe() generates no image.

Facts, prices and quotes are from the OpenAI API docs, the OpenAPI spec (2.3.0)
and the model pages as fetched 2026-08-20; those pages carry no publication date.
"""

import base64
import os

from .. import _http, postprocess
from .base import (AuthError, BackendError, Capabilities, ModerationBlocked,
                   PaymentRequired, RateLimited, UnsupportedOption, nearest_size)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_KEY_ENV = "OPENAI_API_KEY"

#: gpt-image-1.5 is the docs' own recommendation for transparent icons and the
#: cheapest model that also accepts input_fidelity, which the anchor path needs.
DEFAULT_MODEL = "gpt-image-1.5"

#: "auto" is legal and is the API default, but then cost_per_image is unknowable
#: until the bill arrives. medium is named explicitly so a batch can be budgeted.
#: The gpt-image-1-era claim that transparency "works best with medium or high"
#: is gone from the current docs and is UNVERIFIED; it is not why this is medium.
DEFAULT_QUALITY = "medium"

#: The API default is "low". high is the entire style-strength surface -- there is
#: no ControlNet, no IP-Adapter, no LoRA -- so the anchor path opts into it and
#: says so in a note, because it raises the input-token bill on every icon.
DEFAULT_INPUT_FIDELITY = "high"

#: Documented latency reaches two minutes on a complex prompt, so 300s is not a
#: ceiling worth defending. This is a socket timeout and is independent of the
#: 429 backoff in _http, which sleeps between attempts rather than inside one.
DEFAULT_TIMEOUT = 600

#: Tier 1 is five images a minute, so a 429 during an 88-icon batch is normal
#: operation. _http honours Retry-After up to 120s; six attempts covers a full
#: rate-limit window plus slack without turning a dead key into a long hang.
DEFAULT_RETRIES = 6

GENERATIONS = "/images/generations"
EDITS = "/images/edits"
FILES = "/files"
MODEL_LIST = "/models"

MAX_REFS = 16
MAX_N = 10

#: Every option key accepted, from the constructor or from a request's `options`.
#: Anything else is UnsupportedOption: a typo'd -O must fail loudly rather than
#: vanish and quietly change nothing.
OPTIONS = frozenset([
    "api_key", "api_key_env", "base_url", "model", "quality", "output_format",
    "input_fidelity", "moderation", "ref_file_ids", "timeout", "retries",
])

FORMATS = ("png", "jpeg", "webp")
QUALITIES = ("low", "medium", "high", "auto")
MODERATIONS = ("auto", "low")
FIDELITIES = ("high", "low")

#: The only sizes gpt-image-1 / 1.5 / 1-mini accept, and the only ones the JSON
#: body of /v1/images/edits documents for any model. "auto" is deliberately not
#: offered: Capabilities.sizes is integer pairs, and a caller who cannot name the
#: size cannot budget for it either.
FIXED_SIZES = ((1024, 1024), (1536, 1024), (1024, 1536))

#: gpt-image-2 also takes arbitrary WIDTHxHEIGHT -- edges a multiple of 16, max
#: edge 3840, aspect within 1:3..3:1, 655,360 to 8,294,400 pixels. Only the sizes
#: the docs list by name are offered, because snapping an arbitrary request into
#: that rule set is guesswork; a caller who wants 2048x1152 can ask for it.
GPT_IMAGE_2_SIZES = FIXED_SIZES + ((2048, 2048), (2048, 1152),
                                   (3840, 2160), (2160, 3840))

#: model id -> what that model can actually do. `input_fidelity` is three-valued
#: because the API is: "send" (gpt-image-1, 1.5), "reject" (1-mini, an error),
#: "forced" (gpt-image-2 always works at high fidelity and rejects the key).
#: `cost` is USD for one 1024x1024 image at that quality, from the guide's cost
#: table as fetched 2026-08-20. Non-square sizes are priced separately and are not
#: carried here -- Capabilities holds one number and a square is the icon case.
#: For gpt-image-2 the non-square rows are genuinely cheaper than the square one;
#: the docs confirm that is real behaviour, not a typo in the source table.
MODELS = {
    "gpt-image-2": {
        "transparent": False,
        "input_fidelity": "forced",
        "sizes": GPT_IMAGE_2_SIZES,
        "refs": MAX_REFS,
        "cost": {"low": 0.006, "medium": 0.053, "high": 0.211},
    },
    "gpt-image-1.5": {
        "transparent": True,
        "input_fidelity": "send",
        "sizes": FIXED_SIZES,
        "refs": MAX_REFS,
        "cost": {"low": 0.009, "medium": 0.034, "high": 0.133},
    },
    "gpt-image-1": {
        "transparent": True,
        "input_fidelity": "send",
        "sizes": FIXED_SIZES,
        "refs": MAX_REFS,
        "cost": {"low": 0.011, "medium": 0.042, "high": 0.167},
    },
    "gpt-image-1-mini": {
        "transparent": True,
        "input_fidelity": "reject",
        "sizes": FIXED_SIZES,
        "refs": MAX_REFS,
        "cost": {"low": 0.005, "medium": 0.011, "high": 0.036},
    },
}

#: Dated snapshots share their base model's row. Listed rather than matched on a
#: prefix: "gpt-image-1-mini" would prefix-match "gpt-image-1", and a snapshot
#: published after this table was written has nothing to inherit anyway.
ALIASES = {
    "gpt-image-2-2026-04-21": "gpt-image-2",
    "gpt-image-1.5-2025-12-16": "gpt-image-1.5",
}

#: What an unrecognised model id gets: no transparency, no input_fidelity, one
#: reference image, the three sizes every image model here accepts, no price.
UNKNOWN = {"transparent": False, "input_fidelity": "reject",
           "sizes": FIXED_SIZES, "refs": 1, "cost": {}}

VERIFY_HINT = (
    "OpenAI may require API Organization Verification before any GPT Image model "
    "works, including gpt-image-2, gpt-image-1.5, gpt-image-1 and gpt-image-1-mini.\n"
    "  It is a manual identity step in the developer console -- no retry and no "
    "second key gets past it:\n"
    "  https://help.openai.com/en/articles/10910291-api-organization-verification")


class OpenAIBackend(object):
    """GPT Image over /v1/images/generations, or /edits when the request has refs."""

    def __init__(self, **options):
        _check_options(options)
        self.options = dict(options)
        self._caps = None
        #: None until a transparent render has been checked, then True/False. The
        #: caller reads it to explain why an icon went through the flood fill.
        self.transparency_verified = None

    # --- capabilities ---------------------------------------------------

    @property
    def capabilities(self):
        """Answerable with the server off, and with no key present.

        This describes the *configured* instance. A request that overrides `model`
        in its options is still checked against that model's row before anything
        is sent, so nothing illegal goes out; only the preflight report printed
        before the batch describes the instance rather than that one call.
        """
        if self._caps is None:
            self._caps = self._build_caps()
        return self._caps

    def _build_caps(self):
        model = self.options.get("model", DEFAULT_MODEL)
        quality = self.options.get("quality", DEFAULT_QUALITY)
        row, known = _row(model)
        cost = row["cost"].get(quality)

        notes = []
        if not known:
            notes.append(
                "model %r is not in this backend's table (built 2026-08-20), so "
                "the conservative row is used: no transparency, no "
                "input_fidelity, one reference image, no price" % model)
        notes.append(
            "no OpenAI image endpoint has a seed parameter; consistency has to "
            "come from one shared prompt scaffold plus an anchor icon sent "
            "through /v1/images/edits")
        if row["transparent"]:
            notes.append(
                "background=transparent is documented, but the only evidence it "
                "yields real alpha headlessly is one OpenAI forum thread -- "
                "UNVERIFIED. Every render is checked with has_alpha() and a miss "
                "is reported so the flood-fill cutout takes over")
        else:
            notes.append(
                "%s rejects background=transparent, so postprocess.cutout keys "
                "the backdrop out instead" % model)
        if row["input_fidelity"] == "send":
            notes.append(
                "reference renders send input_fidelity=high (the API default is "
                "low). It is the only style-strength knob there is, and it raises "
                "the input-token bill on every icon; -O input_fidelity=low turns "
                "it down")
        elif row["input_fidelity"] == "reject":
            notes.append(
                "%s does not accept input_fidelity, so a reference image carries "
                "only as much style as the default low setting" % model)
        else:
            notes.append(
                "%s always reads references at high fidelity and rejects "
                "input_fidelity, so the surcharge applies whether or not it is "
                "wanted" % model)
        if row["sizes"] != FIXED_SIZES:
            notes.append(
                "the JSON body of /v1/images/edits documents only 1024x1024, "
                "1536x1024 and 1024x1536, so a reference render snaps to those "
                "even where %s accepts more on /v1/images/generations" % model)
        notes.append(
            "an anchor can be uploaded once with upload_reference() and reused as "
            "-O ref_file_ids=file-...; otherwise its base64 is re-sent with every "
            "icon in the set")
        notes.append(
            "throughput is rated in images per minute (5/min at tier 1), so a 429 "
            "part way through a batch is normal and is backed off, not raised")
        if cost is None:
            notes.append(
                "quality=%r has no published per-image price here, so the cost of "
                "a batch cannot be estimated before it runs" % quality)

        return Capabilities(
            name="openai/%s" % model,
            seed=False,
            deterministic=False,
            negative_prompt=False,
            transparent=row["transparent"],
            reference_images=row["refs"],
            batch=True,                       # n is 1-10 in one call
            sizes=row["sizes"],
            cost_per_image=cost,
            notes=tuple(notes),
        )

    # --- generation -----------------------------------------------------

    def generate(self, request):
        opts = self._merged(request.options)
        model = opts.get("model", DEFAULT_MODEL)
        row = _row(model)[0]
        body, path = self._body(request, opts, model, row)
        doc = self._post(path, body, opts)
        images = self._images(doc, opts)
        self._check_alpha(doc, request, row, model, images)
        return images

    def _body(self, request, opts, model, row):
        file_ids = _as_ids(opts.get("ref_file_ids"))
        editing = bool(request.refs) or bool(file_ids)

        prompt = request.prompt
        if request.negative:
            # This API has no negative-prompt parameter. preflight normally strips
            # the field and tells the caller to phrase exclusions affirmatively; a
            # direct generate() would otherwise drop it in silence, so it is
            # folded into the prompt the way the docs suggest.
            prompt = "%s. Avoid: %s" % (prompt.rstrip(". "), request.negative)

        fmt = opts.get("output_format", "png")
        if fmt not in FORMATS:
            raise BackendError("output_format must be one of %s, not %r"
                               % (", ".join(FORMATS), fmt))
        if request.transparent and fmt == "jpeg":
            raise BackendError("jpeg carries no alpha channel; with "
                               "background=transparent output_format must be png "
                               "or webp")

        sizes = FIXED_SIZES if editing else row["sizes"]
        width, height = _pick_size(sizes, request.size)

        body = {
            "model": model,                   # never omitted: see module docstring
            "prompt": prompt,
            "n": min(max(int(request.count), 1), MAX_N),
            "size": "%dx%d" % (width, height),
            "output_format": fmt,
        }

        quality = opts.get("quality", DEFAULT_QUALITY)
        if quality:
            if quality not in QUALITIES:
                raise BackendError("quality must be one of %s, not %r (hd and "
                                   "standard are dall-e only)"
                                   % (", ".join(QUALITIES), quality))
            body["quality"] = quality

        moderation = opts.get("moderation")
        if moderation:
            if moderation not in MODERATIONS:
                raise BackendError("moderation must be one of %s, not %r"
                                   % (", ".join(MODERATIONS), moderation))
            body["moderation"] = moderation

        if request.transparent and row["transparent"]:
            body["background"] = "transparent"

        if editing:
            body["images"] = _image_refs(request.refs, file_ids)
            fidelity = opts.get("input_fidelity")
            if fidelity and fidelity not in FIDELITIES:
                raise BackendError("input_fidelity must be high or low, not %r"
                                   % fidelity)
            gate = row["input_fidelity"]
            if gate == "send":
                body["input_fidelity"] = fidelity or DEFAULT_INPUT_FIDELITY
            elif gate == "reject" and fidelity:
                raise BackendError(
                    "%s rejects input_fidelity; drop the option or switch to "
                    "gpt-image-1.5 or gpt-image-1" % model)
            # "forced": gpt-image-2 reads every input at high fidelity and errors
            # on the key, so an explicit request for it is met, not refused.

        # response_format is never set on either path. Not as None either: a
        # serialiser that emitted it as null would earn the same HTTP 400.
        return body, EDITS if editing else GENERATIONS

    def _images(self, doc, opts):
        items = doc.get("data") or []
        if not items:
            raise BackendError("OpenAI returned no image: %s" % str(doc)[:200])
        out = []
        for n, item in enumerate(items):
            payload = item.get("b64_json")
            if payload:
                raw = base64.b64decode(payload)
            elif item.get("url"):
                # url is documented as unsupported for gpt-image-*, which always
                # inlines base64. It still turns up, because response_format is
                # never sent and a legacy dall-e model then falls back to its own
                # default of url. No auth header: it is a pre-signed link.
                raw = _http.request_bytes(
                    item["url"], timeout=_int(opts, "timeout", DEFAULT_TIMEOUT),
                    retries=2)
            else:
                raise BackendError("image %d carries neither b64_json nor url: %s"
                                   % (n, str(item)[:200]))
            out.append(postprocess.to_png(raw))
        return out

    def _check_alpha(self, doc, request, row, model, images):
        """Pixels decide, not the echoed field -- see the module docstring."""
        self.transparency_verified = None
        if not (request.transparent and row["transparent"]):
            return
        self.transparency_verified = all(postprocess.has_alpha(png)
                                         for png in images)
        if not self.transparency_verified:
            print("warn  openai/%s: background=transparent came back opaque "
                  "(server echoed background=%r); postprocess.cutout keys the "
                  "backdrop out instead" % (model, doc.get("background")))

    # --- transport ------------------------------------------------------

    def _post(self, path, body, opts):
        try:
            return _http.request_json(
                "%s%s" % (_base(opts), path), body,
                headers={"Authorization": "Bearer %s" % _key(opts)},
                timeout=_int(opts, "timeout", DEFAULT_TIMEOUT),
                retries=_int(opts, "retries", DEFAULT_RETRIES))
        except BackendError as exc:
            hint = _verification_hint(exc)
            if hint is None:
                raise                 # 429, 402 and moderation stay as classified
            raise AuthError(hint) from exc

    def _merged(self, request_options):
        _check_options(request_options or {})
        merged = dict(self.options)
        merged.update(request_options or {})
        return merged

    # --- anchors --------------------------------------------------------

    def upload_reference(self, data, filename="anchor.png", **options):
        """Upload one reference and get a file id back, to send it once per set
        rather than once per icon.

        The consuming half is documented: `images: [{"file_id": "file-..."}]` on
        /v1/images/edits. The upload half is only half documented -- the research
        this backend was written from names POST /v1/files with purpose "vision"
        but not the multipart field names, so `file` and the `id` in the reply are
        the Files API's long-standing shape and are UNVERIFIED here. Nothing on
        the default path calls this; a set that never calls it just pays the
        base64 inflation instead.
        """
        opts = self._merged(options)
        doc = _http.post_multipart(
            "%s%s" % (_base(opts), FILES),
            {"purpose": "vision"}, {"file": (filename, data)},
            headers={"Authorization": "Bearer %s" % _key(opts)},
            timeout=_int(opts, "timeout", DEFAULT_TIMEOUT))
        file_id = doc.get("id")
        if not file_id:
            raise BackendError("upload returned no file id: %s" % str(doc)[:200])
        return file_id

    # --- probe ----------------------------------------------------------

    @classmethod
    def probe(cls, **options):
        """Reachability and credentials, generating nothing.

        GET /models is free and returns no image, which matters on a backend where
        a probe that rendered something would cost real money on every --dry-run.
        It cannot prove the organisation is verified for GPT Image models -- that
        only surfaces on a real generation -- so verification is named when a
        failure mentions it, never asserted on success.
        """
        _check_options(options)
        try:
            key = _key(options)
        except AuthError as exc:
            return False, str(exc)
        try:
            _http.request_json("%s%s" % (_base(options), MODEL_LIST),
                               headers={"Authorization": "Bearer %s" % key},
                               timeout=30, retries=1)
        except BackendError as exc:
            return False, _verification_hint(exc) or str(exc)
        model = options.get("model", DEFAULT_MODEL)
        known = _row(model)[1]
        return True, ("reachable, key accepted, model %s%s"
                      % (model, "" if known else
                         " (not in this backend's table -- conservative "
                         "capabilities apply)"))


# --- helpers ------------------------------------------------------------

def _row(model):
    """(facts, is_known) for a model id. Snapshots resolve to their base model."""
    row = MODELS.get(ALIASES.get(model, model))
    if row is None:
        return UNKNOWN, False
    return row, True


def _pick_size(sizes, size):
    """nearest_size() takes a Capabilities, but the edits path has to snap to a
    different list from the one capabilities advertises, so it gets a stand-in."""
    return nearest_size(Capabilities(name="openai", sizes=sizes), size)


def _image_refs(refs, file_ids):
    """`images` for /v1/images/edits: ids where the anchor was uploaded once,
    base64 data URLs otherwise. Each item is exactly one of the two."""
    items = [{"file_id": fid} for fid in file_ids]
    room = max(MAX_REFS - len(items), 0)     # over 16 is a 400; preflight warned
    for data in tuple(refs)[:room]:
        items.append({"image_url": "data:%s;base64,%s"
                                   % (_mime(data),
                                      base64.b64encode(data).decode("ascii"))})
    if not items:
        raise BackendError("the edits path needs at least one reference image")
    return items


def _mime(data):
    """png, webp or jpg is all /v1/images/edits takes, and a data URL has to
    declare which; sniff rather than trust a caller holding only bytes."""
    if data[:8] == postprocess.PNG_MAGIC:
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise BackendError("reference image is not png, jpeg or webp; "
                       "/v1/images/edits accepts nothing else")


def _as_ids(value):
    """-O only ever delivers a string, so accept one id, a comma list, or a
    sequence that came out of a config profile."""
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(value)


def _check_options(options):
    unknown = sorted(set(options) - OPTIONS)
    if unknown:
        raise UnsupportedOption(
            "openai backend has no option %s; it accepts %s"
            % (", ".join(unknown), ", ".join(sorted(OPTIONS))))


def _base(opts):
    return (opts.get("base_url") or DEFAULT_BASE_URL).rstrip("/")


def _int(opts, name, default):
    value = opts.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise BackendError("%s must be a whole number of seconds, not %r"
                           % (name, value))


def _key(opts):
    env = opts.get("api_key_env") or DEFAULT_KEY_ENV
    key = opts.get("api_key") or os.environ.get(env)
    if not key:
        raise AuthError("no OpenAI API key: set %s, or pass -O api_key=sk-..." % env)
    return key


def _verification_hint(exc):
    """None unless this failure smells like the organisation-verification wall.

    Users hit that wall before they hit any code problem, and a bare 403 tells
    them nothing. The exact error code for it is not in any doc here, so matching
    the word is the best signal available -- which is why this only ever *adds* an
    explanation to a message that already mentions verification, and never
    reclassifies a 429, a 402 or a moderation refusal that _http already got right.
    """
    if isinstance(exc, (RateLimited, PaymentRequired, ModerationBlocked)):
        return None
    if "verif" not in str(exc).lower():
        return None
    return "%s\n  %s" % (exc, VERIFY_HINT)
