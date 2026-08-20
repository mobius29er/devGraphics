"""
Headless client for a local InvokeAI install (Community Edition), driven by
hand-built execution graphs because there is no other way in.

Know the cost before reading further. InvokeAI has no "generate this prompt"
route, and no route that runs a saved workflow either -- the workflows router is
CRUD only, and the request for an execute endpoint (#5719) has been open since
2024-02-14. Generation is one POST whose body carries a complete node/edge
execution graph, the same graph the React frontend compiles out of its UI state.
So this module builds an SDXL txt2img graph by hand: six nodes, ten edges, every
field name read off the v6.13.8 source. That is why this file is several times
the size of the Fooocus one, and that size is not accidental complexity waiting
to be tidied away.

What the graph buys is real. Exact pixel sizes, with no HTML-laden aspect-ratio
dropdown to substring-match. A seed whose noise tensor is built on CPU precisely
so it reproduces across Windows, macOS and Linux -- a better match for this
project's promise than anything Fooocus offers. And server-side batching:
batch.data expands one POST into N queue items.

The API is not a supported public surface and the maintainers say so plainly.
psychedelicious, 2024-09-30: "the HTTP API is not intended for public
consumption. We don't often make breaking changes to it, but we don't hesitate
to do so when it serves the needs of the application." The recommended discovery
method is literally browser dev tools. Everything here was read against tag
v6.13.8 with drift notes from 6.14.0-rc2, so probe() reports the server version
against that range instead of letting a renamed field surface as a 422 from
somewhere deep inside the graph.

That drift is not hypothetical: QUEUE_ITEM_STATUS gained "waiting" between 6.13
and 6.14. The poll loop below therefore enumerates TERMINAL statuses and treats
anything it does not recognise as still running. Written the other way round --
"not pending and not in_progress means done" -- it would read results that do not
exist yet from a 6.14 server.

Three more things bite, each handled at its site with a comment: session.results
is keyed by the executor's PREPARED node id, not the id you wrote; enqueue
truncates silently at max_queue_size and the only evidence is enqueued <
requested inside a 200 response; and every node caches its output by default in a
512-entry cache, so re-running an identical graph hands back the same image name
in milliseconds with no GPU work.

Alpha: l2i VAE-decodes to opaque RGB and carries no transparency parameter. Real
in-graph alpha exists (apply_mask_to_image merges a mask into RGBA) but costs
three or four more nodes plus a segmentation model installed on the server, to
reproduce what postprocess.cutout() already does in PIL with no dependencies.
capabilities says transparent=False and means it.
"""

import random
import time
import urllib.parse
import warnings

from .. import _http
from ..postprocess import to_png
from .base import BackendError, Capabilities, UnsupportedOption

DEFAULT_HOST = "127.0.0.1:9090"

#: DEFAULT_QUEUE_ID in session_queue_common.py, and the frontend hardcodes
#: /api/v1/queue/default/. Nothing in the API creates another queue, so there is
#: no discovery step -- just an override for a server that grew one.
DEFAULT_QUEUE_ID = "default"

ORIGIN = "devgraphics"

#: Read against v6.13.8; 6.14 is covered by the drift this module encodes (the
#: "waiting" status, mainly). Anything else gets a warning, not a refusal --
#: refusing would make this backend useless the week InvokeAI ships 6.15.
TESTED_VERSIONS = ((6, 13), (6, 14))

#: Terminal by enumeration, never by exclusion. See the module docstring.
TERMINAL_STATUSES = frozenset(("completed", "failed", "canceled"))
RUNNING_STATUSES = frozenset(("pending", "in_progress", "waiting"))

#: DenoiseLatentsInvocation's scheduler enum as of v6.13.8. Checked locally
#: because an unknown value comes back as a 422 about the whole graph body.
SCHEDULERS = frozenset((
    "ddim", "ddpm", "deis", "deis_k", "lms", "lms_k", "pndm", "heun", "heun_k",
    "euler", "euler_k", "euler_a", "kdpm_2", "kdpm_2_k", "kdpm_2_a",
    "kdpm_2_a_k", "dpmpp_2s", "dpmpp_2s_k", "dpmpp_2m", "dpmpp_2m_k",
    "dpmpp_2m_sde", "dpmpp_2m_sde_k", "dpmpp_3m", "dpmpp_3m_k", "dpmpp_sde",
    "dpmpp_sde_k", "er_sde", "unipc", "unipc_k", "lcm", "tcd"))

#: Options accepted by the constructor (a config profile's [options] table, or
#: -O key=value). Anything else raises UnsupportedOption: a typo'd -O must fail
#: loudly rather than silently change nothing.
OPTIONS = frozenset((
    "host", "queue_id", "token", "model", "model_hash", "steps", "cfg_scale",
    "scheduler", "use_cache", "board_id", "poll_interval", "timeout"))

#: The subset that also makes sense per Request. Connection-level keys (host,
#: token, queue_id, model) are deliberately absent -- they describe which server
#: this backend drives, not what one render should look like.
REQUEST_OPTIONS = frozenset((
    "steps", "cfg_scale", "scheduler", "use_cache", "board_id"))

#: The node's own default is 10, which is a preview-grade setting. 30 is this
#: module's choice, not a number the API documents; override with -O steps=.
DEFAULT_STEPS = 30
DEFAULT_CFG_SCALE = 7.5
DEFAULT_SCHEDULER = "euler"          # the node default; pinned for reproducibility

#: Per-request socket timeout, distinct from `timeout`, which is the wall-clock
#: budget for one whole generate() including every poll.
HTTP_TIMEOUT = 120

#: Only used to pick a default when the caller supplied no seed. The noise node
#: validator applies seed % (SEED_MAX + 1) server-side, and the research names
#: SEED_MAX without giving its value, so a seed the caller *did* supply is passed
#: through untouched rather than wrapped against a number we would be guessing.
RANDOM_SEED_BOUND = 0xFFFFFFFF

NODE_MODEL = "sdxl_model"
NODE_POS = "pos_cond"
NODE_NEG = "neg_cond"
NODE_NOISE = "noise"
NODE_DENOISE = "denoise"
NODE_L2I = "l2i"

#: Graph.validate_self checks unique ids, id/key agreement, edge endpoints, field
#: existence, acyclicity and type compatibility -- but NOT that a connection-only
#: field actually has an edge into it. A graph missing the vae edge therefore
#: enqueues perfectly cleanly and dies at run time as a failed queue item. So the
#: wiring is checked here, before it goes on the wire.
REQUIRED_CONNECTIONS = (
    (NODE_POS, "clip"),
    (NODE_POS, "clip2"),
    (NODE_NEG, "clip"),
    (NODE_NEG, "clip2"),
    (NODE_DENOISE, "unet"),
    (NODE_DENOISE, "positive_conditioning"),
    (NODE_DENOISE, "negative_conditioning"),
    (NODE_DENOISE, "noise"),
    (NODE_L2I, "latents"),
    (NODE_L2I, "vae"),
)


class InvokeAIBackend(object):
    """Drives one InvokeAI server over its REST API."""

    def __init__(self, **options):
        _check_options(options, OPTIONS, "constructor")
        self.host = options.get("host", DEFAULT_HOST)
        self.queue_id = options.get("queue_id", DEFAULT_QUEUE_ID)
        self.token = options.get("token")
        self.model_name = options.get("model")
        # BLAKE3, and the reason a get_by_hash route exists at all: `key` is a
        # UUID that changes when the user reinstalls the same checkpoint, `hash`
        # does not. So the hash is what a config file stores.
        self.model_hash = options.get("model_hash")
        self.steps = int(options.get("steps", DEFAULT_STEPS))
        self.cfg_scale = float(options.get("cfg_scale", DEFAULT_CFG_SCALE))
        self.scheduler = options.get("scheduler", DEFAULT_SCHEDULER)
        self.use_cache = bool(options.get("use_cache", True))
        self.board_id = options.get("board_id")
        self.poll_interval = float(options.get("poll_interval", 1.0))
        self.timeout = float(options.get("timeout", 900))
        _check_scheduler(self.scheduler)
        # Resolved on first use and cached. The constructor must not touch the
        # network: capabilities has to be answerable with the server switched
        # off, or --dry-run needs the very thing it exists to avoid.
        self._model = None
        self._version = None

    # --- contract -------------------------------------------------------

    @property
    def capabilities(self):
        return Capabilities(
            name="invokeai",
            seed=True,
            # Genuinely deterministic, and engineered to be: noise.use_cpu
            # defaults true with the source comment "Use CPU for noise generation
            # (for reproducible results across platforms)".
            deterministic=True,
            negative_prompt=True,
            transparent=False,
            reference_images=0,
            batch=True,
            sizes=(),             # any (w, h); generate() rounds to a multiple of 8
            cost_per_image=None,  # local GPU, no metering
            notes=(
                "graph shape is pinned to InvokeAI 6.13.x-6.14.x field names; the "
                "maintainers do not treat this API as a stable public surface",
                "node results are cached (512 entries), so an identical prompt and "
                "seed returns the same image without re-rendering -- pass -O "
                "use_cache=false, or DELETE /api/v1/app/invocation_cache",
                "count>1 is one enqueue with a per-image seed of seed+i, because N "
                "identical graphs would all hit that cache and return one image",
                "img2img, ControlNet, IP-Adapter and T2I-Adapter are all reachable "
                "through this API but are not wired up here, so reference_images=0",
            ),
        )

    def generate(self, request):
        """Enqueue one batch, poll it out, and return PNG bytes per image."""
        options = dict(request.options or {})
        _check_request_options(options)
        steps = int(options.get("steps", self.steps))
        cfg_scale = float(options.get("cfg_scale", self.cfg_scale))
        scheduler = options.get("scheduler", self.scheduler)
        use_cache = bool(options.get("use_cache", self.use_cache))
        board_id = options.get("board_id", self.board_id)
        _check_scheduler(scheduler)

        count = max(1, int(request.count))
        seed = request.seed
        if seed is None:
            seed = random.randint(0, RANDOM_SEED_BOUND)

        self._warn_on_untested_version()
        graph = self._build_graph(
            prompt=request.prompt,
            negative=request.negative or "",
            seed=int(seed),
            width=_multiple_of_8(request.size[0]),
            height=_multiple_of_8(request.size[1]),
            steps=steps,
            cfg_scale=cfg_scale,
            scheduler=scheduler,
            use_cache=use_cache,
            board_id=board_id,
        )
        _require_connections(graph)

        batch = {"graph": graph, "runs": 1, "origin": ORIGIN}
        if count > 1:
            # One POST expands server-side into `count` sessions. Varying the
            # seed is not cosmetic: with use_cache on, "runs": count would render
            # once and hand back the same image_name count times.
            batch["data"] = [[{"node_path": NODE_NOISE,
                               "field_name": "seed",
                               "items": [int(seed) + i for i in range(count)]}]]

        deadline = time.time() + self.timeout
        item_ids = self._enqueue(batch, count)
        out = []
        for item_id in item_ids:          # item_ids order matches the expansion
            name = self._await_image(item_id, deadline)
            out.append(self._fetch_image(name))
        return out

    @classmethod
    def probe(cls, **options):
        """Reachability, version and model resolution. Generates nothing.

        Deliberately two cheap GETs: /api/v1/app/version, then the SDXL model
        list. Never /openapi.json -- the generated TypeScript types alone are
        1.42 MB and the JSON is larger, because Graph.nodes inlines a union of
        roughly 250 invocation types.
        """
        try:
            backend = cls(**options)
        except (UnsupportedOption, ValueError) as exc:
            return False, str(exc)

        try:
            version = backend._fetch_version()
        except BackendError as exc:
            return False, "invokeai unreachable at %s: %s" % (backend.host, exc)

        notes = ["invokeai %s at %s" % (version, backend.host)]
        parsed = _version_tuple(version)
        if parsed is None or parsed not in TESTED_VERSIONS:
            notes.append("UNTESTED version -- this backend was built against %s "
                         "and the API carries no stability promise"
                         % _tested_range())
        try:
            model = backend._resolve_model()
        except BackendError as exc:
            return False, "; ".join(notes + [str(exc)])
        notes.append("model %s (hash %s)" % (model.get("name"), model.get("hash")))
        return True, "; ".join(notes)

    # --- graph ----------------------------------------------------------

    def _build_graph(self, prompt, negative, seed, width, height, steps,
                     cfg_scale, scheduler, use_cache, board_id):
        """The SDXL txt2img graph the frontend builds, minus its two `string`
        primitives (which exist only to be batch-substitution targets) and minus
        its two `collect` nodes (denoise_latents accepts a bare ConditioningField,
        not only a list). Six nodes, ten edges -- one per connection-only field.
        """
        model = self._resolve_model()

        model_node = {
            "id": NODE_MODEL,
            "type": "sdxl_model_loader",
            # ModelIdentifierField wants all five verbatim. `key` is resolved at
            # run time from the stable hash or name, never stored in config.
            "model": {"key": model["key"], "hash": model["hash"],
                      "name": model["name"], "base": model["base"],
                      "type": model["type"]},
            "use_cache": use_cache,
        }

        # SDXL runs TWO text encoders, and sdxl_compel_prompt carries one field
        # per encoder: `prompt` feeds CLIP-L, `style` feeds CLIP-G. Leaving
        # `style` empty does not fail -- the node quietly reuses `prompt` -- so
        # the graph still renders, just not the image the UI would produce, since
        # the frontend wires one string node into BOTH fields. That divergence is
        # silent, survives a fixed seed, and would surface as 88 icons subtly off
        # the look tuned in the UI, with nothing in any error message pointing at
        # it. Both fields, both nodes, always.
        pos_node = {"id": NODE_POS, "type": "sdxl_compel_prompt",
                    "prompt": prompt, "style": prompt, "use_cache": use_cache}
        neg_node = {"id": NODE_NEG, "type": "sdxl_compel_prompt",
                    "prompt": negative, "style": negative, "use_cache": use_cache}
        # original_width/original_height/target_width/target_height are SDXL's
        # size micro-conditioning embeddings and all default to 1024. They are
        # left unset: the research says leave them at defaults for 1024 icons and
        # does not say what the frontend passes for other sizes, so setting one
        # would mean inventing a value.

        noise_node = {
            "id": NODE_NOISE,
            "type": "noise",
            "seed": seed,
            # Size lives HERE for SD1.5/SDXL. DenoiseLatentsInvocation has no
            # width/height at all -- only the FLUX/SD3/CogView4/Qwen/Z-Image
            # denoisers do -- so setting them on denoise_latents is a 422.
            "width": width,
            "height": height,
            "use_cpu": True,        # the whole basis of cross-platform seeds
            "noise_type": "SD",     # correct for SD1.5/SDXL, not the FLUX path
            "use_cache": use_cache,
        }

        denoise_node = {
            "id": NODE_DENOISE,
            "type": "denoise_latents",
            "steps": steps,
            "cfg_scale": cfg_scale,
            "cfg_rescale_multiplier": 0,
            "scheduler": scheduler,
            # addTextToImage.ts sets these explicitly for txt2img rather than
            # trusting the defaults; do the same so a default change is inert.
            "denoising_start": 0.0,
            "denoising_end": 1.0,
            "use_cache": use_cache,
        }

        l2i_node = {
            "id": NODE_L2I,
            "type": "l2i",
            "fp32": False,
            # False so the saved image is a real gallery asset rather than a
            # hidden intermediate.
            "is_intermediate": False,
            "use_cache": use_cache,
        }
        if board_id:
            l2i_node["board"] = {"board_id": board_id}

        nodes = {}
        for node in (model_node, pos_node, neg_node, noise_node, denoise_node,
                     l2i_node):
            nodes[node["id"]] = node

        edges = [
            _edge(NODE_MODEL, "unet", NODE_DENOISE, "unet"),
            _edge(NODE_MODEL, "clip", NODE_POS, "clip"),
            _edge(NODE_MODEL, "clip2", NODE_POS, "clip2"),
            _edge(NODE_MODEL, "clip", NODE_NEG, "clip"),
            _edge(NODE_MODEL, "clip2", NODE_NEG, "clip2"),
            _edge(NODE_MODEL, "vae", NODE_L2I, "vae"),
            _edge(NODE_POS, "conditioning", NODE_DENOISE, "positive_conditioning"),
            _edge(NODE_NEG, "conditioning", NODE_DENOISE, "negative_conditioning"),
            _edge(NODE_NOISE, "noise", NODE_DENOISE, "noise"),
            _edge(NODE_DENOISE, "latents", NODE_L2I, "latents"),
        ]
        return {"id": "devgraphics_sdxl_txt2img", "nodes": nodes, "edges": edges}

    # --- queue ----------------------------------------------------------

    def _enqueue(self, batch, expected):
        # The body is NOT a bare Batch -- it is Body_enqueue_batch. Posting the
        # Batch at the root gets a 422 that reads like a schema mismatch deep
        # inside the graph and wastes an hour.
        result = _http.request_json(
            self._url("/api/v1/queue/%s/enqueue_batch" % self.queue_id),
            payload={"batch": batch, "prepend": False},
            headers=self._headers(), timeout=HTTP_TIMEOUT)

        enqueued = result.get("enqueued")
        requested = result.get("requested")
        item_ids = result.get("item_ids") or []
        # create_session_nfv_tuples stops generating sessions once max_queue_size
        # is reached and the route still answers 2xx. enqueued < requested is the
        # only signal anywhere that images were dropped.
        if enqueued is not None and requested is not None and enqueued != requested:
            raise BackendError(
                "invokeai enqueued %s of %s requested sessions -- the queue hit "
                "max_queue_size and truncated silently; drain the queue or lower "
                "count" % (enqueued, requested))
        if len(item_ids) != expected:
            raise BackendError(
                "invokeai returned %d queue item id(s) for %d requested image(s)"
                % (len(item_ids), expected))
        return item_ids

    def _await_image(self, item_id, deadline):
        url = self._url("/api/v1/queue/%s/i/%s" % (self.queue_id, item_id))
        warned = set()
        while True:
            item = _http.request_json(url, headers=self._headers(),
                                      timeout=HTTP_TIMEOUT)
            status = item.get("status")
            if status in TERMINAL_STATUSES:
                break
            if status not in RUNNING_STATUSES and status not in warned:
                # A status this build knows and this module does not. Treating it
                # as terminal would read results that do not exist yet -- exactly
                # what "waiting" did to pollers when 6.14 added it -- so keep
                # polling, and say so once.
                warned.add(status)
                warnings.warn("invokeai queue item %s reported unknown status %r; "
                              "treating it as still running (this build is newer "
                              "than %s)" % (item_id, status, _tested_range()))
            if time.time() >= deadline:
                raise BackendError(
                    "invokeai queue item %s still %r after %.0fs; check the "
                    "processor is running (GET /api/v1/queue/%s/status)"
                    % (item_id, status, self.timeout, self.queue_id))
            time.sleep(self.poll_interval)

        if status != "completed":
            # The enqueue response looked perfectly healthy even when the graph
            # was missing a connection-only edge; this is where that surfaces, so
            # all three error fields are worth reading.
            raise BackendError(
                "invokeai queue item %s %s: %s %s"
                % (item_id, status, item.get("error_type") or "",
                   item.get("error_message") or item.get("error_traceback") or ""))
        return _image_name(item, NODE_L2I)

    def _fetch_image(self, image_name):
        raw = _http.request_bytes(
            self._url("/api/v1/images/i/%s/full"
                      % urllib.parse.quote(image_name, safe="")),
            headers=self._headers(), timeout=HTTP_TIMEOUT)
        # The route declares media_type image/png, so this is a magic-byte check
        # and nothing more -- cheap insurance, not a transcode.
        return to_png(raw)

    # --- server facts, all fetched lazily -------------------------------

    def _fetch_version(self):
        if self._version is None:
            doc = _http.request_json(self._url("/api/v1/app/version"),
                                     headers=self._headers(),
                                     timeout=HTTP_TIMEOUT)
            self._version = doc.get("version") or "unknown"
        return self._version

    def _warn_on_untested_version(self):
        parsed = _version_tuple(self._fetch_version())
        if parsed is None or parsed not in TESTED_VERSIONS:
            warnings.warn(
                "invokeai %s is outside the tested range %s; this backend hard-"
                "codes v6.13.8 node and field names, and the maintainers make "
                "breaking API changes without notice"
                % (self._version, _tested_range()))

    def _resolve_model(self):
        """Find the SDXL checkpoint, by hash first and name second.

        Uses the list route rather than get_by_attrs so a wrong name can print
        the names that do exist. Note the trailing slash: the router prefix is
        /v2/models and the route is "/", so omitting it redirects or 404s
        depending on the client.
        """
        if self._model is not None:
            return self._model
        doc = _http.request_json(
            self._url("/api/v2/models/?base_models=sdxl&model_type=main"),
            headers=self._headers(), timeout=HTTP_TIMEOUT)
        models = doc.get("models") or []
        if not models:
            raise BackendError(
                "invokeai has no SDXL main model installed; install one in the "
                "UI or via POST /api/v2/models/install")

        if self.model_hash:
            found = _first(models, "hash", self.model_hash)
            if found is None:
                raise BackendError("no SDXL model with hash %r on %s; installed: %s"
                                   % (self.model_hash, self.host, _names(models)))
        elif self.model_name:
            found = _first(models, "name", self.model_name)
            if found is None:
                raise BackendError("no SDXL model named %r on %s; installed: %s"
                                   % (self.model_name, self.host, _names(models)))
        elif len(models) == 1:
            found = models[0]
        else:
            raise BackendError(
                "%s has %d SDXL models; pick one with -O model_hash=... (stable "
                "across reinstalls) or -O model=...: %s"
                % (self.host, len(models), _names(models)))

        missing = [f for f in ("key", "hash", "name", "base", "type")
                   if not found.get(f)]
        if missing:
            raise BackendError("invokeai model record is missing %s; "
                               "ModelIdentifierField needs all five fields"
                               % ", ".join(missing))
        self._model = found
        return found

    # --- plumbing -------------------------------------------------------

    def _url(self, path):
        if self.host.startswith("http://") or self.host.startswith("https://"):
            return "%s%s" % (self.host.rstrip("/"), path)
        return "http://%s%s" % (self.host, path)

    def _headers(self):
        """A bearer token only matters when the server runs multiuser:true.

        The default install has no auth at all, so this is None almost always.
        Note that /images/i/{name}/full stays unauthenticated in BOTH modes by
        design (a browser cannot put a Bearer header on an <img src>), so only
        the enqueue/poll half ever needs the token -- it is sent everywhere
        anyway because an ignored header costs nothing.

        Limit worth knowing: a mutating request returns a refreshed token in the
        X-Refreshed-Token response header, and _http hands back bodies only, so a
        run longer than the one-day token lifetime cannot renew itself here.
        """
        if not self.token:
            return None
        return {"Authorization": "Bearer %s" % self.token}


# --- module-level helpers -----------------------------------------------

def _edge(source_id, source_field, dest_id, dest_field):
    return {"source": {"node_id": source_id, "field": source_field},
            "destination": {"node_id": dest_id, "field": dest_field}}


def _require_connections(graph):
    """Check what Graph.validate_self will not: that every connection-only field
    has an edge into it, and that each nodes-dict key equals its node's own id
    (_validate_node_id_mapping raises NodeIdMismatchError otherwise).
    """
    for key, node in graph["nodes"].items():
        if key != node.get("id"):
            raise BackendError("graph node key %r does not match node id %r"
                               % (key, node.get("id")))
    wired = set()
    for edge in graph["edges"]:
        dest = edge["destination"]
        wired.add((dest["node_id"], dest["field"]))
    missing = [pair for pair in REQUIRED_CONNECTIONS if pair not in wired]
    if missing:
        raise BackendError(
            "graph is missing edges into connection-only field(s) %s; InvokeAI "
            "would accept this enqueue and then fail at run time"
            % ", ".join("%s.%s" % pair for pair in missing))


def _image_name(item, node_id):
    """Pull the finished image name out of a completed queue item.

    session.results is keyed by PREPARED node ids -- the ids the executor
    materialises -- not by the ids written into the graph, and that indirection
    is the single most common way to get this wrong. prepared_source_mapping is
    what translates prepared id -> our id. Filtering on the output type alone is
    only safe because this graph has exactly one image producer, so it is the
    fallback rather than the primary route.
    """
    session = item.get("session") or {}
    results = session.get("results") or {}
    mapping = session.get("prepared_source_mapping") or {}

    for prepared_id, result in results.items():
        if result.get("type") == "image_output" and mapping.get(prepared_id) == node_id:
            name = (result.get("image") or {}).get("image_name")
            if name:
                return name
    for result in results.values():
        if result.get("type") == "image_output":
            name = (result.get("image") or {}).get("image_name")
            if name:
                return name
    raise BackendError("invokeai queue item %s completed with no image_output in "
                       "session.results" % item.get("item_id"))


def _check_options(options, allowed, where):
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise UnsupportedOption(
            "invokeai %s does not accept %s; accepted keys: %s"
            % (where, ", ".join(unknown), ", ".join(sorted(allowed))))


def _check_request_options(options):
    connection = sorted(set(options) & (OPTIONS - REQUEST_OPTIONS))
    if connection:
        raise UnsupportedOption(
            "invokeai option(s) %s describe the server, not one request; set "
            "them when the backend is constructed" % ", ".join(connection))
    _check_options(options, REQUEST_OPTIONS, "request options")


def _check_scheduler(name):
    if name not in SCHEDULERS:
        raise UnsupportedOption(
            "invokeai scheduler %r is not one of: %s"
            % (name, ", ".join(sorted(SCHEDULERS))))


def _multiple_of_8(value):
    """noise.width/height are gt=0 and multiple_of=8 (LATENT_SCALE_FACTOR)."""
    return max(8, int(round(int(value) / 8.0)) * 8)


def _version_tuple(text):
    parts = []
    for chunk in str(text).split(".")[:2]:
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break                      # tolerate 6.14.0-rc2 and 6.14.0a1
            digits += char
        if not digits:
            return None
        parts.append(int(digits))
    if len(parts) != 2:
        return None
    return (parts[0], parts[1])


def _tested_range():
    return "-".join("%d.%d" % v for v in (TESTED_VERSIONS[0],
                                          TESTED_VERSIONS[-1]))


def _first(models, field, value):
    for model in models:
        if model.get(field) == value:
            return model
    return None


def _names(models):
    return ", ".join(sorted(str(m.get("name")) for m in models))
