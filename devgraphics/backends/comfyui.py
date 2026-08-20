"""
Headless client for a stock ComfyUI server: HTTP + WebSocket, no SDK, no auth.

Four decisions carry this module.

**The request is the graph.** ComfyUI has no prompt, seed or width parameter --
POST /prompt takes an entire node graph, and you change a render by mutating that
dict by node id and input name (prompt["6"]["inputs"]["text"]). So this ships a
hand-written API-format template and injects into it, and anyone with their own
LoRA stack points `workflow` at their own export and remaps the ids with
-O node_prompt=12. Converting the UI-format JSON people actually share was
rejected: the docs call that "a script that strips the x, y, width fields", but
it means rebuilding every link from a flat links array, mapping POSITIONAL
widgets_values onto named inputs using each node's declared input order from
/object_info, dropping muted and bypassed nodes, and inlining subgraph
definitions -- a compiler, not a field-strip. UI files are refused by name.

**Done is `executing` with node null, never a new filename.** ComfyUI caches node
outputs: resubmit a byte-identical graph and nothing renders, an execution_cached
frame lists the skipped nodes, and /history for the NEW prompt_id points at the
ORIGINAL file. Waiting for a file to appear would hang forever on the second run
of any icon, which is the normal case in a resumable batch. execution_success is
the wrong sentinel too -- it fires inside PromptExecutor.execute, before
task_done() populates history, so a /history fetched on it can legitimately
answer {}. main.py sends {"type": "executing", "data": {"node": null}} after
task_done, and only to the client_id that queued the job.

**Names come back from the server; they are never built here.** folder_paths
builds the checkpoint list with os.path.relpath, so models/checkpoints/SDXL/foo
is "SDXL\\foo.safetensors" on Windows and "SDXL/foo.safetensors" on Linux.
ckpt_name has no VALIDATE_INPUTS override, so the exact string is checked against
that list, and a hand-built forward-slash version 400s with value_not_in_list --
an error whose details degrade to "(list of length 137)" above 20 options and so
never says what would have been acceptable. Same for the subfolder /history hands
back for /view. Everything is echoed verbatim; only the matching is forgiving.

**Nothing about the install is known until it is asked.** Native RGBA needs core
nodes that landed around 0.21.0 plus a 444 MB birefnet.safetensors that may not
be downloaded, and that is a fact about one server -- which is exactly why
Capabilities is per instance. It is feature-detected against /object_info rather
than parsed out of comfyui_version, cached on the instance, and answers the
conservative False until the server actually answers.

Stock ComfyUI has no API key, no token and no Authorization header: anything that
can reach the port can queue arbitrary graphs, which is why --listen defaults to
loopback. Pointing `host` at a LAN box is the caller's risk to manage.
"""

import copy
import json
import urllib.parse
import uuid

from .._http import post_multipart, request_bytes, request_json
from ..postprocess import to_png
from .base import (BackendError, Capabilities, MissingDependency,
                   UnsupportedOption)

try:
    from websocket import (WebSocketException, WebSocketTimeoutException,
                           create_connection)
except ImportError:                                       # pragma: no cover
    # websocket-client lives in the `local` extra, because six of the seven
    # backends never open a socket. Failing here would make `import devgraphics`
    # and `devgraphics backends` collapse for an OpenAI user who has no reason to
    # own it, so the module stays importable and the name is a stub that explains
    # itself the moment a ComfyUI render actually needs it.
    class WebSocketException(Exception):
        pass

    class WebSocketTimeoutException(WebSocketException):
        pass

    def create_connection(*_args, **_kwargs):
        raise MissingDependency("comfyui", "websocket-client", "local")

DEFAULT_HOST = "127.0.0.1:8188"

#: Shipped API-format graphs. Read out of the package with importlib.resources so
#: a wheel install works, not just a source checkout.
TEMPLATE = "sdxl_txt2img.api.json"
TEMPLATE_RGBA = "sdxl_txt2img_rgba.api.json"

#: Which node of the shipped template carries which knob. A user's own export
#: overrides the ids; the input names next to each injection are fixed by the
#: node class, so pointing node_prompt at something that is not a text encoder is
#: caught by _set() rather than silently written into an input nobody reads.
DEFAULT_NODES = {"prompt": "6", "negative": "7", "seed": "3", "size": "5",
                 "checkpoint": "4", "image": ""}

#: (option, input name, cast). All of these live on the same node as the seed,
#: because in every SDXL graph that node is the KSampler.
SAMPLER_INPUTS = (("steps", "steps", int),
                  ("cfg", "cfg", float),
                  ("sampler", "sampler_name", str),
                  ("scheduler", "scheduler", str),
                  ("denoise", "denoise", float))

#: Request.options keys this backend accepts. Anything else is a typo'd -O and
#: has to fail loudly: a node id that silently changes nothing is how you find
#: out, 88 icons later, that every one of them carries the template's prompt.
OPTIONS = frozenset(["workflow", "checkpoint", "steps", "cfg", "sampler",
                     "scheduler", "denoise"]
                    + ["node_" + key for key in DEFAULT_NODES])

#: Feature detection and liveness have to answer fast, or --dry-run stops being
#: the cheap thing it exists to be. Generation timeouts are a different order --
#: the first render after a server start is 30-120s of silent checkpoint loading.
PROBE_TIMEOUT = 5.0

UI_FORMAT = (
    "%s is a UI-format workflow, which ComfyUI's /prompt does not accept: it has "
    "top-level 'nodes' / 'links' arrays, where API format is a flat\n"
    "  {\"<node id>\": {\"class_type\": ..., \"inputs\": {...}}}  dict.\n"
    "  Re-export it from ComfyUI:  File -> Export Workflow (API)\n"
    "  On older builds: Settings -> tick 'Enable Dev mode Options', and a\n"
    "  'Save (API Format)' button appears next to Save.\n"
    "  devgraphics will not convert one into the other: links, positional widget\n"
    "  values and subgraph definitions all have to be resolved against the node\n"
    "  schemas, which is a compiler rather than a field-strip.")

#: The half of a /prompt rejection ComfyUI leaves out. _http already classified
#: the status code and dug the message out of {"error": {...}}, but the per-node
#: detail sits in a sibling "node_errors" key it never sees.
PROMPT_HINTS = (
    ("value_not_in_list",
     "an input is not a member of its combo list, and ComfyUI hides that list "
     "above 20 options. These names are os.path.relpath strings -- on Windows a "
     "checkpoint in a subfolder contains a BACKSLASH. probe() prints the "
     "installed names verbatim."),
    ("prompt_no_outputs",
     "the graph has no output node; at least one SaveImage is required."),
    ("missing_node_type",
     "that node class is not installed on this server -- either a custom node, "
     "or a ComfyUI older than the one the workflow was exported from."),
    ("required_input_missing",
     "an input has neither a value nor a link, so the workflow is incomplete."),
)


class ComfyUIBackend:
    """ComfyUI behind the Backend contract.

    `workflow` is the escape hatch that makes this backend worth having: point it
    at your own API-format export and remap whichever node ids carry the prompt,
    the negative, the seed, the size and the checkpoint. The defaults are the
    shipped template's ids, which are also the ids ComfyUI's own default workflow
    uses, so an unmodified export usually needs no remapping at all.

    `node_image` is empty by default and that is what makes reference_images 0:
    the shipped template has no LoadImage node, so there is nowhere to put an
    uploaded reference. Set it (comma-separated for several) and the capability
    appears -- the per-instance half of Capabilities, doing real work.
    """

    def __init__(self, host=DEFAULT_HOST, timeout=900, prefix="",
                 workflow=None, checkpoint=None,
                 node_prompt=DEFAULT_NODES["prompt"],
                 node_negative=DEFAULT_NODES["negative"],
                 node_seed=DEFAULT_NODES["seed"],
                 node_size=DEFAULT_NODES["size"],
                 node_checkpoint=DEFAULT_NODES["checkpoint"],
                 node_image=DEFAULT_NODES["image"],
                 steps=None, cfg=None, sampler=None, scheduler=None,
                 denoise=None):
        self.host = host
        self.timeout = float(timeout)
        # Every route is also registered under /api, and reverse proxies and the
        # desktop app usually expose only that form. Neither is deprecated.
        self.prefix = prefix
        self.workflow = workflow
        self.checkpoint = checkpoint
        self.nodes = {"prompt": str(node_prompt), "negative": str(node_negative),
                      "seed": str(node_seed), "size": str(node_size),
                      "checkpoint": str(node_checkpoint), "image": node_image}
        self.steps = steps
        self.cfg = cfg
        self.sampler = sampler
        self.scheduler = scheduler
        self.denoise = denoise
        # Nothing in here touches the network; see the module docstring. Both
        # caches start empty and fill on first use.
        self._combos = {}
        self._graphs = {}

    # --- contract -------------------------------------------------------

    @property
    def capabilities(self):
        return Capabilities(
            name="comfyui",
            seed=True,
            deterministic=True,
            negative_prompt=True,
            transparent=self._native_alpha(),
            reference_images=len(_ids(self.nodes["image"])),
            batch=True,
            sizes=(),                     # EmptyLatentImage takes real integers
            cost_per_image=None,
            notes=(
                "Node outputs are cached hard: an identical graph produces no new "
                "file, and /history for the new job points at the original one. "
                "Idempotent re-runs are free, but a re-render needs a changed "
                "seed or prompt.",
                "Native alpha is feature-detected per server, from "
                "/object_info/LoadBackgroundRemovalModel. It needs ComfyUI "
                ">= ~0.21.0 and birefnet.safetensors (444 MB) in "
                "models/background_removal/; without it postprocess.cutout() "
                "keys the backdrop out instead.",
                "Checkpoint names are echoed from /object_info verbatim, because "
                "they are os.path.relpath strings: a nested checkpoint contains "
                "a backslash on Windows and a slash on Linux, and the wrong one "
                "is rejected.",
                "Determinism is per install: noise is generated on the CPU from "
                "torch.manual_seed(seed), so the same seed, graph, checkpoint and "
                "machine reproduce the image -- another GPU or torch version does "
                "not have to.",
                "Any width and height, 16..16384, rounded down to a multiple of 8 "
                "here because the server would floor it into the latent silently. "
                "SDXL still degrades far off 1024x1024, so render there and let "
                "postprocess downscale.",
                "No authentication exists in stock ComfyUI: anything that can "
                "reach the port can queue work.",
            ),
        )

    def generate(self, request):
        unknown = sorted(set(request.options) - OPTIONS)
        if unknown:
            raise UnsupportedOption(
                "comfyui does not accept %s; accepted keys: %s"
                % (", ".join(unknown), ", ".join(sorted(OPTIONS))))

        options = request.options
        source = options.get("workflow", self.workflow)
        graph = self._graph(source, request.transparent)
        nodes = dict(self.nodes)
        for key in nodes:
            if "node_" + key in options:
                nodes[key] = str(options["node_" + key])

        _set(graph, nodes["prompt"], "text", request.prompt)
        _set(graph, nodes["negative"], "text", request.negative or "")
        if request.seed is not None:
            # 0 .. 2**64-1 server-side. Folding beats a 400 on a caller whose
            # seed came out of a hash.
            _set(graph, nodes["seed"], "seed", int(request.seed) % (2 ** 64))
        width, height = _latent_size(request.size)
        _set(graph, nodes["size"], "width", width)
        _set(graph, nodes["size"], "height", height)
        if request.count > 1 or _has(graph, nodes["size"], "batch_size"):
            _set(graph, nodes["size"], "batch_size", int(request.count))
        for key, name, cast in SAMPLER_INPUTS:
            value = options.get(key, getattr(self, key))
            if value is not None:
                _set(graph, nodes["seed"], name, _cast(cast, key, value))
        _set(graph, nodes["checkpoint"], "ckpt_name",
             self._resolve_checkpoint(graph, nodes["checkpoint"],
                                      options.get("checkpoint", self.checkpoint)))
        if request.transparent and not source:
            _set_all(graph, "LoadBackgroundRemovalModel", "bg_removal_name",
                     self._bg_model())
        self._attach_refs(graph, nodes["image"], request.refs)

        # to_png because a user's own graph may end in SaveAnimatedWEBP or a
        # SaveImage fork; bytes that are already PNG come back untouched.
        return [to_png(self._fetch(item)) for item in self._run(graph)]

    @classmethod
    def probe(cls, host=DEFAULT_HOST, prefix="", **_options):
        """Reachability, version and the exact checkpoint names. Renders nothing.

        Two cheap GETs. /system_stats is the true liveness check -- GET /prompt is
        cheaper still but only reports queue depth. The checkpoint list is worth
        a second request because a byte-exact ckpt_name is the one thing a user
        cannot guess, and an install with no checkpoints at all is a failure that
        would otherwise surface as an opaque 400 on the first icon.

        The rest of the option table is accepted and ignored: nothing about a
        workflow can be confirmed without spending a render.
        """
        try:
            stats = request_json(_url(host, prefix, "/system_stats"),
                                 timeout=PROBE_TIMEOUT, retries=0)
        except Exception as exc:
            return False, "cannot reach ComfyUI at %s: %s" % (host, exc)
        system = stats.get("system") or {}
        line = "ComfyUI %s (python %s, torch %s)" % (
            system.get("comfyui_version") or "unknown",
            str(system.get("python_version") or "?").split(" ")[0],
            system.get("pytorch_version") or "?")
        devices = [d.get("name") for d in stats.get("devices") or []
                   if d.get("name")]
        if devices:
            line += " on %s" % ", ".join(devices)
        names = _combo(host, prefix, "CheckpointLoaderSimple", "ckpt_name")
        if not names:
            return False, ("%s -- but models/checkpoints is empty, so there is "
                           "nothing to render with" % line)
        return True, "%s; %d checkpoint(s), e.g. %r" % (line, len(names), names[0])

    # --- graph ----------------------------------------------------------

    def _graph(self, source, transparent):
        """A fresh copy of the graph this render starts from.

        Parsed once per source and deep-copied per call: injection mutates, and
        88 icons sharing one dict would each inherit the previous one's prompt.
        """
        key = source or (TEMPLATE_RGBA if transparent else TEMPLATE)
        if key not in self._graphs:
            self._graphs[key] = (_load_workflow(source) if source
                                 else _packaged(key))
        return copy.deepcopy(self._graphs[key])

    def _resolve_checkpoint(self, graph, node_id, wanted):
        """The ckpt_name to inject, echoed verbatim from /object_info.

        The matching is deliberately loose and the answer deliberately exact: a
        user who types the forward-slash spelling of a nested checkpoint (or the
        wrong case) gets the server's own string injected, rather than a
        value_not_in_list 400 that will not even print the alternatives.
        """
        current = _current(graph, node_id, "ckpt_name")
        installed = self._server_combo("CheckpointLoaderSimple", "ckpt_name")
        if not installed:
            # Server not answering. Let POST /prompt produce the real error a
            # moment later rather than inventing one about checkpoints.
            return wanted or current
        if wanted:
            match = _match(wanted, installed)
            if match is None:
                raise BackendError(
                    "no checkpoint named %r on ComfyUI at %s\n  installed: %s"
                    % (wanted, self.host, _listing(installed)))
            return match
        if current in installed:
            return current
        if len(installed) == 1:
            return installed[0]
        raise BackendError(
            "the workflow names checkpoint %r, which this server does not have. "
            "Choose one with -O checkpoint=NAME\n  installed: %s"
            % (current, _listing(installed)))

    def _attach_refs(self, graph, node_image, refs):
        if not refs:
            return
        targets = _ids(node_image)
        if len(refs) > len(targets):
            raise BackendError(
                "%d reference image(s) but %d LoadImage node(s) configured. The "
                "shipped template has none: export a workflow containing "
                "LoadImage and point -O node_image=<id> at it (comma-separated "
                "for several)." % (len(refs), len(targets)))
        for node_id, data in zip(targets, refs):
            _set(graph, node_id, "image", self._upload(data))

    def _bg_model(self):
        """The background-removal model to name, verbatim, or a real explanation.

        Asked of LoadBackgroundRemovalModel rather than RemoveBackground because
        its combo answers both questions at once: an empty options list means the
        node exists but models/background_removal/ is empty, which fails just as
        hard as the node being absent and is a completely different fix.
        """
        names = self._server_combo("LoadBackgroundRemovalModel",
                                   "bg_removal_name")
        if not names:
            raise BackendError(
                "this ComfyUI cannot produce a real alpha channel: either it "
                "predates the core RemoveBackground nodes (~0.21.0) or "
                "models/background_removal/ is empty.\n"
                "  Put birefnet.safetensors (444 MB, "
                "huggingface.co/Comfy-Org/BiRefNet) there, or drop transparent "
                "and let postprocess.cutout() key the backdrop out instead.")
        for name in names:
            if "birefnet" in name.lower():
                return name
        # Whatever else lives in that folder. lucida.safetensors ships in the same
        # HF repo, but whether the loader accepts it is UNVERIFIED, so it is only
        # ever the fallback.
        return names[0]

    def _native_alpha(self):
        """Whether THIS server can put a real alpha channel in the PNG.

        A custom workflow answers for itself: if it already joins an alpha
        channel, devgraphics does not get a vote and does not need the server.
        Otherwise it is a probe, and an unanswered probe means False.
        """
        if self.workflow:
            graph = self._graph(self.workflow, False)
            return any(node.get("class_type") == "JoinImageWithAlpha"
                       for node in graph.values())
        return bool(self._server_combo("LoadBackgroundRemovalModel",
                                       "bg_removal_name"))

    def _server_combo(self, class_name, input_name):
        """_combo, cached on the instance.

        88 icons must not mean 88 /object_info round trips. A failed answer is
        deliberately not cached: a server that was merely switched off when
        --dry-run asked for capabilities gets asked again when it matters, which
        is the whole reason the conservative default is safe.
        """
        key = (class_name, input_name)
        if key not in self._combos:
            names = _combo(self.host, self.prefix, class_name, input_name)
            if names is None:
                return None
            self._combos[key] = names
        return self._combos[key]

    # --- transport ------------------------------------------------------

    def _run(self, graph):
        """Queue the graph, wait for it, and return the /history image entries.

        The socket is opened BEFORE the POST on purpose: the queue can start --
        and on a fully cached graph, finish -- before a socket opened afterwards
        is registered, and then the sentinel being waited for is one that has
        already been sent. Note clientId (camelCase) in the query string against
        client_id (snake_case) in the body: they must carry the same uuid, and a
        mismatch is silent, leaving a healthy socket that never delivers a frame.
        """
        client_id = str(uuid.uuid4())
        payload = {"prompt": graph, "client_id": client_id,
                   # Minting the id here means frames can be filtered without
                   # waiting on the response. Servers too old to honour it just
                   # generate their own, which is why the response wins below.
                   "prompt_id": str(uuid.uuid4())}
        ws = create_connection(_ws_url(self.host, self.prefix, client_id),
                               timeout=self.timeout)
        try:
            try:
                # retries=0: a retried POST queues the job a second time.
                queued = request_json(_url(self.host, self.prefix, "/prompt"),
                                      payload, timeout=60, retries=0)
            except BackendError as exc:
                raise _prompt_error(exc)
            prompt_id = queued.get("prompt_id") or payload["prompt_id"]
            self._wait(ws, prompt_id)
        finally:
            ws.close()
        return self._outputs(prompt_id)

    def _wait(self, ws, prompt_id):
        while True:
            try:
                frame = ws.recv()
            except WebSocketTimeoutException:
                raise BackendError(
                    "no frame from ComfyUI for %gs. The first render after the "
                    "server starts spends 30-120s loading the checkpoint with "
                    "the socket completely silent, so raise timeout= before "
                    "concluding it hung." % self.timeout)
            except WebSocketException as exc:
                raise BackendError("ComfyUI websocket failed: %s" % exc)
            if isinstance(frame, bytes):
                continue                # binary preview; carries no prompt_id
            try:
                message = json.loads(frame)
            except ValueError:
                continue
            data = message.get("data") or {}
            # status frames are broadcast to every socket, and
            # execution_interrupted is broadcast too, so somebody else's job must
            # not be able to end this wait.
            if data.get("prompt_id") != prompt_id:
                continue
            kind = message.get("type")
            if kind == "executing" and data.get("node") is None:
                return
            if kind == "execution_error":
                raise BackendError(
                    "ComfyUI node %s (%s) failed: %s: %s"
                    % (data.get("node_id"), data.get("node_type"),
                       data.get("exception_type"),
                       data.get("exception_message")))
            if kind == "execution_interrupted":
                raise BackendError(
                    "ComfyUI execution was interrupted at node %s (%s)"
                    % (data.get("node_id"), data.get("node_type")))

    def _outputs(self, prompt_id):
        """The image entries for a finished job.

        One fetch, no poll: the sentinel this waited for is sent after
        task_done(), which is what populates history, so it is already there.
        A cached job is indistinguishable here, and that is the point -- its
        entry names the file the ORIGINAL run wrote.
        """
        doc = request_json(_url(self.host, self.prefix,
                                "/history/%s" % urllib.parse.quote(prompt_id)),
                           timeout=60, retries=1)
        entry = doc.get(prompt_id)
        if entry is None:
            raise BackendError(
                "ComfyUI has no history for %s. History is capped and evicted "
                "FIFO, so a very long batch can lose it." % prompt_id)
        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            raise BackendError("ComfyUI reported the job failed: %s"
                               % json.dumps(status.get("messages"))[:400])
        found = []
        for _node_id, output in (entry.get("outputs") or {}).items():
            found.extend(output.get("images") or [])
        # PreviewImage writes type "temp" and ComfyUI wipes that directory on
        # startup, so prefer saved output -- but a graph that only previews still
        # gets its bytes rather than a confusing "no images".
        images = [i for i in found if i.get("type") == "output"] or found
        if not images:
            raise BackendError(
                "ComfyUI finished but produced no image. A graph whose only "
                "output node is not a SaveImage will do this.")
        return images

    def _fetch(self, item):
        """The bytes of one /history output entry.

        The three values go back exactly as they arrived. No channel= and no
        preview=: channel=rgb strips the alpha a 444 MB model may just have been
        spent producing, and preview=webp;90 re-encodes to lossy WebP, which
        hands postprocess.cutout() the ringing thresh=42 keys on. Omit both and
        aiohttp serves the file untouched.
        """
        query = urllib.parse.urlencode({"filename": item.get("filename", ""),
                                        "subfolder": item.get("subfolder", ""),
                                        "type": item.get("type", "output")})
        return request_bytes(_url(self.host, self.prefix, "/view?" + query),
                             timeout=120, retries=2)

    def _upload(self, data):
        """POST /upload/image; return what LoadImage.image should be set to.

        overwrite=true against a random name rather than a stable one: without it
        the server hashes any existing file and either de-duplicates or appends
        " (1)", so the value to inject is whatever came back -- under the key
        "name", not "filename". A just-uploaded name is safe to use immediately
        even though /object_info's combo list is stale, because LoadImage
        declares VALIDATE_INPUTS and execution.py therefore skips the membership
        check for it.
        """
        name = "devgraphics_%s.png" % uuid.uuid4().hex[:12]
        doc = post_multipart(
            _url(self.host, self.prefix, "/upload/image"),
            {"type": "input", "subfolder": "", "overwrite": "true"},
            {"image": (name, to_png(data))},
            timeout=self.timeout)
        got = doc.get("name")
        if not got:
            raise BackendError("ComfyUI accepted the upload but named no file: "
                               "%s" % json.dumps(doc)[:200])
        subfolder = doc.get("subfolder") or ""
        return "%s/%s" % (subfolder, got) if subfolder else got


# --- workflow files -----------------------------------------------------

def _packaged(name):
    """One of the shipped templates, read from the package.

    importlib.resources rather than __file__, so this works from a wheel or a zip
    import and not only from a source checkout. workflows/ is package *data*
    under devgraphics rather than a package of its own, so it is reached by
    joinpath rather than named as an import.
    """
    from importlib.resources import files
    handle = files("devgraphics").joinpath("workflows").joinpath(name)
    return _check_api_format(json.loads(handle.read_text(encoding="utf-8")),
                             "the shipped %s" % name)


def _load_workflow(path):
    """A user's own export, parsed and shape-checked."""
    try:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
    except OSError as exc:
        raise BackendError("cannot read workflow %s: %s" % (path, exc))
    except ValueError as exc:
        raise BackendError("workflow %s is not valid JSON: %s" % (path, exc))
    return _check_api_format(doc, path)


def _check_api_format(doc, where):
    """Refuse a UI export before it becomes an unreadable 400.

    The two formats are told apart by structure, not by filename: in API format
    every value is an object, so a top-level "nodes" or "links" that is an ARRAY
    can only be the UI save format.
    """
    if not isinstance(doc, dict):
        raise BackendError("%s is not a workflow: expected a JSON object, got %s"
                           % (where, type(doc).__name__))
    if isinstance(doc.get("nodes"), list) or isinstance(doc.get("links"), list):
        raise BackendError(UI_FORMAT % where)
    for node_id, node in doc.items():
        if not isinstance(node, dict) or "class_type" not in node:
            raise BackendError(
                "%s is not an API-format workflow: entry %r has no class_type. "
                "API format is a flat {node id: {class_type, inputs}} dict."
                % (where, node_id))
    return doc


# --- graph editing ------------------------------------------------------

def _set(graph, node_id, name, value):
    """Write one input, refusing to invent one.

    Nodes read their declared inputs by name and ignore everything else, so a
    node id pointed at the wrong node would otherwise produce a graph that
    /prompt happily accepts and renders without the caller's prompt in it.
    """
    node = graph.get(str(node_id))
    if node is None:
        raise BackendError("this workflow has no node %r; it has %s"
                           % (node_id, ", ".join(sorted(graph))))
    inputs = node.get("inputs")
    if not isinstance(inputs, dict) or name not in inputs:
        raise BackendError(
            "node %s (%s) has no input %r; its inputs are %s"
            % (node_id, node.get("class_type"), name,
               ", ".join(sorted(inputs or ())) or "(none)"))
    inputs[name] = value


def _set_all(graph, class_type, name, value):
    """Same, but addressed by node class -- for inputs nobody should have to
    remap, such as which matting model a shipped template loads."""
    for node in graph.values():
        inputs = node.get("inputs") or {}
        if node.get("class_type") == class_type and name in inputs:
            node["inputs"][name] = value


def _has(graph, node_id, name):
    return name in ((graph.get(str(node_id)) or {}).get("inputs") or {})


def _current(graph, node_id, name):
    return ((graph.get(str(node_id)) or {}).get("inputs") or {}).get(name)


def _latent_size(size):
    """Width and height as ComfyUI will actually use them.

    EmptyLatentImage declares step 8, but execution.py validates only min and
    max, so 1020 is accepted and then floored into a 127-tall latent and a 1016px
    image. Rounding here makes that visible instead. 16..16384 is MAX_RESOLUTION.
    """
    out = []
    for value in (int(size[0]), int(size[1])):
        out.append(max(16, min(16384, value - value % 8)))
    return out[0], out[1]


def _cast(cast, key, value):
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise BackendError("-O %s=%r is not a %s" % (key, value, cast.__name__))


def _ids(value):
    """Node ids from a string, so -O node_image="10,11" reaches two LoadImages."""
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part) for part in value]


# --- server facts -------------------------------------------------------

def _combo(host, prefix, class_name, input_name):
    """The options of one combo input, or None if the server did not answer.

    Always the per-class route: bare /object_info serialises every installed node
    class and is megabytes on a real install. An unknown class is answered with
    {} and a 200, which is what makes this a feature test.
    """
    try:
        doc = request_json(_url(host, prefix, "/object_info/" + class_name),
                           timeout=PROBE_TIMEOUT, retries=0)
    except BackendError:
        return None
    info = doc.get(class_name) or {}
    spec = ((info.get("input") or {}).get("required") or {}).get(input_name)
    return _combo_options(spec)


def _combo_options(spec):
    """Two shapes are current on 0.33.x and a client has to handle both.

    Legacy V1 nodes (CheckpointLoaderSimple) put the list in element 0:
    [["a.safetensors", "b.safetensors"], {"tooltip": ...}]. V3 nodes -- which is
    most of comfy_extras, including LoadBackgroundRemovalModel -- say
    ["COMBO", {"options": [...], ...}] instead.
    """
    if not isinstance(spec, list) or not spec:
        return []
    if isinstance(spec[0], list):
        return list(spec[0])
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return list(spec[1].get("options") or [])
    return []


def _match(wanted, installed):
    """Exact first, then forgiving about separator and case -- and the answer is
    always the server's own spelling. See the module docstring for why."""
    if wanted in installed:
        return wanted
    normalised = _norm(wanted)
    for name in installed:
        if _norm(name) == normalised:
            return name
    return None


def _norm(name):
    return name.replace("\\", "/").lower()


def _listing(names, limit=20):
    shown = ", ".join(repr(n) for n in names[:limit])
    return shown if len(names) <= limit else "%s ... (%d total)" % (shown,
                                                                   len(names))


def _prompt_error(exc):
    text = str(exc)
    for needle, hint in PROMPT_HINTS:
        if needle in text:
            return BackendError("%s\n  %s" % (text, hint))
    return exc


# --- urls ---------------------------------------------------------------

def _base(host):
    """`host` may be "127.0.0.1:8188" or a full URL, which is how TLS is reached
    without inventing a second option for it (--tls-keyfile serves https)."""
    host = host.rstrip("/")
    return host if "://" in host else "http://" + host


def _url(host, prefix, path):
    return "%s%s%s" % (_base(host), _prefix(prefix), path)


def _ws_url(host, prefix, client_id):
    base = _base(host)
    scheme = "wss" if base.startswith("https://") else "ws"
    return "%s://%s%s/ws?clientId=%s" % (scheme, base.split("://", 1)[1],
                                         _prefix(prefix),
                                         urllib.parse.quote(client_id))


def _prefix(prefix):
    prefix = (prefix or "").strip("/")
    return "/" + prefix if prefix else ""
