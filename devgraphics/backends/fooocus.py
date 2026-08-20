"""
Headless client for a stock Fooocus install (no Fooocus-API fork needed), and the
Backend adapter over it.

Stock Fooocus exposes zero named Gradio endpoints -- /config reports 77
dependencies and not one `api_name` -- so `gradio_client` cannot bind to any of
them and this drives the raw queue protocol instead. The generate flow is two
chained dependencies that share a session_hash:

    [67] get_task(153 controls) -> gr.State
    [68] generate_clicked(gr.State) -> html, preview, progress_gallery, gallery

Calling the first alone silently produces nothing, which is exactly what an early
version of this client did. gr.State never crosses the wire; Gradio keeps it
server-side keyed by session_hash, so we pass null for state inputs and reuse one
hash for both calls. 152 of the 153 inputs have usable defaults in /config -- the
gr.State is the only one that does not -- so the vector is filled from those and
the handful we care about are overridden by label.

/config is fetched lazily and cached on the instance, never in __init__.
Capabilities has to be answerable with the GPU box switched off or --dry-run needs
the very thing it exists to avoid, and the fn_index layout check is a fact about
the *server*, so it belongs on first fetch rather than at construction.

Two dropdowns are matched by substring rather than equality because their choices
carry HTML: Performance, and the aspect ratio, whose label spells the size with a
U+00D7 multiplication sign rather than an ASCII "x".

SDXL emits no alpha, so this backend declares transparent=False and
postprocess.cutout() keys the charcoal backdrop out downstream. Which styles you
pick decides whether that works: measured, "Fooocus V2" + "Fooocus Sharp" leaves a
near-flat backdrop that keys cleanly, while "Sticker Designs" and "Simple Vector
Art" -- whose names promise the opposite -- lay down an asphalt texture that
defeats the flood fill. See docs/findings.md.

OS-agnostic: pure Python + websocket-client, no shell-outs, no platform paths.
"""

import json
import os
import urllib.parse
import uuid

from .._http import request_bytes, request_json
from ..postprocess import to_png
from .base import (BackendError, Capabilities, MissingDependency,
                   UnsupportedOption)

try:
    from websocket import create_connection
    HAVE_WEBSOCKET = True
except ImportError:                                       # pragma: no cover
    # websocket-client lives in the `local` extra: only Fooocus and ComfyUI open
    # a socket, and an OpenAI user has no reason to own it. Keeping the module
    # importable means `devgraphics backends` still lists and describes Fooocus
    # on a machine that cannot drive it.
    HAVE_WEBSOCKET = False

    def create_connection(*_args, **_kwargs):
        raise _no_transport()


def _no_transport():
    return MissingDependency("fooocus", "websocket-client", "local")


def _require_transport():
    """Refuse before any network I/O, not at the socket.

    generate() fetches /config first, so without this the missing package
    surfaces as "cannot reach 127.0.0.1:7865: connection refused" and sends
    someone hunting a server that was never the problem. A missing package is
    deterministic and theirs to fix; an unreachable host might be transient. The
    deterministic answer has to win.
    """
    if not HAVE_WEBSOCKET:
        raise _no_transport()

GET_TASK = 67
GENERATE = 68

#: Option keys FooocusBackend accepts. Anything else is a typo'd -O and has to
#: fail loudly: a style list that is silently ignored is how you find out, 88
#: icons later, that the whole set was rendered in the wrong style.
OPTIONS = frozenset(("styles", "sharpness", "guidance", "performance"))


class FooocusError(RuntimeError):
    pass


class Fooocus:
    def __init__(self, host="127.0.0.1:7865", timeout=900):
        self.host = host
        self.timeout = timeout
        self._cfg = None
        self._comps = None

    # --- plumbing -------------------------------------------------------

    @property
    def cfg(self):
        """/config, fetched once and cached on the instance.

        The layout check lives here rather than in __init__ for the reason in the
        module docstring: a constructor that talks to the network makes offline
        capabilities impossible. It also then fails on the right thing -- a
        mismatched Fooocus build is discovered when you first try to drive it,
        not when you merely name it.
        """
        if self._cfg is None:
            cfg = self._get_json("/config")
            deps = cfg.get("dependencies") or []
            if len(deps) <= GENERATE:
                raise FooocusError("unexpected Fooocus build: only %d dependencies"
                                   % len(deps))
            inputs = deps[GET_TASK].get("inputs") or []
            if len(inputs) < 100:
                raise FooocusError(
                    "fn_index %d is not get_task (%d inputs); Fooocus layout changed"
                    % (GET_TASK, len(inputs))
                )
            self._cfg = cfg
        return self._cfg

    @property
    def comps(self):
        """id -> component, so a control can be found by its label."""
        if self._comps is None:
            self._comps = {c["id"]: c for c in self.cfg["components"]}
        return self._comps

    def _get_json(self, path):
        # request_json rather than urllib directly, so a refused connection
        # arrives as BackendError("cannot reach ...") like every other backend's
        # does. One retry only: probe() should answer quickly when the box is off.
        return request_json("http://%s%s" % (self.host, path), timeout=60, retries=1)

    def _props(self, cid):
        return self.comps.get(cid, {}).get("props", {})

    def defaults(self, fn_index):
        """Every input for a dependency, prefilled from its component default."""
        return [self._props(cid).get("value") for cid in self.cfg["dependencies"][fn_index]["inputs"]]

    def _label_index(self, fn_index, label):
        for n, cid in enumerate(self.cfg["dependencies"][fn_index]["inputs"]):
            if (self._props(cid).get("label") or "") == label:
                return n
        raise FooocusError("no input labelled %r on fn_index %d" % (label, fn_index))

    def _choice_matching(self, fn_index, label, needle):
        """Radio/dropdown choices carry HTML, so match on a substring."""
        idx = self._label_index(fn_index, label)
        cid = self.cfg["dependencies"][fn_index]["inputs"][idx]
        for choice in self._props(cid).get("choices", []):
            value = choice[0] if isinstance(choice, (list, tuple)) else choice
            if needle in str(value):
                return idx, value
        raise FooocusError("no %s choice containing %r" % (label, needle))

    def _call(self, fn_index, data, session_hash, on_progress=None):
        ws = create_connection("ws://%s/queue/join" % self.host, timeout=self.timeout)
        try:
            while True:
                msg = json.loads(ws.recv())
                kind = msg.get("msg")
                if kind == "send_hash":
                    ws.send(json.dumps({"fn_index": fn_index, "session_hash": session_hash}))
                elif kind == "send_data":
                    ws.send(
                        json.dumps(
                            {
                                "fn_index": fn_index,
                                "data": data,
                                "session_hash": session_hash,
                                "event_data": None,
                            }
                        )
                    )
                elif kind == "process_generating":
                    if on_progress:
                        on_progress(msg.get("output", {}))
                elif kind == "process_completed":
                    out = msg.get("output", {})
                    if out.get("error"):
                        raise FooocusError(str(out["error"]))
                    return out
                elif kind in ("queue_full",):
                    raise FooocusError("Fooocus queue is full")
        finally:
            ws.close()

    # --- generation -----------------------------------------------------

    def generate(
        self,
        prompt,
        negative="",
        styles=None,
        size="1024\u00d71024",   # U+00D7, not an ASCII "x" -- see _choice_matching
        count=1,
        seed=None,
        performance="Speed",
        sharpness=None,
        guidance=None,
        on_progress=None,
    ):
        """Render `prompt` and return a list of local file paths on the server."""
        session = uuid.uuid4().hex
        data = self.defaults(GET_TASK)
        data[0] = None  # gr.State, resolved server-side

        data[2] = prompt
        data[3] = negative

        if styles is not None:
            data[self._label_index(GET_TASK, "Selected Styles")] = list(styles)

        idx, value = self._choice_matching(GET_TASK, "Performance", performance)
        data[idx] = value

        idx, value = self._choice_matching(GET_TASK, "Aspect Ratios", size)
        data[idx] = value

        data[self._label_index(GET_TASK, "Image Number")] = count
        data[self._label_index(GET_TASK, "Seed")] = str(seed if seed is not None else 0)

        if sharpness is not None:
            data[self._label_index(GET_TASK, "Image Sharpness")] = float(sharpness)
        if guidance is not None:
            data[self._label_index(GET_TASK, "Guidance Scale")] = float(guidance)

        # Random-seed checkbox would override an explicit seed, so force it off
        # whenever the caller pinned one (reproducibility is the whole point).
        if seed is not None:
            for n, cid in enumerate(self.cfg["dependencies"][GET_TASK]["inputs"]):
                if (self._props(cid).get("label") or "") == "Random":
                    data[n] = False

        self._call(GET_TASK, data, session)
        out = self._call(GENERATE, [None], session, on_progress=on_progress)
        return self._paths(out.get("data", []))

    @staticmethod
    def _paths(payload):
        """Pull file paths out of the finished-gallery output.

        Fooocus yields gr.update(...) rather than bare values, so the gallery
        arrives as {"__type__": "update", "value": [{"name": ..., "is_file": 1}]}.
        Walk the whole tree instead of assuming a fixed depth.
        """
        found = []

        def walk(node):
            if isinstance(node, dict):
                name = node.get("name") or node.get("path")
                if isinstance(name, str) and name:
                    found.append(name)
                elif "value" in node:
                    walk(node["value"])
            elif isinstance(node, list):
                for child in node:
                    walk(child)

        walk(payload)
        return found

    def fetch(self, remote_path):
        """The bytes of one generated file, without writing it anywhere.

        Split out of download() because the Backend contract is bytes: a hosted
        API has no path to hand back, so nothing above this layer should have to
        learn that Fooocus does.
        """
        url = "http://%s/file=%s" % (self.host, urllib.parse.quote(remote_path))
        return request_bytes(url, timeout=120, retries=2)

    def download(self, remote_path, dest):
        """fetch(), then write it. The path-first API predates the Backend
        contract; iconset still uses it to keep the raw 1024s on disk."""
        data = self.fetch(remote_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        return dest


class FooocusBackend:
    """Fooocus behind the Backend contract: bytes out, option keys checked in.

    Thin on purpose. The queue protocol lives in Fooocus above; this is the seam
    between Request/Capabilities and that client, and the one place where a
    Fooocus-specific failure becomes a BackendError.
    """

    def __init__(self, host="127.0.0.1:7865", timeout=900,
                 styles=("Fooocus V2", "Fooocus Sharp"), performance="Speed",
                 sharpness=None, guidance=None):
        self.host = host
        self.timeout = timeout
        self.styles = _as_styles(styles)
        self.performance = performance
        # sharpness and guidance are per-request knobs that also have to be
        # settable once: a config profile's [options] table reaches the
        # constructor, and a whole set wants one sharpness, not one per icon.
        # Request.options still overrides these per call.
        self.sharpness = sharpness
        self.guidance = guidance
        # Eager construction is only safe because Fooocus.__init__ no longer
        # fetches /config. It also keeps one config cache per backend instance
        # rather than one per icon.
        self.client = Fooocus(host=host, timeout=timeout)

    @property
    def capabilities(self):
        return Capabilities(
            name="fooocus",
            seed=True,
            deterministic=True,
            negative_prompt=True,
            transparent=False,
            reference_images=0,
            batch=True,
            sizes=((1024, 1024),),
            cost_per_image=None,
            notes=(
                "SDXL emits no alpha channel, so nothing here is transparent; "
                "postprocess.cutout() keys the backdrop out instead.",
                "Measured: 'Fooocus V2' + 'Fooocus Sharp' leaves a near-flat dark "
                "backdrop that keys out cleanly. 'Sticker Designs' and 'Simple "
                "Vector Art' lay down an asphalt texture that defeats the flood "
                "fill, despite what their names promise.",
                "1024x1024 only. The Aspect Ratios dropdown really does offer many "
                "ratios, but reading them means fetching /config, and capabilities "
                "must answer with the server switched off -- so the square one this "
                "pipeline wants is pinned instead.",
                "The ImagePrompt controls are among the 153 inputs but this client "
                "does not wire them, so there is no reference image here; a fixed "
                "seed carries consistency instead.",
                "Determinism is per install: same seed, same checkpoint, same "
                "Fooocus version. Seed 77777 means nothing on another backend.",
            ),
        )

    def generate(self, request):
        _require_transport()
        unknown = sorted(set(request.options) - OPTIONS)
        if unknown:
            raise UnsupportedOption(
                "fooocus does not accept %s; accepted keys: %s"
                % (", ".join(unknown), ", ".join(sorted(OPTIONS))))

        # The Aspect Ratios dropdown is matched by substring, and its choices
        # spell the size with U+00D7 MULTIPLICATION SIGN where the "x" would go:
        # "1024\u00d71024 <span ...> \u2223 1:1</span>". An ASCII needle matches
        # nothing at all. Written as an escape so this file stays ASCII.
        size = "%d\u00d7%d" % (request.size[0], request.size[1])

        options = request.options
        try:
            paths = self.client.generate(
                prompt=request.prompt,
                negative=request.negative,
                styles=_as_styles(options.get("styles", self.styles)),
                size=size,
                count=request.count,
                seed=request.seed,
                performance=options.get("performance", self.performance),
                sharpness=options.get("sharpness", self.sharpness),
                guidance=options.get("guidance", self.guidance),
            )
        except FooocusError as exc:
            raise BackendError("fooocus: %s" % exc) from exc

        if not paths:
            raise BackendError("fooocus returned no image path; the real error is "
                               "on the Fooocus console")

        # to_png because output_format is a host-side Fooocus setting: a box set
        # to jpeg would otherwise hand cutout() the exact ringing thresh=42 keys
        # on. Bytes that are already PNG come back untouched.
        #
        # Not truncated to request.count -- Image Number is honoured server-side,
        # and dropping an image the GPU already rendered would hide a mismatch
        # rather than surface it.
        return [to_png(self.client.fetch(p)) for p in paths]

    @classmethod
    def probe(cls, host="127.0.0.1:7865", timeout=900, styles=None,
              performance=None):
        """Reachability and layout, from /config alone. Never renders anything.

        Reports the missing transport before trying the host: answering "up" on a
        machine that cannot open a socket would be the wrong kind of true.

        `styles` and `performance` are accepted so probe() takes the same option
        table as the constructor, and then ignored: Fooocus drops an unknown style
        name silently rather than rejecting it, so nothing about them can be
        confirmed without spending a generation.
        """
        if not HAVE_WEBSOCKET:
            return False, str(_no_transport())
        try:
            cfg = Fooocus(host=host, timeout=timeout)._get_json("/config")
        except Exception as exc:
            return False, "cannot reach Fooocus at %s: %s" % (host, exc)
        deps = cfg.get("dependencies") or []
        line = "gradio %s, %d dependencies" % (cfg.get("version") or "unknown",
                                               len(deps))
        if len(deps) <= GENERATE:
            return False, ("%s -- fn_index %d is missing, so this is not a layout "
                           "this client can drive" % (line, GENERATE))
        return True, line


def _as_styles(value):
    """Style names as a list.

    A string is split on commas rather than passed straight through, because `-O
    styles="Fooocus V2,Fooocus Sharp"` arrives as one string and list() on a
    string yields single characters -- which Fooocus accepts and then renders with
    no style at all.
    """
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value)
