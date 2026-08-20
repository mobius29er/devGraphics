"""
Headless client for a stock Fooocus install (no Fooocus-API fork needed).

Stock Fooocus exposes zero named Gradio endpoints, so this drives the raw
queue protocol instead. The generate flow is two chained dependencies that
share a session_hash:

    [67] get_task(153 controls) -> gr.State
    [68] generate_clicked(gr.State) -> html, preview, progress_gallery, gallery

gr.State never crosses the wire; Gradio keeps it server-side keyed by
session_hash, so we pass null for state inputs and reuse one hash for both
calls. Every other control is filled from the component defaults published
in /config, so we only override the handful we actually care about.

OS-agnostic: pure Python + websocket-client, no shell-outs, no platform paths.
"""

import json
import os
import ssl
import urllib.parse
import urllib.request
import uuid

from websocket import create_connection

GET_TASK = 67
GENERATE = 68


class FooocusError(RuntimeError):
    pass


class Fooocus:
    def __init__(self, host="127.0.0.1:7865", timeout=900):
        self.host = host
        self.timeout = timeout
        self.cfg = self._get_json("/config")
        self.comps = {c["id"]: c for c in self.cfg["components"]}
        deps = self.cfg["dependencies"]
        if len(deps) <= GENERATE:
            raise FooocusError("unexpected Fooocus build: only %d dependencies" % len(deps))
        if len(deps[GET_TASK]["inputs"]) < 100:
            raise FooocusError(
                "fn_index %d is not get_task (%d inputs); Fooocus layout changed"
                % (GET_TASK, len(deps[GET_TASK]["inputs"]))
            )

    # --- plumbing -------------------------------------------------------

    def _get_json(self, path):
        url = "http://%s%s" % (self.host, path)
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)

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
        size="1024×1024",
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

    def download(self, remote_path, dest):
        url = "http://%s/file=%s" % (self.host, urllib.parse.quote(remote_path))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
        return dest
