"""
Tests for the Fooocus client and its Backend adapter.

Nothing here touches the network, and one test enforces that rather than assuming
it: every transport the module imports is replaced with something that raises, and
a FooocusBackend is then constructed and asked for its capabilities. That is the
regression the lazy /config property exists to prevent -- --dry-run and the
capability report have to work with the GPU box switched off.

The rest asserts positions in the 153-slot data vector by index. Fooocus has no
named endpoints, so the only contract with the server is "slot 7 is Image Number";
a test that checked "generate() called _call" would pass while sending the seed to
the sharpness control.

The fake /config is a stripped copy of the real one: same shape (components with
props, dependencies listing component ids), same 153 inputs on fn_index 67, same
HTML-laden aspect-ratio choices carrying a U+00D7 multiplication sign, which is
the reason the client matches those by substring at all.
"""

import io
import json
import os
import tempfile
import unittest
import urllib.parse
from unittest import mock

from PIL import Image

from devgraphics.backends import fooocus
from devgraphics.backends.base import (BackendError, Capabilities, Request,
                                       UnsupportedOption)
from devgraphics.backends.fooocus import Fooocus, FooocusBackend, FooocusError

PNG_MAGIC = bytes((0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A))

# The dropdown labels really do carry HTML and a U+00D7 MULTIPLICATION SIGN.
# Escaped so this file stays ASCII; the wire carries the character itself.
SQUARE = "1024\u00d71024 <span style='color: grey;'> \u2223 1:1</span>"
PORTRAIT = "896\u00d71152 <span style='color: grey;'> \u2223 7:9</span>"

REMOTE = "C:\\Fooocus\\outputs\\2026-08-20\\0001.png"

#: Slot -> label, for the controls the client reaches by name. The real build
#: scatters these through 153 inputs; the exact indices are arbitrary here, which
#: is the point -- the client must find them by label, not by position.
LABELS = {
    4: "Selected Styles",
    5: "Performance",
    6: "Aspect Ratios",
    7: "Image Number",
    8: "Seed",
    9: "Random",
    10: "Image Sharpness",
    11: "Guidance Scale",
}

CHOICES = {
    4: ["Fooocus V2", "Fooocus Sharp", "Sticker Designs", "Simple Vector Art"],
    5: ["Quality", "Speed", "Extreme Speed"],
    6: [PORTRAIT, SQUARE],
}

#: Published defaults. Deliberately not the values the tests ask for, so a slot
#: that was never written is visibly different from one that was.
VALUES = {
    0: None,            # the gr.State -- the one input with no usable default
    2: "",
    3: "",
    4: ["Fooocus V2"],
    5: "Quality",
    6: PORTRAIT,
    7: 2,
    8: "0",
    9: True,
    10: 2.0,
    11: 4.0,
}

#: A finished generate_clicked payload: four gr.update() wrappers, the gallery
#: last. Copied in shape from docs/findings.md -- the paths are on the Fooocus
#: host, and only a second request turns them into bytes.
COMPLETED = {
    "data": [
        {"__type__": "update", "value": "<div>Finished</div>"},
        {"__type__": "update", "visible": False},
        {"__type__": "update", "visible": False, "value": None},
        {"__type__": "update",
         "value": [{"name": REMOTE, "data": None, "is_file": True}]},
    ]
}


def make_config(inputs=153, dependencies=77):
    components = []
    ids = []
    for n in range(inputs):
        cid = 1000 + n
        ids.append(cid)
        props = {"value": VALUES.get(n, "default-%d" % n)}
        if n in LABELS:
            props["label"] = LABELS[n]
        if n in CHOICES:
            props["choices"] = CHOICES[n]
        components.append({"id": cid, "props": props})
    deps = [{"inputs": [], "outputs": []} for _ in range(dependencies)]
    if dependencies > fooocus.GENERATE:
        deps[fooocus.GET_TASK] = {"inputs": ids, "outputs": [ids[0]]}
        deps[fooocus.GENERATE] = {"inputs": [ids[0]], "outputs": ids[:4]}
    return {"version": "3.41.2", "components": components, "dependencies": deps}


def image_bytes(fmt):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (13, 13, 13)).save(buf, fmt)
    return buf.getvalue()


class Recorder:
    """Stands in for the queue protocol, and remembers what it was sent."""

    def __init__(self, output=None):
        self.calls = []
        self.output = output if output is not None else COMPLETED

    def __call__(self, fn_index, data, session_hash, on_progress=None):
        self.calls.append((fn_index, list(data), session_hash))
        return self.output if fn_index == fooocus.GENERATE else {}


class FooocusTestCase(unittest.TestCase):

    def client(self, config=None, output=None):
        """A Fooocus wired to a fake /config and a fake queue."""
        client = Fooocus(host="127.0.0.1:7865")
        self.fetches = []
        self.configs = []

        def get_json(path):
            self.configs.append(path)
            return make_config() if config is None else config

        client._get_json = get_json
        self.calls = Recorder(output)
        client._call = self.calls
        return client

    def no_network(self):
        """Every transport this module imported, replaced with an explosion."""
        def boom(*args, **kwargs):
            raise AssertionError("network call: %r" % (args,))
        for name in ("request_json", "request_bytes", "create_connection"):
            patcher = mock.patch.object(fooocus, name, boom)
            patcher.start()
            self.addCleanup(patcher.stop)


class TestOfflineConstruction(FooocusTestCase):

    def test_constructing_anything_touches_no_transport(self):
        self.no_network()
        client = Fooocus(host="10.0.0.9:7865")
        backend = FooocusBackend(host="10.0.0.9:7865")
        self.assertEqual(client.host, "10.0.0.9:7865")
        self.assertIsInstance(backend.capabilities, Capabilities)

    def test_capabilities_with_the_server_switched_off(self):
        backend = FooocusBackend()

        def dead(path):
            raise BackendError("cannot reach http://127.0.0.1:7865/config")

        backend.client._get_json = dead
        caps = backend.capabilities
        self.assertEqual(caps.name, "fooocus")
        self.assertTrue(caps.seed)
        self.assertTrue(caps.deterministic)
        self.assertTrue(caps.negative_prompt)
        self.assertTrue(caps.batch)
        self.assertFalse(caps.transparent)
        self.assertEqual(caps.reference_images, 0)
        self.assertEqual(caps.sizes, ((1024, 1024),))
        self.assertIsNone(caps.cost_per_image)

    def test_capabilities_carry_the_measured_style_findings(self):
        notes = " ".join(FooocusBackend().capabilities.notes)
        self.assertIn("Fooocus Sharp", notes)
        self.assertIn("Sticker Designs", notes)
        self.assertIn("Simple Vector Art", notes)
        self.assertIn("asphalt", notes)
        self.assertIn("cutout", notes)
        self.assertIn("1024x1024", notes)
        # Windows console: a note is printed, so it has to be printable there.
        notes.encode("ascii")

    def test_unsupported_option_is_rejected_before_any_transport(self):
        self.no_network()
        backend = FooocusBackend()
        request = Request(prompt="a flame", options={"stylez": ["Fooocus V2"]})
        with self.assertRaises(UnsupportedOption) as caught:
            backend.generate(request)
        self.assertIn("stylez", str(caught.exception))
        self.assertIn("sharpness", str(caught.exception))   # lists what is legal


class TestConfig(FooocusTestCase):

    def test_config_is_fetched_once_and_cached(self):
        client = self.client()
        client.cfg
        client.comps
        client.defaults(fooocus.GET_TASK)
        client.generate(prompt="a flame", size=SQUARE)
        self.assertEqual(self.configs, ["/config"])

    def test_short_dependency_list_raises_on_use_not_construction(self):
        client = self.client(config=make_config(dependencies=12))
        with self.assertRaises(FooocusError) as caught:
            client.cfg
        self.assertIn("12 dependencies", str(caught.exception))

    def test_wrong_fn_index_layout_raises_on_use_not_construction(self):
        client = self.client(config=make_config(inputs=20))
        with self.assertRaises(FooocusError) as caught:
            client.cfg
        self.assertIn("not get_task", str(caught.exception))


class TestDataVector(FooocusTestCase):

    def test_every_slot_the_client_sets(self):
        client = self.client()
        client.generate(prompt="a flame", negative="photo, 3d render",
                        styles=["Fooocus V2", "Fooocus Sharp"], size=SQUARE,
                        count=3, seed=77777, performance="Speed",
                        sharpness=3.5, guidance=5)

        self.assertEqual([c[0] for c in self.calls.calls],
                         [fooocus.GET_TASK, fooocus.GENERATE])
        fn_index, data, session = self.calls.calls[0]
        self.assertEqual(len(data), 153)
        self.assertIsNone(data[0])                       # gr.State stays null
        self.assertEqual(data[2], "a flame")
        self.assertEqual(data[3], "photo, 3d render")
        self.assertEqual(data[4], ["Fooocus V2", "Fooocus Sharp"])
        self.assertEqual(data[5], "Speed")               # not "Extreme Speed"
        self.assertEqual(data[6], SQUARE)
        self.assertEqual(data[7], 3)
        self.assertEqual(data[8], "77777")               # a string, as the box is
        self.assertIs(data[9], False)                    # Random forced off
        self.assertEqual(data[10], 3.5)
        self.assertEqual(data[11], 5.0)
        self.assertIsInstance(data[11], float)
        self.assertEqual(data[12], "default-12")         # untouched slots survive
        self.assertEqual(data[152], "default-152")

    def test_both_calls_share_one_session_hash(self):
        client = self.client()
        client.generate(prompt="a flame", size=SQUARE)
        first, second = self.calls.calls
        self.assertEqual(first[2], second[2])
        self.assertEqual(second[1], [None])              # gr.State, server-side

    def test_no_seed_leaves_the_random_checkbox_alone(self):
        client = self.client()
        client.generate(prompt="a flame", size=SQUARE)
        _fn, data, _session = self.calls.calls[0]
        self.assertEqual(data[8], "0")
        self.assertIs(data[9], True)                     # published default

    def test_omitted_knobs_keep_the_published_defaults(self):
        client = self.client()
        client.generate(prompt="a flame", size=SQUARE)
        _fn, data, _session = self.calls.calls[0]
        self.assertEqual(data[4], ["Fooocus V2"])
        self.assertEqual(data[10], 2.0)
        self.assertEqual(data[11], 4.0)

    def test_unknown_aspect_ratio_names_the_label(self):
        client = self.client()
        with self.assertRaises(FooocusError) as caught:
            client.generate(prompt="a flame", size="640x480")
        self.assertIn("Aspect Ratios", str(caught.exception))


class TestBackendGenerate(FooocusTestCase):

    def backend(self, image=None, output=None):
        backend = FooocusBackend()
        backend.client = self.client(output=output)
        self.fetched = []

        def fetch(remote_path):
            self.fetched.append(remote_path)
            return image if image is not None else image_bytes("PNG")

        backend.client.fetch = fetch
        return backend

    def test_request_reaches_the_data_vector(self):
        backend = self.backend()
        backend.generate(Request(prompt="a flame", negative="photo", seed=77777,
                                 size=(1024, 1024), count=2))
        _fn, data, _session = self.calls.calls[0]
        self.assertEqual(data[2], "a flame")
        self.assertEqual(data[3], "photo")
        self.assertEqual(data[4], ["Fooocus V2", "Fooocus Sharp"])   # the default
        self.assertEqual(data[6], SQUARE)                # matched via U+00D7
        self.assertEqual(data[7], 2)
        self.assertEqual(data[8], "77777")
        self.assertIs(data[9], False)

    def test_options_override_the_instance_defaults(self):
        backend = self.backend()
        backend.generate(Request(prompt="a flame", options={
            "styles": ["Simple Vector Art"], "performance": "Quality",
            "sharpness": 6, "guidance": 7}))
        _fn, data, _session = self.calls.calls[0]
        self.assertEqual(data[4], ["Simple Vector Art"])
        self.assertEqual(data[5], "Quality")
        self.assertEqual(data[10], 6.0)
        self.assertEqual(data[11], 7.0)

    def test_a_comma_separated_style_string_is_split_not_exploded(self):
        backend = self.backend()
        backend.generate(Request(prompt="a flame",
                                 options={"styles": "Fooocus V2, Fooocus Sharp"}))
        _fn, data, _session = self.calls.calls[0]
        self.assertEqual(data[4], ["Fooocus V2", "Fooocus Sharp"])

    def test_gallery_becomes_png_bytes(self):
        png = image_bytes("PNG")
        backend = self.backend(image=png)
        out = backend.generate(Request(prompt="a flame"))
        self.assertEqual(self.fetched, [REMOTE])
        self.assertEqual(out, [png])
        self.assertTrue(all(isinstance(b, bytes) for b in out))

    def test_a_jpeg_host_is_transcoded(self):
        # output_format is a Fooocus setting, so a host can hand back JPEG. The
        # cutout downstream keys on exactly the ringing JPEG puts round an outline.
        jpeg = image_bytes("JPEG")
        backend = self.backend(image=jpeg)
        out = backend.generate(Request(prompt="a flame"))
        self.assertNotEqual(out[0][:8], jpeg[:8])
        self.assertEqual(out[0][:8], PNG_MAGIC)

    def test_two_images_come_back_in_gallery_order(self):
        second = "C:\\Fooocus\\outputs\\2026-08-20\\0002.png"
        output = {"data": [{"__type__": "update", "value": "<div>done</div>"},
                           {"__type__": "update", "visible": False},
                           {"__type__": "update", "visible": False},
                           {"__type__": "update",
                            "value": [{"name": REMOTE, "is_file": True},
                                      {"name": second, "is_file": True}]}]}
        backend = self.backend(output=output)
        out = backend.generate(Request(prompt="a flame", count=2))
        self.assertEqual(self.fetched, [REMOTE, second])
        self.assertEqual(len(out), 2)

    def test_an_empty_gallery_is_a_backend_error(self):
        output = {"data": [{"__type__": "update", "value": None}]}
        backend = self.backend(output=output)
        with self.assertRaises(BackendError):
            backend.generate(Request(prompt="a flame"))

    def test_a_fooocus_error_becomes_a_backend_error(self):
        backend = self.backend()

        def explode(*args, **kwargs):
            raise FooocusError("Fooocus queue is full")

        backend.client._call = explode
        with self.assertRaises(BackendError) as caught:
            backend.generate(Request(prompt="a flame"))
        self.assertIn("queue is full", str(caught.exception))


class TestFetchAndDownload(FooocusTestCase):

    def test_fetch_returns_bytes_and_quotes_the_path(self):
        seen = {}

        def fake_bytes(url, **kwargs):
            seen["url"] = url
            return b"raw-png-bytes"

        with mock.patch.object(fooocus, "request_bytes", fake_bytes):
            data = Fooocus(host="127.0.0.1:7865").fetch(REMOTE)
        self.assertEqual(data, b"raw-png-bytes")
        self.assertEqual(seen["url"], "http://127.0.0.1:7865/file=%s"
                         % urllib.parse.quote(REMOTE))

    def test_download_writes_what_fetch_returned(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dest = os.path.join(tmp.name, "raw", "flame.png")
        client = Fooocus()
        client.fetch = lambda remote_path: b"raw-png-bytes"
        self.assertEqual(client.download(REMOTE, dest), dest)
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), b"raw-png-bytes")


class TestProbe(FooocusTestCase):

    def test_probe_reports_gradio_version_and_dependency_count(self):
        with mock.patch.object(fooocus, "request_json",
                               lambda url, **kw: make_config()):
            ok, detail = FooocusBackend.probe(host="127.0.0.1:7865")
        self.assertTrue(ok)
        self.assertEqual(detail, "gradio 3.41.2, 77 dependencies")

    def test_probe_generates_nothing(self):
        def boom(*args, **kwargs):
            raise AssertionError("probe opened a websocket")

        with mock.patch.object(fooocus, "create_connection", boom):
            with mock.patch.object(fooocus, "request_json",
                                   lambda url, **kw: make_config()):
                ok, _detail = FooocusBackend.probe()
        self.assertTrue(ok)

    def test_probe_fails_softly_when_nothing_is_listening(self):
        def dead(url, **kwargs):
            raise BackendError("cannot reach %s: [Errno 111] refused" % url)

        with mock.patch.object(fooocus, "request_json", dead):
            ok, detail = FooocusBackend.probe(host="10.0.0.9:7865")
        self.assertFalse(ok)
        self.assertIn("10.0.0.9:7865", detail)

    def test_probe_rejects_a_layout_it_cannot_drive(self):
        with mock.patch.object(fooocus, "request_json",
                               lambda url, **kw: make_config(dependencies=12)):
            ok, detail = FooocusBackend.probe()
        self.assertFalse(ok)
        self.assertIn("12 dependencies", detail)

    def test_probe_takes_the_same_options_as_the_constructor(self):
        with mock.patch.object(fooocus, "request_json",
                               lambda url, **kw: make_config()):
            ok, _detail = FooocusBackend.probe(host="127.0.0.1:7865", timeout=30,
                                               styles=["Fooocus V2"],
                                               performance="Speed")
        self.assertTrue(ok)


class TestQueueVisibility(unittest.TestCase):
    """Gradio's `estimation` frame, which the client used to drop on the floor.

    Found against a real install, not by reading the code: Fooocus had been up
    for four hours with a stalled queue, and every client -- this one and the 0.1
    original -- blocked silently in recv() behind three jammed jobs. The frame
    that said "rank 2 of 3" was arriving the whole time and nothing looked at it.
    A waiting client and a wedged server were indistinguishable.
    """

    def client(self):
        return fooocus.Fooocus(host="127.0.0.1:7865")

    def test_a_queue_position_is_reported(self):
        seen = []
        self.client()._queued(
            {"rank": 2, "queue_size": 3, "rank_eta": 26.9}, seen.append)
        self.assertEqual(len(seen), 1)
        # Human-facing, so one-based: "position 3 of 3", not "rank 2".
        self.assertIn("position 3 of 3", seen[0]["message"])
        self.assertIn("27s", seen[0]["message"])
        self.assertEqual(seen[0]["queue"], (2, 3))

    def test_the_message_says_what_a_stuck_queue_means(self):
        seen = []
        self.client()._queued({"rank": 0, "queue_size": 1}, seen.append)
        self.assertIn("stalled", seen[0]["message"],
                      "a queue that never moves must not read as normal waiting")

    def test_an_unchanged_position_is_not_repeated(self):
        """Gradio re-sends estimation frequently; 900 identical lines is noise."""
        seen = []
        client = self.client()
        for _ in range(5):
            client._queued({"rank": 1, "queue_size": 2}, seen.append)
        self.assertEqual(len(seen), 1)

    def test_movement_up_the_queue_is_reported(self):
        seen = []
        client = self.client()
        client._queued({"rank": 2, "queue_size": 3}, seen.append)
        client._queued({"rank": 1, "queue_size": 2}, seen.append)
        client._queued({"rank": 0, "queue_size": 1}, seen.append)
        self.assertEqual(len(seen), 3)
        self.assertEqual(client.queue_position, (0, 1))

    def test_a_frame_without_a_rank_is_ignored(self):
        seen = []
        client = self.client()
        client._queued({"queue_size": 3}, seen.append)
        self.assertEqual(seen, [])
        self.assertIsNone(client.queue_position)

    def test_the_estimation_frame_is_handled_in_the_call_loop(self):
        """The reason all of the above matters: _call must not drop the frame."""
        client = self.client()
        frames = [
            {"msg": "send_hash"},
            {"msg": "estimation", "rank": 1, "queue_size": 2},
            {"msg": "send_data"},
            {"msg": "process_completed", "output": {"data": []}},
        ]
        seen = []
        with mock.patch.object(fooocus, "create_connection",
                               lambda *a, **k: _Scripted(frames)):
            client._call(67, [None], "hash", on_progress=seen.append)
        self.assertTrue(any("queue" in item for item in seen),
                        "the estimation frame never reached on_progress")


class _Scripted(object):
    """A websocket that replays a fixed list of frames."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []

    def recv(self):
        if not self.frames:
            raise AssertionError("client kept reading past the last frame")
        return json.dumps(self.frames.pop(0))

    def send(self, payload):
        self.sent.append(json.loads(payload))

    def close(self):
        pass


if __name__ == "__main__":
    unittest.main()
