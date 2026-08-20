"""Tests for the InvokeAI backend.

Nothing here touches the network: devgraphics._http.request_json and
request_bytes are replaced by a fake server whose payloads are shaped from the
research report (EnqueueBatchResult, SessionQueueItem with session.results keyed
by prepared node ids). The assertions are on the REQUEST the backend builds --
exact URLs, exact node fields, exact edges -- because that request is the part a
version bump breaks.
"""

import io
import unittest
import warnings
from unittest import mock

from PIL import Image

from devgraphics import _http
from devgraphics.backends import invokeai
from devgraphics.backends.base import (BackendError, Capabilities, Request,
                                       UnsupportedOption)


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (17, 17, 17)).save(buf, "PNG")
    return buf.getvalue()


PNG = _png_bytes()

MODEL = {
    "key": "b1d9e1f2-0000-4c1a-9d0f-3f7c2b6a5e10",   # uuid, changes on reinstall
    "hash": "blake3:6c6f6e6768617368",               # stable identity
    "name": "juggernautXL_v8Rundiffusion",
    "base": "sdxl",
    "type": "main",
    "path": "models/sdxl/main/juggernautXL_v8Rundiffusion.safetensors",
    "format": "checkpoint",
}

IMAGE_NAME = "a1b2c3d4-1111-2222-3333-444455556666.png"


def image_result(name=IMAGE_NAME):
    return {"type": "image_output",
            "image": {"image_name": name},
            "width": 1024, "height": 1024}


def queue_item(item_id, status, results=None, mapping=None, error=None):
    """A SessionQueueItem, trimmed to the fields this backend reads."""
    item = {
        "item_id": item_id, "status": status, "batch_id": "batch-1",
        "session_id": "session-1", "queue_id": "default", "priority": 0,
        "error_type": None, "error_message": None, "error_traceback": None,
        "session": {
            "id": "session-1",
            "results": results if results is not None else {},
            "errors": {},
            "prepared_source_mapping": mapping if mapping is not None else {},
        },
    }
    if error:
        item["error_type"], item["error_message"] = error
    return item


class FakeServer(object):
    """Routes on URL, records every call, never opens a socket."""

    def __init__(self, version="6.13.8", models=None, statuses=None,
                 enqueued=None, requested=None, item_ids=None, item=None):
        self.version = version
        self.models = models if models is not None else [dict(MODEL)]
        self.statuses = statuses or ["completed"]
        self.enqueued = enqueued
        self.requested = requested
        self.item_ids = item_ids
        self.item = item
        self.json_calls = []
        self.byte_calls = []
        self._polls = {}

    # signature mirrors _http.request_json, which the backend calls by keyword
    def request_json(self, url, payload=None, headers=None, method=None,
                     timeout=None, retries=None, sleep=None):
        self.json_calls.append((url, payload))
        if "/api/v1/app/version" in url:
            return {"version": self.version}
        if "/api/v2/models/" in url:
            return {"models": self.models}
        if "enqueue_batch" in url:
            return self._enqueue(payload)
        if "/queue/" in url and "/i/" in url:
            return self._poll(int(url.rsplit("/", 1)[1]))
        raise AssertionError("unexpected URL: %s" % url)

    def request_bytes(self, url, payload=None, headers=None, method=None,
                      timeout=None, retries=None, sleep=None):
        self.byte_calls.append(url)
        return PNG

    def _enqueue(self, payload):
        data = payload["batch"].get("data")
        count = len(data[0][0]["items"]) if data else 1
        ids = self.item_ids if self.item_ids is not None else \
            [41 + n for n in range(count)]
        return {"queue_id": "default",
                "enqueued": count if self.enqueued is None else self.enqueued,
                "requested": count if self.requested is None else self.requested,
                "batch": dict(payload["batch"], batch_id="batch-1"),
                "priority": 0,
                "item_ids": ids}

    def _poll(self, item_id):
        seen = self._polls.get(item_id, 0)
        self._polls[item_id] = seen + 1
        status = self.statuses[min(seen, len(self.statuses) - 1)]
        if self.item is not None and status == "completed":
            return self.item
        if status != "completed":
            return queue_item(item_id, status)
        name = "%s-%s" % (item_id, IMAGE_NAME)
        return queue_item(item_id, "completed",
                          results={"prepared-%s" % item_id: image_result(name)},
                          mapping={"prepared-%s" % item_id: invokeai.NODE_L2I})

    def polls(self, item_id):
        return self._polls.get(item_id, 0)

    def urls(self):
        return [url for url, _ in self.json_calls]

    def enqueue_payload(self):
        for url, payload in self.json_calls:
            if "enqueue_batch" in url:
                return payload
        raise AssertionError("nothing was enqueued")


class InvokeAITestCase(unittest.TestCase):

    def install(self, server):
        for name in ("request_json", "request_bytes"):
            patcher = mock.patch.object(_http, name, getattr(server, name))
            patcher.start()
            self.addCleanup(patcher.stop)
        return server

    def backend(self, **options):
        options.setdefault("poll_interval", 0)
        return invokeai.InvokeAIBackend(**options)


class TestCapabilities(InvokeAITestCase):

    def test_answerable_with_the_server_switched_off(self):
        def explode(*_args, **_kwargs):
            raise AssertionError("capabilities must not touch the network")

        for name in ("request_json", "request_bytes"):
            patcher = mock.patch.object(_http, name, explode)
            patcher.start()
            self.addCleanup(patcher.stop)

        caps = self.backend(host="192.0.2.1:9090").capabilities
        self.assertIsInstance(caps, Capabilities)
        self.assertEqual(caps.name, "invokeai")
        self.assertTrue(caps.seed)
        self.assertTrue(caps.deterministic)      # noise.use_cpu defaults true
        self.assertTrue(caps.negative_prompt)
        self.assertTrue(caps.batch)
        self.assertFalse(caps.transparent)       # l2i decodes to opaque RGB
        self.assertEqual(caps.sizes, ())         # any size, rounded to /8
        self.assertIsNone(caps.cost_per_image)   # local GPU
        self.assertTrue(any("use_cache" in note for note in caps.notes))


class TestGraph(InvokeAITestCase):

    def request(self, **kwargs):
        kwargs.setdefault("prompt", "flat vector sticker icon of a flame")
        kwargs.setdefault("negative", "photo, realistic, text")
        kwargs.setdefault("seed", 77777)
        return Request(**kwargs)

    def test_happy_path_enqueue_body(self):
        server = self.install(FakeServer())
        out = self.backend().generate(self.request())

        self.assertEqual(out, [PNG])

        urls = server.urls()
        self.assertEqual(
            urls[0], "http://127.0.0.1:9090/api/v1/app/version")
        self.assertEqual(
            urls[1],
            "http://127.0.0.1:9090/api/v2/models/?base_models=sdxl&model_type=main")
        self.assertEqual(
            urls[2],
            "http://127.0.0.1:9090/api/v1/queue/default/enqueue_batch")
        self.assertEqual(
            urls[3], "http://127.0.0.1:9090/api/v1/queue/default/i/41")
        self.assertEqual(
            server.byte_calls,
            ["http://127.0.0.1:9090/api/v1/images/i/41-%s/full" % IMAGE_NAME])

        payload = server.enqueue_payload()
        # Body_enqueue_batch, not a bare Batch.
        self.assertEqual(sorted(payload), ["batch", "prepend"])
        self.assertIs(payload["prepend"], False)
        batch = payload["batch"]
        self.assertEqual(batch["runs"], 1)
        self.assertEqual(batch["origin"], "devgraphics")
        self.assertNotIn("data", batch)          # count == 1 needs no expansion

    def test_graph_nodes(self):
        server = self.install(FakeServer())
        self.backend().generate(self.request())
        graph = server.enqueue_payload()["batch"]["graph"]
        nodes = graph["nodes"]

        self.assertEqual(sorted(nodes), sorted([
            invokeai.NODE_MODEL, invokeai.NODE_POS, invokeai.NODE_NEG,
            invokeai.NODE_NOISE, invokeai.NODE_DENOISE, invokeai.NODE_L2I]))
        for key, node in nodes.items():
            self.assertEqual(key, node["id"])    # NodeIdMismatchError guard

        model = nodes[invokeai.NODE_MODEL]
        self.assertEqual(model["type"], "sdxl_model_loader")
        self.assertEqual(model["model"], {
            "key": MODEL["key"], "hash": MODEL["hash"], "name": MODEL["name"],
            "base": "sdxl", "type": "main"})

        noise = nodes[invokeai.NODE_NOISE]
        self.assertEqual(noise["type"], "noise")
        self.assertEqual(noise["seed"], 77777)
        self.assertEqual((noise["width"], noise["height"]), (1024, 1024))
        self.assertIs(noise["use_cpu"], True)
        self.assertEqual(noise["noise_type"], "SD")

        denoise = nodes[invokeai.NODE_DENOISE]
        self.assertEqual(denoise["type"], "denoise_latents")
        self.assertEqual(denoise["steps"], invokeai.DEFAULT_STEPS)
        self.assertEqual(denoise["cfg_scale"], invokeai.DEFAULT_CFG_SCALE)
        self.assertEqual(denoise["scheduler"], invokeai.DEFAULT_SCHEDULER)
        self.assertEqual(denoise["denoising_start"], 0.0)
        self.assertEqual(denoise["denoising_end"], 1.0)
        # size lives on `noise`; width on denoise_latents is a 422 for SDXL
        self.assertNotIn("width", denoise)
        self.assertNotIn("height", denoise)

        l2i = nodes[invokeai.NODE_L2I]
        self.assertEqual(l2i["type"], "l2i")
        self.assertIs(l2i["is_intermediate"], False)
        self.assertNotIn("board", l2i)

    def test_both_sdxl_conditioning_fields_are_set(self):
        """SDXL has two text encoders; an empty `style` silently diverges."""
        server = self.install(FakeServer())
        request = self.request(prompt="a flame", negative="photo, text")
        self.backend().generate(request)
        nodes = server.enqueue_payload()["batch"]["graph"]["nodes"]

        pos = nodes[invokeai.NODE_POS]
        neg = nodes[invokeai.NODE_NEG]
        self.assertEqual(pos["type"], "sdxl_compel_prompt")
        self.assertEqual(pos["prompt"], "a flame")
        self.assertEqual(pos["style"], "a flame")
        self.assertEqual(neg["prompt"], "photo, text")
        self.assertEqual(neg["style"], "photo, text")

    def test_edges_cover_every_connection_only_field(self):
        server = self.install(FakeServer())
        self.backend().generate(self.request())
        edges = server.enqueue_payload()["batch"]["graph"]["edges"]

        self.assertEqual(len(edges), 10)
        wired = set()
        for edge in edges:
            wired.add((edge["destination"]["node_id"],
                       edge["destination"]["field"]))
        for pair in invokeai.REQUIRED_CONNECTIONS:
            self.assertIn(pair, wired)
        self.assertIn(
            {"source": {"node_id": invokeai.NODE_MODEL, "field": "clip2"},
             "destination": {"node_id": invokeai.NODE_NEG, "field": "clip2"}},
            edges)

    def test_missing_edge_is_caught_before_enqueue(self):
        """Graph.validate_self would accept this and fail at run time."""
        self.install(FakeServer())
        backend = self.backend()
        graph = backend._build_graph(
            prompt="p", negative="n", seed=1, width=1024, height=1024,
            steps=30, cfg_scale=7.5, scheduler="euler", use_cache=True,
            board_id=None)
        graph["edges"] = [e for e in graph["edges"]
                          if e["destination"]["field"] != "vae"]
        with self.assertRaises(BackendError) as caught:
            invokeai._require_connections(graph)
        self.assertIn("l2i.vae", str(caught.exception))

    def test_node_key_must_match_node_id(self):
        graph = {"nodes": {"a": {"id": "b"}}, "edges": []}
        with self.assertRaises(BackendError):
            invokeai._require_connections(graph)

    def test_size_is_rounded_to_a_multiple_of_eight(self):
        server = self.install(FakeServer())
        self.backend().generate(self.request(size=(1000, 1013)))
        noise = server.enqueue_payload()["batch"]["graph"]["nodes"]["noise"]
        self.assertEqual((noise["width"], noise["height"]), (1000, 1016))

    def test_board_and_use_cache_options(self):
        server = self.install(FakeServer())
        backend = self.backend(use_cache=False, board_id="board-7")
        backend.generate(self.request())
        nodes = server.enqueue_payload()["batch"]["graph"]["nodes"]
        self.assertEqual(nodes[invokeai.NODE_L2I]["board"],
                         {"board_id": "board-7"})
        for node in nodes.values():
            self.assertIs(node["use_cache"], False)


class TestResults(InvokeAITestCase):

    def test_results_are_keyed_by_prepared_node_id(self):
        """results keys are the executor's ids; translate via the mapping."""
        item = queue_item(
            41, "completed",
            results={
                "prep-noise": {"type": "noise_output"},
                # a decoy image producer that is NOT our terminal node, listed
                # first so a type-only filter would take it
                "prep-other": image_result("wrong.png"),
                "prep-l2i": image_result("right.png"),
            },
            mapping={"prep-noise": invokeai.NODE_NOISE,
                     "prep-other": "some_other_l2i",
                     "prep-l2i": invokeai.NODE_L2I})
        server = self.install(FakeServer(item=item))
        self.backend().generate(Request(prompt="x", seed=1))
        self.assertEqual(
            server.byte_calls,
            ["http://127.0.0.1:9090/api/v1/images/i/right.png/full"])

    def test_type_filter_is_the_fallback_when_no_mapping(self):
        item = queue_item(41, "completed",
                          results={"prep-l2i": image_result("only.png")},
                          mapping={})
        server = self.install(FakeServer(item=item))
        self.backend().generate(Request(prompt="x", seed=1))
        self.assertEqual(
            server.byte_calls,
            ["http://127.0.0.1:9090/api/v1/images/i/only.png/full"])

    def test_no_image_output_raises(self):
        item = queue_item(41, "completed",
                          results={"prep-noise": {"type": "noise_output"}})
        self.install(FakeServer(item=item))
        with self.assertRaises(BackendError) as caught:
            self.backend().generate(Request(prompt="x", seed=1))
        self.assertIn("image_output", str(caught.exception))


class TestPolling(InvokeAITestCase):

    def test_unknown_status_does_not_end_the_poll(self):
        """A 6.15 status this module has never heard of means "keep waiting"."""
        server = self.install(FakeServer(
            statuses=["pending", "waiting", "quantum_superposition",
                      "in_progress", "completed"]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = self.backend().generate(Request(prompt="x", seed=1))

        self.assertEqual(out, [PNG])
        self.assertEqual(server.polls(41), 5)
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("quantum_superposition" in m for m in messages),
                        messages)
        # "waiting" is a known 6.14 status, so it must not warn
        self.assertFalse(any("'waiting'" in m for m in messages), messages)

    def test_failed_item_reports_the_servers_error(self):
        item = queue_item(41, "failed",
                          error=("InvocationError", "vae is None"))
        server = self.install(FakeServer(statuses=["failed"], item=item))
        server.item = item
        server._poll = lambda item_id: item
        with self.assertRaises(BackendError) as caught:
            self.backend().generate(Request(prompt="x", seed=1))
        self.assertIn("vae is None", str(caught.exception))
        self.assertIn("failed", str(caught.exception))

    def test_deadline_is_enforced(self):
        self.install(FakeServer(statuses=["in_progress"]))
        backend = self.backend(timeout=0)
        with self.assertRaises(BackendError) as caught:
            backend.generate(Request(prompt="x", seed=1))
        self.assertIn("in_progress", str(caught.exception))


class TestBatch(InvokeAITestCase):

    def test_truncated_enqueue_is_detected(self):
        """enqueued < requested is the only signal that sessions were dropped."""
        self.install(FakeServer(enqueued=2, requested=3))
        with self.assertRaises(BackendError) as caught:
            self.backend().generate(Request(prompt="x", seed=1, count=3))
        self.assertIn("max_queue_size", str(caught.exception))

    def test_item_id_count_mismatch_is_detected(self):
        self.install(FakeServer(item_ids=[41]))
        with self.assertRaises(BackendError) as caught:
            self.backend().generate(Request(prompt="x", seed=1, count=3))
        self.assertIn("queue item id", str(caught.exception))

    def test_count_becomes_one_batch_with_distinct_seeds(self):
        server = self.install(FakeServer())
        out = self.backend().generate(Request(prompt="x", seed=100, count=3))

        self.assertEqual(out, [PNG, PNG, PNG])
        batch = server.enqueue_payload()["batch"]
        self.assertEqual(batch["data"],
                         [[{"node_path": invokeai.NODE_NOISE,
                            "field_name": "seed",
                            "items": [100, 101, 102]}]])
        self.assertEqual(len([u for u, _ in server.json_calls
                              if "enqueue_batch" in u]), 1)
        for item_id in (41, 42, 43):
            self.assertEqual(server.polls(item_id), 1)


class TestOptions(InvokeAITestCase):

    def test_unknown_constructor_option_raises(self):
        with self.assertRaises(UnsupportedOption) as caught:
            invokeai.InvokeAIBackend(hots="127.0.0.1:9090")
        self.assertIn("hots", str(caught.exception))

    def test_unknown_request_option_raises(self):
        self.install(FakeServer())
        request = Request(prompt="x", seed=1, options={"stpes": 30})
        with self.assertRaises(UnsupportedOption) as caught:
            self.backend().generate(request)
        self.assertIn("stpes", str(caught.exception))

    def test_connection_option_is_rejected_per_request(self):
        self.install(FakeServer())
        request = Request(prompt="x", seed=1, options={"host": "elsewhere"})
        with self.assertRaises(UnsupportedOption) as caught:
            self.backend().generate(request)
        self.assertIn("constructed", str(caught.exception))

    def test_request_options_override_instance_defaults(self):
        server = self.install(FakeServer())
        backend = self.backend(steps=12)
        backend.generate(Request(prompt="x", seed=1,
                                 options={"steps": 40, "scheduler": "dpmpp_2m_k"}))
        denoise = server.enqueue_payload()["batch"]["graph"]["nodes"]["denoise"]
        self.assertEqual(denoise["steps"], 40)
        self.assertEqual(denoise["scheduler"], "dpmpp_2m_k")

    def test_unknown_scheduler_raises(self):
        with self.assertRaises(UnsupportedOption):
            invokeai.InvokeAIBackend(scheduler="euler_ancestral_karras")


class TestModelResolution(InvokeAITestCase):

    def test_hash_wins_over_name_and_position(self):
        other = dict(MODEL, key="k2", hash="blake3:other", name="otherXL")
        server = self.install(FakeServer(models=[other, dict(MODEL)]))
        backend = self.backend(model_hash=MODEL["hash"])
        backend.generate(Request(prompt="x", seed=1))
        model = server.enqueue_payload()["batch"]["graph"]["nodes"]["sdxl_model"]
        self.assertEqual(model["model"]["name"], MODEL["name"])

    def test_ambiguous_install_asks_for_a_selector(self):
        other = dict(MODEL, key="k2", hash="blake3:other", name="otherXL")
        self.install(FakeServer(models=[other, dict(MODEL)]))
        with self.assertRaises(BackendError) as caught:
            self.backend().generate(Request(prompt="x", seed=1))
        self.assertIn("model_hash", str(caught.exception))

    def test_unknown_name_lists_what_is_installed(self):
        self.install(FakeServer())
        with self.assertRaises(BackendError) as caught:
            self.backend(model="notInstalledXL").generate(
                Request(prompt="x", seed=1))
        self.assertIn(MODEL["name"], str(caught.exception))

    def test_model_is_resolved_once_and_cached(self):
        server = self.install(FakeServer())
        backend = self.backend()
        backend.generate(Request(prompt="x", seed=1))
        backend.generate(Request(prompt="y", seed=2))
        self.assertEqual(len([u for u, _ in server.json_calls
                              if "/api/v2/models/" in u]), 1)


class TestProbe(InvokeAITestCase):

    def test_probe_reports_version_and_model_without_generating(self):
        server = self.install(FakeServer())
        ok, message = invokeai.InvokeAIBackend.probe()
        self.assertTrue(ok)
        self.assertIn("6.13.8", message)
        self.assertIn(MODEL["name"], message)
        self.assertFalse(any("enqueue" in url for url in server.urls()))
        self.assertEqual(server.byte_calls, [])

    def test_probe_warns_on_an_untested_version(self):
        self.install(FakeServer(version="7.1.0"))
        ok, message = invokeai.InvokeAIBackend.probe()
        self.assertTrue(ok)
        self.assertIn("UNTESTED", message)
        self.assertIn("6.13-6.14", message)

    def test_probe_accepts_a_release_candidate(self):
        self.install(FakeServer(version="6.14.0rc2"))
        ok, message = invokeai.InvokeAIBackend.probe()
        self.assertTrue(ok)
        self.assertNotIn("UNTESTED", message)

    def test_probe_reports_an_unreachable_server(self):
        server = FakeServer()

        def unreachable(*_args, **_kwargs):
            raise BackendError("cannot reach http://127.0.0.1:9090: refused")

        self.install(server)
        patcher = mock.patch.object(_http, "request_json", unreachable)
        patcher.start()
        self.addCleanup(patcher.stop)
        ok, message = invokeai.InvokeAIBackend.probe()
        self.assertFalse(ok)
        self.assertIn("unreachable", message)

    def test_probe_rejects_bad_options_without_a_request(self):
        server = self.install(FakeServer())
        ok, message = invokeai.InvokeAIBackend.probe(hsot="x")
        self.assertFalse(ok)
        self.assertIn("hsot", message)
        self.assertEqual(server.json_calls, [])

    def test_generate_warns_once_on_an_untested_version(self):
        self.install(FakeServer(version="6.99.0"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.backend().generate(Request(prompt="x", seed=1))
        self.assertTrue(any("6.99.0" in str(w.message) for w in caught))


class TestAuth(InvokeAITestCase):

    def test_bearer_token_is_sent_when_configured(self):
        recorded = []

        def request_json(url, payload=None, headers=None, **kwargs):
            recorded.append(headers)
            return FakeServer.request_json(server, url, payload=payload,
                                           headers=headers, **kwargs)

        server = FakeServer()
        self.install(server)
        patcher = mock.patch.object(_http, "request_json", request_json)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.backend(token="jwt-123").generate(Request(prompt="x", seed=1))
        self.assertTrue(all(h == {"Authorization": "Bearer jwt-123"}
                            for h in recorded), recorded)

    def test_no_token_means_no_headers(self):
        self.assertIsNone(self.backend()._headers())

    def test_host_may_carry_a_scheme(self):
        backend = self.backend(host="https://invoke.example/")
        self.assertEqual(backend._url("/api/v1/app/version"),
                         "https://invoke.example/api/v1/app/version")


if __name__ == "__main__":
    unittest.main()
