"""ComfyUI backend tests.

Nothing here touches a socket. The module imported request_json / request_bytes /
post_multipart / create_connection by name, so those four names are swapped on
the module itself and a FakeServer answers by URL with payloads shaped like the
ones in the research report.

The assertions are deliberately on the REQUEST -- which node id got which value,
which query params the /view URL carries -- because that is the whole surface
this backend has. ComfyUI takes no named parameters at all; if the injection
lands in the wrong place, the server renders the template's prompt happily and
the mistake shows up as 88 identical icons rather than as an error.
"""

import json
import os
import shutil
import tempfile
import unittest

from devgraphics.backends import comfyui
from devgraphics.backends.base import (BackendError, Capabilities, Request,
                                       UnsupportedOption)

PNG = b"\x89PNG\r\n\x1a\n" + b"not really an image, but the magic is what counts"

PROMPT_ID = "11111111-1111-4111-8111-111111111111"

#: One SaveImage output, as /history reports it when filename_prefix contains a
#: slash: the subfolder comes back separately and must go to /view verbatim.
IMAGE = {"filename": "dg_00001_.png", "subfolder": "devgraphics", "type": "output"}


class FakeSocket:
    def __init__(self, frames):
        self.frames = list(frames)
        self.closed = False

    def recv(self):
        if not self.frames:
            raise AssertionError("recv() called past the done sentinel")
        return self.frames.pop(0)

    def close(self):
        self.closed = True


class FakeServer:
    """A ComfyUI that answers by URL. Every payload copied from the report."""

    def __init__(self):
        self.calls = []                      # (kind, url, payload), in order
        self.checkpoints = ["sd_xl_base_1.0.safetensors"]
        self.bg_models = []
        self.upload = {"name": "devgraphics_ab12cd34ef56.png",
                       "subfolder": "", "type": "input"}
        self.images = [IMAGE]
        self.posted = None
        self.socket = None
        self.frames = [
            # Broadcast to every socket and carrying no prompt_id at all.
            json.dumps({"type": "status",
                        "data": {"status": {"exec_info": {"queue_remaining": 1}},
                                 "sid": "someone-else"}}),
            # Somebody else's job, which must not end our wait.
            json.dumps({"type": "executing",
                        "data": {"node": None, "prompt_id": "other-job"}}),
            json.dumps({"type": "execution_start",
                        "data": {"prompt_id": PROMPT_ID, "timestamp": 1}}),
            json.dumps({"type": "executing",
                        "data": {"node": "3", "display_node": "3",
                                 "prompt_id": PROMPT_ID}}),
            b"\x00\x00\x00\x01\x00\x00\x00\x02binary preview frame",
            json.dumps({"type": "execution_success",
                        "data": {"prompt_id": PROMPT_ID, "timestamp": 2}}),
            json.dumps({"type": "executing",
                        "data": {"node": None, "prompt_id": PROMPT_ID}}),
        ]

    # --- transport doubles ---------------------------------------------

    def request_json(self, url, payload=None, **_kwargs):
        self.calls.append(("json", url, payload))
        if "/object_info/CheckpointLoaderSimple" in url:
            # V1 shape: the options list is element 0.
            return {"CheckpointLoaderSimple": {
                "input": {"required": {"ckpt_name": [self.checkpoints,
                                                     {"tooltip": "the model"}]}},
                "output": ["MODEL", "CLIP", "VAE"]}}
        if "/object_info/LoadBackgroundRemovalModel" in url:
            if not self.bg_models:
                return {}                    # unknown class: 200 with {}
            # V3 shape: options live in element 1.
            return {"LoadBackgroundRemovalModel": {
                "input": {"required": {"bg_removal_name":
                                       ["COMBO", {"options": self.bg_models}]}}}}
        if "/object_info/" in url:
            return {}
        if url.endswith("/system_stats"):
            return {"system": {"comfyui_version": "0.33.1",
                               "python_version": "3.12.7 (main)",
                               "pytorch_version": "2.6.0"},
                    "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 4090"}]}
        if url.endswith("/prompt"):
            self.posted = payload
            return {"prompt_id": PROMPT_ID, "number": 3.0, "node_errors": {}}
        if "/history/" in url:
            return {PROMPT_ID: {
                "prompt": [3.0, PROMPT_ID, {}, {}, ["9"]],
                "outputs": {"9": {"images": list(self.images)}},
                "status": {"status_str": "success", "completed": True,
                           "messages": [["execution_start", {}],
                                        ["execution_success", {}]]}}}
        raise AssertionError("unexpected request_json url: %s" % url)

    def request_bytes(self, url, payload=None, **_kwargs):
        self.calls.append(("bytes", url, payload))
        return PNG

    def post_multipart(self, url, fields, files, **_kwargs):
        self.calls.append(("multipart", url, (fields, files)))
        return dict(self.upload)

    def create_connection(self, url, **_kwargs):
        self.calls.append(("ws", url, None))
        self.socket = FakeSocket(self.frames)
        return self.socket

    # --- helpers -------------------------------------------------------

    def urls(self, kind):
        return [url for got, url, _payload in self.calls if got == kind]


class ComfyBase(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()
        for name in ("request_json", "request_bytes", "post_multipart",
                     "create_connection"):
            original = getattr(comfyui, name)
            self.addCleanup(setattr, comfyui, name, original)
            setattr(comfyui, name, getattr(self.server, name))
        self.tmp = tempfile.mkdtemp(prefix="dg-comfy-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write(self, name, doc):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        return path


class HappyPath(ComfyBase):
    def setUp(self):
        ComfyBase.setUp(self)
        self.backend = comfyui.ComfyUIBackend()
        self.out = self.backend.generate(Request(
            prompt="flat vector sticker icon of a flame",
            negative="photo, realistic, watermark",
            seed=77777, size=(1024, 1024), count=1))

    def test_returns_png_bytes(self):
        self.assertEqual(self.out, [PNG])

    def test_prompt_and_negative_land_on_their_own_nodes(self):
        graph = self.server.posted["prompt"]
        self.assertEqual(graph["6"]["inputs"]["text"],
                         "flat vector sticker icon of a flame")
        self.assertEqual(graph["7"]["inputs"]["text"],
                         "photo, realistic, watermark")

    def test_seed_size_and_batch(self):
        graph = self.server.posted["prompt"]
        self.assertEqual(graph["3"]["inputs"]["seed"], 77777)
        self.assertEqual(graph["5"]["inputs"]["width"], 1024)
        self.assertEqual(graph["5"]["inputs"]["height"], 1024)
        self.assertEqual(graph["5"]["inputs"]["batch_size"], 1)

    def test_untouched_inputs_survive(self):
        # Links are ["<src node id>", slot] pairs; mangling one is the classic
        # way to turn a working template into a 400 nobody can read.
        graph = self.server.posted["prompt"]
        self.assertEqual(graph["3"]["inputs"]["latent_image"], ["5", 0])
        self.assertEqual(graph["3"]["inputs"]["model"], ["4", 0])
        self.assertEqual(graph["8"]["inputs"]["vae"], ["4", 2])
        self.assertEqual(graph["3"]["inputs"]["steps"], 30)

    def test_checkpoint_is_echoed_from_object_info(self):
        graph = self.server.posted["prompt"]
        self.assertEqual(graph["4"]["inputs"]["ckpt_name"],
                         "sd_xl_base_1.0.safetensors")

    def test_client_id_matches_the_socket_query(self):
        # clientId is camelCase in the query string, client_id snake_case in the
        # body. A mismatch is silent: the socket never delivers a frame.
        client_id = self.server.posted["client_id"]
        self.assertIn("?clientId=%s" % client_id, self.server.urls("ws")[0])

    def test_socket_opens_before_the_prompt_is_queued(self):
        kinds = [kind for kind, url, _p in self.server.calls
                 if kind == "ws" or (kind == "json" and url.endswith("/prompt"))]
        self.assertEqual(kinds[:2], ["ws", "json"])
        self.assertTrue(self.server.socket.closed)

    def test_view_url_carries_the_history_values_verbatim(self):
        view = [u for u in self.server.urls("bytes") if "/view?" in u]
        self.assertEqual(
            view,
            ["http://127.0.0.1:8188/view?filename=dg_00001_.png"
             "&subfolder=devgraphics&type=output"])
        # channel=rgb would strip alpha and preview=webp would re-encode lossily.
        self.assertNotIn("channel", view[0])
        self.assertNotIn("preview", view[0])

    def test_only_the_per_class_object_info_route_is_used(self):
        for url in self.server.urls("json"):
            if "/object_info" in url:
                self.assertRegex(url, r"/object_info/\w+$")


class Options(ComfyBase):
    def test_unknown_option_is_refused(self):
        backend = comfyui.ComfyUIBackend()
        with self.assertRaises(UnsupportedOption) as caught:
            backend.generate(Request(prompt="x", options={"ckpt": "foo"}))
        self.assertIn("ckpt", str(caught.exception))
        self.assertIn("checkpoint", str(caught.exception))
        self.assertIsNone(self.server.posted)

    def test_sampler_knobs_are_cast_and_injected(self):
        backend = comfyui.ComfyUIBackend()
        backend.generate(Request(prompt="x", options={"steps": "12",
                                                      "cfg": "5.5",
                                                      "sampler": "euler"}))
        sampler = self.server.posted["prompt"]["3"]["inputs"]
        self.assertEqual(sampler["steps"], 12)
        self.assertEqual(sampler["cfg"], 5.5)
        self.assertEqual(sampler["sampler_name"], "euler")

    def test_remapped_node_id_is_used(self):
        backend = comfyui.ComfyUIBackend()
        backend.generate(Request(prompt="x", options={"node_prompt": "7",
                                                      "node_negative": "6"}))
        graph = self.server.posted["prompt"]
        self.assertEqual(graph["7"]["inputs"]["text"], "x")

    def test_node_id_that_does_not_exist_fails_loudly(self):
        backend = comfyui.ComfyUIBackend(node_prompt="99")
        with self.assertRaises(BackendError) as caught:
            backend.generate(Request(prompt="x"))
        self.assertIn("no node '99'", str(caught.exception))

    def test_size_is_rounded_to_the_step_comfyui_does_not_validate(self):
        backend = comfyui.ComfyUIBackend()
        backend.generate(Request(prompt="x", size=(1020, 769)))
        latent = self.server.posted["prompt"]["5"]["inputs"]
        self.assertEqual((latent["width"], latent["height"]), (1016, 768))

    def test_count_becomes_batch_size(self):
        backend = comfyui.ComfyUIBackend()
        self.server.images = [dict(IMAGE), dict(IMAGE, filename="dg_00002_.png")]
        out = backend.generate(Request(prompt="x", count=2))
        latent = self.server.posted["prompt"]["5"]["inputs"]
        self.assertEqual(latent["batch_size"], 2)
        self.assertEqual(len(out), 2)


class Checkpoints(ComfyBase):
    def test_windows_backslash_name_is_echoed_not_rebuilt(self):
        # folder_paths uses os.path.relpath, so a nested checkpoint is reported
        # with a backslash on Windows and the forward-slash spelling 400s.
        self.server.checkpoints = ["SDXL\\juggernautXL_v8.safetensors",
                                   "sd15.ckpt"]
        backend = comfyui.ComfyUIBackend(
            checkpoint="sdxl/juggernautXL_v8.safetensors")
        backend.generate(Request(prompt="x"))
        self.assertEqual(self.server.posted["prompt"]["4"]["inputs"]["ckpt_name"],
                         "SDXL\\juggernautXL_v8.safetensors")

    def test_unknown_checkpoint_lists_what_is_installed(self):
        self.server.checkpoints = ["a.safetensors", "b.safetensors"]
        backend = comfyui.ComfyUIBackend(checkpoint="nope.safetensors")
        with self.assertRaises(BackendError) as caught:
            backend.generate(Request(prompt="x"))
        self.assertIn("a.safetensors", str(caught.exception))
        self.assertIn("b.safetensors", str(caught.exception))

    def test_the_list_is_fetched_once_per_instance(self):
        backend = comfyui.ComfyUIBackend()
        backend.generate(Request(prompt="a"))
        backend.generate(Request(prompt="b"))
        self.assertEqual([u for u in self.server.urls("json")
                          if "/object_info/" in u],
                         ["http://127.0.0.1:8188/object_info/"
                          "CheckpointLoaderSimple"])

    def test_sole_installed_checkpoint_replaces_the_template_placeholder(self):
        self.server.checkpoints = ["dreamshaperXL.safetensors"]
        comfyui.ComfyUIBackend().generate(Request(prompt="x"))
        self.assertEqual(self.server.posted["prompt"]["4"]["inputs"]["ckpt_name"],
                         "dreamshaperXL.safetensors")


class Caching(ComfyBase):
    """A byte-identical graph renders nothing and writes no new file."""

    def test_cached_run_completes_on_the_sentinel(self):
        self.server.frames = [
            json.dumps({"type": "execution_start",
                        "data": {"prompt_id": PROMPT_ID, "timestamp": 1}}),
            json.dumps({"type": "execution_cached",
                        "data": {"nodes": ["3", "4", "5", "6", "7", "8", "9"],
                                 "prompt_id": PROMPT_ID, "timestamp": 1}}),
            # No per-node "executing" frames at all, and /history for this new
            # prompt_id points at the file the FIRST run wrote.
            json.dumps({"type": "executing",
                        "data": {"node": None, "prompt_id": PROMPT_ID}}),
        ]
        out = comfyui.ComfyUIBackend().generate(Request(prompt="x", seed=77777))
        self.assertEqual(out, [PNG])
        self.assertIn("filename=dg_00001_.png",
                      [u for u in self.server.urls("bytes") if "/view?" in u][0])

    def test_execution_error_becomes_a_backend_error(self):
        self.server.frames = [
            json.dumps({"type": "execution_error",
                        "data": {"prompt_id": PROMPT_ID, "node_id": "3",
                                 "node_type": "KSampler",
                                 "exception_type": "torch.OutOfMemoryError",
                                 "exception_message": "CUDA out of memory"}}),
        ]
        with self.assertRaises(BackendError) as caught:
            comfyui.ComfyUIBackend().generate(Request(prompt="x"))
        self.assertIn("KSampler", str(caught.exception))
        self.assertIn("CUDA out of memory", str(caught.exception))


class WorkflowFiles(ComfyBase):
    def test_ui_format_is_refused_with_both_menu_wordings(self):
        path = self.write("ui.json", {
            "id": "8f0c", "last_node_id": 9, "last_link_id": 9,
            "nodes": [{"id": 3, "type": "KSampler", "pos": [10, 20],
                       "widgets_values": [77777, "randomize", 30, 7.0]}],
            "links": [[1, 4, 0, 3, 0, "MODEL"]],
            "groups": [], "config": {}, "extra": {}, "version": 0.4})
        backend = comfyui.ComfyUIBackend(workflow=path)
        with self.assertRaises(BackendError) as caught:
            backend.generate(Request(prompt="x"))
        message = str(caught.exception)
        self.assertIn("Export Workflow (API)", message)     # current wording
        self.assertIn("Save (API Format)", message)         # older wording
        self.assertIn("Dev mode Options", message)
        self.assertIn(path, message)
        self.assertIsNone(self.server.posted)

    def test_a_flat_dict_without_class_type_is_refused(self):
        path = self.write("odd.json", {"3": {"inputs": {"seed": 1}}})
        with self.assertRaises(BackendError) as caught:
            comfyui.ComfyUIBackend(workflow=path).generate(Request(prompt="x"))
        self.assertIn("class_type", str(caught.exception))

    def test_custom_workflow_is_driven_by_its_own_node_ids(self):
        graph = comfyui._packaged(comfyui.TEMPLATE)
        graph["77"] = graph.pop("6")
        path = self.write("custom.api.json", graph)
        backend = comfyui.ComfyUIBackend(workflow=path, node_prompt="77")
        backend.generate(Request(prompt="a rocket"))
        self.assertEqual(self.server.posted["prompt"]["77"]["inputs"]["text"],
                         "a rocket")


class References(ComfyBase):
    def workflow_with_loadimage(self):
        graph = comfyui._packaged(comfyui.TEMPLATE)
        graph["10"] = {"class_type": "LoadImage",
                       "inputs": {"image": "example.png"}}
        return self.write("refs.api.json", graph)

    def test_upload_result_name_is_injected(self):
        self.server.upload = {"name": "devgraphics_ab12 (1).png",
                              "subfolder": "", "type": "input"}
        backend = comfyui.ComfyUIBackend(workflow=self.workflow_with_loadimage(),
                                         node_image="10")
        self.assertEqual(backend.capabilities.reference_images, 1)
        backend.generate(Request(prompt="x", refs=(PNG,)))
        # The response key is "name", not "filename", and the server may have
        # renamed the file out from under us.
        self.assertEqual(self.server.posted["prompt"]["10"]["inputs"]["image"],
                         "devgraphics_ab12 (1).png")
        _kind, url, (fields, files) = [c for c in self.server.calls
                                       if c[0] == "multipart"][0]
        self.assertEqual(url, "http://127.0.0.1:8188/upload/image")
        self.assertEqual(fields, {"type": "input", "subfolder": "",
                                  "overwrite": "true"})
        self.assertEqual(list(files), ["image"])

    def test_more_refs_than_load_image_nodes_is_an_error(self):
        backend = comfyui.ComfyUIBackend()
        self.assertEqual(backend.capabilities.reference_images, 0)
        with self.assertRaises(BackendError) as caught:
            backend.generate(Request(prompt="x", refs=(PNG,)))
        self.assertIn("node_image", str(caught.exception))


class NativeAlpha(ComfyBase):
    def test_absent_nodes_mean_transparent_false(self):
        self.assertFalse(comfyui.ComfyUIBackend().capabilities.transparent)

    def test_installed_model_means_transparent_true_and_is_cached(self):
        self.server.bg_models = ["birefnet.safetensors"]
        backend = comfyui.ComfyUIBackend()
        self.assertTrue(backend.capabilities.transparent)
        before = len(self.server.calls)
        self.assertTrue(backend.capabilities.transparent)
        self.assertEqual(len(self.server.calls), before)

    def test_rgba_template_is_wired_and_names_the_model_verbatim(self):
        # birefnet is preferred over whatever else is in the folder even when it
        # is not first, because it is the architecture the loader sniffs for.
        self.server.bg_models = ["lucida.safetensors", "BiRefNet-v2.safetensors"]
        comfyui.ComfyUIBackend().generate(Request(prompt="x", transparent=True))
        graph = self.server.posted["prompt"]
        self.assertEqual(graph["10"]["inputs"]["bg_removal_name"],
                         "BiRefNet-v2.safetensors")
        self.assertEqual(graph["11"]["class_type"], "RemoveBackground")
        # InvertMask is mandatory: RemoveBackground returns a FOREGROUND mask and
        # JoinImageWithAlpha computes alpha = 1.0 - mask.
        self.assertEqual(graph["12"]["class_type"], "InvertMask")
        self.assertEqual(graph["12"]["inputs"]["mask"], ["11", 0])
        self.assertEqual(graph["13"]["inputs"]["alpha"], ["12", 0])
        self.assertEqual(graph["13"]["inputs"]["image"], ["8", 0])
        self.assertEqual(graph["9"]["inputs"]["images"], ["13", 0])

    def test_transparent_without_the_model_explains_which_half_is_missing(self):
        with self.assertRaises(BackendError) as caught:
            comfyui.ComfyUIBackend().generate(Request(prompt="x",
                                                      transparent=True))
        message = str(caught.exception)
        self.assertIn("birefnet.safetensors", message)
        self.assertIn("background_removal", message)

    def test_a_custom_workflow_answers_for_itself(self):
        path = self.write("rgba.api.json", comfyui._packaged(comfyui.TEMPLATE_RGBA))
        backend = comfyui.ComfyUIBackend(workflow=path)
        self.assertTrue(backend.capabilities.transparent)
        self.assertEqual(self.server.calls, [])      # answered off the file


class Offline(unittest.TestCase):
    """The constructor and capabilities must survive the server being off."""

    def explode(self, *args, **kwargs):
        raise AssertionError("the network was touched")

    def refuse(self, *args, **kwargs):
        raise BackendError("cannot reach http://127.0.0.1:8188: "
                           "[Errno 111] Connection refused")

    def swap(self, request_json):
        for name, value in (("request_json", request_json),
                            ("request_bytes", self.explode),
                            ("post_multipart", self.explode),
                            ("create_connection", self.explode)):
            original = getattr(comfyui, name)
            self.addCleanup(setattr, comfyui, name, original)
            setattr(comfyui, name, value)

    def test_constructor_makes_no_request(self):
        self.swap(self.explode)
        comfyui.ComfyUIBackend(host="10.0.0.9:8188", checkpoint="x.safetensors")

    def test_capabilities_answers_with_the_server_switched_off(self):
        self.swap(self.refuse)
        caps = comfyui.ComfyUIBackend().capabilities
        self.assertIsInstance(caps, Capabilities)
        self.assertEqual(caps.name, "comfyui")
        self.assertTrue(caps.seed)
        self.assertTrue(caps.negative_prompt)
        self.assertFalse(caps.transparent)       # conservative until answered
        self.assertEqual(caps.sizes, ())         # any width/height
        self.assertIsNone(caps.cost_per_image)

    def test_a_failed_probe_is_not_cached_as_false(self):
        self.swap(self.refuse)
        backend = comfyui.ComfyUIBackend()
        self.assertFalse(backend.capabilities.transparent)
        self.assertEqual(backend._combos, {})

    def test_probe_reports_unreachable_without_rendering(self):
        self.swap(self.refuse)
        ok, message = comfyui.ComfyUIBackend.probe(host="10.0.0.9:8188")
        self.assertFalse(ok)
        self.assertIn("cannot reach ComfyUI at 10.0.0.9:8188", message)


class Probe(ComfyBase):
    def test_probe_reports_version_and_a_real_checkpoint_name(self):
        self.server.checkpoints = ["SDXL\\juggernautXL_v8.safetensors"]
        ok, message = comfyui.ComfyUIBackend.probe()
        self.assertTrue(ok)
        self.assertIn("ComfyUI 0.33.1", message)
        self.assertIn("juggernautXL_v8.safetensors", message)
        self.assertIsNone(self.server.posted)        # nothing was rendered
        self.assertEqual(self.server.urls("ws"), [])

    def test_probe_fails_when_there_is_nothing_to_render_with(self):
        self.server.checkpoints = []
        ok, message = comfyui.ComfyUIBackend.probe()
        self.assertFalse(ok)
        self.assertIn("models/checkpoints is empty", message)


class Plumbing(unittest.TestCase):
    def test_combo_shapes(self):
        self.assertEqual(comfyui._combo_options([["a", "b"], {"tooltip": "x"}]),
                         ["a", "b"])
        self.assertEqual(comfyui._combo_options(["COMBO", {"options": ["c"]}]),
                         ["c"])
        self.assertEqual(comfyui._combo_options(["INT", {"default": 1}]), [])
        self.assertEqual(comfyui._combo_options(None), [])

    def test_urls(self):
        self.assertEqual(comfyui._url("127.0.0.1:8188", "", "/prompt"),
                         "http://127.0.0.1:8188/prompt")
        self.assertEqual(comfyui._url("127.0.0.1:8188", "api", "/prompt"),
                         "http://127.0.0.1:8188/api/prompt")
        self.assertEqual(comfyui._ws_url("https://box:8188", "/api/", "abc"),
                         "wss://box:8188/api/ws?clientId=abc")

    def test_prompt_error_adds_the_detail_comfyui_withholds(self):
        raised = comfyui._prompt_error(
            BackendError("HTTP 400: Value not in list [value_not_in_list]"))
        self.assertIn("BACKSLASH", str(raised))


if __name__ == "__main__":
    unittest.main()
