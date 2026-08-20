"""Unit tests for the AUTOMATIC1111 backend.

Nothing here touches the network: `devgraphics.backends.a1111.request_json` is
replaced with a recorder, so every test asserts either on the request the backend
built or on its parsing of a response payload shaped like the one in the research
report -- {"images": [<base64>], "parameters": {...}, "info": "<json string>"}.
"""

import base64
import io
import json
import unittest
from unittest import mock

from PIL import Image

from devgraphics.backends import a1111
from devgraphics.backends.base import BackendError, Capabilities, Request, \
    UnsupportedOption


def _png(colour=(200, 60, 40), size=(8, 8)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, "PNG")
    return buf.getvalue()


def _jpeg(colour=(30, 30, 30), size=(8, 8)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, "JPEG")
    return buf.getvalue()


def _b64(data):
    return base64.b64encode(data).decode("ascii")


#: The seed a1111 reports back when -1 was sent. Deliberately not the seed any
#: test asks for, so a test that "passes" by echoing the request fails instead.
REAL_SEED = 2246598734

INFO = {
    "prompt": "flat vector sticker icon of a flame",
    "negative_prompt": "photo, realistic, 3d render",
    "seed": REAL_SEED,
    "all_seeds": [REAL_SEED],
    "subseed": 1234567,
    "subseed_strength": 0,
    "width": 1024,
    "height": 1024,
    "sampler_name": "DPM++ 2M",
    "cfg_scale": 7.0,
    "steps": 20,
    "batch_size": 1,
}


def response(images=None, info=None):
    """A realistic /sdapi/v1/txt2img body. `info` is a JSON *string*, as on the wire."""
    if images is None:
        images = [_b64(_png())]
    return {"images": images,
            "parameters": {"prompt": "flat vector sticker icon of a flame",
                           "seed": -1, "width": 1024, "height": 1024},
            "info": json.dumps(INFO if info is None else info)}


class Recorder:
    """Stands in for _http.request_json and remembers how it was called."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def __call__(self, url, payload=None, headers=None, method=None,
                 timeout=300, retries=4, sleep=None):
        self.calls.append({"url": url, "payload": payload, "headers": headers,
                           "method": method, "timeout": timeout,
                           "retries": retries})
        if self.error is not None:
            raise self.error
        return self.result

    @property
    def payload(self):
        return self.calls[-1]["payload"]


def patched(recorder):
    return mock.patch.object(a1111, "request_json", recorder)


def no_network():
    """Any HTTP call at all is a test failure."""
    def boom(*args, **kwargs):
        raise AssertionError("this code path must not touch the network")
    return mock.patch.object(a1111, "request_json", boom)


class RequestBodyTests(unittest.TestCase):

    def test_happy_path_body(self):
        rec = Recorder(response())
        backend = a1111.A1111Backend()
        with patched(rec):
            backend.generate(Request(prompt="a flame", negative="photo, blurry",
                                     seed=77777, size=(1024, 768)))

        self.assertEqual(rec.calls[0]["url"],
                         "http://127.0.0.1:7860/sdapi/v1/txt2img")
        self.assertEqual(rec.payload, {
            "prompt": "a flame",
            "negative_prompt": "photo, blurry",
            "seed": 77777,
            "width": 1024,
            "height": 768,
            "batch_size": 1,
            "n_iter": 1,
        })

    def test_unset_seed_is_minus_one(self):
        rec = Recorder(response())
        with patched(rec):
            a1111.A1111Backend().generate(Request(prompt="x"))
        self.assertEqual(rec.payload["seed"], -1)

    def test_optional_knobs_are_omitted_unless_asked_for(self):
        rec = Recorder(response())
        with patched(rec):
            a1111.A1111Backend().generate(Request(prompt="x"))
        for key in ("sampler_name", "steps", "cfg_scale", "override_settings"):
            self.assertNotIn(key, rec.payload)

    def test_options_are_typed_not_passed_through_as_strings(self):
        rec = Recorder(response())
        backend = a1111.A1111Backend(steps="30", cfg_scale="4.5",
                                     sampler_name="DPM++ 2M")
        with patched(rec):
            backend.generate(Request(prompt="x"))
        self.assertEqual(rec.payload["steps"], 30)
        self.assertEqual(rec.payload["cfg_scale"], 4.5)
        self.assertEqual(rec.payload["sampler_name"], "DPM++ 2M")

    def test_request_options_override_constructor_options(self):
        rec = Recorder(response())
        backend = a1111.A1111Backend(steps=20)
        with patched(rec):
            backend.generate(Request(prompt="x", options={"steps": 40}))
        self.assertEqual(rec.payload["steps"], 40)

    def test_checkpoint_is_pinned_per_request(self):
        rec = Recorder(response())
        backend = a1111.A1111Backend(checkpoint="juggernautXL_v8Rundiffusion")
        with patched(rec):
            backend.generate(Request(prompt="x"))
        self.assertEqual(rec.payload["override_settings"],
                         {"sd_model_checkpoint": "juggernautXL_v8Rundiffusion"})
        # and never by mutating global state
        self.assertNotIn("/sdapi/v1/options", rec.calls[0]["url"])

    def test_explicit_override_settings_merge_and_win(self):
        rec = Recorder(response())
        backend = a1111.A1111Backend(
            checkpoint="ignored",
            override_settings={"sd_model_checkpoint": "pinned",
                               "CLIP_stop_at_last_layers": 2})
        with patched(rec):
            backend.generate(Request(prompt="x"))
        self.assertEqual(rec.payload["override_settings"],
                         {"sd_model_checkpoint": "pinned",
                          "CLIP_stop_at_last_layers": 2})

    def test_count_drives_batch_size(self):
        rec = Recorder(response(images=[_b64(_png()) for _ in range(4)]))
        with patched(rec):
            a1111.A1111Backend().generate(Request(prompt="x", count=4))
        self.assertEqual(rec.payload["batch_size"], 4)
        self.assertEqual(rec.payload["n_iter"], 1)

    def test_pinned_batch_size_still_delivers_count(self):
        rec = Recorder(response(images=[_b64(_png()) for _ in range(4)]))
        backend = a1111.A1111Backend(batch_size=2)
        with patched(rec):
            out = backend.generate(Request(prompt="x", count=3))
        self.assertEqual(rec.payload["batch_size"], 2)
        self.assertEqual(rec.payload["n_iter"], 2)     # 2*2 >= 3
        self.assertEqual(len(out), 3)                  # the fourth is dropped


class ResponseTests(unittest.TestCase):

    def test_images_come_back_as_png_bytes(self):
        raw = _png()
        rec = Recorder(response(images=[_b64(raw)]))
        with patched(rec):
            out = a1111.A1111Backend().generate(Request(prompt="x"))
        self.assertEqual(out, [raw])
        self.assertEqual(out[0][:8], bytes((0x89, 0x50, 0x4E, 0x47,
                                            0x0D, 0x0A, 0x1A, 0x0A)))

    def test_non_png_bytes_are_transcoded(self):
        rec = Recorder(response(images=[_b64(_jpeg())]))
        with patched(rec):
            out = a1111.A1111Backend().generate(Request(prompt="x"))
        self.assertEqual(out[0][:8], bytes((0x89, 0x50, 0x4E, 0x47,
                                            0x0D, 0x0A, 0x1A, 0x0A)))

    def test_data_uri_prefix_is_tolerated(self):
        raw = _png()
        rec = Recorder(response(images=["data:image/png;base64," + _b64(raw)]))
        with patched(rec):
            out = a1111.A1111Backend().generate(Request(prompt="x"))
        self.assertEqual(out, [raw])

    def test_real_seed_is_recovered_from_the_info_string(self):
        rec = Recorder(response())
        backend = a1111.A1111Backend()
        with patched(rec):
            backend.generate(Request(prompt="x"))          # seed=-1 on the wire
        self.assertEqual(rec.payload["seed"], -1)
        self.assertEqual(backend.last_seed, REAL_SEED)
        self.assertEqual(backend.last_info["all_seeds"], [REAL_SEED])

    def test_info_really_is_double_encoded(self):
        """A dict where the wire carries a string means the second json.loads is
        missing; last_seed must not silently come from anywhere else."""
        doc = response()
        doc["info"] = INFO                                 # not a JSON string
        rec = Recorder(doc)
        backend = a1111.A1111Backend()
        with patched(rec):
            backend.generate(Request(prompt="x"))
        self.assertIsNone(backend.last_seed)
        self.assertEqual(backend.last_info, {})

    def test_unparsable_info_does_not_break_generation(self):
        doc = response()
        doc["info"] = "<html>proxy ate it</html>"
        rec = Recorder(doc)
        backend = a1111.A1111Backend()
        with patched(rec):
            out = backend.generate(Request(prompt="x"))
        self.assertEqual(len(out), 1)
        self.assertIsNone(backend.last_seed)

    def test_empty_image_list_raises(self):
        rec = Recorder(response(images=[]))
        with patched(rec):
            with self.assertRaises(BackendError):
                a1111.A1111Backend().generate(Request(prompt="x"))


class OptionTests(unittest.TestCase):

    def test_unknown_constructor_option(self):
        with self.assertRaises(UnsupportedOption) as ctx:
            a1111.A1111Backend(sampler="DPM++ 2M")
        self.assertIn("sampler", str(ctx.exception))

    def test_unknown_request_option(self):
        rec = Recorder(response())
        with patched(rec):
            with self.assertRaises(UnsupportedOption):
                a1111.A1111Backend().generate(
                    Request(prompt="x", options={"cfg": 7}))
        self.assertEqual(rec.calls, [])                    # nothing was sent

    def test_non_numeric_number(self):
        rec = Recorder(response())
        backend = a1111.A1111Backend(steps="twenty")
        with patched(rec):
            with self.assertRaises(UnsupportedOption):
                backend.generate(Request(prompt="x"))

    def test_override_settings_must_be_a_table(self):
        backend = a1111.A1111Backend(override_settings="sd_model_checkpoint=x")
        rec = Recorder(response())
        with patched(rec):
            with self.assertRaises(UnsupportedOption):
                backend.generate(Request(prompt="x"))


class CapabilitiesTests(unittest.TestCase):

    def test_answerable_with_the_server_switched_off(self):
        with no_network():
            caps = a1111.A1111Backend(host="10.0.0.9:7860").capabilities
        self.assertIsInstance(caps, Capabilities)
        self.assertEqual(caps.name, "a1111")
        self.assertTrue(caps.seed)
        self.assertTrue(caps.negative_prompt)
        self.assertTrue(caps.batch)
        self.assertEqual(caps.sizes, ())                   # any (w, h)
        self.assertIsNone(caps.cost_per_image)

    def test_seed_without_determinism(self):
        with no_network():
            caps = a1111.A1111Backend().capabilities
        self.assertTrue(caps.seed)
        self.assertFalse(caps.deterministic)

    def test_no_native_alpha_or_reference_images(self):
        with no_network():
            caps = a1111.A1111Backend().capabilities
        self.assertFalse(caps.transparent)
        self.assertEqual(caps.reference_images, 0)

    def test_notes_name_the_pinned_checkpoint(self):
        with no_network():
            caps = a1111.A1111Backend(checkpoint="sd_xl_base_1.0").capabilities
        self.assertTrue(any("sd_xl_base_1.0" in n for n in caps.notes))

    def test_preflight_passes_a_seeded_negative_prompted_request(self):
        from devgraphics.backends.base import preflight
        with no_network():
            caps = a1111.A1111Backend().capabilities
        waivers = preflight(caps, Request(prompt="x", seed=1, negative="photo",
                                          size=(768, 768), count=2))
        self.assertEqual(waivers, ())


class ProbeTests(unittest.TestCase):

    def test_404_names_the_api_flag(self):
        rec = Recorder(error=BackendError("HTTP 404: Not Found"))
        with patched(rec):
            ok, message = a1111.A1111Backend.probe()
        self.assertFalse(ok)
        self.assertIn("--api", message)
        self.assertIn("/sdapi/v1/*", message)

    def test_probe_is_a_bodyless_get_that_does_not_generate(self):
        rec = Recorder({"sd_model_checkpoint": "juggernautXL_v8Rundiffusion"})
        with patched(rec):
            ok, message = a1111.A1111Backend.probe(host="box:7860")
        self.assertTrue(ok)
        self.assertEqual(len(rec.calls), 1)
        call = rec.calls[0]
        self.assertEqual(call["url"], "http://box:7860/sdapi/v1/options")
        self.assertIsNone(call["payload"])                 # GET, no body
        self.assertEqual(call["retries"], 0)
        self.assertIn("juggernautXL_v8Rundiffusion", message)

    def test_unreachable_host(self):
        rec = Recorder(error=BackendError("cannot reach http://box:7860: "
                                          "[Errno 111] Connection refused"))
        with patched(rec):
            ok, message = a1111.A1111Backend.probe(host="box:7860")
        self.assertFalse(ok)
        self.assertIn("Connection refused", message)

    def test_probe_rejects_a_typo(self):
        with no_network():
            with self.assertRaises(UnsupportedOption):
                a1111.A1111Backend.probe(host="box:7860", chekpoint="x")

    def test_generate_404_names_the_api_flag_too(self):
        rec = Recorder(error=BackendError("HTTP 404: Not Found"))
        with patched(rec):
            with self.assertRaises(BackendError) as ctx:
                a1111.A1111Backend().generate(Request(prompt="x"))
        self.assertIn("--api", str(ctx.exception))


class TransportTests(unittest.TestCase):

    def test_no_auth_header_by_default(self):
        rec = Recorder(response())
        with mock.patch.dict("os.environ", {}, clear=True):
            with patched(rec):
                a1111.A1111Backend().generate(Request(prompt="x"))
        self.assertEqual(rec.calls[0]["headers"], {})

    def test_api_auth_from_the_environment(self):
        rec = Recorder(response())
        with mock.patch.dict("os.environ", {a1111.AUTH_ENV: "bob:hunter2"}):
            with patched(rec):
                a1111.A1111Backend().generate(Request(prompt="x"))
        self.assertEqual(rec.calls[0]["headers"],
                         {"Authorization": "Basic " + _b64(b"bob:hunter2")})

    def test_full_url_host_is_left_alone(self):
        rec = Recorder(response())
        with patched(rec):
            a1111.A1111Backend(host="https://gpu.lan:7860/").generate(
                Request(prompt="x"))
        self.assertEqual(rec.calls[0]["url"],
                         "https://gpu.lan:7860/sdapi/v1/txt2img")

    def test_generation_timeout_is_not_the_http_default(self):
        rec = Recorder(response())
        with patched(rec):
            a1111.A1111Backend(timeout=60).generate(Request(prompt="x"))
        self.assertEqual(rec.calls[0]["timeout"], 60.0)


if __name__ == "__main__":
    unittest.main()
