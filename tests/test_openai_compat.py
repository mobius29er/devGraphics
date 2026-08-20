"""Unit tests for the generic OpenAI-compatible backend.

Nothing here touches the network: devgraphics._http.request_json and
request_bytes are replaced for every test, and the default replacement fails the
test if it is called at all, so "capabilities without a network call" is checked
by construction rather than by hoping.

Response payloads are copied from the shapes in the provider research: xAI's
{"data": [{"url", "b64_json", "mime_type", ...}], "usage": {...}} and Together's
{"id", "model", "object": "list", "data": [{"index", "type", ...}]}.
"""

import base64
import io
import os
import unittest
from unittest import mock

from PIL import Image

from devgraphics import _http
from devgraphics.backends import openai_compat
from devgraphics.backends.base import (AuthError, BackendError, Capabilities,
                                       Request, UnsupportedOption)

KEY_ENV = {"XAI_API_KEY": "xai-test", "TOGETHER_API_KEY": "together-test",
           "OPENAI_API_KEY": "sk-test", "DEEPINFRA_API_KEY": "di-test"}


def png_bytes(colour=(200, 40, 40)):
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), colour).save(buf, "PNG")
    return buf.getvalue()


def jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, "JPEG")
    return buf.getvalue()


def b64_response(data=None):
    """xAI's documented generate shape, base64 arm."""
    payload = base64.b64encode(data if data is not None else png_bytes())
    return {"data": [{"url": None,
                      "b64_json": payload.decode("ascii"),
                      "mime_type": "image/png",
                      "storage_error": None}],
            "usage": {"cost_in_usd_ticks": 40000, "total_tokens": None}}


def url_response(url="https://imgen.x.ai/tmp/abc.png"):
    return {"created": 1766000000,
            "data": [{"url": url, "b64_json": None, "revised_prompt": None}]}


def together_response(data=None):
    payload = base64.b64encode(data if data is not None else png_bytes())
    return {"id": "abc", "model": "black-forest-labs/FLUX.1-schnell",
            "object": "list",
            "data": [{"index": 0, "type": "b64_json",
                      "b64_json": payload.decode("ascii")}]}


class Recorder(object):
    """Stands in for _http.request_json / request_bytes."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, url, payload=None, headers=None, **kw):
        self.calls.append({"url": url, "payload": payload,
                           "headers": headers or {}, "kw": kw})
        if isinstance(self.result, Exception):
            raise self.result
        if callable(self.result):
            return self.result(url, payload, headers, **kw)
        return self.result

    @property
    def last(self):
        return self.calls[-1]


def explode(*args, **kw):
    raise AssertionError("the network was touched: %r" % (args,))


class CompatTestCase(unittest.TestCase):
    """Every test starts with both HTTP entry points booby-trapped."""

    def setUp(self):
        patcher = mock.patch.multiple(_http, request_json=explode,
                                      request_bytes=explode)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, KEY_ENV)
        env.start()
        self.addCleanup(env.stop)

    def backend(self, **options):
        options.setdefault("base_url", "https://api.example.com/v1")
        options.setdefault("model", "some-image-model")
        return openai_compat.OpenAICompatBackend(**options)

    def post(self, backend, request, result):
        """Run generate() with request_json returning `result`."""
        rec = Recorder(result)
        with mock.patch.object(_http, "request_json", rec):
            images = backend.generate(request)
        return rec, images


# --- construction and capabilities --------------------------------------

class TestCapabilities(CompatTestCase):

    def test_capabilities_need_no_network(self):
        # setUp has already made any HTTP call an AssertionError.
        caps = self.backend().capabilities
        self.assertIsInstance(caps, Capabilities)
        self.assertEqual("some-image-model", caps.name.split("(")[1].rstrip(")"))

    def test_capabilities_need_no_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            caps = self.backend().capabilities
        self.assertFalse(caps.seed)

    def test_conservative_by_default(self):
        caps = self.backend().capabilities
        self.assertFalse(caps.seed)
        self.assertFalse(caps.deterministic)
        self.assertFalse(caps.negative_prompt)
        self.assertFalse(caps.transparent)
        self.assertEqual(0, caps.reference_images)
        self.assertEqual((), caps.sizes)
        self.assertTrue(any("cannot be introspected" in n for n in caps.notes))

    def test_capability_overrides(self):
        caps = self.backend(supports_seed=True,
                            supports_negative_prompt=True).capabilities
        self.assertTrue(caps.seed)
        self.assertTrue(caps.negative_prompt)
        self.assertFalse(caps.deterministic)   # asserted != reproducible
        self.assertTrue(any("you asserted" in n for n in caps.notes))

    def test_overrides_accept_strings_from_dash_o(self):
        caps = self.backend(supports_seed="true",
                            supports_transparent="false").capabilities
        self.assertTrue(caps.seed)
        self.assertFalse(caps.transparent)

    def test_bad_flag_value_is_rejected(self):
        with self.assertRaises(ValueError):
            self.backend(supports_seed="maybe")

    def test_grok_notes_name_the_retired_ids_and_dates(self):
        notes = " ".join(openai_compat.OpenAICompatBackend(
            preset="grok", model="grok-imagine-image-2.0").capabilities.notes)
        self.assertIn("grok-2-image-1212", notes)
        self.assertIn("2026-02-28", notes)
        self.assertIn("grok-imagine-image-pro", notes)
        self.assertIn("2026-05-15", notes)

    def test_together_notes_name_the_response_format_spelling(self):
        notes = " ".join(openai_compat.OpenAICompatBackend(
            preset="together", model="black-forest-labs/FLUX.1-schnell",
        ).capabilities.notes)
        self.assertIn("base64", notes)
        self.assertIn("b64_json", notes)
        self.assertIn("api.together.ai", notes)   # the preset carries .xyz

    def test_known_grok_price_is_reported(self):
        caps = openai_compat.OpenAICompatBackend(
            preset="grok", model="grok-imagine-image-quality").capabilities
        self.assertEqual(0.05, caps.cost_per_image)

    def test_unknown_model_has_no_price_but_says_so(self):
        caps = self.backend().capabilities
        self.assertIsNone(caps.cost_per_image)
        self.assertTrue(any("cost_per_image" in n for n in caps.notes))


class TestPresets(CompatTestCase):

    def test_preset_supplies_base_url_and_env(self):
        b = openai_compat.OpenAICompatBackend(preset="grok",
                                              model="grok-imagine-image-2.0")
        self.assertEqual("https://api.x.ai/v1", b.base_url)
        self.assertEqual("XAI_API_KEY", b.api_key_env)
        self.assertEqual("aspect_ratio", b.size_param)

    def test_preset_values_come_from_base_not_a_local_copy(self):
        from devgraphics.backends import base as base_module
        for name, cfg in base_module.COMPAT_PRESETS.items():
            b = openai_compat.OpenAICompatBackend(preset=name, model="m")
            self.assertEqual(cfg["base_url"].rstrip("/"), b.base_url)
            self.assertEqual(cfg["api_key_env"], b.api_key_env)

    def test_explicit_options_beat_the_preset(self):
        b = openai_compat.OpenAICompatBackend(
            preset="together", model="m", base_url="http://127.0.0.1:8000/v1",
            api_key_env="LOCAL_KEY")
        self.assertEqual("http://127.0.0.1:8000/v1", b.base_url)
        self.assertEqual("LOCAL_KEY", b.api_key_env)

    def test_unknown_preset_is_rejected(self):
        with self.assertRaises(ValueError):
            openai_compat.OpenAICompatBackend(preset="fireworks", model="m")

    def test_model_is_required(self):
        with self.assertRaises(ValueError):
            openai_compat.OpenAICompatBackend(preset="grok")

    def test_base_url_is_required(self):
        with self.assertRaises(ValueError):
            openai_compat.OpenAICompatBackend(model="m")


class TestOptionRejection(CompatTestCase):

    def test_unknown_constructor_option(self):
        with self.assertRaises(UnsupportedOption) as caught:
            self.backend(sampler="euler")
        self.assertIn("sampler", str(caught.exception))

    def test_typo_on_a_real_option_is_not_silently_ignored(self):
        with self.assertRaises(UnsupportedOption):
            self.backend(support_seed=True)

    def test_unknown_request_option(self):
        b = self.backend()
        request = Request(prompt="a flame", options={"styles": ["sticker"]})
        with self.assertRaises(UnsupportedOption):
            b.generate(request)

    def test_request_option_can_override_the_model(self):
        b = self.backend()
        rec, _ = self.post(b, Request(prompt="a flame",
                                      options={"model": "other-model"}),
                           b64_response())
        self.assertEqual("other-model", rec.last["payload"]["model"])
        self.assertEqual("some-image-model", b.model)   # instance untouched


# --- the request --------------------------------------------------------

class TestRequestBody(CompatTestCase):

    def test_happy_path_body(self):
        b = self.backend()
        rec, _ = self.post(b, Request(prompt="a flame", size=(1024, 1024)),
                           b64_response())
        self.assertEqual({"model": "some-image-model", "prompt": "a flame",
                          "size": "1024x1024"}, rec.last["payload"])
        self.assertEqual("Bearer sk-test", rec.last["headers"]["Authorization"])

    def test_response_format_is_never_sent(self):
        b = self.backend(preset=None, supports_seed=True,
                         supports_negative_prompt=True,
                         supports_transparent=True)
        request = Request(prompt="a flame", negative="text, watermark", seed=77777,
                          transparent=True, count=3)
        rec, _ = self.post(b, request, b64_response())
        self.assertNotIn("response_format", rec.last["payload"])

    def test_response_format_only_arrives_if_the_user_forces_it(self):
        b = self.backend(extra_body={"response_format": "base64"})
        rec, _ = self.post(b, Request(prompt="a flame"), together_response())
        self.assertEqual("base64", rec.last["payload"]["response_format"])

    def test_seed_and_negative_are_withheld_unless_asserted(self):
        b = self.backend()
        rec, _ = self.post(b, Request(prompt="a flame", seed=77777,
                                      negative="text", transparent=True),
                           b64_response())
        for key in ("seed", "negative_prompt", "background"):
            self.assertNotIn(key, rec.last["payload"])

    def test_asserted_capabilities_reach_the_body(self):
        b = self.backend(supports_seed=True, supports_negative_prompt=True,
                         supports_transparent=True)
        rec, _ = self.post(b, Request(prompt="a flame", seed=77777,
                                      negative="text, watermark",
                                      transparent=True),
                           b64_response())
        payload = rec.last["payload"]
        self.assertEqual(77777, payload["seed"])
        self.assertEqual("text, watermark", payload["negative_prompt"])
        self.assertEqual("transparent", payload["background"])

    def test_n_only_when_batching(self):
        b = self.backend()
        rec, _ = self.post(b, Request(prompt="a flame"), b64_response())
        self.assertNotIn("n", rec.last["payload"])
        rec, _ = self.post(b, Request(prompt="a flame", count=4), b64_response())
        self.assertEqual(4, rec.last["payload"]["n"])

    def test_extra_body_passes_through_and_wins(self):
        b = self.backend(extra_body={"steps": 30, "size": "512x512"})
        rec, _ = self.post(b, Request(prompt="a flame", size=(1024, 1024)),
                           b64_response())
        self.assertEqual(30, rec.last["payload"]["steps"])
        self.assertEqual("512x512", rec.last["payload"]["size"])

    def test_extra_body_accepts_a_json_string_from_dash_o(self):
        b = self.backend(extra_body='{"steps": 30, "guidance_scale": 3.5}')
        rec, _ = self.post(b, Request(prompt="a flame"), b64_response())
        self.assertEqual(30, rec.last["payload"]["steps"])
        self.assertEqual(3.5, rec.last["payload"]["guidance_scale"])

    def test_extra_body_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            self.backend(extra_body="steps=30")


class TestSizeSpelling(CompatTestCase):

    def test_size_string(self):
        b = self.backend(size_param="size")
        rec, _ = self.post(b, Request(prompt="p", size=(1536, 1024)),
                           b64_response())
        self.assertEqual("1536x1024", rec.last["payload"]["size"])

    def test_width_height_integers(self):
        b = self.backend(size_param="width_height")
        rec, _ = self.post(b, Request(prompt="p", size=(768, 1024)),
                           b64_response())
        payload = rec.last["payload"]
        self.assertEqual(768, payload["width"])
        self.assertEqual(1024, payload["height"])
        self.assertNotIn("size", payload)

    def test_aspect_ratio_enum(self):
        b = self.backend(size_param="aspect_ratio")
        rec, _ = self.post(b, Request(prompt="p", size=(1024, 1024)),
                           b64_response())
        self.assertEqual("1:1", rec.last["payload"]["aspect_ratio"])
        self.assertNotIn("size", rec.last["payload"])
        rec, _ = self.post(b, Request(prompt="p", size=(1920, 1080)),
                           b64_response())
        self.assertEqual("16:9", rec.last["payload"]["aspect_ratio"])

    def test_none_sends_no_dimension_at_all(self):
        b = self.backend(size_param="none")
        rec, _ = self.post(b, Request(prompt="p", size=(1024, 1024)),
                           b64_response())
        self.assertEqual({"model": "some-image-model", "prompt": "p"},
                         rec.last["payload"])

    def test_unknown_size_param(self):
        with self.assertRaises(ValueError):
            self.backend(size_param="dimensions")


class TestUrlJoining(CompatTestCase):

    def test_no_trailing_slash(self):
        b = self.backend(base_url="https://api.x.ai/v1")
        rec, _ = self.post(b, Request(prompt="p"), b64_response())
        self.assertEqual("https://api.x.ai/v1/images/generations",
                         rec.last["url"])

    def test_trailing_slash(self):
        b = self.backend(base_url="https://api.x.ai/v1/")
        rec, _ = self.post(b, Request(prompt="p"), b64_response())
        self.assertEqual("https://api.x.ai/v1/images/generations",
                         rec.last["url"])

    def test_several_trailing_slashes(self):
        b = self.backend(base_url="http://127.0.0.1:1234/v1///")
        rec, _ = self.post(b, Request(prompt="p"), b64_response())
        self.assertEqual("http://127.0.0.1:1234/v1/images/generations",
                         rec.last["url"])

    def test_extra_path_segment_is_preserved(self):
        b = openai_compat.OpenAICompatBackend(preset="deepinfra", model="m")
        rec, _ = self.post(b, Request(prompt="p"), b64_response())
        self.assertEqual(
            "https://api.deepinfra.com/v1/openai/images/generations",
            rec.last["url"])


# --- the response -------------------------------------------------------

class TestResponseParsing(CompatTestCase):

    def test_b64_json_becomes_png_bytes(self):
        want = png_bytes()
        b = self.backend()
        _, images = self.post(b, Request(prompt="p"), b64_response(want))
        self.assertEqual([want], images)

    def test_together_shape_with_index_and_type_keys(self):
        want = png_bytes((7, 7, 7))
        b = openai_compat.OpenAICompatBackend(preset="together", model="m")
        _, images = self.post(b, Request(prompt="p"), together_response(want))
        self.assertEqual([want], images)

    def test_url_is_fetched_without_the_api_key(self):
        want = png_bytes((9, 90, 9))
        fetch = Recorder(want)
        b = self.backend()
        with mock.patch.object(_http, "request_bytes", fetch):
            _, images = self.post(b, Request(prompt="p"),
                                  url_response("https://cdn.example.net/x.png"))
        self.assertEqual([want], images)
        self.assertEqual("https://cdn.example.net/x.png", fetch.last["url"])
        self.assertNotIn("Authorization", fetch.last["headers"])

    def test_jpeg_is_transcoded_to_png(self):
        b = self.backend()
        _, images = self.post(b, Request(prompt="p"),
                              b64_response(jpeg_bytes()))
        self.assertEqual(1, len(images))
        self.assertTrue(images[0].startswith(b"\x89PNG\r\n\x1a\n"))

    def test_every_returned_image_comes_back(self):
        doc = {"data": [b64_response()["data"][0],
                        b64_response(png_bytes((1, 2, 3)))["data"][0]]}
        b = self.backend()
        _, images = self.post(b, Request(prompt="p", count=2), doc)
        self.assertEqual(2, len(images))
        self.assertNotEqual(images[0], images[1])

    def test_empty_data_is_an_error(self):
        b = self.backend()
        with self.assertRaises(BackendError):
            self.post(b, Request(prompt="p"), {"created": 1, "data": []})

    def test_a_body_that_is_not_the_openai_shape_is_an_error(self):
        b = self.backend()
        with self.assertRaises(BackendError):
            self.post(b, Request(prompt="p"), ["not", "the", "contract"])

    def test_entry_without_url_or_b64_is_an_error(self):
        b = self.backend()
        with self.assertRaises(BackendError) as caught:
            self.post(b, Request(prompt="p"),
                      {"data": [{"index": 0, "type": "url"}]})
        self.assertIn("response_format", str(caught.exception))


class TestErrors(CompatTestCase):

    def test_missing_key_is_an_auth_error_before_any_request(self):
        b = self.backend()
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AuthError) as caught:
                b.generate(Request(prompt="p"))
        self.assertIn("OPENAI_API_KEY", str(caught.exception))

    def test_404_explains_the_retired_ids(self):
        b = self.backend(base_url="https://api.x.ai/v1",
                         model="grok-2-image-1212")
        rec = Recorder(BackendError("HTTP 404: not found"))
        with mock.patch.object(_http, "request_json", rec):
            with self.assertRaises(BackendError) as caught:
                b.generate(Request(prompt="p"))
        message = str(caught.exception)
        self.assertIn("grok-2-image-1212", message)
        self.assertIn("2026-02-28", message)

    def test_400_points_at_the_body_keys_it_sent(self):
        b = self.backend(size_param="width_height")
        rec = Recorder(BackendError("HTTP 400: unknown field 'width'"))
        with mock.patch.object(_http, "request_json", rec):
            with self.assertRaises(BackendError) as caught:
                b.generate(Request(prompt="p"))
        self.assertIn("width_height", str(caught.exception))

    def test_classified_errors_pass_through_untouched(self):
        b = self.backend()
        original = AuthError("HTTP 401: bad key")
        rec = Recorder(original)
        with mock.patch.object(_http, "request_json", rec):
            with self.assertRaises(AuthError) as caught:
                b.generate(Request(prompt="p"))
        self.assertIs(original, caught.exception)


class TestProbe(CompatTestCase):

    def models(self, *ids):
        return {"object": "list",
                "data": [{"id": i, "object": "model"} for i in ids]}

    def test_probe_lists_models_and_never_generates(self):
        rec = Recorder(self.models("grok-imagine-image-2.0", "grok-4"))
        with mock.patch.object(_http, "request_json", rec):
            ok, message = openai_compat.OpenAICompatBackend.probe(
                preset="grok", model="grok-imagine-image-2.0")
        self.assertTrue(ok, message)
        self.assertEqual("https://api.x.ai/v1/models", rec.last["url"])
        self.assertIsNone(rec.last["payload"])          # a GET, not a generation
        self.assertEqual(1, len(rec.calls))

    def test_probe_flags_a_model_the_endpoint_does_not_list(self):
        rec = Recorder(self.models("grok-imagine-image-2.0"))
        with mock.patch.object(_http, "request_json", rec):
            ok, message = openai_compat.OpenAICompatBackend.probe(
                preset="grok", model="grok-2-image-1212")
        self.assertTrue(ok)
        self.assertIn("retired", message)

    def test_probe_without_the_env_var_makes_no_request(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            ok, message = openai_compat.OpenAICompatBackend.probe(
                preset="grok", model="grok-imagine-image-2.0")
        self.assertFalse(ok)
        self.assertIn("XAI_API_KEY", message)

    def test_probe_reports_an_unreachable_endpoint(self):
        rec = Recorder(BackendError("cannot reach host"))
        with mock.patch.object(_http, "request_json", rec):
            ok, message = openai_compat.OpenAICompatBackend.probe(
                base_url="http://127.0.0.1:9/v1", model="m")
        self.assertFalse(ok)
        self.assertIn("cannot reach", message)

    def test_probe_reports_a_bad_option_rather_than_raising(self):
        ok, message = openai_compat.OpenAICompatBackend.probe(
            base_url="https://api.example.com/v1", model="m", sampler="euler")
        self.assertFalse(ok)
        self.assertIn("sampler", message)


class TestContract(CompatTestCase):

    def test_load_accepts_this_backend(self):
        from devgraphics.backends import base as base_module
        backend = base_module.load("openai-compatible", preset="grok",
                                   model="grok-imagine-image-2.0")
        self.assertIsInstance(backend, openai_compat.OpenAICompatBackend)

    def test_registered_under_the_documented_name(self):
        from devgraphics.backends import base as base_module
        self.assertEqual("devgraphics.backends.openai_compat:OpenAICompatBackend",
                         base_module.BUILTIN["openai-compatible"])


if __name__ == "__main__":
    unittest.main()
