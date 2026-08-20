"""Gemini backend tests. No network: request_json is replaced everywhere.

Assertions are on the exact request this code builds and on response payloads
copied from the research report, because those two are the parts that cannot be
checked any other way without spending 6.7 cents a call.
"""

import base64
import io
import os
import unittest
from unittest import mock

from PIL import Image

from devgraphics.backends import gemini
from devgraphics.backends.base import (BackendError, PaymentRequired, Request,
                                       UnsupportedOption)
from devgraphics.backends.gemini import GeminiBackend
from devgraphics.postprocess import PNG_MAGIC

FINAL = (12, 200, 40)
DRAFT = (200, 12, 40)

URL_INTERACTIONS = "https://generativelanguage.googleapis.com/v1beta/interactions"
URL_CONTENT = ("https://generativelanguage.googleapis.com/v1/models/"
               "gemini-3.1-flash-image:generateContent")


def _jpeg(colour):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), colour).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def _png(colour):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), colour).save(buf, "PNG")
    return buf.getvalue()


def _b64(data):
    return base64.b64encode(data).decode("ascii")


def _which(png):
    """FINAL or DRAFT, whichever the decoded image is nearer to."""
    pixel = Image.open(io.BytesIO(png)).convert("RGB").getpixel((8, 8))
    dist = lambda c: sum((a - b) ** 2 for a, b in zip(pixel, c))
    return FINAL if dist(FINAL) <= dist(DRAFT) else DRAFT


def interaction(final=None, status="completed"):
    """The Interactions success payload, shaped exactly as the research records it.

    The leading step is a `thought` carrying an interim render: a walk that grabs
    the first image block in the document returns that draft instead of the icon.
    """
    return {
        "created": "2025-11-26T12:25:15Z",
        "id": "v1_ChdPU0F4YWFtNkFwS2kxZThQZ05lbXdROBIX",
        "model": "gemini-3.1-flash-image",
        "object": "interaction",
        "status": status,
        "steps": [
            {"type": "thought",
             "summary": [{"type": "image", "mime_type": "image/jpeg",
                          "data": _b64(_jpeg(DRAFT))}]},
            {"type": "model_output",
             "content": [
                 {"type": "text", "text": "Here is your generated image:"},
                 {"type": "image", "mime_type": "image/jpeg",
                  "data": _b64(final if final is not None else _jpeg(FINAL))}]},
        ],
        "updated": "2025-11-26T12:25:19Z",
        "usage": {"total_input_tokens": 7, "total_output_tokens": 20,
                  "total_thought_tokens": 22, "total_tokens": 49},
    }


def candidates(final=None):
    """The generateContent success payload, with a leading text part and a
    thought part ahead of the real image."""
    return {"candidates": [{"content": {"parts": [
        {"text": "Here is your generated image:"},
        {"thought": True,
         "inlineData": {"mimeType": "image/jpeg", "data": _b64(_jpeg(DRAFT))}},
        {"inlineData": {"mimeType": "image/jpeg",
                        "data": _b64(final if final is not None
                                     else _jpeg(FINAL))}},
    ]}}]}


class Recorder(object):
    """Stands in for request_json and remembers what it was handed."""

    def __init__(self, response=None):
        self.response = response
        self.calls = []

    def __call__(self, url, payload=None, **kwargs):
        self.calls.append({"url": url, "payload": payload, "kwargs": kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    @property
    def call(self):
        return self.calls[-1]


class GeminiTestCase(unittest.TestCase):

    def patch(self, response=None):
        rec = Recorder(response)
        original = gemini.request_json
        gemini.request_json = rec
        self.addCleanup(setattr, gemini, "request_json", original)
        return rec

    def offline(self):
        """Any HTTP at all is a test failure."""
        def boom(*args, **kwargs):
            raise AssertionError("this path must not touch the network")
        original = gemini.request_json
        gemini.request_json = boom
        self.addCleanup(setattr, gemini, "request_json", original)


class TestInteractionsRequest(GeminiTestCase):

    def test_body_url_and_auth_header(self):
        rec = self.patch(interaction())
        GeminiBackend(api_key="test-key").generate(Request(prompt="a flame"))

        self.assertEqual(rec.call["url"], URL_INTERACTIONS)
        self.assertEqual(rec.call["payload"], {
            "model": "gemini-3.1-flash-image",
            "input": [{"type": "text", "text": "a flame"}],
            "response_format": {"type": "image", "aspect_ratio": "1:1",
                                "image_size": "1K"},
            "store": False,
        })
        self.assertEqual(rec.call["kwargs"]["headers"],
                         {"x-goog-api-key": "test-key"})

    def test_reference_images_are_inlined_after_the_prompt(self):
        rec = self.patch(interaction())
        ref = _png((7, 7, 7))
        GeminiBackend(api_key="k").generate(
            Request(prompt="a flame", refs=(ref,)))

        self.assertEqual(rec.call["payload"]["input"], [
            {"type": "text", "text": "a flame"},
            {"type": "image", "mime_type": "image/png", "data": _b64(ref)},
        ])

    def test_size_picks_an_aspect_enum_and_a_tier(self):
        rec = self.patch(interaction())
        back = GeminiBackend(api_key="k")

        # 1536 sits exactly between the 1K and 2K tiers; the tie resolves down.
        back.generate(Request(prompt="p", size=(1024, 1536)))
        self.assertEqual(rec.call["payload"]["response_format"],
                         {"type": "image", "aspect_ratio": "2:3",
                          "image_size": "1K"})

        back.generate(Request(prompt="p", size=(1800, 1800)))
        self.assertEqual(rec.call["payload"]["response_format"]["image_size"],
                         "2K")

        back.generate(Request(prompt="p", size=(2048, 2048)))
        self.assertEqual(rec.call["payload"]["response_format"]["image_size"],
                         "2K")

    def test_both_spellings_of_the_512_tier_go_through_verbatim(self):
        rec = self.patch(interaction())
        back = GeminiBackend(api_key="k")
        for asked in ("512", "512px"):
            back.generate(Request(prompt="p", options={"image_size": asked}))
            self.assertEqual(
                rec.call["payload"]["response_format"]["image_size"], asked)

    def test_lite_model_never_asks_for_a_tier_it_lacks(self):
        rec = self.patch(interaction())
        GeminiBackend(api_key="k", model="gemini-3.1-flash-lite-image").generate(
            Request(prompt="p", size=(4096, 4096)))
        self.assertEqual(rec.call["payload"]["response_format"]["image_size"],
                         "1K")

    def test_store_is_sent_and_overridable(self):
        rec = self.patch(interaction())
        GeminiBackend(api_key="k", store=True).generate(Request(prompt="p"))
        self.assertIs(rec.call["payload"]["store"], True)

        GeminiBackend(api_key="k").generate(
            Request(prompt="p", options={"store": "false"}))
        self.assertIs(rec.call["payload"]["store"], False)

    def test_system_instruction_and_thinking_level(self):
        rec = self.patch(interaction())
        GeminiBackend(api_key="k", system_instruction="flat vector scaffold",
                      thinking_level="high").generate(Request(prompt="p"))
        self.assertEqual(rec.call["payload"]["system_instruction"],
                         "flat vector scaffold")
        self.assertEqual(rec.call["payload"]["generation_config"],
                         {"thinking_level": "high"})


class TestSeed(GeminiTestCase):

    def test_request_seed_is_never_sent(self):
        """capabilities.seed is False, so a seed must not reach the wire dressed
        up as a consistency promise."""
        rec = self.patch(interaction())
        back = GeminiBackend(api_key="k")
        self.assertFalse(back.capabilities.seed)
        self.assertFalse(back.capabilities.deterministic)

        back.generate(Request(prompt="p", seed=77_777))
        self.assertNotIn("generation_config", rec.call["payload"])
        self.assertNotIn("seed", str(rec.call["payload"]))

    def test_explicit_seed_option_is_sent_and_changes_no_promise(self):
        rec = self.patch(interaction())
        back = GeminiBackend(api_key="k")
        back.generate(Request(prompt="p", options={"seed": 12345}))
        self.assertEqual(rec.call["payload"]["generation_config"], {"seed": 12345})
        self.assertFalse(back.capabilities.seed)

    def test_seed_reaches_generation_config_on_generatecontent_too(self):
        rec = self.patch(candidates())
        GeminiBackend(api_key="k", surface="generatecontent").generate(
            Request(prompt="p", seed=77_777, options={"seed": 5}))
        self.assertEqual(rec.call["payload"]["generationConfig"]["seed"], 5)


class TestResponseWalk(GeminiTestCase):

    def test_thought_images_are_skipped(self):
        self.patch(interaction())
        out = GeminiBackend(api_key="k").generate(Request(prompt="p"))
        self.assertEqual(len(out), 1)
        self.assertEqual(_which(out[0]), FINAL)

    def test_jpeg_is_transcoded_to_png(self):
        self.patch(interaction())
        out = GeminiBackend(api_key="k").generate(Request(prompt="p"))
        self.assertEqual(out[0][:8], PNG_MAGIC)
        self.assertEqual(Image.open(io.BytesIO(out[0])).format, "PNG")
        self.assertEqual(Image.open(io.BytesIO(out[0])).size, (16, 16))

    def test_a_png_response_passes_through_unchanged(self):
        png = _png(FINAL)
        self.patch(interaction(final=png))
        out = GeminiBackend(api_key="k").generate(Request(prompt="p"))
        self.assertEqual(out[0], png)

    def test_unfinished_status_is_an_error(self):
        self.patch(interaction(status="failed"))
        with self.assertRaises(BackendError) as ctx:
            GeminiBackend(api_key="k").generate(Request(prompt="p"))
        self.assertIn("failed", str(ctx.exception))

    def test_budget_exceeded_is_fatal_for_the_batch(self):
        self.patch(interaction(status="budget_exceeded"))
        with self.assertRaises(PaymentRequired):
            GeminiBackend(api_key="k").generate(Request(prompt="p"))

    def test_no_image_block_raises_rather_than_returning_nothing(self):
        self.patch({"status": "completed", "id": "x", "steps": [
            {"type": "model_output",
             "content": [{"type": "text", "text": "I cannot do that"}]}]})
        with self.assertRaises(BackendError) as ctx:
            GeminiBackend(api_key="k").generate(Request(prompt="p"))
        self.assertIn("no image block", str(ctx.exception))

    def test_count_does_not_loop_here(self):
        """batch=False means the caller loops; looping here too would multiply."""
        rec = self.patch(interaction())
        out = GeminiBackend(api_key="k").generate(Request(prompt="p", count=4))
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(len(out), 1)


class TestGenerateContentSurface(GeminiTestCase):

    def test_body_is_camelcase_and_the_model_is_in_the_path(self):
        rec = self.patch(candidates())
        GeminiBackend(api_key="k", surface="generatecontent").generate(
            Request(prompt="a flame"))

        self.assertEqual(rec.call["url"], URL_CONTENT)
        self.assertEqual(rec.call["payload"], {
            "contents": [{"role": "user", "parts": [{"text": "a flame"}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "responseFormat": {"image": {"aspectRatio": "1:1",
                                             "imageSize": "1K"}}},
        })

    def test_refs_go_in_as_inline_data(self):
        rec = self.patch(candidates())
        ref = _png((9, 9, 9))
        GeminiBackend(api_key="k", surface="generate_content").generate(
            Request(prompt="p", refs=(ref,)))
        self.assertEqual(rec.call["payload"]["contents"][0]["parts"][1],
                         {"inline_data": {"mime_type": "image/png",
                                          "data": _b64(ref)}})

    def test_walk_skips_the_text_part_and_the_thought_part(self):
        self.patch(candidates())
        out = GeminiBackend(api_key="k", surface="generatecontent").generate(
            Request(prompt="p"))
        self.assertEqual(len(out), 1)
        self.assertEqual(_which(out[0]), FINAL)
        self.assertEqual(out[0][:8], PNG_MAGIC)

    def test_store_is_not_a_generatecontent_field(self):
        self.offline()
        with self.assertRaises(UnsupportedOption):
            GeminiBackend(api_key="k", surface="generatecontent", store=False)
        back = GeminiBackend(api_key="k", surface="generatecontent")
        with self.assertRaises(UnsupportedOption):
            back.generate(Request(prompt="p", options={"store": True}))


class TestCapabilities(GeminiTestCase):

    def test_answerable_with_no_key_and_no_network(self):
        self.offline()
        with mock.patch.dict(os.environ, {}, clear=True):
            for model in gemini.MODELS:
                caps = GeminiBackend(model=model).capabilities
                self.assertEqual(caps.name, model)
                self.assertFalse(caps.seed)
                self.assertFalse(caps.transparent)
                self.assertFalse(caps.negative_prompt)
                self.assertFalse(caps.batch)
                self.assertIsNotNone(caps.cost_per_image)

    def test_reference_images_are_per_model(self):
        self.offline()
        counts = {m: GeminiBackend(api_key="k", model=m).capabilities.reference_images
                  for m in gemini.MODELS}
        self.assertEqual(counts, {"gemini-3.1-flash-image": 14,
                                  "gemini-3.1-flash-lite-image": 14,
                                  "gemini-3-pro-image": 11,
                                  "gemini-2.5-flash-image": 3})

    def test_only_flash_image_advertises_style_references(self):
        self.offline()
        notes = " ".join(GeminiBackend(api_key="k").capabilities.notes).lower()
        self.assertIn("style references", notes)

        for model in ("gemini-3.1-flash-lite-image", "gemini-3-pro-image"):
            notes = " ".join(
                GeminiBackend(api_key="k", model=model).capabilities.notes).lower()
            self.assertIn("no style-reference slot", notes)
            self.assertIn("gemini-3.1-flash-image", notes)

    def test_notes_carry_the_jpeg_watermark_and_storage_caveats(self):
        self.offline()
        notes = " ".join(GeminiBackend(api_key="k").capabilities.notes).lower()
        self.assertIn("not measured", notes)
        self.assertIn("thresh=42", notes)
        self.assertIn("synthid", notes)
        self.assertIn("previous_interaction_id", notes)
        self.assertIn("free tier", notes)

    def test_sizes_and_cost_track_the_model(self):
        self.offline()
        flash = GeminiBackend(api_key="k").capabilities
        self.assertEqual(flash.sizes, ((512, 512), (1024, 1024), (2048, 2048),
                                       (4096, 4096)))
        self.assertAlmostEqual(flash.cost_per_image, 0.067)

        lite = GeminiBackend(api_key="k",
                             model="gemini-3.1-flash-lite-image").capabilities
        self.assertEqual(lite.sizes, ((1024, 1024),))
        self.assertAlmostEqual(lite.cost_per_image, 0.0336)

    def test_an_unknown_model_id_is_allowed_and_says_so(self):
        self.offline()
        caps = GeminiBackend(api_key="k",
                             model="gemini-4-flash-image").capabilities
        self.assertEqual(caps.sizes, ())
        self.assertIsNone(caps.cost_per_image)
        self.assertIn("conservative guess", " ".join(caps.notes))


class TestRejections(GeminiTestCase):

    def test_imagen_is_refused_with_the_shutdown_date(self):
        self.offline()
        for model in ("imagen-4.0-generate-001", "imagen-4.0-ultra-generate-001",
                      "imagen-4.0-fast-generate-001"):
            with self.assertRaises(BackendError) as ctx:
                GeminiBackend(api_key="k", model=model)
            self.assertIn("2026-08-17", str(ctx.exception))
            self.assertIn("gemini-3.1-flash-image", str(ctx.exception))

    def test_probe_refuses_imagen_without_calling_out(self):
        self.offline()
        ok, message = GeminiBackend.probe(api_key="k",
                                          model="imagen-4.0-generate-001")
        self.assertFalse(ok)
        self.assertIn("2026-08-17", message)

    def test_unknown_request_option_raises(self):
        self.offline()
        back = GeminiBackend(api_key="k")
        with self.assertRaises(UnsupportedOption) as ctx:
            back.generate(Request(prompt="p", options={"negative_prompt": "no"}))
        self.assertIn("negative_prompt", str(ctx.exception))
        self.assertIn("aspect_ratio", str(ctx.exception))   # lists what it takes

    def test_unknown_constructor_option_raises(self):
        self.offline()
        with self.assertRaises(TypeError):
            GeminiBackend(api_key="k", nagative="typo")

    def test_bad_enum_values_raise(self):
        self.offline()
        back = GeminiBackend(api_key="k")
        with self.assertRaises(UnsupportedOption):
            back.generate(Request(prompt="p", options={"image_size": "1k"}))
        with self.assertRaises(UnsupportedOption):
            back.generate(Request(prompt="p", options={"aspect_ratio": "7:3"}))
        with self.assertRaises(UnsupportedOption):
            GeminiBackend(api_key="k", surface="rest")

    def test_thinking_level_is_flash_image_only(self):
        self.offline()
        with self.assertRaises(UnsupportedOption) as ctx:
            GeminiBackend(api_key="k", model="gemini-3-pro-image",
                          thinking_level="high")
        self.assertIn("gemini-3.1-flash-image", str(ctx.exception))

    def test_strip_ratios_are_flash_image_only(self):
        self.offline()
        GeminiBackend(api_key="k")._aspect("21:9", (1024, 1024))
        GeminiBackend(api_key="k")._aspect("8:1", (1024, 1024))
        with self.assertRaises(UnsupportedOption):
            GeminiBackend(api_key="k", model="gemini-3-pro-image")._aspect(
                "8:1", (1024, 1024))

    def test_too_many_reference_images_is_loud(self):
        self.offline()
        back = GeminiBackend(api_key="k", model="gemini-2.5-flash-image")
        with self.assertRaises(BackendError) as ctx:
            back.generate(Request(prompt="p", refs=tuple([_png(FINAL)] * 4)))
        self.assertIn("at most 3", str(ctx.exception))

    def test_missing_key_is_an_authorerror_before_any_request(self):
        self.offline()
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(BackendError) as ctx:
                GeminiBackend().generate(Request(prompt="p"))
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_key_is_read_from_the_environment(self):
        rec = self.patch(interaction())
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "from-env"},
                             clear=True):
            GeminiBackend().generate(Request(prompt="p"))
        self.assertEqual(rec.call["kwargs"]["headers"]["x-goog-api-key"],
                         "from-env")


class TestProbe(GeminiTestCase):

    def test_lists_models_and_generates_nothing(self):
        rec = self.patch({"models": [
            {"name": "models/gemini-3.1-flash-image"},
            {"name": "models/gemini-3-pro-image"}]})
        ok, message = GeminiBackend.probe(api_key="k")

        self.assertTrue(ok)
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(rec.call["url"],
                         "https://generativelanguage.googleapis.com/v1beta/models")
        self.assertIsNone(rec.call["payload"])          # a GET, so no body
        self.assertNotIn("interactions", rec.call["url"])
        self.assertIn("gemini-3.1-flash-image", message)

    def test_model_absent_from_the_listing_fails(self):
        self.patch({"models": [{"name": "models/gemini-2.5-flash"}]})
        ok, message = GeminiBackend.probe(api_key="k",
                                          model="gemini-3.1-flash-image")
        self.assertFalse(ok)
        self.assertIn("billing", message)

    def test_no_key_fails_without_a_request(self):
        self.offline()
        with mock.patch.dict(os.environ, {}, clear=True):
            ok, message = GeminiBackend.probe()
        self.assertFalse(ok)
        self.assertIn("GEMINI_API_KEY", message)


if __name__ == "__main__":
    unittest.main()
