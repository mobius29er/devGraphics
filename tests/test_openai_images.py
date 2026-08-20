"""What this backend puts on the wire, and what it makes of what comes back.

Every test here asserts on a request body or on a response payload copied from the
research report; nothing reaches the network, because _http's three entry points
are replaced for the duration. That matters more than usual on this backend: the
failures it is written against -- `response_format` earning a hard 400, a model
that silently refuses transparency, a batch of 88 icons re-uploading the same
anchor -- are all invisible to a test that only checks the return type.

stdlib unittest so `python -m unittest` works with nothing installed; pytest
collects these classes too.
"""

import base64
import contextlib
import io
import os
import unittest
from unittest import mock

from PIL import Image

from devgraphics import _http
from devgraphics.backends import openai_images as oai
from devgraphics.backends.base import (AuthError, BackendError,
                                       ModerationBlocked, PaymentRequired,
                                       RateLimited, Request, UnsupportedOption)


def _png(color, size=(8, 8)):
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, "PNG")
    return buf.getvalue()


def _jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (12, 34, 56)).save(buf, "JPEG")
    return buf.getvalue()


CLEAR = _png((0, 0, 0, 0))              # real alpha, as background=transparent promises
OPAQUE = _png((255, 128, 0, 255))       # what the forum thread says often arrives
JPEG = _jpeg()


def _reply(png=CLEAR, background="transparent"):
    """The ImagesResponse shape from the research report, with real bytes in it."""
    return {
        "created": 1713833628,
        "data": [{"b64_json": base64.b64encode(png).decode("ascii")}],
        "background": background,
        "output_format": "png",
        "size": "1024x1024",
        "quality": "medium",
        "usage": {"total_tokens": 100, "input_tokens": 50, "output_tokens": 50,
                  "input_tokens_details": {"text_tokens": 10, "image_tokens": 40}},
    }


class Capture(object):
    """Stands in for _http.request_json and remembers what it was handed."""

    def __init__(self, reply=None, raises=None):
        self.reply = _reply() if reply is None else reply
        self.raises = raises
        self.calls = []

    def __call__(self, url, payload=None, **kw):
        self.calls.append({"url": url, "body": payload, "kw": kw})
        if self.raises is not None:
            raise self.raises
        return self.reply

    @property
    def url(self):
        return self.calls[-1]["url"]

    @property
    def body(self):
        return self.calls[-1]["body"]

    @property
    def kw(self):
        return self.calls[-1]["kw"]


def _boom(*args, **kw):
    raise AssertionError("a network call escaped the test")


class OpenAITest(unittest.TestCase):

    def setUp(self):
        env = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
        env.start()
        self.addCleanup(env.stop)

    def run_generate(self, request, cap=None, **options):
        cap = Capture() if cap is None else cap
        backend = oai.OpenAIBackend(**options)
        with mock.patch.object(_http, "request_json", cap):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                images = backend.generate(request)
        return images, cap, backend, out.getvalue()


# --- the plain path -----------------------------------------------------

class Generations(OpenAITest):

    def test_body_is_exact(self):
        request = Request(prompt="a flame", size=(1024, 1024), transparent=True)
        _images, cap, _b, _o = self.run_generate(request)
        self.assertEqual(cap.url,
                         "https://api.openai.com/v1/images/generations")
        self.assertEqual(cap.body, {
            "model": "gpt-image-1.5",
            "prompt": "a flame",
            "n": 1,
            "size": "1024x1024",
            "output_format": "png",
            "quality": "medium",
            "background": "transparent",
        })
        self.assertEqual(cap.kw["headers"],
                         {"Authorization": "Bearer sk-test"})

    def test_response_format_is_never_sent(self):
        """A hard 400 on gpt-image-*, and not as null either."""
        _i, cap, _b, _o = self.run_generate(Request(prompt="a flame"))
        self.assertNotIn("response_format", cap.body)

    def test_model_is_always_named(self):
        """Omitting it routes to dall-e-2 on this endpoint and to gpt-image-1.5
        on the other one."""
        _i, cap, _b, _o = self.run_generate(Request(prompt="a flame"))
        self.assertEqual(cap.body["model"], "gpt-image-1.5")

    def test_b64_json_becomes_png_bytes(self):
        images, _c, _b, _o = self.run_generate(Request(prompt="a flame"))
        self.assertEqual(images, [CLEAR])

    def test_url_is_fetched_when_that_is_what_came_back(self):
        """response_format is never sent, so a legacy model answers with its own
        default of url -- and with jpeg bytes behind it."""
        cap = Capture({"created": 1, "data": [{"url": "https://cdn/img.jpg"}]})
        fetch = mock.Mock(return_value=JPEG)
        backend = oai.OpenAIBackend()
        with mock.patch.object(_http, "request_json", cap):
            with mock.patch.object(_http, "request_bytes", fetch):
                images = backend.generate(Request(prompt="a flame"))
        self.assertEqual(fetch.call_args[0][0], "https://cdn/img.jpg")
        self.assertNotIn("headers", fetch.call_args[1])   # pre-signed link
        self.assertEqual(images[0][:8], bytes((0x89, 0x50, 0x4E, 0x47,
                                               0x0D, 0x0A, 0x1A, 0x0A)))

    def test_no_image_in_the_reply_is_an_error(self):
        cap = Capture({"created": 1, "data": []})
        with self.assertRaises(BackendError):
            self.run_generate(Request(prompt="a flame"), cap=cap)

    def test_size_snaps_to_an_offered_one(self):
        request = Request(prompt="a flame", size=(128, 128))
        _i, cap, _b, _o = self.run_generate(request)
        self.assertEqual(cap.body["size"], "1024x1024")

    def test_gpt_image_2_reaches_its_larger_sizes(self):
        request = Request(prompt="a flame", size=(3840, 2160))
        _i, cap, _b, _o = self.run_generate(request, model="gpt-image-2")
        self.assertEqual(cap.body["size"], "3840x2160")

    def test_count_is_capped_at_the_api_maximum(self):
        _i, cap, _b, _o = self.run_generate(Request(prompt="a flame", count=88))
        self.assertEqual(cap.body["n"], 10)

    def test_negative_is_folded_into_the_prompt(self):
        """There is no negative-prompt parameter; losing it silently is worse."""
        request = Request(prompt="a flame.", negative="text, watermark")
        _i, cap, _b, _o = self.run_generate(request)
        self.assertEqual(cap.body["prompt"], "a flame. Avoid: text, watermark")

    def test_timeout_budget_leaves_room_for_429_backoff(self):
        _i, cap, _b, _o = self.run_generate(Request(prompt="a flame"))
        self.assertEqual(cap.kw["timeout"], 600)
        self.assertEqual(cap.kw["retries"], 6)


# --- the anchor path ----------------------------------------------------

class Edits(OpenAITest):

    def test_body_is_exact(self):
        request = Request(prompt="a target", refs=(CLEAR,), transparent=True)
        _i, cap, _b, _o = self.run_generate(request)
        self.assertEqual(cap.url, "https://api.openai.com/v1/images/edits")
        self.assertEqual(cap.body, {
            "model": "gpt-image-1.5",
            "prompt": "a target",
            "n": 1,
            "size": "1024x1024",
            "output_format": "png",
            "quality": "medium",
            "background": "transparent",
            "input_fidelity": "high",
            "images": [{"image_url": "data:image/png;base64,%s"
                                     % base64.b64encode(CLEAR).decode("ascii")}],
        })

    def test_data_url_declares_the_real_mime_type(self):
        request = Request(prompt="a target", refs=(JPEG,))
        _i, cap, _b, _o = self.run_generate(request)
        self.assertTrue(cap.body["images"][0]["image_url"]
                        .startswith("data:image/jpeg;base64,"))

    def test_a_reference_that_is_not_an_image_is_refused(self):
        request = Request(prompt="a target", refs=(b"not an image at all",))
        with self.assertRaises(BackendError):
            self.run_generate(request)

    def test_response_format_is_never_sent(self):
        request = Request(prompt="a target", refs=(CLEAR,))
        _i, cap, _b, _o = self.run_generate(request)
        self.assertNotIn("response_format", cap.body)

    def test_file_ids_are_sent_instead_of_base64(self):
        """The reason this exists: an 88-icon set uploads its anchor once."""
        request = Request(prompt="a target")
        _i, cap, _b, _o = self.run_generate(
            request, ref_file_ids="file-abc123, file-def456")
        self.assertEqual(cap.url, "https://api.openai.com/v1/images/edits")
        self.assertEqual(cap.body["images"],
                         [{"file_id": "file-abc123"}, {"file_id": "file-def456"}])

    def test_upload_reference_posts_multipart_once(self):
        post = mock.Mock(return_value={"id": "file-abc123", "object": "file"})
        backend = oai.OpenAIBackend()
        with mock.patch.object(_http, "post_multipart", post):
            file_id = backend.upload_reference(CLEAR, filename="anchor.png")
        self.assertEqual(file_id, "file-abc123")
        self.assertEqual(post.call_args[0][0],
                         "https://api.openai.com/v1/files")
        self.assertEqual(post.call_args[0][1], {"purpose": "vision"})
        self.assertEqual(post.call_args[0][2], {"file": ("anchor.png", CLEAR)})

    def test_size_snaps_to_what_the_json_edits_schema_documents(self):
        """gpt-image-2 takes 3840x2160 on /generations; the edits JSON body
        documents only the three fixed sizes, so it snaps by aspect."""
        request = Request(prompt="a banner", size=(3840, 2160), refs=(CLEAR,))
        _i, cap, _b, _o = self.run_generate(request, model="gpt-image-2")
        self.assertEqual(cap.body["size"], "1536x1024")

    def test_sixteen_references_is_the_ceiling(self):
        request = Request(prompt="a target", refs=tuple([CLEAR] * 20))
        _i, cap, _b, _o = self.run_generate(request)
        self.assertEqual(len(cap.body["images"]), 16)


# --- the model table ----------------------------------------------------

class ModelTable(OpenAITest):

    def test_capabilities_answers_with_the_server_off(self):
        with mock.patch.object(_http, "request_json", _boom):
            with mock.patch.object(_http, "request_bytes", _boom):
                with mock.patch.object(_http, "post_multipart", _boom):
                    caps = oai.OpenAIBackend(model="gpt-image-1").capabilities
        self.assertEqual(caps.name, "openai/gpt-image-1")
        self.assertTrue(caps.transparent)
        self.assertFalse(caps.seed)
        self.assertEqual(caps.sizes, oai.FIXED_SIZES)

    def test_capabilities_needs_no_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(oai.OpenAIBackend().capabilities.transparent)

    def test_gpt_image_2_refuses_transparency(self):
        caps = oai.OpenAIBackend(model="gpt-image-2").capabilities
        self.assertFalse(caps.transparent)
        self.assertTrue(any("rejects background=transparent" in n
                            for n in caps.notes))

    def test_gpt_image_2_is_never_asked_for_transparency(self):
        request = Request(prompt="a flame", transparent=True)
        _i, cap, backend, _o = self.run_generate(request, model="gpt-image-2")
        self.assertNotIn("background", cap.body)
        self.assertIsNone(backend.transparency_verified)

    def test_gpt_image_1_mini_rejects_input_fidelity(self):
        caps = oai.OpenAIBackend(model="gpt-image-1-mini").capabilities
        self.assertTrue(caps.transparent)
        self.assertTrue(any("does not accept input_fidelity" in n
                            for n in caps.notes))
        request = Request(prompt="a target", refs=(CLEAR,))
        _i, cap, _b, _o = self.run_generate(request, model="gpt-image-1-mini")
        self.assertNotIn("input_fidelity", cap.body)
        with self.assertRaises(BackendError):
            self.run_generate(request, model="gpt-image-1-mini",
                              input_fidelity="high")

    def test_gpt_image_2_forces_high_fidelity_so_the_key_is_dropped(self):
        request = Request(prompt="a target", refs=(CLEAR,))
        _i, cap, _b, _o = self.run_generate(request, model="gpt-image-2",
                                            input_fidelity="high")
        self.assertNotIn("input_fidelity", cap.body)

    def test_a_snapshot_inherits_its_base_model(self):
        caps = oai.OpenAIBackend(model="gpt-image-1.5-2025-12-16").capabilities
        self.assertTrue(caps.transparent)
        self.assertEqual(caps.cost_per_image, 0.034)
        self.assertFalse(any("not in this backend's table" in n
                             for n in caps.notes))

    def test_an_unknown_model_gets_the_conservative_row(self):
        caps = oai.OpenAIBackend(model="gpt-image-9").capabilities
        self.assertFalse(caps.transparent)
        self.assertEqual(caps.reference_images, 1)
        self.assertIsNone(caps.cost_per_image)
        self.assertEqual(caps.sizes, oai.FIXED_SIZES)
        self.assertTrue(any("not in this backend's table" in n
                            for n in caps.notes))

    def test_cost_per_image_follows_model_and_quality(self):
        self.assertEqual(
            oai.OpenAIBackend(model="gpt-image-1-mini",
                              quality="low").capabilities.cost_per_image, 0.005)
        self.assertEqual(
            oai.OpenAIBackend(model="gpt-image-1",
                              quality="high").capabilities.cost_per_image, 0.167)

    def test_quality_auto_admits_it_cannot_be_priced(self):
        caps = oai.OpenAIBackend(quality="auto").capabilities
        self.assertIsNone(caps.cost_per_image)
        self.assertTrue(any("cannot be estimated" in n for n in caps.notes))


# --- transparency is a request, not a guarantee -------------------------

class Transparency(OpenAITest):

    def test_real_alpha_passes(self):
        request = Request(prompt="a flame", transparent=True)
        _i, _c, backend, out = self.run_generate(request)
        self.assertTrue(backend.transparency_verified)
        self.assertEqual(out, "")

    def test_opaque_bytes_are_flagged_for_the_cutout_fallback(self):
        request = Request(prompt="a flame", transparent=True)
        cap = Capture(_reply(OPAQUE, background="opaque"))
        _i, _c, backend, out = self.run_generate(request, cap=cap)
        self.assertFalse(backend.transparency_verified)
        self.assertIn("came back opaque", out)

    def test_jpeg_cannot_carry_the_alpha_that_was_asked_for(self):
        request = Request(prompt="a flame", transparent=True)
        with self.assertRaises(BackendError):
            self.run_generate(request, output_format="jpeg")


# --- options and errors -------------------------------------------------

class OptionsAndErrors(OpenAITest):

    def test_unknown_constructor_option_is_rejected(self):
        with self.assertRaises(UnsupportedOption):
            oai.OpenAIBackend(qualtiy="high")

    def test_unknown_request_option_is_rejected(self):
        request = Request(prompt="a flame", options={"seed": 77777})
        with self.assertRaises(UnsupportedOption):
            self.run_generate(request)

    def test_a_request_option_overrides_the_instance(self):
        request = Request(prompt="a flame", options={"model": "gpt-image-1"})
        _i, cap, _b, _o = self.run_generate(request, model="gpt-image-1.5")
        self.assertEqual(cap.body["model"], "gpt-image-1")

    def test_bad_option_values_are_refused_before_the_call(self):
        with self.assertRaises(BackendError):
            self.run_generate(Request(prompt="a flame"), quality="hd")
        with self.assertRaises(BackendError):
            self.run_generate(Request(prompt="a flame"), output_format="gif")

    def test_missing_key_is_an_auth_error_not_a_401(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AuthError):
                self.run_generate(Request(prompt="a flame"))

    def test_a_custom_key_env_is_honoured(self):
        with mock.patch.dict(os.environ, {"WORK_KEY": "sk-work"}):
            _i, cap, _b, _o = self.run_generate(Request(prompt="a flame"),
                                                api_key_env="WORK_KEY")
        self.assertEqual(cap.kw["headers"]["Authorization"], "Bearer sk-work")

    def test_moderation_block_is_not_rewrapped(self):
        """Non-retryable, and _http already classified it; wrapping it back into
        a generic BackendError would put it in the retry path."""
        blocked = ModerationBlocked("HTTP 400: [moderation_blocked]")
        with self.assertRaises(ModerationBlocked) as caught:
            self.run_generate(Request(prompt="a flame"),
                              cap=Capture(raises=blocked))
        self.assertIs(caught.exception, blocked)

    def test_rate_limit_and_payment_survive_unchanged(self):
        for err in (RateLimited("HTTP 429", retry_after=12.0),
                    PaymentRequired("HTTP 402")):
            with self.assertRaises(type(err)) as caught:
                self.run_generate(Request(prompt="a flame"),
                                  cap=Capture(raises=err))
            self.assertIs(caught.exception, err)

    def test_organisation_verification_is_named(self):
        """Users hit this wall before they hit any code problem."""
        raw = AuthError("HTTP 403: Your organization must be verified to use "
                        "the model `gpt-image-1.5`.")
        with self.assertRaises(AuthError) as caught:
            self.run_generate(Request(prompt="a flame"), cap=Capture(raises=raw))
        message = str(caught.exception)
        self.assertIn("Organization Verification", message)
        self.assertIn("10910291", message)


# --- probe --------------------------------------------------------------

class Probe(OpenAITest):

    def test_probe_generates_no_image(self):
        cap = Capture({"object": "list", "data": [{"id": "gpt-image-1.5"}]})
        with mock.patch.object(_http, "request_json", cap):
            ok, message = oai.OpenAIBackend.probe()
        self.assertTrue(ok)
        self.assertEqual(cap.url, "https://api.openai.com/v1/models")
        self.assertIsNone(cap.body)
        self.assertIn("gpt-image-1.5", message)

    def test_probe_without_a_key_says_so_rather_than_calling(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(_http, "request_json", _boom):
                ok, message = oai.OpenAIBackend.probe()
        self.assertFalse(ok)
        self.assertIn("OPENAI_API_KEY", message)

    def test_probe_reports_the_verification_wall(self):
        raw = AuthError("HTTP 403: organization must be verified")
        with mock.patch.object(_http, "request_json", Capture(raises=raw)):
            ok, message = oai.OpenAIBackend.probe()
        self.assertFalse(ok)
        self.assertIn("Organization Verification", message)

    def test_probe_rejects_a_typo_rather_than_ignoring_it(self):
        with self.assertRaises(UnsupportedOption):
            oai.OpenAIBackend.probe(api_ky="sk-test")


if __name__ == "__main__":
    unittest.main()
