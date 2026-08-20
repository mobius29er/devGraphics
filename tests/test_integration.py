"""
End-to-end tests for the seam between config, backend, loop and lockfile.

Every other test file mocks one module's transport. These drive the whole
pipeline against a fake backend, because the failures that actually bite live
between modules: a starter config naming an option no backend accepts, a dry run
that forgives a waiver the real run treats as fatal, a lockfile that records a
profile the next run then reports as drift against itself.

No network, no GPU, no API key. The fake backends draw with PIL.
"""

import io
import json
import os
import unittest

from PIL import Image, ImageDraw

from devgraphics import config, consistency, iconset, lockfile, pricing
from devgraphics.backends import base
from devgraphics.backends.base import Capabilities, Request, UnsupportedOption

OPTIONS = frozenset(("tint",))


def _icon(seed, subject, size=(256, 256), bg=(13, 13, 13)):
    """A synthetic render: dark backdrop, one centred blob, deterministic."""
    im = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(im)
    shade = (seed + sum(bytearray(subject.encode("utf-8")))) % 60
    draw.ellipse((60, 60, size[0] - 60, size[1] - 60),
                 fill=(255 - shade, 122, 40))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


class SeededBackend:
    """A local-style backend: honours a seed, so no anchor is needed."""

    def __init__(self, tint=0):
        self.tint = tint
        self.calls = []
        self.last_seed = None

    @property
    def capabilities(self):
        return Capabilities(name="fake-seeded", seed=True, deterministic=True,
                            negative_prompt=True, batch=True,
                            notes=("synthetic backend for tests",))

    def generate(self, request):
        unknown = set(request.options) - OPTIONS
        if unknown:
            raise UnsupportedOption("fake-seeded rejects %s" % sorted(unknown))
        self.calls.append(request)
        self.last_seed = request.seed
        return [_icon(request.seed or 0, request.prompt)]

    @classmethod
    def probe(cls, **_options):
        return True, "fake, always up"


class SeedlessBackend:
    """A hosted-style backend: no seed, but it takes style references."""

    def __init__(self):
        self.calls = []
        self._n = 0

    @property
    def capabilities(self):
        return Capabilities(name="fake-seedless", seed=False,
                            negative_prompt=False, transparent=True,
                            reference_images=4, cost_per_image=0.01)

    def generate(self, request):
        self.calls.append(request)
        self._n += 1
        return [_icon(self._n * 7, request.prompt)]


class Harness(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.out = os.path.join(self.tmp, "assets")
        self.subjects = {"fire": "a flame", "gem": "a cut gemstone",
                         "rocket": "a rocket ship"}
        self.logged = []

    def log(self, line=""):
        self.logged.append(str(line))

    def profile(self, **over):
        prof = config.resolve({}, None, dict(over))
        prof["output"] = dict(prof["output"], size=64)
        return prof

    def run_set(self, backend, profile=None, **kwargs):
        profile = profile or self.profile()
        original = iconset.build
        iconset.build = lambda _p: backend
        try:
            return iconset.generate(self.subjects, self.out, profile=profile,
                                    log=self.log, **kwargs)
        finally:
            iconset.build = original


class TestSeededRun(Harness):
    def test_generates_renders_and_records(self):
        backend = SeededBackend()
        made = self.run_set(backend)

        self.assertEqual(sorted(made), ["fire", "gem", "rocket"])
        for slug, path in made.items():
            self.assertTrue(os.path.exists(path), slug)
            self.assertEqual(Image.open(path).size, (64, 64))
            self.assertEqual(Image.open(path).mode, "RGBA")
            # The backdrop was keyed out, so the corners must be transparent.
            self.assertEqual(Image.open(path).convert("RGBA").getpixel((0, 0))[3], 0)
        # Full-resolution originals are kept: the tracer wants the edge detail.
        self.assertTrue(os.path.exists(os.path.join(self.out, "raw", "fire.png")))

    def test_every_subject_shares_one_seed(self):
        backend = SeededBackend()
        self.run_set(backend, self.profile(seed=4242))
        seeds = set(call.seed for call in backend.calls)
        self.assertEqual(seeds, {4242},
                         "a shared seed is the whole consistency lever")

    def test_second_run_skips_finished_icons(self):
        first = SeededBackend()
        self.run_set(first)
        second = SeededBackend()
        made = self.run_set(second)
        self.assertEqual(len(made), 3)
        self.assertEqual(second.calls, [],
                         "an interrupted batch must resume for free")

    def test_force_regenerates(self):
        self.run_set(SeededBackend())
        again = SeededBackend()
        self.run_set(again, force=True)
        self.assertEqual(len(again.calls), 3)

    def test_lockfile_records_the_profile_and_the_assets(self):
        self.run_set(SeededBackend(), self.profile(seed=99, backend="fooocus"))
        lock = lockfile.read(self.out)
        self.assertIsNotNone(lock)
        self.assertEqual(lock["profile"]["seed"], 99)
        self.assertEqual(set(lock["assets"]), {"fire", "gem", "rocket"})
        entry = lock["assets"]["fire"]
        self.assertEqual(entry["source"], "generated")
        self.assertEqual(entry["subject"], "a flame")
        self.assertEqual(entry["seed_used"], 99)
        self.assertIn("png_sha256", entry)
        self.assertGreater(entry["bg_share"], 0.3)

    def test_unknown_backend_option_is_a_clean_error(self):
        backend = SeededBackend()
        profile = self.profile()
        profile["options"] = {"nonsense": 1}
        request = Request(prompt="x", options={"nonsense": 1})
        with self.assertRaises(UnsupportedOption):
            backend.generate(request)


class TestDriftGate(Harness):
    """The guard that exists because finished icons are silently skipped."""

    def test_changing_the_seed_stops_the_next_run(self):
        self.run_set(SeededBackend(), self.profile(seed=1))
        with self.assertRaises(iconset.SetError) as caught:
            self.run_set(SeededBackend(), self.profile(seed=2))
        self.assertIn("seed", str(caught.exception))

    def test_changing_the_backend_stops_the_next_run(self):
        self.run_set(SeededBackend(), self.profile(backend="fooocus"))
        with self.assertRaises(iconset.SetError):
            self.run_set(SeededBackend(), self.profile(backend="a1111"))

    def test_allow_drift_proceeds_and_says_so(self):
        self.run_set(SeededBackend(), self.profile(seed=1))
        self.logged = []
        self.run_set(SeededBackend(), self.profile(seed=2), force=True,
                     allow_drift=True)
        self.assertTrue(any("seed" in line for line in self.logged))

    def test_an_unchanged_profile_is_not_drift(self):
        profile = self.profile(seed=7)
        self.run_set(SeededBackend(), profile)
        self.run_set(SeededBackend(), self.profile(seed=7), force=True)

    def test_hand_authored_assets_are_never_regenerated(self):
        self.run_set(SeededBackend())
        lock = lockfile.read(self.out)
        lock["assets"]["check"] = lockfile.hand(b"<svg/>", png="icons/check.svg")
        with open(lockfile.path(self.out), "w", encoding="utf-8") as handle:
            json.dump(lock, handle)

        subjects = dict(self.subjects, check="a check mark tick")
        backend = SeededBackend()
        original = iconset.build
        iconset.build = lambda _p: backend
        try:
            made = iconset.generate(subjects, self.out, profile=self.profile(),
                                    log=self.log)
        finally:
            iconset.build = original
        self.assertNotIn("check", made)
        self.assertEqual(backend.calls, [])


class TestSeedlessRun(Harness):
    """The hosted path: no seed, so an anchor and best-of-n stand in for it."""

    def test_no_seed_and_no_anchor_refuses(self):
        with self.assertRaises(iconset.SetError) as caught:
            self.run_set(SeedlessBackend())
        self.assertIn("consistency lever", str(caught.exception))

    def test_allow_drift_overrides_the_refusal(self):
        made = self.run_set(SeedlessBackend(), allow_drift=True)
        self.assertEqual(len(made), 3)

    def test_an_anchor_downgrades_the_refusal_to_a_warning(self):
        made = self.run_set(SeedlessBackend(), self.profile(anchor="fire"))
        self.assertEqual(len(made), 3)
        self.assertTrue(any("carries the style" in line for line in self.logged))

    def test_the_anchor_is_rendered_first_and_then_referenced(self):
        backend = SeedlessBackend()
        self.run_set(backend, self.profile(anchor="gem"))
        self.assertIn("a cut gemstone", backend.calls[0].prompt)
        self.assertEqual(backend.calls[0].refs, (),
                         "the anchor has nothing to reference but itself")
        for call in backend.calls[1:]:
            self.assertEqual(len(call.refs), 1)

    def test_best_of_n_costs_n_calls(self):
        backend = SeedlessBackend()
        self.run_set(backend, self.profile(anchor="fire", n=3))
        # 3 subjects x 3 candidates. The multiplier must be exactly visible.
        self.assertEqual(len(backend.calls), 9)

    def test_best_of_n_is_clamped_where_a_seed_already_pins_the_draw(self):
        backend = SeededBackend()
        self.run_set(backend, self.profile(n=4))
        self.assertEqual(len(backend.calls), 3)
        self.assertTrue(any("n=4 ignored" in line for line in self.logged))

    def test_a_seedless_backend_never_receives_a_seed(self):
        backend = SeedlessBackend()
        self.run_set(backend, self.profile(anchor="fire", seed=77777))
        for call in backend.calls:
            self.assertIsNone(call.seed)
            self.assertEqual(call.negative, "")


class TestPlanAndCost(Harness):
    def test_plan_buckets_and_ordering(self):
        work = iconset.plan(self.subjects, self.out,
                            self.profile(anchor="rocket"))
        self.assertEqual(work["todo"][0], "rocket")
        self.assertEqual(work["cached"], [])
        self.assertEqual(work["hand"], [])

    def test_local_backends_cost_nothing(self):
        work = iconset.plan(self.subjects, self.out, self.profile(backend="fooocus"))
        self.assertFalse(work["cost"], "a local backend must never quote a price")

    def test_best_of_n_multiplies_the_call_count(self):
        work = iconset.plan(self.subjects, self.out, self.profile(n=4))
        self.assertEqual(work["calls"], 12)

    def test_money_carries_its_currency(self):
        self.assertTrue(pricing.money(0.324).startswith("$"))


class TestScaffold(Harness):
    def test_placeholders_are_filled(self):
        profile = self.profile()
        profile["scaffold"] = "icon of {subject} on {bg_hex} in {palette}"
        profile["palette"] = ["#FF7A28", "#FFC400"]
        request = iconset._request(profile, "a flame")
        self.assertIn("a flame", request.prompt)
        self.assertIn("#0D0D0D", request.prompt)
        self.assertIn("#FF7A28", request.prompt)

    def test_an_unknown_placeholder_does_not_abort_the_batch(self):
        profile = self.profile()
        profile["scaffold"] = "icon of {subject}, {typo_nobody_defined}"
        request = iconset._request(profile, "a flame")
        self.assertIn("{typo_nobody_defined}", request.prompt)


class TestAuditIntegration(Harness):
    def test_audit_runs_over_a_finished_set(self):
        made = self.run_set(SeededBackend())
        result = consistency.audit(made)
        text = consistency.report(result)
        self.assertIn("semantics", text,
                      "the report must keep saying what it cannot catch")


class TestStarterConfigs(unittest.TestCase):
    """Every starter config must survive its own backend.

    This is the test that would have caught the shipped openai starter naming
    two options the openai backend rejects.
    """

    def test_every_starter_config_loads_and_builds(self):
        import tempfile
        for name in ("fooocus", "comfyui", "a1111", "openai", "gemini",
                     "invokeai", "openai-compatible"):
            with self.subTest(backend=name):
                tmp = tempfile.mkdtemp()
                path = os.path.join(tmp, "devgraphics.toml")
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(config.starter_toml(name))
                cfg = config.load(path)
                config.validate(cfg, path)
                profile = config.resolve(cfg)
                iconset.build(profile)      # constructs; touches no network


class TestBackendRegistry(unittest.TestCase):
    def test_every_builtin_imports_and_honours_the_contract(self):
        for name in base.BUILTIN:
            with self.subTest(backend=name):
                # openai-compatible is the one backend with no default endpoint:
                # a generic client cannot guess whose /v1/images/generations it
                # is meant to be talking to, so it insists on a preset or URL.
                options = ({"preset": "grok", "model": "grok-imagine-image-2.0"}
                           if name == "openai-compatible" else {})
                backend = base.load(name, **options)
                caps = backend.capabilities
                self.assertIsInstance(caps, Capabilities)
                self.assertTrue(callable(backend.generate))
                self.assertTrue(hasattr(backend, "probe"),
                                "%s needs a probe that cannot cost money" % name)

    def test_a_dotted_path_needs_no_packaging(self):
        backend = base.load("tests.test_integration:SeededBackend")
        self.assertEqual(backend.capabilities.name, "fake-seeded")

    def test_an_unknown_name_lists_what_it_knows(self):
        with self.assertRaises(base.BackendNotFound) as caught:
            base.load("stable-diffusion-webui")
        self.assertIn("comfyui", str(caught.exception))

    def test_model_maps_onto_each_backend_own_spelling(self):
        self.assertEqual(base.MODEL_OPTION.get("comfyui"), "checkpoint")
        self.assertIsNone(base.MODEL_OPTION["fooocus"])
        self.assertEqual(base.MODEL_OPTION.get("openai", "model"), "model")

    def test_a_model_fooocus_cannot_apply_is_flagged_not_silently_dropped(self):
        profile = config.resolve({}, None, {"backend": "fooocus",
                                            "model": "juggernautXL"})
        backend = iconset.build(profile)              # builds; does not raise
        self.assertEqual(iconset.advisory_model(profile), "juggernautXL")
        self.assertNotIn("model", getattr(backend, "_options", {}))

    def test_a_model_the_backend_can_apply_is_not_advisory(self):
        profile = config.resolve({}, None, {"backend": "openai",
                                            "model": "gpt-image-1"})
        self.assertIsNone(iconset.advisory_model(profile))


if __name__ == "__main__":
    unittest.main()
