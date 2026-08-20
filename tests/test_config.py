"""Tests for config, keys, lockfile and pricing.

None of these modules touches the network, and that is part of what is being
tested: `test_import_pulls_in_no_backend` asserts that importing them does not
drag in websocket-client, because `--dry-run` and `devgraphics init` have to
work on a machine with nothing installed and nothing switched on.

Written as unittest so it runs under both `python -m unittest` and pytest.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from devgraphics import config, keys, lockfile, pricing          # noqa: E402
from devgraphics.backends.base import AuthError                  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:                                      # pragma: no cover
    import tomli as tomllib


def write(directory, name, text):
    path = os.path.join(directory, name)
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


class TempCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)


# --- discovery ----------------------------------------------------------

DOTTED = 'default_profile = "dotted"\n[profile.dotted]\nbackend = "comfyui"\n'
PLAIN = 'default_profile = "plain"\n[profile.plain]\nbackend = "fooocus"\n'
PYPROJ = ('[project]\nname = "site"\n\n'
          '[tool.devgraphics]\ndefault_profile = "pyproj"\n'
          '[tool.devgraphics.profile.pyproj]\nbackend = "openai"\n')


class TestDiscovery(TempCase):
    def test_dotfile_beats_plain_file(self):
        write(self.dir, ".devgraphics.toml", DOTTED)
        write(self.dir, "devgraphics.toml", PLAIN)
        path, cfg = config.find(self.dir)
        self.assertTrue(path.endswith(".devgraphics.toml"))
        self.assertEqual(cfg["default_profile"], "dotted")

    def test_standalone_beats_pyproject(self):
        write(self.dir, "devgraphics.toml", PLAIN)
        write(self.dir, "pyproject.toml", PYPROJ)
        path, cfg = config.find(self.dir)
        self.assertTrue(path.endswith("devgraphics.toml"))
        self.assertEqual(cfg["default_profile"], "plain")

    def test_pyproject_is_the_fallback_and_is_unwrapped(self):
        write(self.dir, "pyproject.toml", PYPROJ)
        path, cfg = config.find(self.dir)
        self.assertTrue(path.endswith("pyproject.toml"))
        self.assertEqual(cfg["default_profile"], "pyproj")
        self.assertNotIn("tool", cfg)          # [tool.devgraphics] is stripped

    def test_pyproject_without_our_table_does_not_count(self):
        # A JS-adjacent Python project whose pyproject has nothing to do with us
        # must not stop the walk; the real config is one directory up.
        write(self.dir, "devgraphics.toml", PLAIN)
        nested = os.path.join(self.dir, "packages", "site")
        write(nested, "pyproject.toml", '[project]\nname = "unrelated"\n')
        path, cfg = config.find(nested)
        self.assertTrue(path.endswith("devgraphics.toml"))
        self.assertEqual(cfg["default_profile"], "plain")

    def test_walks_up_from_a_nested_directory(self):
        write(self.dir, "devgraphics.toml", PLAIN)
        deep = os.path.join(self.dir, "a", "b", "c")
        os.makedirs(deep)
        path, _cfg = config.find(deep)
        self.assertEqual(os.path.dirname(path), os.path.abspath(self.dir))

    def test_explicit_path_wins_over_discovery(self):
        write(self.dir, "devgraphics.toml", PLAIN)
        other = write(self.dir, "other.toml", DOTTED)
        path, cfg = config.discover(explicit=other, start=self.dir)
        self.assertEqual(path, other)
        self.assertEqual(cfg["default_profile"], "dotted")

    def test_explicit_path_that_is_missing_is_an_error(self):
        with self.assertRaises(config.ConfigError):
            config.discover(explicit=os.path.join(self.dir, "nope.toml"))

    def test_broken_toml_names_the_file(self):
        path = write(self.dir, "devgraphics.toml", "backend = \n")
        with self.assertRaises(config.ConfigError) as caught:
            config.load(path)
        self.assertIn("devgraphics.toml", str(caught.exception))


# --- profiles and extends ----------------------------------------------

EXTENDS = """
default_profile = "base"

[profile.base]
backend  = "fooocus"
seed     = 77777
scaffold = "flat icon of {subject}"

[profile.base.options]
host        = "127.0.0.1:7865"
performance = "Speed"

[profile.base.output]
size = 128

[profile.hosted]
extends = "base"
backend = "openai"
model   = "gpt-image-1.5"

[profile.hosted.options]
quality = "low"

[profile.sharper]
extends = "base"

[profile.sharper.options]
sharpness = 4.0
"""


class TestProfiles(unittest.TestCase):
    def setUp(self):
        self.cfg = tomllib.loads(EXTENDS)
        config.validate(self.cfg, "test")

    def test_defaults_fill_in(self):
        prof = config.resolve({})
        self.assertEqual(prof["backend"], "fooocus")
        self.assertEqual(prof["render"], (1024, 1024))
        self.assertEqual(prof["output"]["size"], 128)
        self.assertEqual(prof["postprocess"]["thresh"], 42)

    def test_default_profile_is_used_when_none_is_named(self):
        prof = config.resolve(self.cfg)
        self.assertEqual(prof["seed"], 77777)
        self.assertEqual(prof["options"]["host"], "127.0.0.1:7865")

    def test_extends_inherits_the_look(self):
        prof = config.resolve(self.cfg, "hosted")
        self.assertEqual(prof["scaffold"], "flat icon of {subject}")
        self.assertEqual(prof["seed"], 77777)
        self.assertEqual(prof["output"]["size"], 128)

    def test_options_do_not_inherit_across_a_backend_change(self):
        # The whole point: ComfyUI has no `performance` and OpenAI has no `host`,
        # so a deep merge would hand OpenAI two keys it will reject.
        prof = config.resolve(self.cfg, "hosted")
        self.assertEqual(prof["backend"], "openai")
        self.assertEqual(prof["options"], {"quality": "low"})

    def test_options_do_inherit_when_the_backend_is_unchanged(self):
        prof = config.resolve(self.cfg, "sharper")
        self.assertEqual(prof["options"]["host"], "127.0.0.1:7865")
        self.assertEqual(prof["options"]["sharpness"], 4.0)

    def test_a_cli_backend_override_also_drops_the_options(self):
        prof = config.resolve(self.cfg, "base", {"backend": "gemini"})
        self.assertEqual(prof["backend"], "gemini")
        self.assertEqual(prof["options"], {})

    def test_a_cli_override_keeps_options_when_the_backend_is_unchanged(self):
        prof = config.resolve(self.cfg, "base", {"backend": "fooocus",
                                                 "seed": 4242})
        self.assertEqual(prof["seed"], 4242)
        self.assertEqual(prof["options"]["performance"], "Speed")

    def test_unset_argparse_flags_do_not_blank_the_profile(self):
        prof = config.resolve(self.cfg, "base",
                              {"seed": None, "model": None,
                               "output": {"size": None, "svg": "flat"}})
        self.assertEqual(prof["seed"], 77777)
        self.assertEqual(prof["output"]["size"], 128)
        self.assertEqual(prof["output"]["svg"], "flat")

    def test_flat_cli_flags_are_routed_to_their_sub_table(self):
        prof = config.resolve(self.cfg, "base",
                              {"size": 64, "svg": "flat", "host": "10.0.0.4:7865"})
        self.assertEqual(prof["output"]["size"], 64)
        self.assertEqual(prof["output"]["svg"], "flat")
        self.assertEqual(prof["options"]["host"], "10.0.0.4:7865")
        self.assertEqual(prof["options"]["performance"], "Speed")

    def test_an_override_the_resolver_does_not_know_is_an_error(self):
        # Silently ignoring it would still land it in the digest and report
        # drift on the next run.
        with self.assertRaises(config.ConfigError):
            config.resolve(self.cfg, "base", {"sead": 1})

    def test_a_top_level_key_passed_as_an_override_says_so(self):
        with self.assertRaises(config.ConfigError) as caught:
            config.resolve(self.cfg, "base", {"outdir": "assets"})
        self.assertIn("top-level", str(caught.exception))

    def test_extends_cycle_is_caught(self):
        cfg = tomllib.loads('[profile.a]\nextends = "b"\n'
                            '[profile.b]\nextends = "a"\n')
        with self.assertRaises(config.ConfigError) as caught:
            config.resolve(cfg, "a")
        self.assertIn("cycle", str(caught.exception))

    def test_self_extends_is_caught(self):
        cfg = tomllib.loads('[profile.a]\nextends = "a"\n')
        with self.assertRaises(config.ConfigError):
            config.resolve(cfg, "a")

    def test_unknown_profile_suggests_a_real_one(self):
        with self.assertRaises(config.ConfigError) as caught:
            config.resolve(self.cfg, "hostd")
        self.assertIn("hosted", str(caught.exception))

    def test_render_string_becomes_two_ints(self):
        prof = config.resolve({}, None, {"render": "1536x1024"})
        self.assertEqual(prof["render"], (1536, 1024))

    def test_render_nonsense_is_rejected(self):
        with self.assertRaises(config.ConfigError):
            config.resolve({}, None, {"render": "big"})

    def test_palette_and_bg_are_normalised(self):
        prof = config.resolve({}, None, {"bg_hex": "#0d0d0d",
                                         "palette": ["#ff7a28"]})
        self.assertEqual(prof["bg_hex"], "#0D0D0D")
        self.assertEqual(prof["palette"], ["#FF7A28"])

    def test_a_bad_colour_is_rejected(self):
        with self.assertRaises(config.ConfigError):
            config.resolve({}, None, {"bg_hex": "charcoal"})

    def test_resolve_does_not_mutate_the_defaults(self):
        prof = config.resolve({})
        prof["output"]["size"] = 999
        prof["options"]["host"] = "elsewhere"
        self.assertEqual(config.DEFAULTS["output"]["size"], 128)
        self.assertEqual(config.DEFAULTS["options"], {})


# --- validation ---------------------------------------------------------

class TestValidation(unittest.TestCase):
    def test_unknown_top_level_key_suggests_the_right_one(self):
        cfg = tomllib.loads('defualt_profile = "x"\n')
        with self.assertRaises(config.ConfigError) as caught:
            config.validate(cfg, "test.toml")
        self.assertIn("default_profile", str(caught.exception))

    def test_a_key_named_api_key_is_a_hard_error(self):
        secret = "sk-proj-0123456789abcdefghijklmnop"
        cfg = tomllib.loads('[backend.openai]\napi_key = "%s"\n' % secret)
        with self.assertRaises(config.ConfigError) as caught:
            config.validate(cfg, "test.toml")
        message = str(caught.exception)
        self.assertIn("api_key_env", message)
        self.assertNotIn(secret, message)      # never echo it back

    def test_a_key_shaped_value_is_caught_wherever_it_hides(self):
        cfg = tomllib.loads('[profile.p.options]\n'
                            'authorization_header = "sk-live-0123456789abcdefgh"\n')
        with self.assertRaises(config.ConfigError) as caught:
            config.validate(cfg, "test.toml")
        self.assertIn("looks like", str(caught.exception))

    def test_naming_a_variable_is_fine(self):
        cfg = tomllib.loads('[backend.openai]\napi_key_env = "OPENAI_API_KEY"\n')
        self.assertIs(config.validate(cfg, "test.toml"), cfg)

    def test_a_backend_option_at_profile_level_says_where_it_goes(self):
        cfg = tomllib.loads('[profile.p]\nstyles = ["Fooocus V2"]\n')
        with self.assertRaises(config.ConfigError) as caught:
            config.validate(cfg, "test.toml")
        self.assertIn("[profile.p.options]", str(caught.exception))

    def test_unknown_profile_key_suggests(self):
        cfg = tomllib.loads('[profile.p]\nscaffld = "x"\n')
        with self.assertRaises(config.ConfigError) as caught:
            config.validate(cfg, "test.toml")
        self.assertIn("scaffold", str(caught.exception))

    def test_price_override_needs_a_number(self):
        cfg = tomllib.loads('[price."openai:gpt-image-1.5"]\nas_of = "2026-08-20"\n')
        with self.assertRaises(config.ConfigError):
            config.validate(cfg, "test.toml")

    def test_price_key_must_be_backend_colon_model(self):
        cfg = tomllib.loads('[price.openai]\nper_image = 0.01\n')
        with self.assertRaises(config.ConfigError):
            config.validate(cfg, "test.toml")

    def test_a_config_from_the_future_is_refused(self):
        cfg = {"config_version": 99}
        with self.assertRaises(config.ConfigError):
            config.validate(cfg, "test.toml")


# --- digest and starter -------------------------------------------------

class TestDigest(unittest.TestCase):
    def test_stable_across_key_order(self):
        prof = config.resolve({})
        shuffled = dict(reversed(list(prof.items())))
        self.assertEqual(config.digest(prof), config.digest(shuffled))

    def test_moves_when_the_seed_moves(self):
        a = config.resolve({}, None, {"seed": 1})
        b = config.resolve({}, None, {"seed": 2})
        self.assertNotEqual(config.digest(a), config.digest(b))

    def test_moves_when_a_backend_option_moves(self):
        a = config.resolve({}, None, {"options": {"performance": "Speed"}})
        b = config.resolve({}, None, {"options": {"performance": "Quality"}})
        self.assertNotEqual(config.digest(a), config.digest(b))

    def test_to_dict_is_json_able_and_readable(self):
        doc = config.to_dict(config.resolve({}))
        self.assertEqual(doc["render"], "1024x1024")
        json.dumps(doc)          # would raise on a tuple-keyed or exotic value


class TestStarter(unittest.TestCase):
    def test_every_starter_parses_validates_and_resolves(self):
        for backend in ("fooocus", "comfyui", "openai", "gemini", "invokeai"):
            text = config.starter_toml(backend)
            self.assertTrue(text.isascii(), backend)   # Windows console
            cfg = tomllib.loads(text)
            config.validate(cfg, "starter")
            prof = config.resolve(cfg)
            self.assertEqual(prof["backend"], backend)
            self.assertEqual(prof["render"], (1024, 1024))

    def test_the_comments_carry_the_measured_findings(self):
        text = config.starter_toml("fooocus")
        self.assertIn("Sticker Designs", text)         # the style trap
        self.assertIn("defeats the flood fill", text)
        self.assertIn("#0D0D0D", text)                 # what cutout keys out
        self.assertIn("hand-author symbols", text)

    def test_the_openai_starter_names_the_transparency_trap(self):
        text = config.starter_toml("openai")
        self.assertIn("gpt-image-1.5", text)
        self.assertIn("gpt-image-2", text)             # named as the one to avoid
        self.assertIn("api_key_env", text)
        self.assertNotIn("api_key =", text)            # never a value

    def test_the_scaffold_survives_the_round_trip(self):
        cfg = tomllib.loads(config.starter_toml("fooocus"))
        prof = config.resolve(cfg)
        self.assertEqual(prof["scaffold"], config.SCAFFOLD)
        self.assertEqual(prof["negative"], config.NEGATIVE)


# --- keys ---------------------------------------------------------------

class TestKeys(unittest.TestCase):
    def test_explicit_variable_wins_over_config_and_convention(self):
        cfg = {"backend": {"openai": {"api_key_env": "FROM_CONFIG"}}}
        env = {"MY_VAR": "explicit", "FROM_CONFIG": "config",
               "OPENAI_API_KEY": "conventional"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(keys.resolve("openai", "MY_VAR", cfg),
                             ("explicit", "MY_VAR"))

    def test_config_variable_wins_over_convention(self):
        cfg = {"backend": {"openai": {"api_key_env": "FROM_CONFIG"}}}
        env = {"FROM_CONFIG": "config", "OPENAI_API_KEY": "conventional"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(keys.resolve("openai", None, cfg),
                             ("config", "FROM_CONFIG"))

    def test_conventional_variable_is_the_fallback(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True):
            self.assertEqual(keys.resolve("openai"), ("k", "OPENAI_API_KEY"))

    def test_gemini_prefers_gemini_then_google(self):
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "g"}, clear=True):
            self.assertEqual(keys.resolve("gemini"), ("g", "GOOGLE_API_KEY"))
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "g",
                                          "GEMINI_API_KEY": "m"}, clear=True):
            self.assertEqual(keys.resolve("gemini"), ("m", "GEMINI_API_KEY"))

    def test_compat_presets_come_from_the_backend_table(self):
        with mock.patch.dict(os.environ, {"XAI_API_KEY": "x"}, clear=True):
            self.assertEqual(keys.resolve("grok"), ("x", "XAI_API_KEY"))

    def test_missing_key_names_the_variable_to_set(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AuthError) as caught:
                keys.resolve("openai")
        self.assertIn("OPENAI_API_KEY", str(caught.exception))

    def test_an_empty_variable_counts_as_missing(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "  "}, clear=True):
            with self.assertRaises(AuthError):
                keys.resolve("openai")

    def test_a_key_value_in_the_config_is_refused_here_too(self):
        cfg = {"backend": {"openai": {"api_key": "sk-whatever"}}}
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True):
            with self.assertRaises(config.ConfigError):
                keys.resolve("openai", None, cfg)


ENV_FILE = """
# a comment
OPENAI_API_KEY=from-file
export GEMINI_API_KEY="quoted-from-file"
EMPTY=
SPACED = padded

not a pair
"""


class TestEnvFile(TempCase):
    def test_the_real_environment_wins_over_the_file(self):
        path = write(self.dir, ".env", ENV_FILE)
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "from-shell"},
                             clear=True):
            applied = keys.load_env_file(path)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "from-shell")
            self.assertNotIn("OPENAI_API_KEY", applied)
            self.assertEqual(os.environ["GEMINI_API_KEY"], "quoted-from-file")

    def test_parsing_details(self):
        path = write(self.dir, ".env", ENV_FILE)
        with mock.patch.dict(os.environ, {}, clear=True):
            keys.load_env_file(path)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "from-file")
            self.assertEqual(os.environ["GEMINI_API_KEY"], "quoted-from-file")
            self.assertEqual(os.environ["EMPTY"], "")
            self.assertEqual(os.environ["SPACED"], "padded")
            self.assertNotIn("not a pair", os.environ)

    def test_env_files_are_never_loaded_by_themselves(self):
        # No import-time or resolve-time .env pickup: a run must not behave
        # differently because of which directory it started in.
        write(self.dir, ".env", "OPENAI_API_KEY=sneaky\n")
        cwd = os.getcwd()
        os.chdir(self.dir)
        self.addCleanup(os.chdir, cwd)
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AuthError):
                keys.resolve("openai")


# --- lockfile -----------------------------------------------------------

class TestLockfile(TempCase):
    def setUp(self):
        super(TestLockfile, self).setUp()
        self.profile = config.resolve({}, None, {
            "backend": "fooocus", "model": "juggernautXL_v8Rundiffusion",
            "seed": 77777, "options": {"host": "127.0.0.1:7865"}})

    def _write(self, profile=None, assets=None, previous=None):
        return lockfile.write(self.dir, "brand-icons", profile or self.profile,
                              assets or {}, previous=previous)

    def test_round_trip(self):
        entry = lockfile.generated("a flame", 77777, b"pretend png",
                                   png="icons/fire.png", bg_share=0.731)
        self._write(assets={"fire": entry})
        doc = lockfile.read(self.dir)
        self.assertEqual(doc["lockfile_version"], lockfile.LOCKFILE_VERSION)
        self.assertEqual(doc["profile_name"], "brand-icons")
        self.assertEqual(doc["profile"]["render"], "1024x1024")
        self.assertEqual(doc["assets"]["fire"]["seed_used"], 77777)
        self.assertEqual(doc["assets"]["fire"]["bg_share"], 0.731)
        self.assertEqual(doc["assets"]["fire"]["png_sha256"],
                         lockfile.sha256(b"pretend png"))
        self.assertTrue(doc["assets"]["fire"]["generated"].endswith("Z"))

    def test_the_note_refuses_to_promise_pixels(self):
        self._write()
        doc = lockfile.read(self.dir)
        self.assertIn("does NOT guarantee byte-identical", doc["note"])
        self.assertNotIn("reproducible", json.dumps(doc).lower())

    def test_missing_lock_reads_as_none(self):
        self.assertIsNone(lockfile.read(self.dir))

    def test_a_corrupt_lock_is_not_silently_ignored(self):
        write(self.dir, lockfile.NAME, "{not json")
        with self.assertRaises(ValueError):
            lockfile.read(self.dir)

    def test_hand_authored_entries_survive_a_later_partial_run(self):
        first = {"check": lockfile.hand(b"<svg/>", png="icons/check.svg")}
        self._write(assets=first)
        previous = lockfile.read(self.dir)
        self._write(assets={"fire": lockfile.generated("a flame", 1, b"x")},
                    previous=previous)
        doc = lockfile.read(self.dir)
        self.assertEqual(sorted(doc["assets"]), ["check", "fire"])
        self.assertEqual(doc["assets"]["check"]["source"], "hand")
        self.assertEqual(lockfile.hand_slugs(doc), ["check"])

    def test_no_drift_when_nothing_changed(self):
        self._write()
        self.assertEqual(lockfile.compare(lockfile.read(self.dir), self.profile), [])

    def test_a_changed_backend_is_detected(self):
        # The realistic failure: 40 icons on Fooocus in January, the next 48 on
        # OpenAI in March. Every run succeeded; the set stopped being a set.
        self._write()
        moved = config.resolve({}, None, {"backend": "openai",
                                          "model": "gpt-image-1.5"})
        changes = lockfile.compare(lockfile.read(self.dir), moved)
        text = "\n".join(changes)
        self.assertIn("backend", text)
        self.assertIn("fooocus -> openai", text)
        self.assertIn("model", text)

    def test_a_changed_seed_is_detected(self):
        self._write()
        moved = dict(self.profile, seed=12345)
        changes = lockfile.compare(lockfile.read(self.dir), moved)
        self.assertEqual(len(changes), 1)
        self.assertIn("77777 -> 12345", changes[0])

    def test_a_changed_scaffold_is_detected_and_clipped(self):
        self._write()
        moved = dict(self.profile, scaffold="something else entirely " * 10)
        changes = lockfile.compare(lockfile.read(self.dir), moved)
        self.assertEqual(len(changes), 1)
        self.assertIn("scaffold", changes[0])
        self.assertLess(len(changes[0]), 130)

    def test_an_options_only_change_still_shows_up(self):
        self._write()
        moved = config.resolve({}, None, {
            "backend": "fooocus", "model": "juggernautXL_v8Rundiffusion",
            "seed": 77777, "options": {"host": "10.0.0.4:7865"}})
        changes = lockfile.compare(lockfile.read(self.dir), moved)
        self.assertEqual(len(changes), 1)
        self.assertIn("options changed", changes[0])

    def test_drift_report_tells_the_user_what_to_do(self):
        self._write()
        moved = dict(self.profile, seed=1)
        report = lockfile.drift_report(
            self.dir, lockfile.compare(lockfile.read(self.dir), moved))
        self.assertIn("--allow-drift", report)
        self.assertIn(lockfile.NAME, report)

    def test_nothing_to_compare_against_is_not_drift(self):
        self.assertEqual(lockfile.compare(None, self.profile), [])

    def test_paths_are_recorded_with_forward_slashes(self):
        entry = lockfile.generated("a flame", 1, b"x", png="icons\\fire.png")
        self.assertEqual(entry["png"], "icons/fire.png")


# --- pricing ------------------------------------------------------------

class TestPricing(unittest.TestCase):
    def test_a_known_model_multiplies_out(self):
        usd, provenance = pricing.estimate("openai", "gpt-image-1.5", 88)
        self.assertAlmostEqual(usd, 0.792, places=4)
        self.assertIn("as_of", provenance)
        self.assertIn("estimate, not a quote", provenance)

    def test_best_of_n_multiplies_the_bill(self):
        usd, provenance = pricing.estimate("openai", "gpt-image-1.5", 88, n=3)
        self.assertAlmostEqual(usd, 2.376, places=4)
        self.assertIn("n=3", provenance)

    def test_an_unknown_model_is_none_and_says_so(self):
        usd, provenance = pricing.estimate("openai", "gpt-image-99", 88)
        self.assertIsNone(usd)
        self.assertIn("unknown", provenance)
        self.assertIn("gpt-image-99", provenance)

    def test_an_unknown_backend_is_none(self):
        usd, _ = pricing.estimate("recraft", "whatever", 10)
        self.assertIsNone(usd)

    def test_a_local_backend_is_free_not_unknown(self):
        usd, provenance = pricing.estimate("fooocus", None, 88)
        self.assertEqual(usd, 0.0)
        self.assertIn("locally", provenance)

    def test_the_config_override_wins_and_is_credited(self):
        overrides = {"openai:gpt-image-1.5": {"per_image": 0.02,
                                              "as_of": "2027-01-01"}}
        usd, provenance = pricing.estimate("openai", "gpt-image-1.5", 10,
                                           overrides=overrides)
        self.assertAlmostEqual(usd, 0.2, places=4)
        self.assertIn("your config", provenance)
        self.assertIn("2027-01-01", provenance)

    def test_an_override_can_price_a_model_the_table_never_heard_of(self):
        overrides = {"openai-compatible:some-new-model": {"per_image": 0.5}}
        usd, provenance = pricing.estimate("openai-compatible", "some-new-model",
                                           4, overrides=overrides)
        self.assertEqual(usd, 2.0)
        self.assertIn("undated", provenance)

    def test_every_shipped_row_carries_a_date_and_a_source(self):
        for key, row in pricing.TABLE.items():
            usd, as_of, note = row
            self.assertGreater(usd, 0, key)
            self.assertRegex(as_of, r"^\d{4}-\d{2}-\d{2}$")
            self.assertTrue(note.strip(), key)

    def test_known_models_lists_by_backend(self):
        self.assertIn("gpt-image-1.5", pricing.known_models("openai"))
        self.assertEqual(pricing.known_models("fooocus"), [])


# --- no network, no heavy imports ---------------------------------------

class TestImports(unittest.TestCase):
    def test_import_pulls_in_no_backend(self):
        """--dry-run and `init` must work with nothing installed and nothing on.

        A subprocess rather than a sys.modules check, because another test in
        the same session may already have imported a backend.
        """
        code = ("import sys;"
                "import devgraphics.config, devgraphics.keys,"
                " devgraphics.lockfile, devgraphics.pricing;"
                "bad = [m for m in sys.modules"
                " if m.startswith('websocket') or m.startswith('PIL')"
                " or m == 'devgraphics.backends.fooocus'];"
                "print(bad)")
        out = subprocess.check_output([sys.executable, "-c", code], cwd=REPO)
        self.assertEqual(out.decode().strip(), "[]")


if __name__ == "__main__":
    unittest.main()
