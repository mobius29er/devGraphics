"""
Tests for what happens when an optional dependency is not installed.

`pip install devgraphics` brings Pillow and a TOML parser and nothing else. That
is deliberate -- five of the seven backends are stdlib urllib and base64, so
making everyone carry a websocket client and a Rust wheel would be a tax on the
majority to serve the minority.

The cost of that choice is a failure mode nobody exercises on a developer machine,
where everything is installed. These tests are the exercise. They simulate the
absent module rather than trusting that the try/except reads correctly, because
the whole point is that a stranger hits this path and we never will.

Two properties matter and they pull in opposite directions:

  The module must still IMPORT. `devgraphics backends` has to list and describe
  Fooocus on a machine that cannot drive it, or the discovery command breaks for
  exactly the person trying to work out what they need.

  The failure must still HAPPEN, at the first moment it is real, and say what to
  install. A stub that silently no-ops would be worse than an ImportError.
"""

import contextlib
import importlib
import sys
import unittest
from unittest import mock

from devgraphics.backends.base import MissingDependency

HAVE_TOMLI = importlib.util.find_spec("tomli") is not None


class Absent(object):
    """A meta_path finder that makes one module unimportable.

    sys.modules[name] = None also makes `import name` raise, but it leaves the
    entry behind for anything that looks the module up directly. Refusing at the
    finder is closer to the real situation: the package is simply not installed.
    """

    def __init__(self, name):
        self.name = name

    def find_module(self, fullname, path=None):           # pragma: no cover
        return None                                        # legacy API, unused

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name or fullname.startswith(self.name + "."):
            # ModuleNotFoundError specifically, not a bare ImportError: that is
            # what a genuinely absent package raises, and config.py catches the
            # narrower type. A test that raised the parent would pass against
            # code that cannot survive the real thing.
            raise ModuleNotFoundError("No module named %r" % fullname,
                                      name=fullname)
        return None


@contextlib.contextmanager
def absent(*names):
    """Make each named package unimportable for the duration of the block.

    A context manager rather than a helper that wraps one reload, because the
    imports being tested are not all at module scope. `to_svg` imports vtracer
    *inside the function*, so blocking only during a reload proves nothing --
    the finder is long gone by the time the call happens. CI caught exactly that:
    the test passed here because vtracer is not installed on this machine, and
    panicked on a runner where it is.
    """
    finders = [Absent(n) for n in names]
    removed = {}
    for name in names:
        for key in [k for k in list(sys.modules)
                    if k == name or k.startswith(name + ".")]:
            removed[key] = sys.modules.pop(key)
    sys.meta_path[:0] = finders
    try:
        yield
    finally:
        for finder in finders:
            sys.meta_path.remove(finder)
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(removed)


def reload_without(module_name, missing):
    """Re-import `module_name` as if `missing` were not installed.

    The caller must reload it again afterwards, or every later test in the
    process sees the crippled version. `Restoring` does that.
    """
    module = importlib.import_module(module_name)   # so reload has a target
    with absent(missing):
        return importlib.reload(module)


class Restoring(unittest.TestCase):
    """Reloads leak. Put every module back before the next test runs."""

    MODULES = ()

    def tearDown(self):
        for name in self.MODULES:
            if name in sys.modules:
                importlib.reload(sys.modules[name])


class TestWithoutWebsocketClient(Restoring):
    MODULES = ("devgraphics.backends.fooocus", "devgraphics.backends.comfyui")

    def test_fooocus_still_imports_and_describes_itself(self):
        fooocus = reload_without("devgraphics.backends.fooocus", "websocket")
        backend = fooocus.FooocusBackend()
        caps = backend.capabilities
        self.assertEqual(caps.name, "fooocus")
        self.assertTrue(caps.seed)
        self.assertTrue(caps.notes, "describe must still work without transport")

    def test_fooocus_says_what_to_install_when_it_tries_to_connect(self):
        fooocus = reload_without("devgraphics.backends.fooocus", "websocket")
        with self.assertRaises(MissingDependency) as caught:
            fooocus.create_connection("ws://127.0.0.1:7865/queue/join")
        message = str(caught.exception)
        self.assertIn("websocket-client", message)
        self.assertIn("devgraphics[local]", message)
        self.assertIn("fooocus", message)

    def test_comfyui_still_imports_and_describes_itself(self):
        comfyui = reload_without("devgraphics.backends.comfyui", "websocket")
        caps = comfyui.ComfyUIBackend().capabilities
        self.assertEqual(caps.name, "comfyui")

    def test_comfyui_says_what_to_install_when_it_tries_to_connect(self):
        comfyui = reload_without("devgraphics.backends.comfyui", "websocket")
        with self.assertRaises(MissingDependency) as caught:
            comfyui.create_connection("ws://127.0.0.1:8188/ws")
        self.assertIn("devgraphics[local]", str(caught.exception))

    def test_the_exception_names_stay_usable_in_except_clauses(self):
        """comfyui catches two websocket exceptions by name.

        Without stand-ins, the except clause raises NameError while handling
        something else -- the worst possible way to report a missing package.
        """
        comfyui = reload_without("devgraphics.backends.comfyui", "websocket")
        self.assertTrue(issubclass(comfyui.WebSocketException, Exception))
        self.assertTrue(issubclass(comfyui.WebSocketTimeoutException,
                                   comfyui.WebSocketException))
        try:
            raise comfyui.WebSocketTimeoutException("timed out")
        except comfyui.WebSocketException:
            pass

    def test_generate_refuses_before_it_touches_the_network(self):
        """The deterministic failure has to beat the transient one.

        generate() fetches /config over HTTP before it ever opens a socket, so
        without an up-front guard a missing package surfaces as "cannot reach
        127.0.0.1:7865: connection refused" and sends someone debugging a server
        that was never the problem. CI found this on a real core-only install.
        """
        from devgraphics.backends.base import Request

        for module, name in (("devgraphics.backends.fooocus", "fooocus"),
                             ("devgraphics.backends.comfyui", "comfyui")):
            with self.subTest(backend=name):
                crippled = reload_without(module, "websocket")
                backend = getattr(crippled, "FooocusBackend"
                                  if name == "fooocus" else "ComfyUIBackend")()

                def explode(*_a, **_k):
                    raise AssertionError("%s reached the network before "
                                         "reporting the missing package" % name)

                with mock.patch.object(crippled, "request_json", explode):
                    with self.assertRaises(MissingDependency) as caught:
                        backend.generate(Request(prompt="a flame", seed=1))
                self.assertIn("devgraphics[local]", str(caught.exception))

    def test_probe_reports_the_missing_package_rather_than_the_host(self):
        """Answering "up" on a machine that cannot open a socket is the wrong
        kind of true."""
        for module, name in (("devgraphics.backends.fooocus", "fooocus"),
                             ("devgraphics.backends.comfyui", "comfyui")):
            with self.subTest(backend=name):
                crippled = reload_without(module, "websocket")
                cls = getattr(crippled, "FooocusBackend"
                              if name == "fooocus" else "ComfyUIBackend")
                ok, message = cls.probe()
                self.assertFalse(ok)
                self.assertIn("websocket-client", message)
                self.assertIn("devgraphics[local]", message)

    def test_a_hosted_backend_is_unaffected(self):
        """The whole reason for the split: no socket, no dependency."""
        reload_without("devgraphics.backends.fooocus", "websocket")
        openai = reload_without("devgraphics.backends.openai_images", "websocket")
        self.assertTrue(openai.OpenAIBackend().capabilities.transparent)


class TestWithoutVtracer(Restoring):
    MODULES = ("devgraphics.vectorize",)

    def test_importing_devgraphics_does_not_need_the_rust_wheel(self):
        vectorize = reload_without("devgraphics.vectorize", "vtracer")
        self.assertIn("flat", vectorize.PRESETS)

    def test_to_svg_says_what_to_install(self):
        """The block has to hold across the CALL: the import is inside to_svg."""
        vectorize = importlib.import_module("devgraphics.vectorize")
        with absent("vtracer"):
            with self.assertRaises(ImportError) as caught:
                vectorize.to_svg("in.png", "out.svg")
        message = str(caught.exception)
        self.assertIn("devgraphics[svg]", message)
        self.assertIn("--svg", message)

    def test_a_bad_preset_is_still_caught_before_the_import(self):
        """Argument validation must not hide behind a missing dependency.

        Deliberately checked with vtracer present as well as absent: a caller who
        typoed a preset deserves the same error either way, and it must arrive
        before anything touches the filesystem.
        """
        vectorize = importlib.import_module("devgraphics.vectorize")
        with self.assertRaises(ValueError):
            vectorize.to_svg("in.png", "out.svg", preset="nonsense")
        with absent("vtracer"):
            with self.assertRaises(ValueError):
                vectorize.to_svg("in.png", "out.svg", preset="nonsense")


class TestWithoutTomllib(Restoring):
    """The 3.9 and 3.10 path, which never runs on a modern interpreter."""

    MODULES = ("devgraphics.config",)

    @unittest.skipUnless(HAVE_TOMLI, "tomli only installs below 3.11, by design")
    def test_it_falls_back_to_the_tomli_backport(self):
        """Only meaningful where the backport is actually present.

        On 3.11+ pyproject deliberately does not install tomli, so blocking
        tomllib there leaves no parser at all -- which is the *next* test, not a
        failure of this one. CI caught this: it passed here only because this
        machine happens to have tomli sitting in its environment.
        """
        config = reload_without("devgraphics.config", "tomllib")
        self.assertIsNotNone(
            config.tomllib,
            "with tomllib absent and tomli installed, config must fall back")
        self.assertEqual(config.tomllib.__name__, "tomli")

    def test_with_neither_parser_the_error_says_what_to_install(self):
        module = importlib.import_module("devgraphics.config")
        with absent("tomllib", "tomli"):
            config = importlib.reload(module)
            self.assertIsNone(config.tomllib)
            with self.assertRaises(config.ConfigError) as caught:
                config.load("devgraphics.toml")
            self.assertIn("tomli", str(caught.exception))


class TestDeclaredExtras(unittest.TestCase):
    """Guard the split itself.

    A dependency drifting back into `dependencies` would undo all of the above
    silently, and nothing else in the suite would notice.
    """

    def setUp(self):
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib
        with open("pyproject.toml", "rb") as handle:
            self.project = tomllib.load(handle)["project"]

    def _names(self, requirements):
        out = set()
        for line in requirements:
            out.add(line.split(";")[0].strip().split(">")[0].split("=")[0]
                    .split("<")[0].split("[")[0].strip().lower())
        return out

    def test_core_is_pillow_and_a_toml_parser_only(self):
        self.assertEqual(self._names(self.project["dependencies"]),
                         {"pillow", "tomli"})

    def test_the_transport_and_the_tracer_are_extras(self):
        extras = self.project["optional-dependencies"]
        self.assertIn("websocket-client", self._names(extras["local"]))
        self.assertIn("vtracer", self._names(extras["svg"]))

    def test_all_covers_every_optional_runtime_dependency(self):
        extras = self.project["optional-dependencies"]
        union = self._names(extras["local"]) | self._names(extras["svg"])
        self.assertEqual(self._names(extras["all"]), union)

    def test_dev_can_run_the_whole_suite(self):
        """CI installs [dev]; it has to pull in what the tests import."""
        dev = self._names(self.project["optional-dependencies"]["dev"])
        self.assertTrue({"pytest", "websocket-client", "vtracer"} <= dev)

    def test_the_backport_is_conditional(self):
        line = [d for d in self.project["dependencies"]
                if d.lower().startswith("tomli")][0]
        self.assertIn("python_version", line,
                      "tomli must not install on 3.11+, where tomllib is stdlib")


if __name__ == "__main__":
    unittest.main()
