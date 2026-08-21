"""
Command line entry point.

Four verbs, and the shape of them is dictated by one thing: some backends cost
money and the old CLI had no concept of that. `gen --dry-run` prints the whole
plan and the bill and generates nothing; a paid run asks before spending unless
told not to. Nothing here can quietly turn 88 icons into $12.

Backend-specific settings all go through one repeatable `-O key=value` rather than
a flag per backend. ComfyUI needs a workflow path and node ids, InvokeAI a queue
id, OpenAI a quality tier, Fooocus a performance mode; a flag for each would make
this module know every backend's schema, which is the opposite of pluggable.

`devgraphics manifest.json ...` still means `devgraphics gen manifest.json ...`,
because that spelling is in the 0.1 README.
"""

import argparse
import json
import os
import sys

from . import config, consistency, iconset, keys, lockfile, pricing
from .backends import base

VERBS = ("gen", "backends", "init", "audit", "glyphs")

#: argparse dest -> config key. Only these reach config.resolve(); handing over
#: the whole namespace would push --dry-run and --yes into the profile, into the
#: digest, and back out as spurious drift on the next run.
OVERRIDES = ("backend", "model", "seed", "anchor", "n", "render",
             "size", "svg", "sheet", "host")


def option(text):
    """`-O key=value`. Coerce the obvious scalars; everything else stays text."""
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected key=value, got %r" % text)
    key, raw = text.split("=", 1)
    low = raw.strip().lower()
    if low in ("true", "false"):
        return key.strip(), low == "true"
    for cast in (int, float):
        try:
            return key.strip(), cast(raw)
        except ValueError:
            pass
    return key.strip(), raw


def build_parser():
    p = argparse.ArgumentParser(
        prog="devgraphics",
        description="Generate a consistent asset set from a manifest, on the "
                    "image generator of your choice.")
    p.add_argument("--version", action="version",
                   version="devgraphics %s" % _version())

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", metavar="PATH",
                        help="devgraphics.toml (default: nearest one above cwd)")
    common.add_argument("--env-file", metavar="PATH", action="append",
                        help="load KEY=VALUE lines; the real environment still "
                             "wins. Repeatable, never automatic.")
    sub = p.add_subparsers(dest="verb")

    g = sub.add_parser("gen", parents=[common], help="generate the set")
    g.add_argument("manifest", nargs="?",
                   help='JSON: {"slug": "prompt fragment"}. Default: the config '
                        "`manifest` key.")
    g.add_argument("-o", "--outdir")
    g.add_argument("-p", "--profile", help="style profile from the config")
    g.add_argument("--backend", help="name, entry point, or module:Class")
    g.add_argument("--model", help="checkpoint or hosted model id")
    g.add_argument("-O", "--opt", type=option, action="append", default=[],
                   metavar="KEY=VALUE",
                   help="backend option; repeatable. See `devgraphics backends "
                        "NAME --describe`.")
    g.add_argument("--host", help="host:port for a local backend")
    g.add_argument("--seed", type=int)
    g.add_argument("--render", help="generation size, WxH (the tracer wants full res)")
    g.add_argument("--size", type=int, help="finished PNG edge length")
    g.add_argument("--svg", help="also trace each icon to SVG (vtracer preset)")
    g.add_argument("--sheet", action="store_true", default=None,
                   help="write a contact sheet for drift review")
    g.add_argument("--anchor", metavar="SLUG",
                   help="render this icon first and pass it as a style reference "
                        "to every other icon, on backends that take one")
    g.add_argument("-n", type=int, dest="n",
                   help="candidates per icon; the closest to the anchor is kept. "
                        "Multiplies hosted cost by exactly n.")
    g.add_argument("--api-key-env", metavar="NAME",
                   help="environment variable holding the key (never the key)")
    g.add_argument("--dry-run", action="store_true",
                   help="print the plan and the estimate; generate nothing")
    g.add_argument("--max-spend", type=float, metavar="USD",
                   help="refuse to start if the estimate exceeds this")
    g.add_argument("-y", "--yes", action="store_true",
                   help="skip the paid-run confirmation")
    g.add_argument("--allow-drift", action="store_true",
                   help="proceed despite a fatal waiver or a lockfile mismatch")
    g.add_argument("--no-lock", action="store_true",
                   help="do not write devgraphics.lock.json")
    g.add_argument("--force", action="store_true",
                   help="regenerate assets that already exist")
    g.add_argument("--only", metavar="SLUG[,SLUG...]",
                   help="re-render just these subjects, whether or not they "
                        "exist. The iteration loop: fix one bad prompt fragment "
                        "without re-running the set.")
    g.add_argument("--audit", action="store_true",
                   help="run the numeric drift audit after generating")

    b = sub.add_parser("backends", parents=[common],
                       help="list backends, or probe one")
    b.add_argument("name", nargs="?")
    b.add_argument("-O", "--opt", type=option, action="append", default=[],
                   metavar="KEY=VALUE")
    b.add_argument("--probe", action="store_true",
                   help="check reachability and credentials; never generates")
    b.add_argument("--describe", action="store_true",
                   help="print the backend's -O options and capabilities")

    i = sub.add_parser("init", parents=[common],
                       help="write a starter devgraphics.toml")
    i.add_argument("--backend", default="fooocus")
    i.add_argument("-o", "--outdir", default=".", help="where to write it")
    i.add_argument("--force", action="store_true")

    a = sub.add_parser("audit", parents=[common],
                       help="score a finished set for style drift")
    a.add_argument("outdir", nargs="?", default="assets")

    y = sub.add_parser("glyphs", parents=[common],
                       help="author SVG for the symbols diffusion models cannot draw")
    y.add_argument("manifest", help='JSON: {"slug": "what it is"}')
    y.add_argument("-o", "--outdir", default="assets")
    y.add_argument("--style", help="glyph grid: %s" % ", ".join(_glyph_styles()))
    y.add_argument("--model")
    y.add_argument("--palette", action="append",
                   help="hex colour; default is currentColor, which is themeable")
    y.add_argument("--force", action="store_true",
                   help="overwrite existing glyphs (they are not reproducible)")
    y.add_argument("--probe", action="store_true",
                   help="check credentials and model id only")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # 0.1 compat: `devgraphics manifest.json ...` means `devgraphics gen ...`.
    # Only mis-fires if a flag VALUE equals a verb, e.g. `--model gen`.
    if argv and not argv[0].startswith("-") and argv[0] not in VERBS:
        argv.insert(0, "gen")

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verb is None:
        parser.print_help()
        return 2

    for path in getattr(args, "env_file", None) or []:
        keys.load_env_file(path)

    handler = {"gen": cmd_gen, "backends": cmd_backends, "init": cmd_init,
               "audit": cmd_audit, "glyphs": cmd_glyphs}[args.verb]
    try:
        return handler(args)
    except (config.ConfigError, iconset.SetError, base.BackendError,
            base.BackendNotFound, base.UnsupportedOption, FileNotFoundError,
            ImportError, TypeError, ValueError) as exc:
        # A bad option is a user mistake, not a crash. UnsupportedOption is a
        # ValueError and TypeError is what base.load() raises for a constructor
        # that rejected an -O key, so both arrive here rather than as a traceback.
        print("\n%s" % exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; finished icons are kept and the next run resumes",
              file=sys.stderr)
        return 130


# --- gen ----------------------------------------------------------------

def cmd_gen(args):
    path, cfg = config.discover(args.config)
    profile = config.resolve(cfg, args.profile, _overrides(args))
    profile["_name"] = args.profile or cfg.get("default_profile")
    profile["_prices"] = config.prices(cfg)

    manifest = args.manifest or cfg.get("manifest")
    if not manifest:
        raise config.ConfigError(
            "no manifest given and no `manifest` key in the config"
            + (" (%s)" % path if path else "; run `devgraphics init`"))
    subjects = _manifest(manifest)
    outdir = args.outdir or cfg.get("outdir") or "assets"

    # Built before anything is printed so a bad -O key fails immediately rather
    # than after a plan block that implies the run is about to work. Constructors
    # touch no network, so this costs nothing even with the server down.
    backend = iconset.build(profile)

    only = tuple(s.strip() for s in (args.only or "").split(",") if s.strip())
    work = iconset.plan(subjects, outdir, profile, force=args.force, only=only)
    print(_plan_block(profile, work, outdir))

    if args.dry_run:
        caps = backend.capabilities
        request, waived = iconset.waivers(caps, profile,
                                          strict=len(work["todo"]) > 1)
        print()
        print(base.report(caps, waived, request))
        print("\nnothing generated (--dry-run)")
        return 0

    if not _affordable(work, args):
        return 1

    made = iconset.generate(subjects, outdir, profile=profile,
                            force=args.force, allow_drift=args.allow_drift,
                            write_lock=not args.no_lock, only=only)

    svg = profile["output"].get("svg")
    if svg:
        # Tracing runs after the expensive part, so a missing vtracer wheel must
        # not throw away a batch that already cost money and 25 minutes. Warn,
        # keep the PNGs, and let the user re-trace later for free.
        try:
            from .vectorize import to_svg
            for slug in made:
                raw = os.path.join(outdir, "raw", slug + ".png")
                if os.path.exists(raw):
                    to_svg(raw, os.path.join(outdir, slug + ".svg"), preset=svg)
        except (ImportError, ValueError) as exc:
            print("\nSVG tracing skipped; the PNGs are written and unaffected.\n"
                  "%s" % exc, file=sys.stderr)

    if profile["output"].get("sheet") and made:
        iconset.contact_sheet(made, os.path.join(outdir, "sheet.png"))

    if args.audit and made:
        print()
        iconset.audit(made, outdir=outdir)

    missing = [s for s in subjects if s not in made and s not in work["hand"]]
    print("\n%d/%d generated -> %s" % (len(made), len(subjects), outdir))
    if work["hand"]:
        print("hand-authored, never regenerated: %s" % ", ".join(work["hand"]))
    if missing:
        print("missing: %s" % ", ".join(missing))
    return 0 if not missing else 1


def _plan_block(profile, work, outdir):
    lines = ["backend   %s%s" % (profile.get("backend"),
                                 "/" + profile["model"] if profile.get("model") else "")]
    if work["anchor"]:
        lines.append("anchor    %s (rendered first, then used as a style reference)"
                     % work["anchor"])
    lines.append("plan      %d cached, %d hand-authored, %d to generate%s"
                 % (len(work["cached"]), len(work["hand"]), len(work["todo"]),
                    " x n=%d" % work["n"] if work["n"] > 1 else ""))
    advisory = iconset.advisory_model(profile)
    if advisory:
        lines.append("          model recorded but not applied -- %s takes no "
                     "model over the wire" % profile.get("backend"))
    if profile.get("backend") in pricing.LOCAL:
        lines.append("estimate  free (local backend)")
    elif work["cost"] is None:
        lines.append("estimate  unknown -- no published price for this model")
    else:
        lines.append("estimate  %d calls, %s" % (work["calls"],
                                                 pricing.money(work["cost"])))
        lines.append("          %s" % work["cost_note"])
    lines.append("outdir    %s" % outdir)
    return "\n".join(lines)


def _affordable(work, args):
    """Nothing spends money without either a flag or an answered question."""
    cost = work["cost"]
    if not cost:
        return True
    if args.max_spend is not None and cost > args.max_spend:
        print("\nestimate %s exceeds --max-spend %s; nothing generated"
              % (pricing.money(cost), pricing.money(args.max_spend)),
              file=sys.stderr)
        return False
    if args.yes:
        return True
    if not sys.stdin.isatty():
        print("\nthis run costs about %s and stdin is not a terminal, so the "
              "confirmation cannot be asked.\nre-run with --yes, or --dry-run to "
              "see the plan." % pricing.money(cost), file=sys.stderr)
        return False
    answer = input("\nthis run costs about %s. continue? [y/N] "
                   % pricing.money(cost))
    return answer.strip().lower() in ("y", "yes")


# --- backends -----------------------------------------------------------

def cmd_backends(args):
    if args.name:
        return _one_backend(args)
    for name in base.available():
        print("%-18s %s" % (name, base.BUILTIN.get(name, "third-party")))
    print("\n`devgraphics backends NAME --describe` for options and capabilities,")
    print("`--probe` to check reachability. A dotted module:Class path also works.")
    return 0


def _one_backend(args):
    options = dict(args.opt or [])
    try:
        backend = base.load(args.name, **options)
    except Exception as exc:
        print("%-10s error  %s" % (args.name, exc), file=sys.stderr)
        return 1

    caps = backend.capabilities
    if args.describe or not args.probe:
        print(_describe(args.name, backend, caps))
    if args.probe:
        probe = getattr(backend, "probe", None)
        if probe is None:
            print("%-10s no probe; nothing to check without generating" % args.name)
            return 0
        ok, message = probe(**options)
        print("%-10s %-5s %s" % (args.name, "up" if ok else "down", message))
        return 0 if ok else 1
    return 0


def _describe(name, backend, caps):
    lines = ["%s" % caps.name,
             "  seed              %s%s" % (_yn(caps.seed),
                                           "" if not caps.seed else
                                           ", deterministic" if caps.deterministic
                                           else ", not deterministic across hosts"),
             "  negative prompt   %s" % _yn(caps.negative_prompt),
             "  native alpha      %s" % _yn(caps.transparent),
             "  style references  %s" % (caps.reference_images or "no"),
             "  batch in one call %s" % _yn(caps.batch),
             "  sizes             %s" % (", ".join("%dx%d" % s for s in caps.sizes)
                                         if caps.sizes else "any"),
             "  cost per image    %s" % (pricing.money(caps.cost_per_image)
                                         if caps.cost_per_image else "free")]
    module = sys.modules[type(backend).__module__]
    accepted = getattr(module, "OPTIONS", None)
    if accepted:
        lines.append("  -O options        %s" % ", ".join(sorted(accepted)))
    for note in caps.notes:
        lines.append("  note  %s" % note)
    return "\n".join(lines)


def _yn(value):
    return "yes" if value else "no"


# --- init / audit / glyphs ----------------------------------------------

def cmd_init(args):
    dest = os.path.join(args.outdir, "devgraphics.toml")
    if os.path.exists(dest) and not args.force:
        print("%s already exists; pass --force to overwrite" % dest,
              file=sys.stderr)
        return 1
    os.makedirs(args.outdir, exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(config.starter_toml(args.backend))
    print("wrote %s" % dest)
    print("edit the scaffold and palette, then: devgraphics gen")
    return 0


def cmd_audit(args):
    icons = os.path.join(args.outdir, "icons")
    if not os.path.isdir(icons):
        raise FileNotFoundError("no icons in %s -- generate a set first" % icons)
    paths = dict((os.path.splitext(f)[0], os.path.join(icons, f))
                 for f in sorted(os.listdir(icons)) if f.endswith(".png"))
    if not paths:
        raise FileNotFoundError("no PNGs in %s" % icons)

    print(consistency.report(
        consistency.audit(paths, bg_shares=iconset.shares(args.outdir))))
    return 0


def cmd_glyphs(args):
    from . import glyphs

    if args.probe:
        ok, message = glyphs.probe(model=args.model or glyphs.MODEL)
        print(message)
        return 0 if ok else 1

    subjects = _manifest(args.manifest)
    options = {}
    if args.palette:
        options["palette"] = args.palette
    made = glyphs.author(subjects, style=args.style, model=args.model or glyphs.MODEL,
                         outdir=os.path.join(args.outdir, "icons"),
                         force=args.force, **options)
    print("\n%d/%d authored -> %s" % (len(made), len(subjects), args.outdir))
    print("SVG from a language model is not reproducible: there is no seed and "
          "sampling parameters are rejected.\nReview these, commit them, and treat "
          "regeneration as a deliberate act (--force).")
    return 0 if len(made) == len(subjects) else 1


# --- shared -------------------------------------------------------------

def _overrides(args):
    over = dict((name, getattr(args, name, None)) for name in OVERRIDES)
    options = dict(args.opt or [])
    if getattr(args, "api_key_env", None):
        options["api_key_env"] = args.api_key_env
    if options:
        over["options"] = options
    return over


def _manifest(path):
    try:
        with open(path, encoding="utf-8") as handle:
            subjects = json.load(handle)
    except OSError as exc:
        raise FileNotFoundError("could not read manifest: %s" % exc)
    except ValueError as exc:
        raise config.ConfigError("%s is not valid JSON: %s" % (path, exc))
    if not isinstance(subjects, dict) or not subjects:
        raise config.ConfigError(
            "%s must be a non-empty JSON object of slug -> prompt fragment" % path)
    return subjects


def _glyph_styles():
    from . import glyphs
    return sorted(glyphs.STYLES)


def _version():
    from . import __version__
    return __version__


if __name__ == "__main__":
    sys.exit(main())
