"""Command line entry point: generate an icon set from a JSON manifest."""

import argparse
import json
import os
import sys

from .iconset import contact_sheet, generate
from .vectorize import PRESETS, to_svg


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="devgraphics",
        description="Generate a consistent icon set from a local Fooocus install.",
    )
    p.add_argument("manifest", help='JSON file: {"slug": "prompt fragment", ...}')
    p.add_argument("-o", "--outdir", default="assets", help="output directory")
    p.add_argument("--size", type=int, default=128, help="PNG edge length")
    p.add_argument("--seed", type=int, default=77_777, help="fixed seed holds style across subjects")
    p.add_argument("--host", default="127.0.0.1:7865", help="Fooocus host:port")
    p.add_argument("--svg", choices=sorted(PRESETS), help="also trace each icon to SVG")
    p.add_argument("--sheet", action="store_true", help="write a contact sheet for drift review")
    args = p.parse_args(argv)

    try:
        with open(args.manifest, encoding="utf-8") as f:
            subjects = json.load(f)
    except (OSError, ValueError) as exc:
        p.error("could not read manifest: %s" % exc)

    if not isinstance(subjects, dict) or not subjects:
        p.error("manifest must be a non-empty JSON object of slug -> prompt fragment")

    made = generate(subjects, args.outdir, size=args.size, seed=args.seed, host=args.host)

    if args.svg:
        for slug in made:
            raw = os.path.join(args.outdir, "raw", slug + ".png")
            if os.path.exists(raw):
                to_svg(raw, os.path.join(args.outdir, slug + ".svg"), preset=args.svg)

    if args.sheet and made:
        contact_sheet(made, os.path.join(args.outdir, "sheet.png"))

    missing = [s for s in subjects if s not in made]
    print("\n%d/%d generated -> %s" % (len(made), len(subjects), args.outdir))
    if missing:
        print("missing: %s" % ", ".join(missing))
    return 0 if len(made) == len(subjects) else 1


if __name__ == "__main__":
    sys.exit(main())
