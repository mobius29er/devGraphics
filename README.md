# devGraphics

Production-ready graphics from a diffusion model you already run locally, at zero
marginal cost.

If you have Fooocus (or any SDXL UI) on your machine, you have an image generator.
What you don't have is a way to turn "I need 88 consistent icons" into 88 files
your build can actually use. devGraphics is that layer: headless generation,
background removal, trimming, resizing, and vectorising — scriptable, repeatable,
no API keys, no per-image billing.

**Status: early.** The Fooocus backend works end to end. Everything here is
documented against measured behaviour, including the parts that don't work.

## Why this exists

Local image generation is a solved commodity — ComfyUI, A1111, InvokeAI and
`diffusers` all do it well. The gap is everything *after* the render:

- **Consistency across a set.** Independently-styled assets read as AI slop. A
  coherent set reads as a design system.
- **Alpha.** SDXL emits no transparency. Icons need it.
- **Resolution independence.** Raster icons don't survive a redesign.
- **Codebase integration.** Assets need names, sizes, formats, and a place to go.

That pipeline is the product. The generator underneath is interchangeable.

## Install

```sh
pip install -e .
```

Requires a running Fooocus instance. Stock Fooocus is fine — no API fork needed.

```sh
python entry_with_update.py --listen
```

Pure Python plus one Rust wheel (`vtracer`); no shell-outs and no platform paths,
so Windows, macOS and Linux behave identically.

## Use

```python
from devgraphics import generate, contact_sheet, to_svg

icons = generate(
    {"fire": "a flame", "trophy": "a trophy cup", "rocket": "a rocket ship"},
    outdir="assets",
    size=128,
)
contact_sheet(icons, "assets/sheet.png")      # eyeball drift on your real background
to_svg("assets/raw/fire.png", "assets/fire.svg", preset="flat")
```

Roughly 13–15 s per 1024×1024 image on a mid-range GPU, plus ~3 s of
post-processing. Finished icons are skipped on re-run, so a batch is resumable.

## How the Fooocus backend works

Stock Fooocus publishes **zero named Gradio endpoints**, so `gradio_client` has
nothing to bind to. This drives the raw queue protocol instead. Generation is two
chained dependencies sharing one `session_hash`:

```
[67] get_task(153 controls) -> gr.State
[68] generate_clicked(gr.State) -> html, preview, progress_gallery, gallery
```

`gr.State` never crosses the wire — Gradio holds it server-side keyed by
`session_hash` — so we pass `null` for state inputs and reuse one hash across both
calls. The other 152 controls are filled from the defaults published in `/config`,
leaving only the handful worth overriding. Output arrives as `gr.update(...)`
wrappers, so the gallery is at `output[3]["value"]`, not `output[3]`.

Those `fn_index` values are pinned to a Fooocus layout and are validated at
construction; a mismatched build raises rather than generating garbage.

## What works, and what doesn't

Measured, not assumed. Full detail in [docs/findings.md](docs/findings.md).

| Subject type | Result |
|---|---|
| Pictorial objects — flame, target, chart, magnifier, trophy, rocket | **Good.** Clean, on-style, usable. |
| Abstract glyphs — check mark, lightning bolt, arrow, X | **Fails.** Generic sticker blobs, not the symbol. |

This is the single most important thing to know before planning a set. SDXL draws
*things* well and *symbols* badly. Budget hand-authored SVG for the glyph subset;
it is usually the smaller half by count and the larger half by usage.

Style choice matters more than expected. `Fooocus V2 + Fooocus Sharp` with an
explicit palette produced flat, clean backgrounds. `Sticker Designs` and
`Simple Vector Art` — despite sounding ideal — produced heavy asphalt textures
that defeat background removal.

## Roadmap

- Image Prompt / PyraCanny reference images to lock style across a set
- SVG path simplification (traced output is ~29 KB vs ~1 KB hand-authored)
- ComfyUI and `diffusers` backends
- Manifest-driven sets and codebase integration

## License

MIT
