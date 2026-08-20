# devGraphics

A consistent asset set for your app, from the image generator of your choice.

Local image generation is a solved commodity. What you don't have is a way to turn
"I need 88 consistent icons" into 88 files your build can actually use — with
alpha, at the right size, named after your components, and still looking like each
other six months later. devGraphics is that layer.

Point it at a local GPU or a hosted API. The pipeline after the render is the same
either way, and that pipeline is the product.

**Status: early.** Everything here is documented against measured behaviour,
including the parts that don't work. Seven backends ship; only the Fooocus path has
been run against real hardware for a full set.

## Install

```sh
pip install -e .
```

Three dependencies, all pure Python except one Rust wheel for tracing, and adding
six backends added none of them. Every hosted backend is stdlib `urllib` and
`base64`. Windows, macOS and Linux behave identically.

## Use

```sh
devgraphics init                 # write a commented devgraphics.toml
devgraphics gen                  # generate the set
devgraphics gen --audit          # ...and score it for style drift
```

Or from Python:

```python
from devgraphics import generate, contact_sheet, to_svg

icons = generate({"fire": "a flame", "trophy": "a trophy cup"}, outdir="assets")
contact_sheet(icons, "assets/sheet.png")
to_svg("assets/raw/fire.png", "assets/fire.svg", preset="flat")
```

Finished icons are skipped on re-run, so an interrupted batch costs nothing to
resume.

## Backends

```sh
devgraphics backends                      # what's available
devgraphics backends comfyui --describe   # options, capabilities, caveats
devgraphics backends openai --probe       # reachable? credentials set?
devgraphics gen --backend comfyui
```

| | seed | negative | alpha | style ref | cost / image |
| --- | --- | --- | --- | --- | --- |
| `fooocus` | yes | yes | no | no | free |
| `comfyui` | yes | yes | yes¹ | yes | free |
| `invokeai` | yes | yes | no | yes | free |
| `a1111` | yes | yes | no | yes | free |
| `openai` | **no** | **no** | yes² | 16 | $0.005–$0.25 |
| `gemini` | **no** | **no** | **no** | ≤3³ | $0.03–$0.24 |
| `openai-compatible` | varies | varies | no | no | varies |

¹ with a 444 MB matting model installed  ² `gpt-image-1`/`1.5`/`1-mini`; `gpt-image-2`
**removed** it  ³ on one model only — the cheap Lite variant has no style-reference slot

`openai-compatible` is the "any endpoint" answer: `--backend openai-compatible -O
preset=grok`, or point `base_url` anywhere that speaks `POST /v1/images/generations`.
Writing your own takes two methods and no packaging: `--backend mypkg.thing:MyClass`.

Full matrix, cost tables, security notes and what got deferred:
[docs/backends.md](docs/backends.md).

**Claude is not on that list, because Anthropic has no image generation API** — not
an endpoint, not a beta. It has a different job here; see below.

## Consistency is the whole point

88 independently styled icons read as AI slop. One coherent set reads as a design
system. Three levers do that work, and which ones you get depends on the backend:

1. **A fixed seed on a pinned checkpoint.** Measured, works, local backends only.
2. **An anchor image as a style reference.** What replaces the seed when there
   isn't one: `--anchor fire` renders your hero icon first and conditions every
   other icon on it.
3. **The prompt scaffold.** The floor, and the only lever that works everywhere.

devGraphics refuses to generate a multi-icon set with none of the three. A seed sent
to a backend that has no seed parameter is a hard error, not a silent drop — that
one failure quietly destroys everything the tool promises. `--allow-drift`
overrides it.

```text
$ devgraphics gen -p brand-icons-hosted --dry-run
backend   openai/gpt-image-1.5
anchor    fire (rendered first, then used as a style reference)
plan      0 cached, 0 hand-authored, 12 to generate x n=3
estimate  36 calls, $0.324

backend: openai/gpt-image-1.5
  warn  seed=77777: dropped -- no seed parameter, so every icon becomes an
        independent draw; the --anchor reference carries the style instead
  note  background=transparent is documented, but the only evidence it yields
        real alpha headlessly is one forum thread -- UNVERIFIED. Every render
        is checked and a miss falls back to the flood-fill cutout

nothing generated (--dry-run)
```

Nothing spends money without `--dry-run` first or an answered prompt. `--max-spend`
is a hard stop.

### Two guards, because skipping makes drift invisible

**The lockfile.** `assets/devgraphics.lock.json` records the backend, model, seed
and scaffold that produced a set. The realistic failure isn't "the seed got
dropped", it's "40 icons on Fooocus in January and the next 48 on OpenAI in March".
No capability check catches that. This does.

**The audit.** `devgraphics audit` scores six PIL-only features per icon against
the set's own median and flags the outliers. ~7 ms an icon. It sees colour and
morphology, **not semantics** — a perfectly on-palette render of the wrong object
scores clean.

## What works, and what doesn't

Measured, not assumed. Full detail in [docs/findings.md](docs/findings.md).

| Subject type | Result |
| --- | --- |
| Pictorial objects — flame, target, chart, magnifier, trophy, rocket | **Good.** |
| Abstract glyphs — check mark, lightning bolt, arrow, X | **Fails.** Generic blobs. |

SDXL draws *things* well and *symbols* badly. Six retries across three style
combinations all produced the same striped blob. That subset is small by variety
and large by usage — in the set that prompted this work, the check mark alone was
43 of 470 emoji uses.

So `devgraphics glyphs` asks Claude to write the SVG source directly. Hand-authored
SVG is ~1 KB and inherits `currentColor`; traced SVG is ~29 KB and cannot. It is
not reproducible — there's no seed and sampling parameters are rejected — so it
refuses to overwrite an existing glyph without `--force`. Generate once, review,
commit.

**Do the newer hosted models fix abstract glyphs? Unknown, and we don't claim they
do.** No benchmark isolates symbol fidelity from text rendering.
`examples/glyph-probe.json` holds the eight symbols this repo measured failing —
run it against any backend in 90 seconds and record the answer.

Until then: **generate things, hand-author symbols.**

Style choice also matters more than expected. `Fooocus V2 + Fooocus Sharp` produced
flat, clean backgrounds. `Sticker Designs` and `Simple Vector Art` — despite
sounding ideal — produced heavy asphalt textures that defeat background removal.

## Roadmap

- Stability AI, once its docs are fetchable enough to verify against
- SVG path simplification (traced output is ~29 KB vs ~1 KB hand-authored)
- Measuring the abstract-glyph probe across every backend

## License

Apache License 2.0 — see [LICENSE](LICENSE). It carries an express patent grant
with defensive termination, which MIT does not, and section 5 licenses inbound
contributions under the same terms so third-party backends need no CLA.

If you redistribute devGraphics, section 4 asks three things: ship the LICENSE,
mark any files you changed, and reproduce [NOTICE](NOTICE) wherever your product
already lists third-party attributions.

devGraphics talks to Fooocus, ComfyUI (GPL-3.0) and AUTOMATIC1111 (AGPL-3.0) over
HTTP only. No code from any of them is linked or vendored here, so none of their
copyleft terms reach this source — and nothing from those repositories should ever
be pasted into it.
