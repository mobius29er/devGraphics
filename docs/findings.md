# Findings

Measured against a stock Fooocus install (Gradio 3.41.2, `juggernautXL_v8Rundiffusion`,
Speed performance, 1024×1024) while building an icon set for a real site. Recorded
so the next set doesn't relearn it.

## Timing

| Stage | Cost |
| --- | --- |
| Generation, 1024×1024, Speed | 13–15 s |
| Background cutout (PIL flood fill, 1 MP) | ~2.7 s |
| Vectorise (vtracer, `flat`) | ~0.1 s |

An 88-icon set is therefore ~25 minutes of wall clock, not a day. Batches are
resumable — existing outputs are skipped — so an interrupted run costs nothing.

## Subject accuracy is the binding constraint

Six diverse subjects, identical scaffold, fixed seed 77777:

| Subject | Prompt fragment | Result |
| --- | --- | --- |
| flame | `a flame` | Good |
| target | `a bullseye target with an arrow in it` | Good |
| chart | `a bar chart with three ascending bars` | Good |
| magnifier | `a magnifying glass` | Good |
| check mark | `a check mark tick` | **Wrong** — rounded blob |
| lightning bolt | `a lightning bolt` | **Wrong** — arrow/cursor |

Retried the two failures across three style combinations (`Fooocus V2 + Sticker
Designs`, `Simple Vector Art`, both together). **All six retries failed**, each
producing the same vague striped rounded-rectangle. This is not a prompt-tuning
problem; SDXL has no reliable prior for simple abstract symbols.

Practical rule: **generate things, hand-author symbols.** Check marks, X marks,
arrows, bolts, plus/minus and bullets should be SVG written by hand. They are also
typically the highest-frequency icons in a UI — in the set that prompted this
work, the check mark alone accounted for 43 of 470 emoji uses — so the failing
subset is small by variety and large by volume.

## Style selection is counter-intuitive

| Styles | Background | Verdict |
| --- | --- | --- |
| `Fooocus V2` + `Fooocus Sharp` | Near-flat dark charcoal | **Best.** Keys out cleanly. |
| `Fooocus V2` + `Sticker Designs` | Heavy asphalt texture | Defeats flood fill |
| `Simple Vector Art` | Heavy asphalt texture | Defeats flood fill |
| `Sticker Designs` + `Simple Vector Art` | Heavy asphalt texture | Defeats flood fill |

The styles whose names promise flat vector output deliver the opposite. Negative
prompting `texture, vignette` did not rescue them.

Consistency levers that did work: one shared prompt scaffold, one shared style
list, and a fixed seed across all subjects. SDXL keeps a recognisable look across
different subjects at a shared seed.

## Background removal

Flood-fill from the border, don't threshold globally. An icon's own dark
outline strokes sit close to a dark backdrop, so a global colour-distance key eats
them; flood fill only removes the *connected* outer region, leaving interior darks
intact.

Flood fill alone is not enough. Speckle whose contrast against the backdrop
exceeds the threshold survives it, and seeding from the whole border rather than
the corners does **not** help — measured background share moved by less than a
point (fire 73.0% → 73.1%, bolt 85.0% → 85.2%) while the grey smudges remained.
Raising the threshold trades those smudges for eaten outline strokes.

What works is discarding disconnected blobs: label the connected components of the
opaque mask and keep only those at least ~15% the size of the largest. Measured
runs dropped 664–1627 fragments per image and cleared the smudges completely.
Label on a downscaled mask (256x256) — a megapixel flood in pure Python is far too
slow, and that resolution is ample to separate an icon from speckle.

Typical background share: 73–87% of pixels for a centred icon. A figure far below
that range (the bullseye measured 34%) means the subject fills the frame, which is
worth flagging — it usually indicates the model ignored "single centered object".

## Vectorising

vtracer presets on one traced icon:

| Preset | Size | Paths |
| --- | --- | --- |
| `fine` | 385.5 KB | 670 |
| `smooth` | 49.4 KB | 34 |
| `flat` | 29.0 KB | 18 |

Trace from the **full-resolution** cutout, not a downscaled PNG — the tracer wants
the edge detail. Below 64 px there is no visible difference between `smooth` and
`flat`, so `flat` is the default.

Caveat worth stating plainly: traced SVG is ~29 KB against ~1 KB for a
hand-authored icon, and its many-coloured paths cannot inherit `currentColor`.
Resolution independence is real; CSS theming is not.

## Driving stock Fooocus

- `/config` reports **77 dependencies and zero `api_name` values** — `gradio_client`
  cannot bind to any of them. The raw queue protocol is the only route.
- Generation is two chained dependencies, not one: `get_task` (153 inputs → state)
  then `generate_clicked` (state → 4 outputs). Calling the first alone silently
  produces nothing, which is exactly what an early version of this client did.
- `gr.State` is server-side, keyed by `session_hash`. Pass `null`; reuse the hash.
- 152 of 153 inputs have usable defaults in `/config`. Only the `gr.State` lacks one.
- Outputs are `gr.update(...)` wrappers: the gallery is `output[3]["value"]`.
- Aspect-ratio choices carry HTML (`1024×1024 <span ...> ∣ 1:1</span>`), so match on
  a substring rather than equality.
