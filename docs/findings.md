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

### Second machine, 2026-08-20: RTX 5070 Ti, Fooocus 2.5.5, torch 2.11.0+cu128

| Stage | Cost |
| --- | --- |
| First icon of a run | 104 s |
| Every icon after it | 23–27 s |
| Re-run of a finished set (all cached) | 0.87 s |

The first-icon penalty is checkpoint loading, and the rest of the gap is Fooocus
reloading it **every render**. Sampling `nvidia-smi` at 0.5 s through one render:

```text
17.4s  util 43%  vram 8227 MiB  power  59 W    <- loading
19.2s  util 96%  vram 9375 MiB  power 200 W    <- sampling starts
23.4s  util 99%  vram 9375 MiB  power 257 W
26.4s  util 99%  vram 9375 MiB  power 251 W
28.2s  util 53%  vram 5490 MiB  power  64 W    <- unloaded again
```

Only ~8 s of a ~25 s render is diffusion; mean utilisation across the whole thing
is 16.5%. That is why the GPU looks idle if you glance at a monitor — and on
Windows, Task Manager's GPU graph defaults to the **3D** engine, while PyTorch
work appears under **Compute_0**.

The reload is a default, not a limit. `args_manager.py` sets
`always_offload_from_vram = not args.disable_offload_from_vram`, so offloading is
**on** unless you turn it off, and with it on `free_memory()` skips its
`if get_free_memory(device) > memory_required: break` early exit and evicts
unconditionally. Launching with `--disable-offload-from-vram` keeps the eviction
logic but only runs it under real pressure. Peak measured VRAM was 9,375 MiB of
16,303, so a 16 GB card has room to spare.

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

### The rule is narrower than "things vs symbols"

A second set, 2026-08-20, seven pictorial subjects on the same scaffold and seed.
All seven are *things*, so the original rule predicted seven successes. Four:

| Subject | Prompt fragment | Result |
| --- | --- | --- |
| fire | `a flame` | Good |
| target | `a bullseye target with an arrow in it` | Good |
| chart | `a bar chart with three ascending bars` | Good (bars descend) |
| magnifier | `a magnifying glass` | Good |
| sparkle | `a four-pointed sparkle star burst` | **Drifted** — generic five-pointed star |
| link | `two interlocking chain links` | **Wrong** — symmetrical abstract blob |
| quill | `a hand holding a pen, writing` | **Wrong** — unreadable hand |

Two failure modes the first set never probed, both worth stating separately from
the glyph rule:

- **Hands.** Diffusion's oldest weakness, and an emoji set walks straight into it
  — ✍ 👋 🤝 👍 are all common. Treat any hand as hand-authored.
- **Relational geometry.** `link` is a *thing*, but the thing is a topological
  claim: two objects passing through each other. The model draws one compact
  object well and a relationship between two objects badly.

`sparkle` is a third, milder case: the model has a strong prior for "star" that
overrides a specific point count. Expect count and arrangement to be ignored.

Revised rule: **generate single compact objects.** Hand-author glyphs, hands, and
anything whose meaning is a relationship rather than a shape.

The background-share floor caught one of these on its own — `link` came back at
47% and `target` at 34%, both under the 60% floor, flagged as "subject fills the
frame" with no human looking. It did **not** catch `quill` at 82%, which is the
limit restated: the audit sees colour and morphology, never semantics.

## The glyph rule is SDXL's, not diffusion's

Measured 2026-08-20 against `openai` / `gpt-image-1-mini`, low quality, $0.005 an
image. `examples/glyph-probe.json` -- the eight symbols this repo recorded SDXL
failing -- run twice.

| Subject | SDXL (Fooocus) | gpt-image-1-mini, no anchor | gpt-image-1-mini, `--anchor check` |
| --- | --- | --- | --- |
| check, bolt, arrow, cross, plus, minus, bullet, chevron | 0/8 | **8/8 correct** | **0/8 -- all eight came back check marks** |

Two findings, and the second matters more than the first.

**Hosted models do fix abstract glyphs.** Every symbol SDXL could not draw across
six retries and three style combinations came back correct and on-style, for four
cents. "Generate things, hand-author symbols" is a fact about SDXL, not about
diffusion, and the README should no longer imply otherwise. Hand-authored SVG is
still smaller and still inherits `currentColor`; that argument survives on its own
merits, and the accuracy argument does not.

**The anchor destroys subject fidelity on weak subjects.** The same eight
subjects, same model, same scaffold, with `--anchor check` added: all eight
rendered as check marks. The reference carried the silhouette, not just the
palette. It is not a general failure -- an earlier run with `--anchor fire`
produced a correct check mark and correct interlocking links -- so the rule is
about *semantic distance*: a flame anchoring a check mark leaves room to disagree,
a check mark anchoring seven other abstract glyphs does not.

That is the documented risk made concrete. docs/backends.md already warns that
"pushing reference conditioning hard homogenises silhouettes as well as style";
this is what that looks like when it happens.

**The audit called the broken run perfect.** All eight anchored icons scored
within tolerance, background shares 70-79%, zero flags -- because they *were*
consistent. They were consistently the wrong object. This is the stated limit of
`consistency.py` reproduced exactly: it sees colour and morphology, never
semantics. Nothing except looking at the contact sheet would have caught it.

Practical rules that follow:

- On a hosted model, try without `--anchor` first. A shared scaffold alone held
  eight icons together well enough to read as a set.
- Reserve `--anchor` for pictorial subjects with strong, distinct silhouettes.
- Look at the sheet. The numbers cannot do this one.

One threshold note: `minus` and `bullet` came back at 56% and 55% background
share, under the 60% floor, and both are correct. The floor is tuned for a centred
SDXL render with margin; a glyph that legitimately fills its frame trips it.
Treat sub-60% as "look at this", which is what it says, not as "wrong".

### The same model traces terribly

Ten icons, `gpt-image-1-mini`, no anchor: 10/10 subjects correct, including every
one SDXL failed -- check, cross, warning, interlocking links, a genuinely
four-pointed sparkle, and a hand holding a pen. Consistent palette and outline
with no seed and no anchor, from the shared scaffold alone. $0.05 and 2.6 minutes.

Then vtracer, `flat` preset, same settings for both sets:

| | paths | size |
| --- | --- | --- |
| Fooocus `chart.svg` | 8 | 27 KB |
| Fooocus `fire.svg` | 6 | 27 KB |
| gpt-image-1-mini `chart.svg` | **1174** | **200 KB** |
| gpt-image-1-mini `check.svg` | **650** | **266 KB** |

Two orders of magnitude, and the cause is visible on the contact sheet: the hosted
model renders **gradient** fills. Every band in a gradient becomes its own traced
region. The scaffold does say "solid flat colour fill", but this model has no
negative prompt to reinforce it with, and it read "sticker" as "shaded sticker".

So the two backends are good at opposite halves of the job:

| | subject accuracy | traces to SVG |
| --- | --- | --- |
| SDXL / Fooocus | things only | **excellently** -- 5-11 paths |
| gpt-image-1-mini | **everything tried** | badly -- 600-1200 paths |

Practical: take PNGs from a hosted model and SVGs from a local one, or push the
scaffold much harder on flatness before tracing hosted output. And this restores
the case for `devgraphics glyphs` on its own terms -- a hand-authored check mark
is ~1 KB against 266 KB traced, and it inherits `currentColor`. That argument was
never about accuracy; now that accuracy is solved, it is the only argument left,
and it is a strong one.

### background=transparent is real

Recorded as UNVERIFIED on forum evidence alone. Measured: the raw bytes from
`gpt-image-1-mini` with `background=transparent` and `output_format=png` carry a
genuine alpha channel -- `postprocess.has_alpha()` returns True before any cutout
runs, and `render_bytes()` skips the flood fill entirely. It works headlessly, no
prompt reinforcement needed.

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

## A stalled Fooocus is invisible without the estimation frame

Measured 2026-08-20 on an install that had been up four hours and had stopped
servicing its queue. Every client blocked in `recv()` with no output — the current
one and the 0.1 original, extracted from git and run against the same server to
tell "our bug" from "their bug" apart. They behaved identically, which is the
answer: the client was fine.

Tracing the exchange frame by frame showed Gradio answering correctly the whole
time:

```text
[0.0s] websocket open
[0.0s] <- send_hash {}
[0.0s] -> send_hash fn_index=67
[0.0s] <- estimation {'rank': 2, 'queue_size': 3, 'rank_eta': 26.9}
```

`estimation` carries the queue position and arrives the moment Gradio enqueues
you. Dropping it makes a wedged server and a hung client indistinguishable.
The client now reports it, and only when `rank > 0`: rank 0 is every healthy
render, and both `get_task` and `generate_clicked` get their own frame, so
reporting it unconditionally printed four irrelevant lines per two icons.

Diagnosing this from outside, in order: `nvidia-smi` for GPU load, working-set
size for whether a checkpoint is resident (39 MB means it is not; ~1.6 GB means it
is), and the process's CPU delta over a few seconds. A Fooocus that answers
`/config` but shows 0.00 CPU is wedged, and only a restart clears it.
