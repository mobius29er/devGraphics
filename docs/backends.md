# Choosing a backend

devGraphics generates a *set*, not an image. Everything below is organised around
the one question that matters for a set: what holds it together, and what does the
backend you picked take away.

Researched 2026-08-20 against first-party documentation. Cells marked
**UNVERIFIED** could not be confirmed against a primary source and are flagged in
the code as well as here. Two things moved *during* the research window, which is
the best argument there is for reading the date on this page: Imagen 4 shut down
on 2026-08-17, and `gpt-image-2` **removed** the transparency support
`gpt-image-1` has.

## Capability matrix

| Backend | seed | negative | native alpha | style ref | exact size | cost / image | setup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fooocus` | yes | yes | no | not wired | 1024² only | free | high — GPU, ~7 GB checkpoint |
| `comfyui` | yes | yes | yes, with a 444 MB matting model | ControlNet core, IP-Adapter is third-party | any, ×8 | free | medium — needs an API-format graph (we ship one) |
| `invokeai` | yes | yes | no (3–4 extra nodes) | needs adapter models | any, ×8 | free | very high — hand-built graph, API is unsupported by its maintainers |
| `a1111` | yes | yes | no | img2img | any | free | low — launch with `--api` |
| `openai` | **no** | **no** | `gpt-image-1`/`1.5`/`1-mini` only | 16 images via edits | 3 fixed sizes | $0.005–$0.25 | low |
| `gemini` | **no** | **no** | **no, in any model** | ≤3, on one model only | ratio × tier | $0.03–$0.24 | low |
| `openai-compatible` (Grok, Together, DeepInfra, …) | per-endpoint | per-endpoint | no | no | per-endpoint | varies | low |

Verified negatives, recorded so nobody re-derives them:

- **Anthropic / Claude has no image generation.** Not an endpoint, not a tool, not
  a beta. The vision docs say it verbatim: Claude "cannot generate, produce, edit,
  manipulate, or create images." It is not in the backend list and never will be.
  See [Glyphs](#glyphs-the-thing-claude-is-actually-for) for what it *is* good for
  here.
- **Imagen 4** (`imagen-4.0-*:predict`) passed its shutdown date on 2026-08-17.
  The backend rejects an `imagen-*` model id with that date rather than 404ing.
- **xAI** `grok-2-image-1212` was deprecated 2026-02-28 and `grok-imagine-image-pro`
  retired 2026-05-15. Two retirements in six months is why xAI is a `base_url`
  swap on `openai-compatible` rather than a module of its own.
- **Fireworks, LM Studio and vLLM do not serve `/v1/images/generations`.** A
  generic OpenAI-compatible client cannot reach them, whatever their chat API does.
- **Ollama** image generation is experimental and macOS-only, and is absent from
  its own OpenAI-compatibility page. **UNVERIFIED**; not advertised.

## What actually holds a set together

Three levers, in descending order of how well they work.

**A fixed seed, on a pinned checkpoint.** This is the measured, working strategy
and it is why local backends are the default. One prompt scaffold, one style list,
one seed across every subject; SDXL keeps a recognisable look across different
subjects at a shared seed. Available on `fooocus`, `comfyui`, `invokeai`, `a1111`.

Caveats worth knowing before you trust it: ComfyUI generates noise on CPU so a
seed is GPU-independent, but cross-machine *pixel* identity still is not
guaranteed. ComfyUI and InvokeAI both cache aggressively — a byte-identical graph
produces no new render and hands back the original filename in milliseconds. A1111
makes no cross-platform determinism promise at all, which is why its capabilities
report `seed=True, deterministic=False`; those are two different promises and the
code keeps them apart.

**An anchor image as a style reference.** The replacement when there is no seed.
Render one hero icon, then condition every other icon on it: `--anchor fire`. On
`openai` that is the edits endpoint with the anchor inlined; on `gemini` it is the
dedicated style-reference slot, which exists on **one** model — defaulting to the
cheap Lite variant for cost silently discards the entire mechanism, so the backend
says so out loud.

Two honest limits. There is no vendor-neutral reference *weight*, so this gives
materially less control than a seed. And pushing reference conditioning hard
homogenises silhouettes along with style.

**The prompt scaffold.** The floor, and the only lever that works everywhere. Make
it carry design-system vocabulary no model can invent: an explicit hex palette, a
uniform stroke weight, a grid, and the backdrop colour the cutout will key out.
Material specifies a 24 dp canvas with uniform 2 dp strokes; IBM Carbon a 32×32
artboard with 2 px padding and 2 px strokes. Those numbers belong in your scaffold.

**Seeds do not port across backends.** Seed 77777 on SDXL and seed 77777 on a
hosted API have nothing to do with each other. Generate one set on one backend;
the lockfile stops you doing otherwise.

## What devGraphics guarantees regardless of backend

This is the actual product. It runs after generation and works identically
everywhere.

1. **`postprocess.render()` normalisation** — framing, scale, centring, alpha.
   Only one hosted API returns real alpha and no local SDXL install returns any;
   keying the backdrop out is not a workaround for SDXL, it is the equaliser that
   makes seven backends emit the same artefact.
2. **The lockfile drift gate** — `assets/devgraphics.lock.json` records
   `{backend, model, seed, scaffold digest}` and compares on re-run. Finished icons
   are skipped, so the realistic failure is not "the seed got dropped", it is "40
   icons were made on Fooocus in January and the next 48 on OpenAI in March". No
   capability check catches that. This does.
3. **The numeric drift audit** (`--audit`) — six PIL-only features per icon, gated
   on a modified z-score against the set's own median. ~7 ms per icon, so ~0.6 s
   for a set against ~25 minutes of generation.
4. **`--snap-palette`** — quantise every icon onto your declared palette. Opt-in:
   hard quantisation without dither bands antialiased edges.

**The audit sees colour and morphology, not semantics.** A perfectly on-palette,
on-stroke render of the wrong object scores clean. It catches style drift, which is
what this tool promises. It does not catch subject failure, which
[findings.md](findings.md) identifies as the binding constraint.

## Cost

For the 88-icon set in [findings.md](findings.md):

| Backend | 88 icons | |
| --- | --- | --- |
| any local backend | **$0.00** | ~25 min; ~7 GB checkpoint, 10–16 GB VRAM |
| `openai` `gpt-image-1-mini` low | $0.44 | cheapest transparent-capable path |
| `openai` `gpt-image-1.5` low / med / high | $0.79 / $2.99 / $11.70 | 1024² is the only square size |
| `openai-compatible` → Grok | $1.76 – $4.40 | no seed, no negative, no alpha |
| `gemini` `3.1-flash-lite-image` | ~$2.96 | **no style-reference slot** — the lever is gone |
| `gemini` `3.1-flash-image` @ 1K | ~$5.90 | + non-disableable thinking tokens, billed |

Multipliers the headline numbers hide: `-n 3` means 264 billed generations, not 88.
Reference-conditioned calls cost more than plain ones. OpenAI's lowest tier is a
handful of images per minute, so 88 icons is a long stretch of pure throttling —
`--dry-run` estimates it. Gemini has no free tier for any image model.

`--dry-run` prints the estimate and generates nothing. `--max-spend` is a hard stop.

## Glyphs: the thing Claude is actually for

[findings.md](findings.md) measured SDXL failing on check marks and lightning bolts
across six retries and three style combinations — the same striped blob every time.
That subset is small by variety and large by usage: in the set that prompted this
work, the check mark alone was 43 of 470 emoji uses.

`devgraphics glyphs` asks Claude to write the SVG source directly. Hand-authored
SVG is ~1 KB and inherits `currentColor`; traced SVG is ~29 KB and cannot. It is
not a backend and is not in `--backend`, because Claude generates no images.

Claude-authored SVG is **not reproducible** — there is no seed, and sampling
parameters are rejected on current models. Generate once, review, commit to git.
The command refuses to overwrite an existing glyph without `--force` for exactly
that reason.

**Do the newer hosted models fix abstract glyphs? Unknown — we do not claim they
do.** No benchmark isolates abstract-symbol fidelity from typographic text
rendering, and text rendering demonstrably improved while symbol shape went
unmeasured. Instead of guessing, `examples/glyph-probe.json` holds the eight
symbols this repo measured failing; run it against any backend in about 90 seconds
and record the answer in findings.md the way everything else here was recorded.
Until then the documented default stands: **generate things, hand-author symbols.**

## Deferred, and why

**Stability AI** is the best hosted fit on paper — the only one where a real
`seed` *and* a real `negative_prompt` both survive — and is blocked on
documentation, not effort. Its docs render client-side and `/openapi.json` 404s, so
the `style_preset` enum, the response field names and the credit costs are all
secondary-source. It also requires `multipart/form-data` even with no file
attached. Cheap to add the day someone can `curl` it once.

**Replicate and fal** are ~50 lines each and structurally awkward: the `input`
object is per-model, so an SDXL wrapper and a FLUX wrapper on the same platform
disagree about whether `negative_prompt` exists. A "consistent set" promise breaks
the moment the model changes, so they need a per-model capability map — a second
copy of the thing `base.Capabilities` exists to avoid.

**`diffusers`** is an in-process library, not an HTTP surface, and drags multi-GB
CUDA-variant torch wheels into a project whose selling point is three pure-Python
dependencies behaving identically on three OSes. An optional extra with its own
code path, or not at all.

## Writing your own backend

Implement `capabilities` and `generate(request)`; nothing needs to import
devGraphics. Point `--backend` at a dotted path — `--backend mypkg.thing:MyBackend`
— and no packaging is involved at all. To publish one, register it under the
`devgraphics.backends` entry-point group.

The contract, the rules a constructor must obey, and why `check()` is not an
`isinstance` test against a runtime-checkable Protocol are all in
[`devgraphics/backends/base.py`](../devgraphics/backends/base.py).

## Security

Local backends have **no authentication**. Stock ComfyUI has none at all, and
anything that can reach the port can queue arbitrary graphs and read files under
its output, input and temp directories. A1111 has none unless launched with
`--api-auth`. InvokeAI has none in a default single-user install. If you point
`--host` at anything other than loopback, network protection is your
responsibility.

API keys are read from environment variables and never from the config file. A
value that looks like a live key anywhere in `devgraphics.toml` is a hard error.
There is no keyring integration: it is a native dependency, and `chmod 0600` is a
no-op on Windows, so a "secure" store would be a false promise inside the exact
OS-parity story this project sells.
