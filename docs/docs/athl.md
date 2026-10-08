# athenea light groups: the `.lights.usda` sidecar and the `.athl` format

A car's lamps over its relit cloud: switched, dimmed, colour-shifted and
sequenced at run time, with no re-bake and no rays. This page covers the
design (athenea's proposals 062, 063, 066, 067, 069 and 076; plan-athenea.md
phase 6), the two sidecar files, the WebGPU plugin that reads them, and what
athenea has to bake before the real Corvette can be lit this way.

| | Code |
|---|---|
| `.lights.usda` reader (USD text subset) | `src/webgpu/athenea/usda.ts` |
| Sidecar model, rules, curves, `w_k(t)` | `src/webgpu/athenea/lightSidecar.ts` (`LightRig`) |
| `.athl` writer and reader, sparse layers, form factor | `rust/spark-lib/src/athl.rs` |
| WASM decode | `rust/spark-rs/src/lib.rs` (`decode_athl`, `athl_header`, `decode_athl_layer`) |
| `.athl` in the browser, GPU packing, page ranges | `src/webgpu/athenea/athl.ts` (`AthlStore`) |
| The shader (an `ISplatColour`) | `slang/athenea_adapter/lights.slang` |
| The plugin | `src/webgpu/athenea/lightsPlugin.ts` (`atheneaLightsPlugin`) |
| Synthetic asset tool | `rust/build-lod/src/bin/athenea-lights.rs` |
| Example | `examples/webgpu/athenea-lights.html` |

## The idea

Light is linear (062 §1). If each group of lamps `k` (low beam, DRL, tail
lamps...) has its contribution to every splat baked once, per unit of the
lamp's radiance, then every state of the car is a weighted sum:

```
colour = TX_base + Σ_k w_k · (direct_k · profile_k + layer_k)
```

- `TX_base`: the cloud relit by the dome with **every lamp off**
  (`atheneaRelightPlugin`, phase 4), or the raster's light.
- `w_k`: group `k`'s weight this frame, linear RGB: its level from the
  state, the rules and the lamp's time curve, times its radiance and colour.
  Computed on the CPU once a frame; it is the only thing that changes when a
  lamp is switched (062 §7.1: one weight moves every term at once).
- `direct_k`: the light from the group's emitter polygons, computed at run
  time (060): LTC over each polygon. Today the cosine lobe only (diffuse:
  Lambert's closed form, the polygon clipped to the horizon), times the
  splat's albedo. Unshadowed.
- `profile_k`: what the lens does to the light leaving the lamp (076): a tint
  (a) and, for headlamps, a beam map (b), read in the direction from the
  lamp to the splat.
- `layer_k`: the baked part, sparse over the splats the group reaches: the
  light sent on after bouncing (indirect), the emitter's own glow (emission)
  and a view-dependent reflected field.

## The two files

066 ships a sidecar beside the cloud, loadable apart from it; a cloud
without one is a normal cloud with its lamps off.

```
corvette.athc            the cloud, TX baked with every lamp off
corvette.lights.usda     groups, states, rules, sequences, curves (text, editable)
corvette.lights.athl     per group: polygons, lens profile, sparse layers (binary, GPU)
```

### `.lights.usda`

Plain USD text. Until athenea's `AtheneaLightGroupAPI` schema exists, the
data are namespaced attributes on `Scope` prims, as 067's draft writes them
(that draft, verbatim, is `test/fixtures/athenea-lights/corvette-067.lights.usda`
and reads as written). A prim with the API applied reads the same. The
reader takes one layer as written: it does not compose (`subLayers`,
references and variants are kept as metadata, an `over` reads like a `def`).

| Prim | Attribute | |
|---|---|---|
| group | `rel athenea:lightGroup:members` | source prims (traceability) |
| | `int2[] athenea:lightGroup:splatRanges` | the emitting splats, `[lo, hi)` |
| | `token athenea:lightGroup:function` | `lowBeam`, `highBeam`, `daytimeRunning`, `tail`, `indicator`... (an indicator blinks) |
| | `token athenea:lightGroup:side` | `left`, `right` |
| | `token athenea:lightGroup:technology` | `led` (default), `halogen`, `xenon`, `blinker`, `lightGuide` |
| | `float athenea:lightGroup:radiance` | the lamp on, in nits (063); default 1 |
| | `color3f athenea:lightGroup:color` | default (1, 1, 1) |
| | `float athenea:lightGroup:temperatureK` | blackbody colour (athenea's Krystek fit, luminance 1), times `color` |
| | `float athenea:lightGroup:startTemperatureK` | xenon's cold start (4300 K), halogen's red glow (1000 K) |
| | `float athenea:lightGroup:riseSeconds`, `fallSeconds` | LED ramp; halogen time constant; xenon warm-up |
| | `float athenea:lightGroup:delaySeconds` | lag after a change (sequential segments) |
| | `float athenea:lightGroup:blinkHz`, `dutyCycle` | blinker, default 1.5 Hz (90 a minute, UN R48) and 0.5 |
| | `color3f athenea:lightGroup:lensTint` | 076 (a), over the `.athl`'s |
| | `bool athenea:lightGroup:initialOn` | |
| | `asset shaping:ies:file` | 076's author profile (read; not evaluated yet) |
| state | `dictionary athenea:lightState:targets` | `{ double <group> = level }`; levels above 1 for a shared lamp (brake = 4 × tail) |
| | `token athenea:lightState:base` | a state its targets start from |
| rule | `token athenea:lightRule:when` | condition: group names (on = target > 0), `state:<name>`, `!`, `&&`, `\|\|`, parentheses |
| | `token` / `token[] athenea:lightRule:target` | the groups it acts on |
| | `token athenea:lightRule:mode` | `scale` (default: × `scale`), `set`, `max`, `sync` (blink in phase with the first group of `when`) |
| | `double athenea:lightRule:scale` | |
| sequence | `double[] athenea:lightSequence:times`, `token[] ...:states` | `(time, state)` keys |
| | `bool ...:loop`, `double ...:duration` | |
| | `double athenea:lightSequence:level:<group>.timeSamples` | a keyframed curve, over the state's target |
| sidecar | `string athenea:lightSidecar:cloudHash` | the cloud it was made for (see Validity) |
| | `float athenea:lightSidecar:nitsPerUnit` | nits per scene unit of light (default 1) |
| | `asset athenea:lightSidecar:athl`, `token ...:defaultState` | |

Evaluation (`LightRig.evaluate(t)`): the targets come from the state (or the
sequence's state at that time, then its curves), then manual levels, then
the rules in prim order, each condition read on the targets as the earlier
rules left them. Each group then moves from where it was to its target
along its lamp's curve, from the time its target changed:

| Lamp | Up | Down | Colour |
|---|---|---|---|
| LED, light guide | linear ramp over `riseSeconds` | over `fallSeconds` | constant |
| halogen | first order, τ = `riseSeconds` | τ = `fallSeconds` | from `startTemperatureK` with the fourth root of the output (a heuristic; 062 marks the timing unverified) |
| xenon | strikes at 30 %, smoothstep to 100 % over `riseSeconds` | linear | 4300 K → `temperatureK` over the warm-up |
| blinker | square wave `blinkHz`, `dutyCycle`, phase from switch-on (or `sync`) | | |

`w_k(t) = level_k(t) · radiance_k / nitsPerUnit · colour_k(t)`.

### `.athl`, version 1

Little-endian; sections 16-byte aligned. Written and read by
`rust/spark-lib/src/athl.rs` (`AthlFile`), round-trip byte-exact.

**Index space.** The cloud's page-aligned virtual order (`athc.rs`
`VirtualTree::of_file(file, true)`, the pager's): the LoD's merged nodes
from 0, the splats from `splatBase` (a multiple of 65 536). Chunks of 65 536
are the pager's pages. A merged node's layer value is the weighted mean of
the splats under it, weighted as athenea's LoD merges colours (opacity × the
area of the two longest axes; `virtual_values`). A whole-file decode (merged
nodes then splats, back to back) is mapped by the shader with `merged` and
`splatBase`; a source in plain order (no LoD) with `merged = splatBase = 0`.

**Header** (128 bytes)

| Offset | Type | |
|---|---|---|
| 0 | `u32` | magic `"ATHL"` |
| 4 | `u32` | version (1) |
| 8 | `u32` | flags (0) |
| 12 | `u32` | groups (≤ 16) |
| 16 | `u32` | elementCount = splatBase + splatCount |
| 20 | `u32` | merged |
| 24 | `u32` | splatBase |
| 28 | `u32` | splatCount |
| 32 | `u32` | chunkSplats (65 536) |
| 36 | `u32` | blockSplats (256) |
| 40 | `u32` | sections |
| 48 | `u64` | section table offset (128) |
| 56 | `u64` | cloudHash |
| 64 | `u64` | bakeHash |

**Section table**, 40 bytes an entry: `u32 tag`, `u16 group`, `u16 kind`,
`u32 chunk`, `u32 count`, `u64 offset`, `u64 bytes`, 8 reserved. Sections in
order: `GRPS`, `POLY`, the `PROF`s, then the `LAYR`s **sorted by chunk, group,
kind**, so the layers of one page, every group, are one byte range (one Range
request; `athlChunkRange`). Unknown tags are skipped.

| Section | Content |
|---|---|
| `GRPS` | per group, 128 bytes: name (32, NUL-padded, as the Blender light group and the `.usda` prim), `u32 flags` (1: polygons two-sided), `u32 polygonFirst`, `u32 polygonCount`, `i32 profile` (−1: none), `f32[3] tint` (076 (a)), `f32[3] origin`, `f32[3]` axes x, y, z (the lamp's frame in object space; z is the beam), `f32 radiance` (what the layers were baked for; 1 = per unit) |
| `POLY` | per polygon, 128 bytes: `u32 group`, `u32 vertexCount` (3–8), `f32[3] radiance` (relative, within the group), pad, 8 × `f32[3]` vertices (object space; the front face is counter-clockwise) |
| `PROF` | 48-byte head: `u32 width`, `u32 height`, `f32 lon0, lon1, lat0, lat1` (radians: longitude from +z towards +x, latitude towards +y, in the lamp's frame), `f32[3] outside`; then `width × height` texels of RGB f16, row 0 at `lat0`. 076's headlamp map is 128 × 64 over ±40° × (−10°, +5°) |
| `LAYR` | one group's layer of one kind over one chunk: `u32 components`, `u32 blocks`, `u32 group \| kind << 16`, `u32 chunk`; `u16` block ids (increasing, < 256; padded to 4 bytes); then the blocks dense, `blocks × 256 × components` f16 |

Layer kinds (all per unit of the group's radiance, so that `w_k` is the
lamp's real radiance in every term, 062 §7.1.2):

| Kind | | Components |
|---|---|---|
| 0 indirect | radiance each splat sends on from the group after ≥ 1 bounce (062's layer); with no polygons, all of the group's light | 3 |
| 1 emission | the emitter splats' own glow, behind their lens | 3 |
| 2 field | view-dependent light leaving the splat: 9 real harmonics (degree 2) × RGB, coefficient-major, of the direction from the eye to the splat, in Spark's basis and signs (`0.282095`, `−0.488603 y`, `0.488603 z`, `−0.488603 x`, ...) | 27 |

A field here is outgoing radiance, the product of the bake, not athenea's
TX reflected-field coupling (48 values against the sky): a per-group bake
gives the lit result directly, and the web needs only to add it.

**Sparsity.** A block of 256 splats is kept when any of its values passes the
bake's threshold (062 §3: e.g. 0.5 % of the group's maximum). In the
virtual order a block is Morton-contiguous, so a lamp's reach is a few
blocks. The synthetic car: 13 749 splats + 5 585 merged nodes, 76 blocks a
chunk pair; each group's indirect layer keeps 42–52 of them.

**Validity** (066 §2.4, §5.3). `cloudHash` is FNV-1a 64 of the cloud's first
4096 bytes (its header page: counts, bounds, table offsets; for v3 its header
and directory), cheap to check in a browser that has fetched them
(`athcCloudHash`). The `.usda` repeats it. `setLights(asset, athl,
{ cloudHash })` refuses a sidecar whose hashes disagree, with a warning. The
header page does not see a re-bake with the same structure: `bakeHash`
(free-form) is for that.

### On the GPU

`AthlStore.pack()` lays a `.athl` out as one storage buffer
(`atheneaLightData`): a header, a 32-word record per group (with, per kind, a
table of chunks), per chunk a 256-entry directory of blocks, the blocks as
f16 pairs, the polygons and the profiles (texels padded to two words). A
lookup is three loads: chunk table, block directory, value. Paging keeps
only the chunks held: `openAthl(urlRange(url))` reads the header, groups,
polygons and profiles (one range) and returns an `AthlPager`, which fetches
a chunk's layers (one range, every group) when the cloud's pages want it and
drops them when the page leaves; the packed buffer is rebuilt on change. For
a paged cloud the uniform `pageChunk` maps each pool page to its `.athl`
chunk (`athlPagerChunks(pager, splats)`); a whole cloud with an `AthlPager`
fetches every chunk.

The shader is generate's `ISplatColour` after the relight (preset
`athenea-relight-lights`) or the raster (`athenea-raster-lights`), or alone
(`athenea-lights`). It binds two storage buffers (the `.athl`, the
attribute pool for normals) and a uniform block with the 16 weights. With
the relight's, generate binds 8 storage buffers.

## The API

```js
import { atheneaRelightPlugin, atheneaLightsPlugin, atheneaOutputPlugin,
         athcCloudHash, decodeAthl } from "@sparkjsdev/spark/webgpu";

const lights = atheneaLightsPlugin({ sidecar: usdaText });
lights.setLights(car, await decodeAthl(athlBytes), { cloudHash: athcCloudHash(athcFirst4k) });
// or paged with the cloud (SplatMesh({ paged: true })):
lights.setLights(car, await openAthl(urlRange(athlUrl)), {
  cloudHash, pageChunks: () => athlPagerChunks(spark.webgpu.lod.pager, car.paged) });
host.register(atheneaRelightPlugin({ hdri })).register(lights).register(atheneaOutputPlugin());

lights.setLightState("noche_ciudad");   // 066's named states
lights.playSequence("bienvenida");      // the sequence's timeline, on the frame clock
lights.setGroupLevel("pilotos", 4);     // a hand override (null: back to the state)
lights.setTerms({ field: false });      // terms on and off (debug)
lights.master = 0.5;                    // a dimmer over every group
lights.lightGroups;                     // [{ name, level, target, weight, baked }]
lights.ui;                              // ControlSpecs: state, sequence, a slider a group, terms
```

A frame whose weights did not change is not regenerated (`onFrame` returns
"clean"). Commands are stamped with the frame's clock, so a curve starts
from the frame the command came in.

## Validation

`test/gpu/atheneaLights.test.ts` (Dawn):

- the shader against the CPU (the same form factor, profile lookup and layer
  reads) at two splats: within half-float rounding;
- **linearity** (068 V5): image(A) + image(B) − image(off) = image(A + B),
  and image(2A) − image(off) = 2 (image(A) − image(off)), to 2e-3 in an
  rgba16float target;
- **isolation** (068 V4 in spirit): switching a group changes only the
  pixels of the splats it reaches;
- after the relight plugin (on a 10-storage-buffer device): the groups'
  light adds to the relit colour;
- the synthetic asset through the real WASM (`atheneaLightsAsset.test.ts`):
  the three files agree (names, hashes), the layers are sparse, one page's
  byte range decodes alone, and an `AthlPager` fetches exactly the ranges of
  the chunks asked for and drops the rest.

`test/unit/atheneaLights.test.ts`: 067's draft read as written; states,
rules, LED ramp, blackbody colour; blinkers at 90 a minute, repeaters in
phase, DRL dimmed on the indicator's side; xenon and halogen warm-up;
sequence with a keyframed curve and loop. `rust/spark-lib athl` tests:
round trip, block sparsity, form factor against numerical integration.

068 V4 proper (each group against Cycles' `Combined_<group>` and athenea's
path tracer per group) waits for athenea's bakes.

## What athenea must produce for the real Corvette

The web side reads; athenea bakes. For `corvette.lights.athl`:

1. **The base TX with every lamp off** (062 §8): the `.athc` the dome relights.
   The emitters' materials at emission 0, as the asset already has them (067 §1).
2. **Per-group bakes (proposal 069).** A group id (0–15) on every light and
   emissive material (from the `.lights.usda` members or Blender's
   `object.lightgroup`), and the integrator summing each contribution into
   its group's buffer as well as the total. For the cloud: per splat of the
   `.athc`, per group, the RGB radiance leaving the splat from that group,
   **per unit of the group's radiance**, split as
   - **indirect** (≥ 1 bounce; if the direct is not separated, all of it,
     and then the `.athl` carries no polygons for that group),
   - **emission** (the emitter splats seen through their lens),
   - optionally a **view-dependent field** per splat (degree-2 harmonics of
     the view direction) for glossy paint and chrome near a lamp.
   Separating direct and indirect (069 §3.3) lets the web compute the direct
   with LTC and keep the bake small.
3. **The emitter polygons per group** (060 §4): the emissive faces of
   `Main_Lens_Reflecotr`, `Mini_Lens_Reflectors`, `Headlights_Housing`,
   `Rear_Light_Relfectors(.001)`, `Rear_Light_Tube`, clustered to a few
   3–8-vertex polygons per group (athenea's light BVH clusters), in the
   cloud's object space, with relative radiance.
4. **The lens per group** (076): a tint (the mean transmittance of
   `Glass_Red`, `Glass_Clear_Ridged`...) for every group, and for the low and
   high beams a 128 × 64 beam map in the lamp's frame (origin, axes) — or the
   author's IES (`shaping:ies:file`) when the geometry gives no realistic
   cut-off.
5. **The `.lights.usda`** of 067 appendix B with the group names as the
   Blender light groups, the calibrated radiances (063), the cloud's hash.

**Interim, before 069:** athenea's `lightGroup:NAME` AOVs (or Cycles'
`Combined_<group>` passes, 067 §3) rendered from the cloud's own splat
positions are not per-splat data; the conversion would need a per-splat
gather (render the AOV from a set of views and project back to the splats,
the way mesh2splat fits colours). Simplest interim: mesh2splat run once per
group with only that group lit and the dome black, its colours (linear,
per unit radiance) taken as that group's indirect + emission layer, no
polygons. The converter `rust/build-lod` would then build the `.athl` from
those clouds (same splats, same order): `sparse_layers` + `virtual_values`.

## From athenea's per-group clouds (the real Corvette)

athenea bakes the interim of 069: a base cloud with every lamp off and, per
group, a cloud of the **same gaussians in the same order** baked with only
that group lit at `emission_luminance` 1 and the dome black (its linear
colour, `0.5 + SH0 · dc`, is the group's light per unit of luminance,
through the lenses, the emitter baked white: the sidecar's `emissionColor`
is the group's colour in `w_k`). `usd-athc` builds the
`.athl` beside the `.athc`:

```sh
usd-athc base_tx.usdc car.athc --transfer 16 --gzip --drop-backs 0.008 \
    --light-layer cruce=layer_cruce.usdc ... --lights-usda corvette.lights.usda \
    [--athl car.lights.athl] [--light-threshold 1e-4] [--light-floor 0] [--light-verify NAME]
athc-convert car.athc light.athc --gzip --keep-splats 1000000 --creases 0.03 \
    --athl car.lights.athl --athl-out light.lights.athl
```

- **Alignment.** Every step that selects, reorders or merges splats reports
  what it did (`pack_streams_kept`, `drop_hidden_backs_kept`, `crop_box_kept`,
  `reduce_thin_runs` / `reduce_cells_runs`, `lod_order`), and
  `athl::SplatSources` composes them: each built splat is a weighted mean
  of source splats (weights as the colour was merged), so a layer follows
  the cloud exactly. Prim masks (`--only-prim`, `--exclude-prim`) are
  applied to the layer as to the base; a layer is refused if its positions
  differ from the base's. `--light-verify NAME` compares a layer with the
  file's own colours (give the base as its own layer: 0 difference).
  `athc-convert --athl` carries layers through a LoD cut: each element the
  weighted mean of the original splats under it (`athc::cut_sources`,
  `CreaseCut::sources`), merged nodes with the original weights
  (`virtual_values_weighted`), the cloud hash re-stamped.
- **Unbaked gaussians.** A gaussian the bake did not reach keeps an
  all-zero SH in every layer (the first Corvette bake: 3 782 of 2 998 230,
  those its transfer missed). Read as `0.5 + SH0 · 0` that is mid grey: 5 000
  nits under a 10 000-nit low beam, white specks all over the car. usd-athc
  reads an all-zero SH as no light (a baked black is `dc = -0.5 / SH0`) and
  reports the count (`unbaked`).
- **Scale.** The values are stored times the group's radiance from the
  sidecar (`radiance`, or athenea's `emissionLuminance`), with GRPS
  `radiance` set to it, so the plugin's `w / radiance` gives the same light
  and f16 keeps the faint bounce (per unit of 10 000 nits it would be
  subnormal). No polygons, no profile, tint 1: kind 0 is all of the group.
- **Sparsity.** `--light-threshold` (radiance units) drops blocks with no
  value past it; `--light-floor` first zeroes splats fainter than it (the
  bake's grainy faint indirect).
- **The floor.** The lamps' light on the ground comes as an additive cloud
  (athenea's `ground_base.usdc`, black, and `ground_<group>.usdc`), built as
  its own part with its `.athl` (`additive: true` in the scene). Additive
  splats (`SplatMesh.additive`, WebGPU, ext accumulator, Gaussian splats):
  generate's `GEN_ADDITIVE` marks the splat (the high half of its alpha
  word), the draw writes its premultiplied light with alpha 0 and the tile
  rasterizer leaves the transmittance: light added, nothing covered
  (`test/gpu/additive.test.ts`). The page switches the relight off for it
  (`host.disable(relight.id, mesh)`: per-asset settings take the SplatMesh)
  so only the lamps light it, and hides it with the ground.
  athenea's raster composites a catcher's gaussians among themselves
  (`lampTransmittance`, splat_blend.slang), so the renderer draws additive
  splats as a layer of their own: a pass of them alone (DRAW_ADDITIVE_ONLY)
  into a cleared float texture, blended as any splats; that layer's light
  added to the frame (alpha untouched); the other splats drawn over it
  (DRAW_NO_ADDITIVE). In three's pass (renderInPass) the layer is drawn
  before it, without the scene's depth, and added in it before the splats.
  A frame with additive splats takes the quad draw (not the tiles).
  Against athenea's day render, lamp-only, front floor: 0.86 of athenea
  (2.14 summed); at pixel (63, 719) 0.0935 against athenea's 0.104 and
  0.091 predicted from the data by its blend. `usd-athc --light-additive`
  (`athl::additive_shares`, coverage / sum at each splat's centre) is the
  view-independent approximation for a renderer without the layer; unused.
  A layer can also be read from athenea's native cloud:
  `--light-layer g=file.usdc#athenea:splat:lightLayer:g`.
- **Glass.** athenea adds the lamps to a thin sheet's colour before
  scaling it to the sheet alpha (`colour += lampLight; colour *= alphaOwn /
  alpha`); the lights plugin scales a sheet's lamp light the same way
  (the headlight glow was 4x athenea's, now 1.4x on the light set).
- **Sidecar forms.** athenea's `corvette.lights.usda` writes states as
  `double athenea:lightState:target:<group>` (any prim under `LightStates`
  is a state; none: every group off) and the radiance as
  `athenea:lightGroup:emissionLuminance`; both are read.
- `scripts/build-corvette-lights.mjs` does the detailed and light sets in one
  command (`corvette.html?set=lights`).

## Open issues

- **Specular direct**: LTC with the GGX lobe needs athenea's fitted tables
  (060: 64 × 64 `M⁻¹`), which athenea does not have yet (only Zeltner's sheen
  LTC). Today the direct term is diffuse; glossy reflections of a lamp come
  only from a baked field.
- **Shadows on the direct**: 060's quotient through the open-direction bits
  is not done; a lamp lights through the body if the polygons see past it.
- **One lamp frame per group**: a group with two headlamps uses one profile
  origin (their midpoint); 076's far-field approximation, and two
  symmetrical lamps, make this close, but a per-polygon frame would be exact.
- **IES** profiles are read from the `.usda` but not evaluated.
- **Paging** follows the cloud's pages by polling the pager's page table
  each frame (`athlPagerChunks`), not by events: a page's layers arrive a
  frame or more after its splats (drawn without the group's layers
  meanwhile). The `.athl` pages are not part of AA's attribute budget
  (`attribPaging.ts`); they are small (a few blocks a page) but uncounted.
- **The relight pass on a paged cloud** dispatches one thread per pool slot:
  at the default pool (256 pages) that is 65 536 workgroups, one over
  WebGPU's limit, and the frame is invalid. The example's `?paged=1` uses a
  16-page pool. The fix belongs in `relightPlugin.ts` (slice the dispatch,
  or run over the resident pages only).
- **Frame skip**: a weight change regenerates the splats (the colour is made
  in generate), and the renderer re-sorts with it; nothing else is redone.
- The floor's additive layer (062 §4) would be its own catcher cloud with its
  own `.athl`: not exercised.
