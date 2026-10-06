# WebGPU performance audit

An algorithm and cost review of the WebGPU pipeline, stage by stage, with what was measured, what was changed, and what is left. Thread AP, 2026-10-06, on an Apple GPU (Chrome, headless, `--disable-gpu-vsync`) and Dawn/Metal for the tests.

**How the numbers were taken.** The GPU was shared for most of the session with a long athenea job (GPU utilisation 94–99% before our work started), which stretches every pass several times over and unevenly: the same 1M-splat draw read 15 ms on an idle GPU and 100–125 ms under that load. So:

- stage times are GPU timestamps (`profile`), medians of 15–40 frames, each frame waited on and read back before the next;
- every comparison is an interleaved A/B in the same page load (configurations alternated 3–5 times), and only ratios are quoted from loaded runs;
- where an idle window came (`idle.sh` waits for utilisation < 30% for 10 s), the absolute numbers are from it and marked *idle*;
- memory is measured on the CPU (the WASM decoder in Node), which the shared GPU doesn't affect.

Scripts (session scratchpad, `ap/`): `corvab.mjs` (Corvette dev page, interleaved configurations, `window.__athenea.frame`), `prof.mjs` (`compare-webgpu.html` `__profile`), `dropped.mjs` (what the draw drops, from the projector's records), `heap.mjs` (WASM heap per `.athc`).

## Summary

| Stage | Algorithm today | Measured | Finding | Done here |
|---|---|---|---|---|
| Relight blend term | AtheneaRelightBlend in every fragment, reading a 16 B record | **a third of a Corvette frame** without `pixelDetail`, where it does nothing | dead per-fragment work | out of the draw variant without `pixelDetail`: Corvette frame −35% (light) / −37% (HD), idle |
| Relight (per eye) | `relitSplat` for every splat of the cloud, each frame the eye moves | 5.8 ms of a 23 ms Corvette light frame (idle) | ran over interior LoD nodes and splats the LoD doesn't draw; kept terms rerun on every LoD update | per-eye pass over the LoD list only: −5% light, −17% HD (idle); −52% with the mobile cap |
| Relight records | 3 × 16 B per splat written and copied by generate | — | dead traffic without `pixelDetail` | written / copied only under a mark |
| `.athc` decode | whole file buffered (Vec doubling), decoded, attribute words kept in the WASM heap | trim 199 MB heap for 0.69M splats | buffer up to 2× the file; output held twice | exact reserve + words straight to JS: −26…30% peak; `workerPool.trim()` after load |
| Sort | stable LSD radix, 8 × 4-bit passes over the active keys | 2M: 1.41 ms (idle) | float keys need 32 bits for full precision | 24-bit depth-range keys, 6 passes: −19…27% |
| Draw (quads) | instanced quads, `projectSplat` in each of 4 vertices | 80–85% of the frame | per-primitive raster is most of it; doing the vertex work once saves 0–17% of the draw, unevenly | compute projector, opt-in (`projectOnce`) |
| Generate | one thread per output splat, cull = draw's early outs | 1M: 0.2–0.4 ms idle | fine | — |
| Draw-side drops after sort | — | 0–0.5% of sorted splats | lossless screen-space cull would gain < 1% | not done (AQ #3a) |
| LoD | WASM traversal, budget + 1 px cut (`lodRenderScale`) | see LoD section | Corvette budget = whole car, cut at 1 px | measured; recommendation |
| Sky | full prefilter on image / rotation / intensity change | once per change | fine (rotation is baked, as athenea) | — |
| Display / FX | full-screen passes | small | fine | — |

## Generate

**Algorithm.** `slang/kernels/generate.slang`: one thread per output splat of a mesh; remap through `lodIndices`, decode (packed/ext), colour plugin (SH or the relit colour), object → world, recolour, dyno hooks, metric, and the draw's early outs (`GEN_CULL`). Each mesh writes its own accumulator range.

**Cost.** Idle: 0.41 ms for 2M synthetic, 0.07 ms for valley. Loaded Corvette (light, 1.08M drawn): about a fifth of the relight pass. Memory bound: 32 B in, 32 B out, 4 B metric per splat (ext), plus the colour plugin's reads.

**Issues.** With the relight colour plugin, generate copied three 16 B per-pixel records per splat from the relight buffer to the draw's slots, used only with `pixelDetail`. Now it copies the head (16 B) and the other two only under its mark.

## Sort

**Algorithm.** `sort_radix.slang` + `GpuSorter.ts`: count and compact the active (finite) metrics in index order, then a stable LSD radix sort, 4-bit digits, histogram → multi-level scan → scatter (ballot ranks where `subgroups` exist), dispatched indirectly over the active blocks only; the active count goes straight into the indirect draw arguments. The per-pass scan runs over `BINS × numBlocks` of the full count, but only the active prefix is meaningful (the stale tail sits after it in the bin-major layout), so it is correct; it is a few µs.

**Keys.** The key was the metric's inverted float bits: 32 bits (8 passes) for full precision; `sortBits` 24 drops 8 low bits (2⁻¹⁵ relative). **Change:** with generate's cull on and a perspective camera, every active metric lies in a range known on the CPU (depth + 100 between the planes, or the radial distance up to the far plane's corners past `clipXY`), so `compactKeys` keys it by its place in that range on 24 bits (`KEY_DEPTH`): 6 passes, a uniform step of (range)/2²⁴ — 1.2·10⁻⁵ for the Corvette's 0.05–200 m, 1.1·10⁻⁴ for 0.1–1000 radial — where 24 float bits step 3·10⁻⁵ *relative* (3 mm at 100 m). `depthKeys` (default true) applies it while the step is ≤ 2.5·10⁻⁴; the range is taken from the camera of the generate being sorted (sorts deferred by `minSortIntervalMs` use their own metrics' range).

| Radix stages (histogram + scan + scatter), GPU ms | float 32-bit | depth 24-bit | |
|---|---|---|---|
| 2M synthetic, 1280×720 (idle) | 1.41 | 1.14 | −19% |
| valley (idle) | 0.41 | 0.33 | −21% |
| 1M synthetic (loaded GPU, ratio only) | 25.96 | 19.07 | −27% |
| valley (loaded GPU, ratio only) | 13.88 | 10.76 | −22% |

Parity (WebGPU vs WebGL, mean /255, 10 cases): unchanged — hello-world 0.04 (0.03), multiple-splats 0.12, extsplats 0.13, nonlod 0.16, lod-example 0.25, multi-lod 0.61 (0.66), depth-of-field 1.00, splat-shader-effects 0.72, procedural-splats 3.08, sogs 0.97. GPU test: the depth-key order matches a CPU reference with the kernel's f32 maths exactly, with clamping and ties.

**Options left (ranked).**

1. 8-bit digits with subgroups (4 or 3 passes): −40–60% of sort on Chrome; Safari keeps 4-bit. Medium effort.
2. 20-bit camera-weighted keys (PlayCanvas): 5 passes; changes ties within ~0.2 mm at 200 m. Would need an odd pass count (the result in `vals[1]`) or a copy.
3. `minSortIntervalMs` > 0 by default: the sort is now ~6–8% of a frame; deferring it saves that while moving at the cost of a stale order. Not worth a default change.

## Draw

**Algorithm.** `splat_draw.slang`: one instance per sorted slot, a 4-vertex strip; the vertex stage runs `projectSplat` (decode, covariance, eigen-decomposition, alpha cut, the falloff-based footprint cut thread V added); the fragment evaluates the Gaussian, discards outside the ellipse and under `minAlpha`, and blends (premultiplied).

**Cost.** 80–85% of the frame, as threads F and V found. Two measurements of where:

- What the draw drops after the sort (projector records): 0 of 1M synthetic, 1 786 of 367 162 on valley (0.5%), 347 of 96 061 on penguin. The blur keeps every footprint at least ~1.5 px, so almost nothing is sub-pixel. **A lossless screen-space cull before the sort (AQ #3a) would remove < 1%.**
- Quad area: 102 M px² for 1M synthetic at 1280×720 (110× the screen), 55 M px² for valley (60×). That raster, mostly of overlapping semi-transparent quads, is the cost.
- `minContribution` (alpha × area < 3 px², AQ #3b) would drop 31–32% of the splats on valley and penguin — but those are the small ones, a small share of the area, and the image changes (lossy). Left as an option to add, opt-in.

**Projector (AQ #1).** `projectOnce` (opt-in): a compute pass after the sort runs `projectSplat` once per sorted slot into a 64 B record in draw order, and `splatVertexProjected` reads it — no `ordering` or splat fetch in the vertex stage. Bit-identical image on Dawn/Metal (GPU test, 8-bit and half-float). Measured on the Apple GPU, three interleaved runs, all on the loaded GPU (draw pass ms, then the frame with the projector's own pass):

| | vertex projection | projector: draw | + project pass | net frame |
|---|---|---|---|---|
| 1M synthetic, run 1 | 123.7 | 125.5 | 6.0 | +1% |
| 1M synthetic, run 2 | 96.7 | 87.0 | 5.0 | −2% (total 127.6 → 124.5) |
| 1M synthetic, run 3 | 104.0 | 85.9 | 5.7 | −6% (132.3 → 124.3) |
| valley, run 1 | 58.0 | 54.7 | 2.0 | +1% |
| valley, run 2 | 49.0 | 42.1 | 2.0 | −12% |
| valley, run 3 | 52.4 | 48.3 | 2.0 | −5% |

For scale, the same 1M draw at 320×180 instead of 1280×720 (1/16 of the pixels) took 81.6 ms instead of 104: four fifths of that draw do not scale with the pixels. Doing the vertex work once takes 10–17% off the draw in two runs of three, less the projector's own pass; the rest is per primitive (binning and setup of 2M triangles), which only fewer splats removes. It stays opt-in: the gain is small and not consistent, it costs 64 B a splat (330 MB for the HD Corvette), and the quads' raster, not their vertex shading, is what costs on a tile-based GPU. Other GPUs (discrete, where PlayCanvas and Babylon ship it) may differ.

**Options left.**

1. Tiles with sub-tile masks (AQ #4) to make `rasterizer: "auto"` pick tiles on captured scenes: the only lever on the dominant cost besides drawing fewer splats.
2. Fewer splats: the LoD cut (below) and `minContribution` (opt-in).
3. `nointerpolation` on `rgba` (AQ #11): trivially safe, not measurable here.

## Relight (athenea)

**Algorithm.** `relightPlugin.ts` + `slang/athenea_adapter/relight.slang`, a compute pass before generate:

- the sky, when its image, rotation, intensity or exposure change (`AtheneaSky.prepare`: area mips, sun extraction, 16 SH, residual, 8 GGX levels; baked as athenea bakes them, so a rotation re-prepares);
- the kept terms (`splatTransferViewless`, athenea's txCache) per splat, when the sky, lights, the mesh's place or data change;
- the colour (`relitSplat`) per splat, when the eye moves (or, with `pixelDetail`, the projection); generate's colour plugin reads it by source index.

**Issues found.**

1. The per-eye pass relit **every splat of the source**, interior LoD nodes and splats the LoD cut doesn't draw included. The Corvette light set holds 1.46M source splats and draws 1.08M; the HD set holds 7.28M and draws 5.11M; with the mobile cap (600k) 1.46M were relit for 0.6M drawn, and a paged pool (the pawn) relit the whole pool.
2. The kept terms' key held the LoD indices, so they were recomputed on **every LoD update** (each moving frame): 132 kept-term passes in 45 orbit frames, measured.
3. Without `pixelDetail` the pass still wrote 3 × 16 B per splat of per-pixel records, generate copied them, and the draw ran the relight blend term, which reads 16 B per fragment to find no mark.

**Changes.**

- The per-eye entries take a list (`listCount`, `listBase`): the LoD indices, copied after the records in the relit buffer when the cut draws fewer splats than the cloud holds (`relightList`). The relit buffer gains 4 B a splat; no new binding (the pass uses 10, the Chrome/Mac limit). Generate reads only drawn splats, so the image is the same (GPU test: drawn colours bit-equal, undrawn untouched, kept terms equal).
- The kept terms no longer key on the LoD indices; a paged pool's page upload bumps its sources' `version` instead (`WgpuSplatPager`), which covers what a slot holds.
- Records only under `kRelightSlope`; generate copies only under the mark; `SplatPlugin.blendActive` leaves the relight blend term out of the draw variant without `pixelDetail`.
- `WgpuSplatRenderer.passTimestampWrites`: the relight pass is timed as `relight` with `profile`.

| Corvette, orbit, relight pass GPU ms (interleaved, loaded GPU) | before | after | |
|---|---|---|---|
| light (1.46M source, 1.08M drawn) | 73.4 | 54.6 | −26% |
| light, `?lod=600000` (mobile cap) | 71.3 | 28.4 | −60% |
| HD (7.28M source, 5.11M drawn) | 212.6 | 31.0 | −85% |

Those were taken under the athenea job, which exaggerates every difference. In an idle window (1280×800, orbit, 24–32 frames, 4 interleaved rounds of four configurations: before, list only, blend term out only, both):

| Corvette, idle | before | list | blend out | both |
|---|---|---|---|---|
| light: frame wall ms | 23.4 | 23.1 | 15.4 | **15.3** |
| light: relight pass ms | 5.85 | 5.53 | 5.86 | 5.55 |
| HD: frame wall ms | 109.5 | 102.9 | 73.7 | **68.6** |
| HD: relight pass ms | 29.5 | 24.6 | 29.6 | 24.7 |
| HD: compute total (relight + generate + sort) ms | 38.4 | 33.5 | 38.5 | 33.4 |

With the mobile cap (`?lod=600000`, 0.6M drawn of 1.46M), idle: relight 6.31 → 3.01 ms (−52%), frame 16.7 → 13.9 ms with the list alone, **9.5 ms** with both (−43%).

So the largest single win of this audit is the blend term: without `pixelDetail` it marked nothing, yet every fragment of every relit splat ran it and read its record — 8 ms of a 23 ms frame on the light set, 36 ms of 110 on the HD set. The LoD list matters most where the cut is far from the cloud (HD, a budget, a paged pool); on the light set at its default distance the splats it skips are cheap ones (interior nodes).

**Options left.**

1. **Relit buffer memory.** 80 B per source splat plus 48 B per drawn slot for the pixel records: 611 MB + 245 MB on the HD set. Without `pixelDetail` only the colour, kept terms and centre are read: a stride of 3 (and 1 record) would save ~400 MB there. Medium effort (stride as a uniform in the pass and the colour plugin).
2. **Frustum list.** Relight only splats in view: needs the view direction in the key (a pure rotation would then rerun), and a stale colour when a splat enters the view; a small gain for the car, which is mostly in view.
3. `pixelDetail` (11 vs 95 fps, measured by an earlier thread) is the expensive path; its per-fragment work is AtheneaRelightBlend's sky read per pixel.

## LoD

**Algorithm.** `WgpuLod` runs Spark's WASM traversal in a worker: a priority refinement from the roots until the budget (`lodSplatCount`) is reached or every node's projected size (2 × mean scale, expanded by LoD opacity, over distance, foveated) is under `pixelScaleLimit` = 1 px × `lodRenderScale`.

**On the Corvette.** The page sets `lodSplatCount` to at least the car's splat count, so the budget never binds and the cut is the 1-px one. Measured (idle, orbit, with this audit's changes), `lodRenderScale` 1 against 2:

| | drawn, scale 1 | drawn, scale 2 | frame ms, 1 → 2 |
|---|---|---|---|
| light, default distance (2.2×) | 1.08M | 0.94M (−13%) | 15.7 → 13.6 |
| light, `?distance=5` | 1.08M | 0.79M (−27%) | 15.3 → 10.9 |
| HD, `?distance=4` | 3.43M | 1.12M (−67%) | 47.3 → 14.3 |

The light set is already its detailed tree cut above the finest levels, so its leaves are near the 1-px size at the default distance and even far off the cut barely moves (1.08M at 2.3× the distance). On the HD set seen from afar, a 2-px cut draws a third of the splats and the frame is 3.3× faster: that is the "≈4× fewer splats far away" of an athenea-like cut.

**Options.** `lodRenderScale` is the existing knob for an athenea-like coarser cut (AK's proposal): splats per pixel fall with its square. Whether the image holds is a visual call for the user (shells: AK found the merged levels close at every distance since the coverage fix).

## Paging and upload

`WgpuSplatPager` keeps pages in storage-buffer pools (65 536 splats a page), uploads with `writeBuffer`, interleaves ext pages on the CPU (an 8-word copy per splat), and widens the SH pool's stride on demand (`pool_restride`). `PagedAttribPool` pages attributes by stream group with a byte budget per tier. Nothing here showed in the frame profiles; the CPU interleave of ext pages is the only per-splat CPU work (could be two `writeBuffer`s with a strided kernel, not worth it at page rates).

## `.athc` decode and site loading

**Algorithm.** The worker buffers the whole file, `read_v3` decompresses every section into an `AthcFile`, the decoder computes the merged levels' coverage and the virtual tree, and emits every block into the receiver: ext splats into JS arrays, attributes into the receiver.

**Issues.** (1) The buffer grew by doubling: up to twice the file, plus a copy at each growth. (2) The receiver kept the packed attribute words in the WASM heap until the end, next to the decoded file. A WASM heap never shrinks, so each worker keeps its largest peak for its lifetime, and the site's four workers kept them all.

**Changes.** `athc_v3::file_bytes` reads the file's size from its tables and the decoder reserves it (page-aligned) once the tables are in; `DecodedAttribs` writes the words straight into JS `Uint32Array`s page by page; `corvette.js` calls `workerPool.trim()` after loading on every device (the mobile path already did; the LoD traversal's worker is not a free one).

| WASM heap peak (MB) | before | after |
|---|---|---|
| light: trim (0.69M splats, 30.9 MB file) | 199 | 144 |
| light: body / paint / catcher | 106 / 72 / 32 | 75 / 59 / 30 |
| HD: paint (2.73M, 125.5 MB) | 823 | 610 |
| HD: trim / body | 749 / 302 | 540 / 230 |
| retained by 4 workers after loading, light / HD | ~410 / ~1 960 | 0 (trimmed) |

Decode time is unchanged (trim 0.5 s, HD paint 2.0 s in Node).

**Left.** The decoded `AthcFile` (~110 MB of the trim's 144) is whole in the heap while the receiver is filled. Decoding block by block would need the merged levels' coverage first (it reads the splats); a two-pass decoder could bring the peak near the largest block. High effort.

## Sky, display, FX

`AtheneaSky.prepare` runs only when the dome changes; its 4k source is uploaded once per image (`rgba32float` with mips, ~170 MB on the GPU), and the page's backdrop holds its own float texture of the same image. `AtheneaDisplay` and the colour corrector are full-screen passes per frame, small next to the draw. Nothing to change.

## Recommendations, ranked

| # | Change | Gain | Risk |
|---|---|---|---|
| 1 | Tile rasterizer sub-tile masks (AQ #4), then `auto` by default where `timestamp-query` exists | the draw: up to 2–4× on dense views | medium |
| 2 | Coarser LoD cut on the Corvette (`lodRenderScale` 2, or a cut in pixels per merged cell as athenea) | HD far: −67% splats, 3.3× frame; light: −13…27% | visual: needs the user's eye |
| 3 | Relit buffer stride 3 without `pixelDetail` | ~400 MB on the HD set | low |
| 4 | 8-bit radix digits with subgroups | −40–60% sort on Chrome | medium |
| 5 | Opt-in `minContribution` cull in generate | −30% splats on captured scenes, lossy | low |
| 6 | Two-pass `.athc` decode | heap peak → largest block | medium |
| 7 | `projectOnce` by default on non-Apple GPUs, after measuring there | unknown | low |
