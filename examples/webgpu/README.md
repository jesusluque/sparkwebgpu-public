# Spark on WebGPU

`SparkRenderer` draws with Spark's WebGPU backend when its renderer is three's
`WebGPURenderer`. An app written for WebGL changes two lines:

```diff
-import * as THREE from "three";
+import * as THREE from "three/webgpu";
 ...
-const renderer = new THREE.WebGLRenderer();
+const renderer = new THREE.WebGPURenderer();
 const spark = new SparkRenderer({ renderer });
 scene.add(spark);
 scene.add(new SplatMesh({ url }));
 renderer.setAnimationLoop(() => renderer.render(scene, camera));
```

`"three"` and `"three/webgpu"` share `three.core.js`, so Spark and the app see
the same three classes. With an import map, add
`"three/webgpu": ".../three.webgpu.js"` and `"three/tsl": ".../three.tsl.js"`.
`new SparkRenderer({ renderer, backend: "webgpu" })` requires a WebGPURenderer.
If the browser has no WebGPU, WebGPURenderer falls back to WebGL2. Spark can't
draw on that fallback and logs an error.

The backend is a file of its own, `dist/spark.webgpu.module.js`, which
`SparkRenderer` loads the first time it gets a WebGPURenderer: splats appear
once `spark.webgpuReady` resolves. `await loadWebGPU()` first, or import
`"@sparkjsdev/spark/webgpu"`, to have `spark.webgpu` from the constructor on.
The WebGPU-only API (`WgpuSplatRenderer`, `GpuSplatSource`, `WgpuLod`, `fx`,
`SplatAttributes`, `splatRequiredLimits`, ...) comes from
`"@sparkjsdev/spark/webgpu"`; the pages here map it in their import maps.

### Clear colour

WebGPURenderer clears the canvas to transparent black, while
WebGLRenderer's default is opaque black, so on a page with a light
background a port shows the page through. The ported pages call
`renderer.setClearColor(0x000000, 1)` after creating the renderer, unless the
original asked for `alpha: true`.

## How it works

`SparkRenderer` is still a transparent mesh in the scene, but three draws
nothing for it. three calls its `onBeforeRender` while it records its render
pass. At that point `SparkWebGPU` (`src/webgpu/SparkWebGPU.ts`) does three
things:

- It maps the scene's visible `SplatMesh`es and `SplatGenerator`s to
  `WgpuSplatRenderer` meshes. PackedSplats and ExtSplats are uploaded once and
  shared. LoD and paged meshes go through `WgpuLod`. Modifiers, edits and
  skinning go through `splatMeshDyno`.
- It submits generate and sort.
- It draws the splats against three's depth buffer.

- **On the canvas** (with or without `antialias`), three renders into a
  linear half-float target and then converts it to the canvas. WebGL Spark
  blends splats in sRGB space on its canvas. So where the SparkRenderer comes
  in three's transparent order, `SparkWebGPU` ends three's pass, draws the
  splats over an 8-bit sRGB copy of the target, clamped as the canvas
  shows it (transmittance in alpha), writes that back to the target
  (`src/webgpu/SrgbComposite.ts`) and resumes three's pass, as three's own
  `copyFramebufferToTexture` does. Transparent objects sorted after the
  splats (by `renderOrder` or depth) draw over them, as on WebGL.
- **With tone mapping**, three/webgpu tone-maps the whole picture in its
  output pass (it has no per-material `toneMapped`), while WebGL Spark
  draws untone-mapped splats over the tone-mapped picture. So the composite
  blends the layer over the picture as it will show, `srgb(T(dst))`, and
  writes back `T⁻¹` of the result (`src/webgpu/toneMapping.ts` inverts each
  of three's operators), which the output pass maps to the composited
  colour. Transparent objects in front still draw over the splats, blended
  in three's linear space as all of three/webgpu's are. Colours an operator
  never produces stay out of reach: AgX and ACES desaturate bright colours
  and AgX's white is 0.997, so such splat colours show as the nearest the
  operator gives. With a custom tone mapping node, a linear output colour
  space or `rawColor`, the splats go on the canvas after the output pass
  instead, over all transparent objects.
- **Into a RenderTarget or through `PostProcessing`'s `pass()`**, the splats
  are drawn inside three's own pass, sorted with the transparent objects,
  and blend in the target's linear space, as WebGL Spark does in render
  targets. `new SparkRenderer({ srgbBlend: true })` (or
  `WgpuSplatRenderer`'s `srgbBlend` option) blends them in sRGB there too, as
  on WebGL's canvas: envmap.html's main view goes from 7.62 to 1.54/255
  against WebGL with it.

## Differences to know

- **`WebGPURenderer` defaults.** The canvas clears to transparent: set
  `scene.background` or `renderer.setClearColor(color, 1)` if the page isn't
  black. `render()` does nothing before `await renderer.init()`, but
  `setAnimationLoop` waits for init. Apps that render on demand should await
  `init()` first (see `on-demand.html`).
- **Custom dyno code.** Dynos with hand-written GLSL run as they are: their
  GLSL is translated to WGSL (see `splat-shader-effects.html` and the other
  ports, which keep their original GLSL). What the translator doesn't cover
  fails with the GLSL line and column; such a dyno can pass
  `wgsl: { globals, statements }` to `new dyno.Dyno(...)`. A graph that fails
  to compile drops its mesh and logs the error.
- **three's ShaderMaterial objects.** `ShaderMaterial` objects such as `Sky`
  have node equivalents (`SkyMesh`; see `sogs.html`).
- **Large paged scenes.** The page pool is limited by the device's
  `maxStorageBufferBindingSize` (128 MB by default, 32–42 pages). Pass
  `requiredLimits: splatRequiredLimits(adapter)` to `WebGPURenderer` to get
  more pages. The SH pool is sized for the SH degrees the pages carry.
- **Many splats.** With default limits one draw holds 4.19M splats in the
  ext accumulator; past that `WgpuSplatRenderer` switches to the packed one
  (`accumulator: "auto"`), up to 8.39M. `splatRequiredLimits` raises both.
- **Render targets on meshes.** The `target` option, `renderTarget` and
  `readTarget` work as on WebGL. three's WebGPURenderer (r180) shows a
  target's texture upside down on a mesh's UVs compared to WebGL, and doesn't
  pick up a `map` that replaced a plain texture after the material compiled
  (see `multiple-viewpoints.html`).
- **Raycasting.** Raycasting works for packed, ext, LoD and paged (`.rad`)
  meshes, with SparkRenderer's coarser LoD raycast selection (`lodRaycast`).
  `WgpuSplatPager` keeps the page pool's core data on the CPU for it, as
  WebGL's pager keeps its texture data.

- **Offscreen renders.** The `target` option with `renderTarget()` /
  `readTarget()`, and `renderCubeMap()` / `readCubeTargets()`, work as on
  WebGL: the SparkRenderer in the scene draws as the one rendering
  (`sparkOverride`). `rawColor: true` writes splat colours as they are,
  with no colour conversion, for data packed into RGB (render-cube-depth).
  `renderEnvMap()` prefilters with three/webgpu's `PMREMGenerator`: Spark's
  `THREE.PMREMGenerator` when "three" resolves to three/webgpu, or the one
  passed as `renderEnvMap({ PMREMGenerator })`. Without either (Vite resolves
  Spark's "three" to three's WebGL build) it returns the cube map, which
  three/webgpu's materials prefilter themselves. `getLodTreeLevel()` works
  as on WebGL.
- **Baking colours.** `RgbaArray.render()` runs GLSL readers. On WebGPU,
  `spark.getRgba({ generator })` gives a generator's RGBA as it generates
  it (modifiers, recolor), in a GPU-backed `RgbaArray` usable as
  `splatRgba` and readable with `read()`; `fromPackedSplats()` decodes on
  the CPU (splat-painter).
- **Portals.** The splat shader has `DISK_PORTAL_FRAGMENT_SHADER`'s disk
  clip built in: a SparkRenderer with the `diskCenter`, `diskNormal`,
  `diskRadius` and `diskTwoSided` extraUniforms and no `fragmentShader`
  gets it (portal.html). `SparkPortals` works unchanged.
- **three ShaderMaterials** become TSL node materials. three/webgpu converts
  every material's output to sRGB, which WebGL's ShaderMaterial and
  MeshDepthMaterial skip (splat-portal, render-cube-depth). three/webgpu has
  no `MeshDepthMaterial`.
- **Generator time.** On WebGL generators see SparkRenderer's `Timer`
  (which an inverted `ownsTimer` used to leave at 0); on WebGPU,
  `performance.now()`.
- **three versions.** The WebGPU path uses some of three r180's internals
  (render context, backend pass state, pipeline cache), all in
  `src/webgpu/threeInternals.ts`, which throws `ThreeInternalsError` on
  another three revision or when an internal it reads is gone.

## Not supported yet

These throw or warn:

- custom `vertexShader`/`fragmentShader` (the portal disk clip is built in);
- WebXR and array cameras;
- `covSplats`, `enable2DGS`, `accumExtSplats`;
- SplatMeshes with a custom `SplatSource`.

## Performance

`compare-webgl.html` and `compare-webgpu.html` render the same scene with
each backend (`?n=` synthetic splats, `?file=`, `?lod=1`, `?rad=1` for the
paged hobbiton scene). `window.__fps(seconds)` measures the animation loop
with the object turning, so every frame regenerates and re-sorts
(`__fps(seconds, 0)`: held still, the draw alone); run Chrome with
`--disable-gpu-vsync --disable-frame-rate-limit`. Measure each page in a
browser of its own, one at a time: pages left open in other browsers skew
each other's fps by up to 2x. On WebGPU,
`?profile=1` turns on `WgpuSplatRenderer`'s `profile` option (timestamp
queries, where the browser has them) and `window.__profile(frames)` returns
the median GPU milliseconds of generate, each sort stage and the draw.
`?opts=<JSON>` sets any other renderer option.

Where the time goes (Apple GPU, 1280×720): the draw is 80–85% of the GPU
time. It is bound by rasterizing and blending the quads, not by the vertex
shader or by what it reads: the packed accumulator, flat varyings and
quads cut to the minAlpha radius all measured the same. The sort is next
(~1 ms per million splats sorted), generate well under 1 ms. JS costs
0.1–0.5 ms a frame. So the WebGPU path does less work instead: generate
drops the splats the draw would skip (outside the frustum, under
minAlpha; `cull`, default on), and the sort only sorts the rest.

Medians of 3 interleaved runs, fps, headless Chrome on an Apple GPU shared
with other jobs (so ±10%), 1280×720, before and after the culling and
compacted sort:

| Scene | WebGL | WebGPU before | WebGPU after |
|---|---|---|---|
| synthetic 1M | 127 | 107 | 106 |
| synthetic 2M | 54 | 62 | 59 |
| synthetic 4M | 14 | 27 | 28 |
| penguin | 252 | 664 | 607 |
| robot-head | 594 | 839 | 705 |
| valley | 200 | 350 | 362 |
| hobbiton .rad, 2.5M LoD budget | – | 68 | 64 |

GPU time per frame (sequential, `__profile`): hobbiton 5.9 → 5.5 ms (draw
3.9 → 3.5), valley 2.8 → 2.9 ms. The hobbiton LoD selection keeps ~0.2M of
its 2.5M slots active and the traversal already culls the frustum, and
inactive keys sort almost for free (all equal, coherent scatter), so the
compacted sort mostly helps when many active splats are off screen.

Later (October 2026, M5 Pro), GPU ms per frame with `__profile`:

- **The draw** is bound by rasterizing many small overlapping quads, not by
  the fragment shader: a constant colour without discards costs the same,
  quads at half size (a quarter of the pixels) still cost 60%, and an
  octagon in place of the quad (17% fewer pixels, 6 triangles) was slower
  (synthetic 1M 10.2 → 12.4 ms). Cutting the footprint where `splatAlpha`
  falls under `minAlpha` (the image is unchanged) only pays off on faint
  splats: penguin's draw 1.14 → 1.02 ms, the synthetic clouds and valley
  unchanged.
- **The sort's scatter** ranks keys with subgroup ballots where the device
  has subgroups: the whole sort 28–37% faster (1M: 1.08 → 0.76 ms; hobbiton
  at 2.5M: 3.1 → 2.0 ms). 8-bit digits (4 passes instead of 8) were slower
  (1M scatter 0.43 → 0.57 ms): the 256-bin ranking and scattered writes
  cost more than the passes saved.
- **`minSortIntervalMs`** (as SparkRenderer's): while only the camera and
  transforms move, the GPU sort runs at most that often and the draw reuses
  the last order, as WebGL draws its last worker sort; the next still frame
  sorts exactly. Default 0. At 16 ms, penguin turning goes 645 → 720 fps.
- **Generate** skipping SH and accumulator writes for culled splats made no
  difference on hobbiton (its cost is the gather through the LoD indices).
- **`rasterizer: "tiles"`** stays 2–2.5x faster on the dense synthetic
  clouds (1M: span 3.6 ms against ~8.5 ms) and slower on captured scenes
  (valley 4.2 against 2.7 ms), where `tiles.blend` dominates.

GPU ms per frame (`__profile` span, object turning, medians of 4 rounds
interleaved between the two builds, other jobs on the GPU), before and after
the subgroup scatter and the footprint cut:

| Scene | before | after | sort before → after |
|---|---|---|---|
| synthetic 1M | 9.19 | 8.67 | 1.08 → 0.76 |
| synthetic 2M | 19.87 | 19.22 | 1.96 → 1.42 |
| synthetic 4M | 42.88 | 42.94 | 5.39 → 3.75 |
| synthetic 1M close-up (`dist=1.2`) | 37.17 | 35.11 | 1.65 → 1.13 |
| penguin | 2.29 | 2.30 | 0.36 → 0.34 |
| valley | 3.39 | 4.13 (draw noise) | 0.63 → 0.61 |
| hobbiton .rad, 1M | 7.23 | 6.70 | 0.94 → 0.68 |
| hobbiton .rad, 2.5M | 18.65 | 16.47 | 3.12 → 1.96 |

fps on a shared GPU varied 2x between runs of the same build, in headless
Chrome unthrottled and paced (`__bench`) alike, so these GPU times are the
comparison to trust.

## Examples

The ports with GLSL dynos keep the original GLSL; WebGPU translates it to
WGSL (src/dyno/wgsl/glslToWgsl.ts).

| Example | Changed besides the renderer | Mean diff vs WebGL (/255) |
|---|---|---|
| hello-world.html | – | 0.03 |
| multiple-splats.html | – | 0.12 |
| raycasting.html (clicked) | – | 0.08 |
| interactivity.html (GLB, lights, shadows) | asset paths | 0.16 |
| extsplats.html | – | 0.13 |
| nonlod.html | – | 0.16 |
| lod-example.html (examples/lod) | – | 0.25 |
| multi-lod.html | – | 0.58 |
| streaming-lod.html | adapter storage limits | 0.82 (fully streamed) |
| depth-of-field.html | – | 0.97 |
| on-demand.html | `await renderer.init()`, adapter storage limits | 0.74 (fully streamed) |
| splat-shader-effects.html | – (GLSL dyno translated) | 0.72 (the suite measures 0.72 on every build back to 3c6f02a) |
| procedural-splats.html | – | random stars differ |
| sogs.html | `SkyMesh` for `Sky` | 0.97 |
| debug-color.html, lod-on-demand.html | – | 0.13, 0.23 |
| particle-animation.html | – | 3.09 (time-driven noise) |
| glsl.html | opaque clear color | 0.14 |
| dynamic-lighting.html (lighting on, debug SDFs) | – | 0.22, 0.23 (Safari 0.21, 0.22) |
| mobile-joystick.html (still, after look + walk) | opaque clear color | 0.36, 0.38 |
| multiple-viewpoints.html (still, turning) | the screens' map and UVs (three, see above) | 0.02, 0.05 |
| interactive-deform.html (still, drag, bounce, rotate) | main.js inlined | 0.08, 0.10, 0.09, 0.12 |
| interactive-ripples.html (still, 30 and 90 frames after a click) | main.js inlined | 0.33, 0.33, 0.34 |
| interactive-holes.html (still, 1 click, 5 clicks, reset) | – | 0.52, 0.68, 0.65, 0.52 |
| splat-dissolve-effects.html (t = 0, 10, 40 s) | opaque clear color | 0.15, 0.39, 0.32 |
| splat-reveal-effects.html (Magic, Spread, Unroll, Twister, Rain) | opaque clear color | 0.11, 0.41, 0.36, 2.40, 2.34 (fast particles) |
| lofi.html (default, next/prev world, bad weather, mid-transition) | – | 0.58, 0.76, 0.32, 1.02, 0.84 |
| splat-transitions.html (spherical, explosion, flow, morph; 11 states) | – | 0.12–0.39 |

`spark-renderer.html` puts a cube through a splat mesh. Use
`?backend=webgl` for the WebGL version, `&mode=target` or `&mode=post` for the
in-pass paths, and `&msaa=1` for antialiasing.
| particle-simulation.html | – | 0.20 (clock at 0) |
| splat-flow.html | – | 0.09, 0.42 moving |
| viewer.html (`?url=`, file input) | clear colour | 0.48, 0.17 |
| splat-painter.html (paint, erase, undo strokes) | `spark.getRgba` for `RgbaArray.render` | 0.57; exported SPZ: same splats, 88 bytes of 40 MB differ |
| render-cube-depth.html (depth on) | clip z mapped to WebGL's for the dyno, TSL depth material, linear output + `rawColor` in depth mode | 1.05 off, 6.01 on (low packed-depth bits); cube readback means within 0.3/255 |
| portal.html | no fragmentShader (built-in disk clip) | 0.32 (local assets substituted) |
| newportal.html | adapter storage limits | 0.33 with a non-paged scene; 0.00 on the suite's substitute .rad, which shows mostly the dark portal disk on both |
| splat-portal.html (+ splat-portal/main.js) | TSL portal material | 0.58 |
| editor.html (normal colour, clip, grid) | – | 0.01 empty, 0.46 with a file, 0.30 with debug options |

`spark-renderer.html` puts a cube through a splat mesh. Use
`?backend=webgl` for the WebGL version, and `&mode=target`, `&mode=post` or
`&msaa=1` for the in-pass paths, `&srgb=1` for `srgbBlend`. Against WebGL,
outside the transparent plane (three's own blending differs there): canvas
and MSAA 0.31/255; target and post 0.84 linear, 0.30 with `&srgb=1`. A
transparent plane with `renderOrder = 1` over the splats now draws over
them, as on WebGL (it went under them before).

Parity numbers use a frozen clock (`performance.now`, `Date.now`, rAF time)
and a seeded `Math.random`.
