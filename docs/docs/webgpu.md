# WebGPU backend

Spark can draw splats with WebGPU as well as WebGL2. When the `renderer` you give `SparkRenderer` is three.js's `WebGPURenderer`, the scene's `SplatMesh`es and `SplatGenerator`s go through Spark's WebGPU backend: the same scene graph and the same API, with the GPU work done by compute and render pipelines whose kernels are written in [Slang](https://shader-slang.org) and compiled ahead of time to WGSL.

WebGL2 stays the default. Nothing changes for apps that keep `THREE.WebGLRenderer`.

What the backend does differently from WebGL Spark:

- **Generate** runs as a compute pass, one thread per splat, instead of a fragment-shader pass into a texture.
- **Sort** is a GPU radix sort in the same frame, with the draw count written straight into indirect draw arguments. WebGL Spark reads the distances back and sorts in a worker, a frame or more behind.
- **Splat data** lives in storage buffers instead of texture arrays, including the LoD page pool.
- **Custom dynos** compile to WGSL as well as GLSL.

How it works inside is in [WebGPU architecture](webgpu-architecture.md).

## Switching to WebGPU

Two lines change: the three.js import and the renderer.

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

`"three"` and `"three/webgpu"` share `three.core.js`, so Spark and your app see the same three.js classes. With an import map, also map `"three/webgpu"` to `three.webgpu.js` and `"three/tsl"` to `three.tsl.js`.

`new SparkRenderer({ renderer, backend: "webgpu" })` asks for WebGPU explicitly and throws if `renderer` isn't a `WebGPURenderer`. Without `backend`, Spark picks WebGPU whenever the renderer is a `WebGPURenderer`.

If the browser has no WebGPU, `WebGPURenderer` falls back to WebGL2. Spark can't draw on that fallback and logs an error, so check `navigator.gpu` first if you need to choose a renderer yourself.

The migration guide in the repository, `examples/webgpu/README.md`, lists every difference with examples, and the [WebGPU examples](../../examples/webgpu/) include ports of most of Spark's examples, each changed only where the guide says.

### Loading the backend

The WebGPU backend is a separate file, `spark.webgpu.module.js` next to `spark.module.js`, so WebGL apps don't download it. `SparkRenderer` loads it the first time it's given a `WebGPURenderer`, and draws nothing until it has: `spark.webgpu` is set and `spark.webgpuReady` resolves once it's loaded. A render loop needs no change. Code that reads `spark.webgpu`, or renders a single frame with `renderTarget()`, should `await spark.webgpuReady` first.

To have `spark.webgpu` from the constructor on, load the backend first, with `await loadWebGPU()` or by importing the WebGPU API:

```typescript
import { SparkRenderer, loadWebGPU } from "@sparkjsdev/spark";
await loadWebGPU();
// or: import { WgpuSplatRenderer, fx } from "@sparkjsdev/spark/webgpu";
```

`loadWebGPU()` resolves to the `"@sparkjsdev/spark/webgpu"` module, which holds the WebGPU-only API: `WgpuSplatRenderer`, `GpuSplatSource`, `WgpuLod`, `WgpuSplatPager`, `WgpuReadTarget`, `WgpuCubeMap`, `fx`, `SplatAttributes`, `capabilitiesOf`, `splatRequiredLimits` and the rest. With an import map, map `"@sparkjsdev/spark/webgpu"` to `spark.webgpu.module.js`; `spark.module.js` finds it on its own.

## Browser support

| Browser | Status |
| ------- | ------ |
| Chrome (macOS) | Verified: all ported examples, parity and benchmarks were measured here. |
| Safari 26 (macOS) | Safari 26.5 ran `index.html` and `fx.html` at about 100 fps with no errors (checked through `caps.html?pages=`). |
| Other browsers and platforms | Not tested. |

The kernels need no optional WebGPU feature. In particular the GPU sort uses no subgroup operations, which Safari doesn't have. Spark also stays within the spec's default limits: at most 8 storage buffers per shader stage (Safari's default) and 128 MiB per storage binding, unless you ask the adapter for more (see [Limits](#limits)).

`examples/webgpu/caps.html` prints what a browser's adapter and device offer and probes the features Spark relies on. `?pages=a.html|b.html` loads other pages in a frame and reports whether they rendered, and `?report=<url>` posts the result, for browsers that can't be automated. `tools/safari/run.mjs` runs example pages in Safari through `safaridriver` when Safari's remote automation is on.

## What's supported

On `WebGPURenderer`, `SparkRenderer` supports:

- `SplatMesh` with PackedSplats and ExtSplats, spherical harmonics up to degree 3, `recolor`, `opacity` and object transforms;
- Level-of-Detail and paged `.rad` streaming (`WgpuLod`, `WgpuSplatPager`), with the same WASM traversal as WebGL Spark;
- dyno generators, `objectModifiers` and `worldModifiers`, `SplatEdit`, `SplatSkinning`, `splatRgba`, and custom dynos that give WGSL (see [Custom dynos](#custom-dynos));
- depth of field (`focalDistance`, `apertureAngle`), `sortRadial`, `maxStdDev`, `minAlpha`, `falloff` and the other draw options;
- drawing on the canvas (with or without `antialias`), into `RenderTarget`s, and through `PostProcessing`'s `pass()`, depth-tested against three's scene;
- the `target` option with `renderTarget()` / `readTarget()`, `renderCubeMap()` / `readCubeTargets()`, and `getRgba()`;
- `SparkPortals` and the portal disk clip (`diskCenter`, `diskNormal`, `diskRadius` and `diskTwoSided` extra uniforms);
- raycasting against packed, ext and LoD meshes;
- `covSplats` (covariance splats from generate to the draw: non-uniform scales and shears, CovSplat modifiers; a visible `covSplats` SplatMesh turns them on by itself), `enable2DGS`, and `accumExtSplats`, which forces the ext accumulator (without it WebGPU picks ext whenever it fits a storage binding, rather than WebGL's packed default);
- `SplatMesh`es with a custom `SplatSource`: its `fetchSplat` graph runs as the generate kernel's source, followed by the mesh's own pipeline;
- `ArrayCamera`s: each sub-camera generates, sorts and draws into its viewport once three has submitted its frame (so transparent objects in front of the splats end up under them, as with custom tone mapping).

The lower-level `WgpuSplatRenderer` has the WebGPU-only extras: [per-Gaussian attributes](webgpu-attributes.md) with picking, and [aofx post effects](webgpu-fx.md).

### Not supported yet

These throw or warn on WebGPU:

- `renderEnvMap` (use `WgpuCubeMap.renderEnvMap`) and `getLodTreeLevel`;
- a custom `vertexShader` or `fragmentShader` on `SparkRenderer` (the portal disk clip is built in);
- WebXR. three.js's `WebGPURenderer` throws in `XRManager.setSession` on its WebGPU backend, and Chrome and Safari on macOS have no WebXR binding for WebGPU, so XR on WebGPU isn't possible there today. `SparkXr` reports `not_supported` instead of showing a button that fails;
- raycasting against paged (`.rad`) meshes, whose splats exist only on the GPU.

## Custom dynos

Built-in dynos and modifiers already have WGSL. A `Dyno` with hand-written GLSL (`new dyno.Dyno({ globals, statements })` or `dyno.dyno(...)`) runs on WebGPU as it is: its GLSL is translated to WGSL when the graph compiles.

```typescript
const effect = new dyno.Dyno({
  inTypes: { gsplat: dyno.Gsplat, t: "float" },
  outTypes: { gsplat: dyno.Gsplat },
  globals: () => [dyno.unindent(`
    vec3 wave(vec3 p, float t) {
      p.xz *= 1.0 + 0.1 * sin(t + p.y);
      return p;
    }
  `)],
  statements: ({ inputs, outputs }) => dyno.unindentLines(`
    ${outputs.gsplat} = ${inputs.gsplat};
    ${outputs.gsplat}.center = wave(${inputs.gsplat}.center, ${inputs.t});
  `),
});
// In a modifier: gsplat = effect.apply({ gsplat, t }).gsplat;
```

The translator takes the GLSL ES 3.0 that dynos use: scalar, vector, matrix, struct and array types; functions with `in`/`out`/`inout` parameters and overloads; `const` and global variables; swizzles, including assignments to them; `?:`, compound assignments, `++`/`--`; `if`, `for`, `while`, `do`, `switch`, `break`, `continue`, `return` (a bare `return;` in statements returns the graph's output); object-like `#define`s; and the built-in functions, with GLSL's `mod`, `smoothstep`, `atan(y, x)`, `modf` and the rest mapped to their WGSL equivalents. `dot` becomes an explicit forward `fma` chain, the rounding of GLSL's and Chrome's `dot` on Metal: Safari's own WGSL `dot` rounds differently by an ulp now and then, which a `fract(sin(dot(p, k)) * 43758.5453)` hash turns into a different random number. `texture`, `textureLod`, `texelFetch` and `textureSize` work on sampler inputs, with `texture` sampling level 0, since dynos run in compute shaders. Uniform arrays (a `DynoUniform` with a `count`) and the `Gsplat`, `CovSplat` and splatDefines helpers are available.

What it doesn't cover fails the compile with the line, column and construct: derivatives (`dFdx`, `fwidth`), `discard`, `uniform` declarations and preprocessor conditionals, `?:` on structs, assignments inside expressions, `switch` fallthrough, and out arguments that aren't local variables. Only plain `Dyno`s are translated: a subclass of `Dyno` (or a dyno that needs WGSL of its own) gives `wgsl: { globals, statements }` next to the GLSL, with the same inputs and outputs. A graph that fails to compile drops its mesh and logs the error.

## Limits

A WebGPU device gets the spec's default limits unless it asks for more, and three.js's `WebGPURenderer` asks for none. With the defaults one storage binding is 128 MiB, which bounds:

- the splats drawn in one frame: 4.19M in the 32-byte ext accumulator. Past that, `WgpuSplatRenderer` (`accumulator: "auto"`, the default) switches to the 16-byte packed accumulator, with centres relative to the camera, up to 8.39M;
- the LoD page pool: 32 to 42 pages of 65,536 splats, depending on the encoding. The SH pool is sized for the SH degrees the pages carry.

Ask the adapter for its limits with `splatRequiredLimits`:

```typescript
import { splatRequiredLimits } from "@sparkjsdev/spark/webgpu";

const adapter = await navigator.gpu.requestAdapter();
const renderer = new THREE.WebGPURenderer({
  requiredLimits: adapter ? splatRequiredLimits(adapter) : {},
});
```

`capabilitiesOf(device)` reports what a device allows in Spark's terms (whether the GPU sort fits, `maxSplats`, `maxSplatsPacked`, `maxSplatsWithSh`, optional features). When the GPU sort doesn't fit, `WgpuSplatRenderer` falls back to a CPU sort and says why.

## Performance

Frame rates in headless Chrome on an Apple GPU at 1280×720 without vsync, with the object turning so that every frame regenerates and re-sorts (medians of three runs, ±10%):

| Scene | WebGL (fps) | WebGPU (fps) | WebGPU vs WebGL |
| ----- | ----------- | ------------ | --------------- |
| penguin | 252 | 607 | 2.4× |
| robot-head | 594 | 705 | 1.2× |
| valley | 200 | 362 | 1.8× |
| synthetic 1M splats | 127 | 106 | 0.8× |
| synthetic 2M splats | 54 | 59 | 1.1× |
| synthetic 4M splats | 14 | 28 | 2.0× |

Earlier runs on the same machine measured real scenes at 1.6–2.8× and 1–2M synthetic splats at about 0.9×. Your numbers depend on the GPU, the screen size and how much of the screen the splats cover.

On WebGPU the draw takes 80–85% of the GPU time, bound by rasterizing and blending the splat quads (about 4 ms per million splats drawn on that Apple GPU), and the sort most of the rest (about 1 ms per million splats sorted). So the backend draws and sorts only what can be seen: generate drops the splats the draw would skip (outside the frustum, under `minAlpha`; `WgpuSplatRenderer`'s `cull` option, on by default), and the sort only sorts the rest.

`examples/webgpu/compare-webgl.html` and `compare-webgpu.html` render the same scene on each backend (`?n=` for a synthetic cloud, `?file=`, `?lod=1`, `?rad=1`), and `window.__fps(seconds)` measures the animation loop. On WebGPU, `?profile=1` turns on `WgpuSplatRenderer`'s `profile` option, which times generate, each sort stage and the draw with timestamp queries into `stats.gpuMs`, and `window.__profile(frames)` returns the medians. The Performance section of `examples/webgpu/README.md` has the latest measurements.

Like WebGL Spark, the WebGPU renderer skips generate and sort on frames where the camera, the meshes, their transforms and colours, and their dyno uniforms didn't change, and redraws the last order. For splats changed some other way, call `WgpuSplatRenderer.markDirty()` or set its `alwaysGenerate` option.

## Known differences

- **Clear colour.** `WebGPURenderer` clears the canvas to transparent black, where `WebGLRenderer` clears to opaque black, so a page with a light background shows through. Set `scene.background` or call `renderer.setClearColor(0x000000, 1)`.
- **Initialization.** `render()` does nothing before `await renderer.init()`. `setAnimationLoop` waits for it; apps that render on demand should await `init()` first.
- **Colour-space blending.** WebGL Spark blends splats in sRGB space on the canvas, but three.js's `WebGPURenderer` renders into a linear half-float target and converts at the end. On the canvas Spark ends three's pass where the `SparkRenderer` comes in the transparent order, draws the splats into an 8-bit layer of their own and composites it in sRGB, so transparent objects after the splats still draw over them. Into render targets the splats blend in the target's linear space, as WebGL Spark does there; `new SparkRenderer({ srgbBlend: true })` blends them in sRGB there too.
- **HDR.** `new SparkRenderer({ hdr: true })` keeps the splats' light in float end to end, for scene-referred pipelines with a display transform at the end: colours above 1 are kept per splat (the half-float "ext" accumulator, never the 8-bit packed one while it fits a binding), and the splats blend in linear light straight into the float target, the canvas's half-float frame included, with no 8-bit layer over a clamped copy of the background (`srgbBlend` is ignored) and three's tone mapping, if any, applied to them in its output pass. Plugins that make linear light (the athenea raster, relight, lights and output plugins: `SplatPlugin.hdr`) imply it.
- **Tone mapping.** Splats aren't tone mapped, as on WebGL. With tone mapping on, they're drawn after three's output pass, over all transparent objects.
- **three.js r180 internals.** Drawing inside three's render pass uses parts of `WebGPURenderer` that aren't public API (the render context, the backend's per-resource data and its utils). Spark is tested with three.js r180; other versions may need changes.
- **three.js materials.** `ShaderMaterial`s become TSL node materials (`Sky` becomes `SkyMesh`), and three's WebGPU backend converts every material's output to sRGB, which WebGL's `ShaderMaterial` and `MeshDepthMaterial` skip. `three/webgpu` has no `MeshDepthMaterial`.
- **Render targets on meshes.** three's `WebGPURenderer` shows a render target's texture upside down on a mesh's UVs compared to WebGL, and doesn't pick up a `map` that replaced a plain texture after the material compiled.
- **Generator time.** Generators and `onFrame` get `performance.now()` on WebGPU; WebGL `SparkRenderer`'s own timer stays at 0.
