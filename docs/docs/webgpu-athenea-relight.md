# athenea relight (WebGPU)

A cloud that athenea baked with a TX transfer (`athenea mesh2splat`, converted to `.athc` by `usd-athc`) can be lit by any dome, an analytic sun and lights, the way athenea's raster lights it. The shading is athenea's own `splat_relight.slang` (vendored, called as it is); the plugin builds its inputs from the mesh and prepares the sky the way athenea's `technique::Environment` does.

```js
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { plugins, atheneaRelightPlugin, atheneaOutputPlugin, splatRequiredLimits } from "@sparkjsdev/spark/webgpu";

// The relight pass binds ten storage buffers: ask the adapter for them.
const renderer = new THREE.WebGPURenderer({ requiredLimits: splatRequiredLimits(adapter) });
const body = new SplatMesh({ url: "body-t64-gz.athc", extSplats: true });
const relight = atheneaRelightPlugin({ hdri: { width, height, data } }); // linear lat-long, top row first
const host = new plugins.PluginHost({ capabilities: splats.capabilities });
host.register(relight).register(atheneaOutputPlugin({ view: "agx" })).attach(splats);

relight.set({ rotation: 0.5, intensity: 1.2 });                       // the dome
relight.set({ sun: { direction: [0.3, 0.8, 0.5], intensity: 3 } });   // a UsdLux distant light
relight.setIor(glassHead, 1.5);                                       // per cloud
relight.setCatcher(ground, true);                                     // a shadow catcher
```

`examples/webgpu/athenea-relight.html` shows the OpenChessSet pawn (`?base=` where its files are) under the HDRIs of its `pawn.json`, with the dome switchable and turnable, an analytic sun and athenea's display transform.

## What runs, and when

Before generate, a pass of the plugin (`passes`):

1. **The sky** (`AtheneaSky`, `slang/athenea_adapter/env.slang`), when the image, its turn or its intensity change. athenea's `env_sun`, `env_project`, `env_residual` and `env_prefilter` (txf 89a04d9), with athenea's lat-long area mips, reading the dome from an rgba32float chain where athenea reads its texture table:
    - the sun is found (median and peak), summed over the projection's texels and handed over as a light (`envSun`);
    - sixteen harmonics of the sky without it (`envSh`);
    - the sky without its sun, with its own mips, convolved with GGX into eight levels of an octahedral map (`envTexels`), at athenea's base side for the image (1024 for a 2k sky).
2. **The kept terms** (athenea's `txCache`, `splatTransferViewless`), when the sky, the lights, a mesh's place or its data change: the body's light under the sky, the reflected field's coupling to it, and the sun's share.
3. **The colour of each splat** (`relitSplat`), when the eye moves or anything above changes.

generate's colour plugin (`relight_colour.slang`, preset `athenea-relight`) then takes that colour, in linear light, encoded for Spark's draw. Draw into a HalfFloat target and end with `atheneaOutputPlugin`, as with the raster plugin.

The streams are read from the mesh's attribute pool by name (`normalOct`, `pbr`, `lobes`, `emission`, `transfer`, `shadowBits`). athenea's functions index the pool directly: a transfer starts at its attribute's word times two (halves), the cells at the `shadowBits` word. A paged pool (`gpuBuffer`) is bound as it is.

Per transfer, the kernel is athenea's: a TX transfer with cells (`kTransfer` 2: field, cells, sun through the cells, light bounce), the first transfer (1), or none (0: the dome's harmonics and map and the lights alone).

## Validation

- **Sky**, `test/gpu/atheneaSky.test.ts`: athenea's `test_environment.cpp` on Dawn with its `environment_check` kernels. A sky of one colour reads `pi L` to 0.09 %; the halves read up 1.000, down 0.000, rough 0.995 (athenea: 0.997); the sun's irradiance matches the closed form; levels 2-6 hold the source's light to 0.07 %, and the sky without its sun once the sun is taken out; three lamps are not a sun.
- **Materials**, `test/gpu/atheneaRelight.test.ts`: twelve materials (matte, polish, conductor, Schlick metal, lacquer with a coat, sheen, tinted specular, glass, thin sheet, emitter, stored normal, mirror) with a TX transfer, under a dome with a sun, an analytic sun and a sphere lamp, on an object turned, moved and scaled. Against splat_project's `relitToward` on athenea's own buffer layout: within 3.4e-4 relative (the kept terms are halves). Shadow catchers against `catcherOpacity` within 2e-3.
- **athenea's frames** (`ATHENEA_BATTERY=1`, on a machine with athenea's renders), measured as athenea measures them:

| scene | ours / athenea raster | ours / path traced | athenea raster / path traced |
|---|---|---|---|
| paint ball green, san giuseppe | ratio 1.007, relMSE 0.0033 | 0.739, 0.0125 | 0.735, 0.0103 |
| paint ball white, san giuseppe | 1.030, 0.0056 | 0.769, 0.2577 | 0.747, 0.2603 |
| green, white dome | 0.961, 0.0035 | 0.828, 0.0104 | 0.861, 0.0067 |
| white, white dome | 0.994, 0.0035 | 0.863, 0.0424 | 0.868, 0.0387 |
| floating ball, sun only | 1.003, 0.0046 | 0.948, 0.1688 | 0.945, 0.1628 |
| pawn body (autoshop_01) | 1.004, 0.0034 | 1.008, 0.0428 | 1.004, 0.0400 |
| pawn glass top | 0.917, 0.0866 | 0.710, 0.1604 | 0.775, 0.0598 |

  The balls are measured in athenea's box (192 x 192 + 96 + 80 of 384 x 384), the pawn over each material's pixels (validate's measure). athenea's raster is the s95 TX build.

## Differences from the native raster

- **Per-pixel slope and the sharp coat** need the cloud's curvature, which neither `.athc` v2 nor v3 carries: every lobe is read at the splat's centre (`noSlope`), as athenea draws a cloud without curvature. The same missing curvature is why a glass head bends once rather than as a lens (the pawn's top, ratio 0.917).
- **The colour's harmonics** beyond the constant term are not read (the clouds athenea converts from meshes are degree 0).
- **Rays**: nothing is traced (no shadow factors, traced reflections or glass exits), as in athenea's raster.
- **IES profiles**: the lights module's tables are bound but empty; lights are spheres, disks, rects, cylinders and distant lights.
- **One dome**, as athenea's raster keeps (the first prepared dome).
- **Footprint**: Spark's splat shape (see the raster page); at the edges of a ball this shows as a lighter rim.
