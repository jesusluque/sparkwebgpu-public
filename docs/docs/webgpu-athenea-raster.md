# athenea raster (WebGPU)

On the WebGPU backend, a `.athc` written by athenea can be drawn the way athenea's own rasteriser draws it: colours in linear light, emission, per-group overrides, and athenea's display transform (exposure in stops, then Standard, AgX or ACES 2.0). It is built from two plugins on the `PluginHost` and runs alongside Spark's renderer; without them nothing changes.

```js
import { PagedSplats, SplatMesh, SparkRenderer } from "@sparkjsdev/spark";
import { fx, plugins, atheneaRasterPlugin, atheneaOutputPlugin } from "@sparkjsdev/spark/webgpu";

const spark = new SparkRenderer({ renderer, pagedExtSplats: true });
const mesh = new SplatMesh({ url, paged: new PagedSplats({ rootUrl: url, athcKeepLinear: true }) });

const raster = atheneaRasterPlugin();
const output = atheneaOutputPlugin({ view: "agx", exposure: 0 });
// Once spark.webgpu.splats exists (after the first frame):
const host = new plugins.PluginHost({ capabilities: splats.capabilities });
host.register(raster).register(output).attach(splats);
host.applyFx(chain); // an fx.FxChain over a HalfFloat RenderTarget

// Each frame: render into the half-float target, then
chain.applyToRenderTarget(renderer, target);
```

`examples/webgpu/athenea-raster.html` puts these pieces together.

## Linear light

athenea blends every splat in linear Rec.709 (`splat/frame.slang`). A capture's colours are sRGB-encoded and are decoded one splat at a time (`cloudLight`). A cloud that athenea wrote, flagged `linear` (bit 1 of the `.athc` flags), already holds light.

- The raster plugin (`slang/athenea_adapter/raster.slang`, an `ISplatColour`) converts each splat's colour to linear light in generate. It uses athenea's `cloudLight`: an sRGB decode, unless the stored colours are linear already.
- By default the `.athc` decoder encodes a linear cloud to sRGB for Spark's blend. `PagedSplats({ athcKeepLinear: true })` keeps the colours linear instead: every ATHV page carries the `ATHV_KEEP_LINEAR` decode flag. The plugin reads `PagedSplats.athcStoredLinear`, or takes `raster.setStoredLinear(mesh, true)`.
- Linear colours need `pagedExtSplats` (half floats). Otherwise 8-bit pages lose the darks and clip anything above 1.
- Before the draw, the plugin writes the colour back with Spark's transfer function (gamma 2.2). The draw undoes it exactly when it draws into a linear target, so the blend sums athenea's light. On an 8-bit sRGB canvas, the encoded colours blend the way Spark (and athenea's `displayBlend`) blends them.

Without `athcKeepLinear`, the plugin still undoes the decoder's encode with the same sRGB curve. That is exact for the base colour and correct to first order for the harmonics (the decoder scales them by the curve's slope).

## Emission

The `emission` stream (RGB9E5, athenea's `unpackRgb9e5`) is added to the linear colour, times `raster.emission` (default 1). In athenea's native raster, emission is added only by the relit path, and only where a body is not `lit`: a radiance bake already holds it. For a cloud baked that way, set `raster.setEmission(0)`.

## Ids, picking and overrides

A `.athc` keeps no Cryptomatte id. athenea's `cloudCrypto`, the prim id its matte and its override table use, comes from USD and is not written to the file. Since phase 2, every decoded `.athc` instead carries `athcGroup` (u32 × 2). It holds `[lo, hi)`, the finest LoD groups an element covers:

- a splat covers its own group, `[g, g + 1)` (the file's chunk tail);
- a merged level node covers its subtree, because groups are Morton-sorted and a subtree's groups are therefore contiguous.

So the ids are spatial cells of athenea's octree, not prims. A group at any level is a range of finest groups.

- **Picking.** `pickAtheneaGroup(attributes, x, y)` reads `SplatAttributes`' id target (the existing MRT picking path) and returns the front-most splat and its `groups`. For a paged pool, the group is read back from the GPU.
- **Overrides.** `raster.setOverrides(mesh, [{ groups: [lo, hi], tint, replace, opacity }])` sets the overrides for one mesh, up to 16 of them. They follow athenea's `splat_override.slang`:
  - `tint` multiplies the colour, or replaces it when `replace` is set (`splatOverrideColour`);
  - `opacity` scales alpha, and 0 hides the range;
  - the first row that applies wins.
  
  A row applies to an element whose whole range lies inside it. A coarse LoD node that straddles the edge of a range therefore keeps its colour until the LoD refines past it. A hidden splat is picked through, to whatever is behind it.

## Output transform

`AtheneaDisplay` is an aofx effect, so it runs in an `FxChain`. It applies athenea's display transform (`technique/display.slang`, vendored together with `aces2.slang` and `aces2_prepare.slang`):

1. The premultiplied picture is composited over `background`.
2. It is multiplied by 2^`exposure`.
3. It goes through a view: Standard (clips), AgX (Wrensch's fit of Sobotka's sigmoid) or ACES 2.0 (the Academy's output transform, with its tables built on the GPU by athenea's prepare passes for `peakLuminance` and the display's primaries).
4. It is encoded for the display: sRGB, BT.1886, Display P3, or linear P3 with headroom.

By default the effect writes a value meant for an output that sRGB-encodes what it receives (three's canvas output). Set `encoded: 1` to write code values for a unorm texture that is shown as it is. OpenColorIO views are native-only.

`atheneaOutputPlugin` adds the effect as the last step of the host's chain. It publishes exposure, view and display controls, and the step's parameters stay live.

## Validation

There are no reference renders of athenea's fixtures in its build tree, so the checks run athenea's own math on Dawn:

- Display transform (`test/gpu/atheneaDisplay.test.ts`):
  - bit-identical to athenea's `displayColour`, called with its own globals, on 509 HDR colours for 9 combinations of view, display and exposure;
  - Standard and AgX within 2.3e-6 of the formulas in JS;
  - ACES 2.0 shows 18% grey at 0.0999995 of reference white (athenea's test: 0.0999993).
- Raster (`test/gpu/atheneaRaster.test.ts`):
  - the centre pixel of a four-splat stack is within 1.0e-3 (relative; half floats) of `splat_blend`'s front-to-back walk, with the hardware and tile rasterisers and with sRGB and linear clouds;
  - with overrides, a hidden range and emission ×2, it stays within 2e-3;
  - Spark's own colour path misses by more than 1e-2.

## Differences from the native raster

- **Footprint.** Spark's splat shape is kept: anti-aliasing blur, `maxStdDev` cut-off, and no 0.99 alpha cap or 1/255 cut. The two rasterisers differ at splat edges, not in the colour path.
- **Relighting.** See [athenea relight](webgpu-athenea-relight.md).
