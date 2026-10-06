# Per-Gaussian attributes (WebGPU)

Besides centre, scales, rotation, colour and spherical harmonics, splats can carry any number of extra **attributes**: a normal, a semantic label, an id, a feature vector. Each attribute has a name, a storage format and a number of components. Attributes load from PLY and `.rad` files, survive LoD building with a merge rule of your choice, and can be read by any Slang kernel and shown or picked in the WebGPU draw.

The code is in `src/webgpu/attributes/` (TypeScript), `slang/core/attrib.slang` (decoding on the GPU) and `rust/spark-lib/src/attrib.rs` (LoD merging and `.rad` chunks). `examples/webgpu/attributes.html` shows all of it.

## The schema

An attribute is described by an `AttributeSpec`:

```typescript
interface AttributeSpec {
  name: string;
  format: AttribFormat;   // "f32" | "f16" | "unorm8" | "snorm8" | "u8" | "u16" | "u32"
  components: number;     // 1 or more
  lodMerge?: LodMerge;    // "weightedMean" | "normalizeMean" | "max" | "mode" | "first"
  toDraw?: boolean;       // readable in the draw shaders
  direction?: boolean;    // a 3-vector that turns with its object (normals)
}
```

### Formats

| Format | Bytes per component | Values you write | Value read in Slang |
| ------ | ------------------- | ---------------- | ------------------- |
| `f32` | 4 | any float | the float |
| `f16` | 2 | any float | the float, at half precision |
| `unorm8` | 1 | `[0, 1]` | `[0, 1]` in steps of 1/255 |
| `snorm8` | 1 | `[-1, 1]` | `[-1, 1]` in steps of 1/127 |
| `u8` | 1 | integers `0..255` | the integer |
| `u16` | 2 | integers `0..65535` | the integer |
| `u32` | 4 | integers | the integer |

Components are packed little-endian at their format's size, and each attribute starts on a word boundary. `snorm8 × 3` is a compact normal, `u8` or `u16` a label, `f16 × 8` a feature vector.

### Setting attributes

Attributes of a set of splats live in an `AttribPool`, one column per attribute, where index `i` is splat `i`. On a `GpuSplatSource` (the GPU copy of a PackedSplats or ExtSplats), `setAttribute` creates the pool if needed:

```typescript
source.setAttribute("normal", normals, "snorm8", 3, { toDraw: true, direction: true });
source.setAttribute("label", labels, "u8", 1, { toDraw: true });
source.setAttribute("feature", features, "f16", 8, { toDraw: true });

source.attribs.getAttribute("label", 42); // [3]
```

`data` holds `count × components` values. When `format` is left out it follows the array type, and `lodMerge` defaults to `normalizeMean` for directions, `mode` for the integer formats and `weightedMean` otherwise. `setAttribute` bumps the source's version, so the renderer regenerates; after changing an `AttribPool` directly, call the renderer's `markDirty()`.

## Loading from files

### PLY

A binary PLY's vertex properties that Spark doesn't use for the splat itself (anything other than position, colour, opacity, scale, rotation, `f_dc_*` and `f_rest_*`) become attributes:

- `nx`, `ny`, `nz` become `normal`, a 3-component direction. An all-zero normal is dropped: 3DGS trainers write `nx = ny = nz = 0`.
- `name_0`, `name_1`, ... of one type become `name` with that many components.
- Anything else becomes an attribute of its own, with one component.

Unsigned integer properties keep their size (`uchar` → `u8`, `ushort` → `u16`, `uint` → `u32`); everything else is stored as `f32`. All PLY attributes are `toDraw`.

```typescript
const packed = new PackedSplats({ url: "scene.ply" });
await packed.initialized;
packed.extra.attribs;   // the decoder's specs and values (AttribPool.from() packs them)
const source = GpuSplatSource.fromPackedSplats(device, packed); // source.attribs
```

The WASM decoder reads the extra properties (`rust/spark-lib/src/ply.rs`), as `f64` values, and every result of the loader worker (plain, `lod: true` packed or ext) carries the decoder's specs and values as `extra.attribs` (`AttribValues`): the worker imports nothing from `src/webgpu`. `AttribPool.from()` packs them on the WebGPU side (`GpuSplatSource`, `WgpuSplatPager`), so WebGL users never pack them. The decoder keeps an all-zero normal out of its output from the first batch on, adding it zeroed if a non-zero one shows up later: a standard 3DGS PLY costs nothing extra (500k splats: 19.1 → 7.6 MB of WASM memory). `GsplatArray`/`CsplatArray` expose `get_attrib_specs()` and `get_attribs()` for JS callers of `set_attribs()`.

### .rad

A `.rad` file lists its attribute schema in its metadata (`attributes: [{ name, format, components, lodMerge }]`), and each chunk stores every attribute as an `attrib` property named after it, in its own format (`f32`, `f16`, `r8` for `unorm8`, `s8` for `snorm8`, `u32` for the integer formats), gzip-compressed. `build-lod` writes the attributes of a PLY into the `.rad` it builds. A `.rad` loaded as PackedSplats or ExtSplats carries them in `extra.attribs`.

A `.rad` file doesn't store the `direction` flag: a 3-component attribute with `normalizeMean` is read back as a direction. Files with attributes need a decoder that knows the `attrib` property; older ones reject it.

Paged `.rad` streaming (`PagedSplats` through `WgpuLod`) pages attributes with the splats: `WgpuSplatPager` keeps a third pool (`PagedAttribPool`), with the schema from the `.rad` metadata (or the first chunk), and writes each chunk's attributes at its page base, converted to the schema's formats. `SplatAttributes` binds that pool directly instead of uploading it.

### Paging by stream group

The attribute pool is one storage binding, so it is sized within a byte budget: the device tier's (`tiers.ts` `attribBudget`: 128 MiB at T1, 768 MiB at T2, 1.5 GiB at T3), never more than the device's binding size. `attributes/attribPaging.ts` plans it by **group** of streams:

| Group | Attributes | Fetched | Device tier |
| ----- | ---------- | ------- | ----------- |
| `core` | every attribute outside the groups below (a `.rad`'s, a `.athc`'s `athcGroup`) | with the splats | any |
| `material` | `normalOct`, `emission`, `pbr`, `lobes` | `.athc` v3 section `MATL` (data tier 2) | T1 |
| `relight` | `shadowBits`, `transfer` | `SHAD`, `TXDI`, `TXIN`, `TXFD` (data tier 3) | T2 |

In that order, a group that fits the rest of the budget at the page pool's full capacity is interleaved in the pool and arrives with its pages, as before. One that does not is **paged**: it gets as many pages as the budget leaves (`slots`), in a region of its own per attribute (`ATTRIB_PAGED` in `slang/core/attrib.slang`): a page table (one word per page of the pool: its slot, or `ATTRIB_NONE`) and the slots' records. A group no page of which fits is left out.

`AttribResidency` gives a paged group's slots to the pages the LoD traversal ranks first (the pager's fetch priority, each time it drives its fetchers). A page that arrives with the group's data takes a free slot; a wanted page without it is **upgraded**: `PagedSplats.fetchStreams` fetches just that group (a `.athc` v3 chunk: one Range request of its sections; v2: of its arrays; a merged page or a `.rad` chunk: decoded again), and takes a free slot or one held by a page no longer wanted. Nothing waits: a splat whose page is not resident reads as zeros, `attribResident(pool, id, splat)` says so, and readers fall back (`slang/athenea_adapter/athc.slang`: the frame's normal, a capture's plain pbr and lobes, no emission, every direction open, `athcTransferResident` false so a relight keeps the captured colour).

Which streams of the groups load: `pagedAttributes.attributes` (`"all"` or names), else the `PluginHost`'s `attributeDemand()` (the `requires.attributes` and `requires.reads` of the plugins the device runs; the athenea raster reads `emission` and `athcGroup`), else, with no plugins, everything but the relight streams. The transfer can be kept shorter (`transferForm`): its layouts are prefixes of each other (`transfer_layout.slang`): `"direct"` keeps 16 values (9 at degree 2), `"indirect"` 64 (36) with the indirect half, `"full"` (default) all 112 (84) with the reflected field; the shorter form is read as that layout.

```js
new SparkRenderer({
  renderer, backend: "webgpu",
  pagedAttributes: {
    budgetBytes: 512 * 1024 * 1024, // default: the tier's, within the binding size
    pages: { relight: 24 },          // pages a group keeps (overrides the budget)
    attributes: "all",               // default: the plugins' demand
    transferForm: "indirect",
  },
});
```

A TX cloud (every_stream's streams: `athcGroup` 8 B, material 24 B, shadow bits 32 B, transfer 224 B a splat) over the default 256 pages would need 4.8 GB whole. Planned:

| Device | Pages | core + material | relight | Pool |
| ------ | ----- | --------------- | ------- | ---- |
| T1 (128 MiB), SH3 | 42 | whole, 88 MB | not loaded (T2) | 88 MB |
| T2 (768 MiB budget) | 256 | whole, 537 MB | 15 pages (0.98M splats), 252 MB | 789 MB |
| T2, `"indirect"` | 256 | 537 MB | 25 pages, 262 MB | 799 MB |
| T2, `"direct"` | 256 | 537 MB | 63 pages, 264 MB | 801 MB |
| T3 (1.5 GiB budget) | 256 | 537 MB | 63 pages (4.1M splats), 1.06 GB | 1.6 GB |

## LoD merge rules

When a LoD tree is built (`lod: true`, or `build-lod` for `.rad`), each parent splat merges its children, and each attribute is merged by its `lodMerge` rule with the same normalized weights the splat merge uses (area × opacity):

| Rule | Parent value |
| ---- | ------------ |
| `weightedMean` | the weighted mean of each component |
| `normalizeMean` | the weighted mean, renormalized to unit length. Children are first flipped into the heaviest child's hemisphere, since a splat's normal has no inside or outside |
| `max` | the maximum of each component |
| `mode` | the whole value with the largest total weight (labels); ties go to the earliest child |
| `first` | the first child's value |

Values are merged as `f64`, so integer labels and ids up to 2<sup>53</sup> merge exactly. The tiny, Bhattacharyya and quick LoD builders all carry attributes. The LoD splats of a PackedSplats or ExtSplats built with `lod: true` carry the merged attributes in `lodSplats.extra.attribs`, and `WgpuLod` draws them with their LoD indices.

## Reading attributes in Slang

On the GPU an attribute pool is one `uint` buffer that carries its own descriptor table, so one compiled kernel serves every schema:

```
[0] strideWords  [1] numAttribs  [2] headerWords  [3] count (splats)
[4 + 4k ..]      attribute k: offsetWords, format, components, flags
[headerWords + splat * strideWords + offsetWords ..]  attribute k's data
```

`AttribPool.pack()` builds it, and `slang/core/attrib.slang` reads it:

```hlsl
import core.attrib;

StructuredBuffer<uint> attribs;

// Components 4·comp4 .. 4·comp4 + 3 of attribute `id` of `splat`, as floats
// (missing components are 0).
float4 feature = readAttrib(attribs, featureId, splat, 1u);
// One component's stored bits: for u8, u16 and u32 the integer, exactly.
uint label = readAttribU(attribs, labelId, splat, 0u);
```

`writeAttrib` writes into a `RWStructuredBuffer<uint>` pool, and `attribDesc`, `attribCount` and `attribSplats` read the table. Attribute ids are resolved by name on the host: `AttribPool.id(name)` for a source pool, `SplatAttributes.drawId(name)` for the draw pool. Integer formats come back from `readAttrib` as floats of their values; use `readAttribU` for labels and ids beyond 2<sup>24</sup>. An unknown id reads as zeros.

## Drawing and picking

`SplatAttributes` (`src/webgpu/attributes/SplatAttributes.ts`) adds attributes to a `WgpuSplatRenderer`:

- After generate, `slang/kernels/attrib_gather.slang` copies every mesh's `toDraw` attributes into a **draw pool** indexed like the accumulator. Its schema is the union of the meshes' `toDraw` attributes (the first mesh with a name sets its format); a mesh without one gets zeros. It follows the mesh's LoD indices, converts formats where they differ, and turns direction attributes into world space.
- The draw uses `slang/draw/splat_attrib_draw.slang`, which reads up to four **slots** of four components each into flat varyings. Colour modes use them: `splat` (the splat's own colour), `label` (a palette by the integer in `colorSlot`, with an optional `highlight`), `relight` (Lambert lighting with `colorSlot` as the normal), `project` (`rgb = offset + rows · (slot, slot + 1)`, for feature vectors) and `value` (the slot as a colour).
- Extra render targets (multiple render targets in one pass): `id`, an `rgba8unorm` target holding the front-most splat whose alpha reaches `pickAlpha`, and `slot0` / `slot1`, `rgba16float` targets with slots 0 and 1 alpha-composited like the colour.

`pick(x, y)` reads the id target at a pixel of the last frame and returns the mesh, the source splat index (through its LoD indices) and the accumulator slot, or `null`.

`SplatAttributes` works with `WgpuSplatRenderer.render()`, which opens its own render pass. `SparkRenderer` on WebGPU draws in three's pass and doesn't use it.

## Example

Six labelled shapes whose splats carry a normal, a label and an 8-component feature, coloured by label and picked by clicking (from `examples/webgpu/attributes.html`):

```typescript
import * as THREE from "three/webgpu";
import { GpuSplatSource, SplatAttributes, WgpuSplatRenderer } from "@sparkjsdev/spark/webgpu";

const renderer = new THREE.WebGPURenderer();
await renderer.init();
const splats = new WgpuSplatRenderer(renderer);
const attributes = new SplatAttributes(splats, {
  // Slot 0: label, 1: normal, 2-3: the feature's two groups of four.
  slots: ["label", "normal", { name: "feature", comp4: 0 }, { name: "feature", comp4: 1 }],
  colorMode: "label",
  colorSlot: 0,
  targets: { id: true },
});

const source = GpuSplatSource.fromExt(splats.device, extA, extB, count);
source.setAttribute("normal", normals, "snorm8", 3, { toDraw: true, direction: true });
source.setAttribute("label", labels, "u8", 1, { toDraw: true });
source.setAttribute("feature", features, "f16", 8, { toDraw: true });
splats.add(source, new THREE.Object3D());

// Relight by the normal in slot 1 instead:
// Object.assign(attributes.options, { colorMode: "relight", colorSlot: 1 });

renderer.domElement.addEventListener("click", async (event) => {
  const ratio = renderer.getPixelRatio();
  const hit = await attributes.pick(event.offsetX * ratio, event.offsetY * ratio);
  if (hit) {
    const [label] = hit.mesh.source.attribs.getAttribute("label", hit.index);
    attributes.options.highlight = label;
  }
});

renderer.setAnimationLoop(() => {
  renderer.setRenderTarget(target);   // a RenderTarget with a DepthTexture
  renderer.render(scene, camera);
  splats.render(camera, target);
  renderer.setRenderTarget(null);
  output.render(renderer);            // a QuadMesh showing target.texture
});
```

With `?source=ply` the example writes the same scene as a PLY with `nx`/`ny`/`nz`, `label` and `feature_0`..`feature_7` properties and loads it through `PackedSplats`, which gives the same attributes from the file.
