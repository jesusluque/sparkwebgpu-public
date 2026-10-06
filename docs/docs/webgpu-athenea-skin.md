# athenea skinning on WebGPU

A cloud a skeleton carries, as athenea makes one (`athenea mesh2splat
--skinned`, with `--transfer` its transfer as two zonal lobes in each
gaussian's frame), posed every frame on the GPU by athenea's own skinner and
relit by athenea's relighting (plan-athenea.md, phase 5).

```js
import { atheneaSkinPlugin, atheneaRelightPlugin, readAthcSkeleton, plugins } from "@sparkjsdev/spark/webgpu";

const url = "sparrow.athc";
const skeleton = await readAthcSkeleton({ url });           // the ATSK blob (athc-v3.md)
const bird = new SplatMesh({ url, extSplats: true });       // decodes skinInfluences / skinGradients
const skin = atheneaSkinPlugin({ skeleton });                // plays clip 0, looping
const relight = atheneaRelightPlugin({ hdri, frame: stage });
host.register(skin).register(relight).attach(spark.webgpu.splats);
skin.set({ playing: false, time: 20 });                      // a time code
skin.set({ jacobian: false });                               // the blend of the joints' linear parts alone
skin.set({ bindPose: true });                                // the cloud as converted
```

Example: `examples/webgpu/athenea-skin.html` (the skinned, relit sparrow;
`?url=` any skinned `.athc`, falls back to the test's skinned corner).

## From athenea to the web

```sh
athenea mesh2splat SparrowBird.usda --skinned --transfer --time 1 \
    --bake-samples 64 --bake-extra 0 --bake-bounces 1 --resolution 160 -o sparrow160tx.usdc
usd-athc sparrow160tx.usdc sparrow160tx.athc --clip flight [--add-clip land=other.usdc] [--gzip]
```

`usd-athc` reads the UsdSkel binding and athenea's `AtheneaSplatSkinningAPI`
off the `ParticleField` and writes the `SKIN` section (athenea's packed
influences and its weight gradients) and the skeleton with one clip of the
cached `skinningXforms` ([the format](athc-v3.md#a-skinned-cloud-skin-and-the-skeleton)).
A cloud converted without `--transfer` (no bake) is carried the same way and
shaded by its material.

## What runs, and when

Per frame, when the time code, the clip or the options changed
(`atheneaSkinPlugin`, `src/webgpu/athenea/skinPlugin.ts`):

1. The joints' transforms at the time code: each matrix element linearly
   between the clip's two samples about it (USD's interpolation of matrix
   samples, which is what athenea's Hydra hands its skinner), held outside.
2. `slang/athenea_adapter/skin.slang`, a splat a thread, before the relight
   pass and generate: athenea's `scene/splat_skin.slang` `splatSkin`
   (vendored at txf 89a04d9), restated over Spark's buffers --
   - the centre by the linear blend of the joints (`X_k G p` weighted);
   - the frame by the whole Jacobian of that blend, `J = Σ w_k A_k L(G) +
     Σ (X_k q) ∇w_kᵀ` (7dff879), on the gaussian's two rest axes, and the
     posed in-plane covariance `J E S² Eᵀ Jᵀ` decomposed exactly into the
     two axes and sizes (the third size is the conversion's);
   - the shading normal by the cofactor of the same Jacobian.
3. The posed splats go back into the mesh's source (ExtSplats or
   PackedSplats, in place; a copy of the rest splats is kept), the posed
   normals into the attribute pool (`normalOct`), and the source's version
   is bumped: generate, the sort, the draw and the relight read the pose.
   The relight's `splatTransferFrame` turns a zonal transfer (and its open
   directions) with the posed frame, as athenea's does.

The kernel is restated rather than dispatched as vendored because athenea's
binds eleven storage buffers (its shutter's second pose and the motion it
writes among them) and Chrome on a Mac gives a compute stage ten; the test
runs athenea's own kernel against it (below).

The attribute pool goes to the GPU whole (`AttribPool.gpuBuffer`), so every
reader binds the buffer the skinner writes. A paged cloud (`PagedSplats`) is
drawn in its bind pose: its pages do not bring `SKIN`.

## Validation

`test/gpu/atheneaSkin.test.ts` (Dawn):

- **athenea's own kernel.** `slang/tests/athenea_splat_skin.slang` compiles
  the vendored `splat_skin.slang` unchanged; the test aliases its
  `skinningXformsEnd` and `motion` bindings as athenea's host binds them when
  no shutter is open (so it fits Dawn's ten) and runs it on the same 4096
  splats (athenea's own case from 7dff879: a strip bent a quarter turn
  across a ramp of weights, plus a sheared and a non-uniformly scaled
  joint). Posed centres: identical (max 6e-8). The posed covariance against
  athenea's statement of it in float64 (`J E S² Eᵀ Jᵀ`): p99 6.5e-3 here,
  4.1e-3 athenea's kernel (both what f16 sizes and a 10-bit rotation leave);
  between the two, p99 8e-3. Normals: 6e-5 rad from float64. None off by
  more than 3% (athenea's metric); with the Jacobian off, 2048 of 4096 are
  (athenea's own number for its old kernel).
- **athenea's skinned corner** (`test/fixtures/athc/skinned_corner.athc`,
  athenea's `tests/data/skinned_corner.usda` converted by athenea with its
  test's arguments): decoded by the WASM reader, posed at time 1 by the
  plugin and relit, it shades as the rest cloud turned by the joint's
  rotation (athenea's fe1f6e6 check): worst 9e-4 relative over 1397
  elements.

`test/gpu/atheneaSkinFrames.test.ts` (opt-in, `ATHENEA_SKIN_FRAMES=1`, data
in `~/luc/sparkwebGPU/athenea-skin-data`): the relit sparrow
(sparrow160tx: 81 785 splats, 609 joints, 61 samples) against athenea's
Hydra raster of the same `.usdc` (`athenea stage --technique raster`, a white
dome, `sparrow_ref.usda`'s camera), over the bird's pixels:

| time code | relMSE (posed) | mean ratio | blend only | bind pose | t ± 2 |
|---|---|---|---|---|---|
| 1 | 0.0053 | 1.010 | 0.0211 | 7.93 | |
| 20 | 0.0015 | 1.010 | 0.0320 | 4.65 | 1.11 / 5.07 |
| 40 | 0.0016 | 1.009 | 0.0285 | 6.16 | |

## Spark's SplatSkinning

Spark's own `SplatSkinning` (dual quaternion, and linear blend on
covariance splats) runs on WebGPU through the dyno backend (its WGSL
versions in `src/SplatSkinning.ts`); see the dyno parity cases
(`dyno:skin`).
