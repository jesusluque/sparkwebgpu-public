# athenea's spheres: the whole pipeline against athenea

A battery that runs athenea's TX balls through everything a page runs, and compares the result with athenea in float (linear light, before any display transform):

1. athenea's USD cloud (`mesh2splat --transfer`, builds tx-s94/s95, `tx-transfer-complete` 262f918) converted by `usd-athc` with the full transfer (112 values), the open-direction cells, the material layers (lobes) and the curvature;
2. the `.athc` decoded by the WASM decoder as a whole-file `SplatMesh` decodes it (`decode_to_extsplats`);
3. relit by `atheneaRelightPlugin` and drawn by `WgpuSplatRenderer` into a half-float target, with `pixelDetail` (athenea's raster always shades the sharp lobes and the slope per pixel).

It's in `test/gpu/atheneaSpheres.test.ts` and runs one sphere at a time at 384 × 384:

```sh
for c in green white; do usd-athc ~/luc/athenea-renders/spheres-web/paint-balls/ball_$c.usdc $DIR/$c.athc --json $DIR/$c.json; done
usd-athc ~/luc/athenea-renders/paint/s88/float.usdc $DIR/float.athc
for m in paint chrome rubber glass; do usd-athc ~/luc/athenea-renders/spheres-web/tx-balls/${m}_tx.usdc $DIR/tx_$m.athc --json $DIR/tx_$m.json; done
ATHENEA_SPHERES=$DIR [ATHENEA_SPHERES_OUT=<dir>] npm run test:gpu -- atheneaSpheres
```

The test takes about two minutes. `ATHENEA_SPHERES_OUT` writes every frame (ours, athenea's raster, the path-traced ground truth) as PFM (float) and the numbers as `spheres.txt`.

## The paint balls against athenea's raster and the path-traced ground truth

The paint balls are metallic paint at roughness 0.3421, with no coat, lying on a grey floor. athenea's stage camera is `/Camera` in `cloud_<c>.usda`: 35 mm at (0, 0.4, 4.2), pitched −8°. The skies are the full-resolution `san_giuseppe_bridge.hdr`, athenea's white dome, and `sun_only.exr` (1024 × 512, a 4 × 4 sun at 36000 on black). Each ball is compared with athenea's raster (s95) and with the unbiased path trace (s94; s83 for the white dome).

Two regions are measured:

- **Box:** athenea's own measure, 192 × 192 + 96 + 80. Its corners fall outside the ball, on the floor, which athenea draws as a mesh and we don't draw at all. There, our frame is laid over athenea's own frame, so the corners compare like with like.
- **Ball:** the pixels whose rays meet the sphere of radius 0.97.

`<c>_mask.exr` in `spheres-web` isn't usable: it is 1.0 in every channel of every pixel.

The table below is per pixel (`pixelDetail`), with the ratio of means and athenea's relMSE, (a − b)² / (b² + 0.01):

| ball, sky | region | ours / athenea raster | ours / GT | athenea raster / GT (NOTES) |
|---|---|---|---|---|
| green, san giuseppe | box | **1.002**, 0.0000 | 0.736, 0.0104 | 0.735, 0.0103 (0.735) |
| | ball | 1.002, 0.0000 | 0.698, 0.0119 | 0.696, 0.0119 |
| white, san giuseppe | box | **1.003**, 0.0000 | 0.749, 0.2599 | 0.747, 0.2603 (0.747) |
| | ball | 1.003, 0.0000 | 0.726, 0.2452 | 0.724, 0.2458 |
| green, white dome | box | **0.995**, 0.0001 | 0.857, 0.0070 | 0.861, 0.0067 (0.861) |
| | ball | 0.996, 0.0000 | 0.849, 0.0084 | 0.852, 0.0082 |
| white, white dome | box | **0.997**, 0.0000 | 0.865, 0.0389 | 0.868, 0.0387 (0.868) |
| | ball | 0.997, 0.0000 | 0.870, 0.0351 | 0.872, 0.0348 |
| floating, sun only | box | **0.997**, 0.0000 | 0.943, 0.1613 | 0.945, 0.1628 (0.945) |
| | ball | 0.997, 0.0001 | 0.946, 0.1666 | 0.949, 0.1683 |

Ours over the ground truth is athenea's own ratio, within 0.004. What both lose against the path tracer is athenea's known loss: the floor's reflection in the lower half of a ball that stands on it (TX096). The floating ball, with no floor, is at 0.945.

Without `pixelDetail`, the sharp lobes are read once at each splat's centre and spread over the footprint. Then ours / athenea is 1.027–1.032 for the balls under san giuseppe and 1.003–1.011 under the sun. Under the white dome it's unchanged, because a sky of one colour has nothing for the mirror to turn to.

## The TX balls: per splat against athenea's relitSplat

athenea stores no raster frames for `tx-balls`: its test `tx_conversions_render_like_the_mesh` path-traces the mesh on every run. Rendering athenea's raster here would mean running athenea on the GPU, so these balls are compared per splat instead.

The reference is athenea's `relitSplat`, run on athenea's own buffer layout (`slang/tests/athenea_relight.slang`). That layout is built from the USD arrays alone, with no usd-athc and no WASM:

- the material packed as `splat_encoding.slang` packs it;
- the lobes as `packing.slang` packs them;
- the transfer rounded to halves (to nearest even, as athenea uploads it);
- the curvature in f32.

Its new entry `referenceRelightPixel` adds splat_project's per-pixel records: `slopes`, `sharpPolish`, the mark, and the colour with the centre's sharp reading taken out.

The skies are rebuilt from athenea's test source (tests/usd/test_usd.cpp, read only):

- **Pale:** `tests/data/lobes/sphere.usda`'s DomeLight, colour (0.6, 0.75, 1) at 0.6, plus its DistantLight: intensity 3, angle 4°, rotateXYZ (−40, 30, 0), `normalize` off, so the intensity is the disc's radiance, as athenea's LightTable passes it on.
- **Window:** the 64 × 32 PNG the test writes, read as sRGB, at intensity 1.5.
- **Lamp:** no dome, a SphereLight of radius 0.3 at 40, at (2.2, 2.5, 1.5).

The glass gets the cloud's `ior` of 1.45 through `setIor`.

| ball | sky | streams | mean luminance ours / athenea | mean relative error | records per pixel (sharp / sloped) |
|---|---|---|---|---|---|
| paint (Schlick metal under a 0.12–0.14 coat) | pale | alike | 0.99997 | 5.4e-4 | 94 024 / 0 |
| | window | | 1.00013 | 4.2e-4 | 94 024 / 0 |
| | lamp | | 1.00028 | 3.1e-4 | 0 / 94 024 |
| chrome (metal, 0.13–0.15) | pale | alike | 1.00003 | 3.1e-5 | 85 623 / 8 401 |
| | window | | 1.00003 | 2.3e-5 | 85 623 / 8 401 |
| | lamp | | 1.00010 | 3.6e-4 | 0 / 94 024 |
| rubber (0.6, sheen) | pale | alike | 1.00004 | 2.1e-5 | 0 / 94 024 |
| | window | | 1.00006 | 2.3e-5 | 0 / 94 024 |
| | lamp | | 1.00014 | 5.7e-5 | 0 / 94 024 |
| glass (1.45, lens) | pale | alike | 1.00030 | 4.0e-4 | 85 180 / 8 844 |
| | window | | 1.00034 | 4.9e-4 | 85 180 / 8 844 |
| | lamp | | 1.00026 | 1.2e-3 | 0 / 94 024 |
| paint ball green | san giuseppe | alike | 1.00012 | 3.4e-5 | 92 303 / 1 721 |

How to read the table:

- **Streams:** every stream the relight reads matches athenea's packing of the USD values. pbr (with the Schlick bit), lobes, transfer, cells and curvature are identical. The octahedral normal is within one step of 1/65535. The colour is within f16, after our sRGB decode.
- **Splats seen edge on:** splats with |n·v| < 0.02, 33 to 69 per ball, turn their face over between the two sides for one quantization step of their rotation. They draw as a line, so they're counted and left out. No other splat is more than 1% apart.
- **Per-pixel records:** every splat gets the same mark (sharp, sloped or none). The records (mirror, its steps, the lobes' weights and roughness, the sun through them, the slope) are compared on our own splats, meaning the `.athc`'s quantized rotation fed to both sides. There they are athenea's to within half-float precision: no record is more than 2% apart.
- **Independent geometry:** from the USD floats, 2–3% of the records move by more than 2%. A splat's step one pixel away follows the rotation's quantization, and the `.athc` quantizes twice: athenea's ten-bit smallest three, then Spark's ext octahedral encoding. athenea's own raster quantizes once.
- **The frame:** the coated paint ball placed in a stage that is turned (a Z-up stage turned to Y-up, then turned and moved again), with the eye and the sun carried along and relit with `frame`. It matches the unturned ball exactly: colours and per-pixel records.

The shaderballs (`shaderballs-m3`: brass, copper, brick, wood, marble, onyx under autoshop_01) come without their clouds. Only athenea's own numbers are left (relMSE against the GT 0.011–0.256), so there's nothing of ours to measure against them.

## What was found

- **The slope's fall was measured without athenea's dilation.** A splat's colour ramp across its footprint (SplatSlope, below roughness 0.5 where no lobe is sharp) is held so that its fall over three sigma is at most the colour. splat_project measures that fall on the 2D covariance after its third of a pixel of dilation (`a += 0.3; c += 0.3`). The relight pass measured it on the bare covariance, so a footprint of a pixel or two kept more of a crease's ramp than athenea does. The rubber ball with its curvature ×40 (a crease): 56 632 of 94 024 records more than 2% apart before, 0 after. Fixed in `slang/athenea_adapter/relight.slang` (`projectForSlope`); test: `atheneaSpheres` "a crease".
- **usd-athc** now reports the cloud layer's `upAxis` in `--json` (and in `--list`). That says whether a cloud needs `frame`.
- **In the test's reference, not in Spark:** three's `DataUtils.toHalfFloat` truncates subnormals, and a transfer is full of them. One splat of the rubber ball read 0.0044 against 3e-7 until the reference rounded to nearest even, as athenea's upload and our `.athc` both do.

Checked and alike, with nothing to fix:

- **The Schlick bit:** pbr bit 25, as `splat_encoding.slang` packs it.
- **The coat and polish read per pixel:** `AtheneaRelightBlend` is splat_blend's sharp branch: the same records and the dome's own sun only. Like athenea, an analytic DistantLight stays in the centre's colour.
- **The analytic sun through the cells**, as athenea applies it.
- **Chrome:** metal 1, roughness about 0.14 after the curvature's widening.
- **Glass:** transmission, the lens from the curvature, and the specular index from the lobes once the cloud has an index.
- **HDR:** the 36 000 sun texels survive from the sky's upload (rgba32float) to the half-float target.

## What the .athc does not carry

Of the primvars athenea's raster reads, all of these are carried: normal, metallic, roughness, transmission, schlickMetal, thinWalled, the nine layer arrays, transferDirect, Indirect and Reflected, shadowBits, curvature, and `linear`. Left out:

- **`ior`:** the cloud's index, a constant written whenever the conversion met a glass. `usd-athc --json` reports it as `constants.ior`, and the page passes it on with `setIor`, as `corvette.json`'s `ior` does. Without it, nothing bends. athenea bends only where the cloud has an index, and then by each splat's `specularIor`. The glass ball under the window changes by 138% of its mean luminance without it.
- **`relight`:** whether the cloud is relit. The page decides that.
- **`cryptoObject` / `cryptoManifest`:** the matte's ids. Reported in `--json`; athenea's overrides read them, the relight doesn't.
- **`emission` and `transferZonal`:** not carried yet. usd-athc warns. None of the spheres has them.

Also, athenea's bake writes `sheenColor` 0 for the rubber, whose OpenPBR material asks for a sheen of 0.5. athenea's test notes that `sheen_*` isn't declared by OpenPBR. That's athenea's side: the cloud is carried as baked.

A whole-file decode encodes a linear cloud's colours to sRGB for Spark's blend, and the plugin reads them that way by default (`setStoredLinear` false). Only a paged cloud keeps them linear.
