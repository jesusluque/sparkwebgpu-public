# athenea `.athc`: version 2 as read here, and version 3 by sections

sparkwebGPU reads athenea's native cloud format directly: no flatten to
`.ply`/`.spz`, so the material, transfer, emission and normals stay. This page
describes what the WebGPU path reads (version 1 and 2, written by athenea's
`Athc.cpp`) and the version 3 layout sparkwebGPU writes for the web
(athenea's proposal 083), together with the code that implements each.

| | Code |
|---|---|
| v1/v2 reader and writer, LoD tree, ATHV pages | `rust/spark-lib/src/athc.rs` |
| v3 reader and writer, gzip | `rust/spark-lib/src/athc_v3.rs` |
| Converter CLI | `rust/build-lod/src/bin/athc-convert.rs` |
| athenea's stream packing and LoD build, on the CPU | `rust/spark-lib/src/athc_build.rs` |
| USD (`.usdc`) TX cloud to `.athc` | `rust/build-lod/src/bin/usd-athc.rs` |
| Paging from the browser | `src/athc.ts`, `src/PagedSplats.ts` |
| Streams paged by group, within a budget | `src/webgpu/attributes/attribPaging.ts`, `PagedAttribPool.ts`, `AttribResidency.ts` |
| athenea's decoders over the attribute pool | `slang/athenea_adapter/athc.slang` |
| athenea-web `scene.json` | `src/webgpu/athenea/sceneJson.ts` |

```js
new SplatMesh({ url: "cloud.athc" });               // whole file, athenea's levels as the LoD tree
new SplatMesh({ url: "cloud.athc", paged: true });  // one Range request per 65 536-splat chunk (v3: per chunk and tier)
```

## Version 2 (athenea's)

Checked against athenea's `modules/lod/src/Athc.cpp` at `txf` 89a04d9:
rewriting `test/fixtures/athc/two_cards.athc` (written by athenea's
mesh2splat test) gives the same bytes. Little-endian. Every block starts on a
4096-byte page and the file ends on one.

```text
page 0   FileHeader, 104 bytes
           0 "ATHC"          4 version (1, 2)    8 count
          12 restPerColour  16 shWords          20 levels
          24 chunkSplats    28 chunks           32 finestGroups
          36 boundsLo[3]    48 extent           52 boundsMin[3]
          64 boundsMax[3]   76 flags            80 levelTable (u64)
          88 chunkTable     96 starts
         ExtraHeader, 32 bytes at 104, only when flags has bit 4 or 5:
           pbrWords (1), lobesWords (0/3), transferCount, transferWords
           (= ceil(count / 2)), shadowWords (0/2/8/32), 3 x padding
page 1   LevelEntry[levels]: level, groups, offset (u64)
         ChunkEntry[chunks]: offset (u64), count, padding
aligned  starts: u32 per finest-level group, its first splat
aligned  each level, coarsest first: a block of `groups` elements
aligned  each chunk: a block of `count` splats
```

A block is arrays over all its n elements, one after the other:

| Array | Words | Present | Contents (athenea `packing.slang`) |
|---|---|---|---|
| positions | 4 | always | x, y, z (f32), opacity (linear) |
| shape | 4 | always | rotation smallest-three 10+10+10 bits + index; f16 ln-scales; f16 base colour (0.5 + SH0 · dc) |
| sh | shWords | always | f16 rest coefficients, rgb per basis (one dummy word at degree 0) |
| tail | 1 | always | levels: the group's octree cell code; chunks: the splat's finest group |
| normals | 1 | bit 0 | octahedral 2 × unorm16 |
| emission | 1 | bit 2 | RGB9E5, linear |
| pbr | 1 | bit 4 | metallic, roughness, transmission bytes; thin-walled bit 24, Schlick bit 25 |
| lobes | 3 | bit 4, lobesWords 3 | `packLobes` |
| transfer | transferWords | bit 5 | f16 values, two a word |
| shadowBits | shadowWords | bit 5, shadowWords > 0 | open directions |

Flag bit 1 says the colours are linear light (no array). Bit 3 is reserved by
athenea (proposal 009); unknown bits are refused, as athenea refuses them.
A file is version 2 exactly when its flags are not zero. Crypto ids,
curvature and visibility are **not** in a `.athc` v2: athenea keeps them in
the USD stage only.

### As a Spark LoD tree

athenea's levels are an octree over the Morton-sorted splats: a level-r
group's cell code is the top 3r bits of its splats' Morton codes, so a group's
children are the next level's groups with code `c·8 .. c·8+7` (contiguous) and
a finest group's children are splats `starts[g] .. starts[g+1]`. The decoder
makes that one Spark tree, rooted at index 0:

```text
0                root: level 1's group, or one merged from level 1's groups
                 (athenea's moments: weights opacity × area of the two longest axes)
1 ..             each level's groups, coarsest first
splatBase ..     the splats (whole file: right after; paged: from the next
                 multiple of 65 536, so each athenea chunk is one Spark page)
```

Paged, the merged pages are built once from the file's first bytes (headers
through the levels, one Range request) in a loader worker; each splat page is
one Range request of its chunk, wrapped in a 160-byte `ATHV` head (format in
`athc.rs`) that the loader worker decodes like any other page.

The streams become attributes of the splats' `AttribPool`: `normalOct`,
`emission`, `pbr`, `lobes` (u32, as stored), `transfer` (f16 × count) and
`shadowBits` (u32), plus `athcGroup` (u32 x 2) on every file: the finest
LoD groups `[lo, hi)` an element covers, a splat its own and a merged node
its subtree, which is what a `.athc` can be picked and overridden by (it keeps
no Cryptomatte id; see [athenea raster](webgpu-athenea-raster.md)). Merged
ATHV pages carry these ranges after their block. `slang/athenea_adapter/athc.slang` reads the streams with
athenea's own unpack functions. A linear cloud's colours are sRGB-encoded at
decode (the base exactly, rest harmonics through the curve's slope) unless
the ATHV head's decode flags (byte 152) carry `ATHV_KEEP_LINEAR`
(`PagedSplats({ athcKeepLinear: true })`, `DecodeOptions::keep_linear`), for
the linear blend of the athenea raster.

## Version 3 (sparkwebGPU's, for the web)

Proposal 083: inside every block, the data a tier needs is a **prefix**, so
each (block, tier) is **one HTTP Range request**, and a reader that moves up a
tier fetches only the sections after the prefix it has.

Version 3 rearranges a v2 file; it does not change a value. A section is
encoding 0 (v2's words) or one of two lossless rearrangements of them
(below), so `v2 → v3 → v2` is the same file byte for byte (tested on both
fixtures, with and without compression, in every encoding). The encodings
and the per-section compression are where 083's smaller sections fit.

```text
page 0   header, 160 bytes
           0 "ATH3"   4 version 3   8 flags (v2's bits)   12 count
          16 chunkSplats   20 chunks   24 levels   28 finestGroups
          32 restPerColour 36 shWords  40 transferCount 44 shadowWords
          48 pbrWords      52 lobesWords 56 transferWords 60 sectionCount
          64 boundsLo[3]   76 extent   80 boundsMin[3]  92 boundsMax[3]
         104 blockCount (levels + chunks)   108 curvatureWords (0, 2)
         112 sectionTable (u64)  120 blockIndex  128 starts  136 dataStart
         144 skinInfluences (0: not skinned)  148 skinGradientWords
         152 skeleton (u64, the ATSK blob's offset; 0: none)
         160 occlusion codebook (u64 offset; 0: none)   168 its entries   172 0
             (only when the section table starts at 176 or later: a header
             of 160 bytes has no codebook)
@160/176 section table, 32 bytes each:
           id (fourcc), tier, encoding (0 v2 words, 1 byte planes,
           2 delta planes, 3 CPCA: transfer sections only, lossy),
           compression (0 none, 1 gzip), words a
           element, index width (OCCL: bytes an index needs, 1/2/4; else 0), 2 x 0
         block index, (32 + 16 × sectionCount) bytes each:
           kind (0 level, 1 chunk), level (1.. or 0), first element, n,
           bounding sphere of the centres (x, y, z, r),
           per section: offset (u64, absolute), stored bytes, raw bytes
aligned  starts: u32 per finest group (as v2)
16-aligned  a skinned cloud's skeleton (ATSK, below)
16-aligned  the occlusion codebook: entries x 8 u32 (below)
aligned  blocks: levels coarsest first, then chunks; each on its own page,
         its sections contiguous in table order
```

| Section | id | Tier | Bytes a splat | Contents |
|---|---|---|---|---|
| S0 | `CORE` | 1 | 36 | positions, shape, tail (as three arrays) |
| S0p | `SHRS` | 1 | 4 · shWords | rest harmonics |
| S0l | `LODS` | 1 | 4 | a merged node's LoD size by error, f32 (0 in a chunk: geometric); flag bit 6 (sparkwebGPU's, below) |
| S0e | `LODE` | 1 | 8 | a merged node's LoD error cut bounds: f16 eps, f16 rho; f16 theta, 8+8-bit cone axis (0 in a chunk); flag bit 7 (sparkwebGPU's, below) |
| S0o | `OCCL` | 1 | 4 | athenea's occlusion mask: the element's codebook entry + 1 (0: none), in a word; flag bit 9 (below) |
| S0m | `LODM` | 1 | 4 | a merged node's second opacity, f32 (its mass ratio held at 0.99; 0 for a leaf and for none); flag bit 11 (below) |
| S0t | `TREE` | 1 | 8 | a paged tree's children: their page, the first one's place and their count; flag bit 10 (below) |
| S0s | `SKIN` | 1 | 4 · (influences + gradientWords) | a skinned cloud's rig (sparkwebGPU's, below) |
| S4 | `MATL` | 2 | 4 · (normals + emission + pbr + lobes) | the material streams, as present |
| S3 | `SHAD` | 3 | 4 · shadowWords | open-direction bits |
| S3c | `CURV` | 3 | 8 | the curvature: athenea's shape operator, three f16 (sparkwebGPU's) |
| S1 | `TXDI` | 3 | 4 · ⌈direct / 2⌉ | the transfer's direct half (9 or 16 values; zonal: all 10) |
| S2 | `TXIN` | 3 | 4 · (2 · direct − ⌈direct / 2⌉) | through the indirect half |
| S2f | `TXFD` | 3 | the rest | the reflected field |

A section is left out when it would be empty. Data tier 1 (the splats as
captured) is `CORE + SHRS`, tier 2 adds the material, tier 3 the relight
streams. The order makes each thing a reader may want one contiguous run:
the material alone; the shadow bits and the curvature with any prefix of the transfer that is
a layout of its own (`transfer_layout.slang`: 16 direct values, 64 with the
indirect half, 112 with the field; 9, 36, 84 at degree 2). For
`every_stream.athc` (SH3, TX transfer of 112 values, 16×16 shadow bits,
material): tier 1 is 128 bytes a splat, tier 2 152, tier 3 408.

### Encodings: byte planes before the gzip

A browser has gzip and nothing better (`DecompressionStream`: no zstd, brotli
only in some engines), so the lever is what gzip is given. Encoding 1 stores
each array of a section (CORE: positions, shape, tail; MATL: each stream it
holds; the rest: one array) transposed by bytes: byte 0 of every element,
then byte 1, and so on (`athc_v3.rs encode_section`). The high byte of an
f16 (sign, exponent, two bits of mantissa) or of an f32 barely changes from
a splat to the next in Morton order, and now sits in runs gzip finds; the
low, noisy bytes no longer break them. Encoding 2 does the same with each
16-bit lane replaced by its difference from the previous element's
(wrapping). Both are undone after the gunzip: in WASM for whole files and
for the pages the loader worker decodes (`CORE`, `SHRS`, the levels), on
the main thread for the streams a chunk page puts together (`src/athc.ts
decodeAthcSection`). In an ATHV kind-2 page an entry's compression word
carries the encoding in its high 16 bits. Readers before these encodings
refuse such a file (`parse_v3`), as they should.

`athc-convert --gzip --planes` (`write_v3_smallest`) writes every section in
whichever encoding stores it smallest over the whole file (the shadow bits,
random bits, stay 0; so do transfers whose low bits are noise either way).
Measured on the published clouds (gzip, MB):

| Cloud | Total | CORE | MATL | CURV | TXDI | TXIN |
|---|---|---|---|---|---|---|
| Corvette light (v3-light, 7 files) | 68.1 → 60.9 (−10.6 %) | −16 % | −15 % | −18 % | −5 % | |
| Corvette HD (v2-hd, 7 files) | 288.7 → 248.6 (−13.9 %) | −18 % | −19 % | 0 | −13 % | |
| its paint (1.92M splats) | 125.5 → 97.2 (−23 %) | −23 % | −28 % | 0 | −26 % | |
| pawn body t16 | 45.3 → 34.4 (−24 %) | −24 % | −21 % | | −28 % | |
| pawn top t16 | 44.1 → 29.5 (−33 %) | −31 % | −32 % | | −38 % | |
| pawn body t64 | 102.8 → 85.3 (−17 %) | −24 % | −21 % | 0 | −27 % | −12 % |

The Corvette's trim and wheels (athenea's whole-car bake) keep their
transfer as it is. The shadow bits never gain.

### Encoding 3: clustered PCA of the transfer (lossy)

The transfer is most of a relit cloud (30–70 % of its bytes), and its
values are smooth functions of a splat's surroundings: neighbours' transfers
are close, and a block's span few directions. Encoding 3 (`athc_cpca.rs`,
Sloan et al. 2003) stores a transfer section (`TXDI`, `TXIN`, `TXFD`; never
another) of one block as:

- up to K clusters (k-means on an even sample, then every element to its
  nearest centre; K at most the block's elements / 128);
- per cluster its mean and its first m principal directions, f16, m the
  fewest (at most M) whose truncation takes at most 40 % of the budget;
- per element its cluster (u8) and m coefficients, quantized with a step per
  cluster proportional to the cluster's RMS (the same relative error in dim
  clusters as in bright ones), zig-zag integers of 1–3 bytes stored as byte
  planes for the gzip.

The budget is relative: the block's sum of squared errors over its sum of
squared values, `E` (default 1e-5), checked on the decoded halves (the step
shrinks until it holds). The SH transfers' values are orthonormal harmonic
coefficients, so this is the transfer's L2 error over the sphere: what the
sky's dot product, the sun's point evaluation and the field's reflection
read. A zonal transfer (10 values: octahedral axes and band coefficients) is
clustered with each value weighted by its RMS and its error counted on the
nine harmonics it stands for. A block that cannot meet the budget, or that
byte planes store smaller, keeps its values exactly (mode 0).

```text
mode 0   u32 0, the halves as byte planes (encoding 1)
mode 1   u32 1, K, M, D (= 2 x words), B (coefficient bytes, 1..3)
         K f32 steps; K x (1 + M) x D f16 (mean, then M directions), padded to 4
         n u8 clusters; M x B planes of n bytes (byte b of zigzag(q_j))
value    acc = mean[k][d]; acc = acc + (q_j * step_k) * dir[k][j][d] for j < M,
         each product and sum rounded to f32; the half nearest (ties even)
```

The decode is that exact f32 arithmetic in a fixed order, so the Rust
decoder (WASM: whole files and kind-2 pages) and the TypeScript one (`src/
athc.ts decodeAthcCpca`: streams a chunk page fetches on its own) give the
same halves bit for bit (`test/unit/athcCpca.test.ts`). What comes out is
the section's words as before: attribute pools, paging by stream group,
residency (`athcTransferResident`) and the transfer forms (`transferForm`:
16, 64, 112, each a prefix of sections) are unchanged
(`test/gpu/athcPaging.test.ts` pages a CPCA file like the others). Merged
LoD nodes are blocks too and are compressed the same way. Readers before
encoding 3 refuse such a file (`parse_v3`).

```sh
athc-convert in.athc out.athc --gzip --planes --cpca 16,48        # K 16, M 48, E 1e-5
athc-convert in.athc out.athc --gzip --planes --cpca 16,48,3e-5   # a looser budget
```

Measured (thread BF, `research/simplify-measurements.md`): the transfer
sections take 4–7× fewer bytes (TX 112 of the pawn's top: 157 → 28 MB); a
relit splat through athenea's `relitSplat` on Dawn moves by a relMSE of
4e-8 – 3e-5 (`test/gpu/athcCpcaRelight.test.ts`, opt-in); the Corvette light set 67.9 → 49.5 MB, the pawn page 281 → 84 MB, the sparrow −6 %.

On the GPU the transfer can also stay as these clusters
(`atheneaRelightPlugin({ transferResident: "cpca" })`, thread BP): the pool
holds `transferCpca` (a cluster entry and the i16 coefficients a section, or
its halves where clusters are not smaller), a table of the clusters sits
beside it, and the relight pass rebuilds the halves a batch at a time
(`slang/athenea_adapter/cpca.slang`), the same as the decoder's to 1e-11
relMSE. The pawn's top (TX 112) takes 148 B a splat instead of 224 (164
when paged, M fixed at 32), at about 6 ns a splat each relit frame; a t16
transfer gains nothing.

### A skinned cloud (`SKIN` and the skeleton)

athenea keeps a cloud a skeleton carries (`athenea mesh2splat --skinned`)
only as a USD stage: UsdSkel's binding on the `ParticleField`
(`skel:jointIndices`, `skel:jointWeights`, elementSize 4;
`skel:geomBindTransform`; `skel:joints`; `skel:skeleton`) plus its own
`AtheneaSplatSkinningAPI`: `jointWeightGradients` (six halves a gaussian:
d w_k / d u and d w_k / d v along its first two rest axes for the first three
joints, 7dff879) and `skinningXforms`, the joints' transforms (bind space to
the skeleton's, `matrix4d[]`) time-sampled at the instants the conversion
read them. Its `.athc` writer refuses `--skinned`. sparkwebGPU carries it:

- **`SKIN`**, tier 1 (it is geometry: without it the cloud is drawn in its
  bind pose), right after `SHRS`. Per element, `influences` words, one an
  influence: joint u16 (low) | weight unorm16 (high) -- athenea's packed
  layout (`common/influences.slang`, t22 390670e), the running sum of the
  weights rounded so a gaussian's steps add up to its weights' sum to one
  step -- then `gradientWords` (0 or influences − 1) words of two halves,
  `jointWeightGradients` as the file holds them. A merged LoD node carries
  the opacity × weight blend of its splats' joints (the heaviest
  `influences`, renormalised) and zero gradients. Header words 144 and 148;
  in memory and in ATHV pages the v2 extra header's seventh and eighth words
  (0 in athenea's files). A v2 file has no room for it.
- **The skeleton**, an `ATSK` blob after the starts (inside the bytes a pager
  reads first; header word 152 says where):

```text
0   "ATSK"  4 version 1  8 bytes (the whole blob)  12 joints
16  clips   20 influences a splat   24 gradient words a splat   28 0
32  geomBindTransform, 16 f32 (USD rows: p' = p M)
96  joint names' bytes   100 skeleton path's bytes   104 0 0
112 the joint names ('\n' between), the skeleton path; padded to 4
    each clip: name bytes, samples, timeCodesPerSecond (f32), 0;
      its name, padded to 4; the sample times (f32 time codes, increasing);
      the transforms, f32 x samples x joints x 16 (USD rows, skinningXforms)
```

A pose between two samples is each matrix element linearly between them
(USD's interpolation of matrix samples, which is what athenea's Hydra hands
its skinner). The decoder makes the attributes `skinInfluences` (u32 ×
influences) and `skinGradients` (u32 × gradientWords); `athc_skeleton()`
(WASM) and `readAthcSkeleton({ url })` read the blob. Code:
`rust/spark-lib/src/athc_skin.rs`; the GPU side is
[athenea skinning](webgpu-athenea-skin.md).

Paged (`PagedSplats`), a skinned cloud is drawn in its bind pose for now:
`SKIN` is not fetched with the pages. `SKIN` takes any encoding (`athc-convert
--planes` keeps the skeleton: `write_v3_smallest_with`), and `file_bytes`
counts the skeleton blob.

### LoD sizes by error (`LODS`)

Spark's traversal (spark-rs `lod_tree.rs`, the same code for WebGL and
`WgpuLod`) refines the node with the largest `size / distance` first under
its splat budget, and stops at a pixel. The decoder's size is geometric
(splat_encode `encode_lod_tree`: twice the mean scale, grown for a LoD
opacity past 1). With header flag bit 6 a file carries, in a tier-1 section
right after the harmonics, one f32 for every element: for a level group the
size the decoder writes into the LoD tree in place of the geometric one, 0
(every chunk element) for a splat that keeps its own. sparkwebGPU writes it
as the geometric size scaled up by the group's error against all the splats
under it (`athc_lod_error.rs level_lod_sizes`: colour and material
variance, normal spread, thickness; `clamp(((v + v0) / v0)^0.5, 1, 4)`),
made monotone up the tree, so a group that looks like its splats keeps its
size and Spark's one-pixel limit and `lodScale` mean what they did, and
the budget goes where the error is. Measured with `athc_measure trav`
(research/simplify-measurements.md § BE): -13 to -29% relMSE at equal
budget on corvette-v6e-light, -9 to -35% on the pawn.

`athc-convert in out --lod-sizes [--lod-size-opts k=v,...]` adds them to any
file (no rebuild); a file that has them keeps them, recomputed after a cut,
unless `--no-lod-sizes`; `usd-athc --lod-sizes` writes them at conversion. A
merged ATHV page carries its nodes' sizes after its group ranges, f32 each,
with bit 1 of its decode flags (word 152, `ATHV_LOD_SIZES`); the browser
fetches `LODS` with the levels and never with a chunk. A v2 file never holds
them, and a reader before them refuses the flag.

### LoD error cut bounds (`LODE`)

The run-time error cut (thread CG, `athc_lod_cut.rs`), athenea's research
100 and `lod_tree_cut.slang` (treeErrorOwn / treeErrorUp / treeErrorKeep,
treeBoundsUp for the cone) after Nanite: with header flag bit 7 a file
carries, in a tier-1 section after `LODS`, two words for every element; a
level group's are

```text
word 0   f16 eps (low) | f16 rho (high)        metres, each rounded up
word 1   f16 theta (low) | cone axis (high)    radians; octahedral 2 x unorm8 (x low)
                                               theta = +inf (0x7c00, axis 0): never culled
```

and a chunk element's (a splat, a leaf) are 0. With `r(n) = |scale(n)|`
(`sqrt(tr Sigma)`), a leaf's eps 0 and own sphere `2 max sigma`:

```text
eps(n)   = max(r(n) |cost(n)|, max over children eps(c))
rho(n)   = max(2 max sigma(n), max over children |mu(n) - mu(c)| + rho(c))
theta(n) = max over children theta(c) + angle(n_c, n_n)      (leaf: 0)
P(n)     = eps(n) ppu / max(d(n) - rho(n), 1e-3)             (pixels)
drawn    : P(n) <= tau < P(parent)
culled   : a node the cut draws, wider than lodBackCullMinPx (1) on screen,
           with n . v < -sin(theta + asin(rho / d))
```

`eps` grows up the tree and each sphere holds its children's, so `P` is
monotone and the cut is consistent without walking up. `cost` is athenea's
`lodCost` when the levels are her merge tree (`usd-athc --tree --lod-errors`:
computed over her whole binary tree, then read at the level groups that are
her nodes), otherwise ours: `athc_merge::group_cost` of the node's children
in the `.athc` tree, BC's merge cost without its on-screen weight (Runnalls'
bound per unit mass with the covariances floored at 1% of the merged mean
variance, plus 128 x the mass-weighted variance of the linear base colour
and of the key mix, plus 32 x that of metallic, sqrt roughness and
transmission). The two costs are on different scales: a tau for one is not
a tau for the other. Her trees default to her bounded form (treeErrorBounded
with Decimate.cpp's weights: `r (|cost| + 10 s_c + theta^2 / alpha)`, a
merge past the guards (cost < 0) `max(4 e, 2 r)`; s_c the spread of the
stored base colour, clamped at 0, climbed as the cone is; alpha
`max(roughness, 0.02)^2`, roughness 0.5 without a material); ours to
`r |cost|`. `--lod-error-weights colour=,normal=,guard=,floor=` (or `own`
for treeErrorOwn) changes either. The cone axis is the node's normal (the
normals stream), rounded to 8 bits a component with theta widened by the
rounding. Only a closed piece carries one: athenea's constant
`primvars:athenea:splat:closed` 1 (mesh2splat found no open edge; the
USD's doubleSided is set on too many meshes to tell), or `--closed`;
`--double-sided` takes it away whatever the primvar says. Glass, thin
walls and clouds without normals never carry one. The decoder's
synthetic root takes level 1's largest eps, its sphere round theirs, no
cone.

Spark's traversal (spark-rs `lod_tree.rs`) takes them as an opt-in mode:
`WgpuLod`'s `lodErrorTau` (SparkRenderer's, tau in pixels, 0 off) refines
a node with bounds while `P > tau` (a node without them, a tree without
them, by the size cut), and `lodBackCull` leaves out the nodes the cut
draws whose whole cone faces away, when they are wider than
`lodBackCullMinPx` (1) on screen: a node the cut refines is never culled,
so full detail never is (culling it brightened the grille past its leaves:
their back faces show through the semi-transparent fronts). A merged ATHV page carries them after its sizes (or
its group ranges), two words a node, with bit 2 of its decode flags
(`ATHV_LOD_ERRORS`); the decoded splats carry them as `extra.lodError`; the
browser fetches `LODE` with the levels and never with a chunk. A v2 file
never holds them, and a reader before them refuses the flag.

`usd-athc --lod-errors` (any conversion) and `athc-convert --lod-errors`
write them; a file that has them keeps them unless its tree changes (a cut,
`--error`, `--rebuild-frame`: then ours), `--no-lod-errors` drops them.

### A producer's own coverage (flag bit 8)

athenea's `decimate --tree-coverage` gives each merge of her tree, as its
opacity, what its leaves cover composited on its plane (an 8 x 8 grid over
its +-2 sigma, `1 - prod(1 - alpha G)`). Header flag bit 8
(`FLAG_NODE_COVERAGE`) says the levels' opacities are such a coverage:
kept as stored -- never replaced by the mass ratio (`uncap_levels`), and
the decoder's widening only orients those nodes (`widen_levels`), it does
not divide them by an area it adds. `usd-athc --tree` sets it when her
tree marks it -- `coverage` is then the composited coverage and the mass
ratio has moved to `coverageMass` (pieces/t21) -- or with `--node-coverage`
(then the opacities are read as they are), and reads `coverage` as the
merges' opacity.

### Two opacities a merged node (`LODM`, flag bit 11)

athenea's coverage trees (pieces/t47) carry both opacities of each merge:
`opacity` is the composited coverage of its leaves fitted over its core,
`primvars:athenea:splat:coverageMass` the mass ratio W / A. Her raster
ranks the coverage first near and the mass far; the switch below lets a
player pick by size on screen. In Spark's mass-keeping draw the mass ratio
held at 0.99 was as good or better nearly everywhere, and the switch gained
at most 0.06 dB (research/errcut.md, round 4), so it is off by default. With
header flag bit 11 (`FLAG_LOD_MASS`) a file keeps both: the stored opacity
(the coverage, with `FLAG_NODE_COVERAGE`) and, in a tier-1 section after
`OCCL`, one f32 an element: a level group's mass ratio clamped to
[1.01/255, 0.99], 0 for a group without one; a chunk's (a leaf, one
opacity) all 0, which a reader takes as none. `usd-athc --tree` writes it
whenever the tree has `coverageMass` (not with `--mass-capped`,
`--mass-coverage` or `--no-lod-mass`). A merged ATHV page carries its nodes'
after the occlusion indices, f32 each, with decode flag bit 4
(`ATHV_LOD_MASS`); the browser fetches `LODM` with the levels and never with
a chunk; the decoder's synthetic root has none.

In a paged file (flag bit 10, error-band pages) `LODM` is one more tier-1
section of every page, beside `TREE`: each page carries its own nodes'
values (0 for its leaves and for the synthetic root), fetched with the
page's tree sections.

The decoder hands it to the receiver (`set_lod_mass`). ExtSplats keep it in
the high half of word a.w, as f16 (the low half is the opacity; a source
splat's high half is otherwise 0); PackedSplats have no room and drop it.
The player draws a node that has one at its **mass by default**
(`WgpuSplatRenderer` `nodeOpacity: "mass"`; `"coverage"` draws the stored
opacity), in generate and in the relight pass, so a merged glass sheet's
alpha is the one drawn. The opt-in switch `opacitySwitchPx` T > 0 (off by
default) instead draws the mass while the node's size on screen, 2 x its
widest sigma x focal / distance (object space, as the LoD traversal's size
/ distance), is at most T pixels, and the coverage above. Leaves keep their
own. A v2 file never holds it, and a reader before it refuses the flag.

### Occlusion masks (`OCCL` and the codebook, flag bit 9)

athenea's `decimate --occlusion-mask N` (athenea-pieces 1000bdf) gives each
gaussian of her tree an occlusion mask: a 16 x 16 octahedral grid of
directions in the asset's frame (bit y * 16 + x, 1 open: some leaf under it
is seen from there), OR'd up the tree, one cell of margin, all open on
glass, then put in a codebook of N entries (each the OR of its members, so
it never closes what a node has open). `usd-athc --tree` reads
`primvars:athenea:splat:occlusionIndex` (int a gaussian) and
`occlusionCodebook` (constant int[], N x 8 u32) and writes:

- `OCCL`, every block (levels and chunks): entry + 1, 0 for none (the
  decoder's synthetic root, the groups made above her roots: never culled),
  one word an element. The section table's sixth word says how many bytes
  the values need (two with 256 entries, because of the 0); the other bytes
  are 0, and `--planes` stores them as next to nothing. A codebook of 1024 to
  4096 entries changes nothing but that width.
- The codebook after the starts and the skeleton (the tables a pager reads
  first), its offset and entries in the header at 160.

The traversal (`lodOcclusionCull`, spark-rs `lod_tree.rs occluded`, the
same rule as `athc_lod_cut::occlusion_hides`) leaves a node out with its
subtree when the eye is outside the asset's box (the header's boundsMin /
boundsMax, the splats' centres) and the four cells toward the eye --
athenea's `splatCellsOpen`: `octEncode(dir) * 16 - 0.5`, the cells clamped
to [0, 15], those of bilinear weight 0 left out -- are all closed in its
entry; `dir` from the node's centre toward the eye in the tree's object
space (the file's frame). Nothing else: the margin and glass are in the
masks. Merged ATHV pages carry the indices after the errors (decode flag
bit 3, `ATHV_OCCLUSION`); decoded splats as `extra.lodOcclusion`, a whole
file's codebook and box as `extra.lodOcclusionCodebook` /
`lodOcclusionBounds`.

### Error-band pages (`TREE`, `PAGE`, flag bit 10)

athenea's `decimate --tree-pages P` pages her whole merge tree (leaves and
merges) by ERROR BANDS, as Nanite does: a node is needed exactly when its
parent's projected error passes tau, so each node goes to the band (an
octave) of its PARENT's error, the bands coarsest first, each band's nodes in
leaf order, the whole order cut into pages of P nodes
(`primvars:athenea:splat:lodPage`), with a table a player tests a page by
without reading its nodes (`lodPageInfo`: band, nodes, first;
`lodPageBounds`: a box, the largest and smallest parent error). `usd-athc
--tree tree.usdc out.athc --pages [--gzip --planes --cpca K,M]` (thread CK,
`athc_pages.rs`) writes that tree page by page; `FLAG_PAGED` (1024) says so:

- every block is a page (block kind 2, `first` its first node in page
  order); no levels, no starts, `chunks` the pages, `chunkSplats` the
  largest page;
- every node, leaves and merges, with all its sections (tier 1 is `CORE`,
  `SHRS`, `LODE`, `OCCL`, `TREE`); the merges' `LODE` are athenea's bounded errors over her whole
  tree (as `--lod-errors`), a leaf's 0; the merges turned so their third
  axis is the normal (`orient_merged`, what the decoder does to a
  producer's coverage), stored as the decoder will read them;
- **`TREE`**, tier 1, two words a node: the page its children are in, then
  the first one's place in it (low 16 bits) and their count (high 16; 0 a
  leaf). Spark's tree names children as one run, so the packer keeps every
  node's children in ONE page, next to one another: a page boundary that
  splits siblings moves the later ones into the earliest one's page (their
  parent is in it or before it, their children after it); a parent her
  order put after one of its children (it happens where the two share a
  band and a first leaf and the boundary falls between them: 92 of the
  hood's 648k nodes) is brought forward to the child's page; within a page,
  the root and the runs whose parent is in an earlier page first (merges
  before leaves, by first leaf), then breadth first. A forest (her trees
  have 14 to 251 roots) gets a root merged over its roots (`merge_root`) at
  the head of page 0;
- **`PAGE`**, the page table, among the tables (a pager reads it with the
  header): its offset (u64), rows and words a row at header byte 176 (the
  header is 192 bytes; the codebook's 16 bytes at 160, 0 without one), a
  row of 16 words a page:

```text
0  band (athenea's bandOf of the largest parent error)   1 nodes   2 first node
3  box lo xyz   6 box hi xyz (f32)
9  largest parent error   10 smallest (f32; a root's 3e38)
11 the page's first byte (u64)   13 the bytes of its tier 1, tier 2, tier 3
```

The data is laid out BY TIER, not page by page: every page's tier-1
sections, page after page, then every page's tier 2, then tier 3 (no page
alignment in between). A page's sections of one tier are contiguous, so a
run of consecutive pages is one Range request a tier, and a reader of tier
1 alone never reads a byte of the material or the transfer. (Page by page,
a Range over pages p..q read q-p pages' transfers in between: on the six
t42 pieces a tier-1 reader fetched 74 MB for the near view instead of 12.)

The table is recomputed from the nodes as the traversal reads them (the
LODE words, f16 rounded up: within 0.3 % of athenea's but where the packer
moved nodes), and its box holds the PARENTS' spheres (each widened by the
f16 a centre is in the traversal), not the nodes' own: a node is refined by
its parent's projected error from the parent's sphere, so this box makes the
page test conservative against the traversal -- every node the cut can reach
is in a needed page. It is up to 1.07-1.23x wider than athenea's
(`usd-athc --page-box-own`: hers). A page is needed when

```text
epsMax · ppu / max(dist(eye, box), near) > tau        (page 0 always)
```

Decoded whole (`new SplatMesh({ url })`), the nodes are in page order and a
node's children are named by their place in the file: the reference, every
node resident. Paged (`PagedSplats`), page p is Spark chunk p of a sparse
virtual space (child `p << 16 | place`), held in a pool slot of
`WgpuLodOptions.bandPageSplats` (3072) splats: `WgpuBandPager` (below).

#### Streaming them (`WgpuBandPager`)

`WgpuLod` gives a paged tree a pool of small slots (`bandPageSplats`,
`maxPagedSplats` the residency budget in nodes) and its own pager
(`src/webgpu/WgpuBandPager.ts`). After each traversal:

- the needed set from each tree's table alone (the eye in its object
  space, tau as the error cut's `lodErrorTau`, over the mesh's lodScale);
- the missing needed pages asked for by priority, the error a page removes
  per byte it costs (`epsMin` from its box over the bytes its fetch reads),
  nearer first on ties, consecutive pages of a tree merged into one Range
  request (at most 8 MiB), within the pool's free capacity;
- the pool evicts pages not needed from the highest index down (the finest
  bands first), never page 0, never one a resident page depends on (whose
  nodes' children are in it);
- `neededFill` (the needed set over the pool) drives tau's breathing
  (`lodErrorTauAuto`) when the needed set does not fit.

The traversal cuts within the resident pages: a node whose children's page
is absent is drawn as a leaf (spark-rs `lod_tree.rs`, the tree's `slot`).

#### The compact tier (`CMPC`, tier 4)

`usd-athc --tree tree.usdc out.athc --pages --compact-basis basis.json`
(thread CN, `athc_compact.rs`; v2 thread CQ) adds a FOURTH tier for the
compact relit path (`src/webgpu/athenea/compact.ts`, research/iphone-compact.md):
what a phone draws of a relit page, without a byte of tiers 1 to 3.

- **`CMPC`**, the last section, tier 4, laid out after tier 3 (a run of
  consecutive pages is one Range), 56 bytes a node (14 words, encoding 0) as
  byte planes, plane k at `n x (widths before k)`:

```text
posX posY posZ  u16 x3 in the page's box     albedo  u8 x3 sRGB
opacity         u8 code (sqrt to 1, then LoD opacity 1..2 by 1/63)
scale           u8 x3 ln in the page's range (0 = exactly 0)
rotation        u32 the ext quaternion word  normal  3 bytes, octahedral 12+12
curvature       s8 x3 signed log in the page's range
material        u16 into the piece's (or the asset's) table   cluster u8 (joint CPCA)
weights         3 x u32: 8 int4 a block (direct, indirect, reflected)
shadow          u16 into the piece's (or the asset's) dictionary  -- the GPU's 40
-- the traversal's 16, one byte a plane:
centre          i16 x3 (lo, hi planes): the f16 centre the decoder rounds the
                node's position to, as a step in f16 order from
                f16(the compact position) (-0 just below +0); 0 for a leaf
child page      u16 (lo, hi): the children's page minus the node's own
children        u32 (4 planes): place (bits 0-11) | count (12-23, 0 a leaf)
                | flags (bit 24: a merge without LODE, eps holds its LoD size)
eps, rho        f16 each (lo, hi planes): LODE's bounds; 0 for a leaf
```

  The traversal reads exactly what the error cut reads (spark-rs
  `lod_tree.rs`): a merge's centre, eps and rho and its children; of a leaf
  only that it is one (its pixel scale decides nothing: a leaf in the
  frontier is drawn whether it pops or not, as the pool never binds the
  cut's count). The centre is exact (bit for bit the f16 of v1's LoD tree
  words) because the correction is taken against the same f32 decode of the
  compact position on both sides (`athc_compact::compact_position`, shared by
  the writer and the WASM). Not kept: LODE's back-face cone (theta, axis;
  athenea's trees hold none, theta infinite: a compact traversal does not
  cull back faces), a LODE merge's size (so `lodErrorSizeK` has no size to
  guard with), the occlusion index. v1 (thread CN) kept the LoD tree's 4
  words and LODE's 2 (24 bytes, 64 a node): the same cut, node for node
  (`test/unit/compactWords.test.ts`).
- **the page's head** in its page-table row: rows are 32 words, word 16 the
  bytes of its `CMPC`, words 17-28 the head (box lo, hi; ln scale lo, hi;
  the curvature's range; three words a reader fills: the slot's codebook
  and materials offsets);
- **the piece's tables** after the page table (offset u64 and bytes at
  header byte 192, the version at 204, the header 208 bytes): `CMPT` v2, 24
  fixed words then the plane widths:

```text
0 "CMPT"  1 version 2  2 node bytes (56)  3 planes (31)  4 head words (12)
5 dictionary entries  6 materials  7 clusters  8 normal bits  9 weight bits
10 flags: 1 linear, 2 curvature, 4 the basis in the asset table,
   8 the dictionary in it, 16 the materials in it
11 dictionary at  12 materials at  13 basis at  14 basis words  15 words
16 17 the asset table's id (u64)  18 its name at  19 its name's bytes (<= 64)
```

  the shadow dictionary (k-means in Hamming space over every node, 1024
  entries, 12 rounds; kept exactly when there are no more distinct masks),
  the materials (every distinct pbr + lobes, 4 words each), the joint CPCA
  basis in f16 (504 words a cluster) -- each absent (0 entries) where the
  flags give it to the asset table.

#### An asset's pieces (`.cmpa`)

athenea's whole car comes as one tree a part (by cryptoObject); every part
carries the same CPCA basis, 516 KB in f16, which was 3.1 of the 3.6 MB of
tables the six t48 pieces read. The **asset table** holds it once: a sidecar
file beside the pieces (`car.cmpa`), which each piece names (`CMPT` words
18-19) and identifies (words 16-17, its id); a reader fetches it once and
the pool keeps one basis (and one dictionary and material table when shared):

```text
0 "CMPA"  1 version 1  2 words  3 flags (1 basis, 2 dictionary, 4 materials)
4 clusters  5 dictionary entries  6 materials  7 basis at  8 basis words
9 dictionary at  10 materials at  11 12 id: FNV-1a 64 of the blob with these two words 0
13-15 0; the arrays at 16-word boundaries
```

```sh
# The pieces one by one, sharing the basis (car.cmpa written by the first,
# checked by the others):
usd-athc --tree tree_Hood.usdc Hood.athc --pages --gzip --planes --transfer full \
    --compact-basis basis.json --compact-asset out/car.cmpa
# Or afterwards, over pieces written alone (v1 or v2): one table with the
# basis, ONE shadow dictionary (4096 entries, k-means over every node of every
# piece) and ONE material table; each piece written again to D (tiers 1-3 byte
# for byte; its v1 traversal words checked node by node against v2's):
athc-convert --compact-asset car.cmpa --out-dir D Hood.athc Front_Fender.athc ...
#   --asset-entries N (0: each piece keeps its dictionary)  --piece-materials
athc-convert --compact-rebase v1.athc v2.athc     # a v1 file to v2, its own tables
```

A dictionary and a material table per piece or per asset, measured on the six
t48 pieces (1.71M nodes; per piece: 6 x 1024 entries, 6406 materials):

| | bytes | mean Hamming distance to the mask (of 256) |
|---|---|---|
| dictionaries per piece (6 x 1024) | 196 608 | 1.00 |
| one per asset, 1024 / 2048 / 4096 / 6144 | 32 768 / 65 536 / 131 072 / 196 608 | 1.73 / 1.37 / 1.07 / 0.91 |
| materials per piece / per asset | 102 496 / 94 368 | exact |

Per asset is cheaper (4096 entries: two thirds of the bytes for about the
same distance; 6144: the same bytes, closer), and a piece's own dictionary
costs 32 KB whatever its size (Door_001: 10k nodes, 3 bytes a node), so the
whole car's ~100 parts would read ~3 MB of dictionaries. `athc-convert
--compact-asset` shares both by default; `usd-athc --compact-asset` alone
(pieces converted one at a time) shares the basis, and the dictionary and
materials too when the table already holds them (every material of the
piece must be in it).

**Our own basis** (thread CR): `fit-compact-basis <piece.athc>... --out
basis` (`athc_basis_fit.rs`) fits a joint CPCA in athenea's layout (C 256,
K 8, int4 weights) to the full transfer of every node of the pieces given:
per-block energy norms, k-means of the joint vectors (k-means++ seeds, Lloyd
over every node), per-cluster PCA of each block, and the scale per
direction with the least error through the int4 code and the GPU's int8
grid. `--compare other.json` prints the same per-piece, per-block relMSE
for another basis, and `--held-out piece.athc` measures a piece it was not
fitted on. Fitted on the six t48 pieces it beats q97 on all of them, but a
held-out piece comes out 10 to 400 times worse (relMSE 1.6e-2 to 0.58):
fit it on every piece of the asset it will encode.

A compact reader (`CompactBandStreamer`, `src/webgpu/athenea/compactPages.ts`;
`openCompactAsset(url)` and the option `asset` for shared tables) reads the
tables with the page table, then only `CMPC` runs: each page goes into a
pool slot of `slotNodes` (3072) nodes, its head then its planes, the GPU's 40
bytes a node; the position and traversal planes (22 bytes a node) go to the
WASM traversal (`update_compact_lod_trees`, one shared compact pool tree, a
slot a page, 14 bytes a node where a full tree's `LodSplat` takes 24). The
page plan, the eviction order and tau's fit are `WgpuBandPager`'s
(`planBand`). The traversal's cut, slot indices, is the compact path's draw
list (`CompactRelitPath.setPagedCloud` + `setDrawList`); compact.slang reads a
slot's planes and head when `slotNodes` > 0.

### The curvature (`CURV`)

athenea keeps a per-splat curvature in USD only
(`primvars:athenea:splat:curvature`, 3 floats: the 2×2 symmetric shape
operator in the splat's first two axes, `xx xy yy`, in 1/metres, measured on
the mesh's normal). Its raster reads it for the per-pixel slope of a
reflection, the sharp coat and polish, and the far face of a solid glass
(`lensExit`). sparkwebGPU keeps it in a section of its own, tier 3, right
after `SHAD` (it is read with the relight streams: `Want` brings it
whenever it brings the shadow bits), as three f16 in two words. The
fourth half is 0 for a splat and, for a merged group written by
sparkwebGPU (`athc.rs store_normal_variance`, `truncate_creases`), the
variance of its splats' normals (1 − |opacity×area-weighted mean normal|²,
LEAN's second moment), which the relight's footprint prefilter widens the
lobes by where it exceeds what the curvature implies
(`relight.slang footprintRoughness`); the attribute is four halves. Header word 108 is its words a splat (0 or 2). In memory and in
ATHV pages it is the v2 extra header's sixth word (padding, 0, in every file
athenea writes); a v2 file never carries it (`--v2` and `athc-convert --v2`
drop it, and a v2 written from a cloud with it is the file without it). A
merged group's curvature is the opacity-weighted mean of its splats', as
the transfer is merged: the mean of the half traces, which is what a lens
reads, is exact; the anisotropic part is in each splat's own frame and only
indicative. It decodes to the attribute `curvature` (f16 × 3) and pages in
the `relight` group.

Files written before the three tiers (`TXIN` holding the field, `SHAD` and
`MATL` after it, all tier 2: the first pawn conversions) are still read and
paged: a reader takes each block's sections from its table. Their material
is not a prefix of its own, so a page that wants the material and not the
transfer reads the transfer too; convert them again to page them well.

Compression is per section, gzip (RFC 1952) as `DecompressionStream("gzip")`
reads it, since HTTP compression does not apply to Range responses.

### Paging it in the browser

`PagedSplats` reads the tables (header through the starts, one Range
request) and, once the pager knows which attributes it keeps
(`WgpuSplatPager.attributePlan`, see [attributes](webgpu-attributes.md#paging-by-stream-group)),
opens the file:

- the **levels**, one Range request per level block, from `CORE` through
  the last section the kept streams need, sent to the loader worker as
  kind-2 ATHV pages (`athc_v3.rs athv_sections_page`: the stored sections
  and a table of them), from which `athc3_merged_pages` builds the merged
  pages as for v2;
- each **chunk page**, one Range request from `CORE` through the last
  section of the streams that page should bring (`streamsToFetch`: the
  groups held at full capacity, and a paged group while it has a free
  slot). The splats (`CORE`, `SHRS`) go to the worker as a kind-2 page; the
  streams are put together on the main thread as attribute columns of
  their stored words (gunzipped with `DecompressionStream`), with no decode:
  `MATL`'s arrays, `SHAD`, and the transfer's sections joined per splat and
  cut to the form kept;
- an **upgrade** (a page the traversal ranks among a paged group's first
  `slots` that did not bring the group): one Range request of that group's
  sections of the chunk (`fetchAthcStreams`).

So a T1 device fetches 128 bytes a splat of every_stream, plus 24 when a
plugin reads the material; the relight sections only for the pages that
hold them. v2 files page as before (a chunk read whole), and their upgrades
read the arrays of the group (one Range request).

The WASM decoder reads a v3 file whole (`new SplatMesh({ url })`) as it reads
a v2 one.

### Converter

```sh
cargo run -p build-lod --bin athc-convert -- in.athc out.athc [--gzip]  # v1/v2 → v3 (test/fixtures/athc/every_stream.v3.athc: --gzip)
cargo run -p build-lod --bin athc-convert -- in.athc out.athc --v2      # v3 → v2
cargo run -p build-lod --bin athc-convert -- in.athc --info             # sections, encodings, stored bytes, tier sizes
cargo run -p build-lod --bin athc-convert -- in.athc out.athc --gzip --planes   # each section in its smallest encoding
# a light cloud cut from a detailed one, keeping finer groups (down to the
# splats) where the normals spread (athc::truncate_creases):
cargo run -p build-lod --bin athc-convert -- hd.athc light.athc --gzip --planes --keep-splats 200000 --creases 0.03 --crease-depth 1
#   --rebuild-frame SEED: build the levels again in an octree frame turned and shifted by SEED
#   --lod-sizes: the merged nodes' LoD sizes by error (LODS, above)
#   --lod-errors: the merged nodes' error cut bounds (LODE, above)
```

The WASM decoder reads a v3 file whole (`new SplatMesh({ url })`) as it reads
a v2 one.

### From USD: athenea's relightable clouds

athenea writes a cloud with a transfer only as a USD stage (its `.athc`
writer refuses `--transfer`). `usd-athc` reads the
`ParticleField3DGaussianSplat` from the `.usdc` (the `openusd` crate),
packs it as athenea's `scene/streams` + `splat_decode` do, builds the levels
as `LodBuilder` does (`athc_build.rs`; on `two_cards.athc` it gives athenea's
groups, cells and order, each merged Gaussian within f32 rounding) and writes
v3, or v2 with `--v2`.

```sh
cargo run --release -p build-lod --bin usd-athc -- M_Pawn_Body_W.usdc body.athc --gzip --json body.json
#   --transfer full|112|84|64|36|16|9|none   which of the transfer's values to keep
#   --no-shadow --no-material --no-normals --max-sh N --chunk N --prim /World/Splats
cargo run --release -p build-lod --bin usd-athc -- cloud.usdc --list   # attributes, types, lengths
# athenea's merge tree (`athenea decimate --tree`): its leaves as the splats, or
# the cut to N by max-subtree cost, the levels its own nodes (below)
cargo run --release -p build-lod --bin usd-athc -- --tree tree.usdc out.athc [--keep-splats N --error] --lod-sizes --gzip
```

**A merge tree** (`--tree`, surfels-web/NOTES.md "Merge tree"): one cloud
with the constant `lodTree`, its leaves then its merges (children before
parents), every node with its own primvars already merged by athenea,
`lodParent` (a root its own index) and `lodCost` (not monotone). The splats
are the leaves, or with `--keep-splats N` the nodes of the cut to at most N
by the max cost over each subtree. Each level above is a cut of the same
tree: from the level below, the tree's merges are taken cheapest first
while a group holds at most eight of that level's groups (the codes' three
bits), until about four times fewer are left
(`athc_merge::levels_from_tree`). A group that is a node of the tree is
written as athenea's row (every section), not merged again; above the
tree's roots the groups are merged here. Leaves the validation drops leave
the tree; a merge it would drop is refused. Fixture:
`test/fixtures/athc/merge_tree.usda` (`merge_tree.py`).

| USD attribute (`primvars:athenea:splat:` for the athenea ones) | `.athc` |
|---|---|
| `positions`, `opacities` (linear) | positions; opacity under 1/255 or a non-finite record is dropped, as athenea's validate does |
| `orientations` (quatf, w first), `scales` (linear) | shape: smallest three, f16 ln scales |
| `radiance:sphericalHarmonicsCoefficients` (DC first), `…Degree` | base colour 0.5 + SH0 · dc, rest harmonics |
| `linear` | flag bit 1 |
| `normal` | normals |
| `metallic`, `roughness`, `transmission` | pbr |
| `transferDirect` (16), `transferIndirect` (48), `transferReflected` (48) | transfer, 112 f16 in that order (`transfer_layout.slang`) |
| `shadowBits` (int, 8 a splat) | shadowBits |
| `curvature` (3 a splat) | `CURV`, three f16 (v3 only; `--no-curvature` leaves it out) |
| `transferZonal` (10 a splat) | the transfer, 10 values (two zonal lobes in the splat's frame), in place of `transferDirect` |
| `skel:jointIndices`, `skel:jointWeights` (or `athenea:splat:joint*`), `jointWeightGradients` | `SKIN` (v3 only) |
| `skel:joints`, `skel:geomBindTransform`, `skinningXforms` (time samples) | the skeleton, one clip (`--clip NAME`; `--add-clip NAME=other.usdc` adds another conversion's samples of the same rig) |
| `cryptoObject`, `cryptoManifest`, `ior`, `relight` | no room in a `.athc`: `--json` reports the constants |

A reduced transfer keeps a layout athenea reads: 64 is the direct and
indirect halves without the reflected field, 16 the direct half alone, 84
and 36 the same truncated to degree 2 (the first coefficients of each half).

### Not yet

- 083's quantized S0 (~20 bytes: positions relative to the block's sphere,
  8-bit log scales) and flattened SH0p: new `encoding` codes. (034's lossy
  transfer is encoding 3, above.)
- The light-group sidecar `.athl` (066, 069): its own file.
