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
@160     section table, 32 bytes each:
           id (fourcc), tier, encoding (0 v2 words, 1 byte planes,
           2 delta planes), compression (0 none, 1 gzip), words a
           element, 3 x 0
         block index, (32 + 16 × sectionCount) bytes each:
           kind (0 level, 1 chunk), level (1.. or 0), first element, n,
           bounding sphere of the centres (x, y, z, r),
           per section: offset (u64, absolute), stored bytes, raw bytes
aligned  starts: u32 per finest group (as v2)
aligned  blocks: levels coarsest first, then chunks; each on its own page,
         its sections contiguous in table order
```

| Section | id | Tier | Bytes a splat | Contents |
|---|---|---|---|---|
| S0 | `CORE` | 1 | 36 | positions, shape, tail (as three arrays) |
| S0p | `SHRS` | 1 | 4 · shWords | rest harmonics |
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
```

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
| `cryptoObject`, `cryptoManifest`, `ior`, `relight` | no room in a `.athc`: `--json` reports the constants |

A reduced transfer keeps a layout athenea reads: 64 is the direct and
indirect halves without the reflected field, 16 the direct half alone, 84
and 36 the same truncated to degree 2 (the first coefficients of each half).

### Not yet

- 083's quantized S0 (~20 bytes: positions relative to the block's sphere,
  8-bit log scales) and flattened SH0p, and the compressed (lossy) transfer
  of 034: new `encoding` codes.
- The light-group sidecar `.athl` (066, 069): its own file.
