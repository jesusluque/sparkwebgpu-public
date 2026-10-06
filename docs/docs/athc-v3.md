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
| Paging from the browser | `src/athc.ts`, `src/PagedSplats.ts` |
| athenea's decoders over the attribute pool | `slang/athenea_adapter/athc.slang` |
| athenea-web `scene.json` | `src/webgpu/athenea/sceneJson.ts` |

```js
new SplatMesh({ url: "cloud.athc" });               // whole file, athenea's levels as the LoD tree
new SplatMesh({ url: "cloud.athc", paged: true });  // one Range request per 65 536-splat chunk
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

Version 3 rearranges a v2 file; it does not change a value. Every section is
encoding 0 (v2's words), so `v2 → v3 → v2` is the same file byte for byte
(tested on both fixtures, with and without compression). The encodings and
the per-section compression are where 083's smaller sections fit later.

```text
page 0   header, 160 bytes
           0 "ATH3"   4 version 3   8 flags (v2's bits)   12 count
          16 chunkSplats   20 chunks   24 levels   28 finestGroups
          32 restPerColour 36 shWords  40 transferCount 44 shadowWords
          48 pbrWords      52 lobesWords 56 transferWords 60 sectionCount
          64 boundsLo[3]   76 extent   80 boundsMin[3]  92 boundsMax[3]
         104 blockCount (levels + chunks)   108 0
         112 sectionTable (u64)  120 blockIndex  128 starts  136 dataStart
@160     section table, 32 bytes each:
           id (fourcc), tier, encoding (0: v2 words), compression
           (0 none, 1 gzip), words a element, 3 x 0
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
| S1 | `TXDI` | 2 | 4 · ⌈direct / 2⌉ | the transfer's first words: the direct half (9 or 16 values; zonal: all 10) |
| S2 | `TXIN` | 2 | the rest | indirect half and reflected field |
| S3 | `SHAD` | 2 | 4 · shadowWords | open-direction bits |
| S4 | `MATL` | 2 | 4 · (normals + emission + pbr + lobes) | the material streams, as present |

A section is left out when it would be empty. Tier 1 (the splats as
captured) is `CORE + SHRS`; tier 2 (relit) is everything. For
`every_stream.athc` (SH3, TX transfer of 112 values, 16×16 shadow bits,
material): tier 1 is 128 bytes a splat, tier 2 408.

Compression is per section, gzip (RFC 1952) as `DecompressionStream("gzip")`
reads it, since HTTP compression does not apply to Range responses.

### Converter

```sh
cargo run -p build-lod --bin athc-convert -- in.athc out.athc [--gzip]  # v1/v2 → v3
cargo run -p build-lod --bin athc-convert -- in.athc out.athc --v2      # v3 → v2
cargo run -p build-lod --bin athc-convert -- in.athc --info             # sections, tier sizes
```

The WASM decoder reads a v3 file whole (`new SplatMesh({ url })`) as it reads
a v2 one.

### Not yet

- Paging v3 by tier in the browser: the block index already gives each
  (block, tier) its range; `src/athc.ts` pages v2 today.
- 083's quantized S0 (~20 bytes: positions relative to the block's sphere,
  8-bit log scales) and flattened SH0p, and the compressed transfer of 034:
  new `encoding` codes.
- The light-group sidecar `.athl` (066, 069): its own file.
