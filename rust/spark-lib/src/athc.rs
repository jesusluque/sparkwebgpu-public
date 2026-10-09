//! athenea's `.athc`: a splat cloud with athenea's levels of detail.
//!
//! The layout below was read off athenea's own reader and writer
//! (`modules/lod/src/Athc.cpp` and `shaders/athenea/common/packing.slang` at
//! txf 89a04d9), not from a summary of them; `AthcFile::write` writes what
//! `writeAthc` writes, byte for byte (tests check it against a file athenea
//! wrote). Little-endian throughout.
//!
//! ```text
//! page 0      FileHeader (104 bytes); ExtraHeader (32) right after it when
//!             flags has bit 4 or 5
//! page 1      LevelEntry[levels] (16 each), then ChunkEntry[chunks] (16 each)
//! aligned     starts: u32[finestGroups], each finest group's first splat
//! aligned     each level, coarsest first: one block of `groups` elements
//! aligned     each chunk (`chunkSplats` splats, the last holds the rest):
//!             one block of `count` elements
//! ```
//! "aligned" is the next multiple of 4096: every block starts on a page, and
//! the file ends on one. A block is arrays one after the other, each over all
//! of its n elements:
//!
//! ```text
//! positions  f32 x4   x, y, z, opacity (linear 0..1)
//! shape      u32 x4   [0] rotation, smallest-three 10+10+10 bits + the index
//!                         of the dropped component (x y z w = 0 1 2 3)
//!                     [1] f16 ln sx | f16 ln sy << 16
//!                     [2] f16 ln sz | f16 base r << 16
//!                     [3] f16 base g | f16 base b << 16
//!                     (base = 0.5 + SH0 * dc, as Spark's rgb)
//! sh         u32 x shWords   f16 rest coefficients, rgb per basis, two a
//!                     word (low half first); one dummy word at degree 0
//! tail       u32      levels: the group's octree cell code at its level;
//!                     chunks: the splat's finest-level group
//! normals    u32      flags bit 0: octahedral 2 x unorm16 (x low, y high)
//! emission   u32      flags bit 2: linear radiance, RGB9E5
//! pbr        u32 x pbrWords (1)        flags bit 4: metallic | roughness << 8
//!                     | transmission << 16 | thinWalled << 24 | schlick << 25
//! lobes      u32 x lobesWords (0 or 3) packing.slang `packLobes`
//! transfer   u32 x transferWords       flags bit 5: transferCount f16 values,
//!                     two a word (9, 36, 84, 16, 64, 112, or 10 zonal)
//! shadowBits u32 x shadowWords (0, 2, 8 or 32): open directions, a bit a cell
//! curvature  u32 x curvatureWords (0 or 2): sparkwebGPU's, never in a v2
//!                     file (see below)
//! ```
//! `curvature` is not athenea's: it is the per-splat shape operator athenea
//! keeps in USD (`primvars:athenea:splat:curvature`, 3 floats: the 2x2
//! symmetric shape operator in the splat's first two axes, xx xy yy) as
//! three f16 (two words, the last half 0). A `.athc` v3 carries it in its
//! own section (`CURV`, athc_v3.rs); in memory and in ATHV pages its words a
//! splat are `ExtraHeader::curvature_words`, the extra header's sixth word
//! (padding, 0, in every v2 file athenea writes). `AthcFile::write` (v2)
//! leaves it out, so a v2 file stays athenea's byte for byte.
//! Flag bit 1 (linear) says the colours are linear light rather than sRGB;
//! it adds no array. Bit 3 is reserved (athenea's proposal 009) and, like
//! every bit this reader does not know, refused: it may add an array.
//! Version 1 files have no flags (the word was padding, always 0); a file is
//! written as version 2 if and only if its flags are not 0.
//!
//! The levels are an octree over the Morton-sorted splats: a level-r group's
//! cell code is the top 3r bits of its splats' 30-bit Morton codes, so the
//! children of a group with code c are the next level's groups with code
//! c * 8 .. c * 8 + 7, contiguous because each level is sorted, and the
//! children of a finest-level group g are splats starts[g] .. starts[g + 1].
//! That is a Spark LoD tree once it has a single root: see `VirtualTree`.

use std::f32::consts::FRAC_1_SQRT_2;

use anyhow::{anyhow, bail, Result};
use glam::{Mat3, Quat, Vec3A};
use half::f16;
use serde::Serialize;

use crate::attrib::{AttribSpec, LodMerge};
use crate::decoder::{ChunkReceiver, SplatInit, SplatProps, SplatReceiver};
use crate::symmat3::SymMat3;

pub const ATHC_MAGIC: u32 = u32::from_le_bytes(*b"ATHC");
/// One virtual chunk of a .athc, as the pager loads it (`AthcVirtual`).
pub const ATHV_MAGIC: u32 = u32::from_le_bytes(*b"ATHV");
pub const PAGE: u64 = 4096;
pub const HEADER_BYTES: usize = 104;
pub const EXTRA_HEADER_BYTES: usize = 32;
pub const VERSION: u32 = 2;
pub const OLDEST_VERSION: u32 = 1;

pub const FLAG_NORMALS: u32 = 1;
pub const FLAG_LINEAR: u32 = 2;
pub const FLAG_EMISSION: u32 = 4;
pub const FLAG_MATERIAL: u32 = 16;
pub const FLAG_TRANSFER: u32 = 32;
/// sparkwebGPU's LoD sizes by error (`athc_lod_error`): a v3 file's `LODS`
/// section, or a merged ATHV page's sizes. Never in a v2 file.
pub const FLAG_LOD_SIZE: u32 = 64;
pub const KNOWN_FLAGS: u32 = FLAG_NORMALS | FLAG_LINEAR | FLAG_EMISSION | FLAG_MATERIAL | FLAG_TRANSFER | FLAG_LOD_SIZE;

/// Spark's LoD pages (and athenea's default chunk): 65 536 splats.
pub const PAGE_SPLATS: u32 = 65536;

/// A surfel's flat axis (2DGS: a scale of exactly 0) as an f16 ln scale: the
/// most negative finite half, which `exp()` takes back to exactly 0 in f32.
/// ln(0) = -inf would be an f16 infinity, which WGSL's `packHalf2x16` leaves
/// indeterminate; athenea writes its own finite sentinel (-65000) the same way.
pub const SURFEL_LN: f32 = -65504.0;

/// A scale as the f16 ln a shape word holds: [`SURFEL_LN`] for 0 (or less),
/// its log otherwise, never below the sentinel.
pub fn ln_scale(s: f32) -> f32 {
    if s > 0.0 { s.ln().max(SURFEL_LN) } else { SURFEL_LN }
}

/// Whether an f16 ln scale is a surfel's flat axis: at or below the
/// sentinels (ours, athenea's -65000, or an older file's -inf).
pub fn is_flat_ln(ln: f32) -> bool {
    ln <= -60000.0
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AthcHeader {
    pub version: u32,
    pub count: u32,
    pub rest_per_colour: u32,
    pub sh_words: u32,
    pub levels: u32,
    pub chunk_splats: u32,
    pub chunks: u32,
    pub finest_groups: u32,
    pub bounds_lo: [f32; 3],
    pub extent: f32,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub flags: u32,
    pub level_table: u64,
    pub chunk_table: u64,
    pub starts: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtraHeader {
    pub pbr_words: u32,
    pub lobes_words: u32,
    pub transfer_count: u32,
    pub transfer_words: u32,
    pub shadow_words: u32,
    /// sparkwebGPU's curvature stream (0 or 2 words a splat): the sixth
    /// word, 0 in athenea's files; never written to a v2 file.
    pub curvature_words: u32,
    /// sparkwebGPU's skin (`athc_skin`): influences a splat (the seventh
    /// word) and gradient words a splat (the eighth), 0 in athenea's files;
    /// never written to a v2 file.
    pub skin_influences: u32,
    pub skin_gradient_words: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct LevelEntry {
    pub level: u32,
    pub groups: u32,
    pub offset: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ChunkEntry {
    pub offset: u64,
    pub count: u32,
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}
fn f32_at(b: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

pub fn aligned(at: u64) -> u64 {
    at.div_ceil(PAGE) * PAGE
}

impl AthcHeader {
    pub fn has(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }

    /// Spark's SH degree for `restPerColour` (0, 3, 8, 15).
    pub fn sh_degree(&self) -> usize {
        match self.rest_per_colour {
            15 => 3,
            8 => 2,
            3 => 1,
            _ => 0,
        }
    }

    fn from_bytes(b: &[u8]) -> Self {
        let f3 = |at: usize| [f32_at(b, at), f32_at(b, at + 4), f32_at(b, at + 8)];
        Self {
            version: u32_at(b, 4),
            count: u32_at(b, 8),
            rest_per_colour: u32_at(b, 12),
            sh_words: u32_at(b, 16),
            levels: u32_at(b, 20),
            chunk_splats: u32_at(b, 24),
            chunks: u32_at(b, 28),
            finest_groups: u32_at(b, 32),
            bounds_lo: f3(36),
            extent: f32_at(b, 48),
            bounds_min: f3(52),
            bounds_max: f3(64),
            flags: u32_at(b, 76),
            level_table: u64_at(b, 80),
            chunk_table: u64_at(b, 88),
            starts: u64_at(b, 96),
        }
    }

    pub fn to_bytes(&self) -> [u8; HEADER_BYTES] {
        let mut b = [0u8; HEADER_BYTES];
        b[0..4].copy_from_slice(b"ATHC");
        let words = [
            self.version,
            self.count,
            self.rest_per_colour,
            self.sh_words,
            self.levels,
            self.chunk_splats,
            self.chunks,
            self.finest_groups,
        ];
        for (k, w) in words.iter().enumerate() {
            b[4 + 4 * k..8 + 4 * k].copy_from_slice(&w.to_le_bytes());
        }
        let floats = [self.bounds_lo, [self.extent, 0.0, 0.0], self.bounds_min, self.bounds_max];
        let mut at = 36;
        for (k, f) in floats.iter().enumerate() {
            let n = if k == 1 { 1 } else { 3 };
            for v in &f[..n] {
                b[at..at + 4].copy_from_slice(&v.to_le_bytes());
                at += 4;
            }
        }
        b[76..80].copy_from_slice(&self.flags.to_le_bytes());
        b[80..88].copy_from_slice(&self.level_table.to_le_bytes());
        b[88..96].copy_from_slice(&self.chunk_table.to_le_bytes());
        b[96..104].copy_from_slice(&self.starts.to_le_bytes());
        b
    }
}

impl ExtraHeader {
    fn from_bytes(b: &[u8]) -> Self {
        Self {
            pbr_words: u32_at(b, 0),
            lobes_words: u32_at(b, 4),
            transfer_count: u32_at(b, 8),
            transfer_words: u32_at(b, 12),
            shadow_words: u32_at(b, 16),
            curvature_words: u32_at(b, 20),
            skin_influences: u32_at(b, 24),
            skin_gradient_words: u32_at(b, 28),
        }
    }

    pub fn to_bytes(&self) -> [u8; EXTRA_HEADER_BYTES] {
        let mut b = [0u8; EXTRA_HEADER_BYTES];
        for (k, w) in [
            self.pbr_words,
            self.lobes_words,
            self.transfer_count,
            self.transfer_words,
            self.shadow_words,
            self.curvature_words,
            self.skin_influences,
            self.skin_gradient_words,
        ]
        .iter()
            .enumerate()
        {
            b[4 * k..4 * k + 4].copy_from_slice(&w.to_le_bytes());
        }
        b
    }

    pub fn words(&self) -> u32 {
        self.pbr_words + self.lobes_words + self.transfer_words + self.shadow_words + self.curvature_words + self.skin_words()
    }

    /// Skin words a splat (`athc_skin`): influences and gradients.
    pub fn skin_words(&self) -> u32 {
        self.skin_influences + self.skin_gradient_words
    }
}

/// Bytes an element of a block takes, extras included (`elementBytesOf`).
pub fn element_bytes(h: &AthcHeader, x: &ExtraHeader) -> u64 {
    16 + 16
        + 4 * h.sh_words as u64
        + 4
        + if h.has(FLAG_NORMALS) { 4 } else { 0 }
        + if h.has(FLAG_EMISSION) { 4 } else { 0 }
        + 4 * x.words() as u64
}

/// The header page's two headers, checked as Athc.cpp's `parse` checks them:
/// magic, version, flags, extra header consistency and counts.
pub fn parse_headers(bytes: &[u8]) -> Result<(AthcHeader, ExtraHeader)> {
    if bytes.len() < HEADER_BYTES + EXTRA_HEADER_BYTES {
        bail!("not a readable .athc (shorter than its header)");
    }
    if u32_at(bytes, 0) != ATHC_MAGIC {
        bail!("not a readable .athc (no ATHC magic)");
    }
    let mut h = AthcHeader::from_bytes(bytes);
    if h.version < OLDEST_VERSION || h.version > VERSION {
        bail!("not a readable .athc (version {}, this reads {} to {})", h.version, OLDEST_VERSION, VERSION);
    }
    if h.version < 2 {
        h.flags = 0;
    }
    let mut x = ExtraHeader::default();
    if h.flags & (FLAG_MATERIAL | FLAG_TRANSFER) != 0 {
        x = ExtraHeader::from_bytes(&bytes[HEADER_BYTES..]);
    }
    // The curvature word is read whatever the flags: an ATHV head carries
    // it for a v3 cloud; a v2 file holds 0 there (padding, or the page's
    // zeros after a FileHeader alone).
    x.curvature_words = u32_at(bytes, HEADER_BYTES + 20);
    if x.curvature_words != 0 && x.curvature_words != 2 {
        bail!("not a readable .athc (curvature of {} words a splat)", x.curvature_words);
    }
    // So are the skin words (athc_skin): 0 in a v2 file.
    x.skin_influences = u32_at(bytes, HEADER_BYTES + 24);
    x.skin_gradient_words = u32_at(bytes, HEADER_BYTES + 28);
    if x.skin_influences > 16 || x.skin_gradient_words != 0 && x.skin_gradient_words + 1 != x.skin_influences {
        bail!(
            "not a readable .athc (skin of {} influences and {} gradient words)",
            x.skin_influences,
            x.skin_gradient_words
        );
    }
    if h.flags & (FLAG_MATERIAL | FLAG_TRANSFER) != 0 {
        if h.has(FLAG_MATERIAL) != (x.pbr_words == 1)
            || (x.lobes_words != 0 && x.lobes_words != 3)
            || h.has(FLAG_TRANSFER) != (x.transfer_words != 0)
            || x.transfer_words > 64
            || ![0, 2, 8, 32].contains(&x.shadow_words)
            || x.transfer_words != x.transfer_count.div_ceil(2)
        {
            bail!("not a readable .athc (inconsistent extra header)");
        }
    }
    let unknown = h.flags & !KNOWN_FLAGS;
    if unknown != 0 {
        let bits: Vec<String> = (0..32).filter(|b| unknown & (1 << b) != 0).map(|b| b.to_string()).collect();
        bail!(
            "not a readable .athc (unknown flag bits {}; this reads bits 0 (normals), 1 (linear), 2 (emission), 4 (material), 5 (transfer) and 6 (LoD sizes))",
            bits.join(", ")
        );
    }
    if h.count == 0
        || h.levels == 0
        || h.chunk_splats == 0
        || h.sh_words == 0
        || h.sh_words > 64
        || h.chunks as u64 != (h.count as u64).div_ceil(h.chunk_splats as u64)
    {
        bail!("not a readable .athc (inconsistent counts)");
    }
    Ok((h, x))
}

/// Everything but the blocks: the headers and tables, from the file's first
/// bytes (through the end of the chunk table; `prefix_bytes` says how many),
/// checked against the file's length as Athc.cpp checks them.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AthcLayout {
    pub header: AthcHeader,
    pub extra: ExtraHeader,
    pub levels: Vec<LevelEntry>,
    pub chunks: Vec<ChunkEntry>,
    pub element_bytes: u64,
    pub file_bytes: u64,
}

impl AthcLayout {
    /// The bytes `parse` needs: through the end of the chunk table.
    pub fn prefix_bytes(bytes: &[u8]) -> Result<u64> {
        let (h, _) = parse_headers(bytes)?;
        Ok((h.level_table + 16 * h.levels as u64).max(h.chunk_table + 16 * h.chunks as u64).max(PAGE))
    }

    pub fn parse(bytes: &[u8], file_bytes: u64) -> Result<Self> {
        if (bytes.len() as u64) < PAGE.min(file_bytes) || file_bytes < PAGE {
            bail!("not a readable .athc (shorter than its header)");
        }
        let (h, x) = parse_headers(bytes)?;
        let within = |offset: u64, size: u64| offset <= file_bytes && size <= file_bytes - offset;
        if !within(h.level_table, h.levels as u64 * 16)
            || !within(h.chunk_table, h.chunks as u64 * 16)
            || !within(h.starts, h.finest_groups as u64 * 4)
        {
            bail!("not a readable .athc (tables past the end)");
        }
        let need = Self::prefix_bytes(bytes)?;
        if (bytes.len() as u64) < need {
            bail!("need the first {} bytes of the .athc for its tables", need);
        }
        let levels: Vec<LevelEntry> = (0..h.levels as usize)
            .map(|l| {
                let at = h.level_table as usize + 16 * l;
                LevelEntry { level: u32_at(bytes, at), groups: u32_at(bytes, at + 4), offset: u64_at(bytes, at + 8) }
            })
            .collect();
        let chunks: Vec<ChunkEntry> = (0..h.chunks as usize)
            .map(|c| {
                let at = h.chunk_table as usize + 16 * c;
                ChunkEntry { offset: u64_at(bytes, at), count: u32_at(bytes, at + 8) }
            })
            .collect();
        let per = element_bytes(&h, &x);
        for (l, e) in levels.iter().enumerate() {
            if e.groups == 0
                || !within(e.offset, e.groups as u64 * per)
                || (l > 0 && e.level != levels[l - 1].level + 1)
                || e.level == 0
                || e.level > 10
            {
                bail!("not a readable .athc (level {})", l);
            }
        }
        if levels.last().unwrap().groups != h.finest_groups {
            bail!("not a readable .athc (finest level's groups)");
        }
        for (c, e) in chunks.iter().enumerate() {
            let expected = h.chunk_splats.min(h.count - c as u32 * h.chunk_splats);
            if e.count != expected || !within(e.offset, e.count as u64 * per) {
                bail!("not a readable .athc (chunk {})", c);
            }
        }
        Ok(Self { header: h, extra: x, levels, chunks, element_bytes: per, file_bytes })
    }

    /// Where the levels' blocks end: what a stream reads before any chunk
    /// (headers, tables, starts and every level).
    pub fn levels_end(&self) -> u64 {
        let last = self.levels.last().unwrap();
        last.offset + last.groups as u64 * self.element_bytes
    }

    pub fn max_sh_degree(&self) -> usize {
        self.header.sh_degree()
    }
}

/// The arrays of one block, owned. An absent optional array is empty.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AthcBlock {
    pub n: usize,
    pub positions: Vec<f32>,
    pub shape: Vec<u32>,
    pub sh: Vec<u32>,
    pub tail: Vec<u32>,
    pub normals: Vec<u32>,
    pub emission: Vec<u32>,
    pub pbr: Vec<u32>,
    pub lobes: Vec<u32>,
    pub transfer: Vec<u32>,
    pub shadow_bits: Vec<u32>,
    /// sparkwebGPU's curvature, two words a splat (three f16), or empty.
    pub curvature: Vec<u32>,
    /// sparkwebGPU's skin (`athc_skin`): influences then gradient words a
    /// splat, or empty.
    pub skin: Vec<u32>,
    /// sparkwebGPU's LoD size by error (`athc_lod_error`, v3 section LODS):
    /// one a merged node, 0 for a splat (its geometric size), or empty.
    pub lod_size: Vec<f32>,
}

fn words_of(b: &[u8], at: &mut usize, n: usize) -> Vec<u32> {
    let out = b[*at..*at + 4 * n].chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect();
    *at += 4 * n;
    out
}

impl AthcBlock {
    /// A block of `n` elements at the start of `bytes` (`blockAt`).
    pub fn read(bytes: &[u8], n: usize, h: &AthcHeader, x: &ExtraHeader) -> Result<Self> {
        let need = n as u64 * element_bytes(h, x);
        if (bytes.len() as u64) < need {
            bail!("a block of {} elements needs {} bytes, has {}", n, need, bytes.len());
        }
        let mut at = 0;
        let positions = words_of(bytes, &mut at, 4 * n).into_iter().map(f32::from_bits).collect();
        let shape = words_of(bytes, &mut at, 4 * n);
        let sh = words_of(bytes, &mut at, h.sh_words as usize * n);
        let tail = words_of(bytes, &mut at, n);
        let opt = |on: bool, at: &mut usize, words: u32| if on { words_of(bytes, at, words as usize * n) } else { Vec::new() };
        let normals = opt(h.has(FLAG_NORMALS), &mut at, 1);
        let emission = opt(h.has(FLAG_EMISSION), &mut at, 1);
        // The extras in `extrasOf`'s order: pbr, lobes, transfer, shadowBits.
        let pbr = opt(x.pbr_words > 0, &mut at, x.pbr_words);
        let lobes = opt(x.lobes_words > 0, &mut at, x.lobes_words);
        let transfer = opt(x.transfer_words > 0, &mut at, x.transfer_words);
        let shadow_bits = opt(x.shadow_words > 0, &mut at, x.shadow_words);
        let curvature = opt(x.curvature_words > 0, &mut at, x.curvature_words);
        let skin = opt(x.skin_words() > 0, &mut at, x.skin_words());
        Ok(Self { n, positions, shape, sh, tail, normals, emission, pbr, lobes, transfer, shadow_bits, curvature, skin, lod_size: Vec::new() })
    }

    /// Every array after positions, in file order.
    fn arrays(&self) -> [&[u32]; 11] {
        [
            &self.shape,
            &self.sh,
            &self.tail,
            &self.normals,
            &self.emission,
            &self.pbr,
            &self.lobes,
            &self.transfer,
            &self.shadow_bits,
            &self.curvature,
            &self.skin,
        ]
    }

    pub fn write(&self, out: &mut Vec<u8>) {
        for v in &self.positions {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for words in self.arrays() {
            for w in words {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
    }

    /// Elements start .. start + n.
    pub fn slice(&self, start: usize, n: usize) -> Self {
        let cut = |v: &Vec<u32>| -> Vec<u32> {
            if v.is_empty() {
                return Vec::new();
            }
            let per = v.len() / self.n;
            v[start * per..(start + n) * per].to_vec()
        };
        Self {
            n,
            positions: self.positions[start * 4..(start + n) * 4].to_vec(),
            shape: cut(&self.shape),
            sh: cut(&self.sh),
            tail: cut(&self.tail),
            normals: cut(&self.normals),
            emission: cut(&self.emission),
            pbr: cut(&self.pbr),
            lobes: cut(&self.lobes),
            transfer: cut(&self.transfer),
            shadow_bits: cut(&self.shadow_bits),
            curvature: cut(&self.curvature),
            skin: cut(&self.skin),
            lod_size: if self.lod_size.is_empty() { Vec::new() } else { self.lod_size[start..start + n].to_vec() },
        }
    }

    pub fn append(&mut self, other: &Self) {
        self.n += other.n;
        self.positions.extend_from_slice(&other.positions);
        self.shape.extend_from_slice(&other.shape);
        self.sh.extend_from_slice(&other.sh);
        self.tail.extend_from_slice(&other.tail);
        self.normals.extend_from_slice(&other.normals);
        self.emission.extend_from_slice(&other.emission);
        self.pbr.extend_from_slice(&other.pbr);
        self.lobes.extend_from_slice(&other.lobes);
        self.transfer.extend_from_slice(&other.transfer);
        self.shadow_bits.extend_from_slice(&other.shadow_bits);
        self.curvature.extend_from_slice(&other.curvature);
        self.skin.extend_from_slice(&other.skin);
        // A block with sizes and one without: the latter's are 0 (geometric).
        if !self.lod_size.is_empty() || !other.lod_size.is_empty() {
            self.lod_size.resize(self.n - other.n, 0.0);
            if other.lod_size.is_empty() {
                self.lod_size.resize(self.n, 0.0);
            } else {
                self.lod_size.extend_from_slice(&other.lod_size);
            }
        }
    }

    /// Words per element of an array of this block (0 when absent).
    fn per(&self, v: &[u32]) -> usize {
        v.len().checked_div(self.n).unwrap_or(0)
    }
}

/// A whole .athc in memory.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AthcFile {
    pub header: AthcHeader,
    pub extra: ExtraHeader,
    /// Each level's number and its groups, coarsest first.
    pub levels: Vec<(u32, AthcBlock)>,
    pub starts: Vec<u32>,
    pub chunks: Vec<AthcBlock>,
}

impl AthcFile {
    pub fn read(bytes: &[u8]) -> Result<Self> {
        let layout = AthcLayout::parse(bytes, bytes.len() as u64)?;
        let h = layout.header;
        let x = layout.extra;
        let levels = layout
            .levels
            .iter()
            .map(|e| Ok((e.level, AthcBlock::read(&bytes[e.offset as usize..], e.groups as usize, &h, &x)?)))
            .collect::<Result<Vec<_>>>()?;
        let mut at = h.starts as usize;
        let starts = words_of(bytes, &mut at, h.finest_groups as usize);
        let chunks = layout
            .chunks
            .iter()
            .map(|e| AthcBlock::read(&bytes[e.offset as usize..], e.count as usize, &h, &x))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { header: h, extra: x, levels, starts, chunks })
    }

    /// The file `writeAthc` writes for this cloud: version, flags, counts and
    /// every offset recomputed from the arrays; bounds as they are.
    pub fn write(&self) -> Result<Vec<u8>> {
        if self.has_curvature() {
            return self.without_curvature().write();
        }
        if self.has_skin() {
            return self.without_skin().write();
        }
        if self.over_capped() {
            return self.capped().write();
        }
        let first = self.chunks.first().ok_or_else(|| anyhow!("a .athc holds at least one chunk"))?;
        let last_level = &self.levels.last().ok_or_else(|| anyhow!("a .athc holds at least one level"))?.1;
        let mut h = self.header;
        let mut x = ExtraHeader::default();
        if !first.pbr.is_empty() {
            x.pbr_words = 1;
            x.lobes_words = first.per(&first.lobes) as u32;
        }
        if !first.transfer.is_empty() {
            x.transfer_count = self.extra.transfer_count;
            x.transfer_words = first.per(&first.transfer) as u32;
            x.shadow_words = first.per(&first.shadow_bits) as u32;
        }
        h.flags = (if first.normals.is_empty() { 0 } else { FLAG_NORMALS })
            | (h.flags & FLAG_LINEAR)
            | (if first.emission.is_empty() { 0 } else { FLAG_EMISSION })
            | (if x.pbr_words > 0 { FLAG_MATERIAL } else { 0 })
            | (if x.transfer_words > 0 { FLAG_TRANSFER } else { 0 });
        h.version = if h.flags != 0 { VERSION } else { OLDEST_VERSION };
        h.count = self.chunks.iter().map(|c| c.n as u32).sum();
        h.sh_words = (first.sh.len() / first.n.max(1)) as u32;
        h.levels = self.levels.len() as u32;
        h.chunks = self.chunks.len() as u32;
        h.finest_groups = last_level.n as u32;
        h.level_table = PAGE;
        h.chunk_table = h.level_table + 16 * h.levels as u64;
        h.starts = aligned(h.chunk_table + 16 * h.chunks as u64);
        let per = element_bytes(&h, &x);
        let mut at = aligned(h.starts + 4 * h.finest_groups as u64);
        let mut level_entries = Vec::new();
        for (level, block) in &self.levels {
            level_entries.push(LevelEntry { level: *level, groups: block.n as u32, offset: at });
            at = aligned(at + block.n as u64 * per);
        }
        let mut chunk_entries = Vec::new();
        for block in &self.chunks {
            chunk_entries.push(ChunkEntry { offset: at, count: block.n as u32 });
            at = aligned(at + block.n as u64 * per);
        }
        let mut out = Vec::with_capacity(at as usize);
        out.extend_from_slice(&h.to_bytes());
        if h.flags & (FLAG_MATERIAL | FLAG_TRANSFER) != 0 {
            out.extend_from_slice(&x.to_bytes());
        }
        out.resize(h.level_table as usize, 0);
        for e in &level_entries {
            out.extend_from_slice(&e.level.to_le_bytes());
            out.extend_from_slice(&e.groups.to_le_bytes());
            out.extend_from_slice(&e.offset.to_le_bytes());
        }
        for e in &chunk_entries {
            out.extend_from_slice(&e.offset.to_le_bytes());
            out.extend_from_slice(&e.count.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
        }
        out.resize(h.starts as usize, 0);
        for s in &self.starts {
            out.extend_from_slice(&s.to_le_bytes());
        }
        for ((_, block), e) in self.levels.iter().zip(&level_entries) {
            out.resize(e.offset as usize, 0);
            block.write(&mut out);
        }
        for (block, e) in self.chunks.iter().zip(&chunk_entries) {
            out.resize(e.offset as usize, 0);
            block.write(&mut out);
        }
        out.resize(aligned(out.len() as u64) as usize, 0);
        Ok(out)
    }

    /// Whether a block carries sparkwebGPU's curvature stream.
    pub fn has_curvature(&self) -> bool {
        self.chunks.iter().chain(self.levels.iter().map(|(_, b)| b)).any(|b| !b.curvature.is_empty())
    }

    /// Whether a block carries sparkwebGPU's skin stream.
    pub fn has_skin(&self) -> bool {
        self.chunks.iter().chain(self.levels.iter().map(|(_, b)| b)).any(|b| !b.skin.is_empty())
    }

    /// The cloud without its skin (what a v2 file holds).
    pub fn without_skin(&self) -> Self {
        let mut out = self.clone();
        out.extra.skin_influences = 0;
        out.extra.skin_gradient_words = 0;
        for b in out.chunks.iter_mut().chain(out.levels.iter_mut().map(|(_, b)| b)) {
            b.skin.clear();
        }
        out
    }

    /// The cloud without its curvature (what a v2 file holds).
    pub fn without_curvature(&self) -> Self {
        let mut out = self.clone();
        out.extra.curvature_words = 0;
        for b in out.chunks.iter_mut().chain(out.levels.iter_mut().map(|(_, b)| b)) {
            b.curvature.clear();
        }
        out
    }

    /// The splats of every chunk, as one block.
    /// Whether a merged opacity is over athenea's 0.99 (`uncap_levels`): a
    /// level's, or a splat's past 1 (a truncated cloud's splats are merged).
    pub fn over_capped(&self) -> bool {
        let over = |b: &AthcBlock, cap: f32| b.positions.chunks_exact(4).any(|p| p[3] > cap);
        self.levels.iter().any(|(_, b)| over(b, 0.99)) || self.chunks.iter().any(|b| over(b, 1.0))
    }

    /// Merged opacities at most athenea's 0.99, as a v2 file holds them.
    pub fn capped(&self) -> Self {
        let mut out = self.clone();
        for (b, cap) in out.chunks.iter_mut().map(|b| (b, 1.0)).chain(out.levels.iter_mut().map(|(_, b)| (b, 0.99))) {
            for p in b.positions.chunks_exact_mut(4) {
                if p[3] > cap {
                    p[3] = 0.99;
                }
            }
        }
        out
    }

    pub fn splats(&self) -> AthcBlock {
        let mut all = AthcBlock::default();
        for c in &self.chunks {
            all.append(c);
        }
        all
    }
}

// --- packing.slang, on the CPU ---------------------------------------------

/// `decodeQuaternion`: smallest-three word to (x, y, z, w).
pub fn decode_quaternion(word: u32) -> [f32; 4] {
    let largest = (word & 3) as usize;
    let small = [(word >> 22) & 1023, (word >> 12) & 1023, (word >> 2) & 1023]
        .map(|v| (v as f32 / 1023.0 * 2.0 - 1.0) * FRAC_1_SQRT_2);
    let rest = (1.0 - small.iter().map(|v| v * v).sum::<f32>()).max(0.0).sqrt();
    let mut q = [0.0; 4];
    let mut next = 0;
    for (k, v) in q.iter_mut().enumerate() {
        if k == largest {
            *v = rest;
        } else {
            *v = small[next];
            next += 1;
        }
    }
    q
}

/// `encodeQuaternion`: (x, y, z, w) to its smallest-three word.
pub fn encode_quaternion(q: [f32; 4]) -> u32 {
    let len = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    let mut q = q.map(|v| v / len);
    let mut largest = 0;
    for k in 1..4 {
        if q[k].abs() > q[largest].abs() {
            largest = k;
        }
    }
    if q[largest] < 0.0 {
        q = q.map(|v| -v);
    }
    let mut word = largest as u32;
    let mut shift = 22;
    for (k, v) in q.iter().enumerate() {
        if k == largest {
            continue;
        }
        let unit = (v / FRAC_1_SQRT_2 * 0.5 + 0.5).clamp(0.0, 1.0);
        word |= ((unit * 1023.0).round() as u32) << shift;
        shift -= 10;
    }
    word
}

pub fn low_half(word: u32) -> f32 {
    f16::from_bits(word as u16).to_f32()
}
pub fn high_half(word: u32) -> f32 {
    f16::from_bits((word >> 16) as u16).to_f32()
}
pub fn pack_halves(a: f32, b: f32) -> u32 {
    f16::from_f32(a).to_bits() as u32 | ((f16::from_f32(b).to_bits() as u32) << 16)
}

/// `unpackNormal`: octahedral 2 x unorm16 to a unit vector.
pub fn unpack_normal(word: u32) -> [f32; 3] {
    let f = [(word & 0xffff) as f32 / 65535.0 * 2.0 - 1.0, (word >> 16) as f32 / 65535.0 * 2.0 - 1.0];
    let mut n = [f[0], f[1], 1.0 - f[0].abs() - f[1].abs()];
    let t = (-n[2]).clamp(0.0, 1.0);
    n[0] += if n[0] >= 0.0 { -t } else { t };
    n[1] += if n[1] >= 0.0 { -t } else { t };
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    n.map(|v| v / l)
}

/// `packNormal`.
pub fn pack_normal(n: [f32; 3]) -> u32 {
    let l1 = n[0].abs() + n[1].abs() + n[2].abs();
    let n = if l1 > 1.0e-20 { n.map(|v| v / l1) } else { [0.0, 0.0, 1.0] };
    let mut p = [n[0], n[1]];
    if n[2] < 0.0 {
        let sx = if n[0] >= 0.0 { 1.0 } else { -1.0 };
        let sy = if n[1] >= 0.0 { 1.0 } else { -1.0 };
        p = [(1.0 - n[1].abs()) * sx, (1.0 - n[0].abs()) * sy];
    }
    let q = p.map(|v| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 65535.0).round() as u32);
    q[0] | (q[1] << 16)
}

/// `unpackRgb9e5`: linear RGB.
pub fn unpack_rgb9e5(word: u32) -> [f32; 3] {
    let step = 2f32.powi((word >> 27) as i32 - 24);
    [(word & 511) as f32 * step, ((word >> 9) & 511) as f32 * step, ((word >> 18) & 511) as f32 * step]
}

fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 { 12.92 * v.max(0.0) } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// d(sRGB)/d(linear) at v, to carry a linear cloud's rest harmonics over.
fn linear_to_srgb_slope(v: f32) -> f32 {
    if v <= 0.003_130_8 { 12.92 } else { 1.055 / 2.4 * v.max(1e-6).powf(1.0 / 2.4 - 1.0) }
}

// --- the LoD tree ----------------------------------------------------------

/// athenea's levels as one Spark LoD tree, in a virtual index space:
///
/// ```text
/// 0                     the root: level 1's only group, or one merged from
///                       level 1's groups when it has several
/// level_base[l] ..      level l's groups, coarsest level first
/// merged ..             (paged: a gap to the next multiple of 65 536)
/// splat_base ..         the cloud's splats, chunk after chunk
/// ```
///
/// With `splat_base` a multiple of 65 536 each athenea chunk of 65 536
/// splats is exactly one Spark page, read with one range request.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VirtualTree {
    pub synth_root: bool,
    pub level_base: Vec<u32>,
    pub merged: u32,
    pub splat_base: u32,
    pub count: u32,
    /// Children of each merged node (virtual indices), `merged` of each.
    #[serde(skip)]
    pub child_start: Vec<u32>,
    #[serde(skip)]
    pub child_count: Vec<u16>,
    /// Each merged node's finest groups, `[lo, hi)` as two words: a node
    /// covers the finest groups lo .. hi (Morton order keeps a subtree's
    /// groups contiguous). A splat's is its own finest group, `[g, g + 1)`.
    /// What the `athcGroup` attribute carries (`GROUP_ATTRIBUTE`).
    #[serde(skip)]
    pub group_range: Vec<u32>,
    /// Finest groups with more than 65 535 splats: Spark's child count is
    /// 16 bits, so the rest of those splats are left out of the tree.
    pub clipped_groups: u32,
}

impl VirtualTree {
    /// `cells`: each level's cell codes (`tail` of its block), coarsest first.
    pub fn build(cells: &[&[u32]], starts: &[u32], count: u32, page_aligned: bool) -> Result<Self> {
        let levels = cells.len();
        if levels == 0 {
            bail!(".athc without levels");
        }
        for (l, c) in cells.iter().enumerate() {
            if c.windows(2).any(|w| w[0] >= w[1]) {
                bail!(".athc level {} is not in Morton order", l);
            }
        }
        let synth_root = cells[0].len() > 1;
        let mut level_base = Vec::with_capacity(levels);
        let mut next = if synth_root { 1u32 } else { 0 };
        for c in cells {
            level_base.push(next);
            next += c.len() as u32;
        }
        let merged = next;
        let splat_base = if page_aligned { merged.div_ceil(PAGE_SPLATS) * PAGE_SPLATS } else { merged };
        let mut child_start = vec![0u32; merged as usize];
        let mut child_count = vec![0u16; merged as usize];
        if synth_root {
            child_start[0] = level_base[0];
            child_count[0] = cells[0].len() as u16;
        }
        for l in 0..levels - 1 {
            let (parents, children) = (cells[l], cells[l + 1]);
            let mut j = 0;
            for (i, &code) in parents.iter().enumerate() {
                let first = j;
                while j < children.len() && children[j] >> 3 == code {
                    j += 1;
                }
                if j == first {
                    bail!(".athc level {} group {} (cell {}) has no children", l, i, code);
                }
                let node = (level_base[l] + i as u32) as usize;
                child_start[node] = level_base[l + 1] + first as u32;
                child_count[node] = (j - first) as u16;
            }
            if j != children.len() {
                bail!(".athc level {} has groups without a parent", l + 1);
            }
        }
        let finest = cells[levels - 1];
        if starts.len() != finest.len() {
            bail!(".athc starts: {} for {} finest groups", starts.len(), finest.len());
        }
        let mut clipped_groups = 0;
        for g in 0..starts.len() {
            let end = if g + 1 < starts.len() { starts[g + 1] } else { count };
            if end <= starts[g] || end > count {
                bail!(".athc finest group {} holds splats {}..{} of {}", g, starts[g], end, count);
            }
            let n = end - starts[g];
            if n > u16::MAX as u32 {
                clipped_groups += 1;
            }
            let node = (level_base[levels - 1] + g as u32) as usize;
            child_start[node] = splat_base + starts[g];
            child_count[node] = n.min(u16::MAX as u32) as u16;
        }
        // Finest groups under each merged node, bottom up: a finest-level
        // group is its own, a parent spans its first child's to its last's.
        let mut group_range = vec![0u32; 2 * merged as usize];
        for g in 0..finest.len() {
            let node = (level_base[levels - 1] + g as u32) as usize;
            group_range[2 * node] = g as u32;
            group_range[2 * node + 1] = g as u32 + 1;
        }
        let parents = (0..levels - 1).rev().flat_map(|l| (0..cells[l].len()).map(move |i| (l, i)));
        let roots = synth_root.then_some(0usize);
        for node in parents.map(|(l, i)| (level_base[l] + i as u32) as usize).chain(roots) {
            let first = child_start[node] as usize;
            let last = first + child_count[node] as usize - 1;
            group_range[2 * node] = group_range[2 * first];
            group_range[2 * node + 1] = group_range[2 * last + 1];
        }
        Ok(Self {
            synth_root,
            level_base,
            merged,
            splat_base,
            count,
            child_start,
            child_count,
            group_range,
            clipped_groups,
        })
    }

    pub fn of_file(file: &AthcFile, page_aligned: bool) -> Result<Self> {
        let cells: Vec<&[u32]> = file.levels.iter().map(|(_, b)| &b.tail[..]).collect();
        Self::build(&cells, &file.starts, file.header.count, page_aligned)
    }

    /// Virtual pages: those of the merged nodes, then one per athenea chunk
    /// (when `splat_base` is page-aligned and chunks are 65 536 splats).
    pub fn merged_pages(&self) -> u32 {
        self.merged.div_ceil(PAGE_SPLATS)
    }
}

/// The root merged from level 1's groups, as athenea merges a group
/// (lod_common.slang): weights opacity x the area of the two longest axes,
/// moments of position and covariance, colours and harmonics by weight,
/// opacity W / own area (athenea caps it at 0.99; see `coverage_ratios`).
/// Everything else is the heaviest group's.
pub fn merge_root(level1: &AthcBlock, sh_words: usize) -> AthcBlock {
    let n = level1.n;
    let mut weights = Vec::with_capacity(n);
    let mut covs = Vec::with_capacity(n);
    let mut sum_w = 0.0f64;
    let mut share = 0.0f64;
    let mut mu = [0.0f64; 3];
    for i in 0..n {
        let p = &level1.positions[i * 4..i * 4 + 4];
        let s = &level1.shape[i * 4..i * 4 + 4];
        let q = decode_quaternion(s[0]);
        let scale = [low_half(s[1]).exp(), high_half(s[1]).exp(), low_half(s[2]).exp()];
        let mut sorted = scale;
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let w = (p[3] * sorted[0] * sorted[1]).max(1e-20) as f64;
        weights.push(w);
        share += (merged_share(level1, i, p[3]) * sorted[0] * sorted[1]) as f64;
        covs.push(SymMat3::new_scale_quaternion(Vec3A::from_array(scale), Quat::from_array(q)));
        sum_w += w;
        for d in 0..3 {
            mu[d] += w * p[d] as f64;
        }
    }
    let mu = mu.map(|v| v / sum_w);
    let mut cov = SymMat3::new_zeros();
    let mut base = [0.0f64; 3];
    let mut rest = vec![0.0f64; sh_words * 2];
    let mut heaviest = 0;
    for i in 0..n {
        let w = weights[i];
        if w > weights[heaviest] {
            heaviest = i;
        }
        let p = &level1.positions[i * 4..i * 4 + 3];
        let d = [p[0] as f64 - mu[0], p[1] as f64 - mu[1], p[2] as f64 - mu[2]];
        let outer = SymMat3::new([
            (d[0] * d[0]) as f32,
            (d[1] * d[1]) as f32,
            (d[2] * d[2]) as f32,
            (d[0] * d[1]) as f32,
            (d[0] * d[2]) as f32,
            (d[1] * d[2]) as f32,
        ]);
        let wn = (w / sum_w) as f32;
        cov.add_weighted(&covs[i], wn);
        cov.add_weighted(&outer, wn);
        let s = &level1.shape[i * 4..i * 4 + 4];
        let rgb = [high_half(s[2]), low_half(s[3]), high_half(s[3])];
        for c in 0..3 {
            base[c] += w * rgb[c] as f64;
        }
        for k in 0..sh_words * 2 {
            let word = level1.sh[i * sh_words + k / 2];
            rest[k] += w * if k % 2 == 0 { low_half(word) } else { high_half(word) } as f64;
        }
    }
    let (values, vectors) = cov.positive_eigens();
    let mut axes = Mat3::from_cols(vectors[0].into(), vectors[1].into(), vectors[2].into());
    if axes.determinant() < 0.0 {
        axes = Mat3::from_cols(vectors[0].into(), vectors[1].into(), (-vectors[2]).into());
    }
    let q = Quat::from_mat3(&axes).normalize();
    let scale = values.map(|v| v.max(1e-12).sqrt());
    let mut sorted = scale;
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    // Uncapped: see `coverage_ratios`.
    let opacity = if is_sheet(level1, heaviest) {
        (share as f32) / (sorted[0] * sorted[1]).max(1e-20) + SHEET_PAD
    } else {
        (sum_w as f32) / (sorted[0] * sorted[1]).max(1e-20)
    };
    let base = base.map(|v| (v / sum_w) as f32);
    let mut root = level1.slice(heaviest, 1);
    root.positions = vec![mu[0] as f32, mu[1] as f32, mu[2] as f32, opacity];
    root.shape = vec![
        encode_quaternion(q.to_array()),
        pack_halves(ln_scale(scale[0]), ln_scale(scale[1])),
        pack_halves(ln_scale(scale[2]), base[0]),
        pack_halves(base[1], base[2]),
    ];
    root.sh = (0..sh_words)
        .map(|k| pack_halves((rest[2 * k] / sum_w) as f32, (rest[2 * k + 1] / sum_w) as f32))
        .collect();
    root.tail = vec![0];
    // A LoD size at least its children's (athc_lod_error).
    if !level1.lod_size.is_empty() {
        root.lod_size = vec![level1.lod_size.iter().cloned().fold(0.0, f32::max)];
    }
    root
}

/// The merged nodes in virtual order (root, then every level), one block,
/// each widened to tile its cell (`widen_merged`).
pub fn merged_block(file: &AthcFile, tree: &VirtualTree) -> AthcBlock {
    let extent = file.header.extent;
    let mut all = if tree.synth_root {
        let mut root = merge_root(&file.levels[0].1, file.header.sh_words as usize);
        widen_merged(&mut root, extent);
        root
    } else {
        AthcBlock::default()
    };
    for (level, block) in &file.levels {
        let mut block = block.clone();
        widen_merged(&mut block, extent / (1u64 << *level) as f32);
        all.append(&block);
    }
    all
}

// --- coverage: merged opacity past athenea's 0.99 ---------------------------
//
// athenea gives a group opacity min(W / A, 0.99): W the sum over its splats of
// opacity x the area of their two longest axes, A its own two-axis area. On a
// closed shell W / A is 2.5 at the finest level and 10-12 from the third up
// (the splats overlap several deep), and capping it at 0.99 throws most of the
// group away: a surface tiled by gaussians of peak 0.99, each its cell wide,
// is half transparent once a cell is a pixel or less (the antialiasing pays a
// sub-pixel gaussian back by its mass, 0.99 x 2 pi sigma^2 = 0.52 of its
// cell). Seen far off, the Corvette's paint let through 12% of what was behind
// it at 14 m and 26% at 29 m, where its own splats let through 0.04%.
//
// So the ratio is kept whole (`coverage_ratios`, `uncap_levels`): a .athc this
// crate writes as v3 may carry merged opacities above 1, and the decoder turns
// any opacity above 1 into Spark's LoD opacity (`spark_lod_opacity`), which
// Spark draws as 1 - (1 - g)^ratio: `ratio` gaussians composited, the way the
// splats under the group composite. A v2 file is written capped, as athenea's.

/// athenea's coverage alpha on a thin sheet of glass (mesh2splat's
/// m2sCoverageAlpha): what the sheet reflects head on, plus the 1/255 it adds
/// for the radius the gaussian would be cut at. splat_project draws a sheet
/// at its opacity less that 1/255 (`alphaOwn`), so the 1/255 is a splat's,
/// not an area's: a merged sheet whose opacity is W / A of its splats' whole
/// opacities kept the 1/255 of every splat under it as reflection. On the
/// Corvette's windshield (0.0057 a splat, 0.0018 of it reflection) the
/// merged levels reflected 2.2-2.9 times what their splats do, and the
/// headlights 2.7-3.2: bright dots wherever the LoD drew a merged cell
/// among splats. A sheet's coverage is merged as its reflection (opacity
/// less the 1/255) and the 1/255 added back once.
pub const SHEET_PAD: f32 = 1.0 / 255.0;

/// Whether element i is a thin sheet of glass (splat_project's `sheet`:
/// thin-walled, transmission past one half).
pub fn is_sheet(block: &AthcBlock, i: usize) -> bool {
    if block.pbr.is_empty() || block.n == 0 {
        return false;
    }
    let words = block.pbr.len() / block.n;
    let w = block.pbr[i * words];
    (w >> 24) & 1 == 1 && ((w >> 16) & 255) as f32 / 255.0 > 0.5
}

/// What of element i's opacity merges by area: all of it, or a sheet's
/// reflection (`SHEET_PAD`).
fn merged_share(block: &AthcBlock, i: usize, o: f32) -> f32 {
    if is_sheet(block, i) {
        (o - SHEET_PAD).max(0.0)
    } else {
        o
    }
}

/// The two longest of a shape word's three scales, multiplied: a gaussian's
/// area (a surfel's too, whose third is 0).
pub fn two_axis_area(shape: &[u32]) -> f32 {
    let mut s = [low_half(shape[1]).exp(), high_half(shape[1]).exp(), low_half(shape[2]).exp()];
    s.sort_by(|a, b| b.total_cmp(a));
    s[0] * s[1]
}

/// Each level's groups' W / A (see above), from the splats up, without
/// athenea's 0.99: what the group's opacity is before the cap.
pub fn coverage_ratios(file: &AthcFile) -> Vec<Vec<f32>> {
    let levels = file.levels.len();
    let mut weights: Vec<Vec<f64>> = file.levels.iter().map(|(_, b)| vec![0.0; b.n]).collect();
    if levels == 0 {
        return Vec::new();
    }
    // The finest level: its splats, run after run (chunks are Morton order).
    let count = file.header.count as usize;
    let mut g = 0usize;
    let mut at = 0usize;
    let finest = &mut weights[levels - 1];
    for chunk in &file.chunks {
        for i in 0..chunk.n {
            while g + 1 < file.starts.len() && file.starts[g + 1] as usize <= at {
                g += 1;
            }
            if g < finest.len() && at < count {
                let o = merged_share(chunk, i, chunk.positions[i * 4 + 3].max(0.0));
                let w = o * two_axis_area(&chunk.shape[i * 4..i * 4 + 4]);
                finest[g] += w as f64;
            }
            at += 1;
        }
    }
    // Coarser levels: a group's children are the next level's groups whose
    // cell code, shifted down three bits, is its own.
    for l in (0..levels - 1).rev() {
        let (parents, children) = (&file.levels[l].1.tail, &file.levels[l + 1].1.tail);
        let mut j = 0;
        for (i, &code) in parents.iter().enumerate() {
            let mut w = 0.0;
            while j < children.len() && children[j] >> 3 == code {
                w += weights[l + 1][j];
                j += 1;
            }
            weights[l][i] = w;
        }
    }
    file.levels
        .iter()
        .zip(&weights)
        .map(|((_, b), w)| {
            (0..b.n)
                .map(|i| {
                    let r = (w[i] as f32 / two_axis_area(&b.shape[i * 4..i * 4 + 4]).max(1e-30)).max(0.0);
                    if is_sheet(b, i) { r + SHEET_PAD } else { r }
                })
                .collect()
        })
        .collect()
}

/// Every level's opacities replaced by their uncapped ratio, and every
/// merged glass spread to its splats' area (`spread_translucent`).
pub fn uncap_levels(file: &mut AthcFile) {
    let ratios = coverage_ratios(file);
    let areas = translucent_areas(file);
    for (((_, block), r), a) in file.levels.iter_mut().zip(ratios).zip(areas) {
        for (i, o) in r.into_iter().enumerate() {
            block.positions[i * 4 + 3] = o;
            spread_translucent(block, i, a[i]);
        }
    }
}

// --- merged glass: its splats' area, not its moments' ----------------------
//
// splat_project draws a sheet at athenea's kSheetAlpha (0.1) whatever its
// own opacity, its colour scaled by alphaOwn / 0.1: each gaussian of a sheet
// takes a tenth of what is behind it (and of the sheet's own gaussians
// behind it) over its footprint. What a sheet lets through is so set by how
// much gaussian area covers it -- the sum of its splats' two-axis areas --
// not by their opacity. A merged sheet with its moments' area has a quarter
// of its splats' (the Corvette's windshield cut to a fifth by error: 0.64
// against 2.47 m2), so it let through about three times as much of what is
// behind it and occluded its own reflection less: thread BD measured the
// error cut's glass 1.2-1.6 times as bright as the full cloud, where a
// thinning that keeps the area (every fifth splat, its axes x sqrt 5) was
// 0.84-1.02. So a merged sheet's long axes grow (both by one factor, the
// thin one is left) until its area is its splats', its reflection divided
// by the area gained; a sheet already as large is left as it is.
//
// Solid glass (transmission past one half, not thin: the Corvette's tinted
// windows, 0.145 a splat) has the same trouble the other way round: its
// splats composite, 1 - prod(1 - a_i), and a merge that keeps their mass in
// one gaussian of the moments' area (0.55 on the tinted windows, a quarter
// of their area) covers more than they did and shows more of its own light.
// Spreading it to its splats' area (as a sheet) is wrong though: solid glass
// is stacked through its thickness (a lamp lens, a tinted window: the
// splats' areas sum to 3-9 times the moments'), so the grown gaussian
// spills past the glass, a lamp's lens light with it -- bright discs round
// every lit lens (thread BJ: night relMSE 85-249 against 2-8). Solid glass
// so keeps its moments' size and composites its splats over their overlap:
// n = their area over its, its opacity 1 - (1 - mean)^n with the mean their
// area-weighted opacity (the optical depth sum a_i (-ln(1 - o_i)) kept).

/// Whether element i is glass that merges by area (`spread_translucent`):
/// a thin sheet, or solid glass (transmission past one half).
pub fn is_translucent(block: &AthcBlock, i: usize) -> bool {
    if block.pbr.is_empty() || block.n == 0 {
        return false;
    }
    let words = block.pbr.len() / block.n;
    ((block.pbr[i * words] >> 16) & 255) as f32 / 255.0 > 0.5
}

/// Each level's groups' sum of their glass splats' two-axis areas (0 where
/// a group has none; `is_translucent`).
pub fn translucent_areas(file: &AthcFile) -> Vec<Vec<f64>> {
    let levels = file.levels.len();
    let mut areas: Vec<Vec<f64>> = file.levels.iter().map(|(_, b)| vec![0.0; b.n]).collect();
    if levels == 0 {
        return areas;
    }
    let count = file.header.count as usize;
    let mut g = 0usize;
    let mut at = 0usize;
    let finest = &mut areas[levels - 1];
    for chunk in &file.chunks {
        for i in 0..chunk.n {
            while g + 1 < file.starts.len() && file.starts[g + 1] as usize <= at {
                g += 1;
            }
            if g < finest.len() && at < count && is_translucent(chunk, i) {
                finest[g] += two_axis_area(&chunk.shape[i * 4..i * 4 + 4]) as f64;
            }
            at += 1;
        }
    }
    for l in (0..levels - 1).rev() {
        let (parents, children) = (&file.levels[l].1.tail, &file.levels[l + 1].1.tail);
        let mut j = 0;
        for (i, &code) in parents.iter().enumerate() {
            let mut a = 0.0;
            while j < children.len() && children[j] >> 3 == code {
                a += areas[l + 1][j];
                j += 1;
            }
            areas[l][i] = a;
        }
    }
    areas
}

/// Element i, merged glass whose splats' two-axis areas sum to `area`: a
/// sheet with its long axes grown to that area and its reflection (its
/// opacity less `SHEET_PAD`) divided by the area gained; solid glass with its
/// opacity composited over the splats' overlap (see above). Anything else is
/// left.
pub fn spread_translucent(block: &mut AthcBlock, i: usize, area: f64) {
    if area <= 0.0 || !is_translucent(block, i) {
        return;
    }
    let sheet = is_sheet(block, i);
    let w = &mut block.shape[i * 4..i * 4 + 4];
    let mut s = [low_half(w[1]).exp(), high_half(w[1]).exp(), low_half(w[2]).exp()];
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
    let before = (s[order[0]] * s[order[1]]) as f64;
    if before.is_nan() || before <= 0.0 || area <= before {
        return;
    }
    let o = &mut block.positions[i * 4 + 3];
    if !sheet {
        // A node already covered past 1 (Spark's LoD opacity, the ratio
        // `coverage_ratios` gave it) keeps it: relight.slang's
        // coveredGlassAlpha draws that coverage wider than a gaussian, which
        // is what closes the gaps between a level's cells on a closed solid
        // (the pawn's glass head, thread AU). Composited to under 1 it let
        // 5-45% more of the room through from 1-3 m (thread BL).
        if *o >= 1.0 {
            return;
        }
        let mean = (*o as f64 * before / area).clamp(0.0, 1.0);
        *o = (1.0 - (1.0 - mean).powf(area / before)) as f32;
        return;
    }
    *o = ((*o - SHEET_PAD).max(0.0) as f64 * before / area) as f32 + SHEET_PAD;
    let grow = (area / before).sqrt() as f32;
    for &k in &order[..2] {
        s[k] *= grow;
    }
    w[1] = pack_halves(ln_scale(s[0]), ln_scale(s[1]));
    w[2] = pack_halves(ln_scale(s[2]), high_half(w[2]));
}

/// How far a merged gaussian's two long axes are widened, as a fraction of
/// its cell's edge added in quadrature (`widen_merged`).
pub const MERGED_FILL: f32 = 0.35;

/// athenea's merged gaussian is its group's moments: on a flat, evenly
/// covered cell of edge L its long axes are L / sqrt(12), and a surface tiled
/// by such gaussians stays open where four cells meet, however opaque each
/// is (2.45 sigma from every centre). Spark's own merge (gsplat.rs
/// `new_merged`) widens by half its step; here the two long axes grow by
/// `MERGED_FILL` x L in quadrature, the thin one is left as it is (a shell
/// stays a shell), and the opacity is divided by the area gained so the
/// group keeps its weight -- except that a capped one (athenea's 0.99, a
/// file whose coverage is not known) stays at least 0.99. Measured on the
/// Corvette (paint and body, CPU raster of Spark's draw): the light let
/// through at 58 m goes from 5.2% to 0.6%.
pub fn widen_merged(block: &mut AthcBlock, edge: f32) {
    orient_merged(block);
    let grow = (MERGED_FILL * edge).powi(2);
    for i in 0..block.n {
        let sheet = is_sheet(block, i);
        let w = &mut block.shape[i * 4..i * 4 + 4];
        let mut s = [low_half(w[1]).exp(), high_half(w[1]).exp(), low_half(w[2]).exp()];
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
        let before = s[order[0]] * s[order[1]];
        for &k in &order[..2] {
            s[k] = (s[k] * s[k] + grow).sqrt();
        }
        let after = s[order[0]] * s[order[1]];
        w[1] = pack_halves(ln_scale(s[0]), ln_scale(s[1]));
        w[2] = pack_halves(ln_scale(s[2]), high_half(w[2]));
        let o = &mut block.positions[i * 4 + 3];
        if sheet {
            // The reflection keeps its weight; the 1/255 stays one.
            *o = (*o - SHEET_PAD).max(0.0) * before / after.max(1e-30) + SHEET_PAD;
            continue;
        }
        let widened = *o * before / after.max(1e-30);
        *o = if *o >= 0.98 { widened.max(0.99) } else { widened };
    }
}

/// A splat's third axis is its surface's normal, the way mesh2splat lays
/// it: athenea's relighting reads the curvature "on the mesh's normal" and
/// turns it to the face the eye sees by the sign of that axis against the
/// shading normal (splat_project: `faces`), and takes the shape operator in
/// the first two axes. A merged gaussian's axes are its moments'
/// eigenvectors, in no order and of either sign: on the pawn's glass head
/// the third axis was the thinnest for 29% of the merged nodes and pointed
/// along the stored normal for 50% -- half of every LoD level's cells read
/// the head as a convex lens and half as a concave one (a checker of dark
/// and bright cells, and no lens image). So a merged gaussian with a normal
/// gets its axes turned (the same gaussian) until its third axis is the one
/// nearest its stored normal, pointing the same way; without normals, the
/// thinnest axis third.
pub fn orient_merged(block: &mut AthcBlock) {
    let normals = !block.normals.is_empty();
    for i in 0..block.n {
        let w = &mut block.shape[i * 4..i * 4 + 4];
        let q = decode_quaternion(w[0]);
        let m = Mat3::from_quat(Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize());
        let mut axes = [m.x_axis, m.y_axis, m.z_axis];
        let mut ln = [low_half(w[1]), high_half(w[1]), low_half(w[2])];
        let normal = if normals { Some(glam::Vec3::from_array(unpack_normal(block.normals[i]))) } else { None };
        // The axis to put third.
        let third = match normal {
            Some(n) => (0..3).max_by(|&a, &b| axes[a].dot(n).abs().total_cmp(&axes[b].dot(n).abs())).unwrap(),
            None => (0..3).min_by(|&a, &b| ln[a].total_cmp(&ln[b])).unwrap(),
        };
        // A cyclic turn keeps the frame right-handed.
        let turn = (third + 1) % 3;
        let mut changed = third != 2;
        if changed {
            axes = [axes[turn], axes[(turn + 1) % 3], axes[(turn + 2) % 3]];
            ln = [ln[turn], ln[(turn + 1) % 3], ln[(turn + 2) % 3]];
        }
        // Against the normal: the third and the first axes turned over (a
        // half turn about the second), the same gaussian.
        if let Some(n) = normal {
            if axes[2].dot(n) < 0.0 {
                axes[2] = -axes[2];
                axes[0] = -axes[0];
                changed = true;
            }
        }
        if !changed {
            continue;
        }
        let q = Quat::from_mat3(&Mat3::from_cols(axes[0], axes[1], axes[2])).normalize();
        w[0] = encode_quaternion(q.to_array());
        w[1] = pack_halves(ln[0], ln[1]);
        w[2] = pack_halves(ln[2], high_half(w[2]));
    }
}

/// Spark's LoD opacity for an opacity `o` past 1 (tsplat.rs
/// `encode_lod_opacity`): D = sqrt(1 + e ln o), at most 5, stored as
/// 1 + (D - 1) / 4, which the draw reads as a gaussian composited `o` times.
/// At most 1 it is `o` itself.
pub fn spark_lod_opacity(o: f32) -> f32 {
    if o.is_nan() || o <= 1.0 {
        return o;
    }
    let d = (1.0 + std::f32::consts::E * o.ln()).sqrt().min(5.0);
    1.0 + 0.25 * (d - 1.0)
}

/// The cloud as its levels from the coarsest down to level index `keep`
/// (0 the coarsest), whose groups become the splats: the full cloud less its
/// finest detail, each kept group merged as athenea merges (its attributes
/// too) and with its whole coverage (`uncap_levels`).
pub fn truncate_levels(file: &AthcFile, keep: usize) -> Result<AthcFile> {
    if keep == 0 || keep >= file.levels.len() {
        bail!("keep levels 1 .. {} as merged levels (asked {})", file.levels.len() - 1, keep);
    }
    let mut full = file.clone();
    uncap_levels(&mut full);
    let mut splats = full.levels[keep].1.clone();
    // The kept groups are drawn as splats, never through `merged_block`:
    // widened here, once.
    widen_merged(&mut splats, full.header.extent / (1u64 << full.levels[keep].0) as f32);
    let parents = &full.levels[keep - 1].1.tail;
    let mut starts = Vec::with_capacity(parents.len());
    let mut j = 0;
    for &code in parents {
        starts.push(j as u32);
        while j < splats.n && splats.tail[j] >> 3 == code {
            splats.tail[j] = starts.len() as u32 - 1;
            j += 1;
        }
    }
    if j != splats.n {
        bail!("level {} has groups without a parent", keep);
    }
    let per = file.header.chunk_splats.max(1) as usize;
    let chunks: Vec<AthcBlock> = (0..splats.n).step_by(per).map(|s| splats.slice(s, per.min(splats.n - s))).collect();
    let mut out = AthcFile {
        header: full.header,
        extra: full.extra,
        levels: full.levels[..keep].to_vec(),
        starts,
        chunks,
    };
    out.header.count = splats.n as u32;
    out.header.chunks = out.chunks.len() as u32;
    out.header.levels = keep as u32;
    out.header.finest_groups = out.levels[keep - 1].1.n as u32;
    Ok(out)
}

/// Each level's groups' normal moments from the splats up: the weight W
/// (opacity x area of the two long axes, as athenea's moments weigh) and
/// the weighted sum of the unit normals, `[W, Sx, Sy, Sz]` (f64). Empty
/// for a cloud without normals.
pub fn normal_moments(file: &AthcFile) -> Vec<Vec<[f64; 4]>> {
    let levels = file.levels.len();
    if levels == 0 || file.chunks.iter().all(|c| c.normals.is_empty()) {
        return Vec::new();
    }
    let mut m: Vec<Vec<[f64; 4]>> = file.levels.iter().map(|(_, b)| vec![[0.0; 4]; b.n]).collect();
    let mut g = 0usize;
    let mut at = 0usize;
    for chunk in &file.chunks {
        for i in 0..chunk.n {
            while g + 1 < file.starts.len() && file.starts[g + 1] as usize <= at {
                g += 1;
            }
            if g < m[levels - 1].len() && !chunk.normals.is_empty() {
                let w = (chunk.positions[i * 4 + 3].max(0.0) * two_axis_area(&chunk.shape[i * 4..i * 4 + 4])) as f64;
                let n = unpack_normal(chunk.normals[i]);
                let e = &mut m[levels - 1][g];
                e[0] += w;
                for k in 0..3 {
                    e[k + 1] += w * n[k] as f64;
                }
            }
            at += 1;
        }
    }
    for l in (0..levels - 1).rev() {
        let (parents, children) = (&file.levels[l].1.tail, &file.levels[l + 1].1.tail);
        let mut j = 0;
        for (i, &code) in parents.iter().enumerate() {
            let mut e = [0.0f64; 4];
            while j < children.len() && children[j] >> 3 == code {
                for k in 0..4 {
                    e[k] += m[l + 1][j][k];
                }
                j += 1;
            }
            m[l][i] = e;
        }
    }
    m
}

/// A group's normal spread from its moments: 1 - |mean normal|, 0 for a
/// flat group, 1 - cos(phi / 2) for two equal faces phi apart (a 45 degree
/// crease: 0.076), 1 for two opposite faces.
pub fn normal_spread(m: &[f64; 4]) -> f32 {
    if m[0] <= 0.0 {
        return 0.0;
    }
    let len = (m[1] * m[1] + m[2] * m[2] + m[3] * m[3]).sqrt() / m[0];
    (1.0 - len).clamp(0.0, 1.0) as f32
}

/// The variance of a group's normals, 1 - |mean normal|^2 (the trace of
/// their covariance; LEAN's second moment less the mean's square): what a
/// merged gaussian's lobes are widened by (relight.slang
/// footprintRoughness), kept in the curvature's fourth half.
pub fn normal_variance(m: &[f64; 4]) -> f32 {
    if m[0] <= 0.0 {
        return 0.0;
    }
    let len2 = (m[1] * m[1] + m[2] * m[2] + m[3] * m[3]) / (m[0] * m[0]);
    (1.0 - len2).clamp(0.0, 1.0) as f32
}

/// Writes each merged level's normal variance (`normal_variance`) into the
/// fourth half of its curvature (0 in every file written before; the
/// splats keep 0: their own spread is the curvature's). A cloud without
/// curvature or normals is left as it is.
pub fn store_normal_variance(file: &mut AthcFile) {
    let m = normal_moments(file);
    if m.is_empty() {
        return;
    }
    for ((_, b), m) in file.levels.iter_mut().zip(&m) {
        if b.curvature.len() != 2 * b.n {
            continue;
        }
        for i in 0..b.n {
            let w = &mut b.curvature[2 * i + 1];
            *w = pack_halves(low_half(*w), normal_variance(&m[i]));
        }
    }
}

/// What [`truncate_creases`] made of a cloud.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct CreaseCut {
    /// Elements of the cut from each level kept as merged (index into the
    /// file's levels, from `keep`), then real splats.
    pub per_level: Vec<usize>,
    pub splats: usize,
    /// The original splats `[lo, hi)` each element of the cut stands for.
    #[serde(skip)]
    pub sources: Vec<[u32; 2]>,
}

/// The splats (file order, `[lo, hi)`) under group `g` of level index `l`
/// (into `file.levels`): the run of its finest groups.
pub fn group_splats(file: &AthcFile, l: usize, g: usize) -> [u32; 2] {
    let (finest_level, finest) = file.levels.last().map(|(r, b)| (*r, &b.tail)).expect("levels");
    let (level, code) = (file.levels[l].0, file.levels[l].1.tail[g]);
    let shift = 3 * (finest_level - level);
    let lo = finest.partition_point(|&c| (c >> shift) < code);
    let hi = finest.partition_point(|&c| (c >> shift) <= code);
    let n = file.header.count;
    let start = |k: usize| file.starts.get(k).copied().unwrap_or(n);
    [start(lo), start(hi)]
}

/// [`truncate_levels`]' elements as the original splats each stands for.
pub fn cut_sources(file: &AthcFile, keep: usize) -> Vec<[u32; 2]> {
    (0..file.levels[keep].1.n).map(|g| group_splats(file, keep, g)).collect()
}

/// [`truncate_levels`], aware of creases: a group of level `keep` whose
/// normals spread more than `max_spread` ([`normal_spread`]) is replaced by
/// its children, recursively, down to the splats themselves, so that a
/// crease keeps finer (in the end real) splats instead of one merged
/// gaussian straddling it with the faces' mean normal (the merged cell's
/// centre off both faces, its normal neither's: the light Corvette's door
/// sill read as a sawtooth along its crease). Each element keeps its own
/// level's widening; the merged ones carry their normal variance in the
/// curvature's fourth half (`store_normal_variance`). `max_spread` >= 1
/// is `truncate_levels` (plus the variance). `max_depth` bounds how many
/// levels below `keep` a crease may go (the splats count as one more than
/// the finest level).
pub fn truncate_creases(file: &AthcFile, keep: usize, max_spread: f32, max_depth: usize) -> Result<(AthcFile, CreaseCut)> {
    if keep == 0 || keep >= file.levels.len() {
        bail!("keep levels 1 .. {} as merged levels (asked {})", file.levels.len() - 1, keep);
    }
    let mut full = file.clone();
    uncap_levels(&mut full);
    store_normal_variance(&mut full);
    let moments = normal_moments(&full);
    let spread = |l: usize, g: usize| moments.get(l).map_or(0.0, |m| normal_spread(&m[g]));
    let levels = full.levels.len();
    let finest_level = full.levels[levels - 1].0;
    let all = full.splats();
    let count = all.n;
    // Each level's children ranges in the next (codes are sorted).
    let child_range = |l: usize, g: usize| -> (usize, usize) {
        let code = full.levels[l].1.tail[g];
        let next = &full.levels[l + 1].1.tail;
        let lo = next.partition_point(|&c| (c >> 3) < code);
        let hi = next.partition_point(|&c| (c >> 3) <= code);
        (lo, hi)
    };
    // The cut, in Morton order: (level index or usize::MAX for a splat, index).
    let mut cut: Vec<(usize, usize)> = Vec::new();
    let mut stack: Vec<(usize, usize)> = (0..full.levels[keep].1.n).rev().map(|g| (keep, g)).collect();
    while let Some((l, g)) = stack.pop() {
        if spread(l, g) <= max_spread || l - keep >= max_depth {
            cut.push((l, g));
        } else if l + 1 < levels {
            let (lo, hi) = child_range(l, g);
            stack.extend((lo..hi).rev().map(|c| (l + 1, c)));
        } else {
            let lo = full.starts[g] as usize;
            let hi = full.starts.get(g + 1).map_or(count, |&s| s as usize);
            cut.extend((lo..hi).map(|s| (usize::MAX, s)));
        }
    }
    // Each kept level widened once, as truncate_levels widens its one.
    let widened: Vec<AthcBlock> = (keep..levels)
        .map(|l| {
            let mut b = full.levels[l].1.clone();
            widen_merged(&mut b, full.header.extent / (1u64 << full.levels[l].0) as f32);
            b
        })
        .collect();
    let mut stats = CreaseCut { per_level: vec![0; levels - keep], splats: 0, sources: Vec::with_capacity(cut.len()) };
    let mut splats = AthcBlock::default();
    let mut codes = Vec::with_capacity(cut.len());
    let parent_level = full.levels[keep - 1].0;
    // Runs of the same source are appended together.
    let mut run: Option<(usize, usize, usize)> = None;
    let flush = |run: Option<(usize, usize, usize)>, splats: &mut AthcBlock| {
        if let Some((l, lo, hi)) = run {
            let mut part = if l == usize::MAX { all.slice(lo, hi - lo) } else { widened[l - keep].slice(lo, hi - lo) };
            part.tail.clear();
            if l == usize::MAX && part.curvature.is_empty() && !widened[0].curvature.is_empty() {
                part.curvature = vec![0; 2 * part.n];
            }
            splats.append(&part);
        }
    };
    for &(l, i) in &cut {
        let (code, level) = if l == usize::MAX {
            (full.levels[levels - 1].1.tail[all.tail[i] as usize], finest_level)
        } else {
            (full.levels[l].1.tail[i], full.levels[l].0)
        };
        codes.push(code >> (3 * (level - parent_level)));
        if l == usize::MAX {
            stats.splats += 1;
            stats.sources.push([i as u32, i as u32 + 1]);
        } else {
            stats.per_level[l - keep] += 1;
            stats.sources.push(group_splats(&full, l, i));
        }
        run = match run {
            Some((rl, lo, hi)) if rl == l && hi == i => Some((rl, lo, hi + 1)),
            other => {
                flush(other, &mut splats);
                Some((l, i, i + 1))
            }
        };
    }
    flush(run, &mut splats);
    // Tails: the index of the parent group at level keep - 1.
    let parents = &full.levels[keep - 1].1.tail;
    let mut starts = Vec::with_capacity(parents.len());
    let mut j = 0;
    splats.tail = vec![0; splats.n];
    for &code in parents {
        starts.push(j as u32);
        while j < splats.n && codes[j] == code {
            splats.tail[j] = starts.len() as u32 - 1;
            j += 1;
        }
    }
    if j != splats.n {
        bail!("the cut has elements without a parent at level {}", keep - 1);
    }
    let per = file.header.chunk_splats.max(1) as usize;
    let chunks: Vec<AthcBlock> = (0..splats.n).step_by(per).map(|s| splats.slice(s, per.min(splats.n - s))).collect();
    let mut out = AthcFile { header: full.header, extra: full.extra, levels: full.levels[..keep].to_vec(), starts, chunks };
    out.header.count = splats.n as u32;
    out.header.chunks = out.chunks.len() as u32;
    out.header.levels = keep as u32;
    out.header.finest_groups = out.levels[keep - 1].1.n as u32;
    Ok((out, stats))
}

// --- ATHV: one page of a virtual tree --------------------------------------

/// Bytes before an ATHV page's own data.
pub const ATHV_HEAD: usize = 160;
pub const ATHV_MERGED: u32 = 0;
pub const ATHV_SPLATS: u32 = 1;

/// One page of the virtual tree, self-contained for a decoder:
///
/// ```text
/// 0    "ATHV"
/// 4    kind: 0 merged nodes, 1 splats (an athenea chunk as it is in the file)
/// 8    base: the virtual index of its first element
/// 12   n: its elements
/// 16   the .athc's FileHeader and ExtraHeader (136 bytes, as in the file)
/// 152  decode flags (ATHV_KEEP_LINEAR), 0
/// 160  kind 0: child_count u32[n], child_start u32[n] (virtual), the
///      block, then group ranges u32[2n] (VirtualTree::group_range);
///      kind 1: the block (the file's chunk bytes)
/// ```
pub fn athv_head(kind: u32, base: u32, n: u32, header_page: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ATHV_HEAD);
    out.extend_from_slice(b"ATHV");
    out.extend_from_slice(&kind.to_le_bytes());
    out.extend_from_slice(&base.to_le_bytes());
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&header_page[..HEADER_BYTES + EXTRA_HEADER_BYTES]);
    out.resize(ATHV_HEAD, 0);
    out
}

/// The merged pages of a file's virtual tree (page-aligned splats), from
/// the file's bytes through `AthcLayout::levels_end`.
pub fn athv_merged_pages(prefix: &[u8], file_bytes: u64) -> Result<(VirtualTree, Vec<Vec<u8>>)> {
    let layout = AthcLayout::parse(prefix, file_bytes)?;
    if (prefix.len() as u64) < layout.levels_end() {
        bail!("need the first {} bytes of the .athc for its levels", layout.levels_end());
    }
    let h = layout.header;
    let x = layout.extra;
    let levels = layout
        .levels
        .iter()
        .map(|e| Ok((e.level, AthcBlock::read(&prefix[e.offset as usize..], e.groups as usize, &h, &x)?)))
        .collect::<Result<Vec<_>>>()?;
    let mut at = h.starts as usize;
    let starts = words_of(prefix, &mut at, h.finest_groups as usize);
    let file = AthcFile { header: h, extra: x, levels, starts, chunks: Vec::new() };
    merged_pages_of(&file, prefix)
}

/// The paged virtual tree of a cloud's levels (`file.chunks` unused) and
/// its merged pages, their heads carrying `header_page`'s first 136 bytes.
pub fn merged_pages_of(file: &AthcFile, header_page: &[u8]) -> Result<(VirtualTree, Vec<Vec<u8>>)> {
    let prefix = header_page;
    let tree = VirtualTree::of_file(file, true)?;
    let merged = merged_block(file, &tree);
    let mut pages = Vec::new();
    for p in 0..tree.merged_pages() {
        let base = p * PAGE_SPLATS;
        let n = (tree.merged - base).min(PAGE_SPLATS);
        let mut out = athv_head(ATHV_MERGED, base, n, prefix);
        for k in base..base + n {
            out.extend_from_slice(&(tree.child_count[k as usize] as u32).to_le_bytes());
        }
        for k in base..base + n {
            out.extend_from_slice(&tree.child_start[k as usize].to_le_bytes());
        }
        merged.slice(base as usize, n as usize).write(&mut out);
        for w in &tree.group_range[2 * base as usize..2 * (base + n) as usize] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        if !merged.lod_size.is_empty() {
            out[152..156].copy_from_slice(&ATHV_LOD_SIZES.to_le_bytes());
            for v in &merged.lod_size[base as usize..(base + n) as usize] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        pages.push(out);
    }
    Ok((tree, pages))
}

// --- decoding into Spark ----------------------------------------------------

/// The attributes a block carries, in the order `set_attrib` numbers them.
pub fn attrib_specs(h: &AthcHeader, x: &ExtraHeader) -> Vec<AttribSpec> {
    let spec = |name: &str, format: &str, components: u32, lod_merge| AttribSpec {
        name: name.to_string(),
        format: format.to_string(),
        components: components as usize,
        lod_merge,
    };
    let mut out = Vec::new();
    if h.has(FLAG_NORMALS) {
        out.push(spec("normalOct", "u32", 1, LodMerge::First));
    }
    if h.has(FLAG_EMISSION) {
        out.push(spec("emission", "u32", 1, LodMerge::First));
    }
    if x.pbr_words > 0 {
        out.push(spec("pbr", "u32", 1, LodMerge::First));
    }
    if x.lobes_words > 0 {
        out.push(spec("lobes", "u32", x.lobes_words, LodMerge::First));
    }
    if x.transfer_words > 0 {
        out.push(spec("transfer", "f16", x.transfer_count, LodMerge::WeightedMean));
    }
    if x.shadow_words > 0 {
        out.push(spec("shadowBits", "u32", x.shadow_words, LodMerge::First));
    }
    if x.curvature_words > 0 {
        out.push(spec(CURVATURE_ATTRIBUTE, "f16", 4, LodMerge::WeightedMean));
    }
    // The skin as stored (athc_skin): merged nodes carry their own.
    if x.skin_influences > 0 {
        out.push(spec(crate::athc_skin::SKIN_INFLUENCES_ATTRIBUTE, "u32", x.skin_influences, LodMerge::First));
    }
    if x.skin_gradient_words > 0 {
        out.push(spec(crate::athc_skin::SKIN_GRADIENTS_ATTRIBUTE, "u32", x.skin_gradient_words, LodMerge::First));
    }
    // Every .athc: which of athenea's finest LoD groups a splat is in.
    out.push(spec(GROUP_ATTRIBUTE, "u32", 2, LodMerge::First));
    out
}

/// The attribute every decoded .athc carries: `[lo, hi)`, the finest LoD
/// groups an element covers -- its own group for a splat, a node's subtree
/// for athenea's merged levels (`VirtualTree::group_range`). A .athc keeps
/// no Cryptomatte id (athenea's `cloudCrypto` lives in USD, not the file),
/// so these are the ids it can be picked and overridden by: spatial cells
/// of the Morton octree rather than prims. Any level's group is a range.
pub const GROUP_ATTRIBUTE: &str = "athcGroup";

/// The curvature stream's attribute: athenea's per-splat shape operator in
/// the splat's first two axes (xx, xy, yy), 1/metres, on the mesh's normal,
/// and a merged splat's normal variance (`store_normal_variance`; 0 for a splat).
pub const CURVATURE_ATTRIBUTE: &str = "curvature";

#[derive(Clone, Copy, Debug, Default)]
pub struct DecodeOptions {
    /// Keep a linear cloud's colours linear. By default they are encoded to
    /// sRGB (the base exactly, the rest harmonics through the curve's slope
    /// at the base), which is what Spark's blend expects.
    pub keep_linear: bool,
}

/// An ATHV head's decode flags (word 152): bit 0, keep a linear cloud's
/// colours linear (`DecodeOptions::keep_linear`).
pub const ATHV_KEEP_LINEAR: u32 = 1;
/// An ATHV merged page's decode flags, bit 1: its nodes' LoD sizes
/// (`AthcBlock::lod_size`, f32 each) follow its group ranges.
pub const ATHV_LOD_SIZES: u32 = 2;

/// Feeds `block`'s elements to `receiver` as splats base .. base + n, with
/// `children` (counts, virtual starts) when they are tree nodes.
pub fn emit_block<T: SplatReceiver>(
    receiver: &mut T,
    base: usize,
    block: &AthcBlock,
    h: &AthcHeader,
    x: &ExtraHeader,
    children: Option<(&[u16], &[u32])>,
    groups: Option<&[u32]>,
    lod_tree: bool,
    options: DecodeOptions,
) {
    let degree = h.sh_degree();
    let srgb = h.has(FLAG_LINEAR) && !options.keep_linear;
    let sh_words = h.sh_words as usize;
    let mut done = 0;
    while done < block.n {
        let count = (block.n - done).min(PAGE_SPLATS as usize);
        let mut center = vec![0.0; count * 3];
        let mut opacity = vec![0.0; count];
        let mut rgb = vec![0.0; count * 3];
        let mut scale = vec![0.0; count * 3];
        let mut quat = vec![0.0; count * 4];
        let sizes = [0, 9, 24, 45];
        let mut rest = vec![0.0f32; count * sizes[degree]];
        for i in 0..count {
            let e = done + i;
            let p = &block.positions[e * 4..e * 4 + 4];
            center[i * 3..i * 3 + 3].copy_from_slice(&p[..3]);
            // Past 1 (merged coverage, `coverage_ratios`): Spark's LoD opacity.
            opacity[i] = spark_lod_opacity(p[3]);
            let s = &block.shape[e * 4..e * 4 + 4];
            quat[i * 4..i * 4 + 4].copy_from_slice(&decode_quaternion(s[0]));
            scale[i * 3] = low_half(s[1]).exp();
            scale[i * 3 + 1] = high_half(s[1]).exp();
            scale[i * 3 + 2] = low_half(s[2]).exp();
            let mut colour = [high_half(s[2]), low_half(s[3]), high_half(s[3])];
            let mut slope = [1.0; 3];
            if srgb {
                for c in 0..3 {
                    slope[c] = linear_to_srgb_slope(colour[c]);
                    colour[c] = linear_to_srgb(colour[c]);
                }
            }
            rgb[i * 3..i * 3 + 3].copy_from_slice(&colour);
            let n = sizes[degree];
            for k in 0..n {
                let word = block.sh[e * sh_words + k / 2];
                let v = if k % 2 == 0 { low_half(word) } else { high_half(word) };
                rest[i * n + k] = v * slope[k % 3];
            }
        }
        let n = sizes[degree];
        let split = |from: usize, per: usize| -> Vec<f32> {
            (0..count).flat_map(|i| rest[i * n + from..i * n + from + per].to_vec()).collect()
        };
        let sh1 = if degree >= 1 { split(0, 9) } else { Vec::new() };
        let sh2 = if degree >= 2 { split(9, 15) } else { Vec::new() };
        let sh3 = if degree >= 3 { split(24, 21) } else { Vec::new() };
        let (child_count, child_start): (Vec<u16>, Vec<usize>) = match children {
            Some((counts, starts)) => (
                counts[done..done + count].to_vec(),
                starts[done..done + count].iter().map(|&s| s as usize).collect(),
            ),
            None if lod_tree => (vec![0; count], vec![0; count]),
            None => (Vec::new(), Vec::new()),
        };
        receiver.set_batch(base + done, count, &SplatProps {
            center: &center,
            opacity: &opacity,
            rgb: &rgb,
            scale: &scale,
            quat: &quat,
            sh1: &sh1,
            sh2: &sh2,
            sh3: &sh3,
            child_count: &child_count,
            child_start: &child_start,
        });
        done += count;
    }
    // The extras, as attributes: words as they are, the transfer and the
    // curvature as halves (set_attrib_words: a WASM receiver keeps them packed).
    let mut k = 0;
    for (on, values, per) in [
        (h.has(FLAG_NORMALS), &block.normals, 1),
        (h.has(FLAG_EMISSION), &block.emission, 1),
        (x.pbr_words > 0, &block.pbr, 1),
        (x.lobes_words > 0, &block.lobes, x.lobes_words as usize),
    ] {
        if on {
            receiver.set_attrib_words(k, base, block.n, values, per, false);
            k += 1;
        }
    }
    if x.transfer_words > 0 {
        receiver.set_attrib_words(k, base, block.n, &block.transfer, x.transfer_count as usize, true);
        k += 1;
    }
    if x.shadow_words > 0 {
        receiver.set_attrib_words(k, base, block.n, &block.shadow_bits, x.shadow_words as usize, false);
        k += 1;
    }
    if x.curvature_words > 0 {
        receiver.set_attrib_words(k, base, block.n, &block.curvature, 4, true);
        k += 1;
    }
    if x.skin_influences > 0 {
        let (i, g) = (x.skin_influences as usize, x.skin_gradient_words as usize);
        // A block without the stream (a page that left SKIN out) reads as
        // carried by nothing: zero weights, which the skinner leaves at rest.
        let whole = block.skin.len() == block.n * (i + g);
        let column = |from: usize, per: usize| -> Vec<u32> {
            if !whole {
                return vec![0; block.n * per];
            }
            (0..block.n).flat_map(|e| block.skin[e * (i + g) + from..e * (i + g) + from + per].to_vec()).collect()
        };
        receiver.set_attrib_words(k, base, block.n, &column(0, i), i, false);
        k += 1;
        if g > 0 {
            receiver.set_attrib_words(k, base, block.n, &column(i, g), g, false);
            k += 1;
        }
    }
    // LoD sizes by error, for the nodes that have them (0: geometric).
    if lod_tree && block.lod_size.len() == block.n && block.lod_size.iter().any(|&v| v > 0.0) {
        receiver.set_lod_size(base, block.n, &block.lod_size);
    }
    // GROUP_ATTRIBUTE: given for merged nodes; a splat's tail is its group.
    let ranges: Vec<u32> = match groups {
        Some(g) => g.to_vec(),
        None => block.tail.iter().flat_map(|&g| [g, g + 1]).collect(),
    };
    receiver.set_attrib_words(k, base, block.n, &ranges, 2, false);
}

fn begin<T: SplatReceiver>(receiver: &mut T, num_splats: usize, h: &AthcHeader, x: &ExtraHeader) -> Result<()> {
    // A LoD tree: receivers mark the encoding lodOpacity, which is what makes
    // the loader treat the result as LoD splats, and keeps opacities up to 2:
    // merged coverage past 1 (`spark_lod_opacity`).
    receiver.init_splats(&SplatInit { num_splats, max_sh_degree: h.sh_degree(), lod_tree: true })?;
    let specs = attrib_specs(h, x);
    if !specs.is_empty() {
        receiver.init_attribs(&specs);
    }
    Ok(())
}

/// Decodes a whole .athc, or one ATHV page, into a Spark receiver: the
/// virtual tree's merged nodes and the splats (a whole file: no gap), with
/// child counts and starts, and the extras as attributes.
pub struct AthcDecoder<T: SplatReceiver> {
    splats: T,
    buffer: Vec<u8>,
    /// The whole file's size was read from its tables and reserved.
    reserved: bool,
    pub options: DecodeOptions,
    /// Set after finish(): the tree a whole file was decoded with.
    pub tree: Option<VirtualTree>,
}

impl<T: SplatReceiver> AthcDecoder<T> {
    pub fn new(splats: T) -> Self {
        Self { splats, buffer: Vec::new(), reserved: false, options: DecodeOptions::default(), tree: None }
    }

    pub fn into_splats(self) -> T {
        self.splats
    }

    fn finish_file(&mut self) -> Result<()> {
        // The file's bytes go as soon as they are read, and each chunk once
        // it is handed over: a whole file with a full transfer (the
        // Corvette's paint, 1.9M splats, 112 halves each) is most of a
        // wasm32 heap, read and decoded side by side.
        let bytes = std::mem::take(&mut self.buffer);
        let mut file = if u32_at(&bytes, 0) == crate::athc_v3::ATH3_MAGIC {
            crate::athc_v3::read_v3(&bytes)?
        } else {
            AthcFile::read(&bytes)?
        };
        drop(bytes);
        let (h, x) = (file.header, file.extra);
        // The whole file is here, splats too: the merged levels get their
        // whole coverage, whatever a v2 file capped (`coverage_ratios`).
        uncap_levels(&mut file);
        let tree = VirtualTree::of_file(&file, false)?;
        let merged = merged_block(&file, &tree);
        begin(&mut self.splats, (tree.merged + h.count) as usize, &h, &x)?;
        emit_block(
            &mut self.splats,
            0,
            &merged,
            &h,
            &x,
            Some((&tree.child_count, &tree.child_start)),
            Some(&tree.group_range),
            true,
            self.options,
        );
        drop(merged);
        file.levels = Vec::new();
        let mut at = tree.splat_base as usize;
        for chunk in std::mem::take(&mut file.chunks) {
            emit_block(&mut self.splats, at, &chunk, &h, &x, None, None, true, self.options);
            at += chunk.n;
        }
        self.tree = Some(tree);
        self.splats.finish()
    }

    fn finish_page(&mut self) -> Result<()> {
        let b = &self.buffer;
        if b.len() < ATHV_HEAD {
            bail!("ATHV page shorter than its header");
        }
        let kind = u32_at(b, 4);
        if kind == crate::athc_v3::ATHV_SECTIONS {
            // A v3 block's sections (athc_v3::athv_sections_page).
            let (h, x, block, flags) = crate::athc_v3::read_sections_page(b)?;
            let mut options = self.options;
            options.keep_linear |= flags & ATHV_KEEP_LINEAR != 0;
            begin(&mut self.splats, block.n, &h, &x)?;
            emit_block(&mut self.splats, 0, &block, &h, &x, None, None, true, options);
            return self.splats.finish();
        }
        let base = u32_at(b, 8);
        let n = u32_at(b, 12) as usize;
        let mut page = vec![0u8; HEADER_BYTES + EXTRA_HEADER_BYTES];
        page.copy_from_slice(&b[16..16 + HEADER_BYTES + EXTRA_HEADER_BYTES]);
        let (h, x) = parse_headers(&page)?;
        let _ = base; // a page's elements are its own 0..n; the tree's starts are virtual
        let mut at = ATHV_HEAD;
        let children = if kind == ATHV_MERGED {
            if b.len() < at + 8 * n {
                bail!("ATHV page shorter than its tree");
            }
            let counts: Vec<u16> = words_of(b, &mut at, n).into_iter().map(|c| c as u16).collect();
            let starts = words_of(b, &mut at, n);
            Some((counts, starts))
        } else if kind == ATHV_SPLATS {
            None
        } else {
            bail!("ATHV page of kind {}", kind);
        };
        let block = AthcBlock::read(&b[at..], n, &h, &x)?;
        // Merged pages carry their nodes' group ranges after the block.
        let mut block = block;
        let groups = if kind == ATHV_MERGED {
            let mut after = at + n * element_bytes(&h, &x) as usize;
            if b.len() < after + 8 * n {
                bail!("ATHV page shorter than its group ranges");
            }
            let ranges = words_of(b, &mut after, 2 * n);
            // ... and their LoD sizes, when the page says so.
            if u32_at(b, 152) & ATHV_LOD_SIZES != 0 {
                if b.len() < after + 4 * n {
                    bail!("ATHV page shorter than its LoD sizes");
                }
                block.lod_size = words_of(b, &mut after, n).into_iter().map(f32::from_bits).collect();
            }
            Some(ranges)
        } else {
            None
        };
        let mut options = self.options;
        options.keep_linear |= u32_at(b, 152) & ATHV_KEEP_LINEAR != 0;
        begin(&mut self.splats, n, &h, &x)?;
        let children = children.as_ref().map(|(c, s)| (&c[..], &s[..]));
        emit_block(&mut self.splats, 0, &block, &h, &x, children, groups.as_deref(), true, options);
        self.splats.finish()
    }
}

impl<T: SplatReceiver> ChunkReceiver for AthcDecoder<T> {
    fn push(&mut self, bytes: &[u8]) -> Result<()> {
        self.buffer.extend_from_slice(bytes);
        // A v3 file says its size in its tables: reserve it once, so the
        // buffer holds the file and not up to twice it (Vec doubling), and
        // is not copied as it grows. Its sections are decoded next to it.
        if !self.reserved && self.buffer.len() >= 144 && u32_at(&self.buffer, 0) == crate::athc_v3::ATH3_MAGIC {
            match crate::athc_v3::tables_bytes(&self.buffer) {
                Ok(tables) if self.buffer.len() as u64 >= tables => {
                    self.reserved = true;
                    if let Ok(end) = crate::athc_v3::file_bytes(&self.buffer) {
                        // Files end padded to a whole page.
                        let want = aligned(end) as usize;
                        if want > self.buffer.len() && want < (1usize << 31) {
                            let mut whole = Vec::with_capacity(want);
                            whole.extend_from_slice(&self.buffer);
                            self.buffer = whole;
                        }
                    }
                }
                Ok(_) => {}
                Err(_) => self.reserved = true,
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.buffer.len() < 4 {
            bail!(".athc shorter than its magic");
        }
        let result = match u32_at(&self.buffer, 0) {
            ATHC_MAGIC | crate::athc_v3::ATH3_MAGIC => self.finish_file(),
            ATHV_MAGIC => self.finish_page(),
            _ => Err(anyhow!("neither an .athc (v1, v2, v3) nor an ATHV page")),
        };
        self.buffer = Vec::new();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::SplatInit;

    #[test]
    fn merged_solid_glass_keeps_a_coverage_past_one() {
        // One solid glass node (transmission 1, not thin), 1 cm x 1 cm x 1 mm,
        // whose splats' areas sum to four times its own.
        let mut b = AthcBlock { n: 1, positions: vec![0.0, 0.0, 0.0, 1.5], shape: vec![0, 0, 0, 0], pbr: vec![255 << 16], ..Default::default() };
        b.shape[1] = pack_halves(ln_scale(0.01), ln_scale(0.01));
        b.shape[2] = pack_halves(ln_scale(0.001), 0.0);
        let area = 4.0 * two_axis_area(&b.shape) as f64;
        let shape = b.shape.clone();
        // A coverage past 1 (Spark's LoD opacity, relight's coveredGlassAlpha) is kept.
        spread_translucent(&mut b, 0, area);
        assert_eq!(b.positions[3], 1.5);
        // Under 1 its splats composite over their overlap: less than their sum, its size kept.
        b.positions[3] = 0.4;
        spread_translucent(&mut b, 0, area);
        let o = b.positions[3];
        assert!(o < 0.4 && o > 0.3, "{o}");
        assert_eq!(b.shape, shape);
    }

    const TWO_CARDS: &[u8] = include_bytes!("../../../test/fixtures/athc/two_cards.athc");

    /// Everything a decoder hands a receiver.
    #[derive(Default)]
    struct Collect {
        init: Option<SplatInit>,
        center: Vec<f32>,
        opacity: Vec<f32>,
        rgb: Vec<f32>,
        scale: Vec<f32>,
        quat: Vec<f32>,
        sh1: Vec<f32>,
        child_count: Vec<u16>,
        child_start: Vec<usize>,
        specs: Vec<AttribSpec>,
        attribs: Vec<Vec<f64>>,
        finished: bool,
    }

    impl SplatReceiver for Collect {
        fn init_splats(&mut self, init: &SplatInit) -> Result<()> {
            let n = init.num_splats;
            self.init = Some(init.clone());
            self.center = vec![0.0; n * 3];
            self.opacity = vec![0.0; n];
            self.rgb = vec![0.0; n * 3];
            self.scale = vec![0.0; n * 3];
            self.quat = vec![0.0; n * 4];
            self.sh1 = vec![0.0; n * 9];
            self.child_count = vec![0; n];
            self.child_start = vec![0; n];
            Ok(())
        }
        fn finish(&mut self) -> Result<()> {
            self.finished = true;
            Ok(())
        }
        fn set_batch(&mut self, base: usize, count: usize, b: &SplatProps) {
            self.center[base * 3..(base + count) * 3].copy_from_slice(b.center);
            self.opacity[base..base + count].copy_from_slice(b.opacity);
            self.rgb[base * 3..(base + count) * 3].copy_from_slice(b.rgb);
            self.scale[base * 3..(base + count) * 3].copy_from_slice(b.scale);
            self.quat[base * 4..(base + count) * 4].copy_from_slice(b.quat);
            if !b.sh1.is_empty() {
                self.sh1[base * 9..(base + count) * 9].copy_from_slice(b.sh1);
            }
            self.child_count[base..base + count].copy_from_slice(b.child_count);
            self.child_start[base..base + count].copy_from_slice(b.child_start);
        }
        fn set_center(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_opacity(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_rgb(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_rgba(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_scale(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_quat(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn init_attribs(&mut self, specs: &[AttribSpec]) {
            let n = self.opacity.len();
            self.specs = specs.to_vec();
            self.attribs = specs.iter().map(|s| vec![0.0; n * s.components]).collect();
        }
        fn set_attrib(&mut self, attrib: usize, base: usize, count: usize, values: &[f64]) {
            let c = self.specs[attrib].components;
            self.attribs[attrib][base * c..(base + count) * c].copy_from_slice(&values[..count * c]);
        }
    }

    fn decode(bytes: &[u8]) -> Collect {
        let mut d = AthcDecoder::new(Collect::default());
        d.push(bytes).unwrap();
        d.finish().unwrap();
        d.into_splats()
    }

    #[test]
    fn two_cards_header_is_what_athenea_wrote() {
        let layout = AthcLayout::parse(TWO_CARDS, TWO_CARDS.len() as u64).unwrap();
        let h = layout.header;
        assert_eq!(h.version, 2);
        assert_eq!(h.flags, FLAG_NORMALS);
        assert_eq!((h.count, h.rest_per_colour, h.sh_words), (1352, 0, 1));
        assert_eq!((h.levels, h.chunk_splats, h.chunks, h.finest_groups), (6, 65536, 1, 527));
        assert_eq!((h.level_table, h.chunk_table, h.starts), (4096, 4096 + 6 * 16, 8192));
        assert_eq!(h.bounds_min, [0.019_531_248, 0.019_531_248, -4.0]);
        assert_eq!(layout.extra, ExtraHeader::default());
        // positions 16 + shape 16 + sh 4 + tail 4 + normals 4
        assert_eq!(layout.element_bytes, 44);
        let levels: Vec<(u32, u32)> = layout.levels.iter().map(|e| (e.level, e.groups)).collect();
        assert_eq!(levels[..4], [(1, 3), (2, 3), (3, 10), (4, 36)]);
        assert_eq!(levels.last().unwrap().1, 527);
        assert_eq!(layout.levels[0].offset, 0x3000);
        assert_eq!(layout.chunks, vec![ChunkEntry { offset: layout.chunks[0].offset, count: 1352 }]);
        // Every block on its own page.
        assert!(layout.levels.iter().all(|e| e.offset % PAGE == 0));
        assert_eq!(layout.chunks[0].offset % PAGE, 0);
        assert_eq!(TWO_CARDS.len() as u64 % PAGE, 0);
    }

    #[test]
    fn two_cards_round_trips_byte_for_byte() {
        let file = AthcFile::read(TWO_CARDS).unwrap();
        let written = file.write().unwrap();
        assert_eq!(written.len(), TWO_CARDS.len());
        assert!(written == TWO_CARDS, "rewritten .athc differs from athenea's");
        assert_eq!(AthcFile::read(&written).unwrap(), file);
    }

    #[test]
    fn two_cards_levels_are_an_octree() {
        let file = AthcFile::read(TWO_CARDS).unwrap();
        // The splats' tail is their finest group, in order, matching starts.
        let splats = file.splats();
        let mut g = 0usize;
        for s in 0..splats.n {
            while g + 1 < file.starts.len() && file.starts[g + 1] as usize <= s {
                g += 1;
            }
            assert_eq!(splats.tail[s] as usize, g, "splat {s}");
        }
        let tree = VirtualTree::of_file(&file, false).unwrap();
        assert!(tree.synth_root, "level 1 holds 3 groups");
        assert_eq!(tree.merged, 1 + file.levels.iter().map(|(_, b)| b.n as u32).sum::<u32>());
        assert_eq!(tree.clipped_groups, 0);
        // Every node but the root is some node's child exactly once.
        let total = (tree.merged + file.header.count) as usize;
        let mut parents = vec![0u32; total];
        for k in 0..tree.merged as usize {
            for c in tree.child_start[k]..tree.child_start[k] + tree.child_count[k] as u32 {
                parents[c as usize] += 1;
            }
        }
        assert_eq!(parents[0], 0);
        assert!(parents[1..].iter().all(|&p| p == 1));
        let paged = VirtualTree::of_file(&file, true).unwrap();
        assert_eq!(paged.splat_base, PAGE_SPLATS);
    }

    #[test]
    fn two_cards_decodes_as_a_lod_tree() {
        let file = AthcFile::read(TWO_CARDS).unwrap();
        let out = decode(TWO_CARDS);
        let tree = VirtualTree::of_file(&file, false).unwrap();
        let init = out.init.as_ref().unwrap();
        assert!(out.finished && init.lod_tree);
        assert_eq!(init.num_splats, (tree.merged + 1352) as usize);
        assert_eq!(init.max_sh_degree, 0);
        assert_eq!(out.specs.len(), 2);
        assert_eq!(out.specs[0].name, "normalOct");
        assert_eq!(out.specs[1].name, GROUP_ATTRIBUTE);
        let splats = file.splats();
        for s in [0usize, 700, 1351] {
            let v = tree.merged as usize + s;
            assert_eq!(out.center[v * 3..v * 3 + 3], splats.positions[s * 4..s * 4 + 3]);
            assert_eq!(out.opacity[v], splats.positions[s * 4 + 3]);
            assert_eq!(out.scale[v * 3], low_half(splats.shape[s * 4 + 1]).exp());
            assert_eq!(out.rgb[v * 3 + 2], high_half(splats.shape[s * 4 + 3]));
            let q = &out.quat[v * 4..v * 4 + 4];
            assert!((q.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-5);
            assert_eq!(out.attribs[0][v], splats.normals[s] as f64);
            // Its finest group, as a range of one.
            let g = splats.tail[s] as f64;
            assert_eq!(out.attribs[1][v * 2..v * 2 + 2], [g, g + 1.0]);
            assert_eq!(out.child_count[v], 0);
            // A converted card: normals unit, colours in 0..1.
            let n = unpack_normal(splats.normals[s]);
            assert!(((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) - 1.0).abs() < 1e-5);
        }
        // The root covers everything; its opacity is a merge's.
        assert!(out.opacity[0] > 0.0 && out.opacity[0] <= 0.99);
        assert_eq!(out.child_count[0], 3);
        assert_eq!(out.child_start[0], 1);
        let finest = tree.level_base[5] as usize;
        assert_eq!(out.child_start[finest], tree.splat_base as usize);
        // Group ranges: the root covers every finest group; a node covers
        // its children's; a finest-level node is its own group.
        let range = |v: usize| (out.attribs[1][v * 2] as u32, out.attribs[1][v * 2 + 1] as u32);
        assert_eq!(range(0), (0, file.header.finest_groups));
        assert_eq!(range(finest + 3), (3, 4));
        for v in 0..tree.merged as usize {
            let (first, n) = (out.child_start[v], out.child_count[v] as usize);
            if v >= finest {
                continue;
            }
            assert_eq!(range(v).0, range(first).0, "node {v}");
            assert_eq!(range(v).1, range(first + n - 1).1, "node {v}");
        }
    }

    #[test]
    fn coverage_past_one_is_sparks_lod_opacity() {
        assert_eq!(spark_lod_opacity(0.4), 0.4);
        assert_eq!(spark_lod_opacity(1.0), 1.0);
        // Spark draws stored s > 1 as D = 4 s - 3 and 1 - (1 - g)^exp((D^2 - 1) / e):
        // the ratio comes back as the exponent.
        for o in [1.5f32, 3.0, 12.0, 100.0] {
            let d = 4.0 * spark_lod_opacity(o) - 3.0;
            assert!((((d * d - 1.0) / std::f32::consts::E).exp() - o).abs() < 1e-3 * o, "{o}");
        }
        assert_eq!(spark_lod_opacity(1e9), 2.0);
    }

    #[test]
    fn levels_keep_their_coverage_in_v3_and_cap_it_in_v2() {
        let mut file = AthcFile::read(TWO_CARDS).unwrap();
        let capped = file.clone();
        uncap_levels(&mut file);
        let ratios = coverage_ratios(&file);
        // The uncapped ratio is the capped opacity wherever that was under the cap.
        for ((_, b), r) in capped.levels.iter().zip(&ratios) {
            for i in 0..b.n {
                let was = b.positions[i * 4 + 3];
                if was < 0.98 {
                    assert!((r[i] - was).abs() < 2e-2 * was.max(0.05), "{} {}", r[i], was);
                } else {
                    assert!(r[i] >= 0.98 - 1e-3);
                }
            }
        }
        assert!(file.over_capped(), "two cards: overlapping splats cover past 1");
        let v3 = crate::athc_v3::read_v3(&crate::athc_v3::write_v3(&file, crate::athc_v3::COMPRESSION_NONE).unwrap()).unwrap();
        for ((_, a), (_, b)) in v3.levels.iter().zip(&file.levels) {
            assert_eq!(a.positions, b.positions);
        }
        let v2 = AthcFile::read(&file.write().unwrap()).unwrap();
        assert!(!v2.over_capped());
        // Splats as they were (at most 1); only merged opacities are capped.
        assert_eq!(v2.chunks, file.chunks);
    }

    #[test]
    fn truncated_levels_are_a_cloud_of_merged_splats() {
        let file = AthcFile::read(TWO_CARDS).unwrap();
        let keep = file.levels.len() - 1;
        let cut = truncate_levels(&file, keep).unwrap();
        let kept = &file.levels[keep].1;
        assert_eq!(cut.header.count as usize, kept.n);
        assert_eq!(cut.levels.len(), keep);
        // The levels kept are the file's, with their whole coverage.
        for ((la, a), (lb, b)) in cut.levels.iter().zip(&file.levels) {
            assert_eq!((la, &a.shape, &a.tail), (lb, &b.shape, &b.tail));
        }
        // Each new splat's tail is its parent group, and starts are where
        // each parent's run begins.
        let splats = cut.splats();
        for g in 0..cut.starts.len() {
            let end = if g + 1 < cut.starts.len() { cut.starts[g + 1] } else { cut.header.count };
            assert!(end > cut.starts[g]);
            assert!(splats.tail[cut.starts[g] as usize..end as usize].iter().all(|&t| t == g as u32));
        }
        // Their coverage is kept whole, through v3 and the decoder.
        let ratios = coverage_ratios(&file);
        assert!(ratios[keep].iter().any(|&o| o > 1.0));
        let bytes = crate::athc_v3::write_v3(&cut, crate::athc_v3::COMPRESSION_GZIP).unwrap();
        let back = crate::athc_v3::read_v3(&bytes).unwrap();
        assert_eq!(back.splats().positions, splats.positions);
        let out = decode(&bytes);
        let tree = VirtualTree::of_file(&back, false).unwrap();
        assert_eq!(out.opacity.len(), (tree.merged + cut.header.count) as usize);
        assert!(out.opacity[tree.merged as usize..].iter().any(|&a| a > 1.0));
        assert!(out.opacity.iter().all(|&a| a <= 2.0));
    }

    #[test]
    fn a_crease_aware_cut_keeps_finer_splats_where_normals_spread() {
        const HOOD: &[u8] = include_bytes!("../../../test/fixtures/athc/hood_t16.athc");
        let file = crate::athc_v3::read_v3(HOOD).unwrap();
        let keep = file.levels.len() - 3;
        let plain = truncate_levels(&file, keep).unwrap();
        // No bound: the plain cut, plus each merged splat's normal variance.
        let (same, stats) = truncate_creases(&file, keep, 1.0, usize::MAX).unwrap();
        assert_eq!((stats.per_level[0], stats.splats), (file.levels[keep].1.n, 0));
        let (a, b) = (plain.splats(), same.splats());
        assert_eq!((&a.positions, &a.shape, &a.tail, &a.transfer), (&b.positions, &b.shape, &b.tail, &b.transfer));
        assert_eq!(plain.starts, same.starts);
        let moments = normal_moments(&file);
        for i in 0..b.n {
            assert_eq!(a.curvature[2 * i], b.curvature[2 * i]);
            assert_eq!(a.curvature[2 * i + 1] & 0xffff, b.curvature[2 * i + 1] & 0xffff);
            let v = high_half(b.curvature[2 * i + 1]);
            assert!((v - normal_variance(&moments[keep][i])).abs() <= 1e-3 * v.max(1e-2), "{i}: {v}");
        }
        assert!((0..b.n).any(|i| high_half(b.curvature[2 * i + 1]) > 0.0));
        // A bound of 0: every group with any spread goes down, to the splats.
        let (fine, stats) = truncate_creases(&file, keep, 0.0, usize::MAX).unwrap();
        assert!(stats.splats > 0 && fine.header.count > plain.header.count);
        let one = truncate_creases(&file, keep, 0.0, 1).unwrap().1;
        assert_eq!((one.per_level.len(), one.splats), (stats.per_level.len(), 0));
        assert!(one.per_level[1] > 0);
        // Still a tree: each parent's run, tails its index; and every splat's
        // weight is there once (the cut covers the cloud).
        let splats = fine.splats();
        for g in 0..fine.starts.len() {
            let end = if g + 1 < fine.starts.len() { fine.starts[g + 1] } else { fine.header.count };
            assert!(end > fine.starts[g]);
            assert!(splats.tail[fine.starts[g] as usize..end as usize].iter().all(|&t| t == g as u32));
        }
        let bytes = crate::athc_v3::write_v3(&fine, crate::athc_v3::COMPRESSION_GZIP).unwrap();
        let back = crate::athc_v3::read_v3(&bytes).unwrap();
        assert_eq!(back.splats().positions, splats.positions);
        let tree = VirtualTree::of_file(&back, false).unwrap();
        assert_eq!(decode(&bytes).opacity.len(), (tree.merged + fine.header.count) as usize);
    }

    #[test]
    fn athv_pages_decode_as_the_whole_file() {
        let whole = decode(TWO_CARDS);
        let layout = AthcLayout::parse(TWO_CARDS, TWO_CARDS.len() as u64).unwrap();
        let prefix = &TWO_CARDS[..layout.levels_end() as usize];
        let (tree, pages) = athv_merged_pages(prefix, TWO_CARDS.len() as u64).unwrap();
        assert_eq!(pages.len(), 1);
        let merged = decode(&pages[0]);
        let m = tree.merged as usize;
        assert_eq!(merged.opacity.len(), m);
        // A synthesised root merges level 1 by opacity: the whole file has
        // the levels' uncapped coverage (`coverage_ratios`), a page of a v2
        // file its capped opacities, so the root may sit elsewhere.
        let from = if tree.synth_root { 3 } else { 0 };
        assert_eq!(merged.center[from..], whole.center[from..m * 3]);
        assert_eq!(merged.child_count[..], whole.child_count[..m]);
        // Children of the finest groups point at page 1, where the splats are.
        for k in 0..m {
            let shift = if whole.child_start[k] >= m { PAGE_SPLATS as usize - m } else { 0 };
            assert_eq!(merged.child_start[k], whole.child_start[k] + shift);
        }
        let e = layout.chunks[0];
        let mut page = athv_head(ATHV_SPLATS, tree.splat_base, e.count, TWO_CARDS);
        page.extend_from_slice(&TWO_CARDS[e.offset as usize..(e.offset + e.count as u64 * layout.element_bytes) as usize]);
        let splats = decode(&page);
        assert_eq!(splats.center[..], whole.center[m * 3..]);
        assert_eq!(splats.attribs[0][..], whole.attribs[0][m..]);
        assert!(splats.child_count.iter().all(|&c| c == 0));
        // Group ranges, merged and splats, as the whole file has them.
        assert_eq!(merged.attribs[1][..], whole.attribs[1][..2 * m]);
        assert_eq!(splats.attribs[1][..], whole.attribs[1][2 * m..]);
    }

    #[test]
    fn v3_section_pages_decode_as_the_chunk() {
        use crate::athc_v3::*;
        let file = every_stream();
        let bytes = file.write().unwrap();
        let layout = AthcLayout::parse(&bytes, bytes.len() as u64).unwrap();
        let (tree, _) = athv_merged_pages(&bytes[..layout.levels_end() as usize], bytes.len() as u64).unwrap();
        let e = layout.chunks[0];
        let mut page = athv_head(ATHV_SPLATS, tree.splat_base, e.count, &bytes);
        page.extend_from_slice(&bytes[e.offset as usize..(e.offset + e.count as u64 * layout.element_bytes) as usize]);
        let whole = decode(&page);
        let v3 = write_v3(&file, COMPRESSION_GZIP).unwrap();
        let l3 = parse_v3(&v3).unwrap();
        let b = l3.blocks.iter().find(|b| b.kind == 1).unwrap();
        let section_page = |want: Want, flags: u32| {
            let picks = wanted_sections(&l3.sections, &l3.extra, want);
            let parts: Vec<_> = picks
                .iter()
                .map(|&k| {
                    let s = b.spans[k];
                    (&l3.sections[k], &v3[s.offset as usize..s.offset as usize + s.stored as usize], s.raw)
                })
                .collect();
            athv_sections_page(&l3.v2_headers(), tree.splat_base, b.n, flags, want.transfer_values, &parts)
        };
        let all = decode(&section_page(Want::all(&l3.extra), 0));
        assert_eq!(all.center, whole.center);
        assert_eq!(all.rgb, whole.rgb);
        assert_eq!(all.sh1, whole.sh1);
        assert_eq!(all.specs, whole.specs);
        assert_eq!(all.attribs, whole.attribs);
        // The splats alone: the same splats, and only their groups.
        let core = decode(&section_page(Want::default(), ATHV_KEEP_LINEAR));
        assert_eq!(core.center, whole.center);
        assert_eq!(core.specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), [GROUP_ATTRIBUTE]);
        assert_eq!(core.attribs[0], whole.attribs[6]);
        assert_eq!(core.rgb[1000 * 3], 0.25, "decode flags kept");
        // A shorter transfer: its first 64 values.
        let short = decode(&section_page(Want { material: false, transfer_values: 64 }, 0));
        let names: Vec<&str> = short.specs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["transfer", "shadowBits", GROUP_ATTRIBUTE]);
        assert_eq!(short.specs[0].components, 64);
        for i in [0usize, 777] {
            assert_eq!(short.attribs[0][i * 64..i * 64 + 64], whole.attribs[4][i * 112..i * 112 + 64]);
        }
    }

    #[test]
    fn athv_keep_linear_flag() {
        let bytes = every_stream().write().unwrap();
        let layout = AthcLayout::parse(&bytes, bytes.len() as u64).unwrap();
        let (tree, mut pages) = athv_merged_pages(&bytes[..layout.levels_end() as usize], bytes.len() as u64).unwrap();
        let e = layout.chunks[0];
        let mut page = athv_head(ATHV_SPLATS, tree.splat_base, e.count, &bytes);
        page.extend_from_slice(&bytes[e.offset as usize..(e.offset + e.count as u64 * layout.element_bytes) as usize]);
        let srgb = decode(&page);
        assert!((srgb.rgb[1000 * 3] - linear_to_srgb(0.25)).abs() < 1e-6);
        page[152..156].copy_from_slice(&ATHV_KEEP_LINEAR.to_le_bytes());
        assert_eq!(decode(&page).rgb[1000 * 3], 0.25);
        let before = decode(&pages[0]).rgb;
        pages[0][152..156].copy_from_slice(&ATHV_KEEP_LINEAR.to_le_bytes());
        let kept = decode(&pages[0]).rgb;
        assert!(before.iter().zip(&kept).any(|(a, b)| a != b));
        assert!(kept.iter().zip(&before).all(|(k, b)| (linear_to_srgb(*k) - b).abs() < 1e-5));
    }

    #[test]
    fn unknown_flag_bits_are_refused_by_name() {
        // athenea's own test: version 2, normals and bit 3 (test_lod.cpp).
        let mut page = vec![0u8; 4096];
        page[..4].copy_from_slice(b"ATHC");
        page[4..8].copy_from_slice(&2u32.to_le_bytes());
        page[76..80].copy_from_slice(&(1u32 | 1 << 3).to_le_bytes());
        let err = parse_headers(&page).unwrap_err().to_string();
        assert!(err.contains("unknown flag bits 3;"), "{err}");
        let mut v3 = TWO_CARDS.to_vec();
        v3[4..8].copy_from_slice(&3u32.to_le_bytes());
        assert!(AthcFile::read(&v3).unwrap_err().to_string().contains("version 3"));
        // A version 1 file's flags word is padding: ignored.
        let mut v1 = TWO_CARDS.to_vec();
        v1[4..8].copy_from_slice(&1u32.to_le_bytes());
        let read = AthcFile::read(&v1).unwrap();
        assert_eq!(read.header.flags, 0);
    }

    /// A cloud carrying every stream, with athenea's test values
    /// (test_lod.cpp, "a transfer and a material go through ... a .athc").
    fn every_stream() -> AthcFile {
        every_stream_of(70_000)
    }

    fn every_stream_of(count: u32) -> AthcFile {
        const ONE: u32 = 0x3C00_3C00; // two halves of 1.0
        const OPEN: u32 = 0xFFFF_FFFF;
        const MATERIAL: u32 = 0x00FF_8040;
        let block = |n: usize, tail: Vec<u32>| {
            let mut b = AthcBlock { n, ..Default::default() };
            for i in 0..n {
                let t = i as f32 / n as f32;
                b.positions.extend_from_slice(&[t, 1.0 - t, 0.5 * t, 0.25 + 0.5 * t]);
                b.shape.extend_from_slice(&[
                    encode_quaternion([0.1, 0.2, 0.3 + t, 0.9]),
                    pack_halves((0.01f32 + t).ln(), 0.02f32.ln()),
                    pack_halves(0.03f32.ln(), 0.25),
                    pack_halves(0.5, 0.75),
                ]);
                // degree 3: 45 halves in 23 words
                b.sh.extend((0..23).map(|k| pack_halves(0.01 * k as f32, -0.01 * k as f32 - t)));
                b.normals.push(pack_normal([t, 0.5, -1.0]));
                b.emission.push((15 << 27) | (256 << 18) | (128 << 9) | 64);
                b.pbr.push(MATERIAL);
                b.lobes.extend_from_slice(&[0xFF00_8040, 0x9930_4010, 0x0102_0304]);
                b.transfer.extend(std::iter::repeat_n(ONE, 56));
                b.shadow_bits.extend(std::iter::repeat_n(OPEN, 8));
            }
            b.tail = tail;
            b
        };
        let starts: Vec<u32> = (0..8).map(|g| g * (count / 8)).collect();
        let splat_groups: Vec<u32> = (0..count).map(|s| (s / (count / 8)).min(7)).collect();
        let mut header = AthcHeader {
            rest_per_colour: 15,
            chunk_splats: 65536,
            bounds_lo: [0.0; 3],
            extent: 1.0,
            bounds_min: [0.0; 3],
            bounds_max: [1.0; 3],
            flags: FLAG_LINEAR,
            ..Default::default()
        };
        header.flags |= FLAG_LINEAR;
        let all = block(count as usize, splat_groups);
        AthcFile {
            header,
            extra: ExtraHeader { transfer_count: 112, ..Default::default() },
            levels: vec![(1, block(1, vec![0])), (2, block(8, (0..8).collect()))],
            starts,
            chunks: (0..count as usize)
                .step_by(65536)
                .map(|at| all.slice(at, (count as usize - at).min(65536)))
                .collect(),
        }
    }

    #[test]
    fn every_stream_writes_as_athenea_does() {
        let file = every_stream();
        let bytes = file.write().unwrap();
        assert_eq!(&bytes[..4], b"ATHC");
        assert_eq!(u32_at(&bytes, 4), 2);
        assert_eq!(u32_at(&bytes, 76), FLAG_NORMALS | FLAG_LINEAR | FLAG_EMISSION | FLAG_MATERIAL | FLAG_TRANSFER);
        // The extra header right after the header: pbr 1, lobes 3, 112 values
        // in 56 words, 8 shadow words.
        assert_eq!((104..124).step_by(4).map(|at| u32_at(&bytes, at)).collect::<Vec<_>>(), vec![1, 3, 112, 56, 8]);
        let layout = AthcLayout::parse(&bytes, bytes.len() as u64).unwrap();
        // 32 + 23 * 4 + 4 + 4 + 4 + 4 * (1 + 3 + 56 + 8)
        assert_eq!(layout.element_bytes, 32 + 92 + 12 + 4 * 68);
        assert_eq!(layout.chunks.iter().map(|c| c.count).collect::<Vec<_>>(), vec![65536, 4464]);
        assert!(layout.chunks.iter().all(|c| c.offset % PAGE == 0));
        let read = AthcFile::read(&bytes).unwrap();
        let mut expected = file.clone();
        expected.header = read.header;
        expected.extra = read.extra;
        assert_eq!(read, expected);
        assert_eq!(read.extra.transfer_words, 56);
        // Rewriting what was read is the identity.
        assert!(read.write().unwrap() == bytes);
    }

    #[test]
    fn every_stream_decodes_to_attributes() {
        let bytes = every_stream().write().unwrap();
        let out = decode(&bytes);
        let names: Vec<&str> = out.specs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["normalOct", "emission", "pbr", "lobes", "transfer", "shadowBits", GROUP_ATTRIBUTE]);
        assert_eq!(out.init.as_ref().unwrap().max_sh_degree, 3);
        let tree_root_children = out.child_count[0];
        assert_eq!(tree_root_children, 8, "level 1 has one group: it is the root");
        let v = 9 + 1000; // a splat
        assert_eq!(out.attribs[2][v], 0x00FF_8040 as f64);
        assert_eq!(out.attribs[3][v * 3..v * 3 + 3], [0xFF00_8040u32 as f64, 0x9930_4010u32 as f64, 0x0102_0304 as f64]);
        assert!(out.attribs[4][v * 112..v * 112 + 112].iter().all(|&t| t == 1.0));
        assert!(out.attribs[5][v * 8..v * 8 + 8].iter().all(|&b| b == 0xFFFF_FFFFu32 as f64));
        assert_eq!(unpack_rgb9e5(out.attribs[1][v] as u32), [64.0 / 512.0, 128.0 / 512.0, 256.0 / 512.0]);
        // Linear: the base colour goes out sRGB-encoded.
        assert!((out.rgb[v * 3] - linear_to_srgb(0.25)).abs() < 1e-6);
        assert!((out.rgb[v * 3 + 1] - linear_to_srgb(0.5)).abs() < 1e-6);
        let mut keep = AthcDecoder::new(Collect::default());
        keep.options.keep_linear = true;
        keep.push(&bytes).unwrap();
        keep.finish().unwrap();
        assert_eq!(keep.into_splats().rgb[v * 3], 0.25);
    }

    /// test/fixtures/athc/every_stream.athc is this cloud of 300 splats:
    /// the GPU and browser tests read it. ATHC_WRITE_FIXTURES=1 rewrites it.
    #[test]
    fn every_stream_fixture_is_current() {
        let bytes = every_stream_of(300).write().unwrap();
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test/fixtures/athc/every_stream.athc");
        if std::env::var("ATHC_WRITE_FIXTURES").is_ok() {
            std::fs::write(path, &bytes).unwrap();
        }
        assert!(std::fs::read(path).unwrap() == bytes, "stale fixture: ATHC_WRITE_FIXTURES=1 cargo test");
    }

    #[test]
    fn v3_decodes_as_v2() {
        let v3 = crate::athc_v3::write_v3(&AthcFile::read(TWO_CARDS).unwrap(), crate::athc_v3::COMPRESSION_GZIP).unwrap();
        let (a, b) = (decode(TWO_CARDS), decode(&v3));
        assert_eq!(a.center, b.center);
        assert_eq!(a.child_start, b.child_start);
        assert_eq!(a.attribs, b.attribs);
    }

    #[test]
    fn packing_matches_packing_slang() {
        for q in [[0.0, 0.0, 0.0, 1.0], [0.5, -0.5, 0.5, 0.5], [0.1, 0.7, -0.2, 0.3]] {
            let len = (q.iter().map(|v: &f32| v * v).sum::<f32>()).sqrt();
            let d = decode_quaternion(encode_quaternion(q));
            let dot: f32 = (0..4).map(|k| d[k] * q[k] / len).sum();
            assert!(dot.abs() > 0.9999, "{q:?} -> {d:?}");
        }
        for n in [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.6, -0.8, 0.0], [-0.3, 0.4, -0.866]] {
            let back = unpack_normal(pack_normal(n));
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((0..3).all(|k| (back[k] - n[k] / l).abs() < 1e-3), "{n:?} -> {back:?}");
        }
        // e = 15: a step of 2^-9.
        assert_eq!(unpack_rgb9e5((15 << 27) | 256), [0.5, 0.0, 0.0]);
    }
}
