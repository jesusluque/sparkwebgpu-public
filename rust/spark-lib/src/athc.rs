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
//! ```
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
pub const KNOWN_FLAGS: u32 = FLAG_NORMALS | FLAG_LINEAR | FLAG_EMISSION | FLAG_MATERIAL | FLAG_TRANSFER;

/// Spark's LoD pages (and athenea's default chunk): 65 536 splats.
pub const PAGE_SPLATS: u32 = 65536;

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
        }
    }

    pub fn to_bytes(&self) -> [u8; EXTRA_HEADER_BYTES] {
        let mut b = [0u8; EXTRA_HEADER_BYTES];
        for (k, w) in [self.pbr_words, self.lobes_words, self.transfer_count, self.transfer_words, self.shadow_words]
            .iter()
            .enumerate()
        {
            b[4 * k..4 * k + 4].copy_from_slice(&w.to_le_bytes());
        }
        b
    }

    pub fn words(&self) -> u32 {
        self.pbr_words + self.lobes_words + self.transfer_words + self.shadow_words
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
            "not a readable .athc (unknown flag bits {}; this reads bits 0 (normals), 1 (linear), 2 (emission), 4 (material) and 5 (transfer))",
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
        Ok(Self { n, positions, shape, sh, tail, normals, emission, pbr, lobes, transfer, shadow_bits })
    }

    /// Every array after positions, in file order.
    fn arrays(&self) -> [&[u32]; 9] {
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

    /// The splats of every chunk, as one block.
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
        Ok(Self { synth_root, level_base, merged, splat_base, count, child_start, child_count, clipped_groups })
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
/// opacity min(W / own area, 0.99). Everything else is the heaviest group's.
pub fn merge_root(level1: &AthcBlock, sh_words: usize) -> AthcBlock {
    let n = level1.n;
    let mut weights = Vec::with_capacity(n);
    let mut covs = Vec::with_capacity(n);
    let mut sum_w = 0.0f64;
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
    let opacity = ((sum_w as f32) / (sorted[0] * sorted[1]).max(1e-20)).min(0.99);
    let base = base.map(|v| (v / sum_w) as f32);
    let mut root = level1.slice(heaviest, 1);
    root.positions = vec![mu[0] as f32, mu[1] as f32, mu[2] as f32, opacity];
    root.shape = vec![
        encode_quaternion(q.to_array()),
        pack_halves(scale[0].ln(), scale[1].ln()),
        pack_halves(scale[2].ln(), base[0]),
        pack_halves(base[1], base[2]),
    ];
    root.sh = (0..sh_words)
        .map(|k| pack_halves((rest[2 * k] / sum_w) as f32, (rest[2 * k + 1] / sum_w) as f32))
        .collect();
    root.tail = vec![0];
    root
}

/// The merged nodes in virtual order (root, then every level), one block.
pub fn merged_block(file: &AthcFile, tree: &VirtualTree) -> AthcBlock {
    let mut all = if tree.synth_root {
        merge_root(&file.levels[0].1, file.header.sh_words as usize)
    } else {
        AthcBlock::default()
    };
    for (_, level) in &file.levels {
        all.append(level);
    }
    all
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
/// 152  0, 0
/// 160  kind 0: child_count u32[n], child_start u32[n] (virtual), then
///      the block; kind 1: the block (the file's chunk bytes)
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
    let tree = VirtualTree::of_file(&file, true)?;
    let merged = merged_block(&file, &tree);
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
    out
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DecodeOptions {
    /// Keep a linear cloud's colours linear. By default they are encoded to
    /// sRGB (the base exactly, the rest harmonics through the curve's slope
    /// at the base), which is what Spark's blend expects.
    pub keep_linear: bool,
}

/// Feeds `block`'s elements to `receiver` as splats base .. base + n, with
/// `children` (counts, virtual starts) when they are tree nodes.
pub fn emit_block<T: SplatReceiver>(
    receiver: &mut T,
    base: usize,
    block: &AthcBlock,
    h: &AthcHeader,
    x: &ExtraHeader,
    children: Option<(&[u16], &[u32])>,
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
            opacity[i] = p[3];
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
    // The extras, as attributes: words as they are, the transfer as halves.
    let mut k = 0;
    let words = |v: &[u32]| -> Vec<f64> { v.iter().map(|&w| w as f64).collect() };
    for (on, values) in [
        (h.has(FLAG_NORMALS), &block.normals),
        (h.has(FLAG_EMISSION), &block.emission),
        (x.pbr_words > 0, &block.pbr),
        (x.lobes_words > 0, &block.lobes),
    ] {
        if on {
            receiver.set_attrib(k, base, block.n, &words(values));
            k += 1;
        }
    }
    if x.transfer_words > 0 {
        let tc = x.transfer_count as usize;
        let tw = x.transfer_words as usize;
        let mut values = vec![0.0f64; block.n * tc];
        for i in 0..block.n {
            for c in 0..tc {
                let word = block.transfer[i * tw + c / 2];
                values[i * tc + c] = if c % 2 == 0 { low_half(word) } else { high_half(word) } as f64;
            }
        }
        receiver.set_attrib(k, base, block.n, &values);
        k += 1;
    }
    if x.shadow_words > 0 {
        receiver.set_attrib(k, base, block.n, &words(&block.shadow_bits));
    }
}

fn begin<T: SplatReceiver>(receiver: &mut T, num_splats: usize, h: &AthcHeader, x: &ExtraHeader) -> Result<()> {
    // A LoD tree: receivers mark the encoding lodOpacity, which is what makes
    // the loader treat the result as LoD splats. athenea's merged opacities
    // stay at most 0.99, so the extended range goes unused.
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
    pub options: DecodeOptions,
    /// Set after finish(): the tree a whole file was decoded with.
    pub tree: Option<VirtualTree>,
}

impl<T: SplatReceiver> AthcDecoder<T> {
    pub fn new(splats: T) -> Self {
        Self { splats, buffer: Vec::new(), options: DecodeOptions::default(), tree: None }
    }

    pub fn into_splats(self) -> T {
        self.splats
    }

    fn finish_file(&mut self) -> Result<()> {
        let file = AthcFile::read(&self.buffer)?;
        let (h, x) = (file.header, file.extra);
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
            true,
            self.options,
        );
        let mut at = tree.splat_base as usize;
        for chunk in &file.chunks {
            emit_block(&mut self.splats, at, chunk, &h, &x, None, true, self.options);
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
        begin(&mut self.splats, n, &h, &x)?;
        let children = children.as_ref().map(|(c, s)| (&c[..], &s[..]));
        emit_block(&mut self.splats, 0, &block, &h, &x, children, true, self.options);
        self.splats.finish()
    }
}

impl<T: SplatReceiver> ChunkReceiver for AthcDecoder<T> {
    fn push(&mut self, bytes: &[u8]) -> Result<()> {
        self.buffer.extend_from_slice(bytes);
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.buffer.len() < 4 {
            bail!(".athc shorter than its magic");
        }
        let result = match u32_at(&self.buffer, 0) {
            ATHC_MAGIC => self.finish_file(),
            ATHV_MAGIC => self.finish_page(),
            _ => Err(anyhow!("neither an .athc nor an ATHV page")),
        };
        self.buffer = Vec::new();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::SplatInit;

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
        assert_eq!(out.specs.len(), 1);
        assert_eq!(out.specs[0].name, "normalOct");
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
        assert_eq!(merged.center[..], whole.center[..m * 3]);
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
        assert_eq!(names, ["normalOct", "emission", "pbr", "lobes", "transfer", "shadowBits"]);
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
