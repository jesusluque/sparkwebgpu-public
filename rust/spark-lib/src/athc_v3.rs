//! `.athc` version 3: a v2 cloud rearranged by sections, so that what a tier
//! needs of a block (a level or a chunk) is one prefix of it, read with one
//! HTTP Range request (athenea's proposal 083). docs/docs/athc-v3.md is the
//! design; this is its reader and writer, and the v2 <-> v3 converter.
//!
//! Version 3 changes where the bytes are, not what they are: every section
//! here is encoding 0, the v2 words as they are, so v2 -> v3 -> v2 gives the
//! same file byte for byte. Quantized encodings (083's 20-byte S0) and
//! per-section gzip are the format's room to grow: gzip is implemented
//! (`compression` 1, what a browser's DecompressionStream reads), smaller
//! encodings are not yet.

use anyhow::{anyhow, bail, Result};
use miniz_oxide::deflate::compress_to_vec;
use serde::{Serialize, Serializer};
use miniz_oxide::inflate::decompress_to_vec;

use crate::athc::{
    aligned, AthcBlock, AthcFile, AthcHeader, ExtraHeader, FLAG_EMISSION, FLAG_MATERIAL, FLAG_NORMALS, FLAG_TRANSFER,
    PAGE,
};

pub const ATH3_MAGIC: u32 = u32::from_le_bytes(*b"ATH3");
pub const V3_VERSION: u32 = 3;
pub const V3_HEADER_BYTES: usize = 160;
pub const SECTION_ENTRY_BYTES: usize = 32;
pub const BLOCK_ENTRY_HEAD: usize = 32;
pub const BLOCK_ENTRY_PER_SECTION: usize = 16;

pub const COMPRESSION_NONE: u32 = 0;
pub const COMPRESSION_GZIP: u32 = 1;
pub const ENCODING_V2_WORDS: u32 = 0;

/// Sections, in the order they sit in every block (tier order). Within the
/// relight tier the shadow bits come first and the transfer in the order of
/// its layouts (transfer_layout.slang), so every reduced form a reader may
/// want (`Want`) is one contiguous run: shadow and the direct half, then
/// the indirect half, then the reflected field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectionId {
    /// positions (f32 x4), shape (u32 x4), tail (u32): 36 bytes.
    Core,
    /// The rest harmonics, `shWords` words.
    Sh,
    /// normals, emission, pbr, lobes: those the flags say, in that order.
    Material,
    /// The open-direction bits.
    Shadow,
    /// sparkwebGPU's curvature (athenea's shape operator, three f16 in two
    /// words): read with the relight streams, so it sits right after the
    /// shadow bits.
    Curvature,
    /// The transfer's first ceil(direct / 2) words.
    TransferDirect,
    /// The transfer's words through the indirect half (2 * direct words).
    TransferIndirect,
    /// The rest: the reflected field.
    TransferField,
}

impl SectionId {
    pub fn code(self) -> u32 {
        u32::from_le_bytes(*match self {
            Self::Core => b"CORE",
            Self::Sh => b"SHRS",
            Self::TransferDirect => b"TXDI",
            Self::TransferIndirect => b"TXIN",
            Self::TransferField => b"TXFD",
            Self::Shadow => b"SHAD",
            Self::Curvature => b"CURV",
            Self::Material => b"MATL",
        })
    }

    pub const ALL: [SectionId; 8] = [
        Self::Core,
        Self::Sh,
        Self::Material,
        Self::Shadow,
        Self::Curvature,
        Self::TransferDirect,
        Self::TransferIndirect,
        Self::TransferField,
    ];

    pub fn from_code(code: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.code() == code)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Core => "CORE",
            Self::Sh => "SHRS",
            Self::Material => "MATL",
            Self::Shadow => "SHAD",
            Self::Curvature => "CURV",
            Self::TransferDirect => "TXDI",
            Self::TransferIndirect => "TXIN",
            Self::TransferField => "TXFD",
        }
    }

    /// The data tier that first needs it: 1 the splats as captured, 2 their
    /// materials, 3 relit (shadow bits and transfer).
    pub fn tier(self) -> u32 {
        match self {
            Self::Core | Self::Sh => 1,
            Self::Material => 2,
            _ => 3,
        }
    }

    pub fn is_transfer(self) -> bool {
        matches!(self, Self::TransferDirect | Self::TransferIndirect | Self::TransferField)
    }
}

impl Serialize for SectionId {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

/// The transfer's direct half in values (transfer_layout.slang), the zonal
/// transfer (10 values) whole.
pub fn transfer_direct_values(count: u32) -> u32 {
    match count {
        10 => 10,
        16 | 64 | 112 => 16,
        _ => 9,
    }
}

/// Where a transfer of `count` values splits into its three sections, in
/// words of a splat's row: [0, direct) [direct, indirect) [indirect, words).
/// A half the layout does not keep is empty.
pub fn transfer_split_words(count: u32) -> [u32; 3] {
    let words = count.div_ceil(2);
    let d = transfer_direct_values(count);
    let direct = d.div_ceil(2).min(words);
    let indirect = if count >= 4 * d { (2 * d).min(words) } else { direct };
    [direct, indirect, words]
}

/// The shorter transfers a reader may keep of one of `count` values:
/// prefixes that are layouts of their own (transfer_layout.slang), shortest
/// first, `count` itself last (a TX transfer of 112: 16 direct, 64 with
/// the indirect half, 112 with the reflected field).
pub fn transfer_forms(count: u32) -> Vec<u32> {
    let d = transfer_direct_values(count);
    if count == 10 || count == d {
        return vec![count];
    }
    let mut out = vec![d, 4 * d];
    if count > 4 * d {
        out.push(count);
    }
    out
}

/// What a reader keeps of a cloud's optional streams: the material streams
/// (normals, emission, pbr, lobes) or not, and the relight streams (shadow
/// bits and a transfer of `transfer_values`, one of `transfer_forms`) or not
/// (0). The splats themselves (`CORE`, `SHRS`) are always kept.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Want {
    pub material: bool,
    pub transfer_values: u32,
}

impl Want {
    /// Everything the cloud has.
    pub fn all(x: &ExtraHeader) -> Self {
        Self { material: true, transfer_values: x.transfer_count }
    }

    /// Whether section `id` is needed for this.
    pub fn needs(&self, id: SectionId, x: &ExtraHeader) -> bool {
        let [direct, indirect, _] = transfer_split_words(x.transfer_count);
        let words = self.transfer_values.div_ceil(2);
        match id {
            SectionId::Core | SectionId::Sh => true,
            SectionId::Material => self.material,
            SectionId::Shadow | SectionId::Curvature => self.transfer_values > 0,
            SectionId::TransferDirect => words > 0,
            SectionId::TransferIndirect => words > direct,
            SectionId::TransferField => words > indirect,
        }
    }

    /// Checks the transfer form against the cloud's.
    pub fn check(&self, x: &ExtraHeader) -> Result<()> {
        if self.transfer_values != 0 && !transfer_forms(x.transfer_count).contains(&self.transfer_values) {
            bail!(
                ".athc: a transfer of {} values keeps no form of {} (forms: {:?})",
                x.transfer_count,
                self.transfer_values,
                transfer_forms(x.transfer_count)
            );
        }
        Ok(())
    }
}

/// The v2 headers of what `want` keeps of a cloud: the streams left out are
/// gone from the flags and the extra header, the transfer is the form kept.
pub fn reduced_headers(h: &AthcHeader, x: &ExtraHeader, want: Want) -> (AthcHeader, ExtraHeader) {
    let mut h = *h;
    let mut x = *x;
    if !want.material {
        h.flags &= !(FLAG_NORMALS | FLAG_EMISSION | FLAG_MATERIAL);
        x.pbr_words = 0;
        x.lobes_words = 0;
    }
    let values = want.transfer_values.min(x.transfer_count);
    if values == 0 {
        h.flags &= !FLAG_TRANSFER;
        x.transfer_count = 0;
        x.transfer_words = 0;
        x.shadow_words = 0;
        x.curvature_words = 0;
    } else {
        x.transfer_count = values;
        x.transfer_words = values.div_ceil(2);
    }
    h.version = if h.flags != 0 { 2 } else { 1 };
    (h, x)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Section {
    pub id: SectionId,
    pub tier: u32,
    pub encoding: u32,
    pub compression: u32,
    pub words: u32,
}

/// The sections of a file written before the three tiers (the first v3:
/// `TXIN` holding the field too, then `SHAD` and `MATL`, all tier 2). Still
/// read, and paged by the browser: a reader takes what a block's table says.
pub fn legacy_sections_of(h: &AthcHeader, x: &ExtraHeader, compression: u32) -> Vec<Section> {
    let direct = transfer_direct_values(x.transfer_count).div_ceil(2).min(x.transfer_words);
    let material = (if h.has(FLAG_NORMALS) { 1 } else { 0 })
        + (if h.has(FLAG_EMISSION) { 1 } else { 0 })
        + x.pbr_words
        + x.lobes_words;
    [
        (SectionId::Core, 9, 1),
        (SectionId::Sh, h.sh_words, 1),
        (SectionId::TransferDirect, direct, 2),
        (SectionId::TransferIndirect, x.transfer_words - direct, 2),
        (SectionId::Shadow, x.shadow_words, 2),
        (SectionId::Material, material, 2),
    ]
    .into_iter()
    .filter(|&(_, words, _)| words > 0)
    .map(|(id, words, tier)| Section { id, tier, encoding: ENCODING_V2_WORDS, compression, words })
    .collect()
}

/// The sections a cloud with these headers has, in block order.
pub fn sections_of(h: &AthcHeader, x: &ExtraHeader, compression: u32) -> Vec<Section> {
    let [direct, indirect, words] =
        if x.transfer_words > 0 { transfer_split_words(x.transfer_count) } else { [0, 0, 0] };
    let material = (if h.has(FLAG_NORMALS) { 1 } else { 0 })
        + (if h.has(FLAG_EMISSION) { 1 } else { 0 })
        + x.pbr_words
        + x.lobes_words;
    [
        (SectionId::Core, 9),
        (SectionId::Sh, h.sh_words),
        (SectionId::Material, material),
        (SectionId::Shadow, x.shadow_words),
        (SectionId::Curvature, x.curvature_words),
        (SectionId::TransferDirect, direct),
        (SectionId::TransferIndirect, indirect - direct),
        (SectionId::TransferField, words - indirect),
    ]
    .into_iter()
    .filter(|&(_, words)| words > 0)
    .map(|(id, words)| Section { id, tier: id.tier(), encoding: ENCODING_V2_WORDS, compression, words })
    .collect()
}

fn put_words(out: &mut Vec<u8>, words: &[u32]) {
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

/// Words `from .. to` of each element of an array of `per` words.
fn columns(v: &[u32], n: usize, per: usize, from: usize, to: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(n * (to - from));
    for i in 0..n {
        out.extend_from_slice(&v[i * per + from..i * per + to]);
    }
    out
}

/// One section of a block, uncompressed: its arrays one after the other.
/// `tx_from`: the first transfer word of section `s` (the words of the
/// transfer sections before it).
fn section_bytes(block: &AthcBlock, s: &Section, x: &ExtraHeader, tx_from: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(block.n * s.words as usize * 4);
    match s.id {
        SectionId::Core => {
            for p in &block.positions {
                out.extend_from_slice(&p.to_le_bytes());
            }
            put_words(&mut out, &block.shape);
            put_words(&mut out, &block.tail);
        }
        SectionId::Sh => put_words(&mut out, &block.sh),
        SectionId::TransferDirect | SectionId::TransferIndirect | SectionId::TransferField => {
            let w = x.transfer_words as usize;
            let from = tx_from as usize;
            put_words(&mut out, &columns(&block.transfer, block.n, w, from, from + s.words as usize));
        }
        SectionId::Shadow => put_words(&mut out, &block.shadow_bits),
        SectionId::Curvature => put_words(&mut out, &block.curvature),
        SectionId::Material => {
            for v in [&block.normals, &block.emission, &block.pbr, &block.lobes] {
                put_words(&mut out, v);
            }
        }
    }
    out
}

// --- gzip (RFC 1952), what DecompressionStream("gzip") reads ---------------

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, t) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *t = c;
    }
    let mut crc = !0u32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

pub fn gzip(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff];
    out.extend_from_slice(&compress_to_vec(data, 6));
    out.extend_from_slice(&crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

pub fn gunzip(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 18 || data[0] != 0x1f || data[1] != 0x8b || data[2] != 8 {
        bail!("not a gzip member");
    }
    let flags = data[3];
    let mut at = 10;
    if flags & 4 != 0 {
        at += 2 + u16::from_le_bytes([data[at], data[at + 1]]) as usize;
    }
    for bit in [8, 16] {
        if flags & bit != 0 {
            at += data[at..].iter().position(|&b| b == 0).ok_or_else(|| anyhow!("gzip name"))? + 1;
        }
    }
    if flags & 2 != 0 {
        at += 2;
    }
    let out = decompress_to_vec(&data[at..data.len() - 8]).map_err(|e| anyhow!("gzip: {:?}", e))?;
    let crc = u32::from_le_bytes(data[data.len() - 8..data.len() - 4].try_into().unwrap());
    if crc != crc32(&out) {
        bail!("gzip: crc mismatch");
    }
    Ok(out)
}

// --- the file ---------------------------------------------------------------

/// Where one section of one block is in the file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SectionSpan {
    pub offset: u64,
    pub stored: u32,
    pub raw: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct BlockEntry {
    /// 0 a level's groups, 1 a chunk of splats.
    pub kind: u32,
    /// The level's number (1 the coarsest); 0 for a chunk.
    pub level: u32,
    /// Its first element: group within the level, or splat.
    pub first: u32,
    pub n: u32,
    /// Centre and radius of a sphere holding every element's centre.
    pub sphere: [f32; 4],
    pub spans: Vec<SectionSpan>,
}

impl BlockEntry {
    /// The one byte range a reader of tier `tier` fetches for this block.
    pub fn tier_range(&self, sections: &[Section], tier: u32) -> (u64, u64) {
        let start = self.spans[0].offset;
        let end = sections
            .iter()
            .zip(&self.spans)
            .filter(|(s, _)| s.tier <= tier)
            .map(|(_, span)| span.offset + span.stored as u64)
            .max()
            .unwrap_or(start);
        (start, end - start)
    }

    /// The one byte range holding the sections `picks` (indices into the
    /// section table): from the first one's start to the last one's end,
    /// with whatever sits between. Empty when `picks` is.
    pub fn range_of(&self, picks: &[usize]) -> (u64, u64) {
        let (Some(&lo), Some(&hi)) = (picks.iter().min(), picks.iter().max()) else {
            return (self.spans[0].offset, 0);
        };
        let start = self.spans[lo].offset;
        (start, self.spans[hi].offset + self.spans[hi].stored as u64 - start)
    }
}

/// The section table's indices `want` needs, in block order.
pub fn wanted_sections(sections: &[Section], x: &ExtraHeader, want: Want) -> Vec<usize> {
    sections.iter().enumerate().filter(|(_, s)| want.needs(s.id, x)).map(|(k, _)| k).collect()
}

fn sphere_of(block: &AthcBlock) -> [f32; 4] {
    if block.n == 0 {
        return [0.0; 4];
    }
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for i in 0..block.n {
        for d in 0..3 {
            lo[d] = lo[d].min(block.positions[i * 4 + d]);
            hi[d] = hi[d].max(block.positions[i * 4 + d]);
        }
    }
    let c = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
    let mut r2 = 0.0f32;
    for i in 0..block.n {
        let p = &block.positions[i * 4..i * 4 + 3];
        r2 = r2.max((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2));
    }
    [c[0], c[1], c[2], r2.sqrt()]
}

/// A v3 file's tables (everything but the data).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct V3Layout {
    pub header: AthcHeader,
    pub extra: ExtraHeader,
    pub sections: Vec<Section>,
    pub blocks: Vec<BlockEntry>,
    pub starts_offset: u64,
}

/// v2 (in memory) to v3 bytes.
pub fn write_v3(file: &AthcFile, compression: u32) -> Result<Vec<u8>> {
    write_v3_as(file, compression, false)
}

/// `write_v3`, or with `legacy` the layout before the three tiers
/// (`legacy_sections_of`), as files written then are (for tests).
pub fn write_v3_as(file: &AthcFile, compression: u32, legacy: bool) -> Result<Vec<u8>> {
    // The v2 writer settles flags, counts and the extra header; the
    // curvature (no v2 holds it) comes back from the cloud as it was.
    let mut v2 = AthcFile::read(&file.write()?)?;
    if file.has_curvature() {
        let per = |b: &AthcBlock| b.curvature.len().checked_div(b.n).unwrap_or(0);
        if file.chunks.iter().chain(file.levels.iter().map(|(_, b)| b)).any(|b| per(b) != 2) {
            bail!(".athc v3: curvature must be two words for every element of every block");
        }
        v2.extra.curvature_words = 2;
        for (to, from) in v2.chunks.iter_mut().zip(&file.chunks) {
            to.curvature = from.curvature.clone();
        }
        for ((_, to), (_, from)) in v2.levels.iter_mut().zip(&file.levels) {
            to.curvature = from.curvature.clone();
        }
    }
    let v2 = v2;
    let (h, x) = (v2.header, v2.extra);
    let sections =
        if legacy { legacy_sections_of(&h, &x, compression) } else { sections_of(&h, &x, compression) };
    let blocks: Vec<(u32, u32, u32, &AthcBlock)> = v2
        .levels
        .iter()
        .map(|(level, b)| (0, *level, 0, b))
        .chain(v2.chunks.iter().enumerate().map(|(c, b)| (1, 0, c as u32 * h.chunk_splats, b)))
        .collect();
    let section_table = V3_HEADER_BYTES as u64;
    let block_index = section_table + (SECTION_ENTRY_BYTES * sections.len()) as u64;
    let entry = BLOCK_ENTRY_HEAD + BLOCK_ENTRY_PER_SECTION * sections.len();
    let starts = aligned(block_index + (entry * blocks.len()) as u64);
    let data_start = aligned(starts + 4 * v2.starts.len() as u64);

    let mut data = Vec::new();
    let mut entries = Vec::new();
    let mut at = data_start;
    for &(kind, level, first, block) in &blocks {
        let mut spans = Vec::new();
        let begin = data.len();
        let mut tx_from = 0;
        for s in &sections {
            let raw = section_bytes(block, s, &x, tx_from);
            if s.id.is_transfer() {
                tx_from += s.words;
            }
            let stored = if compression == COMPRESSION_GZIP { gzip(&raw) } else { raw.clone() };
            spans.push(SectionSpan { offset: at + (data.len() - begin) as u64, stored: stored.len() as u32, raw: raw.len() as u32 });
            data.extend_from_slice(&stored);
        }
        let end = aligned((data.len() - begin) as u64) as usize + begin;
        data.resize(end, 0);
        at = data_start + data.len() as u64;
        entries.push(BlockEntry { kind, level, first, n: block.n as u32, sphere: sphere_of(block), spans });
    }

    let mut out = Vec::with_capacity(data_start as usize + data.len());
    let words = [
        ATH3_MAGIC,
        V3_VERSION,
        h.flags,
        h.count,
        h.chunk_splats,
        h.chunks,
        h.levels,
        h.finest_groups,
        h.rest_per_colour,
        h.sh_words,
        x.transfer_count,
        x.shadow_words,
        x.pbr_words,
        x.lobes_words,
        x.transfer_words,
        sections.len() as u32,
    ];
    put_words(&mut out, &words);
    for f in h.bounds_lo.iter().chain([h.extent].iter()).chain(&h.bounds_min).chain(&h.bounds_max) {
        out.extend_from_slice(&f.to_le_bytes());
    }
    put_words(&mut out, &[blocks.len() as u32, x.curvature_words]);
    for v in [section_table, block_index, starts, data_start] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.resize(V3_HEADER_BYTES, 0);
    for s in &sections {
        put_words(&mut out, &[s.id.code(), s.tier, s.encoding, s.compression, s.words, 0, 0, 0]);
    }
    for e in &entries {
        put_words(&mut out, &[e.kind, e.level, e.first, e.n]);
        for f in e.sphere {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for span in &e.spans {
            out.extend_from_slice(&span.offset.to_le_bytes());
            put_words(&mut out, &[span.stored, span.raw]);
        }
    }
    out.resize(starts as usize, 0);
    put_words(&mut out, &v2.starts);
    out.resize(data_start as usize, 0);
    out.extend_from_slice(&data);
    debug_assert_eq!(out.len() as u64 % PAGE, 0);
    Ok(out)
}

fn u32_at(b: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(b.get(at..at + 4).ok_or_else(|| anyhow!(".athc v3 too short"))?.try_into().unwrap()))
}
fn u64_at(b: &[u8], at: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(b.get(at..at + 8).ok_or_else(|| anyhow!(".athc v3 too short"))?.try_into().unwrap()))
}
fn f32_at(b: &[u8], at: usize) -> Result<f32> {
    Ok(f32::from_bits(u32_at(b, at)?))
}

/// A v3 file's tables, from its first bytes (through the block index).
pub fn parse_v3(bytes: &[u8]) -> Result<V3Layout> {
    if u32_at(bytes, 0)? != ATH3_MAGIC {
        bail!("not an .athc v3 (no ATH3 magic)");
    }
    if u32_at(bytes, 4)? != V3_VERSION {
        bail!("not an .athc v3 (version {})", u32_at(bytes, 4)?);
    }
    let w = |k: usize| u32_at(bytes, 4 * k);
    let f3 = |at: usize| -> Result<[f32; 3]> { Ok([f32_at(bytes, at)?, f32_at(bytes, at + 4)?, f32_at(bytes, at + 8)?]) };
    let header = AthcHeader {
        version: 2,
        flags: w(2)?,
        count: w(3)?,
        chunk_splats: w(4)?,
        chunks: w(5)?,
        levels: w(6)?,
        finest_groups: w(7)?,
        rest_per_colour: w(8)?,
        sh_words: w(9)?,
        bounds_lo: f3(64)?,
        extent: f32_at(bytes, 76)?,
        bounds_min: f3(80)?,
        bounds_max: f3(92)?,
        ..Default::default()
    };
    let extra = ExtraHeader {
        transfer_count: w(10)?,
        shadow_words: w(11)?,
        pbr_words: w(12)?,
        lobes_words: w(13)?,
        transfer_words: w(14)?,
        curvature_words: u32_at(bytes, 108)?,
    };
    if extra.curvature_words != 0 && extra.curvature_words != 2 {
        bail!(".athc v3: curvature of {} words a splat", extra.curvature_words);
    }
    let section_count = w(15)? as usize;
    let block_count = u32_at(bytes, 104)? as usize;
    let section_table = u64_at(bytes, 112)? as usize;
    let block_index = u64_at(bytes, 120)? as usize;
    let starts_offset = u64_at(bytes, 128)?;
    if header.flags & !crate::athc::KNOWN_FLAGS != 0 {
        bail!(".athc v3: unknown flag bits {:#x}", header.flags & !crate::athc::KNOWN_FLAGS);
    }
    if block_count != (header.levels + header.chunks) as usize {
        bail!(".athc v3: {} blocks for {} levels and {} chunks", block_count, header.levels, header.chunks);
    }
    let mut sections = Vec::with_capacity(section_count);
    for k in 0..section_count {
        let at = section_table + SECTION_ENTRY_BYTES * k;
        let code = u32_at(bytes, at)?;
        let id = SectionId::from_code(code)
            .ok_or_else(|| anyhow!(".athc v3: unknown section {:?}", code.to_le_bytes().map(|b| b as char)))?;
        sections.push(Section {
            id,
            tier: u32_at(bytes, at + 4)?,
            encoding: u32_at(bytes, at + 8)?,
            compression: u32_at(bytes, at + 12)?,
            words: u32_at(bytes, at + 16)?,
        });
    }
    let same = |expected: Vec<Section>| {
        expected.len() == sections.len()
            && expected.iter().zip(&sections).all(|(e, s)| e.id == s.id && e.words == s.words && e.tier == s.tier)
    };
    if !same(sections_of(&header, &extra, COMPRESSION_NONE))
        && !same(legacy_sections_of(&header, &extra, COMPRESSION_NONE))
    {
        bail!(".athc v3: its sections are not those its header implies");
    }
    if let Some(s) = sections.iter().find(|s| s.encoding != ENCODING_V2_WORDS || s.compression > COMPRESSION_GZIP) {
        bail!(".athc v3: section {:?} has encoding {} / compression {} this reader does not know", s.id, s.encoding, s.compression);
    }
    let entry = BLOCK_ENTRY_HEAD + BLOCK_ENTRY_PER_SECTION * section_count;
    let mut blocks = Vec::with_capacity(block_count);
    for b in 0..block_count {
        let at = block_index + entry * b;
        let mut spans = Vec::with_capacity(section_count);
        for k in 0..section_count {
            let s = at + BLOCK_ENTRY_HEAD + BLOCK_ENTRY_PER_SECTION * k;
            spans.push(SectionSpan { offset: u64_at(bytes, s)?, stored: u32_at(bytes, s + 8)?, raw: u32_at(bytes, s + 12)? });
        }
        blocks.push(BlockEntry {
            kind: u32_at(bytes, at)?,
            level: u32_at(bytes, at + 4)?,
            first: u32_at(bytes, at + 8)?,
            n: u32_at(bytes, at + 12)?,
            sphere: [f32_at(bytes, at + 16)?, f32_at(bytes, at + 20)?, f32_at(bytes, at + 24)?, f32_at(bytes, at + 28)?],
            spans,
        });
    }
    Ok(V3Layout { header, extra, sections, blocks, starts_offset })
}

/// One block's elements from its sections' bytes (`raw[k]`: section k,
/// uncompressed; sections of a higher tier may be missing, and are then
/// left empty in the block).
pub fn block_from_sections(layout: &V3Layout, n: usize, raw: &[Option<Vec<u8>>]) -> Result<AthcBlock> {
    let (h, x) = (&layout.header, &layout.extra);
    let words = |b: &[u8]| -> Vec<u32> { b.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect() };
    let mut block = AthcBlock { n, ..Default::default() };
    // The transfer's sections, as rows of their words.
    let mut parts: Vec<(usize, Vec<u32>)> = Vec::new();
    for (s, bytes) in layout.sections.iter().zip(raw) {
        let Some(bytes) = bytes else { continue };
        if bytes.len() != n * s.words as usize * 4 {
            bail!(".athc v3: section {:?} holds {} bytes for {} elements", s.id, bytes.len(), n);
        }
        let v = words(bytes);
        match s.id {
            SectionId::Core => {
                block.positions = v[..4 * n].iter().map(|&w| f32::from_bits(w)).collect();
                block.shape = v[4 * n..8 * n].to_vec();
                block.tail = v[8 * n..9 * n].to_vec();
            }
            SectionId::Sh => block.sh = v,
            SectionId::TransferDirect | SectionId::TransferIndirect | SectionId::TransferField => {
                parts.push((s.words as usize, v))
            }
            SectionId::Shadow => block.shadow_bits = v,
            SectionId::Curvature => block.curvature = v,
            SectionId::Material => {
                let mut at = 0;
                let mut take = |on: bool, per: u32| -> Vec<u32> {
                    if !on {
                        return Vec::new();
                    }
                    let out = v[at..at + n * per as usize].to_vec();
                    at += n * per as usize;
                    out
                };
                block.normals = take(h.has(FLAG_NORMALS), 1);
                block.emission = take(h.has(FLAG_EMISSION), 1);
                block.pbr = take(x.pbr_words > 0, x.pbr_words);
                block.lobes = take(x.lobes_words > 0, x.lobes_words);
            }
        }
    }
    // Joined per element; the sections present are a prefix of the three
    // (`Want`), so the row is a prefix of the transfer's.
    if !parts.is_empty() {
        let per: usize = parts.iter().map(|(w, _)| w).sum();
        let mut transfer = Vec::with_capacity(n * per);
        for e in 0..n {
            for (w, v) in &parts {
                transfer.extend_from_slice(&v[e * w..(e + 1) * w]);
            }
        }
        block.transfer = transfer;
    }
    Ok(block)
}

// --- ATHV pages of sections: what the browser pages a v3 file with --------

/// An ATHV page (athc.rs `athv_head`) of kind 2: one block's sections as
/// they are stored, for the decoder to put together.
///
/// ```text
/// 0    "ATHV"   4 kind 2   8 base   12 n
/// 16   the cloud's v2 FileHeader and ExtraHeader (136 bytes; `V3Layout::v2_headers`)
/// 152  decode flags (ATHV_KEEP_LINEAR)
/// 156  transfer values kept (`Want::transfer_values`)
/// 160  m, then m entries: section code, compression, stored bytes, raw bytes
///      the m sections' stored bytes, one after the other
/// ```
///
/// Sections may be any of the block's; the decoder keeps what `Want` they
/// make (material with `MATL`, the relight streams with `SHAD`) and reads the
/// transfer as the form at 156.
pub const ATHV_SECTIONS: u32 = 2;

impl V3Layout {
    /// The 136 bytes of v2 headers (FileHeader, ExtraHeader) of this cloud,
    /// as an ATHV head carries them. Table offsets are 0: a page has none.
    pub fn v2_headers(&self) -> Vec<u8> {
        let mut h = self.header;
        h.version = if h.flags != 0 { 2 } else { 1 };
        let mut out = h.to_bytes().to_vec();
        out.extend_from_slice(&self.extra.to_bytes());
        out
    }
}

/// A kind-2 ATHV page of `n` elements at virtual `base` from stored sections
/// (`(section, stored bytes, raw bytes)`), as the browser builds one.
pub fn athv_sections_page(
    headers: &[u8],
    base: u32,
    n: u32,
    flags: u32,
    transfer_values: u32,
    parts: &[(&Section, &[u8], u32)],
) -> Vec<u8> {
    let mut out = crate::athc::athv_head(ATHV_SECTIONS, base, n, headers);
    out[152..156].copy_from_slice(&flags.to_le_bytes());
    out[156..160].copy_from_slice(&transfer_values.to_le_bytes());
    put_words(&mut out, &[parts.len() as u32]);
    for (s, stored, raw) in parts {
        put_words(&mut out, &[s.id.code(), s.compression, stored.len() as u32, *raw]);
    }
    for (_, stored, _) in parts {
        out.extend_from_slice(stored);
    }
    out
}

/// A kind-2 page's block, with the v2 headers of what it keeps.
pub fn read_sections_page(b: &[u8]) -> Result<(AthcHeader, ExtraHeader, AthcBlock, u32)> {
    let n = u32_at(b, 12)? as usize;
    let (h, x) = crate::athc::parse_headers(b.get(16..152).ok_or_else(|| anyhow!("ATHV page too short"))?)?;
    let flags = u32_at(b, 152)?;
    let transfer_values = u32_at(b, 156)?;
    let m = u32_at(b, 160)? as usize;
    // The page's own sections, words a element from their sizes: either
    // layout of the transfer's sections (sections_of, legacy_sections_of).
    let mut sections = Vec::with_capacity(m);
    let mut raw: Vec<Option<Vec<u8>>> = Vec::with_capacity(m);
    let mut at = 164 + 16 * m;
    let mut present = Vec::new();
    for k in 0..m {
        let e = 164 + 16 * k;
        let code = u32_at(b, e)?;
        let compression = u32_at(b, e + 4)?;
        let stored = u32_at(b, e + 8)? as usize;
        let raw_bytes = u32_at(b, e + 12)? as usize;
        let id = SectionId::from_code(code).ok_or_else(|| anyhow!("ATHV page: unknown section {:#x}", code))?;
        if present.contains(&id) || n == 0 || raw_bytes % (4 * n) != 0 {
            bail!("ATHV page: section {} twice, or not whole words of {} elements", id.name(), n);
        }
        sections.push(Section {
            id,
            tier: id.tier(),
            encoding: ENCODING_V2_WORDS,
            compression: COMPRESSION_NONE,
            words: (raw_bytes / (4 * n)) as u32,
        });
        let data = b.get(at..at + stored).ok_or_else(|| anyhow!("ATHV page shorter than its sections"))?;
        at += stored;
        let bytes = match compression {
            COMPRESSION_NONE => data.to_vec(),
            COMPRESSION_GZIP => gunzip(data)?,
            c => bail!("ATHV page: compression {}", c),
        };
        if bytes.len() != raw_bytes {
            bail!("ATHV page: section {} unpacks to {} bytes, not {}", id.name(), bytes.len(), raw_bytes);
        }
        raw.push(Some(bytes));
        present.push(id);
    }
    for id in [SectionId::Core, SectionId::Sh] {
        if !present.contains(&id) {
            bail!("ATHV page without its {} section", id.name());
        }
    }
    let want = Want {
        material: present.contains(&SectionId::Material),
        transfer_values: if present.contains(&SectionId::Shadow) || x.shadow_words == 0 && present.iter().any(|s| s.is_transfer()) {
            transfer_values
        } else {
            0
        },
    };
    want.check(&x)?;
    let layout = V3Layout { header: h, extra: x, sections, ..Default::default() };
    let mut block = block_from_sections(&layout, n, &raw)?;
    let (rh, mut rx) = reduced_headers(&h, &x, want);
    // A page whose sections leave the curvature out reads without it.
    if block.curvature.is_empty() {
        rx.curvature_words = 0;
    }
    // The transfer as the form kept: the prefix of each row.
    if rx.transfer_words > 0 {
        let have = block.transfer.len() / n.max(1);
        let keep = rx.transfer_words as usize;
        if have < keep {
            bail!("ATHV page: transfer rows of {} words for a form of {}", have, keep);
        }
        if have != keep {
            block.transfer = (0..n).flat_map(|e| block.transfer[e * have..e * have + keep].to_vec()).collect();
        }
    } else {
        block.transfer.clear();
        block.shadow_bits.clear();
        block.curvature.clear();
    }
    if !want.material {
        block.normals.clear();
        block.emission.clear();
        block.pbr.clear();
        block.lobes.clear();
    }
    Ok((rh, rx, block, flags))
}

/// The paged virtual tree of a v3 cloud and its merged pages (kind 0, as
/// `athv_merged_pages` makes them for v2), from its tables (through the
/// starts) and its levels' blocks as kind-2 pages (coarsest first). The
/// pages carry the v2 headers of what those level pages keep, also returned.
pub fn athv_merged_pages_v3(
    tables: &[u8],
    levels: &[&[u8]],
) -> Result<(crate::athc::VirtualTree, Vec<Vec<u8>>, Vec<u8>)> {
    let layout = parse_v3(tables)?;
    let level_blocks: Vec<&BlockEntry> = layout.blocks.iter().filter(|b| b.kind == 0).collect();
    if levels.len() != level_blocks.len() {
        bail!(".athc v3: {} level pages for {} levels", levels.len(), level_blocks.len());
    }
    let mut header = None;
    let mut blocks = Vec::new();
    for (page, entry) in levels.iter().zip(&level_blocks) {
        if u32_at(page, 0)? != crate::athc::ATHV_MAGIC || u32_at(page, 4)? != ATHV_SECTIONS {
            bail!(".athc v3: a level page is not an ATHV page of sections");
        }
        let (h, x, block, _) = read_sections_page(page)?;
        if block.n != entry.n as usize {
            bail!(".athc v3: level {} page of {} groups, not {}", entry.level, block.n, entry.n);
        }
        if header.is_some_and(|hx| hx != (h, x)) {
            bail!(".athc v3: level pages keep different streams");
        }
        header = Some((h, x));
        blocks.push((entry.level, block));
    }
    let (h, x) = header.unwrap_or((layout.header, layout.extra));
    let at = layout.starts_offset as usize;
    let starts = (0..layout.header.finest_groups as usize)
        .map(|g| u32_at(tables, at + 4 * g))
        .collect::<Result<Vec<_>>>()?;
    let file = AthcFile { header: h, extra: x, levels: blocks, starts, chunks: Vec::new() };
    let mut headers = h.to_bytes().to_vec();
    headers.extend_from_slice(&x.to_bytes());
    let (tree, pages) = crate::athc::merged_pages_of(&file, &headers)?;
    Ok((tree, pages, headers))
}

/// The bytes of a v3 file a pager reads before any block: the header, the
/// section table, the block index and the starts.
pub fn tables_bytes(bytes: &[u8]) -> Result<u64> {
    if u32_at(bytes, 0)? != ATH3_MAGIC {
        bail!("not an .athc v3 (no ATH3 magic)");
    }
    u64_at(bytes, 136)
}

/// v3 bytes (the whole file) back to the v2 cloud.
pub fn read_v3(bytes: &[u8]) -> Result<AthcFile> {
    let layout = parse_v3(bytes)?;
    let mut levels = Vec::new();
    let mut chunks = Vec::new();
    for b in &layout.blocks {
        let raw = layout
            .sections
            .iter()
            .zip(&b.spans)
            .map(|(s, span)| {
                let stored = bytes
                    .get(span.offset as usize..span.offset as usize + span.stored as usize)
                    .ok_or_else(|| anyhow!(".athc v3: a section past the end"))?;
                let raw = if s.compression == COMPRESSION_GZIP { gunzip(stored)? } else { stored.to_vec() };
                if raw.len() != span.raw as usize {
                    bail!(".athc v3: section {:?} unpacks to {} bytes, not {}", s.id, raw.len(), span.raw);
                }
                Ok(Some(raw))
            })
            .collect::<Result<Vec<_>>>()?;
        let block = block_from_sections(&layout, b.n as usize, &raw)?;
        if b.kind == 0 {
            levels.push((b.level, block));
        } else {
            chunks.push(block);
        }
    }
    let at = layout.starts_offset as usize;
    let starts = (0..layout.header.finest_groups as usize)
        .map(|g| u32_at(bytes, at + 4 * g))
        .collect::<Result<Vec<_>>>()?;
    Ok(AthcFile { header: layout.header, extra: layout.extra, levels, starts, chunks })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_CARDS: &[u8] = include_bytes!("../../../test/fixtures/athc/two_cards.athc");
    const EVERY: &[u8] = include_bytes!("../../../test/fixtures/athc/every_stream.athc");

    #[test]
    fn v2_to_v3_and_back_is_the_same_file() {
        for (name, v2) in [("two_cards", TWO_CARDS), ("every_stream", EVERY)] {
            for compression in [COMPRESSION_NONE, COMPRESSION_GZIP] {
                let file = AthcFile::read(v2).unwrap();
                let v3 = write_v3(&file, compression).unwrap();
                assert_eq!(&v3[..4], b"ATH3");
                assert_eq!(v3.len() as u64 % PAGE, 0);
                let back = read_v3(&v3).unwrap().write().unwrap();
                assert!(back == v2, "{name} (compression {compression}) did not come back byte for byte");
            }
        }
    }

    #[test]
    fn a_tier_is_one_range_of_a_block() {
        let file = AthcFile::read(EVERY).unwrap();
        let v3 = write_v3(&file, COMPRESSION_NONE).unwrap();
        let layout = parse_v3(&v3).unwrap();
        let ids: Vec<SectionId> = layout.sections.iter().map(|s| s.id).collect();
        use SectionId::*;
        assert_eq!(ids, [Core, Sh, Material, Shadow, TransferDirect, TransferIndirect, TransferField]);
        // 112 values: 16 direct (8 words), 48 indirect (24), 48 of field (24).
        assert_eq!(layout.sections.iter().map(|s| s.words).collect::<Vec<_>>(), [9, 23, 6, 8, 8, 24, 24]);
        assert_eq!(layout.sections.iter().map(|s| s.tier).collect::<Vec<_>>(), [1, 1, 2, 3, 3, 3, 3]);
        for b in &layout.blocks {
            assert_eq!(b.spans[0].offset % PAGE, 0, "every block on its own page");
            let (start, len) = b.tier_range(&layout.sections, 1);
            assert_eq!(start, b.spans[0].offset);
            assert_eq!(len, (b.n as u64) * (9 + 23) * 4, "tier 1: core and harmonics only");
            let (_, material) = b.tier_range(&layout.sections, 2);
            assert_eq!(material, (b.n as u64) * (9 + 23 + 6) * 4);
            let (_, all) = b.tier_range(&layout.sections, 3);
            assert_eq!(all, (b.n as u64) * (9 + 23 + 6 + 8 + 8 + 24 + 24) * 4);
            // A tier-1 reader decodes the block from its prefix alone.
            let prefix = &v3[start as usize..(start + len) as usize];
            let mut raw: Vec<Option<Vec<u8>>> = vec![None; layout.sections.len()];
            let mut at = 0;
            for k in 0..2 {
                let n = b.spans[k].raw as usize;
                raw[k] = Some(prefix[at..at + n].to_vec());
                at += n;
            }
            let block = block_from_sections(&layout, b.n as usize, &raw).unwrap();
            assert_eq!(block.positions.len(), 4 * b.n as usize);
            assert!(block.transfer.is_empty() && block.pbr.is_empty());
            // The relight streams of any form are one range after the material.
            for form in transfer_forms(112) {
                let picks = wanted_sections(&layout.sections, &layout.extra, Want { material: false, transfer_values: form });
                let relight: Vec<usize> = picks.into_iter().filter(|&k| layout.sections[k].tier == 3).collect();
                let (from, bytes) = b.range_of(&relight);
                assert_eq!(from, b.spans[3].offset);
                assert_eq!(bytes, (b.n as u64) * (8 + form.div_ceil(2) as u64) * 4, "form {form}");
            }
        }
        // Chunks know where they are.
        let chunk = layout.blocks.iter().find(|b| b.kind == 1).unwrap();
        assert!(chunk.sphere[3] > 0.0);
    }

    #[test]
    fn transfer_forms_are_layout_prefixes() {
        assert_eq!(transfer_forms(112), [16, 64, 112]);
        assert_eq!(transfer_forms(84), [9, 36, 84]);
        assert_eq!(transfer_forms(64), [16, 64]);
        assert_eq!(transfer_forms(36), [9, 36]);
        assert_eq!(transfer_forms(16), [16]);
        assert_eq!(transfer_forms(10), [10]);
        assert_eq!(transfer_split_words(112), [8, 32, 56]);
        assert_eq!(transfer_split_words(84), [5, 18, 42]);
        assert_eq!(transfer_split_words(36), [5, 18, 18]);
        assert_eq!(transfer_split_words(10), [5, 5, 5]);
    }

    /// Each stored section of block `b`, as a kind-2 page part.
    fn parts<'a>(v3: &'a [u8], layout: &'a V3Layout, b: &BlockEntry, picks: &[usize]) -> Vec<(&'a Section, &'a [u8], u32)> {
        picks
            .iter()
            .map(|&k| {
                let span = b.spans[k];
                (&layout.sections[k], &v3[span.offset as usize..span.offset as usize + span.stored as usize], span.raw)
            })
            .collect()
    }

    #[test]
    fn a_page_of_sections_is_the_chunk_it_keeps() {
        let file = AthcFile::read(EVERY).unwrap();
        for compression in [COMPRESSION_NONE, COMPRESSION_GZIP] {
            let v3 = write_v3(&file, compression).unwrap();
            let layout = parse_v3(&v3).unwrap();
            let headers = layout.v2_headers();
            let (b, chunk) = (layout.blocks.iter().find(|b| b.kind == 1).unwrap(), &file.chunks[0]);
            for want in [
                Want::default(),
                Want { material: true, transfer_values: 0 },
                Want { material: false, transfer_values: 16 },
                Want { material: true, transfer_values: 64 },
                Want::all(&layout.extra),
            ] {
                let picks = wanted_sections(&layout.sections, &layout.extra, want);
                let page = athv_sections_page(&headers, 65536, b.n, 0, want.transfer_values, &parts(&v3, &layout, b, &picks));
                let (h, x, block, _) = read_sections_page(&page).unwrap();
                assert_eq!((h, x), reduced_headers(&layout.header, &layout.extra, want), "{want:?}");
                assert_eq!(block.positions, chunk.positions);
                assert_eq!(block.sh, chunk.sh);
                assert_eq!(block.tail, chunk.tail);
                assert_eq!(!block.pbr.is_empty(), want.material);
                if want.material {
                    assert_eq!((&block.normals, &block.lobes), (&chunk.normals, &chunk.lobes));
                }
                let keep = want.transfer_values.div_ceil(2) as usize;
                let tw = layout.extra.transfer_words as usize;
                let expect: Vec<u32> = (0..chunk.n).flat_map(|e| chunk.transfer[e * tw..e * tw + keep].to_vec()).collect();
                assert_eq!(block.transfer, expect, "{want:?}");
                assert_eq!(block.shadow_bits.is_empty(), want.transfer_values == 0);
                // The decoder's v2 reading of the same block agrees.
                let mut v2 = Vec::new();
                block.write(&mut v2);
                assert_eq!(crate::athc::AthcBlock::read(&v2, block.n, &h, &x).unwrap(), block);
            }
            // Without its core it is not a page.
            let page = athv_sections_page(&headers, 0, b.n, 0, 0, &parts(&v3, &layout, b, &[1]));
            assert!(read_sections_page(&page).is_err());
        }
    }

    #[test]
    fn files_of_the_first_layout_still_read_and_page() {
        let file = AthcFile::read(EVERY).unwrap();
        let v3 = write_v3_as(&file, COMPRESSION_GZIP, true).unwrap();
        let layout = parse_v3(&v3).unwrap();
        use SectionId::*;
        let ids: Vec<SectionId> = layout.sections.iter().map(|s| s.id).collect();
        assert_eq!(ids, [Core, Sh, TransferDirect, TransferIndirect, Shadow, Material]);
        assert!(read_v3(&v3).unwrap().write().unwrap() == EVERY);
        let headers = layout.v2_headers();
        let (b, chunk) = (layout.blocks.iter().find(|b| b.kind == 1).unwrap(), &file.chunks[0]);
        for values in [16, 64, 112] {
            let want = Want { material: true, transfer_values: values };
            let picks = wanted_sections(&layout.sections, &layout.extra, want);
            let page = athv_sections_page(&headers, 0, b.n, 0, values, &parts(&v3, &layout, b, &picks));
            let (_, x, block, _) = read_sections_page(&page).unwrap();
            assert_eq!(x.transfer_count, values);
            let keep = values.div_ceil(2) as usize;
            let expect: Vec<u32> = (0..chunk.n).flat_map(|e| chunk.transfer[e * 56..e * 56 + keep].to_vec()).collect();
            assert_eq!(block.transfer, expect);
            assert_eq!(block.pbr, chunk.pbr);
        }
    }

    #[test]
    fn merged_pages_from_levels_match_v2() {
        for v2 in [TWO_CARDS, EVERY] {
            let file = AthcFile::read(v2).unwrap();
            let layout2 = crate::athc::AthcLayout::parse(v2, v2.len() as u64).unwrap();
            let (tree2, pages2) = crate::athc::athv_merged_pages(&v2[..layout2.levels_end() as usize], v2.len() as u64).unwrap();
            let v3 = write_v3(&file, COMPRESSION_GZIP).unwrap();
            let layout = parse_v3(&v3).unwrap();
            let tables = &v3[..tables_bytes(&v3).unwrap() as usize];
            let headers = layout.v2_headers();
            let all = Want::all(&layout.extra);
            let picks = wanted_sections(&layout.sections, &layout.extra, all);
            let levels: Vec<Vec<u8>> = layout
                .blocks
                .iter()
                .filter(|b| b.kind == 0)
                .map(|b| athv_sections_page(&headers, 0, b.n, 0, all.transfer_values, &parts(&v3, &layout, b, &picks)))
                .collect();
            let refs: Vec<&[u8]> = levels.iter().map(|v| &v[..]).collect();
            let (tree, pages, _) = athv_merged_pages_v3(tables, &refs).unwrap();
            assert_eq!((tree.merged, tree.splat_base), (tree2.merged, tree2.splat_base));
            assert_eq!(pages.len(), pages2.len());
            for (a, b) in pages.iter().zip(&pages2) {
                // The same page but for the head's table offsets.
                assert_eq!(a[..16], b[..16]);
                assert_eq!(a[ATHV_HEAD_END..], b[ATHV_HEAD_END..]);
            }
            // Tier 1 levels: merged pages of the splats alone.
            let core = Want::default();
            let picks = wanted_sections(&layout.sections, &layout.extra, core);
            let levels: Vec<Vec<u8>> = layout
                .blocks
                .iter()
                .filter(|b| b.kind == 0)
                .map(|b| athv_sections_page(&headers, 0, b.n, 0, 0, &parts(&v3, &layout, b, &picks)))
                .collect();
            let refs: Vec<&[u8]> = levels.iter().map(|v| &v[..]).collect();
            let (_, pages, headers) = athv_merged_pages_v3(tables, &refs).unwrap();
            let (h, x) = crate::athc::parse_headers(&headers).unwrap();
            assert_eq!(h.flags & !crate::athc::FLAG_LINEAR, 0);
            assert_eq!(x, ExtraHeader::default());
            assert!(pages[0].len() < pages2[0].len() || layout.sections.len() == 2);
        }
    }

    const ATHV_HEAD_END: usize = crate::athc::ATHV_HEAD;

    #[test]
    fn gzip_is_what_a_browser_reads() {
        let data: Vec<u8> = (0..10_000u32).flat_map(|v| (v % 251).to_le_bytes()).collect();
        let packed = gzip(&data);
        assert_eq!(&packed[..3], &[0x1f, 0x8b, 8]);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(gunzip(&packed).unwrap(), data);
    }
}
