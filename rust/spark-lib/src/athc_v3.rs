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
use miniz_oxide::inflate::decompress_to_vec;

use crate::athc::{
    aligned, AthcBlock, AthcFile, AthcHeader, ExtraHeader, FLAG_EMISSION, FLAG_NORMALS, PAGE,
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

/// Sections, in the order they sit in every block (tier order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectionId {
    /// positions (f32 x4), shape (u32 x4), tail (u32): 36 bytes.
    Core,
    /// The rest harmonics, `shWords` words.
    Sh,
    /// The transfer's first ceil(direct / 2) words.
    TransferDirect,
    /// The rest of the transfer's words (indirect half, reflected field).
    TransferIndirect,
    /// The open-direction bits.
    Shadow,
    /// normals, emission, pbr, lobes: those the flags say, in that order.
    Material,
}

impl SectionId {
    pub fn code(self) -> u32 {
        u32::from_le_bytes(*match self {
            Self::Core => b"CORE",
            Self::Sh => b"SHRS",
            Self::TransferDirect => b"TXDI",
            Self::TransferIndirect => b"TXIN",
            Self::Shadow => b"SHAD",
            Self::Material => b"MATL",
        })
    }

    pub fn from_code(code: u32) -> Option<Self> {
        [Self::Core, Self::Sh, Self::TransferDirect, Self::TransferIndirect, Self::Shadow, Self::Material]
            .into_iter()
            .find(|s| s.code() == code)
    }

    /// The tier that first needs it: 1 the splats as captured, 2 relit.
    pub fn tier(self) -> u32 {
        match self {
            Self::Core | Self::Sh => 1,
            _ => 2,
        }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Section {
    pub id: SectionId,
    pub tier: u32,
    pub encoding: u32,
    pub compression: u32,
    pub words: u32,
}

/// The sections a cloud with these headers has, in block order.
pub fn sections_of(h: &AthcHeader, x: &ExtraHeader, compression: u32) -> Vec<Section> {
    let direct = transfer_direct_values(x.transfer_count).div_ceil(2).min(x.transfer_words);
    let material = (if h.has(FLAG_NORMALS) { 1 } else { 0 })
        + (if h.has(FLAG_EMISSION) { 1 } else { 0 })
        + x.pbr_words
        + x.lobes_words;
    [
        (SectionId::Core, 9),
        (SectionId::Sh, h.sh_words),
        (SectionId::TransferDirect, direct),
        (SectionId::TransferIndirect, x.transfer_words - direct),
        (SectionId::Shadow, x.shadow_words),
        (SectionId::Material, material),
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
fn section_bytes(block: &AthcBlock, s: &Section, x: &ExtraHeader) -> Vec<u8> {
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
        SectionId::TransferDirect | SectionId::TransferIndirect => {
            let w = x.transfer_words as usize;
            let d = transfer_direct_values(x.transfer_count).div_ceil(2).min(x.transfer_words) as usize;
            let (from, to) = if s.id == SectionId::TransferDirect { (0, d) } else { (d, w) };
            put_words(&mut out, &columns(&block.transfer, block.n, w, from, to));
        }
        SectionId::Shadow => put_words(&mut out, &block.shadow_bits),
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SectionSpan {
    pub offset: u64,
    pub stored: u32,
    pub raw: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
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
#[derive(Clone, Debug, Default, PartialEq)]
pub struct V3Layout {
    pub header: AthcHeader,
    pub extra: ExtraHeader,
    pub sections: Vec<Section>,
    pub blocks: Vec<BlockEntry>,
    pub starts_offset: u64,
}

/// v2 (in memory) to v3 bytes.
pub fn write_v3(file: &AthcFile, compression: u32) -> Result<Vec<u8>> {
    // The v2 writer settles flags, counts and the extra header.
    let v2 = AthcFile::read(&file.write()?)?;
    let (h, x) = (v2.header, v2.extra);
    let sections = sections_of(&h, &x, compression);
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
        for s in &sections {
            let raw = section_bytes(block, s, &x);
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
    put_words(&mut out, &[blocks.len() as u32, 0]);
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
    };
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
    let expected = sections_of(&header, &extra, COMPRESSION_NONE);
    if expected.len() != sections.len()
        || expected.iter().zip(&sections).any(|(e, s)| e.id != s.id || e.words != s.words)
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
    let mut direct = Vec::new();
    let mut indirect = Vec::new();
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
            SectionId::TransferDirect => direct = v,
            SectionId::TransferIndirect => indirect = v,
            SectionId::Shadow => block.shadow_bits = v,
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
    if !direct.is_empty() {
        let d = direct.len() / n.max(1);
        let i = indirect.len() / n.max(1);
        block.transfer = (0..n)
            .flat_map(|e| direct[e * d..(e + 1) * d].iter().chain(&indirect[e * i..(e + 1) * i]).copied().collect::<Vec<_>>())
            .collect();
    }
    Ok(block)
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
        assert_eq!(ids, [Core, Sh, TransferDirect, TransferIndirect, Shadow, Material]);
        // 112 values: 16 direct (8 words), then 48 words of indirect and field.
        assert_eq!(layout.sections.iter().map(|s| s.words).collect::<Vec<_>>(), [9, 23, 8, 48, 8, 6]);
        for b in &layout.blocks {
            assert_eq!(b.spans[0].offset % PAGE, 0, "every block on its own page");
            let (start, len) = b.tier_range(&layout.sections, 1);
            assert_eq!(start, b.spans[0].offset);
            assert_eq!(len, (b.n as u64) * (9 + 23) * 4, "tier 1: core and harmonics only");
            let (_, all) = b.tier_range(&layout.sections, 2);
            assert_eq!(all, (b.n as u64) * (9 + 23 + 8 + 48 + 8 + 6) * 4);
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
        }
        // Chunks know where they are.
        let chunk = layout.blocks.iter().find(|b| b.kind == 1).unwrap();
        assert!(chunk.sphere[3] > 0.0);
    }

    #[test]
    fn gzip_is_what_a_browser_reads() {
        let data: Vec<u8> = (0..10_000u32).flat_map(|v| (v % 251).to_le_bytes()).collect();
        let packed = gzip(&data);
        assert_eq!(&packed[..3], &[0x1f, 0x8b, 8]);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(gunzip(&packed).unwrap(), data);
    }
}
