//! `.athl`: the binary half of athenea's light sidecar (proposals 066 §2.4,
//! 069, 076; layout in docs/docs/athl.md).
//!
//! What a cloud's light groups add to it, baked: per group, sparse RGB f16
//! layers over the cloud's splats (the light each splat sends on from the
//! group after bouncing, the emitter's own glow, a view-dependent reflected
//! field), the emitter's polygons for the direct term (LTC, 060) and the
//! lens profile it shines through (076). Every value is per unit of the
//! group's radiance, so the runtime weight `w_k` (the `.lights.usda` half)
//! is the lamp's real radiance in every term at once (062 §7.1).
//!
//! The splat index space is the cloud's page-aligned virtual order
//! (`athc::VirtualTree::of_file(file, true)`): the LoD's merged nodes from
//! 0, the splats from `splat_base` (a multiple of 65 536). Chunks of 65 536
//! are the pager's pages, so the layers of one page, every group, are one
//! contiguous byte range. Inside a chunk a layer is sparse by blocks of 256
//! splats: the ids of the blocks a group reaches, then those blocks dense.

use anyhow::{anyhow, bail, ensure, Result};
use half::f16;
use serde::Serialize;

use crate::athc::{high_half, low_half, AthcFile, VirtualTree, PAGE_SPLATS};

pub const ATHL_MAGIC: u32 = u32::from_le_bytes(*b"ATHL");
pub const ATHL_VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 128;
pub const SECTION_BYTES: usize = 40;
pub const GROUP_BYTES: usize = 128;
pub const POLYGON_BYTES: usize = 128;
pub const PROFILE_HEAD_BYTES: usize = 48;
pub const LAYER_HEAD_BYTES: usize = 16;
pub const CHUNK_SPLATS: u32 = PAGE_SPLATS;
pub const BLOCK_SPLATS: u32 = 256;
pub const BLOCKS_PER_CHUNK: u32 = CHUNK_SPLATS / BLOCK_SPLATS;
pub const MAX_GROUPS: usize = 16;
pub const MAX_POLYGON_VERTICES: usize = 8;
pub const GROUP_NAME_BYTES: usize = 32;

pub const TAG_GROUPS: u32 = u32::from_le_bytes(*b"GRPS");
pub const TAG_POLYGONS: u32 = u32::from_le_bytes(*b"POLY");
pub const TAG_PROFILE: u32 = u32::from_le_bytes(*b"PROF");
pub const TAG_LAYER: u32 = u32::from_le_bytes(*b"LAYR");

/// Layer kinds.
/// The light the group sends on from each splat after bouncing (062's
/// indirect layer); with no polygons, all of the group's light.
pub const KIND_INDIRECT: u16 = 0;
/// The emitter's own visible radiance (its splats, behind their lens).
pub const KIND_EMISSION: u16 = 1;
/// A view-dependent field: the group's light leaving a splat towards the
/// eye, degree-2 real harmonics of the view direction, rgb a coefficient.
pub const KIND_FIELD: u16 = 2;
pub const FIELD_COMPONENTS: u32 = 27;

/// Group flags.
/// The polygons light both sides (else only the side their winding faces).
pub const GROUP_TWO_SIDED: u32 = 1;

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AthlGroup {
    pub name: String,
    pub flags: u32,
    pub polygon_first: u32,
    pub polygon_count: u32,
    /// Index of its lens profile, or -1.
    pub profile: i32,
    /// 076 (a): the lens's mean transmittance.
    pub tint: [f32; 3],
    /// The lamp's frame (076): origin, then axes x, y and z (the beam), in
    /// the cloud's object space.
    pub origin: [f32; 3],
    pub axes: [[f32; 3]; 3],
    /// The radiance the layers were baked for (1: per unit).
    pub radiance: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AthlPolygon {
    pub group: u32,
    /// Relative radiance of this polygon within the group.
    pub radiance: [f32; 3],
    /// 3 to 8 vertices, object space; the front face is counter-clockwise.
    pub vertices: Vec<[f32; 3]>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AthlProfile {
    pub width: u32,
    pub height: u32,
    /// The sector the map covers, radians: longitude about the beam (from
    /// +z towards +x) and latitude (towards +y).
    pub lon: [f32; 2],
    pub lat: [f32; 2],
    /// Outside the sector.
    pub outside: [f32; 3],
    /// rgb a texel, row 0 at lat[0], as f16 bits (3 a texel).
    #[serde(skip)]
    pub texels: Vec<u16>,
}

/// One group's layer of one kind over one chunk: the blocks it reaches.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AthlLayer {
    pub group: u16,
    pub kind: u16,
    pub chunk: u32,
    /// Values a splat: 3 (rgb) or FIELD_COMPONENTS.
    pub components: u32,
    /// Block ids within the chunk (0..256), increasing.
    pub blocks: Vec<u16>,
    /// f16 bits, blocks.len() x 256 x components, block after block.
    #[serde(skip)]
    pub data: Vec<u16>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AthlFile {
    pub flags: u32,
    /// Elements of the virtual order: splat_base + splat_count.
    pub element_count: u32,
    pub merged: u32,
    pub splat_base: u32,
    pub splat_count: u32,
    /// `cloud_hash` of the .athc it was baked for (0: not checked).
    pub cloud_hash: u64,
    /// Whatever identifies the bake (its parameters, 0 if none).
    pub bake_hash: u64,
    pub groups: Vec<AthlGroup>,
    pub polygons: Vec<AthlPolygon>,
    pub profiles: Vec<AthlProfile>,
    pub layers: Vec<AthlLayer>,
}

/// A section table entry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AthlSection {
    pub tag: u32,
    pub group: u16,
    pub kind: u16,
    pub chunk: u32,
    pub count: u32,
    pub offset: u64,
    pub bytes: u64,
}

/// FNV-1a, 64 bits, of the cloud's first 4096 bytes (its header page, which
/// holds its counts, bounds and table offsets; a v3 file's header and
/// directory start). Cheap to check in a browser that has fetched them.
pub fn cloud_hash(athc: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in &athc[..athc.len().min(4096)] {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u16s(out: &mut Vec<u8>, v: &[u16]) {
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    if v.len() % 2 == 1 {
        out.extend_from_slice(&[0, 0]);
    }
}
fn pad16(out: &mut Vec<u8>) {
    out.resize(out.len().div_ceil(16) * 16, 0);
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap())
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
fn f32x3_at(b: &[u8], at: usize) -> [f32; 3] {
    [f32_at(b, at), f32_at(b, at + 4), f32_at(b, at + 8)]
}
fn u16s_at(b: &[u8], at: usize, n: usize) -> Vec<u16> {
    (0..n).map(|i| u16_at(b, at + 2 * i)).collect()
}

/// The header fields, as `AthlFile` holds them, and the section table.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AthlHeader {
    pub version: u32,
    pub flags: u32,
    pub group_count: u32,
    pub element_count: u32,
    pub merged: u32,
    pub splat_base: u32,
    pub splat_count: u32,
    pub chunk_splats: u32,
    pub block_splats: u32,
    /// As hex: JSON numbers would lose bits.
    pub cloud_hash: String,
    pub bake_hash: String,
    pub sections: Vec<AthlSection>,
}

impl AthlHeader {
    /// Bytes from the start to the end of the section table.
    pub fn prefix_bytes(bytes: &[u8]) -> Result<u64> {
        ensure!(bytes.len() >= HEADER_BYTES, ".athl shorter than its header");
        ensure!(u32_at(bytes, 0) == ATHL_MAGIC, "not a .athl (magic)");
        Ok(u64_at(bytes, 48) + SECTION_BYTES as u64 * u32_at(bytes, 40) as u64)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let need = Self::prefix_bytes(bytes)?;
        ensure!(bytes.len() as u64 >= need, ".athl prefix needs {} bytes, has {}", need, bytes.len());
        let version = u32_at(bytes, 4);
        ensure!(version == ATHL_VERSION, ".athl version {} (this reads {})", version, ATHL_VERSION);
        let count = u32_at(bytes, 40) as usize;
        let table = u64_at(bytes, 48) as usize;
        let sections = (0..count)
            .map(|i| {
                let at = table + i * SECTION_BYTES;
                AthlSection {
                    tag: u32_at(bytes, at),
                    group: u16_at(bytes, at + 4),
                    kind: u16_at(bytes, at + 6),
                    chunk: u32_at(bytes, at + 8),
                    count: u32_at(bytes, at + 12),
                    offset: u64_at(bytes, at + 16),
                    bytes: u64_at(bytes, at + 24),
                }
            })
            .collect();
        Ok(Self {
            version,
            flags: u32_at(bytes, 8),
            group_count: u32_at(bytes, 12),
            element_count: u32_at(bytes, 16),
            merged: u32_at(bytes, 20),
            splat_base: u32_at(bytes, 24),
            splat_count: u32_at(bytes, 28),
            chunk_splats: u32_at(bytes, 32),
            block_splats: u32_at(bytes, 36),
            cloud_hash: format!("{:016x}", u64_at(bytes, 56)),
            bake_hash: format!("{:016x}", u64_at(bytes, 64)),
            sections,
        })
    }
}

impl AthlLayer {
    fn byte_len(&self) -> usize {
        LAYER_HEAD_BYTES + self.blocks.len().div_ceil(2) * 4 + self.data.len().div_ceil(2) * 4
    }

    fn write(&self, out: &mut Vec<u8>) {
        put_u32(out, self.components);
        put_u32(out, self.blocks.len() as u32);
        put_u32(out, self.group as u32 | (self.kind as u32) << 16);
        put_u32(out, self.chunk);
        put_u16s(out, &self.blocks);
        put_u16s(out, &self.data);
    }

    /// A LAYR section's bytes (what one Range request of it returns).
    pub fn read(b: &[u8]) -> Result<Self> {
        ensure!(b.len() >= LAYER_HEAD_BYTES, "LAYR section shorter than its head");
        let components = u32_at(b, 0);
        let n = u32_at(b, 4) as usize;
        let gk = u32_at(b, 8);
        let chunk = u32_at(b, 12);
        ensure!(components > 0 && components <= 64, "LAYR with {} components", components);
        ensure!(n <= BLOCKS_PER_CHUNK as usize, "LAYR with {} blocks", n);
        let values = n * BLOCK_SPLATS as usize * components as usize;
        let at = LAYER_HEAD_BYTES + n.div_ceil(2) * 4;
        ensure!(b.len() >= at + values * 2, "LAYR shorter than its {} blocks", n);
        let blocks = u16s_at(b, LAYER_HEAD_BYTES, n);
        ensure!(blocks.windows(2).all(|w| w[0] < w[1]), "LAYR blocks not increasing");
        ensure!(blocks.last().is_none_or(|&x| (x as u32) < BLOCKS_PER_CHUNK), "LAYR block id out of the chunk");
        Ok(Self {
            group: (gk & 0xffff) as u16,
            kind: (gk >> 16) as u16,
            chunk,
            components,
            blocks,
            data: u16s_at(b, at, values),
        })
    }

    /// The value of component `c` of virtual element `index` (0 outside).
    pub fn value(&self, index: u32, c: u32) -> f32 {
        if index / CHUNK_SPLATS != self.chunk {
            return 0.0;
        }
        let local = index % CHUNK_SPLATS;
        let block = (local / BLOCK_SPLATS) as u16;
        match self.blocks.binary_search(&block) {
            Ok(k) => {
                let at = (k * BLOCK_SPLATS as usize + (local % BLOCK_SPLATS) as usize) * self.components as usize
                    + c as usize;
                f16::from_bits(self.data[at]).to_f32()
            }
            Err(_) => 0.0,
        }
    }
}

impl AthlFile {
    pub fn write(&self) -> Result<Vec<u8>> {
        ensure!(self.groups.len() <= MAX_GROUPS, ".athl holds at most {} groups", MAX_GROUPS);
        for g in &self.groups {
            ensure!(g.name.len() < GROUP_NAME_BYTES, "group name '{}' too long", g.name);
            ensure!(
                (g.polygon_first + g.polygon_count) as usize <= self.polygons.len(),
                "group '{}' polygons out of range",
                g.name
            );
            ensure!(g.profile < self.profiles.len() as i32, "group '{}' profile out of range", g.name);
        }
        for p in &self.polygons {
            ensure!(
                (3..=MAX_POLYGON_VERTICES).contains(&p.vertices.len()),
                "a polygon has {} vertices",
                p.vertices.len()
            );
        }
        for p in &self.profiles {
            ensure!(p.texels.len() == 3 * (p.width * p.height) as usize, "profile texels");
        }
        let mut layers: Vec<&AthlLayer> = self.layers.iter().collect();
        // Chunk first: one page's layers, every group, are one range.
        layers.sort_by_key(|l| (l.chunk, l.group, l.kind));
        for l in &layers {
            ensure!((l.group as usize) < self.groups.len(), "layer of group {}", l.group);
            ensure!(
                l.data.len() == l.blocks.len() * BLOCK_SPLATS as usize * l.components as usize,
                "layer data size"
            );
        }
        let sections = 2 + self.profiles.len() + layers.len();
        let mut body = Vec::new();
        let mut table: Vec<AthlSection> = Vec::new();
        let start = HEADER_BYTES + sections * SECTION_BYTES;
        let start = start.div_ceil(16) * 16;
        #[allow(clippy::too_many_arguments)]
        fn open(table: &mut Vec<AthlSection>, start: usize, body: &[u8], tag: u32, group: u16, kind: u16, chunk: u32, count: u32) {
            table.push(AthlSection { tag, group, kind, chunk, count, offset: (start + body.len()) as u64, bytes: 0 });
        }
        fn close(table: &mut [AthlSection], start: usize, body: &[u8]) {
            let last = table.last_mut().unwrap();
            last.bytes = (start + body.len()) as u64 - last.offset;
        }
        open(&mut table, start, &body, TAG_GROUPS, 0, 0, 0, self.groups.len() as u32);
        for g in &self.groups {
            let at = body.len();
            body.extend_from_slice(g.name.as_bytes());
            body.resize(at + GROUP_NAME_BYTES, 0);
            put_u32(&mut body, g.flags);
            put_u32(&mut body, g.polygon_first);
            put_u32(&mut body, g.polygon_count);
            put_u32(&mut body, g.profile as u32);
            for v in g.tint.iter().chain(&g.origin).chain(g.axes.iter().flatten()) {
                put_f32(&mut body, *v);
            }
            put_f32(&mut body, g.radiance);
            body.resize(at + GROUP_BYTES, 0);
        }
        close(&mut table, start, &body);
        pad16(&mut body);
        open(&mut table, start, &body, TAG_POLYGONS, 0, 0, 0, self.polygons.len() as u32);
        for p in &self.polygons {
            let at = body.len();
            put_u32(&mut body, p.group);
            put_u32(&mut body, p.vertices.len() as u32);
            for v in p.radiance {
                put_f32(&mut body, v);
            }
            put_f32(&mut body, 0.0);
            for v in &p.vertices {
                for x in v {
                    put_f32(&mut body, *x);
                }
            }
            body.resize(at + POLYGON_BYTES, 0);
        }
        close(&mut table, start, &body);
        for (k, p) in self.profiles.iter().enumerate() {
            pad16(&mut body);
            open(&mut table, start, &body, TAG_PROFILE, 0, 0, k as u32, p.width * p.height);
            let at = body.len();
            put_u32(&mut body, p.width);
            put_u32(&mut body, p.height);
            for v in p.lon.iter().chain(&p.lat).chain(&p.outside) {
                put_f32(&mut body, *v);
            }
            body.resize(at + PROFILE_HEAD_BYTES, 0);
            put_u16s(&mut body, &p.texels);
            close(&mut table, start, &body);
        }
        for l in &layers {
            pad16(&mut body);
            open(&mut table, start, &body, TAG_LAYER, l.group, l.kind, l.chunk, l.blocks.len() as u32);
            let at = body.len();
            l.write(&mut body);
            debug_assert_eq!(body.len() - at, l.byte_len());
            close(&mut table, start, &body);
        }
        pad16(&mut body);
        let mut out = Vec::with_capacity(start + body.len());
        put_u32(&mut out, ATHL_MAGIC);
        put_u32(&mut out, ATHL_VERSION);
        put_u32(&mut out, self.flags);
        put_u32(&mut out, self.groups.len() as u32);
        put_u32(&mut out, self.element_count);
        put_u32(&mut out, self.merged);
        put_u32(&mut out, self.splat_base);
        put_u32(&mut out, self.splat_count);
        put_u32(&mut out, CHUNK_SPLATS);
        put_u32(&mut out, BLOCK_SPLATS);
        put_u32(&mut out, table.len() as u32);
        put_u32(&mut out, 0);
        out.extend_from_slice(&(HEADER_BYTES as u64).to_le_bytes());
        out.extend_from_slice(&self.cloud_hash.to_le_bytes());
        out.extend_from_slice(&self.bake_hash.to_le_bytes());
        out.resize(HEADER_BYTES, 0);
        for s in &table {
            put_u32(&mut out, s.tag);
            out.extend_from_slice(&s.group.to_le_bytes());
            out.extend_from_slice(&s.kind.to_le_bytes());
            put_u32(&mut out, s.chunk);
            put_u32(&mut out, s.count);
            out.extend_from_slice(&s.offset.to_le_bytes());
            out.extend_from_slice(&s.bytes.to_le_bytes());
            out.extend_from_slice(&[0u8; 8]);
        }
        out.resize(start, 0);
        out.extend_from_slice(&body);
        Ok(out)
    }

    pub fn read(bytes: &[u8]) -> Result<Self> {
        let h = AthlHeader::parse(bytes)?;
        ensure!(h.chunk_splats == CHUNK_SPLATS && h.block_splats == BLOCK_SPLATS, ".athl chunk or block size");
        let section = |s: &AthlSection| -> Result<&[u8]> {
            let (a, b) = (s.offset as usize, (s.offset + s.bytes) as usize);
            bytes.get(a..b).ok_or_else(|| anyhow!(".athl section past the end"))
        };
        let mut file = AthlFile {
            flags: h.flags,
            element_count: h.element_count,
            merged: h.merged,
            splat_base: h.splat_base,
            splat_count: h.splat_count,
            cloud_hash: u64_at(bytes, 56),
            bake_hash: u64_at(bytes, 64),
            ..Default::default()
        };
        for s in &h.sections {
            let b = section(s)?;
            match s.tag {
                TAG_GROUPS => {
                    for i in 0..s.count as usize {
                        let at = i * GROUP_BYTES;
                        ensure!(b.len() >= at + GROUP_BYTES, "GRPS short");
                        let name = &b[at..at + GROUP_NAME_BYTES];
                        let end = name.iter().position(|&c| c == 0).unwrap_or(GROUP_NAME_BYTES);
                        let f = at + GROUP_NAME_BYTES;
                        file.groups.push(AthlGroup {
                            name: String::from_utf8(name[..end].to_vec())?,
                            flags: u32_at(b, f),
                            polygon_first: u32_at(b, f + 4),
                            polygon_count: u32_at(b, f + 8),
                            profile: u32_at(b, f + 12) as i32,
                            tint: f32x3_at(b, f + 16),
                            origin: f32x3_at(b, f + 28),
                            axes: [f32x3_at(b, f + 40), f32x3_at(b, f + 52), f32x3_at(b, f + 64)],
                            radiance: f32_at(b, f + 76),
                        });
                    }
                }
                TAG_POLYGONS => {
                    for i in 0..s.count as usize {
                        let at = i * POLYGON_BYTES;
                        ensure!(b.len() >= at + POLYGON_BYTES, "POLY short");
                        let n = u32_at(b, at + 4) as usize;
                        ensure!((3..=MAX_POLYGON_VERTICES).contains(&n), "a polygon of {} vertices", n);
                        file.polygons.push(AthlPolygon {
                            group: u32_at(b, at),
                            radiance: f32x3_at(b, at + 8),
                            vertices: (0..n).map(|v| f32x3_at(b, at + 24 + 12 * v)).collect(),
                        });
                    }
                }
                TAG_PROFILE => {
                    ensure!(b.len() >= PROFILE_HEAD_BYTES, "PROF short");
                    let (width, height) = (u32_at(b, 0), u32_at(b, 4));
                    let n = 3 * (width * height) as usize;
                    ensure!(b.len() >= PROFILE_HEAD_BYTES + 2 * n, "PROF texels short");
                    file.profiles.push(AthlProfile {
                        width,
                        height,
                        lon: [f32_at(b, 8), f32_at(b, 12)],
                        lat: [f32_at(b, 16), f32_at(b, 20)],
                        outside: f32x3_at(b, 24),
                        texels: u16s_at(b, PROFILE_HEAD_BYTES, n),
                    });
                }
                TAG_LAYER => file.layers.push(AthlLayer::read(b)?),
                _ => {} // unknown sections are skipped
            }
        }
        ensure!(file.groups.len() == h.group_count as usize, ".athl group count");
        Ok(file)
    }
}

/// Sparse layers of one group and kind from values over the virtual
/// elements (`components` a splat, `element_count` splats): a block is kept
/// when any of its values is past `threshold` in magnitude.
pub fn sparse_layers(
    group: u16,
    kind: u16,
    components: u32,
    values: &[f32],
    threshold: f32,
) -> Result<Vec<AthlLayer>> {
    let c = components as usize;
    ensure!(c > 0 && values.len() % c == 0, "values are not a whole number of splats");
    let count = values.len() / c;
    let mut out = Vec::new();
    let chunks = count.div_ceil(CHUNK_SPLATS as usize);
    for chunk in 0..chunks {
        let mut layer = AthlLayer { group, kind, chunk: chunk as u32, components, ..Default::default() };
        for block in 0..BLOCKS_PER_CHUNK as usize {
            let first = chunk * CHUNK_SPLATS as usize + block * BLOCK_SPLATS as usize;
            if first >= count {
                break;
            }
            let last = (first + BLOCK_SPLATS as usize).min(count);
            let span = &values[first * c..last * c];
            if !span.iter().any(|v| v.abs() > threshold) {
                continue;
            }
            layer.blocks.push(block as u16);
            for e in first..first + BLOCK_SPLATS as usize {
                for k in 0..c {
                    let v = if e < count { values[e * c + k] } else { 0.0 };
                    layer.data.push(f16::from_f32(v).to_bits());
                }
            }
        }
        if !layer.blocks.is_empty() {
            out.push(layer);
        }
    }
    Ok(out)
}

/// A per-splat layer (file order, `components` a splat) over the cloud's
/// page-aligned virtual order: the splats at `splat_base`, and each merged
/// node the weighted mean over the splats under it, weighted as athenea's
/// LoD merges colours (lod_common.slang: opacity x the area of the two
/// longest axes).
pub fn virtual_values(file: &AthcFile, tree: &VirtualTree, per_splat: &[f32], components: u32) -> Result<Vec<f32>> {
    let c = components as usize;
    let n = file.header.count as usize;
    ensure!(per_splat.len() == n * c, "a layer of {} values for {} splats x {}", per_splat.len(), n, c);
    ensure!(tree.splat_base % CHUNK_SPLATS == 0, "the tree is not page-aligned");
    let splats = file.splats();
    let mut weight = vec![0f32; n];
    for (i, w) in weight.iter_mut().enumerate() {
        let opacity = splats.positions[4 * i + 3];
        let s = &splats.shape[4 * i..4 * i + 4];
        let mut axes = [low_half(s[1]).exp(), high_half(s[1]).exp(), low_half(s[2]).exp()];
        axes.sort_by(|a, b| b.total_cmp(a));
        *w = opacity * axes[0] * axes[1];
    }
    let mut out = vec![0f32; (tree.splat_base as usize + n) * c];
    out[tree.splat_base as usize * c..].copy_from_slice(per_splat);
    // Prefix sums over splats in file order: a merged node's splats are
    // the contiguous run of its finest groups.
    let mut sum_w = vec![0f64; n + 1];
    let mut sum_v = vec![0f64; (n + 1) * c];
    for i in 0..n {
        sum_w[i + 1] = sum_w[i] + weight[i] as f64;
        for k in 0..c {
            sum_v[(i + 1) * c + k] = sum_v[i * c + k] + weight[i] as f64 * per_splat[i * c + k] as f64;
        }
    }
    let start_of = |g: u32| -> usize {
        if (g as usize) < file.starts.len() {
            file.starts[g as usize] as usize
        } else {
            n
        }
    };
    for m in 0..tree.merged as usize {
        let (lo, hi) = (tree.group_range[2 * m], tree.group_range[2 * m + 1]);
        let (a, b) = (start_of(lo), start_of(hi));
        let w = sum_w[b] - sum_w[a];
        if w <= 0.0 {
            continue;
        }
        for k in 0..c {
            out[m * c + k] = ((sum_v[b * c + k] - sum_v[a * c + k]) / w) as f32;
        }
    }
    Ok(out)
}

// --- the direct term, as the shader evaluates it ----------------------------
// (slang/athenea_adapter/lights.slang; the same arithmetic in f32.)

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm3(a: [f32; 3]) -> [f32; 3] {
    let l = dot3(a, a).sqrt().max(1e-20);
    [a[0] / l, a[1] / l, a[2] / l]
}

/// The diffuse form factor of a polygon seen from `x` with normal `n`:
/// (1/pi) times the integral of the cosine over the polygon -- LTC with
/// the identity transform, Lambert's closed form (060 §2), the polygon
/// first clipped to the horizon. A one-sided emitter lights only the side
/// its counter-clockwise winding faces. The diffuse radiance a splat sends
/// on is albedo x radiance x this.
pub fn polygon_form_factor(x: [f32; 3], n: [f32; 3], vertices: &[[f32; 3]], two_sided: bool) -> f32 {
    if vertices.len() < 3 {
        return 0.0;
    }
    if !two_sided {
        let face = cross3(sub3(vertices[1], vertices[0]), sub3(vertices[2], vertices[0]));
        if dot3(face, sub3(x, vertices[0])) <= 0.0 {
            return 0.0;
        }
    }
    // Sutherland-Hodgman against the plane n . d = 0, d = vertex - x.
    let d: Vec<[f32; 3]> = vertices.iter().map(|v| sub3(*v, x)).collect();
    let mut clipped: Vec<[f32; 3]> = Vec::with_capacity(d.len() + 1);
    for i in 0..d.len() {
        let (a, b) = (d[i], d[(i + 1) % d.len()]);
        let (ha, hb) = (dot3(n, a), dot3(n, b));
        if ha > 0.0 {
            clipped.push(a);
        }
        if (ha > 0.0) != (hb > 0.0) {
            let t = ha / (ha - hb);
            clipped.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]);
        }
    }
    if clipped.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..clipped.len() {
        let a = norm3(clipped[i]);
        let b = norm3(clipped[(i + 1) % clipped.len()]);
        let c = dot3(a, b).clamp(-1.0, 1.0);
        let angle = c.acos();
        let g = cross3(a, b);
        let gl = dot3(g, g).sqrt();
        if gl > 1e-12 {
            sum += angle * dot3(n, g) / gl;
        }
    }
    sum.abs() / (2.0 * std::f32::consts::PI)
}

/// The group's lens profile towards `x` (076: the direction from the lamp's
/// origin, in its frame; longitude from +z towards +x, latitude towards +y),
/// bilinear between texel centres, `outside` past the sector; times the
/// group's tint.
pub fn profile_sample(p: &AthlProfile, g: &AthlGroup, x: [f32; 3]) -> [f32; 3] {
    let d = sub3(x, g.origin);
    let local = norm3([dot3(d, g.axes[0]), dot3(d, g.axes[1]), dot3(d, g.axes[2])]);
    let lon = local[0].atan2(local[2]);
    let lat = local[1].clamp(-1.0, 1.0).asin();
    let u = (lon - p.lon[0]) / (p.lon[1] - p.lon[0]);
    let v = (lat - p.lat[0]) / (p.lat[1] - p.lat[0]);
    if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
        return [p.outside[0] * g.tint[0], p.outside[1] * g.tint[1], p.outside[2] * g.tint[2]];
    }
    let fx = (u * p.width as f32 - 0.5).clamp(0.0, (p.width - 1) as f32);
    let fy = (v * p.height as f32 - 0.5).clamp(0.0, (p.height - 1) as f32);
    let (x0, y0) = (fx.floor() as u32, fy.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(p.width - 1), (y0 + 1).min(p.height - 1));
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let t = |x: u32, y: u32, c: usize| f16::from_bits(p.texels[(3 * (y * p.width + x)) as usize + c]).to_f32();
    let mut out = [0.0; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let a = t(x0, y0, c) + (t(x1, y0, c) - t(x0, y0, c)) * tx;
        let b = t(x0, y1, c) + (t(x1, y1, c) - t(x0, y1, c)) * tx;
        *o = (a + (b - a) * ty) * g.tint[c];
    }
    out
}

/// Checks a whole file's structure (what `read` does not: sizes against
/// the header).
pub fn validate(file: &AthlFile) -> Result<()> {
    if file.splat_base % CHUNK_SPLATS != 0 {
        bail!(".athl splat_base {} is not page-aligned", file.splat_base);
    }
    if file.element_count != file.splat_base + file.splat_count {
        bail!(".athl element count {} != {} + {}", file.element_count, file.splat_base, file.splat_count);
    }
    let chunks = file.element_count.div_ceil(CHUNK_SPLATS);
    for l in &file.layers {
        if l.chunk >= chunks {
            bail!("layer of chunk {} past the {} chunks", l.chunk, chunks);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> AthlFile {
        let n = 70_000u32;
        let mut values = vec![0f32; n as usize * 3];
        for i in [5u32, 300, 66_000, 69_999] {
            values[i as usize * 3] = 0.25;
            values[i as usize * 3 + 2] = 2.0;
        }
        let mut field = vec![0f32; n as usize * 27];
        field[27 * 1000 + 4] = -0.5;
        let mut layers = sparse_layers(1, KIND_INDIRECT, 3, &values, 1e-4).unwrap();
        layers.extend(sparse_layers(0, KIND_FIELD, 27, &field, 1e-4).unwrap());
        AthlFile {
            flags: 0,
            element_count: n,
            merged: 0,
            splat_base: 0,
            splat_count: n,
            cloud_hash: 0x0123_4567_89ab_cdef,
            bake_hash: 7,
            groups: vec![
                AthlGroup {
                    name: "cruce".into(),
                    polygon_first: 0,
                    polygon_count: 1,
                    profile: 0,
                    tint: [1.0, 1.0, 1.0],
                    axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    radiance: 1.0,
                    ..Default::default()
                },
                AthlGroup {
                    name: "pilotos".into(),
                    flags: GROUP_TWO_SIDED,
                    profile: -1,
                    tint: [1.0, 0.05, 0.02],
                    radiance: 1.0,
                    ..Default::default()
                },
            ],
            polygons: vec![AthlPolygon {
                group: 0,
                radiance: [1.0, 1.0, 1.0],
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            }],
            profiles: vec![AthlProfile {
                width: 4,
                height: 2,
                lon: [-0.7, 0.7],
                lat: [-0.17, 0.09],
                outside: [0.0, 0.0, 0.0],
                texels: (0..24).map(|i| f16::from_f32(i as f32 / 8.0).to_bits()).collect(),
            }],
            layers,
        }
    }

    #[test]
    fn round_trips() {
        let file = sample();
        validate(&file).unwrap();
        let bytes = file.write().unwrap();
        let back = AthlFile::read(&bytes).unwrap();
        let mut sorted = file.clone();
        sorted.layers.sort_by_key(|l| (l.chunk, l.group, l.kind));
        assert_eq!(back, sorted);
        assert_eq!(back.write().unwrap(), bytes);
        let h = AthlHeader::parse(&bytes[..AthlHeader::prefix_bytes(&bytes).unwrap() as usize]).unwrap();
        assert_eq!(h.cloud_hash, "0123456789abcdef");
        // Each LAYR section reads alone, as a Range request returns it.
        for s in h.sections.iter().filter(|s| s.tag == TAG_LAYER) {
            let l = AthlLayer::read(&bytes[s.offset as usize..(s.offset + s.bytes) as usize]).unwrap();
            assert_eq!((l.group, l.kind, l.chunk), (s.group, s.kind, s.chunk));
            assert_eq!(l.blocks.len() as u32, s.count);
        }
    }

    #[test]
    fn layers_are_sparse_by_block() {
        let file = sample();
        let indirect: Vec<_> = file.layers.iter().filter(|l| l.kind == KIND_INDIRECT).collect();
        // Splats 5 and 300 are blocks 0 and 1 of chunk 0; 66000 and 69999 are
        // blocks 1 and 17 of chunk 1.
        assert_eq!(indirect.len(), 2);
        assert_eq!(indirect[0].blocks, vec![0, 1]);
        assert_eq!(indirect[1].blocks, vec![1, 17]);
        assert_eq!(indirect[1].value(69_999, 2), 2.0);
        assert_eq!(indirect[1].value(69_998, 2), 0.0);
        assert_eq!(indirect[0].value(300, 0), 0.25);
        assert_eq!(indirect[0].value(66_000, 0), 0.0); // another chunk's
        let field = file.layers.iter().find(|l| l.kind == KIND_FIELD).unwrap();
        assert_eq!(field.value(1000, 4), -0.5);
        assert_eq!(field.data.len(), 256 * 27);
    }

    #[test]
    fn form_factor_of_a_square_overhead() {
        // A unit square 1 above a point facing it, against the integral of
        // cos^2 / r^2 over the square, numerically.
        let v = vec![[-0.5, 1.0, -0.5], [-0.5, 1.0, 0.5], [0.5, 1.0, 0.5], [0.5, 1.0, -0.5]];
        let f = polygon_form_factor([0.0; 3], [0.0, 1.0, 0.0], &v, true);
        let mut sum = 0.0f64;
        let m = 400;
        for i in 0..m {
            for j in 0..m {
                let x = -0.5 + (i as f64 + 0.5) / m as f64;
                let z = -0.5 + (j as f64 + 0.5) / m as f64;
                let r2 = x * x + z * z + 1.0;
                sum += 1.0 / (r2 * r2) / (m * m) as f64; // cos^2 / r^2 with cos = 1/r
            }
        }
        let want = (sum / std::f64::consts::PI) as f32;
        assert!((f - want).abs() < 1e-4, "{f} vs {want}");
        // Facing it from below a one-sided emitter facing up: nothing.
        let up = vec![[-0.5, 1.0, -0.5], [-0.5, 1.0, 0.5], [0.5, 1.0, 0.5], [0.5, 1.0, -0.5]];
        let face = cross3(sub3(up[1], up[0]), sub3(up[2], up[0]));
        assert!(face[1] > 0.0);
        assert_eq!(polygon_form_factor([0.0; 3], [0.0, 1.0, 0.0], &up, false), 0.0);
        // Half below the horizon: clipped to the half above.
        let tilted = polygon_form_factor([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], &v, true);
        assert_eq!(tilted, 0.0); // edge-on
        let half = polygon_form_factor([0.0, 0.5, 0.0], [0.0, 0.0, 1.0], &v, true);
        assert!(half > 0.0 && half < f);
    }

    #[test]
    fn rejects_what_does_not_fit() {
        let mut file = sample();
        file.groups[0].polygon_count = 2;
        assert!(file.write().is_err());
        let bytes = sample().write().unwrap();
        assert!(AthlFile::read(&bytes[..100]).is_err());
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(AthlFile::read(&bad).is_err());
    }
}
