//! The compact relit path's planes as a FOURTH TIER of a paged tree (thread
//! CN): what an iPhone keeps of athenea's band pages.
//!
//! The compact path (src/webgpu/athenea/compact.ts, research/iphone-compact.md)
//! relights each drawn gaussian in its shading pass from ~40 bytes: geometry
//! quantized in a box, athenea's joint CPCA (a cluster and 3 x 8 int4
//! weights), a shadowBits dictionary index and a material index. Here those
//! bytes are written per error-band page (`athc_pages`), in a section of
//! their own, `CMPC`, tier 4, so a reader that draws only the compact path
//! fetches one run of tier 4 per Range and never a byte of tiers 1 to 3:
//!
//! - one quantization box per PAGE (the page head, in the page table's row);
//! - the tables GLOBAL per piece (a file): the shadow dictionary (k-means in
//!   Hamming space over every node, `athc_shadow_vq::kmeans_hamming`), the
//!   material table (every distinct pbr + lobes), athenea's joint CPCA basis
//!   in f16 -- one blob among the file's tables (`CompactTables`), read with
//!   the page table;
//! - per node, `NODE_BYTES` (64) bytes as byte planes in the page's section:
//!   the 40 the GPU keeps (`PLANE_WIDTHS[..15]`) and the 24 the traversal
//!   keeps on the CPU (the LoD tree's 4 words, exactly what the decoder
//!   gives a node, and the 2 LODE words), so the traversal over tier 4 is
//!   the traversal over tiers 1-3.
//!
//! The GPU's decode (slang/athenea_adapter/compact.slang) is the resident
//! compact cloud's (`encodeCompact`), the planes in pool slots.

use anyhow::{anyhow, bail, Result};
use half::f16;
use serde::Serialize;

use crate::athc::{high_half, linear_to_srgb, low_half, spark_lod_opacity, unpack_normal, AthcBlock};
use crate::splat_encode::{
    decode_ext_splat_center, decode_ext_splat_opacity, decode_ext_splat_scale, encode_ext_splat, encode_lod_tree,
    set_lod_tree_size,
};

/// Bytes a node in a `CMPC` section.
pub const NODE_BYTES: usize = 64;
/// Bytes a node in each plane, in the order they are stored: position x, y,
/// z (u16 in the page's box), albedo (3 x u8 sRGB), opacity (the code), ln
/// scales (3 x u8 in the page's range, 0 = exactly 0), rotation (the ext
/// quaternion word), normal (octahedral, 12 bits an axis), curvature (3 x
/// signed log), material (u16), CPCA cluster (u8), the three blocks' 8 int4
/// weights (a word each), shadow dictionary index (u16) -- the GPU's 40 --
/// then the LoD tree's 4 words and the 2 LODE words (the traversal's 24).
pub const PLANE_WIDTHS: [u32; 17] = [2, 2, 2, 3, 1, 3, 4, 3, 3, 2, 1, 4, 4, 4, 2, 16, 8];
pub const PLANES: usize = PLANE_WIDTHS.len();
/// The planes the GPU keeps (the rest are the traversal's).
pub const GPU_PLANES: usize = 15;
pub const P_POS_X: usize = 0;
pub const P_ALBEDO: usize = 3;
pub const P_OPACITY: usize = 4;
pub const P_SCALE: usize = 5;
pub const P_ROTATION: usize = 6;
pub const P_NORMAL: usize = 7;
pub const P_CURVATURE: usize = 8;
pub const P_MATERIAL: usize = 9;
pub const P_CLUSTER: usize = 10;
pub const P_WEIGHTS: usize = 11;
pub const P_SHADOW: usize = 14;
pub const P_LOD_TREE: usize = 15;
pub const P_LOD_ERROR: usize = 16;
/// Words of a page head (the page table row's words 17..29): box lo, hi
/// (f32 x 3 each), ln scale lo, hi, the curvature's range (log2(1 +
/// max / floor)), then three words a reader fills (compact.slang's chunk
/// words 9, 10: the codebook's and the materials' word offsets).
pub const HEAD_WORDS: usize = 12;
/// The blob of the tables (`CompactTables::to_bytes`).
pub const TABLE_MAGIC: u32 = u32::from_le_bytes(*b"CMPT");
pub const TABLE_VERSION: u32 = 1;
const TABLE_FIXED_WORDS: usize = 16;
/// athenea's joint CPCA (inbox/q97): the transfer's three blocks.
pub const JOINT_BLOCKS: [usize; 3] = [16, 48, 48];
pub const JOINT_K: usize = 8;
/// Floats a cluster in the GPU table (cpcaJoint.ts packJointBasis).
pub const CLUSTER_FLOATS: usize = 1008;
/// compact.slang kCompactCurvatureFloor.
pub const CURVATURE_FLOOR: f32 = 0.01;

/// Byte offset of plane `k` in a page section of `n` nodes.
pub fn plane_offset(k: usize, n: usize) -> usize {
    n * PLANE_WIDTHS[..k].iter().sum::<u32>() as usize
}

// --- athenea's joint CPCA basis ---------------------------------------------

/// athenea's joint CPCA basis (cpca_joint_C256_K8.{json,bin}): see
/// src/webgpu/athenea/cpcaJoint.ts.
#[derive(Clone, Debug)]
pub struct JointBasis {
    pub clusters: usize,
    /// [C][112], the blocks over their energy norms.
    pub centroids: Vec<f32>,
    pub energy_norm: [f32; 3],
    /// Per block: [C][D] means, [C][D][K] directions, [C][K] scales.
    pub mean: [Vec<f32>; 3],
    pub dir: [Vec<f32>; 3],
    pub scale: [Vec<f32>; 3],
}

impl JointBasis {
    /// From its .json and .bin (little-endian f32 arrays).
    pub fn parse(json: &str, bin: &[u8]) -> Result<Self> {
        let j: serde_json::Value = serde_json::from_str(json)?;
        let c = j["C"].as_u64().ok_or_else(|| anyhow!("cpca basis: no C"))? as usize;
        if j["K"].as_u64() != Some(JOINT_K as u64) {
            bail!("cpca basis: K {:?}, the record holds {}", j["K"], JOINT_K);
        }
        if c > 256 {
            bail!("cpca basis: {} clusters, the compact cluster is a byte", c);
        }
        let arrays = j["arrays"].as_array().ok_or_else(|| anyhow!("cpca basis: no arrays"))?;
        let array = |name: &str| -> Result<Vec<f32>> {
            let a = arrays
                .iter()
                .find(|a| a["name"].as_str() == Some(name))
                .ok_or_else(|| anyhow!("cpca basis: no {name}"))?;
            let at = a["offset_bytes"].as_u64().unwrap_or(0) as usize;
            let size = a["size_bytes"].as_u64().unwrap_or(0) as usize;
            let b = bin.get(at..at + size).ok_or_else(|| anyhow!("cpca basis: {name} past the end"))?;
            Ok(b.chunks_exact(4).map(|w| f32::from_le_bytes(w.try_into().unwrap())).collect())
        };
        let names = ["transferDirect", "transferIndirect", "transferReflected"];
        let en = array("energy_norm")?;
        let basis = Self {
            clusters: c,
            centroids: array("centroids_joint")?,
            energy_norm: [en[0], en[1], en[2]],
            mean: [array(&format!("{}_mean", names[0]))?, array(&format!("{}_mean", names[1]))?, array(&format!("{}_mean", names[2]))?],
            dir: [array(&format!("{}_basis", names[0]))?, array(&format!("{}_basis", names[1]))?, array(&format!("{}_basis", names[2]))?],
            scale: [array(&format!("{}_scale", names[0]))?, array(&format!("{}_scale", names[1]))?, array(&format!("{}_scale", names[2]))?],
        };
        for b in 0..3 {
            let d = JOINT_BLOCKS[b];
            if basis.mean[b].len() != c * d || basis.dir[b].len() != c * d * JOINT_K || basis.scale[b].len() != c * JOINT_K {
                bail!("cpca basis: block {b} sized wrong");
            }
        }
        if basis.centroids.len() != c * 112 {
            bail!("cpca basis: centroids sized wrong");
        }
        Ok(basis)
    }

    /// athenea's encode (cpcaJoint.ts encodeJointCpca) of 112 values: the
    /// cluster and the 24 weights at `levels` a side (7: int4).
    pub fn encode(&self, x: &[f32], levels: i32) -> (u32, [i32; 24]) {
        let inv = self.energy_norm.map(|e| 1.0 / e as f64);
        let mut z = [0f64; 112];
        for v in 0..112 {
            z[v] = x[v] as f64 * inv[if v < 16 { 0 } else if v < 64 { 1 } else { 2 }];
        }
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for c in 0..self.clusters {
            let cent = &self.centroids[c * 112..c * 112 + 112];
            let (mut dot, mut n2) = (0f64, 0f64);
            for v in 0..112 {
                dot += z[v] * cent[v] as f64;
                n2 += (cent[v] as f64) * (cent[v] as f64);
            }
            let d = n2 - 2.0 * dot;
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        let mut q = [0i32; 24];
        let mut first = 0;
        for b in 0..3 {
            let d = JOINT_BLOCKS[b];
            for k in 0..JOINT_K {
                let mut w = 0f64;
                for i in 0..d {
                    w += self.dir[b][(best * d + i) * JOINT_K + k] as f64
                        * (x[first + i] as f64 - self.mean[b][best * d + i] as f64);
                }
                let s = self.scale[b][best * JOINT_K + k] as f64;
                q[b * JOINT_K + k] =
                    if s > 0.0 { ((w / s) * levels as f64).round().clamp(-levels as f64, levels as f64) as i32 } else { 0 };
            }
            first += d;
        }
        (best as u32, q)
    }

    /// The 112 values weights at `levels` decode to, as the GPU rebuilds
    /// them (each weight back on the int8 grid, round(q x 127 / levels)).
    pub fn decode(&self, cluster: u32, q: &[i32; 24], levels: i32) -> [f32; 112] {
        let c = cluster as usize;
        let mut out = [0f32; 112];
        let mut first = 0;
        for b in 0..3 {
            let d = JOINT_BLOCKS[b];
            for i in 0..d {
                let mut acc = self.mean[b][c * d + i] as f64;
                for k in 0..JOINT_K {
                    let q8 = (q[b * JOINT_K + k] as f64 * 127.0 / levels as f64).round();
                    acc += q8 * self.dir[b][(c * d + i) * JOINT_K + k] as f64 * self.scale[b][c * JOINT_K + k] as f64
                        / 127.0;
                }
                out[first + i] = acc as f32;
            }
            first += d;
        }
        out
    }

    /// The GPU table (cpcaJoint.ts packJointBasis) in f16, two a word:
    /// `CLUSTER_FLOATS / 2` words a cluster.
    pub fn table_f16(&self) -> Vec<u32> {
        let mut halves = vec![0u16; self.clusters * CLUSTER_FLOATS];
        for c in 0..self.clusters {
            let mut at = c * CLUSTER_FLOATS;
            for b in 0..3 {
                let d = JOINT_BLOCKS[b];
                for i in 0..d {
                    halves[at + i] = f16::from_f32(self.mean[b][c * d + i]).to_bits();
                }
                for k in 0..JOINT_K {
                    let s = self.scale[b][c * JOINT_K + k] / 127.0;
                    for i in 0..d {
                        halves[at + (1 + k) * d + i] =
                            f16::from_f32(self.dir[b][(c * d + i) * JOINT_K + k] * s).to_bits();
                    }
                }
                at += d * (JOINT_K + 1);
            }
        }
        halves.chunks(2).map(|p| p[0] as u32 | (p.get(1).copied().unwrap_or(0) as u32) << 16).collect()
    }
}

// --- the normal and the opacity codes (compact.ts) ---------------------------

fn oct_vector(fx: f64, fy: f64) -> [f64; 3] {
    let (mut x, mut y) = (fx, fy);
    let z = 1.0 - fx.abs() - fy.abs();
    let t = (-z).clamp(0.0, 1.0);
    x += if x >= 0.0 { -t } else { t };
    y += if y >= 0.0 { -t } else { t };
    let l = (x * x + y * y + z * z).sqrt();
    let l = if l > 0.0 { l } else { 1.0 };
    [x / l, y / l, z / l]
}

/// compact.ts decodeNormalBits.
pub fn decode_normal_bits(q: u32, bits: u32) -> [f64; 3] {
    let top = ((1u64 << bits) - 1) as f64;
    let x = q & ((1 << bits) - 1);
    let y = q >> bits;
    oct_vector(x as f64 / top * 2.0 - 1.0, y as f64 / top * 2.0 - 1.0)
}

/// compact.ts encodeNormalBits: the best of the four codes around n.
pub fn encode_normal_bits(n: [f32; 3], bits: u32) -> u32 {
    let n = n.map(|v| v as f64);
    let top = ((1u64 << bits) - 1) as f64;
    let l1 = n[0].abs() + n[1].abs() + n[2].abs();
    let l1 = if l1 > 0.0 { l1 } else { 1.0 };
    let (mut px, mut py) = (n[0] / l1, n[1] / l1);
    if n[2] < 0.0 {
        let ox = (1.0 - py.abs()) * if px >= 0.0 { 1.0 } else { -1.0 };
        let oy = (1.0 - px.abs()) * if py >= 0.0 { 1.0 } else { -1.0 };
        px = ox;
        py = oy;
    }
    let ux = (px * 0.5 + 0.5) * top;
    let uy = (py * 0.5 + 0.5) * top;
    let mut best = 0u32;
    let mut best_dot = -2.0;
    for qx in [ux.floor(), ux.ceil()] {
        for qy in [uy.floor(), uy.ceil()] {
            let cx = qx.clamp(0.0, top) as u32;
            let cy = qy.clamp(0.0, top) as u32;
            let code = cx | cy << bits;
            let v = decode_normal_bits(code, bits);
            let d = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
            if d > best_dot {
                best_dot = d;
                best = code;
            }
        }
    }
    best
}

/// compact.ts encodeOpacity: sqrt up to 1, then 1..2 by 1/63.
pub fn encode_opacity(a: f32) -> u8 {
    if a.is_nan() || a <= 0.0 {
        return 0;
    }
    if a <= 1.0 {
        return (a.sqrt() * 191.0).round() as u8;
    }
    (192.0 + ((a - 1.0) * 63.0).round()).min(255.0) as u8
}

pub fn decode_opacity(code: u8) -> f32 {
    if code <= 191 {
        let s = code as f32 / 191.0;
        s * s
    } else {
        1.0 + (code - 192) as f32 / 63.0
    }
}

// --- the tables -----------------------------------------------------------

/// A piece's tables, global to its pages (the blob among the file's tables).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompactTables {
    /// The shadow dictionary: 8 words an entry.
    pub codebook: Vec<u32>,
    /// The materials: pbr then 3 lobes words, 4 words an entry.
    pub materials: Vec<u32>,
    /// athenea's basis, f16 (`JointBasis::table_f16`).
    pub basis: Vec<u32>,
    pub clusters: u32,
    pub normal_bits: u32,
    pub weight_bits: u32,
    /// The albedo is linear light (else sRGB; this writer stores sRGB).
    pub linear: bool,
    pub curvature: bool,
    /// The basis is the asset's (a sidecar `.cmpa`, athc_basis_fit, thread
    /// CR): `basis` then holds the asset table's id (2 words) instead.
    pub asset_basis: bool,
}

/// The piece's tables flag: its basis is in the asset table (CQ's value).
pub const TABLE_ASSET_BASIS: u32 = 4;

impl CompactTables {
    pub fn entries(&self) -> usize {
        self.codebook.len() / 8
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let head = TABLE_FIXED_WORDS + PLANES;
        let codebook_at = head.div_ceil(16) * 16;
        let materials_at = (codebook_at + self.codebook.len()).div_ceil(16) * 16;
        let basis_at = (materials_at + self.materials.len()).div_ceil(16) * 16;
        let total = (basis_at + self.basis.len()).div_ceil(16) * 16;
        let mut w = vec![0u32; total];
        w[..TABLE_FIXED_WORDS].copy_from_slice(&[
            TABLE_MAGIC,
            TABLE_VERSION,
            NODE_BYTES as u32,
            PLANES as u32,
            HEAD_WORDS as u32,
            self.entries() as u32,
            (self.materials.len() / 4) as u32,
            self.clusters,
            self.normal_bits,
            self.weight_bits,
            (self.linear as u32) | (self.curvature as u32) << 1 | if self.asset_basis { TABLE_ASSET_BASIS } else { 0 },
            codebook_at as u32,
            materials_at as u32,
            basis_at as u32,
            self.basis.len() as u32,
            total as u32,
        ]);
        w[TABLE_FIXED_WORDS..head].copy_from_slice(&PLANE_WIDTHS);
        w[codebook_at..codebook_at + self.codebook.len()].copy_from_slice(&self.codebook);
        w[materials_at..materials_at + self.materials.len()].copy_from_slice(&self.materials);
        w[basis_at..basis_at + self.basis.len()].copy_from_slice(&self.basis);
        w.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        let w = |k: usize| -> Result<u32> {
            Ok(u32::from_le_bytes(b.get(4 * k..4 * k + 4).ok_or_else(|| anyhow!("compact tables too short"))?.try_into().unwrap()))
        };
        if w(0)? != TABLE_MAGIC || w(1)? != TABLE_VERSION {
            bail!("not a compact table (CMPT v{TABLE_VERSION})");
        }
        if w(2)? as usize != NODE_BYTES || w(3)? as usize != PLANES || w(4)? as usize != HEAD_WORDS {
            bail!("compact tables of another layout");
        }
        let words = |at: u32, n: usize| -> Result<Vec<u32>> { (0..n).map(|k| w(at as usize + k)).collect() };
        let flags = w(10)?;
        Ok(Self {
            codebook: words(w(11)?, 8 * w(5)? as usize)?,
            materials: words(w(12)?, 4 * w(6)? as usize)?,
            basis: words(w(13)?, w(14)? as usize)?,
            clusters: w(7)?,
            normal_bits: w(8)?,
            weight_bits: w(9)?,
            linear: flags & 1 != 0,
            curvature: flags & 2 != 0,
            asset_basis: flags & TABLE_ASSET_BASIS != 0,
        })
    }
}

// --- the pages ------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CompactOptions {
    /// Centroids of the shadow dictionary (athenea's 1024).
    pub centroids: usize,
    /// k-means rounds (athenea's 12).
    pub rounds: usize,
    pub threads: usize,
    /// Bits an axis of the normal (12).
    pub normal_bits: u32,
    /// Levels a side of the CPCA weights (7: int4).
    pub levels: i32,
}

impl Default for CompactOptions {
    fn default() -> Self {
        Self {
            centroids: 1024,
            rounds: 12,
            threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
            normal_bits: 12,
            levels: 7,
        }
    }
}

/// What building the compact tier measured.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactStats {
    pub nodes: usize,
    pub pages: usize,
    pub materials: usize,
    pub distinct_masks: usize,
    pub codebook_entries: usize,
    /// Hamming distance of each node's mask to its entry: mean, p90, p99, max.
    pub hamming: [f64; 4],
    pub clusters_used: usize,
    /// The transfer through the basis at the weights' levels against the
    /// node's own: sum of squared errors over sum of squares (all 112 values,
    /// and by block).
    pub transfer_rel_mse: f64,
    pub transfer_rel_mse_blocks: [f64; 3],
    /// Bytes of the tables blob: codebook, materials, basis.
    pub table_bytes: [usize; 3],
    pub seconds: f32,
}

/// The compact tier of a paged tree: the tables, each page's `CMPC` section
/// (raw, `NODE_BYTES` a node) and its head.
pub struct CompactPages {
    pub tables: CompactTables,
    pub pages: Vec<Vec<u8>>,
    pub heads: Vec<[u32; HEAD_WORDS]>,
    pub stats: CompactStats,
}

type Mask = [u64; 4];

fn mask_of(words: &[u32]) -> Mask {
    let mut m = [0u64; 4];
    for k in 0..4 {
        m[k] = words[2 * k] as u64 | (words[2 * k + 1] as u64) << 32;
    }
    m
}

fn mask_words(m: &Mask) -> [u32; 8] {
    let mut w = [0u32; 8];
    for k in 0..4 {
        w[2 * k] = m[k] as u32;
        w[2 * k + 1] = (m[k] >> 32) as u32;
    }
    w
}

/// A node's LoD tree words as the decoder gives them (ExtSplats'
/// receiver: the ext splat encoded, read back, `encode_lod_tree`), with its
/// children named `page << 16 | place`.
pub fn lod_tree_words(block: &AthcBlock, i: usize) -> [u32; 4] {
    let p = &block.positions[i * 4..i * 4 + 4];
    let s = &block.shape[i * 4..i * 4 + 4];
    let q = crate::athc::decode_quaternion(s[0]);
    let scale = [low_half(s[1]).exp(), high_half(s[1]).exp(), low_half(s[2]).exp()];
    let (mut a, mut b) = ([0u32; 4], [0u32; 4]);
    encode_ext_splat(&mut a, &mut b, [p[0], p[1], p[2]], spark_lod_opacity(p[3]), [0.0; 3], scale, q);
    let (count, start) = if block.tree.len() == 2 * block.n {
        let (page, place, count) = crate::athc_pages::tree_children(&block.tree[2 * i..2 * i + 2]);
        (count as u16, if count > 0 { page << 16 | place } else { 0 })
    } else {
        (0, 0)
    };
    let mut w = [0u32; 4];
    encode_lod_tree(&mut w, &decode_ext_splat_center(&a), decode_ext_splat_opacity(&a), &decode_ext_splat_scale(&b), count, start);
    if let Some(&size) = block.lod_size.get(i).filter(|&&v| v > 0.0) {
        set_lod_tree_size(&mut w, size);
    }
    w
}

/// The compact tier of the pages of one paged tree (`athc_pages`), its
/// colours linear light when `linear` (stored as sRGB bytes, as the resident
/// path does).
pub fn build_compact(pages: &[AthcBlock], linear: bool, basis: &JointBasis, o: &CompactOptions) -> Result<CompactPages> {
    let t = std::time::Instant::now();
    let total: usize = pages.iter().map(|b| b.n).sum();
    for (p, b) in pages.iter().enumerate() {
        if b.n == 0 {
            continue;
        }
        // A cloud without lobes (no coat or specular layer) keeps zeros.
        if b.normals.len() != b.n || b.pbr.len() != b.n || !(b.lobes.is_empty() || b.lobes.len() == 3 * b.n) {
            bail!("compact: page {p} lacks normals, pbr or lobes (3 words): n {} normals {} pbr {} lobes {}", b.n, b.normals.len(), b.pbr.len(), b.lobes.len());
        }
        if b.shadow_bits.len() != 8 * b.n {
            bail!("compact: page {p} lacks the shadow bits (8 words)");
        }
        if b.transfer.len() != 56 * b.n {
            bail!("compact: page {p}: the transfer must be the full 112 values (usd-athc --transfer full)");
        }
    }
    let curvature = pages.iter().all(|b| b.curvature.len() == 2 * b.n) && pages.iter().any(|b| b.n > 0);

    // The shadow dictionary: every node of the piece.
    let masks: Vec<Mask> = pages.iter().flat_map(|b| b.shadow_bits.chunks_exact(8).map(mask_of)).collect();
    let mut distinct: Vec<Mask> = masks.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let (codebook, labels): (Vec<Mask>, Vec<u32>) = if distinct.len() <= o.centroids {
        let labels = masks.iter().map(|m| distinct.binary_search(m).unwrap() as u32).collect();
        (distinct.clone(), labels)
    } else {
        crate::athc_shadow_vq::kmeans_hamming(&masks, o.centroids, o.rounds, o.threads)
    };
    if codebook.len() > 0x10000 {
        bail!("compact: {} masks, the index is 16 bits", codebook.len());
    }
    let mut ham: Vec<u32> = masks
        .iter()
        .zip(&labels)
        .map(|(m, &l)| {
            let c = &codebook[l as usize];
            (0..4).map(|k| (m[k] ^ c[k]).count_ones()).sum()
        })
        .collect();
    ham.sort_unstable();
    let pct = |p: f64| ham.get(((ham.len() as f64 * p) as usize).min(ham.len().saturating_sub(1))).copied().unwrap_or(0) as f64;
    let hamming = [ham.iter().map(|&v| v as f64).sum::<f64>() / ham.len().max(1) as f64, pct(0.9), pct(0.99), pct(1.0)];

    // The materials: every distinct (pbr, lobes).
    let mut material_of: std::collections::HashMap<[u32; 4], u32> = std::collections::HashMap::new();
    let mut materials: Vec<u32> = Vec::new();

    let mut out_pages = Vec::with_capacity(pages.len());
    let mut heads = Vec::with_capacity(pages.len());
    let mut used = vec![false; basis.clusters];
    let (mut err2, mut ref2) = ([0f64; 3], [0f64; 3]);
    let mut node = 0usize;
    let nb = o.normal_bits;
    let normal_bytes = (2 * nb as usize).div_ceil(8);
    if normal_bytes != PLANE_WIDTHS[P_NORMAL] as usize {
        bail!("compact: a {}-bit normal is {} bytes, the plane {}", nb, normal_bytes, PLANE_WIDTHS[P_NORMAL]);
    }
    let mut x = [0f32; 112];
    for b in pages {
        let n = b.n;
        let mut raw = vec![0u8; n * NODE_BYTES];
        let at = |k: usize| plane_offset(k, n);
        // The page's box and ranges.
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        let (mut ln_lo, mut ln_hi) = (f32::MAX, f32::MIN);
        let mut c_max = 0f32;
        for i in 0..n {
            for k in 0..3 {
                lo[k] = lo[k].min(b.positions[4 * i + k]);
                hi[k] = hi[k].max(b.positions[4 * i + k]);
            }
            let s = &b.shape[4 * i..4 * i + 4];
            for ln in [low_half(s[1]), high_half(s[1]), low_half(s[2])] {
                if ln <= -60000.0 {
                    continue;
                }
                ln_lo = ln_lo.min(ln);
                ln_hi = ln_hi.max(ln);
            }
            if curvature {
                let c = &b.curvature[2 * i..2 * i + 2];
                for v in [low_half(c[0]), high_half(c[0]), low_half(c[1])] {
                    if v.is_finite() {
                        c_max = c_max.max(v.abs());
                    }
                }
            }
        }
        if n == 0 {
            lo = [0.0; 3];
            hi = [0.0; 3];
        }
        if ln_lo.is_nan() || ln_hi.is_nan() || ln_lo > ln_hi {
            ln_lo = 0.0;
            ln_hi = 0.0;
        }
        let range = (1.0 + c_max / CURVATURE_FLOOR).log2();
        let mut head = [0u32; HEAD_WORDS];
        for k in 0..3 {
            head[k] = lo[k].to_bits();
            head[3 + k] = hi[k].to_bits();
        }
        head[6] = ln_lo.to_bits();
        head[7] = ln_hi.to_bits();
        head[8] = range.to_bits();
        for i in 0..n {
            let p = &b.positions[4 * i..4 * i + 4];
            for k in 0..3 {
                let q = if hi[k] > lo[k] { (((p[k] - lo[k]) / (hi[k] - lo[k])) * 65535.0).round() as u16 } else { 0 };
                raw[at(P_POS_X + k) + 2 * i..at(P_POS_X + k) + 2 * i + 2].copy_from_slice(&q.to_le_bytes());
            }
            let s = &b.shape[4 * i..4 * i + 4];
            let mut colour = [high_half(s[2]), low_half(s[3]), high_half(s[3])];
            if linear {
                colour = colour.map(linear_to_srgb);
            }
            for k in 0..3 {
                raw[at(P_ALBEDO) + 3 * i + k] = (colour[k].clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            raw[at(P_OPACITY) + i] = encode_opacity(spark_lod_opacity(p[3]));
            for (k, ln) in [low_half(s[1]), high_half(s[1]), low_half(s[2])].into_iter().enumerate() {
                raw[at(P_SCALE) + 3 * i + k] = if ln <= -60000.0 {
                    0
                } else {
                    1 + if ln_hi > ln_lo { (((ln - ln_lo) / (ln_hi - ln_lo)) * 254.0).round() as u8 } else { 0 }
                };
            }
            let q = crate::athc::decode_quaternion(s[0]);
            let rot = crate::splat_encode::encode_quat_oct101012(q);
            raw[at(P_ROTATION) + 4 * i..at(P_ROTATION) + 4 * i + 4].copy_from_slice(&rot.to_le_bytes());
            let code = encode_normal_bits(unpack_normal(b.normals[i]), nb);
            raw[at(P_NORMAL) + 3 * i..at(P_NORMAL) + 3 * i + 3].copy_from_slice(&code.to_le_bytes()[..3]);
            if curvature && range > 0.0 {
                let c = &b.curvature[2 * i..2 * i + 2];
                for (k, v) in [low_half(c[0]), high_half(c[0]), low_half(c[1])].into_iter().enumerate() {
                    let m = if v.is_finite() {
                        ((127.0 * (1.0 + v.abs() / CURVATURE_FLOOR).log2()) / range).round() as i32
                    } else {
                        0
                    };
                    raw[at(P_CURVATURE) + 3 * i + k] = (if v < 0.0 { -m } else { m }) as u8;
                }
            }
            let lobes = if b.lobes.is_empty() { [0; 3] } else { [b.lobes[3 * i], b.lobes[3 * i + 1], b.lobes[3 * i + 2]] };
            let key = [b.pbr[i], lobes[0], lobes[1], lobes[2]];
            let m = match material_of.get(&key) {
                Some(&m) => m,
                None => {
                    let m = (materials.len() / 4) as u32;
                    if m > 0xffff {
                        bail!("compact: over 65536 materials in a piece");
                    }
                    materials.extend_from_slice(&key);
                    material_of.insert(key, m);
                    m
                }
            };
            raw[at(P_MATERIAL) + 2 * i..at(P_MATERIAL) + 2 * i + 2].copy_from_slice(&(m as u16).to_le_bytes());
            // The transfer: athenea's joint CPCA at `levels`.
            for v in 0..112 {
                let w = b.transfer[56 * i + v / 2];
                x[v] = if v % 2 == 0 { low_half(w) } else { high_half(w) };
            }
            let (cluster, qs) = basis.encode(&x, o.levels);
            used[cluster as usize] = true;
            raw[at(P_CLUSTER) + i] = cluster as u8;
            for blk in 0..3 {
                let mut packed = 0u32;
                for k in 0..JOINT_K {
                    packed |= ((qs[blk * JOINT_K + k] & 0xf) as u32) << (4 * k);
                }
                let off = at(P_WEIGHTS + blk) + 4 * i;
                raw[off..off + 4].copy_from_slice(&packed.to_le_bytes());
            }
            let back = basis.decode(cluster, &qs, o.levels);
            let mut first = 0;
            for blk in 0..3 {
                for v in first..first + JOINT_BLOCKS[blk] {
                    err2[blk] += ((back[v] - x[v]) as f64).powi(2);
                    ref2[blk] += (x[v] as f64).powi(2);
                }
                first += JOINT_BLOCKS[blk];
            }
            raw[at(P_SHADOW) + 2 * i..at(P_SHADOW) + 2 * i + 2].copy_from_slice(&(labels[node] as u16).to_le_bytes());
            let tree = lod_tree_words(b, i);
            for (k, w) in tree.iter().enumerate() {
                raw[at(P_LOD_TREE) + 16 * i + 4 * k..at(P_LOD_TREE) + 16 * i + 4 * k + 4].copy_from_slice(&w.to_le_bytes());
            }
            if b.lod_error.len() == 2 * n {
                for k in 0..2 {
                    let off = at(P_LOD_ERROR) + 8 * i + 4 * k;
                    raw[off..off + 4].copy_from_slice(&b.lod_error[2 * i + k].to_le_bytes());
                }
            }
            node += 1;
        }
        out_pages.push(raw);
        heads.push(head);
    }
    let tables = CompactTables {
        codebook: codebook.iter().flat_map(mask_words).collect(),
        materials,
        basis: basis.table_f16(),
        clusters: basis.clusters as u32,
        normal_bits: nb,
        weight_bits: 4,
        linear: false,
        curvature,
        asset_basis: false,
    };
    let stats = CompactStats {
        nodes: total,
        pages: pages.len(),
        materials: tables.materials.len() / 4,
        distinct_masks: distinct.len(),
        codebook_entries: tables.entries(),
        hamming,
        clusters_used: used.iter().filter(|&&u| u).count(),
        transfer_rel_mse: err2.iter().sum::<f64>() / ref2.iter().sum::<f64>().max(1e-30),
        transfer_rel_mse_blocks: [0, 1, 2].map(|k| err2[k] / ref2[k].max(1e-30)),
        table_bytes: [4 * tables.codebook.len(), 4 * tables.materials.len(), 4 * tables.basis.len()],
        seconds: t.elapsed().as_secs_f32(),
    };
    Ok(CompactPages { tables, pages: out_pages, heads, stats })
}

/// One node of a compact page, decoded as compact.slang reads it (tests).
#[derive(Clone, Debug, PartialEq)]
pub struct CompactNode {
    pub center: [f32; 3],
    pub scales: [f32; 3],
    pub rgb: [f32; 3],
    pub opacity: f32,
    pub rotation: u32,
    pub normal: [f64; 3],
    pub curvature: [f32; 3],
    pub material: [u32; 4],
    pub cluster: u32,
    pub weights: [i32; 24],
    pub cells: [u32; 8],
    pub lod_tree: [u32; 4],
    pub lod_error: [u32; 2],
}

/// Node i of a page section (`raw`, n nodes, `head`) with its piece's tables.
pub fn decode_node(raw: &[u8], n: usize, head: &[u32; HEAD_WORDS], tables: &CompactTables, i: usize) -> CompactNode {
    let field = |k: usize| -> u32 {
        let w = PLANE_WIDTHS[k] as usize;
        let at = plane_offset(k, n) + i * w;
        let mut v = 0u32;
        for b in 0..w.min(4) {
            v |= (raw[at + b] as u32) << (8 * b);
        }
        v
    };
    let word = |k: usize, j: usize| -> u32 {
        let at = plane_offset(k, n) + i * PLANE_WIDTHS[k] as usize + 4 * j;
        u32::from_le_bytes(raw[at..at + 4].try_into().unwrap())
    };
    let f = |k: usize| f32::from_bits(head[k]);
    let center = [0, 1, 2].map(|k| f(k) + field(P_POS_X + k) as f32 * ((f(3 + k) - f(k)) / 65535.0));
    let s = field(P_SCALE);
    let step = (f(7) - f(6)) / 254.0;
    let scales = [0, 1, 2].map(|k| {
        let c = (s >> (8 * k)) & 0xff;
        if c == 0 { 0.0 } else { (f(6) + (c - 1) as f32 * step).exp() }
    });
    let a = field(P_ALBEDO);
    let cw = field(P_CURVATURE);
    let curvature = [0, 1, 2].map(|k| {
        let c = (((cw >> (8 * k)) & 0xff) as u8) as i8 as i32;
        let m = CURVATURE_FLOOR * (2f32.powf(c.abs() as f32 / 127.0 * f(8)) - 1.0);
        if c < 0 { -m } else { m }
    });
    let m = field(P_MATERIAL) as usize * 4;
    let e = field(P_SHADOW) as usize * 8;
    let mut weights = [0i32; 24];
    for blk in 0..3 {
        let w = field(P_WEIGHTS + blk);
        for k in 0..JOINT_K {
            weights[blk * JOINT_K + k] = (((w >> (4 * k)) & 0xf) << 28) as i32 >> 28;
        }
    }
    CompactNode {
        center,
        scales,
        rgb: [a & 0xff, (a >> 8) & 0xff, (a >> 16) & 0xff].map(|v| v as f32 / 255.0),
        opacity: decode_opacity(field(P_OPACITY) as u8),
        rotation: field(P_ROTATION),
        normal: decode_normal_bits(field(P_NORMAL), tables.normal_bits),
        curvature,
        material: [tables.materials[m], tables.materials[m + 1], tables.materials[m + 2], tables.materials[m + 3]],
        cluster: field(P_CLUSTER),
        weights,
        cells: tables.codebook[e..e + 8].try_into().unwrap(),
        lod_tree: [0, 1, 2, 3].map(|j| word(P_LOD_TREE, j)),
        lod_error: [word(P_LOD_ERROR, 0), word(P_LOD_ERROR, 1)],
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::athc_v3::{parse_v3, read_v3, write_v3_paged, write_v3_paged_compact, SectionId, COMPRESSION_GZIP, ENCODING_BYTE_PLANES};

    /// A joint basis of athenea's shape with `c` clusters of deterministic values.
    pub(crate) fn sample_basis(c: usize) -> JointBasis {
        let mut r = 12345u32;
        let mut rand = move || {
            r = r.wrapping_mul(1664525).wrapping_add(1013904223);
            (r >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let blocks = |per: usize| -> [Vec<f32>; 3] { JOINT_BLOCKS.map(|d| vec![0.0; c * d * per]) };
        let mut b = JointBasis {
            clusters: c,
            centroids: vec![0.0; c * 112],
            energy_norm: [1.0, 1.0, 1.0],
            mean: blocks(1),
            dir: blocks(JOINT_K),
            scale: JOINT_BLOCKS.map(|_| vec![0.0; c * JOINT_K]),
        };
        b.centroids.iter_mut().for_each(|v| *v = 0.2 * rand());
        for k in 0..3 {
            b.mean[k].iter_mut().for_each(|v| *v = 0.05 * rand());
            b.dir[k].iter_mut().for_each(|v| *v = rand());
            b.scale[k].iter_mut().for_each(|v| *v = 0.2 + 0.1 * rand());
        }
        b
    }

    #[test]
    fn joint_encode_rounds_to_the_levels_and_decodes_on_the_int8_grid() {
        let basis = sample_basis(4);
        let x: Vec<f32> = (0..112).map(|v| 0.05 * (v as f32 * 0.3).sin()).collect();
        let (c, q) = basis.encode(&x, 7);
        assert!(c < 4);
        assert!(q.iter().all(|&w| (-7..=7).contains(&w)));
        let (c8, q8) = basis.encode(&x, 127);
        assert_eq!(c, c8);
        // The int4 weights are the int8 ones on a coarser grid.
        for k in 0..24 {
            assert!((q[k] as f32 * 127.0 / 7.0 - q8[k] as f32).abs() <= 127.0 / 14.0 + 1.0, "{k} {} {}", q[k], q8[k]);
        }
        // The f16 table holds mean then the scaled directions.
        let t = basis.table_f16();
        assert_eq!(t.len(), 4 * CLUSTER_FLOATS / 2);
        let half = |f: usize| f16::from_bits(if f % 2 == 0 { t[f / 2] as u16 } else { (t[f / 2] >> 16) as u16 }).to_f32();
        let c1 = CLUSTER_FLOATS;
        assert!((half(c1) - basis.mean[0][16]).abs() < 1e-3);
        assert!((half(c1 + 16) - basis.dir[0][(16) * JOINT_K] * basis.scale[0][JOINT_K] / 127.0).abs() < 1e-5);
    }

    #[test]
    fn normal_and_opacity_codes_round_trip() {
        for n in [[0.0f32, 0.0, 1.0], [0.6, -0.48, -0.64], [-1.0, 0.0, 0.0]] {
            let d = decode_normal_bits(encode_normal_bits(n, 12), 12);
            let dot: f64 = (0..3).map(|k| d[k] * n[k] as f64).sum();
            assert!(dot > 0.99999, "{n:?} {d:?}");
        }
        for a in [0.0f32, 0.01, 0.5, 1.0, 1.5, 2.0] {
            assert!((decode_opacity(encode_opacity(a)) - a).abs() < 0.02 + 0.06 * a.sqrt() * (a <= 1.0) as u8 as f32, "{a}");
        }
    }

    /// The fourth tier of a paged tree: written after tiers 1-3 without
    /// touching them, one run over consecutive pages, its tables among the
    /// file's tables, each node decoding to what the tree holds.
    #[test]
    fn a_paged_tree_carries_its_compact_tier() {
        let (paged, file) = crate::athc_pages::tests::sample_relit_paged(1500, 400);
        let basis = sample_basis(6);
        let c = build_compact(&file.chunks, false, &basis, &CompactOptions { threads: 2, ..Default::default() }).unwrap();
        assert_eq!(c.stats.nodes, file.chunks.iter().map(|b| b.n).sum::<usize>());
        // Few distinct masks: kept exactly.
        assert!(c.stats.codebook_entries <= 40 && c.stats.hamming[3] == 0.0);
        assert!(c.stats.materials <= 15);
        let enc = |_: SectionId| ENCODING_BYTE_PLANES;
        let (plain, _) = write_v3_paged(&file, &paged.rows, COMPRESSION_GZIP, &enc, None, None).unwrap();
        let (bytes, _) = write_v3_paged_compact(&file, &paged.rows, COMPRESSION_GZIP, &enc, None, None, &c).unwrap();
        let (lp, layout) = (parse_v3(&plain).unwrap(), parse_v3(&bytes).unwrap());
        let k4 = layout.sections.len() - 1;
        assert_eq!(layout.sections[k4].id, SectionId::Compact);
        assert_eq!(layout.sections[k4].tier, 4);
        assert_eq!(&layout.sections[..k4], &lp.sections[..]);
        // Tiers 1-3: the same stored bytes, page for page.
        for (a, b) in lp.blocks.iter().zip(&layout.blocks) {
            for k in 0..k4 {
                assert_eq!(a.spans[k].stored, b.spans[k].stored);
                let sa = &plain[a.spans[k].offset as usize..][..a.spans[k].stored as usize];
                let sb = &bytes[b.spans[k].offset as usize..][..b.spans[k].stored as usize];
                assert_eq!(sa, sb);
            }
        }
        // The whole file reads as the same tree.
        let (fa, fb) = (read_v3(&plain).unwrap(), read_v3(&bytes).unwrap());
        assert_eq!(fa.chunks.len(), fb.chunks.len());
        for (a, b) in fa.chunks.iter().zip(&fb.chunks) {
            assert_eq!((&a.positions, &a.shape, &a.tree, &a.transfer), (&b.positions, &b.shape, &b.tree, &b.transfer));
        }
        // Tier 4 after tier 3, page after page: one run.
        let last3 = layout.blocks.iter().map(|b| b.spans[k4 - 1].offset + b.spans[k4 - 1].stored as u64).max().unwrap();
        for (p, b) in layout.blocks.iter().enumerate() {
            assert!(b.spans[k4].offset >= last3);
            if p > 0 {
                let a = &layout.blocks[p - 1].spans[k4];
                assert_eq!(b.spans[k4].offset, a.offset + a.stored as u64);
            }
            assert_eq!(layout.pages[p].tier4_bytes, b.spans[k4].stored);
            assert_eq!(layout.pages[p].compact, c.heads[p]);
            assert_eq!(b.spans[k4].raw as usize, b.n as usize * NODE_BYTES);
            assert_eq!(layout.pages[p].tier_bytes, lp.pages[p].tier_bytes);
        }
        // The tables, before the data.
        let [at, n] = layout.compact_table;
        assert!(at > 0 && at + n <= crate::athc_v3::tables_bytes(&bytes).unwrap());
        let tables = CompactTables::from_bytes(&bytes[at as usize..(at + n) as usize]).unwrap();
        assert_eq!(tables, c.tables);
        // Every node as compact.slang reads it.
        for (p, block) in file.chunks.iter().enumerate() {
            let b = &layout.blocks[p];
            let span = b.spans[k4];
            let raw = crate::athc_v3::gunzip(&bytes[span.offset as usize..][..span.stored as usize]).unwrap();
            let head = &layout.pages[p].compact;
            let lo: Vec<f32> = (0..3).map(|k| f32::from_bits(head[k])).collect();
            let hi: Vec<f32> = (0..3).map(|k| f32::from_bits(head[3 + k])).collect();
            for i in 0..block.n {
                let node = decode_node(&raw, block.n, head, &tables, i);
                for k in 0..3 {
                    assert!((node.center[k] - block.positions[4 * i + k]).abs() <= (hi[k] - lo[k]) / 65535.0 + 1e-6);
                }
                assert_eq!(node.lod_tree, lod_tree_words(block, i));
                assert_eq!(node.lod_error.to_vec(), block.lod_error[2 * i..2 * i + 2].to_vec());
                assert_eq!(node.material, [block.pbr[i], block.lobes[3 * i], block.lobes[3 * i + 1], block.lobes[3 * i + 2]]);
                assert_eq!(node.cells.to_vec(), block.shadow_bits[8 * i..8 * i + 8].to_vec());
                let mut x = [0f32; 112];
                for v in 0..112 {
                    let w = block.transfer[56 * i + v / 2];
                    x[v] = if v % 2 == 0 { low_half(w) } else { high_half(w) };
                }
                let (cl, q) = basis.encode(&x, 7);
                assert_eq!((node.cluster, node.weights), (cl, q));
                let n = unpack_normal(block.normals[i]);
                let dot: f64 = (0..3).map(|k| node.normal[k] * n[k] as f64).sum();
                assert!(dot > 0.99999);
                let s = &block.shape[4 * i..4 * i + 4];
                let scale0 = low_half(s[1]).exp();
                assert!((node.scales[0] / scale0 - 1.0).abs() < 0.05, "{} {}", node.scales[0], scale0);
                let k0 = low_half(block.curvature[2 * i]);
                assert!((node.curvature[0] - k0).abs() <= 0.03 * k0.abs() + 0.01, "{} {}", node.curvature[0], k0);
            }
        }
    }
}

#[cfg(test)]
mod fixture {
    /// test/fixtures/athc/compact_pages.athc: a small paged tree with its
    /// compact tier (synthetic streams and basis), for the Dawn test of the
    /// paged compact draw (test/gpu/compactPagesDraw.test.ts). Regenerate:
    /// cargo test -p spark-lib --lib write_compact_fixture -- --ignored
    #[test]
    #[ignore]
    fn write_compact_fixture() {
        use super::*;
        use crate::athc_v3::{write_v3_paged_compact, COMPRESSION_GZIP};
        let (paged, file) = crate::athc_pages::tests::sample_relit_paged_spread(1200, 300, 3.0);
        let rows = paged.rows.clone();
        let basis = tests::sample_basis(8);
        let c = build_compact(&file.chunks, false, &basis, &CompactOptions { threads: 2, ..Default::default() }).unwrap();
        let (bytes, _) = write_v3_paged_compact(&file, &rows, COMPRESSION_GZIP, &|_| 0, None, None, &c).unwrap();
        let out = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test/fixtures/athc/compact_pages.athc");
        std::fs::write(out, &bytes).unwrap();
        eprintln!("{} bytes, {} pages, {} nodes", bytes.len(), rows.len(), c.stats.nodes);
    }
}
