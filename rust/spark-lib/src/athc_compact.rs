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

/// Bytes a node in a `CMPC` section (v2, thread CQ: the GPU's 40, then the
/// traversal's 16).
pub const NODE_BYTES: usize = 64;
/// v2's (thread CQ): no cone, no occlusion index.
pub const NODE_BYTES_V2: usize = 56;
/// Bytes a node in each plane, in the order they are stored: position x, y,
/// z (u16 in the page's box), albedo (3 x u8 sRGB), opacity (the code), ln
/// scales (3 x u8 in the page's range, 0 = exactly 0), rotation (the ext
/// quaternion word), normal (octahedral, 12 bits an axis), curvature (3 x
/// signed log), material (u16), CPCA cluster (u8), the three blocks' 8 int4
/// weights (a word each), shadow dictionary index (u16) -- the GPU's 40 --
/// then the traversal's 16, one byte a plane (`TraversalNode`): the centre's
/// correction (3 x i16, low byte then high), the children's page (u16, from
/// the node's own page), the children word (u32: place, count, flags) and
/// LODE's eps and rho (f16).
///
/// v3 (thread CS) appends 8 one-byte planes to the traversal's: LODE's cone
/// word (theta f16 | the octahedral 8+8 axis; NO_CONE: never culled; a leaf
/// of a closed part carries its own, theta 0), athenea's occlusion index
/// (entry + 1, 0 none; the codebook in the asset table) and 2 bytes spare.
/// The first 31 planes are v2's, at the same offsets.
pub const PLANE_WIDTHS: [u32; 39] = [
    2, 2, 2, 3, 1, 3, 4, 3, 3, 2, 1, 4, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];
/// Planes of a v2 node.
pub const PLANES_V2: usize = 31;
pub const PLANES: usize = PLANE_WIDTHS.len();
/// The planes the GPU keeps (the rest are the traversal's).
pub const GPU_PLANES: usize = 15;
/// Bytes a node the GPU keeps, and the traversal.
pub const GPU_NODE_BYTES: usize = 40;
pub const TRAVERSAL_NODE_BYTES: usize = 24;
pub const TRAVERSAL_NODE_BYTES_V2: usize = 16;
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
/// The traversal's planes (one byte each): centre correction x lo, x hi, y
/// lo, y hi, z lo, z hi; the children's page lo, hi; the children word's
/// four bytes; eps lo, hi; rho lo, hi.
pub const P_CENTRE: usize = 15;
pub const P_CHILD_PAGE: usize = 21;
pub const P_KIDS: usize = 23;
pub const P_EPS: usize = 27;
pub const P_RHO: usize = 29;
/// v3: the cone word's 4 bytes, the occlusion index's 2.
pub const P_CONE: usize = 31;
pub const P_OCCLUSION: usize = 35;
/// The children word: place (bits 0-11) | count (12-23) | flags (24-31).
pub const KIDS_PLACE_BITS: u32 = 12;
pub const KIDS_COUNT_BITS: u32 = 12;
/// A merge without LODE: the eps half is its LoD size (the size cut).
pub const KIDS_SIZE_MODE: u32 = 1 << 24;
/// Words of a page head (the page table row's words 17..29): box lo, hi
/// (f32 x 3 each), ln scale lo, hi, the curvature's range (log2(1 +
/// max / floor)), then three words a reader fills (compact.slang's chunk
/// words 9, 10: the codebook's and the materials' word offsets).
pub const HEAD_WORDS: usize = 12;
/// The blob of the tables (`CompactTables::to_bytes`).
pub const TABLE_MAGIC: u32 = u32::from_le_bytes(*b"CMPT");
/// v2 (thread CQ): the traversal's 16 bytes, and tables that may live in an
/// asset table (`CompactAsset`) the asset's pieces share.
pub const TABLE_VERSION: u32 = 3;
const TABLE_FIXED_WORDS: usize = 24;
/// The piece's tables flags.
pub const TABLE_LINEAR: u32 = 1;
pub const TABLE_CURVATURE: u32 = 2;
/// The basis, the shadow dictionary, the materials: in the asset table.
pub const TABLE_ASSET_BASIS: u32 = 4;
pub const TABLE_ASSET_CODEBOOK: u32 = 8;
pub const TABLE_ASSET_MATERIALS: u32 = 16;
/// v3: the occlusion codebook (and the asset's box) in the asset table.
pub const TABLE_ASSET_OCCLUSION: u32 = 32;
/// The asset table (`CompactAsset::to_bytes`, a sidecar `.cmpa`).
pub const ASSET_MAGIC: u32 = u32::from_le_bytes(*b"CMPA");
pub const ASSET_VERSION: u32 = 1;
const ASSET_FIXED_WORDS: usize = 16;
/// Bytes of the asset's name a piece keeps (UTF-8, zero padded).
pub const ASSET_NAME_BYTES: usize = 64;
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

/// FNV-1a, 64 bits: an asset table's id.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn words_of(b: &[u8]) -> impl Fn(usize) -> Result<u32> + '_ {
    move |k: usize| -> Result<u32> {
        Ok(u32::from_le_bytes(b.get(4 * k..4 * k + 4).ok_or_else(|| anyhow!("compact tables too short"))?.try_into().unwrap()))
    }
}

/// An asset's compact tables, shared by its pieces (thread CQ): athenea's
/// joint CPCA basis once, and -- when its pieces were put together by
/// `athc-convert --compact-asset` -- one shadow dictionary and one material
/// table. A sidecar file (`<asset>.cmpa`); a piece names it and holds its id.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompactAsset {
    /// The basis in f16 (`JointBasis::table_f16`), empty if the pieces hold
    /// their own.
    pub basis: Vec<u32>,
    pub clusters: u32,
    /// The shadow dictionary (8 words an entry), empty: per piece.
    pub codebook: Vec<u32>,
    /// The materials (4 words an entry), empty: per piece.
    pub materials: Vec<u32>,
    /// athenea's occlusion codebook (8 words an entry; thread CS), the same
    /// in every piece, and the asset's box (min xyz, max xyz) its rule tests
    /// the eye against; empty: none.
    pub occlusion: Vec<u32>,
    pub occlusion_box: [f32; 6],
}

impl CompactAsset {
    pub fn flags(&self) -> u32 {
        (!self.basis.is_empty() as u32)
            | (!self.codebook.is_empty() as u32) << 1
            | (!self.materials.is_empty() as u32) << 2
            | (!self.occlusion.is_empty() as u32) << 3
    }

    /// The blob: 16 fixed words (magic, version, words, flags, clusters,
    /// entries, materials, basis at, basis words, codebook at, materials at,
    /// id lo, id hi, 0 0 0), then the arrays at 16-word boundaries. The id
    /// is FNV-1a 64 of the blob with its two words 0.
    pub fn to_bytes(&self) -> Vec<u8> {
        let basis_at = ASSET_FIXED_WORDS;
        let codebook_at = (basis_at + self.basis.len()).div_ceil(16) * 16;
        let materials_at = (codebook_at + self.codebook.len()).div_ceil(16) * 16;
        let occlusion_at = (materials_at + self.materials.len()).div_ceil(16) * 16;
        let occlusion_words = if self.occlusion.is_empty() { 0 } else { self.occlusion.len() + 6 };
        let total = (occlusion_at + occlusion_words).div_ceil(16) * 16;
        let mut w = vec![0u32; total];
        if !self.occlusion.is_empty() {
            // Words 13, 14: the occlusion codebook's offset and entries, the
            // box's 6 floats after it (v1 kept them 0: no occlusion).
            w[13] = occlusion_at as u32;
            w[14] = (self.occlusion.len() / 8) as u32;
            w[occlusion_at..occlusion_at + self.occlusion.len()].copy_from_slice(&self.occlusion);
            for k in 0..6 {
                w[occlusion_at + self.occlusion.len() + k] = self.occlusion_box[k].to_bits();
            }
        }
        w[..13].copy_from_slice(&[
            ASSET_MAGIC,
            ASSET_VERSION,
            total as u32,
            self.flags(),
            self.clusters,
            (self.codebook.len() / 8) as u32,
            (self.materials.len() / 4) as u32,
            basis_at as u32,
            self.basis.len() as u32,
            codebook_at as u32,
            materials_at as u32,
            0,
            0,
        ]);
        w[basis_at..basis_at + self.basis.len()].copy_from_slice(&self.basis);
        w[codebook_at..codebook_at + self.codebook.len()].copy_from_slice(&self.codebook);
        w[materials_at..materials_at + self.materials.len()].copy_from_slice(&self.materials);
        let mut bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let id = fnv1a64(&bytes);
        bytes[44..48].copy_from_slice(&(id as u32).to_le_bytes());
        bytes[48..52].copy_from_slice(&((id >> 32) as u32).to_le_bytes());
        bytes
    }

    /// The asset's id (as its blob holds it).
    pub fn id(&self) -> u64 {
        let b = self.to_bytes();
        u32::from_le_bytes(b[44..48].try_into().unwrap()) as u64 | (u32::from_le_bytes(b[48..52].try_into().unwrap()) as u64) << 32
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        let w = words_of(b);
        if w(0)? != ASSET_MAGIC || w(1)? != ASSET_VERSION {
            bail!("not a compact asset table (CMPA v{ASSET_VERSION})");
        }
        let total = w(2)? as usize;
        if b.len() < 4 * total {
            bail!("compact asset table: {} bytes of {}", b.len(), 4 * total);
        }
        let mut zeroed = b[..4 * total].to_vec();
        zeroed[44..52].fill(0);
        let id = w(11)? as u64 | (w(12)? as u64) << 32;
        if fnv1a64(&zeroed) != id {
            bail!("compact asset table: its id does not match its bytes");
        }
        let words = |at: u32, n: usize| -> Result<Vec<u32>> { (0..n).map(|k| w(at as usize + k)).collect() };
        Ok(Self {
            clusters: w(4)?,
            basis: words(w(7)?, w(8)? as usize)?,
            codebook: words(w(9)?, 8 * w(5)? as usize)?,
            materials: words(w(10)?, 4 * w(6)? as usize)?,
            occlusion: if w(3)? & 8 != 0 { words(w(13)?, 8 * w(14)? as usize)? } else { Vec::new() },
            occlusion_box: if w(3)? & 8 != 0 {
                let at = w(13)? as usize + 8 * w(14)? as usize;
                let mut bx = [0f32; 6];
                for (k, v) in bx.iter_mut().enumerate() {
                    *v = f32::from_bits(w(at + k)?);
                }
                bx
            } else {
                [0.0; 6]
            },
        })
    }
}

/// A piece's tables, global to its pages (the blob among the file's tables).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompactTables {
    /// The shadow dictionary: 8 words an entry (empty: the asset's).
    pub codebook: Vec<u32>,
    /// The materials: pbr then 3 lobes words, 4 words an entry (empty: the asset's).
    pub materials: Vec<u32>,
    /// athenea's basis, f16 (`JointBasis::table_f16`; empty: the asset's).
    pub basis: Vec<u32>,
    pub clusters: u32,
    pub normal_bits: u32,
    pub weight_bits: u32,
    /// The albedo is linear light (else sRGB; this writer stores sRGB).
    pub linear: bool,
    pub curvature: bool,
    /// What the asset table holds (`TABLE_ASSET_*`), its id and its name
    /// (where a reader finds it beside the piece); 0 and "" without.
    pub asset_flags: u32,
    pub asset_id: u64,
    pub asset_name: String,
}

impl CompactTables {
    pub fn entries(&self) -> usize {
        self.codebook.len() / 8
    }

    /// The tables as a reader uses them: the asset's arrays in place of the
    /// ones the piece leaves to it (its id checked).
    pub fn resolve(&self, asset: Option<&CompactAsset>) -> Result<CompactTables> {
        let mut t = self.clone();
        if self.asset_flags == 0 {
            return Ok(t);
        }
        let a = asset.ok_or_else(|| anyhow!("compact tables: the piece's arrays are in asset table {:?}", self.asset_name))?;
        if a.id() != self.asset_id {
            bail!("compact tables: asset table {:016x}, the piece wants {:016x}", a.id(), self.asset_id);
        }
        if self.asset_flags & TABLE_ASSET_BASIS != 0 {
            if a.basis.is_empty() || a.clusters != self.clusters {
                bail!("compact tables: the asset has no basis of {} clusters", self.clusters);
            }
            t.basis = a.basis.clone();
        }
        if self.asset_flags & TABLE_ASSET_CODEBOOK != 0 {
            t.codebook = a.codebook.clone();
        }
        if self.asset_flags & TABLE_ASSET_MATERIALS != 0 {
            t.materials = a.materials.clone();
        }
        t.asset_flags = 0;
        Ok(t)
    }

    /// The blob (v2): 24 fixed words (magic, version, node bytes, planes,
    /// head words, entries, materials, clusters, normal bits, weight bits,
    /// flags, codebook at, materials at, basis at, basis words, total, asset
    /// id lo, hi, asset name at, its bytes, 0 x 4), the plane widths, then
    /// the asset's name and the arrays at 16-word boundaries.
    pub fn to_bytes(&self) -> Vec<u8> {
        let head = TABLE_FIXED_WORDS + PLANES;
        let name_at = head.div_ceil(16) * 16;
        let name = self.asset_name.as_bytes();
        let name_words = if self.asset_flags == 0 { 0 } else { ASSET_NAME_BYTES / 4 };
        let codebook_at = (name_at + name_words).div_ceil(16) * 16;
        let materials_at = (codebook_at + self.codebook.len()).div_ceil(16) * 16;
        let basis_at = (materials_at + self.materials.len()).div_ceil(16) * 16;
        let total = (basis_at + self.basis.len()).div_ceil(16) * 16;
        let mut w = vec![0u32; total];
        let flags = ((self.linear as u32) * TABLE_LINEAR) | ((self.curvature as u32) * TABLE_CURVATURE) | self.asset_flags;
        w[..20].copy_from_slice(&[
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
            flags,
            codebook_at as u32,
            materials_at as u32,
            basis_at as u32,
            self.basis.len() as u32,
            total as u32,
            self.asset_id as u32,
            (self.asset_id >> 32) as u32,
            name_at as u32,
            if name_words > 0 { name.len().min(ASSET_NAME_BYTES) as u32 } else { 0 },
        ]);
        w[TABLE_FIXED_WORDS..head].copy_from_slice(&PLANE_WIDTHS);
        w[codebook_at..codebook_at + self.codebook.len()].copy_from_slice(&self.codebook);
        w[materials_at..materials_at + self.materials.len()].copy_from_slice(&self.materials);
        w[basis_at..basis_at + self.basis.len()].copy_from_slice(&self.basis);
        let mut bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        if name_words > 0 {
            let n = name.len().min(ASSET_NAME_BYTES);
            bytes[4 * name_at..4 * name_at + n].copy_from_slice(&name[..n]);
        }
        bytes
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        let w = words_of(b);
        if w(0)? != TABLE_MAGIC || !(w(1)? == TABLE_VERSION || w(1)? == 2) {
            bail!("not a compact table (CMPT v2 or v{TABLE_VERSION}; v1 files: athc-convert --compact-rebase)");
        }
        let (bytes, planes) = if w(1)? == 2 { (NODE_BYTES_V2, PLANES_V2) } else { (NODE_BYTES, PLANES) };
        if w(2)? as usize != bytes || w(3)? as usize != planes || w(4)? as usize != HEAD_WORDS {
            bail!("compact tables of another layout");
        }
        let words = |at: u32, n: usize| -> Result<Vec<u32>> { (0..n).map(|k| w(at as usize + k)).collect() };
        let flags = w(10)?;
        let (name_at, name_len) = (w(18)? as usize, w(19)? as usize);
        let name = b.get(4 * name_at..4 * name_at + name_len).ok_or_else(|| anyhow!("compact tables: the asset name"))?;
        Ok(Self {
            codebook: words(w(11)?, 8 * w(5)? as usize)?,
            materials: words(w(12)?, 4 * w(6)? as usize)?,
            basis: words(w(13)?, w(14)? as usize)?,
            clusters: w(7)?,
            normal_bits: w(8)?,
            weight_bits: w(9)?,
            linear: flags & TABLE_LINEAR != 0,
            curvature: flags & TABLE_CURVATURE != 0,
            asset_flags: flags & (TABLE_ASSET_BASIS | TABLE_ASSET_CODEBOOK | TABLE_ASSET_MATERIALS | TABLE_ASSET_OCCLUSION),
            asset_id: w(16)? as u64 | (w(17)? as u64) << 32,
            asset_name: String::from_utf8_lossy(name).into_owned(),
        })
    }
}

// --- the traversal's words --------------------------------------------------

/// f16 bits in order (-0 just below +0): a correction is a difference of these.
fn f16_order(x: u16) -> i32 {
    if x & 0x8000 != 0 { -((x & 0x7fff) as i32) - 1 } else { x as i32 }
}

fn f16_from_order(v: i32) -> u16 {
    if v < 0 { 0x8000 | (-(v + 1)) as u16 } else { v as u16 }
}

/// A compact position decoded (u16 `q` in [lo, hi]), as compact.slang and
/// the traversal read it.
#[inline]
pub fn compact_position(q: u16, lo: f32, hi: f32) -> f32 {
    lo + q as f32 * ((hi - lo) / 65535.0)
}

/// What the traversal keeps of a node: exactly what the error cut reads of
/// the LoD tree words and LODE (spark-rs `lod_tree.rs`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TraversalNode {
    /// The centre, f16 bits as the decoder rounds it (a leaf's: its compact
    /// position's; the cut never reads it).
    pub centre: [u16; 3],
    /// The children's page, their first's place in it, their count (0 a leaf).
    pub child_page: u32,
    pub place: u32,
    pub count: u32,
    /// LODE's eps and rho (f16 bits); a merge without LODE (`size_mode`):
    /// eps is its LoD size, rho 0.
    pub eps: u16,
    pub rho: u16,
    pub size_mode: bool,
    /// v3: LODE's cone word (NO_CONE_WORD: never culled; a leaf of a closed
    /// part: theta 0 about its normal), athenea's occlusion index (entry + 1).
    pub cone: u32,
    pub occlusion: u16,
}

/// Bytes of plane `first`.. of the traversal's, its planes from `base`.
fn get_bytes_at(raw: &[u8], base: usize, first: usize, k: usize, n: usize, i: usize) -> u32 {
    let mut v = 0u32;
    for b in 0..k {
        v |= (raw[base + (first + b - P_CENTRE) * n + i] as u32) << (8 * b);
    }
    v
}

/// Bytes a node of the traversal's input (`traversal_input`).
pub const TRAVERSAL_INPUT_BYTES: usize = 6 + TRAVERSAL_NODE_BYTES;
pub const TRAVERSAL_INPUT_BYTES_V2: usize = 6 + TRAVERSAL_NODE_BYTES_V2;

/// What a reader hands the traversal of a page (spark-rs
/// `update_compact_lod_trees`): the position planes then the traversal's,
/// 22 bytes a node.
pub fn traversal_input(raw: &[u8], n: usize) -> Vec<u8> {
    let t = if raw.len() == n * NODE_BYTES_V2 { TRAVERSAL_NODE_BYTES_V2 } else { TRAVERSAL_NODE_BYTES };
    let mut v = raw[..6 * n].to_vec();
    v.extend_from_slice(&raw[plane_offset(P_CENTRE, n)..plane_offset(P_CENTRE, n) + t * n]);
    v
}

fn put_bytes(raw: &mut [u8], first: usize, k: usize, n: usize, i: usize, v: u32) {
    for b in 0..k {
        raw[plane_offset(first + b, n) + i] = (v >> (8 * b)) as u8;
    }
}

/// The compact position's three u16 of node i.
pub fn position_codes(raw: &[u8], n: usize, i: usize) -> [u16; 3] {
    [0, 1, 2].map(|k| {
        let at = plane_offset(P_POS_X + k, n) + 2 * i;
        u16::from_le_bytes([raw[at], raw[at + 1]])
    })
}

/// Node i of page `page` (its section `raw` of n nodes, its head) as the
/// traversal reads it.
pub fn traversal_node(raw: &[u8], n: usize, head: &[u32], page: u32, i: usize) -> TraversalNode {
    traversal_node_at(raw, plane_offset(P_CENTRE, n), n, head, page, i, raw.len() != n * NODE_BYTES_V2)
}

/// `traversal_node` of a page's `traversal_input` (v2's 22 bytes a node or v3's 30).
pub fn traversal_node_input(input: &[u8], n: usize, head: &[u32], page: u32, i: usize) -> TraversalNode {
    traversal_node_at(input, 6 * n, n, head, page, i, input.len() != n * TRAVERSAL_INPUT_BYTES_V2)
}

fn traversal_node_at(raw: &[u8], base: usize, n: usize, head: &[u32], page: u32, i: usize, v3: bool) -> TraversalNode {
    let get_bytes = |first: usize, k: usize, n: usize, i: usize| get_bytes_at(raw, base, first, k, n, i);
    let q = position_codes(raw, n, i);
    let mut centre = [0u16; 3];
    for k in 0..3 {
        let p = compact_position(q[k], f32::from_bits(head[k]), f32::from_bits(head[3 + k]));
        let d = get_bytes(P_CENTRE + 2 * k, 2, n, i) as u16 as i16 as i32;
        centre[k] = f16_from_order(f16_order(f16::from_f32(p).to_bits()) + d);
    }
    let kids = get_bytes(P_KIDS, 4, n, i);
    let count = (kids >> KIDS_PLACE_BITS) & ((1 << KIDS_COUNT_BITS) - 1);
    TraversalNode {
        centre,
        child_page: if count > 0 { (page + get_bytes(P_CHILD_PAGE, 2, n, i)) & 0xffff } else { 0 },
        place: kids & ((1 << KIDS_PLACE_BITS) - 1),
        count,
        eps: get_bytes(P_EPS, 2, n, i) as u16,
        rho: get_bytes(P_RHO, 2, n, i) as u16,
        size_mode: kids & KIDS_SIZE_MODE != 0,
        cone: if v3 { get_bytes(P_CONE, 4, n, i) } else { NO_CONE_WORD },
        occlusion: if v3 { get_bytes(P_OCCLUSION, 2, n, i) as u16 } else { 0 },
    }
}

/// A cone word that never culls (athc_lod_cut::NO_CONE).
pub const NO_CONE_WORD: u32 = crate::athc_lod_cut::NO_CONE as u32;

/// Writes node i's traversal planes into `raw` (its position planes already
/// written): from the LoD tree words the decoder gives it
/// (`lod_tree_words`) and its LODE words. Only what the error cut reads is
/// kept: a merge's centre, eps and rho (or its size, without LODE), its
/// children; a leaf only that it is one. LODE's back-face cone (theta, axis)
/// is not kept (a compact traversal does not cull back faces; athenea's
/// trees hold none: theta infinite).
pub fn put_traversal(
    raw: &mut [u8],
    n: usize,
    head: &[u32],
    page: u32,
    i: usize,
    tree: [u32; 4],
    lode: Option<[u32; 2]>,
    occlusion: u32,
) -> Result<()> {
    let count = tree[2] & 0xffff;
    // v3: the cone (a leaf's [0, 0] LODE: none) and the occlusion index.
    let cone = match lode {
        Some([0, 0]) | None => NO_CONE_WORD,
        Some(e) => e[1],
    };
    if raw.len() == n * NODE_BYTES {
        put_bytes(raw, P_CONE, 4, n, i, cone);
        if occlusion > 0xffff {
            bail!("compact traversal: occlusion index {occlusion} past 16 bits");
        }
        put_bytes(raw, P_OCCLUSION, 2, n, i, occlusion);
    }
    if count == 0 {
        return Ok(());
    }
    let (child_page, place) = (tree[3] >> 16, tree[3] & 0xffff);
    if place >= 1 << KIDS_PLACE_BITS || count >= 1 << KIDS_COUNT_BITS {
        bail!("compact traversal: children at place {place}, count {count}: the word holds 12 bits each");
    }
    let q = position_codes(raw, n, i);
    let centre = [tree[0] as u16, (tree[0] >> 16) as u16, tree[1] as u16];
    for k in 0..3 {
        let p = compact_position(q[k], f32::from_bits(head[k]), f32::from_bits(head[3 + k]));
        let d = f16_order(centre[k]) - f16_order(f16::from_f32(p).to_bits());
        if d < i16::MIN as i32 || d > i16::MAX as i32 {
            bail!("compact traversal: centre correction {d} past 16 bits");
        }
        put_bytes(raw, P_CENTRE + 2 * k, 2, n, i, d as i16 as u16 as u32);
    }
    put_bytes(raw, P_CHILD_PAGE, 2, n, i, child_page.wrapping_sub(page) & 0xffff);
    let e = lode.unwrap_or([0, 0]);
    let size_mode = e[0] == 0;
    put_bytes(raw, P_KIDS, 4, n, i, place | count << KIDS_PLACE_BITS | if size_mode { KIDS_SIZE_MODE } else { 0 });
    let (eps, rho) = if size_mode { (tree[1] >> 16, 0) } else { (e[0] & 0xffff, e[0] >> 16) };
    put_bytes(raw, P_EPS, 2, n, i, eps);
    put_bytes(raw, P_RHO, 2, n, i, rho);
    Ok(())
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
    /// The asset table the piece shares (thread CQ), and its name (where a
    /// reader finds it beside the piece). Its basis must be the piece's; its
    /// shadow dictionary and materials, when it has them, replace the
    /// piece's (every material of the piece must be in it).
    pub asset: Option<CompactAsset>,
    pub asset_name: String,
}

impl Default for CompactOptions {
    fn default() -> Self {
        Self {
            centroids: 1024,
            rounds: 12,
            threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
            normal_bits: 12,
            levels: 7,
            asset: None,
            asset_name: String::new(),
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
    /// Bytes of the tables blob: codebook, materials, basis (0: the asset's).
    pub table_bytes: [usize; 3],
    /// Nodes whose LODE had a back-face cone (theta finite), not kept.
    pub back_cones_dropped: usize,
    /// Merges without LODE (their size kept for the size cut).
    pub size_mode_merges: usize,
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
    let asset = o.asset.as_ref();
    let basis_f16 = basis.table_f16();
    if let Some(a) = asset {
        if !a.basis.is_empty() && (a.basis != basis_f16 || a.clusters as usize != basis.clusters) {
            bail!("compact: the asset table's basis is not this basis");
        }
    }
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
    let asset_codebook: Vec<Mask> = asset.map(|a| a.codebook.chunks_exact(8).map(mask_of).collect()).unwrap_or_default();
    let (codebook, labels): (Vec<Mask>, Vec<u32>) = if !asset_codebook.is_empty() {
        let mut labels = vec![0u32; masks.len()];
        crate::athc_shadow_vq::assign(&masks, &asset_codebook, o.threads, &mut labels);
        (asset_codebook.clone(), labels)
    } else if distinct.len() <= o.centroids {
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
    let asset_materials = asset.is_some_and(|a| !a.materials.is_empty());
    if let Some(a) = asset.filter(|_| asset_materials) {
        for (m, key) in a.materials.chunks_exact(4).enumerate() {
            material_of.entry(key.try_into().unwrap()).or_insert(m as u32);
        }
    }
    let (mut back_cones, mut size_mode) = (0usize, 0usize);
    let mut used_materials = std::collections::HashSet::new();

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
    for (page, b) in pages.iter().enumerate() {
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
                None if asset_materials => bail!("compact: a material of the piece is not in the asset table"),
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
            used_materials.insert(m);
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
            let lode = (b.lod_error.len() == 2 * n).then(|| [b.lod_error[2 * i], b.lod_error[2 * i + 1]]);
            if let Some(e) = lode {
                if e[0] != 0 && f16::from_bits(e[1] as u16).to_f32().is_finite() {
                    back_cones += 1;
                }
            }
            if tree[2] & 0xffff > 0 && lode.is_none_or(|e| e[0] == 0) {
                size_mode += 1;
            }
            let occlusion = b.occlusion.get(i).copied().unwrap_or(0);
            put_traversal(&mut raw, n, &head, page as u32, i, tree, lode, occlusion)?;
            node += 1;
        }
        out_pages.push(raw);
        heads.push(head);
    }
    let mut asset_flags = 0;
    if let Some(a) = asset {
        asset_flags |= if a.basis.is_empty() { 0 } else { TABLE_ASSET_BASIS };
        asset_flags |= if asset_codebook.is_empty() { 0 } else { TABLE_ASSET_CODEBOOK };
        asset_flags |= if asset_materials { TABLE_ASSET_MATERIALS } else { 0 };
    }
    let tables = CompactTables {
        codebook: if asset_flags & TABLE_ASSET_CODEBOOK != 0 { Vec::new() } else { codebook.iter().flat_map(mask_words).collect() },
        materials: if asset_materials { Vec::new() } else { materials },
        basis: if asset_flags & TABLE_ASSET_BASIS != 0 { Vec::new() } else { basis_f16 },
        clusters: basis.clusters as u32,
        normal_bits: nb,
        weight_bits: 4,
        linear: false,
        curvature,
        asset_flags,
        asset_id: if asset_flags != 0 { asset.unwrap().id() } else { 0 },
        asset_name: if asset_flags != 0 { o.asset_name.clone() } else { String::new() },
    };
    let stats = CompactStats {
        nodes: total,
        pages: pages.len(),
        materials: used_materials.len(),
        distinct_masks: distinct.len(),
        codebook_entries: codebook.len(),
        hamming,
        clusters_used: used.iter().filter(|&&u| u).count(),
        transfer_rel_mse: err2.iter().sum::<f64>() / ref2.iter().sum::<f64>().max(1e-30),
        transfer_rel_mse_blocks: [0, 1, 2].map(|k| err2[k] / ref2[k].max(1e-30)),
        table_bytes: [4 * tables.codebook.len(), 4 * tables.materials.len(), 4 * tables.basis.len()],
        back_cones_dropped: back_cones,
        size_mode_merges: size_mode,
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
    pub traversal: TraversalNode,
}

/// Node i of page `page`'s section (`raw`, n nodes, `head`) with its
/// piece's tables (resolved: `CompactTables::resolve`).
pub fn decode_node(raw: &[u8], n: usize, head: &[u32; HEAD_WORDS], tables: &CompactTables, page: u32, i: usize) -> CompactNode {
    let field = |k: usize| -> u32 {
        let w = PLANE_WIDTHS[k] as usize;
        let at = plane_offset(k, n) + i * w;
        let mut v = 0u32;
        for b in 0..w.min(4) {
            v |= (raw[at + b] as u32) << (8 * b);
        }
        v
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
        traversal: traversal_node(raw, n, head, page, i),
    }
}

/// What a node's LoD tree words and LODE give the traversal
/// (`TraversalNode`): the reference `traversal_node` must equal.
pub fn traversal_of_words(raw: &[u8], n: usize, head: &[u32], i: usize, tree: [u32; 4], lode: Option<[u32; 2]>) -> TraversalNode {
    let count = tree[2] & 0xffff;
    let cone = match lode {
        Some([0, 0]) | None => NO_CONE_WORD,
        Some(e) => e[1],
    };
    if count == 0 {
        let q = position_codes(raw, n, i);
        let centre = [0, 1, 2].map(|k| f16::from_f32(compact_position(q[k], f32::from_bits(head[k]), f32::from_bits(head[3 + k]))).to_bits());
        return TraversalNode { centre, cone, ..Default::default() };
    }
    let e = lode.unwrap_or([0, 0]);
    let size_mode = e[0] == 0;
    TraversalNode {
        centre: [tree[0] as u16, (tree[0] >> 16) as u16, tree[1] as u16],
        child_page: tree[3] >> 16,
        place: tree[3] & 0xffff,
        count,
        eps: if size_mode { (tree[1] >> 16) as u16 } else { e[0] as u16 },
        rho: if size_mode { 0 } else { (e[0] >> 16) as u16 },
        size_mode,
        cone,
        occlusion: 0,
    }
}

// --- an asset's pieces ------------------------------------------------------

/// Bytes of a v1 `CMPC` node (thread CN): the GPU's 40, the LoD tree's 4
/// words, the 2 LODE words.
pub const V1_NODE_BYTES: usize = 64;

/// A paged tree with its compact tier written again (thread CQ): the
/// traversal's words in the v2 layout (from a v1 or v2 file), the tables
/// sharing `asset` when given (its basis must be the piece's; its shadow
/// dictionary and materials, when it has them, replace the piece's: the
/// shadow index of each node is its true mask's nearest entry, read from
/// tier 3). Tiers 1-3 are written again byte for byte (checked).
pub fn rebase_compact(bytes: &[u8], asset: Option<&CompactAsset>, asset_name: &str, threads: usize) -> Result<(Vec<u8>, CompactStats)> {
    use crate::athc_v3::{gunzip, parse_v3, read_v3, write_v3_paged_compact, SectionId, COMPRESSION_GZIP};
    let t = std::time::Instant::now();
    let layout = parse_v3(bytes)?;
    let k4 = layout.sections.len().checked_sub(1).filter(|&k| layout.sections[k].id == SectionId::Compact).ok_or_else(|| anyhow!("rebase: no compact tier"))?;
    let [at, len] = layout.compact_table;
    let blob = &bytes[at as usize..(at + len) as usize];
    let w = words_of(blob);
    let version = w(1)?;
    // The piece's own tables (v1: the same fields, no asset).
    let old = match version {
        1 => {
            let words = |at: u32, n: usize| -> Result<Vec<u32>> { (0..n).map(|k| w(at as usize + k)).collect() };
            if w(2)? as usize != V1_NODE_BYTES {
                bail!("rebase: a v1 compact tier of {} bytes a node", w(2)?);
            }
            CompactTables {
                codebook: words(w(11)?, 8 * w(5)? as usize)?,
                materials: words(w(12)?, 4 * w(6)? as usize)?,
                basis: words(w(13)?, w(14)? as usize)?,
                clusters: w(7)?,
                normal_bits: w(8)?,
                weight_bits: w(9)?,
                linear: w(10)? & 1 != 0,
                curvature: w(10)? & 2 != 0,
                ..Default::default()
            }
        }
        2 | 3 => CompactTables::from_bytes(blob)?,
        v => bail!("rebase: compact tables v{v}"),
    };
    if old.asset_flags != 0 {
        bail!("rebase: the piece already shares asset table {:?}", old.asset_name);
    }
    let node_bytes = match version {
        1 => V1_NODE_BYTES,
        2 => NODE_BYTES_V2,
        _ => NODE_BYTES,
    };
    for s in &layout.sections[..k4] {
        if s.encoding == crate::athc_v3::ENCODING_CPCA {
            bail!("rebase: section {:?} in CPCA (write it again with usd-athc)", s.id);
        }
    }
    let compression = layout.sections[k4].compression;
    if layout.sections.iter().any(|s| s.compression != compression) {
        bail!("rebase: sections of different compressions");
    }
    let file = read_v3(bytes)?;
    let skeleton = crate::athc_v3::read_v3_skeleton(bytes)?;
    if let Some(a) = asset {
        if !a.basis.is_empty() && (a.basis != old.basis || a.clusters != old.clusters) {
            bail!("rebase: the asset table's basis is not the piece's");
        }
    }
    let codebook: Vec<Mask> = asset.map(|a| a.codebook.chunks_exact(8).map(mask_of).collect()).unwrap_or_default();
    let mut material_of: std::collections::HashMap<[u32; 4], u32> = std::collections::HashMap::new();
    if let Some(a) = asset {
        for (m, key) in a.materials.chunks_exact(4).enumerate() {
            material_of.entry(key.try_into().unwrap()).or_insert(m as u32);
        }
    }
    let shared_materials = asset.is_some_and(|a| !a.materials.is_empty());
    let (mut pages, mut heads) = (Vec::new(), Vec::new());
    let mut stats = CompactStats { pages: layout.blocks.len(), ..Default::default() };
    let mut used_materials = std::collections::HashSet::new();
    for (p, b) in layout.blocks.iter().enumerate() {
        let n = b.n as usize;
        let span = b.spans[k4];
        let stored = &bytes[span.offset as usize..][..span.stored as usize];
        let src = if compression == COMPRESSION_GZIP { gunzip(stored)? } else { stored.to_vec() };
        if src.len() != n * node_bytes {
            bail!("rebase: page {p}: {} bytes for {n} nodes", src.len());
        }
        let head = layout.pages[p].compact;
        let mut raw = vec![0u8; n * NODE_BYTES];
        // The GPU's planes, as they are (the same widths in both versions).
        raw[..n * GPU_NODE_BYTES].copy_from_slice(&src[..n * GPU_NODE_BYTES]);
        let block = &file.chunks[p];
        let mut labels = vec![0u32; n];
        if !codebook.is_empty() {
            let masks: Vec<Mask> = block.shadow_bits.chunks_exact(8).map(mask_of).collect();
            if masks.len() != n {
                bail!("rebase: page {p} lacks its shadow bits");
            }
            crate::athc_shadow_vq::assign(&masks, &codebook, threads, &mut labels);
        }
        for i in 0..n {
            if !codebook.is_empty() {
                let at = plane_offset(P_SHADOW, n) + 2 * i;
                raw[at..at + 2].copy_from_slice(&(labels[i] as u16).to_le_bytes());
            }
            let at = plane_offset(P_MATERIAL, n) + 2 * i;
            let m = u16::from_le_bytes([raw[at], raw[at + 1]]) as usize;
            used_materials.insert(old.materials[4 * m..4 * m + 4].to_vec());
            if shared_materials {
                let key: [u32; 4] = old.materials[4 * m..4 * m + 4].try_into().unwrap();
                let to = *material_of.get(&key).ok_or_else(|| anyhow!("rebase: page {p}: a material not in the asset table"))?;
                raw[at..at + 2].copy_from_slice(&(to as u16).to_le_bytes());
            }
            // The traversal's words: from what the tree holds (tiers 1-3),
            // as the native writer makes them.
            let tree = lod_tree_words(block, i);
            let lode = (block.lod_error.len() == 2 * n).then(|| [block.lod_error[2 * i], block.lod_error[2 * i + 1]]);
            put_traversal(&mut raw, n, &head, p as u32, i, tree, lode, block.occlusion.get(i).copied().unwrap_or(0))?;
            if version == 1 {
                // A v1 file's own words: the same node for the traversal.
                let word = |off: usize| u32::from_le_bytes(src[off..off + 4].try_into().unwrap());
                let base = n * GPU_NODE_BYTES;
                let t1 = [0, 1, 2, 3].map(|j| word(base + 16 * i + 4 * j));
                let e1 = [word(base + 16 * n + 8 * i), word(base + 16 * n + 8 * i + 4)];
                let a = traversal_of_words(&src, n, &head, i, t1, Some(e1));
                let b2 = traversal_node(&raw, n, &head, p as u32, i);
                if a != b2 {
                    bail!("rebase: page {p} node {i}: the v1 words give {a:?}, tiers 1-3 {b2:?}");
                }
            }
        }
        stats.nodes += n;
        pages.push(raw);
        heads.push(head);
    }
    let mut flags = 0;
    if let Some(a) = asset {
        flags |= if a.basis.is_empty() { 0 } else { TABLE_ASSET_BASIS };
        flags |= if codebook.is_empty() { 0 } else { TABLE_ASSET_CODEBOOK };
        flags |= if shared_materials { TABLE_ASSET_MATERIALS } else { 0 };
    }
    let tables = CompactTables {
        codebook: if flags & TABLE_ASSET_CODEBOOK != 0 { Vec::new() } else { old.codebook.clone() },
        materials: if shared_materials { Vec::new() } else { old.materials.clone() },
        basis: if flags & TABLE_ASSET_BASIS != 0 { Vec::new() } else { old.basis.clone() },
        asset_flags: flags,
        asset_id: if flags != 0 { asset.unwrap().id() } else { 0 },
        asset_name: if flags != 0 { asset_name.to_string() } else { String::new() },
        ..old.clone()
    };
    stats.materials = used_materials.len();
    stats.codebook_entries = if codebook.is_empty() { old.entries() } else { codebook.len() };
    stats.table_bytes = [4 * tables.codebook.len(), 4 * tables.materials.len(), 4 * tables.basis.len()];
    let compact = CompactPages { tables, pages, heads, stats: stats.clone() };
    let encodings: Vec<(SectionId, u32)> = layout.sections.iter().map(|s| (s.id, s.encoding)).collect();
    let encoding = move |id: SectionId| encodings.iter().find(|(s, _)| *s == id).map_or(0, |e| e.1);
    let (out, _) = write_v3_paged_compact(&file, &layout.pages, compression, &encoding, None, skeleton.as_ref(), &compact)?;
    // Tiers 1-3: the same stored bytes, page for page.
    let new = parse_v3(&out)?;
    if new.sections[..k4] != layout.sections[..k4] {
        bail!("rebase: the sections changed");
    }
    for (p, (a, b)) in layout.blocks.iter().zip(&new.blocks).enumerate() {
        for k in 0..k4 {
            let sa = &bytes[a.spans[k].offset as usize..][..a.spans[k].stored as usize];
            let sb = &out[b.spans[k].offset as usize..][..b.spans[k].stored as usize];
            if sa != sb {
                bail!("rebase: page {p} section {:?} written differently", layout.sections[k].id);
            }
        }
    }
    stats.seconds = t.elapsed().as_secs_f32();
    Ok((out, stats))
}

/// An asset table for pieces (paged trees with their compact tier, v1 or
/// v2, each with its own tables): their one basis, and with `entries` > 0
/// one shadow dictionary of that many entries (k-means in Hamming space
/// over every node of every piece, `rounds`) and one material table (every
/// piece's, in order). The pieces must share their basis.
pub fn build_asset(pieces: &[&[u8]], entries: usize, rounds: usize, shared_materials: bool, threads: usize) -> Result<CompactAsset> {
    use crate::athc_v3::parse_v3;
    let mut asset = CompactAsset::default();
    let mut masks: Vec<Mask> = Vec::new();
    let mut seen = std::collections::HashSet::<[u32; 4]>::new();
    for (k, bytes) in pieces.iter().enumerate() {
        let layout = parse_v3(bytes)?;
        let [at, len] = layout.compact_table;
        if at == 0 {
            bail!("asset: piece {k} has no compact tier");
        }
        let blob = &bytes[at as usize..(at + len) as usize];
        let w = words_of(blob);
        // v1 and v2 hold the basis and materials at the same words.
        let words = |at: u32, n: usize| -> Result<Vec<u32>> { (0..n).map(|k| w(at as usize + k)).collect() };
        if w(1)? == 2 && w(10)? & (TABLE_ASSET_BASIS | TABLE_ASSET_CODEBOOK | TABLE_ASSET_MATERIALS) != 0 {
            bail!("asset: piece {k} shares an asset table already");
        }
        let basis = words(w(13)?, w(14)? as usize)?;
        if k == 0 {
            asset.basis = basis;
            asset.clusters = w(7)?;
        } else if basis != asset.basis || w(7)? != asset.clusters {
            bail!("asset: piece {k}'s basis is not piece 0's");
        }
        if shared_materials {
            for key in words(w(12)?, 4 * w(6)? as usize)?.chunks_exact(4) {
                let key: [u32; 4] = key.try_into().unwrap();
                if seen.insert(key) {
                    asset.materials.extend_from_slice(&key);
                }
            }
        }
        if entries > 0 {
            for b in crate::athc_v3::read_v3(bytes)?.chunks {
                masks.extend(b.shadow_bits.chunks_exact(8).map(mask_of));
            }
        }
    }
    if asset.materials.len() / 4 > 0x10000 {
        bail!("asset: {} materials, the index is 16 bits", asset.materials.len() / 4);
    }
    if entries > 0 {
        if entries > 0x10000 {
            bail!("asset: {entries} entries, the index is 16 bits");
        }
        let mut distinct = masks.clone();
        distinct.sort_unstable();
        distinct.dedup();
        let codebook =
            if distinct.len() <= entries { distinct } else { crate::athc_shadow_vq::kmeans_hamming(&masks, entries, rounds, threads).0 };
        asset.codebook = codebook.iter().flat_map(mask_words).collect();
    }
    Ok(asset)
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
                let node = decode_node(&raw, block.n, head, &tables, p as u32, i);
                for k in 0..3 {
                    assert!((node.center[k] - block.positions[4 * i + k]).abs() <= (hi[k] - lo[k]) / 65535.0 + 1e-6);
                }
                let lode = [block.lod_error[2 * i], block.lod_error[2 * i + 1]];
                assert_eq!(node.traversal, traversal_of_words(&raw, block.n, head, i, lod_tree_words(block, i), Some(lode)));
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
mod asset_tests {
    use super::tests::sample_basis;
    use super::*;
    use crate::athc_v3::{parse_v3, write_v3_paged_compact, COMPRESSION_GZIP};

    fn piece(seed_n: usize, page: usize, o: &CompactOptions) -> (Vec<u8>, CompactPages, crate::athc::AthcFile) {
        let (paged, file) = crate::athc_pages::tests::sample_relit_paged(seed_n, page);
        let c = build_compact(&file.chunks, false, &sample_basis(6), o).unwrap();
        let (bytes, _) = write_v3_paged_compact(&file, &paged.rows, COMPRESSION_GZIP, &|_| 1, None, None, &c).unwrap();
        (bytes, c, file)
    }

    fn nodes(bytes: &[u8], asset: Option<&CompactAsset>) -> Vec<CompactNode> {
        let layout = parse_v3(bytes).unwrap();
        let k4 = layout.sections.len() - 1;
        let [at, n] = layout.compact_table;
        let t = CompactTables::from_bytes(&bytes[at as usize..(at + n) as usize]).unwrap().resolve(asset).unwrap();
        let mut out = Vec::new();
        for (p, b) in layout.blocks.iter().enumerate() {
            let span = b.spans[k4];
            let raw = crate::athc_v3::gunzip(&bytes[span.offset as usize..][..span.stored as usize]).unwrap();
            for i in 0..b.n as usize {
                out.push(decode_node(&raw, b.n as usize, &layout.pages[p].compact, &t, p as u32, i));
            }
        }
        out
    }

    #[test]
    fn traversal_corrections_are_exact() {
        // Every f16 the centre could be, against positions around it.
        for bits in [0x0000u16, 0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x1234, 0x9234] {
            assert_eq!(f16_from_order(f16_order(bits)), bits);
        }
        assert_eq!(f16_order(0x8000) + 1, f16_order(0x0000));
        let n = 3;
        let mut raw = vec![0u8; n * NODE_BYTES];
        let head: Vec<u32> = [-2.0f32, -0.001, 0.0, 3.0, 0.002, 1e-3].iter().map(|v| v.to_bits()).collect();
        for i in 0..n {
            for k in 0..3 {
                let at = plane_offset(P_POS_X + k, n) + 2 * i;
                raw[at..at + 2].copy_from_slice(&(i as u16 * 21845 + 7).to_le_bytes());
            }
        }
        let centres = [[0x8000u16, 0x0000, 0x0001], [0x3c00, 0x9000, 0x1400], [0xc000, 0x0400, 0x8001]];
        for i in 0..n {
            let c = centres[i];
            let tree = [c[0] as u32 | (c[1] as u32) << 16, c[2] as u32 | 0x2c00 << 16, 2, 5 << 16 | (100 + i as u32)];
            let lode = if i == 2 { None } else { Some([0x1234 | 0x0567 << 16, 0x7c00]) };
            put_traversal(&mut raw, n, &head, 3, i, tree, lode, 0).unwrap();
            let t = traversal_node(&raw, n, &head, 3, i);
            assert_eq!(t, traversal_of_words(&raw, n, &head, i, tree, lode), "{i}");
            assert_eq!((t.child_page, t.place, t.count), (5, 100 + i as u32, 2));
            assert_eq!(t.size_mode, i == 2);
            assert_eq!(traversal_node_input(&traversal_input(&raw, n), n, &head, 3, i), t);
        }
        // v3 (thread CS): a merge's cone, a closed leaf's own, the occlusion
        // index; a leaf's [0, 0] LODE is no cone.
        let mut raw = vec![0u8; n * NODE_BYTES];
        let cone = 0x3c00 | 0x8080 << 16;
        put_traversal(&mut raw, n, &head, 3, 0, [0, 0, 2, 5 << 16 | 100], Some([0x1234, cone]), 8193).unwrap();
        let t = traversal_node(&raw, n, &head, 3, 0);
        assert_eq!((t.cone, t.occlusion), (cone, 8193));
        put_traversal(&mut raw, n, &head, 3, 1, [0, 0, 0, 0], Some([0, cone]), 7).unwrap();
        let t = traversal_node_input(&traversal_input(&raw, n), n, &head, 3, 1);
        assert_eq!((t.count, t.cone, t.occlusion), (0, cone, 7));
        put_traversal(&mut raw, n, &head, 3, 2, [0, 0, 0, 0], Some([0, 0]), 0).unwrap();
        assert_eq!(traversal_node(&raw, n, &head, 3, 2).cone, NO_CONE_WORD);
        assert!(put_traversal(&mut raw, n, &head, 3, 2, [0, 0, 0, 0], None, 70000).is_err());
    }

    /// Pieces sharing an asset table: the basis once (the nodes as the
    /// standalone piece's), then with the asset's dictionary and materials.
    #[test]
    fn pieces_share_an_asset_table() {
        let o = CompactOptions { threads: 2, ..Default::default() };
        let (alone, c, file) = piece(1500, 400, &o);
        let asset = CompactAsset { basis: sample_basis(6).table_f16(), clusters: 6, ..Default::default() };
        let blob = asset.to_bytes();
        assert_eq!(CompactAsset::from_bytes(&blob).unwrap(), asset);
        let mut bad = blob.clone();
        bad[4 * 16] ^= 1;
        assert!(CompactAsset::from_bytes(&bad).is_err());
        let shared_o = CompactOptions { asset: Some(asset.clone()), asset_name: "car.cmpa".into(), ..o.clone() };
        let (shared, sc, _) = piece(1500, 400, &shared_o);
        assert!(sc.tables.basis.is_empty() && sc.tables.asset_flags == TABLE_ASSET_BASIS);
        assert_eq!(sc.tables.asset_id, asset.id());
        assert_eq!(nodes(&shared, Some(&asset)), nodes(&alone, None));
        assert!(alone.len() - shared.len() >= 4 * c.tables.basis.len() - 64);
        // Without its asset, or with another, a piece is refused.
        let layout = parse_v3(&shared).unwrap();
        let [at, n] = layout.compact_table;
        let t = CompactTables::from_bytes(&shared[at as usize..(at + n) as usize]).unwrap();
        assert_eq!(t.asset_name, "car.cmpa");
        assert!(t.resolve(None).is_err());
        let other = CompactAsset { clusters: 6, basis: sample_basis(5).table_f16(), ..Default::default() };
        assert!(t.resolve(Some(&other)).is_err());
        // The asset's dictionary and materials: every node's cells its
        // mask's nearest entry, its material the same.
        let asset2 = build_asset(&[&alone, &alone], 16, 4, true, 2).unwrap();
        assert_eq!(asset2.codebook.len(), 16 * 8);
        let (rebased, _) = rebase_compact(&alone, Some(&asset2), "car.cmpa", 2).unwrap();
        let native = piece(1500, 400, &CompactOptions { asset: Some(asset2.clone()), asset_name: "car.cmpa".into(), ..o.clone() }).0;
        assert_eq!(rebased, native, "rebase = the native writer");
        let (a, b) = (nodes(&alone, None), nodes(&rebased, Some(&asset2)));
        let masks: Vec<Mask> = file.chunks.iter().flat_map(|b| b.shadow_bits.chunks_exact(8).map(mask_of)).collect();
        let entries: Vec<Mask> = asset2.codebook.chunks_exact(8).map(mask_of).collect();
        for (k, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!((x.center, x.material, x.cluster, x.weights, x.traversal), (y.center, y.material, y.cluster, y.weights, y.traversal));
            let d = |e: &Mask| (0..4).map(|j| (masks[k][j] ^ e[j]).count_ones()).sum::<u32>();
            assert_eq!(d(&mask_of(&y.cells)), entries.iter().map(d).min().unwrap());
        }
    }

    /// CN's v1 fixture: rebased to v2, its traversal checked node by node
    /// against its own v1 words (in rebase_compact), tiers 1-3 unchanged.
    #[test]
    fn a_v1_file_rebases() {
        let v1 = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../test/fixtures/athc/compact_pages.v1.athc")).unwrap();
        let (v3, stats) = rebase_compact(&v1, None, "", 2).unwrap();
        assert_eq!(stats.nodes, parse_v3(&v1).unwrap().header.count as usize);
        let fixture = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../test/fixtures/athc/compact_pages.athc")).unwrap();
        assert_eq!(v3, fixture, "the v3 fixture is the v1 one rebased");
        // CQ's v2 fixture rebases to the same v3 (thread CS).
        let v2 = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../test/fixtures/athc/compact_pages.v2.athc")).unwrap();
        assert_eq!(rebase_compact(&v2, None, "", 2).unwrap().0, fixture, "the v2 fixture rebased is the v3 one");
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
