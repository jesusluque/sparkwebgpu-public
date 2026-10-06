//! Building a `.athc` on the CPU, as athenea builds one on the device.
//!
//! athenea writes a `.athc` from a cloud on the GPU: USD's arrays are laid
//! out as records (`scene/streams.slang`), validated and packed
//! (`scene/splat_validate.slang`, `scene/splat_decode.slang`), and
//! `lod::LodBuilder` sorts them in Morton order and merges every octree level
//! (`lod/lod_*.slang`, `modules/lod/src/Lod.cpp`). This is the same, kernel
//! for kernel and in f32 as the kernels compute, read off athenea's sources at
//! `txf` 89a04d9 (vendored under `slang/athenea/`). It exists because athenea
//! refuses to write a transfer into a `.athc` ("a .athc cannot carry a
//! transfer"), while the format has room for one (flag bit 5): sparkwebGPU
//! writes those itself.
//!
//! Checked against athenea: rebuilding the levels of `two_cards.athc` (which
//! athenea's mesh2splat test wrote) from its own splats gives its levels --
//! the same groups, cells, starts and splat order, and each merged Gaussian
//! within f32 rounding of athenea's (the device sums in its own order).

use anyhow::{bail, Result};
use half::f16;

use crate::athc::{
    encode_quaternion, pack_halves, pack_normal, AthcBlock, AthcFile, AthcHeader, ExtraHeader,
    FLAG_LINEAR, FLAG_MATERIAL, FLAG_NORMALS, FLAG_TRANSFER, OLDEST_VERSION, VERSION,
};

/// `packing.slang`'s kSH0.
pub const SH0: f32 = 0.282_094_8;
/// `lod_common.slang`: 30-bit codes, ten levels.
pub const LOD_LEVELS: u32 = 10;
/// `lod_common.slang`: W, S1 (3), S2 (6), base colour (3).
const MOMENTS_HEAD: usize = 13;

/// A splat cloud as USD's ParticleField holds one: one array per attribute,
/// as `scene::SplatStreams` takes them. Optional arrays are empty when absent.
#[derive(Clone, Debug, Default)]
pub struct CloudStreams {
    pub count: usize,
    /// xyz
    pub positions: Vec<f32>,
    /// xyzw (GfQuat's imaginary part first, as streams.slang reads it)
    pub rotations: Vec<f32>,
    /// xyz, linear
    pub scales: Vec<f32>,
    /// linear
    pub opacities: Vec<f32>,
    /// SH coefficients a splat, DC first: (degree + 1)^2
    pub coefficients: usize,
    /// rgb per coefficient
    pub sh: Vec<f32>,
    /// `primvars:athenea:splat:linear`
    pub linear: bool,
    /// xyz a splat (`primvars:athenea:splat:normal`)
    pub normals: Vec<f32>,
    pub metallic: Vec<f32>,
    pub roughness: Vec<f32>,
    pub transmission: Vec<f32>,
    /// `transferDirect`: 9 or 16 a splat
    pub transfer_direct: Vec<f32>,
    /// `transferIndirect`: 3 x the direct count a splat, rgb a coefficient
    pub transfer_indirect: Vec<f32>,
    /// `transferReflected`: 48 a splat
    pub transfer_reflected: Vec<f32>,
    /// `shadowBits`: 2, 8 or 32 words a splat
    pub shadow_bits: Vec<u32>,
    /// `curvature`: 3 a splat, the shape operator in the splat's first two
    /// axes (xx, xy, yy), on the mesh's normal (sparkwebGPU's `CURV`)
    pub curvature: Vec<f32>,
}

/// Which of a transfer's values are kept (`transfer_layout.slang`: the count
/// is the layout). `Full` keeps what the cloud carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferKeep {
    Full,
    /// One of 9, 36, 84, 16, 64, 112: the first coefficients of each half
    /// (harmonics truncate to a lower degree), the field kept or not.
    Count(u32),
    None,
}

#[derive(Clone, Copy, Debug)]
pub struct BuildOptions {
    /// `LodBuildSettings`
    pub max_group_fraction: f32,
    pub coarsest_level: u32,
    pub chunk_splats: u32,
    /// Rest basis functions kept a colour at most (0, 3, 8, 15).
    pub max_rest: usize,
    pub transfer: TransferKeep,
    pub shadow_bits: bool,
    pub normals: bool,
    pub material: bool,
    /// Keep the curvature (a v3 section of its own; a v2 file drops it).
    pub curvature: bool,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            max_group_fraction: 0.5,
            coarsest_level: 1,
            chunk_splats: 1 << 16,
            max_rest: 15,
            transfer: TransferKeep::Full,
            shadow_bits: true,
            normals: true,
            material: true,
            curvature: true,
        }
    }
}

/// `transfer_layout.slang`
pub fn transfer_direct_count(count: u32) -> u32 {
    if count == 16 || count == 64 || count == 112 {
        16
    } else {
        9
    }
}
pub fn transfer_indirect_count(count: u32) -> u32 {
    let d = transfer_direct_count(count);
    if count >= 4 * d {
        3 * d
    } else {
        0
    }
}
pub fn transfer_has_field(count: u32) -> bool {
    let d = transfer_direct_count(count);
    count == d + 3 * d + 48
}

/// A packed cloud (`scene::GpuSplats`), its arrays in an `AthcBlock` whose
/// tail is unused, plus what says how to read them.
#[derive(Clone, Debug, Default)]
pub struct PackedCloud {
    pub block: AthcBlock,
    pub rest_per_colour: u32,
    pub sh_words: u32,
    pub transfer_count: u32,
    pub linear: bool,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    /// Records dropped by validation (opacity under 1/255, not finite).
    pub dropped: usize,
}

fn half_safe(v: f32) -> f32 {
    v.clamp(-65000.0, 65000.0)
}
fn f16_bits(v: f32) -> u32 {
    f16::from_f32(v).to_bits() as u32
}
fn f16_of(bits: u32) -> f32 {
    f16::from_bits(bits as u16).to_f32()
}
fn saturate(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// `packPbr` (splat_encoding.slang), with no thin wall or Schlick mark.
pub fn pack_pbr(metallic: f32, roughness: f32, transmission: f32) -> u32 {
    let schlick = transmission >= 4.0 - 0.5;
    let rest = if schlick {
        transmission - 4.0
    } else {
        transmission
    };
    let thin = rest >= 2.0 - 0.5;
    let byte = |v: f32| (saturate(v) * 255.0 + 0.5) as u32;
    byte(metallic)
        | (byte(roughness) << 8)
        | (byte(if thin { rest - 2.0 } else { rest }) << 16)
        | if thin { 1 << 24 } else { 0 }
        | if schlick { 1 << 25 } else { 0 }
}

/// `axesOfQuaternion`: m[row][col], the columns the turned basis.
fn axes_of_quaternion(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [x, y, z, w] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|c| c / l)
}

/// The transfer's values a splat in the kept layout, from USD's three arrays.
fn transfer_layout(s: &CloudStreams, keep: TransferKeep) -> Result<(u32, u32, u32, bool)> {
    let n = s.count;
    if s.transfer_direct.is_empty() || keep == TransferKeep::None {
        return Ok((0, 0, 0, false));
    }
    let direct = if s.transfer_direct.len() >= n * 16 {
        16
    } else {
        9
    };
    if s.transfer_direct.len() != n * direct {
        bail!(
            "transferDirect holds {} values for {} splats",
            s.transfer_direct.len(),
            n
        );
    }
    let indirect = if s.transfer_indirect.len() == n * direct * 3 {
        direct * 3
    } else {
        0
    };
    let field = indirect > 0 && s.transfer_reflected.len() == n * 48;
    let full = (direct + indirect + if field { 48 } else { 0 }) as u32;
    let count = match keep {
        TransferKeep::Full => full,
        TransferKeep::Count(c) => c,
        TransferKeep::None => unreachable!(),
    };
    if ![9, 36, 84, 16, 64, 112].contains(&count) {
        bail!("a transfer of {count} values is no layout (9, 36, 84, 16, 64 or 112)");
    }
    let (d, i, f) = (
        transfer_direct_count(count),
        transfer_indirect_count(count),
        transfer_has_field(count),
    );
    if d as usize > direct || i as usize > indirect || (f && !field) {
        bail!("the cloud carries a transfer of {full} values; {count} asks for more");
    }
    Ok((count, d, i, f))
}

/// streams -> records -> validate -> decode: the packed cloud athenea's
/// `CloudLoader::upload(SplatStreams)` makes, compacted.
pub fn pack_streams(s: &CloudStreams, o: &BuildOptions) -> Result<PackedCloud> {
    let n = s.count;
    let need = |name: &str, v: &[f32], per: usize| -> Result<()> {
        if !v.is_empty() && v.len() != n * per {
            bail!("{name} holds {} values for {n} splats of {per}", v.len());
        }
        Ok(())
    };
    need("positions", &s.positions, 3)?;
    need("orientations", &s.rotations, 4)?;
    need("scales", &s.scales, 3)?;
    need("opacities", &s.opacities, 1)?;
    need("normals", &s.normals, 3)?;
    need("curvature", &s.curvature, 3)?;
    if s.positions.is_empty() {
        bail!("no positions");
    }
    if s.coefficients > 0 && s.sh.len() != n * s.coefficients * 3 {
        bail!(
            "{} SH values for {n} splats of {} coefficients",
            s.sh.len(),
            s.coefficients
        );
    }
    let keep = if s.coefficients > 0 {
        (s.coefficients - 1).min(o.max_rest)
    } else {
        0
    };
    let sh_words = if keep == 0 { 1 } else { (keep * 3).div_ceil(2) };
    let pbr = o.material
        && !(s.metallic.is_empty() && s.roughness.is_empty() && s.transmission.is_empty());
    let normals = o.normals && !s.normals.is_empty();
    let curvature = o.curvature && !s.curvature.is_empty();
    let (transfer_count, direct, indirect, field) = transfer_layout(s, o.transfer)?;
    let transfer_words = transfer_count.div_ceil(2) as usize;
    let shadow_words = if transfer_count > 0 && o.shadow_bits && !s.shadow_bits.is_empty() {
        match s.shadow_bits.len() / n {
            w @ (2 | 8 | 32) if s.shadow_bits.len() == n * w => w,
            _ => bail!(
                "shadowBits holds {} words for {n} splats",
                s.shadow_bits.len()
            ),
        }
    } else {
        0
    };
    let src_direct = s.transfer_direct.len() / n;
    let src_indirect = s.transfer_indirect.len() / n;

    let mut b = AthcBlock::default();
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    let mut dropped = 0;
    let mut values = vec![0.0f32; transfer_count as usize];
    for i in 0..n {
        let p = [
            s.positions[i * 3],
            s.positions[i * 3 + 1],
            s.positions[i * 3 + 2],
        ];
        let a = if s.opacities.is_empty() {
            1.0
        } else {
            s.opacities[i]
        };
        let sc = if s.scales.is_empty() {
            [1.0; 3]
        } else {
            [s.scales[i * 3], s.scales[i * 3 + 1], s.scales[i * 3 + 2]]
        };
        let ls = sc.map(|v| v.max(1e-30).ln());
        let finite =
            p.iter().all(|v| v.is_finite()) && ls.iter().all(|v| v.is_finite() && v.abs() < 60.0);
        if !(finite && a >= 1.0 / 255.0) {
            dropped += 1;
            continue;
        }
        b.n += 1;
        b.positions
            .extend_from_slice(&[p[0], p[1], p[2], saturate(a)]);
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
        let q = if s.rotations.is_empty() {
            [0.0, 0.0, 0.0, 1.0]
        } else {
            [
                s.rotations[i * 4],
                s.rotations[i * 4 + 1],
                s.rotations[i * 4 + 2],
                s.rotations[i * 4 + 3],
            ]
        };
        let q_len = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        let q = if q_len > 1e-8 {
            q
        } else {
            [0.0, 0.0, 0.0, 1.0]
        };
        let dc = |c: usize| {
            if s.coefficients > 0 {
                0.5 + SH0 * s.sh[i * s.coefficients * 3 + c]
            } else {
                0.5
            }
        };
        let base = [dc(0), dc(1), dc(2)];
        b.shape.extend_from_slice(&[
            encode_quaternion(q),
            pack_halves(half_safe(ls[0]), half_safe(ls[1])),
            pack_halves(half_safe(ls[2]), half_safe(base[0])),
            pack_halves(half_safe(base[1]), half_safe(base[2])),
        ]);
        if keep == 0 {
            b.sh.push(0);
        } else {
            let row = i * s.coefficients * 3 + 3;
            let halves: Vec<u32> = (0..keep * 3)
                .map(|k| f16_bits(half_safe(s.sh[row + k])) & 0xffff)
                .collect();
            for pair in halves.chunks(2) {
                b.sh.push(pair[0] | pair.get(1).map_or(0, |h| h << 16));
            }
        }
        if pbr {
            let m = if s.metallic.is_empty() {
                0.0
            } else {
                s.metallic[i]
            };
            let r = if s.roughness.is_empty() {
                1.0
            } else {
                s.roughness[i]
            };
            let t = if s.transmission.is_empty() {
                0.0
            } else {
                s.transmission[i]
            };
            b.pbr.push(pack_pbr(m, r, t));
        }
        if normals {
            let mut nv = [s.normals[i * 3], s.normals[i * 3 + 1], s.normals[i * 3 + 2]];
            #[allow(clippy::neg_cmp_op_on_partial_ord)] // a NaN normal takes the axis too
            if !(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2] > 1.0e-20) {
                let qn = normalize4(q);
                let axes = axes_of_quaternion(qn);
                let k = shortest(ls);
                nv = [axes[0][k], axes[1][k], axes[2][k]];
            }
            b.normals.push(pack_normal(normalize3(nv)));
        }
        if transfer_count > 0 {
            let d = direct as usize;
            let ind = indirect as usize;
            values[..d].copy_from_slice(&s.transfer_direct[i * src_direct..i * src_direct + d]);
            values[d..d + ind]
                .copy_from_slice(&s.transfer_indirect[i * src_indirect..i * src_indirect + ind]);
            if field {
                values[d + ind..].copy_from_slice(&s.transfer_reflected[i * 48..i * 48 + 48]);
            }
            for w in 0..transfer_words {
                let lo16 = f16_bits(values[w * 2]) & 0xffff;
                let hi16 = values.get(w * 2 + 1).map_or(0, |v| f16_bits(*v) & 0xffff);
                b.transfer.push((hi16 << 16) | lo16);
            }
        }
        if shadow_words > 0 {
            b.shadow_bits
                .extend_from_slice(&s.shadow_bits[i * shadow_words..(i + 1) * shadow_words]);
        }
        if curvature {
            let k = &s.curvature[i * 3..i * 3 + 3];
            let safe = |v: f32| if v.is_finite() { half_safe(v) } else { 0.0 };
            b.curvature.extend_from_slice(&[
                pack_halves(safe(k[0]), safe(k[1])),
                pack_halves(safe(k[2]), 0.0),
            ]);
        }
    }
    if b.n == 0 {
        bail!("no splat survives validation");
    }
    Ok(PackedCloud {
        block: b,
        rest_per_colour: keep as u32,
        sh_words: sh_words as u32,
        transfer_count,
        linear: s.linear,
        bounds_min: lo,
        bounds_max: hi,
        dropped,
    })
}

fn normalize4(q: [f32; 4]) -> [f32; 4] {
    let l = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    q.map(|v| v / l)
}

/// The shortest of three axes, as the kernels pick it (ties to the first).
fn shortest(s: [f32; 3]) -> usize {
    if s[0] <= s[1] {
        if s[0] <= s[2] {
            0
        } else {
            2
        }
    } else if s[1] <= s[2] {
        1
    } else {
        2
    }
}

/// `morton.slang`
fn spread(v: u32) -> u32 {
    let mut v = v & 0x3ff;
    v = (v | (v << 16)) & 0x030000ff;
    v = (v | (v << 8)) & 0x0300f00f;
    v = (v | (v << 4)) & 0x030c30c3;
    v = (v | (v << 2)) & 0x09249249;
    v
}
pub fn morton30(unit: [f32; 3]) -> u32 {
    let q = unit.map(|u| ((saturate(u) * 1024.0) as u32).min(1023));
    (spread(q[0]) << 2) | (spread(q[1]) << 1) | spread(q[2])
}

/// The block's elements in `order`.
fn reorder(b: &AthcBlock, order: &[u32]) -> AthcBlock {
    let pick = |v: &Vec<u32>| -> Vec<u32> {
        if v.is_empty() {
            return Vec::new();
        }
        let per = v.len() / b.n;
        let mut out = Vec::with_capacity(v.len());
        for &i in order {
            out.extend_from_slice(&v[i as usize * per..(i as usize + 1) * per]);
        }
        out
    };
    let mut positions = Vec::with_capacity(b.positions.len());
    for &i in order {
        positions.extend_from_slice(&b.positions[i as usize * 4..i as usize * 4 + 4]);
    }
    AthcBlock {
        n: b.n,
        positions,
        shape: pick(&b.shape),
        sh: pick(&b.sh),
        tail: Vec::new(),
        normals: pick(&b.normals),
        emission: pick(&b.emission),
        pbr: pick(&b.pbr),
        lobes: pick(&b.lobes),
        transfer: pick(&b.transfer),
        shadow_bits: pick(&b.shadow_bits),
        curvature: pick(&b.curvature),
    }
}

struct Level {
    groups: usize,
    group: Vec<u32>,
    starts: Vec<u32>,
    cells: Vec<u32>,
}

/// `lod_boundaries` + prefix sum + `lod_groups`
fn groups_of(keys: &[u32], level: u32) -> Level {
    let shift = 3 * (LOD_LEVELS - level);
    let mut group = Vec::with_capacity(keys.len());
    let mut starts = Vec::new();
    let mut cells = Vec::new();
    for (i, &k) in keys.iter().enumerate() {
        if i == 0 || (k >> shift) != (keys[i - 1] >> shift) {
            starts.push(i as u32);
            cells.push(k >> shift);
        }
        group.push(starts.len() as u32 - 1);
    }
    Level {
        groups: starts.len(),
        group,
        starts,
        cells,
    }
}

type M3 = [[f32; 3]; 3];

fn mul(a: &M3, b: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            o[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    o
}
fn transpose(a: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            o[r][c] = a[c][r];
        }
    }
    o
}
fn determinant(m: &M3) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// `lod_finalize.slang`'s cyclic Jacobi.
fn jacobi(a: &mut M3) -> M3 {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..6 {
        for pq in 0..3 {
            let p = if pq == 2 { 1 } else { 0 };
            let q = if pq == 0 { 1 } else { 2 };
            let apq = a[p][q];
            if apq.abs() < 1e-30 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            let sg = theta + 1e-30;
            let sign = if sg > 0.0 {
                1.0
            } else if sg < 0.0 {
                -1.0
            } else {
                0.0
            };
            let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            let mut j = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
            j[p][p] = c;
            j[q][q] = c;
            j[p][q] = s;
            j[q][p] = -s;
            *a = mul(&transpose(&j), &mul(a, &j));
            v = mul(&v, &j);
        }
    }
    v
}

/// `packing.slang`'s quaternionOfAxes, (x, y, z, w).
fn quaternion_of_axes(m: &M3) -> [f32; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let q = if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        [
            (m[2][1] - m[1][2]) * s,
            (m[0][2] - m[2][0]) * s,
            (m[1][0] - m[0][1]) * s,
            0.25 / s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt();
        [
            0.25 * s,
            (m[1][0] + m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = 2.0 * (1.0 - m[0][0] + m[1][1] - m[2][2]).sqrt();
        [
            (m[1][0] + m[0][1]) / s,
            0.25 * s,
            (m[2][1] + m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
        ]
    } else {
        let s = 2.0 * (1.0 - m[0][0] - m[1][1] + m[2][2]).sqrt();
        [
            (m[0][2] + m[2][0]) / s,
            (m[2][1] + m[1][2]) / s,
            0.25 * s,
            (m[1][0] - m[0][1]) / s,
        ]
    };
    normalize4(q)
}

struct MomentLayout {
    keep: usize,
    normals: bool,
    stride: usize,
}

/// `lod_leaf_moments.slang`, over the Morton-sorted splats.
fn leaf_moments(s: &AthcBlock, level: &Level, l: &MomentLayout, sh_words: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; level.groups * l.stride];
    for g in 0..level.groups {
        let first = level.starts[g] as usize;
        let end = if g + 1 < level.groups {
            level.starts[g + 1] as usize
        } else {
            s.n
        };
        let at = g * l.stride;
        for i in first..end {
            let p = &s.positions[i * 4..i * 4 + 4];
            let sh = &s.shape[i * 4..i * 4 + 4];
            let sc = [
                f16_of(sh[1] & 0xffff).exp(),
                f16_of(sh[1] >> 16).exp(),
                f16_of(sh[2] & 0xffff).exp(),
            ];
            let q = crate::athc::decode_quaternion(sh[0]);
            let [x, y, z, w] = q;
            let r = [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                ],
                [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                ],
                [
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ];
            let mm: M3 =
                [0, 1, 2].map(|row| [r[row][0] * sc[0], r[row][1] * sc[1], r[row][2] * sc[2]]);
            let cov = mul(&mm, &transpose(&mm));
            let smallest = sc[0].min(sc[1]).min(sc[2]);
            let area = sc[0] * sc[1] * sc[2] / smallest.max(1e-20);
            let wt = p[3] * area;
            m[at] += wt;
            m[at + 1] += wt * p[0];
            m[at + 2] += wt * p[1];
            m[at + 3] += wt * p[2];
            m[at + 4] += wt * (cov[0][0] + p[0] * p[0]);
            m[at + 5] += wt * (cov[0][1] + p[0] * p[1]);
            m[at + 6] += wt * (cov[0][2] + p[0] * p[2]);
            m[at + 7] += wt * (cov[1][1] + p[1] * p[1]);
            m[at + 8] += wt * (cov[1][2] + p[1] * p[2]);
            m[at + 9] += wt * (cov[2][2] + p[2] * p[2]);
            m[at + 10] += wt * f16_of(sh[2] >> 16);
            m[at + 11] += wt * f16_of(sh[3] & 0xffff);
            m[at + 12] += wt * f16_of(sh[3] >> 16);
            for h in 0..l.keep * 3 {
                let word = s.sh[i * sh_words + h / 2];
                let v = if h & 1 == 0 {
                    f16_of(word & 0xffff)
                } else {
                    f16_of(word >> 16)
                };
                m[at + MOMENTS_HEAD + h] += wt * v;
            }
            if l.normals {
                let nv = crate::athc::unpack_normal(s.normals[i]);
                let nb = at + MOMENTS_HEAD + l.keep * 3;
                m[nb] += wt * nv[0];
                m[nb + 1] += wt * nv[1];
                m[nb + 2] += wt * nv[2];
            }
        }
    }
    m
}

/// `lod_merge_moments.slang`
fn merge_moments(level: &Level, fine: &Level, fine_moments: &[f32], stride: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; level.groups * stride];
    for g in 0..level.groups {
        let first = fine.group[level.starts[g] as usize] as usize;
        let end = if g + 1 < level.groups {
            fine.group[level.starts[g + 1] as usize] as usize
        } else {
            fine.groups
        };
        for k in 0..stride {
            let mut sum = 0.0f32;
            for h in first..end {
                sum += fine_moments[h * stride + k];
            }
            m[g * stride + k] = sum;
        }
    }
    m
}

/// `lod_finalize.slang`: one packed Gaussian per group.
fn finalize(m: &[f32], groups: usize, l: &MomentLayout, sh_words: usize) -> AthcBlock {
    let mut b = AthcBlock {
        n: groups,
        ..Default::default()
    };
    for g in 0..groups {
        let at = g * l.stride;
        let weight = m[at].max(1e-30);
        let mu = [m[at + 1] / weight, m[at + 2] / weight, m[at + 3] / weight];
        let mut cov: M3 = [
            [m[at + 4], m[at + 5], m[at + 6]],
            [m[at + 5], m[at + 7], m[at + 8]],
            [m[at + 6], m[at + 8], m[at + 9]],
        ]
        .map(|row| row.map(|v| v / weight));
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] -= mu[r] * mu[c];
            }
        }
        let mut axes = jacobi(&mut cov);
        if determinant(&axes) < 0.0 {
            for row in axes.iter_mut() {
                row[2] = -row[2];
            }
        }
        let s = [cov[0][0], cov[1][1], cov[2][2]].map(|v| v.max(1e-14).sqrt());
        let smallest = s[0].min(s[1]).min(s[2]);
        let area = s[0] * s[1] * s[2] / smallest.max(1e-20);
        let opacity = (m[at] / area.max(1e-30)).min(0.99);
        let base = [
            m[at + 10] / weight,
            m[at + 11] / weight,
            m[at + 12] / weight,
        ];
        b.positions
            .extend_from_slice(&[mu[0], mu[1], mu[2], opacity]);
        let ls = s.map(f32::ln);
        b.shape.extend_from_slice(&[
            encode_quaternion(quaternion_of_axes(&axes)),
            pack_halves(half_safe(ls[0]), half_safe(ls[1])),
            pack_halves(half_safe(ls[2]), half_safe(base[0])),
            pack_halves(half_safe(base[1]), half_safe(base[2])),
        ]);
        if l.normals {
            let nb = at + MOMENTS_HEAD + l.keep * 3;
            let sum = [m[nb], m[nb + 1], m[nb + 2]];
            let k = shortest(s);
            let axis = [axes[0][k], axes[1][k], axes[2][k]];
            let dot = sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2];
            b.normals
                .push(pack_normal(if dot > 1.0e-24 * weight * weight {
                    normalize3(sum)
                } else {
                    axis
                }));
        }
        if l.keep == 0 {
            b.sh.extend(std::iter::repeat_n(0, sh_words));
            continue;
        }
        let halves: Vec<u32> = (0..l.keep * 3)
            .map(|h| f16_bits(half_safe(m[at + MOMENTS_HEAD + h] / weight)) & 0xffff)
            .collect();
        for pair in halves.chunks(2) {
            b.sh.push(pair[0] | pair.get(1).map_or(0, |h| h << 16));
        }
    }
    b
}

/// `lodExtrasMerge`: 0 the first gaussian's words, 1 f16 pairs averaged by
/// opacity, 2 bits set where half the weight has them.
fn extras_merge(
    source: &[u32],
    words: usize,
    mode: u32,
    starts: &[u32],
    splats: &AthcBlock,
) -> Vec<u32> {
    let groups = starts.len();
    let mut target = vec![0u32; groups * words];
    for g in 0..groups {
        let first = starts[g] as usize;
        let end = if g + 1 < groups {
            starts[g + 1] as usize
        } else {
            splats.n
        };
        let out = g * words;
        if mode == 0 || end <= first + 1 {
            target[out..out + words].copy_from_slice(&source[first * words..first * words + words]);
            continue;
        }
        let opacity = |s: usize| splats.positions[s * 4 + 3].max(0.0);
        let mut total = 0.0f32;
        for s in first..end {
            total += opacity(s);
        }
        let scale = if total > 0.0 {
            1.0 / total
        } else {
            1.0 / (end - first) as f32
        };
        let weight = |s: usize| if total > 0.0 { opacity(s) } else { 1.0 };
        for w in 0..words {
            if mode == 1 {
                let (mut lo, mut hi) = (0.0f32, 0.0f32);
                for s in first..end {
                    let word = source[s * words + w];
                    lo += weight(s) * f16_of(word & 0xffff);
                    hi += weight(s) * f16_of(word >> 16);
                }
                target[out + w] = (f16_bits(hi * scale) << 16) | (f16_bits(lo * scale) & 0xffff);
            } else {
                let mut bits = 0u32;
                for bit in 0..32 {
                    let mut set = 0.0f32;
                    for s in first..end {
                        if (source[s * words + w] >> bit) & 1 != 0 {
                            set += weight(s);
                        }
                    }
                    if set * scale >= 0.5 {
                        bits |= 1 << bit;
                    }
                }
                target[out + w] = bits;
            }
        }
    }
    target
}

/// `LodBuilder::build` and `writeAthc`'s header: the `.athc` of a packed
/// cloud, its splats in Morton order, its levels merged.
pub fn build_lod(cloud: &PackedCloud, o: &BuildOptions) -> Result<AthcFile> {
    let src = &cloud.block;
    let n = src.n;
    if n == 0 {
        bail!("an empty cloud has no levels of detail");
    }
    if !src.emission.is_empty() || !src.lobes.is_empty() {
        bail!("emission and lobes are not built here yet");
    }
    let lo = cloud.bounds_min;
    let mut extent = 0.0f32;
    for k in 0..3 {
        extent = extent.max(cloud.bounds_max[k] - cloud.bounds_min[k]);
    }
    let extent = extent.max(1e-6) * 1.0001;

    // Morton order: a stable sort of the codes, as the radix sort is.
    let codes: Vec<u32> = (0..n)
        .map(|i| {
            let p = &src.positions[i * 4..i * 4 + 3];
            let e = extent.max(1e-20);
            morton30([(p[0] - lo[0]) / e, (p[1] - lo[1]) / e, (p[2] - lo[2]) / e])
        })
        .collect();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&i| codes[i as usize]);
    let keys: Vec<u32> = order.iter().map(|&i| codes[i as usize]).collect();
    let splats = reorder(src, &order);

    // Levels, coarse to fine, until one is nearly the splats again.
    let coarsest = o.coarsest_level.clamp(1, LOD_LEVELS);
    let mut levels: Vec<Option<Level>> = (0..=LOD_LEVELS).map(|_| None).collect();
    let mut finest = coarsest;
    for r in coarsest..=LOD_LEVELS {
        let level = groups_of(&keys, r);
        if level.groups as f32 > o.max_group_fraction * n as f32 && r > coarsest {
            break;
        }
        levels[r as usize] = Some(level);
        finest = r;
    }

    let l = MomentLayout {
        keep: cloud.rest_per_colour as usize,
        normals: !splats.normals.is_empty(),
        stride: MOMENTS_HEAD
            + cloud.rest_per_colour as usize * 3
            + if splats.normals.is_empty() { 0 } else { 3 },
    };
    let sh_words = cloud.sh_words as usize;
    let mut stored: Vec<(u32, AthcBlock)> = Vec::new();
    let mut fine_moments = Vec::new();
    for r in (coarsest..=finest).rev() {
        let level = levels[r as usize].as_ref().unwrap();
        let moments = if r == finest {
            leaf_moments(&splats, level, &l, sh_words)
        } else {
            merge_moments(
                level,
                levels[r as usize + 1].as_ref().unwrap(),
                &fine_moments,
                l.stride,
            )
        };
        let mut block = finalize(&moments, level.groups, &l, sh_words);
        let per = |v: &[u32]| v.len() / n;
        if !splats.pbr.is_empty() {
            block.pbr = extras_merge(&splats.pbr, per(&splats.pbr), 0, &level.starts, &splats);
        }
        if !splats.transfer.is_empty() {
            block.transfer = extras_merge(
                &splats.transfer,
                per(&splats.transfer),
                1,
                &level.starts,
                &splats,
            );
        }
        if !splats.shadow_bits.is_empty() {
            block.shadow_bits = extras_merge(
                &splats.shadow_bits,
                per(&splats.shadow_bits),
                2,
                &level.starts,
                &splats,
            );
        }
        // The curvature merged as the transfer is, by opacity: the mean of
        // the shape operators' half traces (what a lens reads) is exact,
        // the rest is each splat's own frame and only indicative.
        if !splats.curvature.is_empty() {
            block.curvature = extras_merge(&splats.curvature, 2, 1, &level.starts, &splats);
        }
        block.tail = level.cells.clone();
        stored.push((r, block));
        fine_moments = moments;
    }
    stored.reverse();
    let finest_level = levels[finest as usize].as_ref().unwrap();

    let chunk_splats = o.chunk_splats.max(1) as usize;
    let mut whole = splats;
    whole.tail = finest_level.group.clone();
    let chunks: Vec<AthcBlock> = (0..n.div_ceil(chunk_splats))
        .map(|c| whole.slice(c * chunk_splats, chunk_splats.min(n - c * chunk_splats)))
        .collect();

    let mut extra = ExtraHeader::default();
    if !whole.pbr.is_empty() {
        extra.pbr_words = 1;
    }
    if !whole.transfer.is_empty() {
        extra.transfer_count = cloud.transfer_count;
        extra.transfer_words = cloud.transfer_count.div_ceil(2);
        extra.shadow_words = (whole.shadow_bits.len() / n) as u32;
    }
    if !whole.curvature.is_empty() {
        extra.curvature_words = 2;
    }
    let flags = (if whole.normals.is_empty() {
        0
    } else {
        FLAG_NORMALS
    }) | (if cloud.linear { FLAG_LINEAR } else { 0 })
        | (if extra.pbr_words > 0 {
            FLAG_MATERIAL
        } else {
            0
        })
        | (if extra.transfer_words > 0 {
            FLAG_TRANSFER
        } else {
            0
        });
    let header = AthcHeader {
        version: if flags != 0 { VERSION } else { OLDEST_VERSION },
        count: n as u32,
        rest_per_colour: cloud.rest_per_colour,
        sh_words: cloud.sh_words,
        levels: stored.len() as u32,
        chunk_splats: chunk_splats as u32,
        chunks: chunks.len() as u32,
        finest_groups: finest_level.groups as u32,
        bounds_lo: lo,
        extent,
        bounds_min: cloud.bounds_min,
        bounds_max: cloud.bounds_max,
        flags,
        ..Default::default()
    };
    Ok(AthcFile {
        header,
        extra,
        levels: stored,
        starts: finest_level.starts.clone(),
        chunks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_CARDS: &[u8] = include_bytes!("../../../test/fixtures/athc/two_cards.athc");

    /// A packed shape's covariance.
    fn cov_of(shape: &[u32]) -> M3 {
        let q = crate::athc::decode_quaternion(shape[0]);
        let s = [
            f16_of(shape[1] & 0xffff).exp(),
            f16_of(shape[1] >> 16).exp(),
            f16_of(shape[2] & 0xffff).exp(),
        ];
        let r = axes_of_quaternion(q);
        let m: M3 = [0, 1, 2].map(|row| [r[row][0] * s[0], r[row][1] * s[1], r[row][2] * s[2]]);
        mul(&m, &transpose(&m))
    }

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    /// athenea's levels of `two_cards.athc`, rebuilt from its own splats.
    #[test]
    fn rebuilds_two_cards_levels_as_athenea_did() {
        let file = AthcFile::read(TWO_CARDS).unwrap();
        let h = file.header;
        let mut splats = file.splats();
        splats.tail.clear();
        let cloud = PackedCloud {
            block: splats,
            rest_per_colour: h.rest_per_colour,
            sh_words: h.sh_words,
            transfer_count: file.extra.transfer_count,
            linear: h.flags & FLAG_LINEAR != 0,
            bounds_min: h.bounds_min,
            bounds_max: h.bounds_max,
            dropped: 0,
        };
        let built = build_lod(&cloud, &BuildOptions::default()).unwrap();
        let b = built.header;
        assert_eq!(
            (b.count, b.levels, b.finest_groups, b.chunks),
            (h.count, h.levels, h.finest_groups, h.chunks)
        );
        assert_eq!(b.bounds_lo, h.bounds_lo);
        assert_eq!(b.extent, h.extent);
        assert_eq!(b.flags, h.flags);
        assert_eq!(built.starts, file.starts);
        // The splats were in Morton order already: the same, tail included.
        assert_eq!(built.chunks, file.chunks);
        let mut differ = 0usize;
        for ((lr, lb), (fr, fb)) in built.levels.iter().zip(&file.levels) {
            assert_eq!(lr, fr);
            assert_eq!(lb.n, fb.n);
            assert_eq!(lb.tail, fb.tail, "cells of level {lr}");
            for (a, e) in lb.positions.iter().zip(&fb.positions) {
                assert!(close(*a, *e, 1e-4), "level {lr} position {a} vs {e}");
            }
            for g in 0..lb.n {
                // The base colour within an f16 step or two.
                for (w, half) in [(2, 16), (3, 0), (3, 16)] {
                    let a = f16_of(lb.shape[g * 4 + w] >> half & 0xffff);
                    let e = f16_of(fb.shape[g * 4 + w] >> half & 0xffff);
                    assert!(close(a, e, 4e-3), "level {lr} group {g} colour: {a} vs {e}");
                }
                // The same covariance: where two axes are as long, which way
                // they point is the rounding's choice, and the thinnest axis
                // of a coarse group is a difference of large moments.
                let (ca, ce) = (cov_of(&lb.shape[g * 4..]), cov_of(&fb.shape[g * 4..]));
                let big = (0..3).map(|k| ce[k][k]).fold(0.0f32, f32::max);
                let diff = (0..9)
                    .map(|k| (ca[k / 3][k % 3] - ce[k / 3][k % 3]).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    diff <= 1e-2 * big,
                    "level {lr} group {g} covariance {ca:?} vs {ce:?}"
                );
                if lb.shape[g * 4..g * 4 + 4] != fb.shape[g * 4..g * 4 + 4] {
                    differ += 1;
                }
                let na = crate::athc::unpack_normal(lb.normals[g]);
                let ne = crate::athc::unpack_normal(fb.normals[g]);
                let nd: f32 = na.iter().zip(&ne).map(|(a, e)| a * e).sum();
                assert!(nd > 0.9999, "level {lr} group {g} normal {na:?} vs {ne:?}");
            }
        }
        // Not to the bit: the device sums in another order (positions one
        // ulp apart), which moves the thin axis of a flat group and the
        // in-plane turn of a round one. 495 of the 715 merged shapes differ
        // in some bit, every one within the tolerances above.
        let total: usize = built.levels.iter().map(|(_, b)| b.n).sum();
        assert!(differ < total);
        // And the file it writes reads back.
        let bytes = built.write().unwrap();
        assert_eq!(AthcFile::read(&bytes).unwrap().write().unwrap(), bytes);
    }

    fn synthetic(n: usize) -> CloudStreams {
        let mut s = CloudStreams {
            count: n,
            coefficients: 1,
            linear: true,
            ..Default::default()
        };
        for i in 0..n {
            let t = i as f32 / n as f32;
            s.positions
                .extend_from_slice(&[t, (t * 37.0).sin() * 0.3, (t * 11.0).cos() * 0.2]);
            s.rotations
                .extend_from_slice(&[0.0, 0.0, (t * 3.0).sin() * 0.5, 1.0]);
            s.scales.extend_from_slice(&[0.01, 0.005, 0.001]);
            s.opacities
                .push(if i % 97 == 5 { 0.0 } else { 0.5 + 0.5 * t });
            s.sh.extend_from_slice(&[t, 1.0 - t, 0.5]);
            s.normals.extend_from_slice(&[0.0, 1.0, 0.0]);
            s.metallic.push(0.0);
            s.roughness.push(0.25);
            s.transmission.push(if i % 2 == 0 { 1.0 } else { 0.0 });
            s.transfer_direct
                .extend((0..16).map(|k| (i + k) as f32 * 1e-3));
            s.transfer_indirect
                .extend((0..48).map(|k| (k as f32) * 1e-2 - t));
            s.transfer_reflected
                .extend((0..48).map(|k| 1.0 + k as f32 * 1e-2));
            s.shadow_bits.extend((0..8).map(|w| {
                if i % 3 == 0 {
                    u32::MAX
                } else {
                    0x5555_0000 + w as u32
                }
            }));
        }
        s
    }

    #[test]
    fn packs_a_transfer_as_athenea_lays_it_out() {
        let s = synthetic(300);
        let p = pack_streams(&s, &BuildOptions::default()).unwrap();
        assert_eq!(p.dropped, 4); // i % 97 == 5: 5, 102, 199, 296
        assert_eq!(p.transfer_count, 112);
        assert_eq!(p.block.transfer.len(), p.block.n * 56);
        // Splat 0: direct 0..16 then indirect then field, f16 pairs, low first.
        let t = &p.block.transfer[..56];
        assert_eq!(f16_of(t[0] & 0xffff), 0.0);
        assert_eq!(f16_of(t[0] >> 16), f16::from_f32(1e-3).to_f32());
        assert_eq!(f16_of(t[8] & 0xffff), 0.0); // indirect[0] = 0 - t, t = 0
        assert_eq!(f16_of(t[32] & 0xffff), 1.0); // field[0]
        assert_eq!(p.block.pbr[0], 0 | (64 << 8) | (255 << 16));
        assert_eq!(p.block.shadow_bits.len(), p.block.n * 8);

        let reduced = BuildOptions {
            transfer: TransferKeep::Count(64),
            ..Default::default()
        };
        let r = pack_streams(&s, &reduced).unwrap();
        assert_eq!(
            (r.transfer_count, r.block.transfer.len() / r.block.n),
            (64, 32)
        );
        assert_eq!(r.block.transfer[..32], p.block.transfer[..32]);
        let deg2 = BuildOptions {
            transfer: TransferKeep::Count(84),
            ..Default::default()
        };
        let d = pack_streams(&s, &deg2).unwrap();
        assert_eq!(d.block.transfer.len() / d.block.n, 42);
        // direct 9, indirect 27 (the first nine coefficients), field 48.
        let v = |w: &[u32], k: usize| {
            f16_of(if k % 2 == 0 {
                w[k / 2] & 0xffff
            } else {
                w[k / 2] >> 16
            })
        };
        let (full, two) = (&p.block.transfer[56..112], &d.block.transfer[42..84]);
        for k in 0..9 {
            assert_eq!(v(two, k), v(full, k));
        }
        for k in 0..27 {
            assert_eq!(v(two, 9 + k), v(full, 16 + k));
        }
        for k in 0..48 {
            assert_eq!(v(two, 36 + k), v(full, 64 + k));
        }
        assert!(pack_streams(
            &s,
            &BuildOptions {
                transfer: TransferKeep::Count(10),
                ..Default::default()
            }
        )
        .is_err());
    }

    #[test]
    fn builds_levels_with_every_extra_and_round_trips() {
        let s = synthetic(5000);
        let o = BuildOptions {
            chunk_splats: 1024,
            ..Default::default()
        };
        let p = pack_streams(&s, &o).unwrap();
        let f = build_lod(&p, &o).unwrap();
        let h = f.header;
        assert_eq!(
            h.flags,
            FLAG_NORMALS | FLAG_LINEAR | FLAG_MATERIAL | FLAG_TRANSFER
        );
        assert_eq!(
            (
                f.extra.transfer_count,
                f.extra.transfer_words,
                f.extra.shadow_words
            ),
            (112, 56, 8)
        );
        assert_eq!(h.chunks as usize, p.block.n.div_ceil(1024));
        // Each level's groups cover the splats; cells grow by a level each.
        for (r, b) in &f.levels {
            assert_eq!(b.transfer.len(), b.n * 56);
            assert_eq!(b.shadow_bits.len(), b.n * 8);
            assert!(b.positions.chunks(4).all(|p| p[3] > 0.0 && p[3] <= 0.99));
            assert!(
                b.tail.windows(2).all(|w| w[0] < w[1]),
                "level {r} cells sorted"
            );
        }
        // The splats are a permutation of the packed ones.
        let all = f.splats();
        assert_eq!(all.n, p.block.n);
        let mut a: Vec<u32> = all.positions.iter().map(|v| v.to_bits()).collect();
        let mut b: Vec<u32> = p.block.positions.iter().map(|v| v.to_bits()).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);
        // Every third splat is all open: a group merges its bits by weight.
        let bytes = f.write().unwrap();
        let back = AthcFile::read(&bytes).unwrap();
        assert_eq!(back.write().unwrap(), bytes);
        assert_eq!(
            (back.levels.clone(), back.chunks.clone()),
            (f.levels.clone(), f.chunks.clone())
        );
        let v3 = crate::athc_v3::write_v3(&back, crate::athc_v3::COMPRESSION_GZIP).unwrap();
        assert_eq!(
            crate::athc_v3::read_v3(&v3).unwrap().write().unwrap(),
            bytes
        );
    }

    #[test]
    fn carries_the_curvature_in_v3_and_drops_it_from_v2() {
        let mut s = synthetic(3000);
        for i in 0..s.count {
            // A ball of radius 1/(10 + i % 7), a little anisotropic.
            let k = 10.0 + (i % 7) as f32;
            s.curvature.extend_from_slice(&[k, 0.25, k + 0.5]);
        }
        let o = BuildOptions {
            chunk_splats: 1024,
            ..Default::default()
        };
        let p = pack_streams(&s, &o).unwrap();
        assert_eq!(p.block.curvature.len(), p.block.n * 2);
        assert_eq!(f16_of(p.block.curvature[0] & 0xffff), 10.0);
        assert_eq!(f16_of(p.block.curvature[1] >> 16), 0.0);
        let f = build_lod(&p, &o).unwrap();
        assert_eq!(f.extra.curvature_words, 2);
        assert!(f.has_curvature());
        for (_, b) in &f.levels {
            assert_eq!(b.curvature.len(), b.n * 2);
            // A merged group's mean curvature is a mean of 10 .. 16.5.
            for w in b.curvature.chunks(2) {
                let h = 0.5 * (f16_of(w[0] & 0xffff) + f16_of(w[1] & 0xffff));
                assert!((10.0..=16.6).contains(&h), "{h}");
            }
        }
        // v2 leaves it out: the same file as the cloud without it.
        let plain = f.without_curvature();
        let v2 = f.write().unwrap();
        assert_eq!(v2, plain.write().unwrap());
        assert_eq!(AthcFile::read(&v2).unwrap().extra.curvature_words, 0);
        // v3 keeps it, in CURV right after the shadow bits.
        let v3 = crate::athc_v3::write_v3(&f, crate::athc_v3::COMPRESSION_GZIP).unwrap();
        let layout = crate::athc_v3::parse_v3(&v3).unwrap();
        assert_eq!(layout.extra.curvature_words, 2);
        let ids: Vec<&str> = layout.sections.iter().map(|s| s.id.name()).collect();
        assert_eq!(
            ids,
            ["CORE", "SHRS", "MATL", "SHAD", "CURV", "TXDI", "TXIN", "TXFD"]
        );
        let back = crate::athc_v3::read_v3(&v3).unwrap();
        assert_eq!(
            (back.levels.clone(), back.chunks.clone()),
            (f.levels.clone(), f.chunks.clone())
        );
        // A v3 without it is the file it was.
        let v3_plain = crate::athc_v3::write_v3(&plain, crate::athc_v3::COMPRESSION_NONE).unwrap();
        assert_eq!(
            crate::athc_v3::read_v3(&v3_plain).unwrap().write().unwrap(),
            v2
        );
        // A page of a chunk's sections keeps it with the relight streams.
        use crate::athc_v3::{athv_sections_page, read_sections_page, wanted_sections, Want};
        let v3u = crate::athc_v3::write_v3(&f, crate::athc_v3::COMPRESSION_NONE).unwrap();
        let lu = crate::athc_v3::parse_v3(&v3u).unwrap();
        let chunk = lu.blocks.iter().find(|b| b.kind == 1).unwrap();
        for (want, kept) in [
            (
                Want {
                    material: true,
                    transfer_values: 0,
                },
                false,
            ),
            (Want::all(&lu.extra), true),
        ] {
            let picks = wanted_sections(&lu.sections, &lu.extra, want);
            let parts: Vec<_> = picks
                .iter()
                .map(|&k| {
                    let span = chunk.spans[k];
                    (
                        &lu.sections[k],
                        &v3u[span.offset as usize..(span.offset + span.stored as u64) as usize],
                        span.raw,
                    )
                })
                .collect();
            let page = athv_sections_page(
                &lu.v2_headers(),
                0,
                chunk.n,
                0,
                want.transfer_values,
                &parts,
            );
            let (_, x, block, _) = read_sections_page(&page).unwrap();
            assert_eq!(x.curvature_words, if kept { 2 } else { 0 });
            assert_eq!(
                block.curvature,
                if kept {
                    f.chunks[0].curvature.clone()
                } else {
                    Vec::new()
                }
            );
        }
        // Decoded, it is an attribute of three halves.
        let specs = crate::athc::attrib_specs(&layout.header, &layout.extra);
        let c = specs
            .iter()
            .find(|s| s.name == crate::athc::CURVATURE_ATTRIBUTE)
            .unwrap();
        assert_eq!((c.format.as_str(), c.components), ("f16", 3));
    }
}
