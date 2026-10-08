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
    encode_quaternion, is_flat_ln, pack_halves, pack_normal, AthcBlock, AthcFile, AthcHeader,
    ExtraHeader, FLAG_LINEAR, FLAG_MATERIAL, FLAG_NORMALS, FLAG_TRANSFER, OLDEST_VERSION, SURFEL_LN, VERSION,
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
    /// `primvars:athenea:splat:coverage`: a merged gaussian's W / A without
    /// athenea's 0.99 cap (`athenea decimate`), empty when absent. Where it
    /// is, it is the opacity (past 1, Spark's LoD opacity when decoded).
    pub coverage: Vec<f32>,
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
    /// `thinWalled` (int[]): a thin wall, carried on the pbr word (bit 24);
    /// read only beside the material, as `GpuClouds` reads it
    pub thin_walled: Vec<u32>,
    /// `schlickMetal` (int[]): a Schlick metal (OpenPBR, glTF), pbr bit 25
    pub schlick_metal: Vec<u32>,
    /// The layers over the base (`packing.slang` `SplatLobes`), one array
    /// each, empty where the stage does not carry it: `specularWeight`,
    /// `specularColor` (3), `specularIor`, `coatWeight`, `coatRoughness`,
    /// `coatIor`, `sheenColor` (3), `sheenRoughness`, `coatDarkening`.
    pub lobes: LobeStreams,
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
    /// `transferZonal`: 10 a splat, two zonal lobes in the splat's frame
    /// (a skinned cloud's transfer); read in place of `transferDirect`.
    pub transfer_zonal: Vec<f32>,
    /// The rig (`athc_skin`): `skel:jointIndices` / `skel:jointWeights`,
    /// `skin_influences` a splat, empty when nothing carries the cloud.
    pub joint_indices: Vec<i32>,
    pub joint_weights: Vec<f32>,
    pub skin_influences: usize,
    /// Joints the skeleton has (a joint past it holds nothing).
    pub joint_count: u32,
    /// `jointWeightGradients`: `2 (skin_influences - 1)` halves a splat (bits),
    /// or empty.
    pub weight_gradients: Vec<u16>,
}

/// The layers' arrays (`GpuClouds.cpp` `lobeArrays`, in its order).
#[derive(Clone, Debug, Default)]
pub struct LobeStreams {
    pub specular_weight: Vec<f32>,
    pub specular_colour: Vec<f32>,
    pub specular_ior: Vec<f32>,
    pub coat_weight: Vec<f32>,
    pub coat_roughness: Vec<f32>,
    pub coat_ior: Vec<f32>,
    pub sheen_colour: Vec<f32>,
    pub sheen_roughness: Vec<f32>,
    pub coat_darkening: Vec<f32>,
}

/// A splat's layers, as `packing.slang`'s `SplatLobes` holds them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplatLobes {
    pub specular_weight: f32,
    pub specular_colour: [f32; 3],
    pub specular_ior: f32,
    pub coat_weight: f32,
    pub coat_roughness: f32,
    pub coat_ior: f32,
    pub coat_darkening: f32,
    pub sheen_colour: [f32; 3],
    pub sheen_roughness: f32,
}

impl SplatLobes {
    /// `plainLobes()`: what a gaussian without layers reflects with, and what
    /// `streams.slang` writes for an array the stage does not carry.
    pub const PLAIN: SplatLobes = SplatLobes {
        specular_weight: 1.0,
        specular_colour: [1.0; 3],
        specular_ior: 1.5,
        coat_weight: 0.0,
        coat_roughness: 0.0,
        coat_ior: 1.5,
        coat_darkening: 0.0,
        sheen_colour: [0.0; 3],
        sheen_roughness: 0.0,
    };
}

/// `lobeByte`
fn lobe_byte(v: f32) -> u32 {
    (saturate(v) * 255.0 + 0.5) as u32
}
/// `iorByte`: `1 + byte / 128`
fn ior_byte(ior: f32) -> u32 {
    ((ior - 1.0) * 128.0 + 0.5).clamp(0.0, 255.0) as u32
}
/// `coatByte`: `1 + bits / 64` in seven bits, darkening in the eighth
fn coat_byte(ior: f32, darkening: f32) -> u32 {
    (((ior - 1.0) * 64.0 + 0.5).clamp(0.0, 127.0) as u32) | if darkening >= 0.5 { 128 } else { 0 }
}

/// `packing.slang` `packLobes` (txf 89a04d9): (specular colour, weight),
/// (coat weight, roughness, coat index and darkening, specular index),
/// (sheen colour, roughness), a byte each.
pub fn pack_lobes(l: &SplatLobes) -> [u32; 3] {
    let w0 = lobe_byte(l.specular_colour[0])
        | (lobe_byte(l.specular_colour[1]) << 8)
        | (lobe_byte(l.specular_colour[2]) << 16)
        | (lobe_byte(l.specular_weight) << 24);
    let w1 = lobe_byte(l.coat_weight)
        | (lobe_byte(l.coat_roughness) << 8)
        | (coat_byte(l.coat_ior, l.coat_darkening) << 16)
        | (ior_byte(l.specular_ior) << 24);
    let w2 = lobe_byte(l.sheen_colour[0])
        | (lobe_byte(l.sheen_colour[1]) << 8)
        | (lobe_byte(l.sheen_colour[2]) << 16)
        | (lobe_byte(l.sheen_roughness) << 24);
    [w0, w1, w2]
}

/// `packing.slang` `unpackLobes`.
pub fn unpack_lobes(w: [u32; 3]) -> SplatLobes {
    let b = |word: u32, at: u32| ((word >> (at * 8)) & 0xff) as f32 / 255.0;
    SplatLobes {
        specular_colour: [b(w[0], 0), b(w[0], 1), b(w[0], 2)],
        specular_weight: b(w[0], 3),
        coat_weight: b(w[1], 0),
        coat_roughness: b(w[1], 1),
        coat_ior: 1.0 + ((w[1] >> 16) & 0x7f) as f32 / 64.0,
        coat_darkening: if (w[1] >> 23) & 1 != 0 { 1.0 } else { 0.0 },
        specular_ior: 1.0 + (w[1] >> 24) as f32 / 128.0,
        sheen_colour: [b(w[2], 0), b(w[2], 1), b(w[2], 2)],
        sheen_roughness: b(w[2], 3),
    }
}

impl LobeStreams {
    /// Whether an array holds a value (or a colour) for each of `n` splats:
    /// `GpuClouds.cpp`'s `whole` (a shorter array is no array).
    fn whole(v: &[f32], n: usize, per: usize) -> bool {
        !v.is_empty() && v.len() >= n * per
    }
    fn arrays(&self) -> [(&Vec<f32>, usize); 9] {
        [
            (&self.specular_weight, 1),
            (&self.specular_colour, 3),
            (&self.specular_ior, 1),
            (&self.coat_weight, 1),
            (&self.coat_roughness, 1),
            (&self.coat_ior, 1),
            (&self.sheen_colour, 3),
            (&self.sheen_roughness, 1),
            (&self.coat_darkening, 1),
        ]
    }
    fn arrays_mut(&mut self) -> [(&mut Vec<f32>, usize, f32); 9] {
        [
            (&mut self.specular_weight, 1, 1.0),
            (&mut self.specular_colour, 3, 1.0),
            (&mut self.specular_ior, 1, 1.5),
            (&mut self.coat_weight, 1, 0.0),
            (&mut self.coat_roughness, 1, 0.0),
            (&mut self.coat_ior, 1, 1.5),
            (&mut self.sheen_colour, 3, 0.0),
            (&mut self.sheen_roughness, 1, 0.0),
            (&mut self.coat_darkening, 1, 0.0),
        ]
    }
    /// The cloud carries layers where any one array is whole (`haveLobes`).
    pub fn present(&self, n: usize) -> bool {
        self.arrays().iter().any(|(v, per)| Self::whole(v, n, *per))
    }
    /// Splat `i`'s record (`streams.slang`): each array where it is whole,
    /// `plainLobes`' value where it is not.
    pub fn at(&self, n: usize, i: usize) -> SplatLobes {
        let one = |v: &Vec<f32>, d: f32| if Self::whole(v, n, 1) { v[i] } else { d };
        let three = |v: &Vec<f32>, d: f32| {
            if Self::whole(v, n, 3) {
                [v[i * 3], v[i * 3 + 1], v[i * 3 + 2]]
            } else {
                [d; 3]
            }
        };
        SplatLobes {
            specular_weight: one(&self.specular_weight, 1.0),
            specular_colour: three(&self.specular_colour, 1.0),
            specular_ior: one(&self.specular_ior, 1.5),
            coat_weight: one(&self.coat_weight, 0.0),
            coat_roughness: one(&self.coat_roughness, 0.0),
            coat_ior: one(&self.coat_ior, 1.5),
            sheen_colour: three(&self.sheen_colour, 0.0),
            sheen_roughness: one(&self.sheen_roughness, 0.0),
            coat_darkening: one(&self.coat_darkening, 0.0),
        }
    }
    /// `other`'s `m` splats after this cloud's `n`: an array one cloud
    /// carries and the other does not is filled with `plainLobes`' value, as
    /// athenea reads a missing array.
    pub fn append(&mut self, n: usize, mut other: LobeStreams, m: usize) {
        let theirs = other.arrays_mut();
        for ((mine, per, d), (them, _, _)) in self.arrays_mut().into_iter().zip(theirs) {
            let a = Self::whole(mine, n, per);
            let b = Self::whole(them, m, per);
            if !a && !b {
                mine.clear();
                continue;
            }
            mine.truncate(if a { n * per } else { 0 });
            if !a {
                mine.resize(n * per, d);
            }
            if b {
                mine.extend_from_slice(&them[..m * per]);
            } else {
                mine.resize((n + m) * per, d);
            }
        }
    }
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
    /// The octree's frame, per object: 0 world-aligned from the cloud's
    /// lower bound (athenea's); otherwise turned by a rotation and shifted
    /// by up to a level-6 cell, both hashed from the seed, so no
    /// world-aligned lattice of merged cells survives (Cook's jitter, as
    /// far as one octree allows: the levels nest, so all of them turn
    /// together). The header's `bounds_lo` and `extent` are then the
    /// turned frame's.
    pub frame_seed: u32,
    /// Merged nodes written as surfels where they and all their members
    /// are discs (`SurfelNodes`; by default where the cloud has surfels).
    pub surfel_nodes: SurfelNodes,
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
            frame_seed: 0,
            surfel_nodes: SurfelNodes::Auto,
        }
    }
}

fn hash_unit(seed: u32, k: u32) -> f32 {
    let mut h = seed.wrapping_mul(0x9E37_79B9) ^ k.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h as f32 / u32::MAX as f32
}

/// The rotation and the shift (a fraction of the frame's extent) of
/// `BuildOptions::frame_seed`.
pub fn octree_frame(seed: u32) -> ([[f32; 3]; 3], [f32; 3]) {
    // A uniform random rotation (Shoemake), from three hashed numbers.
    let (u1, u2, u3) = (hash_unit(seed, 1), hash_unit(seed, 2), hash_unit(seed, 3));
    let tau = std::f32::consts::TAU;
    let q = [
        (1.0 - u1).sqrt() * (tau * u2).sin(),
        (1.0 - u1).sqrt() * (tau * u2).cos(),
        u1.sqrt() * (tau * u3).sin(),
        u1.sqrt() * (tau * u3).cos(),
    ];
    let shift = [4, 5, 6].map(|k| hash_unit(seed, k) / 64.0);
    (axes_of_quaternion(q), shift)
}

/// A built `.athc`'s splats as the packed cloud they were built from (in
/// Morton order), so its levels can be built again (`build_lod`).
pub fn packed_of(file: &AthcFile) -> PackedCloud {
    let mut block = file.splats();
    block.tail.clear();
    PackedCloud {
        block,
        rest_per_colour: file.header.rest_per_colour,
        sh_words: file.header.sh_words,
        transfer_count: file.extra.transfer_count,
        linear: file.header.has(FLAG_LINEAR),
        bounds_min: file.header.bounds_min,
        bounds_max: file.header.bounds_max,
        dropped: 0,
        skin_influences: file.extra.skin_influences,
        skin_gradient_words: file.extra.skin_gradient_words,
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
    /// The skin's layout (`athc_skin`): influences and gradient words a
    /// splat in `block.skin`, 0 for a cloud nothing carries.
    pub skin_influences: u32,
    pub skin_gradient_words: u32,
}

fn half_safe(v: f32) -> f32 {
    v.clamp(SURFEL_LN, -SURFEL_LN)
}

/// A shape word's three scales (a surfel's flat axis exactly 0).
fn scales_of(sh: &[u32]) -> [f32; 3] {
    [f16_of(sh[1] & 0xffff).exp(), f16_of(sh[1] >> 16).exp(), f16_of(sh[2] & 0xffff).exp()]
}

/// A gaussian's area: its two widest scales multiplied, the mass athenea's
/// merges weigh (`opacity x area`). Not s0 s1 s2 / min(s), which is the same
/// for a 3D gaussian and 0 for a surfel (whose thinnest is exactly 0).
fn disc_area(s: [f32; 3]) -> f32 {
    let mut t = s;
    t.sort_by(|a, b| b.total_cmp(a));
    t[0] * t[1]
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

/// `packPbr` (splat_encoding.slang): a thin wall rides on the transmission
/// two higher, a Schlick metal four higher (`streams.slang`).
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
    if keep == TransferKeep::None {
        return Ok((0, 0, 0, false));
    }
    // A zonal transfer (a skinned cloud's) is read in place of the direct
    // half, and kept whole: ten values (`kTransferZonalCount`).
    if !s.transfer_zonal.is_empty() {
        if s.transfer_zonal.len() != n * 10 {
            bail!("transferZonal holds {} values for {n} splats", s.transfer_zonal.len());
        }
        return match keep {
            TransferKeep::Full | TransferKeep::Count(10) => Ok((10, 10, 0, false)),
            TransferKeep::Count(c) => bail!("a zonal transfer keeps its 10 values, not {c}"),
            TransferKeep::None => unreachable!(),
        };
    }
    if s.transfer_direct.is_empty() {
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
    Ok(pack_streams_kept(s, o)?.0)
}

/// [`pack_streams`], and the stream index of each packed record (those that
/// passed validation, in order): what a per-splat array beside the streams
/// (a light layer, `athl`) needs to follow the cloud.
pub fn pack_streams_kept(s: &CloudStreams, o: &BuildOptions) -> Result<(PackedCloud, Vec<u32>)> {
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
    need("coverage", &s.coverage, 1)?;
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
    // The marks and the layers, only beside the material: athenea's
    // `writeAthc` keeps lobes only where it keeps pbr.
    let thin = pbr && !s.thin_walled.is_empty() && s.thin_walled.len() >= n;
    let schlick = pbr && !s.schlick_metal.is_empty() && s.schlick_metal.len() >= n;
    let lobes = pbr && s.lobes.present(n);
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
    let zonal = transfer_count == 10 && !s.transfer_zonal.is_empty();
    // The rig: athenea's packed influences (390670e) and the gradients as
    // the file holds them.
    let k_skin = if s.joint_indices.is_empty() { 0 } else { s.skin_influences };
    if k_skin > 0 && (s.joint_indices.len() != n * k_skin || s.joint_weights.len() != n * k_skin || k_skin > 16) {
        bail!(
            "{} joint indices and {} weights for {n} splats of {k_skin} influences",
            s.joint_indices.len(),
            s.joint_weights.len()
        );
    }
    let g_skin = if k_skin > 1 && s.weight_gradients.len() == n * 2 * (k_skin - 1) { k_skin - 1 } else { 0 };
    if k_skin > 1 && g_skin == 0 && !s.weight_gradients.is_empty() {
        bail!("jointWeightGradients holds {} halves for {n} splats of {k_skin} joints", s.weight_gradients.len());
    }
    let mut skin_out_of_range = 0usize;

    let mut b = AthcBlock::default();
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    let mut dropped = 0;
    let mut kept: Vec<u32> = Vec::with_capacity(n);
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
        // A scale of exactly 0 is a surfel's flat axis (2DGS): kept as the
        // most negative half, which exp() decodes back to 0.
        let ls = sc.map(|v| if v == 0.0 { SURFEL_LN } else { v.max(1e-30).ln() });
        let flat = sc.iter().filter(|&&v| v == 0.0).count();
        let finite = p.iter().all(|v| v.is_finite())
            && flat <= 1
            && sc.iter().zip(ls.iter()).all(|(&v, l)| v == 0.0 || (l.is_finite() && l.abs() < 60.0));
        if !(finite && a >= 1.0 / 255.0) {
            dropped += 1;
            continue;
        }
        kept.push(i as u32);
        b.n += 1;
        // athenea's uncapped coverage, where it writes one, is the
        // merged gaussian's whole mass: kept past 1 (the LoD opacity).
        let o = match s.coverage.get(i) {
            Some(&c) if c.is_finite() && c > 0.0 => c.max(saturate(a)),
            _ => saturate(a),
        };
        b.positions.extend_from_slice(&[p[0], p[1], p[2], o]);
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
            let mark = (if thin && s.thin_walled[i] != 0 { 2.0 } else { 0.0 })
                + (if schlick && s.schlick_metal[i] != 0 { 4.0 } else { 0.0 });
            b.pbr.push(pack_pbr(m, r, t + mark));
        }
        if lobes {
            b.lobes.extend_from_slice(&pack_lobes(&s.lobes.at(n, i)));
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
        if zonal {
            for w in 0..5 {
                let lo16 = f16_bits(half_safe(s.transfer_zonal[i * 10 + 2 * w])) & 0xffff;
                let hi16 = f16_bits(half_safe(s.transfer_zonal[i * 10 + 2 * w + 1])) & 0xffff;
                b.transfer.push((hi16 << 16) | lo16);
            }
        } else if transfer_count > 0 {
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
        if k_skin > 0 {
            let (words, bad) = crate::athc_skin::pack_gaussian(
                &s.joint_indices[i * k_skin..(i + 1) * k_skin],
                &s.joint_weights[i * k_skin..(i + 1) * k_skin],
                s.joint_count.max(1),
            );
            skin_out_of_range += bad as usize;
            b.skin.extend_from_slice(&words);
            for w in 0..g_skin {
                let at = i * 2 * g_skin + 2 * w;
                b.skin.push(s.weight_gradients[at] as u32 | (s.weight_gradients[at + 1] as u32) << 16);
            }
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
    if skin_out_of_range > 0 {
        eprintln!("warning: {skin_out_of_range} splats name a joint past the skeleton or a weight outside [0, 1]");
    }
    Ok((
        PackedCloud {
            block: b,
            rest_per_colour: keep as u32,
            sh_words: sh_words as u32,
            transfer_count,
            linear: s.linear,
            bounds_min: lo,
            bounds_max: hi,
            dropped,
            skin_influences: k_skin as u32,
            skin_gradient_words: g_skin as u32,
        },
        kept,
    ))
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
/// The rows `order` of `b` (any of them, in that order), tail left empty.
pub fn reorder(b: &AthcBlock, order: &[u32]) -> AthcBlock {
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
        n: order.len(),
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
        skin: pick(&b.skin),
        lod_size: Vec::new(),
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
    /// Merged nodes may be written as surfels (`SurfelNodes`).
    planar: bool,
}

impl MomentLayout {
    /// The moments of a cloud of `keep` rest harmonics a colour: the head,
    /// the harmonics, the normals' sum if any, and last the count of members
    /// that are not discs (`is_disc`).
    fn new(keep: usize, normals: bool, planar: bool) -> Self {
        Self { keep, normals, stride: MOMENTS_HEAD + keep * 3 + if normals { 3 } else { 0 } + 1, planar }
    }
    fn not_discs(&self) -> usize {
        self.stride - 1
    }
}

/// athenea's rule (`decimate`, surfels-web/PLAN.md §2): a member is a disc
/// when its thinnest axis is at most this times its middle one, and a
/// cluster of discs whose own thinnest axis is too is written as a surfel.
pub const DISC_RATIO: f32 = 0.12;
/// Our guard on top: the cluster's normals spread at most this (1 - |mean
/// normal|, `athc::normal_spread`: 0.03 is about 28 degrees between two
/// equal faces) and its thin axis within 0.9 of their mean.
pub const DISC_SPREAD: f32 = 0.03;

/// Whether merged LoD nodes are written as surfels where flat.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SurfelNodes {
    /// Where the cloud brings surfels of its own (a scale of exactly 0).
    #[default]
    Auto,
    On,
    Off,
}

impl SurfelNodes {
    fn applies(self, b: &AthcBlock) -> bool {
        match self {
            SurfelNodes::On => true,
            SurfelNodes::Off => false,
            SurfelNodes::Auto => has_surfels(b),
        }
    }
}

/// Whether any splat of `b` is a surfel (an axis exactly 0).
pub fn has_surfels(b: &AthcBlock) -> bool {
    (0..b.n).any(|i| {
        let sh = &b.shape[i * 4..i * 4 + 4];
        is_flat_ln(f16_of(sh[1] & 0xffff)) || is_flat_ln(f16_of(sh[1] >> 16)) || is_flat_ln(f16_of(sh[2] & 0xffff))
    })
}

/// [`DISC_RATIO`] on three scales.
fn is_disc(s: [f32; 3]) -> bool {
    let mut t = s;
    t.sort_by(|a, b| a.total_cmp(b));
    t[0] <= DISC_RATIO * t[1]
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
            let wt = p[3] * disc_area(sc);
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
            if !is_disc(sc) {
                m[at + l.not_discs()] += 1.0;
            }
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
        let area = disc_area(s);
        let opacity = (m[at] / area.max(1e-30)).min(0.99);
        // A cluster of discs that is a disc itself is written as one: its
        // thinnest axis exactly 0 (a surfel), the same area and mass.
        let thin = shortest(s);
        let flat = l.planar && m[at + l.not_discs()] == 0.0 && is_disc(s) && {
            if l.normals {
                let nb = at + MOMENTS_HEAD + l.keep * 3;
                let sum = [m[nb], m[nb + 1], m[nb + 2]];
                let len = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
                let along = (axes[0][thin] * sum[0] + axes[1][thin] * sum[1] + axes[2][thin] * sum[2]).abs();
                1.0 - len / weight <= DISC_SPREAD && along >= 0.9 * len
            } else {
                true
            }
        };
        let base = [
            m[at + 10] / weight,
            m[at + 11] / weight,
            m[at + 12] / weight,
        ];
        b.positions
            .extend_from_slice(&[mu[0], mu[1], mu[2], opacity]);
        let mut ls = s.map(f32::ln);
        if flat {
            ls[thin] = SURFEL_LN;
        }
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

/// `octEncode` (common/octahedral.slang): a unit direction to [0, 1]^2.
pub fn oct_encode(n: [f32; 3]) -> [f32; 2] {
    let l1 = (n[0].abs() + n[1].abs() + n[2].abs()).max(1e-20);
    let n = n.map(|v| v / l1);
    let p = if n[2] >= 0.0 {
        [n[0], n[1]]
    } else {
        let sx = if n[0] >= 0.0 { 1.0 } else { -1.0 };
        let sy = if n[1] >= 0.0 { 1.0 } else { -1.0 };
        [(1.0 - n[1].abs()) * sx, (1.0 - n[0].abs()) * sy]
    };
    p.map(|v| v * 0.5 + 0.5)
}

/// `octDecode`: [0, 1]^2 back to a unit direction.
pub fn oct_decode(f: [f32; 2]) -> [f32; 3] {
    let f = f.map(|v| v * 2.0 - 1.0);
    let mut n = [f[0], f[1], 1.0 - f[0].abs() - f[1].abs()];
    let t = (-n[2]).clamp(0.0, 1.0);
    n[0] += if n[0] >= 0.0 { -t } else { t };
    n[1] += if n[1] >= 0.0 { -t } else { t };
    normalize3(n)
}

/// Element i's frame: its axes as columns (`axesOfQuaternion`), local to world.
fn frame_of(b: &AthcBlock, i: usize) -> M3 {
    axes_of_quaternion(crate::athc::decode_quaternion(b.shape[i * 4]))
}

fn to_world(r: &M3, a: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|row| r[row][0] * a[0] + r[row][1] * a[1] + r[row][2] * a[2])
}

fn to_local(r: &M3, w: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|col| r[0][col] * w[0] + r[1][col] * w[1] + r[2][col] * w[2])
}

fn half_at(words: &[u32], k: usize) -> f32 {
    let w = words[k / 2];
    f16_of(if k & 1 == 0 { w & 0xffff } else { w >> 16 })
}

/// What lies in each member's own frame merged in the world: the zonal
/// transfer's lobes (`zonal`, 10 halves a splat, two lobes of an octahedral
/// axis and three coefficients) or the curvature (2 words: the shape
/// operator S in the first two axes, xx xy yy, and a fourth half). Each
/// member's is turned into the world -- a lobe's axis by its frame
/// (`splatTransferFrame`: R = axesOfQuaternion, columns the axes), the
/// curvature as U S U^T (U its first two axes) -- averaged by mass (opacity
/// x area, as the moments weigh) and turned into the merged element's own
/// frame (`frames`: what it holds once oriented, `athc::orient_merged`).
/// Averaging the stored halves instead mixes frames that turn: a curved
/// panel's or a merged disc's lobes and curvature pointed anywhere.
fn frame_merge(
    source: &[u32],
    words: usize,
    zonal: bool,
    starts: &[u32],
    splats: &AthcBlock,
    frames: &AthcBlock,
) -> Vec<u32> {
    let groups = starts.len();
    let mut target = vec![0u32; groups * words];
    for g in 0..groups {
        let first = starts[g] as usize;
        let end = if g + 1 < groups { starts[g + 1] as usize } else { splats.n };
        let mass = |i: usize| {
            splats.positions[i * 4 + 3].max(0.0) * disc_area(scales_of(&splats.shape[i * 4..i * 4 + 4]))
        };
        let total: f32 = (first..end).map(mass).sum();
        let weight = |i: usize| if total > 0.0 { mass(i) / total } else { 1.0 / (end - first) as f32 };
        let heaviest = (first..end).max_by(|&a, &b| weight(a).total_cmp(&weight(b))).unwrap_or(first);
        let node = frame_of(frames, g);
        let mut halves = vec![0.0f32; words * 2];
        if zonal {
            for lobe in 0..2 {
                let base = lobe * 5;
                let (mut axis, mut z) = ([0.0f32; 3], [0.0f32; 3]);
                for i in first..end {
                    let v = &source[i * words..i * words + words];
                    let w = weight(i);
                    let a = to_world(&frame_of(splats, i), oct_decode([half_at(v, base), half_at(v, base + 1)]));
                    for k in 0..3 {
                        axis[k] += w * a[k];
                        z[k] += w * half_at(v, base + 2 + k);
                    }
                }
                // Opposed axes cancel: the heaviest member's then.
                if axis.iter().map(|v| v * v).sum::<f32>() < 1e-8 {
                    let v = &source[heaviest * words..heaviest * words + words];
                    axis = to_world(&frame_of(splats, heaviest), oct_decode([half_at(v, base), half_at(v, base + 1)]));
                }
                let sq = oct_encode(to_local(&node, normalize3(axis)));
                halves[base] = sq[0];
                halves[base + 1] = sq[1];
                halves[base + 2..base + 5].copy_from_slice(&z);
            }
        } else {
            let mut t = [[0.0f32; 3]; 3];
            let mut rest = 0.0f32;
            for i in first..end {
                let v = &source[i * words..i * words + words];
                let w = weight(i);
                let r = frame_of(splats, i);
                let (xx, xy, yy) = (half_at(v, 0), half_at(v, 1), half_at(v, 2));
                for a in 0..3 {
                    for b in 0..3 {
                        let (u, vv) = ((r[a][0], r[a][1]), (r[b][0], r[b][1]));
                        t[a][b] += w * (u.0 * (xx * vv.0 + xy * vv.1) + u.1 * (xy * vv.0 + yy * vv.1));
                    }
                }
                rest += w * half_at(v, 3);
            }
            // S' = U'^T T U' in the merged element's first two axes.
            let s = |c: usize, d: usize| -> f32 {
                let mut sum = 0.0;
                for a in 0..3 {
                    for b in 0..3 {
                        sum += node[a][c] * t[a][b] * node[b][d];
                    }
                }
                sum
            };
            halves[0] = s(0, 0);
            halves[1] = s(0, 1);
            halves[2] = s(1, 1);
            halves[3] = rest;
        }
        for w in 0..words {
            target[g * words + w] =
                pack_halves(half_safe(halves[2 * w]), half_safe(halves[2 * w + 1]));
        }
    }
    target
}

/// The transfer of `cloud`'s groups: zonal lobes in each one's frame
/// (`frame_merge`), the world's harmonics by opacity (`extras_merge`).
fn merge_transfer(transfer_count: u32, splats: &AthcBlock, starts: &[u32], frames: &AthcBlock) -> Vec<u32> {
    let words = splats.transfer.len() / splats.n;
    if transfer_count == 10 {
        frame_merge(&splats.transfer, words, true, starts, splats, frames)
    } else {
        extras_merge(&splats.transfer, words, 1, starts, splats)
    }
}

/// `LodBuilder::build` and `writeAthc`'s header: the `.athc` of a packed
/// cloud, its splats in Morton order, its levels merged.
/// The order [`build_lod`] puts `cloud`'s splats in (the file's splat order:
/// `file.splats()` record k is `cloud.block` record `order[k]`).
pub fn lod_order(cloud: &PackedCloud, o: &BuildOptions) -> Vec<u32> {
    let (codes, _, _) = lod_codes(cloud, o);
    let mut order: Vec<u32> = (0..codes.len() as u32).collect();
    order.sort_by_key(|&i| codes[i as usize]);
    order
}

/// The octree's frame and each splat's Morton code in it: (codes, lower
/// bound, extent).
fn lod_codes(cloud: &PackedCloud, o: &BuildOptions) -> (Vec<u32>, [f32; 3], f32) {
    let src = &cloud.block;
    let n = src.n;
    // The octree's frame: world-aligned from the lower bound, or turned
    // and shifted (`frame_seed`).
    let (turn, shift) = if o.frame_seed == 0 {
        ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0; 3])
    } else {
        octree_frame(o.frame_seed)
    };
    let framed = |i: usize| -> [f32; 3] {
        let p = &src.positions[i * 4..i * 4 + 3];
        [0, 1, 2].map(|r| turn[r][0] * p[0] + turn[r][1] * p[1] + turn[r][2] * p[2])
    };
    let (mut lo, mut hi) = (cloud.bounds_min, cloud.bounds_max);
    if o.frame_seed != 0 {
        lo = [f32::MAX; 3];
        hi = [f32::MIN; 3];
        for i in 0..n {
            let q = framed(i);
            for k in 0..3 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
    }
    let mut extent = 0.0f32;
    for k in 0..3 {
        extent = extent.max(hi[k] - lo[k]);
    }
    let extent = extent.max(1e-6) * 1.0001 * if o.frame_seed == 0 { 1.0 } else { 1.0 + 1.0 / 64.0 };
    for k in 0..3 {
        lo[k] -= shift[k] * extent;
    }
    let codes: Vec<u32> = (0..n)
        .map(|i| {
            let p = framed(i);
            let e = extent.max(1e-20);
            morton30([(p[0] - lo[0]) / e, (p[1] - lo[1]) / e, (p[2] - lo[2]) / e])
        })
        .collect();
    (codes, lo, extent)
}

pub fn build_lod(cloud: &PackedCloud, o: &BuildOptions) -> Result<AthcFile> {
    let src = &cloud.block;
    let n = src.n;
    if n == 0 {
        bail!("an empty cloud has no levels of detail");
    }
    if !src.emission.is_empty() {
        bail!("emission is not built here yet");
    }
    if !src.lobes.is_empty() && src.pbr.is_empty() {
        bail!("lobes without the material: a .athc keeps them only beside pbr");
    }
    let (codes, lo, extent) = lod_codes(cloud, o);
    build_lod_from_codes(cloud, o, &codes, lo, extent)
}

/// [`build_lod`] over given 30-bit codes instead of the octree's Morton
/// codes: level r's groups are the splats whose codes agree in their top
/// 3r bits (`groups_of`), a group's children the next level's within it.
/// Any hierarchy of at most eight children a group whose leaves all sit at
/// the finest level can be written so (`athc_merge`'s error-driven tree);
/// `lo` and `extent` go to the header (the decoder widens level r's
/// groups by `MERGED_FILL` x extent / 2^r).
pub fn build_lod_from_codes(cloud: &PackedCloud, o: &BuildOptions, codes: &[u32], lo: [f32; 3], extent: f32) -> Result<AthcFile> {
    let src = &cloud.block;
    let n = src.n;
    if codes.len() != n {
        bail!("{} codes for {} splats", codes.len(), n);
    }
    // Morton order: a stable sort of the codes, as the radix sort is.
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

    let l = MomentLayout::new(
        cloud.rest_per_colour as usize,
        !splats.normals.is_empty(),
        o.surfel_nodes.applies(&splats),
    );
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
        // The frame the decoder reads each group in (`orient_merged`, third
        // axis the normal), for what lies in a frame to be merged into it.
        // The levels keep athenea's moments as they are.
        let mut frames = block.clone();
        crate::athc::orient_merged(&mut frames);
        let per = |v: &[u32]| v.len() / n;
        if !splats.pbr.is_empty() {
            block.pbr = extras_merge(&splats.pbr, per(&splats.pbr), 0, &level.starts, &splats);
        }
        // The layers as the material: a cell takes its first gaussian's
        // (`Lod.cpp` `extrasOf`: lobes, 3 words, mode 0).
        if !splats.lobes.is_empty() {
            block.lobes = extras_merge(&splats.lobes, 3, 0, &level.starts, &splats);
        }
        if !splats.transfer.is_empty() {
            block.transfer = merge_transfer(cloud.transfer_count, &splats, &level.starts, &frames);
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
        // The curvature as a tensor in the world, averaged by mass and
        // read in the group's own frame (`frame_merge`).
        if !splats.curvature.is_empty() {
            block.curvature = frame_merge(&splats.curvature, 2, false, &level.starts, &splats, &frames);
        }
        // The skin: each group carried by its splats' heaviest joints.
        if !splats.skin.is_empty() {
            block.skin = crate::athc_skin::skin_merge(
                &splats.skin,
                cloud.skin_influences as usize,
                cloud.skin_gradient_words as usize,
                &level.starts,
                &splats,
            );
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
        extra.lobes_words = if whole.lobes.is_empty() { 0 } else { 3 };
    }
    if !whole.transfer.is_empty() {
        extra.transfer_count = cloud.transfer_count;
        extra.transfer_words = cloud.transfer_count.div_ceil(2);
        extra.shadow_words = (whole.shadow_bits.len() / n) as u32;
    }
    if !whole.curvature.is_empty() {
        extra.curvature_words = 2;
    }
    if !whole.skin.is_empty() {
        extra.skin_influences = cloud.skin_influences;
        extra.skin_gradient_words = cloud.skin_gradient_words;
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

/// The grid key of splat `i` for cells of side `cell` from `lo`.
fn cell_key(b: &AthcBlock, lo: [f32; 3], i: usize, cell: f32) -> u64 {
    let p = &b.positions[i * 4..i * 4 + 3];
    let q = |k: usize| (((p[k] - lo[k]) / cell).max(0.0) as u64).min((1 << 21) - 1);
    (q(0) << 42) | (q(1) << 21) | q(2)
}

/// The cell side that brings `cloud` down to about `target` splats with
/// [`reduce_cells`]: the smallest whose groups are at most `target`.
pub fn cell_for_target(cloud: &PackedCloud, target: usize) -> f32 {
    let src = &cloud.block;
    let lo = cloud.bounds_min;
    let mut extent = 0.0f32;
    for k in 0..3 {
        extent = extent.max(cloud.bounds_max[k] - cloud.bounds_min[k]);
    }
    let extent = extent.max(1e-6);
    let groups_at = |cell: f32| -> usize {
        let mut keys: Vec<u64> = (0..src.n).map(|i| cell_key(src, lo, i, cell)).collect();
        keys.sort_unstable();
        keys.dedup();
        keys.len()
    };
    let (mut small, mut large) = (extent / (1 << 20) as f32, extent);
    for _ in 0..24 {
        let mid = (small * large).sqrt();
        if groups_at(mid) > target {
            small = mid;
        } else {
            large = mid;
        }
    }
    large
}

/// Which input splats each output splat of a reduction was made from:
/// output j is `members[starts[j]..starts[j + 1]]` (the last runs to the
/// end), each with its weight in the merged colour (opacity x the area of
/// the two longest axes, as the moments and `reduce_thin` weigh it). What a
/// per-splat array beside the cloud (a light layer) needs to be merged as
/// the base colour was.
#[derive(Clone, Debug, Default)]
pub struct SplatRuns {
    pub members: Vec<u32>,
    pub starts: Vec<u32>,
    pub weights: Vec<f32>,
}

/// A packed splat's colour weight: opacity x the area of its two longest axes.
fn colour_weight(b: &AthcBlock, i: usize) -> f32 {
    let sh = &b.shape[i * 4..i * 4 + 4];
    b.positions[i * 4 + 3].max(0.0) * disc_area(scales_of(sh))
}

fn runs_of(src: &AthcBlock, order: &[u32], starts: &[u32]) -> SplatRuns {
    SplatRuns {
        members: order.to_vec(),
        starts: starts.to_vec(),
        weights: order.iter().map(|&i| colour_weight(src, i as usize)).collect(),
    }
}

/// A cloud made lighter for the web: the splats are grouped by a grid of
/// cells of side `cell` (from the cloud's lower bound), and each group is
/// merged into one Gaussian as a LoD level merges one (athenea's moments; the
/// material from its first splat, the transfer averaged by opacity, the open
/// directions by vote). `fill` widens the two long axes of each merged
/// Gaussian (a level's moments give a patch's spread, about 0.29 of the cell,
/// which leaves see-through seams when the merged cloud is looked at close
/// up); its opacity is the merged weight over the widened area, as `finalize`
/// computes it, at most 0.99. [`cell_for_target`] finds the cell for a count.
pub fn reduce_cells(cloud: &PackedCloud, cell: f32, fill: f32) -> Result<PackedCloud> {
    Ok(reduce_cells_runs(cloud, cell, fill)?.0)
}

/// [`reduce_cells`], and the runs each merged splat was made from.
pub fn reduce_cells_runs(cloud: &PackedCloud, cell: f32, fill: f32) -> Result<(PackedCloud, SplatRuns)> {
    let src = &cloud.block;
    let n = src.n;
    if !src.emission.is_empty() {
        bail!("emission is not reduced here yet");
    }
    if cell.is_nan() || cell <= 0.0 {
        bail!("the cell must be positive");
    }
    let lo = cloud.bounds_min;
    let key_of = |i: usize, cell: f32| cell_key(src, lo, i, cell);
    let keys: Vec<u64> = (0..n).map(|i| key_of(i, cell)).collect();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&i| keys[i as usize]);
    let splats = reorder(src, &order);
    let mut group = Vec::with_capacity(n);
    let mut starts = Vec::new();
    for (k, &i) in order.iter().enumerate() {
        if k == 0 || keys[i as usize] != keys[order[k - 1] as usize] {
            starts.push(k as u32);
        }
        group.push(starts.len() as u32 - 1);
    }
    let level = Level {
        groups: starts.len(),
        cells: vec![0; starts.len()],
        group,
        starts,
    };
    let l = MomentLayout::new(
        cloud.rest_per_colour as usize,
        !splats.normals.is_empty(),
        SurfelNodes::Auto.applies(&splats),
    );
    let sh_words = cloud.sh_words as usize;
    let moments = leaf_moments(&splats, &level, &l, sh_words);
    let mut block = finalize(&moments, level.groups, &l, sh_words);
    crate::athc::orient_merged(&mut block);
    let runs = runs_of(src, &order, &level.starts);
    if fill != 1.0 {
        let f = fill.max(1e-3);
        for g in 0..block.n {
            let sh = &mut block.shape[g * 4..g * 4 + 4];
            let ls = [
                f16_of(sh[1] & 0xffff),
                f16_of(sh[1] >> 16),
                f16_of(sh[2] & 0xffff),
            ];
            let k = (0..3).min_by(|&a, &b| ls[a].total_cmp(&ls[b])).unwrap();
            let mut grown = ls;
            for (a, v) in grown.iter_mut().enumerate() {
                if a != k {
                    *v += f.ln();
                }
            }
            let base0 = sh[2] >> 16;
            sh[1] = pack_halves(half_safe(grown[0]), half_safe(grown[1]));
            sh[2] = (f16_bits(half_safe(grown[2])) & 0xffff) | (base0 << 16);
            // finalize clamped at 0.99 before the area grew: the weight
            // over the widened area.
            let weight = moments[g * l.stride];
            let area = disc_area(grown.map(f32::exp));
            block.positions[g * 4 + 3] = (weight / area.max(1e-30)).min(0.99);
        }
    }
    drop(moments);
    let per = |v: &[u32]| v.len() / n;
    if !splats.pbr.is_empty() {
        block.pbr = extras_merge(&splats.pbr, per(&splats.pbr), 0, &level.starts, &splats);
    }
    if !splats.lobes.is_empty() {
        block.lobes = extras_merge(&splats.lobes, 3, 0, &level.starts, &splats);
    }
    if !splats.transfer.is_empty() {
        block.transfer = merge_transfer(cloud.transfer_count, &splats, &level.starts, &block);
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
    if !splats.curvature.is_empty() {
        block.curvature = frame_merge(&splats.curvature, 2, false, &level.starts, &splats, &block);
    }
    if !splats.skin.is_empty() {
        block.skin = crate::athc_skin::skin_merge(
            &splats.skin,
            cloud.skin_influences as usize,
            cloud.skin_gradient_words as usize,
            &level.starts,
            &splats,
        );
    }
    Ok((PackedCloud {
        block,
        rest_per_colour: cloud.rest_per_colour,
        sh_words: cloud.sh_words,
        transfer_count: cloud.transfer_count,
        linear: cloud.linear,
        bounds_min: cloud.bounds_min,
        bounds_max: cloud.bounds_max,
        dropped: cloud.dropped,
        skin_influences: cloud.skin_influences,
        skin_gradient_words: cloud.skin_gradient_words,
    }, runs))
}

/// A cloud thinned for the web to about one splat in `ratio`: the splats in
/// Morton order are cut into runs of about `ratio`, and each run keeps its
/// most typical splat (its normal nearest the run's mean; its place, turn,
/// normal and material), its two long axes
/// grown by the square root of the run's length so the run's area stays
/// covered, its base colour and transfer averaged over the run by opacity
/// and its open directions voted (as a LoD level merges them). Unlike
/// [`reduce_cells`], every kept Gaussian has a real splat's shape on the
/// surface, so a surface cut by the grid shows no seams or moiré.
pub fn reduce_thin(cloud: &PackedCloud, ratio: f32) -> Result<PackedCloud> {
    Ok(reduce_thin_runs(cloud, ratio)?.0)
}

/// [`reduce_thin`], and the runs each kept splat stands for.
pub fn reduce_thin_runs(cloud: &PackedCloud, ratio: f32) -> Result<(PackedCloud, SplatRuns)> {
    let src = &cloud.block;
    let n = src.n;
    if !src.emission.is_empty() {
        bail!("emission is not reduced here yet");
    }
    if ratio.is_nan() || ratio < 1.0 {
        bail!("the thinning ratio must be at least 1");
    }
    let target = ((n as f64 / ratio as f64).round() as usize).clamp(1, n);
    let lo = cloud.bounds_min;
    let mut extent = 0.0f32;
    for k in 0..3 {
        extent = extent.max(cloud.bounds_max[k] - cloud.bounds_min[k]);
    }
    let e = extent.max(1e-6) * 1.0001;
    let codes: Vec<u32> = (0..n)
        .map(|i| {
            let p = &src.positions[i * 4..i * 4 + 3];
            morton30([(p[0] - lo[0]) / e, (p[1] - lo[1]) / e, (p[2] - lo[2]) / e])
        })
        .collect();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&i| codes[i as usize]);
    let splats = reorder(src, &order);
    let starts: Vec<u32> = (0..target)
        .map(|j| (j as u64 * n as u64 / target as u64) as u32)
        .collect();
    let end = |j: usize| {
        if j + 1 < target {
            starts[j + 1] as usize
        } else {
            n
        }
    };
    // Each run's most typical splat: the one whose normal is nearest the
    // run's mean (a run across a panel's edge keeps a splat of the panel,
    // not one of the rim folding into the gap); the middle one without normals.
    let reps: Vec<u32> = (0..target)
        .map(|j| {
            let (first, last) = (starts[j] as usize, end(j));
            if splats.normals.is_empty() {
                return ((first + last) / 2) as u32;
            }
            let normal = |i: usize| crate::athc::unpack_normal(splats.normals[i]);
            let mut mean = [0.0f32; 3];
            for i in first..last {
                let (nv, o) = (normal(i), splats.positions[i * 4 + 3].max(0.0));
                for k in 0..3 {
                    mean[k] += o * nv[k];
                }
            }
            (first..last)
                .max_by(|&a, &b| {
                    let dot = |i: usize| {
                        let nv = normal(i);
                        nv[0] * mean[0] + nv[1] * mean[1] + nv[2] * mean[2]
                    };
                    dot(a).total_cmp(&dot(b))
                })
                .unwrap() as u32
        })
        .collect();
    let runs = runs_of(src, &order, &starts);
    let mut block = reorder(&splats, &reps);
    block.n = target; // reorder keeps a permutation's count
    for j in 0..target {
        let (first, last) = (starts[j] as usize, end(j));
        // The run's base colour, by opacity x area as the moments weigh it.
        let (mut w, mut c) = (0.0f32, [0.0f32; 3]);
        for i in first..last {
            let sh = &splats.shape[i * 4..i * 4 + 4];
            let wt = splats.positions[i * 4 + 3].max(0.0) * disc_area(scales_of(sh));
            w += wt;
            c[0] += wt * f16_of(sh[2] >> 16);
            c[1] += wt * f16_of(sh[3] & 0xffff);
            c[2] += wt * f16_of(sh[3] >> 16);
        }
        let sh = &mut block.shape[j * 4..j * 4 + 4];
        let ls = [
            f16_of(sh[1] & 0xffff),
            f16_of(sh[1] >> 16),
            f16_of(sh[2] & 0xffff),
        ];
        let thin = (0..3).min_by(|&a, &b| ls[a].total_cmp(&ls[b])).unwrap();
        let grow = 0.5 * ((last - first) as f32).ln();
        let g = [0, 1, 2].map(|a| if a == thin { ls[a] } else { ls[a] + grow });
        let base = if w > 0.0 {
            c.map(|v| v / w)
        } else {
            [
                f16_of(sh[2] >> 16),
                f16_of(sh[3] & 0xffff),
                f16_of(sh[3] >> 16),
            ]
        };
        sh[1] = pack_halves(half_safe(g[0]), half_safe(g[1]));
        sh[2] = pack_halves(half_safe(g[2]), half_safe(base[0]));
        sh[3] = pack_halves(half_safe(base[1]), half_safe(base[2]));
    }
    let per = |v: &[u32]| v.len() / n;
    if !splats.transfer.is_empty() {
        block.transfer = merge_transfer(cloud.transfer_count, &splats, &starts, &block);
    }
    if !splats.shadow_bits.is_empty() {
        block.shadow_bits = extras_merge(
            &splats.shadow_bits,
            per(&splats.shadow_bits),
            2,
            &starts,
            &splats,
        );
    }
    if !splats.curvature.is_empty() {
        block.curvature = frame_merge(&splats.curvature, 2, false, &starts, &splats, &block);
    }
    if !splats.skin.is_empty() {
        block.skin = crate::athc_skin::skin_merge(
            &splats.skin,
            cloud.skin_influences as usize,
            cloud.skin_gradient_words as usize,
            &starts,
            &splats,
        );
    }
    Ok((PackedCloud {
        block,
        rest_per_colour: cloud.rest_per_colour,
        sh_words: cloud.sh_words,
        transfer_count: cloud.transfer_count,
        linear: cloud.linear,
        bounds_min: cloud.bounds_min,
        bounds_max: cloud.bounds_max,
        dropped: cloud.dropped,
        skin_influences: cloud.skin_influences,
        skin_gradient_words: cloud.skin_gradient_words,
    }, runs))
}

/// How open a splat is to the environment: its shadow bits' open
/// directions, or without them its direct transfer's first value.
fn exposure(b: &AthcBlock, i: usize) -> Option<f32> {
    if !b.shadow_bits.is_empty() {
        let per = b.shadow_bits.len() / b.n;
        let words = &b.shadow_bits[i * per..(i + 1) * per];
        return Some(words.iter().map(|w| w.count_ones()).sum::<u32>() as f32);
    }
    if !b.transfer.is_empty() {
        let per = b.transfer.len() / b.n;
        return Some(f16_of(b.transfer[i * per] & 0xffff));
    }
    None
}

/// Drops the hidden back of a panel with thickness: a splat whose normal
/// faces away from another splat of the cloud no more than `thickness`
/// behind it (back to back: normals opposed, each behind the other, within
/// half the thickness sideways) and less than half as open to the
/// environment as that splat (shadow bits, or the direct transfer). A
/// mesh2splat bake of a solidified panel (Car_Paint_Main: every door and
/// fender is a 3 mm shell) has both faces on the same grid; the inner one
/// is dark (it sees the cabin), and wherever the draw's centre sort
/// interleaves the two (splats as wide as the shell is thick: the LoD's
/// merged levels, a thinned cloud) the back shows through the front in
/// moiré bands. A panel open on both sides keeps both faces. Returns the
/// cloud and how many splats were dropped.
pub fn drop_hidden_backs(cloud: &PackedCloud, thickness: f32) -> Result<(PackedCloud, usize)> {
    let (kept, keep) = drop_hidden_backs_kept(cloud, thickness)?;
    Ok((kept, cloud.block.n - keep.len()))
}

/// [`drop_hidden_backs`], and the input index of each splat kept.
pub fn drop_hidden_backs_kept(cloud: &PackedCloud, thickness: f32) -> Result<(PackedCloud, Vec<u32>)> {
    let src = &cloud.block;
    let n = src.n;
    if src.normals.is_empty() {
        bail!("dropping hidden backs needs normals");
    }
    if exposure(src, 0).is_none() && n > 0 {
        bail!("dropping hidden backs needs shadow bits or a transfer");
    }
    if thickness.is_nan() || thickness <= 0.0 {
        bail!("the shell thickness must be positive");
    }
    let lateral = 0.5 * thickness;
    let apart = thickness / 16.0;
    let cell_of = |i: usize| {
        let p = &src.positions[i * 4..i * 4 + 3];
        [0, 1, 2].map(|k| (p[k] / thickness).floor() as i32)
    };
    let mut cells: std::collections::HashMap<[i32; 3], Vec<u32>> = std::collections::HashMap::new();
    for i in 0..n {
        cells.entry(cell_of(i)).or_default().push(i as u32);
    }
    let normals: Vec<[f32; 3]> = src.normals.iter().map(|&w| crate::athc::unpack_normal(w)).collect();
    let open: Vec<f32> = (0..n).map(|i| exposure(src, i).unwrap_or(0.0)).collect();
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let keep: Vec<u32> = (0..n)
        .filter(|&i| {
            let (pi, ni) = (&src.positions[i * 4..i * 4 + 3], normals[i]);
            let c = cell_of(i);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(list) = cells.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else {
                            continue;
                        };
                        for &j in list {
                            let j = j as usize;
                            let nj = normals[j];
                            if dot(ni, nj) > -0.7 || 2.0 * open[i] >= open[j] {
                                continue;
                            }
                            let pj = &src.positions[j * 4..j * 4 + 3];
                            let d = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
                            let (behind_i, behind_j) = (-dot(d, ni), dot(d, nj));
                            if behind_i < apart || behind_i > thickness || behind_j < apart {
                                continue;
                            }
                            if dot(d, d) - behind_i * behind_i <= lateral * lateral {
                                return false;
                            }
                        }
                    }
                }
            }
            true
        })
        .map(|i| i as u32)
        .collect();
    let mut block = reorder(src, &keep);
    block.n = keep.len();
    Ok((
        PackedCloud {
            block,
            rest_per_colour: cloud.rest_per_colour,
            sh_words: cloud.sh_words,
            transfer_count: cloud.transfer_count,
            linear: cloud.linear,
            bounds_min: cloud.bounds_min,
            bounds_max: cloud.bounds_max,
            dropped: cloud.dropped,
            skin_influences: cloud.skin_influences,
            skin_gradient_words: cloud.skin_gradient_words,
        },
        keep,
    ))
}

/// The splats whose centres lie inside the box `lo`..`hi` (`usd-athc --box`):
/// a small piece of a large bake, every stream kept, for tests.
pub fn crop_box(cloud: &PackedCloud, lo: [f32; 3], hi: [f32; 3]) -> PackedCloud {
    crop_box_kept(cloud, lo, hi).0
}

/// [`crop_box`], and the input index of each splat kept.
pub fn crop_box_kept(cloud: &PackedCloud, lo: [f32; 3], hi: [f32; 3]) -> (PackedCloud, Vec<u32>) {
    let src = &cloud.block;
    let keep: Vec<u32> = (0..src.n)
        .filter(|&i| (0..3).all(|k| src.positions[i * 4 + k] >= lo[k] && src.positions[i * 4 + k] <= hi[k]))
        .map(|i| i as u32)
        .collect();
    let mut block = reorder(src, &keep);
    block.n = keep.len();
    (PackedCloud { block, ..cloud.clone() }, keep)
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
            ..Default::default()
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

    /// A surfel's flat axis (scale exactly 0) survives packing as exp() == 0;
    /// two flat axes (a line) are dropped.
    #[test]
    fn keeps_surfel_zero_scales() {
        let mut s = synthetic(4);
        s.opacities = vec![1.0; 4];
        s.scales[2] = 0.0;
        s.scales[3 + 1] = 0.0;
        s.scales[6] = 0.0;
        s.scales[7] = 0.0;
        let (cloud, kept) = pack_streams_kept(&s, &BuildOptions::default()).unwrap();
        assert_eq!(kept, vec![0, 1, 3]);
        let sc = |i: usize| {
            let sh = &cloud.block.shape[i * 4..i * 4 + 4];
            [f16_of(sh[1] & 0xffff).exp(), f16_of(sh[1] >> 16).exp(), f16_of(sh[2] & 0xffff).exp()]
        };
        assert_eq!(sc(0)[2], 0.0);
        assert_eq!(sc(1)[1], 0.0);
        assert!(sc(2).iter().all(|&v| v > 0.0));
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

    /// The web reductions of usd-athc: counts, grown shapes, streams kept.
    #[test]
    fn thins_and_merges_two_cards_for_the_web() {
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
            ..Default::default()
        };
        let n = cloud.block.n;
        let area = |b: &AthcBlock| -> f32 {
            (0..b.n)
                .map(|i| {
                    let sh = &b.shape[i * 4..i * 4 + 4];
                    let l = [
                        f16_of(sh[1] & 0xffff),
                        f16_of(sh[1] >> 16),
                        f16_of(sh[2] & 0xffff),
                    ];
                    l.iter().sum::<f32>() - l.iter().copied().fold(f32::MAX, f32::min)
                })
                .map(f32::exp)
                .sum()
        };

        let thin = reduce_thin(&cloud, 4.0).unwrap();
        let t = &thin.block;
        assert_eq!(t.n, (n as f32 / 4.0).round() as usize);
        assert_eq!(t.positions.len(), t.n * 4);
        assert_eq!(t.shape.len(), t.n * 4);
        for (a, b) in [
            (&t.normals, &cloud.block.normals),
            (&t.pbr, &cloud.block.pbr),
            (&t.sh, &cloud.block.sh),
        ] {
            assert_eq!(a.len() * n, b.len() * t.n);
        }
        // The kept splats cover about the area the cloud covered.
        let (before, after) = (area(&cloud.block), area(t));
        assert!((after / before - 1.0).abs() < 0.5, "{before} -> {after}");
        // Every kept splat is one of the cloud's, in place.
        for i in 0..t.n {
            let p = &t.positions[i * 4..i * 4 + 3];
            assert!((0..n).any(|j| &cloud.block.positions[j * 4..j * 4 + 3] == p));
        }
        assert_eq!(
            build_lod(&thin, &BuildOptions::default())
                .unwrap()
                .header
                .count as usize,
            t.n
        );

        let cell = cell_for_target(&cloud, n / 4);
        let merged = reduce_cells(&cloud, cell, 1.4).unwrap();
        assert!(
            merged.block.n <= n / 4 && merged.block.n > 0,
            "{}",
            merged.block.n
        );
        assert_eq!(merged.block.shape.len(), merged.block.n * 4);
        assert!(merged
            .block
            .positions
            .chunks(4)
            .all(|p| p[3] > 0.0 && p[3] <= 0.99));
    }

    /// Spark's draw of one gaussian at squared Mahalanobis radius `z2`, for a
    /// decoded opacity (`spark_lod_opacity`: past 1, a LoD coverage).
    fn spark_alpha(stored: f32, z2: f32) -> f32 {
        let g = (-0.5 * z2).exp();
        if stored <= 1.0 {
            return stored * g;
        }
        let d = (stored * 4.0 - 3.0).min(5.0);
        let e = ((d * d - 1.0) / std::f32::consts::E).exp();
        1.0 - (1.0 - g).powf(e)
    }

    /// A merged level drawn by Spark over the middle of a flat sheet: the
    /// mean light let through where the cells are pixels wide (each gaussian
    /// as drawn, composited), and where they are under a pixel (each
    /// gaussian is its mass there: the antialiasing pays a sub-pixel
    /// gaussian back by its integral, `paid` x the peak).
    fn sheet_transmittance(b: &AthcBlock, side: f32, margin: f32) -> (f32, f32) {
        let mut nodes = Vec::new();
        for i in 0..b.n {
            let p = &b.positions[i * 4..i * 4 + 4];
            let c = cov_of(&b.shape[i * 4..i * 4 + 4]);
            // In the sheet's plane (z): the 2D covariance and its inverse.
            let (a, bb, d) = (c[0][0], c[0][1], c[1][1]);
            let det = a * d - bb * bb;
            let stored = crate::athc::spark_lod_opacity(p[3]);
            let mass_peak = if stored <= 1.0 { stored } else { (stored * 4.0 - 3.0).min(5.0) };
            nodes.push((p[0], p[1], d / det, -bb / det, a / det, stored, mass_peak * 2.0 * std::f32::consts::PI * det.sqrt()));
        }
        let (mut t, mut n) = (0.0f64, 0);
        let steps = 64;
        for u in 0..steps {
            for v in 0..steps {
                let x = margin + (side - 2.0 * margin) * (u as f32 + 0.5) / steps as f32;
                let y = margin + (side - 2.0 * margin) * (v as f32 + 0.5) / steps as f32;
                let mut tr = 1.0f32;
                for &(px, py, ia, ib, ic, stored, _) in &nodes {
                    let (dx, dy) = (x - px, y - py);
                    let z2 = ia * dx * dx + 2.0 * ib * dx * dy + ic * dy * dy;
                    if z2 < 25.0 {
                        tr *= 1.0 - spark_alpha(stored, z2).min(1.0);
                    }
                }
                t += tr as f64;
                n += 1;
            }
        }
        let inside = |&&(px, py, ..): &&(f32, f32, f32, f32, f32, f32, f32)| {
            px > margin && px < side - margin && py > margin && py < side - margin
        };
        let mass: f32 = nodes.iter().filter(inside).map(|n| n.6).sum();
        let tau = mass / (side - 2.0 * margin).powi(2);
        ((t / n as f64) as f32, (-tau).exp())
    }

    /// The Corvette, seen far off, let the inside through its paint: a
    /// closed shell's merged levels at athenea's opacity min(W / A, 0.99)
    /// and athenea's moment shape leave most of every cell open. As decoded
    /// (the whole coverage, `athc::uncap_levels`, drawn as Spark's LoD
    /// opacity, and each cell widened, `athc::widen_merged`) the sheet stays
    /// shut at every level, whether its cells are pixels wide or under one.
    /// The pawn's glass head drawn from its merged LoD levels was a checker
    /// of dark and bright cells and showed no lens: the relight reads a
    /// splat's curvature on its third axis' side (`faces`), and a merged
    /// gaussian's axes were its moments' eigenvectors in any order and sign.
    /// As decoded (`athc::orient_merged`, through `widen_merged`) every
    /// merged gaussian of a sphere has its thinnest axis third, along its
    /// stored normal.
    #[test]
    fn merged_levels_keep_the_normal_as_the_third_axis() {
        use crate::athc::{high_half, low_half};
        // A glass ball's cap: 11 mm, splats 0.25 mm on a 0.2 mm lattice of
        // latitude and longitude, their third axis the outward normal.
        let mut s = CloudStreams { coefficients: 1, linear: true, ..Default::default() };
        let r = 0.011f32;
        let step = 0.0002f32;
        let rings = (0.5 * std::f32::consts::PI * r / step) as usize;
        for a in 0..rings {
            let theta = (a as f32 + 0.5) * step / r;
            let around = ((2.0 * std::f32::consts::PI * r * theta.sin() / step) as usize).max(1);
            for b in 0..around {
                let phi = (b as f32 + 0.5) / around as f32 * 2.0 * std::f32::consts::PI;
                let n = glam::Vec3::new(theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos());
                let q = glam::Quat::from_rotation_arc(glam::Vec3::Z, n);
                s.positions.extend_from_slice(&(n * r).to_array());
                s.rotations.extend_from_slice(&q.to_array());
                s.scales.extend_from_slice(&[0.00025, 0.00025, 0.000025]);
                s.opacities.push(0.1448);
                s.sh.extend_from_slice(&[0.5, 0.5, 0.5]);
                s.normals.extend_from_slice(&n.to_array());
                s.count += 1;
            }
        }
        let o = BuildOptions::default();
        let file = build_lod(&pack_streams(&s, &o).unwrap(), &o).unwrap();
        let along = |b: &crate::athc::AthcBlock| {
            let (mut aligned, mut thinnest) = (0usize, 0usize);
            for i in 0..b.n {
                let q = crate::athc::decode_quaternion(b.shape[i * 4]);
                let z = glam::Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize() * glam::Vec3::Z;
                let n = glam::Vec3::from_array(crate::athc::unpack_normal(b.normals[i]));
                if z.dot(n) > 0.9 {
                    aligned += 1;
                }
                let ln = [low_half(b.shape[i * 4 + 1]), high_half(b.shape[i * 4 + 1]), low_half(b.shape[i * 4 + 2])];
                if ln[2] <= ln[0] && ln[2] <= ln[1] {
                    thinnest += 1;
                }
            }
            (aligned as f32 / b.n as f32, thinnest as f32 / b.n as f32)
        };
        let mut worst_before = 1.0f32;
        for (level, block) in &file.levels {
            if block.n < 16 {
                continue;
            }
            let (a0, _) = along(block);
            let mut b = block.clone();
            crate::athc::widen_merged(&mut b, file.header.extent / (1u64 << *level) as f32);
            let (a1, t1) = along(&b);
            eprintln!("level {level} ({} groups): third axis along the normal {a0:.2} as merged, {a1:.2} decoded (thinnest {t1:.2})", block.n);
            worst_before = worst_before.min(a0);
            assert!(a1 > 0.99 && t1 > 0.99, "level {level}: {a1} {t1}");
        }
        assert!(worst_before < 0.7, "the merged axes were the bug: {worst_before}");
    }

    /// A thin sheet of glass is drawn at its opacity less the 1/255 athenea's
    /// coverage adds a splat (splat_project's `alphaOwn`): merged by area
    /// with that 1/255 in, the Corvette's windshield reflected 2.2-2.9 times
    /// as much from its merged levels as from its splats (bright dots where
    /// the LoD mixes them). As decoded, every level reflects what its splats do.
    #[test]
    fn merged_sheets_of_glass_reflect_what_their_splats_do() {
        use crate::athc::{high_half, low_half};
        let mut s = CloudStreams { coefficients: 1, linear: true, ..Default::default() };
        let (side, step) = (0.192f32, 0.002f32);
        let k = (side / step) as usize;
        let alpha = 0.0018 + 1.0 / 255.0;
        for a in 0..k {
            for b in 0..k {
                let jitter = ((a * 7 + b * 13) % 5) as f32 * 0.0001;
                s.positions.extend_from_slice(&[(a as f32 + 0.5) * step + jitter, (b as f32 + 0.5) * step, 0.0]);
                s.rotations.extend_from_slice(&[0.0, 0.0, 0.0, 1.0]);
                s.scales.extend_from_slice(&[0.0018, 0.0018, 0.0002]);
                s.opacities.push(alpha);
                s.sh.extend_from_slice(&[0.5, 0.5, 0.5]);
                s.normals.extend_from_slice(&[0.0, 0.0, 1.0]);
                s.metallic.push(0.0);
                s.roughness.push(0.02);
                s.transmission.push(1.0);
                s.thin_walled.push(1);
                s.count += 1;
            }
        }
        let o = BuildOptions::default();
        let mut file = build_lod(&pack_streams(&s, &o).unwrap(), &o).unwrap();
        assert!(crate::athc::is_sheet(&file.chunks[0], 0));
        let area = |sh: &[u32]| {
            let mut t = [low_half(sh[1]).exp(), high_half(sh[1]).exp(), low_half(sh[2]).exp()];
            t.sort_by(|a, b| b.total_cmp(a));
            (t[0] * t[1]) as f64
        };
        // splat_project's alphaOwn by area: what a block reflects.
        let reflects = |b: &crate::athc::AthcBlock| -> f64 {
            (0..b.n).map(|i| area(&b.shape[i * 4..i * 4 + 4]) * (b.positions[i * 4 + 3] - 1.0 / 255.0).max(0.0) as f64).sum()
        };
        let splats: f64 = file.chunks.iter().map(reflects).sum();
        crate::athc::uncap_levels(&mut file);
        let extent = file.header.extent;
        for (level, block) in &file.levels {
            let mut b = block.clone();
            crate::athc::widen_merged(&mut b, extent / (1u64 << *level) as f32);
            let ratio = reflects(&b) / splats;
            eprintln!("level {level} ({} groups): reflects {ratio:.3} of its splats", b.n);
            assert!((ratio - 1.0).abs() < 0.03, "level {level}: {ratio}");
        }
    }

    #[test]
    fn merged_levels_keep_a_closed_sheet_closed() {
        // A 192 mm sheet of the paint's splats: 2.4 mm wide, 0.24 mm thin,
        // on a 2 mm grid, opaque.
        let mut s = CloudStreams { coefficients: 1, linear: true, ..Default::default() };
        let (side, step) = (0.192f32, 0.002f32);
        let k = (side / step) as usize;
        for a in 0..k {
            for b in 0..k {
                let jitter = ((a * 7 + b * 13) % 5) as f32 * 0.0001;
                s.positions.extend_from_slice(&[(a as f32 + 0.5) * step + jitter, (b as f32 + 0.5) * step, 0.0]);
                s.rotations.extend_from_slice(&[0.0, 0.0, 0.0, 1.0]);
                s.scales.extend_from_slice(&[0.0024, 0.0024, 0.00024]);
                s.opacities.push(1.0);
                s.sh.extend_from_slice(&[0.5, 0.5, 0.5]);
                s.normals.extend_from_slice(&[0.0, 0.0, 1.0]);
                s.count += 1;
            }
        }
        let o = BuildOptions::default();
        let file = build_lod(&pack_streams(&s, &o).unwrap(), &o).unwrap();
        let mut uncapped = file.clone();
        crate::athc::uncap_levels(&mut uncapped);
        let extent = file.header.extent;
        let mut worst_before = 0.0f32;
        for ((level, before), (_, after)) in file.levels.iter().zip(&uncapped.levels) {
            let edge = extent / (1u64 << *level) as f32;
            // Cells from 3 to 30 mm: a few splats to a few hundred each.
            if !(0.003..0.03).contains(&edge) {
                continue;
            }
            let mut after = after.clone();
            crate::athc::widen_merged(&mut after, edge);
            let margin = 1.5 * edge;
            let (t0, s0) = sheet_transmittance(before, side, margin);
            let (t1, s1) = sheet_transmittance(&after, side, margin);
            eprintln!("level {level} (cell {:.1} mm): through as athenea capped {t0:.3} (sub-pixel {s0:.3}), decoded {t1:.4} (sub-pixel {s1:.4})", edge * 1e3);
            worst_before = worst_before.max(t0.max(s0));
            assert!(t1 < 0.01 && s1 < 0.05, "level {level}: {t1} {s1}");
        }
        assert!(worst_before > 0.25, "the capped levels were the bug: {worst_before}");
    }

    #[test]
    fn drops_the_hidden_back_of_a_shell() {
        // Three 3 mm shells on a 2 mm grid: one open outside and closed
        // inside (its back goes), one open on both sides (both stay), one
        // single face (stays).
        let mut s = CloudStreams { coefficients: 1, linear: true, ..Default::default() };
        let mut face = |x0: f32, z: f32, up: bool, open: u32| {
            for a in 0..10 {
                for b in 0..10 {
                    s.positions.extend_from_slice(&[x0 + a as f32 * 0.002, b as f32 * 0.002, z]);
                    s.rotations.extend_from_slice(&[0.0, 0.0, 0.0, 1.0]);
                    s.scales.extend_from_slice(&[0.002, 0.002, 0.0002]);
                    s.opacities.push(1.0);
                    s.sh.extend_from_slice(&[0.5, 0.5, 0.5]);
                    s.normals.extend_from_slice(&[0.0, 0.0, if up { 1.0 } else { -1.0 }]);
                    s.transfer_direct.extend((0..16).map(|_| 0.1));
                    s.shadow_bits.extend((0..8).map(|w| if w < open { u32::MAX } else { 0 }));
                    s.count += 1;
                }
            }
        };
        face(0.0, 0.003, true, 3); // shell 1: outer face, open
        face(0.0, 0.0, false, 0); //           inner face, closed
        face(1.0, 0.003, true, 3); // shell 2: open both sides
        face(1.0, 0.0, false, 2);
        face(2.0, 0.0, true, 0); // a lone face, however closed
        let o = BuildOptions { transfer: TransferKeep::Count(16), ..Default::default() };
        let cloud = pack_streams(&s, &o).unwrap();
        let (kept, dropped) = drop_hidden_backs(&cloud, 0.008).unwrap();
        assert_eq!(dropped, 100);
        let b = &kept.block;
        assert_eq!(b.n, 400);
        for v in [&b.shape, &b.normals, &b.transfer, &b.shadow_bits] {
            assert_eq!(v.len() % b.n, 0);
        }
        // What went is shell 1's inner face, all of it.
        assert!((0..b.n).all(|i| {
            let p = &b.positions[i * 4..i * 4 + 3];
            !(p[0] < 0.5 && p[2] < 0.001)
        }));
        assert!(build_lod(&kept, &o).is_ok());
    }

    #[test]
    fn crops_a_box_with_every_stream() {
        let s = synthetic(3000);
        let o = BuildOptions { transfer: TransferKeep::Count(16), ..Default::default() };
        let cloud = pack_streams(&s, &o).unwrap();
        let b = &cloud.block;
        let (lo, hi) = ([-0.2f32, -0.2, -0.2], [0.3f32, 0.3, 0.3]);
        let inside = |p: &[f32]| (0..3).all(|k| p[k] >= lo[k] && p[k] <= hi[k]);
        let want: Vec<usize> = (0..b.n).filter(|&i| inside(&b.positions[i * 4..i * 4 + 3])).collect();
        assert!(!want.is_empty() && want.len() < b.n);
        let cut = crop_box(&cloud, lo, hi);
        let c = &cut.block;
        assert_eq!(c.n, want.len());
        for (j, &i) in want.iter().enumerate() {
            assert_eq!(c.positions[j * 4..j * 4 + 4], b.positions[i * 4..i * 4 + 4]);
            for (cv, bv) in [(&c.shape, &b.shape), (&c.normals, &b.normals), (&c.transfer, &b.transfer), (&c.shadow_bits, &b.shadow_bits), (&c.pbr, &b.pbr), (&c.lobes, &b.lobes)] {
                let per = bv.len() / b.n;
                assert_eq!(cv[j * per..(j + 1) * per], bv[i * per..(i + 1) * per]);
            }
        }
        assert!(build_lod(&cut, &o).is_ok());
    }

    #[test]
    fn carries_the_curvature_in_v3_and_drops_it_from_v2() {
        let mut s = synthetic(3000);
        // The splats turn about z: their normal is their third axis.
        for n in s.normals.chunks_mut(3) {
            n.copy_from_slice(&[0.0, 0.0, 1.0]);
        }
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
            // A merged group's mean curvature (in its own frame, whose
            // third axis is the normal) is a mean of 10 .. 16.5, a little
            // less where the group's plane tilts off its splats' (by
            // sin^2(tilt) / 2: the frame is the group's eigenvector nearest
            // the normal, up to ~25 degrees off on this curve).
            for w in b.curvature.chunks(2) {
                let h = 0.5 * (f16_of(w[0] & 0xffff) + f16_of(w[1] & 0xffff));
                assert!((8.5..=16.6).contains(&h), "{h}");
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
        // Decoded, it is an attribute of four halves (the shape operator and
        // a merged splat's normal variance).
        let specs = crate::athc::attrib_specs(&layout.header, &layout.extra);
        let c = specs
            .iter()
            .find(|s| s.name == crate::athc::CURVATURE_ATTRIBUTE)
            .unwrap();
        assert_eq!((c.format.as_str(), c.components), ("f16", 4));
    }

    /// The Corvette paint's layers (athenea-renders/tx/clouds/
    /// Car_Paint_Main_tx_m3.usdc, its most common material): a Schlick
    /// metal under a clear coat that darkens what is under it.
    fn paint_lobes(coat: f32) -> SplatLobes {
        SplatLobes {
            specular_weight: 1.0,
            specular_colour: [1.0; 3],
            specular_ior: 1.5,
            coat_weight: coat,
            coat_roughness: 0.046_975_244,
            coat_ior: 1.45,
            coat_darkening: 1.0,
            sheen_colour: [0.0; 3],
            sheen_roughness: 0.5,
        }
    }

    /// `packLobes` on values read from athenea's clouds, against the words
    /// worked out by hand from packing.slang (txf 89a04d9):
    ///   lobeByte(v) = uint(saturate(v) * 255 + 0.5)
    ///   iorByte(i) = uint(clamp((i - 1) * 128 + 0.5, 0, 255))
    ///   coatByte(i, d) = uint(clamp((i - 1) * 64 + 0.5, 0, 127)) | (d >= 0.5 ? 128 : 0)
    #[test]
    fn packs_lobes_as_athenea_does() {
        // Paint: colour ff ff ff, weight ff; coat ff, roughness
        // 0.046975 * 255 + 0.5 = 12.48 -> 0c, coat index 0.45 * 64 + 0.5 =
        // 29.3 -> 1d | 80 = 9d, specular index 0.5 * 128 + 0.5 -> 40; sheen
        // 0, roughness 0.5 * 255 + 0.5 = 128 -> 80.
        assert_eq!(pack_lobes(&paint_lobes(1.0)), [0xffff_ffff, 0x409d_0cff, 0x8000_0000]);
        // paint/s75/lambert.usdc: specular weight 0, coat weight 0, its
        // roughness 0.12821592 -> 33.19 -> 21.
        let lambert = SplatLobes {
            specular_weight: 0.0,
            coat_weight: 0.0,
            coat_roughness: 0.128_215_92,
            ..paint_lobes(0.0)
        };
        assert_eq!(pack_lobes(&lambert), [0x00ff_ffff, 0x409d_2100, 0x8000_0000]);
        // plainLobes packs to what unpacks to it exactly.
        assert_eq!(pack_lobes(&SplatLobes::PLAIN), [0xffff_ffff, 0x4020_0000, 0]);
        assert_eq!(unpack_lobes(pack_lobes(&SplatLobes::PLAIN)), SplatLobes::PLAIN);
        // The edges: indices clamp (3.5 -> 127 in the coat's seven bits,
        // 0.8 -> 0), a darkening under 0.5 sets no bit, values clamp to 0..1.
        let edge = SplatLobes {
            specular_weight: -0.2,
            specular_colour: [0.2, 0.4, 0.6],    // 51.5 -> 51, 102.5 -> 102, 153.5 -> 153
            specular_ior: 2.2,                   // 154.1 -> 154 = 9a
            coat_weight: 1.7,
            coat_roughness: 0.5,                 // 128
            coat_ior: 3.5,
            coat_darkening: 0.49,
            sheen_colour: [0.0, 0.002, 0.998],   // 0, 1.01 -> 1, 254.99 -> 254
            sheen_roughness: 1.2,
        };
        assert_eq!(pack_lobes(&edge), [0x0099_6633, 0x9a7f_80ff, 0xfffe_0100]);
        let ior0 = SplatLobes { specular_ior: 0.8, ..edge };
        assert_eq!(pack_lobes(&ior0)[1] >> 24, 0);
    }

    /// The Schlick and thin marks on the pbr word (`streams.slang`: + 4,
    /// + 2 on the transmission, `packPbr` takes them back off as bits).
    #[test]
    fn packs_the_schlick_and_thin_marks() {
        // lambert.usdc: metallic 0, roughness 0.34377518 -> 88 = 58, Schlick.
        assert_eq!(pack_pbr(0.0, 0.343_775_18, 4.0), 0x0200_5800);
        // The paint: metallic 1, roughness 0.3421304 -> 87.74 -> 87 = 57.
        assert_eq!(pack_pbr(1.0, 0.342_130_4, 4.0), 0x0200_57ff);
        assert_eq!(pack_pbr(0.0, 0.05, 2.0 + 1.0), 0x01ff_0d00);
        assert_eq!(pack_pbr(0.0, 0.05, 4.0 + 2.0 + 0.5), 0x0380_0d00);
    }

    fn with_lobes(n: usize) -> CloudStreams {
        let mut s = synthetic(n);
        s.schlick_metal = (0..n).map(|i| (i % 2) as u32).collect();
        s.thin_walled = (0..n).map(|i| (i % 5 == 0) as u32).collect();
        // Each splat's material its own: the roughness and the coat's
        // roughness name the splat, so a pairing that moves shows.
        s.roughness = (0..n).map(|i| (i % 251) as f32 / 255.0).collect();
        s.lobes.coat_weight = (0..n).map(|i| (i % 3) as f32 * 0.5).collect();
        s.lobes.coat_roughness = (0..n).map(|i| (i % 251) as f32 / 255.0).collect();
        s.lobes.coat_darkening = vec![1.0; n];
        s.lobes.sheen_colour = (0..n).flat_map(|i| [0.1, 0.2, (i % 7) as f32 / 7.0]).collect();
        s
    }

    /// `streams.slang` writes `plainLobes`' value for an array the stage
    /// does not carry; lobes and marks only beside the material.
    #[test]
    fn packs_lobes_and_marks_from_the_streams() {
        let s = with_lobes(300);
        let p = pack_streams(&s, &BuildOptions::default()).unwrap();
        assert_eq!(p.block.lobes.len(), p.block.n * 3);
        // Splat 0 survives: transmission 1 (even), thin, not Schlick.
        let l = unpack_lobes([p.block.lobes[0], p.block.lobes[1], p.block.lobes[2]]);
        assert_eq!((l.specular_weight, l.specular_colour, l.specular_ior), (1.0, [1.0; 3], 1.5));
        assert_eq!((l.coat_weight, l.coat_roughness, l.coat_ior, l.coat_darkening), (0.0, 0.0, 1.5, 1.0));
        assert_eq!(p.block.pbr[0], 0x01ff_0000);
        // Splat 1: Schlick, coat 0.5 (127.5 + 0.5 -> 128), roughness 1/255.
        assert_eq!(p.block.pbr[1], 0x0200_0100);
        // Sheen (0.1, 0.2, 1/7): 26, 51, 36.
        assert_eq!(p.block.lobes[3..6], [0xffff_ffff, 0x40a0_0180, 0x0024_331a]);
        let none = pack_streams(&s, &BuildOptions { material: false, ..Default::default() }).unwrap();
        assert!(none.block.lobes.is_empty() && none.block.pbr.is_empty());
        let mut bare = s.clone();
        bare.lobes = LobeStreams::default();
        assert!(pack_streams(&bare, &BuildOptions::default()).unwrap().block.lobes.is_empty());
        // A shorter array is no array (`whole`): the rest still are.
        let mut short = s.clone();
        short.lobes.coat_weight.pop();
        let q = pack_streams(&short, &BuildOptions::default()).unwrap();
        assert_eq!(unpack_lobes([q.block.lobes[3], q.block.lobes[4], q.block.lobes[5]]).coat_weight, 0.0);
    }

    /// Every (pbr, lobes) pair a cloud holds is one of `source`'s.
    fn pairs_kept(b: &AthcBlock, source: &AthcBlock) {
        let set: std::collections::HashSet<(u32, [u32; 3])> = (0..source.n)
            .map(|i| (source.pbr[i], [source.lobes[i * 3], source.lobes[i * 3 + 1], source.lobes[i * 3 + 2]]))
            .collect();
        assert_eq!(b.lobes.len(), b.n * 3);
        assert_eq!(b.pbr.len(), b.n);
        for i in 0..b.n {
            let pair = (b.pbr[i], [b.lobes[i * 3], b.lobes[i * 3 + 1], b.lobes[i * 3 + 2]]);
            assert!(set.contains(&pair), "splat {i}: {pair:x?} is no source splat's material");
        }
    }

    /// The layers go with the material through the levels (`Lod.cpp`: a
    /// group takes its first gaussian's pbr and lobes alike), the files
    /// (v2, v3) and the web reductions.
    #[test]
    fn carries_lobes_with_the_material_everywhere() {
        let s = with_lobes(5000);
        let o = BuildOptions { chunk_splats: 1024, ..Default::default() };
        let p = pack_streams(&s, &o).unwrap();
        let f = build_lod(&p, &o).unwrap();
        assert_eq!(f.extra.lobes_words, 3);
        assert!(f.header.flags & FLAG_MATERIAL != 0);
        for (_, b) in &f.levels {
            pairs_kept(b, &p.block);
        }
        // A level's group: its first splat's words, as pbr's.
        let splats = f.splats();
        pairs_kept(&splats, &p.block);
        let bytes = f.write().unwrap();
        let back = AthcFile::read(&bytes).unwrap();
        assert_eq!(back.extra.lobes_words, 3);
        assert_eq!(back.levels, f.levels);
        assert_eq!(back.chunks, f.chunks);
        let v3 = crate::athc_v3::write_v3(&back, crate::athc_v3::COMPRESSION_GZIP).unwrap();
        let again = crate::athc_v3::read_v3(&v3).unwrap();
        assert_eq!(again.write().unwrap(), bytes);

        let thin = reduce_thin(&p, 4.0).unwrap();
        pairs_kept(&thin.block, &p.block);
        let cells = reduce_cells(&p, cell_for_target(&p, 800), 1.0).unwrap();
        pairs_kept(&cells.block, &p.block);
        for r in [&thin, &cells] {
            let g = build_lod(r, &o).unwrap();
            assert_eq!(g.extra.lobes_words, 3);
            for (_, b) in &g.levels {
                pairs_kept(b, &p.block);
            }
        }
    }

    #[test]
    fn drops_backs_with_their_lobes() {
        // The shells of drops_the_hidden_back_of_a_shell, each face its own
        // coat: what stays keeps its own.
        let mut s = CloudStreams { coefficients: 1, linear: true, ..Default::default() };
        let mut face = |x0: f32, z: f32, up: bool, open: u32, coat: f32| {
            for a in 0..10 {
                for b in 0..10 {
                    s.positions.extend_from_slice(&[x0 + a as f32 * 0.002, b as f32 * 0.002, z]);
                    s.rotations.extend_from_slice(&[0.0, 0.0, 0.0, 1.0]);
                    s.scales.extend_from_slice(&[0.002, 0.002, 0.0002]);
                    s.opacities.push(1.0);
                    s.sh.extend_from_slice(&[0.5, 0.5, 0.5]);
                    s.normals.extend_from_slice(&[0.0, 0.0, if up { 1.0 } else { -1.0 }]);
                    s.transfer_direct.extend((0..16).map(|_| 0.1));
                    s.shadow_bits.extend((0..8).map(|w| if w < open { u32::MAX } else { 0 }));
                    s.metallic.push(1.0);
                    s.lobes.coat_weight.push(coat);
                    s.count += 1;
                }
            }
        };
        face(0.0, 0.003, true, 3, 1.0);
        face(0.0, 0.0, false, 0, 0.25);
        face(2.0, 0.0, true, 0, 0.5);
        let o = BuildOptions { transfer: TransferKeep::Count(16), ..Default::default() };
        let cloud = pack_streams(&s, &o).unwrap();
        let (kept, dropped) = drop_hidden_backs(&cloud, 0.008).unwrap();
        assert_eq!(dropped, 100);
        pairs_kept(&kept.block, &cloud.block);
        let coats: Vec<u32> = (0..kept.block.n).map(|i| kept.block.lobes[i * 3 + 1] & 0xff).collect();
        assert!(coats.iter().all(|&c| c == 255 || c == 128));
        assert!(build_lod(&kept, &o).unwrap().extra.lobes_words == 3);
    }

    #[test]
    fn appends_lobes_with_athenea_defaults() {
        let mut a = LobeStreams { coat_weight: vec![1.0, 1.0], ..Default::default() };
        let b = LobeStreams { sheen_colour: vec![0.5; 3], ..Default::default() };
        a.append(2, b, 1);
        assert_eq!(a.coat_weight, [1.0, 1.0, 0.0]);
        assert_eq!(a.sheen_colour, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]);
        assert!(a.specular_weight.is_empty());
        assert!(a.present(3));
        assert_eq!(a.at(3, 2).coat_weight, 0.0);
        assert_eq!(a.at(3, 2).sheen_colour, [0.5; 3]);
    }

    /// test/fixtures/athc/lobes_sphere.athc, for test/gpu/athcLobes.test.ts:
    /// a ball of 1200 splats of the Corvette's paint (a Schlick metal under
    /// a darkening clear coat), coated where x > 0 and bare where not, with
    /// a direct transfer (16) and every way out open. UPDATE_FIXTURES=1
    /// writes it; otherwise the committed one must be what this builds.
    #[test]
    fn writes_the_lobes_sphere_fixture() {
        let n = 1200;
        let mut s = CloudStreams { coefficients: 1, linear: true, count: n, ..Default::default() };
        let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
        let y = |k: usize, v: [f32; 3]| -> f32 {
            let [x, y, z] = v;
            match k {
                0 => 0.282_094_8,
                1 => -0.488_602_5 * y,
                2 => 0.488_602_5 * z,
                3 => -0.488_602_5 * x,
                4 => 1.092_548_4 * x * y,
                5 => -1.092_548_4 * y * z,
                6 => 0.315_391_57 * (2.0 * z * z - x * x - y * y),
                7 => -1.092_548_4 * x * z,
                8 => 0.546_274_2 * (x * x - y * y),
                _ => 0.0,
            }
        };
        let band = [1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25];
        for i in 0..n {
            let yy = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - yy * yy).sqrt();
            let t = golden * i as f32;
            let nv = [r * t.cos(), yy, r * t.sin()];
            s.positions.extend(nv.map(|c| 0.5 * c));
            s.normals.extend_from_slice(&nv);
            // The quaternion turning z onto the normal (xyzw).
            let (ax, ay, az) = (-nv[1], nv[0], 0.0f32);
            let w = 1.0 + nv[2];
            let q = if w < 1e-6 { [1.0, 0.0, 0.0, 0.0] } else { normalize4([ax, ay, az, w]) };
            s.rotations.extend_from_slice(&q);
            s.scales.extend_from_slice(&[0.04, 0.04, 0.004]);
            s.opacities.push(0.9);
            s.sh.extend([0.05f32, 0.06, 0.05].map(|c| (c - 0.5) / SH0));
            s.metallic.push(1.0);
            s.roughness.push(0.342_130_4);
            s.transmission.push(0.0);
            s.schlick_metal.push(1);
            let l = paint_lobes(if nv[0] > 0.0 { 1.0 } else { 0.0 });
            s.lobes.specular_weight.push(l.specular_weight);
            s.lobes.specular_colour.extend_from_slice(&l.specular_colour);
            s.lobes.specular_ior.push(l.specular_ior);
            s.lobes.coat_weight.push(l.coat_weight);
            s.lobes.coat_roughness.push(l.coat_roughness);
            s.lobes.coat_ior.push(l.coat_ior);
            s.lobes.sheen_colour.extend_from_slice(&l.sheen_colour);
            s.lobes.sheen_roughness.push(l.sheen_roughness);
            s.lobes.coat_darkening.push(l.coat_darkening);
            s.transfer_direct.extend((0..16).map(|k| if k < 9 { 0.9 * band[k] * y(k, nv) } else { 0.0 }));
            s.shadow_bits.extend([u32::MAX; 8]);
        }
        let o = BuildOptions { transfer: TransferKeep::Count(16), curvature: false, ..Default::default() };
        let p = pack_streams(&s, &o).unwrap();
        assert_eq!(p.block.n, n);
        let f = build_lod(&p, &o).unwrap();
        assert_eq!(f.extra.lobes_words, 3);
        let bytes = crate::athc_v3::write_v3(&f, crate::athc_v3::COMPRESSION_GZIP).unwrap();
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test/fixtures/athc/lobes_sphere.athc");
        if std::env::var("UPDATE_FIXTURES").is_ok() {
            std::fs::write(path, &bytes).unwrap();
        }
        assert_eq!(std::fs::read(path).unwrap(), bytes, "UPDATE_FIXTURES=1 rewrites {path}");
    }

    #[test]
    fn carries_a_skin_through_the_levels_and_v3() {
        use crate::athc_skin::{element_skin, AthcSkeleton, SkinClip};
        // A strip of splats, the first half on joint 0, the second on 1,
        // blended in the middle; gradients on every splat.
        let n = 64;
        let mut s = CloudStreams {
            count: n,
            skin_influences: 2,
            joint_count: 2,
            ..Default::default()
        };
        for i in 0..n {
            s.positions.extend_from_slice(&[i as f32 * 0.1, 0.0, 0.0]);
            s.rotations.extend_from_slice(&[0.0, 0.0, 0.0, 1.0]);
            s.scales.extend_from_slice(&[0.05, 0.05, 0.005]);
            s.opacities.push(1.0);
            let w = (i as f32 / (n - 1) as f32).clamp(0.0, 1.0);
            s.joint_indices.extend_from_slice(&[0, 1]);
            s.joint_weights.extend_from_slice(&[1.0 - w, w]);
            s.weight_gradients.extend_from_slice(&[
                half::f16::from_f32(-0.5).to_bits(),
                half::f16::from_f32(0.25).to_bits(),
            ]);
        }
        let packed = pack_streams(&s, &BuildOptions::default()).unwrap();
        assert_eq!((packed.skin_influences, packed.skin_gradient_words), (2, 1));
        let file = build_lod(&packed, &BuildOptions::default()).unwrap();
        assert!(file.levels.iter().all(|(_, b)| b.skin.len() == b.n * 3));
        let skeleton = AthcSkeleton {
            influences: 2,
            gradient_words: 1,
            joints: vec!["a".into(), "a/b".into()],
            skeleton: "/Skel".into(),
            geom_bind: std::array::from_fn(|k| if k % 5 == 0 { 1.0 } else { 0.0 }),
            clips: vec![SkinClip {
                name: "c".into(),
                time_codes_per_second: 24.0,
                times: vec![0.0],
                xforms: vec![0.0; 32],
            }],
        };
        let v3 = crate::athc_v3::write_v3_skinned(
            &file,
            crate::athc_v3::COMPRESSION_GZIP,
            Some(&skeleton),
        )
        .unwrap();
        let back = crate::athc_v3::read_v3(&v3).unwrap();
        assert_eq!(back.extra.skin_influences, 2);
        assert_eq!(back.splats().skin, file.splats().skin);
        assert_eq!(
            crate::athc_v3::read_v3_skeleton(&v3).unwrap().unwrap(),
            skeleton
        );
        // Every encoding, and the size the tables give, cover the skeleton.
        use crate::athc_v3::{
            file_bytes, parse_v3, read_v3, read_v3_skeleton, tables_bytes, write_v3_full,
            write_v3_smallest_with, COMPRESSION_GZIP, ENCODING_BYTE_PLANES, ENCODING_DELTA_PLANES,
        };
        for planes in [ENCODING_BYTE_PLANES, ENCODING_DELTA_PLANES] {
            let enc = write_v3_full(&file, COMPRESSION_GZIP, false, &|_| planes, Some(&skeleton)).unwrap();
            assert_eq!(read_v3(&enc).unwrap().splats().skin, file.splats().skin);
            assert_eq!(read_v3_skeleton(&enc).unwrap().unwrap(), skeleton);
        }
        let (smallest, _) = write_v3_smallest_with(&file, COMPRESSION_GZIP, Some(&skeleton)).unwrap();
        assert_eq!(read_v3_skeleton(&smallest).unwrap().unwrap(), skeleton);
        assert_eq!(read_v3(&smallest).unwrap().splats().skin, file.splats().skin);
        let tables = tables_bytes(&v3).unwrap() as usize;
        let end = file_bytes(&v3[..tables]).unwrap();
        let at = parse_v3(&v3).unwrap().skeleton_offset;
        assert!(at > 0 && end >= at + skeleton.to_bytes().len() as u64);
        assert_eq!(crate::athc::aligned(end), v3.len() as u64);
        // A splat's influences and gradient as packed; the weights sum to one.
        let (pairs, grads) = element_skin(&back.splats().skin, 2, 1, 0);
        assert!((pairs[0].1 + pairs[1].1 - 1.0).abs() < 1e-6);
        assert_eq!(grads[0], [-0.5, 0.25]);
        // A v2 file drops the skin.
        assert_eq!(
            AthcFile::read(&file.write().unwrap())
                .unwrap()
                .extra
                .skin_influences,
            0
        );
        // A merged root: both joints, no gradient.
        let (pairs, grads) = element_skin(&back.levels[0].1.skin, 2, 1, 0);
        assert!(pairs.iter().all(|p| p.1 > 0.2));
        assert_eq!(grads[0], [0.0, 0.0]);
    }

    #[test]
    fn keeps_a_zonal_transfer_whole() {
        let n = 4;
        let mut s = CloudStreams {
            count: n,
            ..Default::default()
        };
        for i in 0..n {
            s.positions.extend_from_slice(&[i as f32, 0.0, 0.0]);
            s.opacities.push(1.0);
            s.transfer_zonal.extend((0..10).map(|k| k as f32 * 0.125));
        }
        let packed = pack_streams(&s, &BuildOptions::default()).unwrap();
        assert_eq!(packed.transfer_count, 10);
        assert_eq!(packed.block.transfer.len(), n * 5);
        assert_eq!(f16_of(packed.block.transfer[4] >> 16), 1.125);
        let o = BuildOptions {
            transfer: TransferKeep::Count(16),
            ..Default::default()
        };
        assert!(pack_streams(&s, &o).is_err());
    }

    /// A plane of surfels (athenea's 2DGS: the third axis exactly 0, the
    /// normal along it). Their area is that of their two axes: weighed as
    /// s0 s1 s2 / min(s) every surfel weighed 0, and the merged levels sat
    /// at the origin with opacity 0. Now every level's nodes have weight,
    /// sit at their splats' mean, are surfels themselves and, as decoded,
    /// keep the sheet shut.
    #[test]
    fn a_plane_of_surfels_merges_into_surfels_that_cover_it() {
        use crate::athc::{high_half, low_half};
        let mut s = CloudStreams { coefficients: 1, linear: true, ..Default::default() };
        let (side, step) = (0.2f32, 0.002f32);
        let k = (side / step) as usize;
        for a in 0..k {
            for b in 0..k {
                let jitter = ((a * 7 + b * 13) % 5) as f32 * 0.0001;
                s.positions.extend_from_slice(&[(a as f32 + 0.5) * step + jitter, (b as f32 + 0.5) * step, 0.05]);
                // Turned about the normal, as a bake leaves them.
                let q = glam::Quat::from_rotation_z((a * 3 + b) as f32 * 0.37);
                s.rotations.extend_from_slice(&q.to_array());
                s.scales.extend_from_slice(&[0.0024, 0.0016, 0.0]);
                s.opacities.push(0.95);
                s.sh.extend_from_slice(&[0.5, 0.5, 0.5]);
                s.normals.extend_from_slice(&[0.0, 0.0, 1.0]);
                s.count += 1;
            }
        }
        let o = BuildOptions::default();
        let p = pack_streams(&s, &o).unwrap();
        assert_eq!(p.block.n, k * k);
        let file = build_lod(&p, &o).unwrap();
        let splats = file.splats();
        let extent = file.header.extent;
        let mut uncapped = file.clone();
        crate::athc::uncap_levels(&mut uncapped);
        for (l, (level, block)) in uncapped.levels.iter().enumerate() {
            let edge = extent / (1u64 << *level) as f32;
            for g in 0..block.n {
                let pos = &block.positions[g * 4..g * 4 + 4];
                assert!(pos[3] > 0.0, "level {level} group {g}: no weight");
                // At its splats' mean (they all weigh the same).
                let [lo, hi] = crate::athc::group_splats(&file, l, g);
                let mut mean = [0.0f64; 3];
                for i in lo..hi {
                    for c in 0..3 {
                        mean[c] += splats.positions[i as usize * 4 + c] as f64;
                    }
                }
                for c in 0..3 {
                    let m = (mean[c] / (hi - lo) as f64) as f32;
                    assert!((pos[c] - m).abs() <= 1e-4 * edge.max(step), "level {level} group {g}: {pos:?} vs {mean:?}");
                }
                // A surfel, its flat axis the normal.
                let sh = &block.shape[g * 4..g * 4 + 4];
                let ln = [low_half(sh[1]), high_half(sh[1]), low_half(sh[2])];
                assert_eq!(ln.iter().filter(|&&v| v.exp() == 0.0).count(), 1, "level {level} group {g}: {ln:?}");
                let q = crate::athc::decode_quaternion(sh[0]);
                let r = axes_of_quaternion(q);
                let flat = (0..3).find(|&a| ln[a].exp() == 0.0).unwrap();
                assert!(r[2][flat].abs() > 0.999, "level {level} group {g}: flat axis off the normal");
            }
            // As decoded: widened, the whole coverage as LoD opacity.
            let mut b = block.clone();
            crate::athc::widen_merged(&mut b, edge);
            assert!((0..b.n).all(|i| {
                let sh = &b.shape[i * 4..i * 4 + 4];
                low_half(sh[2]).exp() == 0.0 && low_half(sh[2]).is_finite()
            }), "level {level}: decoded with the flat axis third");
            let margin = 1.5 * edge;
            if side - 2.0 * margin < 4.0 * edge {
                continue;
            }
            let (t, sub) = sheet_transmittance(&b, side, margin);
            eprintln!("level {level} (cell {:.1} mm, {} surfels): through {t:.4} (sub-pixel {sub:.4})", edge * 1e3, b.n);
            assert!(t <= 0.01 && sub <= 0.05, "level {level}: {t} {sub}");
        }
    }

    /// A ball of surfels smaller than a cell is no disc: the levels whose
    /// cells hold all of it keep it a 3D gaussian.
    #[test]
    fn a_small_ball_of_surfels_stays_a_gaussian() {
        use crate::athc::{high_half, low_half};
        let mut s = CloudStreams { coefficients: 1, linear: true, ..Default::default() };
        let (r, n) = (0.002f32, 2000);
        let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
        for i in 0..n {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let rr = (1.0 - y * y).sqrt();
            let t = golden * i as f32;
            let nv = glam::Vec3::new(rr * t.cos(), y, rr * t.sin());
            s.positions.extend_from_slice(&(nv * r + glam::Vec3::splat(0.5)).to_array());
            s.rotations.extend_from_slice(&glam::Quat::from_rotation_arc(glam::Vec3::Z, nv).to_array());
            s.scales.extend_from_slice(&[0.0002, 0.0002, 0.0]);
            s.opacities.push(0.9);
            s.sh.extend_from_slice(&[0.5, 0.5, 0.5]);
            s.normals.extend_from_slice(&nv.to_array());
            s.count += 1;
        }
        // A far corner, so the octree's cells are large around the ball.
        s.positions.extend_from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        for _ in 0..2 {
            s.rotations.extend_from_slice(&[0.0, 0.0, 0.0, 1.0]);
            s.scales.extend_from_slice(&[0.001, 0.001, 0.0]);
            s.opacities.push(0.9);
            s.sh.extend_from_slice(&[0.5, 0.5, 0.5]);
            s.normals.extend_from_slice(&[0.0, 0.0, 1.0]);
            s.count += 1;
        }
        let o = BuildOptions::default();
        let file = build_lod(&pack_streams(&s, &o).unwrap(), &o).unwrap();
        let mut checked = 0;
        for (level, block) in &file.levels {
            let edge = file.header.extent / (1u64 << *level) as f32;
            if edge < 4.0 * r {
                continue;
            }
            for g in 0..block.n {
                let p = &block.positions[g * 4..g * 4 + 3];
                if (p[0] - 0.5).abs() > 0.01 {
                    continue; // the corners
                }
                let sh = &block.shape[g * 4..g * 4 + 4];
                let ln = [low_half(sh[1]), high_half(sh[1]), low_half(sh[2])];
                assert!(ln.iter().all(|v| v.exp() > 0.0), "level {level}: the ball became a disc {ln:?}");
                checked += 1;
            }
        }
        assert!(checked > 0);
    }

    fn zonal_words(lobes: [([f32; 3], [f32; 3]); 2], frame: &M3) -> Vec<u32> {
        let mut v = Vec::new();
        for (axis, z) in lobes {
            let sq = oct_encode(to_local(frame, axis));
            v.extend_from_slice(&[sq[0], sq[1], z[0], z[1], z[2]]);
        }
        v.chunks(2).map(|p| pack_halves(p[0], p[1])).collect()
    }

    fn zonal_world(words: &[u32], frame: &M3) -> [([f32; 3], [f32; 3]); 2] {
        [0, 1].map(|lobe| {
            let b = lobe * 5;
            let axis = to_world(frame, oct_decode([half_at(words, b), half_at(words, b + 1)]));
            (axis, [half_at(words, b + 2), half_at(words, b + 3), half_at(words, b + 4)])
        })
    }

    /// Two splats, the second turned: what lies in each one's frame (a
    /// zonal transfer's lobes, the curvature) merged in the world, read in
    /// the merged element's frame. Averaging the stored halves (what
    /// extras_merge did) mixed the two frames.
    #[test]
    fn merges_zonal_lobes_and_curvature_in_the_world() {
        let turns = [
            glam::Quat::from_rotation_z(0.3),
            glam::Quat::from_euler(glam::EulerRot::XYZ, 0.9, -0.4, 2.1),
        ];
        let node_turn = glam::Quat::from_euler(glam::EulerRot::XYZ, -0.2, 1.1, 0.5);
        let shape = |q: glam::Quat| -> [u32; 4] {
            [encode_quaternion(q.normalize().to_array()), pack_halves(0.01f32.ln(), 0.008f32.ln()), pack_halves(0.001f32.ln(), 0.5), pack_halves(0.5, 0.5)]
        };
        let mut splats = AthcBlock { n: 2, ..Default::default() };
        for (i, q) in turns.iter().enumerate() {
            splats.positions.extend_from_slice(&[i as f32 * 0.01, 0.0, 0.0, if i == 0 { 0.9 } else { 0.3 }]);
            splats.shape.extend_from_slice(&shape(*q));
        }
        let mut frames = AthcBlock { n: 1, ..Default::default() };
        frames.positions.extend_from_slice(&[0.005, 0.0, 0.0, 0.9]);
        frames.shape.extend_from_slice(&shape(node_turn));
        let frame = |b: &AthcBlock, i: usize| frame_of(b, i);
        let mass: Vec<f32> = (0..2).map(|i| splats.positions[i * 4 + 3] * 0.01 * 0.008).collect();
        let wsum = mass[0] + mass[1];

        // The same world lobes on both: the merge gives them back.
        let d0 = normalize3([0.3, -0.5, 0.8]);
        let d1 = normalize3([-0.7, 0.1, 0.2]);
        let same = [(d0, [0.8, 0.4, 0.1]), (d1, [0.2, -0.1, 0.05])];
        let mut source = Vec::new();
        for i in 0..2 {
            source.extend(zonal_words(same, &frame(&splats, i)));
        }
        let merged = frame_merge(&source, 5, true, &[0], &splats, &frames);
        let back = zonal_world(&merged, &frame(&frames, 0));
        for (lobe, (axis, z)) in back.iter().enumerate() {
            let dot: f32 = axis.iter().zip(&same[lobe].0).map(|(a, b)| a * b).sum();
            assert!(dot > 0.9995, "lobe {lobe}: {axis:?} vs {:?}", same[lobe].0);
            for k in 0..3 {
                assert!((z[k] - same[lobe].1[k]).abs() < 2e-3, "lobe {lobe} z{k}");
            }
        }
        // As the old merge did it (the halves averaged): another direction.
        let naive = extras_merge(&source, 5, 1, &[0], &splats);
        let naive_axis = zonal_world(&naive, &frame(&frames, 0))[0].0;
        let naive_dot: f32 = naive_axis.iter().zip(&d0).map(|(a, b)| a * b).sum();
        assert!(naive_dot < 0.99, "the bug: {naive_dot}");

        // Different world lobes: the mass-weighted mean axis and coefficients.
        let other = [(normalize3([0.5, 0.5, 0.6]), [0.4, 0.2, 0.3]), (d1, [0.6, 0.1, -0.05])];
        let mut source = zonal_words(same, &frame(&splats, 0));
        source.extend(zonal_words(other, &frame(&splats, 1)));
        let merged = frame_merge(&source, 5, true, &[0], &splats, &frames);
        let back = zonal_world(&merged, &frame(&frames, 0));
        for lobe in 0..2 {
            let a = [0, 1, 2].map(|k| mass[0] * same[lobe].0[k] + mass[1] * other[lobe].0[k]);
            let expect = normalize3(a);
            let dot: f32 = back[lobe].0.iter().zip(&expect).map(|(a, b)| a * b).sum();
            assert!(dot > 0.9995, "lobe {lobe}");
            for k in 0..3 {
                let z = (mass[0] * same[lobe].1[k] + mass[1] * other[lobe].1[k]) / wsum;
                assert!((back[lobe].1[k] - z).abs() < 2e-3, "lobe {lobe} z{k}: {} vs {z}", back[lobe].1[k]);
            }
        }

        // The curvature: one world tensor (a cylinder of radius 1/12 along
        // x, on a surface whose normal is z), stored in each splat's frame.
        let world = |a: usize, b: usize| if a == 1 && b == 1 { 12.0f32 } else { 0.0 };
        let curv_words = |r: &M3| -> [u32; 2] {
            let s = |c: usize, d: usize| -> f32 {
                let mut sum = 0.0;
                for a in 0..3 {
                    for b in 0..3 {
                        sum += r[a][c] * world(a, b) * r[b][d];
                    }
                }
                sum
            };
            [pack_halves(s(0, 0), s(0, 1)), pack_halves(s(1, 1), 0.0)]
        };
        // Both splats in the surface (third axis z), turned about it.
        let mut flat = splats.clone();
        flat.shape.clear();
        for q in [glam::Quat::from_rotation_z(0.3), glam::Quat::from_rotation_z(-1.2)] {
            flat.shape.extend_from_slice(&shape(q));
        }
        let mut source = Vec::new();
        for i in 0..2 {
            source.extend(curv_words(&frame(&flat, i)));
        }
        let mut node = frames.clone();
        node.shape = shape(glam::Quat::from_rotation_z(0.8)).to_vec();
        let merged = frame_merge(&source, 2, false, &[0], &flat, &node);
        let expect = curv_words(&frame(&node, 0));
        for k in 0..3 {
            let (a, e) = (half_at(&merged, k), half_at(&expect, k));
            assert!((a - e).abs() < 0.02, "curvature {k}: {a} vs {e}");
        }
        let naive = extras_merge(&source, 2, 1, &[0], &flat);
        assert!((half_at(&naive, 1) - half_at(&expect, 1)).abs() > 0.5, "the bug");
    }

    /// athenea's uncapped coverage (`athenea:splat:coverage`, W / A of a
    /// merged gaussian) is the opacity where it is there, past 1 included.
    #[test]
    fn reads_the_uncapped_coverage_as_the_opacity() {
        let mut s = synthetic(4);
        s.opacities = vec![0.99, 0.5, 0.99, 0.2];
        s.coverage = vec![2.5, 0.5, f32::NAN, 0.0];
        let p = pack_streams(&s, &BuildOptions::default()).unwrap();
        let o: Vec<f32> = p.block.positions.chunks(4).map(|v| v[3]).collect();
        assert_eq!(o, vec![2.5, 0.5, 0.99, 0.2]);
    }
}
