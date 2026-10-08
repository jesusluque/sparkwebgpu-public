//! Making a built `.athc` lighter without changing what is seen (thread BB,
//! from thread AY's measurements in `research/simplify-measurements.md`):
//!
//! - [`hidden_mask`]: the splats no view of an orbit ever sees. Each view is
//!   drawn on the CPU with Spark's rules ([`crate::cpu_raster`]) over the
//!   whole scene (every part occludes every other), recording each splat's
//!   largest transmittance in front of its centre and its summed
//!   contribution; a splat is hidden when the first stays under
//!   `max_transmittance` and the second under `max_pixels` in every view.
//!   For a skinned cloud the views are repeated over poses of its clips
//!   ([`pose_splats`]) and only what every pose hides is hidden: what the
//!   bind pose hides shows again when the bird moves (AY § 2.4).
//! - [`decimate_opaque`]: an opaque surface baked ~5x oversampled (every point
//!   under five discs, AY § 2.2) thinned to one splat in `ratio` along Morton
//!   runs that stay on one panel (normals within 30 degrees, near, the same
//!   material up to roughness jitter), the kept splat's two long axes grown so the run stays
//!   covered, its colour, transfer, curvature, open directions and skin the
//!   run's merge (`athc_build`'s, as the LoD merges them).
//! - [`select`]: what is kept, every section following the splats.
//!
//! The levels are then built again over the kept splats (`build_lod`), and a
//! `.athl`'s layers follow the splats through [`crate::athl::SplatSources`].

use anyhow::{ensure, Result};

use crate::athc::{pack_halves, unpack_normal, AthcBlock, AthcFile, FLAG_LINEAR};
use crate::athc_build::{
    disc_area, extras_merge, f16_of, frame_merge, half_safe, merge_transfer, morton30, reorder, scales_of,
    PackedCloud, SplatRuns,
};
use crate::athc_skin::{element_skin, AthcSkeleton, SkinClip};
use crate::cpu_raster::{block_splats, par_chunks, render, AtomicF32s, Camera, Proxy, Splat, Track};

/// What counts as hidden. AY measured with 0.02 and 0.05 px; on the light
/// Corvette 0.02 px over 800 directions keeps every held-out view within
/// 3e-5 of relMSE (0.05 px: 6.5e-5 in the default view).
#[derive(Clone, Copy, Debug)]
pub struct HiddenRule {
    /// The largest transmittance in front of a splat's centre, over every view.
    pub max_transmittance: f32,
    /// The splat's whole contribution (px x alpha x T), summed over every view.
    pub max_pixels: f32,
}

impl Default for HiddenRule {
    fn default() -> Self {
        Self { max_transmittance: 0.02, max_pixels: 0.02 }
    }
}

/// Each splat's largest transmittance in front of its centre and its summed
/// contribution over `views`, added into `max_t` / `contrib` (max / sum).
pub fn accumulate_visibility(splats: &[Splat], views: &[Camera], max_t: &mut [f32], contrib: &mut [f32]) {
    let n = splats.len();
    assert!(max_t.len() == n && contrib.len() == n);
    let t = AtomicF32s::new(n);
    let c = AtomicF32s::new(n);
    for cam in views {
        render(splats, cam, Some(&Track { max_t: &t, contrib: &c }));
    }
    for (i, (v, w)) in t.into_vec().into_iter().zip(c.into_vec()).enumerate() {
        max_t[i] = max_t[i].max(v);
        contrib[i] += w;
    }
}

/// The hidden splats of `splats` over `views` (true: hidden).
pub fn hidden_mask(splats: &[Splat], views: &[Camera], rule: HiddenRule) -> Vec<bool> {
    let n = splats.len();
    let (mut t, mut c) = (vec![0f32; n], vec![0f32; n]);
    accumulate_visibility(splats, views, &mut t, &mut c);
    (0..n).map(|i| t[i] < rule.max_transmittance && c[i] < rule.max_pixels).collect()
}

/// A file's splats (its chunks, in file order) as raster splats.
pub fn file_splats(file: &AthcFile, proxy: Proxy) -> Vec<Splat> {
    let linear = file.header.has(FLAG_LINEAR);
    let mut out = Vec::with_capacity(file.header.count as usize);
    for c in &file.chunks {
        block_splats(c, linear, proxy, &mut out);
    }
    out
}

/// A skinned file's influences, a list a splat (file order).
pub fn file_influences(file: &AthcFile) -> Vec<Vec<(u32, f32)>> {
    let inf = file.extra.skin_influences as usize;
    let gw = file.extra.skin_gradient_words as usize;
    let mut out = Vec::with_capacity(file.header.count as usize);
    for c in &file.chunks {
        for i in 0..c.n {
            out.push(element_skin(&c.skin, inf, gw, i).0);
        }
    }
    out
}

/// A 4x4 USD matrix (rows, vectors on the left) as a column-vector 3x4.
fn affine(m: &[f32]) -> [[f32; 4]; 3] {
    let mut a = [[0.0f32; 4]; 3];
    for c in 0..3 {
        for r in 0..4 {
            a[c][r] = m[4 * r + c];
        }
    }
    a
}

fn apply(a: &[[f32; 4]; 3], p: [f32; 3], w: f32) -> [f32; 3] {
    [0, 1, 2].map(|c| a[c][0] * p[0] + a[c][1] * p[1] + a[c][2] * p[2] + a[c][3] * w)
}

/// Symmetric 3x3 (xx yy zz xy xz yz) eigenvalues and eigenvectors (columns), Jacobi.
fn eigen(s: &[f64; 6]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = [[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]];
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..30 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-30 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let sn = t * c;
            for k in 0..3 {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - sn * akq;
                a[k][q] = sn * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - sn * aqk;
                a[q][k] = sn * apk + c * aqk;
            }
            for k in 0..3 {
                let (vkp, vkq) = (v[k][p], v[k][q]);
                v[k][p] = c * vkp - sn * vkq;
                v[k][q] = sn * vkp + c * vkq;
            }
        }
    }
    let vals = [a[0][0], a[1][1], a[2][2]];
    let cols = [[v[0][0], v[1][0], v[2][0]], [v[0][1], v[1][1], v[2][1]], [v[0][2], v[1][2], v[2][2]]];
    (vals, cols)
}

/// A covariance as (quaternion x y z w, scales).
fn orient(cov: &[f64; 6]) -> ([f32; 4], [f32; 3]) {
    let (vals, mut c) = eigen(cov);
    let cr = [
        c[0][1] * c[1][2] - c[0][2] * c[1][1],
        c[0][2] * c[1][0] - c[0][0] * c[1][2],
        c[0][0] * c[1][1] - c[0][1] * c[1][0],
    ];
    if cr[0] * c[2][0] + cr[1] * c[2][1] + cr[2] * c[2][2] < 0.0 {
        c[2] = c[2].map(|v| -v);
    }
    let m = glam::DMat3::from_cols(c[0].into(), c[1].into(), c[2].into());
    let q = glam::DQuat::from_mat3(&m).normalize();
    ([q.x as f32, q.y as f32, q.z as f32, q.w as f32], vals.map(|v| v.max(1e-18).sqrt() as f32))
}

/// `splats` (bind pose, as stored) posed at `clip`'s sample `k` by linear
/// blend skinning of centres and covariances (athenea_adapter/skin.slang
/// without the Jacobian's gradient terms; AY's `posevis`).
pub fn pose_splats(
    splats: &[Splat],
    influences: &[Vec<(u32, f32)>],
    skeleton: &AthcSkeleton,
    clip: &SkinClip,
    k: usize,
) -> Vec<Splat> {
    let joints = skeleton.joint_count();
    let gb = affine(&skeleton.geom_bind);
    let per = joints * 16;
    let x = &clip.xforms[k * per..(k + 1) * per];
    let mats: Vec<[[f32; 4]; 3]> = (0..joints).map(|j| affine(&x[j * 16..j * 16 + 16])).collect();
    par_chunks(splats.len(), |r| {
        r.map(|i| {
            let s = &splats[i];
            let bound = apply(&gb, s.p, 1.0);
            let mut moved = [0.0f32; 3];
            let mut lin = [[0.0f32; 3]; 3];
            let mut total = 0.0;
            for &(j, w) in &influences[i] {
                if w <= 1e-6 || j as usize >= joints {
                    continue;
                }
                let m = &mats[j as usize];
                let c = apply(m, bound, 1.0);
                for d in 0..3 {
                    moved[d] += w * c[d];
                    for e in 0..3 {
                        lin[d][e] += w * m[d][e];
                    }
                }
                total += w;
            }
            let mut out = s.clone();
            if total <= 1e-6 {
                out.p = bound;
                return out;
            }
            out.p = moved;
            let axes = s.axes();
            let mut cov = [0.0f64; 6];
            for k in 0..3 {
                let a = apply(&gb, axes[k], 0.0);
                let v = [0, 1, 2].map(|d| ((lin[d][0] * a[0] + lin[d][1] * a[1] + lin[d][2] * a[2]) * s.s[k]) as f64);
                cov[0] += v[0] * v[0];
                cov[1] += v[1] * v[1];
                cov[2] += v[2] * v[2];
                cov[3] += v[0] * v[1];
                cov[4] += v[0] * v[2];
                cov[5] += v[1] * v[2];
            }
            let (q, sc) = orient(&cov);
            out.q = q;
            out.s = sc;
            out
        })
        .collect::<Vec<Splat>>()
    })
    .concat()
}

/// The sample indices of a clip of `m` samples at which visibility is
/// taken (`samples` evenly spread, first included), and one between them
/// for a held-out check.
pub fn clip_samples(m: usize, samples: usize) -> (Vec<usize>, usize) {
    let s = samples.max(1).min(m.max(1));
    let picks: Vec<usize> = (0..s).map(|j| j * m / s).collect();
    let held = (m / (2 * s)).min(m.saturating_sub(1));
    let held = if picks.contains(&held) { (held + 1).min(m - 1) } else { held };
    (picks, held)
}

/// The splats `keep` of a packed cloud (indices, any order), every section
/// following them.
pub fn select(cloud: &PackedCloud, keep: &[u32]) -> PackedCloud {
    let mut block = reorder(&cloud.block, keep);
    block.n = keep.len();
    PackedCloud { block, ..cloud.clone() }
}

/// How a part is decimated.
#[derive(Clone, Copy, Debug)]
pub struct Decimate {
    /// About one splat kept in `ratio` (a full run's length).
    pub ratio: usize,
    /// A full run's long axes grow by this (AY: ~0.75 sqrt(ratio), 1.3 at 3);
    /// a shorter run of length l by grow sqrt(l / ratio), at least 1.
    pub grow: f32,
    /// Only splats at least this opaque are thinned (AY's 0.99).
    pub min_opacity: f32,
    /// A run's members face within this of its first (cosine).
    pub min_cos: f32,
    /// A run's members lie within this many of its first's long sigma.
    pub reach: f32,
}

impl Decimate {
    pub fn new(ratio: usize) -> Self {
        Self {
            ratio: ratio.max(1),
            grow: (0.75 * (ratio.max(1) as f32).sqrt()).max(1.0),
            min_opacity: 0.99,
            min_cos: 30f32.to_radians().cos(),
            reach: 3.0,
        }
    }
}

/// What [`decimate_opaque`] did.
#[derive(Clone, Debug, Default)]
pub struct DecimateReport {
    pub candidates: usize,
    pub runs: usize,
    pub full_runs: usize,
    pub output: usize,
}

/// The opaque splats of `cloud` for which `pick` holds thinned to about one
/// in `d.ratio` (module docs); returns the cloud and its runs (every output
/// splat a run, [`SplatRuns`]: members, starts, weights by opacity x area).
pub fn decimate_opaque(
    cloud: &PackedCloud,
    d: &Decimate,
    pick: &dyn Fn(&AthcBlock, usize) -> bool,
) -> Result<(PackedCloud, SplatRuns, DecimateReport)> {
    let src = &cloud.block;
    let n = src.n;
    ensure!(src.emission.is_empty(), "emission is not decimated here yet");
    let pbr_per = src.pbr.len().checked_div(n).unwrap_or(0);
    let lo = cloud.bounds_min;
    let mut extent = 0.0f32;
    for k in 0..3 {
        extent = extent.max(cloud.bounds_max[k] - cloud.bounds_min[k]);
    }
    let e = extent.max(1e-6) * 1.0001;
    let pos = |i: usize| [src.positions[i * 4], src.positions[i * 4 + 1], src.positions[i * 4 + 2]];
    let opacity = |i: usize| src.positions[i * 4 + 3];
    let candidate = |i: usize| {
        let o = opacity(i);
        o >= d.min_opacity && o <= 1.0 + 1e-6 && pick(src, i)
    };
    let mut cands: Vec<u32> = (0..n as u32).filter(|&i| candidate(i as usize)).collect();
    let codes: Vec<u32> = cands
        .iter()
        .map(|&i| {
            let p = pos(i as usize);
            morton30([(p[0] - lo[0]) / e, (p[1] - lo[1]) / e, (p[2] - lo[2]) / e])
        })
        .collect();
    let mut by_code: Vec<usize> = (0..cands.len()).collect();
    by_code.sort_by_key(|&k| codes[k]);
    cands = by_code.iter().map(|&k| cands[k]).collect();
    let normal = |i: usize| if src.normals.is_empty() { None } else { Some(unpack_normal(src.normals[i])) };
    let long = |i: usize| {
        let s = scales_of(&src.shape[i * 4..i * 4 + 4]);
        s[0].max(s[1]).max(s[2])
    };
    // The same material: the same transmission and flags (bits 16..), metallic
    // and roughness within 0.1 (a bake's roughness jitters a step or two).
    let material = |i: usize| if pbr_per > 0 { src.pbr[i * pbr_per] } else { 0 };
    let same_material = |a: u32, b: u32| {
        a >> 16 == b >> 16 && (a & 255).abs_diff(b & 255) <= 26 && ((a >> 8) & 255).abs_diff((b >> 8) & 255) <= 26
    };
    // Runs along the Morton order, each closed early where the next splat
    // leaves the first's panel.
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut starts: Vec<u32> = Vec::with_capacity(n);
    let mut report = DecimateReport { candidates: cands.len(), ..Default::default() };
    let mut k = 0;
    while k < cands.len() {
        let a = cands[k] as usize;
        let (pa, na, ra, ma) = (pos(a), normal(a), d.reach * long(a), material(a));
        starts.push(order.len() as u32);
        order.push(a as u32);
        let mut len = 1;
        k += 1;
        while len < d.ratio && k < cands.len() {
            let b = cands[k] as usize;
            let pb = pos(b);
            let dd = (0..3).map(|j| (pb[j] - pa[j]).powi(2)).sum::<f32>().sqrt();
            let facing = match (na, normal(b)) {
                (Some(x), Some(y)) => x[0] * y[0] + x[1] * y[1] + x[2] * y[2] >= d.min_cos,
                _ => true,
            };
            if dd > ra || !facing || !same_material(material(b), ma) {
                break;
            }
            order.push(b as u32);
            len += 1;
            k += 1;
        }
        if len == d.ratio {
            report.full_runs += 1;
        }
        report.runs += 1;
    }
    // Everything else stays, a run of one each.
    let mut is_cand = vec![false; n];
    for &i in &cands {
        is_cand[i as usize] = true;
    }
    for i in 0..n {
        if !is_cand[i] {
            starts.push(order.len() as u32);
            order.push(i as u32);
        }
    }
    let runs = starts.len();
    let end = |j: usize| if j + 1 < runs { starts[j + 1] as usize } else { n };
    let splats = reorder(src, &order);
    // Each run's most typical splat: its normal nearest the run's mean.
    let reps: Vec<u32> = (0..runs)
        .map(|j| {
            let (first, last) = (starts[j] as usize, end(j));
            if last - first == 1 || splats.normals.is_empty() {
                return first as u32;
            }
            let nrm = |i: usize| unpack_normal(splats.normals[i]);
            let mut mean = [0.0f32; 3];
            for i in first..last {
                let v = nrm(i);
                for c in 0..3 {
                    mean[c] += v[c];
                }
            }
            (first..last)
                .max_by(|&x, &y| {
                    let dot = |i: usize| {
                        let v = nrm(i);
                        v[0] * mean[0] + v[1] * mean[1] + v[2] * mean[2]
                    };
                    dot(x).total_cmp(&dot(y))
                })
                .unwrap() as u32
        })
        .collect();
    let mass = |b: &AthcBlock, i: usize| b.positions[i * 4 + 3].max(0.0) * disc_area(scales_of(&b.shape[i * 4..i * 4 + 4]));
    let mut block = reorder(&splats, &reps);
    block.n = runs;
    let per_n = |v: &[u32]| v.len() / n;
    let merged_transfer = (!splats.transfer.is_empty()).then(|| merge_transfer(cloud.transfer_count, &splats, &starts, &block));
    let merged_shadow =
        (!splats.shadow_bits.is_empty()).then(|| extras_merge(&splats.shadow_bits, per_n(&splats.shadow_bits), 2, &starts, &splats));
    let merged_curv = (!splats.curvature.is_empty()).then(|| frame_merge(&splats.curvature, 2, false, &starts, &splats, &block));
    let merged_skin = (!splats.skin.is_empty()).then(|| {
        crate::athc_skin::skin_merge(
            &splats.skin,
            cloud.skin_influences as usize,
            cloud.skin_gradient_words as usize,
            &starts,
            &splats,
        )
    });
    let full_grow = d.grow / (d.ratio as f32).sqrt();
    for j in 0..runs {
        let (first, last) = (starts[j] as usize, end(j));
        if last - first < 2 {
            continue; // a run of one keeps its own words exactly
        }
        let (mut w, mut c) = (0.0f32, [0.0f32; 3]);
        for i in first..last {
            let sh = &splats.shape[i * 4..i * 4 + 4];
            let wt = mass(&splats, i);
            w += wt;
            c[0] += wt * f16_of(sh[2] >> 16);
            c[1] += wt * f16_of(sh[3] & 0xffff);
            c[2] += wt * f16_of(sh[3] >> 16);
        }
        let sh = &mut block.shape[j * 4..j * 4 + 4];
        let ls = [f16_of(sh[1] & 0xffff), f16_of(sh[1] >> 16), f16_of(sh[2] & 0xffff)];
        let thin = (0..3).min_by(|&a, &b| ls[a].total_cmp(&ls[b])).unwrap();
        let grow = (full_grow * ((last - first) as f32).sqrt()).max(1.0).ln();
        let g = [0, 1, 2].map(|a| if a == thin { ls[a] } else { ls[a] + grow });
        let base = if w > 0.0 { c.map(|v| v / w) } else { [f16_of(sh[2] >> 16), f16_of(sh[3] & 0xffff), f16_of(sh[3] >> 16)] };
        sh[1] = pack_halves(half_safe(g[0]), half_safe(g[1]));
        sh[2] = pack_halves(half_safe(g[2]), half_safe(base[0]));
        sh[3] = pack_halves(half_safe(base[1]), half_safe(base[2]));
        let copy = |dst: &mut Vec<u32>, merged: &Option<Vec<u32>>| {
            if let Some(m) = merged {
                let per = m.len() / runs;
                dst[j * per..(j + 1) * per].copy_from_slice(&m[j * per..(j + 1) * per]);
            }
        };
        copy(&mut block.transfer, &merged_transfer);
        copy(&mut block.shadow_bits, &merged_shadow);
        copy(&mut block.curvature, &merged_curv);
        copy(&mut block.skin, &merged_skin);
    }
    report.output = runs;
    let weights: Vec<f32> = order.iter().map(|&i| mass(src, i as usize)).collect();
    let out = PackedCloud { block, ..cloud.clone() };
    Ok((out, SplatRuns { members: order, starts, weights }, report))
}

/// A splat's material flags from its pbr word, for `--decimate` filters:
/// (transmissive, thin wall, Schlick metal).
pub fn material_flags(b: &AthcBlock, i: usize) -> (bool, bool, bool) {
    let per = b.pbr.len().checked_div(b.n).unwrap_or(0);
    if per == 0 {
        return (false, false, false);
    }
    let w = b.pbr[i * per];
    (((w >> 16) & 255) > 127, (w >> 24) & 1 == 1, (w >> 25) & 1 == 1)
}

/// A splat's coat weight (its lobes' second word's low byte, `pack_lobes`), or 0.
pub fn coat_weight(b: &AthcBlock, i: usize) -> f32 {
    if b.lobes.is_empty() {
        return 0.0;
    }
    crate::athc_build::unpack_lobes([b.lobes[i * 3], b.lobes[i * 3 + 1], b.lobes[i * 3 + 2]]).coat_weight
}

// --- a scene: several files that hide each other ------------------------------

/// Which splats of a part `--decimate` thins (besides being opaque).
#[derive(Clone, Debug)]
pub enum DecimateFilter {
    All,
    /// Only splats with a coat.
    Coat,
    /// Only splats lying on another cloud's surface ([`NearSet`]): a part
    /// of a file that holds many materials (the lamps set's `car`) picked by
    /// the part it was in another bake (the detailed set's `paint`).
    Near(std::sync::Arc<NearSet>),
}

impl DecimateFilter {
    pub fn picks(&self, b: &AthcBlock, i: usize) -> bool {
        match self {
            DecimateFilter::All => true,
            DecimateFilter::Coat => coat_weight(b, i) > 0.0,
            DecimateFilter::Near(set) => {
                let p = [b.positions[i * 4], b.positions[i * 4 + 1], b.positions[i * 4 + 2]];
                let n = if b.normals.is_empty() { None } else { Some(unpack_normal(b.normals[i])) };
                set.near(p, n)
            }
        }
    }
}

/// A cloud's splat centres (and normals) on a grid: is a point on its
/// surface (a centre within `reach`, facing within 45 degrees)?
pub struct NearSet {
    reach: f32,
    cells: std::collections::HashMap<[i32; 3], Vec<([f32; 3], Option<[f32; 3]>)>>,
}

impl std::fmt::Debug for NearSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NearSet({} cells, reach {})", self.cells.len(), self.reach)
    }
}

impl NearSet {
    pub fn new(file: &AthcFile, reach: f32) -> Self {
        let s = file.splats();
        let mut cells: std::collections::HashMap<[i32; 3], Vec<_>> = std::collections::HashMap::new();
        for i in 0..s.n {
            let p = [s.positions[i * 4], s.positions[i * 4 + 1], s.positions[i * 4 + 2]];
            let n = if s.normals.is_empty() { None } else { Some(unpack_normal(s.normals[i])) };
            cells.entry(p.map(|v| (v / reach).floor() as i32)).or_default().push((p, n));
        }
        Self { reach, cells }
    }

    pub fn near(&self, p: [f32; 3], n: Option<[f32; 3]>) -> bool {
        let c = p.map(|v| (v / self.reach).floor() as i32);
        let r2 = self.reach * self.reach;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(list) = self.cells.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else { continue };
                    for (q, m) in list {
                        let d2 = (0..3).map(|k| (q[k] - p[k]).powi(2)).sum::<f32>();
                        let facing = match (n, m) {
                            (Some(a), Some(b)) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] > 0.7,
                            _ => true,
                        };
                        if d2 <= r2 && facing {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

/// One file of a scene.
pub struct SceneInput {
    pub name: String,
    pub file: AthcFile,
    /// Written out (pruned and/or decimated); false: it only occludes.
    pub output: bool,
    /// Its hidden splats are dropped (false: kept whole, still occluding).
    pub prune: bool,
    pub decimate: Option<(Decimate, DecimateFilter)>,
}

/// How a scene is pruned.
#[derive(Clone, Debug)]
pub struct PruneOptions {
    pub camera: CamSpec,
    /// Directions at the camera's distance ([`CamSpec::orbit`]).
    pub views: usize,
    pub width: usize,
    /// Views from below too (a cloud in the air); else the upper half.
    pub below: bool,
    pub rule: HiddenRule,
    pub drop_hidden: bool,
    /// A skinned scene (one file): its clips, the samples taken of each and
    /// the directions a pose is looked at from.
    pub clips: Vec<SkinClip>,
    pub skeleton: Option<AthcSkeleton>,
    pub pose_samples: usize,
    pub pose_views: usize,
    pub pose_width: usize,
    pub build: BuildOptions,
}

/// A pruned file, where its splats come from, and what was done.
pub struct PrunedFile {
    pub file: AthcFile,
    /// Its splats (file order) over the input file's splats (file order).
    pub track: SplatSources,
    pub input_splats: usize,
    pub hidden: usize,
    pub decimate: Option<DecimateReport>,
}

pub use crate::cpu_raster::CamSpec;
use crate::athc_build::{build_lod, lod_order, packed_of, BuildOptions};
use crate::athl::SplatSources;

/// The box of the splats' centres (lo, hi).
pub fn splat_bounds(splats: &[Splat]) -> ([f32; 3], [f32; 3]) {
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for s in splats {
        for k in 0..3 {
            lo[k] = lo[k].min(s.p[k]);
            hi[k] = hi[k].max(s.p[k]);
        }
    }
    (lo, hi)
}

/// The hidden splats of every file of a scene (one mask a file, its file order).
pub fn scene_hidden(inputs: &[SceneInput], o: &PruneOptions) -> Result<Vec<Vec<bool>>> {
    let mut splats = Vec::new();
    let mut spans = Vec::new();
    for s in inputs {
        let a = splats.len();
        splats.extend(file_splats(&s.file, Proxy::Albedo));
        spans.push(a..splats.len());
    }
    let n = splats.len();
    let bounds = splat_bounds(&splats);
    let views = o.camera.orbit(o.views, o.width, o.below, bounds);
    let t0 = std::time::Instant::now();
    let (mut t, mut c) = (vec![0f32; n], vec![0f32; n]);
    accumulate_visibility(&splats, &views, &mut t, &mut c);
    eprintln!("visibility: {} views of {} splats at {} px in {:.0}s", views.len(), n, o.width, t0.elapsed().as_secs_f32());
    let mut hidden: Vec<bool> = (0..n).map(|i| t[i] < o.rule.max_transmittance && c[i] < o.rule.max_pixels).collect();
    if !o.clips.is_empty() {
        ensure!(inputs.len() == 1, "poses are taken of a scene of one skinned file");
        let skel = o.skeleton.as_ref().ok_or_else(|| anyhow::anyhow!("clips without a skeleton"))?;
        let influences = file_influences(&inputs[0].file);
        ensure!(influences.len() == n, "the skin covers {} of {} splats", influences.len(), n);
        let pose_views = o.camera.orbit(o.pose_views, o.pose_width, o.below, bounds);
        let (mut pt, mut pc) = (vec![0f32; n], vec![0f32; n]);
        let mut poses = 0;
        for clip in &o.clips {
            let (picks, _) = clip_samples(clip.times.len(), o.pose_samples);
            for k in picks {
                let posed = pose_splats(&splats, &influences, skel, clip, k);
                accumulate_visibility(&posed, &pose_views, &mut pt, &mut pc);
                poses += 1;
            }
            let still = (0..n).filter(|&i| hidden[i] && pt[i] < o.rule.max_transmittance && pc[i] < o.rule.max_pixels).count();
            eprintln!("  {}: {} poses x {} views so far, hidden in all {}", clip.name, poses, pose_views.len(), still);
        }
        for i in 0..n {
            hidden[i] = hidden[i] && pt[i] < o.rule.max_transmittance && pc[i] < o.rule.max_pixels;
        }
    }
    Ok(spans.into_iter().map(|r| hidden[r].to_vec()).collect())
}

/// Prunes and decimates a scene's output files and builds their levels again.
pub fn prune_scene(inputs: &[SceneInput], o: &PruneOptions) -> Result<Vec<Option<PrunedFile>>> {
    let masks = if o.drop_hidden { Some(scene_hidden(inputs, o)?) } else { None };
    let mut out = Vec::new();
    for (k, s) in inputs.iter().enumerate() {
        if !s.output {
            out.push(None);
            continue;
        }
        let n0 = s.file.header.count as usize;
        let mut packed = packed_of(&s.file);
        let mut track = SplatSources::identity(n0);
        let mut hidden = 0;
        if let (Some(m), true) = (&masks, s.prune) {
            let keep: Vec<u32> = (0..n0 as u32).filter(|&i| !m[k][i as usize]).collect();
            hidden = n0 - keep.len();
            packed = select(&packed, &keep);
            track = track.select(&keep)?;
        }
        let mut report = None;
        if let Some((d, filter)) = &s.decimate {
            let (p, runs, r) = decimate_opaque(&packed, d, &|b, i| filter.picks(b, i))?;
            packed = p;
            track = track.merge(&runs.members, &runs.starts, &runs.weights)?;
            report = Some(r);
        }
        let mut build = o.build;
        build.chunk_splats = s.file.header.chunk_splats;
        track = track.select(&lod_order(&packed, &build))?;
        let mut file = build_lod(&packed, &build)?;
        crate::athc::uncap_levels(&mut file);
        out.push(Some(PrunedFile { file, track, input_splats: n0, hidden, decimate: report }));
    }
    Ok(out)
}

/// A `.athl` over `original` carried over the pruned `file`, whose splats
/// come from `original`'s by `track`: each splat's light the weighted mean of
/// its sources' (linear, as colours merge), the merged nodes as the LoD merges
/// them ([`crate::athl::virtual_values`]); the cloud hash re-stamped.
pub fn carry_layers(
    athl: &crate::athl::AthlFile,
    original: &AthcFile,
    file: &AthcFile,
    track: &SplatSources,
    file_bytes: &[u8],
) -> Result<crate::athl::AthlFile> {
    use crate::athc::VirtualTree;
    use crate::athl::{cloud_hash, sparse_layers, validate, virtual_values, AthlFile};
    let old = VirtualTree::of_file(original, true)?;
    ensure!(
        athl.splat_base == old.splat_base && athl.splat_count == original.header.count && athl.merged == old.merged,
        "the .athl is over {} merged + {} splats from {}, the cloud {} + {} from {}",
        athl.merged,
        athl.splat_count,
        athl.splat_base,
        old.merged,
        original.header.count,
        old.splat_base
    );
    ensure!(track.len() == file.header.count as usize, "tracked {} splats, the file has {}", track.len(), file.header.count);
    let tree = VirtualTree::of_file(file, true)?;
    let n = file.header.count;
    let mut out = AthlFile {
        flags: athl.flags,
        element_count: tree.splat_base + n,
        merged: tree.merged,
        splat_base: tree.splat_base,
        splat_count: n,
        cloud_hash: cloud_hash(file_bytes),
        bake_hash: athl.bake_hash,
        groups: athl.groups.clone(),
        polygons: athl.polygons.clone(),
        profiles: athl.profiles.clone(),
        layers: Vec::new(),
    };
    let mut kinds: Vec<(u16, u16)> = athl.layers.iter().map(|l| (l.group, l.kind)).collect();
    kinds.sort();
    kinds.dedup();
    for (group, kind) in kinds {
        let (c, dense) = athl.dense(group, kind).expect("a layer of that group and kind");
        let per = &dense[athl.splat_base as usize * c as usize..];
        let values = track.gather(per, c as usize)?;
        let virt = virtual_values(file, &tree, &values, c)?;
        out.layers.extend(sparse_layers(group, kind, c, &virt, 0.0)?);
    }
    out.layers.sort_by_key(|l| (l.chunk, l.group, l.kind));
    validate(&out)?;
    Ok(out)
}

/// One held-out view of [`check_scene`].
#[derive(Clone, Debug)]
pub struct CheckRow {
    pub view: String,
    /// The output against the input, both at 1x (albedo).
    pub rel_mse: f64,
    /// The same with the normal proxy.
    pub rel_mse_normals: f64,
    /// Mean |ΔT| where either has content.
    pub coverage: f64,
    /// Against the input supersampled 2x: the input's own floor, and the output's.
    pub floor_ss: f64,
    pub out_ss: f64,
}

/// Renders a scene before and after (albedo and the normal proxy) from
/// views no orbit used ([`CamSpec::test_views`], and the same at twice the
/// width for a supersampled reference), AY's metrics in float.
pub fn check_views(
    before: &[Splat],
    after: &[Splat],
    before_n: &[Splat],
    after_n: &[Splat],
    views: &[(String, Camera)],
    views_2x: &[(String, Camera)],
    dump: Option<&str>,
) -> Vec<CheckRow> {
    views
        .iter()
        .zip(views_2x)
        .map(|((name, cam), (_, cam2))| {
            let a = render(before, cam, None);
            let b = render(after, cam, None);
            if let Some(dir) = dump {
                let stem: String = name.split(' ').take(2).collect::<Vec<_>>().join("-");
                let _ = a.save_ppm(&format!("{dir}/{stem}-in.ppm"), None);
                let _ = b.save_ppm(&format!("{dir}/{stem}-out.ppm"), None);
                let _ = b.save_ppm(&format!("{dir}/{stem}-diff.ppm"), Some(&a));
            }
            let an = render(before_n, cam, None);
            let bn = render(after_n, cam, None);
            let reference = render(before, cam2, None).downsample(2);
            CheckRow {
                view: name.clone(),
                rel_mse: b.rel_mse(&a),
                rel_mse_normals: bn.rel_mse(&an),
                coverage: b.alpha_diff(&a),
                floor_ss: a.rel_mse(&reference),
                out_ss: b.rel_mse(&reference),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::athc::VirtualTree;
    use crate::athl::{cloud_hash, sparse_layers, virtual_values, AthlFile, AthlGroup};

    const HOOD: &[u8] = include_bytes!("../../../test/fixtures/athc/hood_t16.athc");
    const SKINNED: &[u8] = include_bytes!("../../../test/fixtures/athc/skinned_corner.athc");

    fn read(bytes: &[u8]) -> AthcFile {
        if bytes[..4] == *b"ATH3" {
            crate::athc_v3::read_v3(bytes).unwrap()
        } else {
            AthcFile::read(bytes).unwrap()
        }
    }

    fn per(v: &[u32], n: usize) -> usize {
        v.len().checked_div(n).unwrap_or(0)
    }

    /// A one-group, three-component .athl over `file` whose splat values are
    /// each splat's centre.
    fn centre_layers(file: &AthcFile, bytes: &[u8]) -> AthlFile {
        let tree = VirtualTree::of_file(file, true).unwrap();
        let n = file.header.count as usize;
        let s = file.splats();
        let values: Vec<f32> = (0..n).flat_map(|i| [s.positions[i * 4], s.positions[i * 4 + 1], s.positions[i * 4 + 2]]).collect();
        let virt = virtual_values(file, &tree, &values, 3).unwrap();
        AthlFile {
            element_count: tree.splat_base + n as u32,
            merged: tree.merged,
            splat_base: tree.splat_base,
            splat_count: n as u32,
            cloud_hash: cloud_hash(bytes),
            groups: vec![AthlGroup { name: "centre".into(), profile: -1, ..Default::default() }],
            layers: sparse_layers(0, 0, 3, &virt, 0.0).unwrap(),
            ..Default::default()
        }
    }

    #[test]
    fn a_splat_under_a_wall_is_hidden() {
        // An opaque floor of discs (2 m, z = 0) and a splat 5 cm under it,
        // looked at from the upper half only: hidden. One beside it, visible.
        let disc = |x: f32, y: f32, z: f32, s: f32| Splat { p: [x, y, z], o: 1.0, q: [0.0, 0.0, 0.0, 1.0], s: [s, s, 0.001], c: [0.5; 3] };
        let mut splats = Vec::new();
        for i in -40..=40 {
            for j in -40..=40 {
                splats.push(disc(i as f32 * 0.05, j as f32 * 0.05, 0.0, 0.05));
            }
        }
        let under = splats.len();
        splats.push(disc(0.0, 0.0, -0.05, 0.02));
        splats.push(disc(0.0, 0.0, 0.05, 0.02));
        let cam = CamSpec { eye: [5.0, 0.0, 2.0], target: [0.0; 3], up: [0.0, 0.0, 1.0], hfov: 40f32.to_radians(), aspect: 1.0 };
        let hidden = hidden_mask(&splats, &cam.orbit(24, 160, false, splat_bounds(&splats)), HiddenRule::default());
        assert!(hidden[under], "the splat under the floor");
        assert!(!hidden[under + 1], "the splat over it");
        assert!(!hidden[40 * 81 + 40], "the floor's middle (under the one over it, around it seen)");
    }

    #[test]
    fn select_and_rebuild_keep_every_section_and_the_layers_aligned() {
        let file = read(HOOD);
        let n = file.header.count as usize;
        let packed = packed_of(&file);
        let keep: Vec<u32> = (0..n as u32).filter(|i| i % 3 != 1).collect();
        let athl = centre_layers(&file, HOOD);
        let inputs = [SceneInput { name: "hood".into(), file: file.clone(), output: true, prune: true, decimate: None }];
        // prune_scene's own path with a mask in place of the views
        let mut track = SplatSources::identity(n).select(&keep).unwrap();
        let cut = select(&packed, &keep);
        let build = BuildOptions { chunk_splats: file.header.chunk_splats, ..Default::default() };
        track = track.select(&lod_order(&cut, &build)).unwrap();
        let mut out = build_lod(&cut, &build).unwrap();
        crate::athc::uncap_levels(&mut out);
        assert_eq!(out.header.count as usize, keep.len());
        let (a, b) = (file.splats(), out.splats());
        for v in [(&a.transfer, &b.transfer), (&a.curvature, &b.curvature), (&a.lobes, &b.lobes), (&a.pbr, &b.pbr), (&a.shadow_bits, &b.shadow_bits), (&a.normals, &b.normals), (&a.sh, &b.sh)] {
            assert_eq!(per(v.0, a.n), per(v.1, b.n));
        }
        assert!(!b.transfer.is_empty());
        // Each kept splat is its source, word for word, wherever the order put it.
        for j in 0..b.n {
            let src = track.index[track.offsets[j] as usize] as usize;
            assert_eq!(&a.positions[src * 4..src * 4 + 4], &b.positions[j * 4..j * 4 + 4]);
            let t = per(&a.transfer, a.n);
            assert_eq!(&a.transfer[src * t..(src + 1) * t], &b.transfer[j * t..(j + 1) * t]);
            assert_eq!(&a.curvature[src * 2..src * 2 + 2], &b.curvature[j * 2..j * 2 + 2]);
        }
        let bytes = crate::athc_v3::write_v3(&out, 0).unwrap();
        let carried = carry_layers(&athl, &file, &out, &track, &bytes).unwrap();
        let (c, dense) = carried.dense(0, 0).unwrap();
        assert_eq!(c, 3);
        for j in 0..b.n {
            let e = carried.splat_base as usize + j;
            for k in 0..3 {
                let (v, p) = (dense[e * 3 + k], b.positions[j * 4 + k]);
                assert!((v - p).abs() <= 2e-3 * p.abs().max(1e-3), "splat {j}: layer {v} at {p}");
            }
        }
        let _ = inputs;
    }

    #[test]
    fn decimating_an_opaque_panel_carries_its_sections() {
        let file = read(HOOD);
        let packed = packed_of(&file);
        let n = packed.block.n;
        let opaque = (0..n).filter(|&i| packed.block.positions[i * 4 + 3] >= 0.99).count();
        let (out, runs, report) = decimate_opaque(&packed, &Decimate::new(3), &|_, _| true).unwrap();
        let m = out.block.n;
        assert_eq!(report.candidates, opaque);
        assert_eq!(m, runs.starts.len());
        assert_eq!(runs.members.len(), n);
        assert!(opaque > 100 && (m as f32) < n as f32 - 0.5 * opaque as f32, "{n} -> {m} ({opaque} opaque)");
        for v in [&out.block.transfer, &out.block.curvature, &out.block.shadow_bits, &out.block.lobes, &out.block.pbr] {
            assert_eq!(v.len() % m, 0);
        }
        // Each run stays on its panel: every member within reach of the kept splat.
        let track = SplatSources::identity(n).merge(&runs.members, &runs.starts, &runs.weights).unwrap();
        let src = &packed.block;
        for j in 0..m {
            let p = &out.block.positions[j * 4..j * 4 + 3];
            let w: f32 = track.weight[track.offsets[j] as usize..track.offsets[j + 1] as usize].iter().sum();
            assert!((w - 1.0).abs() < 1e-4);
            for s in track.offsets[j] as usize..track.offsets[j + 1] as usize {
                let i = track.index[s] as usize;
                let d = (0..3).map(|k| (src.positions[i * 4 + k] - p[k]).powi(2)).sum::<f32>().sqrt();
                assert!(d < 0.05, "run {j}: a member {d} m away");
            }
        }
        // The levels build over it.
        let built = build_lod(&out, &BuildOptions::default()).unwrap();
        assert_eq!(built.header.count as usize, m);
    }

    #[test]
    fn a_skinned_cloud_keeps_its_skin() {
        let file = read(SKINNED);
        assert!(file.has_skin());
        let packed = packed_of(&file);
        let n = packed.block.n;
        let keep: Vec<u32> = (0..n as u32).step_by(2).collect();
        let cut = select(&packed, &keep);
        let built = build_lod(&cut, &BuildOptions { chunk_splats: file.header.chunk_splats, ..Default::default() }).unwrap();
        assert_eq!(built.extra.skin_influences, file.extra.skin_influences);
        assert_eq!(file_influences(&built).len(), keep.len());
        let skel = crate::athc_v3::read_v3_skeleton(SKINNED).unwrap();
        if let Some(s) = skel {
            if let Some(clip) = s.clips.first() {
                let posed = pose_splats(&file_splats(&built, Proxy::Albedo), &file_influences(&built), &s, clip, 0);
                assert_eq!(posed.len(), keep.len());
                assert!(posed.iter().all(|p| p.p.iter().all(|v| v.is_finite())));
            }
        }
    }
}
