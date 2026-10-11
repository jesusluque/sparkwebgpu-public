//! OUR OWN joint CPCA basis for the compact tier (thread CR), fitted to an
//! asset's full transfer, in athenea's layout (`athc_compact::JointBasis`:
//! C clusters, K 8, blocks 16/48/48, int4 weights, an f16 table) so that
//! `build_compact` and the GPU read it unchanged -- and nothing of athenea's
//! q97 basis goes into a published file.
//!
//! The fit (`fit_joint_basis`) over every node of the asset (leaves and the
//! LoD's merged nodes: a cut draws both):
//! 1. per block, the energy norm (the RMS of the block's vector), so that the
//!    joint space weighs the three blocks alike, as athenea's does;
//! 2. k-means (k-means++ seeds on a sample, then Lloyd over every node) of
//!    the joint vectors into C clusters -- the encoder's rule, the nearest
//!    joint centroid;
//! 3. per cluster and block, the mean and the K principal directions (Jacobi)
//!    of its members;
//! 4. per cluster, block and direction, the scale that, after the int4 code
//!    and the GPU's int8 grid (`JointBasis::decode`), leaves the least
//!    squared error over the members (a fraction of the largest weight).
//!
//! The asset table (`AssetBasis`, a sidecar `.cmpa`): the basis ONCE per
//! asset, in thread CQ's CMPA v1 layout (16 fixed words then the f16 table),
//! named by an FNV-1a 64 id; a piece's CMPT then has flag `TABLE_ASSET_BASIS`
//! and, in place of the basis, the two words of that id.

use anyhow::{bail, Result};

use crate::athc_compact::{JointBasis, JOINT_BLOCKS, JOINT_K};

const D: usize = 112;
const BLOCK_FIRST: [usize; 3] = [0, 16, 64];

#[derive(Clone, Debug)]
pub struct FitOptions {
    pub clusters: usize,
    pub iters: usize,
    /// Nodes the k-means++ seeding samples.
    pub seed_sample: usize,
    pub seed: u64,
    pub threads: usize,
    /// Levels a side of the weights (7: int4).
    pub levels: i32,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            clusters: 256,
            iters: 16,
            seed_sample: 200_000,
            seed: 1,
            threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
            levels: 7,
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Splits 0..n into `t` ranges and runs `f` on each, in threads.
fn par<R: Send>(n: usize, t: usize, f: impl Fn(std::ops::Range<usize>) -> R + Sync) -> Vec<R> {
    let t = t.max(1).min(n.max(1));
    let step = n.div_ceil(t);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..t).map(|k| {
            let f = &f;
            s.spawn(move || f(k * step..((k + 1) * step).min(n)))
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// The nearest of `cent` (`c` rows of D, with their squared norms) to z.
#[inline]
fn nearest(cent: &[f32], norms: &[f32], z: &[f32]) -> (usize, f32) {
    let mut best = 0;
    let mut best_d = f32::INFINITY;
    for (c, row) in cent.chunks_exact(D).enumerate() {
        let mut acc = [0f32; 8];
        for (a, b) in row.chunks_exact(8).zip(z.chunks_exact(8)) {
            for k in 0..8 {
                acc[k] += a[k] * b[k];
            }
        }
        let dot: f32 = acc.iter().sum();
        let d = norms[c] - 2.0 * dot;
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    (best, best_d)
}

/// Eigenvectors of the symmetric n x n `a` (row-major), as columns of the
/// returned matrix, ordered by eigenvalue descending (with the values).
pub fn jacobi(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    let mut v = vec![0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let scale: f64 = (0..n).map(|i| a[i * n + i].abs()).sum::<f64>().max(1e-300);
    for _ in 0..100 {
        let mut off = 0f64;
        for p in 0..n {
            for q in p + 1..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off <= 1e-24 * scale * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let sign = if theta < 0.0 { -1.0 } else { 1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j * n + j].partial_cmp(&a[i * n + i]).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0f64; n * n];
    for (col, &i) in order.iter().enumerate() {
        for r in 0..n {
            out[r * n + col] = v[r * n + i];
        }
    }
    (out, order.iter().map(|&i| a[i * n + i]).collect())
}

/// What the fit measured.
#[derive(Clone, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FitReport {
    pub nodes: usize,
    pub clusters_used: usize,
    pub energy_norm: [f32; 3],
    pub kmeans_sse: Vec<f64>,
    pub seconds: f32,
}

/// The joint basis fitted to `x` (112 values a node, flat).
pub fn fit_joint_basis(x: &[f32], o: &FitOptions) -> Result<(JointBasis, FitReport)> {
    let t0 = std::time::Instant::now();
    if x.len() % D != 0 || x.is_empty() {
        bail!("fit: {} values, not a multiple of 112", x.len());
    }
    if o.clusters == 0 || o.clusters > 256 {
        bail!("fit: {} clusters (1..=256: the cluster is a byte)", o.clusters);
    }
    let n = x.len() / D;
    let th = o.threads;
    // 1. Energy norms.
    let e2 = par(n, th, |r| {
        let mut s = [0f64; 3];
        for i in r {
            for b in 0..3 {
                for v in BLOCK_FIRST[b]..BLOCK_FIRST[b] + JOINT_BLOCKS[b] {
                    s[b] += (x[i * D + v] as f64).powi(2);
                }
            }
        }
        s
    });
    let mut en = [0f32; 3];
    for b in 0..3 {
        let s: f64 = e2.iter().map(|s| s[b]).sum();
        en[b] = ((s / n as f64).sqrt() as f32).max(1e-12);
    }
    let inv = en.map(|e| 1.0 / e);
    let zrow = |i: usize, z: &mut [f32]| {
        for v in 0..D {
            let b = if v < 16 { 0 } else if v < 64 { 1 } else { 2 };
            z[v] = x[i * D + v] * inv[b];
        }
    };
    // 2. k-means++ on a sample.
    let mut rng = Rng(o.seed.max(1) * 0x9E37_79B9_7F4A_7C15);
    let m = o.seed_sample.min(n);
    let sample: Vec<usize> = if m == n {
        (0..n).collect()
    } else {
        (0..m).map(|k| ((k as f64 + rng.next()) * n as f64 / m as f64) as usize).map(|i| i.min(n - 1)).collect()
    };
    let mut zs = vec![0f32; m * D];
    for (k, &i) in sample.iter().enumerate() {
        zrow(i, &mut zs[k * D..k * D + D]);
    }
    let c = o.clusters.min(n);
    let mut cent = vec![0f32; c * D];
    let mut dist = vec![f32::INFINITY; m];
    let mut pick = (rng.next() * m as f64) as usize % m;
    for k in 0..c {
        cent[k * D..k * D + D].copy_from_slice(&zs[pick * D..pick * D + D]);
        let row = cent[k * D..k * D + D].to_vec();
        let parts = par(m, th, |r| {
            let mut out = Vec::with_capacity(r.len());
            for j in r {
                let mut s = 0f32;
                for v in 0..D {
                    let e = zs[j * D + v] - row[v];
                    s += e * e;
                }
                out.push(s);
            }
            out
        });
        let mut sum = 0f64;
        for (j, s) in parts.into_iter().flatten().enumerate() {
            if s < dist[j] {
                dist[j] = s;
            }
            sum += dist[j] as f64;
        }
        let mut r = rng.next() * sum;
        pick = m - 1;
        for (j, &d) in dist.iter().enumerate() {
            r -= d as f64;
            if r <= 0.0 {
                pick = j;
                break;
            }
        }
    }
    // Lloyd over every node.
    let mut labels = vec![0u8; n];
    let mut sse_log = Vec::new();
    for it in 0..=o.iters {
        let norms: Vec<f32> = cent.chunks_exact(D).map(|r| r.iter().map(|v| v * v).sum()).collect();
        let parts = par(n, th, |r| {
            let mut acc = vec![0f64; c * D];
            let mut cnt = vec![0usize; c];
            let mut lab = Vec::with_capacity(r.len());
            let mut sse = 0f64;
            let mut far = (0usize, 0f32);
            let mut z = [0f32; D];
            for i in r {
                zrow(i, &mut z);
                let (b, d) = nearest(&cent, &norms, &z);
                let zz: f32 = z.iter().map(|v| v * v).sum();
                let err = (d + zz).max(0.0);
                sse += err as f64;
                if err > far.1 {
                    far = (i, err);
                }
                lab.push(b as u8);
                cnt[b] += 1;
                for v in 0..D {
                    acc[b * D + v] += z[v] as f64;
                }
            }
            (acc, cnt, lab, sse, far)
        });
        let mut acc = vec![0f64; c * D];
        let mut cnt = vec![0usize; c];
        let mut sse = 0f64;
        let mut fars = Vec::new();
        let mut at = 0;
        for (a, k, lab, s, far) in parts {
            for (x, y) in acc.iter_mut().zip(&a) {
                *x += y;
            }
            for (x, y) in cnt.iter_mut().zip(&k) {
                *x += y;
            }
            labels[at..at + lab.len()].copy_from_slice(&lab);
            at += lab.len();
            sse += s;
            fars.push(far);
        }
        sse_log.push(sse / n as f64);
        if it == o.iters {
            break;
        }
        fars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut reseed = fars.into_iter();
        for k in 0..c {
            if cnt[k] == 0 {
                // An empty cluster takes the worst-fitted node of a thread.
                if let Some((i, _)) = reseed.next() {
                    let mut z = [0f32; D];
                    zrow(i, &mut z);
                    cent[k * D..k * D + D].copy_from_slice(&z);
                }
                continue;
            }
            for v in 0..D {
                cent[k * D + v] = (acc[k * D + v] / cnt[k] as f64) as f32;
            }
        }
    }
    // 3. Per cluster and block: mean and covariance.
    let sizes: Vec<usize> = JOINT_BLOCKS.iter().map(|&d| d + d * d).collect();
    let per: usize = sizes.iter().sum();
    let parts = par(n, th, |r| {
        let mut acc = vec![0f64; c * per];
        let mut cnt = vec![0usize; c];
        for i in r {
            let k = labels[i] as usize;
            cnt[k] += 1;
            let mut at = k * per;
            for b in 0..3 {
                let d = JOINT_BLOCKS[b];
                let xs = &x[i * D + BLOCK_FIRST[b]..i * D + BLOCK_FIRST[b] + d];
                for u in 0..d {
                    acc[at + u] += xs[u] as f64;
                }
                at += d;
                for u in 0..d {
                    let xu = xs[u] as f64;
                    for w in u..d {
                        acc[at + u * d + w] += xu * xs[w] as f64;
                    }
                }
                at += d * d;
            }
        }
        (acc, cnt)
    });
    let mut acc = vec![0f64; c * per];
    let mut cnt = vec![0usize; c];
    for (a, k) in parts {
        for (x, y) in acc.iter_mut().zip(&a) {
            *x += y;
        }
        for (x, y) in cnt.iter_mut().zip(&k) {
            *x += y;
        }
    }
    let mut basis = JointBasis {
        clusters: c,
        centroids: vec![0f32; c * D],
        energy_norm: en,
        mean: JOINT_BLOCKS.map(|d| vec![0f32; c * d]),
        dir: JOINT_BLOCKS.map(|d| vec![0f32; c * d * JOINT_K]),
        scale: JOINT_BLOCKS.map(|_| vec![0f32; c * JOINT_K]),
    };
    for k in 0..c {
        if cnt[k] == 0 {
            // Never chosen: far from everything.
            basis.centroids[k * D] = 1e30;
            continue;
        }
        basis.centroids[k * D..k * D + D].copy_from_slice(&cent[k * D..k * D + D]);
        let nk = cnt[k] as f64;
        let mut at = k * per;
        for b in 0..3 {
            let d = JOINT_BLOCKS[b];
            let mu: Vec<f64> = (0..d).map(|u| acc[at + u] / nk).collect();
            at += d;
            let mut cov = vec![0f64; d * d];
            for u in 0..d {
                for w in u..d {
                    let v = acc[at + u * d + w] / nk - mu[u] * mu[w];
                    cov[u * d + w] = v;
                    cov[w * d + u] = v;
                }
            }
            at += d * d;
            let (vecs, _) = jacobi(&cov, d);
            for u in 0..d {
                basis.mean[b][k * d + u] = mu[u] as f32;
                for kk in 0..JOINT_K.min(d) {
                    basis.dir[b][(k * d + u) * JOINT_K + kk] = vecs[u * d + kk] as f32;
                }
            }
        }
    }
    // 4. The scales: the largest weight, then the fraction of it with the
    // least error through the int4 code and the int8 grid.
    let proj = |basis: &JointBasis, i: usize, k: usize, b: usize, kk: usize| -> f64 {
        let d = JOINT_BLOCKS[b];
        let mut w = 0f64;
        for u in 0..d {
            w += basis.dir[b][(k * d + u) * JOINT_K + kk] as f64
                * (x[i * D + BLOCK_FIRST[b] + u] as f64 - basis.mean[b][k * d + u] as f64);
        }
        w
    };
    let slots = c * 3 * JOINT_K;
    let maxes = par(n, th, |r| {
        let mut mx = vec![0f64; slots];
        for i in r {
            let k = labels[i] as usize;
            for b in 0..3 {
                for kk in 0..JOINT_K {
                    let w = proj(&basis, i, k, b, kk).abs();
                    let s = &mut mx[(k * 3 + b) * JOINT_K + kk];
                    if w > *s {
                        *s = w;
                    }
                }
            }
        }
        mx
    });
    let mut mx = vec![0f64; slots];
    for p in &maxes {
        for (a, b) in mx.iter_mut().zip(p) {
            *a = a.max(*b);
        }
    }
    const F: usize = 24;
    let frac = |j: usize| 1.0 - 0.8 * j as f64 / (F - 1) as f64;
    let lv = o.levels as f64;
    let errs = par(n, th, |r| {
        let mut e = vec![0f64; slots * F];
        for i in r {
            let k = labels[i] as usize;
            for b in 0..3 {
                for kk in 0..JOINT_K {
                    let slot = (k * 3 + b) * JOINT_K + kk;
                    let w = proj(&basis, i, k, b, kk);
                    for j in 0..F {
                        let s = mx[slot] * frac(j);
                        let back = if s > 0.0 {
                            let q = ((w / s) * lv).round().clamp(-lv, lv);
                            (q * 127.0 / lv).round() * s / 127.0
                        } else {
                            0.0
                        };
                        e[slot * F + j] += (w - back) * (w - back);
                    }
                }
            }
        }
        e
    });
    let mut err = vec![0f64; slots * F];
    for p in &errs {
        for (a, b) in err.iter_mut().zip(p) {
            *a += b;
        }
    }
    for k in 0..c {
        for b in 0..3 {
            for kk in 0..JOINT_K {
                let slot = (k * 3 + b) * JOINT_K + kk;
                let best = (0..F).min_by(|&p, &q| err[slot * F + p].partial_cmp(&err[slot * F + q]).unwrap()).unwrap();
                basis.scale[b][k * JOINT_K + kk] = (mx[slot] * frac(best)) as f32;
            }
        }
    }
    let report = FitReport {
        nodes: n,
        clusters_used: cnt.iter().filter(|&&v| v > 0).count(),
        energy_norm: en,
        kmeans_sse: sse_log,
        seconds: t0.elapsed().as_secs_f32(),
    };
    Ok((basis, report))
}

/// Per block, the squared error of `x` through `basis` at `levels` and its
/// squared norm (relMSE = err / ref), as `build_compact` measures it.
pub fn transfer_error(basis: &JointBasis, x: &[f32], levels: i32, threads: usize) -> ([f64; 3], [f64; 3]) {
    let n = x.len() / D;
    let parts = par(n, threads, |r| {
        let (mut e, mut f) = ([0f64; 3], [0f64; 3]);
        for i in r {
            let xs = &x[i * D..i * D + D];
            let (c, q) = basis.encode(xs, levels);
            let back = basis.decode(c, &q, levels);
            for b in 0..3 {
                for v in BLOCK_FIRST[b]..BLOCK_FIRST[b] + JOINT_BLOCKS[b] {
                    e[b] += ((back[v] - xs[v]) as f64).powi(2);
                    f[b] += (xs[v] as f64).powi(2);
                }
            }
        }
        (e, f)
    });
    let (mut e, mut f) = ([0f64; 3], [0f64; 3]);
    for (a, b) in parts {
        for k in 0..3 {
            e[k] += a[k];
            f[k] += b[k];
        }
    }
    (e, f)
}

/// The basis as athenea's .json + .bin pair (`JointBasis::parse` reads it).
pub fn basis_json_bin(basis: &JointBasis, meta: serde_json::Value) -> (String, Vec<u8>) {
    let c = basis.clusters;
    let mut bin: Vec<u8> = Vec::new();
    let mut arrays = Vec::new();
    let mut add = |name: String, shape: Vec<usize>, v: &[f32]| {
        arrays.push(serde_json::json!({ "name": name, "shape": shape, "offset_bytes": bin.len(), "size_bytes": 4 * v.len() }));
        for f in v {
            bin.extend_from_slice(&f.to_le_bytes());
        }
    };
    add("centroids_joint".into(), vec![c, D], &basis.centroids);
    add("energy_norm".into(), vec![3], &basis.energy_norm);
    for (b, name) in ["transferDirect", "transferIndirect", "transferReflected"].iter().enumerate() {
        let d = JOINT_BLOCKS[b];
        add(format!("{name}_mean"), vec![c, d], &basis.mean[b]);
        add(format!("{name}_basis"), vec![c, d, JOINT_K], &basis.dir[b]);
        add(format!("{name}_scale"), vec![c, JOINT_K], &basis.scale[b]);
    }
    let mut j = serde_json::json!({
        "endianness": "little",
        "dtype": "float32",
        "C": c,
        "K": JOINT_K,
        "dims": { "transferDirect": 16, "transferIndirect": 48, "transferReflected": 48 },
        "encode": "z = concat(x_d/energy_norm[0], x_i/energy_norm[1], x_r/energy_norm[2]); c = argmin ||z - centroids_joint[c]||^2; per block w = basis[c]^T (x - mean[c]), q = clamp(round(w/scale[c]*7), -7, 7) (int4)",
        "decode": "x = mean[c] + basis[c] (round(q*127/7) * scale[c] / 127)",
        "arrays": arrays,
    });
    if let (Some(o), serde_json::Value::Object(m)) = (j.as_object_mut(), meta) {
        o.extend(m);
    }
    (serde_json::to_string_pretty(&j).unwrap(), bin)
}

// --- the asset table (thread CQ's CMPA v1, the basis alone) -------------------

pub const ASSET_MAGIC: u32 = u32::from_le_bytes(*b"CMPA");
pub const ASSET_VERSION: u32 = 1;
const ASSET_FIXED_WORDS: usize = 16;

/// FNV-1a, 64 bits.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// An asset's basis, shared by its pieces: the blob (16 fixed words: magic,
/// version, words, flags (1: basis), clusters, entries 0, materials 0, basis
/// at, basis words, codebook at, materials at, id lo, id hi, 0 0 0; then the
/// f16 table at 16) and its id (FNV-1a 64 of the blob with the id words 0).
pub fn asset_basis_bytes(table_f16: &[u32], clusters: u32) -> (Vec<u8>, u64) {
    let basis_at = ASSET_FIXED_WORDS;
    let end = (basis_at + table_f16.len()).div_ceil(16) * 16;
    let mut w = vec![0u32; end];
    w[..11].copy_from_slice(&[
        ASSET_MAGIC,
        ASSET_VERSION,
        end as u32,
        1,
        clusters,
        0,
        0,
        basis_at as u32,
        table_f16.len() as u32,
        end as u32,
        end as u32,
    ]);
    w[basis_at..basis_at + table_f16.len()].copy_from_slice(table_f16);
    let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let id = fnv1a64(&bytes);
    w[11] = id as u32;
    w[12] = (id >> 32) as u32;
    (w.iter().flat_map(|v| v.to_le_bytes()).collect(), id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_a_clustered_transfer_and_writes_a_basis_athenea_s_reader_takes() {
        // Two clusters of nodes, each on a 3-dimensional subspace.
        let mut rng = Rng(7);
        let n = 4000;
        let mut x = vec![0f32; n * D];
        for i in 0..n {
            let off = if i % 2 == 0 { 1.0 } else { -2.0 };
            let a = [rng.next() - 0.5, rng.next() - 0.5, rng.next() - 0.5];
            for v in 0..D {
                let s = ((v * 7 + i % 2) as f64).sin();
                x[i * D + v] = (off * s + a[v % 3] * 0.3) as f32;
            }
        }
        let o = FitOptions { clusters: 4, iters: 8, seed_sample: 1000, threads: 4, ..Default::default() };
        let (basis, rep) = fit_joint_basis(&x, &o).unwrap();
        assert!(rep.clusters_used >= 2);
        let (e, f) = transfer_error(&basis, &x, 7, 4);
        for b in 0..3 {
            assert!(e[b] / f[b] < 1e-3, "block {b}: {}", e[b] / f[b]);
        }
        let (json, bin) = basis_json_bin(&basis, serde_json::json!({ "source": "test" }));
        let back = JointBasis::parse(&json, &bin).unwrap();
        assert_eq!(back.table_f16(), basis.table_f16());
        let (blob, id) = asset_basis_bytes(&basis.table_f16(), 4);
        assert_eq!(&blob[..4], b"CMPA");
        let mut zeroed = blob.clone();
        zeroed[44..52].fill(0);
        assert_eq!(fnv1a64(&zeroed), id);
    }
}
