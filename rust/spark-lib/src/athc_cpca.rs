//! Clustered PCA of the transfer (Sloan et al. 2003, "CPCA"): `.athc` v3
//! section encoding 3, lossy, for the transfer sections (`TXDI`, `TXIN`,
//! `TXFD`) only.
//!
//! A section of a block (n elements of D = 2 x words f16 values) is split
//! into K clusters (k-means); each cluster keeps its mean and its first M
//! principal directions (f16), each element its cluster and M coefficients,
//! quantized with a step per cluster proportional to the cluster's RMS (the
//! same relative error everywhere, dim splats included). K is capped by the
//! block's size, M is the fewest directions whose truncation leaves room for
//! the quantization within the error budget (`CpcaOptions::rel_mse`, the
//! block's sum of squared errors over its sum of squared values). A block
//! that cannot meet it, or that is smaller stored as byte planes, keeps its
//! values exactly (mode 0).
//!
//! What a reader gets is the section's words as before (f16 halves): the
//! decode is done once, in the loader, and everything after it (attribute
//! pools, paging, transfer forms) is unchanged. The decode is exact
//! arithmetic in f32 in a fixed order, so the Rust (WASM) and the TypeScript
//! decoders (src/athc.ts) give the same halves bit for bit.
//!
//! Payload (before the section's compression), little endian:
//!
//! ```text
//! mode 0:  u32 0, then the D halves of every element as byte planes (encoding 1)
//! mode 1:  u32 1, u32 K, u32 M, u32 D, u32 B (coefficient bytes, 1..3)
//!          K f32 steps
//!          K x (1 + M) x D f16: each cluster's mean, then its M directions
//!          (padded to 4 bytes)
//!          n u8 clusters
//!          M x B planes of n bytes: byte b of zigzag(q_j) of every element
//! ```
//!
//! Element e, value d: acc = mean\[k\]\[d\]; for j < M, acc = acc +
//! (q_j times step_k) times dir\[k\]\[j\]\[d\], each product and sum
//! rounded to f32; out = f16(acc), round to nearest even.

use anyhow::{bail, Result};
use half::f16;

use crate::athc_v3::{decode_section, encode_section, gzip, ENCODING_BYTE_PLANES};

pub const ENCODING_CPCA: u32 = 3;
const MODE_EXACT: u32 = 0;
const MODE_CPCA: u32 = 1;
/// Fewest elements a cluster is worth (its mean and directions cost bytes).
const MIN_PER_CLUSTER: usize = 128;
/// k-means runs on at most this many elements (evenly spaced), then every
/// element takes its nearest centre.
const KMEANS_SAMPLE: usize = 24_000;
const KMEANS_ITERS: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpcaOptions {
    /// Clusters at most (1..=256).
    pub clusters: usize,
    /// Directions kept at most (a block keeps the fewest that fit).
    pub coeffs: usize,
    /// Error budget of a block: sum of squared errors over sum of squares.
    pub rel_mse: f64,
    /// Keep CPCA even where byte planes would store smaller (tests).
    pub force: bool,
    /// Compare sizes after gzip (the section is gzipped).
    pub gzip: bool,
}

impl Default for CpcaOptions {
    fn default() -> Self {
        Self { clusters: 16, coeffs: 48, rel_mse: 1e-5, force: false, gzip: true }
    }
}

impl CpcaOptions {
    /// `K,M[,E]` as athc-convert's `--cpca` takes it.
    pub fn parse(s: &str) -> Result<Self> {
        let v: Vec<&str> = s.split(',').collect();
        if v.len() < 2 || v.len() > 3 {
            bail!("--cpca K,M[,E]");
        }
        let mut o = Self { clusters: v[0].parse()?, coeffs: v[1].parse()?, ..Default::default() };
        if let Some(e) = v.get(2) {
            o.rel_mse = e.parse()?;
        }
        if o.clusters == 0 || o.clusters > 256 || o.coeffs == 0 || o.rel_mse.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            bail!("--cpca: K in 1..=256, M >= 1, E > 0");
        }
        Ok(o)
    }
}

/// What a block's section became.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CpcaBlockStats {
    pub n: usize,
    /// Mode 1 (else kept exactly).
    pub cpca: bool,
    pub clusters: usize,
    pub coeffs: usize,
    /// Sum of squared errors and of squared values, over the f16 values.
    pub err2: f64,
    pub ref2: f64,
}

fn halves(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2).map(|h| f16::from_bits(u16::from_le_bytes([h[0], h[1]])).to_f32()).collect()
}

fn round16(v: f32) -> f32 {
    f16::from_f32(v).to_f32()
}

/// octahedral.slang octDecode.
fn oct_decode(u: f64, v: f64) -> [f64; 3] {
    let (x, y) = (u * 2.0 - 1.0, v * 2.0 - 1.0);
    let mut n = [x, y, 1.0 - x.abs() - y.abs()];
    let t = (-n[2]).clamp(0.0, 1.0);
    n[0] += if n[0] >= 0.0 { -t } else { t };
    n[1] += if n[1] >= 0.0 { -t } else { t };
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    [n[0] / l, n[1] / l, n[2] / l]
}

/// A zonal transfer's ten values (two lobes: octahedral axis, three band
/// coefficients) as the nine harmonics splat_relight's splatTransferFrame
/// makes of them, in the gaussian's frame (orthonormal: their L2 is the
/// transfer's over the sphere).
pub fn zonal_harmonics(z: &[f32]) -> [f64; 9] {
    const SCALE: [f64; 3] = [3.544_907_701_811_032, 2.046_653_415_892_977, 1.585_330_919_042_404_3];
    let mut sh = [0f64; 9];
    for lobe in 0..2 {
        let l = &z[lobe * 5..lobe * 5 + 5];
        let [x, y, zz] = oct_decode(l[0] as f64, l[1] as f64);
        let basis = [
            0.28209479177387814,
            -0.4886025119029199 * y,
            0.4886025119029199 * zz,
            -0.4886025119029199 * x,
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * zz,
            0.31539156525252005 * (2.0 * zz * zz - x * x - y * y),
            -1.0925484305920792 * x * zz,
            0.5462742152960396 * (x * x - y * y),
        ];
        for k in 0..9 {
            let band = if k == 0 { 0 } else if k < 4 { 1 } else { 2 };
            sh[k] += l[2 + band] as f64 * SCALE[band] * basis[k];
        }
    }
    sh
}

/// The squared error the budget counts, of one element: over the values,
/// or for a zonal transfer (ten values) over its harmonics.
fn metric_err2(x: &[f32], y: &[f32]) -> f64 {
    if x.len() == 10 {
        let (a, b) = (zonal_harmonics(x), zonal_harmonics(y));
        a.iter().zip(&b).map(|(p, q)| (p - q).powi(2)).sum()
    } else {
        x.iter().zip(y).map(|(&p, &q)| (p as f64 - q as f64).powi(2)).sum()
    }
}

fn metric_ref2(x: &[f32]) -> f64 {
    if x.len() == 10 {
        zonal_harmonics(x).iter().map(|v| v * v).sum()
    } else {
        x.iter().map(|&v| v as f64 * v as f64).sum()
    }
}

fn zigzag(q: i32) -> u32 {
    ((q << 1) ^ (q >> 31)) as u32
}

fn unzigzag(z: u32) -> i32 {
    ((z >> 1) as i32) ^ -((z & 1) as i32)
}

fn exact_payload(raw: &[u8], n: usize, words: usize) -> Vec<u8> {
    let mut out = MODE_EXACT.to_le_bytes().to_vec();
    out.extend_from_slice(&encode_section(raw, n, &[words as u32], ENCODING_BYTE_PLANES));
    out
}

/// A tiny deterministic generator (xorshift64*).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn nearest(x: &[f32], centres: &[f32], d: usize) -> usize {
    let mut best = (f32::MAX, 0);
    for (k, c) in centres.chunks_exact(d).enumerate() {
        let e = dist2(x, c);
        if e < best.0 {
            best = (e, k);
        }
    }
    best.1
}

/// Each element's nearest of `centres`, over threads where there are any.
fn assign(x: &[f32], d: usize, centres: &[f32]) -> Vec<u8> {
    let n = x.len() / d;
    let mut out = vec![0u8; n];
    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(16);
        let per = n.div_ceil(threads).max(1024);
        std::thread::scope(|s| {
            for (part, ids) in out.chunks_mut(per).enumerate() {
                s.spawn(move || {
                    for (i, id) in ids.iter_mut().enumerate() {
                        let e = part * per + i;
                        *id = nearest(&x[e * d..(e + 1) * d], centres, d) as u8;
                    }
                });
            }
        });
    }
    #[cfg(target_arch = "wasm32")]
    for (e, id) in out.iter_mut().enumerate() {
        *id = nearest(&x[e * d..(e + 1) * d], centres, d) as u8;
    }
    out
}

/// k-means (k-means++ start, Lloyd) on an even sample; every element's
/// cluster. Empty clusters are dropped (ids renumbered).
fn kmeans(x: &[f32], d: usize, k: usize) -> (usize, Vec<u8>) {
    let n = x.len() / d;
    if k <= 1 {
        return (1, vec![0; n]);
    }
    let stride = n.div_ceil(KMEANS_SAMPLE).max(1);
    let sample: Vec<usize> = (0..n).step_by(stride).collect();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (n as u64) << 7 ^ d as u64);
    let mut centres: Vec<f32> = Vec::with_capacity(k * d);
    let first = sample[(rng.next() % sample.len() as u64) as usize];
    centres.extend_from_slice(&x[first * d..(first + 1) * d]);
    let mut near: Vec<f32> = sample.iter().map(|&e| dist2(&x[e * d..(e + 1) * d], &centres[..d])).collect();
    while centres.len() < k * d {
        let total: f64 = near.iter().map(|&v| v as f64).sum();
        if total <= 0.0 {
            break;
        }
        let mut r = rng.unit() * total;
        let mut pick = sample.len() - 1;
        for (s, &v) in near.iter().enumerate() {
            r -= v as f64;
            if r <= 0.0 {
                pick = s;
                break;
            }
        }
        let e = sample[pick];
        let c = x[e * d..(e + 1) * d].to_vec();
        for (s, &e) in sample.iter().enumerate() {
            near[s] = near[s].min(dist2(&x[e * d..(e + 1) * d], &c));
        }
        centres.extend_from_slice(&c);
    }
    let kk = centres.len() / d;
    let sx: Vec<f32> = sample.iter().flat_map(|&e| x[e * d..(e + 1) * d].iter().copied()).collect();
    for _ in 0..KMEANS_ITERS {
        let ids = assign(&sx, d, &centres);
        let mut sum = vec![0f64; kk * d];
        let mut count = vec![0usize; kk];
        for (s, &id) in ids.iter().enumerate() {
            count[id as usize] += 1;
            for j in 0..d {
                sum[id as usize * d + j] += sx[s * d + j] as f64;
            }
        }
        let mut moved = false;
        for c in 0..kk {
            if count[c] == 0 {
                continue;
            }
            for j in 0..d {
                let v = (sum[c * d + j] / count[c] as f64) as f32;
                moved |= v != centres[c * d + j];
                centres[c * d + j] = v;
            }
        }
        if !moved {
            break;
        }
    }
    let ids = assign(x, d, &centres);
    // Renumber, dropping empty clusters.
    let mut used = vec![0usize; kk];
    for &id in &ids {
        used[id as usize] += 1;
    }
    let mut map = vec![0u8; kk];
    let mut m = 0;
    for c in 0..kk {
        if used[c] > 0 {
            map[c] = m as u8;
            m += 1;
        }
    }
    (m, ids.into_iter().map(|id| map[id as usize]).collect())
}

/// Eigenvalues (descending) and unit eigenvectors (rows) of a symmetric
/// d x d matrix, by cyclic Jacobi.
fn eigen(mut a: Vec<f64>, d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0f64; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    for _sweep in 0..60 {
        let mut off = 0.0;
        let mut diag = 0.0;
        for p in 0..d {
            diag += a[p * d + p] * a[p * d + p];
            for q in p + 1..d {
                off += a[p * d + q] * a[p * d + q];
            }
        }
        if off <= 1e-30 * diag.max(1e-300) {
            break;
        }
        for p in 0..d {
            for q in p + 1..d {
                let apq = a[p * d + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * d + q] - a[p * d + p]) / (2.0 * apq);
                let t = theta.signum().max(0.0).mul_add(2.0, -1.0) / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..d {
                    let akp = a[k * d + p];
                    let akq = a[k * d + q];
                    a[k * d + p] = c * akp - s * akq;
                    a[k * d + q] = s * akp + c * akq;
                }
                for k in 0..d {
                    let apk = a[p * d + k];
                    let aqk = a[q * d + k];
                    a[p * d + k] = c * apk - s * aqk;
                    a[q * d + k] = s * apk + c * aqk;
                }
                for k in 0..d {
                    let vkp = v[k * d + p];
                    let vkq = v[k * d + q];
                    v[k * d + p] = c * vkp - s * vkq;
                    v[k * d + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&i, &j| a[j * d + j].partial_cmp(&a[i * d + i]).unwrap_or(std::cmp::Ordering::Equal));
    let values = order.iter().map(|&i| a[i * d + i].max(0.0)).collect();
    let mut rows = vec![0f64; d * d];
    for (r, &i) in order.iter().enumerate() {
        for k in 0..d {
            rows[r * d + k] = v[k * d + i];
        }
    }
    (values, rows)
}

struct Cluster {
    n: usize,
    mean: Vec<f64>,
    /// Eigenvalues (variance a direction, descending) and directions.
    values: Vec<f64>,
    dirs: Vec<f64>,
    /// Mean squared norm of its elements.
    energy: f64,
}

fn clusters_of(x: &[f32], d: usize, k: usize, ids: &[u8]) -> Vec<Cluster> {
    let n = x.len() / d;
    let mut sum = vec![0f64; k * d];
    let mut count = vec![0usize; k];
    let mut energy = vec![0f64; k];
    for e in 0..n {
        let c = ids[e] as usize;
        count[c] += 1;
        for j in 0..d {
            let v = x[e * d + j] as f64;
            sum[c * d + j] += v;
            energy[c] += v * v;
        }
    }
    let mut cov = vec![0f64; k * d * d];
    let means: Vec<f64> = (0..k * d).map(|i| sum[i] / count[i / d].max(1) as f64).collect();
    let mut row = vec![0f64; d];
    for e in 0..n {
        let c = ids[e] as usize;
        for j in 0..d {
            row[j] = x[e * d + j] as f64 - means[c * d + j];
        }
        let m = &mut cov[c * d * d..(c + 1) * d * d];
        for i in 0..d {
            let ri = row[i];
            if ri == 0.0 {
                continue;
            }
            for j in i..d {
                m[i * d + j] += ri * row[j];
            }
        }
    }
    (0..k)
        .map(|c| {
            let nc = count[c].max(1) as f64;
            let mut m = cov[c * d * d..(c + 1) * d * d].to_vec();
            for i in 0..d {
                for j in i..d {
                    m[i * d + j] /= nc;
                    m[j * d + i] = m[i * d + j];
                }
            }
            let (values, dirs) = eigen(m, d);
            Cluster { n: count[c], mean: means[c * d..(c + 1) * d].to_vec(), values, dirs, energy: energy[c] / nc }
        })
        .collect()
}

/// The f32 reconstruction both decoders compute (see the module's comment).
#[inline]
fn reconstruct(mean: &[f32], dirs: &[f32], coef: &[f32], d: usize, out: &mut [f32]) {
    for v in 0..d {
        let mut acc = mean[v];
        for (j, &c) in coef.iter().enumerate() {
            acc += c * dirs[j * d + v];
        }
        out[v] = acc;
    }
}

struct Quantized {
    payload: Vec<u8>,
    err2: f64,
}

/// Mode 1 payload for these clusters, `m` directions and step scale `s`;
/// None where a coefficient needs more than 3 bytes.
fn quantize(x: &[f32], w: &[f32], d: usize, ids: &[u8], cl: &[Cluster], m: usize, s: f64) -> Option<Quantized> {
    let n = x.len() / d;
    let k = cl.len();
    // The clusters are in the weighted space (y = w x); what is stored is
    // back in the values' own (the directions divided by the weights).
    let means: Vec<Vec<f32>> =
        cl.iter().map(|c| c.mean.iter().enumerate().map(|(v, &u)| round16((u / w[v] as f64) as f32)).collect()).collect();
    let dirs: Vec<Vec<f32>> = cl
        .iter()
        .map(|c| c.dirs[..m * d].iter().enumerate().map(|(i, &u)| round16((u / w[i % d] as f64) as f32)).collect())
        .collect();
    let steps: Vec<f32> = cl
        .iter()
        .map(|c| {
            let st = (s * c.energy.sqrt()) as f32;
            if st > 0.0 && st.is_finite() { st } else { 1.0 }
        })
        .collect();
    let mut qs = vec![0i32; n * m];
    let mut err2 = 0f64;
    let mut maxz = 0u32;
    let mut diff = vec![0f64; d];
    let mut coef = vec![0f32; m];
    let mut rec = vec![0f32; d];
    for e in 0..n {
        let c = ids[e] as usize;
        let xe = &x[e * d..(e + 1) * d];
        for v in 0..d {
            diff[v] = (xe[v] - means[c][v]) as f64 * w[v] as f64;
        }
        for j in 0..m {
            let dir = &cl[c].dirs[j * d..(j + 1) * d];
            let dot: f64 = diff.iter().zip(dir).map(|(&a, &b)| a * b).sum();
            let q = (dot / steps[c] as f64).round();
            if q.abs() >= (1 << 23) as f64 {
                return None;
            }
            let q = q as i32;
            qs[e * m + j] = q;
            maxz = maxz.max(zigzag(q));
            coef[j] = q as f32 * steps[c];
        }
        reconstruct(&means[c], &dirs[c], &coef, d, &mut rec);
        for r in rec.iter_mut() {
            *r = round16(*r);
        }
        err2 += metric_err2(xe, &rec);
    }
    let bytes = if maxz < 1 << 8 { 1 } else if maxz < 1 << 16 { 2 } else { 3 };
    let mut out = Vec::new();
    for w in [MODE_CPCA, k as u32, m as u32, d as u32, bytes] {
        out.extend_from_slice(&w.to_le_bytes());
    }
    for st in &steps {
        out.extend_from_slice(&st.to_le_bytes());
    }
    for c in 0..k {
        for &v in means[c].iter().chain(&dirs[c]) {
            out.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
        }
    }
    out.resize(out.len().div_ceil(4) * 4, 0);
    out.extend_from_slice(ids);
    for j in 0..m {
        for b in 0..bytes {
            out.extend((0..n).map(|e| (zigzag(qs[e * m + j]) >> (8 * b)) as u8));
        }
    }
    Some(Quantized { payload: out, err2 })
}

/// A transfer section's raw bytes (n elements of `words` words) as a CPCA
/// payload, and what it cost.
pub fn encode_cpca(raw: &[u8], n: usize, words: usize, o: &CpcaOptions) -> (Vec<u8>, CpcaBlockStats) {
    let d = 2 * words;
    let exact = |ref2: f64| (exact_payload(raw, n, words), CpcaBlockStats { n, ref2, ..Default::default() });
    if n == 0 || raw.len() != n * d * 2 {
        return exact(0.0);
    }
    let x = halves(raw);
    if x.iter().any(|v| !v.is_finite()) {
        return exact(0.0);
    }
    let ref2: f64 = x.chunks_exact(d).map(metric_ref2).sum();
    if ref2 == 0.0 {
        return exact(0.0);
    }
    let budget = o.rel_mse * ref2;
    // A zonal transfer mixes octahedral coordinates with band coefficients
    // of another scale: each value weighted by its RMS over the block, so
    // the clusters and directions see them alike. Harmonics as they are
    // (orthonormal: equal errors are equal errors over the sphere).
    let mut w = vec![1f32; d];
    if d == 10 {
        for (v, wv) in w.iter_mut().enumerate() {
            let rms = (x.iter().skip(v).step_by(d).map(|&a| a as f64 * a as f64).sum::<f64>() / n as f64).sqrt();
            if rms > 0.0 {
                *wv = (1.0 / rms) as f32;
            }
        }
    }
    let y: Vec<f32> = x.iter().enumerate().map(|(i, &v)| v * w[i % d]).collect();
    let ref2_y: f64 = y.iter().map(|&v| v as f64 * v as f64).sum();
    let budget_y = o.rel_mse * ref2_y;
    let k = o.clusters.min((n / MIN_PER_CLUSTER).max(1)).clamp(1, 256);
    let (k, ids) = kmeans(&y, d, k);
    let cl = clusters_of(&y, d, k, &ids);
    // The fewest directions whose truncation takes at most 40 % of the budget.
    let trunc = |m: usize| -> f64 { cl.iter().map(|c| c.n as f64 * c.values[m..].iter().sum::<f64>()).sum() };
    let Some(m) = (1..=o.coeffs.min(d)).find(|&m| trunc(m) <= 0.4 * budget_y) else {
        return exact(ref2);
    };
    // Quantization error ~ m s^2 ref2 / 12 (a step s x the cluster's RMS).
    let mut s = (12.0 * (budget_y - trunc(m)).max(0.1 * budget_y) / (m as f64 * ref2_y)).sqrt();
    let mut best: Option<Quantized> = None;
    for _ in 0..8 {
        let Some(q) = quantize(&x, &w, d, &ids, &cl, m, s) else { break };
        if q.err2 <= budget {
            best = Some(q);
            break;
        }
        s *= (budget / q.err2).sqrt() * 0.9;
    }
    let Some(q) = best else { return exact(ref2) };
    let stats = CpcaBlockStats { n, cpca: true, clusters: k, coeffs: m, err2: q.err2, ref2 };
    if !o.force {
        let plain = exact_payload(raw, n, words);
        let size = |p: &[u8]| if o.gzip { gzip(p).len() } else { p.len() };
        if size(&plain) <= size(&q.payload) {
            return (plain, CpcaBlockStats { n, ref2, ..Default::default() });
        }
    }
    (q.payload, stats)
}

fn u32_at(b: &[u8], at: usize) -> Result<u32> {
    match b.get(at..at + 4) {
        Some(w) => Ok(u32::from_le_bytes(w.try_into().unwrap())),
        None => bail!(".athc v3: a CPCA section too short"),
    }
}

/// `encode_cpca` undone: the section's raw bytes.
pub fn decode_cpca(stored: &[u8], n: usize, words: usize) -> Result<Vec<u8>> {
    let d = 2 * words;
    match u32_at(stored, 0)? {
        MODE_EXACT => decode_section(&stored[4..], n, &[words as u32], ENCODING_BYTE_PLANES),
        MODE_CPCA => {
            let k = u32_at(stored, 4)? as usize;
            let m = u32_at(stored, 8)? as usize;
            let dd = u32_at(stored, 12)? as usize;
            let bytes = u32_at(stored, 16)? as usize;
            if dd != d || k == 0 || k > 256 || m == 0 || m > d || !(1..=3).contains(&bytes) {
                bail!(".athc v3: a CPCA section of {k} clusters, {m} of {dd} values ({d} expected), {bytes} bytes");
            }
            let steps_at = 20;
            let basis_at = steps_at + 4 * k;
            let ids_at = (basis_at + 2 * k * (1 + m) * d).div_ceil(4) * 4;
            let planes_at = ids_at + n;
            if stored.len() != planes_at + m * bytes * n {
                bail!(".athc v3: a CPCA section of {} bytes, not {}", stored.len(), planes_at + m * bytes * n);
            }
            let f16_at = |i: usize| f16::from_bits(u16::from_le_bytes([stored[basis_at + 2 * i], stored[basis_at + 2 * i + 1]])).to_f32();
            let steps: Vec<f32> =
                (0..k).map(|c| f32::from_le_bytes(stored[steps_at + 4 * c..steps_at + 4 * c + 4].try_into().unwrap())).collect();
            let per = (1 + m) * d;
            let basis: Vec<f32> = (0..k * per).map(f16_at).collect();
            let ids = &stored[ids_at..planes_at];
            let planes = &stored[planes_at..];
            let mut out = vec![0u8; n * d * 2];
            let mut coef = vec![0f32; m];
            let mut rec = vec![0f32; d];
            for e in 0..n {
                let c = ids[e] as usize;
                if c >= k {
                    bail!(".athc v3: a CPCA element in cluster {c} of {k}");
                }
                for (j, cj) in coef.iter_mut().enumerate() {
                    let mut z = 0u32;
                    for b in 0..bytes {
                        z |= (planes[(j * bytes + b) * n + e] as u32) << (8 * b);
                    }
                    *cj = unzigzag(z) as f32 * steps[c];
                }
                let cb = &basis[c * per..(c + 1) * per];
                reconstruct(&cb[..d], &cb[d..], &coef, d, &mut rec);
                for v in 0..d {
                    out[(e * d + v) * 2..(e * d + v) * 2 + 2].copy_from_slice(&f16::from_f32(rec[v]).to_bits().to_le_bytes());
                }
            }
            Ok(out)
        }
        mode => bail!(".athc v3: CPCA mode {mode}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// n elements of d values on a few low-rank families, as halves.
    fn families(n: usize, d: usize) -> Vec<u8> {
        let mut rng = Rng(7);
        let mut out = Vec::new();
        for e in 0..n {
            let fam = e % 3;
            let a = rng.unit() as f32;
            let b = rng.unit() as f32;
            for v in 0..d {
                let t = v as f32 / d as f32;
                let x = match fam {
                    0 => 0.8 + 0.3 * a * (t * 6.0).sin() + 0.05 * b * t,
                    1 => -0.2 + 0.5 * a * t * t - 0.1 * b,
                    _ => 2.0 * a * (t * 3.0).cos() + 0.01 * (rng.unit() as f32 - 0.5),
                };
                out.extend_from_slice(&f16::from_f32(x).to_bits().to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn meets_the_budget_and_round_trips() {
        for (n, words) in [(5000, 8), (3000, 24), (300, 5)] {
            let raw = families(n, 2 * words);
            for e in if words == 5 { [1e-3, 1e-4] } else { [1e-4, 1e-5] } {
                let o = CpcaOptions { clusters: 8, coeffs: 2 * words, rel_mse: e, force: true, gzip: true };
                let (payload, st) = encode_cpca(&raw, n, words, &o);
                assert!(st.cpca, "{n} {words} {e}: {st:?}");
                assert!(st.err2 <= e * st.ref2, "{st:?}");
                let back = decode_cpca(&payload, n, words).unwrap();
                assert_eq!(back.len(), raw.len());
                let (x, y) = (halves(&raw), halves(&back));
                let err2: f64 = x.chunks_exact(2 * words).zip(y.chunks_exact(2 * words)).map(|(a, b)| metric_err2(a, b)).sum();
                assert!((err2 - st.err2).abs() <= 1e-9 * st.ref2, "{err2} {}", st.err2);
            }
        }
    }

    #[test]
    fn keeps_exact_what_it_cannot_fit() {
        let raw = families(1000, 16);
        let o = CpcaOptions { clusters: 4, coeffs: 1, rel_mse: 1e-9, force: true, gzip: true };
        let (payload, st) = encode_cpca(&raw, 1000, 8, &o);
        assert!(!st.cpca);
        assert_eq!(decode_cpca(&payload, 1000, 8).unwrap(), raw);
        let zeros = vec![0u8; 64 * 32];
        let (payload, _) = encode_cpca(&zeros, 64, 8, &CpcaOptions::default());
        assert_eq!(decode_cpca(&payload, 64, 8).unwrap(), zeros);
        assert!(decode_cpca(&payload[..3], 64, 8).is_err());
    }

    #[test]
    fn eigen_of_a_known_matrix() {
        let (v, r) = eigen(vec![2.0, 1.0, 1.0, 2.0], 2);
        assert!((v[0] - 3.0).abs() < 1e-12 && (v[1] - 1.0).abs() < 1e-12);
        assert!((r[0].abs() - 0.5f64.sqrt()).abs() < 1e-12);
    }
}
