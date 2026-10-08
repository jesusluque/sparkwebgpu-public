//! The LoD size of each merged node by its error (thread BE).
//!
//! Spark's traversal refines the node of the largest `size / distance`
//! first, under a splat budget, and stops at a pixel (lod_tree.rs). Its
//! `size` is geometric: twice the mean of the node's scales (splat_encode
//! `encode_lod_tree`). A node that looks like its splats (a flat patch of
//! one colour) is refined as soon as one that does not (a crease, a colour
//! edge) of the same extent.
//!
//! Each merged node's error is measured against all the splats under it
//! (`level_errors`): the variance of their linear base colour, of their
//! material (metallic, sqrt roughness, transmission), the spread of their
//! normals (1 - |mean n|), the node's thickness (smallest axis over its
//! long ones: a crease or a curved patch), and Runnalls' bound B / W on
//! KL(splats || node), all by mass w = opacity x two-axis area. Its size is
//! then (`level_lod_sizes`)
//!
//! ```text
//! size = geometric x clamp(((v + v0) / v0)^gamma, 1, hi)
//! v    = lambda var(c) + lambda_m var(m) + lambda_n spread + lambda_t thick
//! ```
//!
//! made monotone (a node at least as large as its children, as Nanite's
//! error): a node that looks like its splats keeps its geometric size, so
//! Spark's one-pixel limit and `lodScale` mean what they did, and one that
//! does not is refined up to `hi` times sooner. Under a splat budget the
//! traversal spends the splats where the error is.
//!
//! Measured on the CPU (`athc_measure trav`, research/simplify-measurements.md
//! § BE): on corvette-v6e-light the relMSE at a budget of 150k / 300k / 600k
//! drops by 14 / 13 / 29 % from the geometric sizes. A size that is the error
//! alone, `k sqrt(A eps)` with eps BC's merge cost (`level_error_sizes`),
//! was twice as bad: B / W grows with the log of the splats merged, so fine
//! nodes shrank three times against coarse ones, and coarse nodes of faint
//! glass, whose error per area is small, stayed as wide glowing blobs.
//!
//! The sizes are written as the v3 `LODS` section (one f32 an element, 0
//! for a splat: geometric) and read by the decoder in place of the
//! geometric size (`SplatReceiver::set_lod_size`).

use crate::athc::{high_half, low_half, unpack_normal, AthcBlock, AthcFile, FLAG_LINEAR};

/// The pure error size `k sqrt(A eps)` (`level_error_sizes`; measured
/// worse than `level_lod_sizes`, kept for the measurements).
#[derive(Clone, Copy, Debug)]
pub struct SizeOptions {
    /// Weight of the shape term (B / W).
    pub shape: f64,
    /// Weight of the base colour's variance (linear rgb).
    pub lambda: f64,
    /// Weight of the material's variance.
    pub lambda_material: f64,
    /// Weight of the normals' spread (1 - |mean n|).
    pub lambda_normal: f64,
    /// The constant k.
    pub scale: f64,
}

impl Default for SizeOptions {
    fn default() -> Self {
        Self { shape: 1.0, lambda: 128.0, lambda_material: 32.0, lambda_normal: 0.0, scale: 1.0 }
    }
}

/// The error-scaled geometric size (`level_lod_sizes`).
#[derive(Clone, Copy, Debug)]
pub struct LodSizeOptions {
    /// Weight of the base colour's variance (linear rgb).
    pub lambda: f64,
    /// Weight of the material's variance.
    pub lambda_material: f64,
    /// Weight of the normals' spread.
    pub lambda_normal: f64,
    /// Weight of the node's thickness.
    pub lambda_thick: f64,
    /// Weight of B / W.
    pub shape: f64,
    /// The error at which the size grows by sqrt(2) (gamma 0.5).
    pub v0: f64,
    pub gamma: f64,
    /// At most this many times the geometric size.
    pub hi: f64,
}

impl Default for LodSizeOptions {
    /// What measured best on corvette-v6e-light (§ BE).
    fn default() -> Self {
        Self { lambda: 128.0, lambda_material: 32.0, lambda_normal: 4.0, lambda_thick: 2.0, shape: 0.0, v0: 0.05, gamma: 0.5, hi: 4.0 }
    }
}

/// Additive moments of a set of splats.
#[derive(Clone, Copy, Debug, Default)]
struct Acc {
    w: f64,
    m1: [f64; 3],
    m2: [f64; 6],
    c: [f64; 3],
    cc: f64,
    mat: [f64; 3],
    mm: f64,
    n: [f64; 3],
    /// sum w ln det(S_i), each floored at 1% of its own mean variance.
    lndet: f64,
}

impl Acc {
    fn add(&mut self, o: &Acc) {
        self.w += o.w;
        for k in 0..3 {
            self.m1[k] += o.m1[k];
            self.c[k] += o.c[k];
            self.mat[k] += o.mat[k];
            self.n[k] += o.n[k];
        }
        for k in 0..6 {
            self.m2[k] += o.m2[k];
        }
        self.cc += o.cc;
        self.mm += o.mm;
        self.lndet += o.lndet;
    }
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn quat_axes(q: [f32; 4]) -> [[f64; 3]; 3] {
    let [x, y, z, w] = q.map(|v| v as f64);
    let n = (x * x + y * y + z * z + w * w).sqrt().max(1e-20);
    let (x, y, z, w) = (x / n, y / n, z / n, w / n);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
        [2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)],
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

fn det_floor(s: &[f64; 6], eps: f64) -> f64 {
    let (a, b, c) = (s[0] + eps, s[1] + eps, s[2] + eps);
    let (d, e, f) = (s[3], s[4], s[5]);
    (a * (b * c - f * f) - d * (d * c - f * e) + e * (d * f - b * e)).max(1e-300)
}

/// The product of a covariance's two largest eigenvalues.
pub fn two_largest_product(s: &[f64; 6]) -> f64 {
    let tr = s[0] + s[1] + s[2];
    let minors = s[0] * s[1] - s[3] * s[3] + s[0] * s[2] - s[4] * s[4] + s[1] * s[2] - s[5] * s[5];
    let det = (s[0] * (s[1] * s[2] - s[5] * s[5]) - s[3] * (s[3] * s[2] - s[5] * s[4]) + s[4] * (s[3] * s[5] - s[1] * s[4]))
        .max(0.0);
    // Smallest root of x^3 - tr x^2 + minors x - det, by Newton from 0.
    let mut x = 0.0f64;
    for _ in 0..40 {
        let f = ((x - tr) * x + minors) * x - det;
        let d = (3.0 * x - 2.0 * tr) * x + minors;
        if d.abs() < 1e-300 {
            break;
        }
        let nx = x - f / d;
        if (nx - x).abs() <= 1e-15 * tr.abs() {
            x = nx;
            break;
        }
        x = nx;
    }
    let small = x.max(0.0);
    (minors - small * (tr - small)).max(0.0)
}

fn leaf_acc(b: &AthcBlock, i: usize, linear: bool) -> Acc {
    let p = &b.positions[i * 4..i * 4 + 4];
    let w4 = &b.shape[i * 4..i * 4 + 4];
    let q = crate::athc::decode_quaternion(w4[0]);
    let s = [low_half(w4[1]).exp(), high_half(w4[1]).exp(), low_half(w4[2]).exp()].map(|v| v as f64);
    let mut sorted = s;
    sorted.sort_by(|a, b| b.total_cmp(a));
    let w = (p[3].max(0.0) as f64 * sorted[0] * sorted[1]).max(1e-30);
    let a = quat_axes(q);
    let mut cov = [0.0f64; 6];
    for k in 0..3 {
        let v = a[k];
        let s2 = s[k] * s[k];
        cov[0] += s2 * v[0] * v[0];
        cov[1] += s2 * v[1] * v[1];
        cov[2] += s2 * v[2] * v[2];
        cov[3] += s2 * v[0] * v[1];
        cov[4] += s2 * v[0] * v[2];
        cov[5] += s2 * v[1] * v[2];
    }
    let x = [p[0] as f64, p[1] as f64, p[2] as f64];
    let mut c = [high_half(w4[2]), low_half(w4[3]), high_half(w4[3])];
    if !linear {
        c = c.map(|v| srgb_to_linear(v.clamp(0.0, 1.0)));
    }
    let c = c.map(|v| v.max(0.0) as f64);
    let n = if b.normals.is_empty() { [0.0; 3] } else { unpack_normal(b.normals[i]).map(|v| v as f64) };
    let mat = if b.pbr.is_empty() {
        [0.0; 3]
    } else {
        let per = b.pbr.len() / b.n;
        let v = b.pbr[i * per];
        [(v & 255) as f64 / 255.0, (((v >> 8) & 255) as f64 / 255.0).sqrt(), ((v >> 16) & 255) as f64 / 255.0]
    };
    let eps = 0.01 * (cov[0] + cov[1] + cov[2]) / 3.0;
    Acc {
        w,
        m1: x.map(|v| w * v),
        m2: [
            w * (cov[0] + x[0] * x[0]),
            w * (cov[1] + x[1] * x[1]),
            w * (cov[2] + x[2] * x[2]),
            w * (cov[3] + x[0] * x[1]),
            w * (cov[4] + x[0] * x[2]),
            w * (cov[5] + x[1] * x[2]),
        ],
        c: c.map(|v| w * v),
        cc: w * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]),
        mat: mat.map(|v| w * v),
        mm: w * (mat[0] * mat[0] + mat[1] * mat[1] + mat[2] * mat[2]),
        n: n.map(|v| w * v),
        lndet: w * det_floor(&cov, eps.max(1e-30)).ln(),
    }
}

/// What a merged node's error is made of (all per unit of mass).
#[derive(Clone, Copy, Debug, Default)]
pub struct NodeError {
    /// Runnalls' bound over the mass, B / W.
    pub shape: f64,
    /// Variance of the linear base colour (summed over rgb).
    pub colour: f64,
    /// Variance of the material.
    pub material: f64,
    /// 1 - |mean normal|.
    pub spread: f64,
    /// The node's thickness: its smallest axis over the mean of its two
    /// long ones (0 flat).
    pub thick: f64,
    /// sqrt of the product of its two long variances.
    pub area: f64,
}

impl NodeError {
    /// eps (see the module).
    pub fn eps(&self, o: &SizeOptions) -> f64 {
        o.shape * self.shape + o.lambda * self.colour + o.lambda_material * self.material + o.lambda_normal * self.spread
    }
}

fn node_error(a: &Acc, normals: bool) -> NodeError {
    let w = a.w.max(1e-30);
    let mu = a.m1.map(|v| v / w);
    let s = [
        a.m2[0] / w - mu[0] * mu[0],
        a.m2[1] / w - mu[1] * mu[1],
        a.m2[2] / w - mu[2] * mu[2],
        a.m2[3] / w - mu[0] * mu[1],
        a.m2[4] / w - mu[0] * mu[2],
        a.m2[5] / w - mu[1] * mu[2],
    ];
    let eps = (0.01 * (s[0] + s[1] + s[2]) / 3.0).max(1e-30);
    let b = (0.5 * (w * det_floor(&s, eps).ln() - a.lndet) / w).max(0.0);
    let cbar = a.c.map(|v| v / w);
    let cvar = (a.cc / w - (cbar[0] * cbar[0] + cbar[1] * cbar[1] + cbar[2] * cbar[2])).max(0.0);
    let mbar = a.mat.map(|v| v / w);
    let mvar = (a.mm / w - (mbar[0] * mbar[0] + mbar[1] * mbar[1] + mbar[2] * mbar[2])).max(0.0);
    let spread = if normals {
        let l = (a.n[0] * a.n[0] + a.n[1] * a.n[1] + a.n[2] * a.n[2]).sqrt() / w;
        (1.0 - l).max(0.0)
    } else {
        0.0
    };
    let p2 = two_largest_product(&s);
    let tr = s[0] + s[1] + s[2];
    let det = (s[0] * (s[1] * s[2] - s[5] * s[5]) - s[3] * (s[3] * s[2] - s[5] * s[4]) + s[4] * (s[3] * s[5] - s[1] * s[4])).max(0.0);
    let small = if p2 > 0.0 { det / p2 } else { 0.0 };
    let long_mean = 0.5 * (tr - small).max(1e-30);
    NodeError { shape: b, colour: cvar, material: mvar, spread, thick: (small / long_mean).sqrt(), area: p2.sqrt() }
}

/// Each level's groups' errors (coarsest level first, as `file.levels`),
/// each against all the splats under it.
pub fn level_errors(file: &AthcFile) -> Vec<Vec<NodeError>> {
    let levels = file.levels.len();
    if levels == 0 {
        return Vec::new();
    }
    let linear = file.header.flags & FLAG_LINEAR != 0;
    let normals = file.chunks.first().is_some_and(|c| !c.normals.is_empty());
    let mut accs: Vec<Vec<Acc>> = file.levels.iter().map(|(_, b)| vec![Acc::default(); b.n]).collect();
    // The finest level: its runs of splats (chunks are the file's order).
    let finest = levels - 1;
    let count = file.header.count as usize;
    let mut g = 0usize;
    let mut at = 0usize;
    for chunk in &file.chunks {
        for i in 0..chunk.n {
            while g + 1 < file.starts.len() && file.starts[g + 1] as usize <= at {
                g += 1;
            }
            if g < accs[finest].len() && at < count {
                let a = leaf_acc(chunk, i, linear);
                accs[finest][g].add(&a);
            }
            at += 1;
        }
    }
    for l in (0..finest).rev() {
        let (parents, children) = (&file.levels[l].1.tail, &file.levels[l + 1].1.tail);
        let mut j = 0;
        for (i, &code) in parents.iter().enumerate() {
            let mut a = Acc::default();
            while j < children.len() && children[j] >> 3 == code {
                a.add(&accs[l + 1][j]);
                j += 1;
            }
            accs[l][i] = a;
        }
    }
    accs.iter().map(|level| level.iter().map(|a| node_error(a, normals)).collect()).collect()
}

/// Makes per-level sizes monotone: a node at least as large as its
/// largest child.
pub fn monotone_sizes(file: &AthcFile, sizes: &mut [Vec<f32>]) {
    let levels = file.levels.len();
    for l in (0..levels.saturating_sub(1)).rev() {
        let (parents, children) = (&file.levels[l].1.tail, &file.levels[l + 1].1.tail);
        let mut j = 0;
        for i in 0..parents.len() {
            let code = parents[i];
            let mut m = sizes[l][i];
            while j < children.len() && children[j] >> 3 == code {
                m = m.max(sizes[l + 1][j]);
                j += 1;
            }
            sizes[l][i] = m;
        }
    }
}

/// Each level's groups' error sizes `k sqrt(A eps)` (coarsest level first,
/// as `file.levels`), monotone up the tree.
pub fn level_error_sizes(file: &AthcFile, o: &SizeOptions) -> Vec<Vec<f32>> {
    let errs = level_errors(file);
    let mut sizes: Vec<Vec<f32>> =
        errs.iter().map(|l| l.iter().map(|e| (o.scale * (e.area * e.eps(o)).sqrt()) as f32).collect()).collect();
    monotone_sizes(file, &mut sizes);
    sizes
}

impl NodeError {
    /// The factor `level_lod_sizes` scales the geometric size by.
    pub fn size_factor(&self, o: &LodSizeOptions) -> f64 {
        let v = o.lambda * self.colour
            + o.lambda_material * self.material
            + o.lambda_normal * self.spread
            + o.lambda_thick * self.thick
            + o.shape * self.shape;
        ((v + o.v0) / o.v0).powf(o.gamma).clamp(1.0, o.hi)
    }
}

/// Each level's groups' geometric sizes as the decoder makes them: the
/// group widened to its cell (`athc::widen_merged`), its opacity as stored.
pub fn level_geometric_sizes(file: &AthcFile) -> Vec<Vec<f32>> {
    let extent = file.header.extent;
    file.levels
        .iter()
        .map(|(level, block)| {
            let mut b = block.clone();
            crate::athc::widen_merged(&mut b, extent / (1u64 << *level) as f32);
            (0..b.n)
                .map(|i| {
                    let w = &b.shape[i * 4..i * 4 + 4];
                    let s = [low_half(w[1]).exp(), high_half(w[1]).exp(), low_half(w[2]).exp()];
                    geometric_size(s, b.positions[i * 4 + 3])
                })
                .collect()
        })
        .collect()
}

/// Each level's groups' LoD sizes (coarsest level first, as `file.levels`):
/// the geometric size scaled up by the error, monotone up the tree.
pub fn level_lod_sizes(file: &AthcFile, o: &LodSizeOptions) -> Vec<Vec<f32>> {
    let errs = level_errors(file);
    let mut sizes = level_geometric_sizes(file);
    for (l, e) in sizes.iter_mut().zip(&errs) {
        for (s, e) in l.iter_mut().zip(e) {
            *s = (*s as f64 * e.size_factor(o)) as f32;
        }
    }
    monotone_sizes(file, &mut sizes);
    sizes
}

/// The file with every level carrying its LoD sizes (`level_lod_sizes`):
/// written as the v3 `LODS` section.
pub fn with_lod_sizes(file: &AthcFile, o: &LodSizeOptions) -> AthcFile {
    let sizes = level_lod_sizes(file, o);
    let mut out = file.clone();
    for ((_, b), s) in out.levels.iter_mut().zip(sizes) {
        b.lod_size = s;
    }
    for c in out.chunks.iter_mut() {
        c.lod_size.clear();
    }
    out
}

/// Spark's geometric size of an element (splat_encode `encode_lod_tree`):
/// twice the mean of its scales (of its two for a surfel), grown for a LoD
/// opacity past 1.
pub fn geometric_size(scale: [f32; 3], opacity: f32) -> f32 {
    let avg = if scale[0].min(scale[1]).min(scale[2]) <= 0.0 {
        (scale[0] + scale[1] + scale[2]) / 2.0
    } else {
        (scale[0] + scale[1] + scale[2]) / 3.0
    };
    let o = crate::athc::spark_lod_opacity(opacity);
    let expansion = if o <= 1.0 { 1.0 } else { 1.0 + 0.7 * (o * 4.0 - 3.0 - 1.0) };
    2.0 * expansion * avg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::athc::{merged_pages_of, AthcDecoder, VirtualTree};
    use crate::athc_v3::{read_v3, write_v3, COMPRESSION_GZIP};
    use crate::decoder::{ChunkReceiver, SplatProps, SplatReceiver};

    const TWO_CARDS: &[u8] = include_bytes!("../../../test/fixtures/athc/two_cards.athc");

    #[derive(Default)]
    struct Sizes {
        sizes: Vec<(usize, Vec<f32>)>,
        n: usize,
    }

    impl SplatReceiver for Sizes {
        fn set_lod_size(&mut self, base: usize, count: usize, size: &[f32]) {
            self.sizes.push((base, size[..count].to_vec()));
        }
        fn set_batch(&mut self, base: usize, count: usize, _: &SplatProps) {
            self.n = self.n.max(base + count);
        }
        fn set_center(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_opacity(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_rgb(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_rgba(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_scale(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_quat(&mut self, _: usize, _: usize, _: &[f32]) {}
    }

    fn decode(bytes: &[u8]) -> Sizes {
        let mut d = AthcDecoder::new(Sizes::default());
        d.push(bytes).unwrap();
        ChunkReceiver::finish(&mut d).unwrap();
        d.into_splats()
    }

    #[test]
    fn lod_sizes_are_geometric_or_more_and_monotone() {
        let f = AthcFile::read(TWO_CARDS).unwrap();
        let o = LodSizeOptions::default();
        let geo = level_geometric_sizes(&f);
        let sizes = level_lod_sizes(&f, &o);
        for (g, s) in geo.iter().zip(&sizes) {
            for (a, b) in g.iter().zip(s) {
                assert!(*b >= *a * 0.999 && *b > 0.0, "{b} under its geometric {a}");
            }
        }
        for l in 0..f.levels.len() - 1 {
            let (parents, children) = (&f.levels[l].1.tail, &f.levels[l + 1].1.tail);
            for (j, c) in children.iter().enumerate() {
                let i = parents.binary_search(&(c >> 3)).unwrap();
                assert!(sizes[l][i] >= sizes[l + 1][j]);
            }
        }
    }

    #[test]
    fn lod_sizes_go_through_v3_the_decoder_and_merged_pages() {
        let f = AthcFile::read(TWO_CARDS).unwrap();
        let g = with_lod_sizes(&f, &LodSizeOptions::default());
        let bytes = write_v3(&g, COMPRESSION_GZIP).unwrap();
        let back = read_v3(&bytes).unwrap();
        assert!(back.header.has(crate::athc::FLAG_LOD_SIZE));
        for ((_, a), (_, b)) in back.levels.iter().zip(&g.levels) {
            assert_eq!(a.lod_size, b.lod_size);
        }
        assert!(back.chunks.iter().all(|c| c.lod_size.is_empty()));
        // The v2 writer drops them; the same file without them reads as before.
        let v2 = AthcFile::read(&g.write().unwrap()).unwrap();
        assert!(!v2.header.has(crate::athc::FLAG_LOD_SIZE) && v2.levels.iter().all(|(_, b)| b.lod_size.is_empty()));
        assert!(decode(&write_v3(&f, COMPRESSION_GZIP).unwrap()).sizes.is_empty());

        // A whole file: the merged nodes' sizes after the root.
        let tree = VirtualTree::of_file(&g, false).unwrap();
        let whole = decode(&bytes);
        let top = g.levels[0].1.lod_size.iter().cloned().fold(0.0, f32::max);
        let mut expect: Vec<f32> = if tree.synth_root { vec![top] } else { Vec::new() };
        for (_, b) in &g.levels {
            expect.extend_from_slice(&b.lod_size);
        }
        assert_eq!(whole.sizes, vec![(0, expect.clone())]);

        // Paged: every merged page carries its nodes' sizes.
        let mut head = g.header.to_bytes().to_vec();
        head.extend_from_slice(&g.extra.to_bytes());
        let mut hv = g.header;
        hv.flags |= crate::athc::FLAG_LOD_SIZE;
        head[..crate::athc::HEADER_BYTES].copy_from_slice(&hv.to_bytes());
        let (_, pages) = merged_pages_of(&g, &head).unwrap();
        let mut got = Vec::new();
        for p in &pages {
            let s = decode(p);
            assert_eq!(s.sizes.len(), 1);
            got.extend_from_slice(&s.sizes[0].1);
        }
        assert_eq!(got, expect);
    }
}
