//! An error-driven LoD tree for a `.athc`, in place of the octree's fixed
//! cells (thread BC; research/simplify-measurements.md § 5 and "BC").
//!
//! The octree merges whatever falls in a cell: on the Corvette's body half
//! the weight of a mid level mixed normals more than 30 degrees apart and a
//! tenth mixed materials. Here gaussians are merged by pairs, cheapest first
//! (NanoGS; athenea's `decimate`, surfels-web/PLAN.md § 1):
//!
//! * a cluster is its raw moments, weight W = opacity x the area of its two
//!   longest axes (the LoD's mass), so any merge order is exact moment
//!   matching;
//! * candidates are each cluster's nearest neighbours along three shifted
//!   Morton orders of the centres; a pair merges when each is the other's
//!   cheapest, at most half of those pairs a pass, cheapest first;
//! * the cost is an image-space proxy (thread AY's, λ = 128 measured best):
//!   `A_px (B / W + λ w_i w_j / W² |Δc|² + λ_m w_i w_j / W² |Δm|²)`, B
//!   Runnalls' bound on KL(mixture || merged) with the covariances floored
//!   at 1% of the merged mean variance (surfels stay finite), A_px the
//!   merged weight in pixels from the nearest point of an orbit (what no one
//!   sees large merges freely), c the linear base colour, m the material
//!   (metallic, √roughness, transmission);
//! * never merged (until a level cannot reach its count otherwise): across
//!   a key (glass, thin wall, Schlick metal, mirror vs rough, the skin's
//!   dominant joint) or normals more than 60 degrees apart.
//!
//! The binary merges are then written as the `.athc`'s own levels, without
//! any change of format: level r's groups are the codes' top 3r bits
//! (`athc_build::build_lod_from_codes`), so each level is made from the next
//! finer one by pair merges that never put more than eight of its groups in
//! one (three bits), about four into one on average (a surface octree's
//! ratio), and every group's attributes are merged once from its own splats
//! by the builder (moments, harmonics, transfer and curvature in the world
//! frame, lobes, shadow bits, skin) -- a group's splats are one run of the
//! file's order, the tree's depth-first order. The header's `extent` is set
//! so that the decoder's widening (`MERGED_FILL` x extent / 2^r) is that of
//! a cell the size of the level's typical group.
//!
//! A cut (`error_cut`, `athc-convert --keep-splats N --error`) merges the
//! splats down to N clusters by the same cost, with no limit of children:
//! those clusters are the cut cloud's splats (their whole coverage as the
//! LoD opacity, as `truncate_levels` keeps it), and the levels above them
//! are built as above.

use anyhow::{bail, Result};

use crate::athc::{coverage_ratios, high_half, low_half, unpack_normal, AthcBlock, AthcFile, FLAG_LINEAR};
use crate::athc_build::{build_lod_from_codes, packed_of, BuildOptions, PackedCloud, LOD_LEVELS};

/// Where the merges are seen from: an orbit at `eye_dist` around `center`,
/// `fx` pixels per unit of tangent; nothing nearer than `near`.
#[derive(Clone, Copy, Debug)]
pub struct ErrorView {
    pub center: [f64; 3],
    pub eye_dist: f64,
    pub fx: f64,
    pub near: f64,
}

impl ErrorView {
    /// An orbit at `dist` from `center`, `hfov_deg` over `width` pixels.
    pub fn orbit(center: [f64; 3], dist: f64, hfov_deg: f64, width: f64) -> Self {
        let fx = 0.5 * width / (0.5 * hfov_deg.to_radians()).tan();
        Self { center, eye_dist: dist, fx, near: 0.1 * dist }
    }

    /// An orbit at three times the bounds' radius, 39.6 degrees over 1920
    /// pixels (athenea's 50 mm camera).
    pub fn around(lo: [f32; 3], hi: [f32; 3]) -> Self {
        let c = [0, 1, 2].map(|k| 0.5 * (lo[k] as f64 + hi[k] as f64));
        let r = 0.5 * ((0..3).map(|k| (hi[k] as f64 - lo[k] as f64).powi(2)).sum::<f64>()).sqrt();
        Self::orbit(c, 3.0 * r.max(1e-6), 39.6, 1920.0)
    }

    /// Pixels a unit area at `p` covers seen from the nearest point of the
    /// orbit.
    fn pixels(&self, p: [f64; 3]) -> f64 {
        let r = ((p[0] - self.center[0]).powi(2) + (p[1] - self.center[1]).powi(2) + (p[2] - self.center[2]).powi(2)).sqrt();
        let z = (self.eye_dist - r).max(self.near);
        (self.fx / z).powi(2)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ErrorOptions {
    pub view: ErrorView,
    /// Weight of the base colour difference (linear rgb).
    pub lambda: f64,
    /// Weight of the material difference (metallic, √roughness, transmission).
    pub lambda_material: f64,
    /// Weight of the normal difference (|n_i - n_j|²).
    pub lambda_normal: f64,
    /// Merges whose normals are less than this cosine apart are refused
    /// (0.5: 60 degrees).
    pub max_angle_cos: f64,
    /// Groups of a level over groups of the next finer one.
    pub level_ratio: f64,
    /// Neighbours each side along each Morton order.
    pub neighbours: usize,
    /// The coarsest level holds at most this many groups (the root merges
    /// them; codes leave 5 bits for them at ten levels).
    pub top: usize,
    /// Worker threads (1 on the web).
    pub threads: usize,
    /// A cut's clusters widened by this fraction of their own cell's edge
    /// (in quadrature, as `widen_merged`); 0 leaves the moments as they are.
    pub widen: f32,
}

impl ErrorOptions {
    pub fn new(view: ErrorView) -> Self {
        Self {
            view,
            lambda: 128.0,
            lambda_material: 32.0,
            lambda_normal: 0.0,
            max_angle_cos: 0.5,
            level_ratio: 4.0,
            neighbours: 6,
            top: 32,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()).min(12),
            widen: 0.0,
        }
    }
}

/// A cluster: raw moments of its members by weight.
#[derive(Clone, Copy, Debug, Default)]
struct Cl {
    w: f64,
    m1: [f64; 3],
    m2: [f64; 6],
    c: [f64; 3],
    n: [f64; 3],
    mat: [f64; 3],
    key: u64,
    /// Members from the level below (at most `children` in a level step).
    kids: u32,
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
    // Columns: the gaussian's axes.
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
        [2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)],
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// The merge key of element i: glass, thin wall, Schlick, mirror, and the
/// skin's heaviest joint.
fn key_of(b: &AthcBlock, i: usize, skin_per: usize, influences: usize) -> u64 {
    let mut key = 0u64;
    if !b.pbr.is_empty() {
        let per = b.pbr.len() / b.n;
        let w = b.pbr[i * per];
        let glass = ((w >> 16) & 255) > 127;
        let thin = (w >> 24) & 1 == 1;
        let schlick = (w >> 25) & 1 == 1;
        let mirror = ((w >> 8) & 255) < 26;
        key |= glass as u64 | (thin as u64) << 1 | (schlick as u64) << 2 | (mirror as u64) << 3;
    }
    if skin_per > 0 && !b.skin.is_empty() {
        let mut best = (0u32, -1.0f32);
        for k in 0..influences {
            let (joint, w) = crate::athc_skin::unpack_influence(b.skin[i * skin_per + k]);
            if w > best.1 {
                best = (joint, w);
            }
        }
        key |= ((best.0 as u64) + 1) << 8;
    }
    key
}

fn clusters_of(b: &AthcBlock, linear: bool, influences: usize, gradient_words: usize) -> Vec<Cl> {
    let skin_per = influences + gradient_words;
    (0..b.n)
        .map(|i| {
            let p = &b.positions[i * 4..i * 4 + 4];
            let w4 = &b.shape[i * 4..i * 4 + 4];
            let q = crate::athc::decode_quaternion(w4[0]);
            let s = [low_half(w4[1]).exp(), high_half(w4[1]).exp(), low_half(w4[2]).exp()].map(|v| v as f64);
            let mut sorted = s;
            sorted.sort_by(|a, b| b.total_cmp(a));
            // Opacity past 1 is a merged group's coverage: its mass too.
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
            let m2 = [
                w * (cov[0] + x[0] * x[0]),
                w * (cov[1] + x[1] * x[1]),
                w * (cov[2] + x[2] * x[2]),
                w * (cov[3] + x[0] * x[1]),
                w * (cov[4] + x[0] * x[2]),
                w * (cov[5] + x[1] * x[2]),
            ];
            let mut c = [high_half(w4[2]), low_half(w4[3]), high_half(w4[3])];
            if !linear {
                c = c.map(|v| srgb_to_linear(v.clamp(0.0, 1.0)));
            }
            let n = if b.normals.is_empty() { [0.0; 3] } else { unpack_normal(b.normals[i]) };
            let mat = if b.pbr.is_empty() {
                [0.0; 3]
            } else {
                let per = b.pbr.len() / b.n;
                let v = b.pbr[i * per];
                [(v & 255) as f64 / 255.0, (((v >> 8) & 255) as f64 / 255.0).sqrt(), ((v >> 16) & 255) as f64 / 255.0]
            };
            Cl {
                w,
                m1: x.map(|v| w * v),
                m2,
                c: c.map(|v| w * v.max(0.0) as f64),
                n: n.map(|v| w * v as f64),
                mat: mat.map(|v| w * v),
                key: key_of(b, i, skin_per, influences),
                kids: 1,
            }
        })
        .collect()
}

fn add(a: &Cl, b: &Cl) -> Cl {
    let mut r = *a;
    r.w += b.w;
    for k in 0..3 {
        r.m1[k] += b.m1[k];
        r.c[k] += b.c[k];
        r.n[k] += b.n[k];
        r.mat[k] += b.mat[k];
    }
    for k in 0..6 {
        r.m2[k] += b.m2[k];
    }
    r.kids += b.kids;
    r
}

fn mean_cov(c: &Cl) -> ([f64; 3], [f64; 6]) {
    let mu = c.m1.map(|v| v / c.w);
    (
        mu,
        [
            c.m2[0] / c.w - mu[0] * mu[0],
            c.m2[1] / c.w - mu[1] * mu[1],
            c.m2[2] / c.w - mu[2] * mu[2],
            c.m2[3] / c.w - mu[0] * mu[1],
            c.m2[4] / c.w - mu[0] * mu[2],
            c.m2[5] / c.w - mu[1] * mu[2],
        ],
    )
}

fn det_floor(s: &[f64; 6], eps: f64) -> f64 {
    let (a, b, c) = (s[0] + eps, s[1] + eps, s[2] + eps);
    let (d, e, f) = (s[3], s[4], s[5]);
    (a * (b * c - f * f) - d * (d * c - f * e) + e * (d * f - b * e)).max(1e-300)
}

fn len3(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// How strictly a pass refuses merges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Rules {
    /// Keys and normal angles refused.
    Strict,
    /// Keys refused, any angle.
    Keys,
    /// Anything (a coarse level that must reach its count).
    Free,
}

fn cost(a: &Cl, b: &Cl, o: &ErrorOptions, rules: Rules, cap: u32) -> f64 {
    if a.kids + b.kids > cap {
        return f64::INFINITY;
    }
    if rules <= Rules::Keys && a.key != b.key {
        return f64::INFINITY;
    }
    let (na, nb) = (len3(&a.n), len3(&b.n));
    let unit = |v: &[f64; 3], l: f64| if l > 0.0 { v.map(|x| x / l) } else { [0.0; 3] };
    let (ua, ub) = (unit(&a.n, na), unit(&b.n, nb));
    let cos = ua[0] * ub[0] + ua[1] * ub[1] + ua[2] * ub[2];
    if rules == Rules::Strict && na > 0.0 && nb > 0.0 && cos < o.max_angle_cos {
        return f64::INFINITY;
    }
    let m = add(a, b);
    let (mu, s) = mean_cov(&m);
    let (_, sa) = mean_cov(a);
    let (_, sb) = mean_cov(b);
    let eps = 0.01 * (s[0] + s[1] + s[2]) / 3.0;
    let bk = 0.5 * (m.w * det_floor(&s, eps).ln() - a.w * det_floor(&sa, eps).ln() - b.w * det_floor(&sb, eps).ln()) / m.w;
    let share = a.w * b.w / (m.w * m.w);
    let d2 = |x: &[f64; 3], y: &[f64; 3]| {
        let (u, v) = (x.map(|t| t / a.w), y.map(|t| t / b.w));
        (u[0] - v[0]).powi(2) + (u[1] - v[1]).powi(2) + (u[2] - v[2]).powi(2)
    };
    let col = d2(&a.c, &b.c);
    let mat = d2(&a.mat, &b.mat);
    let nrm = if na > 0.0 && nb > 0.0 { 2.0 - 2.0 * cos } else { 0.0 };
    // Merging across a key (only at a coarse level that must) costs as a
    // full colour flip would.
    let keyed = if a.key != b.key { 1.0 } else { 0.0 };
    let apx = m.w * o.view.pixels(mu);
    apx * (bk.max(0.0) + share * (o.lambda * (col + keyed) + o.lambda_material * mat + o.lambda_normal * nrm))
}

/// 21 bits an axis of a point in [lo, lo + ext)^3, interleaved.
fn morton63(p: [f64; 3], lo: [f64; 3], ext: f64, shift: [f64; 3]) -> u64 {
    let mut code = 0u64;
    let q = [0, 1, 2].map(|k| (((p[k] - lo[k] + shift[k]) / ext).clamp(0.0, 0.999_999) * 2_097_152.0) as u64);
    for bit in 0..21 {
        for (k, v) in q.iter().enumerate() {
            code |= ((v >> bit) & 1) << (3 * bit + k);
        }
    }
    code
}

fn par_map<T: Send>(n: usize, threads: usize, f: impl Fn(std::ops::Range<usize>) -> Vec<T> + Sync) -> Vec<T> {
    if threads <= 1 || n < 4096 || cfg!(target_arch = "wasm32") {
        return f(0..n);
    }
    let step = n.div_ceil(threads);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads).map(|k| (k * step).min(n)..((k + 1) * step).min(n)).map(|r| s.spawn(|| f(r))).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

/// Clusters and, for each, what it holds (indices into the level below).
struct Pool {
    cls: Vec<Cl>,
    members: Vec<Vec<u32>>,
    /// Each cluster's node in `tree`, when the merges are recorded.
    node: Vec<u32>,
    tree: Option<MergeTree>,
}

impl Pool {
    fn new(cls: Vec<Cl>, record: bool) -> Self {
        let n = cls.len();
        Self {
            cls,
            members: (0..n as u32).map(|i| vec![i]).collect(),
            node: if record { (0..n as u32).collect() } else { Vec::new() },
            tree: record.then(|| MergeTree::leaves(n)),
        }
    }
}

/// A binary merge tree in athenea's layout (surfels-web/NOTES.md, "Merge
/// tree"): nodes `0 .. leaves` are the input gaussians, then the merges in
/// the order they were made (children always before parents); `parent` per
/// node, a root its own index; `cost` the merge that made the node (0 for a
/// leaf), not monotone -- cut by the max over a node's subtree
/// (`monotone_costs`). athenea's file stores its leaves depth first; here
/// leaf k is the input's gaussian k (`dfs_leaves` gives that order), so a
/// tree athenea wrote is read by `from_parts` with its own leaf order as the
/// input's.
#[derive(Clone, Debug, Default)]
pub struct MergeTree {
    pub leaves: usize,
    pub parent: Vec<u32>,
    pub cost: Vec<f32>,
}

impl MergeTree {
    fn leaves(n: usize) -> Self {
        Self { leaves: n, parent: (0..n as u32).collect(), cost: vec![0.0; n] }
    }

    /// athenea's `lodParent` / `lodCost` over `leaves` leaves.
    pub fn from_parts(leaves: usize, parent: &[i32], cost: &[f32]) -> Result<Self> {
        if parent.len() != cost.len() || parent.len() < leaves {
            bail!("a merge tree of {} parents, {} costs over {} leaves", parent.len(), cost.len(), leaves);
        }
        for (k, &p) in parent.iter().enumerate() {
            if p < 0 || p as usize >= parent.len() || (p as usize != k && (p as usize) < leaves.max(k + 1)) {
                bail!("merge tree node {k}: parent {p} (children come before parents)");
            }
        }
        Ok(Self { leaves, parent: parent.iter().map(|&p| p as u32).collect(), cost: cost.to_vec() })
    }

    fn merge(&mut self, a: u32, b: u32, cost: f64) -> u32 {
        let k = self.parent.len() as u32;
        self.parent[a as usize] = k;
        self.parent[b as usize] = k;
        self.parent.push(k);
        self.cost.push(cost as f32);
        k
    }

    /// Each node's cost made monotone: the max over its subtree.
    pub fn monotone_costs(&self) -> Vec<f32> {
        let mut m = self.cost.clone();
        for k in 0..m.len() {
            let p = self.parent[k] as usize;
            if p != k {
                m[p] = m[p].max(m[k]);
            }
        }
        m
    }

    /// The cut with at most `keep` clusters (at least as many as roots) by
    /// monotone cost: each cluster its leaves.
    pub fn cut(&self, keep: usize) -> Vec<Vec<u32>> {
        let mc = self.monotone_costs();
        let total = self.parent.len();
        let mut kids = vec![0u32; total];
        for k in 0..total {
            let p = self.parent[k] as usize;
            if p != k {
                kids[p] += 1;
            }
        }
        // Merges cheapest first, each taking its children's count down.
        let mut merges: Vec<u32> = (self.leaves as u32..total as u32).collect();
        merges.sort_by(|&a, &b| mc[a as usize].total_cmp(&mc[b as usize]).then(a.cmp(&b)));
        let mut count = self.leaves;
        let mut taken = vec![false; total];
        for &k in &merges {
            if count <= keep {
                break;
            }
            taken[k as usize] = true;
            count -= kids[k as usize] as usize - 1;
        }
        // A taken node's children are taken (monotone costs; ties by index,
        // children first): each node's cluster is its highest taken ancestor.
        let mut cluster = vec![u32::MAX; total];
        let mut out: Vec<Vec<u32>> = Vec::new();
        let mut id = vec![u32::MAX; total];
        for k in (0..total).rev() {
            let p = self.parent[k] as usize;
            let top = if p != k && taken[p] { cluster[p] } else { k as u32 };
            cluster[k] = top;
            if k < self.leaves {
                let t = top as usize;
                if id[t] == u32::MAX {
                    id[t] = out.len() as u32;
                    out.push(Vec::new());
                }
                out[id[t] as usize].push(k as u32);
            }
        }
        for c in out.iter_mut() {
            c.reverse();
        }
        out
    }

    /// The leaves depth first (athenea's file order).
    pub fn dfs_leaves(&self) -> Vec<u32> {
        let total = self.parent.len();
        let mut children: Vec<Vec<u32>> = vec![Vec::new(); total];
        let mut roots = Vec::new();
        for k in 0..total {
            let p = self.parent[k] as usize;
            if p == k {
                roots.push(k as u32);
            } else {
                children[p].push(k as u32);
            }
        }
        let mut out = Vec::with_capacity(self.leaves);
        let mut stack: Vec<u32> = roots.into_iter().rev().collect();
        while let Some(k) = stack.pop() {
            if (k as usize) < self.leaves {
                out.push(k);
            }
            stack.extend(children[k as usize].iter().rev());
        }
        out
    }
}

/// What a run of `greedy` did.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct MergeRun {
    pub from: usize,
    pub to: usize,
    pub passes: usize,
    /// The dearest merge done.
    pub last_cost: f64,
    /// Merges done past the strict rules (keys or angles relaxed).
    pub relaxed: usize,
    /// Clusters left after each pass and the dearest merge so far.
    pub curve: Vec<(usize, f64)>,
}

/// Pairs merged cheapest first until `target` clusters are left or no pair
/// is allowed (`cap` members of the level below a cluster at most).
fn greedy(pool: &mut Pool, target: usize, o: &ErrorOptions, rules: Rules, cap: u32, bounds: ([f64; 3], f64), run: &mut MergeRun) {
    greedy_to(pool, target, f64::INFINITY, o, rules, cap, bounds, run)
}

/// [`greedy`], merging nothing dearer than `max_cost`.
#[allow(clippy::too_many_arguments)]
fn greedy_to(pool: &mut Pool, target: usize, max_cost: f64, o: &ErrorOptions, rules: Rules, cap: u32, bounds: ([f64; 3], f64), run: &mut MergeRun) {
    let (lo, ext) = bounds;
    let shifts = [[0.0, 0.0, 0.0], [ext * 0.0123, ext * 0.0071, ext * 0.0093], [ext * 0.0031, ext * 0.0157, ext * 0.0047]];
    let k = o.neighbours.max(1);
    let mut slow = 0;
    while pool.cls.len() > target {
        let alive = pool.cls.len();
        let cls = &pool.cls;
        let mut best: Vec<(f64, u32)> = vec![(f64::INFINITY, u32::MAX); alive];
        for sh in &shifts {
            let mut order: Vec<(u64, u32)> =
                par_map(alive, o.threads, |r| r.map(|i| (morton63(cls[i].m1.map(|v| v / cls[i].w), lo, ext, *sh), i as u32)).collect());
            order.sort_unstable();
            let found: Vec<(f64, u32)> = par_map(alive, o.threads, |r| {
                r.map(|pos| {
                    let i = order[pos].1 as usize;
                    let mut b = (f64::INFINITY, u32::MAX);
                    for d in 1..=k {
                        for q in [pos.wrapping_sub(d), pos + d] {
                            if q < alive {
                                let j = order[q].1 as usize;
                                let c = cost(&cls[i], &cls[j], o, rules, cap);
                                if c < b.0 || (c == b.0 && (j as u32) < b.1) {
                                    b = (c, j as u32);
                                }
                            }
                        }
                    }
                    b
                })
                .collect()
            });
            for (pos, b) in found.into_iter().enumerate() {
                let i = order[pos].1 as usize;
                if b.0 < best[i].0 || (b.0 == best[i].0 && b.1 < best[i].1) {
                    best[i] = b;
                }
            }
        }
        let mut pairs: Vec<(f64, u32, u32)> = Vec::new();
        for (i, &(c, j)) in best.iter().enumerate() {
            if j != u32::MAX && c.is_finite() && (i as u32) < j && best[j as usize].1 == i as u32 {
                pairs.push((c, i as u32, j));
            }
        }
        if pairs.is_empty() {
            break;
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let need = alive - target;
        let cheap = pairs.partition_point(|p| p.0 <= max_cost);
        let take = need.min(pairs.len().div_ceil(2)).min(cheap);
        if take == 0 {
            break;
        }
        let mut dead = vec![false; alive];
        for &(c, i, j) in &pairs[..take] {
            let (i, j) = (i as usize, j as usize);
            if rules != Rules::Strict {
                let strict = cost(&pool.cls[i], &pool.cls[j], o, Rules::Strict, cap);
                if !strict.is_finite() {
                    run.relaxed += 1;
                }
            }
            pool.cls[i] = add(&pool.cls[i], &pool.cls[j]);
            if let Some(t) = pool.tree.as_mut() {
                pool.node[i] = t.merge(pool.node[i], pool.node[j], c);
            }
            let moved = std::mem::take(&mut pool.members[j]);
            if moved.len() > pool.members[i].len() {
                let kept = std::mem::replace(&mut pool.members[i], moved);
                pool.members[i].extend(kept);
            } else {
                pool.members[i].extend(moved);
            }
            dead[j] = true;
            run.last_cost = run.last_cost.max(c);
        }
        let mut at = 0;
        pool.cls.retain(|_| {
            at += 1;
            !dead[at - 1]
        });
        at = 0;
        pool.members.retain(|_| {
            at += 1;
            !dead[at - 1]
        });
        if !pool.node.is_empty() {
            at = 0;
            pool.node.retain(|_| {
                at += 1;
                !dead[at - 1]
            });
        }
        run.passes += 1;
        run.curve.push((pool.cls.len(), run.last_cost));
        // A tail of passes that each merge almost nothing: stop.
        if take * 2000 < alive {
            slow += 1;
            if slow >= 4 {
                break;
            }
        } else {
            slow = 0;
        }
    }
}

fn bounds_of(cls: &[Cl]) -> ([f64; 3], f64) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for c in cls {
        for k in 0..3 {
            let v = c.m1[k] / c.w;
            lo[k] = lo[k].min(v);
            hi[k] = hi[k].max(v);
        }
    }
    let ext = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max) * 1.1 + 1e-9;
    (lo, ext)
}

/// One level step: the clusters of `pool` merged into about `1 / ratio` as
/// many, at most `cap` a group; the rules relaxed only if the count would
/// stay over `must`.
fn level_step(pool: &mut Pool, o: &ErrorOptions, cap: u32, must: usize) -> MergeRun {
    let from = pool.cls.len();
    let target = ((from as f64 / o.level_ratio).ceil() as usize).max(1);
    let bounds = bounds_of(&pool.cls);
    let mut run = MergeRun { from, ..Default::default() };
    for rules in [Rules::Strict, Rules::Keys, Rules::Free] {
        // Relaxed only as the levels above need, or when nothing merged.
        if rules != Rules::Strict && pool.cls.len() <= must && pool.cls.len() < from {
            break;
        }
        // Relaxed, only as far as the levels above need.
        let goal = if rules == Rules::Strict || pool.cls.len() == from { target } else { target.max(must) };
        greedy(pool, goal, o, rules, cap, bounds, &mut run);
        if pool.cls.len() <= target {
            break;
        }
    }
    run.to = pool.cls.len();
    run
}

/// An error-driven hierarchy: levels finest first, each node its members in
/// the level below (the first level's in the elements).
pub struct Hierarchy {
    /// Per level, each group's members (indices into the finer level, or the
    /// base elements for level 0).
    pub levels: Vec<Vec<Vec<u32>>>,
    /// Each level's groups' centroids and long-axis edge estimate.
    centroids: Vec<Vec<[f64; 3]>>,
    edges: Vec<Vec<f64>>,
    pub runs: Vec<MergeRun>,
}

fn edge_of(c: &Cl) -> f64 {
    // A flat cell of edge L filled evenly has moments L / sqrt(12) along
    // both its long axes: L = sqrt(12 sigma1 sigma2).
    let (_, s) = mean_cov(c);
    let m = glam::DMat3::from_cols_array(&[s[0], s[3], s[4], s[3], s[1], s[5], s[4], s[5], s[2]]);
    // Two largest eigenvalues: their product is the sum of the principal
    // 2x2 minors less what the smallest takes (det / smallest).
    let tr = s[0] + s[1] + s[2];
    let minors = s[0] * s[1] - s[3] * s[3] + s[0] * s[2] - s[4] * s[4] + s[1] * s[2] - s[5] * s[5];
    let det = m.determinant().max(0.0);
    // Smallest root of x^3 - tr x^2 + minors x - det by Newton from 0.
    let mut x = 0.0f64;
    for _ in 0..30 {
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
    let prod = (minors - small * (tr - small)).max(0.0);
    (12.0 * prod.sqrt()).sqrt()
}

/// Builds the levels above `base` (each level step at most eight of the
/// finer level's groups a group), finest first, until at most `o.top`
/// groups or `max_levels` levels. `first` is an already made first level
/// (a cut's clusters), whose members are base elements.
fn hierarchy(base: Vec<Cl>, first: Option<Pool>, o: &ErrorOptions, max_levels: usize) -> Result<Hierarchy> {
    let mut levels = Vec::new();
    let mut centroids = Vec::new();
    let mut edges = Vec::new();
    let mut runs = Vec::new();
    let mut pool = match first {
        Some(p) => p,
        None => {
            let n = base.len();
            let _ = n;
            let mut p = Pool::new(base, false);
            for c in p.cls.iter_mut() {
                c.kids = 1;
            }
            let left = max_levels - 1;
            let must = must_reach(o, left);
            runs.push(level_step(&mut p, o, 8, must));
            p
        }
    };
    loop {
        // `pool` is a level: record it.
        centroids.push(pool.cls.iter().map(|c| c.m1.map(|v| v / c.w)).collect());
        edges.push(pool.cls.iter().map(edge_of).collect());
        levels.push(std::mem::take(&mut pool.members));
        if pool.cls.len() <= o.top && levels.len() >= 2 {
            break;
        }
        if levels.len() >= max_levels {
            if pool.cls.len() > o.top {
                bail!("{} groups left at the coarsest of {} levels (at most {})", pool.cls.len(), max_levels, o.top);
            }
            break;
        }
        // The next level, from this one's groups.
        let n = pool.cls.len();
        pool.members = (0..n as u32).map(|i| vec![i]).collect();
        for c in pool.cls.iter_mut() {
            c.kids = 1;
        }
        let left = max_levels - levels.len() - 1;
        let must = must_reach(o, left);
        let run = level_step(&mut pool, o, 8, must);
        if run.to == run.from {
            bail!("a level step merged nothing ({} groups)", run.from);
        }
        runs.push(run);
    }
    Ok(Hierarchy { levels, centroids, edges, runs })
}

/// The most groups a level may keep with `left` levels to go above it.
fn must_reach(o: &ErrorOptions, left: usize) -> usize {
    (o.top as f64 * 3f64.powi(left as i32)).min(1e15) as usize
}

/// Codes for the base elements: the path of digits from the coarsest
/// level down, children ordered along a Morton curve of their centroids
/// (so the file's order, depth first, keeps neighbours together).
fn codes_of(h: &Hierarchy, base_len: usize) -> Result<Vec<u32>> {
    let d = h.levels.len();
    let top = h.levels[d - 1].len();
    let top_bits = 32 - 3 * (d as u32 - 1);
    if d as u32 > LOD_LEVELS || (top_bits < 32 && top as u64 > 1u64 << top_bits) {
        bail!("{} levels with {} groups at the top do not fit 32-bit codes", d, top);
    }
    // Morton keys of every level's centroids over the whole bounds.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for c in &h.centroids[0] {
        for k in 0..3 {
            lo[k] = lo[k].min(c[k]);
            hi[k] = hi[k].max(c[k]);
        }
    }
    let ext = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max) * 1.01 + 1e-9;
    let mort = |p: &[f64; 3]| morton63(*p, lo, ext, [0.0; 3]);
    // code[level][group], coarsest first.
    let mut code: Vec<Vec<u32>> = h.levels.iter().map(|l| vec![0u32; l.len()]).collect();
    let mut tops: Vec<u32> = (0..top as u32).collect();
    tops.sort_by_key(|&g| mort(&h.centroids[d - 1][g as usize]));
    for (k, &g) in tops.iter().enumerate() {
        code[d - 1][g as usize] = k as u32;
    }
    for l in (1..d).rev() {
        for g in 0..h.levels[l].len() {
            let mut kids = h.levels[l][g].clone();
            if kids.len() > 8 {
                bail!("a group of level {} has {} children", l, kids.len());
            }
            kids.sort_by_key(|&c| mort(&h.centroids[l - 1][c as usize]));
            for (k, &c) in kids.iter().enumerate() {
                code[l - 1][c as usize] = (code[l][g] << 3) | k as u32;
            }
        }
    }
    let mut out = vec![u32::MAX; base_len];
    for (g, members) in h.levels[0].iter().enumerate() {
        for &e in members {
            out[e as usize] = code[0][g];
        }
    }
    if out.contains(&u32::MAX) {
        bail!("an element is in no group");
    }
    Ok(out)
}

/// The header extent whose cells (extent / 2^r, the decoder's widening)
/// match each level's typical group: the geometric mean over the levels of
/// median edge x 2^r.
fn extent_for(h: &Hierarchy, coarsest: u32) -> f32 {
    let d = h.levels.len();
    let mut sum = 0.0;
    let mut count = 0;
    for l in 0..d {
        let r = coarsest + (d - 1 - l) as u32;
        let mut e: Vec<f64> = h.edges[l].iter().cloned().filter(|v| *v > 0.0).collect();
        if e.is_empty() {
            continue;
        }
        e.sort_by(|a, b| a.total_cmp(b));
        let med = e[e.len() / 2];
        sum += (med * (1u64 << r) as f64).ln();
        count += 1;
    }
    if count == 0 {
        1.0
    } else {
        (sum / count as f64).exp() as f32
    }
}

/// What `error_levels` / `error_cut` made.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct ErrorTree {
    /// Groups a level, coarsest first (a cut's finest level is its splats).
    pub groups: Vec<usize>,
    pub runs: Vec<MergeRun>,
    /// The cut's own merges (splats to clusters), when cut.
    pub cut: Option<MergeRun>,
    pub extent: f32,
    /// The original splats (file order) each output splat stands for.
    #[serde(skip)]
    pub sources: Vec<Vec<u32>>,
}

fn options_for(file: &AthcFile, coarsest: u32) -> BuildOptions {
    BuildOptions {
        coarsest_level: coarsest,
        max_group_fraction: 2.0,
        chunk_splats: file.header.chunk_splats,
        ..Default::default()
    }
}

/// The file with its levels built again as an error-driven tree over the
/// same splats (reordered depth first; `sources` maps each back).
pub fn error_levels(file: &AthcFile, o: &ErrorOptions) -> Result<(AthcFile, ErrorTree)> {
    let cloud = packed_of(file);
    let linear = file.header.flags & FLAG_LINEAR != 0;
    let base = clusters_of(&cloud.block, linear, cloud.skin_influences as usize, cloud.skin_gradient_words as usize);
    let n = base.len();
    let h = hierarchy(base, None, o, LOD_LEVELS as usize)?;
    let codes = codes_of(&h, n)?;
    let d = h.levels.len() as u32;
    let coarsest = LOD_LEVELS - d + 1;
    let extent = extent_for(&h, coarsest);
    let mut out = build_lod_from_codes(&cloud, &options_for(file, coarsest), &codes, file.header.bounds_lo, extent)?;
    crate::athc::uncap_levels(&mut out);
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&i| codes[i as usize]);
    let tree = ErrorTree {
        groups: out.levels.iter().map(|(_, b)| b.n).collect(),
        runs: h.runs,
        cut: None,
        extent,
        sources: order.iter().map(|&i| vec![i]).collect(),
    };
    Ok((out, tree))
}

/// The cloud merged down to `keep` splats by the error (each the merge of
/// the original splats it stands for, every section with it), with an
/// error-driven tree of levels above them.
pub fn error_cut(file: &AthcFile, keep: usize, o: &ErrorOptions) -> Result<(AthcFile, ErrorTree)> {
    error_cut_to(file, keep, f64::INFINITY, o)
}

/// [`error_cut`], merging nothing dearer than `max_cost` (the same cost
/// over a scene's parts, when they are seen from the same orbit, spends the
/// splats where the error is): `keep` is then a floor.
pub fn error_cut_to(file: &AthcFile, keep: usize, max_cost: f64, o: &ErrorOptions) -> Result<(AthcFile, ErrorTree)> {
    let cloud = packed_of(file);
    let linear = file.header.flags & FLAG_LINEAR != 0;
    let base = clusters_of(&cloud.block, linear, cloud.skin_influences as usize, cloud.skin_gradient_words as usize);
    let n = base.len();
    if keep == 0 || keep >= n {
        bail!("cut {} splats to {} (keep fewer than all, at least one)", n, keep);
    }
    if keep < 2 * o.top {
        bail!("cut to at least {} splats (the levels above need them)", 2 * o.top);
    }
    // The cut: splats merged to `keep` clusters, no limit of members.
    let bounds = bounds_of(&base);
    let mut pool = Pool::new(base.clone(), true);
    let mut run = MergeRun { from: n, ..Default::default() };
    for rules in [Rules::Strict, Rules::Keys] {
        greedy_to(&mut pool, keep, max_cost, o, rules, u32::MAX, bounds, &mut run);
        if pool.cls.len() <= keep {
            break;
        }
    }
    run.to = pool.cls.len();
    let tree = pool.tree.take().expect("recorded");
    let (out, mut t) = error_cut_tree(file, &tree, keep.max(run.to), o)?;
    t.cut = Some(run);
    Ok((out, t))
}

/// The cut of a merge tree (ours, or athenea's `--tree`, `MergeTree::
/// from_parts`) to `keep` clusters by monotone cost, written as
/// [`error_cut`] writes its own.
pub fn error_cut_tree(file: &AthcFile, tree: &MergeTree, keep: usize, o: &ErrorOptions) -> Result<(AthcFile, ErrorTree)> {
    let cloud = packed_of(file);
    let linear = file.header.flags & FLAG_LINEAR != 0;
    let base = clusters_of(&cloud.block, linear, cloud.skin_influences as usize, cloud.skin_gradient_words as usize);
    let n = base.len();
    if tree.leaves != n {
        bail!("a merge tree over {} leaves for {} splats", tree.leaves, n);
    }
    let members = tree.cut(keep);
    let cls: Vec<Cl> = members
        .iter()
        .map(|m| {
            let mut c = base[m[0] as usize];
            for &e in &m[1..] {
                c = add(&c, &base[e as usize]);
            }
            c
        })
        .collect();
    let clusters = cls.clone();
    let pool = Pool { cls, members, node: Vec::new(), tree: None };
    let h = hierarchy(base, Some(pool), o, LOD_LEVELS as usize)?;
    let codes = codes_of(&h, n)?;
    let d = h.levels.len() as u32;
    let coarsest = LOD_LEVELS - d + 1;
    let extent = extent_for(&h, coarsest);
    let full = build_lod_from_codes(&cloud, &options_for(file, coarsest), &codes, file.header.bounds_lo, extent)?;
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&i| codes[i as usize]);
    // The finest level (the clusters, in code order) becomes the splats.
    let ratios = coverage_ratios(&full);
    let levels = full.levels.len();
    let mut splats = full.levels[levels - 1].1.clone();
    for (i, r) in ratios[levels - 1].iter().enumerate() {
        splats.positions[i * 4 + 3] = *r;
    }
    let mut kept = full.clone();
    for ((_, b), r) in kept.levels.iter_mut().zip(&ratios) {
        for (i, v) in r.iter().enumerate() {
            b.positions[i * 4 + 3] = *v;
        }
    }
    // The clusters' geometry from their f64 moments (the builder sums f32
    // moments of absolute positions: a merged splat's thin axis would be
    // lost in their rounding), each axis's scale replaced where the frame
    // agrees.
    crate::athc::orient_merged(&mut splats);
    refine_scales(&mut splats, &h.levels[0], &clusters, &codes);
    if o.widen > 0.0 {
        widen_own(&mut splats, o.widen);
    }
    // Sources: the cluster each output splat is (the level's groups are in
    // code order, as the splats under them).
    let mut sources = Vec::with_capacity(splats.n);
    let mut at = 0usize;
    while at < n {
        let c = codes[order[at] as usize];
        let mut end = at;
        while end < n && codes[order[end] as usize] == c {
            end += 1;
        }
        sources.push(order[at..end].to_vec());
        at = end;
    }
    if sources.len() != splats.n {
        bail!("{} clusters for {} merged splats", sources.len(), splats.n);
    }
    // Tails: the parent group at the next coarser level.
    let parents = &kept.levels[levels - 2].1.tail;
    let mut starts = Vec::with_capacity(parents.len());
    let mut j = 0;
    let cells = splats.tail.clone();
    for &code in parents {
        starts.push(j as u32);
        while j < splats.n && cells[j] >> 3 == code {
            splats.tail[j] = starts.len() as u32 - 1;
            j += 1;
        }
    }
    if j != splats.n {
        bail!("the cut has clusters without a parent");
    }
    let per = file.header.chunk_splats.max(1) as usize;
    let chunks: Vec<AthcBlock> = (0..splats.n).step_by(per).map(|s| splats.slice(s, per.min(splats.n - s))).collect();
    let mut out = AthcFile { header: kept.header, extra: kept.extra, levels: kept.levels[..levels - 1].to_vec(), starts, chunks };
    out.header.count = splats.n as u32;
    out.header.chunks = out.chunks.len() as u32;
    out.header.levels = (levels - 1) as u32;
    out.header.finest_groups = out.levels[levels - 2].1.n as u32;
    let tree = ErrorTree {
        groups: out.levels.iter().map(|(_, b)| b.n).collect::<Vec<_>>().into_iter().chain([splats.n]).collect(),
        runs: h.runs,
        cut: None,
        extent,
        sources,
    };
    Ok((out, tree))
}

/// Each merged splat's scales from its cluster's f64 covariance along the
/// splat's own (oriented) axes, its opacity rescaled to keep the mass.
fn refine_scales(splats: &mut AthcBlock, groups: &[Vec<u32>], clusters: &[Cl], codes: &[u32]) {
    // Group g of the level (in code order) is the cluster whose members'
    // code it is.
    let mut by_code: Vec<(u32, usize)> = groups.iter().enumerate().map(|(g, m)| (codes[m[0] as usize], g)).collect();
    by_code.sort();
    for (i, &(_, g)) in by_code.iter().enumerate() {
        if i >= splats.n {
            break;
        }
        let c = &clusters[g];
        let (_, s) = mean_cov(c);
        let w = &mut splats.shape[i * 4..i * 4 + 4];
        let a = quat_axes(crate::athc::decode_quaternion(w[0]));
        let mut sc = [0.0f64; 3];
        for k in 0..3 {
            let v = a[k];
            let var = v[0] * (s[0] * v[0] + s[3] * v[1] + s[4] * v[2])
                + v[1] * (s[3] * v[0] + s[1] * v[1] + s[5] * v[2])
                + v[2] * (s[4] * v[0] + s[5] * v[1] + s[2] * v[2]);
            sc[k] = var.max(1e-14).sqrt();
        }
        let old = [low_half(w[1]).exp(), high_half(w[1]).exp(), low_half(w[2]).exp()].map(|v| v as f64);
        let area = |s: [f64; 3]| {
            let mut t = s;
            t.sort_by(|a, b| b.total_cmp(a));
            t[0] * t[1]
        };
        let ratio = area(old) / area(sc).max(1e-30);
        w[1] = crate::athc::pack_halves(crate::athc::ln_scale(sc[0] as f32), crate::athc::ln_scale(sc[1] as f32));
        w[2] = crate::athc::pack_halves(crate::athc::ln_scale(sc[2] as f32), high_half(w[2]));
        let p = &mut splats.positions[i * 4 + 3];
        *p = (*p as f64 * ratio) as f32;
    }
}

/// Each element's two long axes grown by `fill` x its own cell edge in
/// quadrature (`edge_of`), the opacity divided by the area gained.
fn widen_own(b: &mut AthcBlock, fill: f32) {
    for i in 0..b.n {
        let w = &mut b.shape[i * 4..i * 4 + 4];
        let mut s = [low_half(w[1]).exp(), high_half(w[1]).exp(), low_half(w[2]).exp()];
        let mut ord = [0usize, 1, 2];
        ord.sort_by(|&x, &y| s[y].total_cmp(&s[x]));
        let before = s[ord[0]] * s[ord[1]];
        let edge = (12.0 * before).sqrt();
        let grow = (fill * edge).powi(2);
        for &k in &ord[..2] {
            s[k] = (s[k] * s[k] + grow).sqrt();
        }
        let after = s[ord[0]] * s[ord[1]];
        w[1] = crate::athc::pack_halves(crate::athc::ln_scale(s[0]), crate::athc::ln_scale(s[1]));
        w[2] = crate::athc::pack_halves(crate::athc::ln_scale(s[2]), high_half(w[2]));
        b.positions[i * 4 + 3] *= before / after.max(1e-30);
    }
}

/// The packed cloud an `.athc`'s splats are (for callers that build).
pub fn cloud_of(file: &AthcFile) -> PackedCloud {
    packed_of(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::athc::VirtualTree;
    use crate::athc_build::{build_lod, pack_streams, CloudStreams};

    /// A folded sheet: a 60 x 60 grid of discs on the floor (normal +z,
    /// red) and another up a wall (normal -y, blue), meeting at y = 0.
    fn folded() -> AthcFile {
        let mut s = CloudStreams { coefficients: 1, linear: true, ..Default::default() };
        let step = 0.01f32;
        for face in 0..2 {
            for a in 0..60 {
                for b in 0..60 {
                    let (u, v) = (a as f32 * step, (b as f32 + 0.5) * step);
                    if face == 0 {
                        s.positions.extend_from_slice(&[u, v, 0.0]);
                        s.rotations.extend_from_slice(&[0.0, 0.0, 0.0, 1.0]);
                        s.normals.extend_from_slice(&[0.0, 0.0, 1.0]);
                        s.sh.extend_from_slice(&[1.5, -1.5, -1.5]);
                    } else {
                        s.positions.extend_from_slice(&[u, 0.0, v]);
                        s.rotations.extend_from_slice(&[std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2]);
                        s.normals.extend_from_slice(&[0.0, -1.0, 0.0]);
                        s.sh.extend_from_slice(&[-1.5, -1.5, 1.5]);
                    }
                    s.scales.extend_from_slice(&[1.2 * step, 1.2 * step, 0.1 * step]);
                    s.opacities.push(1.0);
                    s.count += 1;
                }
            }
        }
        let p = pack_streams(&s, &BuildOptions::default()).unwrap();
        build_lod(&p, &BuildOptions::default()).unwrap()
    }

    fn options(file: &AthcFile) -> ErrorOptions {
        let mut o = ErrorOptions::new(ErrorView::around(file.header.bounds_min, file.header.bounds_max));
        o.threads = 1;
        o
    }

    fn face_of(file: &AthcFile, i: u32) -> bool {
        unpack_normal(file.splats().normals[i as usize])[2] > 0.5
    }

    #[test]
    fn cuts_by_error_without_crossing_the_fold() {
        let file = folded();
        let (cut, tree) = error_cut(&file, 1200, &options(&file)).unwrap();
        assert_eq!(cut.header.count, 1200);
        // Every original splat in exactly one cluster, no cluster on both faces.
        let mut seen = vec![0u32; file.header.count as usize];
        for list in &tree.sources {
            let face = face_of(&file, list[0]);
            for &i in list {
                seen[i as usize] += 1;
                assert_eq!(face_of(&file, i), face, "a cluster straddles the fold");
            }
        }
        assert!(seen.iter().all(|&k| k == 1));
        // A tree Spark can page, eight children at most, through v3 and back.
        let v = VirtualTree::of_file(&cut, true).unwrap();
        assert!(v.child_count.iter().take(v.merged as usize).all(|&c| (1..=8).contains(&c) || v.synth_root));
        let bytes = crate::athc_v3::write_v3_full(&cut, crate::athc_v3::COMPRESSION_NONE, false, &|_| 0, None).unwrap();
        let back = crate::athc_v3::read_v3(&bytes).unwrap();
        assert_eq!(back.header.count, 1200);
        // The merged splats keep the mass of what they stand for (coverage).
        let w0: f64 = crate::athl::splat_weights(&file).iter().map(|&w| w as f64).sum();
        let w1: f64 = crate::athl::splat_weights(&cut).iter().map(|&w| w as f64).sum();
        assert!((w1 / w0 - 1.0).abs() < 0.02, "mass {w0} -> {w1}");
    }

    #[test]
    fn levels_by_error_keep_every_splat() {
        let file = folded();
        let (out, tree) = error_levels(&file, &options(&file)).unwrap();
        assert_eq!(out.header.count, file.header.count);
        let mut order: Vec<u32> = tree.sources.iter().map(|l| l[0]).collect();
        order.sort();
        assert!(order.iter().enumerate().all(|(k, &i)| k as u32 == i));
        let v = VirtualTree::of_file(&out, true).unwrap();
        let first_merged = if v.synth_root { 1 } else { 0 };
        for k in first_merged..v.level_base[v.level_base.len() - 1] as usize {
            assert!((1..=8).contains(&v.child_count[k]), "node {k}: {} children", v.child_count[k]);
        }
        // The finest groups stay on one face.
        let n = out.header.count as usize;
        for g in 0..out.starts.len() {
            let end = out.starts.get(g + 1).map_or(n, |&s| s as usize);
            let faces: Vec<bool> = (out.starts[g] as usize..end).map(|k| face_of(&file, tree.sources[k][0])).collect();
            assert!(faces.iter().all(|&f| f == faces[0]));
        }
    }

    #[test]
    fn cuts_a_merge_tree_by_its_monotone_costs() {
        // leaves 0..4; 4 = (0, 1) at 1.0; 5 = (2, 3) at 3.0; 6 = (4, 5) at 2.0
        // (cheaper than its child: athenea's costs are not monotone).
        let t = MergeTree::from_parts(4, &[4, 4, 5, 5, 6, 6, 6], &[0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(t.monotone_costs()[6], 3.0);
        let mut c = t.cut(3);
        c.sort();
        assert_eq!(c, vec![vec![0, 1], vec![2], vec![3]]);
        assert_eq!(t.cut(2).len(), 2);
        assert_eq!(t.cut(1), vec![vec![0, 1, 2, 3]]);
        assert_eq!(t.dfs_leaves(), vec![0, 1, 2, 3]);
        assert!(MergeTree::from_parts(4, &[4, 4, 5, 5, 6, 6, 3], &[0.0; 7]).is_err());
    }
}
