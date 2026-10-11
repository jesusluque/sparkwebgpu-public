//! The LoD error cut (thread CG): each merged node of a `.athc` carries a
//! monotone error and the radius of the sphere round its leaves, so that a
//! run-time traversal can draw the cut by projected error, as Nanite does
//! and as athenea's `lod_tree_cut.slang` (`treeErrorOwn` / `treeErrorUp` /
//! `treeErrorKeep`, research 100) defines it:
//!
//! ```text
//! eps(n) = max(r(n) |cost(n)|, max over children eps(c))     a leaf: 0
//! rho(n) = max(2 max sigma(n), max over children |mu(n) - mu(c)| + rho(c))
//! P(n)   = eps(n) ppu / max(d(n) - rho(n), near)              (pixels)
//! drawn  : P(n) <= tau < P(parent)
//! ```
//!
//! `r(n)` is the node's size, `sqrt(tr Sigma)` = `|scale|` of its own
//! gaussian; `ppu` the pixels a unit covers at a distance of one; `d(n)` the
//! eye's distance to the node's centre; `near` 1e-3. `eps` is non-decreasing
//! up the tree and each sphere holds its children's, so `d - rho` is
//! non-increasing up the tree: `P` is monotone and the cut is consistent (each
//! leaf drawn once, by itself or by one ancestor) and needs no walk up.
//!
//! `cost` is athenea's `lodCost` (`usd-athc --tree`: her merge tree, every
//! node of it, `athenea_tree_errors`), or, for a tree of ours (the octree,
//! BC's error levels, a cut), `athc_merge::group_cost` of the node's children
//! in the `.athc` tree (`file_errors`): BC's merge cost without its on-screen
//! weight, over any number of children.
//!
//! Stored per merged node as two f16 in one word (eps low, rho high), each
//! rounded up, in the v3 section `LODE` (flag 128, docs/docs/athc-v3.md) and in
//! merged ATHV pages (`ATHV_LOD_ERRORS`). A splat (a leaf) has none (0).

use half::f16;

use crate::athc::{decode_quaternion, high_half, is_flat_ln, low_half, unpack_normal, AthcBlock};

/// Words a merged node: eps|rho, then theta|normal.
pub const LOD_ERROR_WORDS: usize = 2;

/// athenea's `near` in `projectedError`.
pub const NEAR: f32 = 1.0e-3;

/// theta for a node that is never culled (glass, a thin wall, a double-sided
/// piece, no normals): f16 infinity.
pub const NO_CONE: u16 = 0x7c00;

/// The smallest f16 at or above `v` (v >= 0): rounding up keeps a parent's
/// stored bound at or above its child's.
pub fn f16_up(v: f32) -> f16 {
    if v.is_nan() || v == f32::INFINITY {
        return f16::INFINITY;
    }
    let v = v.max(0.0);
    let h = f16::from_f32(v);
    if h.to_f32() >= v {
        h
    } else {
        f16::from_bits(h.to_bits() + 1)
    }
}

/// One node's eps and rho as a word: f16 eps low, f16 rho high, rounded up.
pub fn pack_error(eps: f32, rho: f32) -> u32 {
    (f16_up(eps).to_bits() as u32) | ((f16_up(rho).to_bits() as u32) << 16)
}

pub fn unpack_error(word: u32) -> (f32, f32) {
    (low_half(word), high_half(word))
}

/// Octahedral 2 x unorm8 (x low): the cone's axis as the traversal reads it.
pub fn pack_normal8(n: [f32; 3]) -> u16 {
    let l1 = n[0].abs() + n[1].abs() + n[2].abs();
    let n = if l1 > 1.0e-20 { n.map(|v| v / l1) } else { [0.0, 0.0, 1.0] };
    let mut p = [n[0], n[1]];
    if n[2] < 0.0 {
        let sx = if n[0] >= 0.0 { 1.0 } else { -1.0 };
        let sy = if n[1] >= 0.0 { 1.0 } else { -1.0 };
        p = [(1.0 - n[1].abs()) * sx, (1.0 - n[0].abs()) * sy];
    }
    let q = p.map(|v| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u16);
    q[0] | (q[1] << 8)
}

pub fn unpack_normal8(bits: u16) -> [f32; 3] {
    let f = [(bits & 255) as f32 / 255.0 * 2.0 - 1.0, (bits >> 8) as f32 / 255.0 * 2.0 - 1.0];
    let mut n = [f[0], f[1], 1.0 - f[0].abs() - f[1].abs()];
    let t = (-n[2]).clamp(0.0, 1.0);
    n[0] += if n[0] >= 0.0 { -t } else { t };
    n[1] += if n[1] >= 0.0 { -t } else { t };
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    n.map(|v| v / l)
}

fn angle(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0).acos()
}

/// The cone word: f16 theta low (rounded up, widened by the axis' 8-bit
/// quantization), the quantized axis high; `NO_CONE` when theta is not
/// finite.
pub fn pack_cone(theta: f32, normal: [f32; 3]) -> u32 {
    if !theta.is_finite() {
        return NO_CONE as u32;
    }
    let q = pack_normal8(normal);
    let widened = theta + angle(normal, unpack_normal8(q)) + 1.0e-4;
    (f16_up(widened).to_bits() as u32) | ((q as u32) << 16)
}

/// (theta, axis); theta infinite: never culled.
pub fn unpack_cone(word: u32) -> (f32, [f32; 3]) {
    (low_half(word), unpack_normal8((word >> 16) as u16))
}

/// Whether a node is wholly back-facing from `eye` (athenea's test,
/// n . v < -sin theta, with the sphere's angular radius added to theta: v
/// differs between the node's leaves by up to asin(rho / d)).
pub fn back_facing(centre: [f32; 3], rho: f32, theta: f32, normal: [f32; 3], eye: [f32; 3]) -> bool {
    let v = [eye[0] - centre[0], eye[1] - centre[1], eye[2] - centre[2]];
    let d = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if !theta.is_finite() || d <= rho {
        return false;
    }
    let spread = theta + (rho / d).asin();
    if spread >= std::f32::consts::FRAC_PI_2 {
        return false;
    }
    (normal[0] * v[0] + normal[1] * v[1] + normal[2] * v[2]) / d < -spread.sin()
}

/// Element i's scales (a surfel's flat axis 0) and centre.
pub fn scales_of(b: &AthcBlock, i: usize) -> [f32; 3] {
    let w = &b.shape[i * 4..i * 4 + 4];
    [low_half(w[1]), high_half(w[1]), low_half(w[2])].map(|l| if is_flat_ln(l) { 0.0 } else { l.exp() })
}

pub fn centre_of(b: &AthcBlock, i: usize) -> [f32; 3] {
    [b.positions[i * 4], b.positions[i * 4 + 1], b.positions[i * 4 + 2]]
}

/// Element i's normal (the normals stream), if the block has one.
pub fn normal_of(b: &AthcBlock, i: usize) -> Option<[f32; 3]> {
    (!b.normals.is_empty()).then(|| unpack_normal(b.normals[i]))
}

/// Its normal, or athenea's fallback without one: the gaussian's third axis.
pub fn facing_of(b: &AthcBlock, i: usize) -> [f32; 3] {
    normal_of(b, i).unwrap_or_else(|| {
        let [x, y, z, w] = decode_quaternion(b.shape[i * 4]);
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)]
    })
}

/// Whether element i may be culled as single-sided: not glass, not a thin
/// wall (both are seen from either side).
pub fn one_sided(b: &AthcBlock, i: usize) -> bool {
    if b.pbr.is_empty() {
        return true;
    }
    let w = b.pbr[i * (b.pbr.len() / b.n)];
    (w >> 24) & 1 == 0 && ((w >> 16) & 255) <= 127
}

/// Element i's base colour as stored, at least 0 (lod_blob's `blobOf`:
/// 0.5 + SH0 dc), and roughness (pbr byte 1; 0.5 without, as athenea's
/// Decimate.cpp fills `roughness` when the cloud has none).
pub fn look_of(b: &AthcBlock, i: usize) -> ([f32; 3], f32) {
    let w = &b.shape[i * 4..i * 4 + 4];
    let rough = if b.pbr.is_empty() { 0.5 } else { ((b.pbr[i * (b.pbr.len() / b.n)] >> 8) & 255) as f32 / 255.0 };
    ([high_half(w[2]).max(0.0), low_half(w[3]).max(0.0), high_half(w[3]).max(0.0)], rough)
}

/// athenea's node size `r`: `sqrt(dot(scale, scale))`, `sqrt(tr Sigma)`.
pub fn size_of(s: [f32; 3]) -> f32 {
    (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt()
}

/// athenea's own sphere: twice the widest axis.
pub fn own_radius(s: [f32; 3]) -> f32 {
    2.0 * s[0].max(s[1]).max(s[2])
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// athenea's bounded error weights (`treeErrorBounded`): e = r (|cost| +
/// colour s_c + normal theta^2 / alpha), a merge past the guards (cost < 0)
/// max(guard e, floor_size r). The default (0, 0, 1, 0) is `treeErrorOwn`.
#[derive(Clone, Copy, Debug)]
pub struct ErrorWeights {
    pub colour: f32,
    pub normal: f32,
    pub guard: f32,
    pub floor_size: f32,
}

impl Default for ErrorWeights {
    fn default() -> Self {
        Self { colour: 0.0, normal: 0.0, guard: 1.0, floor_size: 0.0 }
    }
}

impl ErrorWeights {
    /// athenea's research prototype on the GPU (Decimate.cpp, the cut by
    /// error with `--cut-bounds`): colour 10, normal 1, guard 4, floor 2.
    /// The default for her trees (`usd-athc --tree --lod-errors`).
    pub fn athenea() -> Self {
        Self { colour: 10.0, normal: 1.0, guard: 4.0, floor_size: 2.0 }
    }

    /// `k=v,...` over `self` ("own": treeErrorOwn's weights, 0 0 1 0).
    pub fn parse_over(self, spec: &str) -> anyhow::Result<Self> {
        let mut w = self;
        for kv in spec.split(',').filter(|s| !s.is_empty()) {
            if kv == "own" {
                w = Self::default();
                continue;
            }
            let (k, v) = kv.split_once('=').ok_or_else(|| anyhow::anyhow!("--lod-error-weights k=v,..."))?;
            let v: f32 = v.parse()?;
            match k {
                "colour" | "color" => w.colour = v,
                "normal" => w.normal = v,
                "guard" => w.guard = v,
                "floor" => w.floor_size = v,
                _ => anyhow::bail!("--lod-error-weights: unknown {k}"),
            }
        }
        Ok(w)
    }

    pub fn parse(spec: &str) -> anyhow::Result<Self> {
        let mut w = Self::default();
        for kv in spec.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').ok_or_else(|| anyhow::anyhow!("--lod-error-weights k=v,..."))?;
            let v: f32 = v.parse()?;
            match k {
                "colour" | "color" => w.colour = v,
                "normal" => w.normal = v,
                "guard" => w.guard = v,
                "floor" => w.floor_size = v,
                _ => anyhow::bail!("--lod-error-weights: unknown {k}"),
            }
        }
        Ok(w)
    }

    fn own(&self, size: f32, cost: f32, spread: f32, theta: f32, rough: f32) -> f32 {
        let alpha = rough.max(0.02).powi(2);
        let e = size * (cost.abs() + self.colour * spread + self.normal * theta * theta / alpha);
        if cost < 0.0 { (self.guard * e).max(self.floor_size * size) } else { e }
    }
}

/// The bounds of one node as the climb raises them.
#[derive(Clone, Copy, Debug)]
pub struct NodeBounds {
    pub eps: f32,
    pub rho: f32,
    /// The colour spread s_c and normal cone theta over its leaves.
    pub spread: f32,
    pub theta: f32,
    /// Whether every leaf under it may be culled from behind: a normal,
    /// neither glass nor a thin wall, not a double-sided piece.
    pub cull: bool,
    pub centre: [f32; 3],
    pub colour: [f32; 3],
    pub normal: [f32; 3],
}

impl NodeBounds {
    /// A node before the climb: its own sphere, no spread, no cone.
    pub fn own(b: &AthcBlock, i: usize, double_sided: bool) -> Self {
        let s = scales_of(b, i);
        Self {
            eps: 0.0,
            rho: own_radius(s),
            spread: 0.0,
            theta: 0.0,
            cull: !double_sided && !b.normals.is_empty() && one_sided(b, i),
            centre: centre_of(b, i),
            colour: look_of(b, i).0,
            normal: facing_of(b, i),
        }
    }

    /// Raises this (a parent's) bounds over a child's (treeErrorUp and
    /// treeBoundsUp).
    pub fn cover(&mut self, c: &NodeBounds) {
        self.eps = self.eps.max(c.eps);
        self.rho = self.rho.max(dist(c.centre, self.centre) + c.rho);
        self.spread = self.spread.max(c.spread + dist(c.colour, self.colour));
        self.theta = self.theta.max(c.theta + angle(c.normal, self.normal));
        self.cull &= c.cull;
    }

    pub fn words(&self) -> [u32; 2] {
        [pack_error(self.eps, self.rho), if self.cull { pack_cone(self.theta, self.normal) } else { NO_CONE as u32 }]
    }
}

/// The decoder's synthetic root over level 1 (`athc::merge_root`): no merge
/// cost of its own (it is refined whenever its children's errors are seen),
/// its sphere round level 1's; never culled (its normal is no one's).
pub fn root_error(root: &AthcBlock, level1: &AthcBlock) -> Vec<u32> {
    let c = centre_of(root, 0);
    let mut eps = 0.0f32;
    let mut rho = own_radius(scales_of(root, 0));
    for i in 0..level1.n {
        let (e, r) = unpack_error(level1.lod_error[i * LOD_ERROR_WORDS]);
        eps = eps.max(e);
        rho = rho.max(dist(c, centre_of(level1, i)) + r);
    }
    vec![pack_error(eps, rho), NO_CONE as u32]
}

/// athenea's bounds over her whole merge tree (`lod_tree_cut.slang`:
/// treeErrorOwn, treeBoundsUp, treeErrorBounded, treeErrorUp to a fixed
/// point): `block` holds every node in tree order, children before parents,
/// `parent` a root its own index, nodes `0 .. leaves` the leaves. With
/// children before parents one pass in index order is the fixed point.
pub fn athenea_tree_errors(
    block: &AthcBlock,
    parent: &[u32],
    leaves: usize,
    cost: &[f32],
    weights: &ErrorWeights,
    double_sided: bool,
    closed: Option<&[bool]>,
) -> Vec<NodeBounds> {
    let n = parent.len();
    // `closed`: each node's own mark (athenea:splat:closed); a node is
    // culled only when all its leaves are (the climb ANDs it).
    let mut nb: Vec<NodeBounds> = (0..n)
        .map(|k| {
            let mut b = NodeBounds::own(block, k, double_sided);
            b.cull &= closed.is_none_or(|c| c[k]);
            b
        })
        .collect();
    // The spread, cone and sphere first (treeBoundsUp, with the sphere).
    for k in 0..n {
        let p = parent[k] as usize;
        if p == k || p >= n {
            continue;
        }
        let c = nb[k];
        let q = &mut nb[p];
        q.rho = q.rho.max(dist(c.centre, q.centre) + c.rho);
        q.spread = q.spread.max(c.spread + dist(c.colour, q.colour));
        q.theta = q.theta.max(c.theta + angle(c.normal, q.normal));
        q.cull &= c.cull;
    }
    // Each merge's own error with its bounds, then the climb.
    for k in leaves..n {
        let s = scales_of(block, k);
        let rough = look_of(block, k).1;
        nb[k].eps = weights.own(size_of(s), cost[k], nb[k].spread, nb[k].theta, rough);
    }
    for k in 0..n {
        let p = parent[k] as usize;
        if p != k && p < n {
            nb[p].eps = nb[p].eps.max(nb[k].eps);
        }
    }
    nb
}

#[cfg(feature = "athc")]
pub use build::*;

#[cfg(feature = "athc")]
mod build {
    use super::*;
    use crate::athc::{AthcFile, FLAG_LINEAR};
    use crate::athc_merge::{clusters_of, group_cost, ErrorOptions, ErrorView};
    use anyhow::{bail, Result};

    /// Each level's children in the `.athc` tree: level l's group i holds
    /// `kids[l][i]`, a range of level l + 1's groups, or of the splats (the
    /// file's order, chunks one after the other) for the finest level.
    pub fn level_children(file: &AthcFile) -> Result<Vec<Vec<(u32, u32)>>> {
        let levels = file.levels.len();
        let count: u32 = file.chunks.iter().map(|c| c.n as u32).sum();
        let mut out = Vec::with_capacity(levels);
        for l in 0..levels {
            let parents = &file.levels[l].1.tail;
            let mut kids = Vec::with_capacity(parents.len());
            if l + 1 < levels {
                let children = &file.levels[l + 1].1.tail;
                let mut j = 0usize;
                for &code in parents {
                    let first = j;
                    while j < children.len() && children[j] >> 3 == code {
                        j += 1;
                    }
                    if j == first {
                        bail!(".athc level {l}: cell {code} has no children");
                    }
                    kids.push((first as u32, j as u32));
                }
            } else {
                if file.starts.len() != parents.len() {
                    bail!(".athc: {} starts for {} finest groups", file.starts.len(), parents.len());
                }
                for g in 0..parents.len() {
                    let end = if g + 1 < parents.len() { file.starts[g + 1] } else { count };
                    kids.push((file.starts[g], end));
                }
            }
            out.push(kids);
        }
        Ok(out)
    }

    /// The splats of a file as one block.
    pub fn splats_of(file: &AthcFile) -> AthcBlock {
        let mut all = AthcBlock::default();
        for c in &file.chunks {
            let mut c = c.clone();
            c.lod_size.clear();
            c.lod_error.clear();
            all.append(&c);
        }
        all
    }

    /// What a conversion knows of a level group: athenea's node bounds
    /// (`athenea_tree_errors`), or nothing (computed here).
    pub type Known = Vec<Vec<Option<NodeBounds>>>;

    /// Every merged node's bounds, a level at a time (coarsest first, as
    /// `file.levels`), packed (`LOD_ERROR_WORDS` a node): from the splats up,
    /// each node's own error from `group_cost` of its children in the `.athc`
    /// tree, bounded as athenea's (`ErrorWeights`) -- or, where `known` gives
    /// it (athenea's node), hers, raised to its children's if need be.
    pub fn file_errors(file: &AthcFile, known: Option<&Known>, weights: &ErrorWeights, double_sided: bool) -> Result<Vec<Vec<u32>>> {
        let kids = level_children(file)?;
        let o = ErrorOptions::new(ErrorView::around(file.header.bounds_min, file.header.bounds_max));
        let linear = file.header.has(FLAG_LINEAR);
        let (inf, grad) = (file.extra.skin_influences as usize, file.extra.skin_gradient_words as usize);
        let splats = splats_of(file);
        let mut below_cl = clusters_of(&splats, linear, inf, grad);
        let mut below: Vec<NodeBounds> = (0..splats.n).map(|i| NodeBounds::own(&splats, i, double_sided)).collect();
        drop(splats);
        let mut out = vec![Vec::new(); file.levels.len()];
        for l in (0..file.levels.len()).rev() {
            let b = &file.levels[l].1;
            let cl = clusters_of(b, linear, inf, grad);
            let mut here = Vec::with_capacity(b.n);
            for i in 0..b.n {
                let (a, z) = kids[l][i];
                let mut nb = match known.and_then(|k| k[l][i]) {
                    Some(v) => v,
                    None => {
                        let mut nb = NodeBounds::own(b, i, double_sided);
                        for k in a..z {
                            let c = below[k as usize];
                            let e = nb.eps;
                            nb.cover(&c);
                            nb.eps = e;
                        }
                        let ks: Vec<_> = (a..z).map(|k| &below_cl[k as usize]).collect();
                        let rough = look_of(b, i).1;
                        nb.eps = weights.own(size_of(scales_of(b, i)), group_cost(&ks, &o) as f32, nb.spread, nb.theta, rough);
                        nb
                    }
                };
                for k in a..z {
                    nb.cover(&below[k as usize]);
                }
                here.push(nb);
            }
            out[l] = here.iter().flat_map(|nb| nb.words()).collect();
            below_cl = cl;
            below = here;
        }
        Ok(out)
    }

    /// The file with every level carrying its errors (`file_errors`).
    pub fn with_lod_errors(file: &AthcFile, known: Option<&Known>, weights: &ErrorWeights, double_sided: bool) -> Result<AthcFile> {
        let errors = file_errors(file, known, weights, double_sided)?;
        let mut out = file.clone();
        for ((_, b), e) in out.levels.iter_mut().zip(errors) {
            b.lod_error = e;
        }
        for c in out.chunks.iter_mut() {
            c.lod_error.clear();
        }
        Ok(out)
    }

    /// The (parent, child) pairs whose stored bounds do not hold the child's
    /// (what makes the cut consistent and the cull safe), with the splats as
    /// the finest level's children.
    pub fn monotone_violations(file: &AthcFile) -> Result<usize> {
        let kids = level_children(file)?;
        let splats = splats_of(file);
        let mut bad = 0;
        let w = LOD_ERROR_WORDS;
        for l in 0..file.levels.len() {
            let b = &file.levels[l].1;
            if b.lod_error.len() != b.n * w {
                bail!("level {l} has no LoD errors");
            }
            for i in 0..b.n {
                let (e, r) = unpack_error(b.lod_error[i * w]);
                let c = centre_of(b, i);
                let (a, z) = kids[l][i];
                for k in a..z {
                    let k = k as usize;
                    let (ce, cr, cc) = if l + 1 < file.levels.len() {
                        let nb = &file.levels[l + 1].1;
                        let (x, y) = unpack_error(nb.lod_error[k * w]);
                        (x, y, centre_of(nb, k))
                    } else {
                        let own = NodeBounds::own(&splats, k, false);
                        (0.0, own.rho, own.centre)
                    };
                    if ce > e || dist(c, cc) + cr > r * (1.0 + 1.0 / 1024.0) {
                        bad += 1;
                    }
                }
            }
        }
        Ok(bad)
    }

    /// Each node's leaves (splat range), coarsest level first.
    pub fn leaf_ranges(file: &AthcFile) -> Result<Vec<Vec<(u32, u32)>>> {
        let kids = level_children(file)?;
        let mut out: Vec<Vec<(u32, u32)>> = vec![Vec::new(); kids.len()];
        for l in (0..kids.len()).rev() {
            out[l] = if l + 1 == kids.len() {
                kids[l].clone()
            } else {
                kids[l].iter().map(|&(a, z)| (out[l + 1][a as usize].0, out[l + 1][z as usize - 1].1)).collect()
            };
        }
        Ok(out)
    }

    /// The nodes whose stored cone (theta about the 8-bit axis) leaves out
    /// the normal of a leaf under it.
    pub fn cone_violations(file: &AthcFile) -> Result<usize> {
        let ranges = leaf_ranges(file)?;
        let splats = splats_of(file);
        let mut bad = 0;
        for (l, (_, b)) in file.levels.iter().enumerate() {
            for i in 0..b.n {
                let (t, n) = unpack_cone(b.lod_error[i * LOD_ERROR_WORDS + 1]);
                if !t.is_finite() {
                    continue;
                }
                let (a, z) = ranges[l][i];
                if (a..z).any(|k| normal_of(&splats, k as usize).is_none_or(|m| angle(m, n) > t + 1e-3)) {
                    bad += 1;
                }
            }
        }
        Ok(bad)
    }
}

#[cfg(all(test, feature = "athc"))]
mod tests {
    use super::*;
    use crate::athc::AthcFile;

    fn fixture(name: &str) -> AthcFile {
        let path = format!("{}/../../test/fixtures/athc/{}", env!("CARGO_MANIFEST_DIR"), name);
        let bytes = std::fs::read(path).unwrap();
        if u32::from_le_bytes(bytes[..4].try_into().unwrap()) == crate::athc_v3::ATH3_MAGIC {
            crate::athc_v3::read_v3(&bytes).unwrap()
        } else {
            AthcFile::read(&bytes).unwrap()
        }
    }

    #[test]
    fn f16_up_never_rounds_down() {
        for &v in &[0.0f32, 1e-7, 3.3e-4, 0.1, 0.123_456, 1.0, 7.77, 1234.5] {
            let h = f16_up(v).to_f32();
            assert!(h >= v && h <= v * 1.001 + 1e-7, "{v} -> {h}");
        }
        assert!(f16_up(f32::INFINITY).to_f32().is_infinite());
    }

    fn projected(b: &AthcBlock, i: usize, eye: [f32; 3]) -> f32 {
        let (e, r) = unpack_error(b.lod_error[i * LOD_ERROR_WORDS]);
        e / (dist(centre_of(b, i), eye) - r).max(NEAR)
    }

    /// eps is monotone up the tree, each sphere and cone holds its
    /// children's, and so the projected error is monotone from every eye
    /// and a culled node's leaves all face away.
    #[test]
    fn errors_are_monotone_spheres_and_cones_nest() {
        for name in ["two_cards.athc", "every_stream.v3.athc", "hood_t16.athc", "lobes_sphere.athc"] {
            let f = fixture(name);
            let g = with_lod_errors(&f, None, &ErrorWeights::default(), false).unwrap();
            assert_eq!(monotone_violations(&g).unwrap(), 0, "{name}");
            assert_eq!(cone_violations(&g).unwrap(), 0, "{name}");
            assert!(g.levels.iter().any(|(_, b)| b.lod_error.iter().step_by(2).any(|&w| unpack_error(w).0 > 0.0)), "{name}");
            let kids = level_children(&g).unwrap();
            let splats = splats_of(&g);
            let eyes = [[0.0f32, 0.0, 5.0], [3.0, -2.0, 1.0], [0.1, 0.2, 0.3], [0.0, -6.0, 0.0]];
            for eye in eyes {
                for l in 1..g.levels.len() {
                    let (pb, cb) = (&g.levels[l - 1].1, &g.levels[l].1);
                    for (i, &(a, z)) in kids[l - 1].iter().enumerate() {
                        let p = projected(pb, i, eye);
                        for k in a..z {
                            assert!(projected(cb, k as usize, eye) <= p * 1.002 + 1e-6, "{name} level {l}");
                        }
                    }
                }
                // Every leaf under a culled finest-level node faces away.
                let l = g.levels.len() - 1;
                let b = &g.levels[l].1;
                for (i, &(a, z)) in kids[l].iter().enumerate() {
                    let (_, rho) = unpack_error(b.lod_error[i * 2]);
                    let (t, n) = unpack_cone(b.lod_error[i * 2 + 1]);
                    if back_facing(centre_of(b, i), rho, t, n, eye) {
                        for k in a..z {
                            let c = centre_of(&splats, k as usize);
                            let nn = normal_of(&splats, k as usize).unwrap();
                            let v = [eye[0] - c[0], eye[1] - c[1], eye[2] - c[2]];
                            assert!(nn[0] * v[0] + nn[1] * v[1] + nn[2] * v[2] < 0.0, "{name}: a culled leaf faces the eye");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_flat_sheet_seen_from_behind_is_culled_from_the_front_not() {
        let n = [0.0f32, 0.0, 1.0];
        assert!(back_facing([0.0; 3], 0.1, 0.05, n, [0.0, 0.0, -5.0]));
        assert!(!back_facing([0.0; 3], 0.1, 0.05, n, [0.0, 0.0, 5.0]));
        // Edge on, or within the cone plus the sphere's angle: kept.
        assert!(!back_facing([0.0; 3], 0.1, 0.05, n, [5.0, 0.0, -0.1]));
        assert!(!back_facing([0.0; 3], 0.1, f32::INFINITY, n, [0.0, 0.0, -5.0]));
        // The 8-bit axis plus its quantization error still bounds the cone.
        for &v in &[[0.3f32, -0.5, 0.81], [-0.9, 0.1, -0.42], [0.0, 1.0, 0.0]] {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let v = v.map(|x| x / l);
            let (t, q) = unpack_cone(pack_cone(0.0, v));
            assert!(angle(v, q) <= t, "{v:?}");
        }
    }

    #[derive(Default)]
    struct Errors {
        errors: Vec<(usize, Vec<u32>)>,
    }

    impl crate::decoder::SplatReceiver for Errors {
        fn set_lod_error(&mut self, base: usize, count: usize, error: &[u32]) {
            self.errors.push((base, error[..count * LOD_ERROR_WORDS].to_vec()));
        }
        fn set_batch(&mut self, _: usize, _: usize, _: &crate::decoder::SplatProps) {}
        fn set_center(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_opacity(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_rgb(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_rgba(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_scale(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_quat(&mut self, _: usize, _: usize, _: &[f32]) {}
    }

    fn decode(bytes: &[u8]) -> Errors {
        use crate::decoder::ChunkReceiver;
        let mut d = crate::athc::AthcDecoder::new(Errors::default());
        d.push(bytes).unwrap();
        ChunkReceiver::finish(&mut d).unwrap();
        d.into_splats()
    }

    /// LODE through v3, the whole-file decoder and the merged pages, the
    /// synthetic root's over level 1.
    #[test]
    fn lod_errors_go_through_v3_the_decoder_and_merged_pages() {
        use crate::athc_v3::{read_v3, write_v3, COMPRESSION_GZIP};
        for name in ["two_cards.athc", "hood_t16.athc"] {
            let f = fixture(name);
            let g = with_lod_errors(&f, None, &ErrorWeights::default(), false).unwrap();
            let bytes = write_v3(&g, COMPRESSION_GZIP).unwrap();
            let back = read_v3(&bytes).unwrap();
            assert!(back.header.has(crate::athc::FLAG_LOD_ERROR));
            for ((_, a), (_, b)) in back.levels.iter().zip(&g.levels) {
                assert_eq!(a.lod_error, b.lod_error);
            }
            assert!(back.chunks.iter().all(|c| c.lod_error.is_empty()));
            assert!(!crate::athc::AthcFile::read(&g.write().unwrap()).unwrap().header.has(crate::athc::FLAG_LOD_ERROR));

            let tree = crate::athc::VirtualTree::of_file(&g, false).unwrap();
            let mut expect: Vec<u32> = Vec::new();
            if tree.synth_root {
                let l1 = &g.levels[0].1;
                let (e, _) = (0..l1.n).map(|i| unpack_error(l1.lod_error[2 * i])).fold((0.0f32, 0.0f32), |a, b| (a.0.max(b.0), 0.0));
                let root = decode(&bytes).errors[0].1[..2].to_vec();
                assert_eq!(unpack_error(root[0]).0, e);
                assert_eq!(root[1], NO_CONE as u32);
                expect.extend_from_slice(&root);
            }
            for (_, b) in &g.levels {
                expect.extend_from_slice(&b.lod_error);
            }
            assert_eq!(decode(&bytes).errors, vec![(0, expect.clone())], "{name}");

            let mut head = g.header.to_bytes().to_vec();
            head.extend_from_slice(&g.extra.to_bytes());
            let (_, pages) = crate::athc::merged_pages_of(&g, &head).unwrap();
            let got: Vec<u32> = pages.iter().flat_map(|p| decode(p).errors.into_iter().flat_map(|e| e.1)).collect();
            assert_eq!(got, expect, "{name}");
        }
    }

    /// athenea's bounded error (Decimate.cpp's weights): r (|cost| + 10 s_c
    /// + theta^2 / alpha), a guarded merge (cost < 0) x 4, at least 2 r.
    #[test]
    fn athenea_bounded_weights() {
        let f = fixture("hood_t16.athc");
        let b = &f.chunks[0];
        // A small chain over the first splats: 0 1 -> 4, 2 3 -> 5, 4 5 -> 6.
        let mut block = AthcBlock::default();
        for k in [0, 700, 1400, 2100, 300, 1000, 1700] {
            block.append(&b.slice(k, 1));
        }
        let parent = [4, 4, 5, 5, 6, 6, 6];
        let cost = [0.0, 0.0, 0.0, 0.0, 0.2, -0.3, 0.05];
        let w = ErrorWeights::athenea();
        let nb = athenea_tree_errors(&block, &parent, 4, &cost, &w, false, None);
        let own = |k: usize| {
            let r = size_of(scales_of(&block, k));
            let alpha = look_of(&block, k).1.max(0.02).powi(2);
            r * (cost[k].abs() + 10.0 * nb[k].spread + nb[k].theta * nb[k].theta / alpha)
        };
        assert!(nb[6].theta > 0.0);
        assert!((nb[4].eps - own(4)).abs() <= 1e-6 * own(4).max(1e-6));
        let r5 = size_of(scales_of(&block, 5));
        assert!((nb[5].eps - (4.0 * own(5)).max(2.0 * r5)).abs() <= 1e-6 * nb[5].eps);
        assert_eq!(nb[6].eps, own(6).max(nb[4].eps).max(nb[5].eps));
        assert_eq!(ErrorWeights::athenea().parse_over("own").unwrap().colour, 0.0);
        assert_eq!(ErrorWeights::athenea().parse_over("colour=3").unwrap().guard, 4.0);
    }

    /// A producer's own coverage (FLAG_NODE_COVERAGE) is kept: not uncapped,
    /// not divided by the widening's area, through v3.
    #[test]
    fn node_coverage_is_kept_as_stored() {
        use crate::athc::{merged_block, uncap_levels, VirtualTree, FLAG_NODE_COVERAGE};
        let mut f = fixture("two_cards.athc");
        f.header.flags |= FLAG_NODE_COVERAGE;
        for (_, b) in f.levels.iter_mut() {
            for i in 0..b.n {
                b.positions[i * 4 + 3] = 0.37;
            }
        }
        let mut g = f.clone();
        uncap_levels(&mut g);
        assert!(g.levels.iter().all(|(_, b)| (0..b.n).all(|i| b.positions[i * 4 + 3] == 0.37)));
        let back = crate::athc_v3::read_v3(&crate::athc_v3::write_v3(&g, crate::athc_v3::COMPRESSION_GZIP).unwrap()).unwrap();
        assert!(back.header.has(FLAG_NODE_COVERAGE));
        assert!(back.levels.iter().all(|(_, b)| (0..b.n).all(|i| b.positions[i * 4 + 3] == 0.37)));
        let tree = VirtualTree::of_file(&back, false).unwrap();
        let m = merged_block(&back, &tree);
        let first = if tree.synth_root { 1 } else { 0 };
        assert!((first..m.n).all(|i| m.positions[i * 4 + 3] == 0.37));
        // Without the flag the decoder widens (and uncaps) them.
        let mut plain = back.clone();
        plain.header.flags &= !FLAG_NODE_COVERAGE;
        let m = merged_block(&plain, &tree);
        assert!((first..m.n).any(|i| m.positions[i * 4 + 3] != 0.37));
    }

    /// athenea's definitions on a hand tree: leaves 0, a merge r |cost|
    /// raised by its children, the sphere grown by the child's offset.
    #[test]
    fn athenea_tree_errors_match_lod_tree_cut() {
        // Four leaves, two merges, a root: 0 1 -> 4, 2 3 -> 5, 4 5 -> 6.
        let mut b = AthcBlock::default();
        for (x, s) in [(0.0f32, 0.1f32), (1.0, 0.1), (4.0, 0.2), (5.0, 0.2), (0.5, 0.6), (4.5, 0.7), (2.5, 2.0)] {
            let ln = s.ln();
            let mut one = AthcBlock { n: 1, ..Default::default() };
            one.positions = vec![x, 0.0, 0.0, 1.0];
            one.shape = vec![0, crate::athc::pack_halves(ln, ln), crate::athc::pack_halves(ln, 0.5), crate::athc::pack_halves(0.5, 0.5)];
            b.append(&one);
        }
        let parent = [4, 4, 5, 5, 6, 6, 6];
        let cost = [0.0, 0.0, 0.0, 0.0, 0.5, -3.0, 0.1];
        let nb = athenea_tree_errors(&b, &parent, 4, &cost, &ErrorWeights::default(), false, None);
        // A leaf left open takes the cone from every node over it.
        let open = [true, false, true, true, true, true, true];
        let nb2 = athenea_tree_errors(&b, &parent, 4, &cost, &ErrorWeights::default(), false, Some(&open));
        assert!(!nb2[1].cull && !nb2[4].cull && !nb2[6].cull);
        let size = |k: usize| size_of(scales_of(&b, k));
        for k in 0..4 {
            assert_eq!(nb[k].eps, 0.0);
        }
        assert!((nb[4].eps - size(4) * 0.5).abs() < 1e-6);
        assert!((nb[5].eps - size(5) * 3.0).abs() < 1e-6, "|cost|");
        assert_eq!(nb[6].eps, nb[5].eps.max(size(6) * 0.1), "raised to its children's");
        // Leaf 0: 0.5 off merge 4 with its own 0.2; merge 4's own 2 x 0.6.
        assert!((nb[4].rho - (2.0 * scales_of(&b, 4)[0]).max(0.5 + 2.0 * scales_of(&b, 0)[0])).abs() < 1e-5);
        assert!(nb[6].rho >= 2.0 + nb[5].rho - 1e-5 && nb[6].rho >= nb[4].rho + 2.0 - 1e-5);
        // No normals: never culled.
        assert!(nb.iter().all(|n| !n.cull));
    }
}
