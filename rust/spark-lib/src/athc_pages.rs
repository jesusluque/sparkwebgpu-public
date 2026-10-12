//! athenea's merge tree paged by ERROR BANDS (Nanite's order), as a `.athc`
//! v3 a player streams a page at a time (thread CK).
//!
//! athenea's `decimate --tree-pages P` gives every node of her tree (leaves
//! and merges) a page: the band of its PARENT's error (an octave), the bands
//! coarsest first, each band's nodes in leaf order, the whole order cut into
//! pages of P (`primvars:athenea:splat:lodPage`), and a table a player tests
//! a page by without reading its nodes (`lodPageInfo`: band, nodes, first;
//! `lodPageBounds`: the box of the nodes' spheres and the largest and
//! smallest parent error). A node is needed exactly when its parent's
//! projected error passes tau, so a view needs, where it looks, the pages
//! whose largest parent error, seen from the nearest point of their box,
//! passes tau; a node's parent is in its own page or an earlier one.
//!
//! `pack_pages` writes that tree page by page (`FLAG_PAGED`): each page one
//! block, so one Range request, every section a tier needs contiguous in it.
//! Spark's LoD tree names a node's children as a contiguous run, so the
//! packer keeps each node's children in ONE page, next to one another:
//!
//! - siblings share their parent's error, hence their band; a page boundary
//!   that falls between two of them moves the later ones into the earlier
//!   page (their parent is in it or before it, their children after it);
//! - within a page, sibling runs, those holding merges first;
//! - a forest's roots get a root merged over them (`athc::merge_root`), at
//!   the head of page 0.
//!
//! The `TREE` section says, for each node, the page its children are in and
//! where they start in it (`TREE_WORDS`); the decoder names them as Spark's
//! paged index `page << 16 | place` (each page a 65 536-index chunk of a
//! sparse virtual space), or, decoding the whole file, by their place in
//! the file. The page table (`PageRow`) is recomputed from the nodes as the
//! traversal will see them (the LODE words, f16 rounded up), and its box
//! holds the PARENTS' spheres, so the page test is conservative against
//! the traversal's own refinement test: every node the cut can reach is in
//! a needed page (`PageBox::Parents`; `PageBox::Own` is athenea's box, the
//! nodes' own spheres, for comparison).

use anyhow::{anyhow, bail, Result};
use serde::Serialize;

use crate::athc::{merge_root, orient_merged, AthcBlock};
use crate::athc_build::reorder;
use crate::athc_lod_cut::{pack_error, unpack_error, NodeBounds, LOD_ERROR_WORDS, NO_CONE};

/// Row `j` of `src` over row `i` of `dst` (every stream both carry).
fn copy_row(dst: &mut AthcBlock, i: usize, src: &AthcBlock, j: usize) {
    fn one(d: &mut [u32], dn: usize, i: usize, s: &[u32], sn: usize, j: usize) {
        if d.is_empty() || s.is_empty() {
            return;
        }
        let per = d.len() / dn;
        if s.len() / sn != per {
            return;
        }
        d[i * per..(i + 1) * per].copy_from_slice(&s[j * per..(j + 1) * per]);
    }
    let (dn, sn) = (dst.n, src.n);
    dst.positions[i * 4..i * 4 + 4].copy_from_slice(&src.positions[j * 4..j * 4 + 4]);
    one(&mut dst.shape, dn, i, &src.shape, sn, j);
    one(&mut dst.sh, dn, i, &src.sh, sn, j);
    one(&mut dst.normals, dn, i, &src.normals, sn, j);
    one(&mut dst.emission, dn, i, &src.emission, sn, j);
    one(&mut dst.pbr, dn, i, &src.pbr, sn, j);
    one(&mut dst.lobes, dn, i, &src.lobes, sn, j);
    one(&mut dst.transfer, dn, i, &src.transfer, sn, j);
    one(&mut dst.shadow_bits, dn, i, &src.shadow_bits, sn, j);
    one(&mut dst.curvature, dn, i, &src.curvature, sn, j);
    one(&mut dst.skin, dn, i, &src.skin, sn, j);
    // Never occluded, no second opacity.
    if !dst.occlusion.is_empty() {
        dst.occlusion[i] = 0;
    }
    if !dst.lod_mass.is_empty() {
        dst.lod_mass[i] = 0;
    }
}
use crate::athc_v3::{BlockEntry, Section};

/// Words a node in the `TREE` section: its children's page, then their
/// first one's place in that page (low 16 bits) and count (high 16).
pub const TREE_WORDS: usize = 2;
/// Words a row of the page table (`PageRow::to_words`).
pub const PAGE_ROW_WORDS: usize = 16;
/// Words a row with the compact tier (`athc_compact`, thread CN): word 16
/// the bytes of its tier-4 section, 17..29 its compact head.
pub const PAGE_ROW_WORDS_COMPACT: usize = 32;
/// The parent error of a root, as athenea writes it: always needed.
pub const ROOT_PARENT_ERROR: f32 = 3.0e38;

/// One page of a paged tree, as the file's page table holds it.
///
/// ```text
/// 0  band (the coarsest of its nodes': athenea's bandOf of the parent's error)
/// 1  nodes   2 first node in page order
/// 3  box lo xyz   6 box hi xyz (f32): holds the spheres a node's need is tested on
/// 9  largest parent error   10 smallest (f32, world units; a root's 3e38)
/// 11 the page's first (tier-1) byte (u64, two words)
/// 13 the bytes of its tier-1 sections, tier-2's, tier-3's (the file is laid
///    out by tier: each a run of its own)
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PageRow {
    pub band: u32,
    pub count: u32,
    pub first: u32,
    pub lo: [f32; 3],
    pub hi: [f32; 3],
    pub eps_max: f32,
    pub eps_min: f32,
    pub offset: u64,
    pub tier_bytes: [u32; 3],
    /// The compact tier (rows of `PAGE_ROW_WORDS_COMPACT`): its bytes and
    /// the page's head (`athc_compact::HEAD_WORDS`); 0 without.
    pub tier4_bytes: u32,
    pub compact: [u32; crate::athc_compact::HEAD_WORDS],
}

impl PageRow {
    pub fn to_words(&self) -> [u32; PAGE_ROW_WORDS] {
        let f = |v: f32| v.to_bits();
        [
            self.band,
            self.count,
            self.first,
            f(self.lo[0]),
            f(self.lo[1]),
            f(self.lo[2]),
            f(self.hi[0]),
            f(self.hi[1]),
            f(self.hi[2]),
            f(self.eps_max),
            f(self.eps_min),
            self.offset as u32,
            (self.offset >> 32) as u32,
            self.tier_bytes[0],
            self.tier_bytes[1],
            self.tier_bytes[2],
        ]
    }

    pub fn from_words(w: &[u32]) -> Self {
        let f = |k: usize| f32::from_bits(w[k]);
        Self {
            band: w[0],
            count: w[1],
            first: w[2],
            lo: [f(3), f(4), f(5)],
            hi: [f(6), f(7), f(8)],
            eps_max: f(9),
            eps_min: f(10),
            offset: w[11] as u64 | (w[12] as u64) << 32,
            tier_bytes: [w[13], w[14], w[15]],
            tier4_bytes: w.get(16).copied().unwrap_or(0),
            compact: std::array::from_fn(|k| w.get(17 + k).copied().unwrap_or(0)),
        }
    }

    /// `to_words` with the compact tier's words (`PAGE_ROW_WORDS_COMPACT`).
    pub fn to_words_compact(&self) -> [u32; PAGE_ROW_WORDS_COMPACT] {
        let mut w = [0u32; PAGE_ROW_WORDS_COMPACT];
        w[..PAGE_ROW_WORDS].copy_from_slice(&self.to_words());
        w[16] = self.tier4_bytes;
        w[17..17 + crate::athc_compact::HEAD_WORDS].copy_from_slice(&self.compact);
        w
    }

    /// The page's byte ranges from its block's entry: where it starts, and
    /// the bytes through each tier's last section.
    ///
    /// A paged file lays its data out by tier (`write_v3_paged`): a page's
    /// sections of one tier are contiguous, and tier t of consecutive pages
    /// one run. `offset` is the page's first tier-1 byte, `tier_bytes[t-1]`
    /// the bytes of its tier-t sections (each a run starting at its first
    /// section's span).
    pub fn fill_ranges(&mut self, e: &BlockEntry, sections: &[Section]) {
        self.offset = e.spans.first().map_or(0, |s| s.offset);
        let tier = |t: u32| -> u32 {
            sections.iter().zip(&e.spans).filter(|(s, _)| s.tier.clamp(1, 4) == t).map(|(_, sp)| sp.stored).sum()
        };
        for t in 1..=3u32 {
            self.tier_bytes[t as usize - 1] = tier(t);
        }
        self.tier4_bytes = tier(4);
    }

    /// The nearest point of the box to `eye`, and its distance.
    pub fn distance(&self, eye: [f32; 3]) -> f32 {
        let mut d2 = 0.0f32;
        for k in 0..3 {
            let v = (self.lo[k] - eye[k]).max(0.0).max(eye[k] - self.hi[k]);
            d2 += v * v;
        }
        d2.sqrt()
    }

    /// athenea's test: the page holds a needed node somewhere in it when
    /// its largest parent error, from the box's nearest point, projects
    /// past `tau` pixels (`ppu`: pixels a unit covers at a distance of one).
    pub fn needed(&self, eye: [f32; 3], ppu: f32, tau: f32, near: f32) -> bool {
        self.eps_max * ppu / self.distance(eye).max(near) > tau
    }
}

/// athenea's `bandOf` (lod_pages.slang): the octave of a parent's error.
pub fn band_of(parent_error: f32) -> u32 {
    if parent_error >= ROOT_PARENT_ERROR {
        // A root: her 1e30.
        return band_of(1.0e30);
    }
    if parent_error > 0.0 {
        (parent_error.log2() + 40.0).clamp(0.0, 63.0) as u32
    } else {
        0
    }
}

/// Which spheres a page's box holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub enum PageBox {
    /// The parents' spheres: a node is needed when its parent's projected
    /// error passes tau, measured from the parent's sphere, so this box is
    /// conservative against the traversal (the default).
    #[default]
    Parents,
    /// The nodes' own spheres (athenea's lod_pages.slang).
    Own,
}

/// What `pack_pages` takes: every node of the tree in tree order (children
/// before parents), its parent (a root its own index), athenea's page and
/// bounds (`athc_lod_cut::athenea_tree_errors`).
pub struct PageInput<'a> {
    pub block: &'a AthcBlock,
    pub parent: &'a [u32],
    /// athenea's lodPage per node.
    pub page: &'a [u32],
    /// eps, rho and cone per node (the merges' go into LODE; the leaves'
    /// are left 0 as a chunk's are).
    pub bounds: &'a [NodeBounds],
    /// A value a node carries as its tail word (athcGroup): its row in the
    /// source, say.
    pub ids: &'a [u32],
    /// Each node's first leaf (athenea's lodStart), for the order in a page.
    pub first_leaf: &'a [u32],
    pub sh_words: usize,
    pub page_box: PageBox,
}

/// What the packer did (`pack_pages`).
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PackReport {
    pub nodes: usize,
    pub leaves: usize,
    pub roots: usize,
    pub synthetic_root: bool,
    pub pages: usize,
    pub largest_page: usize,
    pub smallest_page: usize,
    /// Nodes moved from athenea's page: to their earliest sibling's, or
    /// forward to their earliest child's.
    pub moved: usize,
    /// Parents athenea's paging put after a child (brought forward).
    pub parents_after_children: usize,
    /// Sibling runs that crossed a page boundary in athenea's paging.
    pub split_runs: usize,
    pub merges_oriented: usize,
    /// Synthetic nodes over the roots (`pack_pages_with`), and their pages.
    pub synthetic_nodes: usize,
    pub synthetic_pages: usize,
}

/// A paged tree: its pages (blocks with TREE words, LODE, tails), its page
/// table (byte ranges 0 until written), and each node's source index in
/// page order (`u32::MAX` the synthetic root).
pub struct PagedTree {
    pub pages: Vec<AthcBlock>,
    pub rows: Vec<PageRow>,
    pub order: Vec<u32>,
    pub report: PackReport,
}

impl PagedTree {
    /// Global (file) index of each page's first node.
    pub fn firsts(&self) -> Vec<u32> {
        self.rows.iter().map(|r| r.first).collect()
    }
}

/// A node's children in a paged tree (its TREE words): the page, the first
/// one's place in it, their count.
pub fn tree_children(words: &[u32]) -> (u32, u32, u32) {
    (words[0], words[1] & 0xffff, words[1] >> 16)
}

/// A page's children as `emit_block` takes them: counts, and starts named
/// by `name(page, place)`.
pub fn children_of(block: &AthcBlock, name: impl Fn(u32, u32) -> u32) -> (Vec<u16>, Vec<u32>) {
    let mut counts = Vec::with_capacity(block.n);
    let mut starts = Vec::with_capacity(block.n);
    for i in 0..block.n {
        let (page, place, count) = tree_children(&block.tree[i * TREE_WORDS..i * TREE_WORDS + TREE_WORDS]);
        counts.push(count as u16);
        starts.push(if count > 0 { name(page, place) } else { 0 });
    }
    (counts, starts)
}

/// The f16 a node centre is in the traversal's tree (LodSplat) is within
/// this share of the coordinate: a box widened by it holds the centres as
/// the traversal reads them.
const F16_SLACK: f32 = 1.0 / 1024.0;

/// Nodes a page may hold: a pager's slot (WgpuBandPager, CompactBandStreamer).
pub const POOL_SLOT_NODES: usize = 3072;

/// Roots a forest keeps flat under its synthetic root (`pack_pages`); past
/// it they are grouped under a synthetic hierarchy, this many a node.
pub const SYNTHETIC_FANOUT: usize = 2048;

/// The parent error a synthetic node's children see: f16 infinity in LODE,
/// so the error cut always refines it (never drawn, as a root's 3e38).
pub const SYNTHETIC_ERROR: f32 = f32::INFINITY;

/// athenea's tree paged by error bands (see the module), a forest past
/// `SYNTHETIC_FANOUT` roots under a synthetic hierarchy.
pub fn pack_pages(input: &PageInput) -> Result<PagedTree> {
    pack_pages_with(input, SYNTHETIC_FANOUT)
}

/// The synthetic nodes over a forest's roots (thread CX): rows past the
/// input block's, each `merge_root` of its children, never drawn.
struct Synthetic {
    /// Index of the first synthetic node (the input block's n).
    first: usize,
    rows: AthcBlock,
    /// The original roots: the synthetic root is made from them, as a flat
    /// forest's is (the same root, so the same cut).
    roots: Vec<u32>,
    /// Synthetic pages (the hierarchy's, after page 0).
    pages: usize,
}

/// `pack_pages` with the roots past `fanout` (0: never) grouped under a
/// synthetic hierarchy (thread CX: athenea's 5120 forests, 21 065 and 29 213
/// lone roots, which a flat page 0 of 3072 nodes cannot hold).
///
/// The roots in Morton order of their centres, cut into runs of at most
/// `fanout`, each run the children of a synthetic node; again over those
/// nodes until at most `fanout` are left, the new forest's roots (page 0,
/// under the synthetic root). Each run is a page of its own, the levels top
/// down after page 0, then her pages. A synthetic node:
/// - is never drawn: its error is infinite, so the error cut refines it;
/// - has a sphere round its children's, so the page tests hold;
/// - has no cone and no occlusion index (it never culls its children);
/// - carries `merge_root` of its children's rows (the streams a node needs).
///
/// The synthetic root is the flat forest's (made from the original roots),
/// so the cut is the flat forest's, node for node.
pub fn pack_pages_with(input: &PageInput, fanout: usize) -> Result<PagedTree> {
    let n = input.block.n;
    let roots: Vec<u32> = (0..n as u32).filter(|&k| {
        let p = input.parent.get(k as usize).copied().unwrap_or(k) as usize;
        p == k as usize || p >= n
    }).collect();
    if fanout < 2 || roots.len() <= fanout || input.parent.len() != n {
        return pack_inner(input, None);
    }
    let PageInput { block, parent, page, bounds, ids, first_leaf, sh_words, page_box } = *input;
    let mut parent_x = parent.to_vec();
    let mut bounds_x = bounds.to_vec();
    let mut ids_x = ids.to_vec();
    let mut first_x = first_leaf.to_vec();
    let mut rows = AthcBlock::default();
    // Levels bottom up: each level's runs (children of one new node).
    let mut level: Vec<u32> = roots.clone();
    let mut runs_by_level: Vec<Vec<Vec<u32>>> = Vec::new();
    while level.len() > fanout {
        // Morton order of the centres within the level's box.
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for &k in &level {
            let c = bounds_x[k as usize].centre;
            for d in 0..3 {
                lo[d] = lo[d].min(c[d]);
                hi[d] = hi[d].max(c[d]);
            }
        }
        let key = |k: u32| -> u64 {
            let c = bounds_x[k as usize].centre;
            let mut m = 0u64;
            let q: [u64; 3] = [0, 1, 2].map(|d| {
                let span = (hi[d] - lo[d]).max(1e-20);
                (((c[d] - lo[d]) / span).clamp(0.0, 1.0) * 2_097_151.0) as u64
            });
            for b in (0..21).rev() {
                for d in 0..3 {
                    m = (m << 1) | ((q[d] >> b) & 1);
                }
            }
            m
        };
        let mut sorted = level.clone();
        sorted.sort_by_key(|&k| (key(k), k));
        let groups = sorted.len().div_ceil(fanout);
        let per = sorted.len().div_ceil(groups);
        let mut next = Vec::with_capacity(groups);
        let mut runs = Vec::with_capacity(groups);
        for run in sorted.chunks(per) {
            let id = (n + rows.n) as u32;
            // Its rows: merge_root of its children's (originals or synthetic).
            let kids_block = if (run[0] as usize) < n {
                reorder(block, run)
            } else {
                let local: Vec<u32> = run.iter().map(|&k| k - n as u32).collect();
                reorder(&rows, &local)
            };
            let mut row = reorder(&merge_root(&kids_block, sh_words), &[0]);
            if !row.occlusion.is_empty() {
                row.occlusion[0] = 0;
            }
            if !row.lod_mass.is_empty() {
                row.lod_mass[0] = 0;
            }
            if rows.n == 0 {
                rows = row;
            } else {
                rows.append(&row);
            }
            let c = [rows.positions[(rows.n - 1) * 4], rows.positions[(rows.n - 1) * 4 + 1], rows.positions[(rows.n - 1) * 4 + 2]];
            let mut rho = 0.0f32;
            for &k in run {
                let b = &bounds_x[k as usize];
                let d = ((b.centre[0] - c[0]).powi(2) + (b.centre[1] - c[1]).powi(2) + (b.centre[2] - c[2]).powi(2)).sqrt();
                rho = rho.max(d + b.rho);
                parent_x[k as usize] = id;
            }
            parent_x.push(id);
            bounds_x.push(NodeBounds {
                eps: SYNTHETIC_ERROR,
                rho,
                spread: 0.0,
                theta: 0.0,
                cull: false,
                centre: c,
                colour: [0.0; 3],
                normal: [0.0, 0.0, 1.0],
            });
            ids_x.push(u32::MAX);
            first_x.push(run.iter().map(|&k| first_x[k as usize]).min().unwrap());
            next.push(id);
            runs.push(run.to_vec());
        }
        runs_by_level.push(runs);
        level = next;
    }
    // Pages: the new roots in page 0; each run a page, the levels top down;
    // then her pages.
    let mut page_x = vec![0u32; parent_x.len()];
    let mut at = 1u32;
    for runs in runs_by_level.iter().rev() {
        for run in runs {
            for &k in run {
                page_x[k as usize] = at;
            }
            at += 1;
        }
    }
    let original: std::collections::HashSet<u32> = roots.iter().copied().collect();
    for k in 0..n {
        if !original.contains(&(k as u32)) {
            page_x[k] = page[k] + at;
        }
    }
    let synth = Synthetic { first: n, rows, roots, pages: at as usize - 1 };
    let x = PageInput {
        block,
        parent: &parent_x,
        page: &page_x,
        bounds: &bounds_x,
        ids: &ids_x,
        first_leaf: &first_x,
        sh_words,
        page_box,
    };
    pack_inner(&x, Some(&synth))
}

fn pack_inner(input: &PageInput, synth: Option<&Synthetic>) -> Result<PagedTree> {
    let n = input.parent.len();
    let PageInput { parent, page, bounds, ids, first_leaf, .. } = *input;
    let n_block = synth.map_or(n, |s| s.first);
    if input.block.n != n_block || synth.is_some_and(|s| s.first + s.rows.n != n) || page.len() != n || bounds.len() != n || ids.len() != n || first_leaf.len() != n {
        bail!("pack_pages: {} nodes, {} parents, {} pages, {} bounds, {} ids, {} first leaves", input.block.n, parent.len(), page.len(), bounds.len(), ids.len(), first_leaf.len());
    }
    // Children of each node, in tree order.
    let mut kids: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut roots = Vec::new();
    for k in 0..n {
        let p = parent[k] as usize;
        if p == k || p >= n {
            roots.push(k as u32);
        } else {
            if p < k {
                bail!("pack_pages: node {k}'s parent {p} comes before it (children before parents)");
            }
            kids[p].push(k as u32);
        }
    }
    if roots.is_empty() {
        bail!("pack_pages: a tree without a root");
    }
    let pages = page.iter().copied().max().unwrap_or(0) as usize + 1;
    if page.iter().any(|&p| p as usize >= pages) {
        bail!("pack_pages: a node without a page");
    }
    let late_parents = (0..n).filter(|&k| {
        let p = parent[k] as usize;
        p != k && p < n && page[p] > page[k]
    }).count();
    let leaves = (0..n).filter(|&k| kids[k].is_empty()).count();
    // Each sibling run in one page: the earliest of its members'. Its
    // parent is in that page or before it, its children after it.
    let mut at_page: Vec<u32> = page.to_vec();
    let mut split_runs = 0;
    let synthetic = roots.len() > 1;
    let mut runs: Vec<&[u32]> = kids.iter().filter(|c| !c.is_empty()).map(|c| &c[..]).collect();
    if synthetic {
        runs.push(&roots);
    }
    // Until it holds: a parent in its children's page or before it (her
    // order breaks it where a parent and its first child share a band and a
    // first leaf, and the page boundary falls between them: the parent is
    // brought forward), and every run in its earliest member's page.
    let mut parent_moves = 0;
    for round in 0.. {
        let mut changed = false;
        for k in 0..n {
            let p = parent[k] as usize;
            if p != k && p < n && at_page[p] > at_page[k] {
                at_page[p] = at_page[k];
                parent_moves += 1;
                changed = true;
            }
        }
        for run in &runs {
            let lo = run.iter().map(|&c| at_page[c as usize]).min().unwrap();
            for &c in run.iter() {
                if at_page[c as usize] != lo {
                    at_page[c as usize] = lo;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
        if round > n {
            bail!("pack_pages: pages do not settle");
        }
    }
    for run in &runs {
        if run.iter().any(|&c| page[c as usize] != page[run[0] as usize]) {
            split_runs += 1;
        }
    }
    let _ = parent_moves;
    // The roots of a forest go to page 0 with the root merged over them.
    if synthetic {
        for &r in &roots {
            if at_page[r as usize] != 0 {
                at_page[r as usize] = 0;
            }
        }
    }
    let moved = (0..n).filter(|&k| at_page[k] != page[k]).count();
    // Pages the moves emptied (her coarse pages of lone roots, all gone to
    // page 0 under the synthetic root) are dropped, the rest renumbered in
    // order: a page is never empty.
    let pages = {
        let mut used = vec![false; pages];
        for &p in &at_page {
            used[p as usize] = true;
        }
        let mut new_of = vec![0u32; pages];
        let mut next = 0u32;
        for (p, &u) in used.iter().enumerate() {
            new_of[p] = next;
            if u {
                next += 1;
            }
        }
        for p in at_page.iter_mut() {
            *p = new_of[*p as usize];
        }
        next as usize
    };
    // Nodes of each page.
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); pages];
    for k in 0..n {
        members[at_page[k] as usize].push(k as u32);
    }
    // Within a page: the root first, then sibling runs, those with a merge
    // before those of leaves, each by its first leaf.
    const ROOT: u32 = u32::MAX;
    let mut order: Vec<u32> = Vec::with_capacity(n + 1);
    let mut rows = Vec::with_capacity(pages);
    let mut place = vec![(0u32, 0u32); n]; // (page, place)
    let mut root_place = (0u32, 0u32);
    for (p, nodes) in members.iter().enumerate() {
        let first = order.len() as u32;
        let mut head: Vec<u32> = Vec::new();
        if p == 0 {
            if synthetic {
                head.push(ROOT);
            } else {
                head.push(roots[0]);
            }
        }
        // The runs in this page, by their parent.
        let mut seen = vec![false; 0];
        let _ = &mut seen;
        let mut groups: Vec<(bool, u32, Vec<u32>)> = Vec::new();
        let mut done = std::collections::HashSet::new();
        for &k in nodes {
            let k = k as usize;
            let pk = parent[k] as usize;
            let is_root = pk == k || pk >= n;
            if is_root {
                if synthetic && !done.contains(&ROOT) {
                    done.insert(ROOT);
                    let run = roots.clone();
                    let merges = run.iter().any(|&c| !kids[c as usize].is_empty());
                    let lead = run.iter().map(|&c| first_leaf[c as usize]).min().unwrap();
                    groups.push((!merges, lead, run));
                }
                continue;
            }
            if done.insert(pk as u32) {
                let run = kids[pk].clone();
                let merges = run.iter().any(|&c| !kids[c as usize].is_empty());
                let lead = run.iter().map(|&c| first_leaf[c as usize]).min().unwrap();
                groups.push((!merges, lead, run));
            }
        }
        // Runs whose parent is in an earlier page first (those holding a
        // merge before those of leaves, each by its first leaf), then
        // breadth first: a run right after the runs before its parent's.
        let here: std::collections::HashSet<u32> = nodes.iter().copied().collect();
        let (mut outer, mut inner): (Vec<_>, Vec<_>) = groups.into_iter().partition(|g| {
            let c = g.2[0] as usize;
            let pk = parent[c] as usize;
            pk == c || pk >= n || !here.contains(&(pk as u32))
        });
        outer.sort_by_key(|g| (g.0, g.1));
        let mut by_parent: std::collections::HashMap<u32, Vec<u32>> =
            inner.drain(..).map(|g| (parent[g.2[0] as usize], g.2)).collect();
        let mut list = head;
        for (_, _, run) in outer {
            if synthetic && p == 0 && run == roots {
                continue;
            }
            list.extend(run);
        }
        if synthetic && p == 0 {
            list.splice(1..1, roots.iter().copied());
        }
        let mut i = 0;
        while i < list.len() {
            if list[i] != ROOT {
                if let Some(run) = by_parent.remove(&list[i]) {
                    list.extend(run);
                }
            }
            i += 1;
        }
        if !by_parent.is_empty() {
            bail!("pack_pages: page {p} has runs whose parent it does not reach");
        }
        if list.len() != nodes.len() + (p == 0 && synthetic) as usize {
            bail!("pack_pages: page {p} lays out {} of its {} nodes", list.len(), nodes.len());
        }
        if list.len() > POOL_SLOT_NODES {
            // The pagers' slots (WgpuBandPager, CompactBandStreamer) hold
            // 3072 nodes: a larger page (a forest's roots past ~3070, all
            // in page 0) would not load. usd-athc --split-roots.
            bail!("pack_pages: page {p} of {} nodes (a pool slot holds {POOL_SLOT_NODES}; --split-roots)", list.len());
        }
        for (i, &k) in list.iter().enumerate() {
            if k == ROOT {
                root_place = (p as u32, i as u32);
            } else {
                place[k as usize] = (p as u32, i as u32);
            }
        }
        order.extend(&list);
        rows.push(PageRow { count: list.len() as u32, first, ..Default::default() });
    }
    // The rows of every node in page order, the synthetic root made from the roots.
    let real: Vec<u32> = order.iter().map(|&k| if k == ROOT { synth.map_or(roots[0], |s| s.roots[0]) } else if k as usize >= n_block { 0 } else { k }).collect();
    let mut all = reorder(input.block, &real);
    // The synthetic hierarchy's rows (their own, `merge_root` of their children's).
    if let Some(s) = synth {
        for (i, &k) in order.iter().enumerate() {
            if k == ROOT || (k as usize) < n_block {
                continue;
            }
            copy_row(&mut all, i, &s.rows, k as usize - s.first);
        }
    }
    // athenea's merges: the third axis turned to the normal (the decoder's
    // orient_merged for a producer's coverage), the leaves as they are.
    let merges: Vec<u32> = (0..order.len() as u32).filter(|&i| {
        let k = order[i as usize];
        k == ROOT || !kids[k as usize].is_empty()
    }).collect();
    let mut m = reorder(&all, &merges);
    orient_merged(&mut m);
    for (j, &i) in merges.iter().enumerate() {
        let i = i as usize;
        all.shape[i * 4..i * 4 + 4].copy_from_slice(&m.shape[j * 4..j * 4 + 4]);
    }
    let mut root_bounds = None;
    if synthetic {
        // A synthetic hierarchy's root is the flat forest's: of the original roots.
        let over: &[u32] = synth.map_or(&roots, |s| &s.roots);
        let rb = reorder(input.block, over);
        let mut root = merge_root(&rb, input.sh_words);
        orient_merged(&mut root);
        let i = (root_place.1) as usize; // page 0
        // merge_root gives what it merges' streams carry (the heaviest's).
        let one = |dst: &mut Vec<u32>, src: &Vec<u32>| {
            if !dst.is_empty() && !src.is_empty() {
                let per = dst.len() / all.n;
                dst[i * per..(i + 1) * per].copy_from_slice(&src[..per]);
            }
        };
        all.positions[i * 4..i * 4 + 4].copy_from_slice(&root.positions[..4]);
        one(&mut all.shape, &root.shape);
        one(&mut all.sh, &root.sh);
        one(&mut all.normals, &root.normals);
        one(&mut all.emission, &root.emission);
        one(&mut all.pbr, &root.pbr);
        one(&mut all.lobes, &root.lobes);
        one(&mut all.transfer, &root.transfer);
        one(&mut all.shadow_bits, &root.shadow_bits);
        one(&mut all.curvature, &root.curvature);
        one(&mut all.skin, &root.skin);
        // No second opacity of its own (the copied row was its first root's).
        if !all.lod_mass.is_empty() {
            all.lod_mass[i] = 0;
        }
        // No occlusion mask of its own either (the copied row's was its
        // first root's: hidden whenever that root was, it took the whole
        // forest with it; thread CS). Never occluded.
        if !all.occlusion.is_empty() {
            all.occlusion[i] = 0;
        }
        // Its bounds: the roots' largest error, its sphere round theirs, no cone.
        let c = [all.positions[i * 4], all.positions[i * 4 + 1], all.positions[i * 4 + 2]];
        let mut eps = 0.0f32;
        let mut rho = 0.0f32;
        for &r in over {
            let b = &bounds[r as usize];
            eps = eps.max(b.eps);
            let d = ((b.centre[0] - c[0]).powi(2) + (b.centre[1] - c[1]).powi(2) + (b.centre[2] - c[2]).powi(2)).sqrt();
            rho = rho.max(d + b.rho);
        }
        root_bounds = Some((eps, rho, c));
    }
    // Tails, LODE (merges' only) and TREE.
    all.tail = order.iter().map(|&k| if k == ROOT { u32::MAX } else { ids[k as usize] }).collect();
    all.lod_error = vec![0; order.len() * LOD_ERROR_WORDS];
    all.tree = vec![0; order.len() * TREE_WORDS];
    for (i, &k) in order.iter().enumerate() {
        let run: &[u32] = if k == ROOT { &roots } else { &kids[k as usize] };
        if run.is_empty() {
            // A leaf of a closed part (athenea:splat:closed): its own cone,
            // theta 0 about its normal, and no error (eps, rho 0) -- the
            // compact traversal culls it from behind (thread CS).
            if k != ROOT && bounds[k as usize].cull {
                all.lod_error[i * 2 + 1] = bounds[k as usize].words()[1];
            }
            continue;
        }
        let (cp, c0) = place[run[0] as usize];
        for (j, &c) in run.iter().enumerate() {
            if place[c as usize] != (cp, c0 + j as u32) {
                bail!("pack_pages: node {k}'s children are not one run of a page");
            }
        }
        all.tree[i * 2] = cp;
        all.tree[i * 2 + 1] = c0 | (run.len() as u32) << 16;
        let words = if k == ROOT {
            let (eps, rho, _) = root_bounds.unwrap();
            [pack_error(eps, rho), NO_CONE as u32]
        } else {
            bounds[k as usize].words()
        };
        all.lod_error[i * 2..i * 2 + 2].copy_from_slice(&words);
    }
    // The page table from the nodes as the traversal reads them: each node's
    // parent error (f16, rounded up as LODE stores it) and sphere.
    let mut parent_at = vec![u32::MAX; order.len()];
    for i in 0..order.len() {
        let (cp, c0, count) = tree_children(&all.tree[i * 2..i * 2 + 2]);
        for j in 0..count {
            let child = rows[cp as usize].first + c0 + j;
            parent_at[child as usize] = i as u32;
        }
    }
    let sphere = |i: usize| -> ([f32; 3], f32) {
        let k = order[i];
        if k == ROOT {
            let (_, rho, c) = root_bounds.unwrap();
            (c, rho)
        } else {
            let b = &bounds[k as usize];
            (b.centre, b.rho)
        }
    };
    for (p, row) in rows.iter_mut().enumerate() {
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        let (mut eps_max, mut eps_min) = (0.0f32, f32::MAX);
        for i in row.first as usize..(row.first + row.count) as usize {
            let up = parent_at[i];
            let e = if up == u32::MAX { ROOT_PARENT_ERROR } else { unpack_error(all.lod_error[up as usize * 2]).0.min(ROOT_PARENT_ERROR) };
            eps_max = eps_max.max(e);
            eps_min = eps_min.min(e);
            row.band = row.band.max(band_of(e));
            let (c, r) = match input.page_box {
                PageBox::Own => sphere(i),
                // A root's need is its own: its sphere.
                PageBox::Parents if up == u32::MAX => sphere(i),
                PageBox::Parents => {
                    let up = up as usize;
                    (
                        [all.positions[up * 4], all.positions[up * 4 + 1], all.positions[up * 4 + 2]],
                        unpack_error(all.lod_error[up * 2]).1,
                    )
                }
            };
            for d in 0..3 {
                let slack = F16_SLACK * c[d].abs();
                lo[d] = lo[d].min(c[d] - r - slack);
                hi[d] = hi[d].max(c[d] + r + slack);
            }
        }
        if row.count == 0 {
            bail!("pack_pages: page {p} is empty");
        }
        row.lo = lo;
        row.hi = hi;
        row.eps_max = eps_max;
        row.eps_min = eps_min;
    }
    // Split into page blocks.
    let blocks: Vec<AthcBlock> = rows.iter().map(|r| all.slice(r.first as usize, r.count as usize)).collect();
    let report = PackReport {
        nodes: order.len(),
        leaves,
        roots: roots.len(),
        synthetic_root: synthetic,
        pages,
        largest_page: rows.iter().map(|r| r.count as usize).max().unwrap_or(0),
        smallest_page: rows.iter().map(|r| r.count as usize).min().unwrap_or(0),
        moved,
        parents_after_children: late_parents,
        split_runs,
        merges_oriented: merges.len(),
        synthetic_nodes: synth.map_or(0, |s| s.rows.n),
        synthetic_pages: synth.map_or(0, |s| s.pages),
    };
    Ok(PagedTree { pages: blocks, rows, order, report })
}

/// Checks a paged tree's structure: every TREE run inside one page, every
/// node but the root named by exactly one parent, a parent in its child's
/// page or an earlier one, children after their parent within a page.
/// Returns the parent of each node (global index; the root `u32::MAX`).
pub fn check_paged(pages: &[AthcBlock], rows: &[PageRow]) -> Result<Vec<u32>> {
    let total: usize = rows.iter().map(|r| r.count as usize).sum();
    let mut parent = vec![u32::MAX; total];
    let mut first = 0u32;
    for (p, (b, r)) in pages.iter().zip(rows).enumerate() {
        if r.first != first || r.count as usize != b.n || b.tree.len() != b.n * TREE_WORDS {
            bail!("page {p}: first {} count {} (expected first {first}, {} nodes)", r.first, r.count, b.n);
        }
        first += r.count;
    }
    for (p, b) in pages.iter().enumerate() {
        for i in 0..b.n {
            let (cp, c0, count) = tree_children(&b.tree[i * 2..i * 2 + 2]);
            if count == 0 {
                continue;
            }
            let row = rows.get(cp as usize).ok_or_else(|| anyhow!("page {p} node {i}: children in page {cp}, past the last"))?;
            if (cp as usize) < p {
                bail!("page {p} node {i}: children in an earlier page {cp}");
            }
            if c0 + count > row.count {
                bail!("page {p} node {i}: children {c0}+{count} past page {cp}'s {} nodes", row.count);
            }
            let me = rows[p].first + i as u32;
            for j in 0..count {
                let c = (row.first + c0 + j) as usize;
                if parent[c] != u32::MAX {
                    bail!("node {c} has two parents");
                }
                if c as u32 <= me {
                    bail!("node {c} comes before its parent {me}");
                }
                parent[c] = me;
            }
        }
    }
    let roots = parent.iter().filter(|&&p| p == u32::MAX).count();
    if roots != 1 || parent[0] != u32::MAX {
        bail!("{roots} roots (the root is node 0 of page 0)");
    }
    Ok(parent)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::athc::{ln_scale, pack_halves, AthcFile, AthcHeader, ExtraHeader};
    use crate::athc_lod_cut::NodeBounds;

    /// A small binary tree over `leaves` leaves in a row: merges pair
    /// neighbours level by level (children before parents), every node a
    /// round gaussian of opacity 0.5. Returns block, parents, first leaves.
    fn tree(leaves: usize) -> (AthcBlock, Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut parent = Vec::new();
        let mut size = Vec::new();
        let mut start = Vec::new();
        let mut centre: Vec<f32> = Vec::new();
        let mut level: Vec<u32> = (0..leaves as u32).collect();
        for k in 0..leaves {
            parent.push(k as u32);
            size.push(1u32);
            start.push(k as u32);
            centre.push(k as f32 * 0.01);
        }
        while level.len() > 1 {
            let mut next = Vec::new();
            for pair in level.chunks(2) {
                if pair.len() == 1 {
                    next.push(pair[0]);
                    continue;
                }
                let k = parent.len() as u32;
                parent.push(k);
                for &c in pair {
                    parent[c as usize] = k;
                }
                size.push(pair.iter().map(|&c| size[c as usize]).sum());
                start.push(start[pair[0] as usize]);
                centre.push((centre[pair[0] as usize] + centre[pair[1] as usize]) / 2.0);
                next.push(k);
            }
            level = next;
        }
        let n = parent.len();
        let mut b = AthcBlock { n, ..Default::default() };
        for k in 0..n {
            b.positions.extend_from_slice(&[centre[k], 0.0, 0.0, 0.5]);
            let s = 0.004 * (size[k] as f32).sqrt();
            b.shape.extend_from_slice(&[crate::athc::encode_quaternion([0.0, 0.0, 0.0, 1.0]), pack_halves(ln_scale(s), ln_scale(s)), pack_halves(ln_scale(s * 0.1), 0.5), pack_halves(0.5, 0.5)]);
        }
        (b, parent, start, size)
    }

    /// Bounds as athenea's: own sphere, eps growing with the leaves under.
    fn bounds(b: &AthcBlock, parent: &[u32], size: &[u32]) -> Vec<NodeBounds> {
        let n = b.n;
        let mut nb: Vec<NodeBounds> = (0..n).map(|k| NodeBounds::own(b, k, true)).collect();
        for k in 0..n {
            if size[k] > 1 {
                nb[k].eps = 0.001 * size[k] as f32;
            }
        }
        for k in 0..n {
            let p = parent[k] as usize;
            if p != k {
                let c = nb[k];
                nb[p].cover(&c);
            }
        }
        nb
    }

    /// athenea's paging (lod_pages.slang treeBandTable): by (band of the
    /// parent's error, first leaf), the whole order reversed into pages.
    fn her_pages(parent: &[u32], nb: &[NodeBounds], start: &[u32], per: usize) -> Vec<u32> {
        let n = parent.len();
        let mut keys: Vec<(u32, u32, usize)> = (0..n)
            .map(|k| {
                let p = parent[k] as usize;
                let e = if p == k { ROOT_PARENT_ERROR } else { nb[p].eps };
                (band_of(e), start[k], k)
            })
            .collect();
        keys.sort();
        let mut page = vec![0u32; n];
        for (at, &(_, _, k)) in keys.iter().enumerate() {
            page[k] = ((n - 1 - at) / per) as u32;
        }
        page
    }

    #[test]
    fn pages_keep_every_run_of_children_in_one_page() {
        let (b, parent, start, size) = tree(1000);
        let nb = bounds(&b, &parent, &size);
        let page = her_pages(&parent, &nb, &start, 64);
        let ids: Vec<u32> = (0..b.n as u32).collect();
        let input = PageInput { block: &b, parent: &parent, page: &page, bounds: &nb, ids: &ids, first_leaf: &start, sh_words: 0, page_box: PageBox::Parents };
        let paged = pack_pages(&input).unwrap();
        assert!(paged.report.split_runs > 0, "the test should split some runs: {:?}", paged.report);
        let up = check_paged(&paged.pages, &paged.rows).unwrap();
        // The same tree: each node's parent maps back to its parent.
        for (i, &k) in paged.order.iter().enumerate() {
            let p = up[i];
            if p == u32::MAX {
                assert_eq!(parent[k as usize], k);
            } else {
                assert_eq!(paged.order[p as usize], parent[k as usize]);
            }
        }
        // Every node once, pages in band order (coarsest first).
        let mut seen = paged.order.clone();
        seen.sort();
        assert_eq!(seen, ids);
        for w in paged.rows.windows(2) {
            assert!(w[0].band >= w[1].band || w[0].eps_min >= w[1].eps_max * 0.5, "{:?} then {:?}", w[0], w[1]);
        }
        // A page's box holds its nodes' parents' spheres, so a needed node
        // (its parent's error past tau) is always in a needed page.
        let ppu = 1000.0;
        for eye in [[-1.0f32, 0.3, 0.0], [5.0, 0.0, 0.2], [2.0, 0.01, 0.0]] {
            for tau in [0.5f32, 1.0, 4.0] {
                for (i, &k) in paged.order.iter().enumerate() {
                    let p = parent[k as usize] as usize;
                    if p == k as usize {
                        continue;
                    }
                    let (e, r) = unpack_error(nb[p].words()[0]);
                    let c = nb[p].centre;
                    let d = ((c[0] - eye[0]).powi(2) + (c[1] - eye[1]).powi(2) + (c[2] - eye[2]).powi(2)).sqrt();
                    if e * ppu / (d - r).max(1e-3) > tau {
                        let pg = paged.rows.iter().position(|row| (row.first..row.first + row.count).contains(&(i as u32))).unwrap();
                        assert!(paged.rows[pg].needed(eye, ppu, tau, 1e-3), "node {k} needed, page {pg} not");
                    }
                }
            }
        }
    }

    #[test]
    fn a_forest_gets_a_root_at_the_head_of_page_0() {
        let (b, mut parent, start, size) = tree(64);
        // Cut the top: the root's children become roots.
        let top = parent.len() - 1;
        let kids: Vec<usize> = (0..top).filter(|&k| parent[k] as usize == top).collect();
        for k in kids {
            parent[k] = k as u32;
        }
        let b = b.slice(0, top);
        parent.truncate(top);
        let nb = bounds(&b, &parent, &size[..top]);
        let page = her_pages(&parent, &nb, &start[..top], 16);
        let ids: Vec<u32> = (0..b.n as u32).collect();
        let input = PageInput { block: &b, parent: &parent, page: &page, bounds: &nb, ids: &ids, first_leaf: &start[..top], sh_words: 0, page_box: PageBox::Own };
        let paged = pack_pages(&input).unwrap();
        assert!(paged.report.synthetic_root && paged.report.roots == 2);
        assert_eq!(paged.order[0], u32::MAX);
        check_paged(&paged.pages, &paged.rows).unwrap();
        // The root covers both: its error the larger, its sphere round theirs.
        let (e, r) = unpack_error(paged.pages[0].lod_error[0]);
        assert!(e >= nb.iter().map(|b| b.eps).fold(0.0, f32::max));
        assert!(r > 0.3, "{r}");
        assert_eq!(paged.rows[0].eps_max, ROOT_PARENT_ERROR);
    }

    /// A forest: `tree(leaves)` with its top `cut` merges removed (the last
    /// indices), their children roots.
    fn forest(leaves: usize, cut: usize) -> (AthcBlock, Vec<u32>, Vec<u32>, Vec<u32>) {
        let (b, mut parent, start, size) = tree(leaves);
        let keep = parent.len() - cut;
        for k in 0..keep {
            if parent[k] as usize >= keep {
                parent[k] = k as u32;
            }
        }
        parent.truncate(keep);
        (b.slice(0, keep), parent, start[..keep].to_vec(), size[..keep].to_vec())
    }

    /// The error cut over a paged tree as the traversal makes it (refine a
    /// node with children while eps / max(d - rho, near) passes tau), each
    /// cut node by its source index (`order`; the synthetic root u32::MAX),
    /// sorted; and whether a synthetic node (source >= `n`) was ever cut.
    fn cpu_cut(paged: &PagedTree, n: usize, eye: [f32; 3], tau: f32) -> (Vec<u32>, bool) {
        let mut out = Vec::new();
        let mut synthetic_drawn = false;
        let mut stack = vec![0usize];
        let node = |i: usize| -> (usize, usize) {
            let p = paged.rows.iter().position(|r| (r.first..r.first + r.count).contains(&(i as u32))).unwrap();
            (p, i - paged.rows[p].first as usize)
        };
        while let Some(i) = stack.pop() {
            let (p, at) = node(i);
            let b = &paged.pages[p];
            let (cp, c0, count) = tree_children(&b.tree[at * 2..at * 2 + 2]);
            let (eps, rho) = unpack_error(b.lod_error[at * 2]);
            let c = &b.positions[at * 4..at * 4 + 3];
            let d = ((c[0] - eye[0]).powi(2) + (c[1] - eye[1]).powi(2) + (c[2] - eye[2]).powi(2)).sqrt();
            if count > 0 && eps / (d - rho).max(1e-3) > tau {
                let first = paged.rows[cp as usize].first;
                for j in 0..count {
                    stack.push((first + c0 + j) as usize);
                }
            } else {
                let k = paged.order[i];
                if k != u32::MAX && k as usize >= n {
                    synthetic_drawn = true;
                }
                out.push(k);
            }
        }
        out.sort_unstable();
        (out, synthetic_drawn)
    }

    #[test]
    fn a_synthetic_hierarchy_over_the_roots_keeps_the_flat_cut() {
        // ~400 roots (lone leaves and small trees), two levels at fanout 8.
        let (b, parent, start, size) = forest(1024, 400);
        let n = b.n;
        let nb = bounds(&b, &parent, &size);
        let page = her_pages(&parent, &nb, &start, 64);
        let ids: Vec<u32> = (0..n as u32).collect();
        let input = PageInput { block: &b, parent: &parent, page: &page, bounds: &nb, ids: &ids, first_leaf: &start, sh_words: 0, page_box: PageBox::Parents };
        let flat = pack_pages_with(&input, 0).unwrap();
        assert!(flat.report.roots > 200, "{:?}", flat.report);
        assert_eq!(flat.report.synthetic_nodes, 0);
        for fanout in [8usize, 32, 128] {
            let hier = pack_pages_with(&input, fanout).unwrap();
            assert!(hier.report.synthetic_nodes > 0);
            assert!(hier.rows[0].count as usize <= fanout + 1, "page 0 of {}", hier.rows[0].count);
            check_paged(&hier.pages, &hier.rows).unwrap();
            // Every source node once.
            let mut seen: Vec<u32> = hier.order.iter().copied().filter(|&k| k != u32::MAX && (k as usize) < n).collect();
            seen.sort_unstable();
            assert_eq!(seen, ids);
            // Synthetic nodes: no cone, never occluded, an infinite error,
            // a sphere round their children.
            for (p, blk) in hier.pages.iter().enumerate() {
                for at in 0..blk.n {
                    let k = hier.order[(hier.rows[p].first as usize) + at];
                    if k == u32::MAX || (k as usize) < n {
                        continue;
                    }
                    assert_eq!(blk.lod_error[at * 2 + 1], NO_CONE as u32);
                    assert!(unpack_error(blk.lod_error[at * 2]).0.is_infinite());
                    let (cp, c0, count) = tree_children(&blk.tree[at * 2..at * 2 + 2]);
                    assert!(cp as usize > p && count > 0);
                    let (_, rho) = unpack_error(blk.lod_error[at * 2]);
                    let c = &blk.positions[at * 4..at * 4 + 3];
                    let kids = &hier.pages[cp as usize];
                    for j in (c0..c0 + count).map(|j| j as usize) {
                        let kc = &kids.positions[j * 4..j * 4 + 3];
                        let (_, kr) = unpack_error(kids.lod_error[j * 2]);
                        let d = ((kc[0] - c[0]).powi(2) + (kc[1] - c[1]).powi(2) + (kc[2] - c[2]).powi(2)).sqrt();
                        assert!(d <= rho * 1.001 + 1e-6 || kr == 0.0 && d <= rho * 1.001 + 1e-6, "child outside: {d} > {rho}");
                    }
                }
            }
            // The page table still has every needed node's page needed.
            for row in &hier.rows {
                assert!(row.eps_max.is_finite() && row.count > 0);
            }
            // The same cut as flat, everywhere (cuts from the root alone to every leaf).
            let mut sizes = std::collections::BTreeSet::new();
            for eye in [[-1.0f32, 0.3, 0.0], [5.0, 0.0, 0.2], [2.0, 0.01, 0.0], [40.0, 30.0, 0.0], [5.12, 0.0, 0.0]] {
                for tau in [1e-4f32, 1e-3, 1e-2, 0.1, 10.0] {
                    let (a, _) = cpu_cut(&flat, n, eye, tau);
                    let (h, drawn) = cpu_cut(&hier, n, eye, tau);
                    assert!(!drawn, "a synthetic node in the cut (fanout {fanout}, eye {eye:?}, tau {tau})");
                    assert_eq!(a, h, "fanout {fanout}, eye {eye:?}, tau {tau}");
                    sizes.insert(a.len());
                }
            }
            assert!(sizes.len() >= 5 && sizes.contains(&1) && sizes.iter().any(|&s| s > 900), "{sizes:?}");
        }
    }

    #[test]
    fn a_page_of_only_roots_is_dropped_not_left_empty() {
        // athenea's 5120 forests (thousands of lone roots): a coarse page
        // of hers holds nothing but roots, which all go to page 0.
        let (b, mut parent, start, size) = tree(64);
        let top = parent.len() - 1;
        let kids: Vec<usize> = (0..top).filter(|&k| parent[k] as usize == top).collect();
        for &k in &kids {
            parent[k] = k as u32;
        }
        let b = b.slice(0, top);
        parent.truncate(top);
        let nb = bounds(&b, &parent, &size[..top]);
        let her = her_pages(&parent, &nb, &start[..top], 16);
        // Her page 1: the roots alone; the rest one page later.
        let page: Vec<u32> = (0..top).map(|k| if kids.contains(&k) { 1 } else if her[k] == 0 { 0 } else { her[k] + 1 }).collect();
        let ids: Vec<u32> = (0..b.n as u32).collect();
        let input = PageInput { block: &b, parent: &parent, page: &page, bounds: &nb, ids: &ids, first_leaf: &start[..top], sh_words: 0, page_box: PageBox::Parents };
        let paged = pack_pages(&input).unwrap();
        assert!(paged.rows.iter().all(|r| r.count > 0));
        assert_eq!(paged.report.pages, paged.rows.len());
        check_paged(&paged.pages, &paged.rows).unwrap();
        let mut seen = paged.order.clone();
        seen.retain(|&k| k != u32::MAX);
        seen.sort();
        assert_eq!(seen, ids);
    }

    /// A paged tree of `leaves` leaves (pages of `per`) whose nodes carry
    /// every stream the compact tier reads (normals, pbr, lobes, curvature,
    /// shadow bits, a full transfer of 112), deterministic; and its file.
    pub(crate) fn sample_relit_paged(leaves: usize, per: usize) -> (PagedTree, AthcFile) {
        sample_relit_paged_spread(leaves, per, 0.0)
    }

    /// `sample_relit_paged` with the row bent in 3D and stretched `spread`
    /// times along x (0: as `tree` lays it).
    pub(crate) fn sample_relit_paged_spread(leaves: usize, per: usize, spread: f32) -> (PagedTree, AthcFile) {
        let (mut b, parent, start, size) = tree(leaves);
        if spread > 0.0 {
            for i in 0..b.n {
                let x = b.positions[4 * i];
                b.positions[4 * i] = spread * x;
                b.positions[4 * i + 1] = (x * 7.0).sin() * 0.2;
                b.positions[4 * i + 2] = (x * 5.0).cos() * 0.1;
                // Gaussians as much bigger: a few pixels from a few metres.
                let sh = &mut b.shape[4 * i..4 * i + 4];
                let g = (3.0 * spread).ln();
                sh[1] = pack_halves(crate::athc::low_half(sh[1]) + g, crate::athc::high_half(sh[1]) + g);
                sh[2] = pack_halves(crate::athc::low_half(sh[2]) + g, crate::athc::high_half(sh[2]));
            }
        }
        let nb = bounds(&b, &parent, &size);
        let page = her_pages(&parent, &nb, &start, per);
        let ids: Vec<u32> = (0..b.n as u32).collect();
        let mut r = 0x9e3779b9u32;
        let mut rand = move || {
            r ^= r << 13;
            r ^= r >> 17;
            r ^= r << 5;
            r
        };
        for k in 0..b.n {
            let t = k as f32 * 0.37;
            b.normals.push(crate::athc::pack_normal([t.sin() * 0.3, t.cos() * 0.3, 0.9]));
            b.pbr.push(rand() % 5);
            b.lobes.extend_from_slice(&[rand() % 3, 0, 7]);
            b.curvature.extend_from_slice(&[pack_halves(0.5 * t.sin(), -0.02), pack_halves(3.0 + t.cos(), 0.0)]);
            // A few dozen distinct masks, so the dictionary keeps them exactly.
            let m = rand() % 40;
            b.shadow_bits.extend((0..8).map(|w| if w == m % 8 { 0xffff_0000 ^ m } else { m.wrapping_mul(2654435761) ^ w }));
            for v in 0..56 {
                let x = 0.1 * ((k * 7 + v) as f32 * 0.13).sin();
                b.transfer.push(pack_halves(x, 0.5 * x));
            }
        }
        let input = PageInput { block: &b, parent: &parent, page: &page, bounds: &nb, ids: &ids, first_leaf: &start, sh_words: 0, page_box: PageBox::Parents };
        let paged = pack_pages(&input).unwrap();
        let file = AthcFile {
            header: AthcHeader { flags: crate::athc::FLAG_PAGED | crate::athc::FLAG_NODE_COVERAGE, bounds_min: [0.0; 3], bounds_max: [3.0, 0.0, 0.0], ..Default::default() },
            extra: ExtraHeader { transfer_count: 112, ..Default::default() },
            levels: Vec::new(),
            starts: Vec::new(),
            chunks: paged.pages.clone(),
            occlusion_codebook: Vec::new(),
        };
        (paged, file)
    }

    /// A small paged tree written as v3 (gzip, byte planes), for decoder tests.
    pub(crate) fn sample_paged_file() -> Vec<u8> {
        use crate::athc_v3::{write_v3_paged, COMPRESSION_GZIP, ENCODING_BYTE_PLANES};
        let (b, parent, start, size) = tree(300);
        let nb = bounds(&b, &parent, &size);
        let page = her_pages(&parent, &nb, &start, 40);
        let ids: Vec<u32> = (0..b.n as u32).collect();
        let mut b = b;
        b.sh = vec![0; b.n];
        let input = PageInput { block: &b, parent: &parent, page: &page, bounds: &nb, ids: &ids, first_leaf: &start, sh_words: 1, page_box: PageBox::Parents };
        let paged = pack_pages(&input).unwrap();
        let file = AthcFile {
            header: AthcHeader { flags: crate::athc::FLAG_PAGED | crate::athc::FLAG_NODE_COVERAGE, bounds_min: [0.0; 3], bounds_max: [3.0, 0.0, 0.0], ..Default::default() },
            extra: ExtraHeader::default(),
            levels: Vec::new(),
            starts: Vec::new(),
            chunks: paged.pages,
            occlusion_codebook: Vec::new(),
        };
        write_v3_paged(&file, &paged.rows, COMPRESSION_GZIP, &|_| ENCODING_BYTE_PLANES, None, None).unwrap().0
    }

    #[test]
    fn a_paged_file_comes_back_from_v3_with_its_table() {
        use crate::athc_v3::{parse_v3, read_v3, write_v3_paged, COMPRESSION_GZIP, ENCODING_BYTE_PLANES};
        let (mut b, parent, start, size) = tree(300);
        // A second opacity on every merge (LODM), none on the leaves.
        b.lod_mass = (0..b.n).map(|k| if size[k] > 1 { (0.5 + 0.001 * k as f32).to_bits() } else { 0 }).collect();
        let nb = bounds(&b, &parent, &size);
        let page = her_pages(&parent, &nb, &start, 40);
        let ids: Vec<u32> = (0..b.n as u32).collect();
        let input = PageInput { block: &b, parent: &parent, page: &page, bounds: &nb, ids: &ids, first_leaf: &start, sh_words: 0, page_box: PageBox::Parents };
        let paged = pack_pages(&input).unwrap();
        let file = AthcFile {
            header: AthcHeader { flags: crate::athc::FLAG_PAGED | crate::athc::FLAG_NODE_COVERAGE, bounds_min: [0.0; 3], bounds_max: [3.0, 0.0, 0.0], ..Default::default() },
            extra: ExtraHeader::default(),
            levels: Vec::new(),
            starts: Vec::new(),
            chunks: paged.pages.clone(),
            occlusion_codebook: Vec::new(),
        };
        for encoding in [0, ENCODING_BYTE_PLANES] {
            let (bytes, _) = write_v3_paged(&file, &paged.rows, COMPRESSION_GZIP, &|_| encoding, None, None).unwrap();
            let layout = parse_v3(&bytes).unwrap();
            assert_eq!(layout.pages.len(), paged.rows.len());
            assert!(layout.sections.iter().any(|s| s.id == crate::athc_v3::SectionId::Tree));
            assert!(layout.header.has(crate::athc::FLAG_NODE_COVERAGE) && layout.header.has(crate::athc::FLAG_LOD_ERROR));
            for (r, (row, e)) in layout.pages.iter().zip(&layout.blocks).enumerate() {
                let mut want = paged.rows[r];
                want.fill_ranges(e, &layout.sections);
                assert_eq!(*row, want);
                assert_eq!(e.first, row.first);
                assert!(row.tier_bytes[0] > 0 && row.offset == e.spans[0].offset);
            }
            let back = read_v3(&bytes).unwrap();
            assert_eq!(back.chunks.len(), paged.pages.len());
            for (a, b) in back.chunks.iter().zip(&paged.pages) {
                assert_eq!(a.tree, b.tree);
                assert!(a.lod_error == b.lod_error || a.lod_error.is_empty() && b.lod_error.iter().all(|&w| w == 0));
                assert_eq!(a.positions, b.positions);
                assert_eq!(a.tail, b.tail);
                if !a.lod_mass.is_empty() || b.lod_mass.iter().any(|&w| w != 0) {
                    assert_eq!(a.lod_mass, b.lod_mass);
                }
            }
            assert!(layout.header.has(crate::athc::FLAG_LOD_MASS));
            assert_eq!(layout.sections.iter().find(|s| s.id == crate::athc_v3::SectionId::LodMass).map(|s| s.tier), Some(1));
            assert!(back.chunks.iter().any(|c| !c.lod_mass.is_empty()));
            check_paged(&back.chunks, &layout.pages).unwrap();
        }
    }
}
