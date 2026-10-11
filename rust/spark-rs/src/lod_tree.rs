use std::{array, cell::{Ref, RefCell}, collections::BinaryHeap, rc::Rc};

use ahash::{AHashMap, AHashSet};
use glam::{Vec3, Vec3A};
use half::f16;
use itertools::izip;
use js_sys::{Array, Object, Reflect, Uint32Array, Uint8Array};
use ordered_float::OrderedFloat;
use wasm_bindgen::prelude::*;

const MAX_SPLAT_CHUNK: usize = 65536;

/// Words a node of the LoD error cut (spark-lib athc_lod_cut): f16 eps | f16
/// rho, f16 theta | the cone's octahedral 8+8-bit axis.
pub const LOD_ERROR_WORDS: usize = 2;
/// athenea's `near` in `projectedError`.
const ERROR_NEAR: f32 = 1.0e-3;

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
struct FourHeap<T: Ord> {
    data: Vec<T>,
}

#[allow(dead_code)]
impl<T: Ord> FourHeap<T> {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn push(&mut self, value: T) {
        self.data.push(value);
        let mut index = self.data.len() - 1;
        while index > 0 {
            let parent = (index - 1) / 4;
            if self.data[parent] >= self.data[index] {
                break;
            }
            self.data.swap(parent, index);
            index = parent;
        }
    }

    fn peek(&self) -> Option<&T> {
        self.data.first()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn pop(&mut self) -> Option<T> {
        let last = self.data.pop()?;
        if self.data.is_empty() {
            return Some(last);
        }

        let root = std::mem::replace(&mut self.data[0], last);
        let len = self.data.len();
        let mut index = 0usize;
        loop {
            let child0 = index * 4 + 1;
            if child0 >= len {
                break;
            }

            let child_end = (child0 + 4).min(len);
            let mut max_child = child0;
            for child in (child0 + 1)..child_end {
                if self.data[child] > self.data[max_child] {
                    max_child = child;
                }
            }

            if self.data[index] >= self.data[max_child] {
                break;
            }
            self.data.swap(index, max_child);
            index = max_child;
        }

        Some(root)
    }

    fn drain(&mut self) -> std::vec::Drain<'_, T> {
        self.data.drain(..)
    }

    fn clear(&mut self) {
        self.data.clear();
    }
}

type Frontier<T> = BinaryHeap<T>;

#[derive(Debug, Clone, Default)]
struct LodSplat {
    center: [f16; 3],
    size: f16,
    child_start: u32,
    child_count: u16,
    /// The error cut's bounds (0 0: none, the size cut): eps and rho in
    /// object units, theta (infinite: never culled) and its axis.
    eps: f16,
    rho: f16,
    theta: f16,
    cone_axis: u16,
    /// athenea's occlusion mask: its tree's codebook entry + 1, 0 none.
    occlusion: u16,
    /// A quantized compact tree (thread CT): the largest centre error of the
    /// node's page, added to its sphere wherever the cut and the cull read
    /// it, so the quantized cut only refines more and culls less.
    margin: f32,
}

impl LodSplat {
    fn new_f16(center: [f16; 3], size: f16, child_start: u32, child_count: u16) -> Self {
        Self { center, size, child_start, child_count, ..Default::default() }
    }

    fn set_error(&mut self, words: [u32; 2]) {
        self.eps = f16::from_bits((words[0] & 0xffff) as u16);
        self.rho = f16::from_bits((words[0] >> 16) as u16);
        self.theta = f16::from_bits((words[1] & 0xffff) as u16);
        self.cone_axis = (words[1] >> 16) as u16;
    }

    fn has_error(&self) -> bool {
        self.rho.to_bits() != 0 || self.eps.to_bits() != 0
    }

    /// Wholly back-facing from `eye` (athc_lod_cut::back_facing): n . v <
    /// -sin(theta + asin(rho / d)).
    fn back_facing(&self, eye: Vec3A) -> bool {
        let theta = self.theta.to_f32();
        // A leaf (no error) culls with its own cone when it has one (a
        // compact v3 tree's closed parts, thread CS): its sphere 0.
        // ([0, 0] LODE words, theta's bits 0: no cone.)
        if !theta.is_finite() || (!self.has_error() && (self.child_count > 0 || self.theta.to_bits() == 0)) {
            return false;
        }
        let v = eye - self.center();
        let d = v.length();
        let rho = self.rho.to_f32() + self.margin;
        if d <= rho {
            return false;
        }
        let spread = theta + (rho / d).asin();
        if spread >= std::f32::consts::FRAC_PI_2 {
            return false;
        }
        let bits = self.cone_axis;
        let f = [(bits & 255) as f32 / 255.0 * 2.0 - 1.0, (bits >> 8) as f32 / 255.0 * 2.0 - 1.0];
        let mut n = [f[0], f[1], 1.0 - f[0].abs() - f[1].abs()];
        let t = (-n[2]).clamp(0.0, 1.0);
        n[0] += if n[0] >= 0.0 { -t } else { t };
        n[1] += if n[1] >= 0.0 { -t } else { t };
        let n = Vec3A::from_array(n).normalize();
        n.dot(v) / d < -spread.sin()
    }

    #[allow(dead_code)]
    fn new(center: Vec3, size: f32, child_start: u32, child_count: u16) -> Self {
        let center = center.to_array().map(f16::from_f32);
        let size = f16::from_f32(size);
        Self::new_f16(center, size, child_start, child_count)
    }

    fn center(&self) -> Vec3A {
        Vec3A::from_array(self.center.map(|x| x.to_f32()))
    }

    fn size(&self) -> f32 {
        self.size.to_f32()
    }
}

// #[derive(Debug, Clone, Default)]
// struct LodSplat {
//     center: Vec3,
//     size: f32,
//     child_start: u32,
//     child_count: u16,
// }

// impl LodSplat {
//     fn new_f16(center: [f16; 3], size: f16, child_start: u32, child_count: u16) -> Self {
//         let center = Vec3::from_array(center.map(|x| x.to_f32()));
//         let size = size.to_f32();
//         Self::new(center, size, child_start, child_count)
//     }

//     fn new(center: Vec3, size: f32, child_start: u32, child_count: u16) -> Self {
//         Self { center, size, child_start, child_count }
//     }

//     fn center(&self) -> Vec3A {
//         self.center.to_vec3a()
//     }

//     fn size(&self) -> f32 {
//         self.size
//     }
// }

/// A node of a compact tree (thread CQ, spark-lib athc_compact
/// `TraversalNode`): 14 bytes where `LodSplat` takes 24, what the error cut
/// reads and no more -- the centre (f16 x 3), LODE's eps and rho (f16; a
/// merge without LODE: its size in eps, rho's sign bit set), the children's
/// chunk (u16) and their place (12 bits) and count (4 bits, 15: in the
/// tree's `wide` map; 0 a leaf). No back-face cone, no occlusion, no size
/// for a merge with LODE (so no `errorSizeK` guard): a compact traversal
/// draws exactly the nodes of the plain error cut.
///
/// v3 (thread CS) adds 6 bytes: LODE's cone word (lo, hi; a closed part's
/// leaf its own, theta 0) and athenea's occlusion index -- the back-face
/// cull of closed parts and the occlusion cull. v2 trees: NO_CONE, 0.
#[derive(Debug, Clone, Copy, Default)]
struct CompactLodNode([u16; 10]);

const COMPACT_WIDE: u16 = 15;
const COMPACT_SIZE_MODE: u16 = 0x8000;

#[derive(Debug, Clone)]
enum LodNodes {
    Full(Vec<LodSplat>),
    Compact { nodes: Vec<CompactLodNode>, wide: AHashMap<u32, u16>, margins: Vec<f32>, slot: u32 },
    /// 12-byte records (thread CT, QUANT_RECORD_BITS): each pool page's
    /// quantization, and the nodes that do not fit kept whole.
    /// A pool page's records in a block of its own (a page's landing never
    /// moves the others: no copy of the whole tree, no doubled peak).
    Quant { nodes: Vec<Box<[QuantNode]>>, pages: Vec<PageQuant>, full: AHashMap<u32, CompactLodNode>, wide: AHashMap<u32, u16>, slot: u32 },
}

/// A compact node's LodSplat from its 10 words (and its wide count).
fn compact_splat(w: [u16; 10], wide: &AHashMap<u32, u16>, i: usize) -> LodSplat {
    let code = w[6] >> 12;
    let count = if code == COMPACT_WIDE { wide.get(&(i as u32)).copied().unwrap_or(0) } else { code };
    let center = [f16::from_bits(w[0]), f16::from_bits(w[1]), f16::from_bits(w[2])];
    let child_start = if count > 0 { (w[5] as u32) << 16 | (w[6] & 0xfff) as u32 } else { 0 };
    let mut s = LodSplat::new_f16(center, f16::ZERO, child_start, count);
    if w[4] & COMPACT_SIZE_MODE != 0 {
        s.size = f16::from_bits(w[3]);
        s.theta = f16::INFINITY;
    } else {
        s.eps = f16::from_bits(w[3]);
        s.rho = f16::from_bits(w[4]);
        s.theta = f16::from_bits(w[7]);
        s.cone_axis = w[8];
    }
    s.occlusion = w[9];
    s
}

impl Default for LodNodes {
    fn default() -> Self {
        LodNodes::Full(Vec::new())
    }
}

impl LodNodes {
    fn get(&self, i: usize) -> LodSplat {
        match self {
            LodNodes::Full(v) => v[i].clone(),
            LodNodes::Compact { nodes, wide, margins, slot } => {
                let mut s = compact_splat(nodes[i].0, wide, i);
                s.margin = margins.get(i / *slot as usize).copied().unwrap_or(0.0);
                s
            }
            LodNodes::Quant { nodes, pages, full, wide, slot } => {
                let p = &pages[i / *slot as usize];
                match full.get(&(i as u32)) {
                    Some(n) => {
                        let mut s = compact_splat(n.0, wide, i);
                        s.margin = p.margin;
                        s
                    }
                    None => unpack_quant(&nodes[i / *slot as usize][i % *slot as usize], p, &QUANT_RECORD_BITS),
                }
            }
        }
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        match self {
            LodNodes::Full(v) => v.len(),
            LodNodes::Compact { nodes, .. } => nodes.len(),
            LodNodes::Quant { nodes, slot, .. } => nodes.len() * *slot as usize,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct LodTree {
    splats: Rc<RefCell<LodNodes>>,
    page_to_chunk: Vec<u32>,
    chunk_to_page: Vec<u32>,
    /// athenea's occlusion codebook (8 words an entry) and the asset's box.
    occlusion: Option<(Vec<u32>, [f32; 6])>,
    /// Splats a physical page (pool slot) holds: 65 536, or fewer for a
    /// pager of small pages (athenea's error-band pages, thread CK). The
    /// virtual index space keeps 65 536 a chunk (`child >> 16`); a resident
    /// chunk's nodes sit at `page * slot + (child & 0xffff)`.
    slot: u32,
}

/// Whether node `s` is hidden from `eye` by athenea's occlusion mask
/// (`decimate --occlusion-mask`): the eye outside the asset's box, and the
/// four cells toward it on the 16 x 16 octahedral grid (splatCellsOpen's:
/// octEncode(dir) x 16 - 0.5, the cells clamped, those of weight 0 left out)
/// all closed in its entry. `dir` from the node toward the eye, in the
/// asset's own frame (the tree's object space).
///
/// A quantized node (`margin` > 0, thread CU): its exact centre is anywhere
/// within `margin` of the stored one, so it is hidden only if it is hidden
/// from every such centre: every cell any direction from that box toward the
/// eye reads is closed (`occluded_box`). It never culls a node the exact
/// centre keeps.
fn occluded(s: &LodSplat, eye: Vec3A, occlusion: &(Vec<u32>, [f32; 6])) -> bool {
    let (codebook, b) = occlusion;
    if s.occlusion == 0 {
        return false;
    }
    let entry = (s.occlusion - 1) as usize * 8;
    if entry + 8 > codebook.len() {
        return false;
    }
    let inside = eye.x >= b[0] && eye.y >= b[1] && eye.z >= b[2] && eye.x <= b[3] && eye.y <= b[4] && eye.z <= b[5];
    if inside {
        return false;
    }
    let mask = &codebook[entry..entry + 8];
    if s.margin > 0.0 {
        return occluded_box(mask, eye - s.center(), s.margin);
    }
    occluded_dir(mask, eye - s.center())
}

/// Whether cell `bit` of an entry is open.
fn cell_open(mask: &[u32], cx: i32, cy: i32) -> bool {
    let bit = cy as usize * 16 + cx as usize;
    (mask[bit >> 5] >> (bit & 31)) & 1 != 0
}

/// The exact rule for the direction `d` (node toward eye).
fn occluded_dir(mask: &[u32], d: Vec3A) -> bool {
    let l1 = d.x.abs() + d.y.abs() + d.z.abs();
    if l1 <= 1.0e-20 {
        return false;
    }
    let n = d / l1;
    let p = if n.z >= 0.0 {
        [n.x, n.y]
    } else {
        [(1.0 - n.y.abs()) * if n.x >= 0.0 { 1.0 } else { -1.0 }, (1.0 - n.x.abs()) * if n.y >= 0.0 { 1.0 } else { -1.0 }]
    };
    let at = [(p[0] * 0.5 + 0.5) * 16.0 - 0.5, (p[1] * 0.5 + 0.5) * 16.0 - 0.5];
    let lo = [at[0].floor(), at[1].floor()];
    let f = [at[0] - lo[0], at[1] - lo[1]];
    for k in 0..4u32 {
        let (kx, ky) = ((k & 1) as f32, (k >> 1) as f32);
        let w = (if kx > 0.0 { f[0] } else { 1.0 - f[0] }) * (if ky > 0.0 { f[1] } else { 1.0 - f[1] });
        if w <= 0.0 {
            continue;
        }
        let cx = ((lo[0] + kx) as i32).clamp(0, 15);
        let cy = ((lo[1] + ky) as i32).clamp(0, 15);
        if cell_open(mask, cx, cy) {
            return false;
        }
    }
    true
}

/// The conservative rule for a centre known within `margin` of the stored
/// one: `d` (stored centre toward eye) anywhere in the box d +- margin. The
/// box is split at the coordinate planes; in each closed orthant both
/// octahedral coordinates are monotone in every component of d, so their
/// range over the sub-box is that of its 8 corners, and every cell the exact
/// rule may read from it lies in floor(min) ..= ceil(max) of the grid's
/// coordinate (both bilinear cells, clamped). All those cells closed, in
/// every sub-box: hidden. A box holding the eye's own point: never hidden.
fn occluded_box(mask: &[u32], d: Vec3A, margin: f32) -> bool {
    // A hair over the margin (f32 rounding of the exact rule's d).
    let m = margin * 1.001 + 1.0e-6;
    let lo = [d.x - m, d.y - m, d.z - m];
    let hi = [d.x + m, d.y + m, d.z + m];
    if (0..3).all(|k| lo[k] <= 0.0 && hi[k] >= 0.0) {
        return false;
    }
    // Each axis's sub-intervals by sign: + takes [max(lo, 0), hi] (the exact
    // rule's sign of 0 is +), - takes [lo, min(hi, 0)].
    let parts = |k: usize| -> [(f32, f32, f32); 2] {
        [
            if hi[k] >= 0.0 { (1.0, lo[k].max(0.0), hi[k]) } else { (0.0, 0.0, 0.0) },
            if lo[k] < 0.0 { (-1.0, lo[k], hi[k].min(0.0)) } else { (0.0, 0.0, 0.0) },
        ]
    };
    let (px, py, pz) = (parts(0), parts(1), parts(2));
    for &(sx, x0, x1) in &px {
        if sx == 0.0 {
            continue;
        }
        for &(sy, y0, y1) in &py {
            if sy == 0.0 {
                continue;
            }
            for &(sz, z0, z1) in &pz {
                if sz == 0.0 {
                    continue;
                }
                let (mut u0, mut u1, mut v0, mut v1) = (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
                for c in 0..8u32 {
                    let a = if c & 1 == 0 { x0 } else { x1 }.abs();
                    let b = if c & 2 == 0 { y0 } else { y1 }.abs();
                    let z = if c & 4 == 0 { z0 } else { z1 }.abs();
                    let l = a + b + z;
                    if l <= 1.0e-20 {
                        return false;
                    }
                    let (u, v) = if sz > 0.0 { (sx * a / l, sy * b / l) } else { (sx * (1.0 - b / l), sy * (1.0 - a / l)) };
                    u0 = u0.min(u);
                    u1 = u1.max(u);
                    v0 = v0.min(v);
                    v1 = v1.max(v);
                }
                // Grid coordinates, a hair wider for rounding.
                let g = |p: f32| (p * 0.5 + 0.5) * 16.0 - 0.5;
                let (gx0, gx1, gy0, gy1) = (g(u0) - 1.0e-3, g(u1) + 1.0e-3, g(v0) - 1.0e-3, g(v1) + 1.0e-3);
                let (cx0, cx1) = ((gx0.floor() as i32).clamp(0, 15), (gx1.ceil() as i32).clamp(0, 15));
                let (cy0, cy1) = ((gy0.floor() as i32).clamp(0, 15), (gy1.ceil() as i32).clamp(0, 15));
                for cy in cy0..=cy1 {
                    for cx in cx0..=cx1 {
                        if cell_open(mask, cx, cy) {
                            return false;
                        }
                    }
                }
            }
        }
    }
    true
}

/// A tree's occlusion codebook (athenea's, 8 words an entry) and the
/// asset's box (min xyz, max xyz) in the tree's object space.
#[wasm_bindgen]
pub fn set_lod_occlusion_codebook(lod_id: u32, codebook: Uint32Array, bounds: &[f32]) -> Result<(), JsValue> {
    if bounds.len() != 6 {
        return Err(JsValue::from_str("occlusion bounds: 6 floats"));
    }
    STATE.with_borrow_mut(|state| {
        let tree = state.lod_trees.get_mut(&lod_id).ok_or_else(|| JsValue::from_str("no such LoD tree"))?;
        tree.occlusion = Some((codebook.to_vec(), bounds.try_into().unwrap()));
        Ok(())
    })
}

struct LodState {
    next_id: u32,
    lod_trees: AHashMap<u32, LodTree>,
    /// (pixel scale, instance, node, its parent: u32::MAX a root).
    frontier: Frontier<(OrderedFloat<f32>, u32, u32, u32)>,
    /// (instance, node, its parent).
    output: Vec<(u32, u32, u32)>,
    /// The nodes culled (instance, node): pruned subtrees' roots and cut nodes left out.
    culled: Vec<(u32, u32)>,
    touched: Vec<(u32, u32)>,
    touched_set: AHashSet<(u32, u32)>,
    buffer: Vec<u32>,
}

impl LodState {
    fn new() -> Self {
        Self {
            next_id: 1000,
            lod_trees: AHashMap::new(),
            frontier: Frontier::new(),
            output: Vec::new(),
            culled: Vec::new(),
            touched: Vec::new(),
            touched_set: AHashSet::new(),
            buffer: Vec::new(),
        }
    }
}

thread_local! {
    static STATE: RefCell<LodState> = RefCell::new(LodState::new());
}

fn set_lod_tree_data(state: &mut LodState, lod_id: u32, page_base: u32, _chunk_base: u32, count: u32, lod_tree_data: &Uint32Array, lod_error: Option<&Uint32Array>, lod_occlusion: Option<&Uint32Array>) {
    let lod_tree = state.lod_trees.get(&lod_id).unwrap();
    let mut nodes = lod_tree.splats.borrow_mut();
    let LodNodes::Full(splats) = &mut *nodes else {
        panic!("a compact LoD tree takes update_compact_lod_trees");
    };

    if state.buffer.is_empty() {
        state.buffer.resize(MAX_SPLAT_CHUNK * 4, 0);
    }

    if page_base + count > splats.len() as u32 {
        let new_size = (splats.len() * 2).max((page_base + count) as usize);
        splats.resize_with(new_size, Default::default);
    }

    let mut index = 0;
    while index < count {
        let chunk = (count - index).min(MAX_SPLAT_CHUNK as u32);
        let buffer = &mut state.buffer[0..(chunk * 4) as usize];
        lod_tree_data.subarray(index * 4, (index + chunk) * 4).copy_to(buffer);

        for i in 0..chunk {
            let i4 = i * 4;
            let words: [u32; 4] = array::from_fn(|j| buffer[i4 as usize + j]);
            let center = [
                f16::from_bits((words[0] & 0xffff) as u16),
                f16::from_bits((words[0] >> 16) as u16),
                f16::from_bits((words[1] & 0xffff) as u16),
            ];
            let size = f16::from_bits((words[1] >> 16) as u16);
            let child_count = (words[2] & 0xffff) as u16;
            let child_start = words[3];

            splats[(page_base + index + i) as usize] = LodSplat::new_f16(center, size, child_start, child_count);
        }
        if let Some(errors) = lod_error {
            let w = LOD_ERROR_WORDS as u32;
            let mut words = vec![0u32; (chunk * w) as usize];
            errors.subarray(index * w, (index + chunk) * w).copy_to(&mut words);
            for i in 0..chunk as usize {
                splats[page_base as usize + index as usize + i].set_error([words[2 * i], words[2 * i + 1]]);
            }
        }
        if let Some(occlusion) = lod_occlusion {
            let mut words = vec![0u32; chunk as usize];
            occlusion.subarray(index, index + chunk).copy_to(&mut words);
            for i in 0..chunk as usize {
                splats[page_base as usize + index as usize + i].occlusion = words[i].min(u16::MAX as u32) as u16;
            }
        }
        index += chunk;
    }
}

#[wasm_bindgen]
pub fn new_lod_tree(capacity: u32, slot: Option<u32>, compact: Option<bool>) -> Result<Object, JsValue> {
    let slot = slot.filter(|&s| s > 0 && s <= MAX_SPLAT_CHUNK as u32).unwrap_or(MAX_SPLAT_CHUNK as u32);
    STATE.with_borrow_mut(|state| {
        let lod_id = state.next_id;
        let splats = if compact.unwrap_or(false) {
            // Grown with the pages that land (not reserved by the budget).
            let _ = capacity;
            if QUANT.with_borrow(|q| *q == QUANT_RECORD_BITS) {
                LodNodes::Quant { nodes: Vec::new(), pages: Vec::new(), full: AHashMap::new(), wide: AHashMap::new(), slot }
            } else {
                LodNodes::Compact { nodes: Vec::new(), wide: AHashMap::new(), margins: Vec::new(), slot }
            }
        } else {
            LodNodes::Full(Vec::with_capacity(capacity as usize))
        };
        let splats = Rc::new(RefCell::new(splats));
        let page_capacity = capacity.div_ceil(slot);
        let page_to_chunk = Vec::with_capacity(page_capacity as usize);
        let chunk_to_page: Vec<u32> = Vec::with_capacity(page_capacity as usize);
        state.lod_trees.insert(lod_id, LodTree { splats, page_to_chunk, chunk_to_page, occlusion: None, slot });
        state.next_id += 1;

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("lodId"), &JsValue::from(lod_id)).unwrap();

        Ok(result)
    })
}

#[wasm_bindgen]
pub fn new_shared_lod_tree(orig_lod_id: u32) -> Result<Object, JsValue> {
    STATE.with_borrow_mut(|state| {
        let lod_tree = state.lod_trees.get(&orig_lod_id).unwrap();
        let splats = lod_tree.splats.clone();
        let page_to_chunk = Vec::with_capacity(lod_tree.page_to_chunk.capacity());
        let chunk_to_page = Vec::with_capacity(lod_tree.chunk_to_page.capacity());
        let slot = lod_tree.slot;

        let new_lod_id = state.next_id;
        state.next_id += 1;
        state.lod_trees.insert(new_lod_id, LodTree { splats, page_to_chunk, chunk_to_page, occlusion: None, slot });

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("lodId"), &JsValue::from(new_lod_id)).unwrap();
        Ok(result)
    })
}

#[wasm_bindgen]
pub fn init_lod_tree(num_splats: u32, lod_tree: Uint32Array, lod_error: Option<Uint32Array>, lod_occlusion: Option<Uint32Array>) -> Result<Object, JsValue> {
    STATE.with_borrow_mut(|state| {
        let lod_id = state.next_id;
        let pages = num_splats.div_ceil(65536);
        let splats = Rc::new(RefCell::new(LodNodes::Full(Vec::with_capacity(num_splats as usize))));
        let page_to_chunk = (0..pages).collect();
        let chunk_to_page: Vec<u32> = (0..pages).collect();
        state.lod_trees.insert(lod_id, LodTree { splats, page_to_chunk, chunk_to_page, occlusion: None, slot: MAX_SPLAT_CHUNK as u32 });
        state.next_id += 1;

        set_lod_tree_data(state, lod_id, 0, 0, num_splats, &lod_tree, lod_error.as_ref(), lod_occlusion.as_ref());

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("lodId"), &JsValue::from(lod_id)).unwrap();

        Ok(result)
    })
}

#[wasm_bindgen]
pub fn dispose_lod_tree(lod_id: u32) {
    STATE.with_borrow_mut(|state| {
        state.lod_trees.remove(&lod_id);
    })
}

#[wasm_bindgen]
pub fn update_lod_trees(lod_ids: &[u32], page_bases: &[u32], chunk_bases: &[u32], counts: &[u32], lod_trees: &Array, lod_errors: Option<Array>, lod_occlusions: Option<Array>) -> Result<Object, JsValue> {
    STATE.with_borrow_mut(|state| {
        for (k, (&lod_id, &page_base, &chunk_base, &count, lod_tree_data)) in izip!(lod_ids, page_bases, chunk_bases, counts, lod_trees.iter()).enumerate() {
            let lod_error = lod_errors.as_ref().map(|a| a.get(k as u32)).filter(|e| !e.is_falsy()).map(Uint32Array::from);
            let lod_occlusion = lod_occlusions.as_ref().map(|a| a.get(k as u32)).filter(|e| !e.is_falsy()).map(Uint32Array::from);
            let lod_tree = state.lod_trees.get_mut(&lod_id).unwrap();
            let slot = lod_tree.slot;
            // A range of `count` splats from a page: whole pages of the pool,
            // whole chunks of the virtual space (one each for small pages).
            let pages = count.div_ceil(slot).max(1);
            let base_page = page_base / slot;
            let base_chunk = chunk_base >> 16;

            if (base_page + pages) > lod_tree.page_to_chunk.len() as u32 {
                lod_tree.page_to_chunk.resize((base_page + pages) as usize, 0xFFFFFFFF);
            }
            if (base_chunk + pages) > lod_tree.chunk_to_page.len() as u32 {
                lod_tree.chunk_to_page.resize((base_chunk + pages) as usize, 0xFFFFFFFF);
            }

            if lod_tree_data.is_falsy() {
                for page in 0..pages {
                    lod_tree.page_to_chunk[(base_page + page) as usize] = 0xFFFFFFFF;
                    lod_tree.chunk_to_page[(base_chunk + page) as usize] = 0xFFFFFFFF;
                }    
            } else {
                for page in 0..pages {
                    lod_tree.page_to_chunk[(base_page + page) as usize] = base_chunk + page;
                    lod_tree.chunk_to_page[(base_chunk + page) as usize] = base_page + page;
                }

                let lod_tree_data = Uint32Array::from(lod_tree_data);
                set_lod_tree_data(state, lod_id, page_base, chunk_base, count, &lod_tree_data, lod_error.as_ref(), lod_occlusion.as_ref());
            }
        }

        let result = Object::new();
        // for (&lod_id, lod_tree) in state.lod_trees.iter() {
        //     let entry = Object::new();
        //     Reflect::set(&entry, &JsValue::from_str("pageToChunk"), &JsValue::from(lod_tree.page_to_chunk.clone())).unwrap();
        //     Reflect::set(&entry, &JsValue::from_str("chunkToPage"), &JsValue::from(lod_tree.chunk_to_page.clone())).unwrap();
        //     Reflect::set(&result, &JsValue::from_str(lod_id.to_string().as_str()), &JsValue::from(entry)).unwrap();
        // }
        Ok(result)
    })
}

/// Pages of a compact tree (`new_lod_tree(.., compact = true)`, thread CQ):
/// for each range, `pages[k]` the page's traversal input (spark-lib
/// `athc_compact::traversal_input`: its position planes then the
/// traversal's, 22 bytes a node) or falsy to evict it, and its box (lo xyz,
/// hi xyz) in `boxes[6k..]`. A range is one page: `chunk_base >> 16` its
/// page in the tree's virtual space, `page_base` its pool slot's first node.
#[wasm_bindgen]
pub fn update_compact_lod_trees(lod_ids: &[u32], page_bases: &[u32], chunk_bases: &[u32], counts: &[u32], pages: &Array, boxes: &[f32]) -> Result<(), JsValue> {
    if boxes.len() != 6 * lod_ids.len() || page_bases.len() != lod_ids.len() || chunk_bases.len() != lod_ids.len() || counts.len() != lod_ids.len() {
        return Err(JsValue::from_str("update_compact_lod_trees: ranges of different lengths"));
    }
    STATE.with_borrow_mut(|state| {
        for (k, page) in pages.iter().enumerate() {
            let (lod_id, page_base, chunk_base, count) = (lod_ids[k], page_bases[k], chunk_bases[k], counts[k]);
            let lod_tree = state.lod_trees.get_mut(&lod_id).ok_or_else(|| JsValue::from_str("no such LoD tree"))?;
            let slot = lod_tree.slot;
            if count > slot {
                return Err(JsValue::from_str("update_compact_lod_trees: a page past its slot"));
            }
            let (pool_page, chunk) = ((page_base / slot) as usize, (chunk_base >> 16) as usize);
            if pool_page >= lod_tree.page_to_chunk.len() {
                lod_tree.page_to_chunk.resize(pool_page + 1, 0xFFFFFFFF);
            }
            if chunk >= lod_tree.chunk_to_page.len() {
                lod_tree.chunk_to_page.resize(chunk + 1, 0xFFFFFFFF);
            }
            if page.is_falsy() {
                lod_tree.page_to_chunk[pool_page] = 0xFFFFFFFF;
                lod_tree.chunk_to_page[chunk] = 0xFFFFFFFF;
                // A quantized tree frees the page's block.
                if let LodNodes::Quant { nodes, .. } = &mut *lod_tree.splats.borrow_mut() {
                    if let Some(b) = nodes.get_mut(pool_page) {
                        *b = Box::new([]);
                    }
                }
                continue;
            }
            lod_tree.page_to_chunk[pool_page] = chunk as u32;
            lod_tree.chunk_to_page[chunk] = pool_page as u32;
            let input = Uint8Array::from(page).to_vec();
            let n = count as usize;
            if input.len() != n * spark_lib::athc_compact::TRAVERSAL_INPUT_BYTES
                && input.len() != n * spark_lib::athc_compact::TRAVERSAL_INPUT_BYTES_V2
            {
                return Err(JsValue::from_str("update_compact_lod_trees: a page's input of the wrong size"));
            }
            let head: Vec<u32> = boxes[6 * k..6 * k + 6].iter().map(|v| v.to_bits()).collect();
            let mut store = lod_tree.splats.borrow_mut();
            let base = page_base as usize;
            let mut page: Vec<_> = (0..n).map(|i| spark_lib::athc_compact::traversal_node_input(&input, n, &head, chunk as u32, i)).collect();
            let (mut pq, codes) = QUANT.with_borrow(|q| q.apply(&mut page));
            pq.chunk = chunk as u32;
            let compact_node = |t: &spark_lib::athc_compact::TraversalNode| {
                let code = if t.count >= COMPACT_WIDE as u32 { COMPACT_WIDE } else { t.count as u16 };
                CompactLodNode([
                    t.centre[0],
                    t.centre[1],
                    t.centre[2],
                    t.eps,
                    t.rho | if t.size_mode { COMPACT_SIZE_MODE } else { 0 },
                    t.child_page as u16,
                    (t.place as u16 & 0xfff) | code << 12,
                    t.cone as u16,
                    (t.cone >> 16) as u16,
                    t.occlusion,
                ])
            };
            match &mut *store {
                LodNodes::Compact { nodes, wide, margins, .. } => {
                    if base + n > nodes.len() {
                        // Exactly the pages' extent (a slot more at a time): the
                        // pool's budget is not reserved up front (thread CT).
                        // Steps of an eighth: a copy every few pages, at most 1/8 spare.
                        if base + n > nodes.capacity() {
                            nodes.reserve_exact((base + n - nodes.len()).max(nodes.len() / 8));
                        }
                        nodes.resize(base + n, CompactLodNode::default());
                    }
                    if pool_page >= margins.len() {
                        margins.resize(pool_page + 1, 0.0);
                    }
                    margins[pool_page] = pq.margin;
                    for (i, t) in page.iter().enumerate() {
                        let key = (base + i) as u32;
                        if t.count >= COMPACT_WIDE as u32 {
                            wide.insert(key, t.count as u16);
                        } else {
                            wide.remove(&key);
                        }
                        nodes[base + i] = compact_node(t);
                    }
                }
                LodNodes::Quant { nodes, pages, full, wide, .. } => {
                    if pool_page >= nodes.len() {
                        nodes.resize_with(pool_page + 1, || Box::new([]));
                    }
                    // The page's own block, its nodes' size.
                    let mut block = vec![QuantNode::default(); n].into_boxed_slice();
                    if pool_page >= pages.len() {
                        pages.resize(pool_page + 1, PageQuant::default());
                    }
                    pages[pool_page] = pq;
                    for (i, (t, c)) in page.iter().zip(&codes).enumerate() {
                        let key = (base + i) as u32;
                        let packed = if t.count == 0 && (t.eps != 0 || t.rho != 0) { None } else { pack_quant(t, c, chunk as u32) };
                        match packed {
                            Some(q) => {
                                block[i] = q;
                                full.remove(&key);
                                wide.remove(&key);
                            }
                            None => {
                                full.insert(key, compact_node(t));
                                if t.count >= COMPACT_WIDE as u32 {
                                    wide.insert(key, t.count as u16);
                                } else {
                                    wide.remove(&key);
                                }
                            }
                        }
                    }
                    nodes[pool_page] = block;
                }
                LodNodes::Full(_) => return Err(JsValue::from_str("update_compact_lod_trees: not a compact tree")),
            }
        }
        Ok(())
    })
}

#[allow(dead_code)]
struct LodInstance<'a> {
    lod_id: u32,
    splats: Ref<'a, LodNodes>,
    page_to_chunk: &'a [u32],
    chunk_to_page: &'a [u32],
    origin: Vec3A,
    forward: Vec3A,
    right: Vec3A,
    up: Vec3A,
    output: Vec<u32>,
    lod_scale: f32,
    outside_foveate: f32,
    behind_foveate: f32,
    cone_dot0: f32,
    cone_dot: f32,
    cone_foveate: f32,
}

#[allow(dead_code)]
fn children_resident(child_count: u16, child_start: u32, instance: &LodInstance) -> bool {
    // Check endpoints, okay since child_count <= 65535
    for child in [child_start, child_start + child_count as u32 - 1] {
        if !is_resident(child, instance) {
            return false;
        }
    }
    true
}

#[allow(dead_code)]
fn is_resident(index: u32, instance: &LodInstance) -> bool {
    let chunk = (index >> 16) as usize;
    if chunk >= instance.chunk_to_page.len() {
        false
    } else {
        instance.chunk_to_page[chunk] != 0xFFFFFFFF
    }
}

#[wasm_bindgen]
pub fn get_lod_tree_level(lod_id: u32, level: u32) -> anyhow::Result<Object, JsValue> {
    STATE.with_borrow_mut(|state| {
        let LodState { lod_trees, .. } = state;
        let lod_tree = lod_trees.get(&lod_id).unwrap();
        let splats = lod_tree.splats.borrow();

        let root_size = splats.get(0).size();
        let level_size = root_size / (1.25f32.powi(level as i32));

        let mut nodes = vec![0];
        let mut output_nodes = Vec::new();

        while !nodes.is_empty() {
            let mut new_nodes = Vec::new();
            for node in nodes {
                let splat = &splats.get(node as usize);
                let &LodSplat { child_count, child_start, .. } = splat;
                if splat.size() <= level_size {
                    output_nodes.push(node);
                } else {
                    for child in child_start..child_start + child_count as u32 {
                        new_nodes.push(child);
                    }
                }
            }
            nodes = new_nodes;
        }

        let output = Uint32Array::new_with_length(output_nodes.len() as u32);
        for (i, node) in output_nodes.into_iter().enumerate() {
            output.set_index(i as u32, node);
        }

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("indices"), &JsValue::from(output)).unwrap();
        Ok(result)
    })
}

#[wasm_bindgen]
pub fn traverse_lod_trees(
    max_splats: u32, pixel_scale_limit: f32, _last_pixel_limit: Option<f32>,
    lod_ids: &[u32], root_pages: &[u32],
    view_to_objects: &[f32], lod_scales: &[f32],
    behind_foveates: &[f32], cone_foveates: &[f32],
    cone_fov0s: &[f32], cone_fovs: &[f32],
    error_limit: Option<f32>, back_cull_min: Option<f32>, occlusion_cull: Option<bool>,
    error_size_k: Option<f32>, neighbours: Option<bool>,
) -> anyhow::Result<Object, JsValue> {
    // With `neighbours` (the node pool, thread CT): each cut node's parent and
    // its children (when resident), and the culled nodes, by instance.
    let neighbours = neighbours.unwrap_or(false);
    // The error cut (spark-lib athc_lod_cut): a node with bounds is measured
    // by its projected error, P = eps / max(d - rho, near), scaled so that
    // P <= error_limit compares as the size cut's pixel_scale_limit.
    let err_ratio = error_limit.filter(|&e| e > 0.0).map_or(0.0, |e| pixel_scale_limit / e);
    // The combined cut (`error_size_k` k > 0): P = max(P_err, P_size / k), so
    // the error cut never keeps a node more than k tau pixels wide.
    let size_guard = error_size_k.filter(|&k| k > 0.0 && err_ratio > 0.0).map_or(0.0, |k| err_ratio / k);
    // Back-face culling of the cut's coarse nodes: a node the cut draws (not
    // one it refines) whose whole normal cone faces away, when it is wider
    // than `back_cull_min` on screen (the same angle units as the limit: a
    // node at full detail is never culled). None: off.
    // Negative (the compact path, thread CS): any size, and a node wholly
    // facing away is pruned with its subtree (its cone bounds its leaves).
    let prune_backs = back_cull_min.is_some_and(|m| m < 0.0);
    let back_cull_min = back_cull_min.map(|m| m.max(0.0)).filter(|m| m.is_finite());
    let max_splats = max_splats as usize;
    let num_instances = lod_ids.len();
    if view_to_objects.len() != num_instances * 16 {
        return Err(JsValue::from_str("Invalid view_to_objects length"));
    }
    if lod_scales.len() != num_instances {
        return Err(JsValue::from_str("Invalid lod_scales length"));
    }
    if behind_foveates.len() != num_instances {
        return Err(JsValue::from_str("Invalid behind_foveates length"));
    }
    if cone_foveates.len() != num_instances {
        return Err(JsValue::from_str("Invalid cone_foveates length"));
    }
    if cone_fov0s.len() != num_instances {
        return Err(JsValue::from_str("Invalid cone_fov0s length"));
    }
    if cone_fovs.len() != num_instances {
        return Err(JsValue::from_str("Invalid cone_fovs length"));
    }

    STATE.with_borrow_mut(|state| {
        let LodState { lod_trees, frontier, output, culled: culled_out, touched, touched_set, .. } = state;
        culled_out.clear();
        let slots: Vec<u32> = lod_ids.iter().map(|id| lod_trees.get(id).map_or(MAX_SPLAT_CHUNK as u32, |t| t.slot)).collect();
        let instances: Vec<_> = lod_ids.iter().enumerate().map(|(index, &lod_id)| {
            let lod_tree = lod_trees.get(&lod_id).unwrap();
            let LodTree { splats, page_to_chunk, chunk_to_page, .. } = &lod_tree;
            let i16 = index * 16;
            let forward = Vec3A::from_slice(&view_to_objects[(i16 + 8)..(i16 + 11)]).normalize().map(|x| -x);
            let origin = Vec3A::from_slice(&view_to_objects[(i16 + 12)..(i16 + 15)]);
            let lod_scale = lod_scales[index];
            let behind_foveate = behind_foveates[index];
            let cone_foveate = cone_foveates[index];
            let cone_dot0 = if cone_fov0s[index] > 0.0 { (0.5 * cone_fov0s[index].clamp(0.0, 180.0)).to_radians().cos() } else { 1.0 };
            let cone_dot = if cone_fovs[index] > 0.0 { (0.5 * cone_fovs[index].clamp(0.0, 180.0)).to_radians().cos() } else { 1.0 };
            let cone_dot = cone_dot.min(cone_dot0);
            (lod_id, splats.borrow(), page_to_chunk, chunk_to_page, origin, forward, lod_scale, behind_foveate, cone_foveate, cone_dot0, cone_dot, err_ratio, size_guard)
        }).collect();
        // Each instance's occlusion codebook, when culling by it.
        let occlusions: Vec<Option<&(Vec<u32>, [f32; 6])>> = lod_ids
            .iter()
            .map(|id| if occlusion_cull.unwrap_or(false) { lod_trees.get(id).and_then(|t| t.occlusion.as_ref()) } else { None })
            .collect();
        let hidden = |inst: u32, index: u32| -> bool {
            occlusions[inst as usize].is_some_and(|o| occluded(&instances[inst as usize].1.get(index as usize), instances[inst as usize].4, o))
        };
        let mut occluded_count = 0u32;

        let mut num_splats = 0;
        let mut culled_count = 0u32;
        frontier.clear();
        output.clear();
        // Not the budget (the node pool's is millions, 12 bytes each and kept
        // by the heap): the vector grows with the cut (thread CT).
        output.reserve(max_splats.min(1 << 16));
        touched.clear();
        touched_set.clear();

        for (inst_index, instance) in instances.iter().enumerate() {
            let (lod_id, splats, ..) = instance;
            let root_page = root_pages[inst_index];
            let root_page = if root_page == 0xFFFFFFFF { 0 } else { root_page };
            let root_index = root_page * slots[inst_index];
            if hidden(inst_index as u32, root_index) {
                occluded_count += 1;
                continue;
            }
            let pixel_scale = compute_pixel_scale(&splats.get(root_index as usize), instance);
            frontier.push((OrderedFloat(pixel_scale), inst_index as u32, root_index, u32::MAX));
            num_splats += 1;

            if touched_set.insert((*lod_id, 0)) {
                touched.push((*lod_id, 0));
            }
        }
        
        let mut min_pixel_scale = f32::INFINITY;
        let mut leaf_count = 0;
        let culled = |inst: u32, index: u32| -> bool {
            back_cull_min.is_some_and(|min| {
                let i = &instances[inst as usize];
                let s = &i.1.get(index as usize);
                (min <= 0.0 || s.size() / (s.center() - i.4).length().max(1.0e-6) > min) && s.back_facing(i.4)
            })
        };

        while let Some(&(OrderedFloat(pixel_scale), inst_index, paged_index, parent)) = frontier.peek() {
            min_pixel_scale = min_pixel_scale.min(pixel_scale);
            if pixel_scale <= pixel_scale_limit {
                break;
            }

            let instance = &instances[inst_index as usize];
            let (lod_id, splats, _page_to_chunk, chunk_to_page, ..) = instance;
            let LodSplat { child_count, child_start, .. } = splats.get(paged_index as usize);

            // A node wholly facing away (its cone bounds every leaf under it:
            // a closed part's) is gone with its subtree: not refined, its
            // children's pages not asked for (thread CS).
            if prune_backs && culled(inst_index, paged_index) {
                _ = frontier.pop();
                culled_count += 1;
                if neighbours {
                    culled_out.push((inst_index, paged_index));
                }
                continue;
            }

            if child_count == 0 {
                _ = frontier.pop();
                output.push((inst_index, paged_index, parent));
                leaf_count += 1;
                continue;
            }

            let new_num_splats = num_splats - 1 + child_count as usize;
            if new_num_splats > max_splats {
                break;
            }

            _ = frontier.pop();

            let first_chunk = child_start >> 16;
            if touched_set.insert((*lod_id, first_chunk)) {
                touched.push((*lod_id, first_chunk));
            }

            let last_chunk = (child_start + child_count as u32 - 1) >> 16;
            if last_chunk != first_chunk && touched_set.insert((*lod_id, last_chunk)) {
                touched.push((*lod_id, last_chunk));
            }

            if last_chunk as usize >= chunk_to_page.len() {
                if culled(inst_index, paged_index) {
                    culled_count += 1;
                    if neighbours {
                        culled_out.push((inst_index, paged_index));
                    }
                } else {
                    output.push((inst_index, paged_index, parent));
                }
                continue;
            }
            let first_page = chunk_to_page[first_chunk as usize];
            let last_page = chunk_to_page[last_chunk as usize];

            if first_page == 0xFFFFFFFF || last_page == 0xFFFFFFFF {
                if culled(inst_index, paged_index) {
                    culled_count += 1;
                    if neighbours {
                        culled_out.push((inst_index, paged_index));
                    }
                } else {
                    output.push((inst_index, paged_index, parent));
                }
                continue;
            }
            let refined = paged_index;

            let mut hidden_here = 0;
            for child in child_start..child_start + child_count as u32 {
                let child_chunk = (child >> 16) as usize;
                let child_page = chunk_to_page[child_chunk];
                let paged_index = child_page * slots[inst_index as usize] + (child & 0xffff);
                // Occluded from the eye (athenea's mask): gone with its subtree.
                if hidden(inst_index, paged_index) {
                    hidden_here += 1;
                    continue;
                }
                let pixel_scale = compute_pixel_scale(&splats.get(paged_index as usize), instance);
                if pixel_scale <= pixel_scale_limit {
                    // In the cut: drawn, unless a coarse node facing away.
                    if culled(inst_index, paged_index) {
                        culled_count += 1;
                        if neighbours {
                            culled_out.push((inst_index, paged_index));
                        }
                    } else {
                        output.push((inst_index, paged_index, refined));
                    }
                } else {
                    frontier.push((OrderedFloat(pixel_scale), inst_index, paged_index, refined));
                }
            }

            occluded_count += hidden_here;
            num_splats = new_num_splats - hidden_here as usize;
        }

        let output_size = output.len();
        let frontier_size = frontier.len();

        for (_, inst_index, paged_index, parent) in frontier.drain() {
            if culled(inst_index, paged_index) {
                culled_count += 1;
                if neighbours {
                    culled_out.push((inst_index, paged_index));
                }
            } else {
                output.push((inst_index, paged_index, parent));
            }
        }

        let mut instance_counts = vec![0; num_instances];
        for &(inst_index, _, _) in output.iter() {
            instance_counts[inst_index as usize] += 1;
        }

        let mut instance_outputs = Vec::with_capacity(num_instances);
        for counts in instance_counts {
            instance_outputs.push(Vec::with_capacity(counts));
        }

        for &(inst_index, paged_index, _) in output.iter() {
            instance_outputs[inst_index as usize].push(paged_index);
        }
        // Each cut node's parent, its first child and their count (resident
        // children only; u32::MAX none), and the culled nodes, by instance.
        let mut extra: Vec<[Vec<u32>; 4]> = (0..num_instances).map(|_| Default::default()).collect();
        if neighbours {
            for &(inst_index, paged_index, parent) in output.iter() {
                let e = &mut extra[inst_index as usize];
                e[0].push(parent);
                let instance = &instances[inst_index as usize];
                let node = instance.1.get(paged_index as usize);
                let (first, count) = if node.child_count > 0 {
                    let chunk = (node.child_start >> 16) as usize;
                    match instance.3.get(chunk) {
                        Some(&page) if page != 0xFFFFFFFF => (page * slots[inst_index as usize] + (node.child_start & 0xffff), node.child_count as u32),
                        _ => (u32::MAX, 0),
                    }
                } else {
                    (u32::MAX, 0)
                };
                e[1].push(first);
                e[2].push(count);
            }
            for &(inst_index, paged_index) in culled_out.iter() {
                extra[inst_index as usize][3].push(paged_index);
            }
        }

        let instance_indices = Array::new();

        for (inst_index, instance_output) in instance_outputs.iter_mut().enumerate() {
            // instance_output.sort_unstable();
            let rows = instance_output.len().div_ceil(16384);
            let capacity = rows * 16384;
            let output = Uint32Array::new_with_length(capacity as u32);
            output.subarray(0, instance_output.len() as u32).copy_from(instance_output);

            let result = Object::new();
            let lod_id = instances[inst_index].0;
            Reflect::set(&result, &JsValue::from_str("lodId"), &JsValue::from(lod_id)).unwrap();
            Reflect::set(&result, &JsValue::from_str("numSplats"), &JsValue::from(instance_output.len() as u32)).unwrap();
            Reflect::set(&result, &JsValue::from_str("indices"), &JsValue::from(output)).unwrap();
            if neighbours {
                let e = &extra[inst_index];
                for (k, name) in ["parents", "firstKids", "kidCounts", "culled"].into_iter().enumerate() {
                    let a = Uint32Array::new_with_length(e[k].len() as u32);
                    a.copy_from(&e[k]);
                    Reflect::set(&result, &JsValue::from_str(name), &JsValue::from(a)).unwrap();
                }
            }
            instance_indices.push(&JsValue::from(result));
        }

        let out_chunks = Array::new();

        for &(inst_index, chunk) in touched.iter() {
            let pair = Array::new();
            pair.push(&JsValue::from(inst_index));
            pair.push(&JsValue::from(chunk));
            out_chunks.push(&JsValue::from(pair));
        }

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("pixelLimit"), &JsValue::from(min_pixel_scale)).unwrap();
        Reflect::set(&result, &JsValue::from_str("instanceIndices"), &JsValue::from(instance_indices)).unwrap();
        Reflect::set(&result, &JsValue::from_str("chunks"), &JsValue::from(out_chunks)).unwrap();
        Reflect::set(&result, &JsValue::from_str("outputSize"), &JsValue::from(output_size)).unwrap();
        Reflect::set(&result, &JsValue::from_str("frontierSize"), &JsValue::from(frontier_size)).unwrap();
        Reflect::set(&result, &JsValue::from_str("leafCount"), &JsValue::from(leaf_count)).unwrap();
        Reflect::set(&result, &JsValue::from_str("culledCount"), &JsValue::from(culled_count)).unwrap();
        Reflect::set(&result, &JsValue::from_str("occludedCount"), &JsValue::from(occluded_count)).unwrap();
        Ok(result)
    })
}

type Inst<'a> = (u32, Ref<'a, LodNodes>, &'a Vec<u32>, &'a Vec<u32>, Vec3A, Vec3A, f32, f32, f32, f32, f32, f32, f32);

fn compute_pixel_scale<'a>(splat: &LodSplat, instance: &Inst<'a>) -> f32 {
    let &(_, _, _, _, origin, forward, lod_scale, behind_foveate, cone_foveate, cone_dot0, cone_dot, err_ratio, size_guard) = instance;
    let center = splat.center();
    let delta = center - origin;
    // A quantized node's distance less its page's centre error (thread CT).
    let distance = (delta.length() - splat.margin).max(1.0e-6);
    let inv_distance = 1.0 / distance;
    let pixel_scale = if err_ratio > 0.0 && splat.has_error() {
        // athenea's projectedError, at the sphere's nearest point.
        let p = splat.eps.to_f32() / (distance - splat.rho.to_f32()).max(ERROR_NEAR) * err_ratio;
        if size_guard > 0.0 { p.max(splat.size() * inv_distance * size_guard) } else { p }
    } else {
        splat.size() * inv_distance
    };
    let pixel_scale = pixel_scale * lod_scale;

    let forward_dot = delta.dot(forward);
    let foveate = if forward_dot <= 0.0 {
        behind_foveate
    } else {
        let dot = forward_dot * inv_distance;
        if dot >= cone_dot0 {
            1.0
        } else if dot >= cone_dot {
            let t = (dot - cone_dot) / (cone_dot0 - cone_dot);
            cone_foveate + (1.0 - cone_foveate) * t
        } else {
            let t = dot / cone_dot;
            behind_foveate + (cone_foveate - behind_foveate) * t
        }
    };
    foveate * pixel_scale
}

#[wasm_bindgen]
pub fn dynamic_traverse_lod_trees(
    max_splats: u32, pixel_scale_limit: f32, _last_pixel_limit: Option<f32>,
    // lod_instances: &Array,
    lod_ids: &[u32], root_pages: &[u32],
    view_to_objects: &[f32], lod_scales: &[f32],
    behind_foveates: &[f32], cone_foveates: &[f32],
    cone_fov0s: &[f32], cone_fovs: &[f32],
    // readback: Uint32Array,
    // flag: bool,
    error_limit: Option<f32>, _back_cull_min: Option<f32>, _occlusion_cull: Option<bool>,
    error_size_k: Option<f32>,
) -> anyhow::Result<Object, JsValue> {
    // (The dynamic traversal does not cull.)
    let err_ratio = error_limit.filter(|&e| e > 0.0).map_or(0.0, |e| pixel_scale_limit / e);
    // The combined cut (`error_size_k` k > 0): P = max(P_err, P_size / k), so
    // the error cut never keeps a node more than k tau pixels wide.
    let size_guard = error_size_k.filter(|&k| k > 0.0 && err_ratio > 0.0).map_or(0.0, |k| err_ratio / k);

    let max_splats = max_splats as usize;
    let num_instances = lod_ids.len();
    if view_to_objects.len() != num_instances * 16 {
        return Err(JsValue::from_str("Invalid view_to_objects length"));
    }
    if lod_scales.len() != num_instances {
        return Err(JsValue::from_str("Invalid lod_scales length"));
    }
    if behind_foveates.len() != num_instances {
        return Err(JsValue::from_str("Invalid behind_foveates length"));
    }
    if cone_foveates.len() != num_instances {
        return Err(JsValue::from_str("Invalid cone_foveates length"));
    }
    if cone_fov0s.len() != num_instances {
        return Err(JsValue::from_str("Invalid cone_fov0s length"));
    }
    if cone_fovs.len() != num_instances {
        return Err(JsValue::from_str("Invalid cone_fovs length"));
    }

    STATE.with_borrow_mut(|state| {
        let LodState { lod_trees, .. } = state;
        let slots: Vec<u32> = lod_ids.iter().map(|id| lod_trees.get(id).map_or(MAX_SPLAT_CHUNK as u32, |t| t.slot)).collect();
        let instances: Vec<_> = lod_ids.iter().enumerate().map(|(index, &lod_id)| {
            let lod_tree = lod_trees.get(&lod_id).unwrap();
            let LodTree { splats, page_to_chunk, chunk_to_page, .. } = &lod_tree;
            let i16 = index * 16;
            let forward = Vec3A::from_slice(&view_to_objects[(i16 + 8)..(i16 + 11)]).normalize().map(|x| -x);
            let origin = Vec3A::from_slice(&view_to_objects[(i16 + 12)..(i16 + 15)]);
            let lod_scale = lod_scales[index];
            let behind_foveate = behind_foveates[index];
            let cone_foveate = cone_foveates[index];
            let cone_dot0 = if cone_fov0s[index] > 0.0 { (0.5 * cone_fov0s[index]).to_radians().cos() } else { 1.0 };
            let cone_dot = if cone_fovs[index] > 0.0 { (0.5 * cone_fovs[index]).to_radians().cos() } else { 1.0 };
            (lod_id, splats.borrow(), page_to_chunk, chunk_to_page, origin, forward, lod_scale, behind_foveate, cone_foveate, cone_dot0, cone_dot, err_ratio, size_guard)
        }).collect();

        let mut lod_chunk_max: AHashMap<u32, Vec<f32>> = AHashMap::new();

        let mut outputs = Vec::with_capacity(num_instances);
        for (inst_index, instance) in instances.iter().enumerate() {
            let (lod_id, splats, ..) = instance;
            let root_page = root_pages[inst_index];
            let root_page = if root_page == 0xFFFFFFFF { 0 } else { root_page };
            let root_index = root_page * slots[inst_index];
            let root_scale = compute_pixel_scale(&splats.get(root_index as usize), instance);
            let frontier = vec![(root_index, root_scale)];
            let instance_output = Vec::with_capacity(1000);
            outputs.push((instance_output, frontier));

            let chunk_max = lod_chunk_max.entry(*lod_id).or_default();
            if chunk_max.is_empty() {
                chunk_max.resize(1, 0.0);
            }
            chunk_max[0] = f32::INFINITY;
        }


        let mut leaf_count = 0;
        let mut missing_count = 0;
        let mut min_pixel_scale = f32::INFINITY;

        let mut current_scale = pixel_scale_limit * 100.0;

        loop {
            let iterator = instances.iter().zip(outputs);
            outputs = Vec::with_capacity(num_instances);

            let mut output_count = 0;

            for (inst_index, (instance, (mut instance_output, mut stack))) in iterator.enumerate() {
                let (lod_id, splats, _, chunk_to_page, ..) = instance;
                let chunk_max = lod_chunk_max.entry(*lod_id).or_default();
                let mut frontier = Vec::with_capacity(stack.len());

                while let Some((paged_index, pixel_scale)) = stack.pop() {
                    min_pixel_scale = min_pixel_scale.min(pixel_scale);
                    if pixel_scale <= current_scale {
                        frontier.push((paged_index, pixel_scale));
                        continue;
                    }

                    let LodSplat { child_count, child_start, .. } = splats.get(paged_index as usize);
                    if child_count == 0 {
                        instance_output.push((paged_index, pixel_scale));
                        leaf_count += 1;
                        continue;
                    }

                    let first_chunk = child_start >> 16;
                    let last_chunk = (child_start + child_count as u32 - 1) >> 16;

                    if last_chunk as usize >= chunk_max.len() {
                        chunk_max.resize(last_chunk as usize + 1, 0.0);
                    }
                    chunk_max[first_chunk as usize] = chunk_max[first_chunk as usize].max(pixel_scale);
                    chunk_max[last_chunk as usize] = chunk_max[last_chunk as usize].max(pixel_scale);
        
                    if last_chunk as usize >= chunk_to_page.len() {
                        instance_output.push((paged_index, pixel_scale));
                        missing_count += 1;
                        continue;
                    }
                    let first_page = chunk_to_page[first_chunk as usize];
                    let last_page = chunk_to_page[last_chunk as usize];
        
                    if first_page == 0xFFFFFFFF || last_page == 0xFFFFFFFF {
                        instance_output.push((paged_index, pixel_scale));
                        missing_count += 1;
                        continue;
                    }
        
                    for child in child_start..child_start + child_count as u32 {
                        let child_chunk = (child >> 16) as usize;
                        let child_page = chunk_to_page[child_chunk];
                        let paged_index = child_page * slots[inst_index] + (child & 0xffff);
                        let pixel_scale = compute_pixel_scale(&splats.get(paged_index as usize), instance);
                        if pixel_scale <= current_scale {
                            if pixel_scale <= pixel_scale_limit {
                                instance_output.push((paged_index, pixel_scale));
                            } else {
                                frontier.push((paged_index, pixel_scale));
                            }
                        } else {
                            stack.push((paged_index, pixel_scale));
                        }
                    }
                }

                output_count += instance_output.len() + frontier.len();
                outputs.push((instance_output, frontier));
            }

            let ratio = output_count as f32 / max_splats as f32;
            // let next_scale = (0.9 * current_scale * ratio.powf(1.0 / 1.5)).max(pixel_scale_limit);
            let next_scale = 0.99 * current_scale * ratio.powf(1.0 / 2.0);
            // let next_scale = (0.9 * current_scale).max(pixel_scale_limit);
            let next_scale = next_scale.max(0.5 * current_scale);
            let next_scale = next_scale.max(pixel_scale_limit);

            let no_frontier = outputs.iter().all(|(_, frontier)| frontier.is_empty());

            if no_frontier || (next_scale == current_scale) || output_count >= max_splats {
                break;
            }

            current_scale = next_scale;
            // loop
        };

        let mut touched: Vec<_> = lod_chunk_max.into_iter().flat_map(|x| {
            let (lod_id, chunk_max) = x;
            chunk_max.into_iter().enumerate().filter_map(move |(chunk, max)| {
                if max == 0.0 {
                    None
                } else {
                    Some((OrderedFloat(-max), lod_id, chunk as u32))
                }
            })
        }).collect();
        touched.sort_unstable();

        let instance_indices = Array::new();
        let mut output_size = 0;

        for (inst_index, (mut instance_output, frontier)) in outputs.into_iter().enumerate() {
            output_size += frontier.len();
            instance_output.extend(frontier);
            let rows = instance_output.len().div_ceil(16384);
            let capacity = rows * 16384;
            let output = Uint32Array::new_with_length(capacity as u32);
            let output_u32: Vec<u32> = instance_output.into_iter().map(|(paged_index, _)| paged_index).collect();
            output.subarray(0, output_u32.len() as u32).copy_from(&output_u32);

            let result = Object::new();
            let lod_id = instances[inst_index].0;
            Reflect::set(&result, &JsValue::from_str("lodId"), &JsValue::from(lod_id)).unwrap();
            Reflect::set(&result, &JsValue::from_str("numSplats"), &JsValue::from(output_u32.len() as u32)).unwrap();
            Reflect::set(&result, &JsValue::from_str("indices"), &JsValue::from(output)).unwrap();
            instance_indices.push(&JsValue::from(result));
        }

        let out_chunks = Array::new();

        for &(_, lod_id, chunk) in touched.iter() {
            let pair = Array::new();
            pair.push(&JsValue::from(lod_id));
            pair.push(&JsValue::from(chunk));
            out_chunks.push(&JsValue::from(pair));
        }

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("pixelLimit"), &JsValue::from(min_pixel_scale)).unwrap();
        Reflect::set(&result, &JsValue::from_str("instanceIndices"), &JsValue::from(instance_indices)).unwrap();
        Reflect::set(&result, &JsValue::from_str("chunks"), &JsValue::from(out_chunks)).unwrap();
        Reflect::set(&result, &JsValue::from_str("outputSize"), &JsValue::from(output_size)).unwrap();
        Reflect::set(&result, &JsValue::from_str("leafCount"), &JsValue::from(leaf_count)).unwrap();
        Reflect::set(&result, &JsValue::from_str("missingCount"), &JsValue::from(missing_count)).unwrap();
        Ok(result)
    })
}

// ------------------------------------------------- quantized compact nodes

/// The compact traversal's quantization (thread CT): bits for the centre (an
/// axis, in the page's box of centres), LODE's eps and rho (log codes over
/// the page's range, rounded up), the cone's theta (a log code, rounded up)
/// and a merge's octahedral axis (a component); 0 leaves a field exact.
/// Rounding only refines more and culls less: eps, rho and theta up, a
/// merge's axis error added to its theta, the centres' largest error (the
/// page's margin) added to every node's sphere. With the record's widths
/// (QUANT_RECORD_BITS) the nodes are kept as 12-byte records (`QuantNode`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CompactQuant {
    pub centre_bits: u32,
    pub eps_bits: u32,
    pub rho_bits: u32,
    pub theta_bits: u32,
    pub axis_bits: u32,
}

/// The widths of a 12-byte record (QuantNode).
pub const QUANT_RECORD_BITS: CompactQuant = CompactQuant { centre_bits: 10, eps_bits: 8, rho_bits: 7, theta_bits: 6, axis_bits: 5 };

thread_local! {
    static QUANT: RefCell<CompactQuant> = RefCell::new(CompactQuant::default());
}

/// Sets the quantization of the compact pages that land from now on (all 0:
/// exact, 20-byte nodes; QUANT_RECORD_BITS: 12-byte records).
#[wasm_bindgen]
pub fn set_compact_quant(centre_bits: u32, eps_bits: u32, rho_bits: u32, theta_bits: u32, axis_bits: u32) {
    QUANT.with_borrow_mut(|q| *q = CompactQuant { centre_bits, eps_bits, rho_bits, theta_bits, axis_bits });
}

/// The smallest f16 not below x (x >= 0).
fn f16_up(x: f32) -> u16 {
    let h = f16::from_f32(x);
    if h.to_f32() >= x || !h.is_finite() { h.to_bits() } else { h.to_bits() + 1 }
}

/// A log code's range over a page: (lo, step), codes 1..=2^bits-1 (0: zero).
#[derive(Clone, Copy, Debug, Default)]
struct LogRange {
    lo: f32,
    step: f32,
    top: u32,
}

impl LogRange {
    fn over(values: impl Iterator<Item = f32>, bits: u32) -> Self {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for x in values.filter(|&x| x > 0.0 && x.is_finite()) {
            lo = lo.min(x.log2());
            hi = hi.max(x.log2());
        }
        let top = (1u32 << bits) - 1;
        if !lo.is_finite() {
            return Self { lo: 0.0, step: 0.0, top };
        }
        let steps = (top - 1) as f32;
        Self { lo, step: if hi > lo { (hi - lo) / steps } else { 0.0 }, top }
    }

    /// Code `c` (>= 1) as f16 bits.
    fn decode(&self, c: u32) -> u16 {
        f16_up((self.lo + (c - 1) as f32 * self.step).exp2())
    }

    /// The smallest code whose decode is not below x (None: past the top).
    fn encode_up(&self, x: f32) -> Option<u32> {
        let first = if self.step > 0.0 { ((x.log2() - self.lo) / self.step - 1.0e-4).ceil().max(0.0) as u32 + 1 } else { 1 };
        let mut c = first.min(self.top).max(1);
        // Down while the code below still covers x, up until it does.
        while c > 1 && f16::from_bits(self.decode(c - 1)).to_f32() >= x {
            c -= 1;
        }
        while f16::from_bits(self.decode(c)).to_f32() < x {
            if c == self.top {
                return None;
            }
            c += 1;
        }
        Some(c)
    }
}

/// A page's quantization: the centres' box, the log ranges, the margin.
#[derive(Clone, Copy, Debug, Default)]
struct PageQuant {
    lo: [f32; 3],
    span: [f32; 3],
    ctop: f32,
    eps: LogRange,
    rho: LogRange,
    theta: LogRange,
    margin: f32,
    chunk: u32,
}

impl PageQuant {
    fn centre(&self, q: [u32; 3]) -> [u16; 3] {
        [0, 1, 2].map(|k| f16::from_f32(self.lo[k] + q[k] as f32 / self.ctop * self.span[k]).to_bits())
    }
}

/// A node's codes (QUANT_RECORD_BITS) or None: kept whole (`full`).
#[derive(Clone, Copy, Debug, Default)]
struct NodeCodes {
    centre: [u32; 3],
    eps: u32,
    rho: u32,
    /// Merges: 0 no cone, else the code; leaves: theta's class (0 none, 1 +0, 2 -0, 3 coded) and code.
    theta: u32,
    theta_class: u32,
    /// Merges: the axis at axis_bits a component; leaves: the 16-bit word.
    axis: u32,
}

fn oct_axis(bits: u16) -> Vec3A {
    let f = [(bits & 255) as f32 / 255.0 * 2.0 - 1.0, (bits >> 8) as f32 / 255.0 * 2.0 - 1.0];
    let mut n = [f[0], f[1], 1.0 - f[0].abs() - f[1].abs()];
    let t = (-n[2]).clamp(0.0, 1.0);
    n[0] += if n[0] >= 0.0 { -t } else { t };
    n[1] += if n[1] >= 0.0 { -t } else { t };
    Vec3A::from_array(n).normalize()
}

const NO_THETA: u16 = 0x7c00;

fn cone_present(cone: u32) -> bool {
    let th = (cone & 0xffff) as u16;
    th != 0 && th != 0x8000 && f16::from_bits(th).to_f32().is_finite()
}

impl CompactQuant {
    fn on(&self) -> bool {
        self.centre_bits + self.eps_bits + self.rho_bits + self.theta_bits + self.axis_bits > 0
    }

    fn requant_axis(&self, q8: u16) -> (u32, u16) {
        let top = ((1u32 << self.axis_bits) - 1) as f32;
        let qa = ((q8 as f32) * top / 255.0).round() as u32;
        (qa, ((qa as f32) * 255.0 / top).round() as u16)
    }

    /// A page's nodes quantized in place (their values the codes' decodes),
    /// its quantization and each node's codes.
    fn apply(&self, page: &mut [spark_lib::athc_compact::TraversalNode]) -> (PageQuant, Vec<NodeCodes>) {
        let mut pq = PageQuant::default();
        let mut codes = vec![NodeCodes::default(); page.len()];
        if !self.on() || page.is_empty() {
            return (pq, codes);
        }
        if self.centre_bits > 0 && self.centre_bits < 16 {
            let exact: Vec<Vec3A> = page.iter().map(|t| Vec3A::from_array(t.centre.map(|c| f16::from_bits(c).to_f32()))).collect();
            let lo = exact.iter().fold(Vec3A::splat(f32::INFINITY), |a, c| a.min(*c));
            let hi = exact.iter().fold(Vec3A::splat(f32::NEG_INFINITY), |a, c| a.max(*c));
            pq.lo = lo.to_array();
            pq.span = (hi - lo).to_array();
            pq.ctop = ((1u32 << self.centre_bits) - 1) as f32;
            let mut margin = 0.0f32;
            for ((t, c), code) in page.iter_mut().zip(&exact).zip(codes.iter_mut()) {
                let q = [0, 1, 2].map(|k| if pq.span[k] > 0.0 { ((c[k] - pq.lo[k]) / pq.span[k] * pq.ctop).round().clamp(0.0, pq.ctop) as u32 } else { 0 });
                code.centre = q;
                t.centre = pq.centre(q);
                let stored = Vec3A::from_array(t.centre.map(|v| f16::from_bits(v).to_f32()));
                margin = margin.max((stored - *c).length());
            }
            pq.margin = margin * (1.0 + 1.0e-4) + 1.0e-7;
        }
        let merge = |t: &spark_lib::athc_compact::TraversalNode| t.count > 0 && !t.size_mode;
        if self.eps_bits > 0 {
            pq.eps = LogRange::over(page.iter().filter(|t| merge(t)).map(|t| f16::from_bits(t.eps).to_f32()), self.eps_bits);
        }
        if self.rho_bits > 0 {
            pq.rho = LogRange::over(page.iter().filter(|t| merge(t)).map(|t| f16::from_bits(t.rho).to_f32()), self.rho_bits);
        }
        for (t, code) in page.iter_mut().zip(codes.iter_mut()) {
            if !merge(t) {
                continue;
            }
            for (bits, range, v, c) in [(self.eps_bits, pq.eps, &mut t.eps, &mut code.eps), (self.rho_bits, pq.rho, &mut t.rho, &mut code.rho)] {
                let x = f16::from_bits(*v).to_f32();
                if bits == 0 || !(x > 0.0 && x.is_finite()) {
                    continue;
                }
                if let Some(k) = range.encode_up(x) {
                    *c = k;
                    *v = range.decode(k);
                }
            }
        }
        // A merge's axis requantized, its error added to its theta.
        if self.axis_bits > 0 && self.axis_bits < 8 {
            for (t, code) in page.iter_mut().zip(codes.iter_mut()).filter(|(t, _)| t.count > 0 && cone_present(t.cone)) {
                let axis = (t.cone >> 16) as u16;
                let (qx, ax) = self.requant_axis(axis & 255);
                let (qy, ay) = self.requant_axis(axis >> 8);
                let a2 = ax | ay << 8;
                code.axis = qx | qy << self.axis_bits;
                let err = oct_axis(axis).dot(oct_axis(a2)).clamp(-1.0, 1.0).acos();
                let theta = f16::from_bits((t.cone & 0xffff) as u16).to_f32() + err * (1.0 + 1.0e-4) + 1.0e-6;
                t.cone = f16_up(theta) as u32 | (a2 as u32) << 16;
            }
        }
        if self.theta_bits > 0 {
            pq.theta = LogRange::over(page.iter().filter(|t| cone_present(t.cone)).map(|t| f16::from_bits((t.cone & 0xffff) as u16).to_f32()), self.theta_bits);
            for (t, code) in page.iter_mut().zip(codes.iter_mut()) {
                let th = (t.cone & 0xffff) as u16;
                code.theta_class = match th {
                    NO_THETA => 0,
                    0 => 1,
                    0x8000 => 2,
                    _ => 3,
                };
                if !cone_present(t.cone) {
                    continue;
                }
                if let Some(k) = pq.theta.encode_up(f16::from_bits(th).to_f32()) {
                    code.theta = k;
                    t.cone = (t.cone & 0xffff0000) | pq.theta.decode(k) as u32;
                }
            }
        }
        (pq, codes)
    }
}

/// A 12-byte node (QUANT_RECORD_BITS), bit 0 the leaf flag:
/// both: centre 3 x 10 (bits 1..31), occlusion 13 (31..44);
/// a merge: eps 8 (44), rho 7 (52), place 12 (59), the children's page less
/// its own 8 (71), theta 6 (79: 0 no cone), axis 5 + 5 (85);
/// a leaf: theta's class 2 (44), theta 6 (46), axis 16 (52).
/// What does not fit (another child count than 2, a page 255 or more ahead,
/// a size-mode merge, an occlusion index past 13 bits, a code past its top)
/// is kept whole in the tree's `full` map.
#[derive(Clone, Copy, Debug, Default)]
struct QuantNode([u32; 3]);

impl QuantNode {
    fn put(&mut self, at: u32, bits: u32, v: u32) {
        debug_assert!(bits == 32 || v < 1 << bits);
        for b in 0..bits {
            let (w, k) = (((at + b) / 32) as usize, (at + b) % 32);
            if (v >> b) & 1 != 0 {
                self.0[w] |= 1 << k;
            }
        }
    }

    fn get(&self, at: u32, bits: u32) -> u32 {
        let (w, k) = ((at / 32) as usize, at % 32);
        let lo = (self.0[w] as u64) | (self.0.get(w + 1).copied().unwrap_or(0) as u64) << 32;
        ((lo >> k) & ((1u64 << bits) - 1)) as u32
    }
}

/// The record of node `t` (already quantized by `apply`) with its codes, or
/// None when it does not fit.
fn pack_quant(t: &spark_lib::athc_compact::TraversalNode, c: &NodeCodes, chunk: u32) -> Option<QuantNode> {
    let mut q = QuantNode::default();
    if t.occlusion as u32 >= 1 << 13 || t.size_mode {
        return None;
    }
    let leaf = t.count == 0;
    q.put(0, 1, leaf as u32);
    for k in 0..3 {
        q.put(1 + 10 * k as u32, 10, c.centre[k]);
    }
    q.put(31, 13, t.occlusion as u32);
    if leaf {
        if c.theta_class == 3 && c.theta == 0 {
            return None;
        }
        q.put(44, 2, c.theta_class);
        q.put(46, 6, c.theta);
        q.put(52, 16, t.cone >> 16);
    } else {
        let delta = t.child_page.wrapping_sub(chunk) & 0xffff;
        if t.count != 2 || delta >= 255 || c.eps == 0 || c.rho == 0 && t.rho != 0 {
            return None;
        }
        let th = (t.cone & 0xffff) as u16;
        let theta = if th == NO_THETA { 0 } else if cone_present(t.cone) && c.theta > 0 { c.theta } else { return None };
        q.put(44, 8, c.eps);
        q.put(52, 7, c.rho);
        q.put(59, 12, t.place);
        q.put(71, 8, delta);
        q.put(79, 6, theta);
        q.put(85, 10, if theta == 0 { 0 } else { c.axis });
    }
    Some(q)
}

/// A record back to the node's LodSplat (what `apply` wrote back, bit for bit).
fn unpack_quant(q: &QuantNode, p: &PageQuant, bits: &CompactQuant) -> LodSplat {
    let centre = p.centre([q.get(1, 10), q.get(11, 10), q.get(21, 10)]);
    let leaf = q.get(0, 1) != 0;
    let mut s;
    if leaf {
        s = LodSplat::new_f16(centre.map(f16::from_bits), f16::ZERO, 0, 0);
        s.theta = f16::from_bits(match q.get(44, 2) {
            0 => NO_THETA,
            1 => 0,
            2 => 0x8000,
            _ => p.theta.decode(q.get(46, 6)),
        });
        s.cone_axis = q.get(52, 16) as u16;
    } else {
        let child_page = (p.chunk + q.get(71, 8)) & 0xffff;
        s = LodSplat::new_f16(centre.map(f16::from_bits), f16::ZERO, child_page << 16 | q.get(59, 12), 2);
        s.eps = f16::from_bits(p.eps.decode(q.get(44, 8)));
        let rho = q.get(52, 7);
        s.rho = f16::from_bits(if rho == 0 { 0 } else { p.rho.decode(rho) });
        let theta = q.get(79, 6);
        if theta == 0 {
            s.theta = f16::from_bits(NO_THETA);
        } else {
            s.theta = f16::from_bits(p.theta.decode(theta));
            let top = ((1u32 << bits.axis_bits) - 1) as f32;
            let a = q.get(85, 10);
            let ax = ((a & 31) as f32 * 255.0 / top).round() as u16;
            let ay = ((a >> 5) as f32 * 255.0 / top).round() as u16;
            s.cone_axis = ax | ay << 8;
        }
    }
    s.occlusion = q.get(31, 13) as u16;
    s.margin = p.margin;
    s
}

#[cfg(test)]
mod quant_tests {
    use super::*;
    use spark_lib::athc_compact::{traversal_input, TraversalNode};
    use spark_lib::athc_v3::{gunzip, parse_v3, SectionId};

    fn pages() -> Vec<(u32, Vec<TraversalNode>)> {
        let bytes = include_bytes!("../../../test/fixtures/athc/compact_pages.athc");
        let layout = parse_v3(bytes).unwrap();
        let k4 = layout.sections.iter().position(|s| s.id == SectionId::Compact).unwrap();
        layout
            .blocks
            .iter()
            .enumerate()
            .map(|(p, b)| {
                let n = b.n as usize;
                let span = b.spans[k4];
                let raw = gunzip(&bytes[span.offset as usize..][..span.stored as usize]).unwrap();
                let input = traversal_input(&raw, n);
                let head = layout.pages[p].compact;
                (p as u32, (0..n).map(|i| spark_lib::athc_compact::traversal_node_input(&input, n, &head, p as u32, i)).collect())
            })
            .collect()
    }

    /// The 12-byte records decode to what `apply` wrote back, and the codes
    /// only round up: eps, rho, theta not below the exact values, the centres
    /// within the margin.
    #[test]
    fn records_decode_to_the_quantized_nodes_and_round_up() {
        let mut packed = 0;
        let mut total = 0;
        for (chunk, exact) in pages() {
            let mut page = exact.clone();
            let (mut pq, codes) = QUANT_RECORD_BITS.apply(&mut page);
            pq.chunk = chunk;
            for ((t, e), c) in page.iter().zip(&exact).zip(&codes) {
                total += 1;
                let up = |q: u16, x: u16| f16::from_bits(q).to_f32() >= f16::from_bits(x).to_f32();
                assert!(up(t.eps, e.eps) && up(t.rho, e.rho));
                if cone_present(e.cone) {
                    assert!(up(t.cone as u16, e.cone as u16));
                }
                let d = Vec3A::from_array(t.centre.map(|v| f16::from_bits(v).to_f32())) - Vec3A::from_array(e.centre.map(|v| f16::from_bits(v).to_f32()));
                assert!(d.length() <= pq.margin);
                let Some(q) = pack_quant(t, c, chunk) else { continue };
                packed += 1;
                let s = unpack_quant(&q, &pq, &QUANT_RECORD_BITS);
                let w = AHashMap::new();
                let code = t.count as u16;
                let r = compact_splat(
                    [t.centre[0], t.centre[1], t.centre[2], t.eps, t.rho, t.child_page as u16, (t.place as u16 & 0xfff) | code << 12, t.cone as u16, (t.cone >> 16) as u16, t.occlusion],
                    &w,
                    0,
                );
                assert_eq!((s.center, s.child_start, s.child_count, s.occlusion), (r.center, r.child_start, r.child_count, r.occlusion));
                assert_eq!((s.eps.to_bits(), s.rho.to_bits(), s.theta.to_bits()), (r.eps.to_bits(), r.rho.to_bits(), r.theta.to_bits()), "node {t:?}");
                if cone_present(t.cone) {
                    assert_eq!(s.cone_axis, r.cone_axis);
                }
                assert_eq!(s.margin, pq.margin);
            }
        }
        assert!(packed * 10 >= total * 9, "{packed} of {total} in records");
    }

    /// The conservative occlusion against the exact rule on random nodes,
    /// eyes and masks: the quantized rule never hides what the exact one
    /// keeps, and with no margin it is the exact rule.
    #[test]
    fn quantized_occlusion_is_conservative() {
        let mut seed = 0x12345678u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f32 / (1u64 << 53) as f32
        };
        let (mut hidden, mut extra) = (0, 0);
        for _ in 0..200_000 {
            // Masks mostly closed (a few open cells), so occlusion happens.
            let mut mask = [0u32; 8];
            for _ in 0..(rnd() * 6.0) as usize {
                let bit = (rnd() * 256.0) as usize % 256;
                mask[bit >> 5] |= 1 << (bit & 31);
            }
            let c = Vec3A::new(rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0);
            let eye = Vec3A::new(rnd() * 8.0 - 4.0, rnd() * 8.0 - 4.0, rnd() * 8.0 - 4.0);
            let margin = rnd() * 0.05;
            let off = Vec3A::new(rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0) * margin / 3f32.sqrt();
            let exact = occluded_dir(&mask, eye - (c + off));
            let quant = occluded_box(&mask, eye - c, margin);
            assert!(!quant || exact, "culled only by the quantized rule: c {c:?} eye {eye:?} margin {margin}");
            hidden += exact as u32;
            extra += (exact && !quant) as u32;
        }
        assert!(hidden > 1000, "{hidden} hidden");
        eprintln!("hidden {hidden}, kept by the margin {extra}");
    }

    /// thread CU: the whole car's nodes (CU_CAR_DIR: car-compact-tx-v5) at
    /// athenea's Camera, CameraFar, the close-up and random eyes: the nodes
    /// the quantized rule hides that the exact centre keeps (must be 0), CT's
    /// rule (the stored centre, no margin) for comparison, and those the
    /// margin keeps. `cargo test --release -- --ignored car_occlusion --nocapture`
    #[test]
    #[ignore]
    fn car_occlusion() {
        use spark_lib::athc_compact::{decode_runs, CompactAsset};
        let Ok(dir) = std::env::var("CU_CAR_DIR") else { return };
        let asset = CompactAsset::from_bytes(&std::fs::read(format!("{dir}/car.cmpa")).unwrap()).unwrap();
        let occ = (asset.occlusion.clone(), asset.occlusion_box);
        // Y-up world (three) to the Z-up stage: (x, -z, y).
        let st = |p: [f32; 3]| Vec3A::new(p[0], -p[2], p[1]);
        let (cam, tgt, far) = ([4.378f32, 1.676, 5.854], [0.0f32, 0.553, 0.252], [13.4994f32, 4.0228, 17.5284]);
        let close = [0, 1, 2].map(|k| cam[k] + 0.55 * (tgt[k] - cam[k]));
        let mut eyes = vec![("cam".to_string(), st(cam)), ("far".to_string(), st(far)), ("close".to_string(), st(close))];
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f32 / (1u64 << 53) as f32
        };
        for k in 0..24 {
            let (az, el, r) = (rnd() * std::f32::consts::TAU, rnd() * 1.2 - 0.1, 2.0 + rnd() * 18.0);
            eyes.push((format!("rand{k}"), Vec3A::new(r * el.cos() * az.cos(), r * el.cos() * az.sin(), 0.5 + r * el.sin())));
        }
        // [exact, ct, cu, ct-only, cu-only, kept by the margin] an eye
        let mut counts = vec![[0u64; 6]; eyes.len()];
        let mut nodes = 0u64;
        let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "athc")).collect();
        files.sort();
        for f in files {
            let bytes = std::fs::read(&f).unwrap();
            let layout = parse_v3(&bytes).unwrap();
            let k4 = layout.sections.iter().position(|s| s.id == SectionId::Compact).unwrap();
            for (p, b) in layout.blocks.iter().enumerate() {
                let span = b.spans[k4];
                let (n, raw) = decode_runs(&bytes[span.offset as usize..][..span.stored as usize]).unwrap();
                let input = traversal_input(&raw, n);
                let head = layout.pages[p].compact;
                let exact: Vec<TraversalNode> = (0..n).map(|i| spark_lib::athc_compact::traversal_node_input(&input, n, &head, p as u32, i)).collect();
                let mut page = exact.clone();
                let (pq, _) = QUANT_RECORD_BITS.apply(&mut page);
                for (e, q) in exact.iter().zip(&page) {
                    nodes += 1;
                    if e.occlusion == 0 {
                        continue;
                    }
                    let mut se = LodSplat::new_f16(e.centre.map(f16::from_bits), f16::ZERO, 0, 0);
                    se.occlusion = e.occlusion;
                    let mut sq = LodSplat::new_f16(q.centre.map(f16::from_bits), f16::ZERO, 0, 0);
                    sq.occlusion = q.occlusion;
                    let mut sm = sq.clone();
                    sm.margin = pq.margin;
                    for (k, (_, eye)) in eyes.iter().enumerate() {
                        let a = occluded(&se, *eye, &occ);
                        let b = occluded(&sq, *eye, &occ);
                        let c = occluded(&sm, *eye, &occ);
                        let row = &mut counts[k];
                        row[0] += a as u64;
                        row[1] += b as u64;
                        row[2] += c as u64;
                        row[3] += (b && !a) as u64;
                        row[4] += (c && !a) as u64;
                        row[5] += (a && !c) as u64;
                    }
                }
            }
        }
        eprintln!("{nodes} nodes");
        eprintln!("eye: exact-hidden ct-hidden cu-hidden | ct-only cu-only | kept-by-margin");
        for ((name, _), r) in eyes.iter().zip(&counts) {
            eprintln!("{name}: {} {} {} | {} {} | {}", r[0], r[1], r[2], r[3], r[4], r[5]);
        }
        assert!(counts.iter().all(|r| r[4] == 0), "nodes culled only by the quantized rule");
    }
}
