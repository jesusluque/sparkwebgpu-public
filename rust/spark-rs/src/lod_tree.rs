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
        if !self.has_error() || !theta.is_finite() {
            return false;
        }
        let v = eye - self.center();
        let d = v.length();
        let rho = self.rho.to_f32();
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
#[derive(Debug, Clone, Copy, Default)]
struct CompactLodNode([u16; 7]);

const COMPACT_WIDE: u16 = 15;
const COMPACT_SIZE_MODE: u16 = 0x8000;

#[derive(Debug, Clone)]
enum LodNodes {
    Full(Vec<LodSplat>),
    Compact { nodes: Vec<CompactLodNode>, wide: AHashMap<u32, u16> },
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
            LodNodes::Compact { nodes, wide } => {
                let w = nodes[i].0;
                let code = w[6] >> 12;
                let count = if code == COMPACT_WIDE { wide.get(&(i as u32)).copied().unwrap_or(0) } else { code };
                let center = [f16::from_bits(w[0]), f16::from_bits(w[1]), f16::from_bits(w[2])];
                let child_start = if count > 0 { (w[5] as u32) << 16 | (w[6] & 0xfff) as u32 } else { 0 };
                let mut s = LodSplat::new_f16(center, f16::ZERO, child_start, count);
                if w[4] & COMPACT_SIZE_MODE != 0 {
                    s.size = f16::from_bits(w[3]);
                } else {
                    s.eps = f16::from_bits(w[3]);
                    s.rho = f16::from_bits(w[4]);
                    s.theta = f16::INFINITY;
                }
                s
            }
        }
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        match self {
            LodNodes::Full(v) => v.len(),
            LodNodes::Compact { nodes, .. } => nodes.len(),
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
    let d = eye - s.center();
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
        let cx = ((lo[0] + kx) as i32).clamp(0, 15) as usize;
        let cy = ((lo[1] + ky) as i32).clamp(0, 15) as usize;
        let bit = cy * 16 + cx;
        if (codebook[entry + (bit >> 5)] >> (bit & 31)) & 1 != 0 {
            return false;
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
    frontier: Frontier<(OrderedFloat<f32>, u32, u32)>,
    output: Vec<(u32, u32)>,
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
            LodNodes::Compact { nodes: Vec::with_capacity(capacity as usize), wide: AHashMap::new() }
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
                continue;
            }
            lod_tree.page_to_chunk[pool_page] = chunk as u32;
            lod_tree.chunk_to_page[chunk] = pool_page as u32;
            let input = Uint8Array::from(page).to_vec();
            let n = count as usize;
            if input.len() != n * spark_lib::athc_compact::TRAVERSAL_INPUT_BYTES {
                return Err(JsValue::from_str("update_compact_lod_trees: a page's input of the wrong size"));
            }
            let head: Vec<u32> = boxes[6 * k..6 * k + 6].iter().map(|v| v.to_bits()).collect();
            let mut store = lod_tree.splats.borrow_mut();
            let LodNodes::Compact { nodes, wide } = &mut *store else {
                return Err(JsValue::from_str("update_compact_lod_trees: not a compact tree"));
            };
            let base = page_base as usize;
            if base + n > nodes.len() {
                nodes.resize((nodes.len() * 2).max(base + n), CompactLodNode::default());
            }
            for i in 0..n {
                let t = spark_lib::athc_compact::traversal_node_input(&input, n, &head, chunk as u32, i);
                let code = if t.count >= COMPACT_WIDE as u32 { COMPACT_WIDE } else { t.count as u16 };
                if code == COMPACT_WIDE {
                    wide.insert((base + i) as u32, t.count as u16);
                } else {
                    wide.remove(&((base + i) as u32));
                }
                nodes[base + i] = CompactLodNode([
                    t.centre[0],
                    t.centre[1],
                    t.centre[2],
                    t.eps,
                    t.rho | if t.size_mode { COMPACT_SIZE_MODE } else { 0 },
                    t.child_page as u16,
                    (t.place as u16 & 0xfff) | code << 12,
                ]);
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
    error_size_k: Option<f32>,
) -> anyhow::Result<Object, JsValue> {
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
    let back_cull_min = back_cull_min.filter(|m| m.is_finite() && *m >= 0.0);
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
        let LodState { lod_trees, frontier, output, touched, touched_set, .. } = state;
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
        output.reserve(max_splats);
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
            frontier.push((OrderedFloat(pixel_scale), inst_index as u32, root_index));
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
                s.size() / (s.center() - i.4).length().max(1.0e-6) > min && s.back_facing(i.4)
            })
        };

        while let Some(&(OrderedFloat(pixel_scale), inst_index, paged_index)) = frontier.peek() {
            min_pixel_scale = min_pixel_scale.min(pixel_scale);
            if pixel_scale <= pixel_scale_limit {
                break;
            }

            let instance = &instances[inst_index as usize];
            let (lod_id, splats, _page_to_chunk, chunk_to_page, ..) = instance;
            let LodSplat { child_count, child_start, .. } = splats.get(paged_index as usize);

            if child_count == 0 {
                _ = frontier.pop();
                output.push((inst_index, paged_index));
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
                } else {
                    output.push((inst_index, paged_index));
                }
                continue;
            }
            let first_page = chunk_to_page[first_chunk as usize];
            let last_page = chunk_to_page[last_chunk as usize];

            if first_page == 0xFFFFFFFF || last_page == 0xFFFFFFFF {
                if culled(inst_index, paged_index) {
                    culled_count += 1;
                } else {
                    output.push((inst_index, paged_index));
                }
                continue;
            }

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
                    } else {
                        output.push((inst_index, paged_index));
                    }
                } else {
                    frontier.push((OrderedFloat(pixel_scale), inst_index, paged_index));
                }
            }

            occluded_count += hidden_here;
            num_splats = new_num_splats - hidden_here as usize;
        }

        let output_size = output.len();
        let frontier_size = frontier.len();

        for (_, inst_index, paged_index) in frontier.drain() {
            if culled(inst_index, paged_index) {
                culled_count += 1;
            } else {
                output.push((inst_index, paged_index));
            }
        }

        let mut instance_counts = vec![0; num_instances];
        for &(inst_index, _) in output.iter() {
            instance_counts[inst_index as usize] += 1;
        }

        let mut instance_outputs = Vec::with_capacity(num_instances);
        for counts in instance_counts {
            instance_outputs.push(Vec::with_capacity(counts));
        }

        for &(inst_index, paged_index) in output.iter() {
            instance_outputs[inst_index as usize].push(paged_index);
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
    let distance = delta.length().max(1.0e-6);
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
