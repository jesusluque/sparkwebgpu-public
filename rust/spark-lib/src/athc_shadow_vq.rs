//! A dictionary of a TX cloud's open-direction bits (`shadowBits`, 8 words =
//! 256 cells a gaussian): athenea's proposal 096 (a'), k-means in Hamming
//! space per piece, with athenea's own parameters (thread CB, research/
//! iphone-compact.md):
//!   - `centroids` centroids a piece (1024);
//!   - `rounds` rounds of assign + update (12);
//!   - a centroid is the bitwise majority of its members (an empty one keeps
//!     its bits);
//!   - centroid c is seeded from the piece's member floor(c n / C), in cloud
//!     order.
//!
//! A .athc keeps no prim id (athenea's pieces are prims), so a piece here is
//! a run of `piece_splats` splats (16384) in cloud order (the cut's order is a depth-
//! first walk of its tree, so a run is a place on the car); with
//! `piece_splats` 0 it is a material (the rows that share `pbr` and `lobes`:
//! the Corvette's are textured, 22k of them at 1M, so runs it is). Only the
//! splats (the chunks) are fitted; the merged levels above keep their bits. A
//! piece with no more distinct masks than `centroids` keeps them exactly.
//!
//! `shadow_vq` writes each row's centroid in place of its bits, so the file
//! keeps its layout; a reader dedups each piece's masks into a codebook of at
//! most `centroids` and a 10-bit index a gaussian (src/webgpu/athenea/
//! compact.ts). Cloud order: the chunks (the splats), then the levels finest
//! first.

use crate::athc::{AthcBlock, AthcFile};
use anyhow::{bail, Result};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct ShadowVqOptions {
    pub centroids: usize,
    pub rounds: usize,
    /// Splats a piece (a run in cloud order); 0: a piece a material.
    pub piece_splats: usize,
    pub threads: usize,
}

impl Default for ShadowVqOptions {
    fn default() -> Self {
        Self {
            centroids: 1024,
            rounds: 12,
            piece_splats: 16384,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()).min(16),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ShadowVqStats {
    pub pieces: usize,
    /// Pieces kept exactly (no more distinct masks than centroids).
    pub exact_pieces: usize,
    pub rows: usize,
    /// Hamming distance of each splat (chunk row) to its centroid, histogram 0..=256.
    pub splat_hist: Vec<u64>,
    /// The same over every row, levels included.
    pub all_hist: Vec<u64>,
    /// Codebook entries over every piece (distinct masks after the fit).
    pub codebook_entries: usize,
}

impl ShadowVqStats {
    /// The p-th percentile (0..1) of a histogram.
    pub fn percentile(hist: &[u64], p: f64) -> usize {
        let total: u64 = hist.iter().sum();
        if total == 0 {
            return 0;
        }
        let want = (p * total as f64).ceil().max(1.0) as u64;
        let mut acc = 0;
        for (d, &c) in hist.iter().enumerate() {
            acc += c;
            if acc >= want {
                return d;
            }
        }
        hist.len() - 1
    }
    pub fn mean(hist: &[u64]) -> f64 {
        let total: u64 = hist.iter().sum();
        let s: u64 = hist.iter().enumerate().map(|(d, &c)| d as u64 * c).sum();
        s as f64 / total.max(1) as f64
    }
}

type Mask = [u64; 4];

fn mask_of(words: &[u32]) -> Mask {
    let mut m = [0u64; 4];
    for k in 0..4 {
        m[k] = words[2 * k] as u64 | (words[2 * k + 1] as u64) << 32;
    }
    m
}

fn write_mask(m: &Mask, words: &mut [u32]) {
    for k in 0..4 {
        words[2 * k] = m[k] as u32;
        words[2 * k + 1] = (m[k] >> 32) as u32;
    }
}

#[inline]
fn hamming(a: &Mask, b: &Mask) -> u32 {
    (a[0] ^ b[0]).count_ones() + (a[1] ^ b[1]).count_ones() + (a[2] ^ b[2]).count_ones() + (a[3] ^ b[3]).count_ones()
}

/// The nearest centroid of each mask (ties: the lowest index).
pub fn assign(masks: &[Mask], centroids: &[Mask], threads: usize, out: &mut [u32]) {
    let per = masks.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (ms, os) in masks.chunks(per).zip(out.chunks_mut(per)) {
            s.spawn(move || {
                for (m, o) in ms.iter().zip(os.iter_mut()) {
                    let mut best = u32::MAX;
                    let mut at = 0u32;
                    for (c, cm) in centroids.iter().enumerate() {
                        let d = hamming(m, cm);
                        if d < best {
                            best = d;
                            at = c as u32;
                            if d == 0 {
                                break;
                            }
                        }
                    }
                    *o = at;
                }
            });
        }
    });
}

/// k-means in Hamming space over one piece's masks (cloud order): the
/// centroids and each mask's.
pub fn kmeans_hamming(masks: &[Mask], k: usize, rounds: usize, threads: usize) -> (Vec<Mask>, Vec<u32>) {
    let n = masks.len();
    let mut centroids: Vec<Mask> = (0..k).map(|c| masks[c * n / k]).collect();
    let mut label = vec![0u32; n];
    for _ in 0..rounds {
        assign(masks, &centroids, threads, &mut label);
        // Bitwise majority of the members.
        let mut counts = vec![[0u32; 256]; k];
        let mut members = vec![0u32; k];
        for (m, &l) in masks.iter().zip(&label) {
            let c = &mut counts[l as usize];
            members[l as usize] += 1;
            for (w, &word) in m.iter().enumerate() {
                let mut v = word;
                while v != 0 {
                    let b = v.trailing_zeros() as usize;
                    c[w * 64 + b] += 1;
                    v &= v - 1;
                }
            }
        }
        for c in 0..k {
            if members[c] == 0 {
                continue;
            }
            let mut m = [0u64; 4];
            for bit in 0..256 {
                // Strict majority; a tie keeps the bit open (1) as the
                // seed's half does not decide it.
                if 2 * counts[c][bit] >= members[c] {
                    m[bit / 64] |= 1u64 << (bit % 64);
                }
            }
            centroids[c] = m;
        }
    }
    assign(masks, &centroids, threads, &mut label);
    (centroids, label)
}

/// Rows of the file in cloud order: (block, row) with block 0.. the chunks,
/// then the levels finest first.
fn blocks_mut(file: &mut AthcFile) -> Vec<&mut AthcBlock> {
    let mut out: Vec<&mut AthcBlock> = file.chunks.iter_mut().collect();
    let mut levels: Vec<&mut AthcBlock> = file.levels.iter_mut().map(|(_, b)| b).collect();
    levels.reverse();
    out.extend(levels);
    out
}

/// The dictionary of every piece's bits, written back as each row's centroid.
pub fn shadow_vq(file: &mut AthcFile, o: &ShadowVqOptions) -> Result<ShadowVqStats> {
    let words = file.extra.shadow_words as usize;
    if words != 8 {
        bail!("shadow-vq: {words} shadow words a splat (needs a TX cloud's 8)");
    }
    let pbr_words = file.extra.pbr_words as usize;
    let lobes_words = file.extra.lobes_words as usize;
    let chunk_blocks = file.chunks.len();
    let mut blocks = blocks_mut(file);
    // Rows by piece, in cloud order.
    let mut piece_of: HashMap<Vec<u32>, usize> = HashMap::new();
    let mut rows: Vec<Vec<(u32, u32)>> = Vec::new();
    let mut splat = 0usize;
    for (b, block) in blocks.iter().enumerate().take(chunk_blocks) {
        for r in 0..block.n {
            if let Some(p) = splat.checked_div(o.piece_splats) {
                if p == rows.len() {
                    rows.push(Vec::new());
                }
                rows[p].push((b as u32, r as u32));
                splat += 1;
                continue;
            }
            let mut key = Vec::with_capacity(pbr_words + lobes_words);
            key.extend_from_slice(&block.pbr[r * pbr_words..(r + 1) * pbr_words]);
            key.extend_from_slice(&block.lobes[r * lobes_words..(r + 1) * lobes_words]);
            let next = rows.len();
            let p = *piece_of.entry(key).or_insert(next);
            if p == next {
                rows.push(Vec::new());
            }
            rows[p].push((b as u32, r as u32));
        }
    }
    let mut stats = ShadowVqStats {
        pieces: rows.len(),
        splat_hist: vec![0; 257],
        all_hist: vec![0; 257],
        ..Default::default()
    };
    for piece in &rows {
        stats.rows += piece.len();
        let masks: Vec<Mask> = piece
            .iter()
            .map(|&(b, r)| mask_of(&blocks[b as usize].shadow_bits[r as usize * 8..r as usize * 8 + 8]))
            .collect();
        let mut distinct: Vec<Mask> = masks.clone();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() <= o.centroids {
            stats.exact_pieces += 1;
            stats.codebook_entries += distinct.len();
            stats.all_hist[0] += piece.len() as u64;
            stats.splat_hist[0] += piece.iter().filter(|&&(b, _)| (b as usize) < chunk_blocks).count() as u64;
            continue;
        }
        let (centroids, label) = kmeans_hamming(&masks, o.centroids, o.rounds, o.threads);
        let mut used: Vec<Mask> = label.iter().map(|&l| centroids[l as usize]).collect();
        for (k, &(b, r)) in piece.iter().enumerate() {
            let d = hamming(&masks[k], &used[k]) as usize;
            stats.all_hist[d] += 1;
            if (b as usize) < chunk_blocks {
                stats.splat_hist[d] += 1;
            }
            let at = r as usize * 8;
            write_mask(&used[k], &mut blocks[b as usize].shadow_bits[at..at + 8]);
        }
        used.sort_unstable();
        used.dedup();
        stats.codebook_entries += used.len();
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_finds_two_masks() {
        let a: Mask = [0xFFFF, 0, 0, 0];
        let b: Mask = [0, 0, 0xFFFF_0000, 7];
        let mut masks = Vec::new();
        for i in 0..100 {
            let mut m = if i % 2 == 0 { a } else { b };
            m[1] = (i % 3) as u64; // a bit or two of noise
            masks.push(m);
        }
        let (c, l) = kmeans_hamming(&masks, 2, 12, 2);
        for (m, &k) in masks.iter().zip(&l) {
            assert!(hamming(m, &c[k as usize]) <= 2);
        }
        assert_ne!(l[0], l[1]);
    }

    #[test]
    fn majority_is_bitwise() {
        let masks: Vec<Mask> = vec![[0b011, 0, 0, 0], [0b110, 0, 0, 0], [0b010, 0, 0, 0]];
        let (c, _) = kmeans_hamming(&masks, 1, 1, 1);
        assert_eq!(c[0][0], 0b010);
    }
}
