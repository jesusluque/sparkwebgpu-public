//! Basis or bake? (thread CX, the v8 nose speckle.) For a compact .athc
//! (usd-athc --compact-basis, CMPC v4 runs) against its asset table and the
//! basis it was written with: each leaf's FULL transfer (tiers 1-3, as baked)
//! next to the compact reconstruction (its cluster and int4 weights through
//! the basis), and whether either stands out from its neighbours.
//!
//!   athc-tx-diag piece.athc car.cmpa basis.json [--cell 0.005] [--ratio 3] [--top 20] [--box x0,y0,z0,x1,y1,z1]
//!
//! A leaf is "bright" in a field (full or compact) when its brightness (the
//! sum of the DC terms of the indirect and reflected blocks' RGB, the parts
//! a camera sees) passes `ratio` times the median of the leaves in its cell
//! and the 26 around it (cells of `cell` metres) OF THE SAME MATERIAL (a speck,
//! not a trim next to black plastic). Bright in full = the bake;
//! bright only in compact = the basis. Prints counts, the reconstruction
//! error of each set, and the brightest outliers (page, place, centre).

use anyhow::{bail, Context, Result};
use spark_lib::athc::{high_half, low_half};
use spark_lib::athc_compact::{decode_node, decode_runs, CompactAsset, CompactTables, JointBasis, HEAD_WORDS};
use std::collections::HashMap;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |n: &str, d: f64| -> f64 { args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d) };
    let pos: Vec<&String> = args.iter().enumerate().filter(|(i, a)| !a.starts_with("--") && !(*i > 0 && args[i - 1].starts_with("--"))).map(|(_, a)| a).collect();
    let [file, cmpa, basis_json] = pos[..] else { bail!("usage: athc-tx-diag piece.athc car.cmpa basis.json [--cell m] [--ratio r] [--top n]") };
    let (cell, ratio, top) = (opt("--cell", 0.005), opt("--ratio", 3.0), opt("--top", 20.0) as usize);
    // `--box x0,y0,z0,x1,y1,z1`: only the leaves inside (the file's frame).
    let bx: Option<Vec<f32>> = args.iter().position(|a| a == "--box").and_then(|i| args.get(i + 1)).map(|v| v.split(',').filter_map(|t| t.parse().ok()).collect());
    let bytes = std::fs::read(file)?;
    let asset = CompactAsset::from_bytes(&std::fs::read(cmpa)?)?;
    let bin = basis_json.strip_suffix(".json").map(|p| format!("{p}.bin")).context("basis .json")?;
    let basis = JointBasis::parse(&std::fs::read_to_string(basis_json)?, &std::fs::read(bin)?)?;
    let layout = spark_lib::athc_v3::parse_v3(&bytes)?;
    let [at, len] = layout.compact_table;
    let tables = CompactTables::from_bytes(&bytes[at as usize..(at + len) as usize])?.resolve(Some(&asset))?;
    let k4 = layout.sections.iter().position(|s| s.id == spark_lib::athc_v3::SectionId::Compact).context("no CMPC")?;
    let full = spark_lib::athc_v3::read_v3(&bytes)?;
    if full.chunks.len() != layout.blocks.len() {
        bail!("{} chunks, {} blocks", full.chunks.len(), layout.blocks.len());
    }
    // (page, place, centre, full brightness, compact brightness, per-leaf sq error, sq ref)
    let mut leaves: Vec<(u32, u32, [f32; 3], f32, f32, f64, f64)> = Vec::new();
    let mut mats: Vec<[u32; 4]> = Vec::new();
    let bright = |x: &[f32]| -> f32 { (0..3).map(|c| x[16 + c] + x[64 + c]).sum() };
    for (p, (blk, entry)) in full.chunks.iter().zip(&layout.blocks).enumerate() {
        let n = blk.n;
        if blk.transfer.len() != 56 * n {
            bail!("page {p}: not the full transfer");
        }
        let span = entry.spans[k4];
        let (m, raw) = decode_runs(&bytes[span.offset as usize..][..span.stored as usize])?;
        if m != n {
            bail!("page {p}: {m} compact nodes, {n} in the tiers");
        }
        let head: [u32; HEAD_WORDS] = layout.pages[p].compact;
        for i in 0..n {
            let node = decode_node(&raw, n, &head, &tables, p as u32, i);
            if node.traversal.count != 0 {
                continue; // leaves only
            }
            let mut x = [0f32; 112];
            for v in 0..112 {
                let w = blk.transfer[56 * i + v / 2];
                x[v] = if v % 2 == 0 { low_half(w) } else { high_half(w) };
            }
            let y = basis.decode(node.cluster, &node.weights, 7);
            let (mut e, mut r) = (0f64, 0f64);
            for v in 0..112 {
                e += ((y[v] - x[v]) as f64).powi(2);
                r += (x[v] as f64).powi(2);
            }
            let c = [blk.positions[4 * i], blk.positions[4 * i + 1], blk.positions[4 * i + 2]];
            if let Some(b) = &bx {
                if (0..3).any(|k| c[k] < b[k] || c[k] > b[3 + k]) {
                    continue;
                }
            }
            leaves.push((p as u32, i as u32, c, bright(&x), bright(&y), e, r));
            mats.push(node.material);
        }
    }
    // Cells and each leaf's neighbourhood median (full and compact).
    let key = |c: [f32; 3]| [0, 1, 2].map(|k| (c[k] as f64 / cell).floor() as i32);
    let mut cells: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
    for (j, l) in leaves.iter().enumerate() {
        cells.entry(key(l.2)).or_default().push(j);
    }
    let mut out_full = Vec::new();
    let mut out_comp = Vec::new();
    let (mut e_all, mut r_all, mut e_cf, mut r_cf) = (0f64, 0f64, 0f64, 0f64);
    for (j, l) in leaves.iter().enumerate() {
        e_all += l.5;
        r_all += l.6;
        let k = key(l.2);
        let mut fv = Vec::new();
        let mut cv = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = cells.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                        for &o in v {
                            if o != j && mats[o] == mats[j] {
                                fv.push(leaves[o].3);
                                cv.push(leaves[o].4);
                            }
                        }
                    }
                }
            }
        }
        if fv.len() < 8 {
            continue;
        }
        let med = |v: &mut Vec<f32>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let (mf, mc) = (med(&mut fv), med(&mut cv));
        let bf = l.3 > ratio as f32 * mf.max(1e-4);
        let bc = l.4 > ratio as f32 * mc.max(1e-4);
        if bf {
            out_full.push((l.3 / mf.max(1e-4), j));
        }
        if bc && !bf {
            out_comp.push((l.4 / mc.max(1e-4), j));
            e_cf += l.5;
            r_cf += l.6;
        }
    }
    println!("{file}: {} leaves, relMSE {:.4}", leaves.len(), e_all / r_all.max(1e-30));
    println!(
        "bright in the FULL transfer (the bake): {} ({:.3}%); bright only in the COMPACT reconstruction (the basis): {} ({:.3}%), their relMSE {:.4}",
        out_full.len(),
        100.0 * out_full.len() as f64 / leaves.len() as f64,
        out_comp.len(),
        100.0 * out_comp.len() as f64 / leaves.len() as f64,
        e_cf / r_cf.max(1e-30)
    );
    for (name, mut v) in [("full", out_full), ("compact only", out_comp)] {
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (q, j) in v.iter().take(top) {
            let l = &leaves[*j];
            println!("  {name}: x{q:.1} page {} place {} centre {:.4} {:.4} {:.4} full {:.3} compact {:.3}", l.0, l.1, l.2[0], l.2[1], l.2[2], l.3, l.4);
        }
    }
    Ok(())
}
