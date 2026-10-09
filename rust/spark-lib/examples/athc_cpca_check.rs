//! What CPCA of the transfer (athc_cpca) costs, on the CPU, element by
//! element (merged nodes included), the uncompressed file against the
//! compressed one:
//!
//!   cargo run --release --example athc_cpca_check -- ORIGINAL.athc CPCA.athc [HDR ...]
//!
//! - `coef`: relMSE of the transfer's values, a half at a time (direct,
//!   indirect, field). Over the sphere this is the L2 error of the transfer
//!   as a function (the basis is orthonormal).
//! - `sky <hdr> <up>`: what splat_relight's `transferredBody` sums, the
//!   transfer dotted with the sky's sixteen harmonics (rgb; the indirect half
//!   too where kept), the HDRI projected on athenea's basis (`envBasisValue`)
//!   with +Z or +Y up. relMSE over elements, and over the darker half alone.
//! - `point`: the direct half and the field evaluated along 64 directions
//!   (the sun's share, `splatIndirectAlong`, the field's reflection).
//!
//! Prints one JSON line per file pair.

use anyhow::{bail, Result};
use half::f16;
use spark_lib::athc::AthcFile;
use spark_lib::athc_cpca::zonal_harmonics;
use spark_lib::athc_v3::{read_v3, ATH3_MAGIC};

fn read_any(path: &str) -> Result<AthcFile> {
    let bytes = std::fs::read(path)?;
    if u32::from_le_bytes(bytes[..4].try_into().unwrap()) == ATH3_MAGIC {
        read_v3(&bytes)
    } else {
        AthcFile::read(&bytes)
    }
}

/// Every element's transfer (levels then chunks), as floats.
fn transfers(f: &AthcFile) -> (usize, Vec<f32>) {
    let count = f.extra.transfer_count as usize;
    let words = f.extra.transfer_words as usize;
    let mut out = Vec::new();
    for b in f.levels.iter().map(|(_, b)| b).chain(f.chunks.iter()) {
        for e in 0..b.n {
            for v in 0..count {
                let w = b.transfer[e * words + v / 2];
                let h = if v % 2 == 0 { w & 0xffff } else { w >> 16 };
                out.push(f16::from_bits(h as u16).to_f32());
            }
        }
    }
    (count, out)
}

/// splat_relight's envBasisValue, degree 3.
fn basis(d: [f64; 3]) -> [f64; 16] {
    let [x, y, z] = d;
    [
        0.28209479177387814,
        -0.4886025119029199 * y,
        0.4886025119029199 * z,
        -0.4886025119029199 * x,
        1.0925484305920792 * x * y,
        -1.0925484305920792 * y * z,
        0.31539156525252005 * (2.0 * z * z - x * x - y * y),
        -1.0925484305920792 * x * z,
        0.5462742152960396 * (x * x - y * y),
        -0.5900435899266435 * y * (3.0 * x * x - y * y),
        2.890611442640554 * x * y * z,
        -0.4570457994644658 * y * (4.0 * z * z - x * x - y * y),
        0.3731763325901154 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y),
        -0.4570457994644658 * x * (4.0 * z * z - x * x - y * y),
        1.445305721320277 * z * (x * x - y * y),
        -0.5900435899266435 * x * (x * x - 3.0 * y * y),
    ]
}

/// A Radiance .hdr (RGBE, new-style run lengths).
fn read_hdr(path: &str) -> Result<(usize, usize, Vec<[f32; 3]>)> {
    let b = std::fs::read(path)?;
    let mut at = 0;
    let mut line = || {
        let s = at;
        while at < b.len() && b[at] != b'\n' {
            at += 1;
        }
        at += 1;
        String::from_utf8_lossy(&b[s..at - 1]).to_string()
    };
    loop {
        if line().is_empty() {
            break;
        }
    }
    let dims = line();
    let p: Vec<&str> = dims.split_whitespace().collect();
    if p.len() != 4 || p[0] != "-Y" || p[2] != "+X" {
        bail!("{path}: header {dims}");
    }
    let (h, w): (usize, usize) = (p[1].parse()?, p[3].parse()?);
    let mut px = vec![[0f32; 3]; w * h];
    let mut row = vec![0u8; 4 * w];
    for y in 0..h {
        if b[at] != 2 || b[at + 1] != 2 {
            bail!("{path}: only new-style RLE");
        }
        at += 4;
        for c in 0..4 {
            let mut x = 0;
            while x < w {
                let n = b[at] as usize;
                at += 1;
                if n > 128 {
                    let v = b[at];
                    at += 1;
                    for _ in 0..n - 128 {
                        row[4 * x + c] = v;
                        x += 1;
                    }
                } else {
                    for _ in 0..n {
                        row[4 * x + c] = b[at];
                        at += 1;
                        x += 1;
                    }
                }
            }
        }
        for x in 0..w {
            let e = row[4 * x + 3];
            let f = if e == 0 { 0.0 } else { (2f32).powi(e as i32 - 136) };
            px[y * w + x] = [row[4 * x] as f32 * f, row[4 * x + 1] as f32 * f, row[4 * x + 2] as f32 * f];
        }
    }
    Ok((w, h, px))
}

/// The sky's sixteen rgb harmonics, `up` 2 (+Z) or 1 (+Y).
fn sky_sh(path: &str, up: usize) -> Result<[[f64; 3]; 16]> {
    let (w, h, px) = read_hdr(path)?;
    let mut sh = [[0f64; 3]; 16];
    for y in 0..h {
        let theta = std::f64::consts::PI * (y as f64 + 0.5) / h as f64;
        let dw = (2.0 * std::f64::consts::PI / w as f64) * (std::f64::consts::PI / h as f64) * theta.sin();
        for x in 0..w {
            let phi = 2.0 * std::f64::consts::PI * (x as f64 + 0.5) / w as f64;
            let (a, b, c) = (theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos());
            let d = if up == 2 { [a, b, c] } else { [a, c, -b] };
            let yk = basis(d);
            let p = px[y * w + x];
            for k in 0..16 {
                for ch in 0..3 {
                    sh[k][ch] += p[ch] as f64 * yk[k] * dw;
                }
            }
        }
    }
    Ok(sh)
}

/// Ten zonal values an element to nine harmonics (splatTransferFrame).
fn zonal_to_sh(z: &[f32]) -> Vec<f32> {
    z.chunks_exact(10).flat_map(|e| zonal_harmonics(e).map(|v| v as f32)).collect()
}

fn rel(err: f64, re: f64) -> f64 {
    if re > 0.0 { err / re } else { 0.0 }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        bail!("athc_cpca_check ORIGINAL.athc CPCA.athc [HDR ...]");
    }
    let (fa, fb) = (read_any(&args[0])?, read_any(&args[1])?);
    // Everything but the transfer must be what it was.
    let same = fa.header.count == fb.header.count
        && fa.levels.len() == fb.levels.len()
        && fa.chunks.len() == fb.chunks.len()
        && fa.levels.iter().map(|(_, b)| b).chain(&fa.chunks).zip(fb.levels.iter().map(|(_, b)| b).chain(&fb.chunks)).all(|(x, y)| {
            x.n == y.n
                && x.positions == y.positions
                && x.shape == y.shape
                && x.tail == y.tail
                && x.sh == y.sh
                && x.normals == y.normals
                && x.emission == y.emission
                && x.pbr == y.pbr
                && x.lobes == y.lobes
                && x.shadow_bits == y.shadow_bits
                && x.curvature == y.curvature
                && x.skin == y.skin
                && x.lod_size == y.lod_size
        });
    let (count, a) = transfers(&fa);
    let (count_b, b) = transfers(&fb);
    if count != count_b || a.len() != b.len() {
        bail!("the two files' transfers differ in shape ({count} / {count_b} values, {} / {})", a.len(), b.len());
    }
    let n = a.len().checked_div(count).unwrap_or(0);
    let direct = if matches!(count, 16 | 64 | 112) { 16 } else if count == 10 { 10 } else { 9 };
    let indirect = if count >= 4 * direct { 3 * direct } else { 0 };
    let halves = [(0, direct), (direct, direct + indirect), (direct + indirect, count)];
    let mut json = vec![format!("\"original\":{:?},\"cpca\":{:?},\"elements\":{n},\"values\":{count},\"othersIdentical\":{same}", args[0], args[1])];
    // Values.
    let mut coef = Vec::new();
    let (mut te, mut tr) = (0f64, 0f64);
    for (name, (lo, hi)) in ["direct", "indirect", "field"].iter().zip(halves) {
        if hi <= lo {
            continue;
        }
        let (mut e2, mut r2) = (0f64, 0f64);
        for i in 0..n {
            for v in lo..hi {
                let (x, y) = (a[i * count + v] as f64, b[i * count + v] as f64);
                e2 += (x - y).powi(2);
                r2 += x * x;
            }
        }
        te += e2;
        tr += r2;
        coef.push(format!("\"{name}\":{:.3e}", rel(e2, r2)));
    }
    coef.push(format!("\"all\":{:.3e}", rel(te, tr)));
    json.push(format!("\"coef\":{{{}}}", coef.join(",")));
    // A zonal transfer (two lobes: octahedral axis, three band coefficients,
    // in the gaussian's own frame) as the nine harmonics splatTransferFrame
    // makes of it, in that frame; a sky then stands in that frame too.
    let (a, b, count, direct, indirect) = if count == 10 {
        (zonal_to_sh(&a), zonal_to_sh(&b), 9, 9, 0)
    } else {
        (a, b, count, direct, indirect)
    };
    // Along directions (Fibonacci sphere).
    {
        let dirs: Vec<[f64; 16]> = (0..64)
            .map(|k| {
                let z = 1.0 - (2.0 * k as f64 + 1.0) / 64.0;
                let r = (1.0 - z * z).sqrt();
                let phi = k as f64 * 2.399963229728653;
                basis([r * phi.cos(), r * phi.sin(), z])
            })
            .collect();
        let mut point = Vec::new();
        for (name, lo, comps, stride) in [("direct", 0usize, direct.min(16), 1usize), ("field", direct + indirect, 16, 3)] {
            if name == "field" && count != direct + indirect + 48 {
                continue;
            }
            let (mut e2, mut r2) = (0f64, 0f64);
            for i in 0..n {
                for y in &dirs {
                    for ch in 0..stride {
                        let (mut x0, mut x1) = (0f64, 0f64);
                        for k in 0..comps {
                            x0 += a[i * count + lo + k * stride + ch] as f64 * y[k];
                            x1 += b[i * count + lo + k * stride + ch] as f64 * y[k];
                        }
                        e2 += (x0 - x1).powi(2);
                        r2 += x0 * x0;
                    }
                }
            }
            point.push(format!("\"{name}\":{:.3e}", rel(e2, r2)));
        }
        json.push(format!("\"point\":{{{}}}", point.join(",")));
    }
    // Under skies.
    let mut skies = Vec::new();
    let hdrs: &[String] = &args[2..];
    for hdr in hdrs {
        for up in [2usize, 1] {
            let sh = sky_sh(hdr, up)?;
            let mut vals: Vec<(f64, f64)> = Vec::with_capacity(n);
            for i in 0..n {
                let (mut e2, mut r2) = (0f64, 0f64);
                for ch in 0..3 {
                    let (mut x0, mut x1) = (0f64, 0f64);
                    for k in 0..direct.min(16) {
                        let s = sh[k][ch];
                        x0 += a[i * count + k] as f64 * s;
                        x1 += b[i * count + k] as f64 * s;
                    }
                    if indirect > 0 {
                        for k in 0..direct {
                            let s = sh[k][ch];
                            x0 += a[i * count + direct + 3 * k + ch] as f64 * s;
                            x1 += b[i * count + direct + 3 * k + ch] as f64 * s;
                        }
                    }
                    e2 += (x0 - x1).powi(2);
                    r2 += x0 * x0;
                }
                vals.push((e2, r2));
            }
            let all = rel(vals.iter().map(|v| v.0).sum(), vals.iter().map(|v| v.1).sum());
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&i, &j| vals[i].1.partial_cmp(&vals[j].1).unwrap());
            let dark = &order[..n / 2];
            let dark_rel = rel(dark.iter().map(|&i| vals[i].0).sum(), dark.iter().map(|&i| vals[i].1).sum());
            let name = std::path::Path::new(hdr).file_stem().unwrap().to_string_lossy().to_string();
            skies.push(format!(
                "{{\"hdr\":{name:?},\"up\":\"{}\",\"relMSE\":{all:.3e},\"darkHalf\":{dark_rel:.3e}}}",
                if up == 2 { "+Z" } else { "+Y" }
            ));
        }
    }
    json.push(format!("\"sky\":[{}]", skies.join(",")));
    println!("{{{}}}", json.join(","));
    Ok(())
}
