//! A shadow catcher's .athc (athenea mesh2splat --shadow-catcher: black
//! gaussians on the ground plane, each with its open-direction cells) to a
//! small floor-shadow file for the compact pages (thread CW,
//! examples/webgpu-site/ground-shadow.js), which draw no SplatMesh: each
//! gaussian's centre and its footprint on the floor (the covariance's XY
//! block in the stage, Z up), its opacity and its cells. Gaussians whose
//! cells above the horizon are all open never darken anything under any
//! dome and are left out.
//!
//!   cargo run --release --example floor_shadow -- catcher.athc out.bin [--merge N]
//!
//! --merge N: the gaussians merged N x N on the floor (a grid of N times
//! their spacing): moment-matched (centre and covariance weighted by
//! opacity x area), the opacity keeping the group's mass (sum o_i A_i / A;
//! past 1 the layers it stacks), each cell open where most of the group's
//! weight has it open. Smooth where usd-athc --thin (one in N^2 kept, widened) leaves a
//! grain.
//!
//! Format (little endian): "FSH1", u32 count, u32 words (cells / 32), u32
//! side (cells a side of the octahedral square), f32 the floor's height (the
//! gaussians' mean z), then per gaussian 6 f32
//! (x, y, sxx, sxy, syy, opacity) and `words` u32.
use spark_lib::athc::{decode_quaternion, AthcFile};
use std::collections::HashMap;

fn f16_of(bits: u32) -> f32 {
    half::f16::from_bits(bits as u16).to_f32()
}

fn oct_decode(u: f32, v: f32) -> [f32; 3] {
    let (fx, fy) = (u * 2.0 - 1.0, v * 2.0 - 1.0);
    let mut n = [fx, fy, 1.0 - fx.abs() - fy.abs()];
    let t = (-n[2]).clamp(0.0, 1.0);
    n[0] += if n[0] >= 0.0 { -t } else { t };
    n[1] += if n[1] >= 0.0 { -t } else { t };
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    n.map(|c| c / l)
}

/// One floor gaussian: centre, covariance (xx, xy, yy), opacity, cells.
struct Rec {
    p: [f64; 2],
    c: [f64; 3],
    o: f64,
    cells: Vec<u32>,
}

fn area(c: &[f64; 3]) -> f64 {
    (c[0] * c[2] - c[1] * c[1]).max(0.0).sqrt()
}

/// The records merged on a grid of `cell` metres (see --merge).
fn merge(recs: Vec<Rec>, cell: f64, words: usize) -> Vec<Rec> {
    let mut groups: HashMap<(i64, i64), Vec<Rec>> = HashMap::new();
    for r in recs {
        let key = ((r.p[0] / cell).floor() as i64, (r.p[1] / cell).floor() as i64);
        groups.entry(key).or_default().push(r);
    }
    let mut keys: Vec<_> = groups.keys().copied().collect();
    keys.sort();
    keys.into_iter()
        .map(|k| {
            let g = &groups[&k];
            let w: Vec<f64> = g.iter().map(|r| r.o * area(&r.c)).collect();
            let sw = w.iter().sum::<f64>().max(1e-30);
            let mut p = [0.0; 2];
            for (r, wi) in g.iter().zip(&w) {
                p[0] += wi * r.p[0] / sw;
                p[1] += wi * r.p[1] / sw;
            }
            let mut c = [0.0; 3];
            for (r, wi) in g.iter().zip(&w) {
                let d = [r.p[0] - p[0], r.p[1] - p[1]];
                c[0] += wi * (r.c[0] + d[0] * d[0]) / sw;
                c[1] += wi * (r.c[1] + d[0] * d[1]) / sw;
                c[2] += wi * (r.c[2] + d[1] * d[1]) / sw;
            }
            // Past 1: the layers the group stacks (ground-shadow.js draws
            // 1 - (1 - took)^o).
            let o = sw / area(&c).max(1e-30);
            let mut cells = vec![0u32; words];
            for bit in 0..words * 32 {
                let open: f64 = g
                    .iter()
                    .zip(&w)
                    .filter(|(r, _)| (r.cells[bit >> 5] >> (bit & 31)) & 1 == 1)
                    .map(|(_, wi)| wi)
                    .sum();
                if open * 2.0 >= sw {
                    cells[bit >> 5] |= 1 << (bit & 31);
                }
            }
            Rec { p, c, o, cells }
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        args.len() == 2 || (args.len() == 4 && args[2] == "--merge"),
        "usage: floor_shadow catcher.athc out.bin [--merge N]"
    );
    let merge_n: f64 = if args.len() == 4 { args[3].parse()? } else { 1.0 };
    let bytes = std::fs::read(&args[0])?;
    let file = spark_lib::athc_v3::read_v3(&bytes).or_else(|_| AthcFile::read(&bytes))?;
    let b = file.splats();
    let words = b.shadow_bits.len() / b.n.max(1);
    anyhow::ensure!(words >= 8, "no open-direction cells ({words} words a splat)");
    let side = ((words * 32) as f32).sqrt().round() as usize;
    // The cells above the floor (a catcher's normal is +Z in the stage).
    let above: Vec<bool> = (0..side * side)
        .map(|c| oct_decode(((c % side) as f32 + 0.5) / side as f32, ((c / side) as f32 + 0.5) / side as f32)[2] > 0.0)
        .collect();
    let mut recs = Vec::new();
    let mut z_sum = 0f64;
    for i in 0..b.n {
        let cells = &b.shadow_bits[i * words..i * words + words];
        let closed = (0..side * side).any(|c| above[c] && (cells[c >> 5] >> (c & 31)) & 1 == 0);
        let opacity = b.positions[i * 4 + 3];
        if !closed || opacity.is_nan() || opacity <= 0.0 {
            continue;
        }
        let sh = &b.shape[i * 4..i * 4 + 4];
        let s = [f16_of(sh[1] & 0xffff).exp(), f16_of(sh[1] >> 16).exp(), f16_of(sh[2] & 0xffff).exp()];
        let [x, y, z, w] = decode_quaternion(sh[0]);
        // Rotation matrix rows (x, y only: the floor's block).
        let r = [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        ];
        let cov = |a: usize, c: usize| (0..3).map(|k| r[a][k] * r[c][k] * s[k] * s[k]).sum::<f32>() as f64;
        z_sum += b.positions[i * 4 + 2] as f64;
        recs.push(Rec {
            p: [b.positions[i * 4] as f64, b.positions[i * 4 + 1] as f64],
            c: [cov(0, 0), cov(0, 1), cov(1, 1)],
            o: opacity as f64,
            cells: cells.to_vec(),
        });
    }
    let z0 = (z_sum / (recs.len().max(1) as f64)) as f32;
    let baked = recs.len();
    if merge_n > 1.0 {
        // The gaussians' spacing: the box's area over their count.
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for r in &recs {
            for k in 0..2 {
                lo[k] = lo[k].min(r.p[k]);
                hi[k] = hi[k].max(r.p[k]);
            }
        }
        let spacing = ((hi[0] - lo[0]) * (hi[1] - lo[1]) / recs.len().max(1) as f64).sqrt();
        recs = merge(recs, spacing * merge_n, words);
    }
    let mut head = Vec::new();
    head.extend_from_slice(b"FSH1");
    for v in [recs.len() as u32, words as u32, side as u32] {
        head.extend_from_slice(&v.to_le_bytes());
    }
    head.extend_from_slice(&z0.to_le_bytes());
    for r in &recs {
        for v in [r.p[0], r.p[1], r.c[0], r.c[1], r.c[2], r.o] {
            head.extend_from_slice(&(v as f32).to_le_bytes());
        }
        for c in &r.cells {
            head.extend_from_slice(&c.to_le_bytes());
        }
    }
    std::fs::write(&args[1], &head)?;
    eprintln!(
        "{} gaussians written ({} of {} kept: some cell above the floor closed), {} words a splat, side {}, z {:.4}; {} bytes",
        recs.len(),
        baked,
        b.n,
        words,
        side,
        z0,
        head.len()
    );
    Ok(())
}
