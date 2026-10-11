//! A shadow catcher's .athc (athenea mesh2splat --shadow-catcher: black
//! gaussians on the ground plane, each with its open-direction cells) to a
//! small floor-shadow file for the compact pages (thread CW,
//! examples/webgpu-site/ground-shadow.js), which draw no SplatMesh: each
//! gaussian's centre and its footprint on the floor (the covariance's XY
//! block in the stage, Z up), its opacity and its cells. Gaussians whose
//! cells above the horizon are all open never darken anything under any
//! dome and are left out.
//!
//!   cargo run --release --example floor_shadow -- catcher.athc out.bin
//!
//! Format (little endian): "FSH1", u32 count, u32 words (cells / 32), u32
//! side (cells a side of the octahedral square), f32 the floor's height (the
//! gaussians' mean z), then per gaussian 6 f32
//! (x, y, sxx, sxy, syy, opacity) and `words` u32.
use spark_lib::athc::{decode_quaternion, AthcFile};

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

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(args.len() == 2, "usage: floor_shadow catcher.athc out.bin");
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
    let mut out = Vec::new();
    let mut kept = 0u32;
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
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
        let cov = |a: usize, c: usize| (0..3).map(|k| r[a][k] * r[c][k] * s[k] * s[k]).sum::<f32>();
        let p = [b.positions[i * 4], b.positions[i * 4 + 1]];
        z_sum += b.positions[i * 4 + 2] as f64;
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
        for v in [p[0], p[1], cov(0, 0), cov(0, 1), cov(1, 1), opacity] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for c in cells {
            out.extend_from_slice(&c.to_le_bytes());
        }
        kept += 1;
    }
    let mut head = Vec::new();
    head.extend_from_slice(b"FSH1");
    for v in [kept, words as u32, side as u32] {
        head.extend_from_slice(&v.to_le_bytes());
    }
    let z0 = (z_sum / f64::from(kept.max(1))) as f32;
    head.extend_from_slice(&z0.to_le_bytes());
    head.extend_from_slice(&out);
    std::fs::write(&args[1], &head)?;
    eprintln!(
        "{} of {} gaussians kept (some cell above the floor closed), {} words a splat, side {}; x {:.2}..{:.2}, y {:.2}..{:.2}, z {:.4}; {} bytes",
        kept, b.n, words, side, lo[0], hi[0], lo[1], hi[1], z0, head.len()
    );
    Ok(())
}
