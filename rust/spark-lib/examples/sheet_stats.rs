//! Sheet (thin glass) statistics of .athc files: per file, the sheet
//! splats' count, total two-axis area (sum a_i), reflection mass
//! sum (o_i - 1/255) a_i, and the overlap a sheet drawn at athenea's 0.1 has.
use spark_lib::athc::{is_sheet, two_axis_area, SHEET_PAD};
fn main() -> anyhow::Result<()> {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path)?;
        let f = spark_lib::athc_v3::read_v3(&bytes).or_else(|_| spark_lib::athc::AthcFile::read(&bytes))?;
        let (mut n, mut area, mut mass, mut all, mut o_sum) = (0usize, 0f64, 0f64, 0usize, 0f64);
        let mut a3 = 0f64;
        let (mut gn, mut ga, mut gm, mut gt, mut go, mut gover) = (0usize, 0f64, 0f64, 0f64, 0f64, 0usize);
        for c in &f.chunks {
            for i in 0..c.n {
                all += 1;
                if !is_sheet(c, i) {
                    let per = c.pbr.len() / c.n.max(1);
                    let wd = if per > 0 { c.pbr[i * per] } else { 0 };
                    let tr = ((wd >> 16) & 255) as f64 / 255.0;
                    let a = two_axis_area(&c.shape[i * 4..i * 4 + 4]) as f64;
                    let o = c.positions[i * 4 + 3] as f64;
                    gn += 1; ga += a; gm += o.min(1.0) * a; gt += tr; go += o; if o > 1.0 { gover += 1; }
                    continue;
                }
                n += 1;
                let a = two_axis_area(&c.shape[i * 4..i * 4 + 4]) as f64;
                let o = c.positions[i * 4 + 3] as f64;
                area += a;
                o_sum += o;
                mass += (o - SHEET_PAD as f64).max(0.0) * a;
                // 0.1-alpha "blocking" integral: 2 pi sigma^2 x alpha_draw
                let own = (o.min(1.0) - SHEET_PAD as f64).max(0.0);
                a3 += 2.0 * std::f64::consts::PI * a * own.max(0.1);
            }
        }
        println!("{path}\n  splats {all} sheets {n} sum_area {area:.4e} refl_mass {mass:.4e} mean_o {:.5} block {a3:.4e}", o_sum / n.max(1) as f64);
        println!("  other {gn} sum_area {ga:.4e} mass(min o,1) {gm:.4e} mean_o {:.4} mean_transm {:.3} o>1 {gover}", go / gn.max(1) as f64, gt / gn.max(1) as f64);
    }
    Ok(())
}
