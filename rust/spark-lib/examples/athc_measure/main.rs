//! athc-measure: where the published .athc clouds lose quality or waste
//! splats, measured on the CPU (thread AY, research/simplify-measurements.md).
//!
//! Everything here is a proxy: splats are drawn with their base colour
//! (no relighting), the way Spark's draw composites them (EWA, 0.3 px blur,
//! LoD opacity above 1 as 1 - (1 - g)^o), front to back per pixel, in float.
//! Images are compared in linear float (relMSE), never 8-bit.
//!
//! cargo run --release -p spark-lib --example athc_measure -- <command> ...

mod cloud;
mod merge;
mod raster;

use std::time::Instant;

use anyhow::{bail, Result};
use cloud::*;
use raster::*;

fn usage() -> ! {
    eprintln!(
        "athc_measure <stats|visibility|levels|merge|surfel> --asset corvette-hd|corvette-light|corvette-lights|sparrow|sparrow-mobile|pawn [options]"
    );
    std::process::exit(2)
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let cmd = args[0].as_str();
    let asset = arg(&args, "--asset").unwrap_or_else(|| usage());
    let width: usize = arg(&args, "--width").map(|s| s.parse().unwrap()).unwrap_or(1920);
    let t0 = Instant::now();
    let scene = Scene::load(asset)?;
    eprintln!("{}: {} splats in {} parts, loaded in {:.1}s", asset, scene.splats.len(), scene.parts.len(), t0.elapsed().as_secs_f32());
    match cmd {
        "stats" => stats(&scene, width),
        "visibility" => visibility(&scene, &args),
        "levels" => levels(&scene, &args),
        "surfel" => surfel(&scene, width),
        "merge" => merge::run(&scene, &args),
        "bytes" => bytes(&scene),
        "overlap" => overlap(&scene),
        "thin" => thin(&scene, &args),
        "render" => {
            let cam = scene.camera(width, 1.0, 0.0, 0.0);
            let img = render(&scene.splats, &cam, None);
            let out = arg(&args, "--out").unwrap_or("render.png");
            img.save_png(out, 0.5)?;
            Ok(())
        }
        _ => bail!("unknown command {cmd}"),
    }
}

fn pct(n: usize, of: usize) -> String {
    format!("{:.2}%", 100.0 * n as f64 / of.max(1) as f64)
}

/// Opacity, flatness and on-screen size histograms, near-coincident pairs.
fn stats(scene: &Scene, width: usize) -> Result<()> {
    let cam = scene.camera(width, 1.0, 0.0, 0.0);
    println!("## {} — {} splats\n", scene.name, scene.splats.len());
    println!("default camera {}×{} at {:.3} m from the centre\n", cam.w, cam.h, cam.distance_to(scene.center));
    for (pi, part) in scene.parts.iter().enumerate() {
        let ids: Vec<usize> = (0..scene.splats.len()).filter(|&i| scene.splats[i].part == pi as u16).collect();
        println!("### part {} ({} splats, file {})\n", part, ids.len(), scene.files[pi]);
        // Opacity
        let edges = [1.0 / 255.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.9, 0.99, 1.0001, f32::INFINITY];
        let names = ["<1/255", "<0.01", "<0.05", "<0.1", "<0.25", "<0.5", "<0.9", "<0.99", "≈1", ">1 (LoD)"];
        let mut h = [0usize; 10];
        for &i in &ids {
            let o = scene.splats[i].o;
            h[edges.iter().position(|&e| o < e).unwrap_or(9)] += 1;
        }
        println!("| opacity | {} |\n|---|{}", names.join(" | "), "---|".repeat(10));
        println!("| splats | {} |\n", h.iter().map(|&n| pct(n, ids.len())).collect::<Vec<_>>().join(" | "));
        // Flatness: smallest / middle scale.
        let edges = [1e-6f32, 0.01, 0.05, 0.1, 0.2, 0.5, f32::INFINITY];
        let names = ["0 (surfel)", "<0.01", "<0.05", "<0.1", "<0.2", "<0.5", "≥0.5"];
        let mut h = [0usize; 7];
        for &i in &ids {
            let mut s = scene.splats[i].s;
            s.sort_by(|a, b| a.total_cmp(b));
            let r = s[0] / s[1].max(1e-30);
            h[edges.iter().position(|&e| r < e).unwrap_or(6)] += 1;
        }
        println!("| min/mid scale | {} |\n|---|{}", names.join(" | "), "---|".repeat(7));
        println!("| splats | {} |\n", h.iter().map(|&n| pct(n, ids.len())).collect::<Vec<_>>().join(" | "));
        // Anisotropy: mid / max
        let mut h2 = [0usize; 4];
        for &i in &ids {
            let mut s = scene.splats[i].s;
            s.sort_by(|a, b| a.total_cmp(b));
            let r = s[1] / s[2].max(1e-30);
            h2[[0.25f32, 0.5, 0.8, f32::INFINITY].iter().position(|&e| r < e).unwrap()] += 1;
        }
        println!(
            "mid/max scale: <0.25 {}, <0.5 {}, <0.8 {}, ≥0.8 {}\n",
            pct(h2[0], ids.len()),
            pct(h2[1], ids.len()),
            pct(h2[2], ids.len()),
            pct(h2[3], ids.len())
        );
        // On-screen size (largest sigma in px) at the default camera.
        let edges = [0.25f32, 0.5, 1.0, 2.0, 4.0, f32::INFINITY];
        let names = ["<0.25 px", "<0.5", "<1", "<2", "<4", "≥4"];
        let mut h = [0usize; 6];
        let mut behind = 0;
        for &i in &ids {
            let sp = &scene.splats[i];
            match cam.sigma_px(sp) {
                Some(px) => h[edges.iter().position(|&e| px < e).unwrap()] += 1,
                None => behind += 1,
            }
        }
        println!("| largest σ on screen | {} |\n|---|{}", names.join(" | "), "---|".repeat(6));
        println!("| splats | {} |\n", h.iter().map(|&n| pct(n, ids.len())).collect::<Vec<_>>().join(" | "));
        if behind > 0 {
            println!("({} behind the camera)\n", behind);
        }
        // Median sizes
        let mut longs: Vec<f32> = ids.iter().map(|&i| scene.splats[i].s.iter().cloned().fold(0.0, f32::max)).collect();
        longs.sort_by(|a, b| a.total_cmp(b));
        if !longs.is_empty() {
            println!(
                "largest σ (mm): p10 {:.3}, median {:.3}, p90 {:.3}\n",
                1e3 * longs[longs.len() / 10] * scene.unit,
                1e3 * longs[longs.len() / 2] * scene.unit,
                1e3 * longs[longs.len() * 9 / 10] * scene.unit
            );
        }
        // Near-coincident pairs
        let (dups, close) = near_coincident(&scene.splats, &ids);
        println!(
            "near-coincident (centres < 0.1 σ apart, same-size within 1.5×, normals within 25°): {} splats ({}); within 0.5 σ: {}\n",
            dups,
            pct(dups, ids.len()),
            pct(close, ids.len())
        );
    }
    Ok(())
}

/// Splats with a neighbour whose centre is within 0.1 (resp. 0.5) of the
/// smaller long sigma, of similar size and facing.
fn near_coincident(splats: &[Splat], ids: &[usize]) -> (usize, usize) {
    if ids.is_empty() {
        return (0, 0);
    }
    let mut longs: Vec<f32> = ids.iter().map(|&i| splats[i].long()).collect();
    longs.sort_by(|a, b| a.total_cmp(b));
    let cell = longs[longs.len() / 2].max(1e-9);
    let key = |p: [f32; 3]| ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64, (p[2] / cell).floor() as i64);
    let mut grid: std::collections::HashMap<(i64, i64, i64), Vec<u32>> = std::collections::HashMap::new();
    for &i in ids {
        grid.entry(key(splats[i].p)).or_default().push(i as u32);
    }
    let mut dup = 0;
    let mut close = 0;
    for &i in ids {
        let a = &splats[i];
        let (x, y, z) = key(a.p);
        let mut best = f32::INFINITY;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = grid.get(&(x + dx, y + dy, z + dz)) {
                        for &j in v {
                            let j = j as usize;
                            if j == i {
                                continue;
                            }
                            let b = &splats[j];
                            let (la, lb) = (a.long(), b.long());
                            if la.max(lb) > 1.5 * la.min(lb) {
                                continue;
                            }
                            if a.has_normal() && b.has_normal() && dot(a.n, b.n) < 0.906 {
                                continue;
                            }
                            let d = dist(a.p, b.p) / la.min(lb);
                            best = best.min(d);
                        }
                    }
                }
            }
        }
        if best < 0.1 {
            dup += 1;
        }
        if best < 0.5 {
            close += 1;
        }
    }
    (dup, close)
}

/// Splats never visible from an orbit of views, and the error of dropping them.
fn visibility(scene: &Scene, args: &[String]) -> Result<()> {
    let width: usize = arg(args, "--vis-width").map(|s| s.parse().unwrap()).unwrap_or(960);
    let n = scene.splats.len();
    let nviews: usize = arg(args, "--views").map(|s| s.parse().unwrap()).unwrap_or(160);
    let (max_t, contrib, nv) = visibility_of(scene, &scene.splats, nviews, width);
    let views = vec![(); nv];
    println!("## {} — visibility over {} views at {} px wide\n", scene.name, views.len(), width);
    println!("| part | splats | centre always ≥98% covered | ≥99.9% | contribution < 1/255 px (all views) | < 0.05 px | both (hidden and < 0.05 px) |");
    println!("|---|---|---|---|---|---|---|");
    let mut drop = vec![false; n];
    for (pi, part) in scene.parts.iter().enumerate() {
        let ids: Vec<usize> = (0..n).filter(|&i| scene.splats[i].part == pi as u16).collect();
        let hidden = ids.iter().filter(|&&i| max_t[i] < 0.02).count();
        let hidden2 = ids.iter().filter(|&&i| max_t[i] < 0.001).count();
        let tiny = ids.iter().filter(|&&i| contrib[i] < 1.0 / 255.0).count();
        let small = ids.iter().filter(|&&i| contrib[i] < 0.05).count();
        let both = ids.iter().filter(|&&i| max_t[i] < 0.02 && contrib[i] < 0.05).count();
        for &i in &ids {
            if contrib[i] < 0.05 && max_t[i] < 0.02 {
                drop[i] = true;
            }
        }
        println!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            part,
            ids.len(),
            pct(hidden, ids.len()),
            pct(hidden2, ids.len()),
            pct(tiny, ids.len()),
            pct(small, ids.len()),
            pct(both, ids.len())
        );
    }
    // Pruning test on held-out views at full resolution.
    let kept: Vec<Splat> = scene.splats.iter().zip(&drop).filter(|(_, d)| !**d).map(|(s, _)| s.clone()).collect();
    let tiny_only: Vec<Splat> = scene.splats.iter().zip(&contrib).filter(|(_, c)| **c >= 0.05).map(|(s, _)| s.clone()).collect();
    println!(
        "\nprune test (held-out views, 1920 px): drop hidden ∧ < 0.05 px = {} splats ({}); drop all < 0.05 px = {} ({})\n",
        n - kept.len(),
        pct(n - kept.len(), n),
        n - tiny_only.len(),
        pct(n - tiny_only.len(), n)
    );
    println!("| view | relMSE hidden∧small dropped | relMSE all small dropped | Δcoverage (mean |ΔT|) |\n|---|---|---|---|");
    for (name, cam) in scene.test_views(1920) {
        let full = render(&scene.splats, &cam, None);
        let a = render(&kept, &cam, None);
        let b = render(&tiny_only, &cam, None);
        if let Some(dir) = arg(args, "--save") {
            full.save_png(&format!("{dir}/{}-{}-full.png", scene.name, name.split(' ').next().unwrap()), 0.6)?;
            a.save_png(&format!("{dir}/{}-{}-pruned.png", scene.name, name.split(' ').next().unwrap()), 0.6)?;
            a.diff_png(&full, &format!("{dir}/{}-{}-diff.png", scene.name, name.split(' ').next().unwrap()))?;
        }
        println!("| {} | {:.2e} | {:.2e} | {:.2e} / {:.2e} |", name, a.rel_mse(&full), b.rel_mse(&full), a.alpha_diff(&full), b.alpha_diff(&full));
    }
    Ok(())
}

/// The error of turning every splat into an exact surfel (thinnest axis 0).
fn surfel(scene: &Scene, width: usize) -> Result<()> {
    let flat: Vec<Splat> = scene
        .splats
        .iter()
        .map(|s| {
            let mut s = s.clone();
            let k = (0..3).min_by(|&a, &b| s.s[a].total_cmp(&s.s[b])).unwrap();
            s.s[k] = 0.0;
            s
        })
        .collect();
    let flat_some: Vec<Splat> = scene
        .splats
        .iter()
        .map(|s| {
            let mut s = s.clone();
            let mut o = s.s;
            o.sort_by(|a, b| a.total_cmp(b));
            if o[0] / o[1].max(1e-30) < 0.2 {
                let k = (0..3).min_by(|&a, &b| s.s[a].total_cmp(&s.s[b])).unwrap();
                s.s[k] = 0.0;
            }
            s
        })
        .collect();
    println!("## {} — surfel conversion (thin axis set to 0)\n", scene.name);
    println!("| view | relMSE all splats → surfels | relMSE only min/mid < 0.2 | Δcoverage all |\n|---|---|---|---|");
    let _ = width;
    for (name, cam) in scene.test_views(1920) {
        let full = render(&scene.splats, &cam, None);
        let a = render(&flat, &cam, None);
        let b = render(&flat_some, &cam, None);
        println!("| {} | {:.2e} | {:.2e} | {:.2e} |", name, a.rel_mse(&full), b.rel_mse(&full), a.alpha_diff(&full));
    }
    Ok(())
}

/// The LoD tree, level by level.
fn levels(scene: &Scene, _args: &[String]) -> Result<()> {
    println!("## {} — LoD levels\n", scene.name);
    for (pi, lv) in scene.level_data.iter().enumerate() {
        let Some(lv) = lv else { continue };
        println!("### part {}\n", scene.parts[pi]);
        println!("| level | cell (mm) | groups | splats/group | coverage W/A p50 / p90 | colour σ (w-mean) | colour σ p90 | normal spread 1−|n̄| (w-mean) | groups with members > 30° off (w) | > 60° (w) | mixing materials (w) |");
        println!("|---|---|---|---|---|---|---|---|---|---|---|");
        for l in &lv.rows {
            println!(
                "| {} | {:.2} | {} | {:.1} | {:.2} / {:.2} | {:.4} | {:.4} | {:.4} | {:.1}% | {:.1}% | {:.1}% |",
                l.level,
                l.cell * 1e3 * scene.unit,
                l.groups,
                l.splats_per_group,
                l.cov_p50,
                l.cov_p90,
                l.colour_sd,
                l.colour_sd_p90,
                l.normal_spread,
                100.0 * l.crease30,
                100.0 * l.crease60,
                100.0 * l.mixed,
            );
        }
        println!();
    }
    // Image space: each level as the whole cloud, from the distance where its
    // cell is one pixel at 1920 px, against the splats.
    println!("### image error of a whole level vs the splats, default camera at the width (px) where the level's cell is ~1 px, against the splats supersampled 4×\n");
    println!("| level | splats | width (px) | level: relMSE | level: Δcoverage | all splats at that width: relMSE | Δcoverage |\n|---|---|---|---|---|---|---|");
    let finest = scene.cut_levels.len();
    for k in 0..finest {
        let (level, cell, cut) = &scene.cut_levels[k];
        if cut.len() < 1000 {
            continue;
        }
        // The default view at the resolution where the level's cell is one pixel.
        let cam0 = scene.camera(1920, 1.0, 0.0, 0.0);
        let px = cell * cam0.fx / cam0.distance_to(scene.center) / 1920.0;
        let w = ((1.0 / px / 16.0).round() as usize * 16).max(16);
        if !(48..=3840).contains(&w) {
            continue;
        }
        let cam = scene.camera(w, 1.0, 0.0, 0.0);
        let d = cam.distance_to(scene.center);
        // Reference: the splats supersampled 4x (the splats at w alone lose
        // whatever falls under 1/255 a pixel).
        let reference = render(&scene.splats, &scene.camera(4 * w, 1.0, 0.0, 0.0), None).downsample(4);
        let plain = render(&scene.splats, &cam, None);
        let full = reference;
        let img = render(cut, &cam, None);
        if let Some(dir) = arg(_args, "--save") {
            img.save_png(&format!("{dir}/{}-level{}-cut.png", scene.name, level), 0.6)?;
            full.save_png(&format!("{dir}/{}-level{}-full.png", scene.name, level), 0.6)?;
        }
        let _ = d;
        println!(
            "| {} | {} | {} | {:.2e} | {:.2e} | {:.2e} | {:.2e} |",
            level,
            cut.len(),
            w,
            img.rel_mse(&full),
            img.alpha_diff(&full),
            plain.rel_mse(&full),
            plain.alpha_diff(&full)
        );
    }
    Ok(())
}

/// Each splat's largest transmittance in front of its centre and its summed
/// contribution (px) over an orbit of views; and the number of views.
pub fn visibility_of(scene: &Scene, splats: &[Splat], nviews: usize, width: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let below = scene.views_below;
    let n = splats.len();
    let max_t = AtomicF32s::new(n);
    let contrib = AtomicF32s::new(n);
    let mut views = vec![scene.camera(width, 1.0, 0.0, 0.0)];
    // Fibonacci directions over the sphere (the upper half for a car on the
    // ground), at the default distance, and a quarter as many at half of it.
    for (count, scale) in [(nviews, 1.0f32), (nviews / 4, 0.5)] {
        let total = if below { count } else { 2 * count };
        for k in 0..total {
            let zf = 1.0 - 2.0 * (k as f32 + 0.5) / total as f32;
            let el = zf.asin().to_degrees();
            if !below && el < 2.0 {
                continue;
            }
            let az = (k as f32 * 137.507_76) % 360.0 + 0.01;
            views.push(scene.camera(width, scale, az, el));
        }
    }
    let t0 = Instant::now();
    for cam in &views {
        let track = Track { max_t: &max_t, contrib: &contrib };
        render(splats, cam, Some(&track));
    }
    eprintln!("{} views in {:.1}s", views.len(), t0.elapsed().as_secs_f32());
    let max_t = max_t.into_vec();
    let contrib = contrib.into_vec();
    (max_t, contrib, views.len())
}

/// Stored bytes a splat by section, levels and splats apart.
fn bytes(scene: &Scene) -> Result<()> {
    println!("## {} — stored bytes per section (v3)\n", scene.name);
    for (pi, f) in scene.files.iter().enumerate() {
        let path = format!("{}/{}", scene.dir, f);
        let b = std::fs::read(&path)?;
        let Ok(l) = spark_lib::athc_v3::parse_v3(&b) else { continue };
        let total = b.len();
        let splats: u64 = l.blocks.iter().filter(|x| x.kind == 1).map(|x| x.n as u64).sum();
        let groups: u64 = l.blocks.iter().filter(|x| x.kind == 0).map(|x| x.n as u64).sum();
        println!("### {} — {} splats, {} level groups, {:.1} MB, {:.1} B/splat\n", scene.parts[pi], splats, groups, total as f64 / 1e6, total as f64 / splats as f64);
        println!("| section | splats (MB) | levels (MB) | B/splat |\n|---|---|---|---|");
        for (k, sec) in l.sections.iter().enumerate() {
            let (mut sp, mut lv) = (0u64, 0u64);
            for blk in &l.blocks {
                let st = blk.spans[k].stored as u64;
                if blk.kind == 1 { sp += st } else { lv += st }
            }
            println!("| {} | {:.2} | {:.2} | {:.1} |", sec.id.name(), sp as f64 / 1e6, lv as f64 / 1e6, sp as f64 / splats as f64);
        }
        println!();
    }
    Ok(())
}

/// How deep each surface is covered: at each splat's centre, the summed
/// alpha of the other splats of its part facing the same way (within 25°),
/// as their gaussians fall there in their own plane.
fn overlap(scene: &Scene) -> Result<()> {
    println!("## {} — coverage depth (Σ alpha of the others at each centre)\n", scene.name);
    println!("| part | p10 | median | p90 | mean | splats with depth > 2 |\n|---|---|---|---|---|---|");
    for (pi, part) in scene.parts.iter().enumerate() {
        let ids: Vec<usize> = (0..scene.splats.len()).filter(|&i| scene.splats[i].part == pi as u16).collect();
        if ids.is_empty() {
            continue;
        }
        let sp = &scene.splats;
        let mut longs: Vec<f32> = ids.iter().map(|&i| sp[i].long()).collect();
        longs.sort_by(|a, b| a.total_cmp(b));
        let cell = 3.0 * longs[longs.len() * 9 / 10].max(1e-9);
        let key = |p: [f32; 3]| ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64, (p[2] / cell).floor() as i64);
        let mut grid: std::collections::HashMap<(i64, i64, i64), Vec<u32>> = std::collections::HashMap::new();
        for &i in &ids {
            grid.entry(key(sp[i].p)).or_default().push(i as u32);
        }
        let depth: Vec<f32> = raster::par_chunks(ids.len(), |r| {
            r.map(|k| {
                let i = ids[k];
                let a = &sp[i];
                let (x, y, z) = key(a.p);
                let mut sum = 0.0f32;
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            let Some(v) = grid.get(&(x + dx, y + dy, z + dz)) else { continue };
                            for &j in v {
                                let j = j as usize;
                                if j == i {
                                    continue;
                                }
                                let b = &sp[j];
                                if a.has_normal() && b.has_normal() && dot(a.n, b.n) < 0.906 {
                                    continue;
                                }
                                // b's gaussian at a's centre, in b's frame
                                let d = sub(a.p, b.p);
                                let ax = b.axes();
                                let mut q = 0.0;
                                for t in 0..3 {
                                    let u = dot(d, ax[t]) / b.s[t].max(1e-4 * b.long());
                                    q += u * u;
                                }
                                if q < 9.0 {
                                    sum += b.o.min(1.0) * (-0.5 * q).exp();
                                }
                            }
                        }
                    }
                }
                sum
            })
            .collect::<Vec<f32>>()
        })
        .concat();
        let mut d = depth.clone();
        d.sort_by(|a, b| a.total_cmp(b));
        let mean = d.iter().map(|v| *v as f64).sum::<f64>() / d.len() as f64;
        let over2 = d.iter().filter(|v| **v > 2.0).count();
        println!("| {} | {:.2} | {:.2} | {:.2} | {:.2} | {} |", part, d[d.len() / 10], d[d.len() / 2], d[d.len() * 9 / 10], mean, pct(over2, d.len()));
    }
    println!();
    Ok(())
}

/// Opaque splats thinned 1 in k along the Morton order (per part), their
/// two long axes scaled by `grow`: the error of emitting fewer, larger discs.
fn thin(scene: &Scene, args: &[String]) -> Result<()> {
    let width: usize = arg(args, "--width").map(|s| s.parse().unwrap()).unwrap_or(1920);
    let views2 = scene.test_views(2 * width);
    let views = scene.test_views(width);
    let refs: Vec<Image> = views2.iter().map(|(_, c)| render(&scene.splats, c, None).downsample(2)).collect();
    let n = scene.splats.len();
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for s in &scene.splats {
        for d in 0..3 {
            lo[d] = lo[d].min(s.p[d]);
            hi[d] = hi[d].max(s.p[d]);
        }
    }
    let ext = (0..3).map(|d| hi[d] - lo[d]).fold(0.0, f32::max) * 1.01;
    let morton = |p: [f32; 3]| -> u64 {
        let q = [0, 1, 2].map(|d| (((p[d] - lo[d]) / ext).clamp(0.0, 0.999_999) * 2_097_152.0) as u64);
        let mut c = 0u64;
        for b in 0..21 {
            for (k, v) in q.iter().enumerate() {
                c |= ((v >> b) & 1) << (3 * b + k);
            }
        }
        c
    };
    let only: Option<Vec<String>> = arg(args, "--only").map(|s| s.split(',').map(|x| x.to_string()).collect());
    let mut order: Vec<(u16, u64, usize)> = scene.splats.iter().enumerate().map(|(i, s)| (s.part, morton(s.p), i)).collect();
    order.sort_unstable();
    println!("## {} — opaque splats thinned 1 in k (Morton order), long axes ×grow\n", scene.name);
    let names: Vec<String> = views.iter().map(|(n, _)| n.clone()).collect();
    println!("| k | grow | splats | % | {} | mean | mean Δcoverage |\n|---|---|---|---|{}---|---|", names.join(" | "), "---|".repeat(names.len()));
    for (k, grow) in [(1usize, 1.0f32), (2, 1.0), (2, 1.2), (2, 1.41), (3, 1.0), (3, 1.3), (3, 1.73), (4, 1.5), (4, 1.7), (4, 2.0), (6, 1.7), (6, 2.0), (8, 2.0), (8, 2.4)] {
        let mut out = Vec::with_capacity(n);
        let mut run = 0usize;
        let mut last_part = u16::MAX;
        for &(part, _, i) in &order {
            if part != last_part {
                run = 0;
                last_part = part;
            }
            let s = &scene.splats[i];
            if s.o < 0.99 || only.as_ref().is_some_and(|o| !o.contains(&scene.parts[s.part as usize])) {
                out.push(s.clone());
                continue;
            }
            if run % k == 0 {
                let mut t = s.clone();
                let thinnest = (0..3).min_by(|&a, &b| t.s[a].total_cmp(&t.s[b])).unwrap();
                for d in 0..3 {
                    if d != thinnest {
                        t.s[d] *= grow;
                    }
                }
                out.push(t);
            }
            run += 1;
        }
        let imgs: Vec<Image> = views.iter().map(|(_, c)| render(&out, c, None)).collect();
        let errs: Vec<f64> = imgs.iter().zip(&refs).map(|(a, r)| a.rel_mse(r)).collect();
        let dcov = imgs.iter().zip(&refs).map(|(a, r)| a.alpha_diff(r)).sum::<f64>() / refs.len() as f64;
        println!(
            "| {} | {} | {} | {:.1}% | {} | {:.2e} | {:.2e} |",
            k,
            grow,
            out.len(),
            100.0 * out.len() as f64 / n as f64,
            errs.iter().map(|e| format!("{e:.2e}")).collect::<Vec<_>>().join(" | "),
            errs.iter().sum::<f64>() / errs.len() as f64,
            dcov
        );
    }
    Ok(())
}
