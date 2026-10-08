//! Candidate cuts against a reference cloud (thread BC): the same held-out
//! views as `merge`, the reference supersampled 2x, albedo and normal
//! proxies, relMSE in float.
//!
//!   athc_measure cmp --asset corvette-hd --cand v5=corvette@DIR,err=corvette@DIR2 [--save DIR]

use anyhow::Result;

use crate::arg;
use crate::cloud::*;
use crate::raster::*;

pub fn run(scene: &Scene, args: &[String]) -> Result<()> {
    let width: usize = arg(args, "--width").map(|s| s.parse().unwrap()).unwrap_or(1920);
    let views = scene.test_views(width);
    let cam0 = scene.camera(width, 1.0, 0.0, 0.0);
    let d = cam0.distance_to(scene.center);
    println!(
        "error orbit for athc-convert: --error-orbit {:.4},{:.4},{:.4},{:.4},{:.2},{}\n",
        scene.center[0],
        scene.center[1],
        scene.center[2],
        d,
        scene.cam.hfov.to_degrees(),
        width
    );
    let t0 = std::time::Instant::now();
    let views2 = scene.test_views(2 * width);
    let refs: Vec<Image> = views2.iter().map(|(_, c)| render(&scene.splats, c, None).downsample(2)).collect();
    let nfull = normal_coloured(&scene.splats);
    let nrefs: Vec<Image> = views2.iter().map(|(_, c)| render(&nfull, c, None).downsample(2)).collect();
    eprintln!("reference renders {:.1}s", t0.elapsed().as_secs_f32());
    let names: Vec<String> = views.iter().map(|(n, _)| n.clone()).collect();
    println!("reference {} ({} splats) supersampled 2×; views: {}\n", scene.name, scene.splats.len(), names.join(" · "));
    println!(
        "| cloud | splats | {} | mean | mean w/o far | normals mean | Δcov mean |\n|---|---|{}---|---|---|---|",
        names.join(" | "),
        "---|".repeat(names.len())
    );
    let save = arg(args, "--save").map(|s| s.to_string());
    let row = |label: &str, cut: &[Splat]| {
        let imgs: Vec<Image> = views.iter().map(|(_, c)| render(cut, c, None)).collect();
        let errs: Vec<f64> = imgs.iter().zip(&refs).map(|(i, r)| i.rel_mse(r)).collect();
        let cov: f64 = imgs.iter().zip(&refs).map(|(i, r)| i.alpha_diff(r)).sum::<f64>() / imgs.len() as f64;
        let mean = errs.iter().sum::<f64>() / errs.len() as f64;
        let nofar = errs[..errs.len() - 1].iter().sum::<f64>() / (errs.len() - 1) as f64;
        let nc = normal_coloured(cut);
        let nmean = views.iter().zip(&nrefs).map(|((_, c), r)| render(&nc, c, None).rel_mse(r)).sum::<f64>() / views.len() as f64;
        println!(
            "| {} | {} | {} | {:.3e} | {:.3e} | {:.3e} | {:.2e} |",
            label,
            cut.len(),
            errs.iter().map(|e| format!("{e:.2e}")).collect::<Vec<_>>().join(" | "),
            mean,
            nofar,
            nmean,
            cov
        );
        if let Some(dir) = &save {
            for v in [0usize, 4] {
                let _ = imgs[v].save_png(&format!("{dir}/{label}-v{v}.png"), 0.6);
                let _ = imgs[v].diff_png(&refs[v], &format!("{dir}/{label}-v{v}-diff.png"));
            }
        }
    };
    if args.iter().any(|a| a == "--self") {
        row("reference at 1×", &scene.splats);
    }
    if let Some(list) = arg(args, "--cand") {
        for spec in list.split(',') {
            let (label, asset) = spec.split_once('=').unwrap_or((spec, spec));
            let o = Scene::load(asset)?;
            row(label, &o.splats);
        }
    }
    Ok(())
}
