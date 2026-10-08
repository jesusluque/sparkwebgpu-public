//! `athc-convert --drop-hidden / --decimate / --scene` (spark_lib::athc_prune).

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use spark_lib::athc::AthcFile;
use spark_lib::athc_build::BuildOptions;
use spark_lib::athc_prune::{
    carry_layers, check_views, clip_samples, file_influences, file_splats, pose_splats, prune_scene, CamSpec, Decimate,
    DecimateFilter, HiddenRule, NearSet, PruneOptions, SceneInput,
};
use spark_lib::athc_skin::{AthcSkeleton, SkinClip};
use spark_lib::athc_v3::{gunzip, read_v3_skeleton, write_v3_smallest_with, COMPRESSION_GZIP, COMPRESSION_NONE};
use spark_lib::athl::AthlFile;
use spark_lib::cpu_raster::{Proxy, Splat};

/// Flags that take a value.
const VALUED: &[&str] = &[
    "--scene",
    "--occluder",
    "--camera",
    "--views",
    "--vis-width",
    "--hidden-t",
    "--hidden-px",
    "--clips-json",
    "--pose-samples",
    "--pose-views",
    "--pose-width",
    "--decimate",
    "--athl",
    "--athl-out",
    "--check-width",
    "--keep",
    "--report",
    "--check-dir",
];

fn values<'a>(args: &'a [String], f: &str) -> Vec<&'a str> {
    args.iter().enumerate().filter(|(_, a)| *a == f).filter_map(|(i, _)| args.get(i + 1).map(|s| s.as_str())).collect()
}

fn value<'a>(args: &'a [String], f: &str) -> Option<&'a str> {
    values(args, f).into_iter().next()
}

fn parsed<T: std::str::FromStr>(args: &[String], f: &str, default: T) -> Result<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    value(args, f).map_or(Ok(default), |v| v.parse::<T>().with_context(|| f.to_string()))
}

/// A file's part name: its name up to the first '-' or '.'.
pub fn part_of(path: &str) -> String {
    let base = std::path::Path::new(path).file_name().map_or(path.to_string(), |b| b.to_string_lossy().to_string());
    base.split(['-', '.']).next().unwrap_or(&base).to_string()
}

fn read_file(path: &str) -> Result<(AthcFile, Option<AthcSkeleton>, u64)> {
    let bytes = std::fs::read(path).with_context(|| path.to_string())?;
    let file = super::read_any(&bytes)?;
    let skel = if bytes.len() >= 4 && u32::from_le_bytes(bytes[..4].try_into().unwrap()) == spark_lib::athc_v3::ATH3_MAGIC {
        read_v3_skeleton(&bytes)?
    } else {
        None
    };
    Ok((file, skel, bytes.len() as u64))
}

fn camera_of(args: &[String]) -> Result<CamSpec> {
    let v = value(args, "--camera").ok_or_else(|| anyhow!("--drop-hidden and --check need --camera ex,ey,ez,tx,ty,tz,ux,uy,uz,hfov[,aspect]"))?;
    let f: Vec<f32> = v.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>().context("--camera")?;
    if f.len() != 10 && f.len() != 11 {
        bail!("--camera takes 10 or 11 numbers");
    }
    Ok(CamSpec {
        eye: [f[0], f[1], f[2]],
        target: [f[3], f[4], f[5]],
        up: [f[6], f[7], f[8]],
        hfov: f[9].to_radians(),
        aspect: f.get(10).copied().unwrap_or(16.0 / 9.0),
    })
}

fn decimate_of(spec: &str) -> Result<(String, Decimate, DecimateFilter)> {
    let (part, rest) = spec.split_once('=').ok_or_else(|| anyhow!("--decimate PART=K[:GROW][@coat], not {spec}"))?;
    let (rest, filter) = match rest.split_once('@') {
        Some((r, "coat")) => (r, DecimateFilter::Coat),
        Some((r, f)) if f.starts_with("near:") => {
            // near:REF.athc[,REACH_MM]
            let spec = &f[5..];
            let (path, mm) = match spec.rsplit_once(',') {
                Some((p, m)) => (p, m.parse::<f32>().context("--decimate @near reach (mm)")?),
                None => (spec, 5.0),
            };
            let (file, _, _) = read_file(path)?;
            eprintln!("decimate {part}: on the surface of {path} ({} splats) within {mm} mm", file.header.count);
            (r, DecimateFilter::Near(std::sync::Arc::new(NearSet::new(&file, mm / 1000.0))))
        }
        Some((_, f)) => bail!("--decimate filter {f}: @coat or @near:REF.athc[,MM]"),
        None => (rest, DecimateFilter::All),
    };
    let (k, grow) = match rest.split_once(':') {
        Some((k, g)) => (k, Some(g)),
        None => (rest, None),
    };
    let mut d = Decimate::new(k.parse().context("--decimate K")?);
    if let Some(g) = grow {
        d.grow = g.parse().context("--decimate GROW")?;
    }
    Ok((part.to_string(), d, filter))
}

fn clips_of(path: &str, skeleton: &AthcSkeleton) -> Result<Vec<SkinClip>> {
    let json: Value = serde_json::from_slice(&std::fs::read(path).with_context(|| path.to_string())?)?;
    let dir = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."));
    let mut out = Vec::new();
    for c in json["clips"].as_array().ok_or_else(|| anyhow!("{path} has no clips"))? {
        let f = dir.join(c["file"].as_str().ok_or_else(|| anyhow!("a clip without a file"))?);
        let raw = std::fs::read(&f).with_context(|| f.display().to_string())?;
        let raw = if raw.starts_with(&[0x1f, 0x8b]) { gunzip(&raw)? } else { raw };
        let (clip, joints) = SkinClip::from_atcl(&raw)?;
        if joints != skeleton.joint_count() {
            bail!("{}: {} joints, the skeleton has {}", f.display(), joints, skeleton.joint_count());
        }
        out.push(clip);
    }
    Ok(out)
}

fn splats_of(files: &[&AthcFile], proxy: Proxy) -> Vec<Splat> {
    files.iter().flat_map(|f| file_splats(f, proxy)).collect()
}

pub fn run(args: &[String]) -> Result<()> {
    let flag = |f: &str| args.iter().any(|a| a == f);
    let positional: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|&(i, a)| !a.starts_with("--") && !(i > 0 && VALUED.contains(&args[i - 1].as_str())))
        .map(|(_, a)| a)
        .collect();
    // (input path, output path or None for an occluder)
    let mut jobs: Vec<(String, Option<String>)> = Vec::new();
    if let Some(dir) = value(args, "--scene") {
        std::fs::create_dir_all(dir)?;
        for p in &positional {
            let name = std::path::Path::new(p.as_str()).file_name().unwrap().to_string_lossy().to_string();
            jobs.push((p.to_string(), Some(format!("{dir}/{name}"))));
        }
    } else {
        if positional.len() != 2 {
            bail!("usage: athc-convert in.athc out.athc --drop-hidden|--decimate ... | athc-convert --scene OUTDIR a.athc ...");
        }
        jobs.push((positional[0].to_string(), Some(positional[1].to_string())));
    }
    for o in values(args, "--occluder") {
        jobs.push((o.to_string(), None));
    }
    let keep: Vec<&str> = values(args, "--keep");
    let mut decimate = Vec::new();
    for d in values(args, "--decimate") {
        decimate.push(decimate_of(d)?);
    }
    let mut athl_for: Vec<(String, String, String)> = Vec::new(); // part, in, out
    if value(args, "--scene").is_some() {
        for a in values(args, "--athl") {
            let (part, path) = a.split_once('=').ok_or_else(|| anyhow!("--athl PART=IN.athl in a scene"))?;
            let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy().to_string();
            athl_for.push((part.to_string(), path.to_string(), format!("{}/{}", value(args, "--scene").unwrap(), name)));
        }
    } else if let (Some(i), Some(o)) = (value(args, "--athl"), value(args, "--athl-out")) {
        athl_for.push((part_of(&jobs[0].0), i.to_string(), o.to_string()));
    }
    let drop_hidden = flag("--drop-hidden");
    let check = flag("--check");
    let camera = if drop_hidden || check { Some(camera_of(args)?) } else { None };

    let mut inputs = Vec::new();
    let mut skeleton = None;
    let mut input_bytes = Vec::new();
    for (path, out) in &jobs {
        let (file, skel, bytes) = read_file(path)?;
        let part = part_of(path);
        eprintln!("{path}: {} splats, {} levels{}", file.header.count, file.levels.len(), if skel.is_some() { ", skinned" } else { "" });
        if skel.is_some() && out.is_some() {
            if skeleton.is_some() {
                bail!("one skinned file a scene");
            }
            skeleton = skel;
        }
        let dec = decimate.iter().find(|(p, _, _)| *p == part).map(|(_, d, f)| (*d, f.clone()));
        inputs.push(SceneInput { name: part.clone(), file, output: out.is_some(), prune: !keep.contains(&part.as_str()), decimate: dec });
        input_bytes.push(bytes);
    }
    for (p, _, _) in &decimate {
        if !inputs.iter().any(|s| &s.name == p && s.output) {
            bail!("--decimate {p}: no such written part");
        }
    }
    let clips = match (value(args, "--clips-json"), &skeleton) {
        (Some(j), Some(s)) => clips_of(j, s)?,
        (Some(_), None) => bail!("--clips-json without a skinned file"),
        _ => Vec::new(),
    };
    let o = PruneOptions {
        camera: camera.unwrap_or(CamSpec { eye: [0.0; 3], target: [0.0; 3], up: [0.0, 0.0, 1.0], hfov: 1.0, aspect: 1.0 }),
        views: parsed(args, "--views", 800)?,
        width: parsed(args, "--vis-width", 800)?,
        below: flag("--sphere"),
        rule: HiddenRule { max_transmittance: parsed(args, "--hidden-t", 0.02)?, max_pixels: parsed(args, "--hidden-px", 0.02)? },
        drop_hidden,
        clips,
        skeleton: skeleton.clone(),
        pose_samples: parsed(args, "--pose-samples", 6)?,
        pose_views: parsed(args, "--pose-views", 48)?,
        pose_width: parsed(args, "--pose-width", 640)?,
        build: BuildOptions::default(),
    };
    let t0 = std::time::Instant::now();
    let pruned = prune_scene(&inputs, &o)?;
    let compression = if flag("--gzip") { COMPRESSION_GZIP } else { COMPRESSION_NONE };
    let mut report = Vec::new();
    for (k, ((path, out), p)) in jobs.iter().zip(&pruned).enumerate() {
        let (Some(out), Some(p)) = (out, p) else { continue };
        let s = &inputs[k];
        let (bytes, chosen) = write_v3_smallest_with(&p.file, compression, if s.file.has_skin() { skeleton.as_ref() } else { None })?;
        std::fs::write(out, &bytes)?;
        let n = p.file.header.count as usize;
        println!(
            "{} -> {}: {} -> {} splats ({} hidden, {}), {:.1} -> {:.1} MB",
            path,
            out,
            p.input_splats,
            n,
            p.hidden,
            p.decimate.as_ref().map_or("not decimated".to_string(), |d| format!("decimated {} opaque into {} runs", d.candidates, d.runs - (p.input_splats - p.hidden - d.candidates))),
            input_bytes[k] as f64 / 1e6,
            bytes.len() as f64 / 1e6
        );
        let mut entry = json!({
            "part": s.name, "input": path, "output": out,
            "inputSplats": p.input_splats, "hidden": p.hidden, "splats": n,
            "inputBytes": input_bytes[k], "bytes": bytes.len(),
            "levels": p.file.levels.len(),
            "encodings": format!("{chosen:?}"),
        });
        let layout = spark_lib::athc_v3::parse_v3(&bytes)?;
        let tier = |t: u32| -> u64 { layout.blocks.iter().map(|b| b.tier_range(&layout.sections, t).1).sum() };
        entry["tierBytes"] = json!({ "1": tier(1), "2": tier(2), "3": tier(3) });
        if let Some(d) = &p.decimate {
            entry["decimate"] = json!({ "candidates": d.candidates, "fullRuns": d.full_runs, "output": d.output });
        }
        if let Some((_, ai, ao)) = athl_for.iter().find(|(part, _, _)| *part == s.name) {
            let raw = std::fs::read(ai).with_context(|| ai.clone())?;
            let athl = AthlFile::read(&raw)?;
            let carried = carry_layers(&athl, &s.file, &p.file, &p.track, &bytes)?;
            let written = carried.write()?;
            std::fs::write(ao, &written)?;
            // Every layer over the new cloud's virtual order, element for element.
            if carried.element_count != carried.splat_base + n as u32 {
                bail!("{ao}: {} elements for {} + {}", carried.element_count, carried.splat_base, n);
            }
            println!(
                "{} -> {} ({} layers, {:.2} -> {:.2} MB, {} elements = {} merged + {} splats, cloud {:016x})",
                ai,
                ao,
                carried.layers.len(),
                raw.len() as f64 / 1e6,
                written.len() as f64 / 1e6,
                carried.element_count,
                carried.merged,
                carried.splat_count,
                carried.cloud_hash
            );
            let v = verify_layers(&s.file, &athl, &bytes, &written)?;
            println!(
                "  alignment (read back): {} splats at an input splat's place, layer values max |d| {:.2e} (largest value {:.3e}); {} merged runs; hash ok",
                v.0, v.1, v.2, v.3
            );
            entry["athl"] = json!({ "input": ai, "output": ao, "inputBytes": raw.len(), "bytes": written.len(),
                "layers": carried.layers.len(), "elements": carried.element_count, "cloudHash": format!("{:016x}", carried.cloud_hash),
                "verify": { "exactSplats": v.0, "maxAbsDiff": v.1, "maxValue": v.2, "mergedSplats": v.3 } });
        }
        report.push(entry);
    }
    eprintln!("pruned and built in {:.0}s", t0.elapsed().as_secs_f32());
    let mut checks = Vec::new();
    if check {
        let cam = camera.unwrap();
        let width: usize = parsed(args, "--check-width", 1920)?;
        let before: Vec<&AthcFile> = inputs.iter().map(|s| &s.file).collect();
        let after: Vec<&AthcFile> = inputs.iter().zip(&pruned).map(|(s, p)| p.as_ref().map_or(&s.file, |p| &p.file)).collect();
        let bounds = spark_lib::athc_prune::splat_bounds(&splats_of(&before, Proxy::Albedo));
        let views = cam.test_views(width, o.below, bounds);
        let views2 = cam.test_views(2 * width, o.below, bounds);
        let dump = value(args, "--check-dir");
        if let Some(d) = dump {
            std::fs::create_dir_all(d)?;
        }
        if o.clips.is_empty() {
            let rows = check_views(
                &splats_of(&before, Proxy::Albedo),
                &splats_of(&after, Proxy::Albedo),
                &splats_of(&before, Proxy::Normal),
                &splats_of(&after, Proxy::Normal),
                &views,
                &views2,
                dump,
            );
            for r in rows {
                checks.push(json!({ "view": r.view, "relMse": r.rel_mse, "relMseNormals": r.rel_mse_normals, "coverage": r.coverage, "floorSS": r.floor_ss, "outSS": r.out_ss }));
            }
        } else {
            // Held-out poses (between the samples), from the default and the close view.
            let skel = skeleton.as_ref().unwrap();
            let (fb, fa) = (before[0], after[0]);
            let (ib, ia) = (file_influences(fb), file_influences(fa));
            let picks: Vec<usize> = vec![0, 4];
            for clip in &o.clips {
                let (_, held) = clip_samples(clip.times.len(), o.pose_samples);
                let pose = |f: &AthcFile, inf: &[Vec<(u32, f32)>], proxy| pose_splats(&file_splats(f, proxy), inf, skel, clip, held);
                let rows = check_views(
                    &pose(fb, &ib, Proxy::Albedo),
                    &pose(fa, &ia, Proxy::Albedo),
                    &pose(fb, &ib, Proxy::Normal),
                    &pose(fa, &ia, Proxy::Normal),
                    &picks.iter().map(|&i| views[i].clone()).collect::<Vec<_>>(),
                    &picks.iter().map(|&i| views2[i].clone()).collect::<Vec<_>>(),
                    None,
                );
                for r in rows {
                    checks.push(json!({ "view": format!("{} {} (sample {})", clip.name, r.view, held), "relMse": r.rel_mse, "relMseNormals": r.rel_mse_normals, "coverage": r.coverage, "floorSS": r.floor_ss, "outSS": r.out_ss }));
                }
            }
        }
        println!("\n| view | relMSE out vs in | normals | Δcoverage | in vs 2x ref | out vs 2x ref |\n|---|---|---|---|---|---|");
        for c in &checks {
            println!(
                "| {} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} |",
                c["view"].as_str().unwrap(),
                c["relMse"].as_f64().unwrap(),
                c["relMseNormals"].as_f64().unwrap(),
                c["coverage"].as_f64().unwrap(),
                c["floorSS"].as_f64().unwrap(),
                c["outSS"].as_f64().unwrap()
            );
        }
    }
    if let Some(r) = value(args, "--report") {
        let doc = json!({ "args": args, "files": report, "check": checks, "seconds": t0.elapsed().as_secs_f32() });
        std::fs::write(r, serde_json::to_string_pretty(&doc)?)?;
    }
    Ok(())
}

/// Reads the written cloud and layers back and checks them against the
/// input's: the hash, the element counts, and for every output splat that
/// sits exactly where an input splat sat (same centre and opacity bits),
/// every layer's value against that splat's. Returns (splats matched, the
/// largest difference, the largest value, splats with no match: decimated runs).
fn verify_layers(input: &AthcFile, athl_in: &AthlFile, athc_out: &[u8], athl_out: &[u8]) -> Result<(usize, f32, f32, usize)> {
    use spark_lib::athc::VirtualTree;
    let file = super::read_any(athc_out)?;
    let athl = AthlFile::read(athl_out)?;
    if athl.cloud_hash != spark_lib::athl::cloud_hash(athc_out) {
        bail!("the written .athl's cloud hash is not the written cloud's");
    }
    let tree = VirtualTree::of_file(&file, true)?;
    if athl.splat_base != tree.splat_base || athl.merged != tree.merged || athl.splat_count != file.header.count {
        bail!("the written .athl is not over the written cloud's virtual order");
    }
    let key = |b: &spark_lib::athc::AthcBlock, i: usize| -> [u32; 4] { [0, 1, 2, 3].map(|k| b.positions[i * 4 + k].to_bits()) };
    let a = input.splats();
    let mut at = std::collections::HashMap::with_capacity(a.n);
    for i in 0..a.n {
        at.insert(key(&a, i), i);
    }
    let b = file.splats();
    let mut kinds: Vec<(u16, u16)> = athl_in.layers.iter().map(|l| (l.group, l.kind)).collect();
    kinds.sort();
    kinds.dedup();
    let dense: Vec<_> = kinds
        .iter()
        .map(|&(g, k)| (athl_in.dense(g, k).unwrap(), athl.dense(g, k).map(|d| d.1).unwrap_or_default()))
        .collect();
    let (mut matched, mut unmatched, mut worst, mut top) = (0, 0, 0f32, 0f32);
    for j in 0..b.n {
        let Some(&i) = at.get(&key(&b, j)) else {
            unmatched += 1;
            continue;
        };
        matched += 1;
        for ((c, before), after) in &dense {
            let c = *c as usize;
            for k in 0..c {
                let x = before[(athl_in.splat_base as usize + i) * c + k];
                let y = after.get((athl.splat_base as usize + j) * c + k).copied().unwrap_or(0.0);
                worst = worst.max((x - y).abs());
                top = top.max(x.abs());
            }
        }
    }
    Ok((matched, worst, top, unmatched))
}
