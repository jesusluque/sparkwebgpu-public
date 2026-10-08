//! Spark's runtime LoD traversal on the CPU (thread BE): the same
//! priority-by-`size / distance` refinement under a splat budget as
//! spark-rs lod_tree.rs `traverse_lod_trees` (with WgpuLod's default
//! foveation and the one-pixel limit), over every part of a scene at once,
//! with the nodes' sizes geometric (what the decoder computes today) or by
//! error (`athc_lod_error`), the result drawn and compared with the
//! reference as `cmp` does.
//!
//!   athc_measure trav --asset corvette-hd --cand v6e=corvette@DIR \
//!       --budgets 150000,300000,600000 --sizes geo,err,err:lambda=512 [--save DIR]
//!
//! A sizes spec: `geo`, `file` (the LODS sizes the file carries), or
//! `err[:key=value...]` with keys shape, lambda, mat, normal, scale.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use anyhow::{bail, Result};
use spark_lib::athc::{self, AthcFile, VirtualTree};
use spark_lib::athc_lod_error::{geometric_size, level_error_sizes, level_errors, level_lod_sizes, monotone_sizes, LodSizeOptions, SizeOptions};

use crate::arg;
use crate::cloud::*;
use crate::raster::*;

/// One part as Spark's LoD tree: merged nodes then splats.
pub struct Inst {
    pub nodes: Vec<Splat>,
    pub geo: Vec<f32>,
    pub child_start: Vec<u32>,
    pub child_count: Vec<u16>,
    pub file: AthcFile,
    pub tree: VirtualTree,
}

pub fn load_inst(path: &str, part: u16) -> Result<Inst> {
    let file = read_athc(path)?;
    let mut f = file.clone();
    athc::uncap_levels(&mut f);
    let tree = VirtualTree::of_file(&f, false)?;
    let merged = athc::merged_block(&f, &tree);
    let linear = f.header.flags & athc::FLAG_LINEAR != 0;
    let mut nodes = Vec::new();
    block_splats(&merged, linear, part, &mut nodes);
    for c in &f.chunks {
        block_splats(c, linear, part, &mut nodes);
    }
    let total = nodes.len();
    let mut child_start = vec![0u32; total];
    let mut child_count = vec![0u16; total];
    child_start[..tree.merged as usize].copy_from_slice(&tree.child_start);
    child_count[..tree.merged as usize].copy_from_slice(&tree.child_count);
    let geo = nodes.iter().map(|s| geometric_size(s.s, s.o)).collect();
    Ok(Inst { nodes, geo, child_start, child_count, file, tree })
}

impl Inst {
    /// Sizes by error for the merged nodes, geometric for the splats.
    pub fn error_sizes(&self, o: &SizeOptions) -> Vec<f32> {
        let levels = level_error_sizes(&self.file, o);
        self.place(&levels)
    }

    /// Geometric sizes scaled up by the node's appearance error:
    /// geo x clamp(((v + v0) / v0)^gamma, 1, hi), v = lambda var(c) +
    /// mat var(m) + normal spread + thick thickness.
    pub fn mod_sizes(&self, p: &Mod) -> Vec<f32> {
        let errs = level_errors(&self.file);
        let mut levels: Vec<Vec<f32>> = Vec::new();
        for (l, le) in errs.iter().enumerate() {
            let base = self.tree.level_base[l] as usize;
            levels.push(
                le.iter()
                    .enumerate()
                    .map(|(g, e)| {
                        let v = p.lambda * e.colour + p.mat * e.material + p.normal * e.spread + p.thick * e.thick + p.shape * e.shape;
                        let f = ((v + p.v0) / p.v0).powf(p.gamma).clamp(1.0, p.hi);
                        self.geo[base + g] * f as f32
                    })
                    .collect(),
            );
        }
        if p.monotone {
            monotone_sizes(&self.file, &mut levels);
        }
        self.place(&levels)
    }

    pub fn place(&self, levels: &[Vec<f32>]) -> Vec<f32> {
        let mut out = self.geo.clone();
        for (l, sizes) in levels.iter().enumerate() {
            let base = self.tree.level_base[l] as usize;
            out[base..base + sizes.len()].copy_from_slice(sizes);
        }
        if self.tree.synth_root {
            out[0] = levels[0].iter().cloned().fold(0.0, f32::max);
        }
        out
    }

    /// The LODS sizes the file carries (0: geometric), if any.
    pub fn file_sizes(&self) -> Option<Vec<f32>> {
        if self.file.levels.iter().any(|(_, b)| b.lod_size.is_empty()) {
            return None;
        }
        let levels: Vec<Vec<f32>> = self.file.levels.iter().map(|(_, b)| b.lod_size.clone()).collect();
        Some(self.place(&levels))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Mod {
    pub lambda: f64,
    pub mat: f64,
    pub normal: f64,
    pub thick: f64,
    pub shape: f64,
    pub v0: f64,
    pub gamma: f64,
    pub hi: f64,
    pub monotone: bool,
}

#[derive(PartialEq)]
struct Item(f32, u32, u32);
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0)
    }
}

fn pixel_scale(p: [f32; 3], size: f32, cam: &Camera) -> f32 {
    // WgpuLod's defaults: behind 0.2, cone 90 / 120 degrees, cone 0.4.
    let (behind, cone_fov0, cone_fov, cone_foveate) = (0.2f32, 90.0f32, 120.0f32, 0.4f32);
    let cone_dot0 = (0.5 * cone_fov0).to_radians().cos();
    let cone_dot = (0.5 * cone_fov).to_radians().cos().min(cone_dot0);
    let d = sub(p, cam.eye);
    let dist = dot(d, d).sqrt().max(1e-6);
    let ps = size / dist;
    let fd = dot(d, cam.fwd);
    let fov = if fd <= 0.0 {
        behind
    } else {
        let dt = fd / dist;
        if dt >= cone_dot0 {
            1.0
        } else if dt >= cone_dot {
            let t = (dt - cone_dot) / (cone_dot0 - cone_dot);
            cone_foveate + (1.0 - cone_foveate) * t
        } else {
            let t = dt / cone_dot;
            behind + (cone_foveate - behind) * t
        }
    };
    fov * ps
}

/// `traverse_lod_trees`: the elements drawn, per instance.
pub fn traverse(insts: &[(&Inst, &[f32])], cam: &Camera, max_splats: usize) -> Vec<Vec<u32>> {
    let limit = 1.0 / cam.fx;
    let ps = |k: usize, i: u32| pixel_scale(insts[k].0.nodes[i as usize].p, insts[k].1[i as usize], cam);
    let mut frontier = BinaryHeap::new();
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); insts.len()];
    let mut num = 0usize;
    for k in 0..insts.len() {
        frontier.push(Item(ps(k, 0), k as u32, 0));
        num += 1;
    }
    while let Some(&Item(scale, k, i)) = frontier.peek() {
        if scale <= limit {
            break;
        }
        let inst = insts[k as usize].0;
        let (count, start) = (inst.child_count[i as usize], inst.child_start[i as usize]);
        if count == 0 {
            frontier.pop();
            out[k as usize].push(i);
            continue;
        }
        let next = num - 1 + count as usize;
        if next > max_splats {
            break;
        }
        frontier.pop();
        for c in start..start + count as u32 {
            let s = ps(k as usize, c);
            if s <= limit {
                out[k as usize].push(c);
            } else {
                frontier.push(Item(s, k, c));
            }
        }
        num = next;
    }
    for Item(_, k, i) in frontier.drain() {
        out[k as usize].push(i);
    }
    out
}

enum Sizes {
    Geo,
    File,
    Lib,
    Err(SizeOptions),
    Mod(Mod),
}

fn parse_sizes(spec: &str) -> Result<Sizes> {
    let mut it = spec.split(':');
    match it.next() {
        Some("geo") => Ok(Sizes::Geo),
        Some("file") => Ok(Sizes::File),
        Some("lib") => Ok(Sizes::Lib),
        Some("mod") => {
            let mut m = Mod { lambda: 128.0, mat: 32.0, normal: 0.0, thick: 0.0, shape: 0.0, v0: 0.1, gamma: 0.5, hi: 4.0, monotone: true };
            for kv in it {
                let (k, v) = kv.split_once('=').unwrap_or((kv, "1"));
                let v: f64 = v.parse()?;
                match k {
                    "lambda" => m.lambda = v,
                    "mat" => m.mat = v,
                    "normal" => m.normal = v,
                    "thick" => m.thick = v,
                    "shape" => m.shape = v,
                    "v0" => m.v0 = v,
                    "gamma" => m.gamma = v,
                    "hi" => m.hi = v,
                    "monotone" => m.monotone = v != 0.0,
                    _ => bail!("unknown mod option {k}"),
                }
            }
            Ok(Sizes::Mod(m))
        }
        Some("err") => {
            let mut o = SizeOptions::default();
            for kv in it {
                let (k, v) = kv.split_once('=').unwrap_or((kv, "1"));
                let v: f64 = v.parse()?;
                match k {
                    "shape" => o.shape = v,
                    "lambda" => o.lambda = v,
                    "mat" => o.lambda_material = v,
                    "normal" => o.lambda_normal = v,
                    "scale" => o.scale = v,
                    _ => bail!("unknown size option {k}"),
                }
            }
            Ok(Sizes::Err(o))
        }
        _ => bail!("sizes spec {spec}: geo, file, err[:k=v...] or mod[:k=v...]"),
    }
}

pub fn candidate_files(dir: &str) -> Result<Vec<String>> {
    let mut files = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let name = e?.file_name().to_string_lossy().to_string();
        if name.ends_with(".athc") && !name.starts_with("catcher") && !name.starts_with("lamp-floor") {
            files.push(format!("{dir}/{name}"));
        }
    }
    files.sort();
    Ok(files)
}

pub fn run(scene: &Scene, args: &[String]) -> Result<()> {
    let width: usize = arg(args, "--width").map(|s| s.parse().unwrap()).unwrap_or(1920);
    let views = scene.test_views(width);
    let t0 = std::time::Instant::now();
    let views2 = scene.test_views(2 * width);
    let refs: Vec<Image> = views2.iter().map(|(_, c)| render(&scene.splats, c, None).downsample(2)).collect();
    eprintln!("reference renders {:.1}s", t0.elapsed().as_secs_f32());
    let names: Vec<String> = views.iter().map(|(n, _)| n.clone()).collect();
    let budgets: Vec<usize> = arg(args, "--budgets").unwrap_or("300000").split(',').map(|s| s.parse().unwrap()).collect();
    let sizes: Vec<String> = arg(args, "--sizes").unwrap_or("geo,err").split(',').map(|s| s.to_string()).collect();
    let save = arg(args, "--save").map(|s| s.to_string());
    println!("reference {} ({} splats) supersampled 2×; views: {}\n", scene.name, scene.splats.len(), names.join(" · "));
    println!(
        "| cloud | sizes | budget | drawn (mean) | {} | mean | mean w/o far |\n|---|---|---|---|{}---|---|",
        names.join(" | "),
        "---|".repeat(names.len())
    );
    let list = arg(args, "--cand").unwrap_or_else(|| panic!("--cand label=kind@DIR"));
    for spec in list.split(',') {
        let (label, asset) = spec.split_once('=').unwrap_or((spec, spec));
        let dir = asset.split_once('@').map(|(_, d)| d).unwrap_or(asset);
        let files = candidate_files(dir)?;
        let insts: Vec<Inst> = files.iter().enumerate().map(|(k, f)| load_inst(f, k as u16)).collect::<Result<_>>()?;
        for sz in &sizes {
            let opts = parse_sizes(sz)?;
            let per: Vec<Vec<f32>> = insts
                .iter()
                .map(|inst| match &opts {
                    Sizes::Err(o) => inst.error_sizes(o),
                    Sizes::Mod(m) => inst.mod_sizes(m),
                    Sizes::File => inst.file_sizes().unwrap_or_else(|| inst.geo.clone()),
                    Sizes::Geo => inst.geo.clone(),
                    Sizes::Lib => inst.place(&level_lod_sizes(&inst.file, &LodSizeOptions::default())),
                })
                .collect();
            // How the sizes compare with the geometric ones on merged nodes.
            if !matches!(opts, Sizes::Geo) {
                let mut r: Vec<f32> = Vec::new();
                for (inst, s) in insts.iter().zip(&per) {
                    for i in 0..inst.tree.merged as usize {
                        r.push(s[i] / inst.geo[i].max(1e-12));
                    }
                }
                if std::env::var("TRAV_DEBUG").is_ok() {
                    for (f, (inst, s)) in files.iter().zip(insts.iter().zip(&per)) {
                        let mut line = format!("{}:", f.rsplit('/').next().unwrap());
                        for l in 0..inst.tree.level_base.len() {
                            let a = inst.tree.level_base[l] as usize;
                            let b = if l + 1 < inst.tree.level_base.len() { inst.tree.level_base[l + 1] as usize } else { inst.tree.merged as usize };
                            let med = |v: &[f32]| { let mut v = v.to_vec(); v.sort_by(|x, y| x.total_cmp(y)); v[v.len() / 2] };
                            line += &format!(" L{}[{}] geo {:.3} err {:.3};", inst.file.levels[l].0, b - a, med(&inst.geo[a..b]), med(&s[a..b]));
                        }
                        eprintln!("{line}");
                    }
                }
                r.sort_by(|a, b| a.total_cmp(b));
                eprintln!(
                    "{label} {sz}: merged size / geometric p10 {:.3} p50 {:.3} p90 {:.3}",
                    r[r.len() / 10],
                    r[r.len() / 2],
                    r[r.len() * 9 / 10]
                );
            }
            let pairs: Vec<(&Inst, &[f32])> = insts.iter().zip(&per).map(|(i, s)| (i, &s[..])).collect();
            for &budget in &budgets {
                let mut errs = Vec::new();
                let mut drawn = 0usize;
                for (v, (_, cam)) in views.iter().enumerate() {
                    let picks = traverse(&pairs, cam, budget);
                    let mut cut = Vec::new();
                    for (k, p) in picks.iter().enumerate() {
                        for &i in p {
                            cut.push(insts[k].nodes[i as usize].clone());
                        }
                    }
                    drawn += cut.len();
                    let img = render(&cut, cam, None);
                    errs.push(img.rel_mse(&refs[v]));
                    if let Some(dir) = &save {
                        if v == 0 || v == 4 || v == 5 {
                            let tag = format!("{label}-{}-{budget}-v{v}", sz.replace(':', "_"));
                            let _ = img.save_png(&format!("{dir}/{tag}.png"), 0.6);
                            let _ = img.diff_png(&refs[v], &format!("{dir}/{tag}-diff.png"));
                        }
                    }
                }
                let mean = errs.iter().sum::<f64>() / errs.len() as f64;
                let nofar = errs[..errs.len() - 1].iter().sum::<f64>() / (errs.len() - 1) as f64;
                println!(
                    "| {} | {} | {} | {} | {} | {:.3e} | {:.3e} |",
                    label,
                    sz,
                    budget,
                    drawn / views.len(),
                    errs.iter().map(|e| format!("{e:.2e}")).collect::<Vec<_>>().join(" | "),
                    mean,
                    nofar
                );
            }
        }
    }
    Ok(())
}
