//! A greedy, error-driven pairwise merge (NanoGS-like) on the CPU, against
//! the octree LoD cut at equal splat counts.
//!
//! Clusters are raw moments (weight W = opacity x two-axis area, the LoD's),
//! so a merge is exact moment matching whatever the order. The cost of a
//! pair is an image-space proxy:
//!   A_px x ( B / W + lambda x w_i w_j / W^2 |c_i - c_j|^2 )
//! B Runnalls' bound on KL(mixture || merged) with every covariance floored
//! at 1% of the merged one's mean variance (so surfels stay finite), A_px the
//! merged weight in pixels at the nearest the orbit comes (a merge nobody can
//! see costs nothing). Refused: different key (part, glass, thin, Schlick,
//! mirror), normals more than 60 degrees apart. Candidates: neighbours in
//! three shifted Morton orders; disjoint pairs by mutual best; each pass
//! merges at most the cheaper half of them.

use std::time::Instant;

use anyhow::Result;

use crate::arg;
use crate::cloud::*;
use crate::raster::*;

#[derive(Clone, Copy)]
struct Cl {
    w: f64,
    m1: [f64; 3],
    m2: [f64; 6],
    c: [f64; 3],
    n: [f64; 3],
    key: u32,
    members: u32,
    /// the original splat when it is alone
    orig: u32,
}

fn cl_of(s: &Splat, i: usize) -> Cl {
    let w = (s.o as f64) * s.area2() as f64;
    let w = w.max(1e-30);
    let cov = s.cov();
    let p = s.p.map(|v| v as f64);
    let m2 = [
        w * (cov[0] + p[0] * p[0]),
        w * (cov[1] + p[1] * p[1]),
        w * (cov[2] + p[2] * p[2]),
        w * (cov[3] + p[0] * p[1]),
        w * (cov[4] + p[0] * p[2]),
        w * (cov[5] + p[1] * p[2]),
    ];
    Cl {
        w,
        m1: p.map(|v| w * v),
        m2,
        c: s.c.map(|v| w * v as f64),
        n: s.n.map(|v| w * v as f64),
        key: s.key,
        members: 1,
        orig: i as u32,
    }
}

fn mean_cov(c: &Cl) -> ([f64; 3], [f64; 6]) {
    let mu = c.m1.map(|v| v / c.w);
    let s = [
        c.m2[0] / c.w - mu[0] * mu[0],
        c.m2[1] / c.w - mu[1] * mu[1],
        c.m2[2] / c.w - mu[2] * mu[2],
        c.m2[3] / c.w - mu[0] * mu[1],
        c.m2[4] / c.w - mu[0] * mu[2],
        c.m2[5] / c.w - mu[1] * mu[2],
    ];
    (mu, s)
}

fn det_floor(s: &[f64; 6], eps: f64) -> f64 {
    let (a, b, c) = (s[0] + eps, s[1] + eps, s[2] + eps);
    let (d, e, f) = (s[3], s[4], s[5]);
    (a * (b * c - f * f) - d * (d * c - f * e) + e * (d * f - b * e)).max(1e-300)
}

fn add(a: &Cl, b: &Cl) -> Cl {
    let mut r = *a;
    r.w += b.w;
    for k in 0..3 {
        r.m1[k] += b.m1[k];
        r.c[k] += b.c[k];
        r.n[k] += b.n[k];
    }
    for k in 0..6 {
        r.m2[k] += b.m2[k];
    }
    r.members += b.members;
    r
}

struct Ctx {
    eye_dist: f64,
    center: [f64; 3],
    fx: f64,
    near: f64,
    lambda: f64,
    /// 0 Runnalls + colour, 1 plane ISE (alpha, colour, normal)
    kind: u32,
    mu_n: f64,
}

/// A cluster's opacity (W / two-axis area) and unit normal.
fn opacity_of(s: &[f64; 6], w: f64) -> f64 {
    let (vals, _) = eigen(s);
    let mut v = vals.map(|x| x.max(1e-30).sqrt());
    v.sort_by(|a, b| b.total_cmp(a));
    w / (v[0] * v[1]).max(1e-30)
}

/// 2D (in the plane e1, e2) covariance of a 3D one.
fn plane_cov(s: &[f64; 6], e: &[[f64; 3]; 2]) -> [f64; 3] {
    let m = |u: &[f64; 3], v: &[f64; 3]| {
        u[0] * (s[0] * v[0] + s[3] * v[1] + s[4] * v[2]) + u[1] * (s[3] * v[0] + s[1] * v[1] + s[5] * v[2]) + u[2] * (s[4] * v[0] + s[5] * v[1] + s[2] * v[2])
    };
    [m(&e[0], &e[0]), m(&e[0], &e[1]), m(&e[1], &e[1])]
}

/// ∫ exp(-x'A⁻¹x/2) exp(-(x-d)'B⁻¹(x-d)/2) over the plane.
fn overlap(a: &[f64; 3], b: &[f64; 3], d: [f64; 2], floor: f64) -> f64 {
    let fa = [a[0] + floor, a[1], a[2] + floor];
    let fb = [b[0] + floor, b[1], b[2] + floor];
    let da = (fa[0] * fa[2] - fa[1] * fa[1]).max(1e-300);
    let db = (fb[0] * fb[2] - fb[1] * fb[1]).max(1e-300);
    let c = [fa[0] + fb[0], fa[1] + fb[1], fa[2] + fb[2]];
    let dc = (c[0] * c[2] - c[1] * c[1]).max(1e-300);
    let q = (c[2] * d[0] * d[0] - 2.0 * c[1] * d[0] * d[1] + c[0] * d[1] * d[1]) / dc;
    std::f64::consts::TAU * (da * db / dc).sqrt() * (-0.5 * q).exp()
}

fn cost_ise(a: &Cl, b: &Cl, ctx: &Ctx) -> f64 {
    let m = add(a, b);
    let (mu, s) = mean_cov(&m);
    let (vals, cols) = eigen(&s);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&x, &y| vals[y].total_cmp(&vals[x]));
    let e = [cols[order[0]], cols[order[1]]];
    let floor = 1e-4 * vals[order[0]].max(1e-30);
    let parts = [(a, 1.0f64), (b, 1.0), (&m, -1.0)];
    let mut g = Vec::with_capacity(3);
    for (c, sign) in parts {
        let (cm, cs) = mean_cov(c);
        let o = opacity_of(&cs, c.w).min(1.0);
        let col = c.c.map(|v| v / c.w);
        let nl = (c.n[0].powi(2) + c.n[1].powi(2) + c.n[2].powi(2)).sqrt();
        let n = if nl > 0.0 { c.n.map(|v| v / nl) } else { [0.0; 3] };
        let d = [cm[0] - mu[0], cm[1] - mu[1], cm[2] - mu[2]];
        let pd = [d[0] * e[0][0] + d[1] * e[0][1] + d[2] * e[0][2], d[0] * e[1][0] + d[1] * e[1][1] + d[2] * e[1][2]];
        g.push((plane_cov(&cs, &e), pd, sign * o, col, n));
    }
    let mut err = 0.0;
    for x in 0..3 {
        for y in 0..3 {
            let (ax, px, ox, cx, nx) = &g[x];
            let (ay, py, oy, cy, ny) = &g[y];
            let k = overlap(ax, ay, [py[0] - px[0], py[1] - px[1]], floor);
            let ch = 1.0 + ctx.lambda * (cx[0] * cy[0] + cx[1] * cy[1] + cx[2] * cy[2]) + ctx.mu_n * (nx[0] * ny[0] + nx[1] * ny[1] + nx[2] * ny[2]);
            err += ox * oy * k * ch;
        }
    }
    let r = ((mu[0] - ctx.center[0]).powi(2) + (mu[1] - ctx.center[1]).powi(2) + (mu[2] - ctx.center[2]).powi(2)).sqrt();
    let z = (ctx.eye_dist - r).max(ctx.near);
    err.max(0.0) * (ctx.fx / z).powi(2)
}

fn cost(a: &Cl, b: &Cl, ctx: &Ctx) -> f64 {
    if a.key != b.key {
        return f64::INFINITY;
    }
    if ctx.kind == 1 {
        let na = (a.n[0].powi(2) + a.n[1].powi(2) + a.n[2].powi(2)).sqrt();
        let nb = (b.n[0].powi(2) + b.n[1].powi(2) + b.n[2].powi(2)).sqrt();
        if na > 0.0 && nb > 0.0 && (a.n[0] * b.n[0] + a.n[1] * b.n[1] + a.n[2] * b.n[2]) / (na * nb) < 0.5 {
            return f64::INFINITY;
        }
        return cost_ise(a, b, ctx);
    }
    let na = (a.n[0].powi(2) + a.n[1].powi(2) + a.n[2].powi(2)).sqrt();
    let nb = (b.n[0].powi(2) + b.n[1].powi(2) + b.n[2].powi(2)).sqrt();
    if na > 0.0 && nb > 0.0 && (a.n[0] * b.n[0] + a.n[1] * b.n[1] + a.n[2] * b.n[2]) / (na * nb) < 0.5 {
        return f64::INFINITY;
    }
    let m = add(a, b);
    let (mu, s) = mean_cov(&m);
    let (_, sa) = mean_cov(a);
    let (_, sb) = mean_cov(b);
    let eps = 0.01 * (s[0] + s[1] + s[2]) / 3.0;
    let bk = 0.5 * (m.w * det_floor(&s, eps).ln() - a.w * det_floor(&sa, eps).ln() - b.w * det_floor(&sb, eps).ln()) / m.w;
    let ca = a.c.map(|v| v / a.w);
    let cb = b.c.map(|v| v / b.w);
    let dc = (ca[0] - cb[0]).powi(2) + (ca[1] - cb[1]).powi(2) + (ca[2] - cb[2]).powi(2);
    let col = a.w * b.w / (m.w * m.w) * dc;
    let r = ((mu[0] - ctx.center[0]).powi(2) + (mu[1] - ctx.center[1]).powi(2) + (mu[2] - ctx.center[2]).powi(2)).sqrt();
    let z = (ctx.eye_dist - r).max(ctx.near);
    let apx = m.w * (ctx.fx / z).powi(2);
    apx * (bk.max(0.0) + ctx.lambda * col)
}

/// Symmetric 3x3 eigen (Jacobi), values and column vectors.
fn eigen(s: &[f64; 6]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = [[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]];
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..30 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-30 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let sn = t * c;
            for k in 0..3 {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - sn * akq;
                a[k][q] = sn * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - sn * aqk;
                a[q][k] = sn * apk + c * aqk;
            }
            for k in 0..3 {
                let (vkp, vkq) = (v[k][p], v[k][q]);
                v[k][p] = c * vkp - sn * vkq;
                v[k][q] = sn * vkp + c * vkq;
            }
        }
    }
    let vals = [a[0][0], a[1][1], a[2][2]];
    let cols = [[v[0][0], v[1][0], v[2][0]], [v[0][1], v[1][1], v[2][1]], [v[0][2], v[1][2], v[2][2]]];
    (vals, cols)
}

fn quat_from_cols(c: [[f64; 3]; 3]) -> [f32; 4] {
    // columns are the axes; make it right-handed
    let mut c = c;
    let cr = [
        c[0][1] * c[1][2] - c[0][2] * c[1][1],
        c[0][2] * c[1][0] - c[0][0] * c[1][2],
        c[0][0] * c[1][1] - c[0][1] * c[1][0],
    ];
    if cr[0] * c[2][0] + cr[1] * c[2][1] + cr[2] * c[2][2] < 0.0 {
        c[2] = c[2].map(|v| -v);
    }
    let m = glam::DMat3::from_cols(c[0].into(), c[1].into(), c[2].into());
    let q = glam::DQuat::from_mat3(&m).normalize();
    [q.x as f32, q.y as f32, q.z as f32, q.w as f32]
}

/// A covariance as (quaternion x y z w, scales).
pub fn orient(cov: &[f64; 6]) -> ([f32; 4], [f32; 3]) {
    let (vals, cols) = eigen(cov);
    (quat_from_cols(cols), vals.map(|v| v.max(1e-18).sqrt() as f32))
}

fn to_splat(c: &Cl, orig: &[Splat]) -> Splat {
    if c.members == 1 {
        return orig[c.orig as usize].clone();
    }
    let (mu, s) = mean_cov(c);
    let (vals, cols) = eigen(&s);
    let sc = vals.map(|v| v.max(1e-18).sqrt());
    let mut sorted = sc;
    sorted.sort_by(|a, b| b.total_cmp(a));
    let o = c.w / (sorted[0] * sorted[1]).max(1e-30);
    let col = c.c.map(|v| (v / c.w) as f32);
    let nl = (c.n[0].powi(2) + c.n[1].powi(2) + c.n[2].powi(2)).sqrt();
    let n = if nl > 0.0 { c.n.map(|v| (v / nl) as f32) } else { [0.0; 3] };
    Splat {
        p: mu.map(|v| v as f32),
        o: o as f32,
        q: quat_from_cols(cols),
        s: sc.map(|v| v as f32),
        c: col,
        n,
        part: orig[c.orig as usize].part,
        key: c.key,
    }
}

fn morton(p: [f64; 3], lo: [f64; 3], ext: f64, shift: [f64; 3]) -> u64 {
    let mut code = 0u64;
    let q = [0, 1, 2].map(|k| (((p[k] - lo[k] + shift[k]) / ext).clamp(0.0, 0.999_999) * 2_097_152.0) as u64);
    for b in 0..21 {
        for (k, v) in q.iter().enumerate() {
            code |= ((v >> b) & 1) << (3 * b + k);
        }
    }
    code
}

/// Merge down through `targets` (descending), calling `snap` at each.
fn greedy(splats: &[Splat], ctx: &Ctx, targets: &[usize], mut snap: impl FnMut(usize, Vec<Splat>)) {
    let mut cls: Vec<Cl> = splats.iter().enumerate().map(|(i, s)| cl_of(s, i)).collect();
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for s in splats {
        for k in 0..3 {
            lo[k] = lo[k].min(s.p[k] as f64);
            hi[k] = hi[k].max(s.p[k] as f64);
        }
    }
    let ext = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max) * 1.1 + 1e-9;
    let shifts = [[0.0, 0.0, 0.0], [ext * 0.0123, ext * 0.0071, ext * 0.0093], [ext * 0.0031, ext * 0.0157, ext * 0.0047]];
    const K: usize = 6;
    let mut ti = 0;
    let mut pass = 0;
    while ti < targets.len() {
        let alive = cls.len();
        if alive <= targets[ti] {
            snap(targets[ti], cls.iter().map(|c| to_splat(c, splats)).collect());
            ti += 1;
            continue;
        }
        let t0 = Instant::now();
        // candidates and best partner
        let mut best: Vec<(f64, u32)> = vec![(f64::INFINITY, u32::MAX); alive];
        for sh in &shifts {
            let mut order: Vec<(u64, u32)> = par_chunks(alive, |r| {
                r.map(|i| (morton(cls[i].m1.map(|v| v / cls[i].w), lo, ext, *sh), i as u32)).collect::<Vec<_>>()
            })
            .concat();
            order.sort_unstable();
            let found: Vec<Vec<(f64, u32)>> = par_chunks(alive, |r| {
                r.map(|pos| {
                    let i = order[pos].1 as usize;
                    let mut b = (f64::INFINITY, u32::MAX);
                    for d in 1..=K {
                        for q in [pos.wrapping_sub(d), pos + d] {
                            if q < alive {
                                let j = order[q].1 as usize;
                                let c = cost(&cls[i], &cls[j], ctx);
                                if c < b.0 || (c == b.0 && (j as u32) < b.1) {
                                    b = (c, j as u32);
                                }
                            }
                        }
                    }
                    (i, b)
                })
                .map(|(_, b)| b)
                .collect()
            });
            let flat: Vec<(f64, u32)> = found.concat();
            for (pos, b) in flat.into_iter().enumerate() {
                let i = order[pos].1 as usize;
                if b.0 < best[i].0 || (b.0 == best[i].0 && b.1 < best[i].1) {
                    best[i] = b;
                }
            }
        }
        // the cost is symmetric: j's best may be found only from i's side.
        let mut pairs: Vec<(f64, u32, u32)> = Vec::new();
        for i in 0..alive {
            let (c, j) = best[i];
            if j == u32::MAX || !c.is_finite() {
                continue;
            }
            if (i as u32) < j && best[j as usize].1 == i as u32 {
                pairs.push((c, i as u32, j));
            }
        }
        if pairs.is_empty() {
            eprintln!("no more pairs at {alive}");
            break;
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let need = alive - targets[ti];
        let take = need.min(pairs.len().div_ceil(2));
        let mut dead = vec![false; alive];
        for &(_, i, j) in &pairs[..take] {
            cls[i as usize] = add(&cls[i as usize], &cls[j as usize]);
            dead[j as usize] = true;
        }
        let mut k = 0;
        cls.retain(|_| {
            let keep = !dead[k];
            k += 1;
            keep
        });
        pass += 1;
        eprintln!(
            "pass {pass}: {} → {} ({} mutual pairs, merged {}), cost p50 {:.3e} last {:.3e}, {:.1}s",
            alive,
            cls.len(),
            pairs.len(),
            take,
            pairs[take / 2].0,
            pairs[take - 1].0,
            t0.elapsed().as_secs_f32()
        );
    }
}

pub fn run(scene: &Scene, args: &[String]) -> Result<()> {
    let lambda: f64 = arg(args, "--lambda").map(|s| s.parse().unwrap()).unwrap_or(128.0);
    let width: usize = arg(args, "--width").map(|s| s.parse().unwrap()).unwrap_or(1920);
    let n = scene.splats.len();
    let mut targets: Vec<usize> = match arg(args, "--targets") {
        Some(t) => t.split(',').map(|v| v.parse().unwrap()).collect(),
        None => [0.67, 0.5, 0.33, 0.25, 0.2, 0.125, 0.0625].iter().map(|f| (n as f64 * f) as usize).collect(),
    };
    targets.sort_by(|a, b| b.cmp(a));
    let cam0 = scene.camera(width, 1.0, 0.0, 0.0);
    let ctx = Ctx {
        eye_dist: cam0.distance_to(scene.center) as f64,
        center: scene.center.map(|v| v as f64),
        fx: cam0.fx as f64,
        near: cam0.near as f64 * 10.0,
        lambda,
        kind: if arg(args, "--cost") == Some("ise") { 1 } else { 0 },
        mu_n: arg(args, "--mu").map(|s| s.parse().unwrap()).unwrap_or(1.0),
    };
    let views = scene.test_views(width);
    let t0 = Instant::now();
    // References: the full cloud supersampled 2x (box filtered).
    let views2 = scene.test_views(2 * width);
    let refs: Vec<Image> = views2.iter().map(|(_, c)| render(&scene.splats, c, None).downsample(2)).collect();
    let nfull = normal_coloured(&scene.splats);
    let nrefs: Vec<Image> = views2.iter().map(|(_, c)| render(&nfull, c, None).downsample(2)).collect();
    eprintln!("reference renders {:.1}s", t0.elapsed().as_secs_f32());
    println!("## {} — greedy pairwise merge (λ = {lambda}) vs the octree LoD\n", scene.name);
    let names: Vec<String> = views.iter().map(|(n, _)| n.clone()).collect();
    println!("relMSE against the full cloud ({} splats) supersampled 2×, per view: {}\n", n, names.join(" · "));
    println!("| method | splats | % of full | {} | mean | normals: mean relMSE |\n|---|---|---|{}---|---|", names.join(" | "), "---|".repeat(names.len()));
    let row = |label: &str, cut: &[Splat]| {
        let errs: Vec<f64> = views.iter().zip(&refs).map(|((_, c), r)| render(cut, c, None).rel_mse(r)).collect();
        let mean = errs.iter().sum::<f64>() / errs.len() as f64;
        let nc = normal_coloured(cut);
        let nmean = views.iter().zip(&nrefs).map(|((_, c), r)| render(&nc, c, None).rel_mse(r)).sum::<f64>() / views.len() as f64;
        println!(
            "| {} | {} | {:.1}% | {} | {:.2e} | {:.2e} |",
            label,
            cut.len(),
            100.0 * cut.len() as f64 / n as f64,
            errs.iter().map(|e| format!("{e:.2e}")).collect::<Vec<_>>().join(" | "),
            mean,
            nmean
        );
    };
    row("full cloud (at 1×)", &scene.splats);
    // The octree cuts.
    for (level, _cell, cut) in &scene.cut_levels {
        if cut.len() * 50 < n || cut.len() >= n {
            continue;
        }
        row(&format!("octree level {level}"), cut);
    }
    // A published light cloud, when asked.
    if let Some(other) = arg(args, "--compare") {
        let o = Scene::load(other)?;
        row(&format!("published {other}"), &o.splats);
        if !targets.contains(&o.splats.len()) {
            targets.push(o.splats.len());
        }
    }
    // Equal counts as the octree cuts.
    if args.iter().any(|a| a == "--octree-counts") {
        for (_, _, cut) in &scene.cut_levels {
            if cut.len() * 50 >= n && cut.len() < n {
                targets.push(cut.len());
            }
        }
    }
    targets.sort_by(|a, b| b.cmp(a));
    targets.dedup();
    let save = arg(args, "--save").map(|s| s.to_string());
    // Optionally drop what no view sees first (as `visibility`).
    let mut input = scene.splats.clone();
    if let Some(v) = arg(args, "--prune-views") {
        let (max_t, contrib, nv) = crate::visibility_of(scene, &scene.splats, v.parse().unwrap(), 800);
        input = input.into_iter().enumerate().filter(|(i, _)| !(max_t[*i] < 0.02 && contrib[*i] < 0.05)).map(|(_, s)| s).collect();
        eprintln!("pruned over {nv} views: {} → {}", n, input.len());
        row("pruned (hidden ∧ < 0.05 px)", &input);
    }
    // Optionally thin the opaque splats of some parts first (as `thin`).
    if let Some(spec) = arg(args, "--thin-first") {
        // k,grow,part[+part...]
        let f: Vec<&str> = spec.split(',').collect();
        let (k, grow): (usize, f32) = (f[0].parse().unwrap(), f[1].parse().unwrap());
        let parts: Vec<u16> = f[2].split('+').map(|p| scene.parts.iter().position(|x| x == p).unwrap() as u16).collect();
        input = thin_opaque(&input, k, grow, &parts);
        row(&format!("+ thinned {spec}"), &input);
    }
    greedy(&input, &ctx, &targets, |t, cut| {
        row(&format!("greedy → {t}"), &cut);
        if let Some(dir) = &save {
            for v in [0usize, 3] {
                let img = render(&cut, &views[v].1, None);
                let _ = img.save_png(&format!("{dir}/{}-greedy-{t}-v{v}.png", scene.name), 0.6);
                let _ = img.diff_png(&refs[v], &format!("{dir}/{}-greedy-{t}-v{v}-diff.png", scene.name));
                let _ = refs[v].save_png(&format!("{dir}/{}-full-v{v}.png", scene.name), 0.6);
            }
        }
    });
    Ok(())
}

/// The opaque splats of `parts` thinned 1 in k along the Morton order, their
/// long axes scaled by `grow`.
pub fn thin_opaque(splats: &[Splat], k: usize, grow: f32, parts: &[u16]) -> Vec<Splat> {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for s in splats {
        for d in 0..3 {
            lo[d] = lo[d].min(s.p[d] as f64);
            hi[d] = hi[d].max(s.p[d] as f64);
        }
    }
    let ext = (0..3).map(|d| hi[d] - lo[d]).fold(0.0, f64::max) * 1.01 + 1e-9;
    let mut order: Vec<(u16, u64, usize)> = splats.iter().enumerate().map(|(i, s)| (s.part, morton(s.p.map(|v| v as f64), lo, ext, [0.0; 3]), i)).collect();
    order.sort_unstable();
    let mut out = Vec::with_capacity(splats.len());
    let mut run = 0usize;
    let mut last = u16::MAX;
    for &(part, _, i) in &order {
        if part != last {
            run = 0;
            last = part;
        }
        let s = &splats[i];
        if s.o < 0.99 || !parts.contains(&part) {
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
    out
}
