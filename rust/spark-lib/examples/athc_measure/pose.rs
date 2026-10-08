//! Visibility of a skinned cloud over its clips' poses (linear blend
//! skinning of centres and covariances, as athenea_adapter/skin.slang does
//! without the Jacobian's gradient terms): is what the bind pose hides still
//! hidden when the bird moves?

use anyhow::{anyhow, Result};
use spark_lib::athc::FLAG_LINEAR;
use spark_lib::athc_skin::{element_skin, SkinClip};
use spark_lib::athc_v3::{gunzip, read_v3, read_v3_skeleton};

use crate::arg;
use crate::cloud::*;
use crate::raster::*;

/// A 4x4 USD matrix (rows, vectors on the left) as a column-vector 3x4.
fn affine(m: &[f32]) -> [[f32; 4]; 3] {
    // p' = M^T p: row c of the result is column c of M.
    let mut a = [[0.0f32; 4]; 3];
    for c in 0..3 {
        for r in 0..4 {
            a[c][r] = m[4 * r + c];
        }
    }
    a
}

fn apply(a: &[[f32; 4]; 3], p: [f32; 3], w: f32) -> [f32; 3] {
    [0, 1, 2].map(|c| a[c][0] * p[0] + a[c][1] * p[1] + a[c][2] * p[2] + a[c][3] * w)
}

pub fn run(scene: &Scene, args: &[String]) -> Result<()> {
    let path = format!("{}/{}", scene.dir, scene.files[0]);
    let bytes = std::fs::read(&path)?;
    let file = read_v3(&bytes)?;
    let skel = read_v3_skeleton(&bytes)?.ok_or_else(|| anyhow!("no skeleton"))?;
    let inf = file.extra.skin_influences as usize;
    let gw = file.extra.skin_gradient_words as usize;
    let linear = file.header.flags & FLAG_LINEAR != 0;
    // splats and their influences, in chunk order (as Scene loads them)
    let mut skin: Vec<Vec<(u32, f32)>> = Vec::new();
    let mut check = Vec::new();
    for c in &file.chunks {
        block_splats(c, linear, 0, &mut check);
        for i in 0..c.n {
            skin.push(element_skin(&c.skin, inf, gw, i).0);
        }
    }
    assert_eq!(check.len(), scene.splats.len());
    let gb = affine(&skel.geom_bind);
    // the clips
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{}/sparrow.json", scene.dir))?)?;
    let mut clips = Vec::new();
    for c in json["clips"].as_array().unwrap() {
        let f = c["file"].as_str().unwrap();
        let raw = gunzip(&std::fs::read(format!("{}/{}", scene.dir, f))?)?;
        let (clip, joints) = SkinClip::from_atcl(&raw)?;
        assert_eq!(joints, skel.joint_count());
        clips.push(clip);
    }
    let samples: usize = arg(args, "--samples").map(|s| s.parse().unwrap()).unwrap_or(3);
    let nviews: usize = arg(args, "--views").map(|s| s.parse().unwrap()).unwrap_or(80);
    let joints = skel.joint_count();
    let pose_splats = |clip: &SkinClip, k: usize| -> Vec<Splat> {
        let per = joints * 16;
        let x = &clip.xforms[k * per..(k + 1) * per];
        let mats: Vec<[[f32; 4]; 3]> = (0..joints).map(|j| affine(&x[j * 16..j * 16 + 16])).collect();
        let parts = par_chunks(scene.splats.len(), |r| {
            r.map(|i| {
                let s = &scene.splats[i];
                let bound = apply(&gb, s.p, 1.0);
                let mut moved = [0.0f32; 3];
                let mut lin = [[0.0f32; 3]; 3];
                let mut total = 0.0;
                for &(j, w) in &skin[i] {
                    if w <= 1e-6 || j as usize >= joints {
                        continue;
                    }
                    let m = &mats[j as usize];
                    let c = apply(m, bound, 1.0);
                    for d in 0..3 {
                        moved[d] += w * c[d];
                        for e in 0..3 {
                            lin[d][e] += w * m[d][e];
                        }
                    }
                    total += w;
                }
                let mut out = s.clone();
                if total <= 1e-6 {
                    out.p = bound;
                    return out;
                }
                out.p = moved;
                // chain = blended x linear(geomBind); the axes carried by it
                let axes = s.axes();
                let carried: Vec<[f32; 3]> = (0..3)
                    .map(|k| {
                        let a = apply(&gb, axes[k], 0.0);
                        let v = [0, 1, 2].map(|d| lin[d][0] * a[0] + lin[d][1] * a[1] + lin[d][2] * a[2]);
                        v.map(|x| x * s.s[k])
                    })
                    .collect();
                let mut cov = [0.0f64; 6];
                for v in &carried {
                    let v = v.map(|x| x as f64);
                    cov[0] += v[0] * v[0];
                    cov[1] += v[1] * v[1];
                    cov[2] += v[2] * v[2];
                    cov[3] += v[0] * v[1];
                    cov[4] += v[0] * v[2];
                    cov[5] += v[1] * v[2];
                }
                let (q, sc) = crate::merge::orient(&cov);
                out.q = q;
                out.s = sc;
                out
            })
            .collect::<Vec<Splat>>()
        });
        parts.concat()
    };
    let n = scene.splats.len();
    let mut max_t = vec![0.0f32; n];
    let mut contrib = vec![0.0f32; n];
    let (bt, bc, _) = crate::visibility_of(scene, &scene.splats, nviews, 640);
    let bind_hidden: Vec<bool> = (0..n).map(|i| bt[i] < 0.02 && bc[i] < 0.05).collect();
    let mut poses = 0;
    let mut held_out = Vec::new();
    for clip in &clips {
        let m = clip.times.len();
        for s in 0..samples {
            let k = (s * (m - 1)) / samples.max(1);
            let posed = pose_splats(clip, k);
            let (t, c, _) = crate::visibility_of(scene, &posed, nviews, 640);
            for i in 0..n {
                max_t[i] = max_t[i].max(t[i]);
                contrib[i] += c[i];
            }
            poses += 1;
        }
        // a held-out pose: between the samples
        held_out.push((clip.name.clone(), pose_splats(clip, (m - 1) / (2 * samples).max(1) + 1)));
        eprintln!("{}: {} poses so far", clip.name, poses);
    }
    let all_hidden: Vec<bool> = (0..n).map(|i| bind_hidden[i] && max_t[i] < 0.02 && contrib[i] < 0.05).collect();
    let nb = bind_hidden.iter().filter(|b| **b).count();
    let na = all_hidden.iter().filter(|b| **b).count();
    println!("## {} — visibility over {} poses ({} clips × {} samples) + bind pose, {} views each at 640 px\n", scene.name, poses, clips.len(), samples, nviews);
    println!("hidden in the bind pose: {} ({:.1}%); hidden in the bind pose AND every sampled pose: {} ({:.1}%)\n", nb, 100.0 * nb as f64 / n as f64, na, 100.0 * na as f64 / n as f64);
    println!("| held-out pose | relMSE drop bind-hidden | relMSE drop all-pose-hidden | Δcoverage (bind / all) |\n|---|---|---|---|");
    let cam = scene.camera(1920, 1.0, 0.0, 0.0);
    let cam2 = scene.camera(1920, 0.4, 100.0, 30.0);
    for (name, posed) in &held_out {
        for (vn, c) in [("default", &cam), ("close", &cam2)] {
            let full = render(posed, c, None);
            if let Some(dir) = arg(args, "--save") {
                full.save_png(&format!("{dir}/pose-{name}-{vn}.png"), 0.6)?;
            }
            let a: Vec<Splat> = posed.iter().zip(&bind_hidden).filter(|(_, h)| !**h).map(|(s, _)| s.clone()).collect();
            let b: Vec<Splat> = posed.iter().zip(&all_hidden).filter(|(_, h)| !**h).map(|(s, _)| s.clone()).collect();
            let ia = render(&a, c, None);
            let ib = render(&b, c, None);
            println!("| {} ({}) | {:.2e} | {:.2e} | {:.2e} / {:.2e} |", name, vn, ia.rel_mse(&full), ib.rel_mse(&full), ia.alpha_diff(&full), ib.alpha_diff(&full));
        }
    }
    Ok(())
}
