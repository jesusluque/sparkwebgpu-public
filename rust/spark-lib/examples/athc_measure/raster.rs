//! A CPU splat rasterizer with Spark's draw rules, in float.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use crate::cloud::*;

pub fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(12)
}

/// Run `f(range)` over 0..n in parallel chunks, collecting the results in order.
pub fn par_chunks<T: Send>(n: usize, f: impl Fn(std::ops::Range<usize>) -> T + Sync) -> Vec<T> {
    let t = threads();
    let step = n.div_ceil(t).max(1);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..t).map(|k| (k * step).min(n)..((k + 1) * step).min(n)).map(|r| s.spawn(|| f(r))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

pub struct AtomicF32s(Vec<AtomicU32>);
impl AtomicF32s {
    pub fn new(n: usize) -> Self {
        Self((0..n).map(|_| AtomicU32::new(0)).collect())
    }
    pub fn max(&self, i: usize, v: f32) {
        // non-negative floats order like their bits
        self.0[i].fetch_max(v.max(0.0).to_bits(), Ordering::Relaxed);
    }
    pub fn add(&self, i: usize, v: f32) {
        let mut cur = self.0[i].load(Ordering::Relaxed);
        loop {
            let next = (f32::from_bits(cur) + v).to_bits();
            match self.0[i].compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(c) => cur = c,
            }
        }
    }
    pub fn into_vec(self) -> Vec<f32> {
        self.0.into_iter().map(|a| f32::from_bits(a.into_inner())).collect()
    }
}

pub struct Track<'a> {
    pub max_t: &'a AtomicF32s,
    pub contrib: &'a AtomicF32s,
}

#[derive(Clone, Debug)]
pub struct Camera {
    pub eye: [f32; 3],
    pub right: [f32; 3],
    pub upc: [f32; 3],
    pub fwd: [f32; 3],
    pub fx: f32,
    pub cx: f32,
    pub cy: f32,
    pub w: usize,
    pub h: usize,
    pub near: f32,
}

impl Camera {
    pub fn look(eye: [f32; 3], target: [f32; 3], up: [f32; 3], hfov: f32, w: usize, aspect: f32, near: f32) -> Self {
        let fwd = norm(sub(target, eye));
        let mut right = cross(fwd, up);
        if dot(right, right) < 1e-8 {
            right = cross(fwd, [1.0, 0.0, 0.0]);
        }
        let right = norm(right);
        let upc = cross(right, fwd);
        let h = (w as f32 / aspect).round() as usize;
        let fx = (w as f32 / 2.0) / (hfov / 2.0).tan();
        Self { eye, right, upc, fwd, fx, cx: w as f32 / 2.0, cy: h as f32 / 2.0, w, h, near }
    }
    pub fn distance_to(&self, c: [f32; 3]) -> f32 {
        dist(self.eye, c)
    }
    pub fn depth(&self, p: [f32; 3]) -> f32 {
        dot(sub(p, self.eye), self.fwd)
    }
    pub fn sigma_px(&self, s: &Splat) -> Option<f32> {
        let z = self.depth(s.p);
        if z < self.near {
            return None;
        }
        Some(s.long() * self.fx / z)
    }
}

impl Scene {
    /// The default camera pulled back by `scale` (az = el = 0), or an orbit
    /// at `scale` x the default distance, azimuth `az` from the default eye
    /// and elevation `el` above the plane normal to up (degrees).
    pub fn camera(&self, width: usize, scale: f32, az: f32, el: f32) -> Camera {
        let c = &self.cam;
        let d = sub(c.eye, c.target);
        let near = 0.002 * dot(d, d).sqrt();
        let eye = if az == 0.0 && el == 0.0 {
            [c.target[0] + d[0] * scale, c.target[1] + d[1] * scale, c.target[2] + d[2] * scale]
        } else {
            let dd = dot(d, d).sqrt() * scale;
            let up = norm(c.up);
            let h = sub(d, up.map(|u| u * dot(d, up)));
            let h0 = norm(h);
            let h1 = cross(up, h0);
            let (a, e) = (az.to_radians(), el.to_radians());
            let dir = [0, 1, 2].map(|k| e.cos() * (a.cos() * h0[k] + a.sin() * h1[k]) + e.sin() * up[k]);
            [0, 1, 2].map(|k| c.target[k] + dd * dir[k])
        };
        Camera::look(eye, c.target, c.up, c.hfov, width, c.aspect, near)
    }

    /// Views none of the visibility orbit uses, for held-out tests.
    pub fn test_views(&self, width: usize) -> Vec<(String, Camera)> {
        let mut v = vec![("default".to_string(), self.camera(width, 1.0, 0.0, 0.0))];
        let els: [f32; 3] = if self.views_below { [25.0, -25.0, 55.0] } else { [25.0, 12.0, 55.0] };
        for (k, (az, el)) in [(75.0f32, els[0]), (165.0, els[1]), (255.0, els[2])].iter().enumerate() {
            v.push((format!("orbit{} az{} el{}", k + 1, az, el), self.camera(width, 1.0, *az, *el)));
        }
        v.push(("close ×0.4 az100 el30".to_string(), self.camera(width, 0.4, 100.0, 30.0)));
        v.push(("far ×3".to_string(), self.camera(width, 3.0, 0.0, 0.0)));
        v
    }
}

pub struct Image {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<[f32; 3]>,
    pub t: Vec<f32>,
}

pub const BACKGROUND: f32 = 0.18;

impl Image {
    pub fn composite(&self, i: usize) -> [f32; 3] {
        let t = self.t[i];
        self.rgb[i].map(|v| v + t * BACKGROUND)
    }
    /// sum |a - b|^2 / sum |b|^2 over the composited pixels where either
    /// has content.
    pub fn rel_mse(&self, reference: &Image) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..self.rgb.len() {
            if self.t[i] >= 1.0 && reference.t[i] >= 1.0 {
                continue;
            }
            let a = self.composite(i);
            let b = reference.composite(i);
            for k in 0..3 {
                num += ((a[k] - b[k]) as f64).powi(2);
                den += (b[k] as f64).powi(2);
            }
        }
        num / den.max(1e-30)
    }
    /// Mean |ΔT| over pixels where either has content.
    pub fn alpha_diff(&self, reference: &Image) -> f64 {
        let (mut s, mut n) = (0.0f64, 0usize);
        for i in 0..self.t.len() {
            if self.t[i] >= 1.0 && reference.t[i] >= 1.0 {
                continue;
            }
            s += (self.t[i] - reference.t[i]).abs() as f64;
            n += 1;
        }
        s / n.max(1) as f64
    }
    /// Box-filtered down by k (a supersampled reference).
    pub fn downsample(&self, k: usize) -> Image {
        let (w, h) = (self.w / k, self.h / k);
        let mut out = Image { w, h, rgb: vec![[0.0; 3]; w * h], t: vec![0.0; w * h] };
        let inv = 1.0 / (k * k) as f32;
        for y in 0..h {
            for x in 0..w {
                let o = y * w + x;
                for dy in 0..k {
                    for dx in 0..k {
                        let i = (y * k + dy) * self.w + x * k + dx;
                        for c in 0..3 {
                            out.rgb[o][c] += self.rgb[i][c] * inv;
                        }
                        out.t[o] += self.t[i] * inv;
                    }
                }
            }
        }
        out
    }
    /// |difference| ×20 as a grey image, for looking only.
    pub fn diff_png(&self, other: &Image, path: &str) -> anyhow::Result<()> {
        let mut buf = Vec::with_capacity(self.w * self.h);
        for i in 0..self.rgb.len() {
            let (a, b) = (self.composite(i), other.composite(i));
            let d = (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f32::max);
            buf.push(((d * 20.0).min(1.0) * 255.0) as u8);
        }
        image::save_buffer(path, &buf, self.w as u32, self.h as u32, image::ColorType::L8)?;
        Ok(())
    }
    /// For looking only (tone mapped, 8-bit): never compared.
    pub fn save_png(&self, path: &str, exposure: f32) -> anyhow::Result<()> {
        let mut buf = Vec::with_capacity(self.w * self.h * 3);
        for i in 0..self.rgb.len() {
            for v in self.composite(i) {
                let x = v * exposure / (1.0 + v * exposure);
                let s = if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
                buf.push((s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
        }
        image::save_buffer(path, &buf, self.w as u32, self.h as u32, image::ColorType::Rgb8)?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Proj {
    u: f32,
    v: f32,
    conic: [f32; 3],
    a: f32,
    /// LoD exponent (opacity past 1), or 0.
    e: f32,
    cut2: f32,
    c: [f32; 3],
    idx: u32,
}

const TILE: usize = 16;
const BLUR: f32 = 0.3;
/// Spark's default minAlpha (WgpuSplatRenderer).
pub const MIN_ALPHA: f32 = 0.5 / 255.0;

fn project(s: &Splat, cam: &Camera) -> Option<(Proj, f32, f32)> {
    let d = sub(s.p, cam.eye);
    let z = dot(d, cam.fwd);
    if z < cam.near {
        return None;
    }
    let x = dot(d, cam.right);
    let y = dot(d, cam.upc);
    let f = cam.fx;
    let u = cam.cx + f * x / z;
    let v = cam.cy - f * y / z;
    // rows of the Jacobian in world space
    let r0 = [0, 1, 2].map(|k| f / z * cam.right[k] - f * x / (z * z) * cam.fwd[k]);
    let r1 = [0, 1, 2].map(|k| -(f / z * cam.upc[k] - f * y / (z * z) * cam.fwd[k]));
    let axes = s.axes();
    let (mut a, mut b, mut c) = (0.0f32, 0.0f32, 0.0f32);
    for k in 0..3 {
        let t0 = dot(r0, axes[k]) * s.s[k];
        let t1 = dot(r1, axes[k]) * s.s[k];
        a += t0 * t0;
        b += t0 * t1;
        c += t1 * t1;
    }
    let det0 = (a * c - b * b).max(0.0);
    a += BLUR;
    c += BLUR;
    let det = a * c - b * b;
    if det <= 0.0 {
        return None;
    }
    let (alpha, e, maxstd) = if s.o <= 1.0 {
        let al = s.o * (det0 / det).sqrt();
        if al < MIN_ALPHA {
            return None;
        }
        (al, 0.0, 8f32.sqrt().min((2.0 * (al / MIN_ALPHA).ln()).max(0.0).sqrt()))
    } else {
        let dd = (1.0 + std::f32::consts::E * s.o.ln()).sqrt().min(5.0);
        let e = ((dd * dd - 1.0) / std::f32::consts::E).exp();
        (1.0, e, 8f32.sqrt() + 0.7 * (dd - 1.0))
    };
    let mid = 0.5 * (a + c);
    let lmax = mid + (mid * mid - det).max(0.0).sqrt();
    let r = maxstd * lmax.sqrt();
    let conic = [c / det, -b / det, a / det];
    Some((Proj { u, v, conic, a: alpha, e, cut2: maxstd * maxstd, c: s.c, idx: 0 }, r, z))
}

/// Render front to back; with `track`, each splat's largest transmittance at
/// its centre pixel and its summed contribution (pixels x alpha x T).
pub fn render(splats: &[Splat], cam: &Camera, track: Option<&Track>) -> Image {
    let (w, h) = (cam.w, cam.h);
    let (tw, th) = (w.div_ceil(TILE), h.div_ceil(TILE));
    let ntiles = tw * th;
    // Project and list the tiles each splat touches.
    let parts = par_chunks(splats.len(), |r| {
        let mut projs = Vec::new();
        let mut entries: Vec<(u32, f32, u32)> = Vec::new();
        for i in r {
            let Some((mut p, rad, z)) = project(&splats[i], cam) else { continue };
            p.idx = i as u32;
            let x0 = ((p.u - rad).floor().max(0.0) as usize) / TILE;
            let x1 = ((p.u + rad).ceil().min(w as f32 - 1.0).max(-1.0) as isize).max(-1);
            let y0 = ((p.v - rad).floor().max(0.0) as usize) / TILE;
            let y1 = ((p.v + rad).ceil().min(h as f32 - 1.0).max(-1.0) as isize).max(-1);
            if x1 < 0 || y1 < 0 || p.u + rad < 0.0 || p.v + rad < 0.0 || p.u - rad > w as f32 || p.v - rad > h as f32 {
                continue;
            }
            let (x1, y1) = (x1 as usize / TILE, y1 as usize / TILE);
            projs.push(p);
            let pi = projs.len() as u32 - 1;
            for ty in y0..=y1 {
                for tx in x0..=x1 {
                    entries.push(((ty * tw + tx) as u32, z, pi));
                }
            }
        }
        (projs, entries)
    });
    // Flatten: global proj index.
    let mut projs = Vec::new();
    let mut counts = vec![0usize; ntiles + 1];
    let mut base = Vec::new();
    for (p, e) in &parts {
        base.push(projs.len() as u32);
        projs.extend_from_slice(p);
        for x in e {
            counts[x.0 as usize + 1] += 1;
        }
    }
    for t in 0..ntiles {
        counts[t + 1] += counts[t];
    }
    let total = counts[ntiles];
    let mut list: Vec<(f32, u32)> = vec![(0.0, 0); total];
    let mut fill = counts.clone();
    for ((_, e), b) in parts.iter().zip(&base) {
        for x in e {
            let t = x.0 as usize;
            list[fill[t]] = (x.1, x.2 + b);
            fill[t] += 1;
        }
    }
    drop(parts);
    // Sort each tile by depth, in parallel.
    {
        let mut slices: Vec<&mut [(f32, u32)]> = Vec::with_capacity(ntiles);
        let mut rest: &mut [(f32, u32)] = &mut list;
        for t in 0..ntiles {
            let (a, b) = rest.split_at_mut(counts[t + 1] - counts[t]);
            slices.push(a);
            rest = b;
        }
        let next = AtomicUsize::new(0);
        let slots: Vec<std::sync::Mutex<Option<&mut [(f32, u32)]>>> = slices.into_iter().map(|s| std::sync::Mutex::new(Some(s))).collect();
        std::thread::scope(|s| {
            for _ in 0..threads() {
                s.spawn(|| loop {
                    let t = next.fetch_add(1, Ordering::Relaxed);
                    if t >= ntiles {
                        break;
                    }
                    let sl = slots[t].lock().unwrap().take().unwrap();
                    sl.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                });
            }
        });
    }
    // Raster each tile.
    let next = AtomicUsize::new(0);
    let tiles: Vec<(usize, Vec<[f32; 3]>, Vec<f32>)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads())
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let t = next.fetch_add(1, Ordering::Relaxed);
                        if t >= ntiles {
                            break;
                        }
                        let (tx, ty) = (t % tw, t / tw);
                        let entries = &list[counts[t]..counts[t + 1]];
                        let mut rgb = vec![[0.0f32; 3]; TILE * TILE];
                        let mut tt = vec![1.0f32; TILE * TILE];
                        for py in 0..TILE {
                            let y = ty * TILE + py;
                            if y >= h {
                                break;
                            }
                            for px in 0..TILE {
                                let x = tx * TILE + px;
                                if x >= w {
                                    break;
                                }
                                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                                let mut tr = 1.0f32;
                                let mut col = [0.0f32; 3];
                                for &(_, pi) in entries {
                                    let p = &projs[pi as usize];
                                    if let Some(tk) = track {
                                        if p.u.floor() as isize == x as isize && p.v.floor() as isize == y as isize {
                                            tk.max_t.max(p.idx as usize, tr);
                                        }
                                    }
                                    let (dx, dy) = (fx - p.u, fy - p.v);
                                    let power = 0.5 * (p.conic[0] * dx * dx + 2.0 * p.conic[1] * dx * dy + p.conic[2] * dy * dy);
                                    if 2.0 * power > p.cut2 {
                                        continue;
                                    }
                                    let g = (-power).exp();
                                    let alpha = if p.e > 0.0 { 1.0 - (1.0 - g.min(0.9999)).powf(p.e) } else { p.a * g }.min(0.99);
                                    if alpha < MIN_ALPHA {
                                        continue;
                                    }
                                    let k = tr * alpha;
                                    for d in 0..3 {
                                        col[d] += k * p.c[d];
                                    }
                                    if let Some(tk) = track {
                                        tk.contrib.add(p.idx as usize, k);
                                    }
                                    tr *= 1.0 - alpha;
                                    if tr < 1e-4 {
                                        if track.is_none() {
                                            break;
                                        }
                                        // keep going only to record the centres behind
                                        if tr < 1e-6 {
                                            break;
                                        }
                                    }
                                }
                                rgb[py * TILE + px] = col;
                                tt[py * TILE + px] = tr;
                            }
                        }
                        out.push((t, rgb, tt));
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let mut img = Image { w, h, rgb: vec![[0.0; 3]; w * h], t: vec![1.0; w * h] };
    for (t, rgb, tt) in tiles {
        let (tx, ty) = (t % tw, t / tw);
        for py in 0..TILE {
            for px in 0..TILE {
                let (x, y) = (tx * TILE + px, ty * TILE + py);
                if x < w && y < h {
                    img.rgb[y * w + x] = rgb[py * TILE + px];
                    img.t[y * w + x] = tt[py * TILE + px];
                }
            }
        }
    }
    img
}
