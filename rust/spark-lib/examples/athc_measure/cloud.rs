//! The published assets as plain splats, their cameras and their LoD levels.

use anyhow::{bail, Context, Result};
use spark_lib::athc::{self, AthcBlock, AthcFile};

#[derive(Clone, Debug, Default)]
pub struct Splat {
    pub p: [f32; 3],
    /// Opacity; above 1 a LoD opacity (drawn as 1 - (1 - g)^o).
    pub o: f32,
    /// x y z w
    pub q: [f32; 4],
    pub s: [f32; 3],
    /// Linear base colour.
    pub c: [f32; 3],
    /// Stored normal or 0.
    pub n: [f32; 3],
    pub part: u16,
    /// What a merge may not mix: part, glass, thin, Schlick, mirror.
    pub key: u32,
}

impl Splat {
    pub fn long(&self) -> f32 {
        self.s[0].max(self.s[1]).max(self.s[2])
    }
    pub fn has_normal(&self) -> bool {
        self.n != [0.0; 3]
    }
    /// Rotation matrix columns (the splat's axes).
    pub fn axes(&self) -> [[f32; 3]; 3] {
        let [x, y, z, w] = self.q;
        let n = (x * x + y * y + z * z + w * w).sqrt().max(1e-20);
        let (x, y, z, w) = (x / n, y / n, z / n, w / n);
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
            [2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)],
            [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    }
    /// Covariance xx yy zz xy xz yz (f64).
    pub fn cov(&self) -> [f64; 6] {
        let a = self.axes();
        let mut c = [0.0f64; 6];
        for k in 0..3 {
            let v = [a[k][0] as f64, a[k][1] as f64, a[k][2] as f64];
            let s2 = (self.s[k] as f64).powi(2);
            c[0] += s2 * v[0] * v[0];
            c[1] += s2 * v[1] * v[1];
            c[2] += s2 * v[2] * v[2];
            c[3] += s2 * v[0] * v[1];
            c[4] += s2 * v[0] * v[2];
            c[5] += s2 * v[1] * v[2];
        }
        c
    }
    /// The two longest axes, multiplied (athenea's merge weight / opacity).
    pub fn area2(&self) -> f32 {
        let mut s = self.s;
        s.sort_by(|a, b| b.total_cmp(a));
        s[0] * s[1]
    }
}

/// The same splats coloured by their normal (n + 1) / 2: a proxy for what
/// relighting reads.
pub fn normal_coloured(s: &[Splat]) -> Vec<Splat> {
    s.iter()
        .map(|x| {
            let mut x = x.clone();
            x.c = if x.has_normal() { x.n.map(|v| 0.5 + 0.5 * v) } else { [0.5; 3] };
            x
        })
        .collect()
}

pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = sub(a, b);
    dot(d, d).sqrt()
}
pub fn norm(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt().max(1e-30);
    [a[0] / l, a[1] / l, a[2] / l]
}
pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

pub fn block_splats(b: &AthcBlock, linear: bool, part: u16, out: &mut Vec<Splat>) {
    let pbr_words = b.pbr.len().checked_div(b.n).unwrap_or(0);
    for i in 0..b.n {
        let p = &b.positions[i * 4..i * 4 + 4];
        let w = &b.shape[i * 4..i * 4 + 4];
        let q = athc::decode_quaternion(w[0]);
        let s = [athc::low_half(w[1]).exp(), athc::high_half(w[1]).exp(), athc::low_half(w[2]).exp()];
        let mut c = [athc::high_half(w[2]), athc::low_half(w[3]), athc::high_half(w[3])];
        if !linear {
            c = c.map(|v| srgb_to_linear(v.clamp(0.0, 1.0)));
        }
        let c = c.map(|v| v.max(0.0));
        let n = if b.normals.is_empty() { [0.0; 3] } else { athc::unpack_normal(b.normals[i]) };
        let mut key = (part as u32) << 8;
        if pbr_words > 0 {
            let w = b.pbr[i * pbr_words];
            let trans = ((w >> 16) & 255) > 127;
            let thin = (w >> 24) & 1 == 1;
            let schlick = (w >> 25) & 1 == 1;
            let mirror = ((w >> 8) & 255) < 26;
            key |= trans as u32 | (thin as u32) << 1 | (schlick as u32) << 2 | (mirror as u32) << 3;
        }
        out.push(Splat { p: [p[0], p[1], p[2]], o: p[3], q, s, c, n, part, key });
    }
}

pub fn read_athc(path: &str) -> Result<AthcFile> {
    let bytes = std::fs::read(path).with_context(|| path.to_string())?;
    if bytes.len() >= 4 && &bytes[0..4] == b"ATH3" {
        spark_lib::athc_v3::read_v3(&bytes)
    } else {
        AthcFile::read(&bytes)
    }
}

pub struct LevelRow {
    pub level: u32,
    pub cell: f32,
    pub groups: usize,
    pub splats_per_group: f32,
    pub cov_p50: f32,
    pub cov_p90: f32,
    pub colour_sd: f32,
    pub colour_sd_p90: f32,
    pub normal_spread: f32,
    pub crease30: f32,
    pub crease60: f32,
    /// weight share of groups mixing materials (keys)
    pub mixed: f32,
}

pub struct LevelData {
    pub rows: Vec<LevelRow>,
    pub extent: f32,
    /// Each level (number, its widened groups as splats).
    pub cuts: Vec<(u32, Vec<Splat>)>,
}

/// Group statistics of every level from its member splats.
fn level_data(file: &AthcFile, part: u16) -> Result<LevelData> {
    let linear = file.header.flags & athc::FLAG_LINEAR != 0;
    let mut splats = Vec::new();
    for c in &file.chunks {
        block_splats(c, linear, part, &mut splats);
    }
    let lf = file.levels.last().unwrap().0;
    let finest_codes = &file.levels.last().unwrap().1.tail;
    // finest group of each splat
    let count = splats.len();
    let mut group_of = vec![0u32; count];
    let mut g = 0usize;
    for (i, slot) in group_of.iter_mut().enumerate() {
        while g + 1 < file.starts.len() && file.starts[g + 1] as usize <= i {
            g += 1;
        }
        *slot = g as u32;
    }
    let ratios = athc::coverage_ratios(file);
    let mut rows = Vec::new();
    for (li, (level, block)) in file.levels.iter().enumerate() {
        let shift = 3 * (lf - level);
        let codes = &block.tail;
        let idx: Vec<u32> = (0..count)
            .map(|i| {
                let code = finest_codes[group_of[i] as usize] >> shift;
                codes.binary_search(&code).unwrap_or(0) as u32
            })
            .collect();
        let ng = block.n;
        let mut w = vec![0.0f64; ng];
        let mut cs = vec![[0.0f64; 3]; ng];
        let mut cs2 = vec![0.0f64; ng];
        let mut ns = vec![[0.0f64; 3]; ng];
        let mut members = vec![0usize; ng];
        for (i, sp) in splats.iter().enumerate() {
            let k = idx[i] as usize;
            let wi = (sp.o.min(1.0) * sp.area2()) as f64;
            w[k] += wi;
            members[k] += 1;
            for d in 0..3 {
                cs[k][d] += wi * sp.c[d] as f64;
                ns[k][d] += wi * sp.n[d] as f64;
            }
            cs2[k] += wi * dot(sp.c, sp.c) as f64;
        }
        let mut sd = Vec::with_capacity(ng);
        let mut spread_sum = 0.0;
        let mut sd_sum = 0.0;
        let mut wsum = 0.0;
        let mut mean_n = vec![[0.0f32; 3]; ng];
        for k in 0..ng {
            if w[k] <= 0.0 {
                sd.push((0.0f32, 0.0f64));
                continue;
            }
            let m = cs[k].map(|v| v / w[k]);
            let var = (cs2[k] / w[k] - (m[0] * m[0] + m[1] * m[1] + m[2] * m[2])).max(0.0);
            let s = (var / 3.0).sqrt() as f32;
            sd.push((s, w[k]));
            let nl = (ns[k][0].powi(2) + ns[k][1].powi(2) + ns[k][2].powi(2)).sqrt();
            let spread = 1.0 - nl / w[k];
            spread_sum += spread * w[k];
            sd_sum += s as f64 * w[k];
            wsum += w[k];
            mean_n[k] = norm([ns[k][0] as f32, ns[k][1] as f32, ns[k][2] as f32]);
        }
        let mut first_key = vec![u32::MAX; ng];
        let mut mixed = vec![false; ng];
        for (i, sp) in splats.iter().enumerate() {
            let k = idx[i] as usize;
            if first_key[k] == u32::MAX {
                first_key[k] = sp.key;
            } else if first_key[k] != sp.key {
                mixed[k] = true;
            }
        }
        let wmixed: f64 = (0..ng).filter(|&k| mixed[k]).map(|k| w[k]).sum();
        let mut c30 = vec![false; ng];
        let mut c60 = vec![false; ng];
        for (i, sp) in splats.iter().enumerate() {
            if !sp.has_normal() {
                continue;
            }
            let k = idx[i] as usize;
            let d = dot(sp.n, mean_n[k]);
            if d < 0.866 {
                c30[k] = true;
            }
            if d < 0.5 {
                c60[k] = true;
            }
        }
        let wc30: f64 = (0..ng).filter(|&k| c30[k]).map(|k| w[k]).sum();
        let wc60: f64 = (0..ng).filter(|&k| c60[k]).map(|k| w[k]).sum();
        let mut sds: Vec<f32> = sd.iter().map(|x| x.0).collect();
        sds.sort_by(|a, b| a.total_cmp(b));
        let mut r = ratios[li].clone();
        r.sort_by(|a, b| a.total_cmp(b));
        rows.push(LevelRow {
            level: *level,
            cell: file.header.extent / (1u64 << *level) as f32,
            groups: ng,
            splats_per_group: count as f32 / ng as f32,
            cov_p50: r[r.len() / 2],
            cov_p90: r[r.len() * 9 / 10],
            colour_sd: (sd_sum / wsum.max(1e-30)) as f32,
            colour_sd_p90: sds[sds.len() * 9 / 10],
            normal_spread: (spread_sum / wsum.max(1e-30)) as f32,
            crease30: (wc30 / wsum.max(1e-30)) as f32,
            crease60: (wc60 / wsum.max(1e-30)) as f32,
            mixed: (wmixed / wsum.max(1e-30)) as f32,
        });
    }
    // The levels as Spark decodes them: uncapped, widened, oriented.
    let mut f = file.clone();
    athc::uncap_levels(&mut f);
    let mut cuts = Vec::new();
    for (level, block) in &f.levels {
        let mut b = block.clone();
        athc::widen_merged(&mut b, f.header.extent / (1u64 << *level) as f32);
        let mut s = Vec::new();
        block_splats(&b, linear, part, &mut s);
        cuts.push((*level, s));
    }
    Ok(LevelData { rows, extent: file.header.extent, cuts })
}

pub struct CamSpec {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    /// Horizontal field of view, radians.
    pub hfov: f32,
    pub aspect: f32,
}

pub struct Scene {
    pub name: String,
    pub dir: String,
    pub files: Vec<String>,
    pub parts: Vec<String>,
    pub splats: Vec<Splat>,
    pub center: [f32; 3],
    pub radius: f32,
    pub unit: f32,
    pub views_below: bool,
    pub cam: CamSpec,
    pub level_data: Vec<Option<LevelData>>,
    /// Whole-cloud cuts: (level of the largest part, its cell, splats).
    pub cut_levels: Vec<(u32, f32, Vec<Splat>)>,
}

const R2: &str = "/Users/muriel/luc/sparkwebGPU/publish-r2/sparkwebgpu";

fn yup_to_zup(v: [f32; 3]) -> [f32; 3] {
    [v[0], -v[2], v[1]]
}

impl Scene {
    pub fn load(asset: &str) -> Result<Self> {
        let corvette_cam = CamSpec {
            eye: yup_to_zup([4.378, 1.676, 5.854]),
            target: yup_to_zup([0.0, 0.553, 0.252]),
            up: [0.0, 0.0, 1.0],
            hfov: 39.6f32.to_radians(),
            aspect: 16.0 / 9.0,
        };
        let corvette_parts = ["paint", "body", "windshield", "tinted", "headlights", "trim"];
        let (dir, files, cam, below): (String, Vec<(String, String)>, CamSpec, bool) = match asset {
            "corvette-hd" | "corvette-light" => {
                let d = if asset == "corvette-hd" { "corvette-v2-hd" } else { "corvette-v5-light" };
                (
                    format!("{R2}/{d}"),
                    corvette_parts.iter().map(|p| (p.to_string(), format!("{p}-t16-gz.athc"))).collect(),
                    corvette_cam,
                    false,
                )
            }
            "corvette-lights" | "corvette-lights-hd" => {
                let d = if asset == "corvette-lights" { "corvette-lights-light" } else { "corvette-lights-hd" };
                let dir = format!("{R2}/{d}");
                let mut files = Vec::new();
                for e in std::fs::read_dir(&dir)? {
                    let name = e?.file_name().to_string_lossy().to_string();
                    if name.ends_with(".athc") && !name.starts_with("catcher") && !name.starts_with("lamp-floor") {
                        files.push((name.split('-').next().unwrap().to_string(), name));
                    }
                }
                files.sort();
                (dir, files, corvette_cam, false)
            }
            "sparrow" | "sparrow-mobile" => {
                let f = if asset == "sparrow" { "sparrow-a490b6cbcc.athc" } else { "sparrow-mobile-6ca956bf38.athc" };
                let v = 39.59775f32.to_radians();
                let aspect = 16.0 / 9.0;
                (
                    format!("{R2}/sparrow"),
                    vec![("bird".into(), f.into())],
                    CamSpec {
                        eye: [0.38, -0.48, 0.2],
                        target: [0.0, 0.02, 0.03],
                        up: [0.0, 0.0, 1.0],
                        hfov: 2.0 * ((v / 2.0).tan() * aspect).atan(),
                        aspect,
                    },
                    true,
                )
            }
            "pawn" => (
                format!("{R2}/pawn"),
                vec![("body".into(), "body-t16-gz.athc".into()), ("top".into(), "top-t16-gz.athc".into())],
                CamSpec { eye: [0.0, 0.048, 0.18], target: [0.0, 0.048, 0.0], up: [0.0, 1.0, 0.0], hfov: 39.6f32.to_radians(), aspect: 1.0 },
                true,
            ),
            _ => bail!("unknown asset {asset}"),
        };
        let mut splats = Vec::new();
        let mut parts = Vec::new();
        let mut names = Vec::new();
        let mut level_data = Vec::new();
        for (pi, (part, file)) in files.iter().enumerate() {
            let f = read_athc(&format!("{dir}/{file}"))?;
            let linear = f.header.flags & athc::FLAG_LINEAR != 0;
            for c in &f.chunks {
                block_splats(c, linear, pi as u16, &mut splats);
            }
            // Only a cloud whose splats are the baked ones has meaningful levels
            // (a cut cloud's splats are merged groups already).
            let merged = f.chunks.iter().any(|c| c.positions.chunks_exact(4).any(|p| p[3] > 1.0));
            level_data.push(if f.levels.len() > 1 && !merged && !asset.contains("light") && asset != "sparrow-mobile" {
                Some(level_data_of(&f, pi as u16)?)
            } else {
                None
            });
            parts.push(part.clone());
            names.push(file.clone());
        }
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for s in &splats {
            for d in 0..3 {
                lo[d] = lo[d].min(s.p[d]);
                hi[d] = hi[d].max(s.p[d]);
            }
        }
        let center = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
        let radius = dist(lo, hi) / 2.0;
        eprintln!("bounds {:?} .. {:?}", lo, hi);
        // Whole-cloud cuts by the largest part's levels.
        let mut cut_levels = Vec::new();
        if let Some((big, _)) = level_data.iter().enumerate().filter_map(|(i, l)| l.as_ref().map(|l| (i, l))).max_by_key(|(_, l)| l.rows.last().map(|r| r.groups).unwrap_or(0)) {
            let bl = level_data[big].as_ref().unwrap();
            for (level, _) in &bl.cuts {
                let cell = bl.extent / (1u64 << *level) as f32;
                let mut all = Vec::new();
                for (pi, l) in level_data.iter().enumerate() {
                    match l {
                        Some(l) => {
                            // the level whose cell is nearest in log
                            let best = l
                                .cuts
                                .iter()
                                .min_by(|a, b| {
                                    let ca = (l.extent / (1u64 << a.0) as f32 / cell).ln().abs();
                                    let cb = (l.extent / (1u64 << b.0) as f32 / cell).ln().abs();
                                    ca.total_cmp(&cb)
                                })
                                .unwrap();
                            all.extend(best.1.iter().cloned());
                        }
                        None => all.extend(splats.iter().filter(|s| s.part == pi as u16).cloned()),
                    }
                }
                cut_levels.push((*level, cell, all));
            }
        }
        Ok(Self {
            name: asset.to_string(),
            dir: dir.clone(),
            files: names,
            parts,
            splats,
            center,
            radius,
            unit: 1.0,
            views_below: below,
            cam,
            level_data,
            cut_levels,
        })
    }
}

fn level_data_of(f: &AthcFile, part: u16) -> Result<LevelData> {
    level_data(f, part)
}
