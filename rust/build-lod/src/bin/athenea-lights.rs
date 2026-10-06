//! A synthetic light-sidecar asset (plan-athenea.md phase 6): a small "car"
//! cloud with three light groups, written as athenea would ship one --
//!
//!   <out>/<name>.athc           the cloud (normals, linear colours, LoD)
//!   <out>/<name>.lights.athl    per group: emitter polygons, a lens profile,
//!                               sparse layers (emission, indirect, a field)
//!   <out>/<name>.lights.usda    groups, states, rules, a sequence
//!
//! The layers stand in for athenea's per-group bakes (proposal 069), which
//! do not exist yet: the emission layer is the emitter splats' glow, the
//! indirect layer a crude bounce (a tenth of the direct term on the splats
//! near each lamp), the field a view-dependent glint on the bonnet. The
//! direct term is the shader's (LTC over the polygons), not in the file.
//!
//!   athenea-lights [--out DIR] [--name NAME]

use std::path::PathBuf;

use anyhow::Result;
use spark_lib::athc::{high_half, low_half, pack_halves, unpack_normal, VirtualTree};
use spark_lib::athc_build::{build_lod, pack_streams, BuildOptions, CloudStreams, SH0};
use spark_lib::athl::{
    cloud_hash, polygon_form_factor, profile_sample, sparse_layers, validate, virtual_values, AthlFile,
    AthlGroup, AthlPolygon, AthlProfile, FIELD_COMPONENTS, GROUP_TWO_SIDED, KIND_EMISSION, KIND_FIELD,
    KIND_INDIRECT,
};

type V3 = [f32; 3];

struct Builder {
    s: CloudStreams,
    albedo: Vec<V3>,
}

impl Builder {
    fn splat(&mut self, p: V3, scale: V3, rotation: [f32; 4], normal: V3, albedo: V3) {
        self.s.count += 1;
        self.s.positions.extend_from_slice(&p);
        self.s.scales.extend_from_slice(&scale);
        self.s.rotations.extend_from_slice(&rotation);
        self.s.opacities.push(0.95);
        self.s.normals.extend_from_slice(&normal);
        // Linear colour: base = 0.5 + SH0 * dc.
        self.s.sh.extend(albedo.map(|c| (c - 0.5) / SH0));
        self.albedo.push(albedo);
    }

    /// A rectangle of splats: centre, two half-extents (along u and v), the
    /// normal u x v, `step` apart, flat along the normal.
    fn quad(&mut self, centre: V3, u: V3, v: V3, step: f32, albedo: V3) {
        let lu = len(u);
        let lv = len(v);
        let nu = (2.0 * lu / step).round().max(1.0) as i32;
        let nv = (2.0 * lv / step).round().max(1.0) as i32;
        let n = normalize(cross(u, v));
        let rotation = rotation_to(n);
        for i in 0..nu {
            for j in 0..nv {
                let a = ((i as f32 + 0.5) / nu as f32) * 2.0 - 1.0;
                let b = ((j as f32 + 0.5) / nv as f32) * 2.0 - 1.0;
                let p = add(centre, add(scale(u, a), scale(v, b)));
                let s = [lu / nu as f32 * 1.1, lv / nv as f32 * 1.1, 0.004];
                // The splat's x/y are u/v only for axis-aligned quads, which is
                // all this scene has; z along the normal stays thin.
                self.splat(p, aligned_scale(u, v, n, s), rotation, n, albedo);
            }
        }
    }
}

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn len(a: V3) -> f32 {
    dot(a, a).sqrt()
}
fn normalize(a: V3) -> V3 {
    scale(a, 1.0 / len(a).max(1e-20))
}

/// Identity rotation; the scales carry the orientation (axis-aligned quads).
fn rotation_to(_n: V3) -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

/// World-axis scales of an axis-aligned quad: each world axis gets the
/// extent of whichever of u, v, n lies along it.
fn aligned_scale(u: V3, v: V3, n: V3, s: V3) -> V3 {
    let mut out = [0.004; 3];
    for (dir, value) in [(u, s[0]), (v, s[1]), (n, s[2])] {
        let k = (0..3).max_by(|&a, &b| dir[a].abs().total_cmp(&dir[b].abs())).unwrap();
        out[k] = value;
    }
    out
}

struct Lamp {
    name: &'static str,
    /// Emitter rectangles: centre, half-extents u and v (normal u x v).
    quads: Vec<(V3, V3, V3)>,
    emit: V3,
    tint: V3,
    two_sided: bool,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let out = PathBuf::from(arg("--out").unwrap_or_else(|| "examples/webgpu/athenea-lights".into()));
    let name = arg("--name").unwrap_or_else(|| "car".into());
    std::fs::create_dir_all(&out)?;

    // The car faces +z; y is up; metres.
    let lamps = [
        Lamp {
            name: "cruce",
            quads: vec![
                ([0.55, 0.62, 2.005], [0.12, 0.0, 0.0], [0.0, 0.05, 0.0]),
                ([-0.55, 0.62, 2.005], [0.12, 0.0, 0.0], [0.0, 0.05, 0.0]),
            ],
            emit: [1.0, 1.0, 1.0],
            tint: [1.0, 1.0, 1.0],
            two_sided: false,
        },
        Lamp {
            name: "drl",
            quads: vec![([0.0, 0.86, 2.005], [0.7, 0.0, 0.0], [0.0, 0.015, 0.0])],
            emit: [1.0, 1.0, 1.0],
            tint: [1.0, 1.0, 1.0],
            two_sided: false,
        },
        Lamp {
            name: "pilotos",
            // Facing -z: u x v = -z with u = -x.
            quads: vec![
                ([0.55, 0.7, -2.005], [-0.15, 0.0, 0.0], [0.0, 0.06, 0.0]),
                ([-0.55, 0.7, -2.005], [-0.15, 0.0, 0.0], [0.0, 0.06, 0.0]),
            ],
            // White emitters behind red lenses (067 §2): the tint is the lens.
            emit: [1.0, 0.04, 0.02],
            tint: [1.0, 0.04, 0.02],
            two_sided: false,
        },
    ];

    let mut b = Builder { s: CloudStreams { coefficients: 1, linear: true, ..Default::default() }, albedo: Vec::new() };
    let grey = [0.45, 0.45, 0.45];
    // Ground, and a wall ahead for the beam's cut-off.
    b.quad([0.0, 0.0, 1.0], [3.5, 0.0, 0.0], [0.0, 0.0, -5.0], 0.12, grey);
    b.quad([0.0, 1.25, 6.0], [-3.5, 0.0, 0.0], [0.0, 1.25, 0.0], 0.1, [0.7, 0.7, 0.68]);
    // The body: a box over the ground, its faces outward.
    let paint = [0.55, 0.06, 0.05];
    let (x, y0, y1, z) = (0.8, 0.25, 1.0, 2.0);
    let yc = (y0 + y1) / 2.0;
    let hy = (y1 - y0) / 2.0;
    b.quad([0.0, y1, 0.0], [x, 0.0, 0.0], [0.0, 0.0, -z], 0.08, paint); // top (+y)
    b.quad([0.0, yc, z], [x, 0.0, 0.0], [0.0, hy, 0.0], 0.05, paint); // front (+z)
    b.quad([0.0, yc, -z], [-x, 0.0, 0.0], [0.0, hy, 0.0], 0.05, paint); // back (-z)
    b.quad([x, yc, 0.0], [0.0, 0.0, -z], [0.0, hy, 0.0], 0.08, paint); // right (+x)
    b.quad([-x, yc, 0.0], [0.0, 0.0, z], [0.0, hy, 0.0], 0.08, paint); // left (-x)
    // The lamps' own splats, just in front of the body.
    let dark = [0.05, 0.05, 0.05];
    for lamp in &lamps {
        for &(c, u, v) in &lamp.quads {
            b.quad(c, u, v, 0.02, dark);
        }
    }

    let options = BuildOptions { transfer: spark_lib::athc_build::TransferKeep::None, ..Default::default() };
    let packed = pack_streams(&b.s, &options)?;
    let file = build_lod(&packed, &options)?;
    let athc = file.write()?;
    let tree = VirtualTree::of_file(&file, true)?;
    let n = file.header.count as usize;
    let splats = file.splats();

    // What the bake reads off the file (its order is the LoD's, not ours).
    let position = |i: usize| [splats.positions[4 * i], splats.positions[4 * i + 1], splats.positions[4 * i + 2]];
    let normal = |i: usize| unpack_normal(splats.normals[i]);
    let albedo = |i: usize| {
        let s = &splats.shape[4 * i..4 * i + 4];
        [high_half(s[2]), low_half(s[3]), high_half(s[3])]
    };

    // The low beam's profile (076 (b)): 128 x 64 over +-40 deg, -10 .. +5 deg,
    // a cut-off line at -0.6 deg on the left rising 15 deg to +1 deg on the
    // right, soft over a texel, the beam bright in the centre.
    let (w, h) = (128u32, 64u32);
    let lon = [-40f32.to_radians(), 40f32.to_radians()];
    let lat = [-10f32.to_radians(), 5f32.to_radians()];
    let mut texels = Vec::with_capacity((3 * w * h) as usize);
    for j in 0..h {
        for i in 0..w {
            let lo = lon[0] + (i as f32 + 0.5) / w as f32 * (lon[1] - lon[0]);
            let la = lat[0] + (j as f32 + 0.5) / h as f32 * (lat[1] - lat[0]);
            let cut = if lo < 0.0 { -0.6f32.to_radians() } else { (-0.6f32.to_radians() + lo.to_radians().tan() * 0.27).min(1f32.to_radians()) };
            let below = (((cut - la) / 0.25f32.to_radians()).clamp(-1.0, 1.0) + 1.0) * 0.5;
            let spread = (-(lo / 0.35).powi(2)).exp() * 2.2 + 0.25;
            let v = below * spread;
            for c in [v, v, v * 0.97] {
                texels.push((pack_halves(c, 0.0) & 0xffff) as u16);
            }
        }
    }
    let profile = AthlProfile { width: w, height: h, lon, lat, outside: [0.0; 3], texels };

    let mut groups = Vec::new();
    let mut polygons = Vec::new();
    for (k, lamp) in lamps.iter().enumerate() {
        let first = polygons.len() as u32;
        for &(c, u, v) in &lamp.quads {
            polygons.push(AthlPolygon {
                group: k as u32,
                radiance: [1.0, 1.0, 1.0],
                vertices: vec![
                    sub(sub(c, u), v),
                    sub(add(c, u), v),
                    add(add(c, u), v),
                    add(sub(c, u), v),
                ],
            });
        }
        let centre = scale(lamp.quads.iter().fold([0.0; 3], |a, q| add(a, q.0)), 1.0 / lamp.quads.len() as f32);
        let facing = normalize(cross(lamp.quads[0].1, lamp.quads[0].2));
        let up = [0.0, 1.0, 0.0];
        let ax = normalize(cross(up, facing));
        groups.push(AthlGroup {
            name: lamp.name.into(),
            flags: if lamp.two_sided { GROUP_TWO_SIDED } else { 0 },
            polygon_first: first,
            polygon_count: polygons.len() as u32 - first,
            profile: if lamp.name == "cruce" { 0 } else { -1 },
            tint: lamp.tint,
            origin: centre,
            axes: [ax, up, facing],
            radiance: 1.0,
        });
    }
    let profiles = vec![profile];

    let inside = |p: V3, (c, u, v): (V3, V3, V3)| {
        let d = sub(p, c);
        dot(d, u).abs() <= dot(u, u) + 1e-4 && dot(d, v).abs() <= dot(v, v) + 1e-4 && dot(d, normalize(cross(u, v))).abs() < 0.01
    };
    let mut athl = AthlFile {
        flags: 0,
        element_count: tree.splat_base + n as u32,
        merged: tree.merged,
        splat_base: tree.splat_base,
        splat_count: n as u32,
        cloud_hash: cloud_hash(&athc),
        bake_hash: 0x5157_4854, // "synthetic"
        groups: groups.clone(),
        polygons: polygons.clone(),
        profiles: profiles.clone(),
        layers: Vec::new(),
    };
    let mut stats = Vec::new();
    for (k, lamp) in lamps.iter().enumerate() {
        let g = &groups[k];
        let mut emission = vec![0f32; 3 * n];
        let mut indirect = vec![0f32; 3 * n];
        for i in 0..n {
            let p = position(i);
            if lamp.quads.iter().any(|&q| inside(p, q)) {
                emission[3 * i..3 * i + 3].copy_from_slice(&lamp.emit);
                continue;
            }
            // A stand-in bounce: a tenth of the direct light, near the lamp.
            let mut f = 0.0;
            for poly in &polygons[g.polygon_first as usize..(g.polygon_first + g.polygon_count) as usize] {
                f += polygon_form_factor(p, normal(i), &poly.vertices, lamp.two_sided);
            }
            let shape = if g.profile >= 0 { profile_sample(&profiles[g.profile as usize], g, p) } else { [1.0; 3] };
            let a = albedo(i);
            let dist = len(sub(p, g.origin));
            let falloff = (-dist / 1.5).exp();
            for c in 0..3 {
                indirect[3 * i + c] = 0.1 * a[c] * g.tint[c] * shape[c] * f * falloff + 0.02 * a[c] * g.tint[c] * falloff * falloff;
            }
        }
        let emission = virtual_values(&file, &tree, &emission, 3)?;
        let indirect = virtual_values(&file, &tree, &indirect, 3)?;
        let e = sparse_layers(k as u16, KIND_EMISSION, 3, &emission, 1e-3)?;
        let ind = sparse_layers(k as u16, KIND_INDIRECT, 3, &indirect, 2e-3)?;
        stats.push(serde_json::json!({
            "group": lamp.name,
            "emissionBlocks": e.iter().map(|l| l.blocks.len()).sum::<usize>(),
            "indirectBlocks": ind.iter().map(|l| l.blocks.len()).sum::<usize>(),
        }));
        athl.layers.extend(e);
        athl.layers.extend(ind);
    }
    // A glint on the bonnet from the DRL strip: degree-1 lobe towards +z,
    // so it shows looking at the car from the front.
    let mut field = vec![0f32; FIELD_COMPONENTS as usize * n];
    for i in 0..n {
        let p = position(i);
        if (p[1] - 1.0).abs() < 0.01 && p[2] > 1.2 {
            let s = 0.25 * ((p[2] - 1.2) / 0.8);
            let c = FIELD_COMPONENTS as usize * i;
            for ch in 0..3 {
                field[c + ch] = s * 0.282_095; // DC
                field[c + 3 * 2 + ch] = -s * 0.488_603; // Y(1,0) ~ z, Spark's sign
            }
        }
    }
    let field = virtual_values(&file, &tree, &field, FIELD_COMPONENTS)?;
    athl.layers.extend(sparse_layers(1, KIND_FIELD, FIELD_COMPONENTS, &field, 1e-3)?);
    validate(&athl)?;
    let athl_bytes = athl.write()?;

    let usda = format!(
        r#"#usda 1.0
(
    doc = "Synthetic light sidecar for {name}.athc (sparkwebGPU, rust/build-lod athenea-lights): three groups as athenea's 066/067 sidecar describes them."
)

def Scope "Lights"
{{
    string athenea:lightSidecar:cloudHash = "{hash:016x}"
    asset athenea:lightSidecar:athl = @./{name}.lights.athl@
    float athenea:lightSidecar:nitsPerUnit = 1
    token athenea:lightSidecar:defaultState = "aparcado"

    def Scope "LightGroups"
    {{
        def Scope "cruce" (prepend apiSchemas = ["AtheneaLightGroupAPI"])
        {{
            token athenea:lightGroup:function = "lowBeam"
            token athenea:lightGroup:technology = "xenon"
            float athenea:lightGroup:radiance = 6
            float athenea:lightGroup:temperatureK = 6000
            float athenea:lightGroup:riseSeconds = 2
        }}
        def Scope "drl" (prepend apiSchemas = ["AtheneaLightGroupAPI"])
        {{
            token athenea:lightGroup:function = "daytimeRunning"
            token athenea:lightGroup:technology = "led"
            float athenea:lightGroup:radiance = 3
            float athenea:lightGroup:temperatureK = 6500
            float athenea:lightGroup:riseSeconds = 0.15
        }}
        def Scope "pilotos" (prepend apiSchemas = ["AtheneaLightGroupAPI"])
        {{
            token athenea:lightGroup:function = "tail"
            token athenea:lightGroup:technology = "led"
            float athenea:lightGroup:radiance = 2
            float athenea:lightGroup:riseSeconds = 0.08
        }}
    }}

    def Scope "LightStates"
    {{
        def Scope "aparcado"     {{ dictionary athenea:lightState:targets = {{}} }}
        def Scope "diurno"       {{ dictionary athenea:lightState:targets = {{ double drl = 1 }} }}
        def Scope "noche_ciudad" {{ dictionary athenea:lightState:targets = {{ double cruce = 1  double drl = 1  double pilotos = 1 }} }}
        def Scope "frenando"     {{
            token athenea:lightState:base = "noche_ciudad"
            dictionary athenea:lightState:targets = {{ double pilotos = 4 }}
        }}
    }}

    def Scope "LightRules"
    {{
        def Scope "drl_off_with_low_beam" {{
            token athenea:lightRule:when = "cruce"
            token athenea:lightRule:target = "drl"
            double athenea:lightRule:scale = 0
        }}
    }}

    def Scope "Sequences"
    {{
        def Scope "bienvenida" {{
            double[] athenea:lightSequence:times = [0, 1.5, 3, 7, 9]
            token[] athenea:lightSequence:states = ["aparcado", "diurno", "noche_ciudad", "frenando", "noche_ciudad"]
            double athenea:lightSequence:duration = 12
            bool athenea:lightSequence:loop = true
        }}
    }}
}}
"#,
        hash = athl.cloud_hash,
    );

    std::fs::write(out.join(format!("{name}.athc")), &athc)?;
    std::fs::write(out.join(format!("{name}.lights.athl")), &athl_bytes)?;
    std::fs::write(out.join(format!("{name}.lights.usda")), usda)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "splats": n,
            "merged": tree.merged,
            "splatBase": tree.splat_base,
            "athcBytes": athc.len(),
            "athlBytes": athl_bytes.len(),
            "layers": athl.layers.len(),
            "groups": stats,
            "cloudHash": format!("{:016x}", athl.cloud_hash),
        }))?
    );
    Ok(())
}
