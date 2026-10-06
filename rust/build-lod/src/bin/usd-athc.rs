//! athenea's relightable clouds, from USD to `.athc`.
//!
//! athenea writes a cloud with a transfer (`mesh2splat --transfer`) only as a
//! USD stage: a `ParticleField3DGaussianSplat` whose primvars carry what a
//! `.athc` would (`modules/usd/src/ParticleField.cpp` at txf 89a04d9 reads
//! them). This reads one from a `.usdc` (crate file, through the `openusd`
//! crate) and writes the `.athc` athenea's own `writeAthc` would, transfer
//! included, built as `LodBuilder` builds one (`spark_lib::athc_build`), as
//! v3 by sections (docs/docs/athc-v3.md) or v2.
//!
//! ```sh
//! usd-athc in.usdc out.athc [--prim /World/Splats] [--transfer full|112|84|64|36|16|9|none]
//!          [--no-shadow] [--no-material] [--no-lobes] [--no-normals] [--no-curvature] [--max-sh 0|3|8|15]
//!          [--chunk 65536] [--gzip] [--v2] [--json out.json]
//!          [--add more.usdc]... [--only-prim TEXT]... [--exclude-prim TEXT]...
//!          [--thin RATIO | --target SPLATS | --cell SIDE [--fill 1.0]]
//!          [--drop-backs THICKNESS] [--box x0,y0,z0,x1,y1,z1]
//! usd-athc in.usdc --list            the prim's attributes, their types and lengths
//! ```
//!
//! For the web: `--add` appends more clouds of the same streams (one file for
//! several of athenea's per-material clouds; `--add more.usdc::TEXT,TEXT`
//! also drops those prims from that cloud alone, where two bakes share them); `--only-prim` / `--exclude-prim`
//! keep or drop the splats whose `cryptoObject` names a prim whose path
//! contains the text (athenea's `cryptoManifest`); `--cell` merges the splats
//! in each grid cell of that side into one before the LoD build
//! (`athc_build::reduce_cells`; one side for every part of a scene keeps the
//! density even), `--target` finds the cell for about that many splats, and
//! `--fill` widens each merged Gaussian; `--thin` keeps about one splat in
//! that many instead (`athc_build::reduce_thin`: a real splat's shape, grown
//! to its run's area), which keeps surfaces closed. `--drop-backs` (metres,
//! before any reduction) drops the closed back face of a shell up to that
//! thick (`athc_build::drop_hidden_backs`): mesh2splat bakes both faces of
//! a solidified panel, and the dark inner one shows through the outer one
//! in moiré bands wherever the draw's sort interleaves them (LoD levels,
//! thinned clouds). `--box` keeps the splats whose centres lie in that box
//! (after `--drop-backs`): a small piece of a large bake, for tests.
//!
//! USD attribute -> `.athc` array:
//!
//! ```text
//! positions (point3f[] | positionsh)          positions xyz
//! opacities (float[], linear)                 positions w; < 1/255 dropped
//! orientations (quatf[] | quath[])            shape[0] smallest three
//! scales (float3[], linear)                   shape[1..2] f16 ln
//! radiance:sphericalHarmonicsCoefficients     shape base (0.5 + SH0 dc), sh rest
//!   (float3[], DC first), :...Degree (int)
//! primvars:athenea:splat:linear (bool)        flag bit 1
//! primvars:athenea:splat:normal (normal3f[])  normals (octahedral), flag bit 0
//! primvars:athenea:splat:metallic, roughness,
//!   transmission (float[])                    pbr, flag bit 4
//! primvars:athenea:splat:thinWalled,
//!   schlickMetal (int[])                      pbr bits 24, 25
//! primvars:athenea:splat:specularWeight, specularColor, specularIor,
//!   coatWeight, coatRoughness, coatIor, sheenColor, sheenRoughness,
//!   coatDarkening (float[], color3f[])        lobes (3 words, packing.slang
//!                                             packLobes), beside pbr
//! primvars:athenea:splat:transferDirect (float[], 16 a splat), transferIndirect
//!   (48), transferReflected (48)              transfer, 112 f16, flag bit 5
//! primvars:athenea:splat:shadowBits (int[], 8 a splat)   shadowBits
//! primvars:athenea:splat:curvature (float[], 3 a splat)  curvature, 3 f16
//!                                             (v3 section CURV; not in a v2)
//! cryptoObject, cryptoManifest, ior, relight  not in a .athc:
//!                                             reported in --json
//! primvars:athenea:splat:transferZonal       transfer of 10 values (two zonal
//!   (float[], 10 a splat)                     lobes in the splat's frame; a
//!                                             skinned cloud's), in place of
//!                                             transferDirect
//! skel:jointIndices, skel:jointWeights        SKIN: athenea's packed influences
//!   (or athenea:splat:joint*; elementSize k)  (joint u16 | weight unorm16)
//! primvars:athenea:splat:jointWeightGradients SKIN: gradient words (two halves)
//!   (half[], 2 (k - 1) a splat)
//! skel:joints, skel:geomBindTransform,        the skeleton (ATSK blob, v3 only;
//!   primvars:athenea:splat:skinningXforms     athc_skin.rs): joints, bind
//!   (matrix4d[], time sampled)                transform, one clip of samples
//! ```
//!
//! A skinned cloud (`athenea mesh2splat --skinned`) keeps its rig: the
//! `SKIN` section and the skeleton (docs/docs/athc-v3.md). `--clip NAME`
//! names the clip its `skinningXforms` make (default `default`), and
//! `--add-clip NAME=other.usdc` adds the samples another conversion of the
//! same rig holds as a second clip. `--add`, `--cell`, `--target` and
//! `--thin` are refused with a skin; a v2 file drops it. A `.usda` layer is
//! read too.

use std::collections::BTreeMap;

use anyhow::{anyhow, bail, Context, Result};
use openusd::sdf::{self, AbstractData, Value};
use serde_json::{json, Value as Json};
use spark_lib::athc_build::{
    build_lod, cell_for_target, crop_box, drop_hidden_backs, pack_streams, reduce_cells, reduce_thin, BuildOptions,
    CloudStreams, LobeStreams, TransferKeep,
};
use spark_lib::athc_skin::{AthcSkeleton, SkinClip};
use spark_lib::athc_v3::{gzip, parse_v3, write_v3_skinned, COMPRESSION_GZIP, COMPRESSION_NONE};

/// A float array of any of the value types athenea writes (or USD allows).
fn floats(v: &Value) -> Option<Vec<f32>> {
    Some(match v {
        Value::FloatVec(x) => x.clone(),
        Value::HalfVec(x) => x.iter().map(|h| h.to_f32()).collect(),
        Value::DoubleVec(x) => x.iter().map(|d| *d as f32).collect(),
        Value::Vec3fVec(x) => x.iter().flat_map(|v| [v.x, v.y, v.z]).collect(),
        Value::Vec3hVec(x) => x
            .iter()
            .flat_map(|v| [v.x.to_f32(), v.y.to_f32(), v.z.to_f32()])
            .collect(),
        Value::Vec3dVec(x) => x
            .iter()
            .flat_map(|v| [v.x as f32, v.y as f32, v.z as f32])
            .collect(),
        // GfQuat is (w, x, y, z); streams.slang reads xyzw (imaginary first).
        Value::QuatfVec(x) => x.iter().flat_map(|q| [q.x, q.y, q.z, q.w]).collect(),
        Value::QuathVec(x) => x
            .iter()
            .flat_map(|q| [q.x.to_f32(), q.y.to_f32(), q.z.to_f32(), q.w.to_f32()])
            .collect(),
        Value::QuatdVec(x) => x
            .iter()
            .flat_map(|q| [q.x as f32, q.y as f32, q.z as f32, q.w as f32])
            .collect(),
        _ => return None,
    })
}

fn kind(v: &Value) -> String {
    let s = format!("{v:?}");
    let name: String = s.chars().take_while(|c| c.is_alphanumeric()).collect();
    let len = match v {
        Value::FloatVec(x) => x.len(),
        Value::HalfVec(x) => x.len(),
        Value::IntVec(x) => x.len(),
        Value::Vec3fVec(x) => x.len(),
        Value::Vec3hVec(x) => x.len(),
        Value::QuatfVec(x) => x.len(),
        Value::QuathVec(x) => x.len(),
        _ => return format!("{name} {}", s.chars().take(80).collect::<String>()),
    };
    format!("{name}[{len}]")
}

/// One prim's authored attribute defaults, and the time samples of those
/// that have them.
pub struct Prim {
    pub attributes: BTreeMap<String, Value>,
    pub samples: BTreeMap<String, Vec<(f64, Value)>>,
}

/// A `.usdc` or a `.usda` layer.
fn read_layer(path: &str) -> Result<Box<dyn AbstractData>> {
    if path.ends_with(".usda") {
        return Ok(Box::new(openusd::usda::read_file(path).with_context(|| format!("reading {path}"))?));
    }
    openusd::usdc::read_file(path).with_context(|| format!("reading {path}"))
}

/// The layer's `timeCodesPerSecond` (USD's default, 24, where it says none).
fn time_codes_per_second(data: &dyn AbstractData) -> f32 {
    let Ok(root) = sdf::path("/") else { return 24.0 };
    match data.try_field(&root, "timeCodesPerSecond").ok().flatten().map(|v| v.into_owned()) {
        Some(Value::Double(f)) => f as f32,
        Some(Value::Float(f)) => f,
        _ => 24.0,
    }
}

/// A matrix4d as 16 floats, USD rows.
fn matrix(m: &openusd::gf::Matrix4d) -> [f32; 16] {
    m.0.map(|v| v as f32)
}

impl Prim {
    pub fn read(data: &dyn AbstractData, prim: &str) -> Result<Self> {
        let path = sdf::path(prim).map_err(|e| anyhow!("{prim}: {e}"))?;
        let names = match data
            .try_field(&path, "propertyChildren")
            .map_err(|e| anyhow!("{e}"))?
        {
            Some(v) => match v.into_owned() {
                Value::TokenVec(t) => t.into_iter().map(|t| t.to_string()).collect::<Vec<_>>(),
                other => bail!("{prim}: propertyChildren is {other:?}"),
            },
            None => bail!("no prim {prim} with properties"),
        };
        let mut attributes = BTreeMap::new();
        let mut samples = BTreeMap::new();
        for name in names {
            let at = sdf::path(format!("{prim}.{name}")).map_err(|e| anyhow!("{name}: {e}"))?;
            if let Some(v) = data
                .try_field(&at, "default")
                .map_err(|e| anyhow!("{name}: {e}"))?
            {
                attributes.insert(name.clone(), v.into_owned());
            }
            // Only the rig's transforms are read over time.
            if name.ends_with("skinningXforms") {
                if let Some(Value::TimeSamples(t)) =
                    data.try_field(&at, "timeSamples").map_err(|e| anyhow!("{name}: {e}"))?.map(|v| v.into_owned())
                {
                    samples.insert(name, t);
                }
            }
        }
        Ok(Self { attributes, samples })
    }

    /// Keeps the elements of every per-splat array where `keep` is true
    /// (arrays of `count · k` values; scalars and other arrays untouched), so
    /// a big cloud is filtered before its streams are copied out.
    pub fn retain(&mut self, keep: &[bool]) {
        fn filter<T: Clone>(v: &mut Vec<T>, keep: &[bool]) {
            let count = keep.len();
            if count == 0 || v.is_empty() || v.len() % count != 0 {
                return;
            }
            let per = v.len() / count;
            let mut k = 0;
            v.retain(|_| {
                let kept = keep[k / per];
                k += 1;
                kept
            });
        }
        for v in self.attributes.values_mut() {
            match v {
                Value::FloatVec(x) => filter(x, keep),
                Value::HalfVec(x) => filter(x, keep),
                Value::DoubleVec(x) => filter(x, keep),
                Value::IntVec(x) => filter(x, keep),
                Value::UintVec(x) => filter(x, keep),
                Value::Vec3fVec(x) => filter(x, keep),
                Value::Vec3hVec(x) => filter(x, keep),
                Value::Vec3dVec(x) => filter(x, keep),
                Value::QuatfVec(x) => filter(x, keep),
                Value::QuathVec(x) => filter(x, keep),
                Value::QuatdVec(x) => filter(x, keep),
                _ => {}
            }
        }
    }

    fn get(&self, name: &str) -> Option<&Value> {
        self.attributes
            .get(name)
            .or_else(|| self.attributes.get(&format!("primvars:{name}")))
    }

    fn floats(&self, names: &[&str]) -> Result<Vec<f32>> {
        for name in names {
            if let Some(v) = self.get(name) {
                return floats(v).ok_or_else(|| anyhow!("{name} is {}, not floats", kind(v)));
            }
        }
        Ok(Vec::new())
    }

    fn ints(&self, name: &str) -> Result<Vec<u32>> {
        match self.get(name) {
            Some(Value::IntVec(x)) => Ok(x.iter().map(|v| *v as u32).collect()),
            Some(Value::UintVec(x)) => Ok(x.clone()),
            Some(v) => bail!("{name} is {}, not ints", kind(v)),
            None => Ok(Vec::new()),
        }
    }

    fn bool(&self, name: &str) -> bool {
        matches!(self.get(name), Some(Value::Bool(true)))
    }

    fn scalar(&self, name: &str) -> Option<Json> {
        match self.get(name)? {
            Value::Float(f) => Some(json!(f)),
            Value::Double(f) => Some(json!(f)),
            Value::Int(i) => Some(json!(i)),
            Value::Bool(b) => Some(json!(b)),
            Value::String(s) => Some(json!(s)),
            Value::Token(t) => Some(json!(t.to_string())),
            _ => None,
        }
    }

    /// `scene::SplatStreams`, as ParticleField.cpp fills one.
    pub fn streams(&self) -> Result<CloudStreams> {
        let positions = self.floats(&["positions", "positionsh"])?;
        let count = positions.len() / 3;
        if count == 0 {
            bail!("no positions");
        }
        let sh = self.floats(&[
            "radiance:sphericalHarmonicsCoefficients",
            "radiance:sphericalHarmonicsCoefficientsh",
        ])?;
        let coefficients = sh.len() / 3 / count;
        if let Some(Value::Int(degree)) = self.get("radiance:sphericalHarmonicsDegree") {
            let expect = ((degree + 1) * (degree + 1)) as usize;
            if !sh.is_empty() && expect != coefficients {
                bail!("SH degree {degree} but {coefficients} coefficients a splat");
            }
        }
        Ok(CloudStreams {
            count,
            positions,
            rotations: self.floats(&["orientations", "orientationsh"])?,
            scales: self.floats(&["scales", "scalesh"])?,
            opacities: self.floats(&["opacities", "opacitiesh"])?,
            coefficients,
            sh,
            linear: self.bool("athenea:splat:linear"),
            normals: self.floats(&["athenea:splat:normal"])?,
            metallic: self.floats(&["athenea:splat:metallic"])?,
            roughness: self.floats(&["athenea:splat:roughness"])?,
            transmission: self.floats(&["athenea:splat:transmission"])?,
            thin_walled: self.ints("athenea:splat:thinWalled")?,
            schlick_metal: self.ints("athenea:splat:schlickMetal")?,
            lobes: LobeStreams {
                specular_weight: self.floats(&["athenea:splat:specularWeight"])?,
                specular_colour: self.floats(&["athenea:splat:specularColor"])?,
                specular_ior: self.floats(&["athenea:splat:specularIor"])?,
                coat_weight: self.floats(&["athenea:splat:coatWeight"])?,
                coat_roughness: self.floats(&["athenea:splat:coatRoughness"])?,
                coat_ior: self.floats(&["athenea:splat:coatIor"])?,
                sheen_colour: self.floats(&["athenea:splat:sheenColor"])?,
                sheen_roughness: self.floats(&["athenea:splat:sheenRoughness"])?,
                coat_darkening: self.floats(&["athenea:splat:coatDarkening"])?,
            },
            transfer_direct: self.floats(&["athenea:splat:transferDirect"])?,
            transfer_indirect: self.floats(&["athenea:splat:transferIndirect"])?,
            transfer_reflected: self.floats(&["athenea:splat:transferReflected"])?,
            shadow_bits: self.ints("athenea:splat:shadowBits")?,
            curvature: self.floats(&["athenea:splat:curvature"])?,
            transfer_zonal: self.floats(&["athenea:splat:transferZonal"])?,
            joint_indices: self.joint_indices()?,
            joint_weights: self.joint_weights()?,
            skin_influences: self.skin_influences(count)?,
            joint_count: self.joints().len() as u32,
            weight_gradients: match self.get("athenea:splat:jointWeightGradients") {
                Some(Value::HalfVec(h)) => h.iter().map(|v| v.to_bits()).collect(),
                Some(v) => bail!("jointWeightGradients is {}, not halves", kind(v)),
                None => Vec::new(),
            },
        })
    }

    /// `skel:jointIndices` (UsdSkel's binding), or athenea's older name.
    fn joint_indices(&self) -> Result<Vec<i32>> {
        for name in ["skel:jointIndices", "athenea:splat:jointIndices"] {
            match self.get(name) {
                Some(Value::IntVec(x)) => return Ok(x.clone()),
                Some(v) => bail!("{name} is {}, not ints", kind(v)),
                None => {}
            }
        }
        Ok(Vec::new())
    }

    fn joint_weights(&self) -> Result<Vec<f32>> {
        for name in ["skel:jointWeights", "athenea:splat:jointWeights"] {
            if let Some(v) = self.get(name) {
                return floats(v).ok_or_else(|| anyhow!("{name} is {}, not floats", kind(v)));
            }
        }
        Ok(Vec::new())
    }

    /// Influences a splat: the arrays' length over the count (a constant
    /// binding, one set for every splat, is spread to each).
    fn skin_influences(&self, count: usize) -> Result<usize> {
        let n = self.joint_indices()?.len();
        if n == 0 || count == 0 {
            return Ok(0);
        }
        if n % count != 0 {
            bail!("{n} joint indices for {count} splats");
        }
        Ok(n / count)
    }

    /// `skel:joints`.
    fn joints(&self) -> Vec<String> {
        match self.get("skel:joints") {
            Some(Value::TokenVec(t)) => t.iter().map(|t| t.to_string()).collect(),
            Some(Value::StringVec(t)) => t.clone(),
            _ => Vec::new(),
        }
    }

    /// The rig as `athc_skin` keeps it, one clip of the cached transforms,
    /// or None for a cloud nothing carries.
    fn skeleton(
        &self,
        clip: &str,
        fps: f32,
        influences: u32,
        gradient_words: u32,
    ) -> Result<Option<AthcSkeleton>> {
        if influences == 0 {
            return Ok(None);
        }
        let geom_bind = match self
            .get("skel:geomBindTransform")
            .or_else(|| self.get("athenea:splat:geomBindTransform"))
        {
            Some(Value::Matrix4d(m)) => matrix(m),
            Some(v) => bail!("geomBindTransform is {}, not a matrix4d", kind(v)),
            None => matrix(&openusd::gf::Matrix4d::IDENTITY),
        };
        let mut joints = self.joints();
        let clip = self.clip(clip, fps)?;
        let count = clip.xforms.len() / 16 / clip.times.len().max(1);
        if joints.is_empty() {
            joints = (0..count).map(|j| format!("joint{j}")).collect();
        }
        if joints.len() != count {
            bail!("{} joints and {count} skinning transforms", joints.len());
        }
        let skeleton = match self.get("athenea:splat:skeleton") {
            Some(Value::String(s)) => s.clone(),
            _ => String::new(),
        };
        let s = AthcSkeleton {
            influences,
            gradient_words,
            joints,
            skeleton,
            geom_bind,
            clips: vec![clip],
        };
        s.check()?;
        Ok(Some(s))
    }

    /// `skinningXforms`, as a clip: its time samples, or its default alone.
    fn clip(&self, name: &str, fps: f32) -> Result<SkinClip> {
        let key = "primvars:athenea:splat:skinningXforms";
        let mut out = SkinClip {
            name: name.to_string(),
            time_codes_per_second: fps,
            ..Default::default()
        };
        let mut add = |t: f64, v: &Value| -> Result<()> {
            match v {
                Value::Matrix4dVec(m) => {
                    out.times.push(t as f32);
                    out.xforms.extend(m.iter().flat_map(matrix));
                    Ok(())
                }
                other => bail!("skinningXforms is {}, not matrix4d[]", kind(other)),
            }
        };
        if let Some(samples) = self.samples.get(key) {
            for (t, v) in samples {
                add(*t, v)?;
            }
        } else if let Some(v) = self.get("athenea:splat:skinningXforms") {
            add(0.0, v)?;
        } else {
            bail!(
                "a skinned cloud without skinningXforms (resolve its Skeleton's animation first)"
            );
        }
        Ok(out)
    }


    /// What a `.athc` has no room for, for a page to set on the cloud.
    fn constants(&self) -> Json {
        let mut out = serde_json::Map::new();
        for name in [
            "athenea:splat:ior",
            "athenea:splat:relight",
            "athenea:splat:linear",
            "athenea:splat:cryptoManifest",
        ] {
            if let Some(v) = self.scalar(name) {
                out.insert(name.trim_start_matches("athenea:splat:").to_string(), v);
            }
        }
        if self.get("athenea:splat:emission").is_some() {
            eprintln!("warning: athenea:splat:emission is not carried yet");
        }
        if let Some(Value::IntVec(ids)) = self.get("athenea:splat:cryptoObject") {
            let mut distinct: Vec<i32> = ids.clone();
            distinct.sort();
            distinct.dedup();
            out.insert(
                "cryptoObjects".into(),
                json!(distinct
                    .iter()
                    .take(16)
                    .map(|i| format!("{:08x}", *i as u32))
                    .collect::<Vec<_>>()),
            );
        }
        if let Some(Value::Vec3fVec(e)) = self.get("extent") {
            out.insert(
                "extent".into(),
                json!(e.iter().map(|v| [v.x, v.y, v.z]).collect::<Vec<_>>()),
            );
        }
        Json::Object(out)
    }
}

/// `b` after `a`; both must carry the same streams.
fn append(a: &mut CloudStreams, b: CloudStreams) -> Result<()> {
    if a.coefficients != b.coefficients || a.linear != b.linear {
        bail!("clouds with different harmonics or colour spaces");
    }
    macro_rules! cat {
        ($($f:ident),*) => {$(
            if a.$f.is_empty() != b.$f.is_empty() {
                bail!("{} is in one cloud and not the other", stringify!($f));
            }
            a.$f.extend(b.$f);
        )*};
    }
    cat!(
        positions,
        rotations,
        scales,
        opacities,
        sh,
        normals,
        metallic,
        roughness,
        transmission,
        transfer_direct,
        transfer_indirect,
        transfer_reflected,
        shadow_bits,
        curvature
    );
    // Per-splat marks and layers default where one cloud does not carry
    // them, as athenea reads a missing array (0, and `plainLobes`).
    for (mine, theirs) in [
        (&mut a.thin_walled, b.thin_walled),
        (&mut a.schlick_metal, b.schlick_metal),
    ] {
        if mine.is_empty() && theirs.is_empty() {
            continue;
        }
        mine.resize(a.count, 0);
        if theirs.is_empty() {
            mine.resize(a.count + b.count, 0);
        } else {
            mine.extend_from_slice(&theirs[..b.count]);
        }
    }
    a.lobes.append(a.count, b.lobes, b.count);
    a.count += b.count;
    Ok(())
}

/// Which splats to keep by the prims their `cryptoObject` names.
fn prim_mask(
    prim: &Prim,
    count: usize,
    only: &[&str],
    exclude: &[&str],
) -> Result<Option<Vec<bool>>> {
    if only.is_empty() && exclude.is_empty() {
        return Ok(None);
    }
    let Some(Value::IntVec(ids)) = prim.get("athenea:splat:cryptoObject") else {
        bail!("--only-prim / --exclude-prim need athenea:splat:cryptoObject");
    };
    let Some(Value::String(manifest)) = prim.get("athenea:splat:cryptoManifest") else {
        bail!("--only-prim / --exclude-prim need athenea:splat:cryptoManifest");
    };
    let manifest: BTreeMap<String, String> = serde_json::from_str(manifest)?;
    let mut wanted = std::collections::HashMap::new();
    for (path, hash) in &manifest {
        let id = u32::from_str_radix(hash, 16)? as i32;
        let keep = (only.is_empty() || only.iter().any(|t| path.contains(t)))
            && !exclude.iter().any(|t| path.contains(t));
        wanted.insert(id, keep);
    }
    if ids.len() != count {
        bail!("{} crypto ids for {count} splats", ids.len());
    }
    Ok(Some(
        ids.iter()
            .map(|i| *wanted.get(i).unwrap_or(&true))
            .collect(),
    ))
}

fn args_all<'a>(args: &'a [String], flag: &str) -> Vec<&'a str> {
    args.windows(2)
        .filter(|w| w[0] == flag)
        .map(|w| w[1].as_str())
        .collect()
}

fn arg<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

/// The layer's `upAxis` metadata (the pseudo-root's), where it says one.
fn up_axis(data: &dyn AbstractData) -> Option<String> {
    let root = sdf::path("/").ok()?;
    match data.try_field(&root, "upAxis").ok()??.into_owned() {
        Value::Token(t) => Some(t.to_string()),
        Value::String(t) => Some(t),
        _ => None,
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| args.iter().any(|a| a == f);
    let valued = [
        "--prim",
        "--transfer",
        "--max-sh",
        "--chunk",
        "--json",
        "--add",
        "--only-prim",
        "--exclude-prim",
        "--target",
        "--cell",
        "--fill",
        "--thin",
        "--drop-backs",
        "--clip",
        "--add-clip",
    ];
    let mut paths = Vec::new();
    let mut skip = false;
    for a in &args {
        if skip {
            skip = false;
        } else if valued.contains(&a.as_str()) {
            skip = true;
        } else if !a.starts_with("--") {
            paths.push(a.clone());
        }
    }
    let Some(input) = paths.first() else {
        bail!("usage: usd-athc in.usdc out.athc [--prim P] [--transfer full|112|84|64|36|16|9|none] [--no-shadow] [--no-material] [--no-normals] [--no-curvature] [--max-sh N] [--chunk N] [--gzip] [--v2] [--json out.json] | usd-athc in.usdc --list");
    };
    let prim_path = arg(&args, "--prim").unwrap_or("/World/Splats");
    let t = std::time::Instant::now();
    let data = read_layer(input)?;
    let prim = Prim::read(data.as_ref(), prim_path)?;
    let up_axis = up_axis(data.as_ref());
    if flag("--list") {
        if let Some(up) = &up_axis {
            println!("upAxis: {up}");
        }
        for (name, v) in &prim.attributes {
            println!("{name}: {}", kind(v));
        }
        if let Some(Value::String(m)) = prim.get("athenea:splat:cryptoManifest") {
            println!("cryptoManifest: {m}");
        }
        return Ok(());
    }
    let Some(output) = paths.get(1) else {
        bail!("no output path")
    };
    let only = args_all(&args, "--only-prim");
    let exclude = args_all(&args, "--exclude-prim");
    let transfer_arg = arg(&args, "--transfer").unwrap_or("full");
    // A transfer of its direct half only: the other halves need not be read.
    let direct_only = matches!(transfer_arg, "none" | "9" | "16");
    let filtered = |mut prim: Prim, more_excluded: &[&str]| -> Result<CloudStreams> {
        if direct_only {
            for half in ["transferIndirect", "transferReflected"] {
                prim.attributes
                    .remove(&format!("primvars:athenea:splat:{half}"));
            }
        }
        let count = prim.floats(&["positions", "positionsh"])?.len() / 3;
        let exclude: Vec<&str> = exclude.iter().chain(more_excluded).copied().collect();
        if let Some(keep) = prim_mask(&prim, count, &only, &exclude)? {
            prim.retain(&keep);
        }
        prim.streams()
    };
    let constants = prim.constants();
    // The rig, before the prim's arrays are moved into streams.
    let fps = time_codes_per_second(data.as_ref());
    let rig = {
        let count = prim.floats(&["positions", "positionsh"])?.len() / 3;
        let k = prim.skin_influences(count)? as u32;
        let has_gradients = matches!(prim.get("athenea:splat:jointWeightGradients"),
            Some(Value::HalfVec(h)) if k > 1 && h.len() == count * 2 * (k as usize - 1));
        prim.skeleton(arg(&args, "--clip").unwrap_or("default"), fps, k, if has_gradients { k - 1 } else { 0 })?
    };
    let mut streams = filtered(prim, &[])?;
    let mut sources = vec![json!({ "source": input, "splats": streams.count })];
    if rig.is_some() {
        for f in ["--add", "--cell", "--target", "--thin"] {
            if flag(f) {
                bail!("{f} is not supported on a skinned cloud");
            }
        }
    }
    let mut rig = rig;
    for added in args_all(&args, "--add-clip") {
        let Some(r) = rig.as_mut() else { bail!("--add-clip on a cloud nothing carries") };
        let (name, more) = added.split_once('=').ok_or_else(|| anyhow!("--add-clip NAME=file.usdc"))?;
        let other = read_layer(more)?;
        let p = Prim::read(other.as_ref(), prim_path)?;
        if !p.joints().is_empty() && p.joints() != r.joints {
            bail!("{more}: another skeleton's joints");
        }
        r.clips.push(p.clip(name, time_codes_per_second(other.as_ref()))?);
        r.check()?;
    }
    for added in args_all(&args, "--add") {
        // path::TEXT,TEXT drops those prims from this cloud only
        let (more, own) = added.split_once("::").unwrap_or((added, ""));
        let own: Vec<&str> = own.split(',').filter(|t| !t.is_empty()).collect();
        let data = read_layer(more)?;
        let s = filtered(Prim::read(data.as_ref(), prim_path)?, &own)?;
        sources.push(json!({ "source": more, "splats": s.count }));
        append(&mut streams, s)?;
    }
    let read_s = t.elapsed().as_secs_f32();

    let transfer = match transfer_arg {
        "full" => TransferKeep::Full,
        "none" => TransferKeep::None,
        n => TransferKeep::Count(n.parse().context("--transfer")?),
    };
    let mut options = BuildOptions {
        transfer,
        shadow_bits: !flag("--no-shadow"),
        material: !flag("--no-material"),
        normals: !flag("--no-normals"),
        curvature: !flag("--no-curvature"),
        ..Default::default()
    };
    if let Some(m) = arg(&args, "--max-sh") {
        options.max_rest = m.parse().context("--max-sh")?;
    }
    if let Some(c) = arg(&args, "--chunk") {
        options.chunk_splats = c.parse().context("--chunk")?;
    }
    if flag("--no-lobes") {
        // The material's base alone, as before the layers were carried.
        streams.lobes = LobeStreams::default();
    }
    if flag("--material-stats") {
        // The distinct materials a cloud carries: its raw values and the
        // words they pack to, with how many splats have each.
        let n = streams.count;
        let mut seen: BTreeMap<(u32, [u32; 3]), (usize, String)> = BTreeMap::new();
        for i in 0..n {
            let at = |v: &Vec<f32>, d: f32| v.get(i).copied().unwrap_or(d);
            let mark = |v: &Vec<u32>, m: f32| if v.get(i).is_some_and(|&b| b != 0) { m } else { 0.0 };
            let pbr = spark_lib::athc_build::pack_pbr(
                at(&streams.metallic, 0.0),
                at(&streams.roughness, 1.0),
                at(&streams.transmission, 0.0)
                    + mark(&streams.thin_walled, 2.0)
                    + mark(&streams.schlick_metal, 4.0),
            );
            let l = streams.lobes.at(n, i);
            let w = spark_lib::athc_build::pack_lobes(&l);
            let e = seen.entry((pbr, w)).or_insert_with(|| {
                (
                    0,
                    format!(
                        "metallic {} roughness {} transmission {} | {l:?}",
                        at(&streams.metallic, 0.0),
                        at(&streams.roughness, 1.0),
                        at(&streams.transmission, 0.0)
                    ),
                )
            });
            e.0 += 1;
        }
        let mut rows: Vec<_> = seen.into_iter().collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.1 .0));
        for ((pbr, w), (c, raw)) in rows.iter().take(12) {
            println!("{c:>9}  pbr {pbr:08x} lobes {:08x} {:08x} {:08x}  first: {raw}", w[0], w[1], w[2]);
        }
        println!("{} distinct (pbr, lobes) words", rows.len());
        return Ok(());
    }
    let mut packed = pack_streams(&streams, &options)?;
    let source_splats = streams.count;
    drop(streams);
    let drop_backs: Option<f32> = arg(&args, "--drop-backs")
        .map(|t| t.parse())
        .transpose()
        .context("--drop-backs")?;
    let mut dropped_backs = None;
    if let Some(t) = drop_backs {
        let (kept, dropped) = drop_hidden_backs(&packed, t)?;
        packed = kept;
        dropped_backs = Some(dropped);
    }
    if let Some(b) = arg(&args, "--box") {
        let v: Vec<f32> = b.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>().context("--box")?;
        if v.len() != 6 {
            bail!("--box takes x0,y0,z0,x1,y1,z1");
        }
        packed = crop_box(&packed, [v[0], v[1], v[2]], [v[3], v[4], v[5]]);
    }
    let fill: f32 = arg(&args, "--fill")
        .map_or(Ok(1.0), |f| f.parse())
        .context("--fill")?;
    let mut cell: Option<f32> = arg(&args, "--cell")
        .map(|c| c.parse())
        .transpose()
        .context("--cell")?;
    if let Some(t) = arg(&args, "--target") {
        cell = Some(cell_for_target(&packed, t.parse().context("--target")?));
    }
    if let Some(c) = cell {
        packed = reduce_cells(&packed, c, fill)?;
    }
    let thin: Option<f32> = arg(&args, "--thin")
        .map(|t| t.parse())
        .transpose()
        .context("--thin")?;
    if let Some(r) = thin {
        packed = reduce_thin(&packed, r)?;
    }
    let mut file = build_lod(&packed, &options)?;
    // v3 keeps the merged levels' whole coverage (athc::coverage_ratios);
    // a v2 file is written capped at 0.99, as athenea's.
    spark_lib::athc::uncap_levels(&mut file);
    let build_s = t.elapsed().as_secs_f32() - read_s;
    let compression = if flag("--gzip") {
        COMPRESSION_GZIP
    } else {
        COMPRESSION_NONE
    };
    let bytes = if flag("--v2") {
        if rig.is_some() {
            eprintln!("warning: a v2 file has no room for the skin; written without it");
        }
        file.write()?
    } else {
        write_v3_skinned(&file, compression, rig.as_ref())?
    };
    std::fs::write(output, &bytes)?;

    let h = &file.header;
    let mut report = json!({
        "source": input,
        "sources": sources,
        "sourceSplats": source_splats,
        "reduceCell": cell,
        "thin": thin,
        "dropBacks": drop_backs,
        "droppedBacks": dropped_backs,
        "fill": fill,
        "prim": prim_path,
        "output": output,
        "format": if flag("--v2") { "athc v2" } else if compression == COMPRESSION_GZIP { "athc v3 gzip" } else { "athc v3" },
        "bytes": bytes.len(),
        "splats": h.count,
        "dropped": packed.dropped,
        "levels": file.levels.iter().map(|(r, b)| json!([r, b.n])).collect::<Vec<_>>(),
        "chunks": h.chunks,
        "flags": h.flags,
        "shDegree": h.sh_degree(),
        "transferCount": file.extra.transfer_count,
        "shadowWords": file.extra.shadow_words,
        "curvature": file.has_curvature() && !flag("--v2"),
        "lobesWords": file.extra.lobes_words,
        "skin": rig.as_ref().filter(|_| !flag("--v2")).map(|r| json!({
            "joints": r.joints.len(),
            "influences": r.influences,
            "gradientWords": r.gradient_words,
            "skeleton": r.skeleton,
            "clips": r.clips.iter().map(|c| json!({
                "name": c.name,
                "samples": c.times.len(),
                "from": c.times.first(),
                "to": c.times.last(),
                "timeCodesPerSecond": c.time_codes_per_second,
            })).collect::<Vec<_>>(),
        })),
        "boundsMin": h.bounds_min,
        "boundsMax": h.bounds_max,
        "constants": constants,
        // The stage's up axis, as the cloud's layer says it: a TX transfer,
        // its cells and its field are directions of that stage, so a Z-up
        // cloud is relit in a turned frame (atheneaRelightPlugin `frame`).
        "upAxis": up_axis,
        "seconds": { "read": read_s, "build": build_s },
    });
    if !flag("--v2") {
        let layout = parse_v3(&bytes)?;
        let mut sections = serde_json::Map::new();
        for (k, s) in layout.sections.iter().enumerate() {
            let stored: u64 = layout.blocks.iter().map(|b| b.spans[k].stored as u64).sum();
            let raw: u64 = layout.blocks.iter().map(|b| b.spans[k].raw as u64).sum();
            sections.insert(
                format!("{:?}", s.id),
                json!({ "tier": s.tier, "bytesPerSplat": 4 * s.words, "stored": stored, "raw": raw }),
            );
        }
        report["sections"] = Json::Object(sections);
        let tiers: Vec<u64> = [1, 2, 3]
            .iter()
            .map(|&t| {
                layout
                    .blocks
                    .iter()
                    .map(|b| b.tier_range(&layout.sections, t).1)
                    .sum()
            })
            .collect();
        report["tierBytes"] = json!({ "1": tiers[0], "2": tiers[1], "3": tiers[2] });
        // What a whole-file fetch would cost over HTTP compression.
        if compression == COMPRESSION_NONE && flag("--gzip-estimate") {
            report["gzipWhole"] = json!(gzip(&bytes).len());
        }
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    if let Some(j) = arg(&args, "--json") {
        std::fs::write(j, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test/fixtures/athc/tx_cloud.usdc"
    );

    #[test]
    fn reports_the_layers_up_axis() {
        let data = openusd::usdc::read_file(FIXTURE).unwrap();
        // athenea's export writes the stage's: the fixture's is Y-up.
        assert_eq!(up_axis(data.as_ref()).as_deref(), Some("Y"));
    }

    #[test]
    fn reads_a_tx_particle_field() {
        let data = openusd::usdc::read_file(FIXTURE).unwrap();
        let prim = Prim::read(data.as_ref(), "/World/Splats").unwrap();
        let s = prim.streams().unwrap();
        assert_eq!(s.count, 3);
        assert_eq!(s.coefficients, 1);
        assert!(s.linear);
        assert_eq!(&s.rotations[..4], &[0.0, 0.0, 0.0, 1.0]); // (w, x, y, z) = (1, 0, 0, 0)
        assert_eq!(&s.rotations[4..8], &[0.0, 0.0, 0.70710677, 0.70710677]);
        assert_eq!(s.transfer_direct.len(), 3 * 16);
        assert_eq!(s.transfer_indirect.len(), 3 * 48);
        assert_eq!(s.transfer_reflected.len(), 3 * 48);
        assert_eq!(s.shadow_bits.len(), 3 * 8);
        assert_eq!(s.shadow_bits[0], u32::MAX);
        assert_eq!(s.transfer_direct[1], 0.5);
        let c = prim.constants();
        assert_eq!(c["ior"], json!(1.5));

        let options = BuildOptions::default();
        let packed = pack_streams(&s, &options).unwrap();
        assert_eq!((packed.block.n, packed.transfer_count), (3, 112));
        let file = build_lod(&packed, &options).unwrap();
        assert_eq!(file.extra.shadow_words, 8);
        let bytes = spark_lib::athc_v3::write_v3(&file, COMPRESSION_GZIP).unwrap();
        let back = spark_lib::athc_v3::read_v3(&bytes).unwrap();
        assert_eq!(back.splats().transfer.len(), 3 * 56);
    }
}
