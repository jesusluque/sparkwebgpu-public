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
//!
//! Many clips: `--clip-dir DIR [--clips a,b,c|all]` adds the clips of a
//! folder of athenea's rigs (`<clip>_rig.usda`, an `over` of the cloud's
//! prim holding its `skinningXforms`; the older `lrt:` name is read too),
//! `--clip-skeleton layer.usdc[::/Skel]` remaps their joints by name from
//! that Skeleton's order to the cloud's, `--drop-own-clip` drops the clip
//! the cloud itself carries, and `--clip-files DIR` writes every clip as
//! `DIR/<clip>.atcl.gz` (`SkinClip::to_atcl`) and keeps only the first in
//! the `.athc`, for a page that fetches a clip when it plays it.

use std::collections::BTreeMap;

use anyhow::{anyhow, bail, Context, Result};
use openusd::sdf::{self, AbstractData, Value};
use serde_json::{json, Value as Json};
use spark_lib::athc_build::{
    build_lod, cell_for_target, crop_box_kept, drop_hidden_backs_kept, lod_order, pack_streams, pack_streams_kept,
    reduce_cells_runs, reduce_thin_runs, BuildOptions, CloudStreams, LobeStreams, TransferKeep, SH0,
};
use spark_lib::athl::SplatSources;
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
    // By its first bytes: a text layer whatever its name says.
    let mut head = [0u8; 5];
    let text = std::fs::File::open(path)
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut head))
        .is_ok()
        && &head == b"#usda";
    if text || path.ends_with(".usda") {
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
        Self::read_some(data, prim, None)
    }

    /// [`Prim::read`] of the attributes named (with or without `primvars:`)
    /// alone: a light layer's cloud is read for its colours and positions,
    /// not its transfer.
    pub fn read_some(data: &dyn AbstractData, prim: &str, only: Option<&[&str]>) -> Result<Self> {
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
            if let Some(only) = only {
                let bare = name.strip_prefix("primvars:").unwrap_or(&name);
                if !only.contains(&bare) {
                    continue;
                }
            }
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
        // athenea's name, or the older `lrt:` one its first rigs carry.
        let key = ["primvars:athenea:splat:skinningXforms", "primvars:lrt:splat:skinningXforms"]
            .into_iter()
            .find(|k| self.samples.contains_key(*k))
            .unwrap_or("primvars:athenea:splat:skinningXforms");
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

/// Adds the clips a folder of athenea's rigs holds (`<clip>_rig.usda`, each
/// an `over` of the cloud's prim with its `skinningXforms`; or `<clip>.usda`):
/// `names` a comma list, or `all` / none for every rig there, sorted. A rig
/// names no joints; `skeleton` (`layer[::/Skel/Prim]`, default prim
/// `/root/Bird/Bird`) gives the order its transforms index, and they are
/// remapped by name to the cloud's.
fn add_clip_dir(
    r: &mut AthcSkeleton,
    dir: &str,
    names: Option<&str>,
    skeleton: Option<&str>,
    prim_path: &str,
) -> Result<()> {
    let dir = std::path::Path::new(dir);
    let names: Vec<String> = match names {
        None | Some("all") => {
            let mut v: Vec<String> = std::fs::read_dir(dir)
                .with_context(|| format!("{}", dir.display()))?
                .filter_map(|e| e.ok()?.file_name().into_string().ok())
                .filter_map(|f| f.strip_suffix("_rig.usda").map(str::to_string))
                .collect();
            v.sort();
            v
        }
        Some(list) => list.split(',').map(|n| n.trim().to_string()).filter(|n| !n.is_empty()).collect(),
    };
    if names.is_empty() {
        bail!("{}: no clips", dir.display());
    }
    // The order the rigs' transforms are in -> the cloud's joint index.
    let remap: Option<Vec<usize>> = match skeleton {
        None => None,
        Some(s) => {
            let (layer, skel) = s.split_once("::").unwrap_or((s, "/root/Bird/Bird"));
            let data = read_layer(layer)?;
            let p = Prim::read(data.as_ref(), skel)?;
            let order: Vec<String> = match p.get("joints") {
                Some(Value::TokenVec(t)) => t.iter().map(|t| t.to_string()).collect(),
                Some(Value::StringVec(t)) => t.clone(),
                _ => bail!("{s}: no joints"),
            };
            if order.len() != r.joints.len() {
                bail!("{s}: {} joints, the cloud {}", order.len(), r.joints.len());
            }
            let m = order
                .iter()
                .map(|n| r.joints.iter().position(|j| j == n).ok_or_else(|| anyhow!("{s}: joint {n} not in the cloud")))
                .collect::<Result<Vec<_>>>()?;
            let moved = m.iter().enumerate().filter(|(i, j)| i != *j).count();
            eprintln!("{s}: {} joints, {moved} in another order than the cloud's", m.len());
            (moved > 0).then_some(m)
        }
    };
    for name in names {
        let rig = dir.join(format!("{name}_rig.usda"));
        let path = if rig.exists() { rig } else { dir.join(format!("{name}.usda")) };
        let path = path.to_string_lossy().to_string();
        let data = read_layer(&path)?;
        let p = Prim::read(data.as_ref(), prim_path)?;
        if !p.joints().is_empty() && p.joints() != r.joints {
            bail!("{path}: another skeleton's joints");
        }
        let mut c = p.clip(&name, time_codes_per_second(data.as_ref()))?;
        if let Some(m) = &remap {
            let j = r.joints.len();
            let mut x = vec![0.0; c.xforms.len()];
            for s in 0..c.times.len() {
                for (from, &to) in m.iter().enumerate() {
                    x[(s * j + to) * 16..(s * j + to + 1) * 16]
                        .copy_from_slice(&c.xforms[(s * j + from) * 16..(s * j + from + 1) * 16]);
                }
            }
            c.xforms = x;
        }
        r.clips.push(c);
        r.check().with_context(|| path.clone())?;
    }
    Ok(())
}

/// A clip, for the report: its length, how far its last pose is from its
/// first (a loop's seam), and how far it carries the bird: where the
/// pelvis joint's skinning transform takes the bind-space origin (the
/// sparrow's sits between its feet), against the first sample.
fn clip_report(r: &AthcSkeleton, c: &SkinClip) -> Json {
    let j = r.joints.len();
    let n = c.times.len();
    let per = j * 16;
    let first = &c.xforms[..per];
    let last = &c.xforms[(n - 1) * per..n * per];
    // The seam: the largest change of a rotation element, and of a translation.
    let (mut rot, mut tr) = (0.0f32, 0.0f32);
    for k in 0..per {
        let d = (first[k] - last[k]).abs();
        if k % 16 >= 12 { tr = tr.max(d) } else { rot = rot.max(d) }
    }
    let pelvis = r.joints.iter().position(|n| n.ends_with("Pelvis")).unwrap_or(0);
    // p' = p M (USD rows): the origin's image is the translation row.
    let origin = |s: usize| {
        let m = &c.xforms[(s * j + pelvis) * 16..(s * j + pelvis + 1) * 16];
        [m[12], m[13], m[14]]
    };
    let o0 = origin(0);
    let dist = |a: [f32; 3]| ((a[0] - o0[0]).powi(2) + (a[1] - o0[1]).powi(2) + (a[2] - o0[2]).powi(2)).sqrt();
    let travel = dist(origin(n - 1));
    let excursion = (0..n).map(|s| dist(origin(s))).fold(0.0f32, f32::max);
    let span = c.times[n - 1] - c.times[0];
    json!({
        "name": c.name,
        "samples": n,
        "from": c.times.first(),
        "to": c.times.last(),
        "timeCodesPerSecond": c.time_codes_per_second,
        "seconds": span / c.time_codes_per_second.max(1e-6),
        "seamRotation": rot,
        "seamTranslation": tr,
        "rootTravel": travel,
        "rootExcursion": excursion,
    })
}

/// The linear colours (0.5 + SH0 dc, RGB a splat) of a light layer's
/// clouds, one a source cloud, masked as the base's streams were, and a
/// check that they line up: the same count, and the same positions.
fn read_layer_colours(
    files: &[&str],
    prim_path: &str,
    masks: &[(Option<Vec<bool>>, usize)],
    streams: &CloudStreams,
) -> Result<(Vec<f32>, Json)> {
    let mut out = Vec::with_capacity(streams.count * 3);
    let mut worst = 0f32;
    let mut max = 0f32;
    let mut negative = 0usize;
    let mut unbaked = 0usize;
    for (file, (mask, offset)) in files.iter().zip(masks) {
        let data = read_layer(file)?;
        let p = Prim::read_some(
            data.as_ref(),
            prim_path,
            Some(&[
                "positions",
                "positionsh",
                "radiance:sphericalHarmonicsCoefficients",
                "radiance:sphericalHarmonicsCoefficientsh",
                "athenea:splat:linear",
            ]),
        )?;
        let positions = p.floats(&["positions", "positionsh"])?;
        let count = positions.len() / 3;
        let sh = p.floats(&["radiance:sphericalHarmonicsCoefficients", "radiance:sphericalHarmonicsCoefficientsh"])?;
        if count == 0 || sh.len() % (3 * count) != 0 || sh.is_empty() {
            bail!("{file}: {count} splats and {} SH values", sh.len());
        }
        if p.bool("athenea:splat:linear") != streams.linear {
            eprintln!("warning: {file}: linear is {}, the base's {}", p.bool("athenea:splat:linear"), streams.linear);
        }
        let per = sh.len() / count;
        if let Some(m) = mask {
            if m.len() != count {
                bail!("{file}: {count} splats, its base cloud {} (not the same gaussians)", m.len());
            }
        }
        let mut k = *offset;
        for i in 0..count {
            if mask.as_ref().is_some_and(|m| !m[i]) {
                continue;
            }
            if k >= streams.count {
                bail!("{file}: more splats than its base cloud");
            }
            // A gaussian the bake did not reach keeps an all-zero SH, which
            // reads as 0.5 (mid grey, 5000 nits on a 10 000-nit lamp): no
            // light. A baked black is dc = -0.5 / SH0, never 0.
            let empty = sh[i * per..i * per + 3].iter().all(|v| *v == 0.0);
            if empty {
                unbaked += 1;
            }
            for c in 0..3 {
                worst = worst.max((positions[i * 3 + c] - streams.positions[k * 3 + c]).abs());
                let v = if empty { 0.0 } else { 0.5 + SH0 * sh[i * per + c] };
                if v < 0.0 {
                    negative += 1;
                }
                max = max.max(v);
                out.push(v.max(0.0));
            }
            k += 1;
        }
        let expect = masks.iter().map(|(_, o)| *o).find(|&o| o > *offset).unwrap_or(streams.count);
        if k != expect {
            bail!("{file}: {} splats after the mask, its base cloud {}", k - offset, expect - offset);
        }
    }
    if worst > 1e-5 {
        bail!("the layer {:?} is not the base's gaussians: positions differ by up to {worst}", files);
    }
    if unbaked > 0 {
        eprintln!("{files:?}: {unbaked} gaussians with no SH (not baked): no light");
    }
    Ok((out, json!({ "files": files, "maxPositionError": worst, "max": max, "negativeClamped": negative, "unbaked": unbaked })))
}

/// A group's radiance in a `.lights.usda`: `athenea:lightGroup:radiance`,
/// or athenea's `emissionLuminance` (the per-group bakes are per unit of
/// it, the material's emission colour included). A plain text scan of the
/// group's `def Scope "NAME"` block.
fn sidecar_radiance(text: &str, name: &str) -> Option<f32> {
    let head = format!("def Scope \"{name}\"");
    let at = text.find(&head)? + head.len();
    let rest = &text[at..];
    let end = rest.find("def ").unwrap_or(rest.len());
    for line in rest[..end].lines() {
        let line = line.trim();
        for key in ["athenea:lightGroup:radiance =", "athenea:lightGroup:emissionLuminance ="] {
            if let Some(v) = line.split_once(key).map(|(_, v)| v.trim()) {
                if let Ok(v) = v.split_whitespace().next().unwrap_or("").parse::<f32>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// A stand-in for one of athenea's per-group clouds (tests, before its
/// bakes land): a `.usda` layer with the base's positions, in its order,
/// and as linear colour a lamp's glow from a point (`--emit x,y,z,r`): 1 at
/// the point, falling to 0 at r (squared), plus a faint bounce to 2r.
fn fake_layer(prim: &Prim, prim_path: &str, up: Option<&str>, out: &str, emit: &str) -> Result<()> {
    use std::fmt::Write as _;
    let e: Vec<f32> = emit.split(',').map(|v| v.trim().parse()).collect::<Result<_, _>>().context("--emit")?;
    if e.len() != 4 {
        bail!("--emit x,y,z,r");
    }
    let positions = prim.floats(&["positions", "positionsh"])?;
    let n = positions.len() / 3;
    let (mut pos, mut sh) = (String::with_capacity(n * 40), String::with_capacity(n * 40));
    let mut lit = 0;
    for i in 0..n {
        let p = &positions[i * 3..i * 3 + 3];
        let d = ((p[0] - e[0]).powi(2) + (p[1] - e[1]).powi(2) + (p[2] - e[2]).powi(2)).sqrt();
        let core = (1.0 - d / e[3]).max(0.0).powi(2);
        let bounce = 0.01 * (1.0 - d / (2.0 * e[3])).max(0.0);
        let v = core + bounce;
        if v > 0.0 {
            lit += 1;
        }
        let rgb = [v, 0.9 * v, 0.75 * v].map(|c| (c - 0.5) / SH0);
        let sep = if i == 0 { "" } else { ", " };
        write!(pos, "{sep}({:?}, {:?}, {:?})", p[0], p[1], p[2])?;
        write!(sh, "{sep}({:?}, {:?}, {:?})", rgb[0], rgb[1], rgb[2])?;
    }
    let names: Vec<&str> = prim_path.trim_start_matches('/').split('/').collect();
    let mut text = format!("#usda 1.0\n(\n    upAxis = \"{}\"\n)\n\n", up.unwrap_or("Y"));
    for (k, name) in names.iter().enumerate() {
        let _ = writeln!(text, "{}def Xform \"{name}\"\n{}{{", "    ".repeat(k), "    ".repeat(k));
    }
    let pad = "    ".repeat(names.len());
    let _ = writeln!(text, "{pad}bool primvars:athenea:splat:linear = true");
    let _ = writeln!(text, "{pad}point3f[] positions = [{pos}]");
    let _ = writeln!(text, "{pad}float3[] radiance:sphericalHarmonicsCoefficients = [{sh}]");
    for k in (0..names.len()).rev() {
        let _ = writeln!(text, "{}}}", "    ".repeat(k));
    }
    std::fs::write(out, text)?;
    eprintln!("{out}: {n} splats, {lit} lit from ({}, {}, {}) r {}", e[0], e[1], e[2], e[3]);
    Ok(())
}

struct LayerOptions {
    /// A block of 256 is kept when a value passes this (radiance units).
    threshold: f32,
    /// Splats whose largest component (radiance units) is under this are
    /// zeroed first: the bake's grainy faint indirect, dropped.
    floor: f32,
    /// A layer to compare with the file's own base colour (a layer that is
    /// the base cloud itself: an alignment check).
    verify: Option<String>,
}

/// The `.athl` of the built cloud: each layer through the splats' sources
/// (`track`), scaled to its group's radiance (f16 keeps the faint bounce:
/// per unit of a 10 000-nit lamp it would be subnormal), as kind 0 over
/// the virtual order (merged nodes their splats' weighted mean).
fn write_athl(
    file: &spark_lib::athc::AthcFile,
    athc_bytes: &[u8],
    track: &SplatSources,
    layers: &[(String, Vec<f32>, Json)],
    sidecar: Option<&str>,
    o: &LayerOptions,
    path: &str,
) -> Result<Json> {
    use spark_lib::athc::{high_half, low_half, VirtualTree};
    use spark_lib::athl::{cloud_hash, sparse_layers, validate, virtual_values, AthlFile, AthlGroup, KIND_INDIRECT};
    let n = file.header.count as usize;
    if track.len() != n {
        bail!("tracked {} splats, the file has {n}", track.len());
    }
    let tree = VirtualTree::of_file(file, true)?;
    let mut athl = AthlFile {
        element_count: tree.splat_base + n as u32,
        merged: tree.merged,
        splat_base: tree.splat_base,
        splat_count: n as u32,
        cloud_hash: cloud_hash(athc_bytes),
        ..Default::default()
    };
    let mut bake = 0xcbf2_9ce4_8422_2325u64;
    let mut reports = Vec::new();
    for (k, (name, values, check)) in layers.iter().enumerate() {
        let per = track.gather(values, 3)?;
        let mut report = check.clone();
        if o.verify.as_deref() == Some(name) {
            let splats = file.splats();
            let (mut abs, mut rel) = (0f32, 0f32);
            for i in 0..n {
                let s = &splats.shape[4 * i..4 * i + 4];
                let base = [high_half(s[2]), low_half(s[3]), high_half(s[3])];
                for c in 0..3 {
                    let d = (per[3 * i + c] - base[c]).abs();
                    abs = abs.max(d);
                    rel = rel.max(d / base[c].abs().max(1e-3));
                }
            }
            report["verify"] = json!({ "maxAbs": abs, "maxRel": rel });
            eprintln!("verify {name}: layer vs the file's base colour, max abs {abs:.3e}, max rel {rel:.3e}");
        }
        let radiance = sidecar.and_then(|t| sidecar_radiance(t, name)).unwrap_or(1.0);
        let mut scaled: Vec<f32> = per.iter().map(|v| v * radiance).collect();
        let mut floored = 0usize;
        if o.floor > 0.0 {
            for px in scaled.chunks_mut(3) {
                if px.iter().all(|v| v.abs() < o.floor) && px.iter().any(|v| *v != 0.0) {
                    px.fill(0.0);
                    floored += 1;
                }
            }
        }
        let lit = scaled.chunks(3).filter(|p| p.iter().any(|v| *v != 0.0)).count();
        let virt = virtual_values(file, &tree, &scaled, 3)?;
        let sparse = sparse_layers(k as u16, KIND_INDIRECT, 3, &virt, o.threshold)?;
        let blocks: usize = sparse.iter().map(|l| l.blocks.len()).sum();
        let total = (athl.element_count as usize).div_ceil(spark_lib::athl::BLOCK_SPLATS as usize);
        report["group"] = json!(name);
        report["radiance"] = json!(radiance);
        report["litSplats"] = json!(lit);
        report["flooredSplats"] = json!(floored);
        report["blocks"] = json!(blocks);
        report["blockFraction"] = json!(blocks as f64 / total.max(1) as f64);
        reports.push(report);
        athl.layers.extend(sparse);
        for b in name.bytes().chain(radiance.to_le_bytes()) {
            bake = (bake ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
        athl.groups.push(AthlGroup {
            name: name.clone(),
            profile: -1,
            // The bake went through the lenses: no tint over the layer.
            tint: [1.0; 3],
            axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            radiance,
            ..Default::default()
        });
    }
    athl.bake_hash = bake;
    // Sorted by chunk, group, kind (one byte range a page).
    athl.layers.sort_by_key(|l| (l.chunk, l.group, l.kind));
    validate(&athl)?;
    let bytes = athl.write()?;
    std::fs::write(path, &bytes)?;
    Ok(json!({
        "athl": path,
        "bytes": bytes.len(),
        "cloudHash": format!("{:016x}", athl.cloud_hash),
        "merged": tree.merged,
        "splatBase": tree.splat_base,
        "threshold": o.threshold,
        "floor": o.floor,
        "groups": reports,
    }))
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
        "--clip-dir",
        "--clips",
        "--clip-skeleton",
        "--clip-files",
        "--light-layer",
        "--athl",
        "--lights-usda",
        "--light-threshold",
        "--light-floor",
        "--light-verify",
        "--fake-layer",
        "--emit",
        "--dump-layers",
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
    if let Some(out) = arg(&args, "--fake-layer") {
        return fake_layer(&prim, prim_path, up_axis.as_deref(), out, arg(&args, "--emit").unwrap_or("0,0,0,0.5"));
    }
    let Some(output) = paths.get(1) else {
        bail!("no output path")
    };
    let only = args_all(&args, "--only-prim");
    let exclude = args_all(&args, "--exclude-prim");
    let transfer_arg = arg(&args, "--transfer").unwrap_or("full");
    // A transfer of its direct half only: the other halves need not be read.
    let direct_only = matches!(transfer_arg, "none" | "9" | "16");
    let filtered = |mut prim: Prim, more_excluded: &[&str]| -> Result<(CloudStreams, Option<Vec<bool>>)> {
        if direct_only {
            for half in ["transferIndirect", "transferReflected"] {
                prim.attributes
                    .remove(&format!("primvars:athenea:splat:{half}"));
            }
        }
        let count = prim.floats(&["positions", "positionsh"])?.len() / 3;
        let exclude: Vec<&str> = exclude.iter().chain(more_excluded).copied().collect();
        let mask = prim_mask(&prim, count, &only, &exclude)?;
        if let Some(keep) = &mask {
            prim.retain(keep);
        }
        Ok((prim.streams()?, mask))
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
    let (mut streams, mask) = filtered(prim, &[])?;
    let mut sources = vec![json!({ "source": input, "splats": streams.count })];
    // Each source cloud's prim mask and where its splats start in the streams.
    let mut masks: Vec<(Option<Vec<bool>>, usize)> = vec![(mask, 0)];
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
    if let Some(dir) = arg(&args, "--clip-dir") {
        let Some(r) = rig.as_mut() else { bail!("--clip-dir on a cloud nothing carries") };
        add_clip_dir(r, dir, arg(&args, "--clips"), arg(&args, "--clip-skeleton"), prim_path)?;
    }
    if flag("--drop-own-clip") {
        let Some(r) = rig.as_mut() else { bail!("--drop-own-clip on a cloud nothing carries") };
        if r.clips.len() < 2 {
            bail!("--drop-own-clip leaves no clip (add some with --clip-dir or --add-clip)");
        }
        r.clips.remove(0);
    }
    // Every clip in a file of its own (ATCL, gzipped), the .athc keeping the
    // first alone: a page fetches a clip when it plays it.
    let all_clips: Vec<Json> = rig
        .as_ref()
        .map(|r| r.clips.iter().map(|c| clip_report(r, c)).collect())
        .unwrap_or_default();
    let mut clip_files: BTreeMap<String, (String, usize)> = BTreeMap::new();
    if let Some(dir) = arg(&args, "--clip-files") {
        let Some(r) = rig.as_mut() else { bail!("--clip-files on a cloud nothing carries") };
        std::fs::create_dir_all(dir)?;
        for c in &r.clips {
            let file = format!("{}.atcl.gz", c.name);
            let bytes = gzip(&c.to_atcl(r.joints.len()));
            std::fs::write(std::path::Path::new(dir).join(&file), &bytes)?;
            clip_files.insert(c.name.clone(), (file, bytes.len()));
        }
        r.clips.truncate(1);
    }
    for added in args_all(&args, "--add") {
        // path::TEXT,TEXT drops those prims from this cloud only
        let (more, own) = added.split_once("::").unwrap_or((added, ""));
        let own: Vec<&str> = own.split(',').filter(|t| !t.is_empty()).collect();
        let data = read_layer(more)?;
        let (s, mask) = filtered(Prim::read(data.as_ref(), prim_path)?, &own)?;
        sources.push(json!({ "source": more, "splats": s.count }));
        masks.push((mask, streams.count));
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
    // athenea's per-group light clouds (`--light-layer NAME=a.usdc[,b.usdc]`,
    // one file a source cloud, in the order of the input and its --add):
    // the same gaussians in the same order, their linear colours the group's
    // light per unit radiance. Read now, masked as the streams were.
    let layer_specs = args_all(&args, "--light-layer");
    let mut layers: Vec<(String, Vec<f32>, Json)> = Vec::new();
    for spec in &layer_specs {
        let (name, files) = spec.split_once('=').ok_or_else(|| anyhow!("--light-layer NAME=file.usdc[,more.usdc]"))?;
        let files: Vec<&str> = files.split(',').collect();
        if files.len() != masks.len() {
            bail!("--light-layer {name}: {} files for {} source clouds", files.len(), masks.len());
        }
        let (values, check) = read_layer_colours(&files, prim_path, &masks, &streams)?;
        // Diagnostics: the base's positions and this layer, f32 x y z r g b a splat.
        if let Some(dir) = arg(&args, "--dump-layers") {
            let mut out = Vec::with_capacity(streams.count * 24);
            for i in 0..streams.count {
                for v in streams.positions[3 * i..3 * i + 3].iter().chain(&values[3 * i..3 * i + 3]) {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            std::fs::write(std::path::Path::new(dir).join(format!("{name}.f32")), out)?;
        }
        layers.push((name.to_string(), values, check));
    }
    if layers.len() > spark_lib::athl::MAX_GROUPS {
        bail!("at most {} light groups", spark_lib::athl::MAX_GROUPS);
    }
    let (mut packed, kept) = if layers.is_empty() {
        (pack_streams(&streams, &options)?, Vec::new())
    } else {
        pack_streams_kept(&streams, &options)?
    };
    let source_splats = streams.count;
    drop(streams);
    // Where each built splat comes from, for the layers.
    let mut track = (!layers.is_empty()).then(|| SplatSources::identity(source_splats).select(&kept)).transpose()?;
    let drop_backs: Option<f32> = arg(&args, "--drop-backs")
        .map(|t| t.parse())
        .transpose()
        .context("--drop-backs")?;
    let mut dropped_backs = None;
    if let Some(t) = drop_backs {
        let before = packed.block.n;
        let (kept, keep) = drop_hidden_backs_kept(&packed, t)?;
        packed = kept;
        dropped_backs = Some(before - keep.len());
        if let Some(s) = track.as_mut() {
            *s = s.select(&keep)?;
        }
    }
    if let Some(b) = arg(&args, "--box") {
        let v: Vec<f32> = b.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>().context("--box")?;
        if v.len() != 6 {
            bail!("--box takes x0,y0,z0,x1,y1,z1");
        }
        let (cropped, keep) = crop_box_kept(&packed, [v[0], v[1], v[2]], [v[3], v[4], v[5]]);
        packed = cropped;
        if let Some(s) = track.as_mut() {
            *s = s.select(&keep)?;
        }
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
        let (reduced, runs) = reduce_cells_runs(&packed, c, fill)?;
        packed = reduced;
        if let Some(s) = track.as_mut() {
            *s = s.merge(&runs.members, &runs.starts, &runs.weights)?;
        }
    }
    let thin: Option<f32> = arg(&args, "--thin")
        .map(|t| t.parse())
        .transpose()
        .context("--thin")?;
    if let Some(r) = thin {
        let (reduced, runs) = reduce_thin_runs(&packed, r)?;
        packed = reduced;
        if let Some(s) = track.as_mut() {
            *s = s.merge(&runs.members, &runs.starts, &runs.weights)?;
        }
    }
    if let Some(s) = track.as_mut() {
        *s = s.select(&lod_order(&packed, &options))?;
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
    let lights = match &track {
        Some(track) => {
            let athl_path = arg(&args, "--athl").map(str::to_string).unwrap_or_else(|| {
                format!("{}.lights.athl", output.trim_end_matches(".athc"))
            });
            let sidecar = arg(&args, "--lights-usda").map(std::fs::read_to_string).transpose()?;
            let opts = LayerOptions {
                threshold: arg(&args, "--light-threshold").map_or(Ok(1e-4), |v| v.parse()).context("--light-threshold")?,
                floor: arg(&args, "--light-floor").map_or(Ok(0.0), |v| v.parse()).context("--light-floor")?,
                verify: arg(&args, "--light-verify").map(str::to_string),
            };
            let report = write_athl(&file, &bytes, track, &layers, sidecar.as_deref(), &opts, &athl_path)?;
            Some(report)
        }
        None => None,
    };

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
            "clips": all_clips.iter().map(|c| {
                let mut c = c.clone();
                if let Some((file, bytes)) = c["name"].as_str().and_then(|n| clip_files.get(n)) {
                    c["file"] = json!(file);
                    c["fileBytes"] = json!(bytes);
                }
                c
            }).collect::<Vec<_>>(),
            "embeddedClips": r.clips.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
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
    if let Some(l) = lights {
        report["lights"] = l;
    }
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
