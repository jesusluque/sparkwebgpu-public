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
//!          [--no-shadow] [--no-material] [--no-normals] [--no-curvature] [--max-sh 0|3|8|15]
//!          [--chunk 65536] [--gzip] [--v2] [--json out.json]
//!          [--add more.usdc]... [--only-prim TEXT]... [--exclude-prim TEXT]...
//!          [--thin RATIO | --target SPLATS | --cell SIDE [--fill 1.0]]
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
//! to its run's area), which keeps surfaces closed.
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
//! primvars:athenea:splat:transferDirect (float[], 16 a splat), transferIndirect
//!   (48), transferReflected (48)              transfer, 112 f16, flag bit 5
//! primvars:athenea:splat:shadowBits (int[], 8 a splat)   shadowBits
//! primvars:athenea:splat:curvature (float[], 3 a splat)  curvature, 3 f16
//!                                             (v3 section CURV; not in a v2)
//! cryptoObject, cryptoManifest, ior, relight  not in a .athc:
//!                                             reported in --json
//! ```

use std::collections::BTreeMap;

use anyhow::{anyhow, bail, Context, Result};
use openusd::sdf::{self, AbstractData, Value};
use serde_json::{json, Value as Json};
use spark_lib::athc_build::{
    build_lod, cell_for_target, pack_streams, reduce_cells, reduce_thin, BuildOptions,
    CloudStreams, TransferKeep,
};
use spark_lib::athc_v3::{gzip, parse_v3, write_v3, COMPRESSION_GZIP, COMPRESSION_NONE};

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

/// One prim's authored attribute defaults.
pub struct Prim {
    pub attributes: BTreeMap<String, Value>,
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
        for name in names {
            let at = sdf::path(format!("{prim}.{name}")).map_err(|e| anyhow!("{name}: {e}"))?;
            if let Some(v) = data
                .try_field(&at, "default")
                .map_err(|e| anyhow!("{name}: {e}"))?
            {
                attributes.insert(name, v.into_owned());
            }
        }
        Ok(Self { attributes })
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
            transfer_direct: self.floats(&["athenea:splat:transferDirect"])?,
            transfer_indirect: self.floats(&["athenea:splat:transferIndirect"])?,
            transfer_reflected: self.floats(&["athenea:splat:transferReflected"])?,
            shadow_bits: self.ints("athenea:splat:shadowBits")?,
            curvature: self.floats(&["athenea:splat:curvature"])?,
        })
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
        for name in [
            "athenea:splat:specularWeight",
            "athenea:splat:coatWeight",
            "athenea:splat:emission",
            "athenea:splat:transferZonal",
        ] {
            if self.get(name).is_some() {
                eprintln!("warning: {name} is not carried yet");
            }
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
    let data = openusd::usdc::read_file(input).with_context(|| format!("reading {input}"))?;
    let prim = Prim::read(data.as_ref(), prim_path)?;
    if flag("--list") {
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
    let mut streams = filtered(prim, &[])?;
    let mut sources = vec![json!({ "source": input, "splats": streams.count })];
    for added in args_all(&args, "--add") {
        // path::TEXT,TEXT drops those prims from this cloud only
        let (more, own) = added.split_once("::").unwrap_or((added, ""));
        let own: Vec<&str> = own.split(',').filter(|t| !t.is_empty()).collect();
        let data = openusd::usdc::read_file(more).with_context(|| format!("reading {more}"))?;
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
    let mut packed = pack_streams(&streams, &options)?;
    let source_splats = streams.count;
    drop(streams);
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
    let file = build_lod(&packed, &options)?;
    let build_s = t.elapsed().as_secs_f32() - read_s;
    let compression = if flag("--gzip") {
        COMPRESSION_GZIP
    } else {
        COMPRESSION_NONE
    };
    let bytes = if flag("--v2") {
        file.write()?
    } else {
        write_v3(&file, compression)?
    };
    std::fs::write(output, &bytes)?;

    let h = &file.header;
    let mut report = json!({
        "source": input,
        "sources": sources,
        "sourceSplats": source_splats,
        "reduceCell": cell,
        "thin": thin,
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
        "boundsMin": h.bounds_min,
        "boundsMax": h.bounds_max,
        "constants": constants,
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
        let bytes = write_v3(&file, COMPRESSION_GZIP).unwrap();
        let back = spark_lib::athc_v3::read_v3(&bytes).unwrap();
        assert_eq!(back.splats().transfer.len(), 3 * 56);
    }
}
