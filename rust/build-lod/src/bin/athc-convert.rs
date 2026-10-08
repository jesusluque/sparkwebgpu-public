//! .athc v2 <-> v3 (rust/spark-lib/src/athc_v3.rs, docs/docs/athc-v3.md).
//!
//!   athc-convert in.athc out.athc [--gzip]   v1/v2 to v3 (sections by tier: 1 splats, 2 material, 3 relight)
//!   athc-convert in.athc out.athc --v2       v3 (or v2) to v2
//!   athc-convert in.athc --info              headers, sections, tier sizes
//!   --coverage            the merged levels' whole coverage (athc::uncap_levels; v3 only)
//!   --keep-levels N       the levels from the coarsest down to the Nth, whose
//!                         groups become the splats (athc::truncate_levels)
//!   --keep-splats S       the same, N the finest level with at most S groups
//!   --creases D           with --keep-*: groups whose normals spread more than D
//!                         (athc::normal_spread; 0.03 ~ a 28 degree crease) are
//!                         replaced by their children, down to the splats (athc::truncate_creases)
//!   --crease-depth N      at most N levels below the cut (default: down to the splats)
//!   --rebuild-frame SEED  first build the levels again from the splats (athc_build::build_lod)
//!                         in an octree frame turned and shifted by SEED (0: world-aligned)
//!   --planes              each section in the encoding that stores it smallest (athc_v3 write_v3_smallest)
//!   --encoding E          encoding E (0 words, 1 byte planes, 2 delta planes) for every section
//!   --athl IN --athl-out OUT   carry the cloud's light layers (`.athl`, usd-athc --light-layer) to
//!                         the output: each element of a cut the weighted mean of the original
//!                         splats under it, merged nodes as before; the cloud hash re-stamped
//!   --athl-threshold T    a block is kept when a value passes T (default 0: any light)
//!   --error               with --keep-splats S: merge the splats down to S by an image-space
//!                         error (athc_merge::error_cut: greedy pair merges, Runnalls cost x
//!                         on-screen weight, never across glass/thin/Schlick/mirror/joint or
//!                         normals past 60 degrees), the levels above an error-driven tree too
//!   --error-levels        the levels built again as an error-driven tree, every splat kept
//!                         (athc_merge::error_levels; the splats reordered depth first)
//!   --error-orbit X,Y,Z,D[,HFOV,WIDTH]  where the error is seen from: an orbit at D around
//!                         X,Y,Z (default: three bound radii around the bounds' centre), HFOV
//!                         degrees over WIDTH pixels (39.6, 1920). Give a scene's parts the same
//!   --error-lambda L      colour weight (128); --error-material M (32); --error-normal N (0)
//!   --error-widen F       the cut's clusters widened by F x their own cell edge (0)
//!   --error-ratio R       groups of a level over the next coarser (4)
//!   --error-max E         with --error: merge nothing dearer than E (the same E over a scene's
//!                         parts seen from the same orbit spends the splats where the error is;
//!                         --keep-splats is then a floor); --error-curve prints splats left and
//!                         the dearest merge after each pass
//!   --lod-sizes           write each merged node's LoD size by its error (athc_lod_error::
//!                         level_lod_sizes: the geometric size scaled up by the colour, material,
//!                         normal and thickness error of the splats under it, monotone), the v3
//!                         section LODS that Spark's traversal reads in place of the geometric
//!                         size; a file that has them keeps them (recomputed after a cut) unless
//!                         --no-lod-sizes; --lod-size-opts k=v,... (lambda, material, normal,
//!                         thick, shape, v0, gamma, hi) over LodSizeOptions::default()

use anyhow::{bail, Context, Result};
use spark_lib::athl::{cloud_hash, splat_weights, sparse_layers, validate, virtual_values_weighted, AthlFile};
use spark_lib::athc_build::{build_lod, packed_of, BuildOptions};
use spark_lib::athc_lod_error::{with_lod_sizes, LodSizeOptions};
use spark_lib::athc_merge::{error_cut_to, error_levels, ErrorOptions, ErrorView};
use spark_lib::athc::{cut_sources, truncate_creases, truncate_levels, uncap_levels, AthcFile, VirtualTree};
use spark_lib::athc_v3::{
    parse_v3, read_v3, read_v3_skeleton, write_v3_encoded, write_v3_full, write_v3_smallest_with, SectionId, ATH3_MAGIC, COMPRESSION_GZIP, COMPRESSION_NONE,
};

fn read_any(bytes: &[u8]) -> Result<AthcFile> {
    if bytes.len() >= 4 && u32::from_le_bytes(bytes[..4].try_into().unwrap()) == ATH3_MAGIC {
        read_v3(bytes)
    } else {
        AthcFile::read(bytes)
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| args.iter().any(|a| a == f);
    // Paths: what is neither a flag nor a --keep-* flag's value.
    let paths: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|&(i, a)| {
            !a.starts_with("--") && !(i > 0 && (args[i - 1].starts_with("--keep-") || args[i - 1] == "--encoding" || args[i - 1] == "--creases" || args[i - 1] == "--crease-depth" || args[i - 1] == "--rebuild-frame" || args[i - 1].starts_with("--athl") || args[i - 1] == "--lod-size-opts" || (args[i - 1].starts_with("--error-") && args[i - 1] != "--error-levels")))
        })
        .map(|(_, a)| a)
        .collect();
    let Some(input) = paths.first() else {
        bail!("usage: athc-convert in.athc out.athc [--gzip | --v2] | athc-convert in.athc --info");
    };
    let bytes = std::fs::read(input)?;
    let mut file = read_any(&bytes)?;
    let had_lod_sizes = file.levels.iter().any(|(_, b)| !b.lod_size.is_empty());
    let text = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1)).cloned();
    let athl_io = match (text("--athl"), text("--athl-out")) {
        (Some(i), Some(o)) => Some((i, o)),
        (None, None) => None,
        _ => bail!("--athl and --athl-out go together"),
    };
    if athl_io.is_some() && flag("--rebuild-frame") {
        bail!("--rebuild-frame reorders the splats: not with --athl");
    }
    // The uncut cloud, for the layers, and what each output element stands for.
    let original = athl_io.as_ref().map(|_| file.clone());
    let mut ranges: Option<Vec<[u32; 2]>> = None;
    // The original splats (file order) each output element stands for, when
    // not a run of them (an error-driven tree reorders).
    let mut lists: Option<Vec<Vec<u32>>> = None;
    let cut_to = if flag("--error") {
        Some(text("--keep-splats").ok_or_else(|| anyhow::anyhow!("--error goes with --keep-splats N"))?.parse::<usize>()?)
    } else {
        None
    };
    if cut_to.is_some() || flag("--error-levels") {
        let floats = |f: &str| -> Result<Option<Vec<f64>>> {
            text(f).map(|v| v.split(',').map(|x| x.parse::<f64>().map_err(anyhow::Error::from)).collect()).transpose()
        };
        let view = match floats("--error-orbit")? {
            Some(v) if v.len() >= 4 => ErrorView::orbit([v[0], v[1], v[2]], v[3], *v.get(4).unwrap_or(&39.6), *v.get(5).unwrap_or(&1920.0)),
            Some(_) => bail!("--error-orbit X,Y,Z,D[,HFOV,WIDTH]"),
            None => ErrorView::around(file.header.bounds_min, file.header.bounds_max),
        };
        let mut o = ErrorOptions::new(view);
        let num = |f: &str, d: f64| -> Result<f64> { Ok(text(f).map(|v| v.parse::<f64>()).transpose()?.unwrap_or(d)) };
        o.lambda = num("--error-lambda", o.lambda)?;
        o.lambda_material = num("--error-material", o.lambda_material)?;
        o.lambda_normal = num("--error-normal", o.lambda_normal)?;
        o.widen = num("--error-widen", o.widen as f64)? as f32;
        o.level_ratio = num("--error-ratio", o.level_ratio)?;
        o.threads = num("--error-threads", o.threads as f64)? as usize;
        let t = std::time::Instant::now();
        let before = file.header.count;
        let (out, tree) = match cut_to {
            Some(k) => error_cut_to(&file, k, num("--error-max", f64::INFINITY)?, &o)?,
            None => error_levels(&file, &o)?,
        };
        if let Some(c) = &tree.cut {
            println!("error cut: {} -> {} splats in {} passes, dearest merge {:.3e}, {} past the strict rules", c.from, c.to, c.passes, c.last_cost, c.relaxed);
            if flag("--error-curve") {
                let pts: Vec<String> = c.curve.iter().map(|(n, e)| format!("{n}:{e:.4e}")).collect();
                println!("error curve: {}", pts.join(" "));
            }
        }
        for r in &tree.runs {
            println!("  level step {} -> {} in {} passes (dearest {:.3e}, {} relaxed)", r.from, r.to, r.passes, r.last_cost, r.relaxed);
        }
        println!("error-driven levels: groups {:?}, extent {:.4}, {:.1}s", tree.groups, tree.extent, t.elapsed().as_secs_f32());
        lists = Some(tree.sources);
        file = out;
        println!("kept {} levels: {} splats (of {})", file.levels.len(), file.header.count, before);
    }
    if let Some(seed) = args.iter().position(|a| a == "--rebuild-frame").and_then(|i| args.get(i + 1)) {
        let o = BuildOptions { frame_seed: seed.parse()?, chunk_splats: file.header.chunk_splats, ..Default::default() };
        let before: Vec<usize> = file.levels.iter().map(|(_, b)| b.n).collect();
        file = build_lod(&packed_of(&file), &o)?;
        let after: Vec<usize> = file.levels.iter().map(|(_, b)| b.n).collect();
        println!("levels built again (frame {seed}): groups {before:?} -> {after:?}");
    }
    let value = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1)).map(|v| v.parse::<usize>());
    let keep = if lists.is_some() { None } else { match (value("--keep-levels"), value("--keep-splats")) {
        (Some(n), _) => Some(n?),
        (None, Some(s)) => {
            let s = s?;
            let k = file.levels.iter().rposition(|(_, b)| b.n <= s).filter(|&k| k > 0);
            Some(k.ok_or_else(|| anyhow::anyhow!("no level below the coarsest has at most {s} groups"))?)
        }
        _ => None,
    } };
    let creases = args.iter().position(|a| a == "--creases").and_then(|i| args.get(i + 1)).map(|v| v.parse::<f32>());
    if let Some(k) = keep {
        let before = file.header.count;
        if let Some(d) = creases {
            let depth = value("--crease-depth").transpose()?.unwrap_or(usize::MAX);
            let (cut, stats) = truncate_creases(&file, k, d?, depth)?;
            if athl_io.is_some() {
                ranges = Some(stats.sources.clone());
            }
            file = cut;
            println!("crease-aware cut: merged per level from {}: {:?}, real splats {}", k, stats.per_level, stats.splats);
        } else {
            if athl_io.is_some() {
                ranges = Some(cut_sources(&file, k));
            }
            file = truncate_levels(&file, k)?;
        }
        println!("kept {} levels: {} splats (of {})", k, file.header.count, before);
    } else if flag("--coverage") {
        uncap_levels(&mut file);
    }
    if flag("--no-lod-sizes") {
        for (_, b) in file.levels.iter_mut() {
            b.lod_size.clear();
        }
    } else if flag("--lod-sizes") || had_lod_sizes {
        let o = LodSizeOptions::parse(&text("--lod-size-opts").unwrap_or_default())?;
        file = with_lod_sizes(&file, &o);
        println!("LoD sizes by error for {} merged nodes", file.levels.iter().map(|(_, b)| b.n).sum::<usize>());
    }
    for c in file.chunks.iter_mut() {
        c.lod_size.clear();
    }
    let all = value("--encoding").transpose()?.map(|e| e as u32);
    let encoding = move |_: SectionId| all.unwrap_or(0);
    if flag("--info") {
        let h = &file.header;
        let tree = VirtualTree::of_file(&file, true)?;
        println!(
            "{}: {} splats, {} chunks of {}, {} levels ({} merged), SH degree {}, flags {:#x}",
            input, h.count, h.chunks, h.chunk_splats, h.levels, tree.merged, h.sh_degree(), h.flags
        );
        let v3 = if u32::from_le_bytes(bytes[..4].try_into().unwrap()) == ATH3_MAGIC {
            bytes.clone()
        } else {
            write_v3_encoded(&file, COMPRESSION_NONE, &encoding)?
        };
        let layout = parse_v3(&v3)?;
        for (k, s) in layout.sections.iter().enumerate() {
            let stored: u64 = layout.blocks.iter().map(|b| b.spans[k].stored as u64).sum();
            let raw: u64 = layout.blocks.iter().map(|b| b.spans[k].raw as u64).sum();
            println!(
                "  section {:?}: tier {}, {} B a splat, encoding {}, compression {}, stored {} of {} bytes",
                s.id, s.tier, 4 * s.words, s.encoding, s.compression, stored, raw
            );
        }
        for tier in [1, 2, 3] {
            let bytes: u64 = layout.blocks.iter().map(|b| b.tier_range(&layout.sections, tier).1).sum();
            println!("  tier {tier}: {:.1} MB", bytes as f64 / 1e6);
        }
        return Ok(());
    }
    let Some(output) = paths.get(1) else { bail!("no output path") };
    // A skinned v3 cloud keeps its skeleton (athc_skin); a v2 file drops it.
    let skeleton = read_v3_skeleton(&bytes)?;
    let out = if flag("--v2") {
        file.write()?
    } else {
        let compression = if flag("--gzip") { COMPRESSION_GZIP } else { COMPRESSION_NONE };
        if flag("--planes") {
            let (out, chosen) = write_v3_smallest_with(&file, compression, skeleton.as_ref())?;
            println!("encodings: {:?}", chosen);
            out
        } else {
            write_v3_full(&file, compression, false, &encoding, skeleton.as_ref())?
        }
    };
    std::fs::write(output, &out)?;
    println!("{} -> {} ({:.1} MB to {:.1} MB)", input, output, bytes.len() as f64 / 1e6, out.len() as f64 / 1e6);
    if let (Some((athl_in, athl_out)), Some(original)) = (&athl_io, &original) {
        let threshold: f32 = text("--athl-threshold").map_or(Ok(0.0), |v| v.parse()).context("--athl-threshold")?;
        let n0 = original.header.count;
        let lists = lists.unwrap_or_else(|| {
            ranges.unwrap_or_else(|| (0..n0).map(|i| [i, i + 1]).collect()).iter().map(|r| (r[0]..r[1]).collect()).collect()
        });
        let athl = carry_athl(&AthlFile::read(&std::fs::read(athl_in)?)?, original, &file, &lists, &out, threshold)?;
        let written = athl.write()?;
        std::fs::write(athl_out, &written)?;
        println!(
            "{} -> {} ({} layers, {:.2} MB to {:.2} MB, cloud {:016x})",
            athl_in,
            athl_out,
            athl.layers.len(),
            std::fs::metadata(athl_in)?.len() as f64 / 1e6,
            written.len() as f64 / 1e6,
            athl.cloud_hash
        );
    }
    Ok(())
}

/// `athl` (over `original`'s virtual order) over the cut cloud `file`,
/// whose element e stands for `original`'s splats `lists[e]`.
fn carry_athl(
    athl: &AthlFile,
    original: &AthcFile,
    file: &AthcFile,
    ranges: &[Vec<u32>],
    file_bytes: &[u8],
    threshold: f32,
) -> Result<AthlFile> {
    let old = VirtualTree::of_file(original, true)?;
    if athl.splat_base != old.splat_base || athl.splat_count != original.header.count || athl.merged != old.merged {
        bail!(
            "the .athl is over {} merged + {} splats from {}, the cloud {} + {} from {}",
            athl.merged, athl.splat_count, athl.splat_base, old.merged, original.header.count, old.splat_base
        );
    }
    if ranges.len() != file.header.count as usize {
        bail!("{} ranges for {} elements", ranges.len(), file.header.count);
    }
    let tree = VirtualTree::of_file(file, true)?;
    let n = file.header.count;
    let mut out = AthlFile {
        flags: athl.flags,
        element_count: tree.splat_base + n,
        merged: tree.merged,
        splat_base: tree.splat_base,
        splat_count: n,
        cloud_hash: cloud_hash(file_bytes),
        bake_hash: athl.bake_hash,
        groups: athl.groups.clone(),
        polygons: athl.polygons.clone(),
        profiles: athl.profiles.clone(),
        layers: Vec::new(),
    };
    let mut kinds: Vec<(u16, u16)> = athl.layers.iter().map(|l| (l.group, l.kind)).collect();
    kinds.sort();
    kinds.dedup();
    for (group, kind) in kinds {
        let (c, dense) = athl.dense(group, kind).expect("a layer of that group and kind");
        let c = c as usize;
        let per = &dense[athl.splat_base as usize * c..];
        let (values, weights) = cut_values_of(&splat_weights(original), ranges, per, c);
        let virt = virtual_values_weighted(file, &tree, &values, c as u32, Some(&weights))?;
        out.layers.extend(sparse_layers(group, kind, c as u32, &virt, threshold)?);
    }
    out.layers.sort_by_key(|l| (l.chunk, l.group, l.kind));
    validate(&out)?;
    Ok(out)
}

/// `athl::cut_values` over lists of splats: each element's value the
/// weighted mean of its splats', and their summed weight.
fn cut_values_of(w: &[f32], lists: &[Vec<u32>], per_splat: &[f32], c: usize) -> (Vec<f32>, Vec<f32>) {
    let mut values = vec![0f32; lists.len() * c];
    let mut sums = vec![0f32; lists.len()];
    for (e, list) in lists.iter().enumerate() {
        let mut sw = 0f64;
        let mut sv = vec![0f64; c];
        for &i in list {
            let i = i as usize;
            sw += w[i] as f64;
            for k in 0..c {
                sv[k] += w[i] as f64 * per_splat[i * c + k] as f64;
            }
        }
        sums[e] = sw as f32;
        if sw > 0.0 {
            for k in 0..c {
                values[e * c + k] = (sv[k] / sw) as f32;
            }
        }
    }
    (values, sums)
}
