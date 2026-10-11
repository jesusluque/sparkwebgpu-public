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
//!   --cpca K,M[,E]        the transfer's sections as clustered PCA (athc_cpca, encoding 3, lossy):
//!                         per block at most K clusters (<= 256) of at most M directions, within
//!                         a relative squared error E of each block's values (default 1e-5); the
//!                         other sections as --planes / --encoding say. Merged nodes included.
//!                         --cpca-force: CPCA wherever it meets E, even where byte planes are smaller
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
//!
//!   --lod-errors          write each merged node's error cut bounds (athc_lod_cut, the v3 section
//!                         LODE): eps = max(r x cost, children's), r = sqrt(tr Sigma) and cost
//!                         athc_merge::group_cost of its children (BC's merge cost without the
//!                         on-screen weight, any number of children); rho the sphere round its
//!                         leaves; theta the back-face cone of its leaves' normals (none for glass,
//!                         thin walls, no normals; only with --closed, never with --double-sided). A file that has them keeps them
//!                         (athenea's, from usd-athc --tree) unless the tree changes (then ours);
//!                         --no-lod-errors drops them; --lod-error-weights colour=,normal=,guard=,floor=
//!
//!   --shadow-vq C[,R[,P]] the open-direction bits (shadowBits, 8 words) as a dictionary: per
//!                         piece (P splats in cloud order, 16384; 0: a material, pbr + lobes)
//!                         k-means in Hamming space, C centroids (1024), R rounds (12),
//!                         majority centroids, seeds by cloud order; the splats only
//!                         (athc_shadow_vq); each row's bits become its centroid's. Prints the
//!                         Hamming distances (p50/p90/p99 of 256) and the codebook entries
//!
//! Pruning and decimating (thread BB, `spark_lib::athc_prune`), every section
//! following the kept splats (transfer, curvature, lobes, skin, attributes)
//! and the levels built again over them:
//!   athc-convert in.athc out.athc --drop-hidden --camera ...      one file
//!   athc-convert --scene OUTDIR a.athc b.athc ... --drop-hidden   files that hide each other,
//!                         each written to OUTDIR under its own name
//!   --drop-hidden         drop what no view of an orbit sees (CPU raster, Spark's rules): a
//!                         splat whose centre stays under --hidden-t (0.02) of transmittance and
//!                         that adds under --hidden-px (0.02) px over all views
//!   --camera ex,ey,ez,tx,ty,tz,ux,uy,uz,hfov[,aspect]   the scene's camera (degrees; aspect 16/9)
//!   --views N             directions at its distance (800), a quarter as many at half of it and
//!                         a quarter close up (a tenth of it outside the scene's box, 100 degrees)
//!   --vis-width W         px (800)          --sphere   views from below too (default: upper half)
//!   --clips-json J        a skinned cloud: the clips of J (sparrow.json), only what every
//!                         sampled pose hides is dropped (--pose-samples 6, --pose-views 48,
//!                         --pose-width 640)
//!   --occluder X.athc     in the scene, hiding others, not written (repeatable)
//!   --keep PART           written but not pruned (repeatable)
//!   --decimate PART=K[:GROW][@FILTER]   thin PART's opaque splats to about 1 in K along Morton
//!                         runs on one panel, long axes x GROW (0.75 sqrt K); @coat: only coated
//!                         ones; @near:REF.athc[,MM]: only those on REF's surface (within MM, 5),
//!                         a material picked in a file of many by another bake's part
//!   --athl PART=IN.athl   (scene) carry PART's light layers; with one file --athl/--athl-out
//!   --check [--check-width W] [--check-dir D]   held-out views (and poses), AY's relMSE in
//!                         float (1920); D: the views as PPM, for looking only
//!   --report R.json       what was done, a file at a time
//!   PART: a file's name up to its first '-' or '.' (paint-t16-gz.athc: paint)
//!
//! An asset's compact pieces (thread CQ, `spark_lib::athc_compact`; paged trees
//! written by `usd-athc --tree ... --pages --compact-basis`, v1 or v2):
//!   athc-convert --compact-rebase in.athc out.athc   the compact tier in the v3 layout
//!   athc-convert --compact-runs [--run-nodes 64] in.athc out.athc   CMPC v4 (thread CT):
//!                         each page's traversal planes gzipped, the GPU's in deflate runs
//!                         (the traversal's 24 bytes a node: v2's 16, the cone, occlusion), its own tables
//!   athc-convert --compact-asset car.cmpa --out-dir D a.athc b.athc ...   one asset table
//!                         for the pieces (D/car.cmpa: the basis once) and each piece written
//!                         to D under its name, sharing it; tiers 1-3 byte for byte
//!   --table-only          (thread CS) the asset table alone, for pieces in CPCA the rebase
//!                         cannot write again: convert them with usd-athc --compact-asset over it.
//!                         athenea's occlusion codebook goes in it once when every piece has the
//!                         same (CMPA flag 8, with the pieces' box union; --no-asset-occlusion)
//!   --asset-entries N     one shadow dictionary of N entries for every piece (4096; 0: each
//!                         piece keeps its own), k-means over every node (--asset-rounds 12)
//!   --piece-materials     each piece keeps its material table (default: one for the asset)


use anyhow::{bail, Context, Result};

#[path = "athc-convert/prune.rs"]
mod prune;
use spark_lib::athl::{cloud_hash, splat_weights, sparse_layers, validate, virtual_values_weighted, AthlFile};
use spark_lib::athc_build::{build_lod, packed_of, BuildOptions};
use spark_lib::athc_lod_error::{with_lod_sizes, LodSizeOptions};
use spark_lib::athc_merge::{error_cut_to, error_levels, CostKind, ErrorOptions, ErrorView};
use spark_lib::athc::{cut_sources, truncate_creases, truncate_levels, uncap_levels, AthcFile, VirtualTree};
use spark_lib::athc_v3::{
    parse_v3, read_v3, read_v3_skeleton, write_v3_cpca, write_v3_encoded, write_v3_full, write_v3_smallest_with, SectionId, ATH3_MAGIC, COMPRESSION_GZIP, COMPRESSION_NONE,
};
use spark_lib::athc_cpca::CpcaOptions;

fn read_any(bytes: &[u8]) -> Result<AthcFile> {
    if bytes.len() >= 4 && u32::from_le_bytes(bytes[..4].try_into().unwrap()) == ATH3_MAGIC {
        read_v3(bytes)
    } else {
        AthcFile::read(bytes)
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--scene" || a == "--drop-hidden" || a.starts_with("--decimate")) {
        return prune::run(&args);
    }
    if args.iter().any(|a| a == "--compact-runs") {
        let run: usize = args.iter().position(|a| a == "--run-nodes").and_then(|i| args.get(i + 1)).map(|v| v.parse()).transpose()?.unwrap_or(spark_lib::athc_compact::RUN_NODES);
        let paths: Vec<&String> = args.iter().enumerate().filter(|&(i, a)| !a.starts_with("--") && !(i > 0 && args[i - 1] == "--run-nodes")).map(|(_, a)| a).collect();
        let [input, output] = paths[..] else { bail!("usage: athc-convert --compact-runs [--run-nodes N] in.athc out.athc") };
        let (bytes, sizes) = spark_lib::athc_compact::convert_runs(&std::fs::read(input)?, run)?;
        std::fs::write(output, &bytes)?;
        eprintln!("{output}: {} bytes, tier 4 {} -> {} bytes", bytes.len(), sizes[0], sizes[1]);
        return Ok(());
    }
    if args.iter().any(|a| a == "--compact-asset" || a == "--compact-rebase") {
        return compact_asset(&args);
    }
    let flag = |f: &str| args.iter().any(|a| a == f);
    // Paths: what is neither a flag nor a --keep-* flag's value.
    let paths: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|&(i, a)| {
            !a.starts_with("--") && !(i > 0 && (args[i - 1].starts_with("--keep-") || args[i - 1] == "--encoding" || args[i - 1] == "--lod-error-weights" || args[i - 1] == "--cpca" || args[i - 1] == "--shadow-vq" || args[i - 1] == "--creases" || args[i - 1] == "--crease-depth" || args[i - 1] == "--rebuild-frame" || args[i - 1].starts_with("--athl") || args[i - 1] == "--lod-size-opts" || (args[i - 1].starts_with("--error-") && args[i - 1] != "--error-levels")))
        })
        .map(|(_, a)| a)
        .collect();
    let Some(input) = paths.first() else {
        bail!("usage: athc-convert in.athc out.athc [--gzip | --v2] | athc-convert in.athc --info");
    };
    let bytes = std::fs::read(input)?;
    let mut file = read_any(&bytes)?;
    let had_lod_sizes = file.levels.iter().any(|(_, b)| !b.lod_size.is_empty());
    let had_lod_errors = file.levels.iter().any(|(_, b)| !b.lod_error.is_empty());
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
        match text("--error-kind").as_deref() {
            None | Some("screen") => {}
            Some("seen") => o.kind = CostKind::Seen,
            Some(k) => bail!("--error-kind screen|seen, not {k}"),
        }
        o.gloss = num("--error-gloss", o.gloss)?;
        o.gloss_cone = num("--error-gloss-cone", o.gloss_cone)?;
        o.flat = num("--error-flat", o.flat)?;
        o.seen_colour = num("--error-colour", o.seen_colour)?;
        o.seen_material = num("--error-seen-material", o.seen_material)?;
        o.edge = num("--error-edge", o.edge)?;
        o.pass_fraction = num("--error-pass-fraction", o.pass_fraction)?;
        if flag("--error-mutual") {
            o.matching = false;
        }
        match text("--error-surfels").as_deref() {
            None | Some("auto") => {}
            Some("on") => o.surfels = spark_lib::athc_build::SurfelNodes::On,
            Some("off") => o.surfels = spark_lib::athc_build::SurfelNodes::Off,
            Some(k) => bail!("--error-surfels auto|on|off, not {k}"),
        }
        o.surfel_min = num("--error-surfel-min", o.surfel_min as f64)? as u32;
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
    } else if flag("--lod-sizes") || had_lod_sizes && !(flag("--cpca") && keep.is_none() && lists.is_none() && !flag("--rebuild-frame")) {
        // (--cpca alone changes the transfer only: the sizes stay as stored.)
        let o = LodSizeOptions::parse(&text("--lod-size-opts").unwrap_or_default())?;
        file = with_lod_sizes(&file, &o);
        println!("LoD sizes by error for {} merged nodes", file.levels.iter().map(|(_, b)| b.n).sum::<usize>());
    }
    for c in file.chunks.iter_mut() {
        c.lod_size.clear();
    }
    // The error cut's bounds (athc_lod_cut): kept as stored (athenea's, from
    // usd-athc --tree) unless the tree changed, then ours (group_cost).
    let tree_changed = keep.is_some() || lists.is_some() || flag("--rebuild-frame") || flag("--error") || flag("--error-levels");
    if flag("--no-lod-errors") {
        for (_, b) in file.levels.iter_mut() {
            b.lod_error.clear();
        }
    } else if flag("--lod-errors") || had_lod_errors && tree_changed {
        let w = spark_lib::athc_lod_cut::ErrorWeights::parse(&text("--lod-error-weights").unwrap_or_default())?;
        // A cone only for a piece said to be closed (`--closed`; `--double-sided` wins).
        file = spark_lib::athc_lod_cut::with_lod_errors(&file, None, &w, !flag("--closed") || flag("--double-sided"))?;
        println!("LoD errors (ours) for {} merged nodes", file.levels.iter().map(|(_, b)| b.n).sum::<usize>());
    }
    if let Some(spec) = text("--shadow-vq") {
        let mut o = spark_lib::athc_shadow_vq::ShadowVqOptions::default();
        let parts: Vec<&str> = spec.split(',').collect();
        o.centroids = parts[0].parse().context("--shadow-vq C")?;
        if let Some(r) = parts.get(1) {
            o.rounds = r.parse().context("--shadow-vq C,R")?;
        }
        if let Some(p) = parts.get(2) {
            o.piece_splats = p.parse().context("--shadow-vq C,R,P")?;
        }
        let t = std::time::Instant::now();
        let st = spark_lib::athc_shadow_vq::shadow_vq(&mut file, &o)?;
        use spark_lib::athc_shadow_vq::ShadowVqStats as S;
        println!(
            "shadow-vq {} x {} rounds: {} pieces ({} exact), {} rows, {} codebook entries ({:.1} KB); splats Hamming mean {:.2} p50 {} p90 {} p99 {} max {}; all rows p99 {}; {:.1}s",
            o.centroids, o.rounds, st.pieces, st.exact_pieces, st.rows, st.codebook_entries, st.codebook_entries as f64 * 32.0 / 1e3,
            S::mean(&st.splat_hist), S::percentile(&st.splat_hist, 0.5), S::percentile(&st.splat_hist, 0.9),
            S::percentile(&st.splat_hist, 0.99), S::percentile(&st.splat_hist, 1.0), S::percentile(&st.all_hist, 0.99),
            t.elapsed().as_secs_f32()
        );
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
        // The material's own roughness on the pbr word (athenea 35ae3cb,
        // bits 26..31): what the relight's polishField reads.
        let words: Vec<u32> = file.chunks.iter().chain(file.levels.iter().map(|(_, b)| b)).flat_map(|b| {
            let w = b.pbr.len().checked_div(b.n).unwrap_or(0);
            (0..if w > 0 { b.n } else { 0 }).map(move |i| b.pbr[i * w])
        }).collect();
        let marked = words.iter().filter(|&&w| (w >> 26) & 1 != 0).count();
        println!("  material roughness marked on {marked} of {} pbr words", words.len());
        return Ok(());
    }
    let Some(output) = paths.get(1) else { bail!("no output path") };
    // A skinned v3 cloud keeps its skeleton (athc_skin); a v2 file drops it.
    let skeleton = read_v3_skeleton(&bytes)?;
    let out = if flag("--v2") {
        file.write()?
    } else {
        let compression = if flag("--gzip") { COMPRESSION_GZIP } else { COMPRESSION_NONE };
        if let Some(c) = text("--cpca") {
            let o = CpcaOptions { force: flag("--cpca-force"), ..CpcaOptions::parse(&c)? };
            let chosen = if flag("--planes") {
                let (_, chosen) = write_v3_smallest_with(&file, compression, skeleton.as_ref())?;
                chosen
            } else {
                Vec::new()
            };
            let pick = |id: SectionId| chosen.iter().find(|(s, _)| *s == id).map_or(encoding(id), |&(_, e)| e);
            let t = std::time::Instant::now();
            let (out, stats) = write_v3_cpca(&file, compression, &pick, &o, skeleton.as_ref())?;
            for st in &stats {
                println!(
                    "cpca {}: {} blocks ({} exact), {} elements, up to {} clusters x {} directions, relMSE {:.3e}",
                    st.id.name(), st.blocks, st.exact_blocks, st.elements, st.max_clusters, st.max_coeffs, st.rel_mse()
                );
            }
            println!("cpca: {:.1}s", t.elapsed().as_secs_f32());
            out
        } else if flag("--planes") {
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

/// `--compact-rebase` and `--compact-asset` (thread CQ).
fn compact_asset(args: &[String]) -> Result<()> {
    use spark_lib::athc_compact::{build_asset, rebase_compact};
    let value = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1)).cloned();
    let takes = ["--compact-asset", "--out-dir", "--asset-entries", "--asset-rounds"];
    let paths: Vec<&String> =
        args.iter().enumerate().filter(|&(i, a)| !a.starts_with("--") && !(i > 0 && takes.contains(&args[i - 1].as_str()))).map(|(_, a)| a).collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    if args.iter().any(|a| a == "--compact-rebase") {
        let [input, output] = paths[..] else { bail!("usage: athc-convert --compact-rebase in.athc out.athc") };
        let (bytes, stats) = rebase_compact(&std::fs::read(input)?, None, "", threads)?;
        std::fs::write(output, &bytes)?;
        eprintln!("{output}: {} bytes, {}", bytes.len(), serde_json::to_string(&stats)?);
        return Ok(());
    }
    let asset_path = value("--compact-asset").context("--compact-asset car.cmpa")?;
    let out_dir = std::path::PathBuf::from(value("--out-dir").context("--out-dir D")?);
    let entries: usize = value("--asset-entries").map(|v| v.parse()).transpose().context("--asset-entries")?.unwrap_or(4096);
    let rounds: usize = value("--asset-rounds").map(|v| v.parse()).transpose().context("--asset-rounds")?.unwrap_or(12);
    let shared_materials = !args.iter().any(|a| a == "--piece-materials");
    if paths.is_empty() {
        bail!("--compact-asset: no pieces");
    }
    let inputs: Vec<Vec<u8>> = paths.iter().map(std::fs::read).collect::<std::io::Result<_>>()?;
    let refs: Vec<&[u8]> = inputs.iter().map(|b| b.as_slice()).collect();
    let t = std::time::Instant::now();
    let mut asset = build_asset(&refs, entries, rounds, shared_materials, threads)?;
    // athenea's occlusion codebook, once per asset (thread CS): the same in
    // every piece, with the asset's box (the pieces' boxes' union).
    let layouts: Vec<_> = refs.iter().map(|b| spark_lib::athc_v3::parse_v3(b)).collect::<Result<_>>()?;
    if layouts.iter().all(|l| !l.occlusion_codebook.is_empty()) && !args.iter().any(|a| a == "--no-asset-occlusion") {
        if layouts.iter().any(|l| l.occlusion_codebook != layouts[0].occlusion_codebook) {
            bail!("--compact-asset: the pieces' occlusion codebooks differ");
        }
        let mut b = [f32::MAX, f32::MAX, f32::MAX, f32::MIN, f32::MIN, f32::MIN];
        for l in &layouts {
            for k in 0..3 {
                b[k] = b[k].min(l.header.bounds_min[k]);
                b[3 + k] = b[3 + k].max(l.header.bounds_max[k]);
            }
        }
        asset.occlusion = layouts[0].occlusion_codebook.clone();
        asset.occlusion_box = b;
        eprintln!("occlusion: {} entries once, the asset's box {:?}", asset.occlusion.len() / 8, b);
    }
    std::fs::create_dir_all(&out_dir)?;
    let name = std::path::Path::new(&asset_path).file_name().context("--compact-asset: a file name")?.to_string_lossy().into_owned();
    let blob = asset.to_bytes();
    std::fs::write(out_dir.join(&name), &blob)?;
    eprintln!(
        "{name}: {} bytes (basis {} clusters, {} entries, {} materials), id {:016x}, {:.1}s",
        blob.len(),
        asset.clusters,
        asset.codebook.len() / 8,
        asset.materials.len() / 4,
        asset.id(),
        t.elapsed().as_secs_f32()
    );
    // `--table-only`: the asset table alone (pieces written in CPCA, which
    // the rebase cannot write again: convert them with usd-athc
    // --compact-asset over this table).
    if args.iter().any(|a| a == "--table-only") {
        return Ok(());
    }
    let mut report = Vec::new();
    for (path, bytes) in paths.iter().zip(&inputs) {
        let (out, stats) = rebase_compact(bytes, Some(&asset), &name, threads)?;
        let file = std::path::Path::new(path.as_str()).file_name().context("a piece's name")?;
        std::fs::write(out_dir.join(file), &out)?;
        eprintln!("{}: {} -> {} bytes", file.to_string_lossy(), bytes.len(), out.len());
        report.push(serde_json::json!({ "piece": file.to_string_lossy(), "bytes": out.len(), "before": bytes.len(), "compact": stats }));
    }
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "asset": name, "assetBytes": blob.len(), "pieces": report }))?);
    Ok(())
}
