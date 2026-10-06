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

use anyhow::{bail, Result};
use spark_lib::athc_build::{build_lod, packed_of, BuildOptions};
use spark_lib::athc::{truncate_creases, truncate_levels, uncap_levels, AthcFile, VirtualTree};
use spark_lib::athc_v3::{
    parse_v3, read_v3, write_v3_encoded, write_v3_smallest, SectionId, ATH3_MAGIC, COMPRESSION_GZIP, COMPRESSION_NONE,
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
            !a.starts_with("--") && !(i > 0 && (args[i - 1].starts_with("--keep-") || args[i - 1] == "--encoding" || args[i - 1] == "--creases" || args[i - 1] == "--crease-depth" || args[i - 1] == "--rebuild-frame"))
        })
        .map(|(_, a)| a)
        .collect();
    let Some(input) = paths.first() else {
        bail!("usage: athc-convert in.athc out.athc [--gzip | --v2] | athc-convert in.athc --info");
    };
    let bytes = std::fs::read(input)?;
    let mut file = read_any(&bytes)?;
    if let Some(seed) = args.iter().position(|a| a == "--rebuild-frame").and_then(|i| args.get(i + 1)) {
        let o = BuildOptions { frame_seed: seed.parse()?, chunk_splats: file.header.chunk_splats, ..Default::default() };
        let before: Vec<usize> = file.levels.iter().map(|(_, b)| b.n).collect();
        file = build_lod(&packed_of(&file), &o)?;
        let after: Vec<usize> = file.levels.iter().map(|(_, b)| b.n).collect();
        println!("levels built again (frame {seed}): groups {before:?} -> {after:?}");
    }
    let value = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1)).map(|v| v.parse::<usize>());
    let keep = match (value("--keep-levels"), value("--keep-splats")) {
        (Some(n), _) => Some(n?),
        (None, Some(s)) => {
            let s = s?;
            let k = file.levels.iter().rposition(|(_, b)| b.n <= s).filter(|&k| k > 0);
            Some(k.ok_or_else(|| anyhow::anyhow!("no level below the coarsest has at most {s} groups"))?)
        }
        _ => None,
    };
    let creases = args.iter().position(|a| a == "--creases").and_then(|i| args.get(i + 1)).map(|v| v.parse::<f32>());
    if let Some(k) = keep {
        let before = file.header.count;
        if let Some(d) = creases {
            let depth = value("--crease-depth").transpose()?.unwrap_or(usize::MAX);
            let (cut, stats) = truncate_creases(&file, k, d?, depth)?;
            file = cut;
            println!("crease-aware cut: merged per level from {}: {:?}, real splats {}", k, stats.per_level, stats.splats);
        } else {
            file = truncate_levels(&file, k)?;
        }
        println!("kept {} levels: {} splats (of {})", k, file.header.count, before);
    } else if flag("--coverage") {
        uncap_levels(&mut file);
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
    let out = if flag("--v2") {
        file.write()?
    } else {
        let compression = if flag("--gzip") { COMPRESSION_GZIP } else { COMPRESSION_NONE };
        if flag("--planes") {
            let (out, chosen) = write_v3_smallest(&file, compression)?;
            println!("encodings: {:?}", chosen);
            out
        } else {
            write_v3_encoded(&file, compression, &encoding)?
        }
    };
    std::fs::write(output, &out)?;
    println!("{} -> {} ({:.1} MB to {:.1} MB)", input, output, bytes.len() as f64 / 1e6, out.len() as f64 / 1e6);
    Ok(())
}
