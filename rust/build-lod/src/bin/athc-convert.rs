//! .athc v2 <-> v3 (rust/spark-lib/src/athc_v3.rs, docs/docs/athc-v3.md).
//!
//!   athc-convert in.athc out.athc [--gzip]   v1/v2 to v3 (sections by tier: 1 splats, 2 material, 3 relight)
//!   athc-convert in.athc out.athc --v2       v3 (or v2) to v2
//!   athc-convert in.athc --info              headers, sections, tier sizes
//!   --coverage            the merged levels' whole coverage (athc::uncap_levels; v3 only)
//!   --keep-levels N       the levels from the coarsest down to the Nth, whose
//!                         groups become the splats (athc::truncate_levels)
//!   --keep-splats S       the same, N the finest level with at most S groups

use anyhow::{bail, Result};
use spark_lib::athc::{truncate_levels, uncap_levels, AthcFile, VirtualTree};
use spark_lib::athc_v3::{parse_v3, read_v3, write_v3, ATH3_MAGIC, COMPRESSION_GZIP, COMPRESSION_NONE};

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
        .filter(|&(i, a)| !a.starts_with("--") && !(i > 0 && args[i - 1].starts_with("--keep-")))
        .map(|(_, a)| a)
        .collect();
    let Some(input) = paths.first() else {
        bail!("usage: athc-convert in.athc out.athc [--gzip | --v2] | athc-convert in.athc --info");
    };
    let bytes = std::fs::read(input)?;
    let mut file = read_any(&bytes)?;
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
    if let Some(k) = keep {
        let before = file.header.count;
        file = truncate_levels(&file, k)?;
        println!("kept {} levels: {} splats (of {})", k, file.header.count, before);
    } else if flag("--coverage") {
        uncap_levels(&mut file);
    }
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
            write_v3(&file, COMPRESSION_NONE)?
        };
        let layout = parse_v3(&v3)?;
        for s in &layout.sections {
            println!("  section {:?}: tier {}, {} B a splat", s.id, s.tier, 4 * s.words);
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
        write_v3(&file, if flag("--gzip") { COMPRESSION_GZIP } else { COMPRESSION_NONE })?
    };
    std::fs::write(output, &out)?;
    println!("{} -> {} ({:.1} MB to {:.1} MB)", input, output, bytes.len() as f64 / 1e6, out.len() as f64 / 1e6);
    Ok(())
}
