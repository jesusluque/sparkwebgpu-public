//! Per block and transfer section of a CPCA .athc: mode, clusters, directions, bytes.
//!   cargo run --release --example athc_cpca_stats -- FILE.athc
use anyhow::Result;
use spark_lib::athc_v3::{gunzip, parse_v3, ENCODING_CPCA};

fn main() -> Result<()> {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path)?;
        let layout = parse_v3(&bytes)?;
        let mut rows = Vec::new();
        for (bi, b) in layout.blocks.iter().enumerate() {
            for (s, span) in layout.sections.iter().zip(&b.spans) {
                if !s.id.is_transfer() { continue; }
                let stored = &bytes[span.offset as usize..span.offset as usize + span.stored as usize];
                let raw = if s.compression == 1 { gunzip(stored)? } else { stored.to_vec() };
                let w = |k: usize| u32::from_le_bytes(raw[4 * k..4 * k + 4].try_into().unwrap());
                let (mode, k, m, bb) = if s.encoding != ENCODING_CPCA { (9, 0, 0, 0) } else if w(0) == 0 { (0, 0, 0, 0) } else { (1, w(1), w(2), w(4)) };
                rows.push(format!("{bi} {} {} {} {} {:?} d{} mode{} k{} m{} b{}", b.kind, b.level, b.first, b.n, s.id, 2 * s.words, mode, k, m, bb));
            }
        }
        println!("# {path}");
        for r in rows { println!("{r}"); }
    }
    Ok(())
}
