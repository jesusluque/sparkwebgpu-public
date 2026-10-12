//! Fits OUR OWN joint CPCA basis for the compact tier (thread CR,
//! spark_lib::athc_basis_fit) to the full transfer of an asset's pieces.
//!
//!   fit-compact-basis <piece.athc>... --out <dir/basis> [--clusters 256]
//!     [--iters 16] [--seed 1] [--seed-sample 200000] [--compare other.json]
//!     [--report report.json] [--held-out piece.athc]...
//!
//! The pieces: .athc files carrying the full 112-value transfer in their
//! streams (usd-athc --transfer full). Every node is used (leaves and merged
//! nodes). Writes <out>.json + <out>.bin (athenea's layout, which usd-athc
//! --compact-basis reads) and prints, per piece and block (direct, indirect,
//! reflected), the transfer's relMSE through the fitted basis at int4 -- and
//! through `--compare` (e.g. athenea's q97, a LOCAL comparison only).
//! `--held-out <piece.athc>` (repeatable): measured, not fitted on.
//! `--sample S` (thread CS, the whole car): fit on every S-th node of each
//! piece only (RAM: 448 bytes a fitted node), then measure every node, one
//! piece in memory at a time. `piece.athc@S` (thread CX): that piece's own
//! stride, to weigh some pieces more (athenea's 5120 detail beside the
//! 1280 panels).

use anyhow::{bail, Context, Result};
use serde_json::json;
use spark_lib::athc::{high_half, low_half};
use spark_lib::athc_basis_fit::{basis_json_bin, fit_joint_basis, transfer_error, FitOptions};
use spark_lib::athc_compact::JointBasis;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let valued = ["--out", "--clusters", "--iters", "--seed", "--seed-sample", "--compare", "--report", "--held-out", "--sample"];
    let held: Vec<String> =
        args.iter().enumerate().filter(|(_, a)| *a == "--held-out").filter_map(|(i, _)| args.get(i + 1).cloned()).collect();
    let mut files = Vec::new();
    let mut k = 0;
    while k < args.len() {
        if valued.contains(&args[k].as_str()) {
            k += 2;
            continue;
        }
        if args[k].starts_with("--") {
            bail!("unknown option {}", args[k]);
        }
        files.push(args[k].clone());
        k += 1;
    }
    let out = arg("--out").context("--out <dir/basis>")?;
    if files.is_empty() {
        bail!("usage: fit-compact-basis <piece.athc>... --out <dir/basis>");
    }
    let mut o = FitOptions::default();
    if let Some(v) = arg("--clusters") {
        o.clusters = v.parse()?;
    }
    if let Some(v) = arg("--iters") {
        o.iters = v.parse()?;
    }
    if let Some(v) = arg("--seed") {
        o.seed = v.parse()?;
    }
    if let Some(v) = arg("--seed-sample") {
        o.seed_sample = v.parse()?;
    }

    let sample: usize = arg("--sample").map(|v| v.parse()).transpose()?.unwrap_or(1).max(1);
    // A piece's transfer, every node (leaves and merges), as f32.
    let load = |f: &str, stride: usize| -> Result<Vec<f32>> {
        let bytes = std::fs::read(f).with_context(|| f.to_string())?;
        let file = spark_lib::athc_v3::read_v3(&bytes).with_context(|| f.to_string())?;
        drop(bytes);
        let mut x = Vec::new();
        let mut i = 0usize;
        for b in file.levels.iter().map(|(_, b)| b).chain(&file.chunks) {
            if b.n == 0 {
                continue;
            }
            if b.transfer.len() != 56 * b.n {
                bail!("{f}: not the full transfer (112 values a node)");
            }
            for node in b.transfer.chunks_exact(56) {
                if i % stride == 0 {
                    for w in node {
                        let (lo, hi) = (low_half(*w), high_half(*w));
                        x.push(if lo.is_finite() { lo } else { 0.0 });
                        x.push(if hi.is_finite() { hi } else { 0.0 });
                    }
                }
                i += 1;
            }
        }
        Ok(x)
    };
    let stem = |f: &str| std::path::Path::new(f).file_stem().unwrap().to_string_lossy().to_string();

    // The fit's sample: every `sample`-th node of each fitted piece.
    let mut x: Vec<f32> = Vec::new();
    for f in &files {
        let (path, stride) = split_stride(f, sample)?;
        let before = x.len() / 112;
        x.extend(load(path, stride)?);
        eprintln!("{}: {} nodes sampled (1 in {stride})", stem(path), x.len() / 112 - before);
    }
    let (basis, fit) = fit_joint_basis(&x, &o)?;
    eprintln!("fit: {}", serde_json::to_string(&fit)?);
    drop(x);

    let compare = match arg("--compare") {
        Some(j) => {
            let bin = j.strip_suffix(".json").map(|p| format!("{p}.bin")).unwrap_or_else(|| format!("{j}.bin"));
            Some(JointBasis::parse(&std::fs::read_to_string(&j)?, &std::fs::read(&bin)?)?)
        }
        None => None,
    };
    // Every node of every piece, one piece at a time; sums kept for the total.
    let ratio = |e: &[f64; 3], f: &[f64; 3]| {
        json!({ "all": e.iter().sum::<f64>() / f.iter().sum::<f64>().max(1e-30), "direct": e[0] / f[0].max(1e-30), "indirect": e[1] / f[1].max(1e-30), "reflected": e[2] / f[2].max(1e-30) })
    };
    let mut rows = Vec::new();
    let mut tot = [([0f64; 3], [0f64; 3]); 2];
    let mut tot_nodes = 0usize;
    for (k, f) in files.iter().chain(&held).enumerate() {
        let f = split_stride(f, 1)?.0;
        let x = load(f, 1)?;
        let n = x.len() / 112;
        let mut row = json!({ "piece": stem(f), "heldOut": k >= files.len(), "nodes": n });
        for (j, b) in std::iter::once(Some(&basis)).chain(std::iter::once(compare.as_ref())).enumerate() {
            let Some(b) = b else { continue };
            let (e, fs) = transfer_error(b, &x, o.levels, o.threads);
            if k < files.len() {
                for q in 0..3 {
                    tot[j].0[q] += e[q];
                    tot[j].1[q] += fs[q];
                }
            }
            row[if j == 0 { "ours" } else { "compare" }] = ratio(&e, &fs);
        }
        if k < files.len() {
            tot_nodes += n;
        }
        println!("{}", serde_json::to_string(&row)?);
        rows.push(row);
    }
    let mut all = json!({ "piece": "ALL fitted", "heldOut": false, "nodes": tot_nodes, "ours": ratio(&tot[0].0, &tot[0].1) });
    if compare.is_some() {
        all["compare"] = ratio(&tot[1].0, &tot[1].1);
    }
    println!("{}", serde_json::to_string(&all)?);
    rows.push(all);

    if basis.clusters > spark_lib::athc_compact::MAX_CLUSTERS {
        // A measurement only: the compact tier's cluster is a byte.
        if let Some(r) = arg("--report") {
            std::fs::write(r, serde_json::to_string_pretty(&json!({ "fit": fit, "pieces": rows }))?)?;
        }
        eprintln!("{} clusters: measured only (the compact tier stores 512 at most); no basis written", basis.clusters);
        return Ok(());
    }
    let names: Vec<String> = files.iter().map(|f| stem(f)).collect();
    let (j, bin) = basis_json_bin(
        &basis,
        json!({
            "source": "sparkwebGPU fit-compact-basis (thread CR): OUR OWN fit, k-means of the joint vectors then per-cluster PCA, int4 scales by least squared error. Not athenea's basis.",
            "fittedOn": names,
            "nodes": fit.nodes,
            "clustersUsed": fit.clusters_used,
            "weightLevels": o.levels,
            "seed": o.seed,
            "sample": sample,
        }),
    );
    std::fs::write(format!("{out}.json"), j)?;
    std::fs::write(format!("{out}.bin"), bin)?;
    if let Some(r) = arg("--report") {
        std::fs::write(r, serde_json::to_string_pretty(&json!({ "fit": fit, "pieces": rows }))?)?;
    }
    eprintln!("wrote {out}.json and .bin");
    Ok(())
}

/// `piece.athc@S`: the path and its own stride (else `default`).
fn split_stride(f: &str, default: usize) -> Result<(&str, usize)> {
    match f.rsplit_once('@') {
        Some((path, s)) if !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) => Ok((path, s.parse::<usize>()?.max(1))),
        _ => Ok((f, default)),
    }
}
