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

use anyhow::{bail, Context, Result};
use serde_json::json;
use spark_lib::athc::{high_half, low_half};
use spark_lib::athc_basis_fit::{basis_json_bin, fit_joint_basis, transfer_error, FitOptions};
use spark_lib::athc_compact::JointBasis;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let valued = ["--out", "--clusters", "--iters", "--seed", "--seed-sample", "--compare", "--report", "--held-out"];
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

    // The transfer of every node, by piece.
    let mut pieces: Vec<(String, std::ops::Range<usize>)> = Vec::new();
    let mut x: Vec<f32> = Vec::new();
    for f in files.iter().chain(&held) {
        let bytes = std::fs::read(f).with_context(|| f.clone())?;
        let file = spark_lib::athc_v3::read_v3(&bytes).with_context(|| f.clone())?;
        let first = x.len() / 112;
        for b in file.levels.iter().map(|(_, b)| b).chain(&file.chunks) {
            if b.n == 0 {
                continue;
            }
            if b.transfer.len() != 56 * b.n {
                bail!("{f}: not the full transfer (112 values a node)");
            }
            for w in &b.transfer {
                let (lo, hi) = (low_half(*w), high_half(*w));
                x.push(if lo.is_finite() { lo } else { 0.0 });
                x.push(if hi.is_finite() { hi } else { 0.0 });
            }
        }
        let name = std::path::Path::new(f).file_stem().unwrap().to_string_lossy().to_string();
        eprintln!("{name}: {} nodes", x.len() / 112 - first);
        pieces.push((name, first..x.len() / 112));
    }

    let fitted = pieces.iter().take(files.len()).map(|(_, r)| r.end).max().unwrap_or(0);
    let (basis, fit) = fit_joint_basis(&x[..fitted * 112], &o)?;
    eprintln!("fit: {}", serde_json::to_string(&fit)?);

    let compare = match arg("--compare") {
        Some(j) => {
            let bin = j.strip_suffix(".json").map(|p| format!("{p}.bin")).unwrap_or_else(|| format!("{j}.bin"));
            Some(JointBasis::parse(&std::fs::read_to_string(&j)?, &std::fs::read(&bin)?)?)
        }
        None => None,
    };
    let rel = |b: &JointBasis, r: &std::ops::Range<usize>| {
        let (e, f) = transfer_error(b, &x[r.start * 112..r.end * 112], o.levels, o.threads);
        let all = e.iter().sum::<f64>() / f.iter().sum::<f64>().max(1e-30);
        json!({ "all": all, "direct": e[0] / f[0].max(1e-30), "indirect": e[1] / f[1].max(1e-30), "reflected": e[2] / f[2].max(1e-30) })
    };
    let mut rows = Vec::new();
    let all = ("ALL fitted".to_string(), 0..fitted);
    for (k, (name, r)) in pieces.iter().chain(std::iter::once(&all)).enumerate() {
        let row = json!({
            "piece": name,
            "heldOut": k >= files.len() && k < pieces.len(),
            "nodes": r.len(),
            "ours": rel(&basis, r),
            "compare": compare.as_ref().map(|c| rel(c, r)),
        });
        println!("{}", serde_json::to_string(&row)?);
        rows.push(row);
    }

    let names: Vec<&str> = pieces.iter().take(files.len()).map(|(n, _)| n.as_str()).collect();
    let (j, bin) = basis_json_bin(
        &basis,
        json!({
            "source": "sparkwebGPU fit-compact-basis (thread CR): OUR OWN fit, k-means of the joint vectors then per-cluster PCA, int4 scales by least squared error. Not athenea's basis.",
            "fittedOn": names,
            "nodes": fit.nodes,
            "clustersUsed": fit.clusters_used,
            "weightLevels": o.levels,
            "seed": o.seed,
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
