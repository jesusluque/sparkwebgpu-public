#!/usr/bin/env node
// Vendors athenea's Slang into slang/athenea/ from one pinned commit.
//
// athenea (~/luc/athenea) is read, never written: every file comes out of
// `git show <commit>:<path>`, so the branch checked out there does not matter.
// The roots are shaders/athenea/{common,scene,splat,lod} and, of
// technique/, the display transform (display.slang, aces2_prepare.slang: the
// view transforms AgX and ACES 2.0 of the output stage); every module they
// import (transitively) comes along, wherever it lives. The copies keep their
// paths under shaders/athenea/, so `import athenea.common.packing;` resolves
// with `-I slang` exactly as it does with `-I shaders` in athenea.
//
// The copies are never edited by hand: glue goes in slang/athenea_adapter/.
// slang/athenea/VENDOR records the commit and a hash of every file, and
// tools/slang-build skips any directory holding a VENDOR file (athenea's own
// kernels are not ours to ship; the adapter imports what it needs).
//
//   node tools/vendor-athenea.mjs                 vendor from the pinned commit (VENDOR), or txf
//   node tools/vendor-athenea.mjs --ref txf       re-pin to what `txf` is now
//   node tools/vendor-athenea.mjs --check         fail if a copy differs from its pinned source
//   node tools/vendor-athenea.mjs --wgsl          also compile every vendored entry point to
//                                                 WGSL and write slang/athenea/WGSL.md
//   --repo <path>                                 athenea checkout (default ~/luc/athenea)

import { execFileSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { findEntries, findSlangc } from "./slang-build/index.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SLANG_DIR = path.join(ROOT, "slang");
const OUT = path.join(SLANG_DIR, "athenea");
// Directories, or single files (a path ending in .slang).
const ROOTS = [
  "common",
  "scene",
  "splat",
  "lod",
  "technique/display.slang",
  "technique/aces2_prepare.slang",
];
const PREFIX = "shaders/athenea/";

function arg(name) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : undefined;
}
const flag = (name) => process.argv.includes(name);

const repo = arg("--repo") ?? path.join(os.homedir(), "luc/athenea");
const git = (...args) =>
  execFileSync("git", ["-C", repo, ...args], {
    encoding: "utf8",
    maxBuffer: 1 << 28,
  });

function pinned() {
  const vendor = path.join(OUT, "VENDOR");
  if (!fs.existsSync(vendor)) return null;
  return /^commit\s+([0-9a-f]{40})$/m.exec(
    fs.readFileSync(vendor, "utf8"),
  )?.[1];
}

const ref = arg("--ref");
const commit = git("rev-parse", `${ref ?? pinned() ?? "txf"}^{commit}`).trim();
const refName = ref ?? (pinned() ? null : "txf");

const tree = new Set(
  git("ls-tree", "-r", "--name-only", commit, PREFIX)
    .split("\n")
    .filter((p) => p.endsWith(".slang")),
);
const show = (p) => git("show", `${commit}:${p}`);

// `import a.b.c;` is shaders/a/b/c.slang; `import name;` is beside the importer.
const IMPORT_RE = /^\s*(?:__)?import\s+([\w.]+)\s*;/gm;
function importsOf(file, source) {
  const out = [];
  for (const [, mod] of source.matchAll(IMPORT_RE)) {
    const local = path.posix.join(path.posix.dirname(file), `${mod}.slang`);
    const global = `shaders/${mod.replaceAll(".", "/")}.slang`;
    if (!mod.includes(".") && tree.has(local)) out.push(local);
    else if (tree.has(global)) out.push(global);
    else throw new Error(`${file}: cannot resolve import ${mod}`);
  }
  return out;
}

const files = new Map();
const queue = [...tree].filter((p) => {
  const rel = p.slice(PREFIX.length);
  return ROOTS.includes(rel) || ROOTS.includes(rel.split("/")[0]);
});
while (queue.length) {
  const p = queue.shift();
  if (files.has(p)) continue;
  const source = show(p);
  files.set(p, source);
  queue.push(...importsOf(p, source));
}

const sha = (s) => crypto.createHash("sha256").update(s).digest("hex");
const sorted = [...files.keys()].sort();

if (flag("--check")) {
  const bad = sorted.filter((p) => {
    const dest = path.join(OUT, p.slice(PREFIX.length));
    return (
      !fs.existsSync(dest) || fs.readFileSync(dest, "utf8") !== files.get(p)
    );
  });
  if (bad.length) {
    console.error(
      `vendor-athenea: ${bad.length} copies differ from ${commit}:`,
    );
    for (const p of bad) console.error(`  ${p}`);
    process.exit(1);
  }
  console.log(`vendor-athenea: ${sorted.length} files match ${commit}`);
  process.exit(0);
}

fs.rmSync(OUT, { recursive: true, force: true });
for (const p of sorted) {
  const dest = path.join(OUT, p.slice(PREFIX.length));
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, files.get(p));
}

const subject = git("log", "-1", "--format=%s", commit).trim();
const date = git("log", "-1", "--format=%cI", commit).trim();
fs.writeFileSync(
  path.join(OUT, "VENDOR"),
  [
    "# athenea Slang, vendored by tools/vendor-athenea.mjs. Do not edit these",
    "# files: re-run the script. Glue lives in slang/athenea_adapter/.",
    `repo     ${repo.replace(os.homedir(), "~")}`,
    ...(refName ? [`ref      ${refName}`] : []),
    `commit   ${commit}`,
    `date     ${date}`,
    `subject  ${subject}`,
    `roots    ${ROOTS.map((r) => PREFIX + r).join(" ")} (and their imports)`,
    "",
    "# sha256  path (under shaders/athenea/)",
    ...sorted.map((p) => `${sha(files.get(p))}  ${p.slice(PREFIX.length)}`),
    "",
  ].join("\n"),
);
fs.writeFileSync(
  path.join(OUT, "NOTICE"),
  [
    "athenea shaders",
    "Copyright (c) 2026 jesus luque.",
    "",
    "The .slang files under this directory are unmodified copies of",
    `shaders/athenea/ in the athenea repository at commit ${commit}.`,
    "Each keeps its original copyright header. They are vendored into",
    "sparkwebGPU by their author's decision (plan-athenea.md, section 6,",
    "2026-10-06) as a read-only dependency; changes belong upstream in",
    "athenea, adaptations in slang/athenea_adapter/.",
    "",
    "Licence: MIT, by decision of their author (2026-10-06), the same",
    "licence as Spark. athenea's tree at that commit carries no LICENSE",
    "file yet; when one is added upstream it governs these copies.",
    "",
  ].join("\n"),
);
console.log(
  `vendor-athenea: ${sorted.length} files from ${commit.slice(0, 7)} (${subject})`,
);

if (flag("--wgsl")) {
  const exe = findSlangc();
  if (!exe) throw new Error("slangc not found");
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "vendor-athenea-"));
  const rows = [];
  for (const p of sorted) {
    const rel = p.slice(PREFIX.length);
    const entries = findEntries(files.get(p));
    if (entries.length === 0) {
      // A module: compile it for errors alone.
      try {
        execFileSync(
          exe,
          [
            path.join(OUT, rel),
            "-I",
            SLANG_DIR,
            "-target",
            "wgsl",
            "-o",
            path.join(tmp, "m.wgsl"),
          ],
          { stdio: ["ignore", "pipe", "pipe"] },
        );
        rows.push([rel, "module", "ok", ""]);
      } catch (err) {
        const why = firstError(err);
        // A module whose code is all generic or unused emits nothing.
        if (why.includes("cannot write output file")) {
          rows.push([rel, "module", "ok", "no code emitted on its own"]);
        } else {
          rows.push([rel, "module", "fails", why]);
        }
      }
      continue;
    }
    for (const e of entries) {
      try {
        execFileSync(
          exe,
          [
            path.join(OUT, rel),
            "-I",
            SLANG_DIR,
            "-target",
            "wgsl",
            "-entry",
            e.name,
            "-stage",
            e.stage,
            "-o",
            path.join(tmp, `${e.name}.wgsl`),
          ],
          { stdio: ["ignore", "pipe", "pipe"] },
        );
        const wgsl = fs.readFileSync(path.join(tmp, `${e.name}.wgsl`), "utf8");
        const buffers = (wgsl.match(/var<storage/g) ?? []).length;
        rows.push([
          rel,
          e.name,
          "ok",
          buffers > 8
            ? `${buffers} storage buffers (> 8, Safari's default)`
            : `${buffers} storage buffers`,
        ]);
      } catch (err) {
        rows.push([rel, e.name, "fails", firstError(err)]);
      }
    }
  }
  fs.rmSync(tmp, { recursive: true, force: true });
  const ok = rows.filter((r) => r[2] === "ok").length;
  fs.writeFileSync(
    path.join(OUT, "WGSL.md"),
    [
      `# athenea Slang → WGSL (slangc, -target wgsl), commit ${commit.slice(0, 7)}`,
      "",
      `Generated by \`node tools/vendor-athenea.mjs --wgsl\`: ${ok} of ${rows.length} compile.`,
      "Modules without entry points are compiled whole; kernels, one entry at a time.",
      "",
      "| file | entry | result | notes |",
      "|---|---|---|---|",
      ...rows.map(
        (r) => `| ${r.map((c) => c.replaceAll("|", "\\|")).join(" | ")} |`,
      ),
      "",
    ].join("\n"),
  );
  console.log(
    `vendor-athenea: WGSL ${ok}/${rows.length} ok, see slang/athenea/WGSL.md`,
  );
}

function firstError(err) {
  const msg = `${err.stdout ?? ""}${err.stderr ?? ""}`;
  const line =
    msg.split("\n").find((l) => /error/i.test(l)) ?? msg.split("\n")[0];
  // E36107 names the feature in a note: say which.
  const uses = [
    ...new Set([...msg.matchAll(/see using of '(\w+)'/g)].map((m) => m[1])),
  ];
  return (
    line
      .replace(/^.*?slang\/athenea\//, "")
      .replace(
        /^error\[E36107\]: unavailable features in entry point/,
        "E36107: not in WGSL",
      )
      .trim()
      .slice(0, 200) + (uses.length ? `: ${uses.join(", ")}` : "")
  );
}
