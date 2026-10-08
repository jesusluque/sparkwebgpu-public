#!/usr/bin/env node
// A candidate set from a published one and `athc-convert --scene ... --report`
// (thread BB: hidden splats dropped, opaque paint decimated):
//
//   node scripts/build-pruned-set.mjs --source publish-r2/sparkwebgpu/corvette-v2-hd \
//     --pruned /tmp/v6-hd --out publish-r2/sparkwebgpu/corvette-v6-hd [--note "..."]
//
// Copies the source folder (hdri, clips, catcher, CREDITS, ...), puts the
// pruned .athc / .lights.athl in place of the source's (a hashed name gets
// the new bytes' hash, as build-corvette-lights.mjs names them), and updates
// the scene JSON files (every name, and a part's bytes, splats, tierBytes,
// athlBytes; the totals). Nothing is uploaded.
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const args = process.argv.slice(2);
const arg = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};
const source = arg("--source");
const pruned = arg("--pruned");
const out = arg("--out");
const note = arg("--note");
if (!source || !pruned || !out) {
  console.error(
    "usage: build-pruned-set.mjs --source DIR --pruned DIR --out DIR [--note TEXT]",
  );
  process.exit(1);
}
const report = JSON.parse(
  fs.readFileSync(path.join(pruned, "report.json"), "utf8"),
);
const hash10 = (file) =>
  createHash("sha256").update(fs.readFileSync(file)).digest("hex").slice(0, 10);
const HASHED = /^(.*)-[0-9a-f]{10}(\.athc|\.lights\.athl)$/;
const renamed = new Map(); // old name -> new name
const newName = (oldName, file) => {
  const m = oldName.match(HASHED);
  return m ? `${m[1]}-${hash10(file)}${m[2]}` : oldName;
};

fs.mkdirSync(out, { recursive: true });
const byOld = new Map();
for (const f of report.files) {
  const oldName = path.basename(f.input);
  const name = newName(oldName, f.output);
  renamed.set(oldName, name);
  byOld.set(oldName, f);
  fs.copyFileSync(f.output, path.join(out, name));
  if (f.athl) {
    const oldAthl = path.basename(f.athl.input);
    const athl = newName(oldAthl, f.athl.output);
    renamed.set(oldAthl, athl);
    fs.copyFileSync(f.athl.output, path.join(out, athl));
  }
}
// Everything else of the source, as it is.
const copy = (from, to) => {
  for (const e of fs.readdirSync(from, { withFileTypes: true })) {
    if (e.name === ".wrangler" || renamed.has(e.name)) continue;
    const a = path.join(from, e.name);
    const b = path.join(to, e.name);
    if (e.isDirectory()) {
      fs.mkdirSync(b, { recursive: true });
      copy(a, b);
    } else if (!e.name.endsWith(".json")) {
      fs.copyFileSync(a, b);
    }
  }
};
copy(source, out);

const rename = (v) => {
  if (typeof v === "string") return renamed.get(v) ?? v;
  if (Array.isArray(v)) return v.map(rename);
  if (v && typeof v === "object") {
    for (const k of Object.keys(v)) v[k] = rename(v[k]);
  }
  return v;
};
const decimated = (f) =>
  f.decimate
    ? `, opaque splats decimated ${f.decimate.candidates} -> ${f.decimate.output - (f.inputSplats - f.hidden - f.decimate.candidates)}`
    : "";
const pruneNote = (f) =>
  `thread BB: ${f.hidden} splats no view sees dropped (athc-convert --drop-hidden)${decimated(f)}, levels built again; ${f.inputSplats} -> ${f.splats} splats`;
for (const name of fs.readdirSync(source).filter((n) => n.endsWith(".json"))) {
  const json = JSON.parse(fs.readFileSync(path.join(source, name), "utf8"));
  // A part (corvette.json) ...
  for (const p of json.parts ?? []) {
    const f = byOld.get(p.file);
    if (!f) continue;
    p.bytes = f.bytes;
    p.splats = f.splats;
    if (p.tierBytes) p.tierBytes = f.tierBytes;
    if (f.athl) p.athlBytes = f.athl.bytes;
    p.pruned = {
      inputSplats: f.inputSplats,
      hidden: f.hidden,
      decimate: f.decimate ?? null,
    };
    p.origin = `${p.origin ?? ""}${p.origin ? "; " : ""}${pruneNote(f)}`;
  }
  if (json.parts) {
    json.splats = json.parts.reduce((s, p) => s + (p.splats ?? 0), 0);
    json.bytes = json.parts.reduce((s, p) => s + (p.bytes ?? 0), 0);
  }
  // ... or a whole cloud (sparrow.json).
  if (json.cloud && byOld.has(json.cloud)) {
    const f = byOld.get(json.cloud);
    json.cloudBytes = f.bytes;
    json.splats = f.splats;
    json.pruned = {
      inputSplats: f.inputSplats,
      hidden: f.hidden,
      note: pruneNote(f),
    };
  }
  rename(json);
  if (note) json.candidate = note;
  fs.writeFileSync(path.join(out, name), `${JSON.stringify(json, null, 1)}\n`);
}
for (const [a, b] of renamed) console.log(`${a} -> ${b}`);
console.log(`${out}: written`);
