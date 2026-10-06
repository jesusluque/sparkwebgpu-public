// Compiles every .slang file under slang/ that has entry points to WGSL, and
// writes src/webgpu/generated/<path>.ts exporting the code and its reflection.
//
// The reflection walk follows aopenfx/sdk/cmake/AofxKernelTrailer.cmake: entry
// names and workgroup sizes, then each parameter's binding, its storage-buffer
// element size or its uniform-block size. Here it is emitted as typed JSON
// instead of a binary trailer, and the host builds bind-group layouts from it.
//
// The scalar `static const`s of every .slang file (flags, format codes, enum
// values, float limits) also go to src/webgpu/generated/constants.ts, so host code reads
// them from there instead of repeating the numbers.
//
// Plugin presets (slang/plugins/presets.json) add kernel variants under
// src/webgpu/generated/variants/, listed in generated/presets.ts.
//
//   node tools/slang-build/index.mjs           build all
//   node tools/slang-build/index.mjs --check   fail if a generated file is stale
//
// As a Vite plugin: `import { slangBuild } from "./tools/slang-build/index.mjs"`.

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const SLANG_DIR = path.join(ROOT, "slang");
const OUT_DIR = path.join(ROOT, "src/webgpu/generated");

export function findSlangc() {
  const candidates = [
    process.env.SLANGC,
    process.env.SLANG_ROOT && path.join(process.env.SLANG_ROOT, "bin/slangc"),
    path.join(os.homedir(), "tools/slang/bin/slangc"),
  ].filter(Boolean);
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  try {
    return execFileSync("which", ["slangc"], { encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

// A directory with a VENDOR file holds vendored sources (slang/athenea, see
// tools/vendor-athenea.mjs): imported by our modules, never built on their own.
function listSlang(dir) {
  if (!fs.existsSync(dir)) return [];
  if (fs.existsSync(path.join(dir, "VENDOR"))) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) return listSlang(p);
    return e.name.endsWith(".slang") ? [p] : [];
  });
}

// An entry is a function under a run of attributes that includes either
// [shader("stage")] or, as in the aofx kernels (where CMake names the entries),
// [numthreads(...)], which means compute. Modules without one are import-only.
const ENTRY_RE = /((?:\[[^\]]*\]\s*)+)[\w<>,]+\s+(\w+)\s*\(/g;

function stripComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
}

export function findEntries(source) {
  const entries = [];
  for (const m of stripComments(source).matchAll(ENTRY_RE)) {
    const stage =
      /\[shader\("(\w+)"\)\]/.exec(m[1])?.[1] ??
      (/\[numthreads\s*\(/.test(m[1]) ? "compute" : null);
    if (stage) entries.push({ stage, name: m[2] });
  }
  return entries;
}

function scalarBytes(t) {
  if (!t) return 0;
  for (const s of t.sizes ?? []) {
    if (s.kind === "uniform") return s.value;
  }
  return 0;
}

// Scalar type and component count of a uniform field, for packing by name.
function fieldType(t) {
  if (!t) return {};
  if (t.kind === "scalar") return { scalar: t.scalarType, components: 1 };
  if (t.kind === "vector") {
    return { scalar: t.elementType?.scalarType, components: t.elementCount };
  }
  if (t.kind === "matrix") {
    return {
      scalar: t.elementType?.scalarType,
      components: t.rowCount * t.columnCount,
    };
  }
  return { kind: t.kind };
}

// Slang [format] names of storage textures, as WebGPU formats.
const STORAGE_FORMATS = {
  rgba16f: "rgba16float",
  rgba32f: "rgba32float",
  rgba8: "rgba8unorm",
  r32f: "r32float",
};

export function reflect(json) {
  const bindings = [];
  for (const p of json.parameters ?? []) {
    const b = p.binding ?? {};
    const t = p.type ?? {};
    const entry = {
      name: p.name,
      group: b.space ?? 0,
      binding: b.index ?? 0,
    };
    if (t.kind === "resource" && t.baseShape === "structuredBuffer") {
      entry.kind = t.access === "readWrite" ? "storage" : "read-only-storage";
      entry.elementBytes = scalarBytes(t.resultType);
    } else if (t.kind === "constantBuffer" || t.kind === "parameterBlock") {
      entry.kind = "uniform";
      entry.bytes = t.elementVarLayout?.binding?.size ?? 0;
      entry.fields = (t.elementType?.fields ?? []).map((f) => ({
        name: f.name,
        offset: f.binding?.offset ?? 0,
        size: f.binding?.size ?? 0,
        ...fieldType(f.type),
      }));
    } else if (t.kind === "resource" && t.baseShape === "byteAddressBuffer") {
      entry.kind = t.access === "readWrite" ? "storage" : "read-only-storage";
      entry.elementBytes = 4;
    } else if (t.kind === "resource" && t.baseShape === "texture2D") {
      // Bound by the caller (DispatchArgs.bindings): a Texture2D<float> as
      // unfilterable float (depth textures included), a write-only
      // WTexture2D as a storage texture of its [format].
      entry.kind = "external";
      entry.layout =
        t.access === "write"
          ? {
              storageTexture: {
                access: "write-only",
                format: STORAGE_FORMATS[p.format] ?? p.format,
              },
            }
          : { texture: { sampleType: "unfilterable-float" } };
    } else {
      entry.kind = "unsupported";
      entry.type = t.kind;
    }
    bindings.push(entry);
  }
  const entries = (json.entryPoints ?? []).map((e) => ({
    name: e.name,
    stage: e.stage,
    workgroupSize: e.threadGroupSize ?? null,
    // Filled in by compileOne from the entry's own WGSL.
    uses: [],
  }));
  return { entries, bindings };
}

// `[public] static const uint|int|float NAME = EXPR;` where EXPR is numeric
// literals, constants declared before it in the file and arithmetic (integer
// operators for uint/int). Anything else (vectors, calls) is left to the Slang
// side. Floats keep their declared double value (Slang rounds to f32), and
// 1.0 / 0.0 comes out as Infinity.
const CONST_RE =
  /\b(?:public\s+)?static\s+const\s+(uint|int|float)\s+(\w+)\s*=\s*([^;]+);/g;
const INT_EXPR_RE = /^[\s\d()+\-*/%<>|&^~]*$/;
const FLOAT_EXPR_RE = /^[\s\d.e()+\-*/]*$/;
const INT_LITERAL_RE = /\b(0[xX][0-9a-fA-F]+|\d+)[uU]?\b/g;
const FLOAT_LITERAL_RE =
  /(?<![\w.])(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFhH]?(?![\w.])/g;

export function findConstants(source) {
  const out = {};
  for (const [, type, name, expr] of stripComments(source).matchAll(CONST_RE)) {
    const isFloat = type === "float";
    let known = true;
    const js = expr
      .replace(isFloat ? FLOAT_LITERAL_RE : INT_LITERAL_RE, (lit) =>
        String(Number(lit.replace(/[uUfFhH]$/, ""))),
      )
      .replace(/(?<![\d.])\b[A-Za-z_]\w*\b/g, (id) => {
        if (id in out) return `(${out[id]})`;
        known = false;
        return id;
      });
    if (!known || !(isFloat ? FLOAT_EXPR_RE : INT_EXPR_RE).test(js)) continue;
    const v = Function(`"use strict"; return (${js});`)();
    if (isFloat) {
      if (typeof v === "number" && !Number.isNaN(v)) out[name] = v;
      continue;
    }
    if (!Number.isInteger(v)) continue;
    out[name] = type === "uint" ? v >>> 0 : v | 0;
  }
  return out;
}

// "draw/splat_attrib_draw" -> "drawSplatAttribDraw".
function constantsName(rel) {
  return rel
    .split(/[/_]/)
    .map((w, i) => (i === 0 ? w : w[0].toUpperCase() + w.slice(1)))
    .join("");
}

function constantsModule(byRel) {
  const lines = [
    "// Generated by tools/slang-build from the scalar `static const`s of",
    "// slang/**/*.slang — do not edit.",
  ];
  for (const [rel, constants] of byRel) {
    const entries = Object.entries(constants);
    if (entries.length === 0) continue;
    lines.push(
      "",
      `/** slang/${rel}.slang */`,
      `export const ${constantsName(rel)} = {`,
      ...entries.map(([k, v]) => `  ${k}: ${v},`),
      "} as const;",
    );
  }
  return `${lines.join("\n")}\n`;
}

// Writes `dest` unless it already holds `text`; in check mode records it as
// stale instead. True when written.
function writeIfChanged(dest, text, check, stale, rel) {
  const prev = fs.existsSync(dest) ? fs.readFileSync(dest, "utf8") : null;
  if (prev === text) {
    // Unchanged, but mark it current so upToDate() can skip the next build.
    if (!check) fs.utimesSync(dest, new Date(), new Date());
    return false;
  }
  if (check) {
    stale.push(rel);
    return false;
  }
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, text);
  return true;
}

// Relative import of src/webgpu/KernelModule.ts from generated/<rel>.ts.
function kernelModuleImport(rel) {
  const from = path.dirname(path.join(OUT_DIR, rel));
  const to = path.join(OUT_DIR, "..", "KernelModule");
  return path.relative(from, to).split(path.sep).join("/");
}

function slangc(exe, file, entries, wgslPath, jsonPath) {
  const args = [file, "-target", "wgsl", "-I", SLANG_DIR];
  for (const e of entries) args.push("-entry", e.name, "-stage", e.stage);
  args.push("-o", wgslPath);
  if (jsonPath) args.push("-reflection-json", jsonPath);
  try {
    execFileSync(exe, args, { stdio: ["ignore", "pipe", "pipe"] });
  } catch (err) {
    const msg = `${err.stdout ?? ""}${err.stderr ?? ""}`;
    throw new Error(`slangc failed on ${path.relative(ROOT, file)}:\n${msg}`);
  }
  return fs.readFileSync(wgslPath, "utf8");
}

// "group:binding" of every resource a WGSL module declares.
const BINDING_RE =
  /@(binding|group)\((\d+)\)\s*@(binding|group)\((\d+)\)\s*var/g;

export function declaredBindings(wgsl) {
  const out = new Set();
  for (const m of wgsl.matchAll(BINDING_RE)) {
    const v = { [m[1]]: m[2], [m[3]]: m[4] };
    out.add(`${v.group}:${v.binding}`);
  }
  return out;
}

// Size and alignment of a WGSL type, for the workgroup variables slangc
// emits (scalars, atomics, vectors and arrays of them).
function wgslLayout(type) {
  const t = type.replace(/\s+/g, "");
  if (/^(u32|i32|f32|bool)$/.test(t)) return { size: 4, align: 4 };
  if (t === "f16") return { size: 2, align: 2 };
  let m = t.match(/^atomic<(\w+)>$/);
  if (m) return wgslLayout(m[1]);
  m = t.match(/^vec([234])(?:<(\w+)>|([fhiu]))$/);
  if (m) {
    const n = Number(m[1]);
    const c = wgslLayout(
      m[2] ?? { f: "f32", h: "f16", i: "i32", u: "u32" }[m[3]],
    );
    const align = c.size * (n === 3 ? 4 : n);
    return { size: c.size * n, align };
  }
  m = t.match(/^array<(.+),(?:i32|u32)?\(?(\d+)u?\)?>$/);
  if (m) {
    const e = wgslLayout(m[1]);
    const stride = Math.ceil(e.size / e.align) * e.align;
    return { size: stride * Number(m[2]), align: e.align };
  }
  throw new Error(`slang-build: no layout for workgroup type ${type}`);
}

// The workgroup storage an entry needs against maxComputeWorkgroupStorageSize:
// WebGPU counts roundUp(16, size) for each workgroup variable the entry uses,
// and an entry compiled alone declares only those.
export function workgroupStorageBytes(wgsl) {
  let total = 0;
  for (const m of wgsl.matchAll(/var<workgroup>\s*\w+\s*:\s*([^;]+);/g)) {
    total += Math.ceil(wgslLayout(m[1]).size / 16) * 16;
  }
  return total;
}

function compileOne(exe, file) {
  const source = fs.readFileSync(file, "utf8");
  // A wrapper that #includes a vendored kernel (slang/tests) has its entries.
  const included = [...source.matchAll(/^#include\s+"(athenea\/[^"]+)"/gm)].map(
    (m) => fs.readFileSync(path.join(SLANG_DIR, m[1]), "utf8"),
  );
  const entries = findEntries([source, ...included].join("\n"));
  if (entries.length === 0) return null;
  const rel = path.relative(SLANG_DIR, file).replace(/\.slang$/, "");
  return compileEntries(exe, file, rel, entries);
}

// `file`'s entries (found in `source`, which may be another file it
// includes) compiled to the module generated/<outRel>.ts; `rel` names it.
function compileEntries(exe, file, outRel, entries, name = outRel) {
  const rel = outRel;
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "slang-build-"));
  const jsonPath = path.join(tmp, "out.json");
  const wgsl = slangc(exe, file, entries, path.join(tmp, "out.wgsl"), jsonPath);
  const reflection = reflect(JSON.parse(fs.readFileSync(jsonPath, "utf8")));

  // slangc's reflection does not always say which bindings an entry uses,
  // but compiled alone an entry's WGSL declares only those.
  for (const entry of reflection.entries) {
    const own =
      entries.length === 1
        ? wgsl
        : slangc(
            exe,
            file,
            [entries.find((e) => e.name === entry.name)],
            path.join(tmp, `${entry.name}.wgsl`),
          );
    const declared = declaredBindings(own);
    entry.uses = reflection.bindings
      .filter((b) => declared.has(`${b.group}:${b.binding}`))
      .map((b) => b.name);
    if (entry.stage === "compute") {
      entry.workgroupStorageBytes = workgroupStorageBytes(own);
    }
  }
  fs.rmSync(tmp, { recursive: true, force: true });

  const ts = [
    "// Generated by tools/slang-build from",
    `// ${name === rel ? `slang/${rel}.slang` : name} — do not edit.`,
    `import type { KernelModule } from "${kernelModuleImport(rel)}";`,
    "",
    `export const wgsl = ${JSON.stringify(wgsl)};`,
    "",
    `export const reflection = ${JSON.stringify(reflection, null, 2)} as const;`,
    "",
    `const module: KernelModule = { name: ${JSON.stringify(name)}, wgsl, reflection };`,
    "export default module;",
    "",
  ].join("\n");
  return { rel, ts };
}

// Presets: plugin sets compiled into kernel variants (slang/plugins/
// presets.json). A plugin adds an ISplatColour to generate and/or an
// IBlendTerm to the draw and tile kernels; a preset's colour plugins make
// one generate variant, its blend plugins one variant of each pixel kernel,
// so presets sharing a list share the variant. Each variant compiles a
// wrapper that imports the plugin modules, defines SPLAT_COLOUR or
// SPLAT_BLEND and #includes the kernel, into
// generated/variants/<plugins>/<kernel>.ts; generated/presets.ts lists them
// with loaders, for src/webgpu/plugins/PluginHost.ts.

const PRESETS_FILE = path.join(SLANG_DIR, "plugins/presets.json");
const VARIANTS_DIR = path.join(OUT_DIR, "variants");
/** Kernels a colour variant recompiles. */
export const COLOUR_KERNELS = ["kernels/generate"];
/** Kernels a blend variant recompiles. */
export const BLEND_KERNELS = [
  "draw/splat_draw",
  "draw/splat_attrib_draw",
  "tiles/tile_raster",
];

export function readPresets(file = PRESETS_FILE) {
  if (!fs.existsSync(file)) return { plugins: {}, presets: [] };
  const json = JSON.parse(fs.readFileSync(file, "utf8"));
  return { plugins: json.plugins ?? {}, presets: json.presets ?? [] };
}

// "example.tint+x" -> "example-tint__x": a directory name per plugin list.
export function variantDir(ids) {
  return ids.map((id) => id.replace(/[^\w-]/g, "-")).join("__");
}

export function colourType(ids, plugins) {
  return ids.reduce(
    (t, id) => `ColourChain<${t}, ${plugins[id].colour}>`,
    "DefaultSplatColour",
  );
}

export function blendType(ids, plugins) {
  return ids
    .map((id) => plugins[id].blend)
    .reduce((t, b) => `BlendChain<${t}, ${b}>`);
}

/**
 * The variants a manifest needs: { colour, blend } lists of
 * { key, plugins, modules, type, kernels }, and each preset with the keys of
 * its variants (null: the default kernels).
 */
export function presetVariants({ plugins, presets }) {
  const colour = new Map();
  const blend = new Map();
  const names = new Set();
  const out = [];
  for (const preset of presets) {
    if (!preset.name || names.has(preset.name) || preset.name === "default") {
      throw new Error(
        `presets.json: bad or repeated preset name '${preset.name}'`,
      );
    }
    names.add(preset.name);
    for (const id of preset.plugins ?? []) {
      const p = plugins[id];
      if (!p)
        throw new Error(
          `presets.json: preset '${preset.name}' has unknown plugin '${id}'`,
        );
      if (!p.module || (!p.colour && !p.blend)) {
        throw new Error(
          `presets.json: plugin '${id}' needs a module and a colour or blend struct`,
        );
      }
    }
    const ids = preset.plugins ?? [];
    const add = (map, kind, list, type, kernels) => {
      if (list.length === 0) return null;
      const key = list.join("+");
      if (!map.has(key)) {
        map.set(key, {
          key,
          kind,
          plugins: list,
          modules: [...new Set(list.map((id) => plugins[id].module))],
          type: type(list, plugins),
          kernels,
        });
      }
      return key;
    };
    out.push({
      name: preset.name,
      plugins: ids,
      colour: add(
        colour,
        "colour",
        ids.filter((id) => plugins[id].colour),
        colourType,
        COLOUR_KERNELS,
      ),
      blend: add(
        blend,
        "blend",
        ids.filter((id) => plugins[id].blend),
        blendType,
        BLEND_KERNELS,
      ),
    });
  }
  return {
    colour: [...colour.values()],
    blend: [...blend.values()],
    presets: out,
  };
}

export function wrapperSource(variant, kernelRel) {
  const macro = variant.kind === "colour" ? "SPLAT_COLOUR" : "SPLAT_BLEND";
  return [
    `// slang-build: ${kernelRel} with ${variant.kind} plugins ${variant.plugins.join(", ")}`,
    ...variant.modules.map((m) => `import ${m};`),
    `#define ${macro} ${variant.type}`,
    `#include "${kernelRel}.slang"`,
    "",
  ].join("\n");
}

function presetsModule(variants, plugins) {
  const imp = (v, k) =>
    `      "${k}": (await import("./variants/${variantDir(v.plugins)}/${k}")).default,`;
  const table = (list) =>
    list.flatMap((v) => [
      `  ${JSON.stringify(v.key)}: {`,
      `    plugins: ${JSON.stringify(v.plugins)},`,
      "    load: async () => ({",
      ...v.kernels.map((k) => imp(v, k)),
      "    }),",
      "  },",
    ]);
  const slang = Object.fromEntries(
    Object.entries(plugins).map(([id, p]) => [
      id,
      { module: p.module, colour: p.colour ?? null, blend: p.blend ?? null },
    ]),
  );
  return [
    "// Generated by tools/slang-build from slang/plugins/presets.json — do",
    "// not edit. Kernel variants per plugin list, loaded on demand.",
    'import type { KernelModule } from "../KernelModule";',
    "",
    "export interface KernelVariant {",
    "  readonly plugins: readonly string[];",
    '  /** The variant\'s kernels, by slang/ path ("kernels/generate"...). */',
    "  load(): Promise<Record<string, KernelModule>>;",
    "}",
    "",
    "/** The Slang side of each plugin: module and extension structs. */",
    `export const pluginSlang: Record<string, { module: string; colour: string | null; blend: string | null }> = ${JSON.stringify(slang, null, 2)};`,
    "",
    `export const presets = ${JSON.stringify(variants.presets, null, 2)} as const;`,
    "",
    '/** generate variants, by colour plugin ids joined with "+". */',
    "export const colourVariants: Record<string, KernelVariant> = {",
    ...table(variants.colour),
    "};",
    "",
    '/** Draw and tile variants, by blend plugin ids joined with "+". */',
    "export const blendVariants: Record<string, KernelVariant> = {",
    ...table(variants.blend),
    "};",
    "",
  ].join("\n");
}

// Compiles every variant; returns the generated paths it wrote or kept.
function buildVariants(exe, variants, check, stale, log) {
  const kept = new Set();
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "slang-preset-"));
  try {
    for (const v of [...variants.colour, ...variants.blend]) {
      for (const kernelRel of v.kernels) {
        const kernelFile = path.join(SLANG_DIR, `${kernelRel}.slang`);
        const entries = findEntries(fs.readFileSync(kernelFile, "utf8"));
        const wrapper = path.join(
          tmp,
          `${variantDir(v.plugins)}-${kernelRel.replace(/\//g, "-")}.slang`,
        );
        fs.writeFileSync(wrapper, wrapperSource(v, kernelRel));
        const outRel = `variants/${variantDir(v.plugins)}/${kernelRel}`;
        const out = compileEntries(
          exe,
          wrapper,
          outRel,
          entries,
          `${kernelRel}+${v.key}`,
        );
        kept.add(path.join(OUT_DIR, `${outRel}.ts`));
        if (
          writeIfChanged(
            path.join(OUT_DIR, `${outRel}.ts`),
            out.ts,
            check,
            stale,
            outRel,
          )
        ) {
          log(`slang-build: ${outRel}`);
        }
      }
    }
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
  return kept;
}

// Removes variant modules no preset makes any more (or, checking, reports them).
function pruneVariants(kept, check, stale, log) {
  if (!fs.existsSync(VARIANTS_DIR)) return;
  for (const f of fs.readdirSync(VARIANTS_DIR, { recursive: true })) {
    const file = path.join(VARIANTS_DIR, String(f));
    if (!file.endsWith(".ts") || kept.has(file)) continue;
    const rel = path.relative(OUT_DIR, file);
    if (check) {
      stale.push(rel);
      continue;
    }
    fs.rmSync(file);
    log(`slang-build: removed ${rel}`);
  }
  // Empty directories left behind.
  const prune = (dir) => {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      if (e.isDirectory()) prune(path.join(dir, e.name));
    }
    if (dir !== VARIANTS_DIR && fs.readdirSync(dir).length === 0)
      fs.rmdirSync(dir);
  };
  if (!check) prune(VARIANTS_DIR);
}

export function buildAll({ check = false, log = console.log } = {}) {
  const exe = findSlangc();
  if (!exe) {
    throw new Error(
      "slangc not found: set SLANGC or SLANG_ROOT, or install to ~/tools/slang/bin",
    );
  }
  const stale = [];
  const constants = [];
  let count = 0;
  for (const file of listSlang(SLANG_DIR)) {
    const rel = path.relative(SLANG_DIR, file).replace(/\.slang$/, "");
    constants.push([rel, findConstants(fs.readFileSync(file, "utf8"))]);
    const out = compileOne(exe, file);
    if (!out) continue;
    count += 1;
    const dest = path.join(OUT_DIR, `${out.rel}.ts`);
    if (writeIfChanged(dest, out.ts, check, stale, out.rel)) {
      log(`slang-build: ${out.rel}`);
    }
  }
  const manifest = readPresets();
  const variants = presetVariants(manifest);
  const kept = buildVariants(exe, variants, check, stale, log);
  count += kept.size;
  pruneVariants(kept, check, stale, log);
  if (
    writeIfChanged(
      path.join(OUT_DIR, "presets.ts"),
      presetsModule(variants, manifest.plugins),
      check,
      stale,
      "presets",
    )
  ) {
    log("slang-build: presets");
  }
  constants.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  const dest = path.join(OUT_DIR, "constants.ts");
  if (
    writeIfChanged(dest, constantsModule(constants), check, stale, "constants")
  ) {
    log("slang-build: constants");
  }
  if (stale.length) {
    throw new Error(`slang-build: stale generated files: ${stale.join(", ")}`);
  }
  return count;
}

// Vite plugin: rebuild on .slang changes in dev. A missing slangc is only a
// warning, since the generated files are committed.
// True when no .slang file is newer than the oldest generated module, so a
// build can skip slangc. Modules import each other, hence the global check.
export function upToDate() {
  const sources = listSlang(SLANG_DIR);
  if (fs.existsSync(PRESETS_FILE)) sources.push(PRESETS_FILE);
  const outputs = fs.existsSync(OUT_DIR)
    ? fs
        .readdirSync(OUT_DIR, { recursive: true })
        .filter((f) => String(f).endsWith(".ts"))
    : [];
  if (sources.length === 0 || outputs.length === 0) return false;
  const newest = Math.max(...sources.map((f) => fs.statSync(f).mtimeMs));
  const oldest = Math.min(
    ...outputs.map((f) => fs.statSync(path.join(OUT_DIR, String(f))).mtimeMs),
  );
  return newest <= oldest;
}

export function slangBuild() {
  const run = () => {
    if (upToDate()) return;
    try {
      buildAll({ log: () => {} });
    } catch (err) {
      console.warn(String(err.message ?? err));
    }
  };
  return {
    name: "slang-build",
    buildStart: run,
    handleHotUpdate({ file }) {
      if (file.endsWith(".slang") || file === PRESETS_FILE) run();
    },
  };
}

if (
  process.argv[1] &&
  fileURLToPath(import.meta.url) === path.resolve(process.argv[1])
) {
  const check = process.argv.includes("--check");
  try {
    const n = buildAll({ check });
    console.log(
      `slang-build: ${n} module(s) ${check ? "up to date" : "built"}`,
    );
  } catch (err) {
    console.error(err.message);
    process.exit(1);
  }
}
