// athenea's .athc (rust/spark-lib/src/athc.rs has the layout): paging it as a
// Spark LoD tree, and decoding the streams it carries beside the splats.
//
// A .athc is one Spark LoD tree in a virtual index space (athc.rs
// VirtualTree): the root and athenea's merged levels first, then, from the
// next multiple of 65 536, the splats, so that each athenea chunk of 65 536
// splats is exactly one Spark page. Opening it reads the file's first bytes
// through the end of its levels (one range request); the merged pages are
// built from them (in a worker) as ATHV blobs, and every splat page is one
// range request of its chunk, wrapped in a 160-byte ATHV head. The loader
// worker decodes both kinds through the usual MultiDecoder.
//
// A version 3 file (docs/docs/athc-v3.md) is paged by sections: what a page
// fetches is the one Range of its block holding the sections it wants. The
// splats (CORE, SHRS) go to the loader worker as a kind-2 ATHV page; the
// streams (MATL, SHAD, TX*) are put together here as attribute columns, as
// their words are stored, with no decode. Which streams a page wants is
// the pager's (WgpuSplatPager.streamsToFetch); a page that did not fetch a
// stream group fetches it later on its own (fetchAthcStreams), one Range.

import { athc_layout, athc_prefix_bytes, athc_skeleton } from "spark-rs";
import { workerPool } from "./SplatWorker";
import * as wasm from "./wasm";
import type { AttribFormat, AttributeSpec } from "./webgpu/attributes/schema";
import { AttribPool, attribWords } from "./webgpu/attributes/schema";

export const ATHC_MAGIC = 0x43485441; // "ATHC"
export const ATH3_MAGIC = 0x33485441; // "ATH3"
export const ATHV_MAGIC = 0x56485441; // "ATHV"
export const ATHV_HEAD = 160;
export const ATHV_SPLATS = 1;
/** A v3 block's sections (rust athc_v3.rs athv_sections_page). */
export const ATHV_SECTIONS = 2;
/** ATHV head decode flags (byte 152): keep a linear cloud's colours linear. */
export const ATHV_KEEP_LINEAR = 1;
export const ATHC_PAGE_SPLATS = 65536;

export const ATHC_FLAGS = {
  normals: 1,
  linear: 2,
  emission: 4,
  material: 16,
  transfer: 32,
} as const;

export type AthcHeader = {
  version: number;
  count: number;
  restPerColour: number;
  shWords: number;
  levels: number;
  chunkSplats: number;
  chunks: number;
  finestGroups: number;
  boundsLo: number[];
  extent: number;
  boundsMin: number[];
  boundsMax: number[];
  flags: number;
  levelTable: number;
  chunkTable: number;
  starts: number;
};

export type AthcLayout = {
  version?: 1 | 2;
  header: AthcHeader;
  extra: {
    pbrWords: number;
    lobesWords: number;
    transferCount: number;
    transferWords: number;
    shadowWords: number;
    /** sparkwebGPU's curvature (v3 section CURV): 0 or 2 words a splat. */
    curvatureWords?: number;
  };
  levels: { level: number; groups: number; offset: number }[];
  chunks: { offset: number; count: number }[];
  elementBytes: number;
  fileBytes: number;
  /** Bytes before the first chunk: headers, tables, starts and levels. */
  levelsEnd: number;
  /** The attributes its splats carry (spark-lib athc::attrib_specs). */
  attribSpecs: {
    name: string;
    format: string;
    components: number;
    lodMerge: string;
  }[];
};

/** A v3 section (athc_v3.rs Section). */
export type AthcSection = {
  id:
    | "CORE"
    | "SHRS"
    | "SKIN"
    | "MATL"
    | "SHAD"
    | "CURV"
    | "TXDI"
    | "TXIN"
    | "TXFD";
  tier: number;
  encoding: number;
  compression: number;
  words: number;
};

/** A v3 block: a level's groups (kind 0) or a chunk of splats (kind 1). */
export type AthcBlockEntry = {
  kind: number;
  level: number;
  first: number;
  n: number;
  sphere: number[];
  spans: { offset: number; stored: number; raw: number }[];
};

/** A version 3 file's tables (athc_layout of an "ATH3" prefix). */
export type AthcV3Layout = {
  version: 3;
  header: AthcHeader;
  extra: AthcLayout["extra"];
  sections: AthcSection[];
  blocks: AthcBlockEntry[];
  startsOffset: number;
  fileBytes: number;
  /** v2 FileHeader + ExtraHeader (136 bytes), as an ATHV head carries them. */
  headers: Uint8Array;
  attribSpecs: AthcLayout["attribSpecs"];
  /** The transfer forms it can be kept as (athc_v3.rs transfer_forms). */
  transferForms: number[];
};

/**
 * The transfer words a splat in TXDI, and through TXIN, as the file's
 * sections hold them (a file of the first v3 layout keeps the field in TXIN).
 */
function transferSplit(layout: AthcV3Layout): [number, number] {
  const words = (id: AthcSection["id"]) =>
    layout.sections.find((s) => s.id === id)?.words ?? 0;
  const direct = words("TXDI");
  return [direct, direct + words("TXIN")];
}

export type AnyAthcLayout = AthcLayout | AthcV3Layout;

export function isAthcV3(layout: AnyAthcLayout): layout is AthcV3Layout {
  return layout.version === 3;
}

/** What a reader keeps of the optional streams (athc_v3.rs Want). */
export type AthcWant = { material: boolean; transferValues: number };

export const ATHC_MATERIAL_STREAMS = [
  "normalOct",
  "emission",
  "pbr",
  "lobes",
] as const;
export const ATHC_RELIGHT_STREAMS = [
  "shadowBits",
  "transfer",
  "curvature",
] as const;

/**
 * The Want of attribute `names` (the pool's specs, so the transfer's
 * components are the form kept): the material if any material stream is
 * named, the relight streams if the transfer or the shadow bits are.
 */
export function athcWantOf(
  layout: AnyAthcLayout,
  specs: readonly Pick<AttributeSpec, "name" | "components">[],
): AthcWant {
  const has = (n: string) => specs.some((s) => s.name === n);
  const transfer = specs.find((s) => s.name === "transfer");
  const relight = ATHC_RELIGHT_STREAMS.some(has);
  return {
    material: ATHC_MATERIAL_STREAMS.some(has),
    transferValues: relight
      ? (transfer?.components ?? layout.extra.transferCount)
      : 0,
  };
}

/** Whether a v3 section is needed for `want` (athc_v3.rs Want::needs). */
export function athcNeeds(
  layout: AthcV3Layout,
  id: AthcSection["id"],
  want: AthcWant,
): boolean {
  const [direct, indirect] = transferSplit(layout);
  const words = Math.ceil(want.transferValues / 2);
  switch (id) {
    case "CORE":
    case "SHRS":
      return true;
    // A skinned cloud's rig (athc_skin.rs): read whole-file only for now;
    // a paged skinned cloud is drawn in its bind pose.
    case "SKIN":
      return false;
    case "MATL":
      return want.material;
    case "SHAD":
    case "CURV":
      return want.transferValues > 0;
    case "TXDI":
      return words > 0;
    case "TXIN":
      return words > direct;
    case "TXFD":
      return words > indirect;
  }
}

export type AthcTree = {
  synthRoot: boolean;
  levelBase: number[];
  merged: number;
  splatBase: number;
  count: number;
  clippedGroups: number;
};

export type AthcPaging = {
  layout: AnyAthcLayout;
  tree: AthcTree;
  /** The FileHeader and ExtraHeader, as an ATHV head carries them. */
  headers: Uint8Array;
  /** ATHV blobs of the merged pages, 0 .. mergedPages - 1. */
  mergedPages: Uint8Array[];
  /** Virtual pages: merged ones, then one per chunk from splatBase / 65 536. */
  pageCount: number;
  /**
   * The pages ask the decoder to keep a linear cloud's colours linear
   * (ATHV_KEEP_LINEAR) instead of encoding them to sRGB for Spark's blend:
   * for a renderer that blends in linear light (the athenea raster plugin).
   */
  keepLinear: boolean;
  /** Version 3: what the merged pages (the levels) were fetched with. */
  levelsWant?: AthcWant;
};

type FetchOptions = {
  url?: string;
  fileBytes?: Uint8Array;
  requestHeader?: Record<string, string>;
  withCredentials?: boolean;
  signal?: AbortSignal;
};

/** Bytes offset .. offset + bytes of the file, and the file's size. */
async function readRange(
  { url, fileBytes, requestHeader, withCredentials, signal }: FetchOptions,
  offset: number,
  bytes: number,
): Promise<{ data: Uint8Array; total: number }> {
  if (fileBytes) {
    return {
      data: fileBytes.subarray(offset, offset + bytes),
      total: fileBytes.length,
    };
  }
  if (!url) throw new Error(".athc: no url or fileBytes");
  const headers = new Headers(requestHeader);
  headers.set("Range", `bytes=${offset}-${offset + bytes - 1}`);
  const response = await fetch(url, {
    headers,
    credentials: withCredentials ? "include" : "same-origin",
    signal,
  });
  if (!response.ok) {
    throw new Error(
      `Failed to fetch "${url}": ${response.status} ${response.statusText}`,
    );
  }
  const body = new Uint8Array(await response.arrayBuffer());
  if (response.status === 206) {
    const total = Number(
      /\/(\d+)\s*$/.exec(response.headers.get("Content-Range") ?? "")?.[1],
    );
    return { data: body, total: Number.isFinite(total) ? total : Number.NaN };
  }
  // A server that ignores Range sends the whole file.
  return { data: body.subarray(offset, offset + bytes), total: body.length };
}

/** The headers and tables of a .athc, from its first bytes. */
export async function readAthcLayout(
  options: FetchOptions,
): Promise<{ layout: AnyAthcLayout; prefix: Uint8Array }> {
  await wasm.initialization;
  const first = await readRange(options, 0, 65536);
  let prefix = first.data;
  const total = first.total;
  if (!Number.isFinite(total)) {
    throw new Error(".athc: the server did not say how long the file is");
  }
  const need = athc_prefix_bytes(prefix.slice(0, 4096));
  if (need > prefix.length) {
    prefix = (await readRange(options, 0, need)).data;
  }
  const layout = athc_layout(prefix, total) as AnyAthcLayout;
  return { layout, prefix: prefix.subarray(0, Math.max(need, 136)) };
}

/** One animation of a skinned .athc (athc_skin.rs SkinClip). */
export type AthcSkinClip = {
  name: string;
  timeCodesPerSecond: number;
  /** Time codes of the samples, increasing. */
  times: Float32Array;
  /** samples x joints x 16: each joint's transform, USD rows (p' = p M). */
  xforms: Float32Array;
};

/**
 * The skeleton of a skinned .athc v3 (athenea's AtheneaSplatSkinningAPI /
 * UsdSkel binding as usd-athc writes it; docs/docs/athc-v3.md, SKIN).
 */
export type AthcSkeleton = {
  /** Influences a splat (attribute skinInfluences, u32 each). */
  influences: number;
  /** Gradient words a splat (attribute skinGradients), 0 or influences - 1. */
  gradientWords: number;
  joints: string[];
  skeleton: string;
  /** skel:geomBindTransform, USD rows. */
  geomBind: Float32Array;
  clips: AthcSkinClip[];
};

/** The skeleton of a skinned .athc (null when it has none), from its tables. */
export async function readAthcSkeleton(
  options: FetchOptions,
): Promise<AthcSkeleton | null> {
  const { prefix } = await readAthcLayout(options);
  return athc_skeleton(prefix) as AthcSkeleton | null;
}

/** Merged nodes and the virtual index of the first splat (athc.rs VirtualTree). */
export function athcSplatBase(layout: AnyAthcLayout): {
  merged: number;
  splatBase: number;
} {
  const groups = isAthcV3(layout)
    ? layout.blocks.filter((b) => b.kind === 0).map((b) => b.n)
    : layout.levels.map((l) => l.groups);
  const merged = (groups[0] > 1 ? 1 : 0) + groups.reduce((a, b) => a + b, 0);
  return {
    merged,
    splatBase: Math.ceil(merged / ATHC_PAGE_SPLATS) * ATHC_PAGE_SPLATS,
  };
}

/** Virtual pages of a .athc: its merged pages, then one per chunk. */
export function athcPageCount(layout: AnyAthcLayout) {
  return (
    athcSplatBase(layout).splatBase / ATHC_PAGE_SPLATS + layout.header.chunks
  );
}

/**
 * Opens a .athc for paging: its layout, its virtual tree and the ATHV blobs
 * of its merged pages (built in a loader worker from the file's levels).
 */
export async function openAthc(
  options: FetchOptions & {
    keepLinear?: boolean;
    /** An already read layout (readAthcLayout). */
    read?: { layout: AnyAthcLayout; prefix: Uint8Array };
    /** Version 3: what the levels keep (default everything). */
    want?: (layout: AnyAthcLayout) => AthcWant;
  },
): Promise<AthcPaging> {
  const read = options.read ?? (await readAthcLayout(options));
  if (isAthcV3(read.layout)) {
    return openAthcV3(options, read.layout, read.prefix);
  }
  const { prefix } = read;
  const layout = read.layout;
  const levels =
    prefix.length >= layout.levelsEnd
      ? prefix.subarray(0, layout.levelsEnd)
      : (await readRange(options, 0, layout.levelsEnd)).data;
  const { tree: treeObject, pages } = await workerPool.withWorker((worker) =>
    worker.call("athcMergedPages", {
      prefix: levels.slice(),
      fileBytes: layout.fileBytes,
    }),
  );
  const tree = treeObject as unknown as AthcTree;
  const chunkPages = layout.header.chunks;
  if (layout.header.chunkSplats !== ATHC_PAGE_SPLATS && chunkPages > 1) {
    throw new Error(
      `.athc: chunks of ${layout.header.chunkSplats} splats; paging needs ${ATHC_PAGE_SPLATS}`,
    );
  }
  const keepLinear = options.keepLinear ?? false;
  if (keepLinear) {
    for (const page of pages) {
      new DataView(page.buffer, page.byteOffset).setUint32(
        152,
        ATHV_KEEP_LINEAR,
        true,
      );
    }
  }
  return {
    layout,
    tree,
    headers: prefix.slice(0, 136),
    mergedPages: pages,
    pageCount: tree.splatBase / ATHC_PAGE_SPLATS + chunkPages,
    keepLinear,
  };
}

/** The kind-2 ATHV page of a v3 block's sections (athc_v3.rs athv_sections_page). */
export function athvSectionsPage(
  layout: AthcV3Layout,
  base: number,
  n: number,
  flags: number,
  transferValues: number,
  parts: { section: AthcSection; stored: Uint8Array; raw: number }[],
): Uint8Array {
  const table = 4 + 16 * parts.length;
  const bytes = parts.reduce((t, p) => t + p.stored.length, 0);
  const out = new Uint8Array(ATHV_HEAD + table + bytes);
  out.set(athvHead(layout.headers, base, n, flags, ATHV_SECTIONS));
  const view = new DataView(out.buffer);
  view.setUint32(156, transferValues, true);
  view.setUint32(ATHV_HEAD, parts.length, true);
  let at = ATHV_HEAD + table;
  parts.forEach(({ section, stored, raw }, k) => {
    const e = ATHV_HEAD + 4 + 16 * k;
    view.setUint32(e, fourcc(section.id), true);
    view.setUint32(
      e + 4,
      (section.compression | (section.encoding << 16)) >>> 0,
      true,
    );
    view.setUint32(e + 8, stored.length, true);
    view.setUint32(e + 12, raw, true);
    out.set(stored, at);
    at += stored.length;
  });
  return out;
}

function fourcc(id: string) {
  return (
    (id.charCodeAt(0) |
      (id.charCodeAt(1) << 8) |
      (id.charCodeAt(2) << 16) |
      (id.charCodeAt(3) << 24)) >>>
    0
  );
}

/**
 * Block `block`'s sections `ids` (those it has), read with one Range
 * request from the first one's start to the last one's end.
 */
async function readSections(
  layout: AthcV3Layout,
  block: AthcBlockEntry,
  ids: readonly AthcSection["id"][],
  options: FetchOptions,
): Promise<{ section: AthcSection; stored: Uint8Array; raw: number }[]> {
  const picks = layout.sections
    .map((section, k) => ({ section, k }))
    .filter(({ section }) => ids.includes(section.id));
  if (!picks.length) return [];
  const first = block.spans[picks[0].k];
  const last = block.spans[picks[picks.length - 1].k];
  const start = first.offset;
  const { data } = await readRange(
    options,
    start,
    last.offset + last.stored - start,
  );
  return picks.map(({ section, k }) => {
    const span = block.spans[k];
    return {
      section,
      stored: data.subarray(
        span.offset - start,
        span.offset - start + span.stored,
      ),
      raw: span.raw,
    };
  });
}

function sectionsFor(layout: AthcV3Layout, want: AthcWant, core: boolean) {
  return layout.sections
    .map((s) => s.id)
    .filter(
      (id) =>
        athcNeeds(layout, id, want) &&
        (core || (id !== "CORE" && id !== "SHRS")),
    );
}

async function openAthcV3(
  options: FetchOptions & {
    keepLinear?: boolean;
    want?: (layout: AnyAthcLayout) => AthcWant;
  },
  layout: AthcV3Layout,
  tables: Uint8Array,
): Promise<AthcPaging> {
  const levelsWant = options.want?.(layout) ?? {
    material: true,
    transferValues: layout.extra.transferCount,
  };
  const keepLinear = options.keepLinear ?? false;
  const flags = keepLinear ? ATHV_KEEP_LINEAR : 0;
  const ids = sectionsFor(layout, levelsWant, true);
  // One Range request a level block.
  const levels = await Promise.all(
    layout.blocks
      .filter((b) => b.kind === 0)
      .map(async (b) =>
        athvSectionsPage(
          layout,
          0,
          b.n,
          flags,
          levelsWant.transferValues,
          await readSections(layout, b, ids, options),
        ),
      ),
  );
  const { tree: treeObject, pages } = await workerPool.withWorker((worker) =>
    worker.call("athc3MergedPages", { tables: tables.slice(), levels }),
  );
  const tree = treeObject as unknown as AthcTree;
  if (
    layout.header.chunkSplats !== ATHC_PAGE_SPLATS &&
    layout.header.chunks > 1
  ) {
    throw new Error(
      `.athc: chunks of ${layout.header.chunkSplats} splats; paging needs ${ATHC_PAGE_SPLATS}`,
    );
  }
  if (keepLinear) {
    for (const page of pages) {
      new DataView(page.buffer, page.byteOffset).setUint32(
        152,
        ATHV_KEEP_LINEAR,
        true,
      );
    }
  }
  return {
    layout,
    tree,
    headers: layout.headers,
    mergedPages: pages,
    pageCount: tree.splatBase / ATHC_PAGE_SPLATS + layout.header.chunks,
    keepLinear,
    levelsWant,
  };
}

/** The ATHV head of a page of `n` splats at virtual index `base`. */
export function athvHead(
  headers: Uint8Array,
  base: number,
  n: number,
  flags = 0,
  kind = ATHV_SPLATS,
) {
  const head = new Uint8Array(ATHV_HEAD);
  const view = new DataView(head.buffer);
  view.setUint32(0, ATHV_MAGIC, true);
  view.setUint32(4, kind, true);
  view.setUint32(8, base, true);
  view.setUint32(12, n, true);
  head.set(headers.subarray(0, 136), 16);
  view.setUint32(152, flags, true);
  return head;
}

/** Virtual page `page` of an opened .athc, as the loader worker decodes it. */
export async function fetchAthcPage(
  paging: AthcPaging,
  page: number,
  options: FetchOptions,
): Promise<Uint8Array> {
  if (page < 0 || page >= paging.pageCount) {
    throw new Error(`.athc page ${page} of ${paging.pageCount}`);
  }
  const firstChunkPage = paging.tree.splatBase / ATHC_PAGE_SPLATS;
  if (page < paging.mergedPages.length) {
    // A copy: the loader worker takes its bytes, and an upgrade decodes it again.
    return paging.mergedPages[page].slice();
  }
  if (page < firstChunkPage) {
    throw new Error(`.athc page ${page} is between the levels and the splats`);
  }
  const chunk = page - firstChunkPage;
  if (isAthcV3(paging.layout)) {
    return (await fetchAthc3Page(paging, page, options, [])).page;
  }
  const { offset, count } = paging.layout.chunks[chunk];
  const bytes = count * paging.layout.elementBytes;
  const { data } = await readRange(options, offset, bytes);
  const out = new Uint8Array(ATHV_HEAD + bytes);
  out.set(
    athvHead(
      paging.headers,
      page * ATHC_PAGE_SPLATS,
      count,
      paging.keepLinear ? ATHV_KEEP_LINEAR : 0,
    ),
  );
  out.set(data, ATHV_HEAD);
  return out;
}

/** Chunk block `chunk` of a v3 layout. */
function chunkBlock(layout: AthcV3Layout, chunk: number): AthcBlockEntry {
  const block = layout.blocks.filter((b) => b.kind === 1)[chunk];
  if (!block) throw new Error(`.athc v3: no chunk ${chunk}`);
  return block;
}

/**
 * Virtual page `page` (a chunk) of a v3 file with the streams `streams`
 * (the pool's specs): one Range request from CORE through the last section
 * they need. The splats as a kind-2 ATHV page for the loader worker, the
 * streams as attribute columns of their stored words.
 */
export async function fetchAthc3Page(
  paging: AthcPaging,
  page: number,
  options: FetchOptions,
  streams: readonly AttributeSpec[],
): Promise<{ page: Uint8Array; streams: AttribPool | null }> {
  const layout = paging.layout as AthcV3Layout;
  const chunk = page - paging.tree.splatBase / ATHC_PAGE_SPLATS;
  const block = chunkBlock(layout, chunk);
  const want = athcWantOf(layout, streams);
  const parts = await readSections(
    layout,
    block,
    sectionsFor(layout, want, true),
    options,
  );
  const core = parts.filter(
    (p) => p.section.id === "CORE" || p.section.id === "SHRS",
  );
  const bytes = athvSectionsPage(
    layout,
    page * ATHC_PAGE_SPLATS,
    block.n,
    paging.keepLinear ? ATHV_KEEP_LINEAR : 0,
    0,
    core,
  );
  const rest = parts.filter((p) => !core.includes(p));
  return {
    page: bytes,
    streams: rest.length
      ? await streamColumns(layout, block.n, rest, streams)
      : null,
  };
}

/**
 * The streams `names` of virtual page `page` on their own (an upgrade of a
 * page that did not fetch them): one Range request of a v3 block's
 * sections, or of a v2 chunk's arrays. Null for a merged page (they are in
 * its ATHV blob: decode it again).
 */
export async function fetchAthcStreams(
  paging: AthcPaging,
  page: number,
  options: FetchOptions,
  specs: readonly AttributeSpec[],
): Promise<AttribPool | null> {
  const chunk = page - paging.tree.splatBase / ATHC_PAGE_SPLATS;
  if (chunk < 0 || !specs.length) return null;
  const { layout } = paging;
  if (isAthcV3(layout)) {
    const block = chunkBlock(layout, chunk);
    const want = athcWantOf(layout, specs);
    const parts = await readSections(
      layout,
      block,
      sectionsFor(layout, want, false),
      options,
    );
    return streamColumns(layout, block.n, parts, specs);
  }
  // Version 2: the block's arrays, one after the other.
  const { offset, count: n } = layout.chunks[chunk];
  const arrays = v2Arrays(layout);
  const wanted = arrays.filter((a) => specs.some((s) => s.name === a.name));
  if (!wanted.length) return null;
  const from = wanted[0].word * n * 4;
  const last = wanted[wanted.length - 1];
  const to = (last.word + last.words) * n * 4;
  const { data } = await readRange(options, offset + from, to - from);
  const words = new Uint32Array(data.slice().buffer);
  const pool = new AttribPool(n);
  for (const spec of specs) {
    const a = wanted.find((w) => w.name === spec.name);
    if (!a) continue;
    const planar = words.subarray(
      a.word * n - from / 4,
      (a.word + a.words) * n - from / 4,
    );
    pool.setColumn({
      spec: { ...spec },
      words: assembleRows(n, [{ data: planar, words: a.words }], spec),
    });
  }
  return pool;
}

/** A v2 block's stream arrays: name, first word (times n) and words a splat. */
function v2Arrays(layout: AthcLayout) {
  const { header: h, extra: x } = layout;
  let word = 4 + 4 + h.shWords + 1;
  const out: { name: string; word: number; words: number }[] = [];
  const add = (on: boolean, name: string, words: number) => {
    if (!on) return;
    out.push({ name, word, words });
    word += words;
  };
  add((h.flags & ATHC_FLAGS.normals) !== 0, "normalOct", 1);
  add((h.flags & ATHC_FLAGS.emission) !== 0, "emission", 1);
  add(x.pbrWords > 0, "pbr", x.pbrWords);
  add(x.lobesWords > 0, "lobes", x.lobesWords);
  add(x.transferWords > 0, "transfer", x.transferWords);
  add(x.shadowWords > 0, "shadowBits", x.shadowWords);
  return out;
}

/** Each splat's row of `spec`: the parts' rows one after the other, the prefix it keeps. */
function assembleRows(
  n: number,
  parts: { data: Uint32Array; words: number }[],
  spec: Pick<AttributeSpec, "format" | "components">,
): Uint32Array {
  const keep = attribWords(spec);
  const out = new Uint32Array(n * keep);
  if (parts.length === 1 && parts[0].words === keep) {
    out.set(parts[0].data.subarray(0, n * keep));
    return out;
  }
  for (let i = 0; i < n; i++) {
    let at = i * keep;
    const end = at + keep;
    for (const { data, words } of parts) {
      if (at >= end) break;
      const take = Math.min(words, end - at);
      out.set(data.subarray(i * words, i * words + take), at);
      at += take;
    }
  }
  // An odd count of halves: the unused half is zero.
  const bits = (spec.components * (spec.format === "f16" ? 16 : 32)) % 32;
  if (bits) for (let i = 0; i < n; i++) out[i * keep + keep - 1] &= 0xffff;
  return out;
}

/** gzip (RFC 1952) as the browser reads it. */
async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as BlobPart])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** A section's arrays, in words an element (athc_v3.rs section_arrays). */
function sectionArrays(layout: AthcV3Layout, section: AthcSection): number[] {
  const { header: h, extra: x } = layout;
  let arrays = [section.words];
  if (section.id === "CORE") arrays = [4, 4, 1];
  if (section.id === "MATL") {
    arrays = [
      (h.flags & ATHC_FLAGS.normals) !== 0 ? 1 : 0,
      (h.flags & ATHC_FLAGS.emission) !== 0 ? 1 : 0,
      x.pbrWords,
      x.lobesWords,
    ].filter((w) => w > 0);
  }
  return arrays.reduce((t, w) => t + w, 0) === section.words
    ? arrays
    : [section.words];
}

/**
 * A section's bytes as stored (after the gunzip) back to its words
 * (athc_v3.rs decode_section): encoding 1 byte planes, 2 byte planes of the
 * 16-bit lanes' differences.
 */
export function decodeAthcSection(
  bytes: Uint8Array,
  n: number,
  arrays: readonly number[],
  encoding: number,
): Uint8Array {
  if (encoding === 0) return bytes;
  if (encoding !== 1 && encoding !== 2) {
    throw new Error(`.athc v3: section encoding ${encoding}`);
  }
  const out = new Uint8Array(bytes.length);
  let at = 0;
  for (const w of arrays) {
    const row = 4 * w;
    if (at + n * row > bytes.length) {
      throw new Error(".athc v3: an encoded section shorter than its arrays");
    }
    if (encoding === 1) {
      for (let b = 0; b < row; b++) {
        const src = at + b * n;
        for (let e = 0; e < n; e++) out[at + e * row + b] = bytes[src + e];
      }
    } else {
      for (let b = 0; b < row; b += 2) {
        const lo = at + b * n;
        const hi = lo + n;
        let prev = 0;
        for (let e = 0; e < n; e++) {
          prev = (prev + (bytes[lo + e] | (bytes[hi + e] << 8))) & 0xffff;
          out[at + e * row + b] = prev & 0xff;
          out[at + e * row + b + 1] = prev >> 8;
        }
      }
    }
    at += n * row;
  }
  return out;
}

/** The streams of `specs` in sections `parts` of a block of `n`, as attribute columns. */
async function streamColumns(
  layout: AthcV3Layout,
  n: number,
  parts: { section: AthcSection; stored: Uint8Array; raw: number }[],
  specs: readonly AttributeSpec[],
): Promise<AttribPool> {
  const raw = new Map<string, Uint32Array>();
  for (const { section, stored } of parts) {
    const bytes = decodeAthcSection(
      section.compression === 1 ? await gunzip(stored) : stored.slice(),
      n,
      sectionArrays(layout, section),
      section.encoding ?? 0,
    );
    raw.set(
      section.id,
      new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4),
    );
  }
  const { header: h, extra: x } = layout;
  // The planar arrays each stream is made of.
  const sources = new Map<string, { data: Uint32Array; words: number }[]>();
  const matl = raw.get("MATL");
  if (matl) {
    let at = 0;
    const take = (on: boolean, name: string, words: number) => {
      if (!on) return;
      sources.set(name, [{ data: matl.subarray(at, at + n * words), words }]);
      at += n * words;
    };
    take((h.flags & ATHC_FLAGS.normals) !== 0, "normalOct", 1);
    take((h.flags & ATHC_FLAGS.emission) !== 0, "emission", 1);
    take(x.pbrWords > 0, "pbr", x.pbrWords);
    take(x.lobesWords > 0, "lobes", x.lobesWords);
  }
  const shad = raw.get("SHAD");
  if (shad) sources.set("shadowBits", [{ data: shad, words: x.shadowWords }]);
  const curv = raw.get("CURV");
  if (curv) sources.set("curvature", [{ data: curv, words: 2 }]);
  // The transfer's sections in file order: direct, then the rest.
  const tx = layout.sections
    .filter((s) => /^TX/.test(s.id) && raw.has(s.id))
    .map((s) => ({ data: raw.get(s.id) as Uint32Array, words: s.words }));
  if (tx.length) sources.set("transfer", tx);
  const pool = new AttribPool(n);
  for (const spec of specs) {
    const src = sources.get(spec.name);
    if (!src) continue;
    const have = src.reduce((t, p) => t + p.words, 0);
    if (have < attribWords(spec)) continue;
    pool.setColumn({ spec: { ...spec }, words: assembleRows(n, src, spec) });
  }
  return pool;
}

/** An attribute spec of a .athc stream, as the layout lists it. */
export function athcStreamSpec(
  layout: AnyAthcLayout,
  name: string,
): AttributeSpec | undefined {
  const s = layout.attribSpecs.find((a) => a.name === name);
  return (
    s && {
      name: s.name,
      format: s.format as AttribFormat,
      components: s.components,
    }
  );
}

// --- the streams beside the splats (athenea packing.slang) -----------------

/** `unpackNormal`: octahedral 2 x unorm16 to a unit vector. */
export function unpackAthcNormal(word: number): [number, number, number] {
  const fx = ((word & 0xffff) / 65535) * 2 - 1;
  const fy = ((word >>> 16) / 65535) * 2 - 1;
  let x = fx;
  let y = fy;
  const z = 1 - Math.abs(fx) - Math.abs(fy);
  const t = Math.min(Math.max(-z, 0), 1);
  x += x >= 0 ? -t : t;
  y += y >= 0 ? -t : t;
  const l = Math.hypot(x, y, z);
  return [x / l, y / l, z / l];
}

/** `unpackRgb9e5`: linear RGB. */
export function unpackRgb9e5(word: number): [number, number, number] {
  const step = 2 ** ((word >>> 27) - 24);
  return [
    (word & 511) * step,
    ((word >>> 9) & 511) * step,
    ((word >>> 18) & 511) * step,
  ];
}

/** splat_encoding.slang's pbr word. */
export function unpackAthcPbr(word: number) {
  return {
    metallic: (word & 0xff) / 255,
    roughness: ((word >>> 8) & 0xff) / 255,
    transmission: ((word >>> 16) & 0xff) / 255,
    thinWalled: ((word >>> 24) & 1) !== 0,
    schlickMetal: ((word >>> 25) & 1) !== 0,
  };
}

/** packing.slang `unpackLobes`. */
export function unpackAthcLobes(w0: number, w1: number, w2: number) {
  const byte = (w: number, at: number) => ((w >>> (at * 8)) & 0xff) / 255;
  return {
    specularColour: [byte(w0, 0), byte(w0, 1), byte(w0, 2)],
    specularWeight: byte(w0, 3),
    coatWeight: byte(w1, 0),
    coatRoughness: byte(w1, 1),
    coatIor: 1 + ((w1 >>> 16) & 0x7f) / 64,
    coatDarkening: ((w1 >>> 23) & 1) !== 0 ? 1 : 0,
    specularIor: 1 + ((w1 >>> 24) & 0xff) / 128,
    sheenColour: [byte(w2, 0), byte(w2, 1), byte(w2, 2)],
    sheenRoughness: byte(w2, 3),
  };
}
