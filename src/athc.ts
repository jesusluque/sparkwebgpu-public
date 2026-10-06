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

import { athc_layout, athc_prefix_bytes } from "spark-rs";
import { workerPool } from "./SplatWorker";
import * as wasm from "./wasm";

export const ATHC_MAGIC = 0x43485441; // "ATHC"
export const ATHV_MAGIC = 0x56485441; // "ATHV"
export const ATHV_HEAD = 160;
export const ATHV_SPLATS = 1;
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
  header: AthcHeader;
  extra: {
    pbrWords: number;
    lobesWords: number;
    transferCount: number;
    transferWords: number;
    shadowWords: number;
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

export type AthcTree = {
  synthRoot: boolean;
  levelBase: number[];
  merged: number;
  splatBase: number;
  count: number;
  clippedGroups: number;
};

export type AthcPaging = {
  layout: AthcLayout;
  tree: AthcTree;
  /** The FileHeader and ExtraHeader, as an ATHV head carries them. */
  headers: Uint8Array;
  /** ATHV blobs of the merged pages, 0 .. mergedPages - 1. */
  mergedPages: Uint8Array[];
  /** Virtual pages: merged ones, then one per chunk from splatBase / 65 536. */
  pageCount: number;
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
): Promise<{ layout: AthcLayout; prefix: Uint8Array }> {
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
  const layout = athc_layout(prefix, total) as AthcLayout;
  return { layout, prefix };
}

/**
 * Opens a .athc for paging: its layout, its virtual tree and the ATHV blobs
 * of its merged pages (built in a loader worker from the file's levels).
 */
export async function openAthc(options: FetchOptions): Promise<AthcPaging> {
  const { layout, prefix } = await readAthcLayout(options);
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
  return {
    layout,
    tree,
    headers: prefix.slice(0, 136),
    mergedPages: pages,
    pageCount: tree.splatBase / ATHC_PAGE_SPLATS + chunkPages,
  };
}

/** The ATHV head of a page of `n` splats at virtual index `base`. */
export function athvHead(headers: Uint8Array, base: number, n: number) {
  const head = new Uint8Array(ATHV_HEAD);
  const view = new DataView(head.buffer);
  view.setUint32(0, ATHV_MAGIC, true);
  view.setUint32(4, ATHV_SPLATS, true);
  view.setUint32(8, base, true);
  view.setUint32(12, n, true);
  head.set(headers.subarray(0, 136), 16);
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
    return paging.mergedPages[page];
  }
  if (page < firstChunkPage) {
    throw new Error(`.athc page ${page} is between the levels and the splats`);
  }
  const chunk = page - firstChunkPage;
  const { offset, count } = paging.layout.chunks[chunk];
  const bytes = count * paging.layout.elementBytes;
  const { data } = await readRange(options, offset, bytes);
  const out = new Uint8Array(ATHV_HEAD + bytes);
  out.set(athvHead(paging.headers, page * ATHC_PAGE_SPLATS, count));
  out.set(data, ATHV_HEAD);
  return out;
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
