// The `.athl` half of athenea's light sidecar on the WebGPU path: its data
// as decoded by the WASM reader (rust/spark-lib/src/athl.rs; layout in
// docs/docs/athl.md), and the one storage buffer the lights plugin reads
// (slang/athenea_adapter/lights.slang), packed from it.
//
// Paging: a .athl's layers are by chunk of 65 536 splats of the cloud's
// virtual order, the pager's pages, each (chunk, every group) one byte range
// (athlChunkRange). An AthlStore holds the chunks fetched so far; dropping
// a chunk frees it; the packed buffer holds what is held. A whole file is
// every chunk at once; openAthl reads the groups, polygons and profiles
// first and an AthlPager then fetches the chunks the cloud's pages want.

import {
  athl_header,
  athl_meta_bytes,
  athl_prefix_bytes,
  decode_athl,
  decode_athl_layer,
  decode_athl_meta,
} from "spark-rs";
import * as wasm from "../../wasm";

export const ATHL_KIND = { indirect: 0, emission: 1, field: 2 } as const;
export const ATHL_CHUNK_SPLATS = 65536;
export const ATHL_BLOCK_SPLATS = 256;
export const ATHL_FIELD_COMPONENTS = 27;
export const ATHL_GROUP_TWO_SIDED = 1;
export const ATHL_MAX_GROUPS = 16;
export const ATHL_NONE = 0xffffffff;

type V3 = [number, number, number];

export interface AthlSection {
  tag: number;
  group: number;
  kind: number;
  chunk: number;
  count: number;
  offset: number;
  bytes: number;
}

export interface AthlHeader {
  version: number;
  flags: number;
  groupCount: number;
  elementCount: number;
  merged: number;
  splatBase: number;
  splatCount: number;
  chunkSplats: number;
  blockSplats: number;
  /** 16 hex digits. */
  cloudHash: string;
  bakeHash: string;
  sections: AthlSection[];
}

export interface AthlGroup {
  name: string;
  flags: number;
  polygonFirst: number;
  polygonCount: number;
  profile: number;
  tint: V3;
  origin: V3;
  axes: [V3, V3, V3];
  radiance: number;
}

export interface AthlPolygon {
  group: number;
  radiance: V3;
  vertices: V3[];
}

export interface AthlProfile {
  width: number;
  height: number;
  lon: [number, number];
  lat: [number, number];
  outside: V3;
  /** f16 bits, rgb a texel, row 0 at lat[0]. */
  texels: Uint16Array;
}

export interface AthlLayer {
  group: number;
  kind: number;
  chunk: number;
  components: number;
  /** Block ids in the chunk, increasing. */
  blocks: Uint16Array;
  /** f16 bits, blocks x 256 x components. */
  data: Uint16Array;
}

export interface AthlData {
  header: Omit<AthlHeader, "sections"> & { sections?: AthlSection[] };
  groups: AthlGroup[];
  polygons: AthlPolygon[];
  profiles: AthlProfile[];
  layers: AthlLayer[];
}

const TAG_LAYER = 0x5259414c; // "LAYR"

/** Decodes a whole .athl (WASM). */
export async function decodeAthl(bytes: Uint8Array): Promise<AthlData> {
  await wasm.initialization;
  return decode_athl(bytes) as AthlData;
}

/** A .athl's header and section table from its first bytes (WASM). */
export async function readAthlHeader(
  fetchRange: (start: number, end: number) => Promise<Uint8Array>,
): Promise<AthlHeader> {
  await wasm.initialization;
  let prefix = await fetchRange(0, 4096);
  const need = athl_prefix_bytes(prefix.slice(0, 128));
  if (need > prefix.length) prefix = await fetchRange(0, need);
  return athl_header(prefix) as AthlHeader;
}

/** One LAYR section's bytes (WASM). */
export async function decodeAthlLayer(section: Uint8Array): Promise<AthlLayer> {
  await wasm.initialization;
  return decode_athl_layer(section) as AthlLayer;
}

/**
 * The byte range holding chunk `chunk`'s layers, every group and kind
 * (they are stored chunk after chunk): [start, end), or null when the
 * chunk has none.
 */
export function athlChunkRange(
  header: AthlHeader,
  chunk: number,
): [number, number] | null {
  const s = header.sections.filter(
    (x) => x.tag === TAG_LAYER && x.chunk === chunk,
  );
  if (!s.length) return null;
  return [
    Math.min(...s.map((x) => x.offset)),
    Math.max(...s.map((x) => x.offset + x.bytes)),
  ];
}

/** The LAYR sections inside a fetched chunk range, as layers. */
export async function decodeAthlChunk(
  header: AthlHeader,
  chunk: number,
  bytes: Uint8Array,
  start: number,
): Promise<AthlLayer[]> {
  const out: AthlLayer[] = [];
  for (const s of header.sections) {
    if (s.tag !== TAG_LAYER || s.chunk !== chunk) continue;
    out.push(
      await decodeAthlLayer(
        bytes.subarray(s.offset - start, s.offset - start + s.bytes),
      ),
    );
  }
  return out;
}

/**
 * FNV-1a, 64 bits, of a cloud's first 4096 bytes: a .athl's cloudHash
 * (rust athl.rs cloud_hash), as 16 hex digits.
 */
export function athcCloudHash(bytes: Uint8Array): string {
  let h = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  const n = Math.min(bytes.length, 4096);
  for (let i = 0; i < n; i++) {
    h ^= BigInt(bytes[i]);
    h = (h * prime) & mask;
  }
  return h.toString(16).padStart(16, "0");
}

// ---------------------------------------------------------------------------
// The GPU layout (lights.slang reads it; one uint buffer):
//
//   [0]  groupCount   [1] chunkCount   [2] polygonOffset   [3] polygonCount
//   [4 .. 16)         reserved
//   [16 + 32 k ..]    group k (ATHL_GROUP_WORDS):
//       0 flags  1 polygonFirst  2 polygonCount  3 profile offset (or NONE)
//       4..6 tint  7 radiance  8..10 origin  12..14 axis x  16..18 axis y
//       20..22 axis z  24 + kind: that kind's chunk table offset (or NONE)
//   chunk table       chunkCount words: the chunk's block directory (or NONE)
//   block directory   256 words: the block's data offset (or NONE)
//   block data        256 x components halves, two to a word
//   polygon           32 words: vertexCount, group, radiance rgb, pad x 3,
//                     then 8 vertices x 3 floats
//   profile           width, height, lon0, lon1, lat0, lat1, outside rgb,
//                     pad x 3, then two words a texel (rg, b0 halves)

export const ATHL_HEADER_WORDS = 16;
export const ATHL_GROUP_WORDS = 32;
export const ATHL_POLYGON_WORDS = 32;
export const ATHL_PROFILE_HEAD_WORDS = 12;

/** The chunks of an .athl held on the CPU, packed for the GPU on demand. */
export class AthlStore {
  readonly groups: AthlGroup[];
  readonly polygons: AthlPolygon[];
  readonly profiles: AthlProfile[];
  readonly header: AthlData["header"];
  /** Bumped by every change of what the packed buffer holds. */
  version = 0;
  private chunks = new Map<number, AthlLayer[]>();

  constructor(data: AthlData) {
    if (data.groups.length > ATHL_MAX_GROUPS) {
      throw new Error(
        `.athl: ${data.groups.length} groups, at most ${ATHL_MAX_GROUPS}`,
      );
    }
    this.header = data.header;
    this.groups = data.groups;
    this.polygons = data.polygons;
    this.profiles = data.profiles;
    for (const l of data.layers) this.addLayers([l]);
  }

  get chunkCount(): number {
    return Math.ceil(this.header.elementCount / ATHL_CHUNK_SPLATS);
  }

  /** The chunks whose layers are held. */
  get heldChunks(): number[] {
    return [...this.chunks.keys()].sort((a, b) => a - b);
  }

  /** Adds a chunk's layers (fetched by range). */
  addLayers(layers: readonly AthlLayer[]) {
    for (const l of layers) {
      const list = this.chunks.get(l.chunk) ?? [];
      const at = list.findIndex(
        (x) => x.group === l.group && x.kind === l.kind,
      );
      if (at >= 0) list[at] = l;
      else list.push(l);
      this.chunks.set(l.chunk, list);
    }
    this.version += 1;
  }

  /** Frees a chunk's layers (its page left the pool). */
  dropChunk(chunk: number) {
    if (this.chunks.delete(chunk)) this.version += 1;
  }

  /** Every layer held. */
  layers(): AthlLayer[] {
    return this.heldChunks.flatMap((c) => this.chunks.get(c) ?? []);
  }

  /** The buffer lights.slang reads. */
  pack(): Uint32Array {
    const G = this.groups.length;
    const chunks = this.chunkCount;
    const layers = this.layers();
    let words = ATHL_HEADER_WORDS + ATHL_GROUP_WORDS * G;
    // Chunk tables, one per (group, kind) with any layer.
    const kinds = new Map<string, AthlLayer[]>();
    for (const l of layers) {
      const key = `${l.group}:${l.kind}`;
      kinds.set(key, [...(kinds.get(key) ?? []), l]);
    }
    const tableAt = new Map<string, number>();
    for (const key of kinds.keys()) {
      tableAt.set(key, words);
      words += chunks;
    }
    const directoryAt = new Map<AthlLayer, number>();
    for (const l of layers) {
      directoryAt.set(l, words);
      words += ATHL_CHUNK_SPLATS / ATHL_BLOCK_SPLATS;
    }
    const dataAt = new Map<AthlLayer, number>();
    for (const l of layers) {
      dataAt.set(l, words);
      words += (l.blocks.length * ATHL_BLOCK_SPLATS * l.components) / 2;
    }
    const polygonAt = words;
    words += ATHL_POLYGON_WORDS * this.polygons.length;
    const profileAt: number[] = [];
    for (const p of this.profiles) {
      profileAt.push(words);
      words += ATHL_PROFILE_HEAD_WORDS + 2 * p.width * p.height;
    }
    const out = new Uint32Array(Math.max(words, 4));
    const f = new Float32Array(out.buffer);
    out[0] = G;
    out[1] = chunks;
    out[2] = polygonAt;
    out[3] = this.polygons.length;
    this.groups.forEach((g, k) => {
      const at = ATHL_HEADER_WORDS + ATHL_GROUP_WORDS * k;
      out[at] = g.flags;
      out[at + 1] = g.polygonFirst;
      out[at + 2] = g.polygonCount;
      out[at + 3] = g.profile >= 0 ? profileAt[g.profile] : ATHL_NONE;
      f.set(g.tint, at + 4);
      f[at + 7] = g.radiance;
      f.set(g.origin, at + 8);
      f.set(g.axes[0], at + 12);
      f.set(g.axes[1], at + 16);
      f.set(g.axes[2], at + 20);
      for (let kind = 0; kind < 3; kind++) {
        out[at + 24 + kind] = tableAt.get(`${k}:${kind}`) ?? ATHL_NONE;
      }
    });
    for (const [key, list] of kinds) {
      const t = tableAt.get(key) as number;
      out.fill(ATHL_NONE, t, t + chunks);
      for (const l of list) out[t + l.chunk] = directoryAt.get(l) as number;
    }
    for (const l of layers) {
      const d = directoryAt.get(l) as number;
      out.fill(ATHL_NONE, d, d + ATHL_CHUNK_SPLATS / ATHL_BLOCK_SPLATS);
      const base = dataAt.get(l) as number;
      const per = (ATHL_BLOCK_SPLATS * l.components) / 2;
      l.blocks.forEach((b, i) => {
        out[d + b] = base + i * per;
      });
      const halves = new Uint16Array(out.buffer, base * 4, l.data.length);
      halves.set(l.data);
    }
    this.polygons.forEach((p, i) => {
      const at = polygonAt + ATHL_POLYGON_WORDS * i;
      out[at] = p.vertices.length;
      out[at + 1] = p.group;
      f.set(p.radiance, at + 2);
      p.vertices.forEach((v, j) => f.set(v, at + 8 + 3 * j));
    });
    this.profiles.forEach((p, i) => {
      const at = profileAt[i];
      out[at] = p.width;
      out[at + 1] = p.height;
      f.set([...p.lon, ...p.lat, ...p.outside], at + 2);
      const t = new Uint16Array(out.buffer, (at + ATHL_PROFILE_HEAD_WORDS) * 4);
      for (let k = 0; k < p.width * p.height; k++) {
        t[4 * k] = p.texels[3 * k];
        t[4 * k + 1] = p.texels[3 * k + 1];
        t[4 * k + 2] = p.texels[3 * k + 2];
      }
    });
    return out;
  }
}

export type RangeFetch = (start: number, end: number) => Promise<Uint8Array>;

/** A range fetch of `url` (HTTP Range; [start, end)). */
export function urlRange(url: string): RangeFetch {
  return async (start, end) => {
    const r = await fetch(url, {
      headers: { Range: `bytes=${start}-${end - 1}` },
    });
    if (!r.ok) throw new Error(`.athl: ${url} ${r.status}`);
    const b = new Uint8Array(await r.arrayBuffer());
    // A server without ranges sends the whole file.
    return r.status === 206 ? b : b.subarray(start, end);
  };
}

/**
 * Keeps an AthlStore holding the chunks a cloud wants (its resident
 * pages'), fetching each missing one by its byte range and dropping those
 * no longer wanted.
 */
export class AthlPager {
  /** Fetches in flight. */
  pending = 0;
  /** At most this many fetches at once. */
  concurrency = 2;
  private wanted = new Set<number>();
  private inFlight = new Set<number>();
  private failed = new Set<number>();
  onChange: (() => void) | null = null;

  constructor(
    readonly store: AthlStore,
    readonly header: AthlHeader,
    readonly fetchRange: RangeFetch,
  ) {}

  /** The chunks to hold from now on. */
  update(chunks: Iterable<number>) {
    this.wanted = new Set(chunks);
    for (const c of this.store.heldChunks) {
      if (!this.wanted.has(c)) {
        this.store.dropChunk(c);
        this.onChange?.();
      }
    }
    const held = new Set(this.store.heldChunks);
    for (const c of this.wanted) {
      if (this.pending >= this.concurrency) break;
      if (held.has(c) || this.inFlight.has(c) || this.failed.has(c)) continue;
      const range = athlChunkRange(this.header, c);
      if (!range) continue; // no group reaches this page
      this.inFlight.add(c);
      this.pending += 1;
      this.fetchRange(range[0], range[1])
        .then((bytes) => decodeAthlChunk(this.header, c, bytes, range[0]))
        .then((layers) => {
          // Still wanted: a page can leave while its layers travel.
          if (this.wanted.has(c)) {
            this.store.addLayers(layers);
            this.onChange?.();
          }
        })
        .catch((e) => {
          this.failed.add(c);
          console.warn(`athl: chunk ${c}: ${e}`);
        })
        .finally(() => {
          this.inFlight.delete(c);
          this.pending -= 1;
          this.update(this.wanted);
        });
    }
  }

  /** Resolves when nothing wanted is still to fetch. */
  async settled(): Promise<void> {
    while (this.pending > 0) await new Promise((r) => setTimeout(r, 5));
  }
}

/**
 * Opens a .athl for paging: its header, then its groups, polygons and
 * profiles (one range); no layers until an AthlPager asks for them.
 */
export async function openAthl(fetchRange: RangeFetch): Promise<AthlPager> {
  await wasm.initialization;
  let prefix = await fetchRange(0, 4096);
  const need = athl_prefix_bytes(prefix.slice(0, 128));
  if (need > prefix.length) prefix = await fetchRange(0, need);
  const header = athl_header(prefix) as AthlHeader;
  const meta = athl_meta_bytes(prefix);
  if (meta > prefix.length) prefix = await fetchRange(0, meta);
  const data = decode_athl_meta(prefix) as AthlData;
  return new AthlPager(new AthlStore(data), header, fetchRange);
}
