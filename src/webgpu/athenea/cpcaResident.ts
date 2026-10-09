// The TX transfer kept on the GPU as clustered PCA (.athc v3 encoding 3,
// rust/spark-lib/src/athc_cpca.rs) instead of its halves: the attribute
// `transferCpca` and a table of the clusters beside the pool, rebuilt a
// batch at a time by the relight pass (slang/athenea_adapter/cpca.slang has
// both layouts; atheneaRelightPlugin({ transferResident: "cpca" })).
//
// A splat's record keeps the transfer's sections (TXDI, TXIN, TXFD: the
// prefix its transfer form keeps) one after the other, in halves:
//   - a section stored smaller as CPCA (1 + M halves against its D): the
//     entry of its cluster (u16) and its M coefficients (i16, the file's
//     quantized integers; M the most any block of the cloud uses);
//   - otherwise its D halves as they are.
// The table holds one entry a cluster (its step, mean and m directions as
// the file's halves) and, for a block the file keeps exact or that this
// layout cannot hold (more directions than M, coefficients past 16 bits, a
// merged root the decoder made), an exact entry: the block's halves, its
// elements pointing at their row in it.
//
// The rebuilt halves are the WASM decoder's bit for bit as far as f32 goes
// (the same products and sums in the same order; a GPU may fuse a product
// into its sum).

import { decodeAthcCpca } from "../../athc";
import type { AttributeColumn, AttributeSpec } from "../attributes/schema";

export const TRANSFER_CPCA = "transferCpca";
/** An exact entry (cpca.slang kCpcaExact). */
export const CPCA_EXACT = 0x80000000;
/** Most coefficients a CPCA section keeps (cpca.slang kCpcaMaxCoefficients). */
export const CPCA_MAX_COEFFICIENTS = 64;
/** Entries a table holds: the record keeps them in 16 bits. */
export const CPCA_MAX_ENTRIES = 65536;

export type CpcaSectionId = "TXDI" | "TXIN" | "TXFD";

export interface CpcaSection {
  id: CpcaSectionId;
  /** Words an element in the file (2 halves each: D = 2 x words). */
  words: number;
  /** Its first word in the transfer (the decoded attribute's). */
  firstWord: number;
  /** Kept as clusters (else its halves). */
  cpca: boolean;
  /** Coefficients a record holds (CPCA). */
  coeffs: number;
  /** Its first half in the record. */
  half: number;
}

/** A CPCA payload's header (athc_cpca.rs; mode 0 is exact byte planes). */
export function cpcaHeader(payload: Uint8Array) {
  const v = new DataView(
    payload.buffer,
    payload.byteOffset,
    payload.byteLength,
  );
  const mode = v.getUint32(0, true);
  if (mode !== 1) return { mode, k: 0, m: 0, d: 0, b: 0 };
  return {
    mode,
    k: v.getUint32(4, true),
    m: v.getUint32(8, true),
    d: v.getUint32(12, true),
    b: v.getUint32(16, true),
  };
}

/**
 * The sections a record keeps for a transfer of `words` words (the form
 * kept: whole sections, TXDI first) out of the file's, and which of them
 * are smaller as clusters, given the most directions any block of each uses
 * (`coeffs`, absent: never CPCA).
 */
export function planCpcaSections(
  fileSections: readonly { id: string; words: number }[],
  words: number,
  coeffs: Partial<Record<CpcaSectionId, number>>,
): CpcaSection[] {
  const out: CpcaSection[] = [];
  let firstWord = 0;
  let half = 0;
  for (const s of fileSections) {
    if (!/^TX(DI|IN|FD)$/.test(s.id)) continue;
    if (firstWord + s.words > words) break;
    const id = s.id as CpcaSectionId;
    const m = coeffs[id];
    const cpca =
      m !== undefined &&
      m > 0 &&
      m <= CPCA_MAX_COEFFICIENTS &&
      1 + m < 2 * s.words;
    out.push({
      id,
      words: s.words,
      firstWord,
      cpca,
      coeffs: cpca ? (m as number) : 0,
      half,
    });
    firstWord += s.words;
    half += cpca ? 1 + (m as number) : 2 * s.words;
  }
  if (firstWord !== words) {
    throw new Error(
      `transferCpca: the file's sections do not make a transfer of ${words} words`,
    );
  }
  return out;
}

/**
 * The cluster table of a cloud's transfer and its records' layout. Entries
 * are added as blocks arrive (a paged cloud's pages); `pack()` is what the
 * GPU reads.
 */
export class CpcaTable {
  /** Words a record (the attribute's components). */
  readonly recordWords: number;
  /** Words of the transfer it rebuilds (the decoded attribute's). */
  readonly transferWords: number;
  private offsets: number[] = [];
  private data = new Uint32Array(1 << 16);
  private used = 0;
  /** Bumped when entries are added. */
  version = 0;

  constructor(
    readonly sections: readonly CpcaSection[],
    /** Values of the transfer (the attribute's components as decoded). */
    readonly transferCount: number,
  ) {
    const halves = sections.reduce(
      (n, s) => n + (s.cpca ? 1 + s.coeffs : 2 * s.words),
      0,
    );
    this.recordWords = Math.ceil(halves / 2);
    this.transferWords = sections.reduce((n, s) => n + s.words, 0);
  }

  get entries() {
    return this.offsets.length;
  }

  /** The attribute the records are. */
  spec(): AttributeSpec {
    return { name: TRANSFER_CPCA, format: "u32", components: this.recordWords };
  }

  /** Bytes of pack(). */
  get bytes() {
    return 4 * (1 + this.offsets.length + this.used);
  }

  private reserve(words: number) {
    if (this.used + words <= this.data.length) return;
    let size = this.data.length;
    while (size < this.used + words) size *= 2;
    const next = new Uint32Array(size);
    next.set(this.data.subarray(0, this.used));
    this.data = next;
  }

  /** An entry of head `head` and `halves` (raw f16 bytes); its number. */
  addEntry(
    head: readonly [number, number, number],
    halves: Uint8Array,
  ): number {
    if (this.offsets.length >= CPCA_MAX_ENTRIES) {
      throw new Error("transferCpca: more than 65 536 table entries");
    }
    const words = 4 + Math.ceil(halves.length / 4);
    this.reserve(words);
    const at = this.used;
    this.data[at] = head[0] >>> 0;
    this.data[at + 1] = head[1] >>> 0;
    this.data[at + 2] = head[2] >>> 0;
    this.data[at + 3] = 0;
    new Uint8Array(this.data.buffer, (at + 4) * 4, halves.length).set(halves);
    this.used += words;
    this.offsets.push(at);
    this.version += 1;
    return this.offsets.length - 1;
  }

  /** [entries, where each starts (from the buffer's start), the entries]. */
  pack(): Uint32Array {
    const e = this.offsets.length;
    const out = new Uint32Array(1 + e + this.used);
    out[0] = e;
    for (let k = 0; k < e; k++) out[1 + k] = 1 + e + this.offsets[k];
    out.set(this.data.subarray(0, this.used), 1 + e);
    return out;
  }

  /**
   * Block rows `first` .. `first + n - 1` of `records` (recordWords a row)
   * from the block's transfer sections as stored, after the gunzip
   * (encoding 3 payloads, or the halves of an unencoded section).
   */
  addBlock(
    records: Uint32Array,
    first: number,
    n: number,
    payloads: ReadonlyMap<string, { bytes: Uint8Array; encoding: number }>,
  ) {
    const h16 = new Uint16Array(records.buffer, records.byteOffset);
    const row = (e: number) => 2 * (first + e) * this.recordWords;
    for (const s of this.sections) {
      const p = payloads.get(s.id);
      if (!p) throw new Error(`transferCpca: a block without ${s.id}`);
      const d = 2 * s.words;
      const halvesOf = () => {
        const raw =
          p.encoding === 3 ? decodeAthcCpca(p.bytes, n, s.words) : p.bytes;
        return new Uint16Array(raw.buffer, raw.byteOffset, n * d);
      };
      if (!s.cpca) {
        const values = halvesOf();
        for (let e = 0; e < n; e++) {
          h16.set(values.subarray(e * d, (e + 1) * d), row(e) + s.half);
        }
        continue;
      }
      if (p.encoding === 3 && this.addClusters(h16, row, n, s, p.bytes))
        continue;
      // Exact: the block's halves in the table, each element its row.
      const values = halvesOf();
      const entry = this.addEntry(
        [CPCA_EXACT, n, d],
        new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
      );
      for (let e = 0; e < n; e++) {
        h16[row(e) + s.half] = entry;
        h16[row(e) + s.half + 1] = e;
        for (let j = 1; j < s.coeffs; j++) h16[row(e) + s.half + 1 + j] = 0;
      }
    }
  }

  /** The clusters of a mode 1 payload, where the record holds them. */
  private addClusters(
    h16: Uint16Array,
    row: (e: number) => number,
    n: number,
    s: CpcaSection,
    bytes: Uint8Array,
  ): boolean {
    const { mode, k, m, d, b } = cpcaHeader(bytes);
    if (mode !== 1 || d !== 2 * s.words || m > s.coeffs) return false;
    if (this.offsets.length + k > CPCA_MAX_ENTRIES) return false;
    const basisAt = 20 + 4 * k;
    const per = (1 + m) * d;
    const idsAt = Math.ceil((basisAt + 2 * k * per) / 4) * 4;
    const planesAt = idsAt + n;
    // The coefficients as i16: a 3-byte one must fit.
    const q = new Int32Array(n * m);
    for (let j = 0; j < m; j++) {
      for (let e = 0; e < n; e++) {
        let z = 0;
        for (let i = 0; i < b; i++) {
          z += bytes[planesAt + (j * b + i) * n + e] * 2 ** (8 * i);
        }
        const v = z & 1 ? -(z + 1) / 2 : z / 2;
        if (v < -32768 || v > 32767) return false;
        q[e * m + j] = v;
      }
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const base = this.offsets.length;
    for (let c = 0; c < k; c++) {
      this.addEntry(
        [m, view.getUint32(20 + 4 * c, true), d],
        bytes.subarray(basisAt + 2 * c * per, basisAt + 2 * (c + 1) * per),
      );
    }
    for (let e = 0; e < n; e++) {
      const at = row(e) + s.half;
      h16[at] = base + bytes[idsAt + e];
      for (let j = 0; j < s.coeffs; j++) {
        h16[at + 1 + j] = j < m ? q[e * m + j] & 0xffff : 0;
      }
    }
    return true;
  }

  /**
   * Row `r` of `records` from the transfer's decoded words (`transfer`,
   * transferWords of them): a merged root the decoder made, say.
   */
  addRow(records: Uint32Array, r: number, transfer: Uint32Array) {
    const h16 = new Uint16Array(records.buffer, records.byteOffset);
    const t16 = new Uint16Array(
      transfer.buffer,
      transfer.byteOffset,
      2 * this.transferWords,
    );
    const at = 2 * r * this.recordWords;
    for (const s of this.sections) {
      const values = t16.subarray(2 * s.firstWord, 2 * (s.firstWord + s.words));
      if (!s.cpca) {
        h16.set(values, at + s.half);
        continue;
      }
      const entry = this.addEntry(
        [CPCA_EXACT, 1, 2 * s.words],
        new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
      );
      h16[at + s.half] = entry;
      h16[at + s.half + 1] = 0;
      for (let j = 1; j < s.coeffs; j++) h16[at + s.half + 1 + j] = 0;
    }
  }

  /** cpca.slang's rebuild of row `r`, on the CPU (tests): the transfer's words. */
  rebuild(records: Uint32Array, r: number): Uint32Array {
    const table = this.pack();
    const t16 = new Uint16Array(table.buffer);
    const h16 = new Uint16Array(records.buffer, records.byteOffset);
    const out = new Uint32Array(this.transferWords);
    const o16 = new Uint16Array(out.buffer);
    const at = 2 * r * this.recordWords;
    const f = Math.fround;
    for (const s of this.sections) {
      const d = 2 * s.words;
      const dst = 2 * s.firstWord;
      if (!s.cpca) {
        o16.set(h16.subarray(at + s.half, at + s.half + d), dst);
        continue;
      }
      const start = table[1 + h16[at + s.half]];
      const head = table[start];
      const data = 2 * (start + 4);
      if (head & CPCA_EXACT) {
        const index = h16[at + s.half + 1];
        o16.set(t16.subarray(data + index * d, data + (index + 1) * d), dst);
        continue;
      }
      const m = head;
      const step = new Float32Array(table.buffer, 4 * (start + 1), 1)[0];
      const coef = new Float32Array(m);
      for (let j = 0; j < m; j++) {
        coef[j] = f(((h16[at + s.half + 1 + j] << 16) >> 16) * step);
      }
      for (let v = 0; v < d; v++) {
        let acc = halfToFloat(t16[data + v]);
        for (let j = 0; j < m; j++) {
          acc = f(acc + f(coef[j] * halfToFloat(t16[data + (1 + j) * d + v])));
        }
        o16[dst + v] = floatToHalf(acc);
      }
    }
    return out;
  }
}

/** The attribute column of `n` rows for a table. */
export function cpcaColumn(
  table: CpcaTable,
  words: Uint32Array,
): AttributeColumn {
  return { spec: table.spec(), words };
}

function halfToFloat(h: number) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >>> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? Number.NaN : s * Number.POSITIVE_INFINITY;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

const F16 = (
  globalThis as unknown as { Float16Array?: Float32ArrayConstructor }
).Float16Array;

/** A float to the nearest half (ties even). */
function floatToHalf(v: number): number {
  if (F16) {
    const a = new F16(1);
    a[0] = v;
    return new Uint16Array(a.buffer)[0];
  }
  const f32 = new Float32Array([v]);
  const x = new Uint32Array(f32.buffer)[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 31) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    let h = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const half = 1 << (shift - 1);
    if (rem > half || (rem === half && h & 1)) h += 1;
    return sign | h;
  }
  let h = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && h & 1)) h += 1;
  return sign | h;
}

/**
 * A whole .athc v3 file's transfer as records and a table: `rows` decoded
 * elements (the WASM decoder's order: a merged root it made, the levels,
 * then the chunks' splats), the transfer form of `values` values, and the
 * decoded transfer (`transfer`, its words a row) for a root the file does
 * not hold.
 */
export async function athcCpcaResident(
  bytes: Uint8Array,
  layout: {
    sections: readonly {
      id: string;
      words: number;
      encoding: number;
      compression: number;
    }[];
    blocks: readonly {
      kind: number;
      n: number;
      spans: readonly { offset: number; stored: number }[];
    }[];
  },
  {
    rows,
    values,
    transfer,
    gunzip,
  }: {
    rows: number;
    values: number;
    transfer?: Uint32Array;
    gunzip: (b: Uint8Array) => Promise<Uint8Array>;
  },
): Promise<{ table: CpcaTable; column: AttributeColumn }> {
  const tx = layout.sections
    .map((s, k) => ({ s, k }))
    .filter(({ s }) => /^TX(DI|IN|FD)$/.test(s.id));
  const blocks = [
    ...layout.blocks.filter((b) => b.kind === 0),
    ...layout.blocks.filter((b) => b.kind === 1),
  ];
  // Every block's transfer sections, unpacked once.
  const payloads: Map<string, { bytes: Uint8Array; encoding: number }>[] = [];
  const coeffs: Partial<Record<CpcaSectionId, number>> = {};
  for (const b of blocks) {
    const map = new Map<string, { bytes: Uint8Array; encoding: number }>();
    for (const { s, k } of tx) {
      const span = b.spans[k];
      const stored = bytes.subarray(span.offset, span.offset + span.stored);
      const raw = s.compression === 1 ? await gunzip(stored) : stored;
      map.set(s.id, { bytes: raw, encoding: s.encoding });
      if (s.encoding === 3) {
        const h = cpcaHeader(raw);
        const id = s.id as CpcaSectionId;
        if (h.mode === 1 && h.b <= 2)
          coeffs[id] = Math.max(coeffs[id] ?? 0, h.m);
      }
    }
    payloads.push(map);
  }
  const sections = planCpcaSections(
    tx.map(({ s }) => s),
    Math.ceil(values / 2),
    coeffs,
  );
  const table = new CpcaTable(sections, values);
  const held = blocks.reduce((n, b) => n + b.n, 0);
  const extra = rows - held;
  if (extra < 0 || extra > 1) {
    throw new Error(`transferCpca: ${rows} rows for ${held} elements`);
  }
  const records = new Uint32Array(rows * table.recordWords);
  if (extra) {
    if (!transfer)
      throw new Error("transferCpca: a merged root without its transfer");
    const stride = Math.ceil(transfer.length / rows);
    table.addRow(records, 0, transfer.subarray(0, stride));
  }
  let at = extra;
  blocks.forEach((b, k) => {
    table.addBlock(records, at, b.n, payloads[k]);
    at += b.n;
  });
  return { table, column: cpcaColumn(table, records) };
}
