// .athc v3 section encoding 3, clustered PCA of the transfer (rust
// athc_cpca.rs): the TypeScript decoder (src/athc.ts decodeAthcCpca, for a
// page that fetches the transfer on its own before the WASM is up) gives the
// same halves as the Rust one (WASM: the loader worker, and
// decodeAthcSection once the WASM is up), bit for bit.
//
// pawn_top_tx.cpca.athc: the pawn's glass top (TX transfer, 112 values: all
// three sections) cut to 1142 splats and 3 levels, transfer as CPCA
// (athc-convert --keep-splats 1500, then --planes --cpca 4,48,1e-4
// --cpca-force); pawn_top_tx.cpca.ref.athc: the same decoded by Rust and
// written back losslessly (byte planes).

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { describe, expect, it, vi } from "vitest";
import { decodeAthcCpca, decodeAthcSection } from "../../src/athc";

// The real WASM (test/unit/setup.ts stubs it), initialized up front.
vi.mock("spark-rs", async () => {
  const { readFileSync } = await import("node:fs");
  const actual = await vi.importActual<Record<string, unknown>>("spark-rs");
  (actual.initSync as (o: { module: Buffer }) => void)({
    module: readFileSync(
      new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
    ),
  });
  return { ...actual, default: async () => ({}) };
});

const fixture = (name: string) =>
  new Uint8Array(
    readFileSync(new URL(`../fixtures/athc/${name}`, import.meta.url)),
  );

/** The transfer sections of every block of a v3 file, decoded. */
function transferSections(bytes: Uint8Array, ts = false) {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const u32 = (at: number) => v.getUint32(at, true);
  const u64 = (at: number) => Number(v.getBigUint64(at, true));
  const sectionCount = u32(60);
  const blockCount = u32(104);
  const table = u64(112);
  const index = u64(120);
  const sections = Array.from({ length: sectionCount }, (_, k) => {
    const at = table + 32 * k;
    const code = String.fromCharCode(...bytes.subarray(at, at + 4));
    return {
      code,
      encoding: u32(at + 8),
      compression: u32(at + 12),
      words: u32(at + 16),
    };
  });
  const out: { code: string; encoding: number; n: number; raw: Uint8Array }[] =
    [];
  const entry = 32 + 16 * sectionCount;
  for (let b = 0; b < blockCount; b++) {
    const at = index + entry * b;
    const n = u32(at + 12);
    sections.forEach((s, k) => {
      if (!s.code.startsWith("TX")) return;
      const span = at + 32 + 16 * k;
      const offset = u64(span);
      const stored = bytes.subarray(offset, offset + u32(span + 8));
      const plain =
        s.compression === 1 ? new Uint8Array(gunzipSync(stored)) : stored;
      const raw =
        ts && s.encoding === 3
          ? decodeAthcCpca(plain, n, s.words)
          : decodeAthcSection(plain, n, [s.words], s.encoding);
      expect(raw.length).toBe(u32(span + 12));
      out.push({ code: s.code, encoding: s.encoding, n, raw });
    });
  }
  return out;
}

describe(".athc v3 CPCA transfer sections", () => {
  it("decode in TypeScript to the halves Rust decodes them to", async () => {
    await (await import("../../src/wasm")).initialization;
    const got = transferSections(fixture("pawn_top_tx.cpca.athc"), true);
    const want = transferSections(fixture("pawn_top_tx.cpca.ref.athc"));
    expect(got.map((s) => s.code)).toEqual(want.map((s) => s.code));
    expect(new Set(got.map((s) => s.code))).toEqual(
      new Set(["TXDI", "TXIN", "TXFD"]),
    );
    expect(got.every((s) => s.encoding === 3)).toBe(true);
    expect(want.every((s) => s.encoding !== 3)).toBe(true);
    let elements = 0;
    for (let k = 0; k < got.length; k++) {
      expect(got[k].n).toBe(want[k].n);
      expect(Buffer.from(got[k].raw).equals(Buffer.from(want[k].raw))).toBe(
        true,
      );
      elements += got[k].n;
    }
    expect(elements).toBeGreaterThan(3 * 1142);
    // The same without Float16Array (older engines: toHalf per value).
    const g = globalThis as { Float16Array?: unknown };
    const saved = g.Float16Array;
    g.Float16Array = undefined;
    try {
      const slow = transferSections(fixture("pawn_top_tx.cpca.athc"), true);
      for (let k = 0; k < got.length; k++)
        expect(Buffer.from(slow[k].raw).equals(Buffer.from(want[k].raw))).toBe(
          true,
        );
    } finally {
      g.Float16Array = saved;
    }
    // decodeAthcSection, the WASM up: the same.
    const viaWasm = transferSections(fixture("pawn_top_tx.cpca.athc"));
    for (let k = 0; k < got.length; k++)
      expect(Buffer.from(viaWasm[k].raw).equals(Buffer.from(want[k].raw))).toBe(
        true,
      );
  });

  it("refuses what is not a CPCA section", () => {
    expect(() => decodeAthcCpca(new Uint8Array(3), 1, 1)).toThrow();
    const bad = new Uint8Array(24);
    new DataView(bad.buffer).setUint32(0, 7, true);
    expect(() => decodeAthcCpca(bad, 1, 1)).toThrow(/mode 7/);
    // Mode 0: the values kept exactly, as byte planes.
    const exact = new Uint8Array([0, 0, 0, 0, 1, 2, 3, 4]);
    expect(Array.from(decodeAthcCpca(exact, 1, 1))).toEqual([1, 2, 3, 4]);
  });
});
