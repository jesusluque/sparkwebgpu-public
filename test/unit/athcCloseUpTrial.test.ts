// Thread CX: athenea's 5120 close-up level (car5120) converted to the compact
// format (car-data/convert8.sh: CMPC v4 runs, against an asset table) parses
// through the viewer's CPU path: the WASM layout (athc_layout), the piece's
// tables resolved against the asset's .cmpa, every page's runs container
// (traversal member gunzipped, every run inflated, unpackRunsContainer as
// CompactBandStreamer lands it) and the WASM traversal over all pages
// resident: a cut at a far and a near view, every key a node of a page.
//
// Opt-in, real data (outside git, never published):
//   CLOSEUP_TRIAL=<dir with car.cmpa and car_*.athc> [CLOSEUP_FILES=a.athc,b.athc]
//   npx vitest run --dir test/unit test/unit/athcCloseUpTrial.test.ts

import {
  closeSync,
  existsSync,
  openSync,
  readFileSync,
  readSync,
  readdirSync,
  statSync,
} from "node:fs";
import { gunzipSync } from "node:zlib";
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import type { AthcPageRow, AthcV3Layout } from "../../src/athc";
import {
  parseRunsContainer,
  unpackRunsContainer,
} from "../../src/webgpu/athenea/compactNodes";
import {
  compactChildPages,
  parseCompactAsset,
  parseCompactTables,
  resolveCompactTables,
} from "../../src/webgpu/athenea/compactPages";
import { makeLodCall } from "./compactLod";

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

// biome-ignore lint/suspicious/noExplicitAny: the WASM bindings
const wasm = (await import("spark-rs")) as unknown as Record<string, any>;
const lod = makeLodCall(wasm);

const SLOT = 3072;
const DIR = process.env.CLOSEUP_TRIAL ?? "";
type Row = AthcPageRow & { compact: number[] };

const ON = DIR !== "" && existsSync(DIR);

describe.skipIf(!ON)("close-up trial (car5120)", () => {
  // The describe's body runs even when skipped: nothing read without data.
  const files = ON
    ? (
        process.env.CLOSEUP_FILES?.split(",") ??
        readdirSync(DIR).filter((f) => f.endsWith(".athc"))
      ).sort()
    : [];
  const asset = ON
    ? parseCompactAsset(new Uint8Array(readFileSync(`${DIR}/car.cmpa`)))
    : undefined;
  it.skipIf(files.length > 0)("needs CLOSEUP_TRIAL", () => {});

  for (const file of files) {
    it(`${file} parses and cuts`, async () => {
      const path = `${DIR}/${file}`;
      const size = statSync(path).size;
      const fd = openSync(path, "r");
      const read = (offset: number, bytes: number) => {
        const out = new Uint8Array(bytes);
        readSync(fd, out, 0, bytes, offset);
        return out;
      };
      try {
        const prefixBytes = wasm.athc_prefix_bytes(read(0, 256)) as number;
        const prefix = read(0, prefixBytes);
        const layout = wasm.athc_layout(prefix, size) as AthcV3Layout;
        const rows = (layout.pages ?? []) as Row[];
        expect(rows.length).toBeGreaterThan(0);
        const k4 = layout.sections.findIndex(
          (s) => (s.id as string) === "CMPC",
        );
        expect(k4).toBeGreaterThanOrEqual(0);
        const t = (layout as unknown as { compactTable: number[] })
          .compactTable;
        const tables = resolveCompactTables(
          parseCompactTables(prefix.slice(t[0], t[0] + t[1])),
          asset,
        );
        expect(tables.basis.length).toBeGreaterThan(0);

        // Every page as the streamer lands it.
        const { lodId } = (await lod("newLodTree", {
          capacity: rows.length * SLOT,
          slot: SLOT,
          compact: true,
        })) as { lodId: number };
        let nodes = 0;
        let runs = 0;
        let stored = 0;
        let largest = 0;
        const lo = [1e9, 1e9, 1e9];
        const hi = [-1e9, -1e9, -1e9];
        const ranges: Record<string, unknown>[] = [];
        for (let page = 0; page < rows.length; page++) {
          const n = rows[page].count;
          expect(n).toBeGreaterThan(0);
          expect(n).toBeLessThanOrEqual(SLOT);
          largest = Math.max(largest, n);
          const span = layout.blocks[page].spans[k4];
          stored += span.stored;
          const c = read(span.offset, span.stored);
          const h = parseRunsContainer(c);
          expect(h.n).toBe(n);
          const trav = new Uint8Array(
            gunzipSync(c.subarray(h.travAt, h.travAt + h.travBytes)),
          );
          const u = unpackRunsContainer(c, h, trav, true);
          expect(u.planes.length).toBe(h.runs);
          const raw = u.raw as Uint8Array;
          for (const p of compactChildPages(raw, n, page)) {
            expect(p).toBeGreaterThan(page);
            expect(p).toBeLessThan(rows.length);
          }
          const box = new Float32Array(
            Uint32Array.from(rows[page].compact.slice(0, 6)).buffer,
          );
          for (let k = 0; k < 3; k++) {
            lo[k] = Math.min(lo[k], box[k]);
            hi[k] = Math.max(hi[k], box[3 + k]);
          }
          ranges.push({
            lodId,
            pageBase: page * SLOT,
            chunkBase: page * 65536,
            count: n,
            input: u.input,
            box,
          });
          nodes += n;
          runs += h.runs;
        }
        await lod("updateCompactLodTrees", { ranges });

        // A cut from far and near, every page resident.
        const c = lo.map((v, k) => (v + hi[k]) / 2);
        const r = Math.hypot(...lo.map((v, k) => hi[k] - v)) / 2;
        const cuts: Record<string, number> = {};
        for (const [name, d, W, H, tau] of [
          ["far", 6, 393, 852, 3],
          ["near", 1.2, 1920, 1080, 1],
        ] as [string, number, number, number, number][]) {
          const camera = new THREE.PerspectiveCamera(40, W / H, 0.001, 1000);
          camera.up.set(0, 0, 1);
          camera.position.fromArray(
            c.map((v, k) => v + d * r * [0.6, -0.7, 0.4][k]),
          );
          camera.lookAt(new THREE.Vector3().fromArray(c));
          camera.updateProjectionMatrix();
          camera.updateMatrixWorld();
          const pixel = (2 * Math.tan((camera.fov * Math.PI) / 360)) / H;
          const { keyIndices } = (await lod("traverseLodTrees", {
            maxSplats: rows.length * SLOT,
            pixelScaleLimit: pixel,
            instances: {
              piece: {
                lodId,
                rootPage: 0,
                viewToObjectCols: camera.matrixWorld.elements.slice(),
                lodScale: 1,
                behindFoveate: 1,
                coneFov0: 0,
                coneFov: 0,
                coneFoveate: 1,
              },
            },
            errorLimit: pixel * tau,
          })) as {
            keyIndices: Record<
              string,
              { numSplats: number; indices: Uint32Array }
            >;
          };
          const k = keyIndices.piece;
          expect(k.numSplats).toBeGreaterThan(0);
          for (const i of k.indices.subarray(0, k.numSplats)) {
            const page = Math.floor(i / SLOT);
            expect(page).toBeLessThan(rows.length);
            expect(i % SLOT).toBeLessThan(rows[page].count);
          }
          cuts[name] = k.numSplats;
        }
        wasm.dispose_lod_tree(lodId);
        console.log(
          `CLOSEUP_TRIAL ${JSON.stringify({ file, bytes: size, prefixBytes, pages: rows.length, nodes, largestPage: largest, runs, cmpcStored: stored, cuts })}`,
        );
      } finally {
        closeSync(fd);
      }
    }, 600_000);
  }
});
