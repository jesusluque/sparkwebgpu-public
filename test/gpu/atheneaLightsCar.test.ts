// A light sidecar built as the Corvette's will be (usd-athc --light-layer,
// scripts/build-corvette-lights.mjs): test/fixtures/athenea-lights/
// headlight-piece.{athc,lights.athl}, a piece of athenea's headlight-cover
// TX bake (usd-athc --drop-backs --box) with three layers: "self" (the base
// cloud given as its own layer: usd-athc --light-verify checked it equals
// the file's colours), "cruce" (a fabricated glow from a point) and "drl"
// (out of reach of the piece: empty), scaled by athenea's C7 sidecar's
// emissionLuminance. Through the real WASM: the .athl's index space is the
// whole-file decode's (merged nodes then splats, what the plugin's whole
// layout needs), its hash the cloud's, its groups the sidecar's.

import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";

vi.mock("spark-rs", async () => await vi.importActual("spark-rs"));
vi.mock("../../src/wasm", () => ({
  initialization: Promise.resolve(),
  isInitialized: () => true,
}));

const wasm = await vi.importActual<typeof import("spark-rs")>("spark-rs");
wasm.initSync({
  module: readFileSync(
    new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
  ),
});

const { AthlStore, athcCloudHash, decodeAthl } = await import(
  "../../src/webgpu/athenea/athl"
);
const { parseLightSidecar } = await import(
  "../../src/webgpu/athenea/lightSidecar"
);

const file = (n: string) =>
  new Uint8Array(
    readFileSync(new URL(`../fixtures/athenea-lights/${n}`, import.meta.url)),
  );
const athc = file("headlight-piece.athc");
const bytes = file("headlight-piece.lights.athl");
const sidecar = parseLightSidecar(
  new TextDecoder().decode(file("corvette-c7.lights.usda")),
);

describe("a car's light layers from usd-athc", () => {
  it("line up with the whole-file decode and the sidecar", async () => {
    const data = await decodeAthl(bytes);
    const { merged, splatBase, splatCount, elementCount, cloudHash } =
      data.header;
    expect(cloudHash).toBe(athcCloudHash(athc.subarray(0, 4096)));
    // The page's decode (SplatMesh, extSplats): merged nodes then splats.
    const decoder = wasm.decode_to_extsplats(
      undefined,
      "cloud.athc",
      undefined,
      undefined,
      undefined,
    );
    decoder.push(athc);
    const decoded = decoder.finish() as { numSplats: number };
    expect(decoded.numSplats).toBe(merged + splatCount);
    expect(splatBase % 65536).toBe(0);
    expect(elementCount).toBe(splatBase + splatCount);

    // Groups as given, radiance the sidecar's luminance (1 for "self").
    expect(data.groups.map((g) => g.name)).toEqual(["self", "cruce", "drl"]);
    const radiance = Object.fromEntries(
      sidecar.groups.map((g) => [g.name, g.radiance]),
    );
    expect(data.groups.map((g) => g.radiance)).toEqual([
      1,
      radiance.cruce,
      radiance.drl,
    ]);
    // "self" covers every splat and merged node: every block of both runs.
    const blocksOf = (g: number) =>
      data.layers
        .filter((l) => l.group === g && l.kind === 0)
        .reduce((a, l) => a + l.blocks.length, 0);
    expect(blocksOf(0)).toBe(
      Math.ceil(merged / 256) + Math.ceil(splatCount / 256),
    );
    expect(blocksOf(1)).toBeGreaterThan(0);
    expect(blocksOf(2)).toBe(0);
    const packed = new AthlStore(data).pack();
    expect(packed[0]).toBe(3);
  });
});

// Opt-in: a set built by scripts/build-corvette-lights.mjs
// (CORVETTE_LIGHTS=<dir with corvette.json>): every part with an .athl
// lines up with its cloud's whole-file decode, its hash and the sidecar.
const SET = process.env.CORVETTE_LIGHTS;
describe.skipIf(!SET)("a built Corvette lights set", () => {
  it("lines up part by part", async () => {
    const dir = new URL(`file://${SET}/`);
    const read = (n: string) => new Uint8Array(readFileSync(new URL(n, dir)));
    const scene = JSON.parse(new TextDecoder().decode(read("corvette.json")));
    const side = parseLightSidecar(
      new TextDecoder().decode(read(scene.lights)),
    );
    const lit = scene.parts.filter((p: { athl?: string }) => p.athl);
    expect(lit.length).toBeGreaterThan(0);
    for (const part of lit) {
      const cloud = read(part.file);
      const data = await decodeAthl(read(part.athl));
      expect(data.header.cloudHash).toBe(
        athcCloudHash(cloud.subarray(0, 4096)),
      );
      const decoder = wasm.decode_to_extsplats(
        undefined,
        "cloud.athc",
        undefined,
        undefined,
        undefined,
      );
      decoder.push(cloud);
      const { numSplats } = decoder.finish() as { numSplats: number };
      expect(numSplats).toBe(data.header.merged + data.header.splatCount);
      expect(data.groups.map((g) => g.name)).toEqual(
        side.groups.map((g) => g.name),
      );
      expect(data.groups.map((g) => g.radiance)).toEqual(
        side.groups.map((g) => g.radiance),
      );
    }
  }, 300_000);
});
