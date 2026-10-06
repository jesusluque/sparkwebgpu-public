// athenea-web's scene.json (copies of two of its published scenes) read as
// its viewer reads them.

import { readFileSync } from "node:fs";
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import {
  applyAtheneaTransform,
  parseAtheneaScene,
  pickAtheneaFile,
} from "../../src/webgpu/athenea/sceneJson";

const json = (name: string) =>
  JSON.parse(
    readFileSync(
      new URL(`../fixtures/athenea-web/${name}.scene.json`, import.meta.url),
      "utf8",
    ),
  );

const URL_BASE = "https://athenea-assets.lucab.co.uk/scenes/pawn-hq/scene.json";

describe("athenea-web scene.json", () => {
  it("resolves files against the scene's folder and keeps absolute ones", () => {
    const s = parseAtheneaScene(json("pawn-hq"), URL_BASE);
    expect(s.id).toBe("pawn-hq");
    const rad = s.files.find((f) => f.format === "rad");
    expect(rad?.url).toBe(
      "https://athenea-assets.lucab.co.uk/scenes/pawn-hq/pawn-lod.rad",
    );
    const abs = parseAtheneaScene(
      { id: "x", files: [{ format: "spz", path: "https://a.b/c.spz" }] },
      URL_BASE,
    );
    expect(abs.files[0].url).toBe("https://a.b/c.spz");
    expect(s.cameras[0]).toEqual({
      name: "Referencia",
      position: [0, 0.048, 0.18],
      target: [0, 0.048, 0],
      fov: 39.6,
    });
    expect(s.transform).toEqual({ scale: 0.01 });
  });

  it("picks .athc first, then the scene's default, then the viewer's order", () => {
    const s = parseAtheneaScene(json("pawn-hq"), URL_BASE);
    // tlod is athenea-web's own LoD prototype: not ours to open.
    expect(pickAtheneaFile(s)?.format).toBe("rad");
    expect(pickAtheneaFile(s, "spz")?.format).toBe("spz");
    const withAthc = parseAtheneaScene(
      {
        ...json("pawn-hq"),
        files: [
          ...json("pawn-hq").files,
          { format: "athc", path: "pawn.athc" },
        ],
      },
      URL_BASE,
    );
    expect(pickAtheneaFile(withAthc)?.url).toMatch(/pawn\.athc$/);
    const hidden = parseAtheneaScene(
      {
        id: "x",
        files: [
          { format: "athc", path: "a.athc", viewer: false },
          { format: "ply", path: "a.ply" },
        ],
      },
      URL_BASE,
    );
    expect(pickAtheneaFile(hidden)?.format).toBe("ply");
  });

  it("applies transforms as athenea-web does: XYZ degrees, a file's own first", () => {
    const s = parseAtheneaScene(json("soar"), URL_BASE);
    expect(s.transform).toEqual({ rotation: [180, 0, 0] });
    const o = new THREE.Object3D();
    applyAtheneaTransform(o, s.transform);
    expect(o.rotation.x).toBeCloseTo(Math.PI);
    const glb = s.files.find((f) => f.format === "glb");
    expect(glb?.transform).toEqual({ rotation: [0, 180, 0] });
  });

  it("refuses what it cannot read", () => {
    expect(() => parseAtheneaScene({ files: [] }, URL_BASE)).toThrow(/no id/);
    expect(() =>
      parseAtheneaScene({ id: "x", files: [{ path: "a" }] }, URL_BASE),
    ).toThrow(/format and path/);
    expect(() =>
      parseAtheneaScene(
        {
          id: "x",
          files: [],
          cameras: [{ position: [0, 0], target: [0, 0, 0] }],
        },
        URL_BASE,
      ),
    ).toThrow(/cameras\[0\]\.position/);
  });
});
