import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { decodeAthcClip } from "../../src/athc";

// test/fixtures/athc/hop.atcl.gz: athc_skin.rs's round-trip clip (2 joints,
// 3 samples, element k = (k % 7) / 4 - 1), gzipped, as usd-athc
// --clip-files writes one.
describe("decodeAthcClip", () => {
  it("reads a gzipped ATCL clip", async () => {
    const bytes = new Uint8Array(
      readFileSync(new URL("../fixtures/athc/hop.atcl.gz", import.meta.url)),
    );
    const { joints, clip } = await decodeAthcClip(bytes);
    expect(joints).toBe(2);
    expect(clip.name).toBe("hop");
    expect(clip.timeCodesPerSecond).toBe(30);
    expect([...clip.times]).toEqual([1, 2, 3]);
    expect(clip.xforms.length).toBe(3 * 2 * 16);
    for (let k = 0; k < clip.xforms.length; k++) {
      expect(clip.xforms[k]).toBe((k % 7) * 0.25 - 1);
    }
  });

  it("refuses another file", async () => {
    await expect(decodeAthcClip(new Uint8Array(64))).rejects.toThrow(/ATCL/);
  });
});
