import { describe, expect, it } from "vitest";
import { relightGrid } from "../../src/webgpu/athenea/relightPlugin";

describe("relightGrid", () => {
  it("keeps one row up to 65 535 workgroups of 256", () => {
    expect(relightGrid(1000)).toEqual([1000, 1]);
    expect(relightGrid(65535 * 256)).toEqual([65535 * 256, 1]);
  });
  it("splits a 256-page pool (65 536 workgroups) into rows", () => {
    const count = 256 * 65536;
    const [x, y] = relightGrid(count);
    expect(x).toBe(65535 * 256);
    expect(y).toBe(2);
    expect(x * y).toBeGreaterThanOrEqual(count);
    expect(Math.ceil(x / 256)).toBeLessThanOrEqual(65535);
  });
});
