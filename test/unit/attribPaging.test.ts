// The attribute pool's sizing by stream group (src/webgpu/attributes/
// attribPaging.ts): a .athc TX cloud at each device tier, demands, forms.

import { describe, expect, it } from "vitest";
import {
  PAGE_SPLATS,
  planAttribPaging,
  transferFormValues,
  transferForms,
} from "../../src/webgpu/attributes/attribPaging";
import type { AttributeSpec } from "../../src/webgpu/attributes/schema";
import { PluginHost } from "../../src/webgpu/plugins/PluginHost";
import { tierOfLimits, tierSpec } from "../../src/webgpu/plugins/tiers";

const MIB = 1024 * 1024;

// every_stream.athc's attributes (rust athc.rs attrib_specs), in file order.
const TX: AttributeSpec[] = [
  { name: "normalOct", format: "u32", components: 1 },
  { name: "emission", format: "u32", components: 1 },
  { name: "pbr", format: "u32", components: 1 },
  { name: "lobes", format: "u32", components: 3 },
  { name: "transfer", format: "f16", components: 112 },
  { name: "shadowBits", format: "u32", components: 8 },
  { name: "athcGroup", format: "u32", components: 2 },
];

const page = (bytes: number) => PAGE_SPLATS * bytes;

describe("planAttribPaging", () => {
  it("drops the relight streams below T2 and keeps the rest whole", () => {
    // T1: 128 MiB bindings, SH3 packed: 42 pages of splats.
    const plan = planAttribPaging(TX, 42, {
      budget: tierSpec(1).attribBudget,
      tier: 1,
    });
    expect(plan.groups.map((g) => [g.name, g.slots, g.paged])).toEqual([
      ["core", 42, false],
      ["material", 42, false],
    ]);
    expect(plan.dropped.map((d) => [d.name, d.reason])).toEqual([
      ["transfer", "demand"],
      ["shadowBits", "demand"],
    ]);
    const asked = planAttribPaging(TX, 42, {
      budget: tierSpec(1).attribBudget,
      tier: 1,
      demand: "all",
    });
    expect(asked.dropped.map((d) => d.reason)).toEqual(["tier", "tier"]);
    expect(plan.bytes).toBe((4 + 4 * 5) * 4 + 42 * page(8 + 4 + 4 + 4 + 12));
    expect(plan.bytes).toBeLessThan(128 * MIB);
  });

  it("pages the transfer in what the budget leaves (T2, T3)", () => {
    const at = (tier: 2 | 3, transferForm?: "full" | "indirect" | "direct") =>
      planAttribPaging(TX, 256, {
        budget: tierSpec(tier).attribBudget,
        tier,
        demand: "all",
        transferForm,
      });
    const t2 = at(2);
    expect(t2.groups.map((g) => [g.name, g.slots, g.paged])).toEqual([
      ["core", 256, false],
      ["material", 256, false],
      ["relight", 15, true],
    ]);
    // Everything whole would be 4.8 GB; the plan stays in the budget.
    const whole = 256 * page(8 + 24 + 32 + 224);
    expect(whole).toBeGreaterThan(4.8e9);
    expect(t2.bytes).toBeLessThanOrEqual(768 * MIB);
    expect(t2.groups[2].bytes).toBe(2 * 4 * 256 + 15 * page(32 + 224));
    // Shorter transfers fit more pages.
    expect(at(2, "indirect").groups[2].slots).toBe(25);
    expect(at(2, "direct").groups[2].slots).toBe(63);
    expect(
      at(2, "direct").specs.find((s) => s.name === "transfer")?.components,
    ).toBe(16);
    expect(at(3).groups[2].slots).toBe(63);
    expect(at(3).bytes).toBeLessThanOrEqual(1536 * MIB);
  });

  it("takes pages as asked, the group's budget be what it may", () => {
    const plan = planAttribPaging(TX, 8, {
      budget: 0,
      tier: 1,
      demand: ["transfer", "athcGroup"],
      pages: { relight: 3, core: 8 },
    });
    expect(plan.specs.map((s) => s.name)).toEqual(["athcGroup", "transfer"]);
    expect(plan.groups.map((g) => [g.name, g.slots, g.paged])).toEqual([
      ["core", 8, false],
      ["relight", 3, true],
    ]);
  });

  it("leaves out a group no page of which fits", () => {
    const plan = planAttribPaging(TX, 256, {
      budget: 256 * page(8 + 24) + 128 + 1e6,
      tier: 2,
      demand: "all",
    });
    expect(plan.groups.map((g) => g.name)).toEqual(["core", "material"]);
    expect(plan.dropped.filter((d) => d.reason === "budget").length).toBe(2);
  });

  it("knows the transfer's forms as athc_v3.rs does", () => {
    expect(transferForms(112)).toEqual([16, 64, 112]);
    expect(transferForms(84)).toEqual([9, 36, 84]);
    expect(transferForms(36)).toEqual([9, 36]);
    expect(transferForms(16)).toEqual([16]);
    expect(transferForms(10)).toEqual([10]);
    expect(transferFormValues(112, "indirect")).toBe(64);
    expect(transferFormValues(16, "indirect")).toBe(16);
  });

  it("measures the tier of a device's limits", () => {
    const limits = (bytes: number) => ({
      maxStorageBufferBindingSize: bytes,
      maxBufferSize: bytes,
    });
    expect(tierOfLimits(limits(128 * MIB))).toBe(1);
    expect(tierOfLimits(limits(1024 * MIB))).toBe(2);
    expect(tierOfLimits(limits(4096 * MIB))).toBe(3);
  });
});

describe("PluginHost.attributeDemand", () => {
  it("lists what the plugins the device runs require or read", () => {
    const host = new PluginHost({ tier: 2 });
    host
      .register({ id: "a", requires: { attributes: ["transfer"] } })
      .register({ id: "b", requires: { reads: ["emission"] } })
      .register({ id: "c", minTier: 3, requires: { attributes: ["lobes"] } })
      .register({
        id: "d",
        enabled: false,
        requires: { attributes: ["pbr"] },
      });
    expect(host.attributeDemand().sort()).toEqual(["emission", "transfer"]);
  });
});
