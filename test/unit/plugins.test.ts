import { describe, expect, it, vi } from "vitest";
import type { KernelModule } from "../../src/webgpu/KernelModule";
import type { WgpuSplatMesh } from "../../src/webgpu/WgpuSplatRenderer";
import { capabilitiesOf } from "../../src/webgpu/capabilities";
import { FxChain } from "../../src/webgpu/fx/FxChain";
import { Gpu } from "../../src/webgpu/fx/Gpu";
import { Invert } from "../../src/webgpu/fx/effects/Invert";
import {
  blendVariants,
  colourVariants,
  presets,
} from "../../src/webgpu/generated/presets";
import {
  PluginHost,
  attributesPlugin,
  dynoPlugin,
  fadePlugin,
  fxPlugin,
  orderPlugins,
  pickVariant,
  tierOf,
  tintPlugin,
} from "../../src/webgpu/plugins";
import type { PluginFrame, SplatPlugin } from "../../src/webgpu/plugins/types";

const MIB = 1024 * 1024;
const DEFAULTS = {
  maxStorageBuffersPerShaderStage: 8,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxStorageBufferBindingSize: 128 * MIB,
  maxBufferSize: 256 * MIB,
};
const caps = (limits: Record<string, number> = {}, features: string[] = []) =>
  capabilitiesOf({
    limits: { ...DEFAULTS, ...limits } as unknown as GPUSupportedLimits,
    features: new Set(features) as unknown as GPUSupportedFeatures,
  });

const p = (id: string, extra: Partial<SplatPlugin> = {}): SplatPlugin => ({
  id,
  ...extra,
});
const ids = (list: readonly SplatPlugin[]) => list.map((x) => x.id);

// Variant tables whose loaders resolve to named stand-ins.
const fakeModule = (name: string) =>
  ({
    name,
    wgsl: "",
    reflection: { entries: [], bindings: [] },
  }) as KernelModule;
function fakeVariants(keys: { colour?: string[][]; blend?: string[][] }) {
  const table = (lists: string[][] = []) =>
    Object.fromEntries(
      lists.map((plugins) => [
        plugins.join("+"),
        {
          plugins,
          load: async () => ({
            "kernels/generate": fakeModule(`generate+${plugins.join("+")}`),
            "draw/splat_draw": fakeModule(`draw+${plugins.join("+")}`),
            "draw/splat_attrib_draw": fakeModule("attrib"),
            "tiles/tile_raster": fakeModule("tiles"),
          }),
        },
      ]),
    );
  return { colour: table(keys.colour), blend: table(keys.blend) };
}

const mesh = (attributes: string[] = []) =>
  ({
    source: {
      attribs: {
        columns: attributes.map((name) => ({ spec: { name } })),
      },
    },
  }) as unknown as WgpuSplatMesh;

describe("tiers", () => {
  it("measures T0-T3 from the storage limits", () => {
    expect(tierOf(null)).toBe(0);
    expect(tierOf(caps())).toBe(1);
    expect(
      tierOf(
        caps({
          maxStorageBufferBindingSize: 1024 * MIB,
          maxBufferSize: 1024 * MIB,
        }),
      ),
    ).toBe(2);
    // Dawn on Apple silicon with the adapter's limits.
    expect(
      tierOf(
        caps({
          maxStorageBufferBindingSize: 4294967292,
          maxBufferSize: 4294967295,
          maxStorageBuffersPerShaderStage: 10,
        }),
      ),
    ).toBe(3);
    // A big binding is no use with a small maxBufferSize.
    expect(tierOf(caps({ maxStorageBufferBindingSize: 4096 * MIB }))).toBe(1);
    expect(tierOf(caps({ maxStorageBuffersPerShaderStage: 4 }))).toBe(0);
  });

  it("rules plugins out by tier and features", () => {
    const host = new PluginHost({ capabilities: caps({}, ["shader-f16"]) });
    host
      .register(p("a"))
      .register(p("big", { minTier: 2 }))
      .register(p("f16", { requires: { features: ["shader-f16"] } }))
      .register(p("sub", { requires: { features: ["subgroups"] } }))
      .register(p("sort", { requires: { features: ["gpuSort"] } }));
    const r = host.resolve();
    expect(r.tier).toBe(1);
    expect(ids(r.active)).toEqual(["a", "f16", "sort"]);
    expect(r.inactive).toEqual([
      { id: "big", reason: "tier", detail: "needs T2, have T1" },
      { id: "sub", reason: "features", detail: "subgroups" },
    ]);
  });

  it("takes an explicit tier", () => {
    const host = new PluginHost({ tier: 3 });
    host.register(p("big", { minTier: 3 }));
    expect(host.isActive("big")).toBe(true);
    expect(new PluginHost().tier).toBe(0);
  });
});

describe("orderPlugins", () => {
  it("keeps registration order without constraints", () => {
    expect(ids(orderPlugins([p("a"), p("b"), p("c")]))).toEqual([
      "a",
      "b",
      "c",
    ]);
  });

  it("sorts by order, then after/before edges win", () => {
    const list = [
      p("a", { order: 10 }),
      p("b"),
      p("c", { order: -1 }),
      p("d", { after: ["a"] }),
      p("e", { before: ["c"] }),
    ];
    expect(ids(orderPlugins(list))).toEqual(["b", "e", "c", "a", "d"]);
  });

  it("ignores unknown ids and reports cycles", () => {
    expect(ids(orderPlugins([p("a", { after: ["zz"] }), p("b")]))).toEqual([
      "a",
      "b",
    ]);
    expect(() =>
      orderPlugins([p("a", { after: ["b"] }), p("b", { after: ["a"] })]),
    ).toThrow(/cycle among a, b/);
  });

  it("leaves the host unchanged when a register would make a cycle", () => {
    const host = new PluginHost({ tier: 1 });
    host.register(p("a", { after: ["b"] }));
    expect(() => host.register(p("b", { after: ["a"] }))).toThrow(/cycle/);
    expect(ids(host.plugins)).toEqual(["a"]);
    expect(() => host.register(p("a"))).toThrow(/already registered/);
  });
});

describe("PluginHost per-asset state", () => {
  it("checks required attributes and sections against each asset", () => {
    const host = new PluginHost({ tier: 1 });
    host.register(p("normals", { requires: { attributes: ["normal"] } }));
    host.register(p("tx", { requires: { sections: ["transfer"] } }));
    const plain = mesh();
    const withNormals = mesh(["normal", "label"]);
    expect(host.isActive("normals", plain)).toBe(false);
    expect(host.isActive("normals", withNormals)).toBe(true);
    expect(host.resolve(plain).inactive[0]).toEqual({
      id: "normals",
      reason: "attributes",
      detail: "normal",
    });
    const athc = {};
    host.setAsset(athc, { sections: ["base", "transfer"] });
    expect(host.isActive("tx", athc)).toBe(true);
    expect(host.isActive("tx", plain)).toBe(false);
    // Without an asset, only the device and the switches count.
    expect(host.isActive("tx")).toBe(true);
  });

  it("switches per asset over global over the plugin default", () => {
    const host = new PluginHost({ tier: 1 });
    host.register(p("a")).register(p("off", { enabled: false }));
    const m1 = mesh();
    const m2 = mesh();
    expect(host.isActive("off", m1)).toBe(false);
    host.enable("off", m1);
    expect(host.isActive("off", m1)).toBe(true);
    expect(host.isActive("off", m2)).toBe(false);
    host.disable("a");
    host.enable("a", m2);
    expect(host.isActive("a", m1)).toBe(false);
    expect(host.isActive("a", m2)).toBe(true);
    host.clearActive("a");
    expect(host.isActive("a", m1)).toBe(true);
    expect(host.resolve(m1).inactive).toEqual([]);
  });

  it("bumps its version on changes that alter the image", () => {
    const host = new PluginHost({ tier: 1 });
    const v0 = host.version;
    host.register(p("a"));
    host.disable("a");
    const v1 = host.version;
    host.disable("a"); // no change
    expect(v1).toBeGreaterThan(v0);
    expect(host.version).toBe(v1);
  });
});

describe("presets", () => {
  it("lists the compiled presets and their variants", () => {
    expect(presets.map((x) => x.name)).toEqual([
      "tint",
      "fade",
      "tint-fade",
      "athenea-raster",
      "athenea-relight",
      "athenea-lights",
      "athenea-raster-lights",
      "athenea-relight-lights",
    ]);
    expect(Object.keys(colourVariants)).toEqual([
      "example.tint",
      "athenea.raster",
      "athenea.relight",
      "athenea.lights",
      "athenea.raster+athenea.lights",
      "athenea.relight+athenea.lights",
    ]);
    expect(Object.keys(blendVariants)).toEqual(["example.fade"]);
  });

  it("picks the exact variant, else the longest inside the list", () => {
    const table = fakeVariants({ colour: [["a"], ["a", "b"], ["c"]] }).colour;
    expect(pickVariant(table, [])).toBe("");
    expect(pickVariant(table, ["a", "b"])).toBe("a+b");
    expect(pickVariant(table, ["a", "x", "b"])).toBe("a+b");
    expect(pickVariant(table, ["b", "a"])).toBe("a");
    expect(pickVariant(table, ["x"])).toBe("");
  });

  it("loads the generated variants", async () => {
    const tint = await colourVariants["example.tint"].load();
    expect(tint["kernels/generate"].name).toBe("kernels/generate+example.tint");
    const generate = tint["kernels/generate"].reflection;
    expect(generate.bindings.map((b) => b.name)).toContain("tintParams");
    const fade = await blendVariants["example.fade"].load();
    expect(Object.keys(fade).sort()).toEqual([
      "draw/splat_attrib_draw",
      "draw/splat_draw",
      "tiles/tile_raster",
    ]);
    for (const module of Object.values(fade)) {
      expect(module.reflection.bindings.map((b) => b.name)).toContain(
        "fadeParams",
      );
    }
  });

  it("refuses plugin Slang that no preset compiled", () => {
    const host = new PluginHost({ tier: 1 });
    expect(() =>
      host.register(p("x", { slang: { module: "plugins.x", colour: "X" } })),
    ).toThrow(/no preset compiles it/);
    expect(() =>
      host.register(
        p("example.tint", {
          slang: { module: "plugins.examples.tint", colour: "Other" },
        }),
      ),
    ).toThrow(/differs/);
    host.register(tintPlugin()).register(fadePlugin());
  });

  it("gives each mesh its colour variant once loaded, defaults otherwise", async () => {
    const host = new PluginHost({ tier: 1 });
    const tint = tintPlugin([0.5, 0.5, 0.5, 1]);
    host.register(tint);
    const m1 = mesh();
    const m2 = mesh();
    host.disable(tint.id, m2);
    // Loading starts on first use; that frame takes the default kernel.
    expect(host.generateKernel(m1)).toBeNull();
    await host.ready([m1, m2]);
    const k = host.generateKernel(m1);
    expect(k?.module.name).toBe("kernels/generate+example.tint");
    expect(new Float32Array(k?.uniforms.tintParams as ArrayBuffer)).toEqual(
      new Float32Array([0.5, 0.5, 0.5, 1]),
    );
    expect(host.generateKernel(m2)).toBeNull();
    tint.setTint([1, 0, 0, 1], m1);
    expect(
      new Float32Array(
        host.generateKernel(m1)?.uniforms.tintParams as ArrayBuffer,
      ),
    ).toEqual(new Float32Array([1, 0, 0, 1]));
  });

  it("reports colour plugins without a variant", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const host = new PluginHost({
      tier: 1,
      variants: fakeVariants({ colour: [["a"]] }),
    });
    // Slang ids are checked against the generated table: skip via no slang
    // check by registering plain objects with slang added afterwards.
    const a = p("a");
    const b = p("b");
    host.register(a).register(b);
    a.slang = { module: "m", colour: "A" };
    b.slang = { module: "m", colour: "B" };
    const r = host.resolve(mesh());
    expect(r.colour).toBe("a");
    expect(ids(r.active)).toEqual(["a"]);
    expect(r.inactive).toEqual([
      { id: "b", reason: "no-variant", detail: "colour" },
    ]);
    expect(warn).toHaveBeenCalledTimes(1);
    host.resolve(mesh());
    expect(warn).toHaveBeenCalledTimes(1);
    warn.mockRestore();
  });

  it("draws with the blend plugins active on any mesh", () => {
    const host = new PluginHost({
      tier: 1,
      variants: fakeVariants({ blend: [["f"], ["f", "g"]] }),
    });
    const f = p("f");
    const g = p("g", { requires: { attributes: ["normal"] } });
    host.register(f).register(g);
    f.slang = { module: "m", blend: "F" };
    g.slang = { module: "m", blend: "G" };
    expect(host.blendKey([mesh()])).toBe("f");
    expect(host.blendKey([mesh(), mesh(["normal"])])).toBe("f+g");
    host.disable("f");
    expect(host.blendKey([mesh(["normal"])])).toBe("");
  });
});

describe("PluginHost frame", () => {
  const frame = (meshes: WgpuSplatMesh[]) =>
    ({
      renderer: { meshes },
      camera: {},
      time: 0,
      deltaTime: 0,
    }) as unknown as PluginFrame;

  it("is dirty on host changes and when a plugin says so", () => {
    const host = new PluginHost({ tier: 1 });
    let say: "dirty" | "clean" = "clean";
    host.register(p("a", { onFrame: () => say }));
    const f = frame([]);
    expect(host.frame(f)).toBe(true); // registration
    expect(host.frame(f)).toBe(false);
    say = "dirty";
    expect(host.frame(f)).toBe(true);
    say = "clean";
    host.disable("a");
    expect(host.frame(f)).toBe(true);
    expect(host.frame(f)).toBe(false);
  });

  it("chains dyno plugins per mesh and leaves meshes' own dynos alone", () => {
    const host = new PluginHost({ tier: 1 });
    const m1 = { a: 1 } as never;
    const m2 = { a: 2 } as never;
    const g = { id: "gen" } as never;
    const update = vi.fn();
    host.register(dynoPlugin("one", { objectModifiers: [m1], update }));
    host.register(
      dynoPlugin("two", {
        objectModifiers: [m2],
        worldModifiers: [m1],
        generator: g,
      }),
    );
    const plain = mesh();
    const own = Object.assign(mesh(), { dyno: { objectModifiers: [] } });
    const ownDyno = own.dyno;
    host.frame(frame([plain, own]));
    expect(plain.dyno?.objectModifiers).toEqual([m1, m2]);
    expect(plain.dyno?.worldModifiers).toEqual([m1]);
    expect(plain.dyno?.generator).toBe(g);
    plain.dyno?.update?.({} as never);
    expect(update).toHaveBeenCalledTimes(1);
    expect(own.dyno).toBe(ownDyno);
    const first = plain.dyno;
    host.disable("one", plain);
    host.frame(frame([plain, own]));
    expect(plain.dyno).toBe(first);
    expect(plain.dyno?.objectModifiers).toEqual([m2]);
    host.disable("two", plain);
    host.frame(frame([plain, own]));
    expect(plain.dyno).toBeUndefined();
  });

  it("runs active plugins' pre-passes in stage order", () => {
    const host = new PluginHost({ tier: 1 });
    const order: string[] = [];
    const pass = (name: string) => ({
      name,
      encode: () => order.push(name),
    });
    host.register(p("b", { passes: [pass("b")], after: ["a"] }));
    host.register(p("a", { passes: [pass("a1"), pass("a2")] }));
    host.register(p("c", { passes: [pass("c")], enabled: false }));
    const encoder = {} as GPUCommandEncoder;
    host.encodePasses(encoder); // no frame yet: nothing
    host.frame(frame([]));
    host.encodePasses(encoder);
    expect(order).toEqual(["a1", "a2", "b"]);
  });

  it("collects controls of active plugins", () => {
    const host = new PluginHost({ tier: 1 });
    const fade = fadePlugin({ fog: 0.25 });
    host.register(fade);
    const controls = host.controls();
    expect(controls.map((c) => `${c.plugin}:${c.id}`)).toEqual([
      "example.fade:fog",
      "example.fade:opacity",
    ]);
    expect(controls[0].get?.()).toBe(0.25);
    controls[0].set?.(0.5);
    expect(fade.options.fog).toBe(0.5);
    host.disable(fade.id);
    expect(host.controls()).toEqual([]);
  });
});

describe("built-in plugins", () => {
  it("adds plugin effects after the chain's own, in stage order", () => {
    const gpu = {} as Gpu;
    vi.spyOn(Gpu, "forDevice").mockReturnValue(gpu);
    const chain = new FxChain(gpu);
    const own = chain.add(new Invert());
    const host = new PluginHost({ tier: 1 });
    host.register(
      fxPlugin("late", [{ effect: new Invert(), name: "x" }], { order: 1 }),
    );
    host.register(
      fxPlugin("early", [{ effect: new Invert() }, { effect: new Invert() }]),
    );
    host.applyFx(chain);
    expect(chain.steps.map((s) => s.instance)).toEqual([
      own.instance,
      "early.0",
      "early.1",
      "late.x",
    ]);
    const steps = [...chain.steps];
    host.applyFx(chain);
    expect(chain.steps).toEqual(steps);
    host.disable("early");
    host.applyFx(chain);
    expect(chain.steps.map((s) => s.instance)).toEqual([
      own.instance,
      "late.x",
    ]);
  });

  it("describes the attribute stage by the attributes it reads", () => {
    const a = attributesPlugin({
      slots: ["normal", { name: "label" }, "normal"],
    });
    expect(a.id).toBe("spark.attributes");
    expect(a.requires?.attributes).toEqual(["normal", "label"]);
    expect(a.stage).toBeNull();
  });
});
