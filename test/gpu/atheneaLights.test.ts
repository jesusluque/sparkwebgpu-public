// athenea's light groups (atheneaLightsPlugin, slang/athenea_adapter/
// lights.slang) through WgpuSplatRenderer on Dawn:
//   - the shader's arithmetic against the CPU (direct LTC form factor
//     through a lens profile, sparse layers, the field) at two splats;
//   - linearity, 068 V5: the image with groups A and B on is the sum of
//     A alone and B alone (less the base once); twice the weight, twice
//     the light;
//   - isolation, 068 V4 in spirit: switching a group changes only the
//     pixels of the splats it reaches;
//   - the chain after the relight plugin (atheneaRelightPlugin), on a
//     device with its ten storage buffers.
//
// Two small splats facing the camera, one left and one right, each alone
// on its pixels, so a pixel's colour is opacity x the splat's light.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { encodeExtSplat } from "../../src/utils";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
} from "../../src/webgpu/WgpuSplatRenderer";
import type { AthlData } from "../../src/webgpu/athenea/athl";
import { atheneaLightsPlugin } from "../../src/webgpu/athenea/lightsPlugin";
import { atheneaRasterPlugin } from "../../src/webgpu/athenea/rasterPlugin";
import { atheneaRelightPlugin } from "../../src/webgpu/athenea/relightPlugin";
import { PluginHost } from "../../src/webgpu/plugins";
import { device, wideDevice } from "./device";

const W = 64;
const H = 48;
const Z = -2;
const OPACITY = 0.8;
const SPLATS = [
  { x: -0.6, albedo: [0.6, 0.5, 0.4] },
  { x: 0.6, albedo: [0.2, 0.3, 0.7] },
] as const;
// Pixel centres of the two splats (fov 50, aspect 4:3).
const halfW = Math.tan((25 * Math.PI) / 180) * 2 * (W / H);
const PIXELS = SPLATS.map((s) => Math.round(W / 2 + (s.x / halfW) * (W / 2)));

const h = (v: number) => THREE.DataUtils.toHalfFloat(v);
const f = (v: number) => THREE.DataUtils.fromHalfFloat(h(v));

// Group a: a one-sided square 0.5 in front of the left splat, facing it,
// through a 2 x 2 lens profile; plus an indirect layer there.
// Group b: an emission layer and a field on the right splat.
const QUAD: [number, number, number][] = [
  [-0.75, -0.15, Z + 0.5],
  [-0.75, 0.15, Z + 0.5],
  [-0.45, 0.15, Z + 0.5],
  [-0.45, -0.15, Z + 0.5],
];
const INDIRECT = [0.1, 0.2, 0.05];
const EMISSION = [0.3, 0.1, 0.02];
const FIELD_DC = 0.2; // red, through the constant harmonic
const PROFILE = [
  [1, 0.5, 0.25],
  [0.8, 0.6, 0.2],
  [0.6, 0.7, 0.3],
  [0.4, 0.8, 0.1],
];
const TINT = [1, 0.9, 0.8];
const LAMP = {
  origin: [-0.6, 0, Z + 0.5],
  axes: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, -1],
  ],
};

function layer(
  group: number,
  kind: number,
  components: number,
  at: number,
  values: number[],
) {
  const data = new Uint16Array(256 * components);
  values.forEach((v, c) => {
    data[at * components + c] = h(v);
  });
  return {
    group,
    kind,
    chunk: 0,
    components,
    blocks: new Uint16Array([0]),
    data,
  };
}

const ATHL: AthlData = {
  header: {
    version: 1,
    flags: 0,
    groupCount: 2,
    elementCount: 2,
    merged: 0,
    splatBase: 0,
    splatCount: 2,
    chunkSplats: 65536,
    blockSplats: 256,
    cloudHash: "0000000000000000",
    bakeHash: "0000000000000000",
  },
  groups: [
    {
      name: "a",
      flags: 0,
      polygonFirst: 0,
      polygonCount: 1,
      profile: 0,
      tint: TINT as [number, number, number],
      origin: LAMP.origin as [number, number, number],
      axes: LAMP.axes as [
        [number, number, number],
        [number, number, number],
        [number, number, number],
      ],
      radiance: 1,
    },
    {
      name: "b",
      flags: 0,
      polygonFirst: 1,
      polygonCount: 0,
      profile: -1,
      tint: [1, 1, 1],
      origin: [0, 0, 0],
      axes: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      radiance: 2, // baked at 2: w is halved for its layers
    },
  ],
  polygons: [{ group: 0, radiance: [1, 1, 1], vertices: QUAD }],
  profiles: [
    {
      width: 2,
      height: 2,
      lon: [-0.4, 0.4],
      lat: [-0.4, 0.4],
      outside: [0, 0, 0],
      texels: new Uint16Array(PROFILE.flat().map(h)),
    },
  ],
  layers: [
    layer(0, 0, 3, 0, INDIRECT),
    layer(1, 1, 3, 1, EMISSION),
    layer(1, 2, 27, 1, [FIELD_DC / 0.28209479, 0, 0]),
  ],
};

const SIDECAR = `#usda 1.0
def Scope "Lights" {
    def Scope "a" { float athenea:lightGroup:radiance = 1.5  float athenea:lightGroup:riseSeconds = 0  float athenea:lightGroup:fallSeconds = 0 }
    def Scope "b" { float athenea:lightGroup:radiance = 1  color3f athenea:lightGroup:color = (1, 0.8, 0.6)  float athenea:lightGroup:riseSeconds = 0  float athenea:lightGroup:fallSeconds = 0 }
    def Scope "off"  { dictionary athenea:lightState:targets = {} }
    def Scope "A"    { dictionary athenea:lightState:targets = { double a = 1 } }
    def Scope "A2"   { dictionary athenea:lightState:targets = { double a = 2 } }
    def Scope "B"    { dictionary athenea:lightState:targets = { double b = 1 } }
    def Scope "AB"   { dictionary athenea:lightState:targets = { double a = 1  double b = 1 } }
}
`;

// --- the CPU side (rust athl.rs polygon_form_factor, profile_sample) -------

type V = [number, number, number];
const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: V, b: V) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V, b: V): V => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const norm = (a: V): V => {
  const l = Math.sqrt(dot(a, a));
  return [a[0] / l, a[1] / l, a[2] / l];
};

function formFactor(x: V, n: V, verts: V[]): number {
  const d = verts.map((v) => sub(v, x));
  if (dot(cross(sub(d[1], d[0]), sub(d[2], d[0])), sub([0, 0, 0], d[0])) <= 0)
    return 0;
  let sum = 0;
  for (let i = 0; i < d.length; i++) {
    const a = norm(d[i]);
    const b = norm(d[(i + 1) % d.length]);
    const g = cross(a, b);
    sum +=
      (Math.acos(Math.min(Math.max(dot(a, b), -1), 1)) * dot(n, g)) /
      Math.sqrt(dot(g, g));
  }
  return Math.abs(sum) / (2 * Math.PI);
}

function profile(x: V): V {
  const d = sub(x, LAMP.origin as V);
  const l = norm([
    dot(d, LAMP.axes[0] as V),
    dot(d, LAMP.axes[1] as V),
    dot(d, LAMP.axes[2] as V),
  ]);
  const lon = Math.atan2(l[0], l[2]);
  const lat = Math.asin(l[1]);
  const u = (lon + 0.4) / 0.8;
  const v = (lat + 0.4) / 0.8;
  const fx = Math.min(Math.max(u * 2 - 0.5, 0), 1);
  const fy = Math.min(Math.max(v * 2 - 0.5, 0), 1);
  const t = (i: number, j: number, c: number) => f(PROFILE[j * 2 + i][c]);
  return [0, 1, 2].map((c) => {
    const a = t(0, 0, c) + (t(1, 0, c) - t(0, 0, c)) * fx;
    const b = t(0, 1, c) + (t(1, 1, c) - t(0, 1, c)) * fx;
    return (a + (b - a) * fy) * TINT[c];
  }) as V;
}

/**
 * Each splat's pixel with groups a and b at levels la, lb: its light times
 * its alpha at that pixel (`alpha`, measured: a small splat's footprint
 * there is not its opacity).
 */
function expected(
  la: number,
  lb: number,
  alpha = [OPACITY, OPACITY],
): number[][] {
  const wa = [1.5 * la, 1.5 * la, 1.5 * la];
  const wb = [1, 0.8, 0.6].map((c) => (c * lb) / 2);
  return SPLATS.map((s, i) => {
    const x: V = [s.x, 0, Z];
    const light = [...s.albedo] as number[];
    if (i === 0) {
      const F = formFactor(x, [0, 0, 1], QUAD);
      const p = profile(x);
      for (let c = 0; c < 3; c++) {
        light[c] += wa[c] * (s.albedo[c] * F * p[c] + f(INDIRECT[c]));
      }
    } else {
      for (let c = 0; c < 3; c++) {
        light[c] += wb[c] * (f(EMISSION[c]) + (c === 0 ? FIELD_DC : 0));
      }
    }
    return light.map((c) => c * alpha[i]);
  });
}

function packNormal(n: V): number {
  const l1 = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2]);
  const m = n.map((v) => v / l1);
  let p = [m[0], m[1]];
  if (m[2] < 0) {
    p = [
      (1 - Math.abs(m[1])) * Math.sign(m[0] || 1),
      (1 - Math.abs(m[0])) * Math.sign(m[1] || 1),
    ];
  }
  const q = p.map((v) =>
    Math.round(Math.min(Math.max(v * 0.5 + 0.5, 0), 1) * 65535),
  );
  return (q[0] | (q[1] << 16)) >>> 0;
}

function harness(d: GPUDevice) {
  const color = d.createTexture({
    size: [W, H],
    format: "rgba16float",
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.TEXTURE_BINDING,
  });
  const target = {
    texture: { colorSpace: THREE.LinearSRGBColorSpace },
    samples: 0,
  } as unknown as THREE.RenderTarget;
  const fakeRenderer = {
    backend: {
      isWebGPUBackend: true,
      device: d,
      context: {} as GPUCanvasContext,
      get: (r: object) =>
        r === target.texture ? { texture: color } : undefined,
    },
  };
  const camera = new THREE.PerspectiveCamera(50, W / H, 0.05, 100);
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();

  const source = () => {
    const a = new Uint32Array(8);
    const b = new Uint32Array(8);
    SPLATS.forEach((s, i) =>
      encodeExtSplat(
        [a, b],
        i,
        s.x,
        0,
        Z,
        0.06,
        0.06,
        0.002,
        0,
        0,
        0,
        1,
        OPACITY,
        ...s.albedo,
      ),
    );
    const src = GpuSplatSource.fromExt(d, a, b, 2);
    src.setAttribute(
      "normalOct",
      [packNormal([0, 0, 1]), packNormal([0, 0, 1])],
      "u32",
      1,
    );
    return src;
  };

  async function image(): Promise<Float32Array> {
    const bytesPerRow = 256 * Math.ceil((W * 8) / 256);
    const buf = d.createBuffer({
      size: bytesPerRow * H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: color }, { buffer: buf, bytesPerRow }, [
      W,
      H,
    ]);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const halves = new Uint16Array(buf.getMappedRange());
    const out = new Float32Array(W * H * 3);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        for (let c = 0; c < 3; c++) {
          out[3 * (y * W + x) + c] = THREE.DataUtils.fromHalfFloat(
            halves[(y * bytesPerRow) / 2 + 4 * x + c],
          );
        }
      }
    }
    buf.unmap();
    buf.destroy();
    return out;
  }

  /** Renders each state in turn with one renderer, as a page switches them. */
  async function render(
    states: string[],
    relight: boolean,
  ): Promise<Float32Array[]> {
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      alwaysGenerate: true,
    });
    const mesh = splats.add(source());
    const host = new PluginHost({ capabilities: splats.capabilities });
    const lights = atheneaLightsPlugin({ sidecar: SIDECAR });
    if (relight) {
      const r = atheneaRelightPlugin({ noDome: true, intensity: 0 });
      r.setStoredLinear(mesh, true);
      host.register(r);
    } else {
      const raster = atheneaRasterPlugin();
      raster.setStoredLinear(mesh, true);
      host.register(raster);
    }
    host.register(lights);
    lights.setStoredLinear(mesh, true);
    lights.setLights(mesh, ATHL);
    host.attach(splats);
    await host.ready();
    const why = host.resolve(mesh).inactive;
    expect(why).toEqual([]);
    const out: Float32Array[] = [];
    for (const s of states) {
      lights.setLightState(s);
      for (let frame = 0; frame < 2; frame++) {
        const enc = d.createCommandEncoder();
        enc
          .beginRenderPass({
            colorAttachments: [
              {
                view: color.createView(),
                loadOp: "clear",
                storeOp: "store",
                clearValue: [0, 0, 0, 1],
              },
            ],
          })
          .end();
        d.queue.submit([enc.finish()]);
        splats.render(camera, target);
        await d.queue.onSubmittedWorkDone();
      }
      out.push(await image());
    }
    host.detach();
    splats.dispose();
    return out;
  }

  const at = (img: Float32Array, i: number) => {
    const p = 3 * ((H / 2) * W + PIXELS[i]);
    return [img[p], img[p + 1], img[p + 2]];
  };
  return { render, at };
}

const worst = (a: number[], b: number[]) =>
  Math.max(
    ...a.map((x, i) => Math.abs(x - b[i]) / Math.max(0.05, Math.abs(b[i]))),
  );

describe.skipIf(!device)("athenea light groups", () => {
  const { render, at } = harness(device as GPUDevice);

  it("lights each splat as the CPU does (direct through the lens, layers, field)", async () => {
    const [off, a, b, ab] = await render(["off", "A", "B", "AB"], false);
    // The splats' alpha at their pixels, from the base colour alone.
    const alpha = [0, 1].map((i) => at(off, i)[0] / SPLATS[i].albedo[0]);
    expect(Math.min(...alpha)).toBeGreaterThan(0.3);
    for (const [img, la, lb] of [
      [off, 0, 0],
      [a, 1, 0],
      [b, 0, 1],
      [ab, 1, 1],
    ] as const) {
      const want = expected(la, lb, alpha);
      for (const i of [0, 1]) {
        if (process.env.ATHENEA_LOG)
          process.stderr.write(`${la}${lb} ${i}: ${at(img, i)} / ${want[i]}\n`);
        expect(worst(at(img, i), want[i])).toBeLessThan(3e-3);
      }
    }
    // The direct term is there (not just the layer).
    const F = formFactor([-0.6, 0, Z], [0, 0, 1], QUAD);
    expect(F).toBeGreaterThan(0.05);
  });

  it("is linear in the groups (068 V5): A + B = AB, 2A = twice A", async () => {
    const [off, a, b, ab, a2] = await render(
      ["off", "A", "B", "AB", "A2"],
      false,
    );
    let sum = 0;
    let twice = 0;
    for (let k = 0; k < off.length; k++) {
      sum = Math.max(sum, Math.abs(a[k] + b[k] - off[k] - ab[k]));
      twice = Math.max(twice, Math.abs(2 * (a[k] - off[k]) - (a2[k] - off[k])));
    }
    // Half floats in the target: an ulp near 1 is 1e-3.
    expect(sum).toBeLessThan(2e-3);
    expect(twice).toBeLessThan(2e-3);
    expect(Math.max(...a.map((v, k) => v - off[k]))).toBeGreaterThan(0.05);
  });

  it("changes only the pixels of the splats a group reaches (068 V4)", async () => {
    const [off, a, b] = await render(["off", "A", "B"], false);
    const changed = (img: Float32Array) => {
      const cols = new Set<number>();
      for (let k = 0; k < img.length; k++) {
        if (Math.abs(img[k] - off[k]) > 1e-4) cols.add(Math.floor(k / 3) % W);
      }
      return [...cols];
    };
    const ca = changed(a);
    const cb = changed(b);
    expect(ca.length).toBeGreaterThan(0);
    expect(cb.length).toBeGreaterThan(0);
    expect(Math.max(...ca)).toBeLessThan(W / 2);
    expect(Math.min(...cb)).toBeGreaterThanOrEqual(W / 2);
  });
});

describe.skipIf(!wideDevice)("athenea light groups after the relight", () => {
  it("adds the groups' light to the relit colour", async () => {
    const { render, at } = harness(wideDevice as GPUDevice);
    const [off, ab] = await render(["off", "AB"], true);
    // The relight without a dome or lights leaves the splats black: the
    // groups' light is all there is; the alpha is the raster's (above).
    const { render: raster } = harness(wideDevice as GPUDevice);
    const [plain] = await raster(["off"], false);
    const alpha = [0, 1].map((i) => at(plain, i)[0] / SPLATS[i].albedo[0]);
    const base = expected(0, 0, alpha);
    const lit = expected(1, 1, alpha);
    for (const i of [0, 1]) {
      const got = at(ab, i).map((v, c) => v - at(off, i)[c]);
      const want = lit[i].map((v, c) => v - base[i][c]);
      if (process.env.ATHENEA_LOG)
        process.stderr.write(
          `relight ${i}: ${at(off, i)} | ${got} / ${want}\n`,
        );
      expect(worst(got, want)).toBeLessThan(5e-3);
    }
  });
});

// The synthetic asset (rust/build-lod athenea-lights) through the real WASM
// reader: the three files agree, the layers are sparse, and one page's
// layers read alone from its byte range decode as in the whole file.
describe("the synthetic light sidecar", async () => {
  const { readFileSync } = await import("node:fs");
  const { vi } = await import("vitest");
  const wasm = await vi.importActual<typeof import("spark-rs")>("spark-rs");
  wasm.initSync({
    module: readFileSync(
      new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
    ),
  });
  const { AthlStore, athcCloudHash, athlChunkRange } = await import(
    "../../src/webgpu/athenea/athl"
  );
  const { parseLightSidecar } = await import(
    "../../src/webgpu/athenea/lightSidecar"
  );
  const file = (n: string) =>
    new Uint8Array(
      readFileSync(
        new URL(`../../examples/webgpu/athenea-lights/${n}`, import.meta.url),
      ),
    );
  const athc = file("car.athc");
  const bytes = file("car.lights.athl");
  const sidecar = parseLightSidecar(
    new TextDecoder().decode(file("car.lights.usda")),
  );

  it("decodes, matches its cloud and its sidecar, and is sparse", () => {
    const data = wasm.decode_athl(bytes) as AthlData & {
      header: { sections: { tag: number }[] };
    };
    expect(data.groups.map((g) => g.name)).toEqual(
      sidecar.groups.map((g) => g.name),
    );
    expect(sidecar.warnings).toEqual([]);
    expect(data.header.cloudHash).toBe(athcCloudHash(athc));
    expect(sidecar.cloudHash).toBe(data.header.cloudHash);
    const { merged, splatBase, splatCount, elementCount } = data.header;
    expect(splatBase % 65536).toBe(0);
    expect(elementCount).toBe(splatBase + splatCount);
    expect(merged).toBeGreaterThan(0);
    // Indirect layers reach some blocks of each chunk, not all.
    const blocks = Math.ceil(merged / 256) + Math.ceil(splatCount / 256);
    for (let g = 0; g < 3; g++) {
      const n = data.layers
        .filter((l) => l.group === g && l.kind === 0)
        .reduce((a, l) => a + l.blocks.length, 0);
      expect(n).toBeGreaterThan(0);
      expect(n).toBeLessThan(blocks);
    }
    const packed = new AthlStore(data).pack();
    expect(packed[0]).toBe(3);
    expect(packed[1]).toBe(Math.ceil(elementCount / 65536));
  });

  it("reads one page's layers from its byte range", () => {
    const data = wasm.decode_athl(bytes) as AthlData;
    const header = wasm.athl_header(
      bytes.slice(0, wasm.athl_prefix_bytes(bytes.slice(0, 128))),
    );
    const chunk = Math.floor(header.splatBase / 65536);
    const range = athlChunkRange(header, chunk) as [number, number];
    const slice = bytes.slice(range[0], range[1]);
    const sections = header.sections.filter(
      (s: { tag: number; chunk: number }) =>
        s.tag === 0x5259414c && s.chunk === chunk,
    );
    expect(sections.length).toBeGreaterThan(0);
    for (const s of sections) {
      const l = wasm.decode_athl_layer(
        slice.subarray(s.offset - range[0], s.offset - range[0] + s.bytes),
      );
      const whole = data.layers.find(
        (x) => x.chunk === chunk && x.group === l.group && x.kind === l.kind,
      );
      expect(whole && [...whole.blocks]).toEqual([...l.blocks]);
      expect(whole?.data.length).toBe(l.data.length);
    }
  });
});
