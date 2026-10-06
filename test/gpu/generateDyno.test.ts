// Dyno generators and modifiers spliced into kernels/generate.slang at its
// hooks (src/webgpu/dyno), against the CPU transform of the source splats.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import {
  SplatEdit,
  SplatEditRgbaBlendMode,
  SplatEditSdf,
  SplatEditSdfType,
  SplatEdits,
} from "../../src/SplatEdit";
import { SplatTransformer } from "../../src/SplatGenerator";
import type { SplatMesh } from "../../src/SplatMesh";
import { SplatSkinning } from "../../src/SplatSkinning";
import * as d from "../../src/dyno";
import { snowBox } from "../../src/generators/snow";
import { staticBox } from "../../src/generators/static";
import { makeDepthColorModifier } from "../../src/modifiers/depthColor";
import { makeNormalColorModifier } from "../../src/modifiers/normalColor";
import { decodeExtSplat, encodeExtSplat } from "../../src/utils";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import { DynoKernels, type WgpuDyno } from "../../src/webgpu/dyno/DynoKernels";
import { kernelsGenerate } from "../../src/webgpu/generated/constants";
import generate from "../../src/webgpu/generated/kernels/generate";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { device, readBack, storage } from "./device";

const {
  GEN_SRC_EXT: SRC_EXT,
  GEN_OUT_EXT: OUT_EXT,
  GEN_SORT_RADIAL: SORT_RADIAL,
  GEN_DYNO_SOURCE: DYNO_SOURCE,
} = kernelsGenerate;
const N = 256;
const extA = new Uint32Array(N * 4);
const extB = new Uint32Array(N * 4);
for (let i = 0; i < N; i++) {
  encodeExtSplat(
    [extA, extB],
    i,
    i * 0.01,
    1,
    -2,
    0.1,
    0.2,
    0.05,
    0,
    0,
    0,
    1,
    0.8,
    0.25,
    0.5,
    0.75,
  );
}
const src = new Uint32Array(N * 8);
for (let i = 0; i < N; i++) {
  src.set(extA.subarray(4 * i, 4 * i + 4), 8 * i);
  src.set(extB.subarray(4 * i, 4 * i + 4), 8 * i + 4);
}

const rotate = new THREE.Quaternion().setFromEuler(
  new THREE.Euler(0.2, 0.9, -0.4),
);
const translate = new THREE.Vector3(0.5, -1, 2);
const scale = 2;
const viewCenter = new THREE.Vector3(0, 0, 5);

function toWorld(c: THREE.Vector3) {
  return c.clone().multiplyScalar(scale).applyQuaternion(rotate).add(translate);
}

describe.skipIf(!device)("generate.slang with dyno", () => {
  const registry = new KernelRegistry(device as GPUDevice);
  const kernels = new DynoKernels(registry, generate, "generate");

  async function run(owner: object, dyno: WgpuDyno, flags = 0) {
    const out = storage(N * 32);
    const metric = storage(N * 4);
    const params = UniformWriter.for(generate).setAll({
      numSplats: N,
      outBase: 0,
      flags: SRC_EXT | OUT_EXT | SORT_RADIAL | flags,
      numSh: 0,
      srcCount: N,
      rotate: rotate.toArray(),
      translateScale: [...translate.toArray(), scale],
      recolor: [1, 1, 1, 1],
      encoding: [0, 1, -12, 9],
      viewCenter: [...viewCenter.toArray(), 0],
      viewDir: [0, 0, -1, 0],
    });
    const { kernel, bindings } = kernels.prepare(owner, dyno);
    kernel.run({
      grid: [N],
      buffers: {
        src: storage(src),
        sh: storage(16),
        lodIndices: storage(16),
        outSplats: out,
        sortMetric: metric,
      },
      uniforms: params.data,
      bindings,
    });
    const words = new Uint32Array(await readBack(out));
    const a = new Uint32Array(N * 4);
    const b = new Uint32Array(N * 4);
    for (let i = 0; i < N; i++) {
      a.set(words.subarray(8 * i, 8 * i + 4), 4 * i);
      b.set(words.subarray(8 * i + 4, 8 * i + 8), 4 * i);
    }
    return {
      splat: (i: number) => decodeExtSplat([a, b], i),
      metric: new Float32Array(await readBack(metric)),
    };
  }

  const offset = d.dynoVec3(new THREE.Vector3(0, 1, 0));
  const objectModifier = d.dynoBlock(
    { gsplat: d.Gsplat },
    { gsplat: d.Gsplat },
    ({ gsplat }) => {
      const { center } = d.splitGsplat(gsplat as never).outputs;
      return {
        gsplat: d.combineGsplat({
          gsplat: gsplat as never,
          center: d.add(center, offset),
          r: d.dynoConst("float", 1),
        }),
      };
    },
  );

  it("applies object modifiers before the transform", async () => {
    const { splat } = await run({}, { objectModifiers: [objectModifier] });
    for (const i of [0, 17, N - 1]) {
      const s = splat(i);
      const e = toWorld(new THREE.Vector3(i * 0.01, 2, -2));
      expect(s.center.distanceTo(e)).toBeLessThan(1e-3);
      expect(s.color.r).toBeCloseTo(1, 2);
      expect(s.color.g).toBeCloseTo(0.5, 2);
    }
  });

  it("runs GLSL object modifiers, translated to WGSL", async () => {
    const glslModifier = d.dynoBlock(
      { gsplat: d.Gsplat },
      { gsplat: d.Gsplat },
      ({ gsplat }) => ({
        gsplat: new d.Dyno({
          inTypes: { gsplat: d.Gsplat, offset: "vec3" },
          outTypes: { gsplat: d.Gsplat },
          inputs: { gsplat, offset },
          globals: () => [
            "vec3 lift(vec3 p, vec3 o) {\n  p.yz += o.yz;\n  return p;\n}",
          ],
          statements: ({ inputs, outputs }) => [
            `${outputs.gsplat} = ${inputs.gsplat};`,
            `${outputs.gsplat}.center = lift(${inputs.gsplat}.center, ${inputs.offset});`,
            `${outputs.gsplat}.rgba.rg = vec2(1.0, ${inputs.gsplat}.rgba.g);`,
          ],
        }).outputs.gsplat,
      }),
    );
    const { splat } = await run({}, { objectModifiers: [glslModifier] });
    for (const i of [0, 17, N - 1]) {
      const s = splat(i);
      const e = toWorld(new THREE.Vector3(i * 0.01, 2, -2));
      expect(s.center.distanceTo(e)).toBeLessThan(1e-3);
      expect(s.color.r).toBeCloseTo(1, 2);
      expect(s.color.g).toBeCloseTo(0.5, 2);
    }
  });

  it("applies world modifiers after it, and drops deactivated splats", async () => {
    const fade = d.dynoFloat(0.5);
    const worldModifier = d.dynoBlock(
      { gsplat: d.Gsplat },
      { gsplat: d.Gsplat },
      ({ gsplat }) => {
        const { opacity, index, flags } = d.splitGsplat(
          gsplat as never,
        ).outputs;
        const odd = d.equal(
          d.imod(index, d.dynoConst("int", 2)),
          d.dynoConst("int", 1),
        );
        return {
          gsplat: d.combineGsplat({
            gsplat: gsplat as never,
            opacity: d.mul(opacity, fade),
            flags: d.select(odd, d.dynoConst("uint", 0), flags),
          }),
        };
      },
    );
    const { splat, metric } = await run(
      {},
      { objectModifiers: [objectModifier], worldModifiers: [worldModifier] },
    );
    for (const i of [0, 2, 100]) {
      const s = splat(i);
      const e = toWorld(new THREE.Vector3(i * 0.01, 2, -2));
      expect(s.center.distanceTo(e)).toBeLessThan(1e-3);
      expect(s.opacity).toBeCloseTo(0.4, 2);
      expect(metric[i]).toBeCloseTo(e.distanceTo(viewCenter), 3);
    }
    for (const i of [1, 3, 255]) {
      expect(metric[i]).toBe(Number.POSITIVE_INFINITY);
      expect(splat(i).opacity).toBe(0);
    }
  });

  it("generates splats with dynoSource", async () => {
    const spacing = d.dynoFloat(0.5);
    const generator = d.dynoBlock(
      { index: "int" },
      { gsplat: d.Gsplat },
      ({ index }) => ({
        gsplat: d.combineGsplat({
          flags: d.dynoLiteral("uint", "GSPLAT_FLAG_ACTIVE"),
          index: index as never,
          center: d.vec3(d.mul(d.float(index as never), spacing)),
          scales: d.dynoConst("vec3", [0.01, 0.02, 0.03]),
          quaternion: d.dynoConst("vec4", [0, 0, 0, 1]),
          rgba: d.dynoConst("vec4", [1, 0, 0, 1]),
        }),
      }),
    );
    const { splat } = await run({}, { generator }, DYNO_SOURCE);
    for (const i of [0, 9, N - 1]) {
      const s = splat(i);
      const e = toWorld(new THREE.Vector3().setScalar(i * 0.5));
      expect(s.center.distanceTo(e)).toBeLessThan(1e-2 * (1 + i * 0.01));
      expect(s.scales.y).toBeCloseTo(0.04, 3);
      expect(s.color.r).toBeCloseTo(1, 2);
    }
  });

  it("shares a pipeline between graphs of the same shape", async () => {
    const make = (y: number) => {
      const v = d.dynoVec3(new THREE.Vector3(0, y, 0));
      return d.dynoBlock(
        { gsplat: d.Gsplat },
        { gsplat: d.Gsplat },
        ({ gsplat }) => ({
          gsplat: d.combineGsplat({
            gsplat: gsplat as never,
            center: d.add(d.splitGsplat(gsplat as never).outputs.center, v),
          }),
        }),
      );
    };
    const a = { objectModifiers: [make(1)] };
    const b = { objectModifiers: [make(3)] };
    const ra = await run(a, a);
    const rb = await run(b, b);
    expect(kernels.prepare(a, a).kernel).toBe(kernels.prepare(b, b).kernel);
    expect(
      ra.splat(5).center.distanceTo(toWorld(new THREE.Vector3(0.05, 2, -2))),
    ).toBeLessThan(1e-3);
    expect(
      rb.splat(5).center.distanceTo(toWorld(new THREE.Vector3(0.05, 4, -2))),
    ).toBeLessThan(1e-3);
  });

  it("applies SplatEdit SDFs (world modifier)", async () => {
    const edits = new SplatEdits({ maxEdits: 1, maxSdfs: 1 });
    const edit = new SplatEdit({
      rgbaBlendMode: SplatEditRgbaBlendMode.MULTIPLY,
    });
    const sdf = new SplatEditSdf({
      type: SplatEditSdfType.SPHERE,
      radius: 0.3,
      color: new THREE.Color(1, 0, 0),
      displace: new THREE.Vector3(0, 0.5, 0),
    });
    sdf.position.copy(toWorld(new THREE.Vector3(1.28, 1, -2)));
    edits.update([{ edit, sdfs: [sdf] }]);
    const worldModifier = d.dynoBlock(
      { gsplat: d.Gsplat },
      { gsplat: d.Gsplat },
      ({ gsplat }) => ({ gsplat: edits.modify(gsplat as never) }),
    );
    const { splat } = await run({}, { worldModifiers: [worldModifier] });
    for (const i of [120, 128, 136]) {
      const s = splat(i);
      const e = toWorld(new THREE.Vector3(i * 0.01, 1, -2)).add(
        new THREE.Vector3(0, 0.5, 0),
      );
      expect(s.color.g).toBeCloseTo(0, 2);
      expect(s.color.r).toBeCloseTo(0.25, 2);
      expect(s.center.distanceTo(e)).toBeLessThan(1e-3);
    }
    for (const i of [0, 100, 160]) {
      expect(splat(i).color.g).toBeCloseTo(0.5, 2);
    }
  });

  it("applies SplatSkinning (object modifier)", async () => {
    const skinning = new SplatSkinning({
      mesh: {} as SplatMesh,
      numSplats: N,
      numBones: 2,
    });
    const identity = new THREE.Quaternion();
    const origin = new THREE.Vector3();
    skinning.setRestQuatPos(0, identity, origin);
    skinning.setRestQuatPos(1, identity, origin);
    const lift = new THREE.Vector3(0, 0, 1);
    const turn = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(0, 0, 1),
      Math.PI / 2,
    );
    skinning.setBoneQuatPos(0, identity, lift);
    skinning.setBoneQuatPos(1, turn, origin);
    for (let i = 0; i < N; i++) {
      skinning.setSplatBones(
        i,
        new THREE.Vector4(i % 2, 0, 0, 0),
        new THREE.Vector4(1, 0, 0, 0),
      );
    }
    skinning.skinTexture.needsUpdate = true;
    skinning.updateBones();
    const objectModifier = d.dynoBlock(
      { gsplat: d.Gsplat },
      { gsplat: d.Gsplat },
      ({ gsplat }) => ({ gsplat: skinning.modify(gsplat as never) }),
    );
    const { splat } = await run({}, { objectModifiers: [objectModifier] });
    for (const i of [0, 1, 50, 51]) {
      const c = new THREE.Vector3(i * 0.01, 1, -2);
      const skinned =
        i % 2 ? c.clone().applyQuaternion(turn) : c.clone().add(lift);
      expect(splat(i).center.distanceTo(toWorld(skinned))).toBeLessThan(1e-3);
    }
  });

  it("compiles Spark's modifiers and generators", async () => {
    const view = new SplatTransformer();
    view.updateFromMatrix(new THREE.Matrix4());
    const noise = staticBox({
      box: new THREE.Box3(
        new THREE.Vector3(-1, -1, -1),
        new THREE.Vector3(1, 1, 1),
      ),
      cells: new THREE.Vector3(8, 8, 4),
      dotScale: 0.01,
    });
    const { snow } = snowBox({});
    for (const dyno of [
      {
        worldModifiers: [
          makeDepthColorModifier(
            view,
            d.dynoFloat(1),
            d.dynoFloat(10),
            d.dynoBool(false),
          ),
          makeNormalColorModifier(view),
        ],
      },
      { generator: noise.generator },
      { generator: snow.generator },
    ]) {
      const { kernel } = kernels.prepare({}, dyno);
      const info = await registry
        .shaderModule(kernel.module)
        .getCompilationInfo();
      expect(info.messages.filter((m) => m.type === "error")).toEqual([]);
    }
    // The static generator's grid, inside its box.
    noise.frameUpdate?.({ object: noise, time: 1 } as never);
    const { splat } = await run(
      {},
      { generator: noise.generator },
      DYNO_SOURCE,
    );
    for (const i of [0, 77, 255]) {
      const c = splat(i).center.clone().sub(translate);
      c.applyQuaternion(rotate.clone().invert()).divideScalar(scale);
      expect(
        Math.max(Math.abs(c.x), Math.abs(c.y), Math.abs(c.z)),
      ).toBeLessThan(1.001);
      expect(splat(i).opacity).toBeCloseTo(1, 2);
    }
  });

  it("reports when a dyno's uniforms change", () => {
    const t = d.dynoFloat(0);
    const mod = d.dynoBlock(
      { gsplat: d.Gsplat },
      { gsplat: d.Gsplat },
      ({ gsplat }) => ({
        gsplat: d.combineGsplat({ gsplat: gsplat as never, opacity: t }),
      }),
    );
    const owner = {};
    const dyno = { objectModifiers: [mod] };
    expect(kernels.changed(owner, dyno)).toBe(true);
    expect(kernels.changed(owner, dyno)).toBe(false);
    t.value = 0.5;
    expect(kernels.changed(owner, dyno)).toBe(true);
    expect(kernels.changed(owner, dyno)).toBe(false);
    dyno.objectModifiers = [mod, mod];
    expect(kernels.changed(owner, dyno)).toBe(true);
  });

  // Two textures uploaded once each have the same version: a swap is still
  // a change.
  it("reports a texture uniform swapped for another", () => {
    const texture = (r: number) => {
      const t = new THREE.DataTexture(new Uint8Array([r, 0, 0, 255]), 1, 1);
      t.needsUpdate = true;
      return t;
    };
    const tex = d.dynoSampler2D(texture(10));
    const mod = d.dynoBlock(
      { gsplat: d.Gsplat },
      { gsplat: d.Gsplat },
      ({ gsplat }) => ({
        gsplat: d.combineGsplat({
          gsplat: gsplat as never,
          rgba: d.texelFetch(tex, d.dynoIvec2([0, 0])),
        }),
      }),
    );
    const owner = {};
    const dyno = { objectModifiers: [mod] };
    expect(kernels.changed(owner, dyno)).toBe(true);
    expect(kernels.changed(owner, dyno)).toBe(false);
    tex.value = texture(200);
    expect(kernels.changed(owner, dyno)).toBe(true);
  });
});
