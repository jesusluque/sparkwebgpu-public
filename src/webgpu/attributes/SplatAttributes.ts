// Per-Gaussian attributes in WgpuSplatRenderer: a stage that, after generate,
// gathers every mesh's toDraw attributes into a draw pool indexed like the
// accumulator (kernels/attrib_gather.slang), and draws with
// draw/splat_attrib_draw.slang: attributes in flat varyings, colour modes from
// them, and extra targets for picking (id) and composited attribute images.
//
//   const attributes = new SplatAttributes(splats, {
//     slots: ["normal", "label"], colorMode: "relight", targets: { id: true },
//   });
//   source.setAttribute("normal", normals, "snorm8", 3, { toDraw: true, direction: true });
//   ...
//   const hit = await attributes.pick(x, y); // { mesh, index } or null

import * as THREE from "three";
import type { KernelRegistry } from "../KernelRegistry";
import type {
  SplatDrawContext,
  SplatDrawVariant,
  SplatMeshRange,
  SplatRendererStage,
  WgpuSplatMesh,
  WgpuSplatRenderer,
} from "../WgpuSplatRenderer";
import {
  drawSplatAttribDraw as attribDraw,
  kernelsAttribGather,
} from "../generated/constants";
import attribDrawModule from "../generated/draw/splat_attrib_draw";
import gatherModule from "../generated/kernels/attrib_gather";
import {
  createReadback,
  createStorage,
  createUniform,
  readAndDestroy,
  reuseTexture,
  upload,
} from "../gpuBuffers";
import {
  type ReflectedRenderPipeline,
  createReflectedRenderPipeline,
} from "../renderPipeline";
import { UniformWriter } from "../uniforms";
import {
  ATTRIB_FORMATS,
  ATTRIB_NONE,
  type AttribPool,
  type AttributeSpec,
  type PoolLayout,
  attribGroupWords,
  poolHeader,
  poolLayout,
  poolWords,
} from "./schema";

const _position = new THREE.Vector3();
const _scale = new THREE.Vector3();
const COLOR_MODES = {
  splat: attribDraw.COLOR_SPLAT,
  label: attribDraw.COLOR_LABEL,
  relight: attribDraw.COLOR_RELIGHT,
  project: attribDraw.COLOR_PROJECT,
  value: attribDraw.COLOR_VALUE,
} as const;
export type AttribColorMode = keyof typeof COLOR_MODES;

/** A varying slot: four components of a draw attribute (comp4 = which four). */
export type AttribSlot = string | { name: string; comp4?: number };

export interface SplatAttributesOptions {
  /** Up to four draw attributes read into the vertex shader's flat varyings. */
  slots?: AttribSlot[];
  /** How slots colour the splats; colorSlot (and the next) feed it. */
  colorMode?: AttribColorMode;
  colorSlot?: number;
  /** Extra render targets: id for picking, slot 0 / slot 1 composited. */
  targets?: { id?: boolean; slot0?: boolean; slot1?: boolean };
  /** A splat is pickable where its alpha reaches this (default 0.3). */
  pickAlpha?: number;
  /** relight: world direction towards the light, and ambient. */
  light?: [number, number, number];
  ambient?: number;
  /** project: rgb = offset + rows . (slot, slot + 1), rows of 8. */
  projection?: [number[], number[], number[]];
  projectionOffset?: [number, number, number];
  /** label: highlight this label, dim the others (null for none). */
  highlight?: number | null;
}

export interface PickResult {
  mesh: WgpuSplatMesh;
  /** Source splat index (through the mesh's LOD indices). */
  index: number;
  /** Accumulator slot. */
  slot: number;
}

interface GpuPool {
  version: number;
  buffer: GPUBuffer;
}

export class SplatAttributes implements SplatRendererStage {
  readonly device: GPUDevice;
  readonly registry: KernelRegistry;
  options: SplatAttributesOptions;

  /** The draw pool's schema: every mesh's toDraw attributes, by name. */
  layout: PoolLayout = poolLayout([]);
  drawPool: GPUBuffer | null = null;
  idTexture: GPUTexture | null = null;
  slotTextures: (GPUTexture | null)[] = [null, null];

  private capacity = -1;
  private layoutKey = "";
  private gpuPools = new WeakMap<AttribPool, GpuPool>();
  private maps = new Map<WgpuSplatMesh, { key: string; buffer: GPUBuffer }>();
  private emptyPool: GPUBuffer;
  private empty: GPUBuffer;
  private uniform: GPUBuffer;
  private pipelines = new Map<string, ReflectedRenderPipeline>();
  private ranges: readonly SplatMeshRange[] = [];

  constructor(
    readonly renderer: WgpuSplatRenderer,
    options: SplatAttributesOptions = {},
  ) {
    this.device = renderer.device;
    this.registry = renderer.registry;
    this.options = options;
    this.emptyPool = upload(
      this.device,
      poolHeader(poolLayout([]), 0),
      "empty attribs",
    );
    this.empty = createStorage(this.device, 16, "empty");
    this.uniform = createUniform(
      this.device,
      UniformWriter.for(attribDrawModule, "attribParams").data.byteLength,
      "attrib draw params",
    );
    renderer.stages.push(this);
  }

  /** The draw-pool attribute id of `name`, or -1. */
  drawId(name: string): number {
    return this.layout.specs.findIndex((s) => s.name === name);
  }

  // A source pool on the GPU, re-uploaded when it changed.
  private gpuPool(pool: AttribPool): GPUBuffer {
    if (pool.gpuBuffer) return pool.gpuBuffer;
    let g = this.gpuPools.get(pool);
    if (!g || g.version !== pool.version) {
      g?.buffer.destroy();
      g = {
        version: pool.version,
        buffer: upload(this.device, pool.pack().words, "attribs"),
      };
      this.gpuPools.set(pool, g);
    }
    return g.buffer;
  }

  // The union of the meshes' toDraw attributes; the first mesh with a name
  // sets its format. Directions arrive in world space, so lose the flag.
  private drawSpecs(): AttributeSpec[] {
    const specs: AttributeSpec[] = [];
    for (const m of this.renderer.meshes) {
      for (const s of m.source.attribs?.schema ?? []) {
        if (s.toDraw && !specs.some((x) => x.name === s.name)) {
          specs.push({ ...s, direction: false });
        }
      }
    }
    return specs;
  }

  private ensurePool(total: number) {
    const specs = this.drawSpecs();
    const key = JSON.stringify(
      specs.map((s) => [s.name, s.format, s.components]),
    );
    if (key === this.layoutKey && total <= this.capacity) return;
    this.layoutKey = key;
    this.layout = poolLayout(specs);
    this.capacity = Math.max(total, Math.ceil(this.capacity * 1.5));
    this.drawPool?.destroy();
    this.drawPool = createStorage(
      this.device,
      poolWords(this.layout, this.capacity) * 4,
      "draw attribs",
    );
    this.device.queue.writeBuffer(
      this.drawPool,
      0,
      poolHeader(this.layout, this.capacity),
    );
    for (const m of this.maps.values()) m.buffer.destroy();
    this.maps.clear();
  }

  // Per draw attribute, the mesh's source attribute id (ATTRIB_NONE if absent).
  private attribMap(mesh: WgpuSplatMesh): GPUBuffer {
    const schema = mesh.source.attribs?.schema ?? [];
    const ids = this.layout.specs.map((s) => {
      const k = schema.findIndex((x) => x.name === s.name);
      return k < 0 ? ATTRIB_NONE : k;
    });
    const key = ids.join(",");
    const cached = this.maps.get(mesh);
    if (cached?.key === key) return cached.buffer;
    cached?.buffer.destroy();
    const buffer = upload(
      this.device,
      new Uint32Array(ids.length ? ids : [ATTRIB_NONE]),
      "attrib map",
    );
    this.maps.set(mesh, { key, buffer });
    return buffer;
  }

  generate(encoder: GPUCommandEncoder, ranges: readonly SplatMeshRange[]) {
    const total = ranges.reduce((n, r) => n + r.count, 0);
    this.ensurePool(total);
    this.ranges = ranges;
    for (const m of this.maps.keys()) {
      if (!ranges.some((r) => r.mesh === m)) {
        this.maps.get(m)?.buffer.destroy();
        this.maps.delete(m);
      }
    }
    if (this.layout.specs.length === 0) return;

    const kernel = this.registry.get(gatherModule, "gatherAttribs");
    const q = new THREE.Quaternion();
    const pass = encoder.beginComputePass({ label: "gather attribs" });
    for (const { mesh, base, count } of ranges) {
      const attribs = mesh.source.attribs;
      // As generate: the rotation of the world matrix the mesh is drawn with.
      mesh.object.matrixWorld.decompose(_position, q, _scale);
      const params = UniformWriter.for(gatherModule).setAll({
        numSplats: count,
        outBase: base,
        flags: mesh.lodIndices
          ? kernelsAttribGather.GATHER_USE_LOD |
            (mesh.lodFaded ? kernelsAttribGather.GATHER_LOD_FADE : 0)
          : 0,
        rotate: [q.x, q.y, q.z, q.w],
      });
      kernel.dispatch(pass, {
        grid: [count],
        buffers: {
          srcAttribs: attribs ? this.gpuPool(attribs) : this.emptyPool,
          lodIndices: mesh.lodBuffer ?? this.empty,
          attribMap: this.attribMap(mesh),
          drawAttribs: this.drawPool as GPUBuffer,
        },
        uniforms: params.data,
      });
    }
    pass.end();
  }

  private texture(
    current: GPUTexture | null,
    format: GPUTextureFormat,
    width: number,
    height: number,
    label: string,
  ): GPUTexture {
    return reuseTexture(this.device, current, {
      label,
      size: [width, height],
      format,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
  }

  draw(context: SplatDrawContext): SplatDrawVariant | null {
    const o = this.options;
    const t = o.targets ?? {};
    const { width, height } = context;
    const premultiplied: GPUBlendState = {
      color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
    };
    // The id's alpha is 0 or 1, so this either replaces or keeps the id.
    const select: GPUBlendState = {
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
    };
    const targets: (GPUColorTargetState | null)[] = [
      context.colorTarget,
      t.id ? { format: "rgba8unorm", blend: select } : null,
      t.slot0 ? { format: "rgba16float", blend: premultiplied } : null,
      t.slot1 ? { format: "rgba16float", blend: premultiplied } : null,
    ];
    while (targets.length > 1 && !targets[targets.length - 1]) targets.pop();

    // A blend-plugin variant when the frame has one (PluginHost).
    const module = context.plugin?.attribDraw ?? attribDrawModule;
    const key = JSON.stringify([module.name, targets, context.depthStencil]);
    let pipeline = this.pipelines.get(key);
    if (!pipeline) {
      pipeline = createReflectedRenderPipeline(this.device, module, {
        vertex: "splatAttribVertex",
        fragment: "splatAttribFragment",
        targets,
        depthStencil: context.depthStencil,
        label: "splat attrib draw",
      });
      this.pipelines.set(key, pipeline);
    }

    const attachment = (tex: GPUTexture): GPURenderPassColorAttachment => ({
      view: tex.createView(),
      loadOp: "clear",
      clearValue: [0, 0, 0, 0],
      storeOp: "store",
    });
    const attachments: (GPURenderPassColorAttachment | null)[] = [];
    if (t.id) {
      this.idTexture = this.texture(
        this.idTexture,
        "rgba8unorm",
        width,
        height,
        "splat ids",
      );
      attachments.push(attachment(this.idTexture));
    } else if (targets.length > 1) {
      attachments.push(null);
    }
    (["slot0", "slot1"] as const).forEach((name, k) => {
      if (t[name]) {
        this.slotTextures[k] = this.texture(
          this.slotTextures[k],
          "rgba16float",
          width,
          height,
          `splat ${name}`,
        );
        attachments.push(attachment(this.slotTextures[k] as GPUTexture));
      } else if (targets.length > 2 + k) {
        attachments.push(null);
      }
    });

    this.writeParams();
    return {
      pipeline,
      buffers: {
        ...context.plugin?.buffers,
        attribs: this.drawPool ?? this.emptyPool,
        attribParams: this.uniform,
      },
      attachments,
    };
  }

  private writeParams() {
    const o = this.options;
    // Per slot: format (ATTRIB_NONE: unused), first word in the record,
    // words and valid components of its comp4 group.
    const { layout } = this;
    const slots = (o.slots ?? []).slice(0, 4).map((s) => {
      const { name, comp4 = 0 } = typeof s === "string" ? { name: s } : s;
      const id = this.drawId(name);
      const spec = layout.specs[id];
      const [first, words] = spec ? attribGroupWords(spec, comp4) : [0, 0];
      if (!words) return [ATTRIB_NONE, 0, 1, 0];
      return [
        ATTRIB_FORMATS[spec.format],
        layout.offsets[id] + first,
        words,
        Math.min(spec.components - 4 * comp4, 4),
      ];
    });
    while (slots.length < 4) slots.push([ATTRIB_NONE, 0, 1, 0]);
    const [r, g, b] = o.projection ?? [[1], [0, 1], [0, 0, 1]];
    const row = (v: number[], k: number) =>
      Array.from({ length: 4 }, (_, i) => v[4 * k + i] ?? 0);
    const light = o.light ?? [0.4, 0.8, 0.5];
    const params = UniformWriter.for(attribDrawModule, "attribParams").setAll({
      slotFormat: slots.map((s) => s[0]),
      slotWord: slots.map((s) => s[1]),
      slotWords: slots.map((s) => s[2]),
      slotCount: slots.map((s) => s[3]),
      strideWords: layout.strideWords,
      headerWords: layout.headerWords,
      colorMode: COLOR_MODES[o.colorMode ?? "splat"],
      colorSlot: o.colorSlot ?? 0,
      pickAlpha: o.pickAlpha ?? 0.3,
      highlight: o.highlight == null ? 0 : o.highlight + 1,
      light: [...light, o.ambient ?? 0.25],
      projectR0: row(r, 0),
      projectR1: row(r, 1),
      projectG0: row(g, 0),
      projectG1: row(g, 1),
      projectB0: row(b, 0),
      projectB1: row(b, 1),
      projectOffset: [...(o.projectionOffset ?? [0.5, 0.5, 0.5]), 0],
    });
    this.device.queue.writeBuffer(this.uniform, 0, params.data);
  }

  /**
   * The front-most splat with alpha >= pickAlpha at pixel (x, y) of the last
   * frame drawn with targets.id, or null.
   */
  async pick(x: number, y: number): Promise<PickResult | null> {
    const tex = this.idTexture;
    if (!tex) throw new Error("SplatAttributes: enable targets.id to pick");
    const ranges = this.ranges;
    const px = Math.floor(x);
    const py = Math.floor(y);
    if (px < 0 || py < 0 || px >= tex.width || py >= tex.height) return null;
    const staging = createReadback(this.device, 256, "pick");
    const encoder = this.device.createCommandEncoder({ label: "pick" });
    encoder.copyTextureToBuffer(
      { texture: tex, origin: [px, py] },
      { buffer: staging, bytesPerRow: 256 },
      [1, 1],
    );
    this.device.queue.submit([encoder.finish()]);
    const [r, g, b, a] = new Uint8Array(await readAndDestroy(staging), 0, 4);
    const id = r | (g << 8) | (b << 16);
    if (a === 0 || id === 0) return null;
    const slot = id - 1;
    for (const { mesh, base, count } of ranges) {
      if (slot >= base && slot < base + count) {
        const i = slot - base;
        return { mesh, index: mesh.lodIndices ? mesh.lodIndices[i] : i, slot };
      }
    }
    return null;
  }

  dispose() {
    const i = this.renderer.stages.indexOf(this);
    if (i >= 0) this.renderer.stages.splice(i, 1);
    this.drawPool?.destroy();
    this.idTexture?.destroy();
    for (const t of this.slotTextures) t?.destroy();
    for (const m of this.maps.values()) m.buffer.destroy();
    this.maps.clear();
    this.emptyPool.destroy();
    this.empty.destroy();
    this.uniform.destroy();
  }
}
