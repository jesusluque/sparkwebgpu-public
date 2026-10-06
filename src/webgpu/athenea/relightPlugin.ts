// athenea's relighting on the PluginHost (plan-athenea.md, phase 4, stage
// 1): a cloud with athenea's streams (a .athc's normals, material, lobes,
// emission, TX transfer and open-direction cells) lit by any dome, an
// analytic sun and lights, the way athenea's raster lights it.
//
//   const relight = atheneaRelightPlugin({ hdri, rotation: 0.4, intensity: 1 });
//   host.register(relight).register(atheneaOutputPlugin({ view: "agx" }));
//   host.attach(splats);
//   relight.set({ sun: { direction: [0.3, 0.8, 0.5], intensity: 3 } });
//
// Per frame, before generate (a PrePass):
//   - the sky, when it changed: AtheneaSky prepares the dome as athenea's
//     technique::Environment does (harmonics, the sun taken out, the GGX
//     levels of the sky without it);
//   - the kept terms, when the sky, the lights, a mesh's place or its data
//     changed: athenea's splatTransferViewless (the body's light under the
//     sky, the field's coupling, the sun's share), a splat each -- athenea's
//     txCache;
//   - each splat's colour for this eye, when the eye moved or anything above
//     changed: athenea's relitSplat (slang/athenea_adapter/relight.slang).
// generate's colour plugin (relight_colour.slang) then takes that colour.
//
// Where a cloud keeps athenea's curvature (.athc v3 CURV, the attribute
// `curvature`), the pass also reads a solid glass's far face from it (a
// lens: lensExit) and, with the last frame's projection, how the surface
// turns a pixel away (SplatSlope): the reflection's slope across the
// footprint, or the sharp coat and polish, go to the draw per splat and are
// shaded per pixel there by the plugin's blend term (AtheneaRelightBlend,
// athenea's splat_blend), as athenea's raster draws them.
//
// A splat whose transfer page a paged pool has not brought keeps its
// captured colour (athcTransferResident) until the page arrives.
//
// The streams are read from the mesh's attribute pool through
// core/attrib.slang, by name, so a paged pool (gpuBuffer) is bound as it is.
// The colour leaves in linear light, encoded for Spark's draw like the raster
// plugin's: draw into a HalfFloat target and end with atheneaOutputPlugin.
//
// Needs 10 storage buffers in a compute stage (athenea's lights module
// brings its IES tables along): a device made with splatRequiredLimits
// (Chrome on a Mac offers 10, Safari 44), tier 2.

import * as THREE from "three";
import type { WgpuSplatMesh, WgpuSplatRenderer } from "../WgpuSplatRenderer";
import { ATTRIB_NONE, type AttribPool } from "../attributes/schema";
import relightModule from "../generated/athenea_adapter/relight";
import { atheneaAdapterRelight } from "../generated/constants";
import { createStorage, upload } from "../gpuBuffers";
import type { PluginFrame, SplatPlugin } from "../plugins/types";
import { UniformWriter } from "../uniforms";
import { AtheneaSky, ENV, type SkyImage } from "./AtheneaSky";
import {
  type AtheneaLightRecord,
  LIGHT_NORMALIZE,
  LIGHT_SHADOW,
  distantMatrix,
  packLightRecords,
} from "./lights";

export const ATHENEA_RELIGHT_ID = "athenea.relight";
/** Storage buffers the relight kernel binds. */
export const RELIGHT_STORAGE_BUFFERS = 10;

const C = atheneaAdapterRelight;
type Rgb = [number, number, number];
type Vec3 = [number, number, number] | THREE.Vector3;

export interface AtheneaSun {
  /** Towards the sun, in the world. */
  direction: Vec3;
  /** Irradiance on a surface facing it (UsdLux distant intensity, normalized). */
  intensity?: number;
  colour?: Rgb;
  /** Angular diameter, radians (default 0.53 degrees). */
  angle?: number;
  /** Shadowed by the cloud's open-direction cells (default true). */
  shadow?: boolean;
}

export interface AtheneaRelightOptions {
  /** The dome's lat-long image (linear radiance); null: its colour alone. */
  hdri?: SkyImage | null;
  /** Turns the dome about +Y, radians. */
  rotation?: number;
  intensity?: number;
  exposure?: number;
  colour?: Rgb;
  /** No dome at all (lights and sun only). */
  noDome?: boolean;
  /** An analytic sun (a UsdLux distant light) beside the dome. */
  sun?: AtheneaSun | null;
  /** More lights, as athenea's records (sphere, rect, disk, distant...). */
  lights?: readonly AtheneaLightRecord[];
  /** Use the transfer's indirect half and reflected field (default true). */
  indirect?: boolean;
  /** The cloud's colours are baked radiance (athenea's litBody). */
  litBody?: boolean;
  /** The cloud's index of refraction (0: nothing bends). */
  ior?: number;
  /** Emission times this (default 1). */
  emission?: number;
  /**
   * Per-pixel slope and sharp coat/polish (needs the cloud's curvature).
   * Costly: every covered pixel shades every splat's lobes, about 8x the
   * frame time on the pawn at full screen (default false).
   */
  pixelDetail?: boolean;
  /**
   * athenea's stage: the object whose space the clouds were baked in, where
   * the page has placed or turned it (the Corvette's Z-up stage under a car
   * turned to three's Y-up). A TX transfer, its cells and its reflected
   * field are directions of the bake's world, and athenea reads them with
   * world directions, so the relighting runs in this object's frame: the
   * clouds placed in it, the dome, the sun and the lights taken into it.
   * Default none: three's world is the stage (a rigid frame; its scale is
   * not undone for the lights).
   */
  frame?: THREE.Object3D | null;
}

type RelightState = Required<Omit<AtheneaRelightOptions, "hdri" | "frame">> & {
  hdri: SkyImage | null;
  frame: THREE.Object3D | null;
};

export interface AtheneaRelightPlugin extends SplatPlugin {
  /** The options in force (change them with set). */
  readonly options: Readonly<RelightState>;
  set(options: AtheneaRelightOptions): void;
  /** Whether `asset`'s stored colours are linear light (else sRGB-encoded). */
  setStoredLinear(asset: object, linear: boolean): void;
  storedLinearOf(mesh: WgpuSplatMesh): boolean;
  /**
   * `asset`'s index of refraction (athenea's per-cloud `ior`, which a .athc
   * does not carry; a glass pawn's head is 1.5), over options.ior.
   */
  setIor(asset: object, ior: number): void;
  /**
   * `asset` is a shadow catcher (athenea mesh2splat --shadow-catcher): drawn
   * black, as opaque as what the object took of the light reaching it
   * (splat_project's catcherOpacity, through its open-direction cells).
   */
  setCatcher(asset: object, catcher: boolean): void;
  /** The prepared dome (after the first frame). */
  readonly sky: AtheneaSky | null;
  /** Dispatches so far. */
  readonly stats: { relit: number; viewless: number; skies: number };
}

interface PoolOnGpu {
  version: number;
  buffer: GPUBuffer;
  owned: boolean;
  ids: Record<string, number>;
  transferCount: number;
  shadowWords: number;
}

interface MeshState {
  relit: GPUBuffer;
  pool: PoolOnGpu | null;
  /** What the kept terms were computed for. */
  viewlessKey: string;
  /** What the colours were computed for. */
  relitKey: string;
}

const STREAMS = [
  "normalOct",
  "emission",
  "pbr",
  "lobes",
  "transfer",
  "shadowBits",
  "curvature",
] as const;

/** Bytes of the draw's per-slot record (relight_colour.slang atheneaPixel). */
const PIXEL_RECORD_BYTES = 48;

function keysOf(mesh: WgpuSplatMesh | undefined): object[] {
  if (!mesh) return [];
  return mesh.object ? [mesh, mesh.object] : [mesh];
}

function pagedStoredLinear(mesh: WgpuSplatMesh): boolean | undefined {
  const paged = (mesh.object as { paged?: { athcStoredLinear?: boolean } })
    ?.paged;
  return paged?.athcStoredLinear;
}

const identities = new WeakMap<object, number>();
let identityCount = 0;
/** A number for an object's identity (a buffer, an index array). */
function identity(o: object | null | undefined): number {
  if (!o) return -1;
  let k = identities.get(o);
  if (k === undefined) {
    identityCount += 1;
    k = identityCount;
    identities.set(o, k);
  }
  return k;
}

/**
 * The relight kernels' grid: one thread a splat, in rows of
 * kRelightDispatchRow threads once a cloud is wider than one dispatch
 * dimension allows (65 535 workgroups of 256).
 */
export function relightGrid(count: number): [number, number] {
  const row = C.kRelightDispatchRow;
  return count <= row ? [count, 1] : [row, Math.ceil(count / row)];
}

/** A zeroed uniform block for the blend term until the first pass fills it. */
function createUniformBlock(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: "relight pixel params",
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

const vec3 = (v: Vec3): THREE.Vector3 =>
  v instanceof THREE.Vector3 ? v.clone() : new THREE.Vector3(...v);

/**
 * athenea's relighting as an ISplatColour plugin of generate (variant
 * "athenea-relight") fed by a pass before it.
 */
export function atheneaRelightPlugin(
  initial: AtheneaRelightOptions = {},
): AtheneaRelightPlugin {
  const options: RelightState = {
    hdri: null,
    rotation: 0,
    intensity: 1,
    exposure: 0,
    colour: [1, 1, 1],
    noDome: false,
    sun: null,
    lights: [],
    indirect: true,
    litBody: false,
    ior: 0,
    emission: 1,
    pixelDetail: false,
    frame: null,
  };
  const linear = new WeakMap<object, boolean>();
  const iors = new WeakMap<object, number>();
  const catchers = new WeakMap<object, boolean>();
  const isCatcher = (mesh: WgpuSplatMesh) =>
    keysOf(mesh).some((k) => catchers.get(k) === true);
  const states = new WeakMap<WgpuSplatMesh, MeshState>();
  const stats = { relit: 0, viewless: 0, skies: 0 };
  let renderer: WgpuSplatRenderer | null = null;
  let sky: AtheneaSky | null = null;
  let lightsBuffer: GPUBuffer | null = null;
  let envOfLight: GPUBuffer | null = null;
  let ies: { records: GPUBuffer; values: GPUBuffer } | null = null;
  let lightCount = 0;
  let envLights = 0;
  /** Bumped when the lights or the sky change. */
  let lightsVersion = 0;
  let lightsDirty = true;
  let dirty = true;
  /** Three's world to athenea's stage (options.frame's inverse), and its turn. */
  let toStage = new THREE.Matrix4();
  let stageTurn = new THREE.Matrix4();
  let stageKey = "";
  /** Brings the stage up to date: a moved frame prepares the sky again. */
  const updateStage = () => {
    const f = options.frame;
    if (f) {
      f.updateWorldMatrix(true, false);
      toStage = f.matrixWorld.clone().invert();
    } else {
      toStage = new THREE.Matrix4();
    }
    const q = new THREE.Quaternion();
    toStage.decompose(new THREE.Vector3(), q, new THREE.Vector3());
    stageTurn = new THREE.Matrix4().makeRotationFromQuaternion(q);
    const key = toStage.elements.map((v) => v.toPrecision(7)).join();
    if (key !== stageKey) {
      stageKey = key;
      if (sky) {
        sky.set({ frame: f ? stageTurn : null });
      }
      lightsDirty = true;
    }
  };

  const lookupLinear = (mesh: WgpuSplatMesh) => {
    for (const k of keysOf(mesh)) {
      const v = linear.get(k);
      if (v !== undefined) return v;
    }
    return pagedStoredLinear(mesh) ?? false;
  };

  const iorOf = (mesh: WgpuSplatMesh) => {
    for (const k of keysOf(mesh)) {
      const v = iors.get(k);
      if (v !== undefined) return v;
    }
    return options.ior;
  };

  const records = (): AtheneaLightRecord[] => {
    const out: AtheneaLightRecord[] = [];
    if (!options.noDome && sky) out.push(sky.record());
    const s = options.sun;
    if (s) {
      out.push({
        kind: "distant",
        flags: LIGHT_NORMALIZE | (s.shadow === false ? 0 : LIGHT_SHADOW),
        sizeX: s.angle ?? (0.53 * Math.PI) / 180,
        colour: (s.colour ?? [1, 1, 1]).map(
          (c) => c * (s.intensity ?? 1),
        ) as Rgb,
        matrix: distantMatrix(vec3(s.direction).applyMatrix4(stageTurn)),
      });
    }
    for (const l of options.lights) {
      out.push(
        options.frame
          ? {
              ...l,
              matrix: toStage.clone().multiply(l.matrix ?? new THREE.Matrix4()),
            }
          : l,
      );
    }
    return out;
  };

  const writeLights = (device: GPUDevice) => {
    const list = records();
    lightCount = list.length;
    envLights = options.noDome ? 0 : 1;
    const words = packLightRecords(list);
    if (!lightsBuffer || lightsBuffer.size !== words.byteLength) {
      lightsBuffer?.destroy();
      lightsBuffer = createStorage(device, words.byteLength, "relight lights");
    }
    device.queue.writeBuffer(lightsBuffer, 0, words);
    const of = new Uint32Array(Math.max(list.length, 1)).fill(ENV.kEnvNone);
    if (envLights) of[0] = 0;
    envOfLight?.destroy();
    envOfLight = upload(device, of, "relight envOfLight");
    lightsVersion += 1;
  };

  const poolOf = (
    device: GPUDevice,
    mesh: WgpuSplatMesh,
    previous: PoolOnGpu | null,
  ): PoolOnGpu | null => {
    const pool: AttribPool | null = mesh.source.attribs;
    if (!pool) return null;
    if (previous && previous.version === pool.version) {
      if (!pool.gpuBuffer || previous.buffer === pool.gpuBuffer) {
        return previous;
      }
    }
    if (previous?.owned) previous.buffer.destroy();
    const components = (name: string) =>
      pool.column(name)?.spec.components ?? 0;
    if (pool.gpuBuffer) {
      const ids: Record<string, number> = {};
      for (const s of STREAMS) {
        const k = pool.id(s);
        ids[s] = k < 0 ? ATTRIB_NONE : k;
      }
      return {
        version: pool.version,
        buffer: pool.gpuBuffer,
        owned: false,
        ids,
        transferCount: components("transfer"),
        shadowWords: components("shadowBits"),
      };
    }
    const { layout, words } = pool.pack((s) =>
      (STREAMS as readonly string[]).includes(s.name),
    );
    const ids: Record<string, number> = {};
    for (const s of STREAMS) {
      const k = layout.specs.findIndex((spec) => spec.name === s);
      ids[s] = k < 0 ? ATTRIB_NONE : k;
    }
    return {
      version: pool.version,
      buffer: upload(device, words, "relight attribs"),
      owned: true,
      ids,
      transferCount: components("transfer"),
      shadowWords: components("shadowBits"),
    };
  };

  let emptyPool: GPUBuffer | null = null;
  /** The draw's records, three uint4 an accumulator slot. */
  let pixel: GPUBuffer | null = null;
  /** The first dome's sun and the map's side, for the blend term. */
  let pixelParams: GPUBuffer | null = null;
  /** The accumulator's layout the pixel records were written for. */
  let pixelLayout = "";

  const slotsOf = (r: WgpuSplatRenderer) =>
    r.meshes.reduce(
      (n, m) => n + (m.lodIndices ? m.lodIndices.length : m.source.count),
      0,
    );

  /** Which mesh holds which accumulator slots, and which are relit. */
  const layoutOf = (r: WgpuSplatRenderer) =>
    r.meshes
      .map(
        (m) =>
          `${identity(m)}:${identity(m.lodIndices)}:${m.lodIndices?.length ?? m.source.count}:${!r.plugins || r.plugins.isActive(ATHENEA_RELIGHT_ID, m)}`,
      )
      .join();

  const pixelBuffer = (device: GPUDevice, slots: number): GPUBuffer => {
    const bytes = Math.max(slots, 1) * PIXEL_RECORD_BYTES;
    if (!pixel || pixel.size < bytes) {
      pixel?.destroy();
      pixel = createStorage(
        device,
        Math.ceil(bytes * (pixel ? 1.5 : 1)),
        "relight pixel",
      );
      pixelLayout = "";
    }
    return pixel;
  };

  const stateOf = (device: GPUDevice, mesh: WgpuSplatMesh): MeshState => {
    const bytes = Math.max(mesh.source.count, 1) * C.kRelightStride * 16;
    let s = states.get(mesh);
    if (!s || s.relit.size !== bytes) {
      s?.relit.destroy();
      s = {
        relit: createStorage(device, bytes, "relit splats"),
        pool: s?.pool ?? null,
        viewlessKey: "",
        relitKey: "",
      };
      states.set(mesh, s);
    }
    s.pool = poolOf(device, mesh, s.pool);
    return s;
  };

  const relightMesh = (
    pass: GPUComputePassEncoder,
    r: WgpuSplatRenderer,
    mesh: WgpuSplatMesh,
    camera: THREE.Camera,
  ) => {
    const { device } = r;
    const state = stateOf(device, mesh);
    const { source, object } = mesh;
    object.updateMatrixWorld();
    // The cloud in athenea's stage (options.frame), and the eye.
    const world = toStage.clone().multiply(object.matrixWorld);
    const eyeInWorld = new THREE.Vector3().setFromMatrixPosition(
      camera.matrixWorld,
    );
    const eye = eyeInWorld.clone().applyMatrix4(toStage);
    const eyeObject = eyeInWorld
      .clone()
      .applyMatrix4(new THREE.Matrix4().copy(object.matrixWorld).invert());
    const pool = state.pool;
    const ids = pool?.ids ?? {};
    const id = (name: string) => ids[name] ?? ATTRIB_NONE;
    const transferCount =
      id("transfer") === ATTRIB_NONE ? 0 : (pool?.transferCount ?? 0);
    const shadowWords =
      id("shadowBits") === ATTRIB_NONE ? 0 : (pool?.shadowWords ?? 0);
    const kTransfer =
      transferCount >= 9 && shadowWords >= 8 ? 2 : transferCount >= 9 ? 1 : 0;
    const storedLinear = lookupLinear(mesh);
    // The frame's projection, for SplatSlope (splat_project's tx3, ty3):
    // the cloud's space to the eye's with depth ahead, and the focal and
    // centre in pixels of the last target drawn.
    const size = r.lastDrawSize;
    const p = camera.projectionMatrix.elements;
    const slopeOn =
      options.pixelDetail && Boolean(size) && id("curvature") !== ATTRIB_NONE;
    const toEye = new THREE.Matrix4().multiplyMatrices(
      camera.matrixWorldInverse,
      object.matrixWorld,
    ).elements;
    const eyeRow = (k: number, sign: number) => [
      sign * toEye[k],
      sign * toEye[k + 4],
      sign * toEye[k + 8],
      sign * toEye[k + 12],
    ];
    const w = size?.width ?? 1;
    const h = size?.height ?? 1;
    const projection = {
      view0: eyeRow(0, 1),
      view1: eyeRow(1, 1),
      view2: eyeRow(2, -1),
      // px = w/2 (P00 x / d - P02 + 1), py = h/2 (1 - P11 y / d + P12), d = -z.
      focal: [
        (w / 2) * p[0],
        -(h / 2) * p[5],
        (w / 2) * (1 - p[8]),
        (h / 2) * (1 + p[9]),
      ],
      viewport: [w, h, 0, 0],
    };
    let flags =
      (source.format === "ext" ? C.kRelightExt : 0) |
      (source.lodOpacity ? C.kRelightLodOpacity : 0) |
      (storedLinear ? C.kRelightLinear : 0) |
      (options.litBody ? C.kRelightLit : 0) |
      (options.indirect ? C.kRelightIndirect : 0) |
      (slopeOn ? C.kRelightSlope : 0);
    const e = world.elements;
    const row = (k: number) => [e[k], e[k + 4], e[k + 8], e[k + 12]];
    const params = UniformWriter.for(relightModule, "relightParams").setAll({
      count: source.count,
      flags,
      lightCount,
      envLights,
      envBaseSide: sky?.baseSide ?? ENV.kEnvCoarseSide,
      normalOct: id("normalOct"),
      emission: id("emission"),
      pbr: id("pbr"),
      lobes: id("lobes"),
      transfer: id("transfer"),
      shadowBits: id("shadowBits"),
      transferCount,
      shadowWords,
      ior: iorOf(mesh),
      emissionScale: options.emission,
      curvature: id("curvature"),
      encoding: source.encoding,
      row0: row(0),
      row1: row(1),
      row2: row(2),
      eyeObject: [eyeObject.x, eyeObject.y, eyeObject.z, 1],
      eyeWorld: [eye.x, eye.y, eye.z, 1],
      ...projection,
    });
    emptyPool ??= upload(device, new Uint32Array([0, 0, 4, 0]), "relight pool");
    const buffers = {
      relightSrc: source.src,
      relightPool: pool?.buffer ?? emptyPool,
      relightLights: lightsBuffer as GPUBuffer,
      envSh: (sky as AtheneaSky).envSh,
      envSun: (sky as AtheneaSky).envSun,
      envTexels: (sky as AtheneaSky).envTexels as GPUBuffer,
      envOfLight: envOfLight as GPUBuffer,
      relit: state.relit,
      iesRecords: (ies as { records: GPUBuffer }).records,
      iesValues: (ies as { values: GPUBuffer }).values,
    };
    const placed = [
      lightsVersion,
      options.indirect,
      options.litBody,
      e.join(),
      identity(source.src),
      source.count,
      source.version,
      pool?.version ?? -1,
      identity(pool?.buffer),
      identity(mesh.lodIndices),
      storedLinear,
    ].join("|");
    if (isCatcher(mesh)) {
      if (state.relitKey === placed) return;
      r.registry.get(relightModule, "atheneaRelightCatcher").dispatch(pass, {
        grid: relightGrid(source.count),
        buffers,
        uniforms: params.data,
      });
      state.relitKey = placed;
      stats.relit += 1;
      return;
    }
    const kept = kTransfer >= 2 && envLights > 0;
    if (kept && state.viewlessKey !== placed) {
      r.registry.get(relightModule, "atheneaRelightViewless").dispatch(pass, {
        grid: relightGrid(source.count),
        buffers,
        uniforms: params.data,
      });
      state.viewlessKey = placed;
      stats.viewless += 1;
    }
    const key = `${placed}|${eye.toArray().join()}|${iorOf(mesh)}|${options.emission}|${slopeOn ? `${toEye.join()}|${p.join()}|${w}x${h}` : ""}`;
    if (state.relitKey === key) return;
    if (kept) flags |= C.kRelightCache;
    params.set("flags", flags);
    const entry =
      kTransfer === 2
        ? "atheneaRelight"
        : kTransfer === 1
          ? "atheneaRelightFirst"
          : "atheneaRelightPlain";
    r.registry.get(relightModule, entry).dispatch(pass, {
      grid: relightGrid(source.count),
      buffers,
      uniforms: params.data,
    });
    state.relitKey = key;
    stats.relit += 1;
  };

  const plugin: AtheneaRelightPlugin = {
    id: ATHENEA_RELIGHT_ID,
    // Linear light past 1: the renderer keeps it in float (SplatPlugin.hdr).
    hdr: true,
    minTier: 2,
    // A pager brings the streams it reads (WgpuSplatPager.attributePlan).
    requires: { reads: STREAMS },
    slang: {
      module: "athenea_adapter.relight_colour",
      colour: "AtheneaRelightColour",
      blend: "AtheneaRelightBlend",
    },
    options,
    stats,
    get sky() {
      return sky;
    },
    passes: [
      {
        name: "athenea relight",
        encode(encoder: GPUCommandEncoder, frame: PluginFrame) {
          const r = frame.renderer;
          const host = r.plugins;
          const meshes = r.meshes.filter(
            (m) => !host || host.isActive(ATHENEA_RELIGHT_ID, m),
          );
          if (!meshes.length) return;
          const { device } = r;
          if (!sky) {
            sky = new AtheneaSky(device, r.registry);
            stageKey = "";
          }
          updateStage();
          if (sky.dirty) {
            sky.set({
              image: options.hdri,
              rotation: options.rotation,
              intensity: options.intensity,
              exposure: options.exposure,
              colour: options.colour,
            });
            sky.prepare(encoder);
            stats.skies += 1;
            lightsDirty = true;
          }
          ies ??= {
            records: createStorage(device, 32, "relight ies records"),
            values: createStorage(device, 16, "relight ies values"),
          };
          if (lightsDirty) {
            writeLights(device);
            lightsDirty = false;
          }
          const camera = frame.camera;
          camera.updateMatrixWorld();
          // The blend term's sun (env_sun's two float4) and map side.
          pixelParams ??= createUniformBlock(device);
          encoder.copyBufferToBuffer(sky.envSun, 0, pixelParams, 0, 32);
          device.queue.writeBuffer(
            pixelParams,
            32,
            new Uint32Array([sky.baseSide, 0, 0, 0]),
          );
          // Slots a mesh not relit here may now hold kept records of a mesh
          // that was: cleared when the accumulator's layout changes (onFrame
          // has asked for the splats to be generated again).
          const layout = layoutOf(r);
          const records = pixelBuffer(device, slotsOf(r));
          if (layout !== pixelLayout) {
            encoder.clearBuffer(records);
            pixelLayout = layout;
          }
          const pass = encoder.beginComputePass({ label: "athenea relight" });
          for (const mesh of meshes) relightMesh(pass, r, mesh, camera);
          pass.end();
        },
      },
    ],
    buffers(stage, ctx): Record<string, GPUBuffer> {
      if (!renderer) return {};
      const records = pixelBuffer(renderer.device, slotsOf(renderer));
      if (stage === "pixel") {
        emptyPool ??= upload(
          renderer.device,
          new Uint32Array([0, 0, 4, 0]),
          "relight pool",
        );
        pixelParams ??= createUniformBlock(renderer.device);
        return {
          atheneaPixel: records,
          atheneaEnvTexels: sky?.envTexels ?? emptyPool,
          atheneaPixelParams: pixelParams,
        };
      }
      if (!ctx.mesh) return {};
      return {
        atheneaRelit: stateOf(renderer.device, ctx.mesh).relit,
        atheneaPixelOut: records,
      };
    },
    attach(r) {
      renderer = r;
      const have = r.device.limits.maxStorageBuffersPerShaderStage;
      if (have < RELIGHT_STORAGE_BUFFERS) {
        console.warn(
          `atheneaRelight: the device has ${have} storage buffers a stage, the relight pass needs ${RELIGHT_STORAGE_BUFFERS} (make it with splatRequiredLimits)`,
        );
        plugin.enabled = false;
      }
      sky?.destroy();
      sky = null;
      lightsDirty = true;
      dirty = true;
    },
    detach() {
      sky?.destroy();
      sky = null;
      lightsBuffer?.destroy();
      lightsBuffer = null;
      envOfLight?.destroy();
      envOfLight = null;
      ies?.records.destroy();
      ies?.values.destroy();
      ies = null;
      emptyPool?.destroy();
      emptyPool = null;
      pixel?.destroy();
      pixel = null;
      pixelParams?.destroy();
      pixelParams = null;
      pixelLayout = "";
      renderer = null;
    },
    onFrame(frame) {
      // A new accumulator layout clears the draw's records (the pass), so
      // the relit meshes' must be written again.
      const was = dirty || layoutOf(frame.renderer) !== pixelLayout;
      dirty = false;
      return was ? "dirty" : "clean";
    },
    set(o) {
      const skyChanged =
        o.hdri !== undefined ||
        o.rotation !== undefined ||
        o.intensity !== undefined ||
        o.exposure !== undefined ||
        o.colour !== undefined;
      if (o.hdri !== undefined) options.hdri = o.hdri;
      if (o.rotation !== undefined) options.rotation = o.rotation;
      if (o.intensity !== undefined) options.intensity = o.intensity;
      if (o.exposure !== undefined) options.exposure = o.exposure;
      if (o.colour) options.colour = [...o.colour];
      if (o.noDome !== undefined) options.noDome = o.noDome;
      if (o.sun !== undefined) options.sun = o.sun;
      if (o.lights) options.lights = [...o.lights];
      if (o.indirect !== undefined) options.indirect = o.indirect;
      if (o.litBody !== undefined) options.litBody = o.litBody;
      if (o.ior !== undefined) options.ior = o.ior;
      if (o.emission !== undefined) options.emission = o.emission;
      if (o.pixelDetail !== undefined) options.pixelDetail = o.pixelDetail;
      if (o.frame !== undefined) options.frame = o.frame;
      if (skyChanged && sky) sky.dirty = true;
      lightsDirty = true;
      dirty = true;
    },
    setStoredLinear(asset, value) {
      linear.set(asset, value);
      dirty = true;
    },
    storedLinearOf: lookupLinear,
    setCatcher(asset, catcher) {
      catchers.set(asset, catcher);
      dirty = true;
    },
    setIor(asset, ior) {
      iors.set(asset, ior);
      dirty = true;
    },
    ui: [
      {
        id: "intensity",
        label: "Dome intensity",
        type: "number",
        min: 0,
        max: 8,
        step: 0.05,
        get: () => options.intensity,
        set: (v) => plugin.set({ intensity: Number(v) }),
      },
      {
        id: "rotation",
        label: "Dome rotation (degrees)",
        type: "number",
        min: -180,
        max: 180,
        step: 1,
        get: () => (options.rotation * 180) / Math.PI,
        set: (v) => plugin.set({ rotation: (Number(v) * Math.PI) / 180 }),
      },
      {
        id: "indirect",
        label: "Indirect and reflected field",
        type: "boolean",
        get: () => options.indirect,
        set: (v) => plugin.set({ indirect: Boolean(v) }),
      },
    ],
  };
  plugin.set(initial);
  return plugin;
}
