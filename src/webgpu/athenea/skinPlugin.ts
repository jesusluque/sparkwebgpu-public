// athenea's skinning on the PluginHost (plan-athenea.md, phase 5): a cloud a
// skeleton carries (athenea `mesh2splat --skinned`, converted by usd-athc to
// a .athc v3 with a SKIN section and its skeleton; docs/docs/athc-v3.md),
// posed every frame on the GPU before the relight and generate.
//
//   const skeleton = await readAthcSkeleton({ url });
//   const skin = atheneaSkinPlugin({ skeleton });           // plays clip 0
//   host.register(skin).register(atheneaRelightPlugin({ hdri }));
//   skin.set({ playing: false, time: 31 });                  // a time code
//
// Per frame, when the pose changed: the joints' transforms at the time code
// (USD's linear interpolation of the clip's samples, as athenea's Hydra
// reads `skinningXforms`), then slang/athenea_adapter/skin.slang -- athenea's
// splat_skin (txf 89a04d9) -- a splat each: the linear blend of its joints
// for the centre, the whole Jacobian of that blend (the weights' gradients,
// 7dff879) for the frame and the shading normal. The posed splats go back
// into the mesh's source in place (the rest copy is kept), and the turned
// normals into its attribute pool, so the relight (a zonal transfer turns
// with the frame: splatTransferFrame), generate, the sort and the draw all
// read the pose. The source's version is bumped, which regenerates and
// relights.
//
// The pool goes to the GPU whole (AttribPool.gpuBuffer, the layout pack()
// gives), so every reader binds the one the skinner writes. Paged pools
// (PagedSplats) are not carried: a paged skinned cloud stays in its bind
// pose.

import type { AthcSkeleton } from "../../athc";
import type { GpuSplatSource } from "../GpuSplatSource";
import type { WgpuSplatMesh, WgpuSplatRenderer } from "../WgpuSplatRenderer";
import { ATTRIB_NONE, type AttribPool } from "../attributes/schema";
import skinModule from "../generated/athenea_adapter/skin";
import { atheneaAdapterSkin } from "../generated/constants";
import { createStorage, upload } from "../gpuBuffers";
import type { PluginFrame, SplatPlugin } from "../plugins/types";
import { UniformWriter } from "../uniforms";

export const ATHENEA_SKIN_ID = "athenea.skin";
export const SKIN_INFLUENCES_ATTRIBUTE = "skinInfluences";
export const SKIN_GRADIENTS_ATTRIBUTE = "skinGradients";
const C = atheneaAdapterSkin;

export interface AtheneaSkinOptions {
  skeleton: AthcSkeleton;
  /** The clip played, by index or name (default 0). */
  clip?: number | string;
  /** Advance with the frame's time (default true). */
  playing?: boolean;
  /** Time codes a second over the clip's own rate (default 1). */
  speed?: number;
  /** The time code shown; while playing, where it starts (default the clip's first). */
  time?: number;
  /** Loop the clip (default true); else hold its last sample. */
  loop?: boolean;
  /**
   * The weights' gradients in the frame (athenea's whole Jacobian, 7dff879).
   * false: the blend of the joints' linear parts alone (default true).
   */
  jacobian?: boolean;
  /** Turn the shading normals (attribute normalOct) with the frame (default true). */
  normals?: boolean;
  /** Show the bind pose (the cloud as converted) instead of the clip. */
  bindPose?: boolean;
  /** Which meshes this skeleton carries (default every mesh with skinInfluences). */
  meshes?: (mesh: WgpuSplatMesh) => boolean;
}

export interface AtheneaSkinPlugin extends SplatPlugin {
  readonly options: AtheneaSkinOptions;
  set(options: Partial<AtheneaSkinOptions>): void;
  /** The time code posed last. */
  readonly timeCode: number;
  /** The clip's range in time codes, and its rate. */
  range(): { from: number; to: number; timeCodesPerSecond: number };
  readonly stats: { posed: number; splats: number };
}

/** The clip `clip` of a skeleton (index or name). */
export function skinClipIndex(
  skeleton: AthcSkeleton,
  clip: number | string = 0,
): number {
  const k =
    typeof clip === "number"
      ? clip
      : skeleton.clips.findIndex((c) => c.name === clip);
  if (k < 0 || k >= skeleton.clips.length) {
    throw new Error(`athenea skin: no clip ${clip}`);
  }
  return k;
}

/**
 * The joints' transforms of `clip` at time code `t` (athc_skin.rs
 * AthcSkeleton::pose): each element linearly between the two samples about
 * `t`, held outside them. joints x 16, USD rows.
 */
export function atheneaSkinPose(
  skeleton: AthcSkeleton,
  clip: number,
  t: number,
): Float32Array {
  const c = skeleton.clips[clip];
  const per = skeleton.joints.length * 16;
  const n = c.times.length;
  let k = 0;
  while (k < n && c.times[k] <= t) k++;
  if (k === 0) return c.xforms.slice(0, per);
  if (k >= n) return c.xforms.slice((n - 1) * per, n * per);
  const a = (t - c.times[k - 1]) / (c.times[k] - c.times[k - 1]);
  const out = new Float32Array(per);
  const x0 = (k - 1) * per;
  const x1 = k * per;
  for (let i = 0; i < per; i++) {
    const p = c.xforms[x0 + i];
    out[i] = p + (c.xforms[x1 + i] - p) * a;
  }
  return out;
}

/** USD rows (vectors on the left) to the kernel's rows (vectors on the right). */
export function transposeRows(m: ArrayLike<number>, count: number) {
  const out = new Float32Array(count * 16);
  for (let j = 0; j < count; j++) {
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        out[j * 16 + r * 4 + c] = m[j * 16 + c * 4 + r];
      }
    }
  }
  return out;
}

interface SkinState {
  rest: GPUBuffer;
  restNormals: GPUBuffer;
  xforms: GPUBuffer;
  pool: AttribPool;
  poolVersion: number;
  ids: { influences: number; gradients: number; normals: number };
  copied: boolean;
  posedKey: string;
}

export function atheneaSkinPlugin(
  initial: AtheneaSkinOptions,
): AtheneaSkinPlugin {
  const options: AtheneaSkinOptions = {
    playing: true,
    speed: 1,
    loop: true,
    jacobian: true,
    normals: true,
    ...initial,
  };
  const states = new WeakMap<GpuSplatSource, SkinState>();
  /** Pool buffers this plugin made; any other gpuBuffer is a pager's. */
  const ours = new WeakSet<GPUBuffer>();
  /** Sources it cannot carry (paged pools), warned once. */
  const refused = new WeakSet<GpuSplatSource>();
  const stats = { posed: 0, splats: 0 };
  let clip = skinClipIndex(options.skeleton, options.clip);
  let timeCode = options.time ?? options.skeleton.clips[clip].times[0];
  let startedAt: number | null = null;
  let startCode = timeCode;
  let rows: Float32Array | null = null;
  let poseVersion = 0;
  let rowsKey = "";

  const range = () => {
    const c = options.skeleton.clips[clip];
    return {
      from: c.times[0],
      to: c.times[c.times.length - 1],
      timeCodesPerSecond: c.timeCodesPerSecond,
    };
  };

  const carries = (r: WgpuSplatRenderer, mesh: WgpuSplatMesh) => {
    const pool = mesh.source.attribs;
    if (!pool || pool.id(SKIN_INFLUENCES_ATTRIBUTE) < 0) return false;
    if (refused.has(mesh.source)) return false;
    if (r.plugins && !r.plugins.isActive(ATHENEA_SKIN_ID, mesh)) return false;
    return options.meshes ? options.meshes(mesh) : true;
  };

  // The pool on the GPU, whole, as every reader binds it (gpuBuffer).
  const stateOf = (
    device: GPUDevice,
    mesh: WgpuSplatMesh,
  ): SkinState | null => {
    const source = mesh.source;
    const pool = source.attribs as AttribPool;
    let s = states.get(source);
    if (s && s.pool === pool && s.poolVersion === pool.version) return s;
    if (pool.gpuBuffer && !ours.has(pool.gpuBuffer)) {
      if (!refused.has(source)) {
        console.warn("athenea skin: a paged pool is not carried (bind pose)");
        refused.add(source);
      }
      return null;
    }
    const joints = options.skeleton.joints.length;
    const { words } = pool.pack();
    pool.gpuBuffer?.destroy();
    pool.gpuBuffer = upload(device, words, "skinned attribs");
    ours.add(pool.gpuBuffer);
    const normalCol = pool.column("normalOct");
    const id = (n: string) => {
      const k = pool.id(n);
      return k < 0 ? ATTRIB_NONE : k;
    };
    if (!s) {
      s = {
        rest: createStorage(device, source.src.size, "skin rest"),
        restNormals: normalCol
          ? upload(device, normalCol.words, "skin rest normals")
          : createStorage(device, 16, "skin rest normals"),
        xforms: createStorage(device, joints * 64, "skin xforms"),
        pool,
        poolVersion: pool.version,
        ids: { influences: 0, gradients: 0, normals: 0 },
        copied: false,
        posedKey: "",
      };
      states.set(source, s);
    } else if (normalCol) {
      s.restNormals.destroy();
      s.restNormals = upload(device, normalCol.words, "skin rest normals");
    }
    s.pool = pool;
    s.poolVersion = pool.version;
    s.ids = {
      influences: id(SKIN_INFLUENCES_ATTRIBUTE),
      gradients: id(SKIN_GRADIENTS_ATTRIBUTE),
      normals: id("normalOct"),
    };
    s.posedKey = "";
    return s;
  };

  const poseNow = (frame: PluginFrame) => {
    const { from, to, timeCodesPerSecond } = range();
    if (options.playing) {
      if (startedAt === null) {
        startedAt = frame.time;
        startCode = timeCode;
      }
      let t =
        startCode +
        (frame.time - startedAt) * timeCodesPerSecond * (options.speed ?? 1);
      const span = to - from;
      if (options.loop !== false && span > 0) {
        t = from + ((((t - from) % span) + span) % span);
      } else {
        t = Math.min(Math.max(t, from), to);
      }
      timeCode = t;
    }
    const key = `${clip}|${timeCode}`;
    if (key !== rowsKey) {
      rowsKey = key;
      rows = transposeRows(
        atheneaSkinPose(options.skeleton, clip, timeCode),
        options.skeleton.joints.length,
      );
      poseVersion += 1;
    }
  };

  const plugin: AtheneaSkinPlugin = {
    id: ATHENEA_SKIN_ID,
    minTier: 1,
    // Before the relight: it reads the posed frames and normals.
    before: ["athenea.relight"],
    requires: { attributes: [SKIN_INFLUENCES_ATTRIBUTE] },
    options,
    stats,
    get timeCode() {
      return timeCode;
    },
    range,
    set(next) {
      if (next.skeleton && next.skeleton !== options.skeleton) {
        throw new Error("athenea skin: one plugin, one skeleton");
      }
      Object.assign(options, next);
      if (next.clip !== undefined) {
        clip = skinClipIndex(options.skeleton, next.clip);
        rowsKey = "";
      }
      if (next.time !== undefined) timeCode = next.time;
      // Playing restarts from where it is.
      startedAt = null;
      poseVersion += 1;
    },
    onFrame(frame) {
      poseNow(frame);
      let dirty = false;
      for (const mesh of frame.renderer.meshes) {
        if (!carries(frame.renderer, mesh)) continue;
        const s = states.get(mesh.source);
        const key = `${poseVersion}|${options.jacobian}|${options.normals}|${options.bindPose}`;
        if (!s || s.posedKey !== key) {
          // The relight and generate see a new cloud.
          mesh.source.version += 1;
          dirty = true;
        }
      }
      return dirty ? "dirty" : undefined;
    },
    passes: [
      {
        name: "athenea skin",
        encode(encoder: GPUCommandEncoder, frame: PluginFrame) {
          const r = frame.renderer;
          const meshes = r.meshes.filter((m) => carries(r, m));
          if (!meshes.length || !rows) return;
          const { device } = r;
          const key = `${poseVersion}|${options.jacobian}|${options.normals}|${options.bindPose}`;
          const work: { mesh: WgpuSplatMesh; s: SkinState }[] = [];
          for (const mesh of meshes) {
            const s = stateOf(device, mesh);
            if (!s || s.posedKey === key) continue;
            if (!s.copied) {
              encoder.copyBufferToBuffer(
                mesh.source.src,
                0,
                s.rest,
                0,
                mesh.source.src.size,
              );
              s.copied = true;
            }
            device.queue.writeBuffer(s.xforms, 0, rows);
            work.push({ mesh, s });
          }
          if (!work.length) return;
          const pass = encoder.beginComputePass({ label: "athenea skin" });
          const kernel = r.registry.get(skinModule, "atheneaSkin");
          for (const { mesh, s } of work) {
            const source = mesh.source;
            const pool = s.pool;
            const params = UniformWriter.for(skinModule, "skinParams");
            const words = (n: string) => pool.column(n)?.spec.components ?? 0;
            const flags =
              (source.format === "ext" ? C.kSkinExt : 0) |
              (options.jacobian !== false ? C.kSkinJacobian : 0) |
              (options.bindPose ? C.kSkinRest : 0) |
              (options.normals !== false && s.ids.normals !== ATTRIB_NONE
                ? C.kSkinNormals
                : 0);
            params
              .set("count", source.count)
              .set("flags", flags)
              .set("perSplat", words(SKIN_INFLUENCES_ATTRIBUTE))
              .set("gradientWords", words(SKIN_GRADIENTS_ATTRIBUTE))
              .set("influences", s.ids.influences)
              .set("gradients", s.ids.gradients)
              .set("normals", s.ids.normals)
              .set("joints", options.skeleton.joints.length)
              .set("encoding", source.encoding);
            const bind = transposeRows(options.skeleton.geomBind, 1);
            for (let k = 0; k < 4; k++) {
              params.set(
                `geomBind${k}`,
                Array.from(bind.subarray(k * 4, k * 4 + 4)),
              );
            }
            const row = C.kSkinDispatchRow;
            kernel.dispatch(pass, {
              grid:
                source.count <= row
                  ? [source.count, 1]
                  : [row, Math.ceil(source.count / row)],
              buffers: {
                skinRest: s.rest,
                skinSrc: source.src,
                skinPool: pool.gpuBuffer as GPUBuffer,
                skinRestNormals: s.restNormals,
                skinXforms: s.xforms,
              },
              uniforms: params.data,
            });
            s.posedKey = key;
            stats.posed += 1;
            stats.splats += source.count;
          }
          pass.end();
        },
      },
    ],
  };
  return plugin;
}
