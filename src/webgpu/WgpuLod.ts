// LoD for WgpuSplatRenderer: SparkRenderer's driveLod, driveLodExclusive,
// updateLodInstances and updateLodIndices without the WebGL parts.
//
// The traversal is the same Rust/WASM code in a worker (traverseLodTrees);
// its result for each mesh becomes the mesh's index list
// (WgpuSplatRenderer.setLodIndices), which generate remaps through. LoD
// PackedSplats/ExtSplats draw their lodSplats; PagedSplats (.rad) draw the
// WgpuSplatPager pool, whose pages the traversal requests and the pager
// streams in.
//
// Usage:
//   const lod = new WgpuLod(splatRenderer);
//   const mesh = await lod.add(new PackedSplats({ url, lod: true }), object);
//   const paged = await lod.add(new PagedSplats({ rootUrl: "x-lod.rad" }), object2);
//   loop: lod.update(camera); renderer.render(...); splatRenderer.render(...);

import * as THREE from "three";
import { ExtSplats } from "../ExtSplats";
import { PackedSplats } from "../PackedSplats";
import { PagedSplats } from "../PagedSplats";
import { SplatWorker } from "../SplatWorker";
import { type AthcV3Layout, isAthcPaged } from "../athc";
import { isAndroid, isIos, isMobile, isOculus, isVisionPro } from "../utils";
import { GpuSplatSource } from "./GpuSplatSource";
import { LodFade } from "./LodFade";
import { WgpuBandPager } from "./WgpuBandPager";
import { WgpuSplatPager } from "./WgpuSplatPager";
import type { WgpuSplatMesh, WgpuSplatRenderer } from "./WgpuSplatRenderer";
import {
  type RadAttributeMeta,
  specsFromRadMeta,
} from "./attributes/PagedAttribPool";
import type { AttribPagingOptions } from "./attributes/attribPaging";
import { canvasContext } from "./threeInternals";

type LodSplats = PackedSplats | ExtSplats | PagedSplats;

export interface WgpuLodOptions {
  /** Target splat count; by default 500K-2.5M by platform, as SparkRenderer. */
  lodSplatCount?: number;
  /** Multiplies the target splat count. @default 1 */
  lodSplatScale?: number;
  /** Minimum splat size in pixels. @default 1 */
  lodRenderScale?: number;
  lodTraverseMode?: "dynamic" | "standard";
  /**
   * The LoD error cut (rust spark-lib athc_lod_cut.rs, athenea's
   * lod_tree_cut.slang): tau in pixels. A tree whose file carries the
   * bounds (`.athc` section LODE: usd-athc / athc-convert --lod-errors) is
   * cut where each node's projected error, eps ppu / max(d - rho, near),
   * is at most tau and its parent's is not; other trees keep the size cut
   * (lodRenderScale). 0: off, the size cut everywhere. @default 0
   */
  lodErrorTau?: number;
  /**
   * With the bounds' back-face cones (LODE): a node the cut draws whose
   * leaves all face away from the eye is left out, when it is coarse
   * (wider than lodBackCullMinPx on screen; a node the cut refines is never
   * culled, so full detail never is). Only closed, non-glass pieces carry a
   * cone (usd-athc: `athenea:splat:closed`). @default false
   */
  lodBackCull?: boolean;
  /** The narrowest node lodBackCull culls, in pixels. @default 1 */
  lodBackCullMinPx?: number;
  /**
   * athenea's occlusion masks (`.athc` section OCCL and codebook, `decimate
   * --occlusion-mask`): a node is left out with its subtree when the eye is
   * outside the asset's box and the four cells toward the eye on the 16 x
   * 16 octahedral grid are all closed in its entry. @default false
   */
  lodOcclusionCull?: boolean;
  /**
   * tau that breathes with a budget (thread CG, Nanite-style), with
   * lodErrorTau > 0: a detail scale s in [min, 1] divides tau (the cut
   * coarsens under pressure, never finer than lodErrorTau). With athenea's
   * band pages (WgpuBandPager) the pool's budget is met at once instead:
   * tau is at least the smallest whose needed pages fit the pool (from the
   * page table), and the scale answers the frame time alone. Every
   * intervalMs at most, s x 0.97 when the frame time (an EMA of update()'s
   * interval) passes frameMs or the page pool is fuller than `high`, s x 1.01
   * when the pool is under `low` and the frame time under 0.85 frameMs;
   * between the two it holds (hysteresis). true: the defaults.
   */
  lodErrorTauAuto?: boolean | LodTauAutoOptions;
  /** @default 0.2 */
  behindFoveate?: number;
  /** @default 90 */
  coneFov0?: number;
  /** @default 120 */
  coneFov?: number;
  /** @default 0.4 */
  coneFoveate?: number;
  /** Paged splats in the ext encoding. @default false */
  pagedExtSplats?: boolean;
  /** Page pool size, capped by the device's storage binding size. */
  maxPagedSplats?: number;
  /**
   * athenea's error-band pages (a paged .athc tree, FLAG_PAGED; thread CK):
   * splats a pool slot holds (at least the file's largest page). The pool
   * is `maxPagedSplats` (its residency budget, in nodes) in slots of this.
   * @default 3072
   */
  bandPageSplats?: number;
  /** @default 3 */
  numLodFetchers?: number;
  /**
   * Paged attributes: the pool's byte budget, pages per stream group,
   * which streams, the transfer form (attributes/attribPaging.ts). By
   * default the device tier's budget and the PluginHost's streams.
   */
  pagedAttributes?: AttribPagingOptions;
  /** @default true */
  enableLodFetching?: boolean;
  /**
   * A tree no mesh draws is released this long after it was last drawn,
   * as SparkRenderer.lodCleanupTimeoutMs. @default 3000
   */
  lodCleanupTimeoutMs?: number;
  /**
   * Splats in each raycast selection (WgpuLodMesh.raycastIndices), a second
   * coarser traversal, as SparkRenderer.lodRaycast; 0 for none (raycasts
   * then use the drawn selection). Default 10K on mobile, else 25K.
   */
  lodRaycast?: number;
  /** Minimum time between raycast traversals. @default 500 */
  lodRaycastIntervalMs?: number;
  /** Called when LoD state changed and a new frame should be rendered. */
  onDirty?: () => void;
  /**
   * LoD transitions fade instead of popping (LodFade): splats entering the
   * cut fade in over this many ms while those leaving it stay, then fade
   * out over as long. Packed/ext LoD splats only (a paged pool may reuse
   * the slots of splats fading out). 0: off, a hard swap as Spark does.
   * @default 0
   */
  lodFadeMs?: number;
}

export interface LodTauAutoOptions {
  /** Frame budget, ms. @default 1000 / 50 */
  frameMs?: number;
  /** Pool fill (pages in use over the pool's) above which it coarsens. @default 0.85 */
  high?: number;
  /** ... and under which it may refine. @default 0.7 */
  low?: number;
  /** Least time between two steps, ms. @default 250 */
  intervalMs?: number;
  /** The coarsest detail scale. @default 0.3 */
  min?: number;
}

/** The breathing tau's controller (lodErrorTauAuto), alone for tests. */
export class LodTauController {
  /** The detail scale: tau is divided by it. */
  scale = 1;
  frameEma = 0;
  private lastFrame = 0;
  private lastStep = Number.NEGATIVE_INFINITY;

  /** A frame at `now` with the pool `fill` (0..1, or null without a pool). */
  frame(now: number, fill: number | null, o: LodTauAutoOptions = {}): boolean {
    const frameMs = o.frameMs ?? 1000 / 50;
    if (this.lastFrame > 0) {
      const dt = Math.min(now - this.lastFrame, 1000);
      this.frameEma = this.frameEma > 0 ? 0.9 * this.frameEma + 0.1 * dt : dt;
    }
    this.lastFrame = now;
    if (now - this.lastStep < (o.intervalMs ?? 250) || this.frameEma === 0) {
      return false;
    }
    const f = fill ?? 0;
    const over = f > (o.high ?? 0.85) || this.frameEma > frameMs;
    const under = f < (o.low ?? 0.7) && this.frameEma < 0.85 * frameMs;
    const before = this.scale;
    if (over) this.scale *= 0.97;
    else if (under) this.scale *= 1.01;
    this.scale = Math.min(1, Math.max(o.min ?? 0.3, this.scale));
    if (this.scale === before) return false;
    this.lastStep = now;
    return true;
  }
}

export interface WgpuLodMeshOptions {
  /** Detail multiplier for this mesh. @default 1 */
  lodScale?: number;
  behindFoveate?: number;
  coneFov0?: number;
  coneFov?: number;
  coneFoveate?: number;
}

export interface WgpuLodMesh extends WgpuLodMeshOptions {
  id: string;
  splats: LodSplats;
  object: THREE.Object3D;
  mesh: WgpuSplatMesh;
  /**
   * False: draw the full-detail splats instead, as SplatMesh.enableLod
   * (nothing when they weren't kept: load with nonLod: true, or paged).
   */
  enableLod: boolean;
  /** Source of the full-detail splats, built when LoD is first disabled. */
  baseSource?: GpuSplatSource;
  lodSource: GpuSplatSource;
  lastIndices: Uint32Array;
  /** False while hidden (setVisible): kept, with its tree, but not drawn. */
  visible: boolean;
  /**
   * The splats to raycast, as SplatMesh.raycastIndices: indices into the
   * LoD splats (paged: the page pool) of the last raycast traversal.
   */
  raycastIndices?: { numSplats: number; indices: Uint32Array };
  /** The LoD fade between cuts (WgpuLodOptions.lodFadeMs). */
  fade?: LodFade;
}

interface TreeRecord {
  lodId: number;
  rootPage?: number;
  /** When a mesh last drew it (performance.now()). */
  lastTouched: number;
}

const NO_SPLATS = new Uint32Array(0);

// A mesh as the worker's traverseLodTrees takes it.
interface LodInstance {
  instanceId: string;
  lodId: number;
  rootPage?: number;
  viewToObjectCols: number[];
  lodScale: number;
  behindFoveate: number;
  coneFov0: number;
  coneFov: number;
  coneFoveate: number;
}

let nextId = 0;

export class WgpuLod {
  readonly options: Required<
    Omit<
      WgpuLodOptions,
      | "lodSplatCount"
      | "onDirty"
      | "lodRaycast"
      | "pagedAttributes"
      | "bandPageSplats"
    >
  > &
    Pick<
      WgpuLodOptions,
      | "lodSplatCount"
      | "onDirty"
      | "lodRaycast"
      | "pagedAttributes"
      | "bandPageSplats"
    >;
  readonly meshes: WgpuLodMesh[] = [];
  pager?: WgpuSplatPager;
  /** Time of the last traversal in ms, and the splats it selected. */
  readonly stats = {
    traverseMs: 0,
    lodSplats: 0,
    traversals: 0,
    culled: 0,
    /** Nodes the occlusion masks hid in the last traversal (with subtrees). */
    occluded: 0,
    /** The tau the last update used (lodErrorTau over the auto scale). */
    tau: 0,
  };
  /** lodErrorTauAuto's controller. */
  readonly tauController = new LodTauController();

  private worker: SplatWorker | null = null;
  private pagerId: number | undefined;
  private trees = new Map<LodSplats, TreeRecord>();
  private lodIdToSplats = new Map<number, LodSplats>();
  private initQueue: LodSplats[] = [];
  private disposeQueue: number[] = [];
  private lodUpdates: {
    lodId: number;
    pageBase: number;
    chunkBase: number;
    count: number;
    lodTreeData?: Uint32Array;
    lodErrorData?: Uint32Array;
    lodOcclusionData?: Uint32Array;
  }[] = [];
  private dirty = true;
  private version = 0;
  private lastVersion = -1;
  private lastLod?: {
    pos: THREE.Vector3;
    quat: THREE.Quaternion;
    pixelScaleLimit: number;
    errorLimit: number;
    backCull: number;
    occlusion: boolean;
    maxSplats: number;
  };
  private lastPixelLimit?: number;
  /** The error cut's limit of the last traversal (band paging tests by it). */
  private lastErrorLimit = 0;
  /** A pixel in the traversal's angle units (tau 1). */
  private pixelAngle = 0;
  private lastRaycastTime = Number.NEGATIVE_INFINITY;
  /** lodBackCull's narrowest node, in the limit's angle units (-1: off). */
  private backCullMin = -1;

  constructor(
    readonly renderer: WgpuSplatRenderer,
    options: WgpuLodOptions = {},
  ) {
    const defaultPages = isMobile() ? (isIos() ? 96 : 128) : 256;
    this.options = {
      lodSplatScale: 1,
      lodRenderScale: 1,
      lodTraverseMode: "standard",
      lodErrorTau: 0,
      lodBackCull: false,
      lodBackCullMinPx: 1,
      lodOcclusionCull: false,
      lodErrorTauAuto: false,
      behindFoveate: 0.2,
      coneFov0: 90,
      coneFov: 120,
      coneFoveate: 0.4,
      pagedExtSplats: false,
      maxPagedSplats: defaultPages * 65536,
      numLodFetchers: 3,
      enableLodFetching: true,
      lodCleanupTimeoutMs: 3000,
      lodRaycastIntervalMs: 500,
      lodFadeMs: 0,
      ...options,
    };
  }

  /**
   * Adds a mesh. PackedSplats/ExtSplats need LoD splats (load with
   * `lod: true` or call createLodSplats); without them they draw in full.
   */
  async add(
    splats: LodSplats,
    object: THREE.Object3D = new THREE.Object3D(),
    options: WgpuLodMeshOptions = {},
  ): Promise<WgpuLodMesh> {
    const { device } = this.renderer;
    let source: GpuSplatSource;
    if (splats instanceof PagedSplats) {
      const { meta } = await splats.getRadMeta();
      // athenea's band pages: a pool of small slots of their own.
      let band: AthcV3Layout | null = null;
      if (splats.athcLayoutPromise) {
        const { layout } = await splats.athcLayoutPromise;
        if (isAthcPaged(layout)) band = layout as AthcV3Layout;
      }
      splats.pager = this.ensurePager(band !== null);
      if (band) (splats.pager as WgpuBandPager).addTree(splats, band);
      // The .rad's extra attributes, paged with its chunks.
      const attributes = (meta as { attributes?: RadAttributeMeta[] })
        .attributes;
      if (attributes) {
        this.ensurePager().setAttribSchema(
          specsFromRadMeta(attributes),
          meta.chunks?.length,
        );
      }
      source = this.ensurePager().source(splats);
    } else {
      await splats.initialized;
      const lodSplats = splats.lodSplats;
      if (lodSplats instanceof PackedSplats) {
        source = GpuSplatSource.fromPackedSplats(device, lodSplats);
      } else if (lodSplats instanceof ExtSplats) {
        source = GpuSplatSource.fromExtSplats(device, lodSplats);
      } else if (splats instanceof PackedSplats) {
        source = GpuSplatSource.fromPackedSplats(device, splats);
      } else {
        source = GpuSplatSource.fromExtSplats(device, splats);
      }
    }
    const mesh = this.renderer.add(source, object);
    const lodMesh: WgpuLodMesh = {
      ...options,
      id: `wgpu-lod-${nextId++}`,
      splats,
      object,
      mesh,
      enableLod: true,
      lodSource: source,
      lastIndices: NO_SPLATS,
      visible: true,
    };
    // Draw nothing until the first traversal picks the splats.
    if (this.lodSplatsOf(lodMesh)) {
      this.renderer.setLodIndices(mesh, new Uint32Array(0));
    }
    this.meshes.push(lodMesh);
    this.version += 1;
    return lodMesh;
  }

  remove(lodMesh: WgpuLodMesh) {
    const i = this.meshes.indexOf(lodMesh);
    if (i < 0) return;
    this.meshes.splice(i, 1);
    this.renderer.remove(lodMesh.mesh);
    if (!(lodMesh.splats instanceof PagedSplats)) {
      lodMesh.lodSource.destroy();
      lodMesh.baseSource?.destroy();
    }
    // Its tree stays for another mesh of the same splats until the timed
    // release in update(), as on WebGL.
    this.version += 1;
  }

  /** Whether update() still has work: meshes, or trees to release. */
  get active(): boolean {
    return (
      this.meshes.length > 0 ||
      this.trees.size > 0 ||
      this.disposeQueue.length > 0
    );
  }

  /**
   * Hides a mesh without removing it: a SplatMesh hidden and shown again
   * keeps its sources, last selection and (until the timed release) tree.
   */
  setVisible(lodMesh: WgpuLodMesh, visible: boolean) {
    if (lodMesh.visible === visible) return;
    lodMesh.visible = visible;
    this.applySelection(lodMesh);
    this.version += 1;
  }

  /**
   * The indices into `splats`' LoD splats of one level of its LoD tree, as
   * SparkRenderer.getLodTreeLevel; null when no mesh has built its tree.
   */
  async getLodTreeLevel(
    splats: LodSplats,
    level: number,
  ): Promise<Uint32Array | null> {
    return this.ensureWorker().exclusive(async (worker) => {
      const record = this.trees.get(splats);
      if (!record) return null;
      const { indices } = await worker.call("getLodTreeLevel", {
        lodId: record.lodId,
        level,
      });
      return indices;
    });
  }

  /** Switches a mesh between its LoD selection and all of its lodSplats. */
  setEnableLod(lodMesh: WgpuLodMesh, enable: boolean) {
    if (lodMesh.enableLod === enable) return;
    lodMesh.enableLod = enable;
    // Without LoD splats the mesh draws in full either way.
    if (!this.lodSplatsOf(lodMesh)) return;
    this.applySelection(lodMesh);
    this.version += 1;
  }

  // Points the mesh at what it draws now: nothing while hidden, its LoD
  // selection, or with LoD off its full-detail splats.
  private applySelection(lodMesh: WgpuLodMesh) {
    const { splats, mesh } = lodMesh;
    lodMesh.fade?.reset();
    if (!lodMesh.visible) {
      this.renderer.setLodIndices(mesh, NO_SPLATS);
    } else if (!this.lodSplatsOf(lodMesh)) {
      this.renderer.setLodIndices(mesh, null);
    } else if (lodMesh.enableLod) {
      mesh.source = lodMesh.lodSource;
      this.renderer.setLodIndices(mesh, lodMesh.lastIndices);
    } else if (!(splats instanceof PagedSplats) && splats.numSplats > 0) {
      lodMesh.baseSource ??=
        splats instanceof PackedSplats
          ? GpuSplatSource.fromPackedSplats(this.renderer.device, splats)
          : GpuSplatSource.fromExtSplats(this.renderer.device, splats);
      mesh.source = lodMesh.baseSource;
      this.renderer.setLodIndices(mesh, null);
    } else {
      this.renderer.setLodIndices(mesh, NO_SPLATS);
    }
  }

  private ensurePager(band = this.pager instanceof WgpuBandPager) {
    if (this.pager && band !== this.pager instanceof WgpuBandPager) {
      throw new Error(
        "WgpuLod: athenea's band-paged trees and other paged splats in one WgpuLod",
      );
    }
    if (!this.pager) {
      const Pager = band ? WgpuBandPager : WgpuSplatPager;
      this.pager = new Pager(this.renderer.device, {
        ...(band && this.options.bandPageSplats
          ? { pageSplats: this.options.bandPageSplats }
          : {}),
        extSplats: this.options.pagedExtSplats,
        maxSplats: this.options.maxPagedSplats,
        numFetchers: this.options.numLodFetchers,
        onUpdate: () => this.setDirty(),
        attributes: this.options.pagedAttributes,
        attributeDemand: () =>
          this.renderer.plugins?.plugins.length
            ? this.renderer.plugins.attributeDemand()
            : null,
      });
    }
    return this.pager;
  }

  // Page uploads write into existing pool buffers, which a renderer that
  // skips unchanged frames can't see.
  private markRendererDirty() {
    this.renderer.markDirty();
  }

  private setDirty() {
    this.options.onDirty?.();
  }

  private lodSplatsOf(m: WgpuLodMesh): LodSplats | undefined {
    if (m.splats instanceof PagedSplats) return m.splats;
    return m.splats.lodSplats ?? undefined;
  }

  defaultSplatTarget() {
    return isOculus()
      ? 500000
      : isVisionPro()
        ? 750000
        : isAndroid()
          ? 1000000
          : isIos()
            ? 1500000
            : 2500000;
  }

  /**
   * LoD or paging work under way or waiting for the next update(): a
   * traversal or worker call in flight, a mesh added or removed, trees to
   * build, tree ranges to send, pages being fetched or waiting to be
   * uploaded, a cut fading in. Its answer arrives through onDirty (the
   * SparkRenderer's dirty flag) and the next render; a caller that skips
   * rendering a still view keeps rendering while this holds.
   */
  get pending(): boolean {
    return (
      this.dirty ||
      this.version !== this.lastVersion ||
      this.initQueue.length > 0 ||
      this.disposeQueue.length > 0 ||
      this.lodUpdates.length > 0 ||
      this.worker?.queue != null ||
      (this.pager?.isPending() ?? false) ||
      this.meshes.some((m) => m.fade?.active ?? false)
    );
  }

  /**
   * Starts a LoD update for this camera if the worker is free: call once per
   * frame before WgpuSplatRenderer.render. `renderSize` is the drawing size
   * in pixels (default: the canvas).
   */
  update(camera: THREE.Camera, renderSize?: { x: number; y: number }) {
    const o = this.options;
    const size = renderSize ?? this.canvasSize();
    const maxSplats =
      (o.lodSplatCount ?? this.defaultSplatTarget()) * o.lodSplatScale;

    let pixelScaleLimit = 0;
    if (camera instanceof THREE.PerspectiveCamera) {
      const tanYfov = Math.tan((0.5 * camera.fov * Math.PI) / 180);
      pixelScaleLimit = (2.0 * tanYfov) / size.y;
    } else if (camera instanceof THREE.OrthographicCamera) {
      const viewHeight = (camera.top - camera.bottom) / camera.zoom;
      const viewWidth = (camera.right - camera.left) / camera.zoom;
      pixelScaleLimit = Math.min(
        viewWidth / Math.max(1, size.x),
        viewHeight / Math.max(1, size.y),
      );
    }
    // The error cut's tau in the same angular units (a pixel's size),
    // breathing with the budget when asked.
    let tau = o.lodErrorTau;
    if (tau > 0 && o.lodErrorTauAuto) {
      const pager = this.pager;
      // A band pager fits its budget by the page table at once (tauToFit);
      // the controller then answers the frame time alone.
      const band = pager instanceof WgpuBandPager;
      const fill = pager && !band ? pager.pagesInUse / pager.maxPages : null;
      this.tauController.frame(
        performance.now(),
        fill,
        o.lodErrorTauAuto === true ? {} : o.lodErrorTauAuto,
      );
      tau /= this.tauController.scale;
      if (band) tau = Math.max(tau, pager.tauToFit());
    }
    this.stats.tau = tau;
    const errorLimit = tau > 0 ? pixelScaleLimit * tau : 0;
    // The narrowest node the back-face cull may drop (-1: off).
    this.backCullMin = o.lodBackCull
      ? pixelScaleLimit * Math.max(0, o.lodBackCullMinPx)
      : -1;
    this.pixelAngle = pixelScaleLimit;
    pixelScaleLimit *= o.lodRenderScale;

    camera.updateMatrixWorld();
    const viewPos = new THREE.Vector3();
    const viewQuat = new THREE.Quaternion();
    camera.matrixWorld.decompose(viewPos, viewQuat, new THREE.Vector3());

    if (this.lastLod) {
      if (
        this.lastLod.pixelScaleLimit !== pixelScaleLimit ||
        this.lastLod.errorLimit !== errorLimit ||
        this.lastLod.backCull !== this.backCullMin ||
        this.lastLod.occlusion !== o.lodOcclusionCull ||
        this.lastLod.maxSplats !== maxSplats
      ) {
        this.dirty = true;
      }
      // Same thresholds as SparkRenderer.driveLod.
      const distanceRamp = Math.max(
        0,
        1 - viewPos.distanceTo(this.lastLod.pos) / 1.0,
      );
      const quatRamp = Math.max(
        0,
        1 - (1 - viewQuat.dot(this.lastLod.quat)) / 0.01,
      );
      if (distanceRamp * quatRamp < 0.999) this.dirty = true;
    }
    if (this.version !== this.lastVersion) {
      this.dirty = true;
      this.lastVersion = this.version;
    }

    const lodMeshes = this.meshes.filter(
      (m) => m.visible && m.enableLod && this.lodSplatsOf(m),
    );
    const now = performance.now();
    this.stepFades(lodMeshes, now);
    for (const m of lodMeshes) {
      m.object.updateMatrixWorld();
      const splats = this.lodSplatsOf(m) as LodSplats;
      const record = this.trees.get(splats);
      if (record) {
        record.lastTouched = now;
      } else if (!this.initQueue.includes(splats)) {
        this.initQueue.push(splats);
      }
      // The pool's SH buffer and the encoding appear with the first pages.
      if (m.splats instanceof PagedSplats && this.pager) {
        const source = this.pager.source(m.splats);
        if (
          source.numSh !== m.lodSource.numSh ||
          source.sh !== m.lodSource.sh
        ) {
          this.markRendererDirty();
        }
        m.mesh.source = m.lodSource = source;
      }
    }

    this.ensureWorker().tryExclusive(async (worker) => {
      try {
        await this.updateExclusive(worker, {
          lodMeshes,
          viewPos,
          viewQuat,
          pixelScaleLimit,
          errorLimit,
          maxSplats,
        });
      } finally {
        if (
          this.dirty ||
          this.initQueue.length > 0 ||
          this.pager?.hasQueued()
        ) {
          this.setDirty();
        }
      }
    });
  }

  /** Whether `m`'s cuts fade (lodFadeMs, and not a paged pool). */
  private fades(m: WgpuLodMesh): boolean {
    return this.options.lodFadeMs > 0 && !(m.splats instanceof PagedSplats);
  }

  /** Draws `m`'s new cut, through its fade when cuts fade. */
  private drawCut(m: WgpuLodMesh) {
    if (!this.fades(m)) {
      m.fade = undefined;
      this.renderer.setLodIndices(m.mesh, m.lastIndices);
      return;
    }
    m.fade ??= new LodFade(this.options.lodFadeMs);
    m.fade.durationMs = this.options.lodFadeMs;
    const { indices, faded } = m.fade.setCut(m.lastIndices, performance.now());
    this.renderer.setLodIndices(m.mesh, indices, faded);
  }

  /** A frame of every fade in progress. */
  private stepFades(lodMeshes: WgpuLodMesh[], now: number) {
    let stepped = false;
    for (const m of lodMeshes) {
      if (!m.fade) continue;
      if (!this.fades(m)) {
        m.fade = undefined;
        this.renderer.setLodIndices(m.mesh, m.lastIndices);
        stepped = true;
        continue;
      }
      m.fade.durationMs = this.options.lodFadeMs;
      const frame = m.fade.step(now);
      if (!frame) continue;
      this.renderer.setLodIndices(m.mesh, frame.indices, frame.faded);
      stepped = true;
    }
    if (stepped) {
      this.markRendererDirty();
      this.setDirty();
    }
  }

  private canvasSize() {
    const canvas = canvasContext(this.renderer.renderer).canvas as {
      width: number;
      height: number;
    };
    return { x: canvas.width, y: canvas.height };
  }

  private ensureWorker() {
    if (!this.worker) this.worker = new SplatWorker();
    return this.worker;
  }

  private async updateExclusive(
    worker: SplatWorker,
    {
      lodMeshes,
      viewPos,
      viewQuat,
      pixelScaleLimit,
      errorLimit,
      maxSplats,
    }: {
      lodMeshes: WgpuLodMesh[];
      viewPos: THREE.Vector3;
      viewQuat: THREE.Quaternion;
      pixelScaleLimit: number;
      errorLimit: number;
      maxSplats: number;
    },
  ) {
    this.releaseTrees(lodMeshes);
    while (this.disposeQueue.length > 0) {
      const lodId = this.disposeQueue.shift() as number;
      await worker.call("disposeLodTree", { lodId });
    }

    if (this.pager && this.pagerId === undefined) {
      const { lodId } = await worker.call("newLodTree", {
        capacity: this.pager.maxSplats,
        slot: this.pager.pageSplats,
      });
      this.pagerId = lodId;
    }

    while (this.initQueue.length > 0) {
      const splats = this.initQueue.shift() as LodSplats;
      if (this.trees.has(splats)) continue;
      let lodId: number;
      if (splats instanceof PagedSplats) {
        ({ lodId } = await worker.call("newSharedLodTree", {
          lodId: this.pagerId as number,
        }));
      } else {
        ({ lodId } = await worker.call("initLodTree", {
          numSplats: splats.numSplats ?? 0,
          lodTree: (splats.extra.lodTree as Uint32Array).slice(),
          lodError: (splats.extra.lodError as Uint32Array | undefined)?.slice(),
          lodOcclusion: (
            splats.extra.lodOcclusion as Uint32Array | undefined
          )?.slice(),
        }));
        const codebook = splats.extra.lodOcclusionCodebook as
          | Uint32Array
          | undefined;
        const bounds = splats.extra.lodOcclusionBounds as
          | Float32Array
          | undefined;
        if (codebook && bounds) {
          await worker.call("setLodOcclusionCodebook", {
            lodId,
            codebook: codebook.slice(),
            bounds: bounds.slice(),
          });
        }
      }
      if (splats instanceof PagedSplats && splats.athcLayoutPromise) {
        // A paged .athc v3's codebook is in its tables.
        const { layout } = await splats.athcLayoutPromise;
        const v3 = layout as {
          occlusionCodebook?: number[];
          header: { boundsMin: number[]; boundsMax: number[] };
        };
        if (v3.occlusionCodebook?.length) {
          await worker.call("setLodOcclusionCodebook", {
            lodId,
            codebook: new Uint32Array(v3.occlusionCodebook),
            bounds: new Float32Array([
              ...v3.header.boundsMin,
              ...v3.header.boundsMax,
            ]),
          });
        }
      }
      this.trees.set(splats, { lodId, lastTouched: performance.now() });
      this.lodIdToSplats.set(lodId, splats);
      this.dirty = true;
    }

    if (this.pager) {
      for (const update of this.pager.consumeLodTreeUpdates()) {
        const {
          splats,
          page,
          chunk,
          numSplats,
          lodTree,
          lodError,
          lodOcclusion,
        } = update;
        const record = this.trees.get(splats);
        if (!record) continue;
        if (lodTree && chunk === 0) {
          record.rootPage = page;
        } else if (!lodTree && chunk === 0 && record.rootPage === page) {
          // Root evicted: skip the mesh until it is refetched, and stop
          // drawing indices into pages that no longer hold its data.
          record.rootPage = undefined;
          for (const m of this.meshes) {
            if (m.splats === splats) {
              m.lastIndices = NO_SPLATS;
              m.raycastIndices = undefined;
              this.renderer.setLodIndices(m.mesh, NO_SPLATS);
            }
          }
        }
        this.lodUpdates.push({
          lodId: record.lodId,
          pageBase: page * this.pager.pageSplats,
          // The virtual space keeps 65 536 indices a chunk.
          chunkBase: chunk * 65536,
          count: numSplats,
          lodTreeData: lodTree,
          lodErrorData: lodError,
          lodOcclusionData: lodOcclusion,
        });
      }
    }
    if (this.lodUpdates.length > 0) {
      const ranges = this.lodUpdates;
      this.lodUpdates = [];
      await worker.call("updateLodTrees", { ranges });
      this.dirty = true;
    }

    if (!this.dirty) return;
    this.dirty = false;
    this.lastLod = {
      pos: viewPos,
      quat: viewQuat,
      pixelScaleLimit,
      errorLimit,
      backCull: this.backCullMin,
      occlusion: this.options.lodOcclusionCull,
      maxSplats,
    };
    await this.traverse(
      worker,
      lodMeshes,
      maxSplats,
      viewPos,
      viewQuat,
      pixelScaleLimit,
      errorLimit,
    );
    this.setDirty();
  }

  // SparkRenderer.cleanupLodTrees: releases the trees that no mesh drawn now
  // uses and none drew for lodCleanupTimeoutMs. Bookkeeping only: update()
  // may run between the worker calls, so the disposals are queued.
  private releaseTrees(lodMeshes: WgpuLodMesh[]) {
    const drawn = new Set(lodMeshes.map((m) => this.lodSplatsOf(m)));
    const expiry = performance.now() - this.options.lodCleanupTimeoutMs;
    for (const [splats, record] of [...this.trees]) {
      if (drawn.has(splats) || record.lastTouched > expiry) continue;
      this.trees.delete(splats);
      this.lodIdToSplats.delete(record.lodId);
      this.disposeQueue.push(record.lodId);
      if (!(splats instanceof PagedSplats)) continue;
      // Its pages go back to the pool: drop selections that point there.
      this.pager?.removeSplats(splats);
      for (const m of this.meshes) {
        if (m.splats !== splats) continue;
        m.lastIndices = NO_SPLATS;
        m.raycastIndices = undefined;
        this.applySelection(m);
      }
    }
  }

  private async traverse(
    worker: SplatWorker,
    lodMeshes: WgpuLodMesh[],
    maxSplats: number,
    viewPos: THREE.Vector3,
    viewQuat: THREE.Quaternion,
    pixelScaleLimit: number,
    errorLimit = 0,
  ) {
    const o = this.options;
    this.lastErrorLimit = errorLimit;
    const cameraToWorld = new THREE.Matrix4().compose(
      viewPos,
      viewQuat,
      new THREE.Vector3(1, 1, 1),
    );
    const byId = new Map<string, WgpuLodMesh>();
    const instances: Record<string, LodInstance> = {};
    for (const m of lodMeshes) {
      const record = this.trees.get(this.lodSplatsOf(m) as LodSplats);
      if (!record) continue;
      const paged = m.splats instanceof PagedSplats;
      if (paged && record.rootPage === undefined) continue;
      byId.set(m.id, m);
      const viewToObject = m.object.matrixWorld
        .clone()
        .invert()
        .multiply(cameraToWorld);
      instances[m.id] = {
        instanceId: m.id,
        lodId: record.lodId,
        rootPage: record.rootPage,
        viewToObjectCols: viewToObject.elements,
        lodScale: m.lodScale ?? 1,
        behindFoveate: m.behindFoveate ?? o.behindFoveate,
        coneFov0: m.coneFov0 ?? o.coneFov0,
        coneFov: m.coneFov ?? o.coneFov,
        coneFoveate: m.coneFoveate ?? o.coneFoveate,
      };
    }

    const t0 = performance.now();
    const { keyIndices, chunks, pixelLimit, culledCount, occludedCount } =
      await worker.call("traverseLodTrees", {
        maxSplats,
        pixelScaleLimit,
        lastPixelLimit: this.lastPixelLimit,
        instances,
        traverseMode: o.lodTraverseMode,
        errorLimit: errorLimit > 0 ? errorLimit : undefined,
        backCullMin: this.backCullMin >= 0 ? this.backCullMin : undefined,
        occlusionCull: o.lodOcclusionCull,
      });
    this.stats.culled = culledCount ?? 0;
    this.stats.occluded = occludedCount ?? 0;
    this.stats.traverseMs = performance.now() - t0;
    this.stats.traversals += 1;
    this.lastPixelLimit = pixelLimit;

    let total = 0;
    for (const [id, { numSplats, indices }] of Object.entries(keyIndices)) {
      const m = byId.get(id);
      // Removed, hidden or switched off while the traversal ran.
      if (!m || !this.meshes.includes(m) || !m.visible || !m.enableLod) {
        continue;
      }
      m.lastIndices = indices.subarray(0, numSplats);
      this.drawCut(m);
      // PagedSplats.update's count, for stats (no texture here).
      if (m.splats instanceof PagedSplats) {
        m.splats.numSplats = numSplats;
        m.splats.dynoNumSplats.value = numSplats;
      }
      total += numSplats;
    }
    this.stats.lodSplats = total;

    const pager = this.pager;
    if (pager) this.fetchPages(pager, lodMeshes, chunks, viewPos);
    await this.raycastTraverse(
      worker,
      instances,
      byId,
      total,
      pixelScaleLimit,
      errorLimit,
    );
  }

  private fetchPages(
    pager: WgpuSplatPager,
    lodMeshes: WgpuLodMesh[],
    chunks: [number, number][],
    viewPos: THREE.Vector3,
  ) {
    // Pages land in the pool in the same task as the indices that use them.
    pager.processUploads();
    this.markRendererDirty();

    const paged = lodMeshes
      .filter((m) => m.splats instanceof PagedSplats)
      .map((m) => ({
        splats: m.splats as PagedSplats,
        distance: m.object
          .getWorldPosition(new THREE.Vector3())
          .distanceTo(viewPos),
      }));
    // Root chunks first, nearest mesh first; then the traversal's requests.
    paged.sort((a, b) => a.distance - b.distance);
    if (pager instanceof WgpuBandPager) {
      // athenea's band pages: the needed set from each tree's page table
      // (the eye in its object space, the error limit over its lodScale).
      // Without the error cut, tau 1.
      const limit =
        this.lastErrorLimit > 0 ? this.lastErrorLimit : this.pixelAngle;
      const views = [];
      for (const m of lodMeshes) {
        if (!(m.splats instanceof PagedSplats) || !pager.isBand(m.splats))
          continue;
        const eye = viewPos
          .clone()
          .applyMatrix4(m.object.matrixWorld.clone().invert());
        views.push({
          splats: m.splats,
          eye: [eye.x, eye.y, eye.z],
          limit: limit / (m.lodScale ?? 1),
          unit: this.pixelAngle / (m.lodScale ?? 1),
        });
      }
      pager.setViews(views);
      pager.autoDrive = this.options.enableLodFetching;
      if (this.options.enableLodFetching) pager.driveFetchers();
      return;
    }
    pager.fetchPriority = paged.map(({ splats }) => ({ splats, chunk: 0 }));
    for (const [lodId, chunk] of chunks) {
      const splats = this.lodIdToSplats.get(lodId);
      if (splats instanceof PagedSplats && chunk !== 0) {
        pager.fetchPriority.push({ splats, chunk });
      }
    }
    const o = this.options;
    pager.autoDrive = o.enableLodFetching;
    if (o.enableLodFetching) pager.driveFetchers();
  }

  // SparkRenderer's raycast selection: a coarser cut, at most every
  // lodRaycastIntervalMs.
  private async raycastTraverse(
    worker: SplatWorker,
    instances: Record<string, LodInstance>,
    byId: Map<string, WgpuLodMesh>,
    total: number,
    pixelScaleLimit: number,
    errorLimit = 0,
  ) {
    const o = this.options;
    const lodRaycast = o.lodRaycast ?? (isMobile() ? 10000 : 25000);
    const now = performance.now();
    if (
      lodRaycast <= 0 ||
      byId.size === 0 ||
      now - this.lastRaycastTime < o.lodRaycastIntervalMs
    ) {
      return;
    }
    this.lastRaycastTime = now;
    const { keyIndices } = await worker.call("traverseLodTrees", {
      maxSplats: Math.min(lodRaycast, Math.round(total * 0.1)),
      pixelScaleLimit,
      instances,
      traverseMode: o.lodTraverseMode,
      errorLimit: errorLimit > 0 ? errorLimit : undefined,
      backCullMin: this.backCullMin >= 0 ? this.backCullMin : undefined,
      occlusionCull: o.lodOcclusionCull,
    });
    for (const [id, { numSplats, indices }] of Object.entries(keyIndices)) {
      const m = byId.get(id);
      if (m && this.meshes.includes(m)) {
        m.raycastIndices = { numSplats, indices };
      }
    }
  }

  dispose() {
    for (const m of [...this.meshes]) this.remove(m);
    this.pager?.dispose();
    this.pager = undefined;
    this.worker?.worker.terminate();
    this.worker = null;
  }
}
