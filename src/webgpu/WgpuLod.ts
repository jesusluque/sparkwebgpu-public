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
import { isAndroid, isIos, isMobile, isOculus, isVisionPro } from "../utils";
import { GpuSplatSource } from "./GpuSplatSource";
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
      "lodSplatCount" | "onDirty" | "lodRaycast" | "pagedAttributes"
    >
  > &
    Pick<
      WgpuLodOptions,
      "lodSplatCount" | "onDirty" | "lodRaycast" | "pagedAttributes"
    >;
  readonly meshes: WgpuLodMesh[] = [];
  pager?: WgpuSplatPager;
  /** Time of the last traversal in ms, and the splats it selected. */
  readonly stats = { traverseMs: 0, lodSplats: 0, traversals: 0 };

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
  }[] = [];
  private dirty = true;
  private version = 0;
  private lastVersion = -1;
  private lastLod?: {
    pos: THREE.Vector3;
    quat: THREE.Quaternion;
    pixelScaleLimit: number;
    maxSplats: number;
  };
  private lastPixelLimit?: number;
  private lastRaycastTime = Number.NEGATIVE_INFINITY;

  constructor(
    readonly renderer: WgpuSplatRenderer,
    options: WgpuLodOptions = {},
  ) {
    const defaultPages = isMobile() ? (isIos() ? 96 : 128) : 256;
    this.options = {
      lodSplatScale: 1,
      lodRenderScale: 1,
      lodTraverseMode: "standard",
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
      splats.pager = this.ensurePager();
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

  private ensurePager() {
    if (!this.pager) {
      this.pager = new WgpuSplatPager(this.renderer.device, {
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
    pixelScaleLimit *= o.lodRenderScale;

    camera.updateMatrixWorld();
    const viewPos = new THREE.Vector3();
    const viewQuat = new THREE.Quaternion();
    camera.matrixWorld.decompose(viewPos, viewQuat, new THREE.Vector3());

    if (this.lastLod) {
      if (
        this.lastLod.pixelScaleLimit !== pixelScaleLimit ||
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
      maxSplats,
    }: {
      lodMeshes: WgpuLodMesh[];
      viewPos: THREE.Vector3;
      viewQuat: THREE.Quaternion;
      pixelScaleLimit: number;
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
        }));
      }
      this.trees.set(splats, { lodId, lastTouched: performance.now() });
      this.lodIdToSplats.set(lodId, splats);
      this.dirty = true;
    }

    if (this.pager) {
      for (const update of this.pager.consumeLodTreeUpdates()) {
        const { splats, page, chunk, numSplats, lodTree } = update;
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
          chunkBase: chunk * this.pager.pageSplats,
          count: numSplats,
          lodTreeData: lodTree,
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
    this.lastLod = { pos: viewPos, quat: viewQuat, pixelScaleLimit, maxSplats };
    await this.traverse(
      worker,
      lodMeshes,
      maxSplats,
      viewPos,
      viewQuat,
      pixelScaleLimit,
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
  ) {
    const o = this.options;
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
    const { keyIndices, chunks, pixelLimit } = await worker.call(
      "traverseLodTrees",
      {
        maxSplats,
        pixelScaleLimit,
        lastPixelLimit: this.lastPixelLimit,
        instances,
        traverseMode: o.lodTraverseMode,
      },
    );
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
      this.renderer.setLodIndices(m.mesh, m.lastIndices);
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
    await this.raycastTraverse(worker, instances, byId, total, pixelScaleLimit);
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
