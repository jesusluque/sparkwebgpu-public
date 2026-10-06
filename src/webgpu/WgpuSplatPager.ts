// SplatPager's page pool in storage buffers instead of WebGL texture arrays.
//
// The fetch queue, page LRU and LoD tree updates are SplatPager's, unchanged.
// Only the residence changes: page p holds splats [p * 65536, (p + 1) * 65536)
// of each pool buffer, in the interleaved layout kernels/generate.slang reads
// (see GpuSplatSource). The LoD traversal already returns pool indices
// (pageBase + offset), so a paged mesh is drawn as an ordinary source over the
// whole pool, remapped through those indices.
//
// One buffer per pool (core, SH, attributes) keeps the storage-buffer
// count of generate unchanged. Each must fit one binding, so the pool is
// capped by the device's maxStorageBufferBindingSize: with default limits
// (128 MiB) that is 8M packed splats without SH, 2.7M with SH3. Request the
// adapter's limits from WebGPURenderer (requiredLimits) for more.
//
// The core pool (not SH) is also kept on the CPU, in SplatPager's texture
// data as on WebGL, for SplatMesh.raycast on paged meshes.
//
// Attributes are sized per group of streams within a byte budget
// (attributes/attribPaging.ts): cheap ones at the pool's capacity, the
// expensive ones (a .athc's transfer) with fewer pages of their own, given
// to the pages the traversal ranks first (attributes/AttribResidency.ts).

import * as THREE from "three";
import type { PagedSplats } from "../PagedSplats";
import { SplatPager, type SplatPagerOptions } from "../SplatPager";
import { GpuSplatSource } from "./GpuSplatSource";
import { KernelRegistry } from "./KernelRegistry";
import { AttribResidency } from "./attributes/AttribResidency";
import { PagedAttribPool } from "./attributes/PagedAttribPool";
import {
  type AttribPagingOptions,
  type AttribPagingPlan,
  attribBudget,
  planAttribPaging,
} from "./attributes/attribPaging";
import { AttribPool, type AttributeSpec } from "./attributes/schema";
import restrideModule from "./generated/kernels/pool_restride";
import { createStorage } from "./gpuBuffers";
import { tierOfLimits } from "./plugins/tiers";

const PAGE_SPLATS = 65536;

export interface WgpuSplatPagerOptions
  extends Omit<SplatPagerOptions, "renderer"> {
  /** How the attribute pool is sized and which streams it loads. */
  attributes?: AttribPagingOptions;
  /**
   * The streams wanted now (PluginHost.attributeDemand), when
   * `attributes.attributes` does not say; null for the default.
   */
  attributeDemand?: () => readonly string[] | null;
}

interface Pool {
  wordsPerSplat: number;
  buffer: GPUBuffer | null;
}

// SplatPager calls this from its constructor for textures we never use.
const unusedRenderer = {
  initTexture() {},
} as unknown as THREE.WebGLRenderer;

export class WgpuSplatPager extends SplatPager {
  readonly device: GPUDevice;
  /** Per-splat words of each pool, by name. */
  readonly pools: Record<"core" | "sh", Pool>;
  // SH degrees the SH pool's stride holds: it is sized for the data that
  // arrives (an SH1 scene takes a third of SH3's memory) and widened when a
  // page with more degrees comes.
  private shDegree = 0;
  private registry: KernelRegistry | null = null;
  /**
   * Extra attributes (src/webgpu/attributes), paged like the splats: the
   * schema from setAttribSchema (the .rad's meta) or the first chunk that
   * has some, the data from each chunk's `attrib` properties.
   */
  attribs: PagedAttribPool | null = null;
  /** Which pages hold the paged attribute groups. */
  residency: AttribResidency<PagedSplats> | null = null;
  private attribSchema: AttributeSpec[] | null = null;
  private attribPlan: AttribPagingPlan | null = null;
  private readonly attribOptions: AttribPagingOptions;
  private readonly attributeDemand?: () => readonly string[] | null;
  /** Splats on each page, as last uploaded. */
  private pageCounts: number[] = [];

  constructor(device: GPUDevice, options: WgpuSplatPagerOptions) {
    const ext = options.extSplats ?? false;
    const maxSh = options.maxSh ?? 3;
    const coreWords = ext ? 8 : 4;
    const shWords = maxSh > 0 ? (ext ? 16 : 12) : 0;
    const limit = Math.min(
      device.limits.maxStorageBufferBindingSize,
      device.limits.maxBufferSize,
    );
    const maxWords = Math.max(coreWords, shWords);
    const fitPages = Math.floor(limit / (PAGE_SPLATS * maxWords * 4));
    const wanted = options.maxSplats ?? 256 * PAGE_SPLATS;
    super({
      ...options,
      renderer: unusedRenderer,
      maxSplats: Math.min(wanted, fitPages * PAGE_SPLATS),
    });
    if (this.maxSplats < wanted) {
      console.warn(
        `WgpuSplatPager: ${this.maxPages} pages fit maxStorageBufferBindingSize (${limit}), not ${wanted / PAGE_SPLATS}`,
      );
    }
    this.device = device;
    this.attribOptions = options.attributes ?? {};
    this.attributeDemand = options.attributeDemand;
    this.pools = {
      core: { wordsPerSplat: coreWords, buffer: null },
      sh: { wordsPerSplat: 0, buffer: null },
    };
    this.ensurePool(this.pools.core, "splat pages");
  }

  private ensurePool(pool: Pool, label: string) {
    if (!pool.buffer) {
      pool.buffer = createStorage(
        this.device,
        this.maxSplats * pool.wordsPerSplat * 4,
        label,
      );
    }
    return pool.buffer;
  }

  /** The pool as a source for WgpuSplatRenderer, with `splats`' encoding. */
  source(splats: PagedSplats): GpuSplatSource {
    const sh = this.pools.sh.buffer;
    const numSh = sh ? Math.min(splats.numSh, splats.maxSh, this.curSh) : 0;
    const e = splats.splatEncoding;
    const source = new GpuSplatSource(
      this.extSplats ? "ext" : "packed",
      this.maxSplats,
      this.pools.core.buffer as GPUBuffer,
      sh,
      numSh,
      e ? [e.sh1Max, e.sh2Max, e.sh3Max] : undefined,
      e ? [e.rgbMin, e.rgbMax, e.lnScaleMin, e.lnScaleMax] : undefined,
      e?.lodOpacity ?? false,
    );
    source.shStride = this.pools.sh.wordsPerSplat / 4;
    source.attribs = this.attribs?.pool ?? null;
    return source;
  }

  /**
   * The attribute schema of the pool (all paged meshes share it), before
   * the first page arrives; later schemas are ignored.
   */
  setAttribSchema(specs: AttributeSpec[]) {
    if (!this.attribSchema && specs.length) this.attribSchema = specs;
  }

  /**
   * How the attribute pool is sized for the schema: which streams it
   * keeps, and the pages of each group (attribPaging.ts). Null before a
   * schema is known.
   */
  attributePlan(): AttribPagingPlan | null {
    if (this.attribPlan || !this.attribSchema) return this.attribPlan;
    const o = this.attribOptions;
    const tier = o.tier ?? tierOfLimits(this.device.limits);
    const limit = Math.min(
      this.device.limits.maxStorageBufferBindingSize,
      this.device.limits.maxBufferSize,
    );
    const options = {
      budget: Math.min(o.budgetBytes ?? attribBudget(this.device, tier), limit),
      tier,
      demand: o.attributes ?? this.attributeDemand?.() ?? null,
      pages: o.pages,
      transferForm: o.transferForm,
    };
    let plan = planAttribPaging(this.attribSchema, this.maxPages, options);
    if (plan.bytes > limit) {
      console.warn(
        `WgpuSplatPager: attribute pages as asked need ${plan.bytes} bytes, over the binding size (${limit}); sizing them by the budget`,
      );
      plan = planAttribPaging(this.attribSchema, this.maxPages, {
        ...options,
        pages: {},
      });
    }
    for (const g of plan.groups) {
      if (g.paged) {
        console.info(
          `WgpuSplatPager: '${g.name}' attributes (${g.specs.map((s) => s.name).join(", ")}) paged in ${g.slots} of ${this.maxPages} pages`,
        );
      }
    }
    this.attribPlan = plan;
    return plan;
  }

  /**
   * The streams a fetch of `chunk` of `splats` should bring with it (a
   * .athc v3 reads them in the same Range request): the groups at full
   * capacity, and paged ones with a slot free.
   */
  streamsToFetch(_splats: PagedSplats, _chunk: number): string[] {
    const plan = this.attributePlan();
    if (!plan) return [];
    const free = new Set(this.residency?.streamsWithFreeSlot() ?? []);
    return plan.groups.flatMap((g) =>
      g.paged && this.residency && !g.specs.some((s) => free.has(s.name))
        ? []
        : g.specs.map((s) => s.name),
    );
  }

  /** Every stream the pool keeps (merged .athc pages carry them all). */
  streamsKept(): string[] {
    return this.attributePlan()?.specs.map((s) => s.name) ?? [];
  }

  private ensureAttribs(chunk: AttribPool | null): PagedAttribPool | null {
    if (this.attribs) return this.attribs;
    if (!this.attribSchema) {
      if (!chunk) return null;
      this.attribSchema = chunk.schema.map((spec) => ({
        ...spec,
        toDraw: true,
      }));
    }
    const plan = this.attributePlan();
    if (!plan?.specs.length) return null;
    this.attribs = new PagedAttribPool(
      this.device,
      plan.specs,
      this.maxSplats,
      plan,
    );
    if (this.attribs.groups.length) {
      this.residency = new AttribResidency(this.attribs, {
        tenant: (page) => this.pageToSplatsChunk[page],
        count: (page) => this.pageCounts[page] ?? 0,
        onUpdate: () => this.onUpdate?.(),
      });
    }
    return this.attribs;
  }

  private uploadAttribs(page: number, count: number, data: unknown) {
    const chunk = data
      ? AttribPool.from(data as Parameters<typeof AttribPool.from>[0])
      : null;
    const attribs = this.ensureAttribs(chunk);
    if (!attribs) return;
    attribs.uploadPage(page * PAGE_SPLATS, count, chunk);
    this.residency?.onPageUpload(page, count, chunk);
  }

  /** Ranks the resident pages for the paged attribute groups, after the core fetches. */
  driveFetchers() {
    super.driveFetchers();
    if (!this.residency) return;
    const order: number[] = [];
    for (const { splats, chunk } of this.fetchPriority) {
      const entry = this.getSplatsChunk(splats, chunk);
      if (entry) order.push(entry.page);
    }
    this.residency.update(order);
  }

  // uint4s per splat for `degree` SH degrees: packed sh1 (padded), sh2, sh3;
  // ext sh1, sh2, sh3a + sh3b.
  private shSlots(degree: number) {
    return degree === 3 && this.extSplats ? 4 : degree;
  }

  // Makes the SH pool hold `degree` degrees, moving the pages already there.
  private ensureShDegree(degree: number) {
    if (degree <= this.shDegree) return;
    const sh = this.pools.sh;
    const oldSlots = sh.wordsPerSplat / 4;
    const newSlots = this.shSlots(degree);
    const old = sh.buffer;
    sh.wordsPerSplat = 4 * newSlots;
    sh.buffer = null;
    const buffer = this.ensurePool(sh, "SH pages");
    if (old) {
      this.registry ??= new KernelRegistry(this.device);
      const kernel = this.registry.get(restrideModule, "restride");
      // In slices: 256 pages need 65536 workgroups, one over the limit.
      const slice =
        this.device.limits.maxComputeWorkgroupsPerDimension *
        (kernel.entry.workgroupSize?.[0] ?? 1);
      for (let first = 0; first < this.maxSplats; first += slice) {
        const count = Math.min(slice, this.maxSplats - first);
        kernel.run({
          grid: [count],
          buffers: { src: old, dst: buffer },
          uniforms: new Uint32Array([count, oldSlots, newSlots, first]),
        });
      }
      old.destroy();
    }
    this.shDegree = degree;
  }

  // No textures: SplatPager's dyno blocks are never compiled on this path.
  protected newUint32ArrayTexture(): THREE.DataArrayTexture {
    return new THREE.DataArrayTexture(new Uint32Array(4), 1, 1, 1);
  }

  protected uploadPage(
    page: number,
    packedArray: Uint32Array,
    shArrays: Array<Uint32Array>,
    extArray?: Uint32Array,
    attribs?: unknown,
  ) {
    const base = page * PAGE_SPLATS;
    const core = this.pools.core;
    const count = packedArray.length / 4;
    this.pageCounts[page] = count;
    this.mirror(this.packedTexture, base, packedArray);
    if (extArray) this.mirror(this.extTexture, base, extArray);
    // Every page, so one without attributes clears the previous tenant's.
    this.uploadAttribs(page, count, attribs);
    if (extArray) {
      // Ext pages arrive as two arrays; the pool interleaves a and b.
      const words = new Uint32Array(count * 8);
      for (let i = 0; i < count; i++) {
        words.set(packedArray.subarray(4 * i, 4 * i + 4), 8 * i);
        words.set(extArray.subarray(4 * i, 4 * i + 4), 8 * i + 4);
      }
      this.write(core, base, words);
    } else {
      this.write(core, base, packedArray);
    }

    const numSh = Math.min(shArrays.length, 3);
    if (numSh === 0 || this.maxSh === 0) return;
    const degree = Math.min(numSh, this.maxSh);
    this.curSh = Math.max(this.curSh, degree);
    this.ensureShDegree(degree);
    const sh = this.pools.sh;
    // Packed: sh1 (2 words, padded to 4), sh2, sh3. Ext: sh1, sh2, sh3a, sh3b.
    const stride = sh.wordsPerSplat;
    const perArray = this.extSplats ? [4, 4, 4, 4] : [2, 4, 4];
    const arrays = shArrays.slice(0, this.shSlots(degree));
    const words = new Uint32Array(count * stride);
    for (let i = 0; i < count; i++) {
      arrays.forEach((array, k) => {
        const n = perArray[k];
        words.set(array.subarray(n * i, n * i + n), stride * i + 4 * k);
      });
    }
    this.write(sh, base, words);
  }

  // Copies a page into the CPU pool SplatMesh.raycast reads, allocated with
  // the first page (SplatPager's would be allocated up front).
  private mirror(
    texture: { value: THREE.DataArrayTexture },
    splatBase: number,
    words: Uint32Array,
  ) {
    const data = texture.value.image.data as Uint32Array;
    if (data.length < this.maxSplats * 4) {
      const pool = new Uint32Array(this.maxSplats * 4);
      texture.value = new THREE.DataArrayTexture(pool, 1, 1, 1);
      pool.set(words, splatBase * 4);
    } else {
      data.set(words, splatBase * 4);
    }
  }

  private write(pool: Pool, splatBase: number, words: Uint32Array) {
    this.device.queue.writeBuffer(
      pool.buffer as GPUBuffer,
      splatBase * pool.wordsPerSplat * 4,
      words.buffer,
      words.byteOffset,
      words.byteLength,
    );
  }

  dispose() {
    super.dispose();
    this.registry?.destroy();
    this.attribs?.destroy();
    this.attribs = null;
    this.residency = null;
    for (const pool of Object.values(this.pools)) {
      pool.buffer?.destroy();
      pool.buffer = null;
    }
  }
}
