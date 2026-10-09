// The extra attributes of paged splats (.rad, .athc), resident on the GPU
// like WgpuSplatPager's core and SH pools: one buffer, the layout
// slang/core/attrib.slang reads. The CPU AttribPool it exposes holds the
// schema only, with gpuBuffer set, so SplatAttributes binds the pool as it is.
//
// Groups of attributes (attribPaging.ts) are stored one of two ways:
//   - interleaved over all pages, page p's chunk at splats [p * 65536, ...),
//     as before: a group at the page pool's full capacity;
//   - paged: a region per attribute with its own `slots` pages and a page
//     table (ATTRIB_PAGED), shared by the group's attributes. A page holds
//     the group only while it owns a slot (assign / release); otherwise the
//     shaders read it as not resident.
//
//   header | interleaved records (maxSplats x stride) |
//   per paged attribute: table (maxPages words), slots x 65536 records

import { createStorage } from "../gpuBuffers";
import {
  type AttribPagingPlan,
  PAGE_SPLATS,
  type PlannedGroup,
} from "./attribPaging";
import {
  ATTRIB_DIRECTION,
  ATTRIB_FORMATS,
  ATTRIB_NONE,
  ATTRIB_PAGED,
  AttribPool,
  type AttributeColumn,
  type AttributeSpec,
  type LodMerge,
  type PoolLayout,
  attribWords,
  columnValues,
  packColumn,
  poolLayout,
  poolWords,
} from "./schema";

/** An attribute as RadMeta.attributes lists it. */
export interface RadAttributeMeta {
  name: string;
  format: string;
  components: number;
  lodMerge?: string;
  /** transferCpca's records by transfer form (AttributeSpec.cpcaForms). */
  cpcaForms?: readonly { values: number; words: number }[];
}

/**
 * Specs from a .rad's attribute list, flagged as attribsFromResult flags a
 * decoded file's: drawable, and renormalized 3-vectors turn with the object.
 */
export function specsFromRadMeta(
  attributes: readonly RadAttributeMeta[],
): AttributeSpec[] {
  return attributes.map(
    ({ name, format, components, lodMerge, cpcaForms }) => ({
      name,
      format: format as AttributeSpec["format"],
      components,
      lodMerge: lodMerge as LodMerge | undefined,
      direction: lodMerge === "normalizeMean" && components === 3,
      toDraw: true,
      ...(cpcaForms ? { cpcaForms } : {}),
    }),
  );
}

/** A paged group's residency: which page owns which slot. */
export class PagedGroup {
  /** Page -> slot, -1 when not resident. */
  readonly slotOf: Int32Array;
  /** Slot -> page, -1 when free. */
  readonly pageOf: Int32Array;
  /** Word offset of each attribute's region (its table first). */
  readonly regions: number[];
  /** Index in the pool's specs of each attribute. */
  readonly ids: number[];

  constructor(
    readonly plan: PlannedGroup,
    maxPages: number,
    regions: number[],
    ids: number[],
  ) {
    this.slotOf = new Int32Array(maxPages).fill(-1);
    this.pageOf = new Int32Array(plan.slots).fill(-1);
    this.regions = regions;
    this.ids = ids;
  }

  get name() {
    return this.plan.name;
  }

  get names() {
    return this.plan.specs.map((s) => s.name);
  }

  resident(page: number) {
    return this.slotOf[page] >= 0;
  }

  freeSlot(): number {
    return this.pageOf.indexOf(-1);
  }

  get used(): number {
    let n = 0;
    for (const p of this.pageOf) if (p >= 0) n++;
    return n;
  }
}

export class PagedAttribPool {
  readonly layout: PoolLayout;
  /** The schema for renderers; its data lives in `buffer`. */
  readonly pool: AttribPool;
  readonly buffer: GPUBuffer;
  readonly maxPages: number;
  /** The paged groups (none when every attribute is interleaved). */
  readonly groups: PagedGroup[] = [];
  /** Bytes of `buffer`. */
  readonly bytes: number;
  readonly plan: AttribPagingPlan | null;
  /** Words of the header: the interleaved records start after it. */
  readonly headerWords: number;

  constructor(
    readonly device: GPUDevice,
    specs: AttributeSpec[],
    readonly maxSplats: number,
    plan: AttribPagingPlan | null = null,
  ) {
    this.plan = plan;
    this.maxPages = Math.ceil(maxSplats / PAGE_SPLATS);
    const all = plan ? plan.specs : specs;
    const pagedGroups = plan ? plan.groups.filter((g) => g.paged) : [];
    const pagedNames = new Set(
      pagedGroups.flatMap((g) => g.specs.map((s) => s.name)),
    );
    // The interleaved attributes' layout, then the paged regions after
    // their records, in descriptor order.
    this.layout = poolLayout(all.filter((s) => !pagedNames.has(s.name)));
    const offsets = new Map(
      this.layout.specs.map((s, k) => [s.name, this.layout.offsets[k]]),
    );
    const headerWords = 4 + 4 * all.length;
    this.headerWords = headerWords;
    let at = headerWords + Math.max(maxSplats, 1) * this.layout.strideWords;
    const regionOf = new Map<string, number>();
    for (const g of pagedGroups) {
      for (const s of g.specs) {
        regionOf.set(s.name, at);
        at += this.maxPages + g.slots * PAGE_SPLATS * attribWords(s);
      }
    }
    const words = new Uint32Array(headerWords);
    words.set([this.layout.strideWords, all.length, headerWords, maxSplats]);
    all.forEach((s, k) => {
      const region = regionOf.get(s.name);
      words.set(
        [
          region ?? (offsets.get(s.name) as number),
          ATTRIB_FORMATS[s.format],
          s.components,
          (s.direction ? ATTRIB_DIRECTION : 0) |
            (region === undefined ? 0 : ATTRIB_PAGED),
        ],
        4 + 4 * k,
      );
    });
    this.bytes = at * 4;
    this.buffer = createStorage(device, this.bytes, "attribute pages");
    device.queue.writeBuffer(this.buffer, 0, words);
    for (const g of pagedGroups) {
      const regions = g.specs.map((s) => regionOf.get(s.name) as number);
      const group = new PagedGroup(
        g,
        this.maxPages,
        regions,
        g.specs.map((s) => all.indexOf(s)),
      );
      // Every page starts not resident.
      const table = new Uint32Array(this.maxPages).fill(ATTRIB_NONE);
      for (const r of regions)
        device.queue.writeBuffer(this.buffer, r * 4, table);
      this.groups.push(group);
    }
    this.pool = new AttribPool(maxSplats);
    this.pool.columns = all.map((spec) => ({
      spec,
      words: new Uint32Array(0),
    }));
    this.pool.gpuBuffer = this.buffer;
  }

  /** Bytes the pool needs for `specs` over `maxSplats` splats, all interleaved. */
  static bytes(specs: AttributeSpec[], maxSplats: number) {
    return poolWords(poolLayout(specs), maxSplats) * 4;
  }

  group(name: string): PagedGroup | undefined {
    return this.groups.find((g) => g.name === name);
  }

  /**
   * Writes `count` splats from `base` (a page) of the interleaved
   * attributes from a chunk's, matched by name and converted to the pool's
   * formats; attributes the chunk lacks are zero.
   */
  uploadPage(base: number, count: number, chunk: AttribPool | null) {
    const { layout } = this;
    // The cluster table of a transferCpca (one a cloud, shared by its pages).
    if (chunk?.cpca) this.pool.cpca = chunk.cpca;
    const stride = layout.strideWords;
    if (stride === 0) return;
    const rows = new Uint32Array(count * stride);
    const n = Math.min(count, chunk?.count ?? 0);
    layout.specs.forEach((spec, k) => {
      const words = columnWords(spec, chunk?.column(spec.name), n);
      if (!words) return;
      const per = attribWords(spec);
      for (let i = 0; i < n; i++) {
        rows.set(
          words.subarray(i * per, (i + 1) * per),
          i * stride + layout.offsets[k],
        );
      }
    });
    this.device.queue.writeBuffer(
      this.buffer,
      (this.headerWords + base * stride) * 4,
      rows,
    );
    this.pool.version += 1;
  }

  /** Whether `chunk` has every attribute of `group`. */
  static covers(group: PagedGroup, chunk: AttribPool | null | undefined) {
    return Boolean(
      chunk &&
        group.plan.specs.every((s) => {
          const col = chunk.column(s.name);
          return col && columnWords(s, col, 0) !== null;
        }),
    );
  }

  /**
   * Gives `page` the free slot `slot` of `group` and writes its `count`
   * splats from `chunk` there; the page then reads as resident.
   */
  assign(
    group: PagedGroup,
    page: number,
    slot: number,
    count: number,
    chunk: AttribPool,
  ) {
    if (chunk.cpca) this.pool.cpca = chunk.cpca;
    if (group.pageOf[slot] >= 0) this.evict(group, group.pageOf[slot]);
    if (group.slotOf[page] >= 0) this.evict(group, page);
    // A page's arrays may be longer than its splats.
    const n = Math.min(count, chunk.count);
    group.plan.specs.forEach((spec, k) => {
      const words = columnWords(spec, chunk.column(spec.name), n);
      if (!words) return;
      const per = attribWords(spec);
      const at = group.regions[k] + this.maxPages + slot * PAGE_SPLATS * per;
      this.device.queue.writeBuffer(this.buffer, at * 4, words, 0, n * per);
    });
    group.slotOf[page] = slot;
    group.pageOf[slot] = page;
    this.writeTable(group, page, slot);
  }

  /** Takes `page`'s slot of `group` away; the page reads as not resident. */
  evict(group: PagedGroup, page: number) {
    const slot = group.slotOf[page];
    if (slot < 0) return;
    group.slotOf[page] = -1;
    group.pageOf[slot] = -1;
    this.writeTable(group, page, ATTRIB_NONE);
  }

  /** A page that changes tenant loses its slots in every group. */
  release(page: number) {
    for (const g of this.groups) this.evict(g, page);
  }

  private writeTable(group: PagedGroup, page: number, value: number) {
    const word = new Uint32Array([value >>> 0]);
    for (const r of group.regions) {
      this.device.queue.writeBuffer(this.buffer, (r + page) * 4, word);
    }
    this.pool.version += 1;
  }

  /** Bytes by part, for budgets and reports. */
  memory() {
    const interleaved =
      Math.max(this.maxSplats, 1) * this.layout.strideWords * 4;
    return {
      total: this.bytes,
      interleaved,
      groups: this.groups.map((g) => ({
        name: g.name,
        slots: g.plan.slots,
        resident: g.used,
        bytes: g.plan.specs.reduce(
          (n, s) =>
            n +
            (this.maxPages + g.plan.slots * PAGE_SPLATS * attribWords(s)) * 4,
          0,
        ),
      })),
    };
  }

  destroy() {
    this.buffer.destroy();
  }
}

/**
 * The words of `col` for `count` splats in `spec`'s format: as they are,
 * converted, or each row's prefix (a shorter transfer form, the same
 * format with fewer components). Null when the column cannot fill it.
 */
export function columnWords(
  spec: AttributeSpec,
  col: AttributeColumn | undefined,
  count: number,
): Uint32Array | null {
  if (!col) return null;
  if (col.spec.components === spec.components) {
    return col.spec.format === spec.format
      ? col.words
      : packColumn(spec, columnValues(col, count), count);
  }
  if (
    col.spec.format !== spec.format ||
    col.spec.components < spec.components
  ) {
    return null;
  }
  const per = attribWords(spec);
  const from = attribWords(col.spec);
  const out = new Uint32Array(count * per);
  for (let i = 0; i < count; i++) {
    out.set(col.words.subarray(i * from, i * from + per), i * per);
  }
  // The last word's unused half: zero, as a pool written from values is.
  const bits = (spec.components * formatBits(spec)) % 32;
  if (bits) {
    const mask = (1 << bits) - 1;
    for (let i = 0; i < count; i++) out[i * per + per - 1] &= mask;
  }
  return out;
}

function formatBits(spec: AttributeSpec) {
  return spec.format === "f32" || spec.format === "u32"
    ? 32
    : spec.format === "f16" || spec.format === "u16"
      ? 16
      : 8;
}
