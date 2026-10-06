// Which pages hold the paged attribute groups (PagedAttribPool), decided
// from the LoD traversal's fetch priority each time the pager drives its
// fetchers: a group of `slots` pages goes to the first `slots` resident
// pages in priority order. A page that arrives with the group's data takes
// a free slot; a wanted page without it is upgraded by fetching just that
// group (PagedSplats.fetchStreams: for a .athc v3 one Range request of its
// sections), taking a free slot or one held by a page no longer wanted.
// Nothing waits on this: a page without its slot reads as not resident and
// the shaders fall back (slang/core/attrib.slang attribResident).

import {
  type PagedAttribPool,
  type PagedGroup,
  PagedAttribPool as Pool,
} from "./PagedAttribPool";
import type { AttribPool } from "./schema";

/** What a resident page holds, as SplatPager.pageToSplatsChunk says. */
export interface PageTenant<S> {
  splats: S;
  chunk: number;
}

export interface StreamSource {
  /**
   * The attributes `names` of chunk `chunk`, fetched on their own; null
   * when the source cannot (the page then waits for its next fetch).
   */
  fetchStreams?(
    chunk: number,
    names: readonly string[],
  ): Promise<AttribPool | null>;
}

export interface AttribResidencyOptions<S> {
  /** The page's tenant now, to drop an upgrade whose page changed hands. */
  tenant(page: number): PageTenant<S> | undefined;
  /** Splats on a page (for a write). */
  count(page: number): number;
  /** Called when an upgrade landed: a frame should be drawn. */
  onUpdate?(): void;
  /** Upgrades in flight at once, over all groups. @default 2 */
  maxFetches?: number;
}

export class AttribResidency<S extends StreamSource> {
  /** Page -> rank in the last priority order (0 first). */
  private rank = new Map<number, number>();
  /** Per group: the pages it should hold now. */
  private wanted = new Map<PagedGroup, Set<number>>();
  private inflight = new Map<string, PageTenant<S>>();
  /** Groups that arrived with their page and took a slot. */
  arrived = 0;
  /** Groups that arrived with their page and found no slot. */
  dropped = 0;
  /** Upgrades that came back, for tests and reports. */
  upgrades = 0;
  failed = 0;

  constructor(
    readonly pool: PagedAttribPool,
    readonly options: AttribResidencyOptions<S>,
  ) {}

  /** Names of the groups `page` would take with its fetch: those with a free slot. */
  streamsWithFreeSlot(): string[] {
    return this.pool.groups
      .filter((g) => g.freeSlot() >= 0)
      .flatMap((g) => g.names);
  }

  /**
   * A page got a new tenant (core upload): it loses its old slots, and
   * takes the groups its chunk carries where a slot is free, or held by a
   * page no longer wanted.
   */
  onPageUpload(page: number, count: number, chunk: AttribPool | null) {
    this.pool.release(page);
    for (const g of this.pool.groups) {
      if (!Pool.covers(g, chunk)) continue;
      const slot = this.slotFor(g, page);
      if (slot >= 0) {
        this.pool.assign(g, page, slot, count, chunk as AttribPool);
        this.arrived += 1;
      } else {
        this.dropped += 1;
      }
    }
  }

  /** A slot of `group` for `page`: a free one, else the worst-ranked page's not wanted. */
  private slotFor(group: PagedGroup, page: number): number {
    const free = group.freeSlot();
    if (free >= 0) return free;
    const wanted = this.wanted.get(group);
    const mine = this.rank.get(page) ?? Number.POSITIVE_INFINITY;
    let worst = -1;
    let worstRank = -1;
    group.pageOf.forEach((holder, slot) => {
      if (wanted?.has(holder)) return;
      const r = this.rank.get(holder) ?? Number.POSITIVE_INFINITY;
      // Only for a page ranked better than the holder (a page nobody
      // ranks yet takes free slots only).
      if (!(r > mine)) return;
      if (r > worstRank) {
        worst = slot;
        worstRank = r;
      }
    });
    return worst;
  }

  /**
   * The resident pages in fetch-priority order (best first): ranks them,
   * says which pages each group wants, and starts upgrades for wanted pages
   * without the group.
   */
  update(order: readonly number[]) {
    this.rank.clear();
    order.forEach((page, k) => {
      if (!this.rank.has(page)) this.rank.set(page, k);
    });
    const max = this.options.maxFetches ?? 2;
    for (const g of this.pool.groups) {
      const wanted = new Set<number>();
      for (const page of order) {
        if (wanted.size >= g.plan.slots) break;
        wanted.add(page);
      }
      this.wanted.set(g, wanted);
      for (const page of wanted) {
        if (g.resident(page)) continue;
        const key = `${g.name}:${page}`;
        if (this.inflight.has(key) || this.inflight.size >= max) continue;
        const tenant = this.options.tenant(page);
        if (!tenant?.splats.fetchStreams) continue;
        this.inflight.set(key, tenant);
        this.upgrade(g, page, tenant).finally(() => this.inflight.delete(key));
      }
    }
  }

  private async upgrade(
    group: PagedGroup,
    page: number,
    tenant: PageTenant<S>,
  ) {
    let data: AttribPool | null = null;
    try {
      data =
        (await tenant.splats.fetchStreams?.(tenant.chunk, group.names)) ?? null;
    } catch (error) {
      if ((error as Error)?.name !== "AbortError") console.warn(error);
      this.failed += 1;
      return;
    }
    const now = this.options.tenant(page);
    if (
      !data ||
      !now ||
      now.splats !== tenant.splats ||
      now.chunk !== tenant.chunk
    ) {
      return;
    }
    if (group.resident(page) || !Pool.covers(group, data)) return;
    const slot = this.slotFor(group, page);
    if (slot < 0) return;
    this.pool.assign(group, page, slot, this.options.count(page), data);
    this.upgrades += 1;
    this.options.onUpdate?.();
  }

  /** Upgrades in flight. */
  get pending() {
    return this.inflight.size;
  }
}
