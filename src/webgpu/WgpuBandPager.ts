// athenea's error-band pages streamed Nanite-style (thread CK).
//
// A paged .athc tree (rust athc_pages.rs, FLAG_PAGED) is every node of
// athenea's merge tree, page by page: coarsest band of parent error first,
// each page a few thousand nodes whose children are one run in one page.
// Its page table is always resident (it is in the file's tables). A view
// needs a page when the largest parent error in it, seen from the nearest
// point of its box, projects past tau:
//
//   needed(p) = epsMax(p) / max(dist(eye, box(p)), near) > errorLimit
//
// (errorLimit = tau in the traversal's angle units over the mesh's
// lodScale: the same test the traversal refines a node by, taken over every
// node in the page; the box holds the parents' spheres, so it never misses a
// node the cut can reach). Page 0 (the root) is always needed.
//
// Each frame (each traversal) this pager:
//   - computes the needed set from the table alone, without reading nodes;
//   - asks for the missing needed pages by priority, the error a page
//     removes per byte it costs (epsMin seen from its box over its bytes;
//     ties nearer first), consecutive pages merged into one Range request,
//     within the pool's capacity (the residency budget);
//   - lets the pool evict pages not needed, from the highest index down
//     (finest first), never one a resident page depends on (whose nodes'
//     children it holds), never page 0;
//   - reports how full the needed set would make the pool, for tau to
//     breathe (WgpuLod's LodTauController) when it does not fit.
//
// The traversal (spark-rs lod_tree.rs) cuts within the resident pages: a
// node whose children's page is absent is drawn as a leaf.
//
// Pages are SplatPager chunks: band page p is chunk p of the mesh's virtual
// index space (65 536 indices a chunk), held in a pool slot of
// `pageSplats` splats (default 3072: athenea's 3000 and the few the packer
// moves to keep a run of children in one page).

import type { PagedSplats } from "../PagedSplats";
import type { AthcPageRow, AthcV3Layout } from "../athc";
import { athcBandPageBytes, athcWantOf } from "../athc";
import { WgpuSplatPager, type WgpuSplatPagerOptions } from "./WgpuSplatPager";

/** The traversal's floor on a distance (lod_tree.rs ERROR_NEAR). */
export const BAND_NEAR = 1e-3;

/** Distance from `eye` to a page's box. */
export function pageDistance(row: AthcPageRow, eye: ArrayLike<number>) {
  let d2 = 0;
  for (let k = 0; k < 3; k++) {
    const v = Math.max(row.lo[k] - eye[k], 0, eye[k] - row.hi[k]);
    d2 += v * v;
  }
  return Math.sqrt(d2);
}

/** athenea's page test (see the module). */
export function pageNeeded(
  row: AthcPageRow,
  eye: ArrayLike<number>,
  limit: number,
  near = BAND_NEAR,
) {
  return row.epsMax / Math.max(pageDistance(row, eye), near) > limit;
}

/**
 * Spark's foveation (lod_tree.rs compute_pixel_scale) of a direction whose
 * cosine to the view's forward is `dot`: 1 inside coneFov0, down to
 * coneFoveate at coneFov, to behindFoveate at 90 degrees and behind.
 */
export interface Foveation {
  coneFov0: number;
  coneFov: number;
  coneFoveate: number;
  behindFoveate: number;
}

export function foveateOf(dot: number, f: Foveation) {
  const dot0 =
    f.coneFov0 > 0
      ? Math.cos((0.5 * Math.min(f.coneFov0, 180) * Math.PI) / 180)
      : 1;
  const dot1 = Math.min(
    f.coneFov > 0
      ? Math.cos((0.5 * Math.min(f.coneFov, 180) * Math.PI) / 180)
      : 1,
    dot0,
  );
  if (dot <= 0) return f.behindFoveate;
  if (dot >= dot0) return 1;
  if (dot >= dot1) {
    const t = (dot - dot1) / Math.max(dot0 - dot1, 1e-9);
    return f.coneFoveate + (1 - f.coneFoveate) * t;
  }
  const t = dot / Math.max(dot1, 1e-9);
  return f.behindFoveate + (f.coneFoveate - f.behindFoveate) * t;
}

/**
 * The most a page's nodes are foveated (its box's nearest direction to the
 * forward): the bound the page test needs, never under any node's.
 */
export function pageFoveate(
  row: AthcPageRow,
  eye: ArrayLike<number>,
  forward: ArrayLike<number>,
  f: Foveation,
) {
  let d2 = 0;
  let r2 = 0;
  let dot = 0;
  const c = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    c[k] = 0.5 * (row.lo[k] + row.hi[k]) - eye[k];
    const h = 0.5 * (row.hi[k] - row.lo[k]);
    d2 += c[k] * c[k];
    r2 += h * h;
    dot += c[k] * forward[k];
  }
  if (d2 <= r2) return 1;
  const d = Math.sqrt(d2);
  const angle = Math.acos(Math.max(-1, Math.min(1, dot / d)));
  const nearest = Math.max(0, angle - Math.asin(Math.sqrt(r2) / d));
  return foveateOf(Math.cos(nearest), f);
}

/** One mesh's pages as the planner sees them. */
export interface BandMesh<K = unknown> {
  key: K;
  rows: readonly AthcPageRow[];
  /** Bytes a fetch of each page reads. */
  pageBytes: readonly number[];
  /** The eye in the tree's object space. */
  eye: ArrayLike<number>;
  /** errorLimit / lodScale: needed when epsMax / dist passes it. */
  limit: number;
  /**
   * The traversal's foveation of each page at most (1: none): epsMax /
   * dist is weighed by it, as the traversal weighs its nodes (default 1).
   */
  foveate?: (page: number) => number;
  resident: (page: number) => boolean;
  inFlight: (page: number) => boolean;
  /** Pages (other than itself) a resident page's nodes have children in. */
  children: (page: number) => readonly number[] | undefined;
  /**
   * Resident pages holding a node of the current cut (never evicted: the
   * dependency rule then keeps their ancestors' pages too). Default none.
   */
  inCut?: (page: number) => boolean;
}

/** What the pool can take, for planBand (the pager's own state). */
export interface BandRoom {
  /** Free slots. */
  free: number;
  /** Pages fetched or being fetched that are not in the pool yet. */
  pending: number;
}

/**
 * A page needed at limit x BAND_KEEP is not evicted (hysteresis: tau
 * breathing a little does not swap pages out and back in).
 */
export const BAND_KEEP = 0.8;

export interface BandPlan<K = unknown> {
  /** Needed pages of each mesh (by key). */
  needed: Map<K, Set<number>>;
  neededPages: number;
  neededNodes: number;
  /** Requests in priority order: consecutive pages of one mesh. */
  requests: { key: K; pages: number[]; priority: number; bytes: number }[];
  /** Resident pages the pool may evict, the first first. */
  evictable: { key: K; page: number }[];
}

/**
 * The needed set, the requests and the eviction order (see the module).
 * `capacity`: pool pages; `maxRangeBytes`: the longest Range merged.
 */
export function planBand<K>(
  meshes: readonly BandMesh<K>[],
  capacity: number,
  maxRangeBytes = 8 << 20,
  room?: BandRoom,
): BandPlan<K> {
  const needed = new Map<K, Set<number>>();
  let neededPages = 0;
  let neededNodes = 0;
  let residentNeeded = 0;
  const missing: {
    m: BandMesh<K>;
    page: number;
    priority: number;
    distance: number;
  }[] = [];
  for (const m of meshes) {
    const set = new Set<number>();
    m.rows.forEach((row, p) => {
      const d = pageDistance(row, m.eye);
      const f = m.foveate ? m.foveate(p) : 1;
      if (p !== 0 && !((f * row.epsMax) / Math.max(d, BAND_NEAR) > m.limit)) {
        return;
      }
      set.add(p);
      neededPages += 1;
      neededNodes += row.count;
      if (m.resident(p) || m.inFlight(p)) {
        residentNeeded += 1;
        return;
      }
      // Error removed per byte; page 0 first of all.
      const priority =
        p === 0
          ? Number.POSITIVE_INFINITY
          : (f * row.epsMin) /
            Math.max(d, BAND_NEAR) /
            Math.max(1, m.pageBytes[p] ?? 1);
      missing.push({ m, page: p, priority, distance: d });
    });
    needed.set(m.key, set);
  }
  missing.sort((a, b) => b.priority - a.priority || a.distance - b.distance);
  // Evictable: resident, not needed (not even at BAND_KEEP x the limit),
  // not in the cut, not page 0, no resident child page that stays; highest
  // index first (finest band first). Never a needed page: a full pool
  // takes no new page at a needed one's expense.
  const evictable: BandPlan<K>["evictable"] = [];
  for (const m of meshes) {
    const set = needed.get(m.key) as Set<number>;
    const out: number[] = [];
    for (let p = m.rows.length - 1; p > 0; p--) {
      if (set.has(p) || !m.resident(p) || m.inCut?.(p)) continue;
      const f = m.foveate ? m.foveate(p) : 1;
      if (pageNeeded(m.rows[p], m.eye, (m.limit * BAND_KEEP) / f)) continue;
      out.push(p);
    }
    const going = new Set(out);
    // Children before parents (higher index first): a page stays while a
    // child page of it stays.
    for (const p of out) {
      const kids = m.children(p) ?? [];
      if (
        kids.some(
          (c) => c !== p && (m.inFlight(c) || (m.resident(c) && !going.has(c))),
        )
      ) {
        going.delete(p);
      }
    }
    for (const p of out)
      if (going.has(p)) evictable.push({ key: m.key, page: p });
  }
  // Across meshes: by page index, highest first.
  evictable.sort((a, b) => b.page - a.page);
  // Within the budget: what is needed and resident (or coming) stays, the
  // best of the rest fills what the pool can take -- its free slots and
  // the evictable pages, less what is still on its way -- and no more (a
  // page fetched without a slot would be fetched again and again).
  const slots = room
    ? Math.max(0, room.free + evictable.length - room.pending)
    : Math.max(0, capacity - residentNeeded);
  const chosen = missing.slice(0, slots);
  // Consecutive chosen pages of a mesh: one Range request, at the place of
  // its best page.
  const byMesh = new Map<BandMesh<K>, Map<number, number>>();
  chosen.forEach((c, rank) => {
    let pages = byMesh.get(c.m);
    if (!pages) {
      pages = new Map();
      byMesh.set(c.m, pages);
    }
    pages.set(c.page, rank);
  });
  const requests: BandPlan<K>["requests"] = [];
  const ranks: number[] = [];
  for (const [m, pages] of byMesh) {
    const sorted = [...pages.keys()].sort((a, b) => a - b);
    let run: number[] = [];
    let bytes = 0;
    let best = Number.POSITIVE_INFINITY;
    const flush = () => {
      if (!run.length) return;
      requests.push({
        key: m.key,
        pages: run,
        priority: Math.max(...run.map((p) => priorityOf(chosen, m, p))),
        bytes,
      });
      ranks.push(best);
      run = [];
      bytes = 0;
      best = Number.POSITIVE_INFINITY;
    };
    for (const p of sorted) {
      const b = m.pageBytes[p] ?? 0;
      if (
        run.length &&
        (p !== run[run.length - 1] + 1 || bytes + b > maxRangeBytes)
      ) {
        flush();
      }
      run.push(p);
      bytes += b;
      best = Math.min(best, pages.get(p) as number);
    }
    flush();
  }
  const order = requests.map((_, k) => k).sort((a, b) => ranks[a] - ranks[b]);
  return {
    needed,
    neededPages,
    neededNodes,
    requests: order.map((k) => requests[k]),
    evictable,
  };
}

function priorityOf<K>(
  chosen: { m: BandMesh<K>; page: number; priority: number }[],
  m: BandMesh<K>,
  p: number,
) {
  return chosen.find((c) => c.m === m && c.page === p)?.priority ?? 0;
}

export interface WgpuBandPagerOptions extends WgpuSplatPagerOptions {
  /** The longest Range request merged from consecutive pages. @default 8 MiB */
  maxRangeBytes?: number;
}

/** A mesh's view for the next plan (WgpuLod.fetchPages). */
export interface BandView {
  splats: PagedSplats;
  /** The eye in the tree's object space. */
  eye: ArrayLike<number>;
  /** errorLimit / lodScale. */
  limit: number;
  /** The limit at tau 1 (a pixel in the traversal's angle units / lodScale). */
  unit?: number;
  /** Pool pages (slots) holding a node of the current cut. */
  cutSlots?: ReadonlySet<number>;
}

/**
 * The smallest tau whose needed set fits `capacity` pages: a page is needed
 * while tau < epsMax / (max(dist, near) unit), so the needed set at tau is
 * the pages over it; tau fits at the (capacity - roots + 1)-th largest
 * threshold. 0 when everything fits.
 */
export function bandTauToFit(
  meshes: readonly {
    rows: readonly AthcPageRow[];
    eye: ArrayLike<number>;
    unit: number;
  }[],
  capacity: number,
): number {
  const t: number[] = [];
  for (const m of meshes) {
    if (!(m.unit > 0)) continue;
    m.rows.forEach((row, p) => {
      if (p === 0) return;
      t.push(
        row.epsMax / Math.max(pageDistance(row, m.eye), BAND_NEAR) / m.unit,
      );
    });
  }
  const room = capacity - meshes.length;
  if (t.length <= room) return 0;
  t.sort((a, b) => b - a);
  return t[Math.max(0, room)] * (1 + 1e-6);
}

/**
 * athenea's order (Nanite's): tau first, then the requests. Each mesh's
 * limit raised to the tau whose needed set fits `capacity` (bandTauToFit,
 * from `unit`, the limit at tau 1), then planBand. Returns the plan and the
 * tau it fitted (0 when everything fits).
 */
export function planBandFitted<K>(
  meshes: readonly (BandMesh<K> & { unit?: number })[],
  capacity: number,
  maxRangeBytes = 8 << 20,
  room?: BandRoom,
): { plan: BandPlan<K>; fit: number } {
  const fit = bandTauToFit(
    meshes
      .filter((m) => (m.unit ?? 0) > 0)
      .map((m) => ({ rows: m.rows, eye: m.eye, unit: m.unit as number })),
    capacity,
  );
  const fitted = meshes.map((m) =>
    m.unit ? { ...m, limit: Math.max(m.limit, fit * m.unit) } : m,
  );
  return { plan: planBand(fitted, capacity, maxRangeBytes, room), fit };
}

/** The pool of athenea's band pages (see the module). */
export class WgpuBandPager extends WgpuSplatPager {
  readonly maxRangeBytes: number;
  /** Each paged tree's table and the bytes a page fetch reads. */
  private tables = new Map<
    PagedSplats,
    { layout: AthcV3Layout; rows: AthcPageRow[]; pageBytes: number[] }
  >();
  /** Child pages of each resident page (its nodes' children), per mesh. */
  private kids = new Map<PagedSplats, Map<number, number[]>>();
  private views: BandView[] = [];
  /** The last plan's needed set over the pool's pages. */
  neededFill = 0;
  readonly bandStats = {
    neededPages: 0,
    neededNodes: 0,
    residentPages: 0,
    residentNodes: 0,
    /** Pages requested, Range requests, bytes read, since creation. */
    pagesRequested: 0,
    ranges: 0,
    bytesRead: 0,
    /** Pages evicted. */
    evicted: 0,
    /** Needed pages not resident nor coming (over the budget). */
    starved: 0,
  };

  constructor(device: GPUDevice, options: WgpuBandPagerOptions) {
    super(device, { pageSplats: 3072, ...options });
    this.maxRangeBytes = options.maxRangeBytes ?? 8 << 20;
  }

  /** Registers a paged tree's table (its layout's PAGE rows). */
  addTree(splats: PagedSplats, layout: AthcV3Layout) {
    const rows = layout.pages ?? [];
    const largest = rows.reduce((n, r) => Math.max(n, r.count), 0);
    if (largest > this.pageSplats) {
      throw new Error(
        `WgpuBandPager: a page of ${largest} nodes, over the pool's ${this.pageSplats}`,
      );
    }
    this.tables.set(splats, { layout, rows, pageBytes: [] });
  }

  isBand(splats: PagedSplats) {
    return this.tables.has(splats);
  }

  /** The views of the next plan: one per drawn band-paged mesh. */
  setViews(views: BandView[]) {
    this.views = views;
  }

  /**
   * The smallest tau whose needed set fits the pool, for the last views
   * (bandTauToFit): what tau breathes to when the budget does not fit.
   */
  tauToFit(): number {
    const meshes = [];
    for (const v of this.views) {
      const t = this.tables.get(v.splats);
      if (t && v.unit) meshes.push({ rows: t.rows, eye: v.eye, unit: v.unit });
    }
    return bandTauToFit(meshes, this.maxPages);
  }

  private pageBytes(splats: PagedSplats) {
    const t = this.tables.get(splats);
    if (!t) return [];
    if (!t.pageBytes.length) {
      const want = athcWantOf(t.layout, this.attributePlan()?.specs ?? []);
      t.pageBytes = t.rows.map((_, p) => athcBandPageBytes(t.layout, p, want));
    }
    return t.pageBytes;
  }

  private inFlight(splats: PagedSplats, page: number) {
    return (
      this.fetchers.some((f) => f.splats === splats && f.chunk === page) ||
      this.fetched.some((f) => f.splats === splats && f.chunk === page)
    );
  }

  /** Plans and starts fetches (in place of SplatPager's chunk queue). */
  driveFetchers() {
    // tau first: the needed set at the views' limits must fit the pool
    // before anything is asked for, or the pool would take pages only to
    // drop needed ones (the limits raised to bandTauToFit's).
    const fit = this.tauToFit();
    const meshes: (BandMesh<PagedSplats> & { unit?: number })[] = [];
    for (const v of this.views) {
      const t = this.tables.get(v.splats);
      if (!t) continue;
      const kids = this.kids.get(v.splats);
      const cut = v.cutSlots;
      meshes.push({
        key: v.splats,
        rows: t.rows,
        pageBytes: this.pageBytes(v.splats),
        eye: v.eye,
        limit: v.limit,
        unit: v.unit,
        resident: (p) => this.getSplatsChunk(v.splats, p) !== undefined,
        inFlight: (p) => this.inFlight(v.splats, p),
        children: (p) => kids?.get(p),
        inCut: cut
          ? (p) => {
              const slot = this.getSplatsChunk(v.splats, p)?.page;
              return slot !== undefined && cut.has(slot);
            }
          : undefined,
      });
    }
    // A fetched page still waiting for a slot that no view needs any more
    // is let go (it would hold the pool's room).
    for (let i = this.fetched.length - 1; i >= 0; i--) {
      const f = this.fetched[i];
      const v = this.views.find((w) => w.splats === f.splats);
      const t = this.tables.get(f.splats);
      if (!v || !t) continue;
      const lim = v.unit ? Math.max(v.limit, fit * v.unit) : v.limit;
      const row = t.rows[f.chunk];
      if (f.chunk !== 0 && row && !pageNeeded(row, v.eye, lim * BAND_KEEP)) {
        this.fetched.splice(i, 1);
      }
    }
    const { plan } = planBandFitted(meshes, this.maxPages, this.maxRangeBytes, {
      free: this.pageFreelist.length,
      pending: this.fetchers.length + this.fetched.length,
    });
    this.neededFill = plan.neededPages / this.maxPages;
    const s = this.bandStats;
    s.neededPages = plan.neededPages;
    s.neededNodes = plan.neededNodes;
    let residentPages = 0;
    let residentNodes = 0;
    for (const [splats, chunks] of this.splatsChunkToPage) {
      const t = this.tables.get(splats);
      chunks.forEach((c, p) => {
        if (!c) return;
        residentPages += 1;
        residentNodes += t?.rows[p]?.count ?? 0;
      });
    }
    s.residentPages = residentPages;
    s.residentNodes = residentNodes;
    // The pool takes its slots from the evictable pages in this order.
    this.freeablePages = plan.evictable
      .map(({ key, page }) => this.getSplatsChunk(key, page)?.page)
      .filter((p): p is number => p !== undefined);
    const requested = plan.requests.reduce((n, r) => n + r.pages.length, 0);
    s.starved = Math.max(
      0,
      plan.neededPages - residentPages - requested - this.fetchers.length,
    );
    for (const r of plan.requests) {
      if (this.fetchers.length >= this.numFetchers) break;
      if (r.pages.some((p) => this.inFlight(r.key, p))) continue;
      this.fetchRun(r.key, r.pages);
    }
  }

  private fetchRun(splats: PagedSplats, pages: number[]) {
    const entries = pages.map((chunk) => ({ splats, chunk }));
    this.fetchers.push(...entries);
    const s = this.bandStats;
    s.pagesRequested += pages.length;
    splats
      .fetchDecodeBandPages(pages)
      .then(
        ({ bytes, ranges, pages: decoded }) => {
          s.bytesRead += bytes;
          s.ranges += ranges;
          if (
            splats.abortController.signal.aborted ||
            !entries.every((e) => this.fetchers.includes(e))
          ) {
            return;
          }
          for (const { chunk, data } of decoded) {
            this.noteChildren(splats, chunk, data.extra.lodTree as Uint32Array);
            this.fetched.push({ splats, chunk, data });
          }
        },
        async (error) => {
          if (error.name !== "AbortError") console.warn(error);
          await new Promise((r) => setTimeout(r, 250 + 500 * Math.random()));
        },
      )
      .finally(() => {
        for (const e of entries) {
          const k = this.fetchers.indexOf(e);
          if (k >= 0) this.fetchers.splice(k, 1);
        }
        this.processFetched();
        this.onUpdate?.();
        if (this.autoDrive) this.driveFetchers();
      });
  }

  /** The pages a page's nodes have their children in. */
  private noteChildren(splats: PagedSplats, chunk: number, tree?: Uint32Array) {
    if (!tree) return;
    const pages = new Set<number>();
    for (let i = 0; i + 3 < tree.length; i += 4) {
      if ((tree[i + 2] & 0xffff) === 0) continue;
      const c = tree[i + 3] >>> 16;
      if (c !== chunk) pages.add(c);
    }
    let byPage = this.kids.get(splats);
    if (!byPage) {
      byPage = new Map();
      this.kids.set(splats, byPage);
    }
    byPage.set(chunk, [...pages]);
  }

  /** A fetched page without a slot waits (SplatPager.keepUnplaced). */
  protected keepUnplaced = true;

  protected allocateFreeable(): number | undefined {
    const page = super.allocateFreeable();
    if (page !== undefined) this.bandStats.evicted += 1;
    return page;
  }

  removeSplats(splats: PagedSplats) {
    super.removeSplats(splats);
    this.tables.delete(splats);
    this.kids.delete(splats);
  }
}
