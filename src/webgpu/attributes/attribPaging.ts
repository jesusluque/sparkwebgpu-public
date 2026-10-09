// How the paged attribute pool of WgpuSplatPager is sized, stream by stream.
//
// The core page pool holds maxPages pages of 65 536 splats. Attributes used
// to be all or nothing: every one at that capacity, or none when they did
// not fit one storage binding (a .athc TX cloud wants 300 bytes a splat:
// 4.8 GB at 256 pages). Now each *group* of streams gets its own capacity
// inside a byte budget (tiers.ts attribBudget, capped by the device's
// binding size):
//
//   core      every attribute outside the groups below (a .rad's, the
//             .athc's finest-group ranges `athcGroup`): always, at full
//             capacity, interleaved as before
//   material  normalOct, emission, pbr, lobes (.athc v3 section MATL,
//             data tier 2): cheap, full capacity when the budget allows
//   relight   shadowBits, curvature and transfer (SHAD, CURV, TXDI/TXIN/TXFD,
//             data tier 3):
//             the expensive ones, with as many pages as the rest of the
//             budget holds; device tier 2 and up
//
// A group at full capacity is interleaved in the pool and arrives with its
// pages. A group with fewer pages than the pool is *paged*: its own region
// with a page table (slang/core/attrib.slang ATTRIB_PAGED), filled for the
// pages the LoD traversal ranks first (AttribResidency), its data fetched
// on its own (.athc v3: one Range request of its sections). Its other pages
// read as not resident, and the shaders fall back.
//
// The transfer can also be kept shorter (`transferForm`): its layouts are
// prefixes of each other (transfer_layout.slang), 16 direct values, 64
// with the indirect half, 112 with the reflected field (9, 36, 84 at
// degree 2); a shorter form is a different layout, read as such.

import { tierSpec } from "../plugins/tiers";
import type { Tier } from "../plugins/types";
import { type AttributeSpec, attribWords } from "./schema";

export const PAGE_SPLATS = 65536;

/** A group of streams paged together, as a .athc v3 data tier is fetched. */
export interface StreamGroupSpec {
  name: string;
  attributes: readonly string[];
  /** The .athc v3 data tier of its sections (1 arrives with the splats). */
  dataTier: number;
  /** Least device tier that loads it. */
  minTier: Tier;
}

export const STREAM_GROUPS: readonly StreamGroupSpec[] = [
  {
    name: "material",
    attributes: ["normalOct", "emission", "pbr", "lobes"],
    dataTier: 2,
    minTier: 1,
  },
  {
    name: "relight",
    attributes: ["shadowBits", "curvature", "transfer", "transferCpca"],
    dataTier: 3,
    minTier: 2,
  },
];

export const CORE_GROUP = "core";

export type TransferForm = "full" | "indirect" | "direct";

export interface AttribPagingOptions {
  /** Bytes for the attribute pool (default: the tier's, capped by the binding size). */
  budgetBytes?: number;
  /**
   * Pages a group keeps, by group name ("material", "relight", "core"):
   * at most the pool's. Groups not named get what the budget leaves.
   */
  pages?: Record<string, number>;
  /**
   * Which streams of the groups to load: "all", or names. By default the
   * PluginHost's (the attributes its plugins require or read) and, with
   * no host, the material ones.
   */
  attributes?: "all" | readonly string[];
  /** The transfer kept (default "full"). */
  transferForm?: TransferForm;
  /** Overrides the device tier (tierOfLimits). */
  tier?: Tier;
}

export interface PlannedGroup {
  name: string;
  specs: AttributeSpec[];
  /** Pages it holds; the pool's for an interleaved group. */
  slots: number;
  /** Fewer pages than the pool: in its own region, with a page table. */
  paged: boolean;
  dataTier: number;
  /** Bytes a splat. */
  splatBytes: number;
  /** Bytes it takes in the pool (tables included). */
  bytes: number;
}

export interface AttribPagingPlan {
  maxPages: number;
  /** The pool's attributes, in descriptor order. */
  specs: AttributeSpec[];
  groups: PlannedGroup[];
  /** Total bytes of the pool, header included. */
  bytes: number;
  budget: number;
  /** Attributes left out, and why. */
  dropped: { name: string; reason: "demand" | "tier" | "budget" }[];
}

/** The forms a transfer of `count` values can be kept as, shortest first (athc_v3.rs transfer_forms). */
export function transferForms(count: number): number[] {
  const d = count === 10 ? 10 : [16, 64, 112].includes(count) ? 16 : 9;
  if (count === 10 || count === d) return [count];
  return count > 4 * d ? [d, 4 * d, count] : [d, 4 * d];
}

export function transferFormValues(count: number, form: TransferForm) {
  const forms = transferForms(count);
  if (form === "direct") return forms[0];
  if (form === "indirect") return forms[Math.min(1, forms.length - 1)];
  return count;
}

export function groupOf(name: string): StreamGroupSpec | undefined {
  return STREAM_GROUPS.find((g) => g.attributes.includes(name));
}

function specBytes(spec: AttributeSpec) {
  return attribWords(spec) * 4;
}

/** Words of a pool's header for `n` attributes (schema.ts poolLayout). */
function headerBytes(n: number) {
  return (4 + 4 * n) * 4;
}

/**
 * Sizes the attribute pool of `maxPages` pages for `specs` (the cloud's
 * schema): which streams it keeps, at how many pages each group, within
 * `budget` bytes. `demand` is the streams of the groups to load (null: the
 * default of AttribPagingOptions.attributes); `tier` the device's.
 * `cloudPages`, the pages of the cloud the schema is from: a paged group
 * never takes more slots than that (a pawn of 16 pages does not reserve 79
 * pages of transfer because the budget would hold them).
 */
export function planAttribPaging(
  specs: readonly AttributeSpec[],
  maxPages: number,
  {
    budget,
    tier,
    demand = null,
    pages = {},
    transferForm = "full",
    cloudPages,
  }: {
    budget: number;
    tier: Tier;
    demand?: "all" | readonly string[] | null;
    pages?: Record<string, number>;
    transferForm?: TransferForm;
    cloudPages?: number;
  },
): AttribPagingPlan {
  const dropped: AttribPagingPlan["dropped"] = [];
  const wanted = (name: string) => {
    // transferCpca stands for the transfer: loaded only when named.
    if (demand === "all") return name !== "transferCpca";
    if (demand) return demand.includes(name);
    return groupOf(name)?.name !== "relight";
  };
  const byGroup = new Map<string, AttributeSpec[]>();
  for (const s of specs) {
    const group = groupOf(s.name);
    if (group && !wanted(s.name)) {
      dropped.push({ name: s.name, reason: "demand" });
      continue;
    }
    if (group && tier < group.minTier && pages[group.name] === undefined) {
      dropped.push({ name: s.name, reason: "tier" });
      continue;
    }
    let spec = s;
    if (s.name === "transfer") {
      const values = transferFormValues(s.components, transferForm);
      if (values !== s.components) spec = { ...s, components: values };
    }
    // The transfer as clusters (transferCpca): the form's record, a prefix.
    if (s.cpcaForms?.length) {
      const full = s.cpcaForms[s.cpcaForms.length - 1].values;
      const values = transferFormValues(full, transferForm);
      const words = s.cpcaForms.find((f) => f.values === values)?.words;
      if (words !== undefined && words !== s.components) {
        spec = { ...s, components: words };
      }
    }
    const key = group?.name ?? CORE_GROUP;
    byGroup.set(key, [...(byGroup.get(key) ?? []), spec]);
  }
  // Core first, then by data tier: what arrives with the splats is never
  // starved by what is fetched on its own.
  const order = [
    CORE_GROUP,
    ...STREAM_GROUPS.map((g) => g.name).filter((n) => byGroup.has(n)),
  ].filter((n) => byGroup.has(n));
  const total = [...byGroup.values()].reduce((n, g) => n + g.length, 0);
  let left = budget - headerBytes(total);
  const pageSplats = PAGE_SPLATS;
  const groups: PlannedGroup[] = [];
  for (const name of order) {
    const groupSpecs = byGroup.get(name) as AttributeSpec[];
    const splatBytes = groupSpecs.reduce((n, s) => n + specBytes(s), 0);
    const full = maxPages * pageSplats * splatBytes;
    const tables = 4 * maxPages * groupSpecs.length;
    const asked = pages[name];
    let slots: number;
    if (asked !== undefined) {
      slots = Math.max(0, Math.min(Math.floor(asked), maxPages));
    } else if (full <= left) {
      slots = maxPages;
    } else {
      slots = Math.max(
        0,
        Math.min(
          maxPages - 1,
          Math.floor((left - tables) / (pageSplats * splatBytes)),
        ),
      );
    }
    const paged = slots < maxPages;
    if (paged && asked === undefined && cloudPages !== undefined) {
      slots = Math.min(slots, cloudPages);
    }
    if (paged && slots === 0 && asked === undefined) {
      for (const s of groupSpecs)
        dropped.push({ name: s.name, reason: "budget" });
      continue;
    }
    const bytes = paged ? tables + slots * pageSplats * splatBytes : full;
    left -= bytes;
    groups.push({
      name,
      specs: groupSpecs,
      slots,
      paged,
      dataTier: groupOf(groupSpecs[0].name)?.dataTier ?? 1,
      splatBytes,
      bytes,
    });
  }
  const kept = groups.flatMap((g) => g.specs);
  const bytes =
    headerBytes(kept.length) + groups.reduce((n, g) => n + g.bytes, 0);
  return { maxPages, specs: kept, groups, bytes, budget, dropped };
}

/** The default budget of a device: its tier's, within one storage binding. */
export function attribBudget(device: GPUDevice, tier: Tier): number {
  const limit = Math.min(
    device.limits.maxStorageBufferBindingSize,
    device.limits.maxBufferSize,
  );
  return Math.min(limit, tierSpec(tier).attribBudget);
}
