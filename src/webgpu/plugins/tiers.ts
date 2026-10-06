// Device tiers for plugins, from capabilities.ts (plan-athenea.md §3):
//
//   T0  no usable WebGPU splat path: Spark's WebGL renderer
//   T1  the spec's default limits (128 MiB storage bindings): standard SH,
//       up to 1.25M splats
//   T2  raised limits (1 GiB bindings and buffers): TX relighting, 2.5M
//   T3  the largest (2 GiB and up): everything, 4.5M and more
//
// A device only has the limits it asked for: three's WebGPURenderer asks for
// none unless given requiredLimits (splatRequiredLimits(adapter)), so the
// same machine measures T1 or T3 depending on how its device was made.

import type { GpuCapabilities } from "../capabilities";
import type { Tier } from "./types";

const MIB = 1024 * 1024;

export interface TierSpec {
  tier: Tier;
  /** Least maxStorageBufferBindingSize and maxBufferSize. */
  minBindingBytes: number;
  /** Splats the tier is sized for (frame-time budgets start from this). */
  splatBudget: number;
  /**
   * Bytes the paged attribute pool may take (WgpuSplatPager, one buffer and
   * binding: never more than the device's binding size). Streams that do
   * not fit at the page pool's capacity are paged with fewer pages
   * (src/webgpu/attributes/attribPaging.ts).
   */
  attribBudget: number;
}

export const TIERS: readonly TierSpec[] = [
  { tier: 0, minBindingBytes: 0, splatBudget: 0, attribBudget: 0 },
  {
    tier: 1,
    minBindingBytes: 128 * MIB,
    splatBudget: 1_250_000,
    attribBudget: 128 * MIB,
  },
  {
    tier: 2,
    minBindingBytes: 1024 * MIB,
    splatBudget: 2_500_000,
    attribBudget: 768 * MIB,
  },
  {
    tier: 3,
    minBindingBytes: 2048 * MIB - 4,
    splatBudget: 4_500_000,
    attribBudget: 1536 * MIB,
  },
];

/** The tier of a device's capabilities; null (no WebGPU) is T0. */
export function tierOf(caps: GpuCapabilities | null | undefined): Tier {
  if (!caps || !caps.vertexStorage) return 0;
  const lim = (k: string, fallback: number) => caps.limits[k] ?? fallback;
  if (lim("maxStorageBuffersPerShaderStage", 8) < 8) return 0;
  const bytes = Math.min(
    lim("maxStorageBufferBindingSize", 128 * MIB),
    lim("maxBufferSize", 256 * MIB),
  );
  let tier: Tier = 0;
  for (const t of TIERS) {
    if (bytes >= t.minBindingBytes) tier = t.tier;
  }
  return tier;
}

/**
 * The tier of a device's limits alone (a GPUDevice's `limits`), for code
 * that has a device but no GpuCapabilities: T1 to T3 by binding size.
 */
export function tierOfLimits(limits: {
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
}): Tier {
  const bytes = Math.min(
    limits.maxStorageBufferBindingSize,
    limits.maxBufferSize,
  );
  let tier: Tier = 1;
  for (const t of TIERS) {
    if (t.tier > 0 && bytes >= t.minBindingBytes) tier = t.tier;
  }
  return tier;
}

export function tierSpec(tier: Tier): TierSpec {
  return TIERS[tier];
}

/**
 * Whether the device has `feature`: a GpuCapabilities flag ("subgroups",
 * "timestampQuery", "gpuSort", "shaderF16"...) or a WebGPU feature name.
 */
export function hasFeature(
  caps: GpuCapabilities | null | undefined,
  feature: string,
): boolean {
  if (!caps) return false;
  const flag = (caps as unknown as Record<string, unknown>)[feature];
  if (typeof flag === "boolean") return flag;
  return caps.features.has(feature);
}
