// Picking athenea's ids through the existing MRT path: SplatAttributes'
// id target names the front-most splat at a pixel (alpha >= pickAlpha, after
// the raster plugin's overrides, so a hidden range is picked through), and
// the splat's athcGroup attribute says which of athenea's finest LoD groups
// it is -- the id a .athc carries in place of a Cryptomatte id, and the key
// of atheneaRasterPlugin's overrides.
//
//   const attributes = attributesPlugin({ targets: { id: true } });
//   ...
//   const hit = await pickAtheneaGroup(attributes.stage, x, y);
//   if (hit) raster.setOverrides(hit.mesh, [{ groups: hit.groups, opacity: 0 }]);

import type { WgpuSplatMesh } from "../WgpuSplatRenderer";
import type {
  PickResult,
  SplatAttributes,
} from "../attributes/SplatAttributes";
import { attribWords, poolLayout } from "../attributes/schema";
import { createReadback, readAndDestroy } from "../gpuBuffers";
import { ATHC_GROUP_ATTRIBUTE } from "./rasterPlugin";

export interface AtheneaPick extends PickResult {
  /** Finest groups [lo, hi) of the picked splat: a splat's own is one. */
  groups: [number, number];
}

/**
 * `mesh`'s finest-group range of source splat `index`: from the pool's
 * columns, or read back from the GPU for a paged pool. Null when the mesh
 * carries no athcGroup.
 */
export async function atheneaGroupOf(
  device: GPUDevice,
  mesh: WgpuSplatMesh,
  index: number,
): Promise<[number, number] | null> {
  const pool = mesh.source.attribs;
  const col = pool?.column(ATHC_GROUP_ATTRIBUTE);
  if (!pool || !col) return null;
  if (col.words.length >= (index + 1) * attribWords(col.spec)) {
    const [lo, hi] = pool.getAttribute(ATHC_GROUP_ATTRIBUTE, index);
    return [lo, hi];
  }
  if (!pool.gpuBuffer) return null;
  // A paged pool lives on the GPU only: its layout is the schema's.
  const layout = poolLayout(pool.schema);
  const k = pool.id(ATHC_GROUP_ATTRIBUTE);
  const word =
    layout.headerWords + index * layout.strideWords + layout.offsets[k];
  const staging = createReadback(device, 8, "athcGroup readback");
  const encoder = device.createCommandEncoder({ label: "athcGroup readback" });
  encoder.copyBufferToBuffer(pool.gpuBuffer, word * 4, staging, 0, 8);
  device.queue.submit([encoder.finish()]);
  const [lo, hi] = new Uint32Array(await readAndDestroy(staging), 0, 2);
  return [lo, hi];
}

/** The front-most splat at pixel (x, y) and its finest groups, or null. */
export async function pickAtheneaGroup(
  attributes: SplatAttributes,
  x: number,
  y: number,
): Promise<AtheneaPick | null> {
  const hit = await attributes.pick(x, y);
  if (!hit) return null;
  const groups = await atheneaGroupOf(attributes.device, hit.mesh, hit.index);
  return groups ? { ...hit, groups } : null;
}
