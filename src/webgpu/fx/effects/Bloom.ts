// Bloom and glow (slang/fx/bloom.slang): the light above a threshold spread
// by a pyramid of halvings summed back up, added to the picture. Run it on
// linear light, before a display transform. A bloom is a high threshold and
// a modest reach; a glow is a low threshold and a wide reach.

import bloomModule from "../../generated/fx/bloom";
import { UniformWriter } from "../../uniforms";
import { Effect, type RegionQuery, type RenderRequest } from "../Effect";
import {
  COLOR_PLANE,
  type EffectDesc,
  type FxBuffer,
  type Rect,
  isValidBuffer,
  paramNumber,
  unionRects,
} from "../types";

/** The deepest pyramid: a 4k frame halved nine times is 8 pixels tall. */
export const BLOOM_MAX_LEVELS = 9;

/**
 * Levels for a reach of `size` (a fraction of the picture's height): the
 * last level's pixel is about that wide.
 */
export function bloomLevels(size: number, width: number, height: number) {
  const reach = Math.max(size, 0) * height;
  let levels = Math.round(Math.log2(Math.max(reach, 2)));
  levels = Math.min(Math.max(levels, 1), BLOOM_MAX_LEVELS);
  // Never below two pixels on either axis.
  const small = Math.min(width, height);
  return Math.max(1, Math.min(levels, Math.floor(Math.log2(small)) - 1));
}

export class Bloom extends Effect {
  protected describeEffect(): EffectDesc {
    return {
      identifier: "org.aopenfx.bloom",
      label: "Bloom",
      grouping: "Filter",
      description:
        "Light above a threshold spreads into its surroundings, as through a lens: a pyramid of 13-tap halvings summed back up with a tent. Scene-referred: apply before the display transform.",
      inputs: [{ name: "Source", label: "Source", passThrough: true }],
      outputs: [COLOR_PLANE],
      params: [
        {
          name: "threshold",
          label: "Threshold",
          hint: "Light (linear, the picture's units) above which it blooms. Zero is a glow of the whole picture.",
          type: "double",
          defaults: [1],
          hardMin: [0],
          displayMax: [20],
        },
        {
          name: "knee",
          label: "Knee",
          hint: "How far below the threshold the bloom fades in, as a fraction of it.",
          type: "double",
          defaults: [0.5],
          hardMin: [0],
          hardMax: [1],
        },
        {
          name: "intensity",
          label: "Intensity",
          hint: "How much of the spread light is added. Zero is the input, not rendered.",
          type: "double",
          defaults: [0.2],
          hardMin: [0],
          displayMax: [2],
        },
        {
          name: "size",
          label: "Size",
          hint: "Reach of the widest level, as a fraction of the picture's height.",
          type: "double",
          defaults: [0.25],
          hardMin: [0],
          displayMax: [1],
        },
        {
          name: "tint",
          label: "Tint",
          hint: "Colour of the added light.",
          type: "colour",
          dimension: 3,
          defaults: [1, 1, 1],
        },
        {
          name: "clamp",
          label: "Clamp",
          hint: "No pixel brighter than this enters the bloom.",
          type: "double",
          defaults: [65504],
          hardMin: [0],
          displayMax: [65504],
        },
      ],
    };
  }

  regionOfDefinition(q: RegionQuery): Rect {
    return unionRects(q.inputRods);
  }

  /** The bloom at a pixel gathers light from the whole picture. */
  regionOfInterest(q: RegionQuery): Rect[] {
    const all = unionRects(q.inputRods);
    return Array.from({ length: Math.max(q.inputRods.length, 1) }, () => all);
  }

  isIdentity(request: RenderRequest) {
    return request.number("intensity", 0.2) <= 0;
  }

  process(request: RenderRequest): boolean {
    const source = request.input("Source")?.buffer;
    const target = request.output()?.buffer;
    if (!isValidBuffer(source) || !isValidBuffer(target)) return false;
    const gpu = request.gpu;
    const down = gpu.load(bloomModule, "bloomDown");
    const up = gpu.load(bloomModule, "bloomUp");
    const p = request.params;
    const threshold = Math.max(paramNumber(p, "threshold", 1), 0);
    const knee = Math.max(paramNumber(p, "knee", 0.5), 0) * threshold;
    const intensity = Math.max(paramNumber(p, "intensity", 0.2), 0);
    const tint = [0, 1, 2].map((k) => paramNumber(p, "tint", 1, k));
    const clampMax = Math.max(paramNumber(p, "clamp", 65504), 0);
    const levels = bloomLevels(
      paramNumber(p, "size", 0.25),
      target.width,
      target.height,
    );

    const uniforms = (
      from: FxBuffer,
      into: FxBuffer,
      extra: Record<string, number | boolean> = {},
    ) =>
      UniformWriter.for(bloomModule).setAll({
        srcWidth: from.width,
        srcHeight: from.height,
        srcStride: from.stride,
        srcOffsetX: 0,
        srcOffsetY: 0,
        auxWidth: 1,
        auxHeight: 1,
        auxStride: 1,
        dstWidth: into.width,
        dstHeight: into.height,
        dstStride: into.stride,
        prefilter: false,
        threshold,
        // A knee of zero is a hard cut; the kernel keeps it off zero.
        knee: Math.max(knee, 1e-4),
        clampMax,
        srcScale: 1,
        auxScale: 1,
        tintR: 1,
        tintG: 1,
        tintB: 1,
        ...extra,
      });

    // Down: level k is the picture halved k times (level 0 the picture).
    const level: FxBuffer[] = [];
    let w = target.width;
    let h = target.height;
    for (let k = 1; k <= levels; k++) {
      w = Math.ceil(w / 2);
      h = Math.ceil(h / 2);
      const into = gpu.scratch(w, h);
      const from = k === 1 ? source : level[k - 2];
      const extra: Record<string, number | boolean> =
        k === 1
          ? {
              prefilter: true,
              srcOffsetX: target.rect.x1 - source.rect.x1,
              srcOffsetY: target.rect.y1 - source.rect.y1,
            }
          : {};
      if (
        !gpu.run(
          down,
          [w, h, 1],
          { src: from, dst: into },
          uniforms(from, into, extra),
        )
      ) {
        return false;
      }
      level.push(into);
    }

    // Up: each level plus the tent of the one below it, from the smallest.
    let coarse = level[levels - 1];
    for (let k = levels - 1; k >= 1; k--) {
      const fine = level[k - 1];
      const into = gpu.scratch(fine.width, fine.height);
      const u = uniforms(fine, into, {
        auxWidth: coarse.width,
        auxHeight: coarse.height,
        auxStride: coarse.stride,
      });
      if (
        !gpu.run(
          up,
          [into.width, into.height, 1],
          { src: fine, aux: coarse, dst: into },
          u,
        )
      ) {
        return false;
      }
      coarse = into;
    }

    // Onto the picture: the sum of `levels` levels, averaged, times the
    // intensity and tint.
    return gpu.run(
      up,
      [target.width, target.height, 1],
      { src: source, aux: coarse, dst: target },
      uniforms(source, target, {
        srcOffsetX: target.rect.x1 - source.rect.x1,
        srcOffsetY: target.rect.y1 - source.rect.y1,
        auxWidth: coarse.width,
        auxHeight: coarse.height,
        auxStride: coarse.stride,
        auxScale: intensity / levels,
        tintR: tint[0],
        tintG: tint[1],
        tintB: tint[2],
      }),
    );
  }
}
