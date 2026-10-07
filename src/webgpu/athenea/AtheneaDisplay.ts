// athenea's display transform as an aofx effect (slang/athenea_adapter/
// display.slang): a linear premultiplied Rec.709 picture over a background,
// exposed in stops, through a view transform -- Standard, AgX or ACES 2.0 --
// into a display's gamut and transfer function. What `athenea view` shows,
// as its DisplayTransform (technique/display.slang) computes it.
//
// It is the output stage of the athenea raster: the splats are blended in
// linear light into a float target (atheneaRasterPlugin), and this turns
// that light into display code values, last in an FxChain:
//
//   chain.add(new AtheneaDisplay(), { view: 1, exposure: 0.5 });
//   chain.applyToRenderTarget(renderer, hdrTarget);
//
// By default the result is written for an output that sRGB-encodes what it
// is given (three's canvas output, an -srgb texture): the code value, with
// the sRGB curve undone, so the encode lands on it. `encoded` writes the
// code value itself, for a unorm texture shown as it is.

import { Effect, type RenderRequest } from "../fx/Effect";
import type { Gpu } from "../fx/Gpu";
import { COLOR_PLANE, type EffectDesc, isValidBuffer } from "../fx/types";
import displayModule from "../generated/athenea_adapter/display";
import { createStorage } from "../gpuBuffers";
import { UniformWriter } from "../uniforms";

/** athenea's ViewTransform (Ocio, 3, is not here: OpenColorIO is native only). */
export const ATHENEA_VIEWS = ["standard", "agx", "aces2"] as const;
export type AtheneaView = (typeof ATHENEA_VIEWS)[number];
/** athenea's DisplayEncoding. */
export const ATHENEA_DISPLAYS = [
  "srgb",
  "rec709",
  "displayP3",
  "linearP3",
] as const;
export type AtheneaDisplayEncoding = (typeof ATHENEA_DISPLAYS)[number];

const clamp = (v: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, v));

/** The ACES 2.0 parameters and tables for one peak and set of primaries. */
interface AcesTables {
  params: GPUBuffer;
  tables: GPUBuffer;
  key: string;
}

const ACES_TABLE_WORDS = 5 * 362; // aces2.slang kTableWords
const acesByGpu = new WeakMap<Gpu, AcesTables>();

function acesFor(gpu: Gpu): AcesTables {
  let aces = acesByGpu.get(gpu);
  if (!aces) {
    const bytes =
      displayModule.reflection.bindings.find((b) => b.name === "displayAces")
        ?.elementBytes ?? 4096;
    aces = {
      params: createStorage(gpu.device, bytes, "athenea aces2 params"),
      tables: createStorage(
        gpu.device,
        ACES_TABLE_WORDS * 4,
        "athenea aces2 tables",
      ),
      key: "",
    };
    acesByGpu.set(gpu, aces);
  }
  return aces;
}

/**
 * Builds the ACES 2.0 tables (athenea's aces2_prepare passes) for `peak`
 * nits and the limiting primaries, unless they are built for those already.
 * Recorded into the Gpu's frame, before whatever reads them.
 */
export function prepareAces2(
  gpu: Gpu,
  peakLuminance: number,
  limiting: "rec709" | "p3d65",
): AcesTables {
  const aces = acesFor(gpu);
  const key = `${peakLuminance}:${limiting}`;
  if (aces.key === key) return aces;
  const u = UniformWriter.for(displayModule, "prepare").setAll({
    peakLuminance,
    limiting: limiting === "p3d65" ? 1 : 0,
  });
  const buffers = {
    acesParamsOut: aces.params,
    acesParamsIn: aces.params,
    acesTablesOut: aces.tables,
  };
  gpu.run(gpu.load(displayModule, "acesPrepareParams"), [1], buffers, u);
  gpu.run(gpu.load(displayModule, "acesPrepareTables"), [360], buffers);
  gpu.run(gpu.load(displayModule, "acesPrepareWrap"), [1], buffers);
  aces.key = key;
  return aces;
}

export class AtheneaDisplay extends Effect {
  protected describeEffect(): EffectDesc {
    return {
      identifier: "athenea.display",
      label: "athenea Display",
      grouping: "Colour",
      description:
        "athenea's display transform: linear premultiplied Rec.709 over a background, times 2^exposure, through Standard, AgX or ACES 2.0, into sRGB, BT.1886, Display P3 or linear P3.",
      inputs: [{ name: "Source", label: "Source", passThrough: true }],
      outputs: [COLOR_PLANE],
      params: [
        {
          name: "exposure",
          label: "Exposure",
          type: "double",
          defaults: [0],
          displayMin: [-8],
          displayMax: [8],
          hint: "In stops: the light is multiplied by 2^exposure before the view transform.",
        },
        {
          name: "view",
          label: "View",
          type: "choice",
          choices: [
            { value: "standard", label: "Standard" },
            { value: "agx", label: "AgX" },
            { value: "aces2", label: "ACES 2.0" },
          ],
          defaults: [1],
          hint: "Standard clips at the display's white; AgX is Sobotka's sigmoid (Wrensch's fit); ACES 2.0 is the Academy's output transform.",
        },
        {
          name: "display",
          label: "Display",
          type: "choice",
          choices: [
            { value: "srgb", label: "sRGB" },
            { value: "rec709", label: "Rec.709 (BT.1886)" },
            { value: "displayP3", label: "Display P3" },
            { value: "linearP3", label: "Linear P3 (extended)" },
          ],
          defaults: [0],
          hint: "The display's primaries and transfer function.",
        },
        {
          name: "background",
          label: "Background",
          type: "colour",
          dimension: 3,
          defaults: [0, 0, 0],
          hint: "Linear light under what the picture leaves uncovered (its alpha).",
        },
        {
          name: "peakLuminance",
          label: "Peak (nits)",
          type: "double",
          defaults: [100],
          displayMin: [100],
          displayMax: [4000],
          hardMin: [1],
          hint: "ACES 2.0's peak luminance: 100 for a standard display. Linear P3 keeps peak / 100 of headroom over reference white.",
        },
        {
          name: "encoded",
          label: "Encoded output",
          type: "boolean",
          defaults: [0],
          hint: "Write the code value as it is (a unorm texture shown directly) rather than for an output that sRGB-encodes it (three's canvas).",
        },
      ],
    };
  }

  process(request: RenderRequest): boolean {
    const source = request.input("Source")?.buffer;
    const target = request.output("Color")?.buffer;
    if (!isValidBuffer(source) || !isValidBuffer(target)) return false;
    const gpu = request.gpu;
    const view = clamp(Math.round(request.number("view", 1)), 0, 2);
    const display = clamp(Math.round(request.number("display", 0)), 0, 3);
    const aces =
      view === 2
        ? prepareAces2(
            gpu,
            Math.max(1, request.number("peakLuminance", 100)),
            display >= 2 ? "p3d65" : "rec709",
          )
        : acesFor(gpu);
    const u = UniformWriter.for(displayModule, "displayParams").setAll({
      width: target.width,
      height: target.height,
      srcStride: source.stride,
      dstStride: target.stride,
      srcOffsetX: target.rect.x1 - source.rect.x1,
      srcOffsetY: target.rect.y1 - source.rect.y1,
      srcWidth: source.width,
      srcHeight: source.height,
      view,
      display,
      exposure: request.number("exposure", 0),
      // DisplayTransform.cpp: the headroom is the peak over 100 nits.
      peakScale: Math.max(request.number("peakLuminance", 100), 1) / 100,
      backgroundR: request.number("background", 0, 0),
      backgroundG: request.number("background", 0, 1),
      backgroundB: request.number("background", 0, 2),
      forSrgbOutput: request.number("encoded", 0) < 0.5,
    });
    return gpu.run(
      gpu.load(displayModule, "atheneaDisplay"),
      [target.width, target.height, 1],
      {
        src: source,
        dst: target,
        displayAces: aces.params,
        displayAcesTables: aces.tables,
      },
      u,
    );
  }
}
