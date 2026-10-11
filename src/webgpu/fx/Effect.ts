// aofx's `Effect` and `RenderRequest` (sdk/include/aofx/Effect.h).
//
// One Effect instance serves every node that uses it: it holds no per-node
// state, because the node's state arrives in the request. Six methods, three
// of them optional:
//
//   describe()              what it is: inputs, parameters, outputs
//   process(request)        do it, through request.gpu
//   isIdentity(request)     nothing to do; the host hands the input back
//   regionOfDefinition(q)   how big the picture is (default: union of inputs)
//   regionOfInterest(q, r)  how much of each input `r` needs (default: `r`)

import type { Gpu } from "./Gpu";
import {
  type EffectDesc,
  type FxBuffer,
  type ParamValues,
  type Rect,
  hasParam,
  paramNumber,
  unionRects,
} from "./types";

/** What a region question carries, before any pixel exists. */
export interface RegionQuery {
  readonly time: number;
  /** Render scale; parameters in pixels are canonical and multiply by it. */
  readonly scaleX: number;
  readonly scaleY: number;
  /** Each input's region of definition, in clip order; empty when unwired. */
  readonly inputRods: readonly Rect[];
  readonly params: ParamValues;
}

export interface InputPlane {
  readonly clip: string;
  readonly plane: string;
  readonly buffer: FxBuffer;
  /** The input's whole region of definition; the buffer may be less. */
  readonly rod: Rect;
  /** Numbers that travelled with the picture (a track, a measurement). */
  readonly values?: ReadonlyMap<string, readonly number[]>;
}

export interface OutputPlane {
  readonly plane: string;
  readonly buffer: FxBuffer;
}

export class RenderRequest {
  time = 0;
  scaleX = 1;
  scaleY = 1;
  /** The node's stable name; key `keep` buffers on it, not on the effect. */
  instance = "";
  renderWindow: Rect;
  outputRod: Rect;
  projectWidth = 0;
  projectHeight = 0;
  inputs: InputPlane[] = [];
  outputs: OutputPlane[] = [];
  params: ParamValues = {};
  /** Numbers this render hangs on its picture, for nodes downstream. */
  readonly produced = new Map<string, readonly number[] | Promise<number[]>>();
  /** Why a refused render was refused, for the host to show. */
  complaint = "";

  constructor(
    readonly gpu: Gpu,
    window: Rect,
  ) {
    this.renderWindow = window;
    this.outputRod = window;
  }

  input(clip: string, plane = "Color"): InputPlane | undefined {
    return this.inputs.find((i) => i.clip === clip && i.plane === plane);
  }

  output(plane = "Color"): OutputPlane | undefined {
    return this.outputs.find((o) => o.plane === plane);
  }

  has(name: string) {
    return hasParam(this.params, name);
  }

  number(name: string, fallback = 0, component = 0): number {
    return paramNumber(this.params, name, fallback, component);
  }

  attach(id: string, values: readonly number[] | Promise<number[]>) {
    this.produced.set(id, values);
  }
}

export abstract class Effect {
  private described: EffectDesc | null = null;

  protected abstract describeEffect(): EffectDesc;

  /** Answered once and cached: describe() cannot know about any one node. */
  describe(): EffectDesc {
    this.described ??= this.describeEffect();
    return this.described;
  }

  /** Default: the union of the inputs, right for anything that moves no pixel. */
  regionOfDefinition(query: RegionQuery): Rect {
    return unionRects(query.inputRods);
  }

  /** Default: the output's own rectangle of every input. */
  regionOfInterest(query: RegionQuery, output: Rect): Rect[] {
    return query.inputRods.map(() => output);
  }

  isIdentity(_request: RenderRequest): boolean {
    return false;
  }

  /**
   * Pixel for pixel, each output pixel from the same pixel of the
   * pass-through input alone: the host may then hand process() that input's
   * own picture as the output (when nothing reads it after), and process()
   * must write it in place (its output buffer is its input's). Saves a
   * picture a frame (6 MB on a phone). Default false.
   */
  get inPlace(): boolean {
    return false;
  }

  abstract process(request: RenderRequest): boolean;
}
