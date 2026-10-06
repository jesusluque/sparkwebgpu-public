// Back-to-front splat order on the GPU with slang/kernels/sort_radix.slang:
// keys from the generate pass's sort metric, compacted to the active splats,
// eight stable 4-bit radix passes dispatched over those alone, and the active
// count written into indirect draw arguments, all recorded into one compute
// pass with no readback.

import type { GpuProfiler } from "./GpuProfiler";
import type { KernelRegistry } from "./KernelRegistry";
import { kernelsSortRadix } from "./generated/constants";
import sortModule from "./generated/kernels/sort_radix";
import scatterSubgroupModule from "./generated/kernels/sort_scatter_subgroup";
import { createStorage as storage } from "./gpuBuffers";
import { UniformWriter } from "./uniforms";

const { TILE, SCAN_CHUNK, BINS, KEY_DEPTH, KEY_FLOAT, DEPTH_BITS, DEPTH_KEYS } =
  kernelsSortRadix;

/**
 * The range every active metric lies in (generate's cull keeps the view
 * depth between the camera's planes): the sort keys the metrics by their
 * place in it on DEPTH_BITS (24) bits, 6 radix passes, instead of their
 * float bits (sort_radix.slang KEY_DEPTH). Metrics outside are clamped.
 */
export interface SortDepthRange {
  lo: number;
  hi: number;
}

export class GpuSorter {
  readonly device: GPUDevice;
  /** Indirect draw arguments: 4 vertices, one instance per active splat. */
  readonly drawArgs: GPUBuffer;
  /** Workgroup counts of the radix passes: one per TILE active keys. */
  readonly dispatchArgs: GPUBuffer;
  private capacity = 0;
  private keys: GPUBuffer[] = [];
  private vals: GPUBuffer[] = [];
  // blockHist, then the chunk sums of each scan level.
  private scanLevels: GPUBuffer[] = [];
  private empty: GPUBuffer;

  /**
   * `label` prefixes the profiler's stage labels. `subgroups`: the scatter
   * ranks keys with subgroup ballots (default: where the device has them).
   */
  constructor(
    readonly registry: KernelRegistry,
    readonly label = "sort",
    readonly subgroups = registry.device.features.has("subgroups"),
  ) {
    this.device = registry.device;
    this.drawArgs = storage(
      this.device,
      16,
      "sort draw args",
      GPUBufferUsage.INDIRECT,
    );
    this.empty = storage(this.device, 16, "sort empty");
    this.dispatchArgs = storage(
      this.device,
      16,
      "sort dispatch args",
      GPUBufferUsage.INDIRECT,
    );
  }

  /** The sorted splat indices; valid for the first drawArgs[1] entries. */
  get ordering(): GPUBuffer {
    return this.vals[0];
  }

  private ensure(count: number) {
    if (count <= this.capacity) return;
    this.destroyBuffers();
    this.capacity = Math.max(count, Math.ceil(this.capacity * 1.5));
    const n = this.capacity;
    for (const k of [0, 1]) {
      this.keys.push(storage(this.device, n * 4, `sort keys ${k}`));
      this.vals.push(storage(this.device, n * 4, `sort vals ${k}`));
    }
    this.scanLevels = createScanLevels(
      this.device,
      BINS * Math.ceil(n / TILE),
      "sort block hist",
    );
  }

  /**
   * Records the sort of `metric[0..count)` into `pass`, on the top `bits` of
   * the key (16, 24 or 32: 4, 6 or 8 passes; even, so the result lands in
   * `ordering`). Dropping low bits only merges splats whose metrics differ
   * by less than 2^-7 (16) or 2^-15 (24) relative.
   */
  encode(
    pass: GPUComputePassEncoder,
    metric: GPUBuffer,
    count: number,
    bits: 16 | 24 | 32 = 32,
    depth?: SortDepthRange | null,
  ) {
    this.encodeStages(() => pass, metric, count, bits, depth);
  }

  /**
   * As encode, with one compute pass per stage on `encoder` so a profiler
   * can time each (labels sort.prepare, sort.histogram, sort.scan,
   * sort.scatter; summed over the radix passes).
   */
  encodeProfiled(
    encoder: GPUCommandEncoder,
    profiler: GpuProfiler,
    metric: GPUBuffer,
    count: number,
    bits: 16 | 24 | 32 = 32,
    depth?: SortDepthRange | null,
  ) {
    this.withStagePasses(encoder, profiler, (stagePass) =>
      this.encodeStages(stagePass, metric, count, bits, depth),
    );
  }

  /**
   * Buffers of at least `capacity` keys and values for encodeKeys, which
   * the caller's kernels fill; their count goes in drawArgs[1].
   */
  keyInput(capacity: number): { keys: GPUBuffer; vals: GPUBuffer } {
    this.ensure(capacity);
    return { keys: this.keys[0], vals: this.vals[0] };
  }

  /**
   * Records a stable sort of the keyInput keys and values on their low
   * `bits` (a multiple of 4), for drawArgs[1] of them (at most `capacity`),
   * into `encoder`: one pass, or one per stage when `profiler` is given.
   * Returns the buffers holding the sorted keys and values.
   */
  encodeKeys(
    encoder: GPUCommandEncoder,
    profiler: GpuProfiler | null,
    capacity: number,
    bits: number,
  ): { keys: GPUBuffer; vals: GPUBuffer } {
    this.ensure(capacity);
    const passes = Math.ceil(bits / 4);
    this.withStagePasses(encoder, profiler, (stagePass) => {
      this.registry
        .get(sortModule, "writeDispatch")
        .dispatch(stagePass("prepare"), {
          grid: [1],
          buffers: {
            sortCount: this.drawArgs,
            dispatchArgs: this.dispatchArgs,
          },
        });
      this.encodeRadixPasses(stagePass, capacity, 0, passes);
    });
    const out = passes & 1;
    return { keys: this.keys[out], vals: this.vals[out] };
  }

  // Calls `record` with a stage-pass function: every stage in one compute
  // pass, or with `profiler`, a timed pass per stage.
  private withStagePasses(
    encoder: GPUCommandEncoder,
    profiler: GpuProfiler | null,
    record: (stagePass: (stage: string) => GPUComputePassEncoder) => void,
  ) {
    let open: GPUComputePassEncoder | null = null;
    let openStage = "";
    record((stage) => {
      if (open && (!profiler || stage === openStage)) return open;
      open?.end();
      openStage = stage;
      const label = `${this.label}${profiler ? `.${stage}` : ""}`;
      open = encoder.beginComputePass({
        label,
        timestampWrites: profiler?.timestampWrites(label),
      });
      return open;
    });
    (open as GPUComputePassEncoder | null)?.end();
  }

  // `pass(stage)` gives the pass to record each stage's dispatches into.
  private encodeStages(
    stagePass: (stage: string) => GPUComputePassEncoder,
    metric: GPUBuffer,
    count: number,
    floatBits: 16 | 24 | 32,
    depth?: SortDepthRange | null,
  ) {
    this.ensure(count);
    const bits = depth ? DEPTH_BITS : floatBits;
    this.device.queue.writeBuffer(
      this.drawArgs,
      0,
      new Uint32Array([4, 0, 0, 0]),
    );
    const numBlocks = Math.ceil(count / TILE);
    const sortParams = (shift: number) =>
      UniformWriter.for(sortModule, "params").setAll({
        count,
        shift,
        numBlocks,
        keyMode: depth ? KEY_DEPTH : KEY_FLOAT,
        keyLo: depth?.lo ?? 0,
        keyScale: depth ? DEPTH_KEYS / Math.max(depth.hi - depth.lo, 1e-30) : 0,
      }).data;
    const get = (entry: string) => this.registry.get(sortModule, entry);

    // Only the active (finite) metrics become keys, in index order: count
    // them per block, scan the counts, write the keys at those offsets.
    const prepare = stagePass("prepare");
    const blocks = {
      grid: [numBlocks * 128] as const,
      uniforms: sortParams(0),
    };
    get("countActive").dispatch(prepare, {
      ...blocks,
      buffers: {
        sortMetric: metric,
        blockHist: this.scanLevels[0],
        drawArgs: this.drawArgs,
      },
    });
    this.encodeScan(prepare, numBlocks);
    get("compactKeys").dispatch(prepare, {
      ...blocks,
      buffers: {
        sortMetric: metric,
        blockHist: this.scanLevels[0],
        keysOut: this.keys[0],
        valsOut: this.vals[0],
      },
    });
    get("writeDispatch").dispatch(prepare, {
      grid: [1],
      buffers: { sortCount: this.drawArgs, dispatchArgs: this.dispatchArgs },
    });

    this.encodeRadixPasses(stagePass, count, 32 - bits, bits / 4);
  }

  // `passes` radix passes from bit `shift` over the active blocks only
  // (dispatchArgs), from keys[0] and vals[0], alternating buffers.
  private encodeRadixPasses(
    stagePass: (stage: string) => GPUComputePassEncoder,
    count: number,
    shift: number,
    passes: number,
  ) {
    const numBlocks = Math.ceil(count / TILE);
    const get = (entry: string) => this.registry.get(sortModule, entry);
    for (let p = 0; p < passes; p++) {
      const src = p & 1;
      const dst = src ^ 1;
      const uniforms = UniformWriter.for(sortModule, "params").setAll({
        count,
        shift: shift + 4 * p,
        numBlocks,
      }).data;
      get("radixHistogram").dispatchIndirect(
        stagePass("histogram"),
        {
          buffers: {
            keysIn: this.keys[src],
            blockHist: this.scanLevels[0],
            sortCount: this.drawArgs,
          },
          uniforms,
        },
        this.dispatchArgs,
      );
      this.encodeScan(stagePass("scan"), BINS * numBlocks);
      const scatter = this.subgroups
        ? this.registry.get(scatterSubgroupModule, "radixScatterSubgroup")
        : get("radixScatter");
      scatter.dispatchIndirect(
        stagePass("scatter"),
        {
          buffers: {
            keysIn: this.keys[src],
            valsIn: this.vals[src],
            keysOut: this.keys[dst],
            valsOut: this.vals[dst],
            blockHist: this.scanLevels[0],
            sortCount: this.drawArgs,
          },
          uniforms,
        },
        this.dispatchArgs,
      );
    }
  }

  // Exclusive scan of scanLevels[0][0..n), level by level.
  private encodeScan(pass: GPUComputePassEncoder, n: number) {
    encodeExclusiveScan(this.registry, pass, this.scanLevels, n, this.empty);
  }

  private destroyBuffers() {
    for (const b of [...this.keys, ...this.vals, ...this.scanLevels])
      b.destroy();
    this.keys = [];
    this.vals = [];
    this.scanLevels = [];
  }

  destroy() {
    this.destroyBuffers();
    this.drawArgs.destroy();
    this.dispatchArgs.destroy();
    this.empty.destroy();
  }
}

/**
 * Buffers for encodeExclusiveScan of `n` u32: the data, then the chunk sums
 * of each level.
 */
export function createScanLevels(
  device: GPUDevice,
  n: number,
  label: string,
): GPUBuffer[] {
  const levels = [storage(device, n * 4, label)];
  let size = n;
  do {
    size = Math.ceil(size / SCAN_CHUNK);
    levels.push(storage(device, size * 4, `${label} chunk sums`));
  } while (size > 1);
  return levels;
}

/**
 * Records an in-place exclusive scan of levels[0][0..n) into `pass`, level
 * by level (sort_radix.slang's scanChunks and addChunkOffsets). `empty`
 * takes the top level's sum when `levels` has no buffer for it.
 */
export function encodeExclusiveScan(
  registry: KernelRegistry,
  pass: GPUComputePassEncoder,
  levels: readonly GPUBuffer[],
  n: number,
  empty: GPUBuffer,
) {
  const scan = registry.get(sortModule, "scanChunks");
  const add = registry.get(sortModule, "addChunkOffsets");
  const sizes: number[] = [];
  let size = n;
  for (let level = 0; ; level++) {
    sizes.push(size);
    const chunks = Math.ceil(size / SCAN_CHUNK);
    scan.dispatch(pass, {
      grid: [chunks * 256],
      buffers: {
        scanData: levels[level],
        chunkSums: levels[level + 1] ?? empty,
      },
      uniforms: UniformWriter.for(sortModule, "scanParams").set("count", size)
        .data,
    });
    if (chunks <= 1) break;
    size = chunks;
  }
  for (let level = sizes.length - 2; level >= 0; level--) {
    add.dispatch(pass, {
      grid: [sizes[level]],
      buffers: {
        scanData: levels[level],
        chunkSums: levels[level + 1],
      },
      uniforms: UniformWriter.for(sortModule, "scanParams").set(
        "count",
        sizes[level],
      ).data,
    });
  }
}
