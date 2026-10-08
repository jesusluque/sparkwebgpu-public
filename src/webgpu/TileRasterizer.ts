// The experimental compute draw of WgpuSplatRenderer (rasterizer: "tiles"):
// slang/tiles/tile_raster.slang bins the depth-sorted splats into 16 x 16
// pixel tiles and blends each tile front to back in a workgroup, into an
// rgba16float texture that composite() then blends over the target with the
// quad draw's blend state. See the Slang file for the stages.
//
// The pair buffers are sized from a readback of the last frames' pair count,
// so a frame needing more pairs than they hold drops its front-most splats'
// later pairs; the next frames grow them.

import type { GpuProfiler } from "./GpuProfiler";
import { GpuSorter, createScanLevels, encodeExclusiveScan } from "./GpuSorter";
import type { KernelModule } from "./KernelModule";
import type { KernelRegistry } from "./KernelRegistry";
import { tilesTileRaster } from "./generated/constants";
import tileModule from "./generated/tiles/tile_raster";
import { createReadback, createStorage, reuseTexture } from "./gpuBuffers";
import { FULLSCREEN_TRIANGLE_WGSL } from "./renderPipeline";
import { UniformWriter } from "./uniforms";

const { TILE_SIZE, TILE_DEPTH_TEST, TILE_CLAMP } = tilesTileRaster;

const COMPOSITE_WGSL = /* wgsl */ `
@group(0) @binding(0) var tiles: texture_2d<f32>;
${FULLSCREEN_TRIANGLE_WGSL}
@fragment
fn fragmentMain(@builtin(position) p: vec4f) -> @location(0) vec4f {
  return textureLoad(tiles, vec2i(p.xy), 0);
}
`;

export interface TileRasterInput {
  /** Back-to-front accumulator indices, and the draw args holding their count in [1]. */
  ordering: GPUBuffer;
  sortCount: GPUBuffer;
  /** Entries of `ordering` the sort can fill. */
  slots: number;
  /** The accumulator and splat_draw's params block, as the quad draw uses. */
  splats: GPUBuffer;
  drawParams: ArrayBuffer;
  width: number;
  height: number;
  /** Depth to test against (less-equal), as the quad draw's depth attachment. */
  depth: GPUTexture | null;
  /** Clamp each splat's colour to 1, as blending into unorm targets does. */
  clamp: boolean;
  profiler: GpuProfiler | null;
  /** A blend-plugin variant of tiles/tile_raster (PluginHost), and its uniform blocks by name. */
  module?: KernelModule;
  pluginUniforms?: Record<string, ArrayBuffer>;
}

export class TileRasterizer {
  readonly device: GPUDevice;
  /** Transmittance under which a pixel stops blending (0: never). */
  minTransmittance = 1 / 255;
  /** Pairs of the last frame read back, and what the buffers hold. */
  readonly stats = { pairs: 0, pairCapacity: 0 };
  /** encode() calls so far, and which one the last read pair count is from. */
  encodes = 0;
  readFrom = -1;
  private copiedFrom = -1;
  private sorter: GpuSorter;
  private slotCapacity = 0;
  private tileSplats: GPUBuffer | null = null;
  private tileExtra: GPUBuffer | null = null;
  private tileRects: GPUBuffer | null = null;
  private scanLevels: GPUBuffer[] = [];
  private pairCapacity = 0;
  private keyTotal: GPUBuffer;
  private keyReadback: GPUBuffer;
  private readback: "idle" | "copied" | "mapping" = "idle";
  private ranges: GPUBuffer | null = null;
  private output: GPUTexture | null = null;
  private noDepth: GPUTexture;
  private empty: GPUBuffer;
  private composites = new Map<string, GPURenderPipeline>();

  constructor(readonly registry: KernelRegistry) {
    this.device = registry.device;
    this.sorter = new GpuSorter(registry, "tiles.sort");
    this.keyTotal = createStorage(this.device, 16, "tile key total");
    this.keyReadback = createReadback(this.device, 16, "tile key readback");
    this.empty = createStorage(this.device, 16, "tile empty");
    this.noDepth = this.device.createTexture({
      label: "tile no depth",
      size: [1, 1],
      format: "r32float",
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  // The largest pair buffer one binding allows.
  private get maxPairs() {
    const { maxStorageBufferBindingSize, maxBufferSize } = this.device.limits;
    return Math.floor(Math.min(maxStorageBufferBindingSize, maxBufferSize) / 4);
  }

  private ensureSlots(slots: number) {
    if (slots <= this.slotCapacity) return;
    this.slotCapacity = Math.max(slots, Math.ceil(this.slotCapacity * 1.5));
    const n = this.slotCapacity;
    for (const b of [this.tileSplats, this.tileExtra, this.tileRects])
      b?.destroy();
    for (const b of this.scanLevels) b.destroy();
    // TILE_SPLAT_WORDS uint4 a slot (a surfel needs the third).
    this.tileSplats = createStorage(this.device, n * 48, "tile splats");
    this.tileExtra = createStorage(this.device, n * 16, "tile extra");
    this.tileRects = createStorage(this.device, n * 8, "tile rects");
    this.scanLevels = createScanLevels(this.device, n, "tile counts");
    // A first guess before any readback: a few tiles a splat.
    this.pairCapacity = Math.max(
      this.pairCapacity,
      Math.min(4 * n, this.maxPairs),
    );
  }

  /**
   * Maps the last copied pair count, once its frame was submitted: grows the
   * pair buffers for the next frames when they were too small. encode()
   * calls it; call it on frames without one too, to size them before the
   * next.
   */
  poll() {
    if (this.readback !== "copied") return;
    this.readback = "mapping";
    this.keyReadback
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const pairs = new Uint32Array(this.keyReadback.getMappedRange())[0];
        this.keyReadback.unmap();
        this.stats.pairs = pairs;
        this.readFrom = this.copiedFrom;
        if (pairs > this.pairCapacity) {
          this.pairCapacity = Math.min(Math.ceil(pairs * 1.25), this.maxPairs);
        }
        this.readback = "idle";
      })
      .catch(() => {
        this.readback = "idle";
      });
  }

  /** Records the tile stages into `encoder`, leaving the image for composite(). */
  encode(encoder: GPUCommandEncoder, input: TileRasterInput) {
    const { width, height, profiler } = input;
    this.poll();
    this.encodes += 1;
    this.ensureSlots(input.slots);
    const tilesX = Math.ceil(width / TILE_SIZE);
    const tilesY = Math.ceil(height / TILE_SIZE);
    const numTiles = tilesX * tilesY;
    const tileBits = Math.max(4, 4 * Math.ceil(Math.log2(numTiles) / 4));
    const pairs = this.pairCapacity;
    this.stats.pairCapacity = pairs;
    const { keys, vals } = this.sorter.keyInput(pairs);
    if (!this.ranges || this.ranges.size < numTiles * 8) {
      this.ranges?.destroy();
      this.ranges = createStorage(this.device, numTiles * 8, "tile ranges");
    }
    this.output = reuseTexture(this.device, this.output, {
      label: "tile image",
      size: [width, height],
      format: "rgba16float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    encoder.clearBuffer(this.ranges, 0, numTiles * 8);

    const module = input.module ?? tileModule;
    const tileParams = UniformWriter.for(module, "tileParams").setAll({
      slots: input.slots,
      tilesX,
      tilesY,
      keyCapacity: pairs,
      width,
      height,
      flags:
        (input.depth ? TILE_DEPTH_TEST : 0) | (input.clamp ? TILE_CLAMP : 0),
      minTransmittance: this.minTransmittance,
    }).data;
    const get = (entry: string) => this.registry.get(module, entry);
    const pass = (label: string) =>
      encoder.beginComputePass({
        label,
        timestampWrites: profiler?.timestampWrites(label),
      });
    // splat_draw's params too, for the footprint and alpha maths.
    const uniforms = {
      ...input.pluginUniforms,
      tileParams,
      params: input.drawParams,
    };
    const buffers: Record<string, GPUBuffer> = {
      ordering: input.ordering,
      sortCount: input.sortCount,
      splats: input.splats,
      tileSplats: this.tileSplats as GPUBuffer,
      tileExtra: this.tileExtra as GPUBuffer,
      tileRects: this.tileRects as GPUBuffer,
      tileCounts: this.scanLevels[0],
      keysOut: keys,
      valsOut: vals,
      keyArgs: this.sorter.drawArgs,
      keyTotal: this.keyTotal,
    };
    const dispatch = (
      p: GPUComputePassEncoder,
      entry: string,
      grid: readonly [number],
    ) => get(entry).dispatch(p, { grid, buffers, uniforms });

    const prepare = pass("tiles.prepare");
    dispatch(prepare, "tilePreprocess", [input.slots]);
    encodeExclusiveScan(
      this.registry,
      prepare,
      this.scanLevels,
      input.slots,
      this.empty,
    );
    dispatch(prepare, "tileTotal", [1]);
    dispatch(prepare, "tileDuplicate", [input.slots]);
    prepare.end();
    if (this.readback === "idle") {
      encoder.copyBufferToBuffer(this.keyTotal, 0, this.keyReadback, 0, 4);
      this.readback = "copied";
      this.copiedFrom = this.encodes - 1;
    }

    const sorted = this.sorter.encodeKeys(encoder, profiler, pairs, tileBits);

    const ranges = pass("tiles.ranges");
    get("tileRanges").dispatchIndirect(
      ranges,
      {
        buffers: {
          sortedKeys: sorted.keys,
          keyCount: this.sorter.drawArgs,
          rangesOut: this.ranges,
        },
      },
      this.sorter.dispatchArgs,
    );
    ranges.end();

    const blend = pass("tiles.blend");
    const depthView = input.depth
      ? input.depth.createView({ aspect: "depth-only" })
      : this.noDepth.createView();
    get("tileBlend").dispatch(blend, {
      grid: [tilesX * TILE_SIZE, tilesY * TILE_SIZE],
      buffers: {
        ordering: input.ordering,
        splats: input.splats,
        splatsIn: this.tileSplats as GPUBuffer,
        extraIn: this.tileExtra as GPUBuffer,
        sortedVals: sorted.vals,
        rangesIn: this.ranges as GPUBuffer,
      },
      uniforms,
      bindings: {
        depthTex: depthView,
        outTex: (this.output as GPUTexture).createView(),
      },
    });
    blend.end();
  }

  /**
   * Draws the tile image into `pass` (a full-screen triangle) with `blend`,
   * the quad draw's blend state for `format`.
   */
  composite(
    pass: GPURenderPassEncoder,
    format: GPUTextureFormat,
    blend: GPUBlendState,
  ) {
    const key = `${format}/${JSON.stringify(blend)}`;
    let pipeline = this.composites.get(key);
    if (!pipeline) {
      const module = this.device.createShaderModule({ code: COMPOSITE_WGSL });
      pipeline = this.device.createRenderPipeline({
        label: "tile composite",
        layout: "auto",
        vertex: { module, entryPoint: "fullscreenVertex" },
        fragment: {
          module,
          entryPoint: "fragmentMain",
          targets: [{ format, blend }],
        },
      });
      this.composites.set(key, pipeline);
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: (this.output as GPUTexture).createView() },
        ],
      }),
    );
    pass.draw(3);
  }

  destroy() {
    for (const b of [
      this.tileSplats,
      this.tileExtra,
      this.tileRects,
      this.ranges,
      this.keyTotal,
      this.keyReadback,
      this.empty,
      ...this.scanLevels,
    ])
      b?.destroy();
    this.output?.destroy();
    this.noDepth.destroy();
    this.sorter.destroy();
  }
}
