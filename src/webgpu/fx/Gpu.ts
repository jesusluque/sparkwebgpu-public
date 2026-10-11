// aofx's `Gpu` (sdk/include/aofx/Effect.h) on WebGPU: what an effect asks the
// host for. Kernels, scratch pictures, kept buffers and readback are the
// host's; an effect describes work and asks for it.
//
// `run` mirrors the aofx host's reflection-trailer check before anything is
// recorded: the buffer count against the kernel's declared buffers, each
// buffer's size against its element size, the uniform block's size, and the
// grid in threads (divided by the kernel's own workgroup size, and refused
// when that is more groups than the device dispatches). It adds the checks
// WebGPU needs on top: a picture buffer too small for its stride and height,
// and a written buffer bound twice, which WebGPU refuses at submit with an
// error far from the cause.

import type { KernelModule } from "../KernelModule";
import type { Grid, Kernel } from "../KernelRegistry";
import { KernelRegistry, usedBindings } from "../KernelRegistry";
import {
  createReadback,
  createStorage,
  readAndDestroy,
  writeWords,
} from "../gpuBuffers";
import { UniformWriter } from "../uniforms";
import { type FxBuffer, PIXEL_BYTES, rect } from "./types";

export type BoundBuffer = FxBuffer | GPUBuffer;

/**
 * Buffers for a dispatch. An array is aofx's positional form: every storage
 * buffer the module declares, in declaration order, whether or not this
 * entry reads it. A record binds by the names in the Slang source.
 */
export type RunBuffers =
  | readonly BoundBuffer[]
  | Readonly<Record<string, BoundBuffer>>;

export type RunUniforms = UniformWriter | ArrayBufferView | ArrayBuffer;

function isFxBuffer(b: BoundBuffer): b is FxBuffer {
  return "stride" in b;
}

function gpuBuffer(b: BoundBuffer): GPUBuffer {
  return isFxBuffer(b) ? b.buffer : b;
}

/** The storage buffers a module declares, in binding order: aofx's positions. */
export function declaredBuffers(module: KernelModule) {
  return module.reflection.bindings
    .filter((b) => b.kind === "storage" || b.kind === "read-only-storage")
    .sort((a, b) => a.group - b.group || a.binding - b.binding);
}

export class Gpu {
  readonly device: GPUDevice;
  /** Bound where a kernel declares a buffer the dispatch never touches. */
  readonly placeholder: GPUBuffer;
  /** Dispatches recorded since the last flush; for tests and stats. */
  dispatches = 0;

  private encoder: GPUCommandEncoder | null = null;
  private pass: GPUComputePassEncoder | null = null;
  private free = new Map<number, GPUBuffer[]>();
  private lent: GPUBuffer[] = [];
  private kept = new Map<string, GPUBuffer>();
  private reads: Array<() => void> = [];

  constructor(readonly registry: KernelRegistry) {
    this.device = registry.device;
    this.placeholder = createStorage(this.device, 16, "fx placeholder");
  }

  static forDevice(device: GPUDevice) {
    return new Gpu(new KernelRegistry(device));
  }

  /** aofx `load`: the entry `entry` of `module`, built on first use. */
  load(module: KernelModule, entry: string): Kernel {
    return this.registry.get(module, entry);
  }

  /** Problems with a dispatch, or an empty list when the kernel would take it. */
  check(
    kernel: Kernel,
    grid: Grid,
    buffers: RunBuffers,
    uniforms?: RunUniforms,
  ): string[] {
    const where = `${kernel.module.name}:${kernel.entry.name}`;
    const errors: string[] = [];
    const declared = declaredBuffers(kernel.module);
    let named: Record<string, BoundBuffer>;
    if (Array.isArray(buffers)) {
      if (buffers.length !== declared.length) {
        errors.push(
          `${where}: ${buffers.length} buffers for the ${declared.length} the kernel declares (${declared.map((b) => b.name).join(", ")}); bindings are positional and belong to the file`,
        );
      }
      named = {};
      declared.forEach((b, i) => {
        if (buffers[i]) named[b.name] = buffers[i];
      });
    } else {
      named = { ...(buffers as Record<string, BoundBuffer>) };
    }

    // Element sizes, and that a picture's rows fit in its buffer.
    for (const b of declared) {
      const bound = named[b.name];
      if (!bound) continue;
      const buf = gpuBuffer(bound);
      const limit = this.device.limits.maxStorageBufferBindingSize;
      if (buf.size > limit) {
        errors.push(
          `${where}: buffer '${b.name}' is ${buf.size} bytes, over the device's maxStorageBufferBindingSize of ${limit}`,
        );
      }
      if (b.elementBytes && buf.size % b.elementBytes !== 0) {
        errors.push(
          `${where}: buffer '${b.name}' is ${buf.size} bytes, not whole ${b.elementBytes}-byte elements`,
        );
      }
      if (isFxBuffer(bound) && bound.width > 0 && bound.height > 0) {
        const need =
          ((bound.height - 1) * bound.stride + bound.width) * PIXEL_BYTES;
        if (bound.stride < bound.width || buf.size < need) {
          errors.push(
            `${where}: picture '${b.name}' of ${bound.width}x${bound.height} stride ${bound.stride} needs ${need} bytes, buffer has ${buf.size}`,
          );
        }
      }
    }

    // A written buffer bound under a second name is refused by WebGPU.
    const used = usedBindings(kernel.module, kernel.entry);
    for (const b of used) {
      if (b.kind !== "storage" || !named[b.name]) continue;
      const buf = gpuBuffer(named[b.name]);
      for (const o of used) {
        if (o !== b && named[o.name] && gpuBuffer(named[o.name]) === buf) {
          errors.push(
            `${where}: '${b.name}' is written and also bound as '${o.name}'; bind the placeholder for an output nobody asked for`,
          );
        }
      }
    }

    const bytes = uniformBytes(uniforms);
    const block = used.find((b) => b.kind === "uniform");
    if (block && bytes !== block.bytes) {
      errors.push(
        `${where}: uniform block '${block.name}' is ${block.bytes} bytes, got ${bytes}`,
      );
    }
    if (!block && bytes > 0) {
      errors.push(`${where}: uniforms given to a kernel without a block`);
    }

    const size = kernel.entry.workgroupSize ?? [1, 1, 1];
    const maxGroups = this.device.limits.maxComputeWorkgroupsPerDimension;
    const [x, y = 1, z = 1] = grid;
    [x, y, z].forEach((n, i) => {
      if (!Number.isInteger(n) || n < 0) {
        errors.push(`${where}: grid ${grid.join("x")} is not whole threads`);
      } else if (Math.ceil(n / (size[i] || 1)) > maxGroups) {
        errors.push(
          `${where}: grid ${grid.join("x")} is ${Math.ceil(n / (size[i] || 1))} groups along ${"xyz"[i]}, over the device's ${maxGroups}`,
        );
      }
    });
    if (errors.length === 0) {
      for (const b of used) {
        if (b.kind !== "uniform" && !named[b.name]) {
          errors.push(`${where}: missing buffer '${b.name}'`);
        }
      }
    }
    return [...new Set(errors)];
  }

  /**
   * aofx `run`: one thread per grid cell, recorded into this frame's compute
   * pass. A dispatch that disagrees with the kernel throws, naming why,
   * instead of rendering black.
   */
  run(
    kernel: Kernel,
    grid: Grid,
    buffers: RunBuffers,
    uniforms?: RunUniforms,
  ): boolean {
    const errors = this.check(kernel, grid, buffers, uniforms);
    if (errors.length) throw new Error(errors.join("\n"));
    const named: Record<string, GPUBuffer> = {};
    if (Array.isArray(buffers)) {
      declaredBuffers(kernel.module).forEach((b, i) => {
        named[b.name] = gpuBuffer(buffers[i]);
      });
    } else {
      for (const [k, v] of Object.entries(
        buffers as Record<string, BoundBuffer>,
      )) {
        named[k] = gpuBuffer(v);
      }
    }
    kernel.dispatch(this.computePass(), {
      grid,
      buffers: named,
      uniforms: uniformData(uniforms),
    });
    this.dispatches += 1;
    return true;
  }

  /**
   * aofx `scratch`: a picture for this frame, at the origin. Returned to the
   * pool at the next flush, so it is recycled rather than reallocated.
   */
  scratch(width: number, height: number): FxBuffer {
    const w = Math.max(0, Math.ceil(width));
    const h = Math.max(0, Math.ceil(height));
    const buffer = this.lend(Math.max(1, w * h) * PIXEL_BYTES);
    return { buffer, width: w, height: h, stride: w, rect: rect(0, 0, w, h) };
  }

  /** A raw buffer of at least `bytes`, recycled at the next flush. */
  lend(bytes: number): GPUBuffer {
    // Buckets an eighth of a power of two apart keep a resizing window from
    // minting a buffer per size without doubling what a frame needs.
    const step = 2 ** Math.max(8, Math.ceil(Math.log2(bytes)) - 3);
    const size = Math.max(16, Math.ceil(bytes / step) * step);
    const list = this.free.get(size);
    const buffer =
      list?.pop() ?? createStorage(this.device, size, "fx scratch");
    this.lent.push(buffer);
    return buffer;
  }

  /**
   * Hands a lent buffer (lend, scratch) back before the frame ends, once
   * nothing recorded after this point reads it: the next lend of its size
   * this frame gets it again. Safe within one frame because every use is
   * recorded in order into one encoder (WebGPU orders the later writes
   * after the earlier reads). A buffer not lent this frame is ignored.
   */
  release(buffer: GPUBuffer | FxBuffer | null | undefined) {
    const b = buffer && "buffer" in buffer ? buffer.buffer : buffer;
    if (!b) return;
    const k = this.lent.indexOf(b as GPUBuffer);
    if (k < 0) return;
    this.lent.splice(k, 1);
    const list = this.free.get(b.size) ?? [];
    list.push(b as GPUBuffer);
    this.free.set(b.size, list);
  }

  /**
   * aofx `keep`: a buffer that outlives the frame, by a key that identifies
   * its contents. Asking again with the same key costs nothing and `data` may
   * be left out once it is there. There is no eviction: `drop` what you
   * replace.
   */
  keep(
    key: string,
    data?: ArrayBufferView | null,
    bytes?: number,
  ): GPUBuffer | null {
    const had = this.kept.get(key);
    if (had) return had;
    const size = bytes ?? data?.byteLength ?? 0;
    if (size <= 0) return null;
    const buffer = createStorage(this.device, size, `fx keep ${key}`);
    if (data) writeWords(this.device, buffer, data);
    this.kept.set(key, buffer);
    return buffer;
  }

  drop(key: string) {
    this.kept.get(key)?.destroy();
    this.kept.delete(key);
  }

  /**
   * The first `bytes` of a buffer as they are after everything recorded so
   * far; resolves once the frame is flushed. aofx's `read` blocks; WebGPU
   * cannot, so an effect that measures hands its answer on later.
   */
  read(source: BoundBuffer, bytes?: number): Promise<ArrayBuffer> {
    const buf = gpuBuffer(source);
    const staging = createReadback(this.device, bytes ?? buf.size, "fx read");
    this.commandEncoder().copyBufferToBuffer(buf, 0, staging, 0, staging.size);
    return new Promise((resolve, reject) => {
      this.reads.push(() => readAndDestroy(staging).then(resolve, reject));
    });
  }

  /** This frame's encoder, with any compute pass ended, for copies. */
  commandEncoder(): GPUCommandEncoder {
    this.pass?.end();
    this.pass = null;
    this.encoder ??= this.device.createCommandEncoder({ label: "fx" });
    return this.encoder;
  }

  private computePass(): GPUComputePassEncoder {
    if (!this.pass) {
      this.pass = this.commandEncoder().beginComputePass({ label: "fx" });
    }
    return this.pass;
  }

  /** Submits what was recorded and recycles this frame's scratch. */
  flush() {
    if (this.encoder) {
      const encoder = this.commandEncoder();
      this.encoder = null;
      this.registry.submit(encoder.finish());
    }
    // Later frames' writes queue behind this frame's reads, so a lent
    // buffer is free to lend again as soon as the frame is submitted.
    for (const b of this.lent) {
      const list = this.free.get(b.size) ?? [];
      list.push(b);
      this.free.set(b.size, list);
    }
    this.lent = [];
    const reads = this.reads;
    this.reads = [];
    for (const r of reads) r();
    this.dispatches = 0;
  }

  destroy() {
    this.pass?.end();
    this.pass = null;
    this.encoder = null;
    for (const list of this.free.values()) for (const b of list) b.destroy();
    for (const b of this.lent) b.destroy();
    for (const b of this.kept.values()) b.destroy();
    this.free.clear();
    this.lent = [];
    this.kept.clear();
    this.placeholder.destroy();
  }
}

function uniformBytes(u?: RunUniforms): number {
  if (!u) return 0;
  return u instanceof UniformWriter ? u.data.byteLength : u.byteLength;
}

function uniformData(
  u?: RunUniforms,
): ArrayBufferView | ArrayBuffer | undefined {
  if (!u) return undefined;
  return u instanceof UniformWriter ? u.data : u;
}
