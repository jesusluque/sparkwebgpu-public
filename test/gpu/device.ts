// A WebGPU device for tests, from Dawn's Node bindings (the `webgpu`
// package). On a machine without a usable adapter `device` is null and the
// GPU suites skip.

import { create, globals } from "webgpu";
import {
  createReadback,
  createStorage,
  readAndDestroy,
  writeWords,
} from "../../src/webgpu/gpuBuffers";

Object.assign(globalThis, globals);

// Dawn tears down its instance when the GPU object is garbage-collected, and
// a device still in use then crashes the process (SIGSEGV, at random). Keep
// both alive for the whole run.
let gpu: GPU | null = null;
let adapter: GPUAdapter | null = null;

async function open(): Promise<GPUDevice | null> {
  try {
    gpu = create([]);
    adapter = await gpu.requestAdapter();
    // Timestamps where the adapter has them (rasterizer "auto", profile).
    const requiredFeatures: GPUFeatureName[] = adapter?.features.has(
      "timestamp-query",
    )
      ? ["timestamp-query"]
      : [];
    return adapter ? await adapter.requestDevice({ requiredFeatures }) : null;
  } catch {
    return null;
  }
}

export const device = await open();

/**
 * A second device with the adapter's storage-buffer count and binding size
 * (what splatRequiredLimits asks for), for kernels past the default
 * eight (the athenea relight pass binds ten); null where there is none.
 */
let wideAdapter: GPUAdapter | null = null;
export const wideDevice: GPUDevice | null = await (async () => {
  if (!gpu || !device) return null;
  try {
    // An adapter makes one device: a second one needs an adapter of its own.
    wideAdapter = await gpu.requestAdapter();
    const most = wideAdapter?.limits.maxStorageBuffersPerShaderStage ?? 0;
    if (!wideAdapter || most <= 8) return null;
    return await wideAdapter.requestDevice({
      requiredFeatures: [...device.features] as GPUFeatureName[],
      requiredLimits: {
        maxStorageBuffersPerShaderStage: most,
        maxStorageBufferBindingSize:
          wideAdapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: wideAdapter.limits.maxBufferSize,
      },
    });
  } catch {
    return null;
  }
})();

// Thin wrappers over the library's own buffer helpers, so tests allocate and
// read back exactly as the renderer does.

/** A storage buffer holding `data`, or `data` zeroed bytes. */
export function storage(data: ArrayBufferView | number): GPUBuffer {
  const d = device as GPUDevice;
  const size = typeof data === "number" ? data : data.byteLength;
  const buffer = createStorage(d, size, "test storage");
  if (typeof data !== "number") writeWords(d, buffer, data);
  return buffer;
}

/** A copy of all of `buffer`. */
export async function readBack(buffer: GPUBuffer): Promise<ArrayBuffer> {
  const d = device as GPUDevice;
  const staging = createReadback(d, buffer.size, "test readback");
  const encoder = d.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
  d.queue.submit([encoder.finish()]);
  return readAndDestroy(staging);
}

/** Uniform block of `count` followed by padding, as the test kernels use. */
export function countParams(count: number): Uint32Array {
  return new Uint32Array([count, 0, 0, 0]);
}
