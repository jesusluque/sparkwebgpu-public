import init_wasm from "spark-rs";
import { wasmBytes } from "./wasmSource";

/**
 * The compiled module, for this thread's instance and the loader workers'
 * (SplatWorker posts it to each).
 */
export const WASM_MODULE: Promise<WebAssembly.Module> = wasmBytes().then(
  (bytes) => WebAssembly.compile(bytes),
);

// Flag indicating if the spark-rs project has been initialized
let initialized = false;
let started: Promise<void> | null = null;

/**
 * This thread's instance, made on first use (thread CU: its memory grows
 * with what it parses, and a page whose parsing runs in the workers never
 * needs one).
 */
function start(): Promise<void> {
  started ??= init_wasm({ module_or_path: WASM_MODULE }).then((_) => {
    initialized = true;
  });
  return started;
}

/**
 * Promise for module instantiation, ensuring calls to
 * imports from the spark-rs project can be used. A thenable: awaiting it
 * starts the instantiation.
 */
export const initialization: PromiseLike<void> = {
  // biome-ignore lint/suspicious/noThenProperty: awaited as a promise
  then: (resolve, reject) => start().then(resolve, reject),
};

/** Starts this thread's instance (no-op once started). */
export function startWasm(): Promise<void> {
  return start();
}

/**
 * Indicates if the wasm module instantiation has completed or not (it
 * starts on first use: `initialization` awaited, or `startWasm`).
 */
export function isInitialized() {
  return initialized;
}
