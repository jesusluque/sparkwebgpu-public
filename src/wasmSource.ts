// The spark-rs WASM's bytes. By default inlined in the bundle (base64), so
// a single file works anywhere; a build with SPARK_WASM=external (the demo
// site's, `npm run build:site`) replaces this module with a fetch of
// spark_rs_bg.wasm beside the bundle (vite.config.ts), which keeps the
// ~2.4 MB literal and its decoded string out of the page's heap (thread CU).
import WASM from "spark-rs/spark_rs_bg.wasm?arraybuffer&base64";

export function wasmBytes(): Promise<ArrayBuffer> {
  return Promise.resolve(WASM);
}
