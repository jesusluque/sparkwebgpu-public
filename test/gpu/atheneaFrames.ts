// Frames and skies for the athenea comparisons (atheneaBattery, atheneaPawn):
// EXR and Radiance HDR read top row first, and a PNG writer for looking at
// what was drawn.

import { readFileSync, writeFileSync } from "node:fs";
import { zlibSync } from "fflate";
import * as THREE from "three";
import { EXRLoader } from "three/examples/jsm/loaders/EXRLoader.js";
import { HDRLoader } from "three/examples/jsm/loaders/HDRLoader.js";
import type { SkyImage } from "../../src/webgpu/athenea/AtheneaSky";

export /** An EXR, top row first (three's EXRLoader hands it bottom row first). */
function loadExr(file: string) {
  const loader = new EXRLoader();
  loader.setDataType(THREE.FloatType);
  const bytes = readFileSync(file);
  const r = loader.parse(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  ) as { width: number; height: number; data: Float32Array };
  const row = r.width * 4;
  const data = new Float32Array(r.data.length);
  for (let y = 0; y < r.height; y++) {
    data.set(
      r.data.subarray((r.height - 1 - y) * row, (r.height - y) * row),
      y * row,
    );
  }
  return { width: r.width, height: r.height, data };
}

export function loadSky(file: string): SkyImage {
  const bytes = readFileSync(file);
  const buffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
  if (file.endsWith(".exr")) {
    const r = loadExr(file);
    return { width: r.width, height: r.height, data: r.data, channels: 4 };
  }
  const loader = new HDRLoader();
  loader.setDataType(THREE.FloatType);
  const r = loader.parse(buffer) as {
    width: number;
    height: number;
    data: Float32Array;
  };
  return { width: r.width, height: r.height, data: r.data, channels: 4 };
}

export function crc32(bytes: Uint8Array) {
  let c = ~0;
  for (const v of bytes) {
    c ^= v;
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return ~c >>> 0;
}

/** An sRGB PNG of a linear frame (top row first). */
export function writePng(
  file: string,
  rgba: Float32Array,
  w: number,
  h: number,
) {
  const raw = new Uint8Array(h * (w * 3 + 1));
  const enc = (v: number) => {
    const x = Math.min(Math.max(v, 0), 1);
    return Math.round(
      255 * (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055),
    );
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 3; c++) {
        raw[y * (w * 3 + 1) + 1 + x * 3 + c] = enc(rgba[(y * w + x) * 4 + c]);
      }
    }
  }
  const chunk = (type: string, data: Uint8Array) => {
    const out = new Uint8Array(12 + data.length);
    const view = new DataView(out.buffer);
    view.setUint32(0, data.length);
    out.set(new TextEncoder().encode(type), 4);
    out.set(data, 8);
    view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
    return out;
  };
  const ihdr = new Uint8Array(13);
  const v = new DataView(ihdr.buffer);
  v.setUint32(0, w);
  v.setUint32(4, h);
  ihdr.set([8, 2, 0, 0, 0], 8);
  const parts = [
    new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", ihdr),
    chunk("IDAT", zlibSync(raw)),
    chunk("IEND", new Uint8Array(0)),
  ];
  writeFileSync(file, Buffer.concat(parts));
}
