// athenea's LightRecord (light/lights.slang, vendored) as bytes, for the
// buffers the relight pass and the env passes read: one record a light,
// 160 bytes, std430 as slangc lays it out (the generated WGSL's
// LightRecord_std430). The light's frame is three rows of light-to-world;
// UsdLux lights shine down their -Z.

import * as THREE from "three";

export const LIGHT_RECORD_BYTES = 160;

/** athenea's light kinds (lights.slang kLight*). */
export const LIGHT_KIND = {
  distant: 0,
  sphere: 1,
  disk: 2,
  rect: 3,
  dome: 4,
  cylinder: 5,
} as const;
export type AtheneaLightKind = keyof typeof LIGHT_KIND;

/** lights.slang flags. */
export const LIGHT_SHADOW = 1;
export const LIGHT_NORMALIZE = 2;
export const LIGHT_TEMPERATURE = 4;
export const LIGHT_IES_NORMALIZE = 8;
export const LIGHT_NONE = 0xffffffff;

type Rgb = [number, number, number];

export interface AtheneaLightRecord {
  kind: AtheneaLightKind;
  flags?: number;
  /** Sphere and disk radius, rect width, distant angular diameter (radians). */
  sizeX?: number;
  /** Rect height. */
  sizeY?: number;
  /** Colour times intensity. */
  colour?: Rgb;
  /** Stops. */
  exposure?: number;
  /** Kelvin (with LIGHT_TEMPERATURE). */
  temperature?: number;
  /** Cosine of the shaping cone's half angle; -1: none. */
  coneCos?: number;
  coneSoftness?: number;
  /** A dome's image (any value but LIGHT_NONE: it has one). */
  texture?: number;
  /** IES profile row; LIGHT_NONE: none. */
  ies?: number;
  iesAngleScale?: number;
  /** Light to world (only its rotation and translation are used). */
  matrix?: THREE.Matrix4;
}

/** Writes `light` at byte `at` of `view`. */
export function writeLightRecord(
  view: DataView,
  at: number,
  light: AtheneaLightRecord,
) {
  const u = (o: number, v: number) => view.setUint32(at + o, v >>> 0, true);
  const f = (o: number, v: number) => view.setFloat32(at + o, v, true);
  u(0, LIGHT_KIND[light.kind]);
  u(4, light.flags ?? 0);
  f(8, light.sizeX ?? 0);
  f(12, light.sizeY ?? 0);
  const c = light.colour ?? [1, 1, 1];
  c.forEach((v, i) => f(16 + 4 * i, v));
  f(28, light.exposure ?? 0);
  f(32, light.temperature ?? 6500);
  f(36, light.coneCos ?? -1);
  f(40, light.coneSoftness ?? 0);
  u(44, light.texture ?? LIGHT_NONE);
  u(48, 0); // sampler
  u(52, LIGHT_NONE); // lightCategory: unlinked, lights everything
  u(56, LIGHT_NONE); // shadowCategory
  f(60, 1); // cumulative
  u(64, light.ies ?? LIGHT_NONE);
  f(68, light.iesAngleScale ?? 0);
  u(72, 0);
  u(76, 0); // group
  const e = (light.matrix ?? new THREE.Matrix4()).elements;
  // Rows of the 3x4: column-major elements, row r is e[r], e[r+4], e[r+8], e[r+12].
  for (let r = 0; r < 3; r++) {
    for (let k = 0; k < 4; k++) f(80 + 16 * r + 4 * k, e[r + 4 * k]);
  }
  // ShadowAPI: black shadows, no distance limit, no falloff.
  f(128, 0);
  f(132, 0);
  f(136, 0);
  f(140, -1);
  f(144, -1);
  f(148, 1);
  u(152, 0);
  u(156, 0);
}

/** Records for `lights`, packed one after the other (one record at least). */
export function packLightRecords(
  lights: readonly AtheneaLightRecord[],
): Uint32Array {
  const bytes = new ArrayBuffer(
    Math.max(lights.length, 1) * LIGHT_RECORD_BYTES,
  );
  const view = new DataView(bytes);
  lights.forEach((l, k) => writeLightRecord(view, k * LIGHT_RECORD_BYTES, l));
  return new Uint32Array(bytes);
}

/**
 * A distant light (UsdLux's sun) whose light travels along `-towards`:
 * its +Z axis is `towards`, the direction to the light.
 */
export function distantMatrix(towards: THREE.Vector3): THREE.Matrix4 {
  const z = towards.clone().normalize();
  const helper =
    Math.abs(z.y) < 0.999
      ? new THREE.Vector3(0, 1, 0)
      : new THREE.Vector3(1, 0, 0);
  const x = new THREE.Vector3().crossVectors(helper, z).normalize();
  const y = new THREE.Vector3().crossVectors(z, x);
  return new THREE.Matrix4().makeBasis(x, y, z);
}
