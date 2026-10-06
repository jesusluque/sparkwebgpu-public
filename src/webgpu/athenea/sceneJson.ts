// athenea-web's scene.json (~/luc/athenea-web, demo/out/<id>/scene.json and
// R2's scenes/<id>/scene.json): one published scene, the files it is
// available as, its cameras, transform and background. Read here so a
// sparkwebGPU page opens what athenea-web publishes the way its viewer does
// (site/src/catalog.js, site/src/renderers/spark.js, site/src/view.js):
//
//   files[].path        relative to the scene.json's folder, or absolute
//   transform           { rotation: [x, y, z] degrees, XYZ order;
//                         position: [x, y, z]; scale: s }, per file or for
//                         the scene (a file's own wins)
//   cameras[]           { name, position, target, fov (vertical, degrees) }
//   background          { color, hdri, exposure, blur }
//   files[].paged       .rad pages unless false (here .athc too)
//   files[].viewer      false: download only
//
// Formats this viewer opens, in the order it prefers them: athenea's own
// .athc first (no flatten: materials, transfer, emission stay), then what
// athenea-web's Spark renderer opens.

import * as THREE from "three";
import { SplatMesh } from "../../SplatMesh";

export const ATHENEA_VIEWABLE = [
  "athc",
  "rad",
  "sog",
  "spz",
  "ply",
  "splat",
  "ksplat",
] as const;

export type AtheneaTransform = {
  rotation?: [number, number, number];
  position?: [number, number, number];
  scale?: number;
};

export type AtheneaCamera = {
  name: string;
  position: [number, number, number];
  target: [number, number, number];
  fov?: number;
};

export type AtheneaSceneFile = {
  format: string;
  path: string;
  /** Resolved against the scene.json's URL. */
  url: string;
  paged?: boolean;
  viewer?: boolean;
  renderer?: string;
  transform?: AtheneaTransform;
  bytes?: number;
  spzVersion?: number;
  note?: string;
  [key: string]: unknown;
};

export type AtheneaScene = {
  id: string;
  title?: string;
  description?: string;
  renderer?: string;
  gaussians?: number;
  shDegree?: number;
  /** The format athenea-web opens first. */
  default?: string;
  transform?: AtheneaTransform;
  cameras: AtheneaCamera[];
  files: AtheneaSceneFile[];
  background: {
    color?: string;
    hdri?: string;
    exposure?: number;
    blur?: number;
  };
  /** The scene.json's own URL; files resolve against it. */
  url: string;
  [key: string]: unknown;
};

function vec3(v: unknown, what: string): [number, number, number] {
  if (
    !Array.isArray(v) ||
    v.length !== 3 ||
    !v.every((x) => typeof x === "number" && Number.isFinite(x))
  ) {
    throw new Error(`scene.json: ${what} is not [x, y, z]`);
  }
  return v as [number, number, number];
}

function transformOf(t: unknown, what: string): AtheneaTransform | undefined {
  if (t == null) return undefined;
  if (typeof t !== "object") throw new Error(`scene.json: ${what}`);
  const o = t as Record<string, unknown>;
  const out: AtheneaTransform = {};
  if (o.rotation != null) out.rotation = vec3(o.rotation, `${what}.rotation`);
  if (o.position != null) out.position = vec3(o.position, `${what}.position`);
  if (o.scale != null) {
    if (typeof o.scale !== "number" || !(o.scale > 0)) {
      throw new Error(`scene.json: ${what}.scale is not a positive number`);
    }
    out.scale = o.scale;
  }
  return out;
}

/** A scene.json's contents, checked, with every file's URL resolved. */
export function parseAtheneaScene(json: unknown, url: string): AtheneaScene {
  if (!json || typeof json !== "object") {
    throw new Error("scene.json: not an object");
  }
  const s = json as Record<string, unknown>;
  if (typeof s.id !== "string" || !s.id) {
    throw new Error("scene.json: no id");
  }
  if (!Array.isArray(s.files)) throw new Error("scene.json: no files");
  const base = new URL(url, globalThis.location?.href ?? "http://localhost/");
  const files = s.files.map((f, k) => {
    if (!f || typeof f.format !== "string" || typeof f.path !== "string") {
      throw new Error(`scene.json: files[${k}] needs format and path`);
    }
    return {
      ...f,
      url: new URL(f.path, base).toString(),
      transform: transformOf(f.transform, `files[${k}].transform`),
    } as AtheneaSceneFile;
  });
  const cameras = (Array.isArray(s.cameras) ? s.cameras : []).map((c, k) => ({
    name: typeof c?.name === "string" ? c.name : `camera ${k + 1}`,
    position: vec3(c?.position, `cameras[${k}].position`),
    target: vec3(c?.target, `cameras[${k}].target`),
    fov: typeof c?.fov === "number" ? c.fov : undefined,
  }));
  return {
    ...s,
    id: s.id,
    transform: transformOf(s.transform, "transform"),
    cameras,
    files,
    background: (s.background as AtheneaScene["background"]) ?? {},
    url: base.toString(),
  };
}

/** Fetches and parses a scene.json. */
export async function loadAtheneaScene(
  url: string,
  init?: RequestInit,
): Promise<AtheneaScene> {
  const response = await fetch(url, { cache: "no-cache", ...init });
  if (!response.ok) {
    throw new Error(`${response.status} reading ${url}`);
  }
  return parseAtheneaScene(await response.json(), response.url || url);
}

/**
 * The file to open: `format` if the scene has it, else the first of
 * ATHENEA_VIEWABLE it has (a .athc before the scene's own default, which
 * athenea-web picks among flattened formats), skipping download-only files.
 */
export function pickAtheneaFile(
  scene: AtheneaScene,
  format?: string,
): AtheneaSceneFile | undefined {
  const viewable = scene.files.filter(
    (f) =>
      f.viewer !== false &&
      (ATHENEA_VIEWABLE as readonly string[]).includes(f.format),
  );
  const rank = (f: AtheneaSceneFile) =>
    f.format === "athc"
      ? -2
      : f.format === scene.default
        ? -1
        : ATHENEA_VIEWABLE.indexOf(f.format as never);
  return (
    (format && viewable.find((f) => f.format === format)) ||
    viewable.sort((a, b) => rank(a) - rank(b))[0]
  );
}

/** athenea-web's applyTransform: XYZ Euler degrees, position, uniform scale. */
export function applyAtheneaTransform(
  object: THREE.Object3D,
  t: AtheneaTransform | undefined,
) {
  if (!t) return;
  const DEG = Math.PI / 180;
  if (t.rotation) {
    object.rotation.set(
      t.rotation[0] * DEG,
      t.rotation[1] * DEG,
      t.rotation[2] * DEG,
      "XYZ",
    );
  }
  if (t.position) object.position.set(...t.position);
  if (t.scale != null) object.scale.setScalar(t.scale);
}

/** Puts `camera` where a scene camera is, looking at its target. */
export function applyAtheneaCamera(
  camera: THREE.PerspectiveCamera,
  c: AtheneaCamera,
) {
  camera.position.set(...c.position);
  camera.lookAt(new THREE.Vector3(...c.target));
  if (c.fov) {
    camera.fov = c.fov;
    camera.updateProjectionMatrix();
  }
  camera.updateMatrixWorld();
}

/**
 * A SplatMesh of the scene's file (pickAtheneaFile), transformed as
 * athenea-web transforms it; .rad and .athc page unless the file says not.
 */
export function atheneaSceneMesh(
  scene: AtheneaScene,
  file = pickAtheneaFile(scene),
): SplatMesh {
  if (!file) throw new Error(`scene ${scene.id}: no file this viewer opens`);
  const pageable = file.format === "rad" || file.format === "athc";
  const mesh = new SplatMesh({
    url: file.url,
    fileName: file.path.split("?")[0].split("/").pop(),
    paged: pageable && file.paged !== false,
  });
  applyAtheneaTransform(mesh, file.transform ?? scene.transform);
  return mesh;
}
