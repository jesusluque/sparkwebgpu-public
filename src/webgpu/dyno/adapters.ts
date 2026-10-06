// WgpuDyno views of Spark's dyno users, so their graphs run on WebGPU with
// the per-frame context SparkRenderer would otherwise update.

import {
  type SplatEdit,
  type SplatEditSdf,
  SplatEdits,
  isSplatEdit,
  isSplatEditSdf,
} from "../../SplatEdit";
import type {
  CovSplatModifier,
  GsplatGenerator,
  GsplatModifier,
  SplatGenerator,
} from "../../SplatGenerator";
import {
  type SplatMesh,
  type SplatSource,
  maybeInjectSplatRgba,
} from "../../SplatMesh";
import { type Dyno, type IOTypes, dynoBlock } from "../../dyno/base";
import { CovSplat, Gsplat, splitGsplat } from "../../dyno/splats";
import { DynoBool } from "../../dyno/uniforms";
import type { WgpuDyno, WgpuDynoFrame } from "./DynoKernels";

// A modifier per wrapped object, so the graph keeps its identity across
// frames and isn't recompiled, until `version` (what its graph is built
// from) changes.
const wrapped = new WeakMap<
  object,
  { version: unknown; modifier: GsplatModifier }
>();
function asModifier(
  owner: object,
  version: unknown,
  modify: (g: Dyno<IOTypes, IOTypes>) => unknown,
): GsplatModifier {
  const cached = wrapped.get(owner);
  if (cached && cached.version === version) return cached.modifier;
  const modifier = dynoBlock(
    { gsplat: Gsplat },
    { gsplat: Gsplat },
    ({ gsplat }) => ({ gsplat: modify(gsplat as never) as never }),
  );
  wrapped.set(owner, { version, modifier });
  return modifier;
}

const wrappedCov = new WeakMap<
  object,
  { version: unknown; modifier: CovSplatModifier }
>();
function asCovModifier(
  owner: object,
  version: unknown,
  modify: (c: Dyno<IOTypes, IOTypes>) => unknown,
): CovSplatModifier {
  const cached = wrappedCov.get(owner);
  if (cached && cached.version === version) return cached.modifier;
  const modifier = dynoBlock(
    { covsplat: CovSplat },
    { covsplat: CovSplat },
    ({ covsplat }) => ({ covsplat: modify(covsplat as never) as never }),
  );
  wrappedCov.set(owner, { version, modifier });
  return modifier;
}

// Per mesh: whether it is drawn through LOD this frame (WgpuDynoFrame.lod).
const lodFlags = new WeakMap<SplatMesh, DynoBool<string>>();
function lodFlag(mesh: SplatMesh) {
  let flag = lodFlags.get(mesh);
  if (!flag) {
    flag = new DynoBool({ value: false });
    lodFlags.set(mesh, flag);
  }
  return flag;
}

// SplatMesh.splatRgba, the baked colours that replace the source's, as the
// first object modifier. By source index: meshes drawn through LOD leave it
// out, as SparkRenderer does.
function splatRgbaModifiers(mesh: SplatMesh): GsplatModifier[] {
  const rgba = mesh.splatRgba;
  if (!rgba) return [];
  return [
    // Keyed by the mesh's flag: meshes may share an RgbaArray.
    asModifier(lodFlag(mesh), rgba.dyno, (g) =>
      maybeInjectSplatRgba(
        g as never,
        rgba.dyno,
        splitGsplat(g as never).outputs.index,
        lodFlag(mesh),
      ),
    ),
  ];
}

/**
 * A SplatMesh's dyno pipeline on WebGPU: skinning and objectModifiers in
 * object space, SDF edits and worldModifiers in world space, as in
 * SplatMesh.constructGenerator, with its dyno context (time, transforms,
 * worldToView...) and SplatEdits updated each frame. Pass it to
 * WgpuSplatRenderer.add with the mesh as the object. Modifier changes are
 * picked up without updateGenerator(). `globalEdits` are SplatEdits outside
 * the mesh that apply to it (SparkRenderer finds those in the scene).
 *
 * A mesh with covSplats (render it with WgpuSplatRenderer's covSplats) runs
 * as SplatMesh.constructCovGenerator: skinning (either mode), its
 * covObjectModifiers, the full transform, SDF edits and covWorldModifiers on
 * the CovSplat; its Gsplat worldModifiers don't apply, as on WebGL.
 */
export function splatMeshDyno(
  mesh: SplatMesh,
  { globalEdits }: { globalEdits?: () => SplatEdit[] } = {},
): WgpuDyno {
  if (mesh.covSplats) return covSplatMeshDyno(mesh, globalEdits);
  return {
    get objectModifiers() {
      const mods = [...(mesh.objectModifiers ?? [])];
      const skinning = mesh.skinning;
      if (skinning) {
        mods.unshift(
          asModifier(skinning, skinning.uniform, (g) =>
            skinning.modify(g as never),
          ),
        );
      }
      return [...splatRgbaModifiers(mesh), ...mods];
    },
    get worldModifiers() {
      const mods = [...(mesh.worldModifiers ?? [])];
      const edits = mesh.rgbaDisplaceEdits;
      if (edits) {
        // Growing the edit capacity replaces the edits uniform array.
        mods.unshift(
          asModifier(edits, edits.dynoEdits, (g) => edits.modify(g as never)),
        );
      }
      return mods;
    },
    update(frame: WgpuDynoFrame) {
      updateSplatMeshContext(mesh, frame);
      updateSplatMeshEdits(mesh, globalEdits?.() ?? []);
    },
  };
}

/**
 * splatMeshDyno for a SplatMesh drawing a custom SplatSource (not a
 * PackedSplats, ExtSplats or PagedSplats): the source's fetchSplat graph is
 * the generator, read per splat index in the generate kernel, then the
 * mesh's pipeline runs as for its own splats (SplatMesh.constructGenerator /
 * constructCovGenerator). Use WgpuSplatRenderer.addGenerator with
 * source.getNumSplats() and the mesh as the object. The source's SH, if it
 * evaluates any, sees the camera in object space, as on WebGL.
 */
export function splatSourceMeshDyno(
  mesh: SplatMesh,
  source: SplatSource,
  options: { globalEdits?: () => SplatEdit[] } = {},
): WgpuDyno {
  const base = splatMeshDyno(mesh, options);
  const { context } = mesh;
  const generator: GsplatGenerator = dynoBlock(
    { index: "int" },
    { gsplat: Gsplat },
    ({ index }) => {
      if (!index) throw new Error("index is undefined");
      source.setMaxSh(mesh.maxSh);
      source.prepareFetchSplat();
      const viewOrigin = mesh.covSplats
        ? context.covViewToObject.offset
        : context.viewToObject.translate;
      return { gsplat: source.fetchSplat({ index, viewOrigin }) };
    },
  );
  // The base's getters stay live: modifiers changed later are picked up.
  return Object.defineProperties(
    { generator },
    Object.fromEntries(
      Object.entries(Object.getOwnPropertyDescriptors(base)).filter(
        ([key]) => key !== "generator",
      ),
    ),
  ) as WgpuDyno;
}

function covSplatMeshDyno(
  mesh: SplatMesh,
  globalEdits?: () => SplatEdit[],
): WgpuDyno {
  return {
    get objectModifiers() {
      return [...splatRgbaModifiers(mesh), ...(mesh.objectModifiers ?? [])];
    },
    get covObjectModifiers() {
      const mods = [...(mesh.covObjectModifiers ?? [])];
      const skinning = mesh.skinning;
      if (skinning) {
        mods.unshift(
          asCovModifier(skinning, skinning.uniform, (c) =>
            skinning.modifyCov(c as never),
          ),
        );
      }
      return mods;
    },
    get covWorldModifiers() {
      const mods = [...(mesh.covWorldModifiers ?? [])];
      const edits = mesh.rgbaDisplaceEdits;
      if (edits) {
        mods.unshift(
          asCovModifier(edits, edits.dynoEdits, (c) =>
            edits.modifyCov(c as never),
          ),
        );
      }
      return mods;
    },
    update(frame: WgpuDynoFrame) {
      updateSplatMeshContext(mesh, frame);
      updateSplatMeshEdits(mesh, globalEdits?.() ?? []);
    },
  };
}

/** The SplatEdits part of SplatMesh.update: gathers edits and their SDFs. */
export function updateSplatMeshEdits(
  mesh: SplatMesh,
  globalEdits: SplatEdit[],
) {
  if (!mesh.editable) return;
  const edits = (mesh.edits ?? []).concat(globalEdits);
  if (!mesh.edits) {
    mesh.traverseVisible((node) => {
      if (isSplatEdit(node)) edits.push(node);
    });
  }
  edits.sort((a, b) => a.ordering - b.ordering);
  const editsSdfs = edits.map((edit) => {
    if (edit.sdfs != null) return { edit, sdfs: edit.sdfs };
    const sdfs: SplatEditSdf[] = [];
    edit.traverseVisible((node) => {
      if (isSplatEditSdf(node)) sdfs.push(node);
    });
    return { edit, sdfs };
  });
  if (editsSdfs.length > 0 && !mesh.rgbaDisplaceEdits) {
    mesh.rgbaDisplaceEdits = new SplatEdits({
      maxEdits: editsSdfs.length,
      maxSdfs: editsSdfs.reduce((n, e) => n + e.sdfs.length, 0),
    });
  }
  mesh.rgbaDisplaceEdits?.update(editsSdfs);
}

/** The dyno uniforms SplatMesh.update sets, for WebGPU frames. */
export function updateSplatMeshContext(mesh: SplatMesh, frame: WgpuDynoFrame) {
  const { context } = mesh;
  lodFlag(mesh).value = frame.lod ?? false;
  context.time.value = frame.time;
  context.deltaTime.value = frame.deltaTime;
  (mesh.constructor as typeof SplatMesh).dynoTime.value = frame.time;
  context.transform.update(mesh);
  const viewToWorld = frame.camera.matrixWorld;
  context.viewToWorld.updateFromMatrix(viewToWorld);
  context.worldToView.updateFromMatrix(viewToWorld.clone().invert());
  const worldToObject = mesh.matrixWorld.clone().invert();
  context.viewToObject.updateFromMatrix(
    worldToObject.clone().multiply(viewToWorld),
  );
  context.covTransform.update(mesh);
  context.covViewToWorld.updateFromMatrix(viewToWorld);
  context.covWorldToView.updateFromMatrix(viewToWorld.clone().invert());
  context.covViewToObject.updateFromMatrix(worldToObject.multiply(viewToWorld));
  context.recolor.value.set(
    mesh.recolor.r,
    mesh.recolor.g,
    mesh.recolor.b,
    mesh.opacity,
  );
}

/**
 * A SplatGenerator (snow, static, user generators) on WebGPU: its generator
 * graph makes the splats, in world space as it applies its own transform, and
 * its frameUpdate runs each frame. Use WgpuSplatRenderer.addGenerator with
 * generator.numSplats.
 */
export function splatGeneratorDyno(generator: SplatGenerator): WgpuDyno {
  return {
    get generator() {
      return generator.generator;
    },
    worldSpace: true,
    update(frame: WgpuDynoFrame) {
      generator.updateMatrixWorld();
      generator.frameUpdate?.({
        renderer: null as never,
        object: generator,
        time: frame.time,
        deltaTime: frame.deltaTime,
        viewToWorld: frame.camera.matrixWorld,
        camera: frame.camera,
        globalEdits: [],
      });
    },
  };
}
