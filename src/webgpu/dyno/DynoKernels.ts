// Dyno graphs in the generate kernel: compiles a mesh's generator and
// modifiers to WGSL, patches them into the generate module at its hooks, and
// gives the dispatch its pipeline and dyno bindings. Pipelines are cached by
// the generated code, so meshes with the same graph structure share one;
// each mesh keeps its own compiled program for its uniform values.

import type * as THREE from "three";
import type {
  CovSplatModifier,
  GsplatGenerator,
  GsplatModifier,
} from "../../SplatGenerator";
import { type Dyno, type IOTypes, dynoBlock } from "../../dyno/base";
import { CovSplat, Gsplat } from "../../dyno/splats";
import { WgslDynoProgram, type WgslFunction } from "../../dyno/wgsl";
import type { KernelModule } from "../KernelModule";
import type { Kernel, KernelRegistry } from "../KernelRegistry";
import { type TextureLayouts, dynoResources, textureLayouts } from "./bindings";
import { type DynoHookFunctions, patchKernel } from "./patchKernel";
import { TextureCache } from "./textures";

export interface WgpuDynoFrame {
  camera: THREE.Camera;
  object: THREE.Object3D;
  /** Seconds, from performance.now(). */
  time: number;
  deltaTime: number;
  /** The mesh is drawn through LOD indices (its source is the LoD tree). */
  lod?: boolean;
}

/**
 * Dyno graphs for one mesh, as SplatMesh / SplatGenerator use them. Arrays
 * may be edited in place: the graph is recompiled when its members change.
 */
export interface WgpuDyno {
  /** Index -> Gsplat, replacing the read from the mesh's source. */
  generator?: GsplatGenerator;
  /** Applied in object space after the read (SplatMesh.objectModifiers). */
  objectModifiers?: GsplatModifier[];
  /** Applied in world space after transform and recolor. */
  worldModifiers?: GsplatModifier[];
  /**
   * With WgpuSplatRenderer's covSplats: CovSplat modifiers in object space
   * (before the full linear transform) and world space, as
   * SplatMesh.covObjectModifiers / covWorldModifiers. A mesh with
   * worldModifiers takes the similarity transform and runs both after it.
   */
  covObjectModifiers?: CovSplatModifier[];
  covWorldModifiers?: CovSplatModifier[];
  /** The generator emits world-space splats itself: skip the transform. */
  worldSpace?: boolean;
  /** Called each frame before generating, e.g. to set uniform values. */
  update?: (frame: WgpuDynoFrame) => void;
}

export interface DynoDispatch {
  kernel: Kernel;
  bindings: Record<string, GPUBindingResource>;
}

interface Compiled {
  graphs: unknown[];
  program: WgslDynoProgram;
  layouts: TextureLayouts;
  hooks: DynoHookFunctions;
  /** The program patched into each base kernel it ran in, by base name. */
  modules: Map<string, KernelModule>;
  /** What the last refresh saw: uniform bytes and texture versions. */
  state?: {
    uniforms: Uint8Array;
    textures: number[];
    sources: (THREE.Texture | undefined)[];
  };
}

function chain(modifiers: GsplatModifier[]): Dyno<IOTypes, IOTypes> {
  return dynoBlock({ gsplat: Gsplat }, { gsplat: Gsplat }, ({ gsplat }) => {
    let g = gsplat;
    for (const modifier of modifiers) {
      g = modifier.apply({ gsplat: g }).gsplat;
    }
    return { gsplat: g };
  }) as unknown as Dyno<IOTypes, IOTypes>;
}

function chainCov(modifiers: CovSplatModifier[]): Dyno<IOTypes, IOTypes> {
  return dynoBlock(
    { covsplat: CovSplat },
    { covsplat: CovSplat },
    ({ covsplat }) => {
      let c = covsplat;
      for (const modifier of modifiers) {
        c = modifier.apply({ covsplat: c }).covsplat;
      }
      return { covsplat: c };
    },
  ) as unknown as Dyno<IOTypes, IOTypes>;
}

function graphsOf(dyno: WgpuDyno): unknown[] {
  return [
    dyno.generator,
    ...(dyno.objectModifiers ?? []),
    null,
    ...(dyno.worldModifiers ?? []),
    null,
    ...(dyno.covObjectModifiers ?? []),
    null,
    ...(dyno.covWorldModifiers ?? []),
  ];
}

function sameGraphs(a: unknown[], b: unknown[]) {
  return a.length === b.length && a.every((x, i) => x === b[i]);
}

export class DynoKernels {
  readonly textures: TextureCache;
  private modules = new Map<string, KernelModule>();
  private compiled = new WeakMap<object, Compiled>();

  constructor(
    readonly registry: KernelRegistry,
    readonly base: KernelModule,
    readonly entry: string,
  ) {
    this.textures = new TextureCache(registry.device);
  }

  /** Whether the mesh's dyno has anything to run. */
  static active(dyno?: WgpuDyno): dyno is WgpuDyno {
    return Boolean(
      dyno &&
        (dyno.generator ||
          dyno.objectModifiers?.length ||
          dyno.worldModifiers?.length ||
          dyno.covObjectModifiers?.length ||
          dyno.covWorldModifiers?.length),
    );
  }

  /**
   * Runs the dyno updaters and reports whether the splats it makes may have
   * changed since the last call: a new graph, other uniform values or a
   * texture update. A renderer that skips unchanged frames regenerates then.
   */
  changed(owner: object, dyno: WgpuDyno): boolean {
    const before = this.compiled.get(owner);
    const compiled = this.compile(owner, dyno);
    const { program } = compiled;
    program.update();
    const uniforms = new Uint8Array(program.packUniforms());
    // The texture objects too: two uploaded once have the same version.
    const sources = program.backend.textures.map(
      (t) => t.uniform.value as THREE.Texture | undefined,
    );
    const textures = sources.map((t) => t?.version ?? -1);
    const last = compiled.state;
    compiled.state = { uniforms, textures, sources };
    return (
      before !== compiled ||
      !last ||
      last.uniforms.length !== uniforms.length ||
      last.uniforms.some((b, i) => b !== uniforms[i]) ||
      last.textures.some((v, i) => v !== textures[i]) ||
      last.sources.some((t, i) => t !== sources[i])
    );
  }

  /**
   * The kernel and dyno bindings for `owner`'s next dispatch, patched into
   * `base` (a plugin preset's generate variant; this.base by default).
   */
  prepare(
    owner: object,
    dyno: WgpuDyno,
    base: KernelModule = this.base,
  ): DynoDispatch {
    const compiled = this.compile(owner, dyno);
    const { program, layouts } = compiled;
    const module = this.patched(compiled, base);
    program.update();
    const bindings = dynoResources(program, layouts, this.textures, (data) =>
      this.registry.uniforms.push(data),
    );
    return { kernel: this.registry.get(module, this.entry), bindings };
  }

  // A WGSL error only shows as an invalid pipeline: say where it is, which
  // matters most for hand-written dyno WGSL.
  private reportErrors(module: KernelModule) {
    const lines = module.wgsl.split("\n");
    this.registry
      .shaderModule(module)
      .getCompilationInfo()
      .then((info) => {
        const errors = info.messages
          .filter((m) => m.type === "error")
          .map(
            (m) =>
              `${m.lineNum}:${m.linePos} ${m.message}\n    ${lines[m.lineNum - 1]?.trim()}`,
          );
        if (errors.length) {
          console.error(
            `${module.name}: dyno WGSL errors\n${errors.join("\n")}`,
          );
        }
      });
  }

  private compile(owner: object, dyno: WgpuDyno): Compiled {
    const graphs = graphsOf(dyno);
    const cached = this.compiled.get(owner);
    if (cached && sameGraphs(cached.graphs, graphs)) return cached;

    const functions: WgslFunction[] = [];
    const hooks: DynoHookFunctions = {};
    if (dyno.generator) {
      functions.push({
        name: "dyno_source",
        graph: dyno.generator as unknown as Dyno<IOTypes, IOTypes>,
        inputs: { index: "int" },
        output: "gsplat",
      });
      hooks.dynoSource = "dyno_source";
    }
    if (dyno.objectModifiers?.length) {
      functions.push({
        name: "dyno_object",
        graph: chain(dyno.objectModifiers),
        inputs: { gsplat: Gsplat },
        output: "gsplat",
      });
      hooks.dynoObjectModifier = "dyno_object";
    }
    if (dyno.worldModifiers?.length) {
      functions.push({
        name: "dyno_world",
        graph: chain(dyno.worldModifiers),
        inputs: { gsplat: Gsplat },
        output: "gsplat",
      });
      hooks.dynoWorldModifier = "dyno_world";
    }
    if (dyno.covObjectModifiers?.length) {
      functions.push({
        name: "dyno_cov_object",
        graph: chainCov(dyno.covObjectModifiers),
        inputs: { covsplat: CovSplat },
        output: "covsplat",
      });
      hooks.dynoCovObjectModifier = "dyno_cov_object";
    }
    if (dyno.covWorldModifiers?.length) {
      functions.push({
        name: "dyno_cov_world",
        graph: chainCov(dyno.covWorldModifiers),
        inputs: { covsplat: CovSplat },
        output: "covsplat",
      });
      hooks.dynoCovWorldModifier = "dyno_cov_world";
    }
    const program = new WgslDynoProgram({ functions, group: 1 });

    const layouts = textureLayouts(program);
    const compiled = { graphs, program, layouts, hooks, modules: new Map() };
    this.compiled.set(owner, compiled);
    return compiled;
  }

  // The compiled program patched into `base`, shared by programs with the
  // same code.
  private patched(compiled: Compiled, base: KernelModule): KernelModule {
    let module = compiled.modules.get(base.name);
    if (module) return module;
    const { program, layouts } = compiled;
    const key = `${base.name}\n${program.code}\n${JSON.stringify(layouts)}`;
    module = this.modules.get(key);
    if (!module) {
      module = patchKernel(
        base,
        program,
        compiled.hooks,
        `${base.name}+dyno${this.modules.size}`,
        layouts,
      );
      this.modules.set(key, module);
      this.reportErrors(module);
    }
    compiled.modules.set(base.name, module);
    return module;
  }
}
