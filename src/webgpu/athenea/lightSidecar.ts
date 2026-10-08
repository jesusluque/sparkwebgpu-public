// athenea's light sidecar (proposals 062 §7, 066, 067 appendix B, 076):
// the `.lights.usda` half -- light groups, named states, rules between
// groups, sequences and each lamp's time curve -- and the evaluator that
// turns them into one weight a group, w_k(t), every frame.
//
// The sidecar is plain USD (usda.ts reads it). Until athenea's schema
// AtheneaLightGroupAPI exists, its data are namespaced attributes on Scope
// prims, as 067's draft writes them; a prim with the API applied is read the
// same way. Attributes read (all optional unless noted):
//
//   a group (a prim with any athenea:lightGroup:* attribute, or the API):
//     rel     athenea:lightGroup:members        source prims (traceability)
//     int2[]  athenea:lightGroup:splatRanges    emitting splats [lo, hi) of the cloud
//     token   athenea:lightGroup:function       lowBeam, highBeam, daytimeRunning, tail, brake, indicator...
//     token   athenea:lightGroup:side           left, right, centre
//     token   athenea:lightGroup:technology     led (default), halogen, xenon, blinker, lightGuide
//     float   athenea:lightGroup:radiance       the lamp's "on" radiance, nits (default 1)
//     float   athenea:lightGroup:emissionLuminance  athenea's corvette.lights.usda: the radiance, where
//                                               `radiance` is not given (its per-group bakes are per unit
//                                               of it, baked white)
//     color3f athenea:lightGroup:emissionColor  athenea's: the colour, where `color` is not given
//     color3f athenea:lightGroup:color          rgb (default 1, 1, 1)
//     float   athenea:lightGroup:temperatureK   blackbody colour (times `color`); xenon/halogen: the final one
//     float   athenea:lightGroup:startTemperatureK  xenon's (default 4300), halogen's cold glow (default 1000)
//     float   athenea:lightGroup:riseSeconds    on ramp (led), time constant (halogen), warm-up (xenon)
//     float   athenea:lightGroup:fallSeconds    off ramp / time constant
//     float   athenea:lightGroup:delaySeconds   lag after a change (sequential segments)
//     float   athenea:lightGroup:blinkHz        blinker frequency (default 1.5: 90 a minute, UN R48)
//     float   athenea:lightGroup:dutyCycle      blinker on fraction (default 0.5)
//     color3f athenea:lightGroup:lensTint       076 (a): the lens's mean transmittance, over the .athl's
//     bool    athenea:lightGroup:initialOn      on at load (level 1)
//     asset   shaping:ies:file                  076's author profile (read, not yet evaluated)
//   a state:    dictionary athenea:lightState:targets = { double <group> = level }
//               or double athenea:lightState:target:<group> = level (athenea's form: USD has no
//               dictionary-valued attributes); any prim under a `LightStates` scope is a state,
//               one with no targets every group off
//               token athenea:lightState:base   another state its targets start from
//   a rule:     token athenea:lightRule:when    condition: group names, state:<name>, !, &&, ||, ( )
//               token athenea:lightRule:target  (or token[]) the groups it acts on
//               double athenea:lightRule:scale  "scale" (default) multiplies, "set" sets, "max" raises to
//               token athenea:lightRule:mode    scale | set | max | sync
//                     sync: the targets blink in phase with the first group named in `when`
//   a sequence: double[] athenea:lightSequence:times, token[] athenea:lightSequence:states
//               bool athenea:lightSequence:loop, double athenea:lightSequence:duration
//               double athenea:lightSequence:level:<group>.timeSamples = { t: level }   a keyframed curve
//   the sidecar: string athenea:lightSidecar:cloudHash (066 §2.4 validity), float
//               athenea:lightSidecar:nitsPerUnit (scene units, default 1), asset athenea:lightSidecar:athl,
//               token athenea:lightSidecar:defaultState
//
// Evaluation (066 §3, 062 §7): the targets come from the state (or the
// sequence's state at that time), then manual levels, then the rules in
// prim order, each condition read on the targets as the earlier rules left
// them; each group then moves from where it was towards its target along
// its technology's curve, from the time the target changed. The weight is
//   w_k(t) = level_k(t) * radiance_k / nitsPerUnit * colour_k(t)
// RGB, with colour_k the group's colour times the blackbody of its
// temperature (athenea's lights.slang blackbody, luminance 1). Rules and
// curves are control, not arithmetic over the cloud: a few dozen scalars a
// frame on the CPU, and the GPU sees only the K weights.

import {
  type UsdLayer,
  type UsdPrim,
  type UsdProperty,
  type UsdValue,
  apiSchemasOf,
  parseUsda,
  walkPrims,
} from "./usda";

type Rgb = [number, number, number];

export type LampTechnology =
  | "led"
  | "halogen"
  | "xenon"
  | "blinker"
  | "lightGuide";

export interface LightGroupSpec {
  name: string;
  /** Index among the sidecar's groups (prim order). */
  index: number;
  path: string;
  members: string[];
  splatRanges: [number, number][];
  function: string;
  side: string;
  technology: LampTechnology;
  radiance: number;
  color: Rgb;
  temperatureK: number | null;
  startTemperatureK: number;
  riseSeconds: number;
  fallSeconds: number;
  delaySeconds: number;
  blinkHz: number;
  dutyCycle: number;
  lensTint: Rgb | null;
  initialOn: boolean;
  iesFile: string | null;
}

export interface LightStateSpec {
  name: string;
  path: string;
  /** group -> level (0 off, 1 on, more for a shared lamp's brighter use). */
  targets: Record<string, number>;
}

export type RuleMode = "scale" | "set" | "max" | "sync";

export interface LightRuleSpec {
  name: string;
  path: string;
  when: string;
  targets: string[];
  scale: number;
  mode: RuleMode;
}

export interface LightSequenceSpec {
  name: string;
  path: string;
  /** (time, state) keys, sorted. */
  keys: { time: number; state: string }[];
  loop: boolean;
  duration: number;
  /** Keyframed levels by group: (time, level), linear between keys. */
  curves: Record<string, [number, number][]>;
}

export interface LightSidecar {
  groups: LightGroupSpec[];
  states: LightStateSpec[];
  rules: LightRuleSpec[];
  sequences: LightSequenceSpec[];
  cloudHash: string | null;
  nitsPerUnit: number;
  athl: string | null;
  defaultState: string | null;
  /** What was read but ignored or suspicious. */
  warnings: string[];
}

const G = "athenea:lightGroup:";
const S = "athenea:lightState:";
const R = "athenea:lightRule:";
const Q = "athenea:lightSequence:";
const C = "athenea:lightSidecar:";

const num = (v: UsdValue | undefined, fallback: number): number =>
  typeof v === "number" && Number.isFinite(v) ? v : fallback;
const str = (v: UsdValue | undefined, fallback: string): string =>
  typeof v === "string" ? v : fallback;
const rgb = (v: UsdValue | undefined): Rgb | null =>
  Array.isArray(v) && v.length === 3 && v.every((x) => typeof x === "number")
    ? (v as Rgb)
    : null;
const strings = (v: UsdValue | undefined): string[] =>
  Array.isArray(v)
    ? v.filter((x): x is string => typeof x === "string")
    : typeof v === "string"
      ? [v]
      : [];

function assetOf(v: UsdValue | undefined): string | null {
  const a =
    v && typeof v === "object" && !Array.isArray(v)
      ? (v as { asset?: unknown }).asset
      : null;
  return typeof a === "string" ? a : null;
}

function attr(p: UsdPrim, name: string): UsdValue | undefined {
  return p.properties.get(name)?.value;
}

const TECHNOLOGIES: readonly LampTechnology[] = [
  "led",
  "halogen",
  "xenon",
  "blinker",
  "lightGuide",
];

/** Defaults by lamp (062 §7.2; search figures, not measurements). */
const RISE: Record<LampTechnology, number> = {
  led: 0.05,
  halogen: 0.08,
  xenon: 4,
  blinker: 0.02,
  lightGuide: 0.05,
};
const FALL: Record<LampTechnology, number> = {
  led: 0.05,
  halogen: 0.12,
  xenon: 0.05,
  blinker: 0.02,
  lightGuide: 0.05,
};

function groupOf(
  p: UsdPrim,
  index: number,
  warnings: string[],
): LightGroupSpec {
  const get = (k: string) => attr(p, G + k);
  let technology = str(get("technology"), "led") as LampTechnology;
  if (!TECHNOLOGIES.includes(technology)) {
    warnings.push(`${p.path}: unknown technology '${technology}', read as led`);
    technology = "led";
  }
  const fn = str(get("function"), "");
  if (
    fn === "indicator" &&
    technology === "led" &&
    get("technology") === undefined
  ) {
    technology = "blinker";
  }
  const members = p.properties.get(`${G}members`);
  const ranges = get("splatRanges");
  const splatRanges: [number, number][] = [];
  if (Array.isArray(ranges)) {
    for (const r of ranges) {
      if (Array.isArray(r) && r.length === 2) {
        splatRanges.push([num(r[0], 0), num(r[1], 0)]);
      }
    }
  }
  const t = get("temperatureK");
  const athenea =
    get("radiance") === undefined && get("emissionLuminance") !== undefined;
  const ies = p.properties.get("shaping:ies:file")?.value;
  return {
    name: p.name,
    index,
    path: p.path,
    members: pathsOf(members),
    splatRanges,
    function: fn,
    side: str(get("side"), ""),
    technology,
    radiance: num(get("radiance"), num(get("emissionLuminance"), 1)),
    // athenea bakes a group white, per unit of emissionLuminance: its
    // emissionColor goes in the weight (corvette-lights NOTES.md).
    color: rgb(get("color")) ?? rgb(get("emissionColor")) ?? [1, 1, 1],
    // athenea's own weight (and its Cycles reference) is emissionLuminance
    // times emissionColor alone: its temperatureK waits for the lamp curves
    // (v2), so a group written that way is not tinted by it.
    temperatureK: typeof t === "number" && !athenea ? t : null,
    startTemperatureK: num(
      get("startTemperatureK"),
      technology === "halogen" ? 1000 : 4300,
    ),
    riseSeconds: Math.max(num(get("riseSeconds"), RISE[technology]), 0),
    fallSeconds: Math.max(num(get("fallSeconds"), FALL[technology]), 0),
    delaySeconds: Math.max(num(get("delaySeconds"), 0), 0),
    blinkHz: Math.max(num(get("blinkHz"), 1.5), 1e-3),
    dutyCycle: Math.min(Math.max(num(get("dutyCycle"), 0.5), 0), 1),
    lensTint: rgb(get("lensTint")),
    initialOn: get("initialOn") === true,
    iesFile: assetOf(ies),
  };
}

function pathsOf(p: UsdProperty | undefined): string[] {
  const v = p?.value;
  const list = Array.isArray(v) ? v : v ? [v] : [];
  return list
    .map((x) => (x && typeof x === "object" && "path" in x ? x.path : null))
    .filter((x): x is string => x !== null);
}

const hasPrefix = (p: UsdPrim, prefix: string) => {
  for (const k of p.properties.keys()) if (k.startsWith(prefix)) return true;
  return false;
};

/** The sidecar's model from a parsed layer. */
export function readLightSidecar(layer: UsdLayer): LightSidecar {
  const out: LightSidecar = {
    groups: [],
    states: [],
    rules: [],
    sequences: [],
    cloudHash: null,
    nitsPerUnit: 1,
    athl: null,
    defaultState: null,
    warnings: [],
  };
  const bases: Record<string, string> = {};
  for (const p of walkPrims(layer.prims)) {
    if (p.specifier === "class") continue;
    if (hasPrefix(p, G) || apiSchemasOf(p).includes("AtheneaLightGroupAPI")) {
      out.groups.push(groupOf(p, out.groups.length, out.warnings));
    }
    const targets = attr(p, `${S}targets`);
    const segments = p.path.split("/");
    if (
      targets !== undefined ||
      hasPrefix(p, `${S}target:`) ||
      segments[segments.length - 2] === "LightStates"
    ) {
      const t: Record<string, number> = {};
      if (targets && typeof targets === "object" && !Array.isArray(targets)) {
        for (const [k, v] of Object.entries(targets)) {
          if (typeof v === "number") t[k] = v;
          else if (typeof v === "boolean") t[k] = v ? 1 : 0;
        }
      }
      for (const [k, prop] of p.properties) {
        if (!k.startsWith(`${S}target:`)) continue;
        const v = prop.value;
        if (typeof v === "number") t[k.slice(`${S}target:`.length)] = v;
        else if (typeof v === "boolean")
          t[k.slice(`${S}target:`.length)] = v ? 1 : 0;
      }
      out.states.push({ name: p.name, path: p.path, targets: t });
      const base = attr(p, `${S}base`);
      if (typeof base === "string") bases[p.name] = base;
    }
    if (hasPrefix(p, R)) {
      const mode = str(attr(p, `${R}mode`), "scale") as RuleMode;
      out.rules.push({
        name: p.name,
        path: p.path,
        when: str(attr(p, `${R}when`), ""),
        targets: strings(attr(p, `${R}target`)),
        scale: num(attr(p, `${R}scale`), mode === "max" ? 1 : 0),
        mode: ["scale", "set", "max", "sync"].includes(mode) ? mode : "scale",
      });
    }
    if (hasPrefix(p, Q)) {
      const times = attr(p, `${Q}times`);
      const states = strings(attr(p, `${Q}states`));
      const keys = (Array.isArray(times) ? times : [])
        .map((t, k) => ({ time: num(t, 0), state: states[k] ?? "" }))
        .filter((k) => k.state)
        .sort((a, b) => a.time - b.time);
      const curves: Record<string, [number, number][]> = {};
      let last = keys.length ? keys[keys.length - 1].time : 0;
      for (const [name, prop] of p.properties) {
        if (!name.startsWith(`${Q}level:`) || !prop.timeSamples) continue;
        const samples = prop.timeSamples
          .filter(([, v]) => typeof v === "number")
          .map(([t, v]) => [t, v as number] as [number, number]);
        curves[name.slice(`${Q}level:`.length)] = samples;
        if (samples.length)
          last = Math.max(last, samples[samples.length - 1][0]);
      }
      out.sequences.push({
        name: p.name,
        path: p.path,
        keys,
        loop: attr(p, `${Q}loop`) === true,
        duration: Math.max(num(attr(p, `${Q}duration`), last), 1e-6),
        curves,
      });
    }
    const hash = attr(p, `${C}cloudHash`);
    if (typeof hash === "string") out.cloudHash = hash;
    const npu = attr(p, `${C}nitsPerUnit`);
    if (typeof npu === "number" && npu > 0) out.nitsPerUnit = npu;
    const athl = attr(p, `${C}athl`);
    if (assetOf(athl) !== null) out.athl = assetOf(athl);
    const def = attr(p, `${C}defaultState`);
    if (typeof def === "string") out.defaultState = def;
  }
  // States built on others: the base's targets, then their own.
  const byName = new Map(out.states.map((s) => [s.name, s]));
  const resolve = (
    s: LightStateSpec,
    seen: Set<string>,
  ): Record<string, number> => {
    const base = bases[s.name];
    if (!base || seen.has(base)) return s.targets;
    const b = byName.get(base);
    if (!b) {
      out.warnings.push(`state ${s.name}: unknown base '${base}'`);
      return s.targets;
    }
    seen.add(s.name);
    return { ...resolve(b, seen), ...s.targets };
  };
  for (const s of out.states) s.targets = resolve(s, new Set());
  const names = new Set(out.groups.map((g) => g.name));
  for (const s of out.states) {
    for (const k of Object.keys(s.targets)) {
      if (!names.has(k))
        out.warnings.push(`state ${s.name}: unknown group '${k}'`);
    }
  }
  for (const r of out.rules) {
    for (const k of r.targets) {
      if (!names.has(k))
        out.warnings.push(`rule ${r.name}: unknown group '${k}'`);
    }
    try {
      compileCondition(r.when, names);
    } catch (e) {
      out.warnings.push(`rule ${r.name}: ${(e as Error).message}`);
    }
  }
  for (const q of out.sequences) {
    for (const k of q.keys) {
      if (!byName.has(k.state)) {
        out.warnings.push(`sequence ${q.name}: unknown state '${k.state}'`);
      }
    }
  }
  return out;
}

/** Parses a `.lights.usda` sidecar. */
export function parseLightSidecar(text: string): LightSidecar {
  return readLightSidecar(parseUsda(text));
}

// ---------------------------------------------------------------------------
// Conditions: `cruce`, `!cruce`, `cruce && !largas`, `state:diurno || (a)`.

type Cond = (on: (group: string) => boolean, state: string | null) => boolean;

export function compileCondition(
  text: string,
  groups: ReadonlySet<string>,
): Cond {
  const tokens = text.match(/!|&&?|\|\|?|\(|\)|[^\s!&|()]+/g) ?? [];
  let at = 0;
  const peek = () => tokens[at];
  const or = (): Cond => {
    let a = and();
    while (peek() === "||" || peek() === "|") {
      at += 1;
      const l = a;
      const r = and();
      a = (on, s) => l(on, s) || r(on, s);
    }
    return a;
  };
  const and = (): Cond => {
    let a = unary();
    while (peek() === "&&" || peek() === "&") {
      at += 1;
      const l = a;
      const r = unary();
      a = (on, s) => l(on, s) && r(on, s);
    }
    return a;
  };
  const unary = (): Cond => {
    const t = tokens[at++];
    if (t === "!") {
      const a = unary();
      return (on, s) => !a(on, s);
    }
    if (t === "(") {
      const a = or();
      if (tokens[at++] !== ")")
        throw new Error(`condition '${text}': missing ')'`);
      return a;
    }
    if (t === undefined) throw new Error(`condition '${text}' ends early`);
    if (t === "true") return () => true;
    if (t === "false") return () => false;
    if (t.startsWith("state:")) {
      const name = t.slice(6);
      return (_on, s) => s === name;
    }
    if (!groups.has(t))
      throw new Error(`condition '${text}': unknown group '${t}'`);
    return (on) => on(t);
  };
  if (!tokens.length) return () => true;
  const c = or();
  if (at !== tokens.length)
    throw new Error(`condition '${text}': unexpected '${tokens[at]}'`);
  return c;
}

// ---------------------------------------------------------------------------
// Colour

/**
 * athenea's blackbody (light/lights.slang, vendored): Krystek's fit of the
 * Planckian locus, to linear Rec.709 normalised to luminance 1.
 */
export function blackbody(kelvin: number): Rgb {
  const t = Math.min(Math.max(kelvin, 1000), 15000);
  const t2 = t * t;
  const f = Math.fround;
  const u =
    (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t2) /
    (1 + 8.42420235e-4 * t + 7.08145163e-7 * t2);
  const v =
    (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t2) /
    (1 - 2.89741816e-5 * t + 1.61456053e-7 * t2);
  const d = 2 * u - 8 * v + 4;
  const x = (3 * u) / d;
  const y = (2 * v) / d;
  const z = 1 - x - y;
  const X = x / Math.max(y, 1e-6);
  const Z = z / Math.max(y, 1e-6);
  const r = 3.2404542 * X - 1.5371385 - 0.4985314 * Z;
  const g = -0.969266 * X + 1.8760108 + 0.041556 * Z;
  const b = 0.0556434 * X - 0.2040259 + 1.0572252 * Z;
  const p = [Math.max(r, 0), Math.max(g, 0), Math.max(b, 0)];
  const lum = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
  return (lum > 0 ? p.map((c) => f(c / lum)) : [1, 1, 1]) as Rgb;
}

// ---------------------------------------------------------------------------
// The evaluator

interface Transition {
  from: number;
  to: number;
  /** When the target changed (the curve starts after the group's delay). */
  t0: number;
  /** Blink phase anchor. */
  phase: number;
}

export interface LightWeights {
  /** Per group: r, g, b, level (the curve's output, before colour). */
  weights: Float32Array;
  /** Per group: the final target (after rules). */
  targets: Float32Array;
}

/**
 * w_k(t) for a sidecar: commands (setState, setLevel, play) are stamped
 * with their time; evaluate(t) gives every group's weight at t. Times are
 * seconds on one clock (the frame's), non-decreasing between evaluations.
 */
export class LightRig {
  readonly sidecar: LightSidecar;
  /** A multiplier on every weight (an exposure or a dimmer). */
  master = 1;
  private readonly index = new Map<string, number>();
  private readonly stateByName: Map<string, LightStateSpec>;
  private readonly rules: { spec: LightRuleSpec; cond: Cond }[];
  private state: string | null = null;
  private manual = new Map<string, number>();
  private sequence: { spec: LightSequenceSpec; start: number } | null = null;
  private changed = 0;
  private readonly transitions: Transition[];
  private lastTime = Number.NEGATIVE_INFINITY;

  constructor(sidecar: LightSidecar, time = 0) {
    this.sidecar = sidecar;
    sidecar.groups.forEach((g, k) => this.index.set(g.name, k));
    this.stateByName = new Map(sidecar.states.map((s) => [s.name, s]));
    const names = new Set(sidecar.groups.map((g) => g.name));
    this.rules = sidecar.rules.flatMap((spec) => {
      try {
        return [{ spec, cond: compileCondition(spec.when, names) }];
      } catch {
        return [];
      }
    });
    this.transitions = sidecar.groups.map((g) => {
      const on = g.initialOn ? 1 : 0;
      return { from: on, to: on, t0: time - 1e6, phase: time };
    });
    if (sidecar.defaultState && this.stateByName.has(sidecar.defaultState)) {
      this.state = sidecar.defaultState;
      // Settled at load: no transition from black.
      const t = this.targetsAt(time).targets;
      t.forEach((v, k) => {
        this.transitions[k] = { from: v, to: v, t0: time - 1e6, phase: time };
      });
    }
    this.changed = time;
    this.lastTime = time;
  }

  get groupNames(): string[] {
    return this.sidecar.groups.map((g) => g.name);
  }
  get stateNames(): string[] {
    return this.sidecar.states.map((s) => s.name);
  }
  get sequenceNames(): string[] {
    return this.sidecar.sequences.map((s) => s.name);
  }
  get currentState(): string | null {
    return this.sequence
      ? this.sequenceState(Math.max(this.lastTime, this.sequence.start))
      : this.state;
  }
  get playing(): string | null {
    return this.sequence?.spec.name ?? null;
  }

  /** Switches to a named state at `time` (stops a sequence; keeps manual levels). */
  setState(name: string | null, time: number) {
    if (name !== null && !this.stateByName.has(name)) {
      throw new Error(
        `LightRig: no state '${name}' (${this.stateNames.join(", ")})`,
      );
    }
    this.state = name;
    this.sequence = null;
    this.changed = time;
  }

  /** Overrides one group's target (null: back to the state's). */
  setLevel(group: string, level: number | null, time: number) {
    if (!this.index.has(group))
      throw new Error(`LightRig: no group '${group}'`);
    if (level === null) this.manual.delete(group);
    else this.manual.set(group, level);
    this.changed = time;
  }

  clearLevels(time: number) {
    this.manual.clear();
    this.changed = time;
  }

  /** Plays a sequence from `time`. */
  play(name: string, time: number) {
    const spec = this.sidecar.sequences.find((s) => s.name === name);
    if (!spec) throw new Error(`LightRig: no sequence '${name}'`);
    this.sequence = { spec, start: time };
    this.changed = time;
  }

  /** Stops the sequence, holding the state it was in. */
  stop(time: number) {
    if (!this.sequence) return;
    this.state = this.sequenceState(time);
    this.sequence = null;
    this.changed = time;
  }

  private sequenceLocal(time: number): number {
    const q = this.sequence as { spec: LightSequenceSpec; start: number };
    const local = Math.max(time - q.start, 0);
    return q.spec.loop
      ? ((local % q.spec.duration) + q.spec.duration) % q.spec.duration
      : local;
  }

  private sequenceState(time: number): string | null {
    if (!this.sequence) return this.state;
    const local = this.sequenceLocal(time);
    let s: string | null = this.state;
    for (const k of this.sequence.spec.keys) if (k.time <= local) s = k.state;
    return s;
  }

  /** The time of the last sequence key at or before `time` (when it changed). */
  private sequenceKeyTime(time: number): number {
    const q = this.sequence;
    if (!q) return this.changed;
    const local = this.sequenceLocal(time);
    let key = 0;
    for (const k of q.spec.keys) if (k.time <= local) key = k.time;
    return time - (local - key);
  }

  /** Targets at `time`, before curves: after state, sequence, manual levels and rules. */
  targetsAt(time: number): { targets: number[]; syncTo: number[] } {
    const K = this.sidecar.groups.length;
    const t = new Array<number>(K).fill(0);
    const stateName = this.sequence ? this.sequenceState(time) : this.state;
    const state = stateName ? this.stateByName.get(stateName) : undefined;
    if (state) {
      for (const [g, v] of Object.entries(state.targets)) {
        const k = this.index.get(g);
        if (k !== undefined) t[k] = v;
      }
    }
    if (this.sequence) {
      const local = this.sequenceLocal(time);
      for (const [g, samples] of Object.entries(this.sequence.spec.curves)) {
        const k = this.index.get(g);
        if (k !== undefined && samples.length)
          t[k] = sampleCurve(samples, local);
      }
    }
    for (const [g, v] of this.manual) t[this.index.get(g) as number] = v;
    const syncTo = new Array<number>(K).fill(-1);
    const on = (g: string) => t[this.index.get(g) as number] > 0;
    for (const { spec, cond } of this.rules) {
      if (!cond(on, stateName)) continue;
      for (const g of spec.targets) {
        const k = this.index.get(g);
        if (k === undefined) continue;
        if (spec.mode === "scale") t[k] *= spec.scale;
        else if (spec.mode === "set") t[k] = spec.scale;
        else if (spec.mode === "max") t[k] = Math.max(t[k], spec.scale);
        else if (spec.mode === "sync") {
          const lead = /[^\s!&|()]+/.exec(spec.when)?.[0];
          const j = lead ? this.index.get(lead) : undefined;
          if (j !== undefined) syncTo[k] = j;
        }
      }
    }
    return { targets: t, syncTo };
  }

  /** Every group's weight at `time` (seconds, the frame's clock). */
  evaluate(time: number, out?: LightWeights): LightWeights {
    const K = this.sidecar.groups.length;
    const result = out ?? {
      weights: new Float32Array(4 * K),
      targets: new Float32Array(K),
    };
    const { targets, syncTo } = this.targetsAt(time);
    // When the targets changed: the command's time, or the sequence key's.
    const when = Math.min(
      this.sequence ? this.sequenceKeyTime(time) : this.changed,
      time,
    );
    for (let k = 0; k < K; k++) {
      const tr = this.transitions[k];
      if (targets[k] !== tr.to) {
        const g = this.sidecar.groups[k];
        const was = levelAt(g, tr, Math.max(when, tr.t0));
        const switchedOn = tr.to <= 0 && targets[k] > 0;
        this.transitions[k] = {
          from: was.base,
          to: targets[k],
          t0: when,
          phase: switchedOn ? when + g.delaySeconds : tr.phase,
        };
      }
    }
    for (let k = 0; k < K; k++) {
      if (syncTo[k] >= 0)
        this.transitions[k].phase = this.transitions[syncTo[k]].phase;
    }
    const npu = this.sidecar.nitsPerUnit;
    for (let k = 0; k < K; k++) {
      const g = this.sidecar.groups[k];
      const { level, kelvin } = levelAt(g, this.transitions[k], time);
      const colour = groupColour(g, kelvin);
      const s = (level * g.radiance * this.master) / npu;
      result.weights[4 * k] = s * colour[0];
      result.weights[4 * k + 1] = s * colour[1];
      result.weights[4 * k + 2] = s * colour[2];
      result.weights[4 * k + 3] = level;
      result.targets[k] = targets[k];
    }
    this.lastTime = time;
    return result;
  }

  /** Whether any group is still moving at `time` (the frame must be redrawn). */
  animating(time: number): boolean {
    if (this.sequence) return true;
    return this.sidecar.groups.some((g, k) => {
      const tr = this.transitions[k];
      if (g.technology === "blinker" && tr.to > 0) return true;
      return time < tr.t0 + g.delaySeconds + settleTime(g, tr);
    });
  }
}

function sampleCurve(s: [number, number][], t: number): number {
  if (t <= s[0][0]) return s[0][1];
  for (let i = 1; i < s.length; i++) {
    if (t <= s[i][0]) {
      const [t0, v0] = s[i - 1];
      const [t1, v1] = s[i];
      return t1 > t0 ? v0 + ((v1 - v0) * (t - t0)) / (t1 - t0) : v1;
    }
  }
  return s[s.length - 1][1];
}

function settleTime(g: LightGroupSpec, tr: Transition): number {
  const up = tr.to > tr.from;
  const d = up ? g.riseSeconds : g.fallSeconds;
  // Exponential curves settle (to 0.1 %) in about 7 time constants.
  return g.technology === "halogen" ? 7 * d : d;
}

const smoothstep = (x: number) => {
  const t = Math.min(Math.max(x, 0), 1);
  return t * t * (3 - 2 * t);
};

/**
 * The lamp's curve (062 §7.2): its level and colour temperature `dt`
 * after a change from `from` to `to`. `base` is the level without the
 * blinker's square wave (what a later change starts from).
 */
export function levelAt(
  g: LightGroupSpec,
  tr: Transition,
  time: number,
): { level: number; base: number; kelvin: number | null } {
  const dt = time - tr.t0 - g.delaySeconds;
  const { from, to } = tr;
  const up = to > from;
  let base: number;
  let kelvin = g.temperatureK;
  if (dt <= 0) {
    base = from;
  } else {
    switch (g.technology) {
      case "halogen": {
        const tau = up ? g.riseSeconds : g.fallSeconds;
        base = tau > 0 ? to + (from - to) * Math.exp(-dt / tau) : to;
        break;
      }
      case "xenon": {
        if (up) {
          // The arc strikes at once; the flux and the colour come up over
          // the warm-up, cool to cold-white (4300 K -> 6000 K).
          const w = g.riseSeconds > 0 ? smoothstep(dt / g.riseSeconds) : 1;
          base = from + (to - from) * (0.3 + 0.7 * w);
          if (kelvin !== null) {
            kelvin = g.startTemperatureK + (kelvin - g.startTemperatureK) * w;
          }
        } else {
          const d = g.fallSeconds;
          base = d > 0 ? from + (to - from) * Math.min(dt / d, 1) : to;
        }
        break;
      }
      default: {
        const d = up ? g.riseSeconds : g.fallSeconds;
        base = d > 0 ? from + (to - from) * Math.min(dt / d, 1) : to;
      }
    }
  }
  if (g.technology === "halogen" && kelvin !== null) {
    // The filament glows red while it heats: temperature with the fourth
    // root of the relative output (Stefan-Boltzmann; a heuristic, 062 §7.2
    // marks the timing unverified).
    const on = Math.max(to, from, 1e-6);
    const rel = Math.min(Math.max(base / on, 0), 1);
    kelvin = g.startTemperatureK + (kelvin - g.startTemperatureK) * rel ** 0.25;
  }
  let level = base;
  if (g.technology === "blinker" && to > 0) {
    const period = 1 / g.blinkHz;
    const local = time - tr.phase;
    if (local < 0) level = 0;
    else {
      const p = (local % period) / period;
      level = p < g.dutyCycle ? base : 0;
    }
  }
  return { level, base, kelvin };
}

/** The group's colour (luminance 1 for a blackbody) at temperature `kelvin`. */
export function groupColour(g: LightGroupSpec, kelvin: number | null): Rgb {
  const c = g.color;
  if (kelvin === null) return [...c] as Rgb;
  const b = blackbody(kelvin);
  return [c[0] * b[0], c[1] * b[1], c[2] * b[2]];
}
