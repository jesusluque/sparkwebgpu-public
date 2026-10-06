// A reader for the subset of USD's text format (.usda) a light sidecar
// uses (athenea proposals 066/067): layer metadata, prims (def, over,
// class) with their metadata and children, attributes with a value or
// timeSamples, relationships and dictionaries. Values are numbers,
// strings, tokens, booleans, tuples, arrays, paths and asset paths.
//
// It reads; it does not compose. subLayers, references, variants and
// payloads are kept as metadata and not followed: a sidecar is one layer
// whose prims are read as written (an `over` is read like a `def`).
//
// Grammar (the part read):
//   layer     := '#usda 1.0' metadata? prim*
//   metadata  := '(' (entry | string)* ')'
//   entry     := ('prepend' | 'append' | 'delete' | 'add' | 'reorder')? name '=' value
//   prim      := ('def' | 'over' | 'class') typeName? string metadata? '{' item* '}'
//   item      := prim | property | 'variantSet' string '=' '{' ... '}' (skipped)
//   property  := qualifier* 'rel' name ('=' value)? metadata?
//              | qualifier* type ('[]')? name ('.timeSamples' | '.connect')? ('=' value)? metadata?
//              | 'dictionary' name '=' dict
//   dict      := '{' (type ('[]')? (name | string) '=' value | 'dictionary' (name | string) '=' dict)* '}'
//   timeSamples := '{' (number ':' value ','?)* '}'

export type UsdValue =
  | number
  | string
  | boolean
  | null
  | UsdValue[]
  | UsdPath
  | UsdAsset
  | UsdDict;

export interface UsdPath {
  path: string;
}
export interface UsdAsset {
  asset: string;
}
export interface UsdDict {
  [key: string]: UsdValue;
}

export interface UsdProperty {
  name: string;
  /** "rel" for a relationship, else the value type ("float", "color3f[]"...). */
  type: string;
  /** "uniform", "custom"... */
  qualifiers: string[];
  value?: UsdValue;
  /** time -> value, sorted by time. */
  timeSamples?: [number, UsdValue][];
  connect?: UsdValue;
  metadata?: Record<string, UsdValue>;
}

export interface UsdPrim {
  specifier: "def" | "over" | "class";
  typeName: string;
  name: string;
  path: string;
  metadata: Record<string, UsdValue>;
  properties: Map<string, UsdProperty>;
  children: UsdPrim[];
}

export interface UsdLayer {
  version: string;
  metadata: Record<string, UsdValue>;
  /** The layer's root prims. */
  prims: UsdPrim[];
}

export class UsdaError extends Error {
  constructor(message: string, line: number) {
    super(`usda line ${line}: ${message}`);
  }
}

type Token =
  | { kind: "punct"; text: string; line: number }
  | { kind: "string"; text: string; line: number }
  | { kind: "number"; value: number; text: string; line: number }
  | { kind: "ident"; text: string; line: number }
  | { kind: "path"; text: string; line: number }
  | { kind: "asset"; text: string; line: number }
  | { kind: "end"; text: ""; line: number };

const PUNCT = new Set(["(", ")", "[", "]", "{", "}", "=", ",", ":", ";"]);

function tokenize(src: string): Token[] {
  const out: Token[] = [];
  let i = 0;
  let line = 1;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    if (c === "\n") {
      line += 1;
      i += 1;
      continue;
    }
    if (c === " " || c === "\t" || c === "\r") {
      i += 1;
      continue;
    }
    if (c === "#") {
      while (i < n && src[i] !== "\n") i += 1;
      continue;
    }
    if (PUNCT.has(c)) {
      out.push({ kind: "punct", text: c, line });
      i += 1;
      continue;
    }
    if (c === '"' || c === "'") {
      const triple = src.startsWith(c.repeat(3), i);
      const quote = triple ? c.repeat(3) : c;
      const start = line;
      i += quote.length;
      let text = "";
      for (;;) {
        if (i >= n) throw new UsdaError("unterminated string", start);
        if (src.startsWith(quote, i)) {
          i += quote.length;
          break;
        }
        const ch = src[i];
        if (ch === "\n") {
          if (!triple) throw new UsdaError("newline in a string", line);
          line += 1;
        }
        if (ch === "\\" && i + 1 < n) {
          const e = src[i + 1];
          text += e === "n" ? "\n" : e === "t" ? "\t" : e === "r" ? "\r" : e;
          i += 2;
          continue;
        }
        text += ch;
        i += 1;
      }
      out.push({ kind: "string", text, line: start });
      continue;
    }
    if (c === "<") {
      const end = src.indexOf(">", i);
      if (end < 0) throw new UsdaError("unterminated path", line);
      out.push({ kind: "path", text: src.slice(i + 1, end), line });
      i = end + 1;
      continue;
    }
    if (c === "@") {
      const triple = src.startsWith("@@@", i);
      const quote = triple ? "@@@" : "@";
      const end = src.indexOf(quote, i + quote.length);
      if (end < 0) throw new UsdaError("unterminated asset path", line);
      out.push({ kind: "asset", text: src.slice(i + quote.length, end), line });
      i = end + quote.length;
      continue;
    }
    const num = /^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/.exec(
      src.slice(i, i + 64),
    );
    if (num && (c !== "+" || num[0].length > 1)) {
      out.push({
        kind: "number",
        value: Number(num[0]),
        text: num[0],
        line,
      });
      i += num[0].length;
      continue;
    }
    if (c === "-" && src.startsWith("-inf", i)) {
      out.push({
        kind: "number",
        value: Number.NEGATIVE_INFINITY,
        text: "-inf",
        line,
      });
      i += 4;
      continue;
    }
    // Names: namespaced (a:b:c) and with a property suffix (.timeSamples).
    const id = /^[A-Za-z_][\w:.]*/.exec(src.slice(i, i + 512));
    if (id) {
      let text = id[0];
      // A trailing ':' belongs to timeSamples punctuation, not to the name.
      while (text.endsWith(":") || text.endsWith(".")) text = text.slice(0, -1);
      out.push({ kind: "ident", text, line });
      i += text.length;
      continue;
    }
    throw new UsdaError(`unexpected '${c}'`, line);
  }
  out.push({ kind: "end", text: "", line });
  return out;
}

const SPECIFIERS = new Set(["def", "over", "class"]);
const QUALIFIERS = new Set(["uniform", "custom", "varying", "config"]);
const LIST_OPS = new Set(["prepend", "append", "delete", "add", "reorder"]);

class Parser {
  private at = 0;
  constructor(private readonly tokens: Token[]) {}

  private peek(k = 0): Token {
    return this.tokens[Math.min(this.at + k, this.tokens.length - 1)];
  }
  private next(): Token {
    const t = this.peek();
    if (t.kind !== "end") this.at += 1;
    return t;
  }
  private is(text: string, k = 0) {
    const t = this.peek(k);
    return (t.kind === "punct" || t.kind === "ident") && t.text === text;
  }
  private expect(text: string) {
    const t = this.next();
    if ((t.kind !== "punct" && t.kind !== "ident") || t.text !== text) {
      throw new UsdaError(
        `expected '${text}', found '${t.kind === "end" ? "end" : t.text}'`,
        t.line,
      );
    }
  }
  private name(): string {
    const t = this.next();
    if (t.kind === "ident" || t.kind === "string") return t.text;
    throw new UsdaError(`expected a name, found '${t.text}'`, t.line);
  }

  layer(version: string): UsdLayer {
    const metadata = this.is("(") ? this.metadata() : {};
    const prims: UsdPrim[] = [];
    while (this.peek().kind !== "end") prims.push(this.prim(""));
    return { version, metadata, prims };
  }

  metadata(): Record<string, UsdValue> {
    this.expect("(");
    const out: Record<string, UsdValue> = {};
    while (!this.is(")")) {
      const t = this.peek();
      if (t.kind === "end")
        throw new UsdaError("unterminated metadata", t.line);
      if (t.kind === "string") {
        // A bare string is the doc.
        out.doc = this.next().text;
        continue;
      }
      if (this.is(";")) {
        this.next();
        continue;
      }
      let op = "";
      if (t.kind === "ident" && LIST_OPS.has(t.text) && !this.is("=", 1)) {
        op = this.next().text;
      }
      let key = this.name();
      // `dictionary customData = {...}` style typed entries.
      if (this.peek().kind === "ident" && !this.is("=")) key = this.name();
      this.expect("=");
      const v = this.value();
      out[op ? `${op} ${key}` : key] = v;
      if (op && !(key in out)) out[key] = v;
    }
    this.expect(")");
    return out;
  }

  prim(parent: string): UsdPrim {
    const spec = this.next();
    if (spec.kind !== "ident" || !SPECIFIERS.has(spec.text)) {
      throw new UsdaError(`expected a prim, found '${spec.text}'`, spec.line);
    }
    let typeName = "";
    if (this.peek().kind === "ident") typeName = this.next().text;
    const nameTok = this.next();
    if (nameTok.kind !== "string") {
      throw new UsdaError("expected the prim's name in quotes", nameTok.line);
    }
    const path = `${parent}/${nameTok.text}`;
    const metadata = this.is("(") ? this.metadata() : {};
    const prim: UsdPrim = {
      specifier: spec.text as UsdPrim["specifier"],
      typeName,
      name: nameTok.text,
      path,
      metadata,
      properties: new Map(),
      children: [],
    };
    this.expect("{");
    while (!this.is("}")) {
      const t = this.peek();
      if (t.kind === "end") throw new UsdaError("unterminated prim", t.line);
      if (t.kind === "ident" && SPECIFIERS.has(t.text)) {
        prim.children.push(this.prim(path));
      } else if (t.kind === "ident" && t.text === "variantSet") {
        this.skipVariantSet();
      } else if (this.is(";")) {
        this.next();
      } else {
        this.property(prim);
      }
    }
    this.expect("}");
    return prim;
  }

  private skipVariantSet() {
    this.next();
    this.next(); // name
    this.expect("=");
    this.skipBalanced();
  }

  private skipBalanced() {
    const open = this.next();
    const close = open.text === "{" ? "}" : open.text === "(" ? ")" : "]";
    let depth = 1;
    while (depth > 0) {
      const t = this.next();
      if (t.kind === "end") throw new UsdaError("unbalanced block", t.line);
      if (t.kind === "punct" && t.text === open.text) depth += 1;
      if (t.kind === "punct" && t.text === close) depth -= 1;
    }
  }

  private property(prim: UsdPrim) {
    const qualifiers: string[] = [];
    while (this.peek().kind === "ident" && QUALIFIERS.has(this.peek().text)) {
      qualifiers.push(this.next().text);
    }
    const typeTok = this.next();
    if (typeTok.kind !== "ident") {
      throw new UsdaError(
        `expected a property, found '${typeTok.text}'`,
        typeTok.line,
      );
    }
    let type = typeTok.text;
    if (type === "dictionary") {
      const name = this.name();
      this.expect("=");
      prim.properties.set(name, {
        name,
        type,
        qualifiers,
        value: this.dict(),
      });
      return;
    }
    if (this.is("[") && this.is("]", 1)) {
      this.next();
      this.next();
      type += "[]";
    }
    let name = this.name();
    let suffix = "";
    for (const s of [".timeSamples", ".connect", ".spline"]) {
      if (name.endsWith(s)) {
        suffix = s;
        name = name.slice(0, -s.length);
      }
    }
    const property: UsdProperty = prim.properties.get(name) ?? {
      name,
      type: type === "rel" ? "rel" : type,
      qualifiers,
    };
    if (this.is("=")) {
      this.next();
      if (suffix === ".timeSamples") {
        property.timeSamples = this.timeSamples();
      } else if (suffix === ".connect") {
        property.connect = this.value();
      } else if (suffix === ".spline") {
        this.skipBalanced();
      } else {
        property.value = this.value();
      }
    }
    if (this.is("(")) property.metadata = this.metadata();
    prim.properties.set(name, property);
  }

  private timeSamples(): [number, UsdValue][] {
    this.expect("{");
    const out: [number, UsdValue][] = [];
    while (!this.is("}")) {
      const t = this.next();
      if (t.kind !== "number") {
        throw new UsdaError(`expected a time, found '${t.text}'`, t.line);
      }
      this.expect(":");
      out.push([t.value, this.value()]);
      if (this.is(",")) this.next();
    }
    this.expect("}");
    return out.sort((a, b) => a[0] - b[0]);
  }

  private dict(): UsdDict {
    this.expect("{");
    const out: UsdDict = {};
    while (!this.is("}")) {
      if (this.is(";") || this.is(",")) {
        this.next();
        continue;
      }
      const type = this.name();
      if (type === "dictionary") {
        const key = this.name();
        this.expect("=");
        out[key] = this.dict();
        continue;
      }
      if (this.is("[") && this.is("]", 1)) {
        this.next();
        this.next();
      }
      const key = this.name();
      this.expect("=");
      out[key] = this.value();
    }
    this.expect("}");
    return out;
  }

  value(): UsdValue {
    const t = this.peek();
    switch (t.kind) {
      case "number":
        this.next();
        return t.value;
      case "string":
        this.next();
        return t.text;
      case "path":
        this.next();
        return { path: t.text };
      case "asset":
        this.next();
        return { asset: t.text };
      case "ident":
        this.next();
        if (t.text === "true") return true;
        if (t.text === "false") return false;
        if (t.text === "None") return null;
        if (t.text === "inf") return Number.POSITIVE_INFINITY;
        if (t.text === "nan") return Number.NaN;
        return t.text;
      case "punct":
        if (t.text === "[" || t.text === "(") {
          this.next();
          const close = t.text === "[" ? "]" : ")";
          const out: UsdValue[] = [];
          while (!this.is(close)) {
            if (this.peek().kind === "end") {
              throw new UsdaError("unterminated list", t.line);
            }
            out.push(this.value());
            if (this.is(",")) this.next();
          }
          this.next();
          return out;
        }
        if (t.text === "{") return this.dict();
        break;
    }
    throw new UsdaError(`unexpected '${t.text || "end"}'`, t.line);
  }
}

/** Parses a .usda layer (the subset described above). */
export function parseUsda(text: string): UsdLayer {
  const header = /^#usda\s+(\S+)/.exec(text);
  if (!header) throw new UsdaError("not a .usda layer (no '#usda' header)", 1);
  return new Parser(tokenize(text)).layer(header[1]);
}

/** Every prim of the layer, depth first, parents before children. */
export function* walkPrims(
  prims: readonly UsdPrim[],
): Generator<UsdPrim, void, undefined> {
  for (const p of prims) {
    yield p;
    yield* walkPrims(p.children);
  }
}

/** The prim's applied API schemas (apiSchemas, any list op). */
export function apiSchemasOf(prim: UsdPrim): string[] {
  const out: string[] = [];
  for (const [k, v] of Object.entries(prim.metadata)) {
    if (k === "apiSchemas" || k.endsWith(" apiSchemas")) {
      if (Array.isArray(v))
        for (const s of v) if (typeof s === "string") out.push(s);
    }
  }
  return [...new Set(out)];
}
