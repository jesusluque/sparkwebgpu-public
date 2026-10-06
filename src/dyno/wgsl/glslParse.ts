// A parser for the GLSL ES 3.0 that dyno snippets are written in: function,
// struct and constant definitions (globals) and statement lists. It builds
// the AST that glslToWgsl.ts translates. No preprocessor beyond object-like
// #defines, which dyno code doesn't otherwise need.

export interface Pos {
  line: number;
  col: number;
}

/** A GLSL construct the translator can't handle, with where it is. */
export class GlslError extends Error {
  constructor(
    message: string,
    readonly pos: Pos,
    source?: string,
  ) {
    const line = source?.split("\n")[pos.line - 1];
    super(
      `GLSL ${pos.line}:${pos.col}: ${message}${
        line != null ? `\n    ${line.trim()}` : ""
      }`,
    );
  }
}

type TokenKind = "id" | "num" | "op" | "eof";

interface Token {
  kind: TokenKind;
  text: string;
  pos: Pos;
}

const OPS = [
  "<<=",
  ">>=",
  "++",
  "--",
  "+=",
  "-=",
  "*=",
  "/=",
  "%=",
  "&=",
  "|=",
  "^=",
  "<<",
  ">>",
  "<=",
  ">=",
  "==",
  "!=",
  "&&",
  "||",
  "^^",
];

function tokenize(src: string): Token[] {
  const tokens: Token[] = [];
  const defines = new Map<string, Token[]>();
  let i = 0;
  let line = 1;
  let lineStart = 0;
  const pos = (): Pos => ({ line, col: i - lineStart + 1 });
  const newline = () => {
    line++;
    lineStart = i + 1;
  };
  // Tokens of the current line go here while reading a #define.
  let defineTarget: Token[] | null = null;
  const push = (t: Token) => {
    if (defineTarget) {
      defineTarget.push(t);
    } else if (t.kind === "id" && defines.has(t.text)) {
      for (const d of defines.get(t.text) as Token[]) {
        tokens.push({ ...d, pos: t.pos });
      }
    } else {
      tokens.push(t);
    }
  };
  while (i < src.length) {
    const c = src[i];
    if (c === "\n") {
      newline();
      i++;
      defineTarget = null;
      continue;
    }
    if (c === " " || c === "\t" || c === "\r") {
      i++;
      continue;
    }
    if (c === "\\" && src[i + 1] === "\n") {
      i++;
      newline();
      i++;
      continue;
    }
    if (c === "/" && src[i + 1] === "/") {
      while (i < src.length && src[i] !== "\n") i++;
      continue;
    }
    if (c === "/" && src[i + 1] === "*") {
      const start = pos();
      const end = src.indexOf("*/", i + 2);
      if (end < 0) throw new GlslError("unterminated comment", start, src);
      for (; i < end + 2; i++) if (src[i] === "\n") newline();
      continue;
    }
    if (c === "#") {
      const start = pos();
      const m = /^#[ \t]*(\w+)[ \t]*/.exec(src.slice(i));
      const directive = m?.[1];
      if (directive === "define") {
        i += (m as RegExpExecArray)[0].length;
        const name = /^\w+/.exec(src.slice(i))?.[0];
        if (!name || src[i + name.length] === "(") {
          throw new GlslError(
            "only object-like #defines are supported",
            start,
            src,
          );
        }
        i += name.length;
        defineTarget = [];
        defines.set(name, defineTarget);
        continue;
      }
      throw new GlslError(
        `preprocessor directive #${directive ?? ""} is not supported`,
        start,
        src,
      );
    }
    const p = pos();
    const num =
      /^(?:0[xX][0-9a-fA-F]+[uU]?|(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?[fF]?|\d+[eE][+-]?\d+[fF]?|\d+[uU]?)/.exec(
        src.slice(i),
      );
    if (num && /[\d.]/.test(c)) {
      push({ kind: "num", text: num[0], pos: p });
      i += num[0].length;
      continue;
    }
    const id = /^[A-Za-z_]\w*/.exec(src.slice(i));
    if (id) {
      push({ kind: "id", text: id[0], pos: p });
      i += id[0].length;
      continue;
    }
    const op =
      OPS.find((o) => src.startsWith(o, i)) ??
      ("(){}[];,.+-*/%<>=!~&|^?:".includes(c) ? c : null);
    if (!op) throw new GlslError(`unexpected character '${c}'`, p, src);
    push({ kind: "op", text: op, pos: p });
    i += op.length;
  }
  tokens.push({ kind: "eof", text: "", pos: pos() });
  return tokens;
}

// AST

export type Expr =
  | { k: "num"; text: string; pos: Pos }
  | { k: "bool"; value: boolean; pos: Pos }
  | { k: "id"; name: string; pos: Pos }
  | { k: "call"; callee: TypeSpec; args: Expr[]; pos: Pos }
  | { k: "method"; obj: Expr; name: string; args: Expr[]; pos: Pos }
  | { k: "member"; obj: Expr; name: string; pos: Pos }
  | { k: "index"; obj: Expr; index: Expr; pos: Pos }
  | { k: "unary"; op: string; arg: Expr; pos: Pos }
  | { k: "postfix"; op: string; arg: Expr; pos: Pos }
  | { k: "binary"; op: string; left: Expr; right: Expr; pos: Pos }
  | { k: "assign"; op: string; target: Expr; value: Expr; pos: Pos }
  | { k: "cond"; test: Expr; then: Expr; else: Expr; pos: Pos }
  | { k: "comma"; exprs: Expr[]; pos: Pos };

/** A type name, with an array size for `float[3]` (null: unsized). */
export interface TypeSpec {
  name: string;
  array?: Expr | null;
  pos: Pos;
}

export interface Declarator {
  name: string;
  array?: Expr | null;
  init?: Expr;
  pos: Pos;
}

export type Stmt =
  | {
      k: "decl";
      isConst: boolean;
      type: TypeSpec;
      vars: Declarator[];
      pos: Pos;
    }
  | { k: "expr"; expr: Expr; pos: Pos }
  | { k: "block"; body: Stmt[]; pos: Pos }
  | { k: "if"; test: Expr; then: Stmt; else?: Stmt; pos: Pos }
  | {
      k: "for";
      init?: Stmt;
      test?: Expr;
      update?: Expr;
      body: Stmt;
      pos: Pos;
    }
  | { k: "while"; test: Expr; body: Stmt; pos: Pos }
  | { k: "do"; test: Expr; body: Stmt; pos: Pos }
  | { k: "return"; value?: Expr; pos: Pos }
  | { k: "break"; pos: Pos }
  | { k: "continue"; pos: Pos }
  | { k: "discard"; pos: Pos }
  | {
      k: "switch";
      disc: Expr;
      cases: { tests: Expr[] | null; body: Stmt[]; pos: Pos }[];
      pos: Pos;
    }
  | { k: "empty"; pos: Pos };

export interface Param {
  name: string;
  type: TypeSpec;
  qual: "in" | "out" | "inout";
  pos: Pos;
}

export type Global =
  | {
      k: "func";
      ret: TypeSpec;
      name: string;
      params: Param[];
      body?: Stmt[];
      pos: Pos;
    }
  | {
      k: "struct";
      name: string;
      fields: { type: TypeSpec; name: string; array?: Expr | null; pos: Pos }[];
      pos: Pos;
    }
  | { k: "decl"; decl: Extract<Stmt, { k: "decl" }> };

const PRECISION = new Set(["highp", "mediump", "lowp"]);
const ASSIGN_OPS = new Set([
  "=",
  "+=",
  "-=",
  "*=",
  "/=",
  "%=",
  "<<=",
  ">>=",
  "&=",
  "|=",
  "^=",
]);
// Binary operator precedence, higher binds tighter.
const BINARY: Record<string, number> = {
  "||": 1,
  "^^": 2,
  "&&": 3,
  "|": 4,
  "^": 5,
  "&": 6,
  "==": 7,
  "!=": 7,
  "<": 8,
  ">": 8,
  "<=": 8,
  ">=": 8,
  "<<": 9,
  ">>": 9,
  "+": 10,
  "-": 10,
  "*": 11,
  "/": 11,
  "%": 11,
};
const STATEMENT_KEYWORDS = new Set([
  "if",
  "for",
  "while",
  "do",
  "return",
  "break",
  "continue",
  "discard",
  "switch",
  "case",
  "default",
  "else",
]);
const UNSUPPORTED_QUALIFIERS = new Set([
  "uniform",
  "attribute",
  "varying",
  "layout",
  "centroid",
  "flat",
  "smooth",
  "invariant",
  "buffer",
  "shared",
]);

class Parser {
  private t = 0;
  constructor(
    private tokens: Token[],
    readonly src: string,
  ) {}

  private get tok() {
    return this.tokens[this.t];
  }
  private peek(n = 1) {
    return this.tokens[Math.min(this.t + n, this.tokens.length - 1)];
  }
  private is(text: string) {
    return this.tok.kind !== "num" && this.tok.text === text;
  }
  private next() {
    return this.tokens[this.t++];
  }
  error(message: string, pos: Pos = this.tok.pos): never {
    throw new GlslError(message, pos, this.src);
  }
  private expect(text: string) {
    if (!this.is(text)) {
      this.error(
        `expected '${text}' but found ${this.tok.kind === "eof" ? "end of input" : `'${this.tok.text}'`}`,
      );
    }
    return this.next();
  }
  private ident(): Token {
    if (this.tok.kind !== "id") {
      this.error(`expected a name but found '${this.tok.text}'`);
    }
    return this.next();
  }

  atEnd() {
    return this.tok.kind === "eof";
  }

  // Declarations start with a qualifier, `Type name` or `Type[N] name`.
  private isDeclStart(): boolean {
    const tok = this.tok;
    if (tok.kind !== "id") return false;
    if (tok.text === "const" || PRECISION.has(tok.text)) return true;
    if (UNSUPPORTED_QUALIFIERS.has(tok.text)) {
      this.error(`'${tok.text}' declarations are not supported in dyno code`);
    }
    if (STATEMENT_KEYWORDS.has(tok.text) || tok.text === "struct") return false;
    if (this.peek().kind === "id") return true;
    if (this.peek().text === "[") {
      // Type[N] name: find the matching ] and check for a name after it.
      let depth = 0;
      for (let j = this.t + 1; j < this.tokens.length; j++) {
        const x = this.tokens[j];
        if (x.text === "[") depth++;
        else if (x.text === "]" && --depth === 0) {
          return this.tokens[j + 1]?.kind === "id";
        } else if (x.kind === "eof" || x.text === ";") return false;
      }
    }
    return false;
  }

  private typeSpec(): TypeSpec {
    while (PRECISION.has(this.tok.text)) this.next();
    const name = this.ident();
    const spec: TypeSpec = { name: name.text, pos: name.pos };
    if (this.is("[")) {
      this.next();
      spec.array = this.is("]") ? null : this.expr();
      this.expect("]");
    }
    return spec;
  }

  // Qualifiers, a type, then one or more `name [N] = init`.
  private declaration(): Extract<Stmt, { k: "decl" }> {
    const pos = this.tok.pos;
    let isConst = false;
    while (this.is("const") || PRECISION.has(this.tok.text)) {
      if (this.next().text === "const") isConst = true;
    }
    const type = this.typeSpec();
    const vars: Declarator[] = [];
    do {
      if (vars.length) this.expect(",");
      const name = this.ident();
      const v: Declarator = { name: name.text, pos: name.pos };
      if (this.is("[")) {
        this.next();
        v.array = this.is("]") ? null : this.expr();
        this.expect("]");
      }
      if (this.is("=")) {
        this.next();
        v.init = this.assignment();
      }
      vars.push(v);
    } while (this.is(","));
    this.expect(";");
    return { k: "decl", isConst, type, vars, pos };
  }

  // Expressions (Pratt parser)

  expr(): Expr {
    const first = this.assignment();
    if (!this.is(",")) return first;
    const exprs = [first];
    while (this.is(",")) {
      this.next();
      exprs.push(this.assignment());
    }
    return { k: "comma", exprs, pos: first.pos };
  }

  private assignment(): Expr {
    const target = this.conditional();
    if (this.tok.kind === "op" && ASSIGN_OPS.has(this.tok.text)) {
      const op = this.next();
      const value = this.assignment();
      return { k: "assign", op: op.text, target, value, pos: op.pos };
    }
    return target;
  }

  private conditional(): Expr {
    const test = this.binary(1);
    if (!this.is("?")) return test;
    const pos = this.next().pos;
    const then = this.expr();
    this.expect(":");
    const otherwise = this.assignment();
    return { k: "cond", test, then, else: otherwise, pos };
  }

  private binary(minPrec: number): Expr {
    let left = this.unary();
    for (;;) {
      const op = this.tok;
      const prec = op.kind === "op" ? BINARY[op.text] : undefined;
      if (prec === undefined || prec < minPrec) return left;
      this.next();
      const right = this.binary(prec + 1);
      left = { k: "binary", op: op.text, left, right, pos: op.pos };
    }
  }

  private unary(): Expr {
    const tok = this.tok;
    if (
      tok.kind === "op" &&
      ["-", "+", "!", "~", "++", "--"].includes(tok.text)
    ) {
      this.next();
      return { k: "unary", op: tok.text, arg: this.unary(), pos: tok.pos };
    }
    return this.postfix(this.primary());
  }

  private args(): Expr[] {
    this.expect("(");
    const args: Expr[] = [];
    if (this.is("void") && this.peek().text === ")") this.next();
    while (!this.is(")")) {
      if (args.length) this.expect(",");
      args.push(this.assignment());
    }
    this.expect(")");
    return args;
  }

  private primary(): Expr {
    const tok = this.tok;
    if (tok.kind === "num") {
      this.next();
      return { k: "num", text: tok.text, pos: tok.pos };
    }
    if (tok.kind === "id") {
      if (tok.text === "true" || tok.text === "false") {
        this.next();
        return { k: "bool", value: tok.text === "true", pos: tok.pos };
      }
      // float[3](...) array constructors
      if (this.peek().text === "[" && this.peek(2).text === "]") {
        const callee = this.typeSpec();
        return { k: "call", callee, args: this.args(), pos: tok.pos };
      }
      if (this.peek().text === "[" && this.isArrayConstructor()) {
        const callee = this.typeSpec();
        return { k: "call", callee, args: this.args(), pos: tok.pos };
      }
      this.next();
      if (this.is("(")) {
        return {
          k: "call",
          callee: { name: tok.text, pos: tok.pos },
          args: this.args(),
          pos: tok.pos,
        };
      }
      return { k: "id", name: tok.text, pos: tok.pos };
    }
    if (this.is("(")) {
      this.next();
      const e = this.expr();
      this.expect(")");
      return e;
    }
    this.error(
      tok.kind === "eof"
        ? "unexpected end of input"
        : `unexpected '${tok.text}'`,
    );
  }

  // Type[N](...): an index expression is never followed by '('.
  private isArrayConstructor(): boolean {
    let depth = 0;
    for (let j = this.t + 1; j < this.tokens.length; j++) {
      const x = this.tokens[j];
      if (x.text === "[") depth++;
      else if (x.text === "]" && --depth === 0) {
        return this.tokens[j + 1]?.text === "(";
      } else if (x.kind === "eof" || x.text === ";") return false;
    }
    return false;
  }

  private postfix(primary: Expr): Expr {
    let e = primary;
    for (;;) {
      const tok = this.tok;
      if (this.is("[")) {
        this.next();
        const index = this.expr();
        this.expect("]");
        e = { k: "index", obj: e, index, pos: tok.pos };
      } else if (this.is(".")) {
        this.next();
        const name = this.ident();
        if (this.is("(")) {
          e = {
            k: "method",
            obj: e,
            name: name.text,
            args: this.args(),
            pos: name.pos,
          };
        } else {
          e = { k: "member", obj: e, name: name.text, pos: name.pos };
        }
      } else if (this.is("++") || this.is("--")) {
        this.next();
        e = { k: "postfix", op: tok.text, arg: e, pos: tok.pos };
      } else {
        return e;
      }
    }
  }

  // Statements

  statement(): Stmt {
    const tok = this.tok;
    const pos = tok.pos;
    if (this.is("{")) return this.block();
    if (this.is(";")) {
      this.next();
      return { k: "empty", pos };
    }
    if (tok.kind === "id") {
      switch (tok.text) {
        case "if": {
          this.next();
          this.expect("(");
          const test = this.expr();
          this.expect(")");
          const then = this.statement();
          if (this.is("else")) {
            this.next();
            return { k: "if", test, then, else: this.statement(), pos };
          }
          return { k: "if", test, then, pos };
        }
        case "for": {
          this.next();
          this.expect("(");
          let init: Stmt | undefined;
          if (this.isDeclStart()) init = this.declaration();
          else if (this.is(";")) this.next();
          else {
            init = { k: "expr", expr: this.expr(), pos: this.tok.pos };
            this.expect(";");
          }
          const test = this.is(";") ? undefined : this.expr();
          this.expect(";");
          const update = this.is(")") ? undefined : this.expr();
          this.expect(")");
          return { k: "for", init, test, update, body: this.statement(), pos };
        }
        case "while": {
          this.next();
          this.expect("(");
          const test = this.expr();
          this.expect(")");
          return { k: "while", test, body: this.statement(), pos };
        }
        case "do": {
          this.next();
          const body = this.statement();
          this.expect("while");
          this.expect("(");
          const test = this.expr();
          this.expect(")");
          this.expect(";");
          return { k: "do", test, body, pos };
        }
        case "return": {
          this.next();
          const value = this.is(";") ? undefined : this.expr();
          this.expect(";");
          return { k: "return", value, pos };
        }
        case "break":
        case "continue":
        case "discard":
          this.next();
          this.expect(";");
          return { k: tok.text, pos };
        case "switch":
          return this.switchStatement();
        case "precision":
          while (!this.is(";")) this.next();
          this.next();
          return { k: "empty", pos };
      }
      if (this.isDeclStart()) return this.declaration();
    }
    const expr = this.expr();
    this.expect(";");
    return { k: "expr", expr, pos };
  }

  private block(): Extract<Stmt, { k: "block" }> {
    const pos = this.expect("{").pos;
    const body: Stmt[] = [];
    while (!this.is("}")) {
      if (this.atEnd()) this.error("missing '}'", pos);
      body.push(this.statement());
    }
    this.next();
    return { k: "block", body, pos };
  }

  private switchStatement(): Stmt {
    const pos = this.next().pos;
    this.expect("(");
    const disc = this.expr();
    this.expect(")");
    this.expect("{");
    const cases: Extract<Stmt, { k: "switch" }>["cases"] = [];
    while (!this.is("}")) {
      const cpos = this.tok.pos;
      let tests: Expr[] | null;
      if (this.is("default")) {
        this.next();
        tests = null;
      } else {
        this.expect("case");
        tests = [this.conditional()];
      }
      this.expect(":");
      const body: Stmt[] = [];
      while (!this.is("case") && !this.is("default") && !this.is("}")) {
        if (this.atEnd()) this.error("missing '}'", pos);
        body.push(this.statement());
      }
      cases.push({ tests, body, pos: cpos });
    }
    this.next();
    return { k: "switch", disc, cases, pos };
  }

  // Globals: functions, structs, constants and variables.

  global(): Global | null {
    const pos = this.tok.pos;
    if (this.is(";")) {
      this.next();
      return null;
    }
    if (this.is("precision")) {
      while (!this.is(";")) this.next();
      this.next();
      return null;
    }
    if (this.is("struct")) {
      this.next();
      const name = this.ident().text;
      this.expect("{");
      const fields: Extract<Global, { k: "struct" }>["fields"] = [];
      while (!this.is("}")) {
        const type = this.typeSpec();
        do {
          if (this.is(",")) this.next();
          const field = this.ident();
          let array: Expr | null | undefined;
          if (this.is("[")) {
            this.next();
            array = this.expr();
            this.expect("]");
          }
          fields.push({ type, name: field.text, array, pos: field.pos });
        } while (this.is(","));
        this.expect(";");
      }
      this.next();
      if (!this.is(";")) this.error("declarators after a struct definition");
      this.next();
      return { k: "struct", name, fields, pos };
    }
    if (!this.isDeclStart()) {
      this.error(`expected a declaration but found '${this.tok.text}'`);
    }
    // A function: Type name ( ...
    const save = this.t;
    const isConst = this.is("const");
    if (!isConst) {
      const ret = this.typeSpec();
      if (this.tok.kind === "id" && this.peek().text === "(") {
        const name = this.next().text;
        return { k: "func", ret, name, ...this.functionRest(), pos };
      }
    }
    this.t = save;
    return { k: "decl", decl: this.declaration() };
  }

  private functionRest(): { params: Param[]; body?: Stmt[] } {
    this.expect("(");
    const params: Param[] = [];
    if (this.is("void") && this.peek().text === ")") this.next();
    while (!this.is(")")) {
      if (params.length) this.expect(",");
      let qual: Param["qual"] = "in";
      while (
        ["const", "in", "out", "inout"].includes(this.tok.text) ||
        PRECISION.has(this.tok.text)
      ) {
        const q = this.next().text;
        if (q === "out" || q === "inout") qual = q;
      }
      const type = this.typeSpec();
      const name = this.ident();
      if (this.is("[")) {
        this.next();
        type.array = this.expr();
        this.expect("]");
      }
      params.push({ name: name.text, type, qual, pos: name.pos });
    }
    this.next();
    if (this.is(";")) {
      this.next();
      return { params };
    }
    const body = this.block().body;
    return { params, body };
  }
}

/** Global definitions: functions, structs, constants. */
export function parseGlobals(src: string): Global[] {
  const p = new Parser(tokenize(src), src);
  const globals: Global[] = [];
  while (!p.atEnd()) {
    const g = p.global();
    if (g) globals.push(g);
  }
  return globals;
}

/** A statement list, as in a function body. */
export function parseStatements(src: string): Stmt[] {
  const p = new Parser(tokenize(src), src);
  const body: Stmt[] = [];
  while (!p.atEnd()) body.push(p.statement());
  return body;
}
