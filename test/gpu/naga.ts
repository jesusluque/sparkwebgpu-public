// Validates WGSL with naga, Firefox's shader compiler, when its CLI is
// installed (`cargo install naga-cli`): Dawn accepts some WGSL naga
// rejects, such as assignments to swizzles.

import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";

function findNaga(): string | null {
  const home = join(homedir(), ".cargo", "bin", "naga");
  if (existsSync(home)) return home;
  const which = spawnSync("which", ["naga"], { encoding: "utf8" });
  return which.status === 0 ? which.stdout.trim() : null;
}

export const naga = findNaga();

let dir: string | null = null;

/** naga's errors for `code`, "" when it validates. */
export function nagaErrors(code: string): string {
  if (!naga) throw new Error("naga is not installed");
  dir ??= mkdtempSync(join(tmpdir(), "spark-naga-"));
  const file = join(dir, "shader.wgsl");
  writeFileSync(file, code);
  const r = spawnSync(naga, [file], { encoding: "utf8" });
  return r.status === 0 ? "" : `${r.stdout}${r.stderr}`;
}
