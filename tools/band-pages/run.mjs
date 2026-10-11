// node tools/band-pages/run.mjs <out.json> [query] [--sweep N | --jump]
// Drives examples/webgpu/band-pages.html (thread CK): athenea's six t42
// pieces as band-paged trees against the same files decoded whole.
//   far    athenea's CameraFar at 393x695 (a phone in portrait), cold start
//   --sweep N (default 6): far to near (/World/Camera) at 1920x1080 in N
//            steps, then back to far
//   --jump:  far, near at 1920x1080 at once, far, near again (with a pool
//            under the near view's needs: evictions and refetches)
// Serve the repo and the .athc with tools/validate-corvette/serve.mjs
// (VALIDATE_PORT, R2_DIR = the folder of <piece>.athc).
import fs from "node:fs";
import { chromium } from "playwright";

const [out, query = "", ...flags] = process.argv.slice(2);
const sweepAt = flags.indexOf("--sweep");
const steps = sweepAt >= 0 ? Number(flags[sweepAt + 1]) : 6;
// athenea's /World/Camera and CameraFar (pieces/t4/cameras.usda), Z-up to
// Y-up: (x, y, z) -> (x, z, -y); horizontal field 39.6 degrees.
const DIR = [0.6080817, 0.1564344, 0.778309];
const NEAR = [4.378188, 1.676328, 5.853825];
const FAR = [13.4994, 4.0228, 17.5284];
const at = (t, w, h) => {
  const p = FAR.map((f, i) => f + (NEAR[i] - f) * t);
  return {
    position: p,
    target: p.map((v, i) => v - DIR[i] * 7.2),
    hfov: 39.6,
    width: w,
    height: h,
  };
};
const browser = await chromium.launch({
  channel: "chrome",
  headless: true,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-dawn-features=allow_unsafe_apis",
    "--disable-gpu-vsync",
  ],
});
const page = await browser.newPage({
  viewport: { width: 1920, height: 1080 },
  deviceScaleFactor: 1,
});
const log = [];
page.on("console", (m) => {
  if (m.type() === "error" || m.type() === "warning")
    log.push(`${m.type()}: ${m.text()}`);
});
page.on("pageerror", (e) => {
  console.log(`pageerror: ${e.stack ?? e.message}`, log.join("\n"));
  process.exit(1);
});
const port = process.env.VALIDATE_PORT ?? 8147;
const t0 = Date.now();
console.log("loading");
setTimeout(
  async () => {
    console.log(
      "TIMEOUT",
      await page
        .evaluate(() => document.getElementById("status")?.textContent)
        .catch(() => "?"),
      log.join("\n"),
    );
    process.exit(2);
  },
  Number(process.env.BAND_TIMEOUT_S ?? 900) * 1000,
).unref();
await page.goto(
  `http://127.0.0.1:${port}/examples/webgpu/band-pages.html?${query}`,
);
await page.waitForFunction(
  () => window.__band?.ready || window.__band?.error,
  null,
  { timeout: 600000 },
);
const err = await page.evaluate(() => window.__band.error);
if (err) {
  console.log("ERROR", err, log.join("\n"));
  await browser.close();
  process.exit(1);
}
console.log(`loaded in ${(Date.now() - t0) / 1000} s`);
const first = await page.evaluate(() => window.__band.stats());
console.log(`pool ${first.poolPages} pages`);
void `${first.poolPages} pages, ${first.bytesPerNode} B a node: ${first.attributes.join(" ")}`;
const result = { query, loadS: (Date.now() - t0) / 1000, runs: [], log };
async function step(name, v, compare = true) {
  const r = await page.evaluate(
    async ({ v, compare }) => {
      const B = window.__band;
      B.view(v);
      const settle = await B.settle();
      const stats = B.stats();
      const cmp = compare ? await B.compare() : null;
      return { settle, stats, cmp };
    },
    { v, compare },
  );
  r.name = name;
  result.runs.push(r);
  const s = r.stats;
  const c = r.cmp;
  const diff = c
    ? c.cuts.reduce((n, x) => n + x.onlyPaged + x.onlyFull, 0)
    : "-";
  console.log(
    `${name}: tau ${s.tau.toFixed(3)} ${s.bytesPerNode} B/node, cut ${s.cut} (ref ${s.refCut}) needed ${s.neededPages}p/${s.neededNodes} resident ${s.residentPages}p/${s.residentNodes} (${(s.residentNodes / Math.max(1, s.cut)).toFixed(2)}x cut, ${(s.residentBytes / 1e6).toFixed(1)} MB) ` +
      `settle ${r.settle.ms.toFixed(0)} ms ${r.settle.frames} frames, +${r.settle.pagesRequested} pages in ${r.settle.ranges} ranges ${(r.settle.bytesRead / 1e6).toFixed(2)} MB, evicted ${r.settle.evicted}; ` +
      `cut diff ${diff}${c?.image ? ` relMSE ${c.image.relMSE.toExponential(2)} max ${c.image.maxDiff.toFixed(4)} ratio ${c.image.ratio.toFixed(4)}` : ""}`,
  );
}
const mode = flags.includes("--jump") ? "jump" : "sweep";
await step("far 393x695 (cold)", at(0, 393, 695));
if (mode === "jump") {
  await step("near 1920x1080 (jump)", at(1, 1920, 1080));
  await step("far 393x695 (jump back)", at(0, 393, 695));
  await step("near 1920x1080 (jump again)", at(1, 1920, 1080));
  // Turned: the camera 1.5 m to its left, looking at the grille end.
  const side = at(1, 1920, 1080);
  side.position = side.position.map(
    (v, i) => v + [-0.7880107, 0, 0.6156615][i] * 1.5,
  );
  side.target = side.target.map(
    (v, i) => v + [-0.7880107, 0, 0.6156615][i] * 2.5,
  );
  await step("side 1920x1080 (jump)", side);
  await step("near 1920x1080 (jump back)", at(1, 1920, 1080));
} else {
  for (let k = 1; k <= steps; k++) {
    await step(
      `sweep ${k}/${steps} 1920x1080`,
      at(k / steps, 1920, 1080),
      k === steps || k === 1,
    );
  }
  await step("far 393x695 (back)", at(0, 393, 695));
}
fs.writeFileSync(out, JSON.stringify(result, null, 2));
if (log.length) console.log(log.slice(0, 20).join("\n"));
await browser.close();
