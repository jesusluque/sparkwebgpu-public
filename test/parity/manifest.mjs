// The parity cases: each WebGPU port in examples/webgpu/ against its WebGL
// original, plus the paired test pages (compare-*, dyno-*, envmap-*,
// spark-renderer ?backend=webgl).
//
// A case (one scenario of an example) takes:
//   readme     the mean diff examples/webgpu/README.md reports (/255)
//   max        { mean, pct }: thresholds. `mean` defaults to readme * 1.5 +
//              0.15; `pct` (% of pixels off by more than 16) is set from
//              measured runs where there's no README number
//   query      appended to both URLs; or webgl / webgpu: full paths
//   t0, seed   the frozen clock's time (ms) and Math.random's seed
//   freeze     false: the real clock (pages that wait on performance.now)
//   manual     no frames until the input script steps them (animations that
//              advance per frame rather than with time)
//   settle     { ready, minWait, maxWait, stable, epsilon } (see run.mjs)
//   act        async (ctx, side) => input script (see drivers.mjs makeCtx;
//              ctx.settle(opts) waits as the runner does before `act`)
//   routes     request substitutions: { path: RegExp, redirect(path) | rewrite(text) }
//              (rewrite takes a JS module's served code, or a page's HTML)
//   needs      ["files"]: a file input, Chrome only
//   hide       selectors hidden before screenshots (counters, timings);
//              lil-gui and stats.js panels are hidden unless showPanels
//   minLit     % of pixels that must be lit on both sides (default 1)
//   known      a failure that's understood and reported: shown as "known"
//              rather than failing the run (say why)
//   allowErrors  don't fail on page errors only the port throws
//   note       why the case is set up the way it is

const margin = (readme) => Math.round((readme * 1.5 + 0.15) * 100) / 100;

function ex(name, scenarios = [{}], common = {}) {
  return scenarios.map((s) => {
    const c = {
      name,
      scenario: "default",
      viewport: { width: 800, height: 600 },
      ...common,
      ...s,
    };
    if (s.id) c.scenario = s.id;
    c.id = c.scenario === "default" ? name : `${name}:${c.scenario}`;
    const q = c.query ?? "";
    c.webgl = (c.webgl ?? `/examples/${name}/index.html`) + q;
    c.webgpu = (c.webgpu ?? `/examples/webgpu/${name}.html`) + q;
    if (c.readme != null) c.max = { mean: margin(c.readme), ...c.max };
    return c;
  });
}

const BUTTERFLY = "https://sparkjs.dev/assets/splats/butterfly.spz";

// Input-script helpers.
const step =
  (n = 3, dt = 1000 / 60) =>
  (ctx) =>
    ctx.step(n, dt);
async function drag(ctx, from, to, frames) {
  await ctx.step(1);
  await ctx.mouse.move(...from);
  await ctx.mouse.down();
  await ctx.step(1);
  await ctx.mouse.move(...to, 4);
  await ctx.step(frames);
}
async function clicks(ctx, points, framesBetween, framesAfter) {
  await ctx.step(1);
  for (const p of points) {
    await ctx.mouse.click(...p);
    await ctx.step(framesBetween);
  }
  await ctx.step(framesAfter);
}

// lofi shuffles its worlds with Math.random at startup, after three and
// Spark have drawn different numbers of randoms on the two backends: reseed
// at the shuffle so both pick the same worlds.
const lofiRoute = {
  path: /^\/examples\/lofi\/util\.js$/,
  rewrite: (code) =>
    code.replace(
      "export function shuffleArray(array) {",
      "let reseeded = false;\nexport function shuffleArray(array) {\n  if (!reseeded) { reseeded = true; window.__reseed?.(4321); }",
    ),
};
const lofiKey = (k) => async (ctx) => {
  await ctx.key.press(k);
  // The transition starts (and logs so) once the next world has loaded,
  // which a still frame doesn't show.
  await ctx.waitForLog("Starting transition");
  // It takes 1500 ms: step through to its end, then frames at that time
  // for the sort to catch up.
  await ctx.step(10, 200);
  await ctx.step(8, 0);
};

// portal and newportal load assets that aren't in the repo or assets.json.
const HOB =
  "https://storage.googleapis.com/forge-dev-public/asundqui/hobbitverse";
const portalRoute = {
  path: /^\/examples\/portal\/[^/]+\.spz$/,
  redirect: (p) => {
    const file = p.split("/").pop();
    return file.startsWith("cozy")
      ? "https://sparkjs.dev/assets/splats/valley.spz"
      : `${HOB}/${file}`;
  },
};
const NEWPORTAL_RAD =
  "https://storage.googleapis.com/forge-dev-public/asundqui/rad/260217/cozy-spaceship_2-lod.rad";
const newportalRoute = {
  path: /^\/examples\/(webgpu\/newportal\.html|newportal\/index\.html)$/,
  rewrite: (html) =>
    html.replace(
      "`${URL_BASE}/cozy_cottage-lod-0.spz`",
      JSON.stringify(NEWPORTAL_RAD),
    ),
};

// splat-transitions: wait for its loading overlay, pick an effect, then set
// the time by steps; settle frames (dt 0) let the transition's async work
// land.
const T_LOADED = () =>
  document.getElementById("loading")?.style.display === "none";
const transition =
  (effect, ...steps) =>
  async (ctx) => {
    if (effect) {
      await ctx.select(".lil-gui select", effect);
      await ctx.wait(200);
      await ctx.waitFor(T_LOADED, null, 120000);
      await ctx.wait(3000);
    }
    for (const [n, dt] of steps) await ctx.step(n, dt);
    for (let i = 0; i < 3; i++) {
      await ctx.step(6, 0);
      await ctx.wait(1500);
    }
  };

// splat-reveal-effects: time and the camera orbit advance per frame, so it
// runs manual and only steps after loading.
const reveal = (effect, frames) => async (ctx) => {
  if (effect !== "Magic") {
    await ctx.select(".lil-gui select", effect);
    await ctx.wait(15000);
  }
  await ctx.step(frames);
};

const dissolve = (secs) => async (ctx) => {
  // The first frame sets the start time; then 4 steps cover `secs`.
  await ctx.step(1);
  await ctx.step(4, (secs * 1000) / 4);
};

async function stroke(ctx, x0, y0, x1, y1) {
  await ctx.mouse.move(x0, y0);
  await ctx.wait(200);
  await ctx.mouse.down();
  for (let i = 1; i <= 8; i++) {
    await ctx.mouse.move(x0 + ((x1 - x0) * i) / 8, y0 + ((y1 - y0) * i) / 8);
    await ctx.wait(100);
  }
  await ctx.mouse.up();
  await ctx.wait(300);
}

// lil-gui controller by its label.
const guiSet = (name, value) => (ctx) =>
  ctx.eval(
    ([name, value]) => {
      const c = [...document.querySelectorAll(".lil-gui .controller")].find(
        (c) => c.querySelector(".name")?.textContent.trim() === name,
      );
      if (!c) throw new Error(`no lil-gui controller ${name}`);
      const input = [...c.querySelectorAll("input")].pop();
      if (input.type === "checkbox") {
        if (input.checked !== value) input.click();
        return;
      }
      input.value = String(value);
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
      input.dispatchEvent(new Event("blur"));
    },
    [name, value],
  );
const guiOpen = (title) => (ctx) =>
  ctx.eval((title) => {
    const t = [...document.querySelectorAll(".lil-gui .title")].find(
      (t) => t.textContent.trim() === title,
    );
    if (!t) throw new Error(`no lil-gui folder ${title}`);
    if (t.parentElement.classList.contains("closed")) t.click();
  }, title);

const DYNO = ["effect", "depth", "normal", "edit", "snow", "skin", "rgba"];
const READY = () => window.__ready;
const PAGED = { stableShots: 6, maxWait: 120000 };

export const cases = [
  ...ex("hello-world", [{ readme: 0.03 }]),
  ...ex("multiple-splats", [{ readme: 0.12 }]),
  ...ex("raycasting", [
    {
      id: "clicked",
      readme: 0.08,
      note: "a click tints a robot for 40 frames: click, then 3 frames",
      act: (ctx) => clicks(ctx, [[160, 380]], 3, 0),
    },
  ]),
  ...ex("interactivity", [{ readme: 0.16 }]),
  ...ex("extsplats", [{ readme: 0.13 }]),
  ...ex("nonlod", [{ readme: 0.16 }]),
  ...ex("lod-example", [{ readme: 0.25, webgl: "/examples/lod/index.html" }]),
  // Paged scenes decode chunks in workers with nothing new on screen for a
  // while: they need a longer calm.
  ...ex("multi-lod", [{ readme: 0.58, settle: PAGED }]),
  ...ex("streaming-lod", [{ readme: 0.67, settle: PAGED }]),
  ...ex("depth-of-field", [{ readme: 0.97 }]),
  ...ex("on-demand", [{ readme: 0.7, hide: ["#stats"], settle: PAGED }]),
  // README said 0.46 before this suite existed; the suite measures 0.72 on
  // every build back to 3c6f02a, so the 0.46 was measured another way.
  ...ex("splat-shader-effects", [{ readme: 0.72, max: { mean: 0.84 } }]),
  ...ex("procedural-splats", [
    { note: "README: random stars differ (drawn in different orders)" },
  ]),
  ...ex("sogs", [{ readme: 0.97 }]),
  ...ex("debug-color", [{ readme: 0.13 }]),
  ...ex("lod-on-demand", [{ readme: 0.23 }]),
  ...ex("particle-animation", [
    { readme: 3.09, note: "time-driven noise moves fast particles" },
  ]),
  ...ex("glsl", [{ readme: 0.16, t0: 2500, act: step(3) }]),
  ...ex("dynamic-lighting", [
    { readme: 0.22, t0: 7000, act: step(3) },
    {
      id: "debug",
      readme: 0.23,
      t0: 7000,
      act: async (ctx) => {
        await ctx.click("#debug");
        await ctx.step(3);
      },
    },
  ]),
  ...ex("mobile-joystick", [
    { readme: 0.36, t0: 3000, act: step(3) },
    {
      id: "walk",
      readme: 0.38,
      t0: 3000,
      act: async (ctx) => {
        await ctx.mouse.move(400, 300);
        await ctx.mouse.down();
        await ctx.mouse.move(300, 280);
        await ctx.step(1);
        await ctx.mouse.up();
        await ctx.key.down("w");
        await ctx.step(30);
        await ctx.key.up("w");
        await ctx.step(3);
      },
    },
  ]),
  ...ex("multiple-viewpoints", [
    { readme: 0.02, t0: 3000, act: step(1, 0) },
    { id: "turning", readme: 0.05, t0: 3000, act: step(30) },
  ]),
  ...ex(
    "interactive-deform",
    [
      { readme: 0.08, act: step(3) },
      {
        id: "drag",
        readme: 0.09,
        act: (c) => drag(c, [400, 380], [700, 100], 3),
      },
      {
        id: "bounce",
        readme: 0.09,
        act: async (c) => {
          await drag(c, [400, 380], [700, 100], 3);
          await c.mouse.up();
          await c.step(4);
        },
      },
      {
        id: "rotate",
        readme: 0.09,
        act: async (c) => {
          await c.step(1);
          await c.key.down("a");
          await c.step(20);
          await c.key.up("a");
          await c.step(2);
        },
      },
    ],
    { t0: 2000 },
  ),
  ...ex(
    "interactive-ripples",
    [
      { readme: 0.33, act: step(3) },
      {
        id: "click30",
        readme: 0.33,
        act: (c) => clicks(c, [[600, 520]], 1, 29),
      },
      {
        id: "click90",
        readme: 0.34,
        act: (c) => clicks(c, [[600, 520]], 1, 89),
      },
    ],
    { t0: 2000 },
  ),
  ...ex(
    "interactive-holes",
    [
      { readme: 0.5, act: step(3) },
      { id: "click", readme: 0.57, act: (c) => clicks(c, [[400, 300]], 1, 20) },
      {
        id: "many",
        readme: 0.63,
        act: (c) =>
          clicks(
            c,
            [
              [400, 300],
              [300, 350],
              [500, 250],
              [400, 300],
              [200, 200],
            ],
            10,
            60,
          ),
      },
      {
        id: "reset",
        readme: 0.5,
        act: async (c) => {
          await clicks(
            c,
            [
              [400, 300],
              [300, 350],
            ],
            10,
            10,
          );
          await c.clickText("Reset Interactive Holes");
          await c.step(60);
        },
      },
    ],
    { t0: 2000 },
  ),
  ...ex("splat-dissolve-effects", [
    { readme: 0.15, act: dissolve(0) },
    { id: "t10", readme: 0.39, act: dissolve(10) },
    { id: "t40", readme: 0.38, act: dissolve(40) },
  ]),
  ...ex(
    "splat-reveal-effects",
    [
      { id: "magic", readme: 0.36, act: reveal("Magic", 200) },
      { id: "spread", readme: 0.41, act: reveal("Spread", 300) },
      { id: "unroll", readme: 0.34, act: reveal("Unroll", 150) },
      { id: "twister", readme: 1.51, act: reveal("Twister", 400) },
      { id: "rain", readme: 2.38, act: reveal("Rain", 400) },
    ],
    {
      manual: true,
      settle: { minWait: 12000, stable: false },
      note: "fast particles in Twister and Rain",
    },
  ),
  ...ex(
    "lofi",
    [
      { readme: 0.56, act: step(3) },
      { id: "next", readme: 1.18, act: lofiKey("]") },
      { id: "prev", readme: 0.98, act: lofiKey("[") },
      {
        id: "weather",
        readme: 1.01,
        act: async (ctx) => {
          await ctx.clickText("Settings");
          await ctx.clickText("Bad weather");
          await lofiKey("]")(ctx);
        },
      },
      {
        id: "mid",
        readme: 0.83,
        act: async (ctx) => {
          await ctx.key.press("]");
          await ctx.waitForLog("Starting transition");
          await ctx.step(4, 200);
          // Frames at the same time, for the sort to catch up with the
          // splats the transition moved.
          await ctx.step(8, 0);
        },
      },
    ],
    { routes: [lofiRoute], t0: 3000, settle: { minWait: 10000 } },
  ),
  ...ex(
    "splat-transitions",
    [
      { id: "spherical", readme: 0.39, act: transition(null, [1, 3000]) },
      { id: "spherical-mid", readme: 0.39, act: transition(null, [1, 1600]) },
      {
        id: "explosion",
        readme: 0.39,
        act: transition("Explosion", [1, 1200], [1, 600]),
      },
      {
        id: "explosion-frames",
        readme: 0.39,
        act: transition("Explosion", [75, 1000 / 60]),
      },
      { id: "flow", readme: 0.39, act: transition("Flow", [1, 4000]) },
      { id: "flow-late", readme: 0.39, act: transition("Flow", [1, 6800]) },
      { id: "morph", readme: 0.39, act: transition("Morph", [1, 2600]) },
      { id: "morph-late", readme: 0.39, act: transition("Morph", [1, 3300]) },
    ],
    {
      settle: { ready: T_LOADED, minWait: 3000, stable: false },
      note: "README gives 0.08-0.39 over 11 states: each uses the top",
    },
  ),
  ...ex("particle-simulation", [{ readme: 0.2, t0: 0, note: "clock at 0" }]),
  ...ex("splat-flow", [
    { readme: 0.32 },
    { id: "moving", t0: 5000, readme: 0.25, note: "the camera mid-move" },
  ]),
  ...ex("viewer", [
    { id: "url", readme: 0.48, query: `?url=${BUTTERFLY}` },
    {
      id: "file",
      readme: 0.17,
      needs: ["files"],
      act: async (ctx) => {
        await ctx.setFiles("#file-input", await ctx.fixture("butterfly.spz"));
        await ctx.wait(4000);
      },
    },
  ]),
  ...ex("splat-painter", [
    {
      id: "strokes",
      readme: 0.57,
      act: async (ctx) => {
        await ctx.key.press("1");
        await stroke(ctx, 250, 300, 450, 350);
        await ctx.key.press("2");
        await stroke(ctx, 300, 450, 500, 420);
        await ctx.key.press("3");
        await stroke(ctx, 400, 330, 400, 460);
        await ctx.key.press("Escape");
        await ctx.mouse.move(700, 580);
        await ctx.wait(2600);
      },
    },
  ]),
  ...ex("render-cube-depth", [
    { readme: 1.05 },
    {
      id: "depth",
      readme: 6.13,
      note: "packed depth: the low bits differ; cube readbacks agree within 0.3",
      act: async (ctx) => {
        await ctx.click("input[type=checkbox]");
        await ctx.wait(5000);
      },
    },
  ]),
  ...ex("portal", [
    {
      readme: 0.32,
      routes: [portalRoute],
      // Meshes fade in over 2.5 s of performance.now from when they load,
      // and an invisible mesh doesn't hold up the settle: step, let late
      // meshes land, step again, so every loaded mesh is fully faded in.
      act: async (ctx) => {
        await ctx.step(4, 1000);
        await ctx.settle({ minWait: 2000 });
        await ctx.step(4, 1000);
      },
      note: "the characters' SplatSkinning setup throws on both backends (PackedSplats has no boneSplats in this version), so only the cottage (valley.spz) shows",
    },
  ]),
  ...ex("newportal", [
    {
      readme: 0.33,
      routes: [newportalRoute],
      settle: PAGED,
      // The substitute .rad around the camera: the view is mostly the
      // portal disk onto its dark interior, on WebGL as well.
      note: "README's 0.33 is with a non-paged scene; this streams a paged one",
    },
  ]),
  ...ex("splat-portal", [{ readme: 0.58 }]),
  ...ex("editor", [
    { id: "empty", readme: 0.01, minLit: 0 },
    { id: "file", readme: 0.46, query: `?url=${BUTTERFLY}` },
    {
      id: "debug",
      readme: 0.3,
      query: `?url=${BUTTERFLY}`,
      act: async (ctx) => {
        await guiOpen("Debug")(ctx);
        await guiSet("Normal color", true)(ctx);
        await guiOpen("Clip Splats")(ctx);
        await guiSet("Enable clip", true)(ctx);
        await guiSet("Max X", 0.1)(ctx);
        await guiSet("Grid opacity", 0.5)(ctx);
        await ctx.mouse.move(400, 590);
        await ctx.wait(3000);
      },
    },
  ]),

  // Paired test pages in examples/webgpu.
  ...ex("envmap", [
    { webgl: "/examples/webgpu/envmap-webgl.html", settle: { ready: READY } },
  ]),
  ...ex(
    "spark-renderer",
    [
      ["canvas", "", ""],
      ["msaa", "&msaa=1", "msaa=1"],
      ["target", "", "mode=target"],
      ["post", "", "mode=post"],
      ["target-srgb", "", "mode=target&srgb=1"],
    ].map(([id, gl, gpu]) => ({
      id,
      webgl: `/examples/webgpu/spark-renderer.html?backend=webgl${gl}`,
      webgpu: `/examples/webgpu/spark-renderer.html?${gpu}`,
    })),
    {
      settle: { ready: READY },
      act: step(3),
      note: "whole frame, transparent plane included (README's numbers are outside it)",
    },
  ),
  // SparkRenderer options and inputs on WebGPU, without the transparent
  // plane: the splats against WebGL's.
  ...ex(
    "spark-renderer",
    [
      ["plain", "", ""],
      ["cov", "&cov=1&scale=1.4,0.6,1", "cov=1&scale=1.4,0.6,1"],
      ["accum-ext", "&accumExt=1", "accumExt=1"],
      ["2dgs", "&n=20000&flat=1&2dgs=1", "n=20000&flat=1&2dgs=1"],
      ["source", "&source=1", "source=1"],
      // Blended in sRGB, as on the canvas.
      ["msaa-target", "&msaa=1", "mode=target&samples=4&msaa=1&srgb=1"],
      // An ArrayCamera against WebGL drawing each view with setViewport...
      ["views", "&views=2", "views=2"],
      // ...and against WebGPU doing the same.
      ["views-seq", "&views=2&array=0", "views=2", "webgpu"],
    ].map(([id, gl, gpu, reference = "webgl"]) => ({
      id,
      webgl: `/examples/webgpu/spark-renderer.html?backend=${reference}&glass=0${gl}`,
      webgpu: `/examples/webgpu/spark-renderer.html?glass=0&${gpu}`,
    })),
    {
      settle: { ready: READY },
      act: step(3),
    },
  ),
  ...ex(
    "compare",
    [
      { id: "butterfly", query: "?file=butterfly.spz" },
      { id: "synthetic", query: "?n=200000" },
      // sogs.html's tower without the sky.
      { id: "sutro", query: "?file=sutro.zip" },
      // These pages wait for LoD to settle on performance.now.
      { id: "lod", query: "?file=valley.spz&lod=1", freeze: false },
      {
        id: "rad",
        query: "?rad=1",
        freeze: false,
        settle: { ready: READY, maxWait: 120000 },
      },
    ],
    {
      webgl: "/examples/webgpu/compare-webgl.html",
      webgpu: "/examples/webgpu/compare-webgpu.html",
      settle: { ready: READY },
    },
  ),
  ...ex(
    "dyno",
    [
      ...DYNO.map((scene) => ({ id: scene, query: `?scene=${scene}&t=1` })),
      { id: "skin-jacobian", query: "?scene=skin&jacobian=1&t=1" },
      { id: "effect-cov", query: "?scene=effect&cov=1&t=1" },
    ],
    {
      webgl: "/examples/webgpu/dyno-webgl.html",
      webgpu: "/examples/webgpu/dyno-webgpu.html",
      settle: { ready: READY },
    },
  ),
];

// Thresholds from measured runs (Chrome 154, Apple GPU, October 2026), each
// the largest of 2-4 runs × 1.5 + a little: the % of pixels off by more than
// 16 for every case (+ 0.5: WebGL's sort lands a frame early or late under
// stepped per-frame animation, e.g. interactive-deform:bounce varies 0-0.32%
// run to run on WebGL alone), and the mean where the README has no number
// (+ 0.15).
const PCT = {
  "hello-world": 0.5,
  "multiple-splats": 0.53,
  "raycasting:clicked": 0.52,
  interactivity: 0.37,
  extsplats: 0.25,
  nonlod: 0.25,
  "lod-example": 0.5,
  "multi-lod": 0.74,
  "depth-of-field": 0.5,
  "splat-shader-effects": 1.33,
  "procedural-splats": 11.29,
  sogs: 1.5,
  "debug-color": 0.5,
  "lod-on-demand": 0.5,
  "particle-animation": 2.31,
  glsl: 0.25,
  "dynamic-lighting": 0.5,
  "dynamic-lighting:debug": 0.5,
  "mobile-joystick": 0.5,
  "mobile-joystick:walk": 0.5,
  "multiple-viewpoints": 0.5,
  "multiple-viewpoints:turning": 0.5,
  "interactive-deform": 0.5,
  "interactive-deform:drag": 0.5,
  "interactive-deform:bounce": 0.5,
  "interactive-deform:rotate": 0.5,
  "interactive-ripples": 0.5,
  "interactive-ripples:click30": 0.5,
  "interactive-ripples:click90": 0.5,
  "interactive-holes": 0.5,
  "interactive-holes:click": 0.68,
  "interactive-holes:many": 0.79,
  "interactive-holes:reset": 0.5,
  "splat-dissolve-effects": 0.8,
  "splat-dissolve-effects:t10": 0.92,
  "splat-dissolve-effects:t40": 0.79,
  "splat-reveal-effects:magic": 0.52,
  "splat-reveal-effects:spread": 0.5,
  "splat-reveal-effects:unroll": 0.5,
  "splat-reveal-effects:twister": 6.96,
  "splat-reveal-effects:rain": 6.16,
  lofi: 0.39,
  "splat-transitions:spherical": 0.64,
  "splat-transitions:spherical-mid": 0.64,
  "splat-transitions:explosion": 0.72,
  "splat-transitions:explosion-frames": 0.71,
  "splat-transitions:flow": 0.56,
  "splat-transitions:flow-late": 0.52,
  "splat-transitions:morph": 0.5,
  "splat-transitions:morph-late": 0.5,
  "particle-simulation": 0.5,
  "splat-flow": 0.5,
  "splat-flow:moving": 0.59,
  "viewer:url": 0.5,
  "viewer:file": 0.5,
  "splat-painter:strokes": 0.5,
  "render-cube-depth": 0.5,
  "render-cube-depth:depth": 16.49,
  portal: 0.39,
  newportal: 0.25,
  "splat-portal": 0.68,
  "editor:empty": 0.5,
  "editor:file": 0.52,
  "editor:debug": 0.5,
  envmap: 2.79,
  "spark-renderer:canvas": 8.42,
  "spark-renderer:msaa": 8.45,
  "spark-renderer:target": 11.58,
  "spark-renderer:post": 11.58,
  "spark-renderer:target-srgb": 8.42,
  "spark-renderer:plain": 0.82,
  "spark-renderer:cov": 0.82,
  "spark-renderer:accum-ext": 0.8,
  // Surfels are exact 2DGS discs with Huang's screen filter in WebGPU
  // (wt-surfel-draw), WebGL's 2DGS quads have none: edge-on and small
  // discs differ by design (4.08% measured).
  "spark-renderer:2dgs": 6.2,
  "spark-renderer:source": 0.82,
  "spark-renderer:msaa-target": 0.85,
  "spark-renderer:views": 1.82,
  "spark-renderer:views-seq": 0.5,
  "compare:butterfly": 0.5,
  "compare:synthetic": 0.5,
  "dyno:effect": 0.5,
  "dyno:depth": 0.5,
  "dyno:normal": 0.5,
  "dyno:edit": 0.5,
  "dyno:snow": 0.5,
  "dyno:skin": 0.5,
  "dyno:rgba": 0.5,
  "dyno:effect-cov": 0.5,
  "streaming-lod": 0.59,
  "on-demand": 0.53,
  "lofi:next": 0.81,
  "lofi:prev": 0.95,
  "lofi:weather": 1.72,
  "compare:sutro": 0.91,
  "compare:lod": 0.5,
  "compare:rad": 0.5,
  "lofi:mid": 0.52,
};
const MEAN = {
  "procedural-splats": 4.77,
  envmap: 2.46,
  "spark-renderer:canvas": 3.78,
  "spark-renderer:msaa": 3.8,
  "spark-renderer:target": 4.53,
  "spark-renderer:post": 4.52,
  "spark-renderer:target-srgb": 3.77,
  "spark-renderer:plain": 0.57,
  "spark-renderer:cov": 0.59,
  "spark-renderer:accum-ext": 0.54,
  "spark-renderer:2dgs": 2.35, // exact surfels vs WebGL's 2DGS quads: 1.56 measured
  "spark-renderer:source": 0.57,
  "spark-renderer:msaa-target": 0.57,
  "spark-renderer:views": 1.13,
  "spark-renderer:views-seq": 0.15,
  "compare:butterfly": 0.2,
  "compare:synthetic": 0.18,
  "dyno:effect": 0.16,
  "dyno:depth": 0.15,
  "dyno:normal": 0.15,
  "dyno:edit": 0.15,
  "dyno:snow": 0.16,
  "dyno:skin": 0.15,
  "dyno:rgba": 0.15,
  "dyno:effect-cov": 0.16,
  "compare:sutro": 1.03,
  "compare:lod": 0.78,
  "compare:rad": 0.7,
};
for (const c of cases) {
  c.max = { mean: MEAN[c.id], pct: PCT[c.id], ...c.max };
}

// Fixture files input scripts load through file inputs (ctx.fixture(name)),
// downloaded once into the output directory.
export const fixtures = {
  "butterfly.spz": BUTTERFLY,
};

// Every route, for the runner's Vite (paths don't overlap between examples).
export const routes = cases.flatMap((c) => c.routes ?? []);
