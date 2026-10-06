// LoD transitions that fade instead of popping (WgpuLodOptions.lodFadeMs).
//
// The LoD traversal hands each mesh a cut: the source indices to draw. Spark
// swaps one cut for the next in a frame, so a node that splits into its
// children (or merges back) pops. With a fade, the splats that enter the cut
// fade in over the fade time while the ones that left it stay drawn, then
// fade out over the next fade time (LODGE's chunk blending, Babylon's
// streamed LoD: "transitions never flash or leave gaps"). Holding the old
// splats until the new are in keeps the surface covered throughout -- a
// cross-fade would dip both to half and show what is behind.
//
// The fade rides in the top byte of each index the GPU reads (generate's
// GEN_LOD_FADE): 0 drawn, 255 gone; the splat's alpha scales by 1 - k/255.
// So indices must be below 2^24 (a 256-page pool is exactly that); a cut with
// a larger index is drawn without a fade.
//
// At rest the drawn indices are the cut itself, in the traversal's order, so
// the image is the same as without a fade.

/** The largest index a fade can ride with (generate's GEN_LOD_INDEX_MASK). */
export const LOD_FADE_MAX_INDEX = 0x00ffffff;
/** A frame's step is held to this much time (a stall does not skip a fade). */
export const LOD_FADE_MAX_STEP_MS = 1000 / 30;

/** What the renderer draws: the plain indices and, while fading, the faded. */
export interface LodFadeFrame {
  indices: Uint32Array;
  faded?: Uint32Array;
}

/** One mesh's LoD fade. */
export class LodFade {
  /** Per source index: how drawn it is, 0 to 1. */
  private weight = new Float32Array(0);
  /** Per source index: in the drawn list (1) or not (0). */
  private drawnFlag = new Uint8Array(0);
  /** Per source index: the stamp of the last cut it was in. */
  private cutStamp = new Uint32Array(0);
  /**
   * Per source index: when it left the cut, ms after `epoch` (NaN: it is
   * in the cut).
   */
  private leftAt = new Float32Array(0);
  private epoch = Number.NaN;
  private stamp = 0;
  private drawn = new Uint32Array(0);
  private drawnCount = 0;
  private cut: Uint32Array = new Uint32Array(0);
  private lastTime = Number.NaN;
  private fading = false;
  private plain = true;

  constructor(public durationMs: number) {}

  /** Whether step() still has a transition to draw. */
  get active(): boolean {
    return this.fading;
  }

  /** Forgets the drawn splats: the next cut is drawn at once. */
  reset() {
    for (let k = 0; k < this.drawnCount; k++) {
      const i = this.drawn[k];
      this.drawnFlag[i] = 0;
      this.weight[i] = 0;
    }
    this.drawnCount = 0;
    this.fading = false;
    this.plain = true;
    this.lastTime = Number.NaN;
  }

  private grow(index: number) {
    if (index < this.weight.length) return;
    const n = Math.max(index + 1, Math.ceil(this.weight.length * 1.5), 1024);
    const weight = new Float32Array(n);
    weight.set(this.weight);
    const drawnFlag = new Uint8Array(n);
    drawnFlag.set(this.drawnFlag);
    const cutStamp = new Uint32Array(n);
    cutStamp.set(this.cutStamp);
    const leftAt = new Float32Array(n).fill(Number.NaN);
    leftAt.set(this.leftAt);
    this.weight = weight;
    this.drawnFlag = drawnFlag;
    this.cutStamp = cutStamp;
    this.leftAt = leftAt;
  }

  private pushDrawn(index: number) {
    if (this.drawnCount === this.drawn.length) {
      const grown = new Uint32Array(Math.max(1024, this.drawn.length * 2));
      grown.set(this.drawn);
      this.drawn = grown;
    }
    this.drawn[this.drawnCount++] = index;
  }

  /**
   * A new cut from the traversal, at time `now` (ms). Returns what to draw
   * now; the transition then goes on in step().
   */
  setCut(cut: Uint32Array, now: number): LodFadeFrame {
    let largest = 0;
    for (let k = 0; k < cut.length; k++) largest = Math.max(largest, cut[k]);
    this.cut = cut;
    if (largest > LOD_FADE_MAX_INDEX || this.durationMs <= 0) {
      this.reset();
      return { indices: cut };
    }
    this.grow(largest);
    if (Number.isNaN(this.epoch)) this.epoch = now;
    this.stamp = (this.stamp + 1) >>> 0 || 1;
    const stamp = this.stamp;
    const first = this.drawnCount === 0;
    for (let k = 0; k < cut.length; k++) {
      const i = cut[k];
      this.cutStamp[i] = stamp;
      this.leftAt[i] = Number.NaN;
      if (!this.drawnFlag[i]) {
        this.drawnFlag[i] = 1;
        // Nothing drawn before (a first cut, a mesh shown again): at once.
        this.weight[i] = first ? 1 : 0;
        this.pushDrawn(i);
      }
    }
    for (let k = 0; k < this.drawnCount; k++) {
      const i = this.drawn[k];
      if (this.cutStamp[i] !== stamp && Number.isNaN(this.leftAt[i])) {
        this.leftAt[i] = now - this.epoch;
      }
    }
    if (Number.isNaN(this.lastTime)) this.lastTime = now;
    this.fading = !first;
    return this.frame(true) as LodFadeFrame;
  }

  /**
   * Advances the transition to `now` (ms): null when nothing changed (at
   * rest), else what to draw.
   */
  step(now: number): LodFadeFrame | null {
    if (!this.fading) return null;
    const dt = Math.min(
      Math.max(now - (Number.isNaN(this.lastTime) ? now : this.lastTime), 0),
      LOD_FADE_MAX_STEP_MS,
    );
    this.lastTime = now;
    const rate = dt / Math.max(this.durationMs, 1e-3);
    const stamp = this.stamp;
    let kept = 0;
    let fading = false;
    for (let k = 0; k < this.drawnCount; k++) {
      const i = this.drawn[k];
      let w = this.weight[i];
      if (this.cutStamp[i] === stamp) {
        if (w < 1) w = Math.min(1, w + rate);
        if (w < 1) fading = true;
      } else {
        // Held until the incoming splats are in, then faded out.
        if (now - this.epoch - this.leftAt[i] >= this.durationMs) w -= rate;
        if (w <= 0) {
          this.weight[i] = 0;
          this.drawnFlag[i] = 0;
          this.leftAt[i] = Number.NaN;
          continue;
        }
        fading = true;
      }
      this.weight[i] = w;
      this.drawn[kept++] = i;
    }
    this.drawnCount = kept;
    this.fading = fading;
    return this.frame(false);
  }

  private frame(force: boolean): LodFadeFrame | null {
    if (!this.fading) {
      // At rest: the cut itself, in the traversal's order.
      if (this.plain && !force) return null;
      this.plain = true;
      return { indices: this.cut };
    }
    this.plain = false;
    const n = this.drawnCount;
    const indices = this.drawn.slice(0, n);
    const faded = new Uint32Array(n);
    for (let k = 0; k < n; k++) {
      const i = indices[k];
      const gone = Math.round((1 - this.weight[i]) * 255);
      faded[k] = (i | (gone << 24)) >>> 0;
    }
    return { indices, faded };
  }
}
