// The node pool's run decoder (thread CU): CMPC v4 runs inflated and their
// nodes' records packed off the main thread. The main thread sends a batch
// of jobs (a run's deflate bytes, the run's place and the nodes it wants)
// and gets the records back in job order; the runs stay inflated here in a
// small LRU, so a run asked for again is packed without its bytes.

import { inflateSync } from "fflate";
import { NODE_RECORD_WORDS, packRunRecords } from "./compactRecords";

/** A run to pack: its key (page x 64 + run), its nodes and first place, the wanted nodes. */
export interface RunJob {
  key: number;
  page: number;
  base: number;
  m: number;
  /** The run's deflate bytes (absent: the worker has it inflated, `cached`). */
  bytes?: Uint8Array;
  /** Or the run already inflated (the main thread's copy). */
  planes?: Uint8Array;
  /** The nodes wanted, as places within the run. */
  locals: Uint16Array;
}

export interface RunBatch {
  id: number;
  jobs: RunJob[];
}

export interface RunReply {
  id: number;
  /** The records, job after job (NODE_RECORD_WORDS x 4 bytes a node). */
  records: Uint8Array;
  /** Jobs whose run was neither sent nor held (their nodes not packed). */
  lost: number[];
  ms: number;
  inflates: number;
}

const LRU = 1024;
const inflated = new Map<number, Uint8Array>();

function planes(job: RunJob): Uint8Array | null {
  const hit = inflated.get(job.key);
  if (hit) {
    inflated.delete(job.key);
    inflated.set(job.key, hit);
    return hit;
  }
  if (!job.bytes && !job.planes) return null;
  const p = job.planes ?? inflateSync(job.bytes as Uint8Array);
  inflated.set(job.key, p);
  while (inflated.size > LRU) {
    inflated.delete(inflated.keys().next().value as number);
  }
  return p;
}

function onBatch(event: MessageEvent<RunBatch | { drop: number[] }>) {
  const d = event.data;
  if ("drop" in d) {
    for (const k of d.drop) inflated.delete(k);
    return;
  }
  const t0 = performance.now();
  const R = NODE_RECORD_WORDS * 4;
  let total = 0;
  for (const j of d.jobs) total += j.locals.length;
  const records = new Uint8Array(total * R);
  const lost: number[] = [];
  let at = 0;
  let inflates = 0;
  d.jobs.forEach((job, x) => {
    const had = inflated.has(job.key);
    const p = planes(job);
    if (!had && p && !job.planes) inflates += 1;
    if (p)
      packRunRecords(p, job.m, job.locals, job.page, job.base, records, at);
    else lost.push(x);
    at += job.locals.length;
  });
  const reply: RunReply = {
    id: d.id,
    records,
    lost,
    ms: performance.now() - t0,
    inflates,
  };
  (self as unknown as Worker).postMessage(reply, [records.buffer]);
}

// A Web Worker only (the module also loads on its own, for the checks).
if (typeof self !== "undefined") self.onmessage = onBatch;
