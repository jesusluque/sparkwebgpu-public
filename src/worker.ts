import init_wasm, {
  sort_splats,
  sort32_splats,
  decode_to_gsplatarray,
  decode_to_csplatarray,
  decode_to_packedsplats,
  new_lod_tree,
  new_shared_lod_tree,
  init_lod_tree,
  dispose_lod_tree,
  set_lod_occlusion_codebook,
  traverse_lod_trees,
  dynamic_traverse_lod_trees,
  type ChunkDecoder,
  tiny_lod_packedsplats,
  bhatt_lod_packedsplats,
  update_compact_lod_trees,
  set_compact_quant,
  update_lod_trees,
  decode_to_extsplats,
  tiny_lod_extsplats,
  bhatt_lod_extsplats,
  get_lod_tree_level,
  athc_merged_pages,
  athc3_merged_pages,
  athc_layout,
  athc_prefix_bytes,
  lod_memory_report,
} from "spark-rs";
import type {
  AttribValues,
  ExtResult,
  PackedResult,
  SplatEncoding,
} from "./defines";

const rpcHandlers = {
  athcLayout,
  athcMergedPages,
  athc3MergedPages,
  sortSplats16,
  sortSplats32,
  loadPackedSplats,
  loadExtSplats,
  tinyLodPackedSplats,
  qualityLodPackedSplats,
  tinyLodExtSplats,
  qualityLodExtSplats,
  newLodTree,
  newSharedLodTree,
  initLodTree,
  disposeLodTree,
  setLodOcclusionCodebook,
  updateLodTrees,
  updateCompactLodTrees,
  traverseLodTrees,
  getLodTreeLevel,
  nextChunk,
  wasmMemory,
  lodMemory,
  setCompactQuant,
};
export type rpcHandlers = typeof rpcHandlers;

let wasmExports: { memory: WebAssembly.Memory } | null = null;
/** The compact traversal's quantization for the pages that land from now on (thread CT). */
function setCompactQuant({ bits }: { bits: number[] }) {
  const b = [0, 1, 2, 3, 4].map((k) => bits[k] ?? 0);
  set_compact_quant(b[0], b[1], b[2], b[3], b[4]);
  return {};
}
/** What the LoD trees hold, bytes (lod_tree.rs lod_memory_report). */
function lodMemory() {
  return {
    ...(lod_memory_report() as Record<string, number>),
    heap: wasmExports?.memory.buffer.byteLength ?? 0,
  };
}
/** The worker's WASM heap, bytes (it never shrinks). */
function wasmMemory() {
  return { bytes: wasmExports?.memory.buffer.byteLength ?? 0 };
}

async function onMessage(event: MessageEvent) {
  const {
    id,
    name,
    args,
  }: { id: unknown; name: keyof typeof rpcHandlers; args: unknown } =
    event.data;
  try {
    const handler = rpcHandlers[name] as (
      args: unknown,
      options: { sendStatus: (data: unknown) => void },
    ) => unknown | Promise<unknown>;
    if (!handler) {
      throw new Error(`Unknown worker RPC: ${name}`);
    }

    const sendStatus = (data: unknown) => {
      self.postMessage(
        { id, status: data },
        { transfer: getTransferable(data) },
      );
    };
    const result = await handler(args, { sendStatus });
    self.postMessage({ id, result }, { transfer: getTransferable(result) });
  } catch (error) {
    console.warn(`Worker error: ${error}`);
    self.postMessage({ id, error }, { transfer: getTransferable(error) });
  }
}

function sortSplats16({
  numSplats,
  readback,
  ordering,
}: {
  numSplats: number;
  readback: Uint16Array<ArrayBuffer>;
  ordering: Uint32Array<ArrayBuffer>;
}) {
  const activeSplats = sort_splats(numSplats, readback, ordering);
  return { activeSplats, readback, ordering };
}

function sortSplats32({
  numSplats,
  readback,
  ordering,
}: {
  numSplats: number;
  readback: Uint32Array<ArrayBuffer>;
  ordering: Uint32Array<ArrayBuffer>;
}) {
  const activeSplats = sort32_splats(numSplats, readback, ordering);
  return { activeSplats, readback, ordering };
}

async function decodeBytesUrl({
  decoder,
  fileBytes,
  url,
  requestHeader,
  withCredentials,
  chunked,
  chunkedLength,
  sendStatus,
}: {
  decoder: ChunkDecoder;
  fileBytes?: Uint8Array;
  url?: string;
  requestHeader?: Record<string, string>;
  withCredentials?: boolean;
  chunked?: boolean;
  chunkedLength?: number;
  sendStatus: (data: unknown) => void;
}) {
  let readStream: ReadableStream<Uint8Array>;
  let streamLength = 0;

  if (fileBytes) {
    readStream = new ReadableStream({
      start(controller) {
        controller.enqueue(fileBytes);
        controller.close();
      },
    });
    streamLength = fileBytes.length;
  } else if (url) {
    const request = new Request(url, {
      headers: requestHeader ? new Headers(requestHeader) : undefined,
      credentials: withCredentials ? "include" : "same-origin",
    });

    const response = await fetch(request);
    if (!response.ok || !response.body) {
      throw new Error(
        `Failed to fetch "${url}": ${response.status} ${response.statusText}`,
      );
    }
    readStream = response.body;
    const contentLength = Number.parseInt(
      response.headers.get("Content-Length") || "0",
    );
    streamLength = Number.isNaN(contentLength) ? 0 : contentLength;
  } else if (chunked) {
    readStream = new ReadableStream({
      async start(controller) {
        async function readNext() {
          const readNextChunk: Promise<Uint8Array> = new Promise((resolve) => {
            nextChunkWaiter = resolve;
          });
          sendStatus({ nextChunk: true });
          const nextChunk = await readNextChunk;

          if (nextChunk.length === 0) {
            controller.close();
            return true;
          }

          controller.enqueue(nextChunk);
          return false;
        }

        let final: boolean;
        do {
          final = await readNext();
        } while (!final);
      },
    });
    streamLength = chunkedLength ?? 0;
  } else {
    throw new Error("No url or fileBytes provided");
  }

  const reader = readStream.getReader();
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      reader.releaseLock();
      break;
    }
    loaded += value.length;
    sendStatus({ loaded, total: streamLength });

    decoder.push(value);
  }

  const decoded = decoder.finish();
  return decoded;
}

type DecodedPackedResult = {
  readonly numSplats: number;
  readonly packed: Uint32Array;
  readonly sh1?: Uint32Array;
  readonly sh2?: Uint32Array;
  readonly sh3?: Uint32Array;
  readonly sh1Codes?: Uint32Array;
  readonly sh2Codes?: Uint32Array;
  readonly sh3Codes?: Uint32Array;
  readonly lodTree?: Uint32Array;
  readonly lodError?: Uint32Array;
  readonly lodOcclusion?: Uint32Array;
  readonly lodOcclusionCodebook?: Uint32Array;
  readonly lodOcclusionBounds?: Float32Array;
  readonly splatEncoding: SplatEncoding;
  readonly attribSpecs?: AttribValues["specs"];
  readonly attribColumns?: AttribValues["values"];
};

function toPackedResult(packed: DecodedPackedResult): PackedResult {
  return {
    numSplats: packed.numSplats,
    packedArray: packed.packed,
    extra: {
      sh1: packed.sh1,
      sh2: packed.sh2,
      sh3: packed.sh3,
      sh1Codes: packed.sh1Codes,
      sh2Codes: packed.sh2Codes,
      sh3Codes: packed.sh3Codes,
      lodTree: packed.lodTree,
      lodError: packed.lodError,
      lodOcclusion: packed.lodOcclusion,
      lodOcclusionCodebook: packed.lodOcclusionCodebook,
      lodOcclusionBounds: packed.lodOcclusionBounds,
      attribs: attribValues(packed),
    },
    splatEncoding: packed.splatEncoding,
  };
}

// The attributes a WASM result carries (decoded from a PLY or .rad, or
// merged by a LOD build), for AttribPool.from() on the WebGPU side.
function attribValues(result: {
  numSplats: number;
  attribSpecs?: AttribValues["specs"];
  attribColumns?: AttribValues["values"];
}): AttribValues | undefined {
  const { attribSpecs: specs, attribColumns: values } = result;
  if (!specs?.length || !values) return undefined;
  return { count: result.numSplats, specs, values };
}

async function loadPackedSplats(
  {
    url,
    requestHeader,
    withCredentials,
    fileBytes,
    fileType,
    pathName,
    chunked,
    chunkedLength,
    encoding,
    lod,
    lodBase,
    lodAbove,
    nonLod,
    sh1Codes,
    sh2Codes,
    sh3Codes,
  }: {
    url?: string;
    requestHeader?: Record<string, string>;
    withCredentials?: boolean;
    fileBytes?: Uint8Array;
    fileType?: string;
    pathName?: string;
    chunked?: boolean;
    chunkedLength?: number;
    encoding?: SplatEncoding;
    lod?: boolean | "quality";
    lodBase?: number;
    lodAbove?: number;
    nonLod?: boolean;
    sh1Codes?: Uint32Array;
    sh2Codes?: Uint32Array;
    sh3Codes?: Uint32Array;
  },
  {
    sendStatus,
  }: {
    sendStatus: (data: unknown) => void;
  },
) {
  // console.log("loadPackedSplats", { url, requestHeader, withCredentials, fileBytes, fileType, pathName, stream, streamLength, encoding, lod, lodBase, lodAbove, nonLod });
  if (!lod) {
    const decoder = decode_to_packedsplats(
      fileType,
      pathName ?? url,
      encoding,
      sh1Codes,
      sh2Codes,
      sh3Codes,
    );
    const decoded = await decodeBytesUrl({
      decoder,
      fileBytes,
      url,
      requestHeader,
      withCredentials,
      chunked,
      chunkedLength,
      sendStatus,
    });
    const result = toPackedResult(decoded as DecodedPackedResult);
    if (result.splatEncoding.lodOpacity) {
      return { lodSplats: result };
    }
    return result;
  }

  const decoder = decode_to_csplatarray(fileType, pathName ?? url, encoding);
  const decoded = await decodeBytesUrl({
    decoder,
    fileBytes,
    url,
    requestHeader,
    withCredentials,
    chunked,
    chunkedLength,
    sendStatus,
  });

  if (decoded.has_lod()) {
    const result = toPackedResult(
      decoded.to_packedsplats_lod() as DecodedPackedResult,
    );
    return { lodSplats: result };
  }

  if (lodAbove !== undefined) {
    if (decoded.len() < lodAbove) {
      return toPackedResult(decoded.to_packedsplats() as DecodedPackedResult);
    }
  }

  let result:
    | (PackedResult & { lodSplats?: PackedResult })
    | { lodSplats?: PackedResult } = {};

  if (nonLod) {
    // Wait until LoD computation is complete before resolving full PackedSplats result
    result = toPackedResult(decoded.to_packedsplats() as DecodedPackedResult);
  }

  const initialSplats = decoded.len();
  const lodName = lod === "quality" ? "Bhatt" : "Tiny";
  console.log(
    `Loaded ${initialSplats} splats. Starting ${lodName} LoD build...`,
  );

  const lodStart = performance.now();
  if (lod === "quality") {
    const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.25));
    decoded.bhatt_lod(base);
  } else {
    const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.5));
    decoded.tiny_lod(base, false);
  }
  const lodDuration = performance.now() - lodStart;

  console.log(
    `${lodName} LoD: ${initialSplats} -> ${decoded.len()} (${lodDuration} ms)`,
  );

  const lodPacked = decoded.to_packedsplats_lod();
  result.lodSplats = toPackedResult(lodPacked as DecodedPackedResult);
  return result as
    | (PackedResult & { lodSplats: PackedResult })
    | { lodSplats: PackedResult };
}

type DecodedExtResult = {
  numSplats: number;
  ext0: Uint32Array;
  ext1: Uint32Array;
  sh1?: Uint32Array;
  sh2?: Uint32Array;
  sh3a?: Uint32Array;
  sh3b?: Uint32Array;
  sh1Codes?: Uint32Array;
  sh2Codes?: Uint32Array;
  sh3Codes?: [Uint32Array, Uint32Array];
  lodTree?: Uint32Array;
  lodError?: Uint32Array;
  lodOcclusion?: Uint32Array;
  lodOcclusionCodebook?: Uint32Array;
  lodOcclusionBounds?: Float32Array;
  attribSpecs?: AttribValues["specs"];
  attribColumns?: AttribValues["values"];
};

function toExtResult(packed: DecodedExtResult): ExtResult {
  return {
    numSplats: packed.numSplats,
    extArrays: [packed.ext0, packed.ext1],
    extra: {
      sh1: packed.sh1,
      sh2: packed.sh2,
      sh3a: packed.sh3a,
      sh3b: packed.sh3b,
      sh1Codes: packed.sh1Codes,
      sh2Codes: packed.sh2Codes,
      sh3Codes: packed.sh3Codes,
      lodTree: packed.lodTree,
      lodError: packed.lodError,
      lodOcclusion: packed.lodOcclusion,
      lodOcclusionCodebook: packed.lodOcclusionCodebook,
      lodOcclusionBounds: packed.lodOcclusionBounds,
      attribs: attribValues(packed),
    },
  };
}

async function loadExtSplats(
  {
    url,
    requestHeader,
    withCredentials,
    fileBytes,
    fileType,
    pathName,
    chunked,
    chunkedLength,
    lod,
    lodBase,
    lodAbove,
    nonLod,
    sh1Codes,
    sh2Codes,
    sh3Codes,
  }: {
    url?: string;
    requestHeader?: Record<string, string>;
    withCredentials?: boolean;
    fileBytes?: Uint8Array;
    fileType?: string;
    pathName?: string;
    chunked?: boolean;
    chunkedLength?: number;
    lod?: boolean | "quality";
    lodBase?: number;
    lodAbove?: number;
    nonLod?: boolean;
    sh1Codes?: Uint32Array;
    sh2Codes?: Uint32Array;
    sh3Codes?: [Uint32Array, Uint32Array];
  },
  {
    sendStatus,
  }: {
    sendStatus: (data: unknown) => void;
  },
) {
  // console.log("loadExtSplats", { url, requestHeader, withCredentials, fileBytes, fileType, pathName, stream, streamLength, lod, lodBase, lodAbove, nonLod });
  if (!lod) {
    const decoder = decode_to_extsplats(
      fileType,
      pathName ?? url,
      sh1Codes,
      sh2Codes,
      sh3Codes,
    );
    const decoded = await decodeBytesUrl({
      decoder,
      fileBytes,
      url,
      requestHeader,
      withCredentials,
      chunked,
      chunkedLength,
      sendStatus,
    });
    const result = toExtResult(decoded as DecodedExtResult);
    if (result.extra.lodTree) {
      return { lodSplats: result };
    }
    return result;
  }

  const decoder = decode_to_gsplatarray(fileType, pathName ?? url);
  const decoded = await decodeBytesUrl({
    decoder,
    fileBytes,
    url,
    requestHeader,
    withCredentials,
    chunked,
    chunkedLength,
    sendStatus,
  });

  if (decoded.has_lod()) {
    return {
      lodSplats: toExtResult(decoded.to_extsplats_lod() as DecodedExtResult),
    };
  }

  if (lodAbove !== undefined) {
    if (decoded.len() < lodAbove) {
      return toExtResult(decoded.to_extsplats() as DecodedExtResult);
    }
  }

  let result:
    | (ExtResult & { lodSplats?: ExtResult })
    | { lodSplats?: ExtResult } = {};

  if (nonLod) {
    // Wait until LoD computation is complete before resolving full PackedSplats result
    result = toExtResult(decoded.to_extsplats() as DecodedExtResult);
  }

  const initialSplats = decoded.len();
  const lodName = lod === "quality" ? "Bhatt" : "Tiny";
  console.log(
    `Loaded ${initialSplats} splats. Starting ${lodName} LoD build...`,
  );

  const lodStart = performance.now();
  if (lod === "quality") {
    const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.75));
    decoded.bhatt_lod(base);
  } else {
    const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.5));
    decoded.tiny_lod(base, false);
  }
  const lodDuration = performance.now() - lodStart;

  console.log(
    `${lodName} LoD: ${initialSplats} -> ${decoded.len()} (${lodDuration} ms)`,
  );

  const lodPacked = decoded.to_extsplats_lod();
  result.lodSplats = toExtResult(lodPacked as DecodedExtResult);
  return result as
    | (ExtResult & { lodSplats: ExtResult })
    | { lodSplats: ExtResult };
}

async function tinyLodPackedSplats({
  numSplats,
  packedArray,
  extra,
  lodBase,
  rgba,
  encoding,
}: {
  numSplats: number;
  packedArray: Uint32Array;
  extra?: Record<string, unknown>;
  lodBase?: number;
  rgba?: Uint8Array;
  encoding: SplatEncoding;
}) {
  const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.5));
  const lodStart = performance.now();
  const filter = false;
  const decoded = tiny_lod_packedsplats(
    numSplats,
    packedArray,
    extra as object,
    base,
    filter,
    rgba,
    encoding,
  );
  const lodDuration = performance.now() - lodStart;
  const result = toPackedResult(decoded as DecodedPackedResult);
  console.log(
    `Tiny LoD: ${numSplats} -> ${result.numSplats} (${lodDuration} ms)`,
  );
  return result;
}

async function qualityLodPackedSplats({
  numSplats,
  packedArray,
  extra,
  lodBase,
  rgba,
  encoding,
}: {
  numSplats: number;
  packedArray: Uint32Array;
  extra?: Record<string, unknown>;
  lodBase?: number;
  rgba?: Uint8Array;
  encoding: SplatEncoding;
}) {
  const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.75));
  const lodStart = performance.now();
  const decoded = bhatt_lod_packedsplats(
    numSplats,
    packedArray,
    extra as object,
    base,
    rgba,
    encoding,
  );
  const lodDuration = performance.now() - lodStart;
  const result = toPackedResult(decoded as DecodedPackedResult);
  console.log(
    `Bhatt LoD: ${numSplats} -> ${result.numSplats} (${lodDuration} ms)`,
  );
  return result;
}

async function tinyLodExtSplats({
  numSplats,
  extArrays,
  extra,
  lodBase,
  rgba,
  encoding,
}: {
  numSplats: number;
  extArrays: [Uint32Array, Uint32Array];
  extra?: Record<string, unknown>;
  lodBase?: number;
  rgba?: Uint8Array;
  encoding: SplatEncoding;
}) {
  const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.5));
  const lodStart = performance.now();
  const filter = false;
  const decoded = tiny_lod_extsplats(
    numSplats,
    extArrays[0],
    extArrays[1],
    extra as object,
    base,
    filter,
    rgba,
  );
  const lodDuration = performance.now() - lodStart;
  const result = toExtResult(decoded as DecodedExtResult);
  console.log(
    `Tiny LoD: ${numSplats} -> ${result.numSplats} (${lodDuration} ms)`,
  );
  return result;
}

async function qualityLodExtSplats({
  numSplats,
  extArrays,
  extra,
  lodBase,
  rgba,
  encoding,
}: {
  numSplats: number;
  extArrays: readonly [Uint32Array, Uint32Array];
  extra?: Record<string, unknown>;
  lodBase?: number;
  rgba?: Uint8Array;
  encoding?: SplatEncoding;
}) {
  const base = Math.max(1.1, Math.min(2.0, lodBase ?? 1.75));
  const lodStart = performance.now();
  const decoded = bhatt_lod_extsplats(
    numSplats,
    extArrays[0],
    extArrays[1],
    extra as object,
    base,
    rgba,
  );
  const lodDuration = performance.now() - lodStart;
  const result = toExtResult(decoded as DecodedExtResult);
  console.log(
    `Bhatt LoD: ${numSplats} -> ${result.numSplats} (${lodDuration} ms)`,
  );
  return result;
}

function newLodTree({
  capacity,
  slot,
  compact,
}: {
  capacity: number;
  /** Splats a pool page holds (default 65 536; athenea's band pages fewer). */
  slot?: number;
  /** A compact tree (14-byte nodes, updateCompactLodTrees; thread CQ). */
  compact?: boolean;
}) {
  const { lodId } = new_lod_tree(capacity, slot, compact) as {
    lodId: number;
  };
  return { lodId };
}

function newSharedLodTree({
  lodId,
}: {
  lodId: number;
}) {
  const { lodId: newLodId } = new_shared_lod_tree(lodId) as { lodId: number };
  return { lodId: newLodId };
}

function initLodTree({
  numSplats,
  lodTree,
  lodError,
  lodOcclusion,
}: {
  numSplats: number;
  lodTree: Uint32Array;
  lodError?: Uint32Array;
  lodOcclusion?: Uint32Array;
}) {
  const { lodId, chunkToPage } = init_lod_tree(
    numSplats,
    lodTree,
    lodError,
    lodOcclusion,
  ) as {
    lodId: number;
    chunkToPage: Uint32Array;
  };
  return { lodId, chunkToPage };
}

/** A tree's occlusion codebook (athenea's masks) and the asset's box. */
function setLodOcclusionCodebook({
  lodId,
  codebook,
  bounds,
}: {
  lodId: number;
  codebook: Uint32Array;
  bounds: Float32Array;
}) {
  set_lod_occlusion_codebook(lodId, codebook, bounds);
}

function disposeLodTree({ lodId }: { lodId: number }) {
  dispose_lod_tree(lodId);
}

function updateLodTrees({
  ranges,
}: {
  ranges: {
    lodId: number;
    pageBase: number;
    chunkBase: number;
    count: number;
    lodTreeData?: Uint32Array;
    lodErrorData?: Uint32Array;
    lodOcclusionData?: Uint32Array;
  }[];
}) {
  const lodIds = new Uint32Array(ranges.map(({ lodId }) => lodId));
  const pageBases = new Uint32Array(ranges.map(({ pageBase }) => pageBase));
  const chunkBases = new Uint32Array(ranges.map(({ chunkBase }) => chunkBase));
  const counts = new Uint32Array(ranges.map(({ count }) => count));
  const lodTreeData = ranges.map(({ lodTreeData }) => lodTreeData);
  const lodErrorData = ranges.some(({ lodErrorData }) => lodErrorData)
    ? ranges.map(({ lodErrorData }) => lodErrorData)
    : undefined;
  const lodOcclusionData = ranges.some(
    ({ lodOcclusionData }) => lodOcclusionData,
  )
    ? ranges.map(({ lodOcclusionData }) => lodOcclusionData)
    : undefined;

  const result = update_lod_trees(
    lodIds,
    pageBases,
    chunkBases,
    counts,
    lodTreeData,
    lodErrorData,
    lodOcclusionData,
  );
}

/**
 * Pages of a compact tree (thread CQ): each its traversal input (position
 * planes then the traversal's, 22 bytes a node; undefined evicts the page)
 * and its box.
 */
function updateCompactLodTrees({
  ranges,
}: {
  ranges: {
    lodId: number;
    pageBase: number;
    chunkBase: number;
    count: number;
    input?: Uint8Array;
    box?: ArrayLike<number>;
  }[];
}) {
  const boxes = new Float32Array(6 * ranges.length);
  ranges.forEach((r, k) => {
    if (r.box) boxes.set(Array.from(r.box).slice(0, 6), 6 * k);
  });
  update_compact_lod_trees(
    new Uint32Array(ranges.map((r) => r.lodId)),
    new Uint32Array(ranges.map((r) => r.pageBase)),
    new Uint32Array(ranges.map((r) => r.chunkBase)),
    new Uint32Array(ranges.map((r) => r.count)),
    ranges.map((r) => r.input) as unknown as never,
    boxes,
  );
}

function traverseLodTrees({
  maxSplats,
  pixelScaleLimit,
  lastPixelLimit,
  instances,
  traverseMode,
  errorLimit,
  backCullMin,
  occlusionCull,
  errorSizeK,
  neighbours,
  unordered,
  stream,
}: {
  maxSplats: number;
  pixelScaleLimit: number;
  lastPixelLimit?: number;
  /** The error cut: tau x the size of a pixel (athc_lod_cut.rs). */
  errorLimit?: number;
  /**
   * Leave out the cut's back-facing nodes wider than this (a pixel's angle
   * x lodBackCullMinPx); undefined: no culling.
   */
  backCullMin?: number;
  /** Leave out what athenea's occlusion masks hide from the eye. */
  occlusionCull?: boolean;
  /** The combined cut: P = max(P_err, P_size / k) (lod_tree.rs). */
  errorSizeK?: number;
  /**
   * Each cut node's parent, first child and children (resident), and the
   * culled nodes, per instance (the node pool, thread CT).
   */
  neighbours?: boolean;
  /**
   * A stack, not the heap (thread CU): the same cut while maxSplats does
   * not bind (else the traversal runs again in order), about 2.4x faster.
   */
  unordered?: boolean;
  /**
   * The cut in chunks (`parts`, any instance; `culled` apart), never whole
   * in the WASM heap (the node pool's worker, thread CU).
   */
  stream?: boolean;
  instances: Record<
    string,
    {
      instanceId: string;
      lodId: number;
      rootPage?: number;
      viewToObjectCols: number[];
      lodScale: number;
      behindFoveate: number;
      coneFov0: number;
      coneFov: number;
      coneFoveate: number;
      /** [tanX, tanY, outside weight] (thread CU round 11; spark-rs Frustum). */
      frustum?: number[];
    }
  >;
  traverseMode: "dynamic" | "standard";
}) {
  const keyInstances = Object.entries(instances);
  const lodIds = new Uint32Array(
    keyInstances.map(([_key, instance]) => instance.lodId),
  );
  const rootPages = new Uint32Array(
    keyInstances.map(([_key, instance]) => instance.rootPage ?? 0xffffffff),
  );
  const viewToObjects = new Float32Array(
    keyInstances.flatMap(([_key, instance]) => {
      if (instance.viewToObjectCols.length !== 16) {
        throw new Error("Incorrect array size for viewToObjectCols");
      }
      return instance.viewToObjectCols;
    }),
  );
  const lodScales = new Float32Array(
    keyInstances.map(([_key, instance]) => instance.lodScale),
  );
  const behindFoveates = new Float32Array(
    keyInstances.map(([_key, instance]) => instance.behindFoveate),
  );
  const coneFov0s = new Float32Array(
    keyInstances.map(([_key, instance]) => instance.coneFov0),
  );
  const coneFovs = new Float32Array(
    keyInstances.map(([_key, instance]) => instance.coneFov),
  );
  const coneFoveates = new Float32Array(
    keyInstances.map(([_key, instance]) => instance.coneFoveate),
  );

  // The view's frustum by instance (none: the cone's foveation only).
  const frustums = keyInstances.some(([_key, i]) => i.frustum)
    ? new Float32Array(
        keyInstances.flatMap(([_key, i]) => i.frustum ?? [0, 0, 1]),
      )
    : undefined;

  const lodFunction =
    traverseMode === "dynamic"
      ? dynamic_traverse_lod_trees
      : traverse_lod_trees;
  const wasm0 = performance.now();
  const result = lodFunction(
    maxSplats,
    pixelScaleLimit,
    lastPixelLimit,
    lodIds,
    rootPages,
    viewToObjects,
    lodScales,
    behindFoveates,
    coneFoveates,
    coneFov0s,
    coneFovs,
    errorLimit,
    backCullMin,
    occlusionCull,
    errorSizeK,
    neighbours,
    unordered,
    stream,
    frustums,
  ) as {
    culledCount?: number;
    occludedCount?: number;
    instanceIndices: {
      lodId: number;
      numSplats: number;
      indices: Uint32Array;
    }[];
    chunks: [number, number][];
    pixelLimit?: number;
  };
  const { instanceIndices, chunks, pixelLimit, culledCount, occludedCount } =
    result;
  const { parts, culled } = result as unknown as {
    parts?: unknown[];
    culled?: unknown[];
  };
  const wasmMs = performance.now() - wasm0;

  const indices = keyInstances.reduce(
    (indices, [key, _instance], index) => {
      indices[key] = instanceIndices[index];
      return indices;
    },
    {} as Record<
      string,
      { lodId: number; numSplats: number; indices: Uint32Array }
    >,
  );
  // console.log(`traverseLodTrees: instanceIndices=${instanceIndices.length}`);
  // console.log(`traverseLodTrees: chunks=${chunks.length}`, JSON.stringify(chunks));
  return {
    keyIndices: indices,
    chunks,
    pixelLimit,
    culledCount,
    occludedCount,
    wasmMs,
    parts,
    culled,
  };
}

function getLodTreeLevel({
  lodId,
  level,
}: {
  lodId: number;
  level: number;
}) {
  return get_lod_tree_level(lodId, level) as { indices: Uint32Array };
}

// A .athc's headers and tables (src/athc.ts readAthcLayout): `need`, the
// bytes they take, and the layout once the prefix holds them (else null).
async function athcLayout({
  prefix,
  fileBytes,
}: {
  prefix: Uint8Array;
  fileBytes: number;
}): Promise<{ need: number; layout: unknown }> {
  const need = athc_prefix_bytes(prefix.subarray(0, 4096));
  if (need > prefix.length) return { need, layout: null };
  return { need, layout: athc_layout(prefix, fileBytes) };
}

// A .athc's merged pages (src/athc.ts), from its bytes through its levels.
async function athcMergedPages({
  prefix,
  fileBytes,
}: {
  prefix: Uint8Array;
  fileBytes: number;
}) {
  const { pages, ...tree } = athc_merged_pages(prefix, fileBytes) as {
    pages: Uint8Array[];
  } & Record<string, unknown>;
  return { tree, pages };
}

// A .athc v3's merged pages (src/athc.ts), from its tables and its levels'
// blocks as kind-2 ATHV pages.
async function athc3MergedPages({
  tables,
  levels,
}: {
  tables: Uint8Array;
  levels: Uint8Array[];
}) {
  const { pages, headers, ...tree } = athc3_merged_pages(tables, levels) as {
    pages: Uint8Array[];
    headers: Uint8Array;
  } & Record<string, unknown>;
  return { tree, pages, headers };
}

let nextChunkWaiter = (_chunk: Uint8Array) => {};

async function nextChunk({ chunk }: { chunk: Uint8Array }) {
  nextChunkWaiter(chunk);
}

// Recursively finds all ArrayBuffers in an object and returns them as an array
// to use as transferable objects to send between workers.
function getTransferable(ctx: unknown): Transferable[] {
  const buffers: Transferable[] = [];
  const seen = new Set();

  function traverse(obj: unknown) {
    if (obj && typeof obj === "object" && !seen.has(obj)) {
      seen.add(obj);

      if (obj instanceof ArrayBuffer) {
        buffers.push(obj);
      } else if (ArrayBuffer.isView(obj)) {
        // Handles TypedArrays and DataView
        buffers.push(obj.buffer as ArrayBuffer);
      } else if (Array.isArray(obj)) {
        obj.forEach(traverse);
      } else {
        Object.values(obj).forEach(traverse);
      }
    }
  }

  traverse(ctx);
  return buffers;
}

async function initialize() {
  let resolveWaitForModule: (value: WebAssembly.Module) => void;
  const waitForModule = new Promise<WebAssembly.Module>((resolve) => {
    resolveWaitForModule = resolve;
  });

  // Hold any messages received while initializing
  const pending: MessageEvent[] = [];
  const bufferMessage = (event: MessageEvent) => {
    // Handle module
    if (event.data.name === "init-wasm") {
      resolveWaitForModule(event.data.module as WebAssembly.Module);
      return;
    }

    pending.push(event);
  };
  self.addEventListener("message", bufferMessage);

  wasmExports = await init_wasm({ module_or_path: await waitForModule });

  self.removeEventListener("message", bufferMessage);
  self.addEventListener("message", onMessage);

  // Process any buffered messages
  for (const event of pending) {
    onMessage(event);
  }
  pending.length = 0;
}

initialize().catch(console.error);
