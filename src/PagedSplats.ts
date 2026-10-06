import * as THREE from "three";

import { decode_rad_header } from "spark-rs";
import { getSplatFileType, getSplatFileTypeFromPath } from "./SplatLoader";
import type { SplatSource } from "./SplatMesh";
import type { SplatPager } from "./SplatPager";
import { workerPool } from "./SplatWorker";
import {
  ATHC_FLAGS,
  ATHC_PAGE_SPLATS,
  type AnyAthcLayout,
  type AthcPaging,
  athcPageCount,
  athcSplatBase,
  athcWantOf,
  fetchAthc3Page,
  fetchAthcPage,
  fetchAthcStreams,
  isAthcV3,
  openAthc,
  readAthcLayout,
} from "./athc";
import {
  DEFAULT_SPLAT_ENCODING,
  LN_SCALE_MAX,
  LN_SCALE_MIN,
  type RadMeta,
  type SplatEncoding,
  SplatFileType,
} from "./defines";
import * as dyno from "./dyno";
import { decodeExtSplat, unpackSplat, uploadU32DataTextureRows } from "./utils";
import * as wasm from "./wasm";
import { AttribPool, type AttributeSpec } from "./webgpu/attributes/schema";

/**
 * What a pager that pages attributes by group (WgpuSplatPager) tells the
 * splats it fetches: the attributes it keeps, and those a page's fetch
 * should bring with it.
 */
interface StreamPager {
  attributePlan?(): { specs: AttributeSpec[] } | null;
  streamsToFetch?(splats: PagedSplats, chunk: number): string[];
}

export interface PagedSplatsOptions {
  pager?: SplatPager;
  rootUrl?: string;
  requestHeader?: Record<string, string>;
  withCredentials?: boolean;
  fileBytes?: Uint8Array;
  fileType?: SplatFileType;
  maxSh?: number;
  /**
   * A .athc flagged linear keeps its colours linear rather than encoded to
   * sRGB (src/athc.ts AthcPaging.keepLinear): for the WebGPU athenea raster,
   * which blends in linear light (atheneaRasterPlugin, storedLinear).
   */
  athcKeepLinear?: boolean;
}

/** A decoded page with stream columns fetched beside it (a .athc v3 page). */
function withStreams<
  T extends { numSplats: number; extra: { attribs?: object } },
>(result: T, streams: AttribPool | null): T {
  if (!streams?.columns.length) return result;
  const attribs = result.extra.attribs
    ? AttribPool.from(
        result.extra.attribs as Parameters<typeof AttribPool.from>[0],
      )
    : new AttribPool(result.numSplats);
  for (const col of streams.columns) attribs.setColumn(col);
  return { ...result, extra: { ...result.extra, attribs } };
}

export class PagedSplats implements SplatSource {
  pager?: SplatPager;
  rootUrl: string;
  requestHeader?: Record<string, string>;
  withCredentials?: boolean;
  fileBytes?: Uint8Array;
  fileType?: SplatFileType;
  athcKeepLinear: boolean;
  /** Once a .athc is open: its colours are linear light as decoded. */
  athcStoredLinear?: boolean;

  numSh: number;
  maxSh: number;
  sh1Codes?: Uint32Array;
  sh2Codes?: Uint32Array;
  sh3Codes?: Uint32Array | [Uint32Array, Uint32Array];

  numSplats: number;
  splatEncoding?: SplatEncoding;
  radMetaPromise?: Promise<{ meta: RadMeta; chunksStart: number }>;
  /** A .athc's virtual tree and merged pages (src/athc.ts). */
  athcPromise?: Promise<AthcPaging>;
  /** A .athc's headers and tables, read before its merged pages. */
  athcLayoutPromise?: Promise<{ layout: AnyAthcLayout; prefix: Uint8Array }>;

  dynoNumSplats: dyno.DynoInt<"numSplats">;
  dynoIndices: dyno.DynoUsampler2D<"indices", THREE.DataTexture>;
  rgbMinMaxLnScaleMinMax: dyno.DynoVec4<
    THREE.Vector4,
    "rgbMinMaxLnScaleMinMax"
  >;
  lodOpacity: dyno.DynoBool<"lodOpacity">;
  dynoNumSh: dyno.DynoInt<"numSh">;
  shMax: dyno.DynoVec3<THREE.Vector3, "shMax">;

  readonly abortController: AbortController = new AbortController();

  constructor(options: PagedSplatsOptions) {
    this.pager = options.pager;
    this.rootUrl = options.rootUrl ?? "";
    this.requestHeader = options.requestHeader;
    this.withCredentials = options.withCredentials;
    this.athcKeepLinear = options.athcKeepLinear ?? false;
    this.numSh = 0;
    this.maxSh = options.maxSh ?? 3;

    this.numSplats = 0;

    this.dynoNumSplats = new dyno.DynoInt({ value: 0 });
    this.dynoIndices = new dyno.DynoUsampler2D({
      value: PagedSplats.emptyIndicesTexture,
    });

    this.rgbMinMaxLnScaleMinMax = new dyno.DynoVec4({
      value: new THREE.Vector4(0.0, 1.0, LN_SCALE_MIN, LN_SCALE_MAX),
    });
    this.lodOpacity = new dyno.DynoBool({
      value: false,
    });

    this.dynoNumSh = new dyno.DynoInt({ value: 0 });
    this.shMax = new dyno.DynoVec3({ value: new THREE.Vector3() });

    this.fileBytes = options.fileBytes;
    this.fileType = options.fileType;
    if (!this.fileType && this.fileBytes) {
      this.fileType = getSplatFileType(this.fileBytes);
    }
    if (!this.fileType && this.rootUrl) {
      this.fileType = getSplatFileTypeFromPath(this.rootUrl);
    }
    if (!this.fileType) {
      throw new Error("Unable to determine file type");
    }
    if (this.fileType === SplatFileType.RAD) {
      this.radMetaPromise = this.getRadMeta();
    }
    if (this.fileType === SplatFileType.ATHC) {
      this.athcLayoutPromise = this.getAthcLayout();
    }
  }

  private fetchOptions() {
    return {
      url: this.rootUrl,
      fileBytes: this.fileBytes,
      requestHeader: this.requestHeader,
      withCredentials: this.withCredentials,
      signal: this.abortController.signal,
    };
  }

  getAthcLayout() {
    this.athcLayoutPromise ??= readAthcLayout(this.fetchOptions()).then(
      (read) => {
        this.athcStoredLinear =
          this.athcKeepLinear &&
          (read.layout.header.flags & ATHC_FLAGS.linear) !== 0;
        return read;
      },
    );
    return this.athcLayoutPromise;
  }

  /** The attributes the pager keeps (null: every stream of the file). */
  private keptSpecs(): AttributeSpec[] | null {
    return (
      (this.pager as StreamPager | undefined)?.attributePlan?.()?.specs ?? null
    );
  }

  /**
   * Opens a .athc for paging. A v3 file's levels are read with the streams
   * the pager keeps: open it once the pager knows them (its schema is set
   * from getRadMeta, before the first fetch).
   */
  getAthc(): Promise<AthcPaging> {
    this.athcPromise ??= this.getAthcLayout()
      .then((read) =>
        openAthc({
          ...this.fetchOptions(),
          read,
          keepLinear: this.athcKeepLinear,
          want: (layout) => {
            const kept = this.keptSpecs();
            return kept
              ? athcWantOf(layout, kept)
              : { material: true, transferValues: layout.extra.transferCount };
          },
        }),
      )
      .then((paging) => {
        this.athcStoredLinear =
          paging.keepLinear &&
          (paging.layout.header.flags & ATHC_FLAGS.linear) !== 0;
        return paging;
      });
    return this.athcPromise;
  }

  dispose() {
    if (this.dynoIndices.value !== PagedSplats.emptyIndicesTexture) {
      this.dynoIndices.value.dispose();
      this.dynoIndices.value = PagedSplats.emptyIndicesTexture;
    }

    this.abortController.abort();
  }

  setMaxSh(maxSh: number) {
    this.maxSh = maxSh;
  }

  getRadMeta(): Promise<{ meta: RadMeta; chunksStart: number }> {
    if (this.radMetaPromise) {
      return this.radMetaPromise;
    }
    if (this.fileType === SplatFileType.ATHC) {
      // A .athc stands in as a .rad of virtual pages (src/athc.ts): what
      // the WebGPU pager reads of a meta is its attribute schema.
      this.radMetaPromise = this.getAthcLayout().then(({ layout }) => {
        const pageCount = athcPageCount(layout);
        return {
          meta: {
            version: 0,
            type: "athc",
            count: athcSplatBase(layout).splatBase + layout.header.count,
            lodTree: true,
            chunkSize: 65536,
            chunks: Array.from({ length: pageCount }, () => ({
              offset: 0,
              bytes: 0,
            })),
            attributes: layout.attribSpecs,
          } as RadMeta,
          chunksStart: 0,
        };
      });
      return this.radMetaPromise;
    }

    this.radMetaPromise = (async () => {
      await wasm.initialization;

      if (this.fileBytes) {
        // Shouldn't be more than 1 MB, so don't send more data than that.
        const metaStart = decode_rad_header(this.fileBytes.slice(0, 1048576));
        if (metaStart) {
          return metaStart;
        }
        throw new Error("Failed to decode RAD header");
      }
      if (!this.rootUrl) {
        throw new Error("No url or fileBytes provided");
      }

      // We don't know how big the header will be. Most likely 64KB will be enough,
      // but try larger blocks in backoff if it wasn't enough.
      for (const tryBytes of [65536, 256 * 1024, 1024 * 1024]) {
        const bytes = await fetchRange({
          url: this.rootUrl,
          requestHeader: this.requestHeader,
          withCredentials: this.withCredentials,
          offset: 0,
          bytes: tryBytes,
        });
        const metaStart = decode_rad_header(bytes);
        if (metaStart) {
          return metaStart;
        }
      }
      throw new Error("Failed to decode RAD header");
    })().then((metaStart) => {
      // console.log("RAD meta: ", metaStart.meta);
      return metaStart;
    });

    this.radMetaPromise.catch((error) => {
      console.error(error);
      // Allow it to be tried again
      // this.radMetaPromise = undefined;
    });

    return this.radMetaPromise;
  }

  chunkUrl(chunk: number): string {
    return this.rootUrl.replace(/-lod-0\./, `-lod-${chunk}.`);
  }

  /**
   * The attributes `names` of `chunk` alone, for a pager that pages them by
   * group (AttribResidency): one Range request of a .athc chunk's streams,
   * else the chunk fetched and decoded again.
   */
  async fetchStreams(
    chunk: number,
    names: readonly string[],
  ): Promise<AttribPool | null> {
    const specs = (this.keptSpecs() ?? []).filter((s) =>
      names.includes(s.name),
    );
    if (this.fileType === SplatFileType.ATHC) {
      const paging = await this.getAthc();
      if (chunk >= paging.tree.splatBase / ATHC_PAGE_SPLATS) {
        return fetchAthcStreams(paging, chunk, this.fetchOptions(), specs);
      }
    }
    const data = await this.fetchDecodeChunk(chunk);
    const attribs = data.extra.attribs;
    return attribs
      ? AttribPool.from(attribs as Parameters<typeof AttribPool.from>[0])
      : null;
  }

  async fetchDecodeChunk(chunk: number) {
    let decodeBytes = undefined;
    let streams: AttribPool | null = null;

    if (this.fileType === SplatFileType.ATHC) {
      const paging = await this.getAthc();
      if (
        isAthcV3(paging.layout) &&
        chunk >= paging.tree.splatBase / ATHC_PAGE_SPLATS
      ) {
        // The splats and the streams the pager wants with them: one Range.
        const pager = this.pager as StreamPager | undefined;
        const names = pager?.streamsToFetch?.(this, chunk);
        const kept = this.keptSpecs();
        const specs = kept
          ? kept.filter((s) => names?.includes(s.name) ?? true)
          : [];
        const page = await fetchAthc3Page(
          paging,
          chunk,
          this.fetchOptions(),
          specs,
        );
        decodeBytes = page.page;
        streams = page.streams;
      } else {
        // One ATHV page: merged nodes, or an athenea v2 chunk read whole.
        decodeBytes = await fetchAthcPage(paging, chunk, this.fetchOptions());
      }
    } else if (this.fileType === SplatFileType.RAD) {
      const { meta, chunksStart } = await this.getRadMeta();
      if (chunk < 0 || chunk >= meta.chunks.length) {
        throw new Error(
          `Chunk index out of range: ${chunk} (max: ${meta.chunks.length - 1})`,
        );
      }
      let { offset, bytes, filename } = meta.chunks[chunk];

      if (filename) {
        if (this.fileBytes) {
          throw new Error("Chunked RAD file not supported with fileBytes");
        }
        const resolvedRoot = new URL(
          this.rootUrl,
          window.location.href,
        ).toString();
        const chunkUrl = new URL(filename, resolvedRoot).toString();
        decodeBytes = await fetchRange({
          url: chunkUrl,
          requestHeader: this.requestHeader,
          withCredentials: this.withCredentials,
          signal: this.abortController.signal,
        });
      } else {
        offset += chunksStart;
        // console.log(`Fetching chunk ${chunk} at offset ${offset} with bytes ${bytes}`);
        if (this.fileBytes) {
          if (offset < 0 || offset + bytes > this.fileBytes.length) {
            throw new Error(
              `Invalid chunk offset or bytes: ${offset} + ${bytes} > ${this.fileBytes.length}`,
            );
          }
          decodeBytes = this.fileBytes.slice(offset, offset + bytes);
        } else if (this.rootUrl) {
          decodeBytes = await fetchRange({
            url: this.rootUrl,
            requestHeader: this.requestHeader,
            withCredentials: this.withCredentials,
            offset,
            bytes,
            signal: this.abortController.signal,
          });
        } else {
          throw new Error("No url or fileBytes provided");
        }
      }
    } else if (this.fileBytes) {
      // Fall through
    } else if (this.rootUrl) {
      const url = this.chunkUrl(chunk);
      const request = new Request(url, {
        headers: this.requestHeader
          ? new Headers(this.requestHeader)
          : undefined,
        credentials: this.withCredentials ? "include" : "same-origin",
        signal: this.abortController.signal,
      });
      const response = await fetch(request);
      if (!response.ok || !response.body) {
        throw new Error(
          `Failed to fetch "${url}": ${response.status} ${response.statusText}`,
        );
      }
      decodeBytes = new Uint8Array(await response.arrayBuffer());
    } else {
      throw new Error("No url or fileBytes provided");
    }

    return await workerPool.withWorker(async (worker) => {
      if (!this.pager) {
        throw new Error("PagedSplats.pager not set");
      }
      if (!this.pager.extSplats) {
        const result = await worker.call("loadPackedSplats", {
          fileBytes: decodeBytes,
          pathName: this.chunkUrl(chunk),
          sh1Codes: this.sh1Codes?.slice(),
          sh2Codes: this.sh2Codes?.slice(),
          sh3Codes: (this.sh3Codes as Uint32Array | undefined)?.slice(),
        });
        if (!("lodSplats" in result)) {
          throw new Error("Loaded chunk does not contain LoD splats");
        }

        const lodSplats = result.lodSplats;
        if (!this.splatEncoding) {
          this.splatEncoding = lodSplats.splatEncoding;

          this.numSh = lodSplats.extra.sh3
            ? 3
            : lodSplats.extra.sh2
              ? 2
              : lodSplats.extra.sh1
                ? 1
                : 0;

          this.rgbMinMaxLnScaleMinMax.value.set(
            this.splatEncoding.rgbMin,
            this.splatEncoding.rgbMax,
            this.splatEncoding.lnScaleMin,
            this.splatEncoding.lnScaleMax,
          );

          this.lodOpacity.value = this.splatEncoding.lodOpacity;

          this.shMax.value.set(
            this.splatEncoding.sh1Max,
            this.splatEncoding.sh2Max,
            this.splatEncoding.sh3Max,
          );
        }
        this.sh1Codes = lodSplats.extra.sh1Codes ?? this.sh1Codes;
        this.sh2Codes = lodSplats.extra.sh2Codes ?? this.sh2Codes;
        this.sh3Codes = lodSplats.extra.sh3Codes ?? this.sh3Codes;
        return withStreams(lodSplats, streams);
      }

      const sh3Codes = this.sh3Codes as [Uint32Array, Uint32Array] | undefined;
      const result = await worker.call("loadExtSplats", {
        fileBytes: decodeBytes,
        pathName: this.chunkUrl(chunk),
        sh1Codes: this.sh1Codes?.slice(),
        sh2Codes: this.sh2Codes?.slice(),
        sh3Codes: sh3Codes
          ? [sh3Codes[0].slice(), sh3Codes[1].slice()]
          : undefined,
      });
      if (!("lodSplats" in result)) {
        throw new Error("Loaded chunk does not contain LoD splats");
      }
      const lodSplats = result.lodSplats;
      if (!this.splatEncoding) {
        this.splatEncoding = DEFAULT_SPLAT_ENCODING;
        this.numSh =
          lodSplats.extra.sh3a && lodSplats.extra.sh3b
            ? 3
            : lodSplats.extra.sh2
              ? 2
              : lodSplats.extra.sh1
                ? 1
                : 0;
      }
      this.sh1Codes = lodSplats.extra.sh1Codes ?? this.sh1Codes;
      this.sh2Codes = lodSplats.extra.sh2Codes ?? this.sh2Codes;
      this.sh3Codes = lodSplats.extra.sh3Codes ?? this.sh3Codes;
      return withStreams(lodSplats, streams);
    });
  }

  /** Draw nothing until the next update(), keeping the indices texture allocated. */
  clear() {
    this.numSplats = 0;
    this.dynoNumSplats.value = 0;
  }

  update(numSplats: number, indices: Uint32Array) {
    if (!this.pager) {
      throw new Error("PagedSplats.pager not set");
    }

    const renderer = this.pager.renderer;
    this.numSplats = numSplats;
    this.dynoNumSplats.value = this.numSplats;
    const rows = Math.ceil(numSplats / 16384);

    let indicesTexture =
      this.dynoIndices.value === PagedSplats.emptyIndicesTexture
        ? undefined
        : this.dynoIndices.value;
    if (indicesTexture && rows > indicesTexture.image.height) {
      indicesTexture.dispose();
      indicesTexture = undefined;
    }

    if (!indicesTexture) {
      indicesTexture = new THREE.DataTexture(
        indices,
        4096,
        rows,
        THREE.RGBAIntegerFormat,
        THREE.UnsignedIntType,
      );
      indicesTexture.internalFormat = "RGBA32UI";
      indicesTexture.needsUpdate = true;
      renderer.initTexture(indicesTexture);
      this.dynoIndices.value = indicesTexture;
    } else {
      const textureIndices = indicesTexture.image.data as Uint32Array;
      textureIndices.set(indices.subarray(0, numSplats));

      uploadU32DataTextureRows(
        renderer,
        indicesTexture,
        4096,
        rows,
        textureIndices,
      );
    }
  }

  prepareFetchSplat() {}

  getNumSplats(): number {
    return this.numSplats;
  }

  hasRgbDir(): boolean {
    if (!this.pager) {
      return false;
    }
    return Math.min(this.numSh, this.maxSh, this.pager.maxSh) > 0;
  }

  getNumSh(): number {
    return this.numSh;
  }

  fetchSplat({
    index,
    viewOrigin,
  }: {
    index: dyno.DynoVal<"int">;
    viewOrigin?: dyno.DynoVal<"vec3">;
  }): dyno.DynoVal<typeof dyno.Gsplat> {
    if (!this.pager) {
      throw new Error("PagedSplats.pager not set");
    }

    const splatIndex = this.pager.readIndex.apply({
      index,
      numSplats: this.dynoNumSplats,
      indices: this.dynoIndices,
    }).index;

    if (!this.pager.extSplats) {
      if (this.hasRgbDir() && viewOrigin) {
        this.dynoNumSh.value = Math.min(
          this.numSh,
          this.maxSh,
          this.pager.maxSh,
        );
        return this.pager.readSplatDir.apply({
          index: splatIndex,
          rgbMinMaxLnScaleMinMax: this.rgbMinMaxLnScaleMinMax,
          lodOpacity: this.lodOpacity,
          viewOrigin,
          numSh: this.dynoNumSh,
          shMax: this.shMax,
        }).gsplat;
      }
      return this.pager.readSplat.apply({
        index: splatIndex,
        rgbMinMaxLnScaleMinMax: this.rgbMinMaxLnScaleMinMax,
        lodOpacity: this.lodOpacity,
      }).gsplat;
    }

    if (this.hasRgbDir() && viewOrigin) {
      this.dynoNumSh.value = Math.min(this.numSh, this.maxSh, this.pager.maxSh);
      return this.pager.readSplatExtDir.apply({
        index: splatIndex,
        viewOrigin,
        numSh: this.dynoNumSh,
      }).gsplat;
    }
    return this.pager.readSplatExt.apply({ index: splatIndex }).gsplat;
  }

  // Iterate over Gsplats index 0..=(this.numSplats-1), unpack each Gsplat
  // and invoke the callback function with the Gsplat attributes.
  forEachSplat(
    callback: (
      index: number,
      center: THREE.Vector3,
      scales: THREE.Vector3,
      quaternion: THREE.Quaternion,
      opacity: number,
      color: THREE.Color,
    ) => void,
  ) {
    if (!this.pager || !this.numSplats) {
      return;
    }
    const extSplats = this.pager.extSplats;
    const indices = this.dynoIndices.value.image.data as Uint32Array;
    const packedSplatArray = this.pager.packedTexture.value.image
      .data as Uint32Array;
    const extPackedSplatArray = this.pager.extTexture.value.image
      .data as Uint32Array;
    const extArrays: [Uint32Array, Uint32Array] = [
      packedSplatArray,
      extPackedSplatArray,
    ];

    for (let i = 0; i < this.numSplats; ++i) {
      const splatIndex = indices[i];
      const unpacked = extSplats
        ? decodeExtSplat(extArrays, splatIndex)
        : unpackSplat(packedSplatArray, splatIndex, this.splatEncoding);
      callback(
        i,
        unpacked.center,
        unpacked.scales,
        unpacked.quaternion,
        unpacked.opacity,
        unpacked.color,
      );
    }
  }

  static emptyIndicesTexture = (() => {
    const emptyArray = new Uint32Array(4096 * 4);
    const texture = new THREE.DataTexture(emptyArray, 4096, 1);
    texture.format = THREE.RGBAIntegerFormat;
    texture.type = THREE.UnsignedIntType;
    texture.internalFormat = "RGBA32UI";
    texture.needsUpdate = true;
    return texture;
  })();
}

async function fetchRange({
  url,
  requestHeader,
  withCredentials,
  offset,
  bytes,
  signal,
}: {
  url: string;
  requestHeader?: Record<string, string>;
  withCredentials?: boolean;
  offset?: number;
  bytes?: number;
  signal?: AbortSignal;
}): Promise<Uint8Array> {
  const request = new Request(url, {
    headers: requestHeader ? new Headers(requestHeader) : undefined,
    credentials: withCredentials ? "include" : "same-origin",
    signal,
  });
  if (offset !== undefined && bytes !== undefined) {
    request.headers.set("Range", `bytes=${offset}-${offset + bytes - 1}`);
  }
  const response = await fetch(request);
  if (!response.ok || !response.body) {
    throw new Error(
      `Failed to fetch "${url}": ${response.status} ${response.statusText}`,
    );
  }
  return new Uint8Array(await response.arrayBuffer());
}
