// Per-Gaussian attributes of any schema, decoded at run time so one compiled
// kernel serves every schema (src/webgpu/attributes/schema.ts packs them).
//
// An attribute pool is one uint buffer that carries its own descriptor table:
//   [0] strideWords   [1] numAttribs   [2] headerWords   [3] count (splats)
//   [4 + 4k ..]       attribute k: offsetWords, format, components, flags
//   [headerWords + splat * strideWords + offsetWords ..]   attribute k data
// Components are packed little-endian at their format's size, an attribute
// starting on a word boundary, so a group of four components (comp4) is 1, 2
// or 4 whole words. Attribute ids are resolved by name on the host.
//
// A paged attribute (flags ATTRIB_PAGED; src/webgpu/attributes/
// PagedAttribPool.ts) is not interleaved: it has its own region of fewer
// pages than the pool's splats, at offsetWords (from the buffer's start):
//   [offsetWords ..]                     a page table, one word per page of
//                                        65 536 splats: its slot, or
//                                        ATTRIB_NONE when not resident
//   [offsetWords + pages + (slot * 65536 + splat % 65536) * words ..]
// where pages = ceil(count / 65536) and words = attribWords(desc). A splat
// whose page is not resident reads as zeros (readAttrib, readAttribU) and
// attribResident says so: readers fall back on their own default.

module attrib;

import core.math;

public static const uint ATTRIB_F32 = 0u;
public static const uint ATTRIB_F16 = 1u;
public static const uint ATTRIB_UNORM8 = 2u;
public static const uint ATTRIB_SNORM8 = 3u;
public static const uint ATTRIB_U8 = 4u;
public static const uint ATTRIB_U16 = 5u;
public static const uint ATTRIB_U32 = 6u;

public static const uint ATTRIB_NONE = 0xffffffffu;
// Flags: a 3-vector that turns with its object (normals, tangents).
public static const uint ATTRIB_DIRECTION = 1u;
// Flags: stored in its own paged region (see above), possibly not resident.
public static const uint ATTRIB_PAGED = 2u;
public static const uint ATTRIB_PAGE_SHIFT = 16u;

public struct AttribDesc {
    public uint offsetWords;
    public uint format;
    public uint components;
    public uint flags;
};

public uint attribFormatBytes(uint format) {
    switch (format) {
    case ATTRIB_F32:
    case ATTRIB_U32:
        return 4u;
    case ATTRIB_F16:
    case ATTRIB_U16:
        return 2u;
    default:
        return 1u;
    }
}

public uint attribWords(AttribDesc d) {
    return (d.components * attribFormatBytes(d.format) + 3u) / 4u;
}

// Four components from the words of one comp4 group; n of them are valid.
public float4 decodeAttrib4(uint format, uint4 w, uint n) {
    float4 r;
    uint4 bytes = (uint4(w.x) >> uint4(0u, 8u, 16u, 24u)) & 0xffu;
    switch (format) {
    case ATTRIB_F32:
        r = asfloat(w);
        break;
    case ATTRIB_F16:
        r = float4(unpackHalf2x16(w.x), unpackHalf2x16(w.y));
        break;
    case ATTRIB_UNORM8:
        r = float4(bytes) / 255.0;
        break;
    case ATTRIB_SNORM8: {
        int4 s = int4(uint4(w.x) << uint4(24u, 16u, 8u, 0u)) >> 24;
        r = max(float4(s) / 127.0, float4(-1.0));
        break;
    }
    case ATTRIB_U8:
        r = float4(bytes);
        break;
    case ATTRIB_U16:
        r = float4(float(w.x & 0xffffu), float(w.x >> 16u), float(w.y & 0xffffu), float(w.y >> 16u));
        break;
    default:
        r = float4(w);
        break;
    }
    return select(uint4(0u, 1u, 2u, 3u) < uint4(n), r, float4(0.0));
}

// Inverse of decodeAttrib4: the words of one comp4 group (1, 2 or 4 used).
public uint4 encodeAttrib4(uint format, float4 v) {
    switch (format) {
    case ATTRIB_F32:
        return asuint(v);
    case ATTRIB_F16:
        return uint4(packHalf2x16(v.xy), packHalf2x16(v.zw), 0u, 0u);
    case ATTRIB_UNORM8:
    case ATTRIB_SNORM8:
    case ATTRIB_U8: {
        uint4 b;
        if (format == ATTRIB_UNORM8) {
            b = uint4(round(saturate(v) * 255.0));
        } else if (format == ATTRIB_SNORM8) {
            b = uint4(int4(round(clamp(v, -1.0, 1.0) * 127.0))) & 0xffu;
        } else {
            b = uint4(clamp(round(v), 0.0, 255.0));
        }
        return uint4(b.x | (b.y << 8u) | (b.z << 16u) | (b.w << 24u), 0u, 0u, 0u);
    }
    case ATTRIB_U16: {
        uint4 h = uint4(clamp(round(v), 0.0, 65535.0));
        return uint4(h.x | (h.y << 16u), h.z | (h.w << 16u), 0u, 0u);
    }
    default:
        return uint4(max(round(v), 0.0));
    }
}

// Word span of comp4 group g: first word relative to the attribute, count.
public uint2 attribGroupWords(AttribDesc d, uint g) {
    uint per = attribFormatBytes(d.format); // words per 4 components
    uint first = g * per;
    uint total = attribWords(d);
    return uint2(first, first < total ? min(per, total - first) : 0u);
}

// The n (1 to 4) words of a comp4 group at word `base`. Unrolled: a loop
// storing w[k] would make the vector a private array.
public uint4 readAttribWords(StructuredBuffer<uint> pool, uint base, uint n) {
    uint4 w = uint4(pool[base], 0u, 0u, 0u);
    if (n > 1u) {
        w.y = pool[base + 1u];
    }
    if (n > 2u) {
        w.z = pool[base + 2u];
    }
    if (n > 3u) {
        w.w = pool[base + 3u];
    }
    return w;
}

// The pool accessors, for read-only and read-write bindings alike.

public uint attribCount(StructuredBuffer<uint> pool) { return pool[1]; }
public uint attribSplats(StructuredBuffer<uint> pool) { return pool[3]; }

public AttribDesc attribDesc(StructuredBuffer<uint> pool, uint id) {
    uint b = 4u + 4u * id;
    return { pool[b], pool[b + 1u], pool[b + 2u], pool[b + 3u] };
}

// The first word of `splat`'s record of the attribute, or ATTRIB_NONE
// when it is paged and its page is not resident.
public uint attribBase(StructuredBuffer<uint> pool, AttribDesc d, uint splat) {
    if ((d.flags & ATTRIB_PAGED) == 0u) {
        return pool[2] + splat * pool[0] + d.offsetWords;
    }
    uint pages = (pool[3] + (1u << ATTRIB_PAGE_SHIFT) - 1u) >> ATTRIB_PAGE_SHIFT;
    uint page = splat >> ATTRIB_PAGE_SHIFT;
    uint slot = page < pages ? pool[d.offsetWords + page] : ATTRIB_NONE;
    if (slot == ATTRIB_NONE) {
        return ATTRIB_NONE;
    }
    uint local = splat & ((1u << ATTRIB_PAGE_SHIFT) - 1u);
    return d.offsetWords + pages + ((slot << ATTRIB_PAGE_SHIFT) + local) * attribWords(d);
}

// Whether attribute `id` of `splat` is there to read: the pool has it and,
// paged, its page is resident. Not resident reads as zeros.
public bool attribResident(StructuredBuffer<uint> pool, uint id, uint splat) {
    if (id >= pool[1]) {
        return false;
    }
    return attribBase(pool, attribDesc(pool, id), splat) != ATTRIB_NONE;
}

// Components 4 * comp4 .. 4 * comp4 + 3 as floats: normalized formats in
// [0, 1] or [-1, 1], integer formats as their values, missing ones zero.
public float4 readAttrib(StructuredBuffer<uint> pool, uint id, uint splat, uint comp4) {
    if (id >= pool[1]) {
        return float4(0.0);
    }
    AttribDesc d = attribDesc(pool, id);
    uint2 span = attribGroupWords(d, comp4);
    uint at = attribBase(pool, d, splat);
    if (at == ATTRIB_NONE) {
        return float4(0.0);
    }
    uint base = at + span.x;
    uint4 w = uint4(0u);
    for (uint k = 0u; k < span.y; k++) {
        w[k] = pool[base + k];
    }
    uint first = 4u * comp4;
    return decodeAttrib4(d.format, w, d.components > first ? d.components - first : 0u);
}

// One component's raw bits (zero-extended; u32 and f32 as stored).
public uint readAttribU(StructuredBuffer<uint> pool, uint id, uint splat, uint comp) {
    if (id >= pool[1]) {
        return 0u;
    }
    AttribDesc d = attribDesc(pool, id);
    if (comp >= d.components) {
        return 0u;
    }
    uint size = attribFormatBytes(d.format);
    uint byte = comp * size;
    uint at = attribBase(pool, d, splat);
    if (at == ATTRIB_NONE) {
        return 0u;
    }
    uint w = pool[at + byte / 4u] >> ((byte & 3u) * 8u);
    return size == 4u ? w : w & ((1u << (8u * size)) - 1u);
}

public uint attribCount(RWStructuredBuffer<uint> pool) { return pool[1]; }

public AttribDesc attribDesc(RWStructuredBuffer<uint> pool, uint id) {
    uint b = 4u + 4u * id;
    return { pool[b], pool[b + 1u], pool[b + 2u], pool[b + 3u] };
}

public uint attribBase(RWStructuredBuffer<uint> pool, AttribDesc d, uint splat) {
    return pool[2] + splat * pool[0] + d.offsetWords;
}

public float4 readAttrib(RWStructuredBuffer<uint> pool, uint id, uint splat, uint comp4) {
    if (id >= pool[1]) {
        return float4(0.0);
    }
    AttribDesc d = attribDesc(pool, id);
    uint2 span = attribGroupWords(d, comp4);
    uint base = attribBase(pool, d, splat) + span.x;
    uint4 w = uint4(0u);
    for (uint k = 0u; k < span.y; k++) {
        w[k] = pool[base + k];
    }
    uint first = 4u * comp4;
    return decodeAttrib4(d.format, w, d.components > first ? d.components - first : 0u);
}

// Encodes and stores one comp4 group. Padding bytes in the attribute's last
// word are overwritten with zeros; neighbouring attributes are untouched.
public void writeAttrib(RWStructuredBuffer<uint> pool, uint id, uint splat, uint comp4, float4 v) {
    if (id >= pool[1]) {
        return;
    }
    AttribDesc d = attribDesc(pool, id);
    uint2 span = attribGroupWords(d, comp4);
    uint base = attribBase(pool, d, splat) + span.x;
    uint first = 4u * comp4;
    uint n = d.components > first ? d.components - first : 0u;
    uint4 w = encodeAttrib4(d.format, select(uint4(0u, 1u, 2u, 3u) < uint4(n), v, float4(0.0)));
    for (uint k = 0u; k < span.y; k++) {
        pool[base + k] = w[k];
    }
}
