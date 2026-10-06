// The draw attributes of one mesh into the frame's draw pool, indexed like
// the accumulator: runs after kernels/generate.slang with the same output
// range and LOD remap, so generate itself stays unchanged.
//
// The draw pool has its own schema, the union of every mesh's toDraw
// attributes; attribMap names, per draw attribute, the source attribute that
// fills it (ATTRIB_NONE: zeros). Equal formats copy words exactly; otherwise
// values convert through float. Direction attributes turn with the object.

import core.math;
import core.attrib;

static const uint GATHER_USE_LOD = 4u; // as GEN_USE_LOD
static const uint GATHER_LOD_FADE = 65536u; // as GEN_LOD_FADE: the top byte is a fade

struct GatherParams {
    uint numSplats;
    uint outBase;
    uint flags;
    uint pad0;
    float4 rotate; // object to world quaternion
};

StructuredBuffer<uint> srcAttribs;
StructuredBuffer<uint> lodIndices;
StructuredBuffer<uint> attribMap;
RWStructuredBuffer<uint> drawAttribs;
ConstantBuffer<GatherParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void gatherAttribs(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.numSplats) {
        return;
    }
    uint index = (params.flags & GATHER_USE_LOD) != 0u ? lodIndices[i] : i;
    if ((params.flags & GATHER_LOD_FADE) != 0u) {
        index &= 0x00FFFFFFu;
    }
    bool valid = index < attribSplats(srcAttribs);
    uint o = params.outBase + i;

    uint n = attribCount(drawAttribs);
    for (uint k = 0u; k < n; k++) {
        AttribDesc dst = attribDesc(drawAttribs, k);
        uint dstBase = attribBase(drawAttribs, dst, o);
        uint words = attribWords(dst);
        uint s = attribMap[k];
        if (!valid || s >= attribCount(srcAttribs)) {
            for (uint w = 0u; w < words; w++) {
                drawAttribs[dstBase + w] = 0u;
            }
            continue;
        }
        AttribDesc src = attribDesc(srcAttribs, s);
        bool turn = (src.flags & ATTRIB_DIRECTION) != 0u;
        if (src.format == dst.format && src.components == dst.components && !turn) {
            // A paged attribute not resident: zeros, as readAttrib gives.
            uint srcBase = attribBase(srcAttribs, src, index);
            if (srcBase == ATTRIB_NONE) {
                for (uint w = 0u; w < words; w++) {
                    drawAttribs[dstBase + w] = 0u;
                }
                continue;
            }
            for (uint w = 0u; w < words; w++) {
                drawAttribs[dstBase + w] = srcAttribs[srcBase + w];
            }
            continue;
        }
        for (uint g = 0u; 4u * g < dst.components; g++) {
            float4 v = readAttrib(srcAttribs, s, index, g);
            if (turn && g == 0u) {
                v.xyz = quatVec(params.rotate, v.xyz);
            }
            writeAttrib(drawAttribs, k, o, g, v);
        }
    }
}
