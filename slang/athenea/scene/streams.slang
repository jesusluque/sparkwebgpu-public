// Copyright (c) 2026 jesus luque.
//
// Clouds as USD holds them -- one array per attribute, float32 or float16 --
// interleaved into raw records on the device, which then take the same
// validate and decode every file format takes. The CPU uploads each array's
// bytes as they are: no loop over elements, no half to float on the host.
//
//   splat record  0..2 position   3 opacity (linear)   4..6 scale (linear)
//                 7 w  8 x  9 y  10 z   11..13 SH DC   14.. rest, rgb per basis
//   point record  0..2 position   3..5 colour (linear)
//
// Blender holds a splat cloud as a PointCloud whose `radiance:base` is the DC
// coefficient and the opacity in one float4, and whose `radiance:sh_N` are one
// array a basis function after DC: `radianceBase` and the planes of `sh`
// (kShPlanes) take them as they come, and this kernel lays them out as the
// records any other layout gives.
module streams;
import athenea.common.packing;
import athenea.common.transfer_layout;

/// Value `index` of a float array: float32 words, or float16 halves packed two
/// to a little-endian word, first in the low bits.
float streamValue(StructuredBuffer<uint> words, bool half, uint index) {
    if (half) {
        const uint word = words[index / 2];
        return (index & 1u) == 0 ? lowHalf(word) : highHalf(word);
    }
    return bit_cast<float>(words[index]);
}

static const uint kPositions = 1;
static const uint kRotations = 2;
static const uint kScales = 4;
static const uint kOpacities = 8;
static const uint kSh = 16;
static const uint kColours = 32;
static const uint kMetallic = 64;
static const uint kRoughness = 128;
static const uint kTransmission = 256;
static const uint kCrypto = 512;
static const uint kTransferDirect = 1024;
static const uint kTransferIndirect = 2048;
static const uint kShadowBits = 4096;
static const uint kThinWalled = 8192;
static const uint kNormals = 16384;
static const uint kEmission = 32768;
// DC rgb and opacity, four floats a splat (Blender's `radiance:base`): in place
// of `opacities` and of the DC in `sh`.
static const uint kBase = 65536;
// `sh` holds planes: one array a basis function after DC, rgb a splat
// (Blender's `radiance:sh_N`), `planeLength` splats each, back to back.
static const uint kShPlanes = 131072;
// The layers over the base, one array each (GpuSplats::lobes): one value a
// splat, three for a colour.
static const uint kSpecularWeight = 1u << 18;
static const uint kSpecularColour = 1u << 19;
static const uint kSpecularIor = 1u << 20;
static const uint kCoatWeight = 1u << 21;
static const uint kCoatRoughness = 1u << 22;
static const uint kCoatIor = 1u << 23;
static const uint kSheenColour = 1u << 24;
static const uint kSheenRoughness = 1u << 25;
static const uint kCoatDarkening = 1u << 26;
// A TX transfer's reflected field, 48 a splat (GpuSplats::transfer, after the
// indirect half).
static const uint kTransferReflected = 1u << 27;
// Ten floats a splat, a zonal transfer: two lobes, each an axis in the
// gaussian's frame and three coefficients (`primvars:athenea:splat:transferZonal`),
// in place of the harmonics of `transferDirect`.
static const uint kTransferZonal = 1u << 28;
// One int a splat, nonzero where its metal is a Schlick (OpenPBR, glTF):
// carried, as a thin wall is, as a mark on the transmission the decode takes
// back off (splat_encoding's kSchlickMetalMark).
static const uint kSchlickMetal = 1u << 29;
static const float kSchlickMetalMark = 4.0;
// What a thin wall adds to the transmission a record carries: splat_encoding's
// kThinWalledMark, which the decode takes back off.
static const float kThinWalledMark = 2.0;

struct StreamParams {
    uint count;          // records in this slice
    uint first;          // the slice's first element
    uint stride;         // floats per record
    uint keep;           // rest basis functions written per colour
    uint coefficients;   // SH coefficients per splat in the stream (DC first)
    uint present;        // which streams are bound (k* bits)
    uint halves;         // which of them are float16
    uint colourMode;     // points: 0 none (white), 1 one colour, 2 one per point
    uint pbrBase;        // where metallic and roughness go in a record, or kNoPbr
    uint cryptoBase;     // where the Cryptomatte id goes, or kNoPbr: its bits, carried as a float
    // Where a gaussian's transfer goes in a record and how many values it is:
    // nine of the direct half, and twenty-seven more where the indirect one
    // is there too; or ten, a zonal transfer (kTransferZonal).
    uint transferBase; uint transferCount;
    // Where the words of which ways out are open go, or kNoPbr, and how
    // many: two (8 x 8 cells), or eight (16 x 16, a TX transfer's).
    uint shadowBase;
    uint shadowWords;
    // Where the three floats of a shading normal go, or kNoPbr.
    uint normalBase;
    // Where the three floats of an emitted radiance go, or kNoPbr.
    uint emissionBase;
    // Where the thirteen floats of the layers over the base go, or kNoPbr.
    uint lobesBase;
    // kShPlanes: the splats a plane holds (the whole cloud's, not the slice's).
    uint planeLength;
    // The element the transfer's streams and the open directions' start at:
    // they are uploaded a slice at a time (the whole TX Corvette's are 7 GB),
    // so they hold the slice's elements alone and `first` is theirs.
    uint transferFirst;
};

static const uint kNoPbr = 0xFFFFFFFFu;

StructuredBuffer<uint>     positions;   // xyz
StructuredBuffer<uint>     rotations;   // xyzw, GfQuat's layout
StructuredBuffer<uint>     scales;      // xyz
StructuredBuffer<uint>     opacities;
StructuredBuffer<uint>     sh;          // rgb per coefficient
StructuredBuffer<uint>     colours;     // rgb
StructuredBuffer<uint>     metallic;    // one a splat, where the stage carries it
StructuredBuffer<uint>     roughness;
StructuredBuffer<uint>     transmission;
StructuredBuffer<uint>     cryptoObject; // one id a splat, as the prim it came from hashes
StructuredBuffer<uint>     transferDirect;   // nine a splat
StructuredBuffer<uint>     transferIndirect; // twenty-seven a splat (forty-eight at degree 3), rgb a coefficient
StructuredBuffer<uint>     transferReflected; // forty-eight a splat, a TX transfer's field
StructuredBuffer<uint>     transferZonal;    // ten a splat: two lobes in the gaussian's frame
StructuredBuffer<uint>     shadowBits;       // two words a splat, or eight, bits
StructuredBuffer<uint>     schlickMetal;     // one int a splat, nonzero for a Schlick metal
StructuredBuffer<uint>     thinWalled;       // one int a splat, nonzero for a thin wall
StructuredBuffer<uint>     normals;          // xyz a splat, the shading normal
StructuredBuffer<uint>     emission;         // rgb a splat, the radiance it gives off
StructuredBuffer<uint>     radianceBase;     // DC rgb, opacity: four a splat (kBase)
StructuredBuffer<uint>     specularWeight;   // the layers over the base, one array each
StructuredBuffer<uint>     specularColour;   // rgb a splat
StructuredBuffer<uint>     specularIor;
StructuredBuffer<uint>     coatWeight;
StructuredBuffer<uint>     coatRoughness;
StructuredBuffer<uint>     coatIor;
StructuredBuffer<uint>     sheenColour;      // rgb a splat
StructuredBuffer<uint>     sheenRoughness;
StructuredBuffer<uint>     coatDarkening;
RWStructuredBuffer<float>  records;
ConstantBuffer<StreamParams> params;

bool present(uint stream) { return (params.present & stream) != 0; }
bool isHalf(uint stream) { return (params.halves & stream) != 0; }

[shader("compute")]
[numthreads(256, 1, 1)]
void splatStreams(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint e = params.first + i;
    const uint r = i * params.stride;
    for (uint k = 0; k < 3; ++k) {
        records[r + k] = streamValue(positions, isHalf(kPositions), e * 3 + k);
        records[r + 4 + k] = present(kScales) ? streamValue(scales, isHalf(kScales), e * 3 + k) : 1.0;
    }
    records[r + 3] = present(kBase)        ? streamValue(radianceBase, isHalf(kBase), e * 4 + 3)
                     : present(kOpacities) ? streamValue(opacities, isHalf(kOpacities), e)
                                           : 1.0;
    if (present(kRotations)) {
        records[r + 7] = streamValue(rotations, isHalf(kRotations), e * 4 + 3);
        for (uint k = 0; k < 3; ++k) {
            records[r + 8 + k] = streamValue(rotations, isHalf(kRotations), e * 4 + k);
        }
    } else {
        records[r + 7] = 1.0;
        records[r + 8] = 0.0;
        records[r + 9] = 0.0;
        records[r + 10] = 0.0;
    }
    const bool planes = present(kShPlanes);
    const bool haveSh = present(kSh) && params.coefficients > 0;
    const uint base = e * params.coefficients;
    for (uint k = 0; k < 3; ++k) {
        // No harmonics: DC 0, which is base colour 0.5, the schema's default.
        records[r + 11 + k] = present(kBase)        ? streamValue(radianceBase, isHalf(kBase), e * 4 + k)
                              : haveSh && !planes ? streamValue(sh, isHalf(kSh), base * 3 + k)
                                                  : 0.0;
    }
    for (uint basis = 0; basis < params.keep; ++basis) {
        // Interleaved, DC first; or a plane a basis function, DC elsewhere.
        const uint at = planes ? (basis * params.planeLength + e) * 3 : (base + 1 + basis) * 3;
        for (uint c = 0; c < 3; ++c) {
            records[r + 14 + basis * 3 + c] = streamValue(sh, isHalf(kSh), at + c);
        }
    }
    // What the gaussian reflects with, after the harmonics: a cloud converted
    // from a mesh brings them, a capture does not.
    if (params.pbrBase != kNoPbr) {
        records[r + params.pbrBase] = present(kMetallic) ? streamValue(metallic, isHalf(kMetallic), e) : 0.0;
        records[r + params.pbrBase + 1] = present(kRoughness) ? streamValue(roughness, isHalf(kRoughness), e) : 1.0;
        // A thin wall rides on the transmission, two higher (splat_encoding).
        const float thin = (present(kThinWalled) && thinWalled[e] != 0u ? kThinWalledMark : 0.0) +
                           (present(kSchlickMetal) && schlickMetal[e] != 0u ? kSchlickMetalMark : 0.0);
        records[r + params.pbrBase + 2] =
            (present(kTransmission) ? streamValue(transmission, isHalf(kTransmission), e) : 0.0) + thin;
    }
    // The id's bits, as the float they already are (the fixup made sure of
    // it): the record is floats, and the decode takes them back as bits.
    if (params.cryptoBase != kNoPbr) {
        records[r + params.cryptoBase] = present(kCrypto) ? bit_cast<float>(cryptoObject[e]) : 0.0;
    }
    // Which ways out are open: bits, carried as the floats they are.
    if (params.shadowBase != kNoPbr) {
        const uint words = params.shadowWords >= 8u ? params.shadowWords : 2u;   // 2, 8 or 32
        for (uint w = 0; w < words; ++w) {
            records[r + params.shadowBase + w] =
                present(kShadowBits) ? bit_cast<float>(shadowBits[(e - params.transferFirst) * words + w]) : 0.0;
        }
    }
    // The shading normal, as the stage holds it; the decode normalises it.
    if (params.normalBase != kNoPbr) {
        for (uint k = 0; k < 3; ++k) {
            records[r + params.normalBase + k] =
                present(kNormals) ? streamValue(normals, isHalf(kNormals), e * 3 + k) : 0.0;
        }
    }
    // The radiance it gives off, as the stage holds it; the decode packs it.
    if (params.emissionBase != kNoPbr) {
        for (uint k = 0; k < 3; ++k) {
            records[r + params.emissionBase + k] =
                present(kEmission) ? streamValue(emission, isHalf(kEmission), e * 3 + k) : 0.0;
        }
    }
    // The layers over the base, in packing's record order; an array the stage
    // does not carry is its default (`plainLobes`).
    if (params.lobesBase != kNoPbr) {
        const uint b = r + params.lobesBase;
        records[b] = present(kSpecularWeight) ? streamValue(specularWeight, isHalf(kSpecularWeight), e) : 1.0;
        for (uint k = 0; k < 3; ++k) {
            records[b + 1 + k] =
                present(kSpecularColour) ? streamValue(specularColour, isHalf(kSpecularColour), e * 3 + k) : 1.0;
            records[b + 8 + k] =
                present(kSheenColour) ? streamValue(sheenColour, isHalf(kSheenColour), e * 3 + k) : 0.0;
        }
        records[b + 4] = present(kSpecularIor) ? streamValue(specularIor, isHalf(kSpecularIor), e) : 1.5;
        records[b + 5] = present(kCoatWeight) ? streamValue(coatWeight, isHalf(kCoatWeight), e) : 0.0;
        records[b + 6] = present(kCoatRoughness) ? streamValue(coatRoughness, isHalf(kCoatRoughness), e) : 0.0;
        records[b + 7] = present(kCoatIor) ? streamValue(coatIor, isHalf(kCoatIor), e) : 1.5;
        records[b + 11] =
            present(kSheenRoughness) ? streamValue(sheenRoughness, isHalf(kSheenRoughness), e) : 0.0;
        records[b + 12] = present(kCoatDarkening) ? streamValue(coatDarkening, isHalf(kCoatDarkening), e) : 0.0;
    }
    // How much of an environment reaches this gaussian: the direct half
    // first, then the indirect one. A cloud that carries neither writes
    // nothing here and the decode writes no buffer.
    if (params.transferCount > 0 && params.transferBase != kNoPbr && present(kTransferZonal)) {
        for (uint k = 0; k < params.transferCount; ++k) {
            records[r + params.transferBase + k] = streamValue(transferZonal, isHalf(kTransferZonal), (e - params.transferFirst) * 10 + k);
        }
    } else if (params.transferCount > 0 && params.transferBase != kNoPbr) {
        const uint direct = transferDirectCount(params.transferCount);
        const uint indirect = transferIndirectCount(params.transferCount);
        for (uint k = 0; k < direct && k < params.transferCount; ++k) {
            records[r + params.transferBase + k] =
                present(kTransferDirect) ? streamValue(transferDirect, isHalf(kTransferDirect), (e - params.transferFirst) * direct + k)
                                         : 0.0;
        }
        for (uint k = direct; k < direct + indirect; ++k) {
            records[r + params.transferBase + k] =
                present(kTransferIndirect)
                    ? streamValue(transferIndirect, isHalf(kTransferIndirect),
                                  (e - params.transferFirst) * indirect + (k - direct))
                    : 0.0;
        }
        for (uint k = direct + indirect; k < params.transferCount; ++k) {
            records[r + params.transferBase + k] =
                present(kTransferReflected)
                    ? streamValue(transferReflected, isHalf(kTransferReflected),
                                  (e - params.transferFirst) * kTransferFieldValues + (k - direct - indirect))
                    : 0.0;
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pointStreams(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint e = params.first + i;
    const uint r = i * 6;
    const uint from = params.colourMode == 2 ? e : 0;
    for (uint k = 0; k < 3; ++k) {
        records[r + k] = streamValue(positions, isHalf(kPositions), e * 3 + k);
        records[r + 3 + k] = params.colourMode != 0 ? streamValue(colours, isHalf(kColours), from * 3 + k) : 1.0;
    }
}
