// Copyright (c) 2026 jesus luque.
//
// An engine image onto a display: what athenea view shows, on the device, with no
// readback. Engine images are bottom row first; a window's are top row first.
//
//   colour   linear premultiplied Rec.709 over a background, times 2^exposure,
//            through a view transform, into the display's gamut and its
//            transfer function
//   depth    view z, grey on a log scale from near (white) to far (black)
//   ids      a colour per id, from a hash; -1 (nothing) is the background
//   vector   float4 entries shown as rgb * scale + bias (normals, primvars)
//
// View transforms: Standard (the display clips), and AgX -- Troy Sobotka's,
// in the analytic form Benjamin Wrensch fitted ("Minimal AgX", 2023): an inset
// matrix, a log2 encoding of [-12.47, 4.03] stops, a sixth-order polynomial
// for the sigmoid, the outset, and 2.2 back to linear.
// Displays: sRGB (IEC 61966-2-1), Rec.709 (BT.1886, a pure 2.4 power), Display
// P3 (P3-D65 primaries, sRGB's transfer).
module display;

import athenea.technique.aces2;

struct DisplayParams {
    uint  width;
    uint  height;
    uint  mode;          // 0 colour, 1 depth, 2 ids, 3 vector, 4 nothing (the background)
    uint  stride;        // entries per pixel in the source
    uint  offset;        // which entry
    uint  view;          // 0 Standard, 1 AgX, 2 ACES 2.0
    uint  display;       // 0 sRGB, 1 Rec.709 (BT.1886), 2 Display P3, 3 linear P3 with extended range
    uint  flipRows;      // 1: the output's row 0 is the source's top row
    float exposure;      // stops
    float nearZ;         // depth: white at this view z
    float farZ;          // depth: black at this one
    float vectorScale;
    float vectorBias;
    float backgroundR; float backgroundG; float backgroundB;   // display-referred linear
    uint  outputWidth;   // the output's size: each output pixel shows the source pixel under it
    uint  outputHeight;
    float peakScale;     // display 3: the headroom, peak luminance over reference white, values are clamped to it
    // Mode 5: one id's own matte instead of the whole picture -- what that id
    // covers, in white on black. 0 shows every id, each in its own colour.
    uint  cryptoIsolate;
    uint  pad2;
};

StructuredBuffer<Aces2Params> acesParams;   // view 2: one, built by aces2_prepare
StructuredBuffer<float>       acesTables;   // and its tables
StructuredBuffer<float4>     colour;   // modes 0 and 3
StructuredBuffer<float>      depth;    // mode 1
StructuredBuffer<uint>       ids;      // mode 2
RWTexture2D<float4>          output;
ConstantBuffer<DisplayParams> params;

float srgbEncode(float x) {
    return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

public float3 transferEncode(float3 c, uint display) {
    if (display == 3) {
        return clamp(c, float3(0.0), float3(params.peakScale));   // linear, the headroom kept, as ACES clamps at its peak
    }
    c = saturate(c);
    if (display == 1) {
        return pow(c, float3(1.0 / 2.4));
    }
    return float3(srgbEncode(c.r), srgbEncode(c.g), srgbEncode(c.b));
}

/// Linear Rec.709 to linear P3-D65 (both D65 white).
public static const float3x3 kRec709ToP3 = float3x3(
    0.8224621, 0.1775380, 0.0000000,
    0.0331941, 0.9668058, 0.0000000,
    0.0170827, 0.0723974, 0.9105199);

public float3 agx(float3 c) {
    // Wrensch's matrices are GLSL column-major applied as M * v: row vectors here.
    static const float3x3 kInset = float3x3(
        0.842479062253094, 0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772, 0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    static const float3x3 kOutset = float3x3(
        1.19687900512017, -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    const float minEv = -12.47393;
    const float maxEv = 4.026069;
    c = mul(max(c, float3(0.0)), kInset);
    c = clamp(log2(max(c, float3(1e-10))), float3(minEv), float3(maxEv));
    c = (c - minEv) / (maxEv - minEv);
    // The sigmoid, by Horner.
    float3 y = 15.5 * c - 40.14;
    y = y * c + 31.96;
    y = y * c - 6.868;
    y = y * c + 0.4298;
    y = y * c + 0.1191;
    y = y * c - 0.00232;
    y = mul(y, kOutset);
    return pow(max(y, float3(0.0)), float3(2.2));
}

public float3 displayColour(float4 premultiplied, float3 background, float exposure, uint view, uint display) {
    float3 c = premultiplied.rgb + background * (1.0 - saturate(premultiplied.a));
    c *= pow(2.0, exposure);
    if (view == 1) {
        c = agx(c);
    }
    if (view == 2) {
        // The output transform limits to the display's primaries itself
        // (its tables were built for them), so P3 needs no matrix here.
        const Aces2Params p = acesParams[0];
        return transferEncode(outputTransformFwd(mul(c, p.sceneToAP0), p, acesTables), display);
    }
    if (display == 2 || display == 3) {
        c = mul(kRec709ToP3, c);
    }
    return transferEncode(c, display);
}

float3 idColour(uint id) {
    uint h = id * 2654435761u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return float3(float(h & 255u), float((h >> 8) & 255u), float((h >> 16) & 255u)) / 255.0 * 0.8 + 0.2;
}

/// The source pixel under output pixel `tid`, false outside the output.
public bool displayPixel(uint3 tid, out uint pixel) {
    pixel = 0;
    if (tid.x >= params.outputWidth || tid.y >= params.outputHeight) {
        return false;
    }
    const uint x = min(uint((float(tid.x) + 0.5) * float(params.width) / float(params.outputWidth)), params.width - 1);
    const uint y = min(uint((float(tid.y) + 0.5) * float(params.height) / float(params.outputHeight)), params.height - 1);
    const uint row = params.flipRows != 0 ? params.height - 1 - y : y;
    pixel = row * params.width + x;
    return true;
}

public bool displayingColour() {
    return params.mode == 0;
}

public float3 displayBackground() {
    return float3(params.backgroundR, params.backgroundG, params.backgroundB);
}

/// The scene-referred linear colour at `pixel` over the background, exposed:
/// what a view transform takes.
public float3 displayScene(uint pixel) {
    const float4 premultiplied = colour[pixel * params.stride + params.offset];
    return (premultiplied.rgb + displayBackground() * (1.0 - saturate(premultiplied.a))) * pow(2.0, params.exposure);
}

/// What every mode but colour shows at `pixel`; `backgroundShown` where
/// nothing is.
public float3 displayOther(uint pixel, float3 backgroundShown) {
    float3 shown = backgroundShown;
    if (params.mode == 1) {
        const float z = depth[pixel];
        if (z > 0.0) {
            const float t = saturate(log(z / params.nearZ) / log(params.farZ / params.nearZ));
            shown = float3(1.0 - t);
        }
    } else if (params.mode == 2) {
        const uint id = ids[pixel * params.stride + params.offset];
        if (id != 0xFFFFFFFFu) {
            shown = idColour(id);
        }
    } else if (params.mode == 3) {
        shown = saturate(colour[pixel * params.stride + params.offset].rgb * params.vectorScale + params.vectorBias);
    } else if (params.mode == 5) {
        // A CRYPTOMATTE, AS A READER PREVIEWS ONE: every id its own colour,
        // mixed by what it covers. Three layers, six ranks, read from the
        // plane the offset addresses -- a picture of the matte, not the matte.
        const uint pixels = params.width * params.height;
        float3 sum = float3(0.0);
        float covered = 0.0;
        float isolated = 0.0;
        for (uint layer = 0; layer < 3; ++layer) {
            const float4 row = colour[layer * pixels + pixel];
            const uint2 pair = uint2(bit_cast<uint>(row.x), bit_cast<uint>(row.z));
            const float2 coverage = float2(row.y, row.w);
            for (uint which = 0; which < 2; ++which) {
                const float weight = max(which == 0 ? coverage.x : coverage.y, 0.0);
                const uint id = which == 0 ? pair.x : pair.y;
                if (weight > 0.0) {
                    sum += idColour(id) * weight;
                    covered += weight;
                    isolated += id == params.cryptoIsolate ? weight : 0.0;
                }
            }
        }
        if (params.cryptoIsolate != 0) {
            // One id alone: the matte a compositor would pull, which is the
            // way to see that the bumper's id really is the bumper's.
            shown = float3(saturate(isolated));
        } else if (covered > 0.0) {
            shown = sum / max(covered, 1.0);
        }
    }   // any other mode: the background (an AOV nothing drew)
    return shown;
}

public void displayWrite(uint3 tid, float3 shown) {
    output[tid.xy] = float4(shown, 1.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void displayTransform(uint3 tid: SV_DispatchThreadID) {
    uint pixel;
    if (!displayPixel(tid, pixel)) {
        return;
    }
    const float3 background = displayBackground();
    float3 shown;
    if (params.mode == 0) {
        shown = displayColour(colour[pixel * params.stride + params.offset], background, params.exposure, params.view,
                              params.display);
    } else {
        shown = displayOther(pixel, transferEncode(background, params.display));
    }
    output[tid.xy] = float4(shown, 1.0);
}
