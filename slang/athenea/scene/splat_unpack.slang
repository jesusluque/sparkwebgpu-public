// Copyright (c) 2026 jesus luque.
//
// A CLOUD ON THE DEVICE BACK INTO RECORDS: the decode undone, so that what a
// kernel made -- a decimation, a cut -- can be written by the same exporter a
// file's records are. Each record is fourteen floats and the harmonics:
// position, opacity (linear), the three log scales, the rotation w x y z, the
// base colour (0.5 + SH0 * dc, which is what the shape keeps), and the rest
// rgb per basis. Nothing is
// computed that the decode did not store: the halves are widened and the
// quaternion unpacked.
module splat_unpack;
import athenea.common.packing;

public struct UnpackParams {
    public uint count;
    public uint keep;       // rest basis functions per colour
    public uint shWords;
    public uint stride;     // floats a record: 14 + 3 * keep
};

StructuredBuffer<float4>     positions;
StructuredBuffer<uint>       shape;
StructuredBuffer<uint>       sh;
RWStructuredBuffer<float>    records;
ConstantBuffer<UnpackParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void splatUnpack(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint at = i * params.stride;
    const uint row = i * 4;
    const float4 p = positions[i];
    const float4 q = decodeQuaternion(shape[row]);   // x y z w
    records[at + 0] = p.x;
    records[at + 1] = p.y;
    records[at + 2] = p.z;
    records[at + 3] = p.w;
    records[at + 4] = lowHalf(shape[row + 1]);
    records[at + 5] = highHalf(shape[row + 1]);
    records[at + 6] = lowHalf(shape[row + 2]);
    records[at + 7] = q.w;
    records[at + 8] = q.x;
    records[at + 9] = q.y;
    records[at + 10] = q.z;
    records[at + 11] = highHalf(shape[row + 2]);
    records[at + 12] = lowHalf(shape[row + 3]);
    records[at + 13] = highHalf(shape[row + 3]);
    for (uint k = 0; k < params.keep * 3; ++k) {
        const uint word = sh[i * params.shWords + k / 2];
        records[at + 14 + k] = (k & 1u) == 0u ? lowHalf(word) : highHalf(word);
    }
}
