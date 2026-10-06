// Copyright (c) 2026 jesus luque.
//
// FEWER GAUSSIANS, KEPT WHERE THEY MATTER: a cloud cut once, by what merging
// would lose, rather than every frame by how far away it is.
//
// The levels of detail already merge every octree cell into one Gaussian by
// moments (lod_common). What a frame's cut asks is how many pixels a cell
// spans; what a decimation asks is whether the merged Gaussian stands for
// what is under it -- and that is a property of the cloud, not of a view.
// Each group gets an error, the largest of, over its children (the finer
// level's groups, or at the finest level its splats):
//
//   colour     the share of the children whose colour or opacity is further
//              than `colourTolerance` from the merged one, over `outliers`.
//              Not the largest difference: a baked cloud's colours carry the
//              noise of the paths that made them, and one glint among a
//              cell's gaussians kept every cell holding one whole (7 % fewer
//              gaussians). Not a mean either: a mean over a cell that is two
//              thirds cobble and one third mortar sits between them, within
//              a lenient tolerance of both, and the mortar went. A share is
//              what tells a glint (a few) from an edge (many).
//   reach      how far a splat under the cell stands from the merged
//              Gaussian's centre, in its own standard deviations, over 1.5:
//              a merged Gaussian is a blob, and one that must stretch past
//              the splats it stands for to hold them is a smear. The splats
//              and not the finer merges: those sit near the middle of a cell
//              however far its splats reach, and a floor of one colour went
//              to four blobs, one a quadrant. Over discs as wide as their
//              spacing this holds a merge to 4 x 4 of them.
//   flatness   where the children are discs -- a surface, which every cloud
//              converted from a mesh is -- how much thicker the merged
//              Gaussian is than they are, against its width, over
//              `flatTolerance`: a curved surface merged into one blob is no
//              longer a surface. The extra and not the whole: a merge is
//              never thinner than its children, and discs a tenth as thick
//              as wide merged into nothing on a flat floor.
//   children   each child's own error
//
//   around     over everything the merge *reaches* -- the splats of its
//              cell and of the cells around it, as many as its tail is long
//              (three at most), that stand within `reach` + 1 of
//              the merge as it is kept, widened to cover as its splats did
//              (lodDecimateShape) -- the share of them where what the merge
//              puts there, its colour difference times its density, is over
//              `colourTolerance`, over `outliers`. Judged on its own cell
//              alone, a merge beside the mortar laid its colour over the
//              joint.
//
// The last-but-one makes an error never smaller than any error under it, so down any
// line of the tree it passes 1 once: a group is kept merged when its error is
// at most 1 and its parent's is not, and a splat is kept as it is when its
// finest group's error is over 1. Every place of the cloud is kept exactly
// once, as a frame's cut keeps it.
module lod_decimate;
import athenea.common.packing;
import athenea.common.morton;
import lod_common;
import lod_blob;

public struct DecimateParams {
    public uint  count;            // groups at this level, or splats
    public uint  children;         // groups at the finer level (0 at the finest: the children are splats)
    public uint  splats;           // the cloud's splats
    public uint  parents;          // groups at the coarser level (0 at the coarsest)
    public float colourTolerance;
    public float flatTolerance;
    public float reach;            // standard deviations a child may stand out
    public uint  finest;           // groups at the finest level (shape)
    public uint  below;            // levels between this one and the finest (shape)
    public uint  level;            // this level's octree level: cells a side are 2^level
    public float outliers;         // the share of what a merge stands for that may differ by more than colourTolerance
    public float extent;           // the bounds' largest edge: a level-L cell is extent / 2^L
    public uint  hasKey;           // 1: `keys` says which records may share a merge
    public uint  hasMaterial;      // 1: `material` is compared as colour is
    public uint  width;            // attributes: words a record
    public uint  mode;             // attributes: how they merge (lodMergeAttribute); gather: 1 for splats
};

// This level's merged Gaussians, and its cells.
StructuredBuffer<float4>          positions;
StructuredBuffer<uint>            shape;
StructuredBuffer<uint>            cells;
// The children: the finer level's Gaussians, cells and errors, or the splats.
StructuredBuffer<float4>          childPositions;
StructuredBuffer<uint>            childShape;
StructuredBuffer<uint>            childCells;
StructuredBuffer<float>           childErrors;
StructuredBuffer<uint>            starts;        // finest level: each group's first splat
RWStructuredBuffer<float>         errors;        // this level's
// The cut.
StructuredBuffer<float>           parentErrors;
StructuredBuffer<uint>            parentCells;
StructuredBuffer<uint>            groupOf;       // per splat: its finest group
StructuredBuffer<float>           finestErrors;
RWStructuredBuffer<uint>          selected;
// The shape a kept merge is written with.
StructuredBuffer<uint>            finestCells;
StructuredBuffer<float4>          widePositions;   // the kept merges, as lodDecimateShape widened them
StructuredBuffer<uint>            wideShape;
StructuredBuffer<float4>          splatPositions;
StructuredBuffer<uint>            splatShape;
RWStructuredBuffer<float4>        keptPositions;
RWStructuredBuffer<uint>          keptShape;
// WHAT A CLOUD CARRIES BESIDE ITS GAUSSIANS, a record at a time.
StructuredBuffer<uint>            recordOf;      // per store splat: the record of the file it came from
StructuredBuffer<uint>            keys;          // per record: what may not be merged across
StructuredBuffer<float4>          material;      // per record: metallic, roughness, transmission
RWStructuredBuffer<uint2>         groupRange;    // per group: its run of store splats
ConstantBuffer<DecimateParams>    params;

Blob mergedAt(uint g) {
    const uint row = g * 4;
    return blobOf(positions[g], shape[row], shape[row + 1], shape[row + 2], shape[row + 3]);
}

Blob childAt(uint c) {
    const uint row = c * 4;
    return blobOf(childPositions[c], childShape[row], childShape[row + 1], childShape[row + 2], childShape[row + 3]);
}

/// Whether a child's colour or opacity is further from the merged one than
/// the tolerance.
float colourApart(Blob merged, Blob child) {
    const float3 dc = abs(child.colour - merged.colour);
    const float da = abs(child.opacity - merged.opacity);
    return max(max(dc.x, dc.y), max(dc.z, da)) > params.colourTolerance ? 1.0 : 0.0;
}

/// A share of outliers as an error, 1 being as many as are allowed.
float shareError(float share) {
    return share / max(params.outliers, 1.0e-6);
}

/// How far a child stands from the merged Gaussian, in its standard
/// deviations, over the reach allowed.
float lost(Blob merged, Blob child) {
    const float3 away = child.centre - merged.centre;
    // In the merged Gaussian's own frame, in its standard deviations.
    const float3 local = float3(dot(away, merged.axes[0]), dot(away, merged.axes[1]), dot(away, merged.axes[2])) /
                         max(merged.scale, float3(1.0e-20));
    return length(local) / params.reach;
}

/// The merge's thinnest axis.
uint thinnest(Blob b) {
    uint thin = 0;
    if (b.scale.y < b.scale[thin]) thin = 1;
    if (b.scale.z < b.scale[thin]) thin = 2;
    return thin;
}

/// A child's variance along `n`, and whether it is a disc.
float childAlong(Blob child, float3 n) {
    float v = 0.0;
    for (uint k = 0; k < 3; ++k) {
        const float c = dot(n, child.axes[k]);
        v += c * c * child.scale[k] * child.scale[k];
    }
    return v;
}

bool isDisc(Blob child) {
    return min(min(child.scale.x, child.scale.y), child.scale.z) <
           0.2 * max(max(child.scale.x, child.scale.y), child.scale.z);
}

/// How much thicker than its children the merge is, against its width, over
/// the tolerance; 0 where the children are not discs. `own` is the mean of
/// the children's variances along the merge's thinnest axis.
float flatError(Blob merged, float own, bool discs) {
    if (!discs) {
        return 0.0;
    }
    const uint thin = thinnest(merged);
    const float wide = max(max(merged.scale.x, merged.scale.y), merged.scale.z);
    const float extra = sqrt(max(merged.scale[thin] * merged.scale[thin] - own, 0.0));
    return extra / max(wide, 1.0e-20) / params.flatTolerance;
}

/// Two store splats whose records say they may not share a merge: another
/// prim's (a Cryptomatte id), another joint's, a sheet beside a solid. Not a
/// tolerance: a merge that names one prim cannot stand for two.
float keyError(uint reference, uint i) {
    return params.hasKey != 0 && keys[recordOf[i]] != keys[recordOf[reference]] ? 1.0e9 : 0.0;
}

/// How far a store splat's material is from the reference's.
float materialApart(uint reference, uint i) {
    if (params.hasMaterial == 0) {
        return 0.0;
    }
    const float3 d = abs(material[recordOf[i]].xyz - material[recordOf[reference]].xyz);
    return max(max(d.x, d.y), d.z);
}

/// The first of `count` sorted codes that is not below `code`.
uint lowerBound(StructuredBuffer<uint> codes, uint count, uint code) {
    uint lo = 0;
    uint hi = count;
    while (lo < hi) {
        const uint mid = (lo + hi) / 2;
        if (codes[mid] < code) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/// A kept merge, as it is written.
Blob wideAt(uint g) {
    const uint row = g * 4;
    return blobOf(widePositions[g], wideShape[row], wideShape[row + 1], wideShape[row + 2], wideShape[row + 3]);
}

/// The splats of the cell at `code`: those of the finest groups inside it.
uint2 splatsOf(uint code) {
    const uint shift = 3 * params.below;
    const uint fa = lowerBound(finestCells, params.finest, code << shift);
    const uint fb = lowerBound(finestCells, params.finest, (code + 1) << shift);
    return uint2(fa < params.finest ? starts[fa] : params.splats, fb < params.finest ? starts[fb] : params.splats);
}

/// How far what the kept merge of cell `code` reaches is from it in colour,
/// over the tolerance: the root mean square over the splats of the cell and
/// of the cells around it within `reach` of it, weighted by its density.
float aroundError(Blob wide, uint code, uint reference) {
    const int side = int(1u << params.level);
    const int3 at = int3(cellIndex(code));
    // As many cells around as the tail is long: the cells next to it were
    // not enough -- a merge 2.6 discs wide reaches 6.4 past its centre, and
    // the next square began at 5.
    const float tail = params.reach + 1.0;
    const float edge = params.extent / float(side);
    const int ring = clamp(int(ceil(tail * max(max(wide.scale.x, wide.scale.y), wide.scale.z) / max(edge, 1.0e-20))),
                           1, 3);
    float apart = 0.0;
    float weights = 0.0;
    for (int dz = -ring; dz <= ring; ++dz) {
        for (int dy = -ring; dy <= ring; ++dy) {
            for (int dx = -ring; dx <= ring; ++dx) {
                const int3 near = at + int3(dx, dy, dz);
                if (any(near < int3(0)) || any(near >= int3(side))) {
                    continue;
                }
                const uint2 run = splatsOf(cellCode(uint3(near)));
                for (uint i = run.x; i < run.y; ++i) {
                    const uint r = i * 4;
                    const Blob splat = blobOf(splatPositions[i], splatShape[r], splatShape[r + 1],
                                              splatShape[r + 2], splatShape[r + 3]);
                    const float3 away = splat.centre - wide.centre;
                    const float3 local = float3(dot(away, wide.axes[0]), dot(away, wide.axes[1]),
                                                dot(away, wide.axes[2])) /
                                         max(wide.scale, float3(1.0e-20));
                    // As far as the merge's tail still shows: at a reach past
                    // 1.5, a merge beside a checker's edge laid a tenth of
                    // its colour over the next square.
                    const float d2 = dot(local, local);
                    if (d2 > tail * tail) {
                        continue;
                    }
                    // WHAT THE MERGE PUTS THERE: its colour, as much as its
                    // density at that splat says, where the splat's own was.
                    // Counted, not weighted by that density: weighted, the
                    // tail -- a merge four discs wide reaches four discs past
                    // its block, where a disc reached one -- was a small share
                    // of a large sum, and a blue merge ate the edge of every
                    // square beside it.
                    const float g = exp(-0.5 * d2) * wide.opacity;
                    const float3 dc = abs(splat.colour - wide.colour);
                    const bool differs = max(max(max(dc.x, dc.y), dc.z), materialApart(reference, i)) * g >
                                         params.colourTolerance;
                    // Within its reach are the splats it stands for, and a
                    // share of them may differ -- noise. Past it, in its
                    // tail, is what stands beside it, and none may: a share
                    // there bounds nothing, since an edge that clips the end
                    // of the tail is a small share of a merge and a line of
                    // its colour across the next square.
                    if (differs && d2 > params.reach * params.reach) {
                        return 1.0e9;
                    }
                    apart += differs ? 1.0 : 0.0;
                    weights += 1.0;
                }
            }
        }
    }
    return weights > 0.0 ? shareError(apart / weights) : 0.0;
}

/// The finest level: each group against its splats, and what it reaches.
[shader("compute")]
[numthreads(64, 1, 1)]
void lodErrorFinest(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const Blob merged = mergedAt(g);
    const float3 n = merged.axes[thinnest(merged)];
    const uint first = starts[g];
    const uint end = g + 1 < params.count ? starts[g + 1] : params.splats;
    float worst = 0.0;
    float own = 0.0;
    bool discs = true;
    for (uint i = first; i < end; ++i) {
        const Blob child = childAt(i);
        worst = max(worst, max(lost(merged, child), keyError(first, i)));
        own += childAlong(child, n);
        discs = discs && isDisc(child);
    }
    worst = max(worst, flatError(merged, own / float(max(end - first, 1u)), discs));
    errors[g] = worst > 1.0 ? worst : max(worst, aroundError(wideAt(g), cells[g], first));
}

/// A coarser level: each group against the finer groups in its cell -- never
/// below their own errors -- and what it reaches.
[shader("compute")]
[numthreads(64, 1, 1)]
void lodErrorLevel(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const Blob merged = mergedAt(g);
    // A child's cell, one level down, is its parent's with three bits more.
    const uint code = cells[g];
    const uint first = lowerBound(childCells, params.children, code << 3);
    const uint end = lowerBound(childCells, params.children, (code + 1) << 3);
    float worst = 0.0;
    for (uint c = first; c < end; ++c) {
        worst = max(worst, childErrors[c]);
    }
    // Reach and flatness against the splats themselves: the finer merges sit
    // near the cell's middle, and are as thick as their own curvature made
    // them.
    if (worst <= 1.0) {
        const float3 n = merged.axes[thinnest(merged)];
        const uint2 run = splatsOf(code);
        float own = 0.0;
        bool discs = true;
        for (uint i = run.x; i < run.y; ++i) {
            const uint r = i * 4;
            const Blob splat = blobOf(splatPositions[i], splatShape[r], splatShape[r + 1], splatShape[r + 2],
                                      splatShape[r + 3]);
            worst = max(worst, max(lost(merged, splat), keyError(run.x, i)));
            own += childAlong(splat, n);
            discs = discs && isDisc(splat);
        }
        worst = max(worst, flatError(merged, own / float(max(run.y - run.x, 1u)), discs));
    }
    errors[g] = worst > 1.0 ? worst : max(worst, aroundError(wideAt(g), code, splatsOf(code).x));
}

/// A group is kept merged where it loses little enough and its parent does
/// not.
[shader("compute")]
[numthreads(256, 1, 1)]
void lodDecimateGroups(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    bool parentKept = false;
    if (params.parents > 0) {
        const uint parent = lowerBound(parentCells, params.parents, cells[g] >> 3);
        parentKept = parent < params.parents && parentCells[parent] == (cells[g] >> 3) &&
                     parentErrors[parent] <= 1.0;
    }
    selected[g] = errors[g] <= 1.0 && !parentKept ? 1u : 0u;
}

/// A splat is kept as it is where its finest group could not stand for it.
[shader("compute")]
[numthreads(256, 1, 1)]
void lodDecimateSplats(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    selected[i] = finestErrors[groupOf[i]] > 1.0 ? 1u : 0u;
}

/// A MERGE THAT COVERS WHAT IT REPLACES.
///
/// The levels merge by moments, which keeps the second moment of what is
/// under a cell: a cell of 2 x 2 gaussians each as wide as their spacing
/// merges into one 0.58 of its own spacing wide. That is right for a cell
/// seen from far enough that it is a pixel, which is what the levels are
/// for, and wrong for a cloud looked at as closely as before: merged
/// neighbours overlap less than the gaussians they replace, and the floor
/// showed its background through a grid of holes. So a kept merge is widened
/// in its plane to overlap its neighbours as its gaussians overlapped
/// theirs. Over the cell's own splats, along each of the merge's two long
/// axes: their positions' variance `v`, the spacing `d` a grid of `n` of
/// them over that spread has (`d^2 = 12 sqrt(v_a v_b) / (n - 1)`, exact for
/// a square grid), their own width `c` along it; the cell then spans
/// `e = sqrt(12 v + d^2)`, and the merge is `c / d` of that -- never
/// narrower than the moments made it. Its opacity is its splats' mean,
/// weighted by their area, since it now covers as they did.
[shader("compute")]
[numthreads(64, 1, 1)]
void lodDecimateShape(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint row = g * 4;
    const Blob merged = mergedAt(g);
    keptPositions[g] = positions[g];
    for (uint w = 0; w < 4; ++w) {
        keptShape[row + w] = shape[row + w];
    }
    // The splats under the cell, which is also what the merge stands for.
    const uint2 run = splatsOf(cells[g]);
    groupRange[g] = run;
    const uint first = run.x;
    const uint end = run.y;
    const uint n = end - first;
    if (n < 2) {
        return;
    }
    // The merge's axes, its thinnest set apart.
    uint thin = 0;
    if (merged.scale.y < merged.scale[thin]) thin = 1;
    if (merged.scale.z < merged.scale[thin]) thin = 2;
    const uint ia = thin == 0 ? 1u : 0u;
    const uint ib = thin == 2 ? 1u : 2u;
    const float3 a = merged.axes[ia];
    const float3 b = merged.axes[ib];
    float2 sum = float2(0.0);
    float2 squares = float2(0.0);
    float2 own = float2(0.0);
    float opacity = 0.0;
    float weights = 0.0;
    for (uint i = first; i < end; ++i) {
        const uint r = i * 4;
        const Blob splat = blobOf(splatPositions[i], splatShape[r], splatShape[r + 1], splatShape[r + 2],
                                  splatShape[r + 3]);
        const float3 away = splat.centre - merged.centre;
        const float2 at = float2(dot(away, a), dot(away, b));
        sum += at;
        squares += at * at;
        float2 width = float2(0.0);
        for (uint k = 0; k < 3; ++k) {
            const float s2 = splat.scale[k] * splat.scale[k];
            width += float2(dot(a, splat.axes[k]), dot(b, splat.axes[k])) *
                     float2(dot(a, splat.axes[k]), dot(b, splat.axes[k])) * s2;
        }
        own += width;
        const float smallest = min(min(splat.scale.x, splat.scale.y), splat.scale.z);
        const float area = splat.scale.x * splat.scale.y * splat.scale.z / max(smallest, 1.0e-20);
        opacity += splat.opacity * area;
        weights += area;
    }
    const float count = float(n);
    const float2 mean = sum / count;
    const float2 spread = max(squares / count - mean * mean, float2(0.0));
    const float2 c = sqrt(own / count);
    const float d2 = 12.0 * sqrt(spread.x * spread.y) / (count - 1.0);
    if (!(d2 > 1.0e-30)) {
        return;
    }
    const float2 span = sqrt(12.0 * spread + d2);
    // Held to twice what the moments made it: splats nearly in a line have
    // almost no spread across it, `d` goes to nothing, and unbounded the
    // merge was a streak across the frame.
    const float2 wanted = min(c / sqrt(d2), float2(2.0)) * span;
    float3 scale = merged.scale;
    scale[ia] = max(scale[ia], min(wanted.x, 2.0 * scale[ia]));
    scale[ib] = max(scale[ib], min(wanted.y, 2.0 * scale[ib]));
    const float3 ls = log(max(scale, float3(1.0e-20)));
    keptShape[row + 1] = packHalves(halfSafe(ls.x), halfSafe(ls.y));
    keptShape[row + 2] = packHalves(halfSafe(ls.z), highHalf(shape[row + 2]));
    keptPositions[g] = float4(positions[g].xyz, weights > 0.0 ? saturate(opacity / weights) : positions[g].w);
}

