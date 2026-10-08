#!/usr/bin/env python3
"""Writes merge_tree.usda: a small merge tree in athenea's layout
(surfels-web/NOTES.md, "Merge tree"; `athenea decimate --tree`), for
`usd-athc --tree` (rust/build-lod/src/bin/usd-athc.rs).

Two 8 x 8 sheets of discs folded at a right angle (floor, normal +z, red
and rough; wall, normal -y, blue and metallic). Leaves first, depth first:
each sheet's discs along a Morton curve. Then the merges, children before
parents: neighbours along that curve, pair by pair, the moment-matched
gaussian of the run (axis-aligned in the sheet's frame, so its moments are
exact without an eigensolver), its colour, normal and material the run's
means by mass, its coverage W / A uncapped. The two sheets never merge (the
fold): two roots. lodCost grows with the depth, not monotone on purpose
(one level's merges are cheaper than the one below)."""

import math
import os

N = 8
STEP = 0.01
SCALE = 0.6 * STEP
THIN = 0.05 * STEP


def morton(a, b):
    code = 0
    for bit in range(8):
        code |= ((a >> bit) & 1) << (2 * bit) | ((b >> bit) & 1) << (2 * bit + 1)
    return code


nodes = []  # dict: face, w, mu(u, v), var(u, v), leaves
for face in range(2):
    cells = sorted(((a, b) for a in range(N) for b in range(N)), key=lambda c: morton(*c))
    for a, b in cells:
        u, v = (a + 0.5) * STEP, (b + 0.5) * STEP
        nodes.append({"face": face, "w": SCALE * SCALE, "mu": (u, v), "var": (SCALE**2, SCALE**2), "depth": 0})
leaves = len(nodes)
parent = list(range(leaves))
cost = [0.0] * leaves
for face in range(2):
    cur = [k for k in range(leaves) if nodes[k]["face"] == face]
    depth = 0
    while len(cur) > 1:
        depth += 1
        nxt = []
        for i in range(0, len(cur), 2):
            x, y = nodes[cur[i]], nodes[cur[i + 1]]
            w = x["w"] + y["w"]
            mu = tuple((x["w"] * x["mu"][d] + y["w"] * y["mu"][d]) / w for d in range(2))
            var = tuple(
                (x["w"] * (x["var"][d] + x["mu"][d] ** 2) + y["w"] * (y["var"][d] + y["mu"][d] ** 2)) / w - mu[d] ** 2
                for d in range(2)
            )
            k = len(nodes)
            nodes.append({"face": face, "w": w, "mu": mu, "var": var, "depth": depth})
            parent[cur[i]] = k
            parent[cur[i + 1]] = k
            parent.append(k)
            # Not monotone: depth 3's merges cost less than depth 2's.
            cost.append(0.5 if depth == 3 else float(depth) + 0.01 * (k % 5))
            nxt.append(k)
        cur = nxt


def fmt(v):
    return "%.7g" % v


positions, orientations, scales, opacities, coverage, sh, normals, metallic, roughness = ([] for _ in range(9))
for n in nodes:
    u, v = n["mu"]
    su, sv = (math.sqrt(x) for x in n["var"])
    if n["face"] == 0:
        positions.append((u, v, 0.0))
        orientations.append((1.0, 0.0, 0.0, 0.0))
        normals.append((0.0, 0.0, 1.0))
        sh.append((1.5, -1.5, -1.5))
        metallic.append(0.0)
        roughness.append(0.8)
    else:
        positions.append((u, 0.0, v))
        # +90 degrees about x: local y to z, local z (the normal) to -y.
        orientations.append((math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0))
        normals.append((0.0, -1.0, 0.0))
        sh.append((-1.5, -1.5, 1.5))
        metallic.append(1.0)
        roughness.append(0.2)
    scales.append((su, sv, THIN))
    opacities.append(1.0 if n["depth"] == 0 else 0.99)
    coverage.append(n["w"] / (su * sv))


def tup(rows):
    return "[" + ", ".join("(" + ", ".join(fmt(x) for x in r) + ")" for r in rows) + "]"


def flat(rows):
    return "[" + ", ".join(fmt(x) for x in rows) + "]"


out = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def ParticleField3DGaussianSplat "Splats"
    {{
        point3f[] positions = {tup(positions)}
        quatf[] orientations = {tup(orientations)}
        float3[] scales = {tup(scales)}
        float[] opacities = {flat(opacities)}
        uniform int radiance:sphericalHarmonicsDegree = 0
        float3[] radiance:sphericalHarmonicsCoefficients = {tup(sh)}
        float[] primvars:athenea:splat:coverage = {flat(coverage)} (interpolation = "vertex")
        normal3f[] primvars:athenea:splat:normal = {tup(normals)} (interpolation = "vertex")
        float[] primvars:athenea:splat:metallic = {flat(metallic)} (interpolation = "vertex")
        float[] primvars:athenea:splat:roughness = {flat(roughness)} (interpolation = "vertex")
        int[] primvars:athenea:splat:lodParent = [{", ".join(str(p) for p in parent)}] (interpolation = "vertex")
        float[] primvars:athenea:splat:lodCost = {flat(cost)} (interpolation = "vertex")
        custom bool primvars:athenea:splat:linear = 1
        custom bool primvars:athenea:splat:lodTree = 1
    }}
}}
"""
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_tree.usda"), "w") as f:
    f.write(out)
print(f"{leaves} leaves, {len(nodes) - leaves} merges")
