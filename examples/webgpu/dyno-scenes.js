// Dyno scenes shared by dyno-webgl.html and dyno-webgpu.html, so both
// backends render the same graphs:
//   ?scene=effect   (default) examples/splat-shader-effects' cat, with
//                   &effect=Electronic|Deep Meditation|Waves|Flare|Disintegrate
//                   &intensity=0.8
//   ?scene=depth    the butterfly with modifiers.setDepthColor (world space)
//   ?scene=normal   the butterfly with modifiers.setWorldNormalColor
//   ?scene=edit     the butterfly with SplatEdit SDFs (a red sphere, a
//                   displacing box)
//   ?scene=snow     generators.snowBox
//   ?scene=skin     the butterfly flapping by linear-blend SplatSkinning
//                   (covariance splats); &jacobian=1 also with the weights'
//                   gradients (weightGradients: the whole Jacobian)
//   ?scene=rgba     the butterfly recoloured by SplatMesh.splatRgba (an
//                   RgbaArray of channel-swapped colours)
//   &cov=1          covariance splats: an ExtSplats mesh with covSplats, and
//                   the renderer's covSplats (implied by scene=skin)
//   &t=<seconds>    a fixed time for screenshots; animates when absent
//   ?w=&h=          canvas size

export function sceneParams() {
  const params = new URLSearchParams(location.search);
  return {
    scene: params.get("scene") ?? "effect",
    effect: params.get("effect") ?? "Disintegrate",
    intensity: Number(params.get("intensity") ?? 0.8),
    time: params.has("t") ? Number(params.get("t")) : null,
    cov: params.get("cov") === "1" || params.get("scene") === "skin",
    jacobian: params.get("jacobian") === "1",
    size: {
      w: Number(params.get("w") ?? 800),
      h: Number(params.get("h") ?? 600),
    },
  };
}

export const EFFECTS = {
  Electronic: 1,
  "Deep Meditation": 2,
  Waves: 3,
  Flare: 4,
  Disintegrate: 5,
};

const GLSL_GLOBALS = /* glsl */ `
vec3 hash(vec3 p) {
  return fract(sin(p*123.456)*123.456);
}

mat2 rot(float a) {
  float s = sin(a), c = cos(a);
  return mat2(c, -s, s, c);
}

vec3 headMovement(vec3 pos, float t) {
  pos.xy *= rot(smoothstep(-1., -2., pos.y) * .2 * sin(t*2.));
  return pos;
}

vec3 breathAnimation(vec3 pos, float t) {
  float b = sin(t*1.5);
  pos.yz *= rot(smoothstep(-1., -3., pos.y) * .15 * -b);
  pos.z += .3;
  pos.y += 1.2;
  pos *= 1. + exp(-3. * length(pos)) * b;
  pos.z -= .3;
  pos.y -= 1.2;
  return pos;
}

vec4 fractal1(vec3 pos, float t, float intensity) {
  float m = 100.;
  vec3 p = pos * .1;
  p.y += .5;
  for (int i = 0; i < 8; i++) {
    p = abs(p) / clamp(abs(p.x * p.y), 0.3, 3.) - 1.;
    p.xy *= rot(radians(90.));
    if (i > 1) m = min(m, length(p.xy) + step(.3, fract(p.z * .5 + t * .5 + float(i) * .2)));
  }
  m = step(m, 0.5) * 1.3 * intensity;
  return vec4(-pos.y * .3, 0.5, 0.7, .3) * intensity + m;
}

vec4 fractal2(vec3 center, vec3 scales, vec4 rgba, float t, float intensity) {
  vec3 pos = center;
  float splatSize = length(scales);
  vec3 p = pos * .65;
  pos.y += 2.;
  float c = 0.;
  float l, l2 = length(p);
  float m = 100.;
  for (int i = 0; i < 10; i++) {
    p.xyz = abs(p.xyz) / dot(p.xyz, p.xyz) - .8;
    l = length(p.xyz);
    c += exp(-1. * abs(l - l2) * (1. + sin(t * 1.5 + pos.y)));
    l2 = length(p.xyz);
    m = min(m, length(p.xyz));
  }
  c = smoothstep(0.3, 0.5, m + sin(t * 1.5 + pos.y * .5)) + c * .1;
  return vec4(vec3(length(rgba.rgb)) * vec3(c, c*c, c*c*c) * intensity,
              rgba.a * exp(-20. * splatSize) * m * intensity);
}

vec4 sin3D(vec3 p, float t) {
  float m = exp(-2. * length(sin(p * 5. + t * 3.))) * 5.;
  return vec4(m) + .3;
}

vec4 disintegrate(vec3 pos, float t, float intensity) {
  vec3 p = pos + (hash(pos) * 2. - 1.) * intensity;
  float tt = smoothstep(-1., 0.5, -sin(t + -pos.y * .5));
  p.xz *= rot(tt * 2. + p.y * 2. * tt);
  return vec4(mix(p, pos, tt), tt);
}

vec4 flare(vec3 pos, float t) {
  vec3 p = vec3(0., -1.5, 0.);
  float tt = smoothstep(-1., .5, sin(t + hash(pos).x));
  tt = tt * tt;
  p.x += sin(t * 2.) * tt;
  p.z += sin(t * 2.) * tt;
  p.y += sin(t) * tt;
  return vec4(mix(pos, p, tt), tt);
}
`;

/**
 * examples/splat-shader-effects' modifier, in GLSL (translated to WGSL on
 * WebGPU). `effect` and `intensity` are a name and a number, or dyno
 * uniforms (dynoInt with an EFFECTS value, dynoFloat) to change them while
 * it runs.
 */
export function makeEffectModifier(dyno, animateT, effect, intensity) {
  return dyno.dynoBlock(
    { gsplat: dyno.Gsplat },
    { gsplat: dyno.Gsplat },
    ({ gsplat }) => {
      const d = new dyno.Dyno({
        inTypes: {
          gsplat: dyno.Gsplat,
          t: "float",
          effectType: "int",
          intensity: "float",
        },
        outTypes: { gsplat: dyno.Gsplat },
        globals: () => [GLSL_GLOBALS],
        statements: ({ inputs, outputs }) =>
          dyno.unindentLines(/* glsl */ `
            ${outputs.gsplat} = ${inputs.gsplat};
            vec3 localPos = ${inputs.gsplat}.center;
            vec3 splatScales = ${inputs.gsplat}.scales;
            vec4 splatColor = ${inputs.gsplat}.rgba;
            if (${inputs.effectType} == 1) {
              ${outputs.gsplat}.center = headMovement(localPos, ${inputs.t});
              vec4 effect1 = fractal1(localPos, ${inputs.t}, ${inputs.intensity});
              ${outputs.gsplat}.rgba.rgba = mix(splatColor, splatColor*effect1, ${inputs.intensity});
            } else if (${inputs.effectType} == 2) {
              vec4 effectColor = fractal2(localPos, splatScales, splatColor, ${inputs.t}, ${inputs.intensity});
              ${outputs.gsplat}.rgba.rgba = mix(splatColor, effectColor, ${inputs.intensity});
              ${outputs.gsplat}.center = breathAnimation(localPos, ${inputs.t});
            } else if (${inputs.effectType} == 3) {
              vec4 effect = sin3D(localPos, ${inputs.t});
              ${outputs.gsplat}.rgba.rgba = mix(splatColor, splatColor*effect, ${inputs.intensity});
              vec3 pos = localPos;
              pos.y += 1.;
              pos *= (1. + effect.x * .05 * ${inputs.intensity});
              pos.y -= 1.;
              ${outputs.gsplat}.center = pos;
            } else if (${inputs.effectType} == 5) {
              vec4 e = disintegrate(localPos, ${inputs.t}, ${inputs.intensity});
              ${outputs.gsplat}.center = e.xyz;
              ${outputs.gsplat}.scales = mix(vec3(.01, .01, .01), ${inputs.gsplat}.scales, e.w);
            } else if (${inputs.effectType} == 4) {
              vec4 e = flare(localPos, ${inputs.t});
              ${outputs.gsplat}.center = e.xyz;
              ${outputs.gsplat}.rgba.rgb = mix(splatColor.rgb, vec3(1.), abs(e.w));
              ${outputs.gsplat}.rgba.a = mix(splatColor.a, 0.3, abs(e.w));
            }
          `),
      });
      gsplat = d.apply({
        gsplat,
        t: animateT,
        effectType:
          typeof effect === "string"
            ? dyno.dynoInt(EFFECTS[effect] ?? 1)
            : effect,
        intensity:
          typeof intensity === "number" ? dyno.dynoFloat(intensity) : intensity,
      }).gsplat;
      return { gsplat };
    },
  );
}

/**
 * Builds the scene: returns { mesh, url } for a SplatMesh to load, or
 * { generator } for a SplatGenerator, plus `tick(seconds)` to animate.
 */
export async function buildScene({ THREE, spark, getAssetFileURL, params }) {
  const { dyno, modifiers, generators, SplatMesh } = spark;
  const animateT = dyno.dynoFloat(params.time ?? 0);
  if (params.scene === "snow") {
    const { snow } = generators.snowBox({
      box: new THREE.Box3(
        new THREE.Vector3(-1.5, -1, -4),
        new THREE.Vector3(1.5, 1.5, -2),
      ),
      density: 4000,
      minScale: 0.004,
      maxScale: 0.012,
    });
    if (params.time != null) {
      // Still, for screenshots: the flakes drift by deltaTime otherwise.
      const update = snow.frameUpdate;
      snow.frameUpdate = (context) =>
        update({ ...context, time: params.time, deltaTime: 0 });
    }
    return { generator: snow, tick: () => {} };
  }
  const file = params.scene === "effect" ? "cat.spz" : "butterfly.spz";
  const mesh = new SplatMesh({
    url: await getAssetFileURL(file),
    extSplats: params.cov,
    covSplats: params.cov,
  });
  if (params.scene === "effect") {
    mesh.quaternion.set(1, 0, 0, 0);
    mesh.position.set(0, -0.7, -2.5);
    mesh.rotation.set(Math.PI, 0.6, 0);
    mesh.scale.set(0.5, 0.5, 0.5);
    mesh.objectModifier = makeEffectModifier(
      dyno,
      animateT,
      params.effect,
      params.intensity,
    );
  } else {
    mesh.quaternion.set(1, 0, 0, 0);
    mesh.position.set(0, 0, -3);
    mesh.rotation.y += 0.6;
    if (params.scene === "depth") {
      modifiers.setDepthColor(mesh, 2, 4, false);
    } else if (params.scene === "normal") {
      modifiers.setWorldNormalColor(mesh);
    } else if (params.scene === "edit") {
      const edit = new spark.SplatEdit({
        rgbaBlendMode: spark.SplatEditRgbaBlendMode.MULTIPLY,
        softEdge: 0.05,
      });
      const sphere = new spark.SplatEditSdf({
        type: spark.SplatEditSdfType.SPHERE,
        radius: 0.25,
        color: new THREE.Color(1, 0.2, 0.2),
      });
      sphere.position.set(0.3, 0.1, 0);
      const box = new spark.SplatEditSdf({
        type: spark.SplatEditSdfType.BOX,
        radius: 0.02,
        color: new THREE.Color(0.3, 1, 0.3),
        displace: new THREE.Vector3(0, 0.15, 0),
      });
      box.scale.set(0.3, 0.2, 0.3);
      box.position.set(-0.35, -0.2, 0);
      edit.add(sphere, box);
      mesh.add(edit);
    } else if (params.scene === "rgba") {
      await mesh.initialized;
      const splats = mesh.extSplats ?? mesh.packedSplats;
      const array = new Uint8Array(splats.numSplats * 4);
      splats.forEachSplat((i, _center, _scales, _quat, opacity, color) => {
        const b = (v) => Math.round(Math.min(1, Math.max(0, v)) * 255);
        array.set([b(color.b), b(color.r), b(color.g), b(opacity)], 4 * i);
      });
      mesh.splatRgba = new spark.RgbaArray({ array });
    } else if (params.scene === "skin") {
      await mesh.initialized;
      const skin = skinWings(THREE, spark, mesh, params.jacobian);
      mesh.updateGenerator();
      return {
        mesh,
        tick: (seconds) => skin(params.time ?? seconds),
      };
    }
  }
  mesh.updateGenerator();
  return {
    mesh,
    tick: (seconds) => {
      if (params.time == null) animateT.value = seconds;
    },
  };
}

// Linear-blend skinning: bone 0 the body, bones 1 and 2 the wings, which
// flap about the body's axis and stretch, blended near the body.
function skinWings(THREE, spark, mesh, jacobian = false) {
  const { SplatSkinning, SplatSkinningMode } = spark;
  const skinning = new SplatSkinning({
    mesh,
    numBones: 3,
    mode: SplatSkinningMode.LINEAR_BLEND,
    weightGradients: jacobian,
  });
  const zero = new THREE.Vector3();
  mesh.extSplats.forEachSplat((index, center) => {
    const w = Math.min(1, Math.abs(center.x) / 0.3);
    const wing = center.x < 0 ? 1 : 2;
    skinning.setSplatBones(
      index,
      new THREE.Vector4(0, wing, 0, 0),
      new THREE.Vector4(1 - w, w, 0, 0),
    );
    if (jacobian) {
      // d w / d x across the ramp, the body's the opposite.
      const dw = Math.abs(center.x) < 0.3 ? Math.sign(center.x) / 0.3 : 0;
      skinning.setSplatWeightGradients(
        index,
        new THREE.Vector3(-dw, 0, 0),
        new THREE.Vector3(dw, 0, 0),
        zero,
      );
    }
  });
  if (jacobian) skinning.gradientTexture.needsUpdate = true;
  for (let bone = 0; bone < 3; bone++) {
    skinning.setRestMatrix(bone, new THREE.Matrix4());
  }
  mesh.skinning = skinning;
  const m = new THREE.Matrix4();
  const s = new THREE.Matrix4();
  return (t) => {
    const angle = 0.6 * Math.sin(2 * t);
    const stretch = 1 + 0.3 * Math.sin(t);
    for (const [bone, sign] of [
      [1, -1],
      [2, 1],
    ]) {
      m.makeRotationY(sign * angle);
      s.makeScale(stretch, 1 / stretch, 1);
      skinning.setBoneMatrix(bone, m.multiply(s));
    }
    skinning.updateBones();
  };
}
