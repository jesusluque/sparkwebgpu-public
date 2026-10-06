//! A skinned cloud in a `.athc` v3: athenea's `AtheneaSplatSkinningAPI` /
//! UsdSkel binding, as sparkwebGPU carries it to the web.
//!
//! athenea keeps a cloud carried by a skeleton only as a USD stage
//! (`athenea mesh2splat --skinned`): per gaussian `skel:jointIndices` and
//! `skel:jointWeights` (elementSize 4), `primvars:athenea:splat:
//! jointWeightGradients` (six halves: d w_k / d restU, d w_k / d restV for
//! the first three joints), the bind transform `skel:geomBindTransform`, the
//! joint names `skel:joints`, and the joints' transforms written down at the
//! instants the conversion read them, `primvars:athenea:splat:
//! skinningXforms` (matrix4d[], time sampled, bind space to the skeleton's).
//! Its `.athc` writer refuses `--skinned`. This module is that rig in a v3
//! file:
//!
//! * a section `SKIN` (tier 1, right after `SHRS`): per element
//!   `influences` words, one an influence -- joint u16 (low) | weight unorm16
//!   (high), athenea's packed layout (`common/influences.slang`, t22
//!   390670e: the running sum of the weights is what is rounded, so a
//!   gaussian's steps add up to its weights' sum to one step) -- then
//!   `gradientWords` words of two halves (`jointWeightGradients` as the file
//!   holds them: low half d/du, high half d/dv). A merged LoD node carries
//!   the opacity-weighted blend of its splats' influences (the four heaviest
//!   joints, renormalised) and no gradient.
//! * the skeleton (`AthcSkeleton`, an `ATSK` blob between the starts and the
//!   first block, so a pager's first Range request brings it): joint names,
//!   bind transform, and one or more clips of sampled joint transforms.
//!
//! The header words that say so: v3 bytes 144 (influences a splat), 148
//! (gradient words a splat), 152 (u64, the blob's offset; 0 none). In memory
//! and in ATHV pages they are the v2 extra header's seventh and eighth words
//! (`ExtraHeader::skin_influences`, `skin_gradient_words`), padding in every
//! file athenea writes; a v2 file never carries them.

use anyhow::{anyhow, bail, Result};
use serde::Serialize;

use crate::athc::{high_half, low_half, AthcBlock};

/// athenea's `kWeightSteps`: a weight's sixteen bits.
pub const WEIGHT_STEPS: f32 = 65535.0;
/// The largest joint a packed word can name, plus one (`kPackedJoints`).
pub const PACKED_JOINTS: u32 = 65536;
/// The attributes a skinned .athc decodes to (u32 words as stored).
pub const SKIN_INFLUENCES_ATTRIBUTE: &str = "skinInfluences";
pub const SKIN_GRADIENTS_ATTRIBUTE: &str = "skinGradients";

pub const ATSK_MAGIC: u32 = u32::from_le_bytes(*b"ATSK");
pub const ATSK_VERSION: u32 = 1;
const ATSK_HEAD: usize = 112;

/// `packInfluence`: `steps` is the weight in 1/65535ths, already rounded.
pub fn pack_influence(joint: u32, steps: u32) -> u32 {
    (joint & 0xffff) | (steps.min(0xffff) << 16)
}

/// `influenceAt` of the packed layout: (joint, weight).
pub fn unpack_influence(word: u32) -> (u32, f32) {
    (word & 0xffff, (word >> 16) as f32 / WEIGHT_STEPS)
}

/// One gaussian's influences as `splatInfluencesPack` (athenea 390670e)
/// packs them: a negative joint is joint 0, a joint the skeleton does not
/// have holds nothing, and the running sum of the weights is rounded.
/// Returns the words and whether anything was out of range.
pub fn pack_gaussian(indices: &[i32], weights: &[f32], joints: u32) -> (Vec<u32>, bool) {
    let mut out = Vec::with_capacity(indices.len());
    let mut sum = 0.0f32;
    let mut so_far = 0u32;
    let mut bad = false;
    for (&j, &w) in indices.iter().zip(weights) {
        let mut joint = j.max(0) as u32;
        let mut weight = w;
        #[allow(clippy::neg_cmp_op_on_partial_ord)] // a NaN weight is out of range too
        if !(weight >= 0.0 && weight <= 1.0) {
            bad = true;
        }
        if joint >= joints {
            bad = true;
            joint = 0;
            weight = 0.0;
        }
        sum += if weight >= 0.0 { weight } else { 0.0 };
        let up_to = (sum.clamp(0.0, 1.0) * WEIGHT_STEPS + 0.5).floor() as u32;
        let steps = up_to.saturating_sub(so_far);
        so_far = so_far.max(up_to);
        out.push(pack_influence(joint, steps));
    }
    (out, bad)
}

/// The skin words of the merged groups `starts` cut `splats` into: each
/// group's joints weighted by its splats' opacity times weight, the
/// `influences` heaviest kept and renormalised (packed as `pack_gaussian`
/// packs them), the gradients zero (a merged node is no triangle's).
pub fn skin_merge(
    skin: &[u32],
    influences: usize,
    gradient_words: usize,
    starts: &[u32],
    splats: &AthcBlock,
) -> Vec<u32> {
    let per = influences + gradient_words;
    let groups = starts.len();
    let mut out = Vec::with_capacity(groups * per);
    let mut acc: Vec<(u32, f32)> = Vec::new();
    for g in 0..groups {
        let first = starts[g] as usize;
        let end = if g + 1 < groups {
            starts[g + 1] as usize
        } else {
            splats.n
        };
        acc.clear();
        for s in first..end.max(first + 1).min(splats.n) {
            let a = splats.positions[s * 4 + 3].max(1e-6);
            for k in 0..influences {
                let (joint, w) = unpack_influence(skin[s * per + k]);
                if w <= 0.0 {
                    continue;
                }
                match acc.iter_mut().find(|(j, _)| *j == joint) {
                    Some(e) => e.1 += a * w,
                    None => acc.push((joint, a * w)),
                }
            }
        }
        acc.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
        acc.truncate(influences);
        let total: f32 = acc.iter().map(|e| e.1).sum();
        let (mut joints, mut weights) = (
            Vec::with_capacity(influences),
            Vec::with_capacity(influences),
        );
        for k in 0..influences {
            match acc.get(k) {
                Some(&(j, w)) if total > 0.0 => {
                    joints.push(j as i32);
                    weights.push(w / total);
                }
                _ => {
                    joints.push(0);
                    weights.push(if k == 0 && total <= 0.0 { 1.0 } else { 0.0 });
                }
            }
        }
        let (words, _) = pack_gaussian(&joints, &weights, PACKED_JOINTS);
        out.extend_from_slice(&words);
        out.extend(std::iter::repeat_n(0u32, gradient_words));
    }
    out
}

/// One animation: the joints' transforms at `times` (time codes).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SkinClip {
    pub name: String,
    pub time_codes_per_second: f32,
    pub times: Vec<f32>,
    /// `times.len() x joints x 16`: each joint's transform as USD holds a
    /// matrix4d (rows, vectors on the left: `p' = p M`), bind space to the
    /// skeleton's (`skinningXforms`).
    pub xforms: Vec<f32>,
}

/// A skinned cloud's skeleton and clips (the `ATSK` blob).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AthcSkeleton {
    pub influences: u32,
    pub gradient_words: u32,
    /// `skel:joints`, in the order the influences index.
    pub joints: Vec<String>,
    /// The Skeleton prim the cloud was converted against (provenance).
    pub skeleton: String,
    /// `skel:geomBindTransform`, USD rows (vectors on the left).
    pub geom_bind: [f32; 16],
    pub clips: Vec<SkinClip>,
}

fn put(out: &mut Vec<u8>, words: &[u32]) {
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
}
fn put_f(out: &mut Vec<u8>, v: &[f32]) {
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
}
fn pad4(out: &mut Vec<u8>) {
    while out.len() % 4 != 0 {
        out.push(0);
    }
}
fn u32_at(b: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        b.get(at..at + 4)
            .ok_or_else(|| anyhow!("ATSK blob too short"))?
            .try_into()
            .unwrap(),
    ))
}
fn f32s(b: &[u8], at: usize, n: usize) -> Result<Vec<f32>> {
    let bytes = b
        .get(at..at + 4 * n)
        .ok_or_else(|| anyhow!("ATSK blob too short"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
        .collect())
}
fn text(b: &[u8], at: usize, n: usize) -> Result<String> {
    let bytes = b
        .get(at..at + n)
        .ok_or_else(|| anyhow!("ATSK blob too short"))?;
    Ok(String::from_utf8(bytes.to_vec())?)
}

impl AthcSkeleton {
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    /// Checks the clips against the joints.
    pub fn check(&self) -> Result<()> {
        let j = self.joints.len();
        if j == 0 || j as u32 > PACKED_JOINTS {
            bail!("a skeleton of {j} joints (1 to {PACKED_JOINTS})");
        }
        if self.influences == 0
            || self.gradient_words != 0 && self.gradient_words + 1 != self.influences
        {
            bail!(
                "{} influences a splat with {} gradient words",
                self.influences,
                self.gradient_words
            );
        }
        for c in &self.clips {
            if c.times.is_empty() || c.xforms.len() != c.times.len() * j * 16 {
                bail!(
                    "clip '{}': {} transforms for {} samples of {j} joints",
                    c.name,
                    c.xforms.len() / 16,
                    c.times.len()
                );
            }
            if c.times.windows(2).any(|w| w[1].partial_cmp(&w[0]) != Some(std::cmp::Ordering::Greater)) {
                bail!("clip '{}': its times do not increase", c.name);
            }
        }
        Ok(())
    }

    /// The `ATSK` blob.
    ///
    /// ```text
    /// 0   "ATSK"  4 version 1  8 bytes (the whole blob)  12 joints
    /// 16  clips   20 influences a splat   24 gradient words a splat   28 0
    /// 32  geomBindTransform, 16 f32 (USD rows)
    /// 96  name bytes (joints, '\n' between)   100 skeleton path bytes   104 0 0
    /// 112 joint names, then the skeleton path; padded to 4
    ///     each clip: name bytes, samples, timeCodesPerSecond (f32), 0;
    ///     its name padded to 4; times (f32 x samples);
    ///     transforms (f32 x samples x joints x 16, USD rows)
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let names = self.joints.join("\n");
        let mut out = Vec::new();
        put(
            &mut out,
            &[
                ATSK_MAGIC,
                ATSK_VERSION,
                0,
                self.joints.len() as u32,
                self.clips.len() as u32,
                self.influences,
                self.gradient_words,
                0,
            ],
        );
        put_f(&mut out, &self.geom_bind);
        put(
            &mut out,
            &[names.len() as u32, self.skeleton.len() as u32, 0, 0],
        );
        debug_assert_eq!(out.len(), ATSK_HEAD);
        out.extend_from_slice(names.as_bytes());
        out.extend_from_slice(self.skeleton.as_bytes());
        pad4(&mut out);
        for c in &self.clips {
            put(
                &mut out,
                &[
                    c.name.len() as u32,
                    c.times.len() as u32,
                    c.time_codes_per_second.to_bits(),
                    0,
                ],
            );
            out.extend_from_slice(c.name.as_bytes());
            pad4(&mut out);
            put_f(&mut out, &c.times);
            put_f(&mut out, &c.xforms);
        }
        let n = out.len() as u32;
        out[8..12].copy_from_slice(&n.to_le_bytes());
        out
    }

    /// The bytes an `ATSK` blob at the start of `b` takes (its header's word).
    pub fn blob_bytes(b: &[u8]) -> Result<usize> {
        if u32_at(b, 0)? != ATSK_MAGIC {
            bail!("not an ATSK skeleton (no magic)");
        }
        Ok(u32_at(b, 8)? as usize)
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        if u32_at(b, 0)? != ATSK_MAGIC {
            bail!("not an ATSK skeleton (no magic)");
        }
        if u32_at(b, 4)? != ATSK_VERSION {
            bail!("ATSK version {} (this reads {ATSK_VERSION})", u32_at(b, 4)?);
        }
        let joints = u32_at(b, 12)? as usize;
        let clips = u32_at(b, 16)? as usize;
        let mut geom_bind = [0.0; 16];
        geom_bind.copy_from_slice(&f32s(b, 32, 16)?);
        let names_len = u32_at(b, 96)? as usize;
        let skel_len = u32_at(b, 100)? as usize;
        let mut at = ATSK_HEAD;
        let names = text(b, at, names_len)?;
        at += names_len;
        let skeleton = text(b, at, skel_len)?;
        at = (at + skel_len).div_ceil(4) * 4;
        let mut out = Self {
            influences: u32_at(b, 20)?,
            gradient_words: u32_at(b, 24)?,
            joints: names.split('\n').map(str::to_string).collect(),
            skeleton,
            geom_bind,
            clips: Vec::with_capacity(clips),
        };
        if out.joints.len() != joints {
            bail!("ATSK: {} joint names for {joints} joints", out.joints.len());
        }
        for _ in 0..clips {
            let name_len = u32_at(b, at)? as usize;
            let samples = u32_at(b, at + 4)? as usize;
            let fps = f32::from_bits(u32_at(b, at + 8)?);
            at += 16;
            let name = text(b, at, name_len)?;
            at = (at + name_len).div_ceil(4) * 4;
            let times = f32s(b, at, samples)?;
            at += 4 * samples;
            let xforms = f32s(b, at, samples * joints * 16)?;
            at += 4 * samples * joints * 16;
            out.clips.push(SkinClip {
                name,
                time_codes_per_second: fps,
                times,
                xforms,
            });
        }
        out.check()?;
        Ok(out)
    }

    /// Clip `clip`'s transforms at time code `t`: USD's linear interpolation
    /// of matrix samples (each element, between the two samples about `t`;
    /// held before the first and after the last), as athenea's Hydra reads
    /// `skinningXforms`. `joints x 16`, USD rows.
    pub fn pose(&self, clip: usize, t: f32) -> Vec<f32> {
        let c = &self.clips[clip];
        let per = self.joints.len() * 16;
        let k = c.times.partition_point(|&x| x <= t);
        if k == 0 {
            return c.xforms[..per].to_vec();
        }
        if k >= c.times.len() {
            return c.xforms[(c.times.len() - 1) * per..].to_vec();
        }
        let (t0, t1) = (c.times[k - 1], c.times[k]);
        let a = (t - t0) / (t1 - t0);
        let (x0, x1) = (
            &c.xforms[(k - 1) * per..k * per],
            &c.xforms[k * per..(k + 1) * per],
        );
        x0.iter().zip(x1).map(|(p, q)| p + (q - p) * a).collect()
    }
}

/// The (joint, weight) pairs and the gradients of element `e` (for tests
/// and the CPU reference): `influences` pairs, `gradient_words` (du, dv).
pub fn element_skin(
    skin: &[u32],
    influences: usize,
    gradient_words: usize,
    e: usize,
) -> (Vec<(u32, f32)>, Vec<[f32; 2]>) {
    let per = influences + gradient_words;
    let row = &skin[e * per..(e + 1) * per];
    let pairs = row[..influences]
        .iter()
        .map(|&w| unpack_influence(w))
        .collect();
    let grads = row[influences..]
        .iter()
        .map(|&w| [low_half(w), high_half(w)])
        .collect();
    (pairs, grads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_the_running_sum_as_athenea_does() {
        // Thirds: each rounded alone would sum to 65535 - 0 or + 1 steps;
        // the running sum keeps exactly 65535.
        let (w, bad) = pack_gaussian(&[3, 7, 9, -1], &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], 10);
        assert!(!bad);
        let steps: u32 = w.iter().map(|x| x >> 16).sum();
        assert_eq!(steps, 65535);
        assert_eq!(
            w.iter().map(|x| x & 0xffff).collect::<Vec<_>>(),
            [3, 7, 9, 0]
        );
        // A joint past the skeleton holds nothing.
        let (w, bad) = pack_gaussian(&[12, 1], &[0.5, 0.5], 10);
        assert!(bad);
        assert_eq!(unpack_influence(w[0]), (0, 0.0));
    }

    #[test]
    fn a_skeleton_goes_through_its_blob() {
        let s = AthcSkeleton {
            influences: 4,
            gradient_words: 3,
            joints: vec!["Root".into(), "Root/Wing".into()],
            skeleton: "/Bird/Skel".into(),
            geom_bind: std::array::from_fn(|k| if k % 5 == 0 { 1.0 } else { 0.0 }),
            clips: vec![SkinClip {
                name: "flap".into(),
                time_codes_per_second: 30.0,
                times: vec![1.0, 2.0],
                xforms: (0..2 * 2 * 16).map(|k| k as f32).collect(),
            }],
        };
        let b = s.to_bytes();
        assert_eq!(AthcSkeleton::blob_bytes(&b).unwrap(), b.len());
        assert_eq!(AthcSkeleton::from_bytes(&b).unwrap(), s);
        // Halfway between the samples, each element halfway.
        let p = s.pose(0, 1.5);
        assert_eq!(p[0], 16.0);
        assert_eq!(p[31], 47.0);
        assert_eq!(s.pose(0, 0.0)[5], 5.0);
        assert_eq!(s.pose(0, 9.0)[5], 37.0);
    }

    #[test]
    fn a_merged_node_keeps_the_heaviest_joints() {
        let mut b = AthcBlock {
            n: 2,
            ..Default::default()
        };
        b.positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5];
        let (a, _) = pack_gaussian(&[1, 2], &[0.5, 0.5], 10);
        let (c, _) = pack_gaussian(&[2, 3], &[0.5, 0.5], 10);
        let skin: Vec<u32> = [a, vec![0], c, vec![0]].concat();
        let out = skin_merge(&skin, 2, 1, &[0], &b);
        let (pairs, grads) = element_skin(&out, 2, 1, 0);
        // joint 2: 0.5 + 0.25, joint 1: 0.5, joint 3: 0.25 (dropped).
        assert_eq!(pairs[0].0, 2);
        assert_eq!(pairs[1].0, 1);
        assert!((pairs[0].1 - 0.6).abs() < 1e-4 && (pairs[1].1 - 0.4).abs() < 1e-4);
        assert_eq!(grads, vec![[0.0, 0.0]]);
    }
}
