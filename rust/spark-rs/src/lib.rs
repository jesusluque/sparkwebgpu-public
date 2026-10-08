
use std::cell::RefCell;
use js_sys::{Array, Float32Array, Float64Array, Object, Reflect, Uint8Array, Uint16Array, Uint32Array};
use spark_lib::attrib::{AttribArray, AttribSpec, LodMerge};
use spark_lib::decoder::{ChunkReceiver, MultiDecoder, SplatEncoding, SplatFileType, SplatGetter};
#[cfg(all(feature = "spz", feature = "gsplat"))]
use spark_lib::spz::SpzEncoder;
#[cfg(feature = "gsplat")]
use spark_lib::gsplat::{GsplatSH1,GsplatSH2,GsplatSH3};
#[cfg(feature = "gsplat")]
use spark_lib::gsplat::GsplatArray as GsplatArrayInner;
#[cfg(feature = "csplat")]
use spark_lib::csplat::CsplatArray as CsplatArrayInner;
use spark_lib::tsplat::TsplatArray;
use wasm_bindgen::prelude::*;

use crate::decoder::ChunkDecoder;
#[cfg(feature = "gsplat")]
use crate::ext_splats::ExtSplatsData;
#[cfg(feature = "csplat")]
use crate::packed_splats::PackedSplatsData;

mod raycast;
use raycast::{raycast_packed_ellipsoids, raycast_ext_ellipsoids};

mod sort;
use sort::{sort_internal, SortBuffers, sort32_internal, Sort32Buffers};

#[cfg(feature = "gsplat")]
mod transform;
#[cfg(feature = "gsplat")]
use transform::{transform_gsplatarray, TransformOptions};

mod decoder;

mod decoded_attribs;
#[cfg(feature = "csplat")]
mod packed_splats;
#[cfg(feature = "gsplat")]
mod ext_splats;

mod lod_tree;

#[wasm_bindgen(start)]
pub fn wasm_start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn simd_enabled() -> bool {
    cfg!(target_feature = "simd128")
}

thread_local! {
    static SORT_BUFFERS: RefCell<SortBuffers> = RefCell::new(SortBuffers::default());
    static SORT32_BUFFERS: RefCell<Sort32Buffers> = RefCell::new(Sort32Buffers::default());
}

macro_rules! stub_fn {
    ($pred:meta, $name:ident) => {
        #[cfg(not($pred))]
        #[wasm_bindgen(variadic)]
        pub fn $name(_args: &JsValue) -> Result<Object, JsValue> {
            Err(JsValue::from(&format!("'{}' is disabled in this build, it requires: {}", stringify!($name), stringify!($pred))))
        }
    };
}

#[wasm_bindgen]
pub fn sort_splats(
    num_splats: u32, readback: Uint16Array, ordering: Uint32Array,
) -> u32 {
    let max_splats = readback.length() as usize;

    let active_splats = SORT_BUFFERS.with_borrow_mut(|buffers| {
        buffers.ensure_size(max_splats);
        let sub_readback = readback.subarray(0, num_splats);
        sub_readback.copy_to(&mut buffers.readback[..num_splats as usize]);

        let active_splats = match sort_internal(buffers, num_splats as usize) {
            Ok(active_splats) => active_splats,
            Err(err) => {
                wasm_bindgen::throw_str(&err.to_string());
            }
        };

        if active_splats > 0 {
            // Copy out ordering result
            let subarray = &buffers.ordering[..active_splats as usize];
            ordering.subarray(0, active_splats).copy_from(subarray);
        }
        active_splats
    });

    active_splats
}

#[wasm_bindgen]
pub fn sort32_splats(
    num_splats: u32, readback: Uint32Array, ordering: Uint32Array,
) -> u32 {
    let max_splats = readback.length() as usize;

    let active_splats = SORT32_BUFFERS.with_borrow_mut(|buffers| {
        buffers.ensure_size(max_splats);
        let sub_readback = readback.subarray(0, num_splats);
        sub_readback.copy_to(&mut buffers.readback[..num_splats as usize]);

        let active_splats = match sort32_internal(buffers, max_splats, num_splats as usize) {
            Ok(active_splats) => active_splats,
            Err(err) => {
                wasm_bindgen::throw_str(&err.to_string());
            }
        };

        if active_splats > 0 {
            // Copy out ordering result
            let subarray = &buffers.ordering[..active_splats as usize];
            ordering.subarray(0, active_splats).copy_from(subarray);
        }
        active_splats
    });

    active_splats
}

#[wasm_bindgen]
#[cfg(feature = "csplat")]
pub fn decode_to_packedsplats(
    file_type: Option<String>, path_name: Option<String>, encoding: JsValue,
    sh1_codes: Option<Uint32Array>, sh2_codes: Option<Uint32Array>, sh3_codes: Option<Uint32Array>,
) -> Result<ChunkDecoder, JsValue> {
    let encoding = if encoding.is_falsy() {
        SplatEncoding::default()
    } else {
        serde_wasm_bindgen::from_value(encoding)?
    };

    let file_type = if let Some(file_type) = file_type {
        match SplatFileType::from_enum_str(&file_type) {
            Ok(file_type) => Some(file_type),
            Err(err) => { return Err(JsValue::from(err.to_string())); },
        }
    } else {
        None
    };

    let mut splats = PackedSplatsData::new(encoding);
    splats.set_sh_codes(sh1_codes, sh2_codes, sh3_codes);

    let decoder = MultiDecoder::new(splats, file_type, path_name.as_deref());
    let on_finish = |receiver: Box<dyn ChunkReceiver>| {
        let decoder: Box<MultiDecoder<PackedSplatsData>> = receiver.into_any().downcast().unwrap();
        let file_type = decoder.file_type.unwrap();
        let object = decoder.into_splats().into_splat_object();
        Reflect::set(&object, &JsValue::from_str("fileType"), &JsValue::from(file_type.to_enum_str())).unwrap();
        Ok(JsValue::from(object))
    };

    let decoder = ChunkDecoder::new(Box::new(decoder), Box::new(on_finish));
    Ok(decoder)
}
stub_fn!(feature = "csplat", decode_to_packedsplats);

#[wasm_bindgen]
#[cfg(feature = "gsplat")]
pub fn decode_to_extsplats(
    file_type: Option<String>, path_name: Option<String>,
    sh1_codes: Option<Uint32Array>, sh2_codes: Option<Uint32Array>, sh3_codes: Option<Array>,
) -> Result<ChunkDecoder, JsValue> {
    let file_type = if let Some(file_type) = file_type {
        match SplatFileType::from_enum_str(&file_type) {
            Ok(file_type) => Some(file_type),
            Err(err) => { return Err(JsValue::from(err.to_string())); },
        }
    } else {
        None
    };

    let mut splats = ExtSplatsData::new();
    splats.set_sh_codes(sh1_codes, sh2_codes, sh3_codes);

    let decoder = MultiDecoder::new(splats, file_type, path_name.as_deref());
    let on_finish = |receiver: Box<dyn ChunkReceiver>| {
        let decoder: Box<MultiDecoder<ExtSplatsData>> = receiver.into_any().downcast().unwrap();
        let file_type = decoder.file_type.unwrap();
        let object = decoder.into_splats().into_splat_object();
        Reflect::set(&object, &JsValue::from_str("fileType"), &JsValue::from(file_type.to_enum_str())).unwrap();
        Ok(JsValue::from(object))
    };

    let decoder = ChunkDecoder::new(Box::new(decoder), Box::new(on_finish));
    Ok(decoder)
}
stub_fn!(feature = "gsplat", decode_to_extsplats);

#[wasm_bindgen]
#[allow(non_snake_case)]
#[cfg(feature = "gsplat")]
pub struct GsplatArray {
    pub numSplats: usize,
    pub maxShDegree: usize,
    inner: GsplatArrayInner,
}

#[cfg(feature = "gsplat")]
impl GsplatArray {
    pub fn new(inner: GsplatArrayInner) -> Self {
        Self {
            numSplats: inner.len(),
            maxShDegree: inner.max_sh_degree,
            inner,
        }
    }
}


#[wasm_bindgen]
#[cfg(feature = "gsplat")]
impl GsplatArray {
    /// Sets extra per-Gaussian attributes, carried through LOD
    /// (src/webgpu/attributes): specs [{name, format, components, lodMerge}],
    /// one Float64Array of values per attribute.
    pub fn set_attribs(&mut self, specs: JsValue, columns: Array) -> Result<(), JsValue> {
        self.inner.attribs = attribs_from_js(specs, &columns, self.inner.len())?;
        Ok(())
    }

    /// The attributes' values, merged and ordered like the splats.
    pub fn get_attribs(&self) -> Array {
        attribs_to_js(&self.inner.attribs)
    }

    /// The attributes' specs [{name, format, components, lodMerge}], as
    /// set_attribs takes them: decoded from the file (PLY, .rad) or set.
    pub fn get_attrib_specs(&self) -> Result<JsValue, JsValue> {
        Ok(serde_wasm_bindgen::to_value(&self.inner.attribs.specs)?)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn has_lod(&self) -> bool {
        self.inner.has_lod_tree()
    }

    // pub fn quick_lod(&mut self, lod_base: f32, merge_filter: bool) {
    //     spark_lib::quick_lod::compute_lod_tree(&mut self.inner, lod_base, merge_filter, |s| web_sys::console::log_1(&JsValue::from(s)));
    //     // spark_lib::quick_lod::compute_lod_tree(&mut self.inner, lod_base, merge_filter, |_s| {});
    // }

    #[cfg(feature = "tiny_lod")]
    pub fn tiny_lod(&mut self, lod_base: f32, merge_filter: bool) {
        // let log = |s: &str| web_sys::console::log_1(&JsValue::from(s));
        let log = |_s: &str| {};
        self.inner.remove_invalid();
        spark_lib::tiny_lod::compute_lod_tree(&mut self.inner, lod_base, merge_filter, log);
        self.inner.encode_lod_opacity();
        spark_lib::chunk_tree::chunk_tree(&mut self.inner, 0, log);
    }

    #[cfg(feature = "bhatt_lod")]
    pub fn bhatt_lod(&mut self, lod_base: f32) {
        // let log = |s: &str| web_sys::console::log_1(&JsValue::from(s));
        let log = |_s: &str| {};
        self.inner.remove_invalid();
        spark_lib::bhatt_lod::compute_lod_tree(&mut self.inner, lod_base, log);
        self.inner.encode_lod_opacity();
        spark_lib::chunk_tree::chunk_tree(&mut self.inner, 0, log);
    }

    #[cfg(feature = "csplat")]
    pub fn to_packedsplats(&self, encoding: JsValue) -> Result<Object, JsValue> {
        let encoding = if encoding.is_falsy() {
            None
        } else {
            Some(serde_wasm_bindgen::from_value(encoding)?)
        };
        let splats = match PackedSplatsData::new_from_tsplat_array(&self.inner, encoding) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    #[cfg(feature = "csplat")]
    pub fn to_packedsplats_lod(&self, encoding: JsValue) -> Result<Object, JsValue> {
        let encoding = if encoding.is_falsy() {
            None
        } else {
            Some(serde_wasm_bindgen::from_value(encoding)?)
        };
        let splats = match PackedSplatsData::new_from_tsplat_array_lod(&self.inner, encoding) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    #[cfg(feature = "gsplat")]
    pub fn to_extsplats(&self) -> Result<Object, JsValue> {
        let splats = match ExtSplatsData::new_from_tsplat_array(&self.inner) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    #[cfg(feature = "gsplat")]
    pub fn to_extsplats_lod(&self) -> Result<Object, JsValue> {
        let splats = match ExtSplatsData::new_from_tsplat_array_lod(&self.inner) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    pub fn inject_rgba8(&mut self, rgba: Uint8Array) {
        self.inner.inject_rgba8(&rgba.to_vec());
    }

    pub fn transform(&mut self, transform: JsValue) -> Result<(), JsValue> {
        let transform_options: TransformOptions = serde_wasm_bindgen::from_value(transform)?;
        transform_gsplatarray(&mut self.inner, transform_options);
        Ok(())
    }

    pub fn concat(&mut self, other: &mut GsplatArray) -> Result<(), JsValue> {
        for i in 0..other.inner.len() {
            let sh1 = if other.maxShDegree >= 1 { other.inner.sh1[i].clone() } else { GsplatSH1::default() };
            let sh2 = if other.maxShDegree >= 2 { other.inner.sh2[i].clone() } else { GsplatSH2::default() };
            let sh3 = if other.maxShDegree >= 3 { other.inner.sh3[i].clone() } else { GsplatSH3::default() };
            self.inner.push_splat(other.inner.get(i).clone(), Some(sh1), Some(sh2), Some(sh3));
        }
        Ok(())
    }

    #[cfg(feature = "spz")]
    pub fn encode_to_spz(mut self, max_sh: u32, fractional_bits: u8, version: Option<u32>) -> Result<Uint8Array, JsValue> {
        self.inner.clamp_sh_degree(max_sh as usize);
        self.maxShDegree = self.inner.max_sh_degree;
        let mut encoder = SpzEncoder::new(self.inner).with_max_sh(max_sh as usize).with_fractional_bits(fractional_bits);
        if let Some(version) = version {
            encoder = encoder.with_version(version);
        }
        let encoded = match encoder.encode() {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(encoded) => encoded
        };
        Ok(Uint8Array::from(encoded.as_slice()))
    }
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct JsAttribSpec {
    name: String,
    format: String,
    components: usize,
    lod_merge: Option<String>,
}

fn attribs_from_js(specs: JsValue, columns: &Array, count: usize) -> Result<AttribArray, JsValue> {
    let specs: Vec<JsAttribSpec> = serde_wasm_bindgen::from_value(specs)?;
    let mut attribs = AttribArray::new();
    for (k, spec) in specs.into_iter().enumerate() {
        let lod_merge = spec.lod_merge.as_deref().unwrap_or("weightedMean");
        let Some(lod_merge) = LodMerge::parse(lod_merge) else {
            return Err(JsValue::from(format!("attribute {}: unknown lodMerge {}", spec.name, lod_merge)));
        };
        let values = Float64Array::new(&columns.get(k as u32)).to_vec();
        if values.len() != count * spec.components {
            return Err(JsValue::from(format!("attribute {}: {} values for {} splats", spec.name, values.len(), count)));
        }
        let spec = AttribSpec { name: spec.name, format: spec.format, components: spec.components, lod_merge };
        attribs.add(spec, values).map_err(|err| JsValue::from(err.to_string()))?;
    }
    Ok(attribs)
}

fn attribs_to_js(attribs: &AttribArray) -> Array {
    attribs.columns.iter().map(|col| JsValue::from(Float64Array::from(&col[..]))).collect()
}

#[wasm_bindgen]
#[cfg(feature = "gsplat")]
pub fn decode_to_gsplatarray(file_type: Option<String>, path_name: Option<String>) -> Result<ChunkDecoder, JsValue> {
    let file_type = if let Some(file_type) = file_type {
        match SplatFileType::from_enum_str(&file_type) {
            Ok(file_type) => Some(file_type),
            Err(err) => { return Err(JsValue::from(err.to_string())); },
        }
    } else {
        None
    };

    let splats = GsplatArrayInner::new();
    let decoder = MultiDecoder::new(splats, file_type, path_name.as_deref());
    let on_finish = |receiver: Box<dyn ChunkReceiver>| {
        let decoder: Box<MultiDecoder<GsplatArrayInner>> = receiver.into_any().downcast().unwrap();
        let gsplats = GsplatArray::new(decoder.into_splats());
        Ok(JsValue::from(gsplats))
    };

    let decoder = ChunkDecoder::new(Box::new(decoder), Box::new(on_finish));
    Ok(decoder)
}
stub_fn!(feature = "gsplat", decode_to_gsplatarray);

#[wasm_bindgen]
#[cfg(all(feature = "csplat", feature = "gsplat"))]
pub fn packedsplats_to_gsplatarray(num_splats: u32, packed: Uint32Array, extra: Option<Object>, encoding: JsValue) -> Result<GsplatArray, JsValue> {
    let encoding = if encoding.is_falsy() {
        SplatEncoding::default()
    } else {
        serde_wasm_bindgen::from_value(encoding)?
    };
    let mut receiver = match PackedSplatsData::from_js_arrays(packed, num_splats as usize, extra.as_ref(), encoding) {
        Ok(receiver) => receiver,
        Err(err) => { return Err(JsValue::from(err.to_string())); }
    };
    let splats = match receiver.to_gsplat_array() {
        Ok(inner) => inner,
        Err(err) => { return Err(JsValue::from(err.to_string())); }
    };
    Ok(GsplatArray::new(splats))
}
stub_fn!(all(feature = "csplat", feature = "gsplat"), packedsplats_to_gsplatarray);

#[wasm_bindgen]
#[allow(non_snake_case)]
#[cfg(feature = "csplat")]
pub struct CsplatArray {
    pub numSplats: usize,
    pub maxShDegree: usize,
    inner: CsplatArrayInner,
}

#[cfg(feature = "csplat")]
impl CsplatArray {
    pub fn new(inner: CsplatArrayInner) -> Self {
        Self {
            numSplats: inner.len(),
            maxShDegree: inner.max_sh_degree,
            inner,
        }
    }
}

#[wasm_bindgen]
#[cfg(feature = "csplat")]
impl CsplatArray {
    /// Sets extra per-Gaussian attributes, carried through LOD
    /// (src/webgpu/attributes): specs [{name, format, components, lodMerge}],
    /// one Float64Array of values per attribute.
    pub fn set_attribs(&mut self, specs: JsValue, columns: Array) -> Result<(), JsValue> {
        self.inner.attribs = attribs_from_js(specs, &columns, self.inner.len())?;
        Ok(())
    }

    /// The attributes' values, merged and ordered like the splats.
    pub fn get_attribs(&self) -> Array {
        attribs_to_js(&self.inner.attribs)
    }

    /// The attributes' specs [{name, format, components, lodMerge}], as
    /// set_attribs takes them: decoded from the file (PLY, .rad) or set.
    pub fn get_attrib_specs(&self) -> Result<JsValue, JsValue> {
        Ok(serde_wasm_bindgen::to_value(&self.inner.attribs.specs)?)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn has_lod(&self) -> bool {
        self.inner.has_children()
    }

    #[cfg(feature = "tiny_lod")]
    pub fn tiny_lod(&mut self, lod_base: f32, merge_filter: bool) {
        // let log = |s: &str| web_sys::console::log_1(&JsValue::from(s));
        let log = |_s: &str| {};
        self.inner.remove_invalid();
        spark_lib::tiny_lod::compute_lod_tree(&mut self.inner, lod_base, merge_filter, log);
        self.inner.encode_lod_opacity();
        spark_lib::chunk_tree::chunk_tree(&mut self.inner, 0, log);
    }

    #[cfg(feature = "bhatt_lod")]
    pub fn bhatt_lod(&mut self, lod_base: f32) {
        // let log = |s: &str| web_sys::console::log_1(&JsValue::from(s));
        let log = |_s: &str| {};
        self.inner.remove_invalid();
        spark_lib::bhatt_lod::compute_lod_tree(&mut self.inner, lod_base, log);
        self.inner.encode_lod_opacity();
        spark_lib::chunk_tree::chunk_tree(&mut self.inner, 0, log);
    }

    pub fn to_packedsplats(&self) -> Result<Object, JsValue> {
        let encoding = self.inner.encoding.clone();
        let splats = match PackedSplatsData::new_from_tsplat_array(&self.inner, encoding) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    pub fn to_packedsplats_lod(&self) -> Result<Object, JsValue> {
        let encoding = self.inner.encoding.clone();
        let splats = match PackedSplatsData::new_from_tsplat_array_lod(&self.inner, encoding) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    #[cfg(feature = "gsplat")]
    pub fn to_extsplats(&self) -> Result<Object, JsValue> {
        let splats = match ExtSplatsData::new_from_tsplat_array(&self.inner) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    #[cfg(feature = "gsplat")]
    pub fn to_extsplats_lod(&self) -> Result<Object, JsValue> {
        let splats = match ExtSplatsData::new_from_tsplat_array_lod(&self.inner) {
            Err(err) => { return Err(JsValue::from(err.to_string())); },
            Ok(splats) => splats,
        };
        Ok(splats.into_splat_object())
    }

    pub fn inject_rgba8(&mut self, rgba: Uint8Array) {
        self.inner.inject_rgba8(&rgba.to_vec());
    }
}

#[wasm_bindgen]
#[cfg(feature = "csplat")]
pub fn decode_to_csplatarray(file_type: Option<String>, path_name: Option<String>, encoding: JsValue) -> Result<ChunkDecoder, JsValue> {
    let file_type = if let Some(file_type) = file_type {
        match SplatFileType::from_enum_str(&file_type) {
            Ok(file_type) => Some(file_type),
            Err(err) => { return Err(JsValue::from(err.to_string())); },
        }
    } else {
        None
    };

    let encoding = if encoding.is_falsy() {
        None
    } else {
        Some(serde_wasm_bindgen::from_value(encoding)?)
    };
    let splats = CsplatArrayInner::new_encoding(encoding);
    let decoder = MultiDecoder::new(splats, file_type, path_name.as_deref());
    let on_finish = |receiver: Box<dyn ChunkReceiver>| {
        let decoder: Box<MultiDecoder<CsplatArrayInner>> = receiver.into_any().downcast().unwrap();
        let gsplats = CsplatArray::new(decoder.into_splats());
        Ok(JsValue::from(gsplats))
    };

    let decoder = ChunkDecoder::new(Box::new(decoder), Box::new(on_finish));
    Ok(decoder)
}
stub_fn!(feature = "csplat", decode_to_csplatarray);

#[wasm_bindgen]
#[cfg(feature = "csplat")]
pub fn packedsplats_to_csplatarray(num_splats: u32, packed: Uint32Array, extra: Option<Object>, encoding: JsValue) -> Result<CsplatArray, JsValue> {
    let encoding = if encoding.is_falsy() {
        SplatEncoding::default()
    } else {
        serde_wasm_bindgen::from_value(encoding)?
    };
    let mut receiver = match PackedSplatsData::from_js_arrays(packed, num_splats as usize, extra.as_ref(), encoding) {
        Ok(receiver) => receiver,
        Err(err) => { return Err(JsValue::from(err.to_string())); }
    };
    let splats = match receiver.to_csplat_array() {
        Ok(inner) => inner,
        Err(err) => { return Err(JsValue::from(err.to_string())); }
    };
    Ok(CsplatArray::new(splats))
}

#[wasm_bindgen]
#[cfg(feature = "gsplat")]
pub fn extsplats_to_gsplatarray(num_splats: u32, ext1: Uint32Array, ext2: Uint32Array, extra: Option<Object>) -> Result<GsplatArray, JsValue> {
    let mut receiver = match ExtSplatsData::from_js_arrays([ext1, ext2], num_splats as usize, extra.as_ref()) {
        Ok(receiver) => receiver,
        Err(err) => { return Err(JsValue::from(err.to_string())); }
    };
    let splats = match receiver.to_gsplat_array() {
        Ok(inner) => inner,
        Err(err) => { return Err(JsValue::from(err.to_string())); }
    };
    Ok(GsplatArray::new(splats))
}
stub_fn!(feature = "gsplat", extsplats_to_gsplatarray);

#[wasm_bindgen]
#[cfg(all(feature = "csplat", feature = "tiny_lod"))]
pub fn tiny_lod_packedsplats(num_splats: u32, packed: Uint32Array, extra: Option<Object>, lod_base: f32, merge_filter: bool, rgba: Option<Uint8Array>, encoding: JsValue) -> Result<Object, JsValue> {
    let mut gs = packedsplats_to_csplatarray(num_splats, packed, extra, encoding)?;
    if let Some(rgba) = rgba {
        gs.inject_rgba8(rgba);
    }
    gs.tiny_lod(lod_base, merge_filter);
    gs.to_packedsplats_lod()
}
stub_fn!(all(feature = "csplat", feature = "tiny_lod"), tiny_lod_packedsplats);

#[wasm_bindgen]
#[cfg(all(feature = "csplat", feature = "bhatt_lod"))]
pub fn bhatt_lod_packedsplats(num_splats: u32, packed: Uint32Array, extra: Option<Object>, lod_base: f32, rgba: Option<Uint8Array>, encoding: JsValue) -> Result<Object, JsValue> {
    let mut gs = packedsplats_to_csplatarray(num_splats, packed, extra, encoding)?;
    if let Some(rgba) = rgba {
        gs.inject_rgba8(rgba);
    }
    gs.bhatt_lod(lod_base);
    gs.to_packedsplats_lod()
}
stub_fn!(all(feature = "csplat", feature = "bhatt_lod"), bhatt_lod_packedsplats);

#[wasm_bindgen]
#[cfg(all(feature = "gsplat", feature = "tiny_lod"))]
pub fn tiny_lod_extsplats(num_splats: u32, ext1: Uint32Array, ext2: Uint32Array, extra: Option<Object>, lod_base: f32, merge_filter: bool, rgba: Option<Uint8Array>) -> Result<Object, JsValue> {
    let mut gs = extsplats_to_gsplatarray(num_splats, ext1, ext2, extra)?;
    if let Some(rgba) = rgba {
        gs.inject_rgba8(rgba);
    }
    gs.tiny_lod(lod_base, merge_filter);
    gs.to_extsplats_lod()
}
stub_fn!(all(feature = "gsplat", feature = "tiny_lod"), tiny_lod_extsplats);

#[wasm_bindgen]
#[cfg(all(feature = "gsplat", feature = "bhatt_lod"))]
pub fn bhatt_lod_extsplats(num_splats: u32, ext1: Uint32Array, ext2: Uint32Array, extra: Option<Object>, lod_base: f32, rgba: Option<Uint8Array>) -> Result<Object, JsValue> {
    let mut gs = extsplats_to_gsplatarray(num_splats, ext1, ext2, extra)?;
    if let Some(rgba) = rgba {
        gs.inject_rgba8(rgba);
    }
    gs.bhatt_lod(lod_base);
    gs.to_extsplats_lod()
}
stub_fn!(all(feature = "gsplat", feature = "bhatt_lod"), bhatt_lod_extsplats);

const RAYCAST_BUFFER_COUNT: usize = 65536;

thread_local! {
    static RAYCAST_BUFFERS: RefCell<(Vec<u32>, Vec<u32>, Vec<f32>)> = RefCell::new((vec![0; RAYCAST_BUFFER_COUNT * 4], vec![0; RAYCAST_BUFFER_COUNT * 4], vec![0.0; RAYCAST_BUFFER_COUNT]));
}

#[wasm_bindgen]
pub fn get_raycast_buffer() -> Uint32Array {
    RAYCAST_BUFFERS.with_borrow_mut(|(buffer, _, _)| {
        unsafe { Uint32Array::view(buffer) }
    })
}

#[wasm_bindgen]
pub fn get_raycast_buffer2() -> Uint32Array {
    RAYCAST_BUFFERS.with_borrow_mut(|(_, buffer, _)| {
        unsafe { Uint32Array::view(buffer) }
    })
}

#[wasm_bindgen]
pub fn raycast_packed_buffer(
    origin_x: f32, origin_y: f32, origin_z: f32,
    dir_x: f32, dir_y: f32, dir_z: f32,
    min_opacity: f32, near: f32, far: f32,
    count: u32,
    ln_scale_min: f32, ln_scale_max: f32, lod_opacity: bool,
) -> Float32Array {
    RAYCAST_BUFFERS.with_borrow_mut(|(buffer, _, distances)| {
        let encoding = SplatEncoding {
            ln_scale_min,
            ln_scale_max,
            lod_opacity,
            ..Default::default()
        };

        distances.clear();
        let subbuffer = &buffer[0..(4 * count as usize)];
        raycast_packed_ellipsoids(
            subbuffer, distances,
            [origin_x, origin_y, origin_z], [dir_x, dir_y, dir_z],
            min_opacity, near, far, &encoding,
        );

        unsafe { Float32Array::view(distances) }
    })
}

#[wasm_bindgen]
pub fn raycast_ext_buffers(
    origin_x: f32, origin_y: f32, origin_z: f32,
    dir_x: f32, dir_y: f32, dir_z: f32,
    min_opacity: f32, near: f32, far: f32,
    count: u32,
) -> Float32Array {
    RAYCAST_BUFFERS.with_borrow_mut(|(buffer, buffer2, distances)| {
        distances.clear();
        let subbuffer = &buffer[0..(4 * count as usize)];
        let subbuffer2 = &buffer2[0..(4 * count as usize)];
        raycast_ext_ellipsoids(
            subbuffer, subbuffer2, distances,
            [origin_x, origin_y, origin_z], [dir_x, dir_y, dir_z],
            min_opacity, near, far,
        );

        unsafe { Float32Array::view(distances) }
    })
}

#[wasm_bindgen]
pub fn raycast_packed_splats(
    origin_x: f32, origin_y: f32, origin_z: f32,
    dir_x: f32, dir_y: f32, dir_z: f32,
    min_opacity: f32, near: f32, far: f32,
    num_splats: u32, packed_splats: Uint32Array,
    ln_scale_min: f32, ln_scale_max: f32, lod_opacity: bool,
) -> Float32Array {
    let mut distances = Vec::<f32>::new();
    let encoding = SplatEncoding {
        ln_scale_min,
        ln_scale_max,
        lod_opacity,
        ..Default::default()
    };

    RAYCAST_BUFFERS.with_borrow_mut(|(buffer, _, _)| {
        let mut base = 0;
        while base < num_splats {
            let chunk_size = (RAYCAST_BUFFER_COUNT as u32).min(num_splats - base);
            let subarray = packed_splats.subarray(4 * base, 4 * (base + chunk_size));
            let subbuffer = &mut buffer[0..(4 * chunk_size as usize)];
            subarray.copy_to(subbuffer);

            raycast_packed_ellipsoids(
                subbuffer, &mut distances,
                [origin_x, origin_y, origin_z], [dir_x, dir_y, dir_z],
                min_opacity, near, far, &encoding,
            );

            base += chunk_size;
        }
    });

    let output = Float32Array::new_with_length(distances.len() as u32);
    output.copy_from(&distances);
    output
}

#[wasm_bindgen]
#[cfg(feature = "rad")]
pub fn decode_rad_header(bytes: Uint8Array) -> Result<JsValue, JsValue> {
    let bytes = bytes.to_vec();
    let meta_chunks_start = match spark_lib::rad::decode_rad_header(&bytes) {
        Ok(meta_chunks_start) => meta_chunks_start,
        Err(err) => { return Err(JsValue::from(err.to_string())); }
    };
    if let Some((meta, chunks_start)) = meta_chunks_start {
        let object = js_sys::Object::new();
        Reflect::set(&object, &JsValue::from_str("meta"), &serde_wasm_bindgen::to_value(&meta)?)?;
        Reflect::set(&object, &JsValue::from_str("chunksStart"), &JsValue::from_f64(chunks_start as f64))?;
        Ok(JsValue::from(object))
    } else {
        Ok(JsValue::null())
    }
}
stub_fn!(feature = "rad", decode_rad_header);

/// The bytes of a .athc that `athc_layout` needs (through its tables), from
/// at least its first page; throws when `bytes` is not a .athc header.
#[wasm_bindgen]
#[cfg(feature = "athc")]
pub fn athc_prefix_bytes(bytes: Uint8Array) -> Result<f64, JsValue> {
    let bytes = bytes.to_vec();
    if bytes.len() >= 4 && u32::from_le_bytes(bytes[..4].try_into().unwrap()) == spark_lib::athc_v3::ATH3_MAGIC {
        // Version 3: through the starts (the header, sections and block index).
        return spark_lib::athc_v3::tables_bytes(&bytes).map(|n| n as f64).map_err(|e| JsValue::from(e.to_string()));
    }
    spark_lib::athc::AthcLayout::prefix_bytes(&bytes)
        .map(|n| n as f64)
        .map_err(|e| JsValue::from(e.to_string()))
}
stub_fn!(feature = "athc", athc_prefix_bytes);

/// A .athc v3 CPCA transfer section (encoding 3, after its gunzip) back to
/// its words (athc_cpca::decode_cpca), for the streams a chunk page fetches
/// on the main thread (src/athc.ts decodeAthcSection): the same halves the
/// TypeScript decoder gives, several times faster.
#[wasm_bindgen]
#[cfg(feature = "athc")]
pub fn athc_decode_cpca(stored: &[u8], n: u32, words: u32) -> Result<Uint8Array, JsValue> {
    let raw = spark_lib::athc_cpca::decode_cpca(stored, n as usize, words as usize).map_err(|e| JsValue::from(e.to_string()))?;
    Ok(Uint8Array::from(&raw[..]))
}
stub_fn!(feature = "athc", athc_decode_cpca);

/// A .athc's headers and tables (AthcLayout) plus `levelsEnd`, the bytes a
/// stream reads before any chunk, from the file's first bytes.
#[wasm_bindgen]
#[cfg(feature = "athc")]
pub fn athc_layout(prefix: Uint8Array, file_bytes: f64) -> Result<JsValue, JsValue> {
    let bytes = prefix.to_vec();
    if bytes.len() >= 4 && u32::from_le_bytes(bytes[..4].try_into().unwrap()) == spark_lib::athc_v3::ATH3_MAGIC {
        // Version 3: its sections and block index, the v2 headers an ATHV
        // page carries, and the attributes as a v2 file of the same cloud.
        let layout = spark_lib::athc_v3::parse_v3(&bytes).map_err(|e| JsValue::from(e.to_string()))?;
        let object = serde_wasm_bindgen::to_value(&layout)?;
        let set = |k: &str, v: &JsValue| Reflect::set(&object, &JsValue::from_str(k), v);
        set("version", &JsValue::from_f64(3.0))?;
        set("fileBytes", &JsValue::from_f64(file_bytes))?;
        set("headers", &Uint8Array::from(&layout.v2_headers()[..]))?;
        let attribs = spark_lib::athc::attrib_specs(&layout.header, &layout.extra);
        set("attribSpecs", &serde_wasm_bindgen::to_value(&attribs)?)?;
        let forms: Vec<u32> = if layout.extra.transfer_words > 0 {
            spark_lib::athc_v3::transfer_forms(layout.extra.transfer_count)
        } else {
            Vec::new()
        };
        set("transferForms", &serde_wasm_bindgen::to_value(&forms)?)?;
        return Ok(object);
    }
    let layout = spark_lib::athc::AthcLayout::parse(&bytes, file_bytes as u64)
        .map_err(|e| JsValue::from(e.to_string()))?;
    let object = serde_wasm_bindgen::to_value(&layout)?;
    Reflect::set(&object, &JsValue::from_str("levelsEnd"), &JsValue::from_f64(layout.levels_end() as f64))?;
    let attribs = spark_lib::athc::attrib_specs(&layout.header, &layout.extra);
    Reflect::set(&object, &JsValue::from_str("attribSpecs"), &serde_wasm_bindgen::to_value(&attribs)?)?;
    Ok(object)
}
stub_fn!(feature = "athc", athc_layout);

/// The skeleton of a skinned .athc v3 (athc_skin.rs), from the file's
/// bytes through its tables (`athc_prefix_bytes`): { influences,
/// gradientWords, joints, skeleton, geomBind (USD rows), clips: [{ name,
/// timeCodesPerSecond, times: Float32Array, xforms: Float32Array (samples x
/// joints x 16, USD rows) }] }, or null when the cloud has none.
#[wasm_bindgen]
#[cfg(feature = "athc")]
pub fn athc_skeleton(prefix: Uint8Array) -> Result<JsValue, JsValue> {
    let bytes = prefix.to_vec();
    let Some(s) =
        spark_lib::athc_v3::read_v3_skeleton(&bytes).map_err(|e| JsValue::from(e.to_string()))?
    else {
        return Ok(JsValue::NULL);
    };
    let object = Object::new();
    let set = |o: &Object, k: &str, v: &JsValue| Reflect::set(o, &JsValue::from_str(k), v);
    set(
        &object,
        "influences",
        &JsValue::from_f64(s.influences as f64),
    )?;
    set(
        &object,
        "gradientWords",
        &JsValue::from_f64(s.gradient_words as f64),
    )?;
    set(&object, "joints", &serde_wasm_bindgen::to_value(&s.joints)?)?;
    set(&object, "skeleton", &JsValue::from_str(&s.skeleton))?;
    set(&object, "geomBind", &Float32Array::from(&s.geom_bind[..]))?;
    let clips = js_sys::Array::new();
    for c in &s.clips {
        let clip = Object::new();
        set(&clip, "name", &JsValue::from_str(&c.name))?;
        set(
            &clip,
            "timeCodesPerSecond",
            &JsValue::from_f64(c.time_codes_per_second as f64),
        )?;
        set(&clip, "times", &Float32Array::from(&c.times[..]))?;
        set(&clip, "xforms", &Float32Array::from(&c.xforms[..]))?;
        clips.push(&clip);
    }
    set(&object, "clips", &clips)?;
    Ok(object.into())
}
stub_fn!(feature = "athc", athc_skeleton);

/// The bytes of a .athl through its section table, from at least its
/// first 128 bytes (docs/docs/athl.md).
#[wasm_bindgen]
#[cfg(feature = "athc")]
pub fn athl_prefix_bytes(bytes: Uint8Array) -> Result<f64, JsValue> {
    spark_lib::athl::AthlHeader::prefix_bytes(&bytes.to_vec())
        .map(|n| n as f64)
        .map_err(|e| JsValue::from(e.to_string()))
}
stub_fn!(feature = "athc", athl_prefix_bytes);

/// A .athl's header and section table, from its prefix: where each
/// group's layers for each page are, for Range requests.
#[wasm_bindgen]
#[cfg(feature = "athc")]
pub fn athl_header(prefix: Uint8Array) -> Result<JsValue, JsValue> {
    let h = spark_lib::athl::AthlHeader::parse(&prefix.to_vec()).map_err(|e| JsValue::from(e.to_string()))?;
    Ok(serde_wasm_bindgen::to_value(&h)?)
}
stub_fn!(feature = "athc", athl_header);

#[cfg(feature = "athc")]
fn athl_layer_object(l: &spark_lib::athl::AthlLayer) -> Result<JsValue, JsValue> {
    let object = serde_wasm_bindgen::to_value(l)?;
    Reflect::set(&object, &JsValue::from_str("blocks"), &Uint16Array::from(&l.blocks[..]))?;
    Reflect::set(&object, &JsValue::from_str("data"), &Uint16Array::from(&l.data[..]))?;
    Ok(object)
}

/// One LAYR section of a .athl (the bytes a Range request of it returns):
/// { group, kind, chunk, components, blocks: Uint16Array, data: Uint16Array }.
#[wasm_bindgen]
#[cfg(feature = "athc")]
pub fn decode_athl_layer(section: Uint8Array) -> Result<JsValue, JsValue> {
    let l = spark_lib::athl::AthlLayer::read(&section.to_vec()).map_err(|e| JsValue::from(e.to_string()))?;
    athl_layer_object(&l)
}
stub_fn!(feature = "athc", decode_athl_layer);

/// The bytes of a .athl `decode_athl_meta` needs, from its prefix.
#[wasm_bindgen]
#[cfg(feature = "athc")]
pub fn athl_meta_bytes(prefix: Uint8Array) -> Result<f64, JsValue> {
    let h = spark_lib::athl::AthlHeader::parse(&prefix.to_vec()).map_err(|e| JsValue::from(e.to_string()))?;
    Ok(spark_lib::athl::AthlFile::meta_bytes(&h) as f64)
}
stub_fn!(feature = "athc", athl_meta_bytes);

/// A .athl without its layers (as decode_athl, `layers` empty), from its
/// first `athl_meta_bytes` bytes: what a paged reader starts from.
#[wasm_bindgen]
#[cfg(feature = "athc")]
pub fn decode_athl_meta(bytes: Uint8Array) -> Result<JsValue, JsValue> {
    athl_object(&bytes.to_vec(), false)
}
stub_fn!(feature = "athc", decode_athl_meta);

/// A whole .athl: { header, groups, polygons, profiles (texels as a
/// Uint16Array of f16 bits), layers (as decode_athl_layer) }.
#[wasm_bindgen]
#[cfg(feature = "athc")]
pub fn decode_athl(bytes: Uint8Array) -> Result<JsValue, JsValue> {
    athl_object(&bytes.to_vec(), true)
}
stub_fn!(feature = "athc", decode_athl);

#[cfg(feature = "athc")]
fn athl_object(bytes: &[u8], with_layers: bool) -> Result<JsValue, JsValue> {
    let header = spark_lib::athl::AthlHeader::parse(bytes).map_err(|e| JsValue::from(e.to_string()))?;
    let file = if with_layers {
        spark_lib::athl::AthlFile::read(bytes)
    } else {
        spark_lib::athl::AthlFile::read_meta(bytes)
    }
    .map_err(|e| JsValue::from(e.to_string()))?;
    spark_lib::athl::validate(&file).map_err(|e| JsValue::from(e.to_string()))?;
    let object = js_sys::Object::new();
    let set = |k: &str, v: &JsValue| Reflect::set(&object, &JsValue::from_str(k), v);
    set("header", &serde_wasm_bindgen::to_value(&header)?)?;
    set("groups", &serde_wasm_bindgen::to_value(&file.groups)?)?;
    set("polygons", &serde_wasm_bindgen::to_value(&file.polygons)?)?;
    let profiles = Array::new();
    for p in &file.profiles {
        let o = serde_wasm_bindgen::to_value(p)?;
        Reflect::set(&o, &JsValue::from_str("texels"), &Uint16Array::from(&p.texels[..]))?;
        profiles.push(&o);
    }
    set("profiles", &profiles)?;
    let layers = Array::new();
    for l in &file.layers {
        layers.push(&athl_layer_object(l)?);
    }
    set("layers", &layers)?;
    Ok(JsValue::from(object))
}

/// The paged virtual tree of a .athc (spark-lib athc::VirtualTree, splats
/// from the first page boundary after the merged nodes) and its merged
/// pages as ATHV blobs, from the file's bytes through `levelsEnd`.
#[wasm_bindgen]
#[cfg(feature = "athc")]
pub fn athc_merged_pages(prefix: Uint8Array, file_bytes: f64) -> Result<JsValue, JsValue> {
    let (tree, pages) = spark_lib::athc::athv_merged_pages(&prefix.to_vec(), file_bytes as u64)
        .map_err(|e| JsValue::from(e.to_string()))?;
    let object = serde_wasm_bindgen::to_value(&tree)?;
    let array = Array::new();
    for page in pages {
        array.push(&Uint8Array::from(&page[..]));
    }
    Reflect::set(&object, &JsValue::from_str("pages"), &array)?;
    Ok(object)
}
stub_fn!(feature = "athc", athc_merged_pages);

/// A v3 .athc's paged virtual tree and merged pages, from its tables
/// (`athc_prefix_bytes`) and its levels' blocks as kind-2 ATHV pages
/// (spark-lib athc_v3::athv_merged_pages_v3), plus `headers`: the v2 headers
/// of the streams those pages keep.
#[wasm_bindgen]
#[cfg(feature = "athc")]
pub fn athc3_merged_pages(tables: Uint8Array, levels: Array) -> Result<JsValue, JsValue> {
    let levels: Vec<Vec<u8>> = levels.iter().map(|v| Uint8Array::from(v).to_vec()).collect();
    let refs: Vec<&[u8]> = levels.iter().map(|v| &v[..]).collect();
    let (tree, pages, headers) = spark_lib::athc_v3::athv_merged_pages_v3(&tables.to_vec(), &refs)
        .map_err(|e| JsValue::from(e.to_string()))?;
    let object = serde_wasm_bindgen::to_value(&tree)?;
    let array = Array::new();
    for page in pages {
        array.push(&Uint8Array::from(&page[..]));
    }
    Reflect::set(&object, &JsValue::from_str("pages"), &array)?;
    Reflect::set(&object, &JsValue::from_str("headers"), &Uint8Array::from(&headers[..]))?;
    Ok(object)
}
stub_fn!(feature = "athc", athc3_merged_pages);
