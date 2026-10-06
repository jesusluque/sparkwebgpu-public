//! A decoder's per-Gaussian attributes on their way to JS (attribSpecs,
//! attribColumns): values (f64, as spark-lib's AttribArray keeps them) or,
//! where the decoder hands them over as they are stored (`.athc` streams:
//! u32 words, f16 two to a word), the packed words themselves, which JS
//! takes as an AttribPool column without packing (spec `packed: true`).
//!
//! A whole .athc with a full TX transfer is 112 halves a splat: as f64 that
//! is 896 bytes a splat, 2.4 GB for the Corvette's paint, more than a wasm32
//! heap holds next to the file (the decoder aborted, "unreachable"); as its
//! words, 224. Columns are allocated when first written.

use js_sys::{Array, Float64Array, Object, Reflect, Uint32Array};
use spark_lib::attrib::{AttribArray, AttribSpec};
use wasm_bindgen::JsValue;

#[derive(Default)]
pub struct DecodedAttribs {
    specs: Vec<AttribSpec>,
    count: usize,
    values: Vec<Vec<f64>>,
    words: Vec<Option<Vec<u32>>>,
}

impl DecodedAttribs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Values computed elsewhere (a LoD build's AttribArray).
    pub fn from_array(attribs: &AttribArray) -> Self {
        Self {
            specs: attribs.specs.clone(),
            count: attribs.len(),
            values: attribs.columns.clone(),
            words: vec![None; attribs.specs.len()],
        }
    }

    pub fn init(&mut self, specs: &[AttribSpec], count: usize) {
        self.specs = specs.to_vec();
        self.count = count;
        self.values = vec![Vec::new(); specs.len()];
        self.words = vec![None; specs.len()];
    }

    pub fn add(&mut self, spec: &AttribSpec, count: usize) {
        self.count = count;
        self.specs.push(spec.clone());
        self.values.push(Vec::new());
        self.words.push(None);
    }

    pub fn set_values(&mut self, attrib: usize, base: usize, count: usize, values: &[f64]) {
        let c = self.specs[attrib].components;
        let col = &mut self.values[attrib];
        if col.is_empty() {
            col.resize(self.count * c, 0.0);
        }
        col[base * c..(base + count) * c].copy_from_slice(&values[..count * c]);
    }

    pub fn set_words(&mut self, attrib: usize, base: usize, count: usize, words: &[u32]) {
        if count == 0 {
            return;
        }
        let per = words.len() / count;
        let total = self.count * per;
        let col = self.words[attrib].get_or_insert_with(|| vec![0; total]);
        col[base * per..(base + count) * per].copy_from_slice(&words[..count * per]);
    }

    /// attribSpecs and attribColumns on `object` (nothing without attributes).
    pub fn set_on(&self, object: &Object) {
        if self.specs.is_empty() {
            return;
        }
        let Ok(specs) = serde_wasm_bindgen::to_value(&self.specs) else {
            return;
        };
        let specs = Array::from(&specs);
        let columns = Array::new();
        for (k, spec) in self.specs.iter().enumerate() {
            if let Some(words) = &self.words[k] {
                Reflect::set(&specs.get(k as u32), &JsValue::from_str("packed"), &JsValue::TRUE).unwrap();
                columns.push(&Uint32Array::from(&words[..]));
            } else if self.values[k].is_empty() {
                columns.push(&Float64Array::new_with_length((self.count * spec.components) as u32));
            } else {
                columns.push(&Float64Array::from(&self.values[k][..]));
            }
        }
        Reflect::set(object, &JsValue::from_str("attribSpecs"), &specs).unwrap();
        Reflect::set(object, &JsValue::from_str("attribColumns"), &columns).unwrap();
    }
}
