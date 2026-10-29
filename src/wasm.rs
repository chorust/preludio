use crate::error::PreludioError;
use crate::forecast::{finish_forecast_from_frames, ExecutionInfo};
use crate::manifest::{Manifest, ModelPackage};
use crate::observation::{FrameMeta, ObservationSequence};
use js_sys::Float32Array;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

fn to_js_err(err: PreludioError) -> JsValue {
    serde_wasm_bindgen::to_value(&err)
        .unwrap_or_else(|_| JsValue::from_str(&format!("serialization error: {:?}", err)))
}

#[derive(Serialize, Deserialize)]
struct SerializedPackage {
    manifest: Manifest,
    model_bytes: Vec<u8>,
}

#[wasm_bindgen(js_name = packageFromParts)]
pub fn package_from_parts(manifest_json: &str, model_bytes: &[u8]) -> Result<JsValue, JsValue> {
    let pkg = ModelPackage::from_parts(manifest_json, model_bytes.to_vec()).map_err(to_js_err)?;
    let serialized = SerializedPackage {
        manifest: pkg.manifest,
        model_bytes: Vec::new(),
    };
    serde_wasm_bindgen::to_value(&serialized).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn validate(manifest_val: JsValue, seq_val: JsValue) -> Result<(), JsValue> {
    let manifest: Manifest = serde_wasm_bindgen::from_value(manifest_val)
        .map_err(|e| JsValue::from_str(&format!("invalid manifest: {}", e)))?;
    let seq: ObservationSequence = serde_wasm_bindgen::from_value(seq_val)
        .map_err(|e| JsValue::from_str(&format!("invalid sequence: {}", e)))?;

    crate::validate::validate(&manifest, &seq).map_err(to_js_err)
}

#[wasm_bindgen(js_name = validateTimes)]
pub fn validate_times(manifest_val: JsValue, frames_val: JsValue) -> Result<(), JsValue> {
    let manifest: Manifest = serde_wasm_bindgen::from_value(manifest_val)
        .map_err(|e| JsValue::from_str(&format!("invalid manifest: {}", e)))?;
    let frames: Vec<FrameMeta> = serde_wasm_bindgen::from_value(frames_val)
        .map_err(|e| JsValue::from_str(&format!("invalid frames: {}", e)))?;

    crate::validate::validate_times(&manifest, &frames).map_err(to_js_err)
}

#[wasm_bindgen]
pub fn preprocess(manifest_val: JsValue, seq_val: JsValue) -> Result<Float32Array, JsValue> {
    let manifest: Manifest = serde_wasm_bindgen::from_value(manifest_val)
        .map_err(|e| JsValue::from_str(&format!("invalid manifest: {}", e)))?;
    let seq: ObservationSequence = serde_wasm_bindgen::from_value(seq_val)
        .map_err(|e| JsValue::from_str(&format!("invalid sequence: {}", e)))?;

    let output = crate::preprocess::preprocess(&manifest, &seq).map_err(to_js_err)?;
    Ok(Float32Array::from(output.as_slice()))
}

#[wasm_bindgen(js_name = finishForecast)]
pub fn finish_forecast(
    pkg_val: JsValue,
    frames_val: JsValue,
    raw: &[f32],
    execution_val: JsValue,
    generated_at: i64,
) -> Result<JsValue, JsValue> {
    let serialized: SerializedPackage = serde_wasm_bindgen::from_value(pkg_val)
        .map_err(|e| JsValue::from_str(&format!("invalid package: {}", e)))?;
    let pkg = ModelPackage {
        manifest: serialized.manifest,
        model_bytes: serialized.model_bytes,
        package_dir: None,
    };
    let frames: Vec<FrameMeta> = serde_wasm_bindgen::from_value(frames_val)
        .map_err(|e| JsValue::from_str(&format!("invalid frames: {}", e)))?;
    let execution: ExecutionInfo = serde_wasm_bindgen::from_value(execution_val)
        .map_err(|e| JsValue::from_str(&format!("invalid execution: {}", e)))?;

    let forecast =
        finish_forecast_from_frames(&pkg, &frames, raw.to_vec(), execution, generated_at)
            .map_err(to_js_err)?;

    let obj = js_sys::Object::new();
    let values_js = Float32Array::from(&forecast.values[..]);
    let meta_js = serde_wasm_bindgen::to_value(&forecast.metadata)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    js_sys::Reflect::set(&obj, &JsValue::from_str("values"), &values_js)
        .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;
    js_sys::Reflect::set(&obj, &JsValue::from_str("metadata"), &meta_js)
        .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;

    Ok(obj.into())
}
