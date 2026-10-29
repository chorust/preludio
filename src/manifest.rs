use crate::error::{ModelUnavailable, PreludioError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::path::PathBuf;

pub const SCHEMA_V1: &str = "preludio.model-package/1";

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub id: String,
    pub model: ModelMeta,
    pub origin: ModelOrigin,
    pub observation: ModelObservation,
    pub tensor_input: TensorMeta,
    pub tensor_output: TensorMeta,
    pub preprocessing: Vec<PreprocessingStep>,
    pub forecast: ModelForecast,
    pub supported_targets: Vec<SupportedTarget>,
}

impl<'de> Deserialize<'de> for Manifest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct ManifestFields {
            schema: String,
            id: String,
            model: ModelMeta,
            origin: ModelOrigin,
            observation: ModelObservation,
            tensor_input: TensorMeta,
            tensor_output: TensorMeta,
            preprocessing: Vec<PreprocessingStep>,
            forecast: ModelForecast,
            supported_targets: Vec<SupportedTarget>,
        }

        let fields = ManifestFields::deserialize(deserializer)?;
        let manifest = Self {
            schema: fields.schema,
            id: fields.id,
            model: fields.model,
            origin: fields.origin,
            observation: fields.observation,
            tensor_input: fields.tensor_input,
            tensor_output: fields.tensor_output,
            preprocessing: fields.preprocessing,
            forecast: fields.forecast,
            supported_targets: fields.supported_targets,
        };
        validate_manifest(&manifest).map_err(<D::Error as serde::de::Error>::custom)?;
        Ok(manifest)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelMeta {
    pub file: String,
    pub sha256: String,
    pub format: String,
    pub opset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelOrigin {
    pub checkpoint: CheckpointMeta,
    pub source_sha256: String,
    pub config_sha256: String,
    pub exporter: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointMeta {
    pub file: String,
    pub step: i64,
    pub sha256: String,
    pub run: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelObservation {
    pub frames: usize,
    pub cadence_seconds: i64,
    pub cadence_tolerance_seconds: i64,
    pub encoding: String,
    pub grid: GridMeta,
    pub missing_policy: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GridMeta {
    pub id: String,
    pub shape: [usize; 2],
    pub georeferenced: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crs: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub affine: Option<[f64; 6]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub axes: Vec<String>,
    pub range: [f64; 2],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "op")]
pub enum PreprocessingStep {
    #[serde(rename = "u8_div_255")]
    U8Div255,
    #[serde(rename = "resize_bilinear_aa")]
    ResizeBilinearAa {
        height: usize,
        width: usize,
        #[serde(default)]
        align_corners: bool,
    },
    #[serde(rename = "round_f16")]
    RoundF16,
}

impl<'de> Deserialize<'de> for PreprocessingStep {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Empty struct variants reject extra fields; serde unit variants ignore them.
        #[derive(Deserialize)]
        #[serde(tag = "op", deny_unknown_fields)]
        enum Step {
            #[serde(rename = "u8_div_255")]
            U8Div255 {},
            #[serde(rename = "resize_bilinear_aa")]
            ResizeBilinearAa {
                height: usize,
                width: usize,
                #[serde(default)]
                align_corners: bool,
            },
            #[serde(rename = "round_f16")]
            RoundF16 {},
        }

        Ok(match Step::deserialize(deserializer)? {
            Step::U8Div255 {} => Self::U8Div255,
            Step::ResizeBilinearAa {
                height,
                width,
                align_corners,
            } => Self::ResizeBilinearAa {
                height,
                width,
                align_corners,
            },
            Step::RoundF16 {} => Self::RoundF16,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelForecast {
    pub lead_minutes: Vec<u32>,
    pub encoding: String,
    pub grid: GridMeta,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupportedTarget {
    pub id: String,
    pub platform: String,
    pub engine: String,
    pub engine_version: String,
    pub provider: String,
    pub threads: TargetThreads,
    pub requires_cross_origin_isolation: bool,
    pub validation: String,
    pub measurement: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetThreads {
    Default,
    Fixed(u32),
}

impl Serialize for TargetThreads {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Default => serializer.serialize_str("default"),
            Self::Fixed(threads) => serializer.serialize_u32(*threads),
        }
    }
}

impl<'de> Deserialize<'de> for TargetThreads {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct TargetThreadsVisitor;

        impl<'visitor> serde::de::Visitor<'visitor> for TargetThreadsVisitor {
            type Value = TargetThreads;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("`default` or a positive integer thread count")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if value == "default" {
                    Ok(TargetThreads::Default)
                } else {
                    Err(E::invalid_value(serde::de::Unexpected::Str(value), &self))
                }
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                let threads = u32::try_from(value)
                    .map_err(|_| E::custom("thread count does not fit in u32"))?;
                if threads == 0 {
                    return Err(E::invalid_value(
                        serde::de::Unexpected::Unsigned(value),
                        &self,
                    ));
                }
                Ok(TargetThreads::Fixed(threads))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                let threads = u32::try_from(value)
                    .map_err(|_| E::custom("thread count must be a positive u32"))?;
                if threads == 0 {
                    return Err(E::invalid_value(
                        serde::de::Unexpected::Signed(value),
                        &self,
                    ));
                }
                Ok(TargetThreads::Fixed(threads))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if value.is_finite()
                    && value.fract() == 0.0
                    && value >= 1.0
                    && value <= u32::MAX as f64
                {
                    Ok(TargetThreads::Fixed(value as u32))
                } else {
                    Err(E::invalid_value(serde::de::Unexpected::Float(value), &self))
                }
            }
        }

        deserializer.deserialize_any(TargetThreadsVisitor)
    }
}

fn validate_manifest(manifest: &Manifest) -> Result<(), String> {
    if manifest.schema != SCHEMA_V1 {
        return Err(format!("schema must be '{SCHEMA_V1}'"));
    }
    if manifest.id.is_empty() {
        return Err("id must contain at least one character".to_string());
    }
    if manifest.model.format != "onnx" {
        return Err("model.format must be 'onnx'".to_string());
    }
    if manifest.model.opset == 0 {
        return Err("model.opset must be at least 1".to_string());
    }
    validate_sha256("model.sha256", &manifest.model.sha256)?;
    validate_sha256("origin.source_sha256", &manifest.origin.source_sha256)?;
    validate_sha256("origin.config_sha256", &manifest.origin.config_sha256)?;
    validate_sha256(
        "origin.checkpoint.sha256",
        &manifest.origin.checkpoint.sha256,
    )?;

    if manifest.observation.frames == 0 {
        return Err("observation.frames must be at least 1".to_string());
    }
    if manifest.observation.cadence_seconds < 1 {
        return Err("observation.cadence_seconds must be at least 1".to_string());
    }
    if manifest.observation.cadence_tolerance_seconds < 0 {
        return Err("observation.cadence_tolerance_seconds must be at least 0".to_string());
    }
    validate_encoding("observation.encoding", &manifest.observation.encoding)?;
    validate_grid("observation.grid", &manifest.observation.grid)?;
    if manifest.observation.missing_policy != "reject" {
        return Err("observation.missing_policy must be 'reject'".to_string());
    }

    validate_tensor("tensor_input", &manifest.tensor_input)?;
    validate_tensor("tensor_output", &manifest.tensor_output)?;
    if manifest.preprocessing.is_empty() {
        return Err("preprocessing must contain at least one step".to_string());
    }
    for (index, step) in manifest.preprocessing.iter().enumerate() {
        if let PreprocessingStep::ResizeBilinearAa {
            height,
            width,
            align_corners,
        } = step
        {
            if *height == 0 || *width == 0 {
                return Err(format!(
                    "preprocessing[{index}] dimensions must be at least 1"
                ));
            }
            if *align_corners {
                return Err(format!(
                    "preprocessing[{index}].align_corners must be false"
                ));
            }
        }
    }

    if manifest.forecast.lead_minutes.is_empty() || manifest.forecast.lead_minutes.contains(&0) {
        return Err("forecast.lead_minutes must contain positive values".to_string());
    }
    validate_encoding("forecast.encoding", &manifest.forecast.encoding)?;
    validate_grid("forecast.grid", &manifest.forecast.grid)?;

    for (index, target) in manifest.supported_targets.iter().enumerate() {
        if target.engine != "onnxruntime" && target.engine != "onnxruntime-web" {
            return Err(format!(
                "supported_targets[{index}].engine is not supported"
            ));
        }
        if target.provider != "cpu" && target.provider != "wasm" {
            return Err(format!(
                "supported_targets[{index}].provider is not supported"
            ));
        }
        if matches!(target.threads, TargetThreads::Fixed(0)) {
            return Err(format!(
                "supported_targets[{index}].threads must be positive"
            ));
        }
    }
    Ok(())
}

fn validate_sha256(path: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "{path} must be 64 lowercase hexadecimal characters"
        ));
    }
    Ok(())
}

fn validate_encoding(path: &str, value: &str) -> Result<(), String> {
    if value != "training-grayscale-u8" && value != "training-grayscale-normalized" {
        return Err(format!("{path} has an unsupported encoding"));
    }
    Ok(())
}

fn validate_grid(path: &str, grid: &GridMeta) -> Result<(), String> {
    if grid.shape.contains(&0) {
        return Err(format!("{path}.shape dimensions must be at least 1"));
    }
    if grid.georeferenced && (grid.crs.is_none() || grid.affine.is_none()) {
        return Err(format!(
            "{path} requires crs and affine when georeferenced is true"
        ));
    }
    Ok(())
}

fn validate_tensor(path: &str, tensor: &TensorMeta) -> Result<(), String> {
    if tensor.dtype != "float32" {
        return Err(format!("{path}.dtype must be 'float32'"));
    }
    if tensor.shape.contains(&0) {
        return Err(format!("{path}.shape dimensions must be at least 1"));
    }
    if tensor.axes.iter().any(|axis| {
        axis != "batch" && axis != "time" && axis != "lead_time" && axis != "y" && axis != "x"
    }) {
        return Err(format!("{path}.axes contains an unsupported axis"));
    }
    if tensor.range[0] > tensor.range[1] {
        return Err(format!(
            "{path}.range lower bound must not exceed its upper bound"
        ));
    }
    Ok(())
}
#[derive(Clone)]
pub struct ModelPackage {
    pub manifest: Manifest,
    pub model_bytes: Vec<u8>,
    pub package_dir: Option<PathBuf>,
}

impl std::fmt::Debug for ModelPackage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelPackage")
            .field("manifest", &self.manifest)
            .field("model_bytes_len", &self.model_bytes.len())
            .field("package_dir", &self.package_dir)
            .finish()
    }
}

impl ModelPackage {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, PreludioError> {
        let dir_path = dir.as_ref();
        let manifest_path = dir_path.join("manifest.json");

        if !manifest_path.exists() {
            return Err(PreludioError::ModelUnavailable(
                ModelUnavailable::FileMissing {
                    path: manifest_path.to_string_lossy().to_string(),
                },
            ));
        }

        let manifest_bytes = std::fs::read(&manifest_path).map_err(|e| {
            PreludioError::ModelUnavailable(ModelUnavailable::ManifestInvalid {
                message: format!("cannot read manifest.json: {}", e),
            })
        })?;

        let manifest_str = std::str::from_utf8(&manifest_bytes).map_err(|e| {
            PreludioError::ModelUnavailable(ModelUnavailable::ManifestInvalid {
                message: format!("manifest.json is not valid UTF-8: {}", e),
            })
        })?;

        let manifest = Self::parse_manifest(manifest_str)?;

        let model_path = dir_path.join(&manifest.model.file);
        if !model_path.exists() {
            return Err(PreludioError::ModelUnavailable(
                ModelUnavailable::FileMissing {
                    path: model_path.to_string_lossy().to_string(),
                },
            ));
        }

        let mut file = std::fs::File::open(&model_path).map_err(|e| {
            PreludioError::ModelUnavailable(ModelUnavailable::EngineLoadFailed {
                message: format!("failed to open model file: {}", e),
            })
        })?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let n = std::io::Read::read(&mut file, &mut buffer).map_err(|e| {
                PreludioError::ModelUnavailable(ModelUnavailable::EngineLoadFailed {
                    message: format!("failed to read model file: {}", e),
                })
            })?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        let actual = format!("{:x}", hasher.finalize());
        if actual != manifest.model.sha256 {
            return Err(PreludioError::ModelUnavailable(
                ModelUnavailable::HashMismatch {
                    expected: manifest.model.sha256.clone(),
                    actual,
                },
            ));
        }

        Ok(Self {
            manifest,
            model_bytes: Vec::new(),
            package_dir: Some(dir_path.to_path_buf()),
        })
    }

    pub fn from_parts(manifest_json: &str, model: Vec<u8>) -> Result<Self, PreludioError> {
        let manifest = Self::parse_manifest(manifest_json)?;
        Self::verify_hash(&manifest, &model)?;

        Ok(Self {
            manifest,
            model_bytes: model,
            package_dir: None,
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn model_bytes(&self) -> &[u8] {
        &self.model_bytes
    }

    fn parse_manifest(manifest_json: &str) -> Result<Manifest, PreludioError> {
        // Fast-path inspect schema before full serde to return UnsupportedSchema
        #[derive(Deserialize)]
        struct RawSchema {
            schema: Option<String>,
        }
        if let Ok(raw) = serde_json::from_str::<RawSchema>(manifest_json) {
            if let Some(schema) = raw.schema {
                if schema != SCHEMA_V1 {
                    return Err(PreludioError::ModelUnavailable(
                        ModelUnavailable::UnsupportedSchema { schema },
                    ));
                }
            } else {
                return Err(PreludioError::ModelUnavailable(
                    ModelUnavailable::ManifestInvalid {
                        message: "missing 'schema' field".to_string(),
                    },
                ));
            }
        }

        serde_json::from_str::<Manifest>(manifest_json).map_err(|e| {
            PreludioError::ModelUnavailable(ModelUnavailable::ManifestInvalid {
                message: e.to_string(),
            })
        })
    }

    fn verify_hash(manifest: &Manifest, model_bytes: &[u8]) -> Result<(), PreludioError> {
        let mut hasher = Sha256::new();
        hasher.update(model_bytes);
        let actual = format!("{:x}", hasher.finalize());

        if actual != manifest.model.sha256 {
            return Err(PreludioError::ModelUnavailable(
                ModelUnavailable::HashMismatch {
                    expected: manifest.model.sha256.clone(),
                    actual,
                },
            ));
        }
        Ok(())
    }
}
