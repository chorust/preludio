use crate::error::{OutputViolation, PreludioError};
use crate::manifest::ModelPackage;
use crate::observation::{FrameMeta, ObservationSequence};
use crate::time::format_rfc3339_utc;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Forecast {
    pub values: Vec<f32>,
    pub metadata: ForecastMetadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastMetadata {
    pub kind: String,
    pub encoding: String,
    pub grid: ForecastGrid,
    pub lead_minutes: Vec<f64>,
    pub valid_times: Vec<String>,
    pub provenance: ForecastProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastProvenance {
    pub input_frames: Vec<InputFrameMeta>,
    pub last_observation_time: String,
    pub model: ModelProvenance,
    pub generated_at: String,
    pub execution: ExecutionInfo,
    pub preludio_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputFrameMeta {
    pub id: String,
    pub source: String,
    pub valid_time: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProvenance {
    pub id: String,
    pub sha256: String,
    pub schema: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionInfo {
    pub target_id: Option<String>,
    pub verified: bool,
    pub engine: Engine,
    pub engine_version: String,
    pub provider: Provider,
    pub threads: u32,
    pub platform: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cross_origin_isolated: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Engine {
    #[serde(rename = "onnxruntime")]
    OnnxRuntime,
    #[serde(rename = "onnxruntime-web")]
    OnnxRuntimeWeb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provider {
    #[serde(rename = "cpu")]
    Cpu,
    #[serde(rename = "wasm")]
    Wasm,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastGrid {
    pub id: String,
    pub shape: [usize; 2],
    #[serde(default)]
    pub georeferenced: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crs: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub affine: Option<[f64; 6]>,
}

pub fn finish_forecast(
    package: &ModelPackage,
    seq: &ObservationSequence,
    raw: Vec<f32>,
    execution: ExecutionInfo,
    generated_at: i64,
) -> Result<Forecast, PreludioError> {
    let frame_metas: Vec<FrameMeta> = seq
        .frames
        .iter()
        .map(|frame| FrameMeta {
            id: frame.id.clone(),
            source: frame.source.clone(),
            valid_time: frame.valid_time,
        })
        .collect();
    finish_forecast_from_frames(package, &frame_metas, raw, execution, generated_at)
}

pub fn finish_forecast_from_frames(
    package: &ModelPackage,
    frames: &[FrameMeta],
    raw: Vec<f32>,
    execution: ExecutionInfo,
    generated_at: i64,
) -> Result<Forecast, PreludioError> {
    crate::validate::validate_times(package.manifest(), frames)?;
    finish_forecast_from_validated_frames(package, frames, raw, execution, generated_at)
}

pub(crate) fn finish_forecast_from_validated_frames(
    package: &ModelPackage,
    frames: &[FrameMeta],
    raw: Vec<f32>,
    execution: ExecutionInfo,
    generated_at: i64,
) -> Result<Forecast, PreludioError> {
    let manifest = package.manifest();
    if frames.len() < 2 {
        return Err(PreludioError::IncompatibleInput(vec![
            crate::error::InputViolation::InsufficientFrames {
                minimum: 2,
                actual: frames.len(),
            },
        ]));
    }

    let output_shape = &manifest.tensor_output.shape;
    if output_shape.len() != 4 || output_shape[0] != 1 || output_shape.contains(&0) {
        return Err(PreludioError::InvalidOutput {
            execution,
            reason: OutputViolation::Shape,
        });
    }
    let expected_len = output_shape
        .iter()
        .try_fold(1usize, |size, &dimension| size.checked_mul(dimension));
    let Some(expected_len) = expected_len else {
        return Err(PreludioError::InvalidOutput {
            execution,
            reason: OutputViolation::Shape,
        });
    };
    let lead_count = output_shape[1];
    if lead_count > isize::MAX as usize / std::mem::size_of::<f64>()
        || lead_count > isize::MAX as usize / std::mem::size_of::<String>()
    {
        return Err(PreludioError::InvalidOutput {
            execution,
            reason: OutputViolation::Shape,
        });
    }
    if raw.len() != expected_len {
        return Err(PreludioError::InvalidOutput {
            execution,
            reason: OutputViolation::Shape,
        });
    }

    let mut non_finite = 0;
    let mut out_of_range = 0;
    let range = manifest.tensor_output.range;
    for &value in &raw {
        if !value.is_finite() {
            non_finite += 1;
        } else if f64::from(value) < range[0] || f64::from(value) > range[1] {
            out_of_range += 1;
        }
    }

    if non_finite > 0 {
        return Err(PreludioError::InvalidOutput {
            execution,
            reason: OutputViolation::NonFinite { count: non_finite },
        });
    }
    if out_of_range > 0 {
        return Err(PreludioError::InvalidOutput {
            execution,
            reason: OutputViolation::OutOfRange {
                count: out_of_range,
            },
        });
    }

    let first_obs_time = frames[0].valid_time;
    let last_obs_time = frames[frames.len() - 1].valid_time;
    let span = (last_obs_time as i128 - first_obs_time as i128) as u128;
    let interval_count = (frames.len() - 1) as u128;
    let max_lead_seconds = (i64::MAX as i128 - last_obs_time as i128) as u128;
    let mut lead_minutes = Vec::with_capacity(lead_count);
    let mut valid_times = Vec::with_capacity(lead_count);
    for lead_index in 1..=lead_count {
        let index = lead_index - 1;
        let numerator = span.checked_mul(lead_index as u128).ok_or_else(|| {
            PreludioError::IncompatibleInput(vec![crate::error::InputViolation::TimeOverflow {
                index,
            }])
        })?;
        let mut lead_seconds = numerator / interval_count;
        if (numerator % interval_count) * 2 >= interval_count {
            lead_seconds = lead_seconds.checked_add(1).ok_or_else(|| {
                PreludioError::IncompatibleInput(vec![crate::error::InputViolation::TimeOverflow {
                    index,
                }])
            })?;
        }
        if lead_seconds > max_lead_seconds {
            return Err(PreludioError::IncompatibleInput(vec![
                crate::error::InputViolation::TimeOverflow { index },
            ]));
        }
        let valid_time = (last_obs_time as i128 + lead_seconds as i128) as i64;
        lead_minutes.push(lead_seconds as f64 / 60.0);
        valid_times.push(format_rfc3339_utc(valid_time)?);
    }

    let mut input_frames = Vec::with_capacity(frames.len());
    for frame in frames {
        input_frames.push(InputFrameMeta {
            id: frame.id.clone(),
            source: frame.source.clone(),
            valid_time: format_rfc3339_utc(frame.valid_time)?,
        });
    }

    let grid = ForecastGrid {
        id: manifest.forecast.grid.id.clone(),
        shape: manifest.forecast.grid.shape,
        georeferenced: false,
        crs: None,
        affine: None,
    };
    let metadata = ForecastMetadata {
        kind: "forecast".to_string(),
        encoding: manifest.forecast.encoding.clone(),
        grid,
        lead_minutes,
        valid_times,
        provenance: ForecastProvenance {
            input_frames,
            last_observation_time: format_rfc3339_utc(last_obs_time)?,
            model: ModelProvenance {
                id: manifest.id.clone(),
                sha256: manifest.model.sha256.clone(),
                schema: manifest.schema.clone(),
            },
            generated_at: format_rfc3339_utc(generated_at)?,
            execution,
            preludio_version: crate::VERSION.to_string(),
        },
    };

    Ok(Forecast {
        values: raw,
        metadata,
    })
}
