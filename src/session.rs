use crate::error::{ModelUnavailable, OutputViolation, PreludioError};
use crate::forecast::{
    finish_forecast_from_validated_frames, Engine, ExecutionInfo, Forecast, Provider,
};
use crate::manifest::{ModelPackage, TargetThreads};
pub use crate::observation::FrameMeta;
use crate::observation::ObservationSequence;
use crate::preprocess::preprocess;
use crate::validate::validate_times;
use ort::ep::CPU;
use ort::session::builder::AutoDevicePolicy;
use ort::session::Session as OrtSession;
use ort::value::Tensor;

const ONNXRUNTIME_VERSION: &str = "1.31.0";
const DEFAULT_RUNTIME_THREADS: u32 = 4;

static ORT_ENVIRONMENT: std::sync::LazyLock<Result<ort::environment::Environment, String>> =
    std::sync::LazyLock::new(|| {
        ort::init()
            .with_name("preludio")
            .build()
            .map_err(|error| format!("cannot init ort environment: {}", error))
    });

fn ort_environment() -> Result<ort::environment::Environment, PreludioError> {
    match &*ORT_ENVIRONMENT {
        Ok(environment) => Ok(environment.clone()),
        Err(message) => Err(PreludioError::ModelUnavailable(
            ModelUnavailable::EngineLoadFailed {
                message: message.clone(),
            },
        )),
    }
}

pub enum TargetPolicy {
    SupportedOnly,
    Validation,
}

pub struct SessionOptions {
    pub policy: TargetPolicy,
    pub threads: Option<u32>,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            policy: TargetPolicy::SupportedOnly,
            threads: None,
        }
    }
}

pub struct Session {
    package: ModelPackage,
    ort_session: std::sync::Mutex<OrtSession>,
    execution: ExecutionInfo,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("package", &self.package)
            .field("execution", &self.execution)
            .finish()
    }
}

fn resolved_threads(declared: TargetThreads) -> u32 {
    match declared {
        TargetThreads::Default => DEFAULT_RUNTIME_THREADS,
        TargetThreads::Fixed(threads) => threads,
    }
}

fn tensor_shape_matches(actual: &ort::value::ValueType, expected: &[usize]) -> bool {
    let Some(actual_shape) = actual.tensor_shape() else {
        return false;
    };
    actual_shape.len() == expected.len()
        && actual_shape
            .iter()
            .zip(expected)
            .all(|(&actual, &expected)| {
                actual == -1 || usize::try_from(actual).ok() == Some(expected)
            })
}

fn validate_tensor_metadata(
    outlets: &[ort::value::Outlet],
    role: &str,
    expected_name: &str,
    expected_shape: &[usize],
) -> Result<(), PreludioError> {
    let Some(outlet) = outlets.iter().find(|outlet| outlet.name() == expected_name) else {
        return Err(PreludioError::ModelUnavailable(
            ModelUnavailable::TensorMismatch {
                message: format!("model missing expected {role} '{expected_name}'"),
            },
        ));
    };
    if !tensor_shape_matches(outlet.dtype(), expected_shape) {
        return Err(PreludioError::ModelUnavailable(
            ModelUnavailable::TensorMismatch {
                message: format!(
                    "model {role} '{expected_name}' has shape {:?}, expected {:?}",
                    outlet.dtype().tensor_shape(),
                    expected_shape
                ),
            },
        ));
    }
    Ok(())
}

impl Session {
    pub fn open(package: ModelPackage, options: SessionOptions) -> Result<Self, PreludioError> {
        let current_platform = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            "macos-arm64".to_string()
        } else {
            format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
        };

        let matched = package.manifest().supported_targets.iter().find(|target| {
            target.platform == current_platform
                && target.engine == "onnxruntime"
                && target.engine_version == ONNXRUNTIME_VERSION
                && target.provider == "cpu"
                && match (options.threads, target.threads) {
                    (Some(requested), TargetThreads::Fixed(declared)) => requested == declared,
                    (Some(_), TargetThreads::Default) => false,
                    (None, _) => true,
                }
        });

        let (target_id, verified, target_threads) = match options.policy {
            TargetPolicy::SupportedOnly => {
                if let Some(target) = matched {
                    (
                        Some(target.id.clone()),
                        true,
                        Some(resolved_threads(target.threads)),
                    )
                } else {
                    let supported = package
                        .manifest()
                        .supported_targets
                        .iter()
                        .map(|target| target.platform.clone())
                        .collect();
                    return Err(PreludioError::ModelUnavailable(
                        ModelUnavailable::TargetNotSupported {
                            platform: current_platform,
                            supported,
                        },
                    ));
                }
            }
            TargetPolicy::Validation => {
                if let Some(target) = matched {
                    (
                        Some(target.id.clone()),
                        true,
                        Some(resolved_threads(target.threads)),
                    )
                } else {
                    (None, false, None)
                }
            }
        };

        let threads = options
            .threads
            .or(target_threads)
            .unwrap_or(DEFAULT_RUNTIME_THREADS);

        let execution = ExecutionInfo {
            target_id,
            verified,
            engine: Engine::OnnxRuntime,
            engine_version: ONNXRUNTIME_VERSION.to_string(),
            provider: Provider::Cpu,
            threads,
            platform: current_platform,
            cross_origin_isolated: None,
        };
        let env = ort_environment()?;
        let builder = OrtSession::builder(&env).map_err(|e| {
            PreludioError::ModelUnavailable(ModelUnavailable::EngineLoadFailed {
                message: format!("cannot create session builder: {}", e),
            })
        })?;
        let builder = builder
            .with_execution_providers([CPU::default().with_arena_allocator(true).build()])
            .map_err(|e| {
                PreludioError::ModelUnavailable(ModelUnavailable::EngineLoadFailed {
                    message: format!("cannot set CPU execution provider: {}", e),
                })
            })?;

        let builder = builder
            .with_auto_device(AutoDevicePolicy::PreferCPU)
            .map_err(|e| {
                PreludioError::ModelUnavailable(ModelUnavailable::EngineLoadFailed {
                    message: format!("cannot set AutoDevicePolicy: {}", e),
                })
            })?;

        let builder = builder.with_memory_pattern(false).map_err(|e| {
            PreludioError::ModelUnavailable(ModelUnavailable::EngineLoadFailed {
                message: format!("cannot set memory pattern: {}", e),
            })
        })?;

        let mut builder = builder.with_intra_threads(threads as usize).map_err(|e| {
            PreludioError::ModelUnavailable(ModelUnavailable::EngineLoadFailed {
                message: format!("cannot set intra threads: {}", e),
            })
        })?;

        let ort_session = if let Some(ref dir) = package.package_dir {
            let model_path = dir.join(&package.manifest.model.file);
            builder.commit_from_file(model_path).map_err(|e| {
                PreludioError::ModelUnavailable(ModelUnavailable::EngineLoadFailed {
                    message: format!("cannot commit model from file: {}", e),
                })
            })?
        } else {
            builder
                .commit_from_memory(&package.model_bytes)
                .map_err(|e| {
                    PreludioError::ModelUnavailable(ModelUnavailable::EngineLoadFailed {
                        message: format!("cannot commit model from memory: {}", e),
                    })
                })?
        };

        // Check ONNX input/output names and shapes against manifest tensor metadata.
        validate_tensor_metadata(
            ort_session.inputs(),
            "input",
            &package.manifest.tensor_input.name,
            &package.manifest.tensor_input.shape,
        )?;
        validate_tensor_metadata(
            ort_session.outputs(),
            "output",
            &package.manifest.tensor_output.name,
            &package.manifest.tensor_output.shape,
        )?;

        Ok(Self {
            package,
            ort_session: std::sync::Mutex::new(ort_session),
            execution,
        })
    }

    pub fn execution(&self) -> &ExecutionInfo {
        &self.execution
    }

    pub fn predict(&self, seq: &ObservationSequence) -> Result<Forecast, PreludioError> {
        if seq.frames.len() < 2 && seq.frames.len() == self.package.manifest.observation.frames {
            return Err(PreludioError::IncompatibleInput(vec![
                crate::error::InputViolation::InsufficientFrames {
                    minimum: 2,
                    actual: seq.frames.len(),
                },
            ]));
        }
        let input_tensor = preprocess(&self.package.manifest, seq)?;

        let frames: Vec<FrameMeta> = seq
            .frames
            .iter()
            .map(|f| FrameMeta {
                id: f.id.clone(),
                source: f.source.clone(),
                valid_time: f.valid_time,
            })
            .collect();

        self.predict_tensor_internal(&input_tensor, &frames)
    }

    pub fn predict_tensor(
        &self,
        input: &[f32],
        frames: &[FrameMeta],
    ) -> Result<Forecast, PreludioError> {
        validate_times(&self.package.manifest, frames)?;
        self.predict_tensor_internal(input, frames)
    }

    fn predict_tensor_internal(
        &self,
        input: &[f32],
        frames: &[FrameMeta],
    ) -> Result<Forecast, PreludioError> {
        let input_shape = &self.package.manifest.tensor_input.shape;
        if input_shape.len() != 4 || input_shape.contains(&0) {
            return Err(PreludioError::IncompatibleInput(vec![
                crate::error::InputViolation::InvalidShape {
                    name: "tensor input".to_string(),
                    shape: input_shape.clone(),
                },
            ]));
        }
        if input_shape
            .iter()
            .any(|&dimension| dimension > i64::MAX as usize)
        {
            return Err(PreludioError::IncompatibleInput(vec![
                crate::error::InputViolation::ShapeOverflow {
                    name: "tensor input".to_string(),
                },
            ]));
        }
        let expected_len = input_shape
            .iter()
            .try_fold(1usize, |size, &dimension| size.checked_mul(dimension))
            .filter(|&length| length <= isize::MAX as usize / std::mem::size_of::<f32>())
            .ok_or_else(|| {
                PreludioError::IncompatibleInput(vec![
                    crate::error::InputViolation::ShapeOverflow {
                        name: "tensor input".to_string(),
                    },
                ])
            })?;
        if input.len() != expected_len {
            return Err(PreludioError::IncompatibleInput(vec![
                crate::error::InputViolation::ValueLength {
                    index: 0,
                    expected: expected_len,
                    actual: input.len(),
                },
            ]));
        }
        let shape_i64: Vec<i64> = input_shape
            .iter()
            .map(|&dimension| dimension as i64)
            .collect();
        let tensor = Tensor::from_array((shape_i64, input.to_vec())).map_err(|e| {
            PreludioError::ExecutionFailed {
                execution: self.execution.clone(),
                message: format!("cannot create input tensor: {}", e),
            }
        })?;
        let input_name = self.package.manifest.tensor_input.name.as_str();
        let output_name = self.package.manifest.tensor_output.name.as_str();

        let mut session = self
            .ort_session
            .lock()
            .map_err(|e| PreludioError::ExecutionFailed {
                execution: self.execution.clone(),
                message: format!("failed to lock session mutex: {}", e),
            })?;
        let outputs = session
            .run(ort::inputs![input_name => tensor])
            .map_err(|e| PreludioError::ExecutionFailed {
                execution: self.execution.clone(),
                message: format!("inference failed: {}", e),
            })?;

        let output_tensor =
            outputs
                .get(output_name)
                .ok_or_else(|| PreludioError::ExecutionFailed {
                    execution: self.execution.clone(),
                    message: format!("missing output tensor '{}'", output_name),
                })?;

        let (runtime_shape, extracted_data) =
            output_tensor.try_extract_tensor::<f32>().map_err(|e| {
                PreludioError::ExecutionFailed {
                    execution: self.execution.clone(),
                    message: format!("failed to extract output tensor: {}", e),
                }
            })?;

        let expected_output_shape = &self.package.manifest().tensor_output.shape;
        if runtime_shape.len() != expected_output_shape.len()
            || runtime_shape
                .iter()
                .zip(expected_output_shape)
                .any(|(&actual, &expected)| usize::try_from(actual).ok() != Some(expected))
        {
            return Err(PreludioError::InvalidOutput {
                execution: self.execution.clone(),
                reason: OutputViolation::Shape,
            });
        }
        let now = time::OffsetDateTime::now_utc().unix_timestamp();
        finish_forecast_from_validated_frames(
            &self.package,
            frames,
            extracted_data.to_vec(),
            self.execution.clone(),
            now,
        )
    }
}
