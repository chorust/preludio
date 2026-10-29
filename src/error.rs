use crate::forecast::ExecutionInfo;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, thiserror::Error, Serialize, Deserialize)]
pub enum PreludioError {
    #[error("model unavailable: {0}")]
    ModelUnavailable(ModelUnavailable),
    #[error("incompatible input: {} violation(s)", .0.len())]
    IncompatibleInput(Vec<InputViolation>),
    #[error("execution failed on {execution:?}: {message}")]
    ExecutionFailed {
        execution: ExecutionInfo,
        message: String,
    },
    #[error("invalid output: {reason:?}")]
    InvalidOutput {
        execution: ExecutionInfo,
        reason: OutputViolation,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
pub enum ModelUnavailable {
    #[error("file missing: {path}")]
    FileMissing { path: String },
    #[error("manifest invalid: {message}")]
    ManifestInvalid { message: String },
    #[error("unsupported schema: {schema}")]
    UnsupportedSchema { schema: String },
    #[error("hash mismatch: expected {expected}, actual {actual}")]
    HashMismatch { expected: String, actual: String },
    #[error("tensor mismatch: {message}")]
    TensorMismatch { message: String },
    #[error("target not supported: platform {platform}")]
    TargetNotSupported {
        platform: String,
        supported: Vec<String>,
    },
    #[error("cross-origin isolation required")]
    IsolationRequired,
    #[error("engine load failed: {message}")]
    EngineLoadFailed { message: String },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
pub enum InputViolation {
    #[error("frame count: expected {expected}, actual {actual}")]
    FrameCount { expected: usize, actual: usize },
    #[error("valid times not strictly increasing at index {index}")]
    NotIncreasing { index: usize },
    #[error("timestamp arithmetic overflow at index {index}")]
    TimeOverflow { index: usize },
    #[error("at least {minimum} frames required to calculate forecast timing, actual {actual}")]
    InsufficientFrames { minimum: usize, actual: usize },
    #[error("invalid {name} shape {shape:?}; dimensions must be positive")]
    InvalidShape { name: String, shape: Vec<usize> },
    #[error("{name} shape arithmetic overflow")]
    ShapeOverflow { name: String },
    #[error("unsupported value type at index {index}: expected {expected}, actual {actual}")]
    UnsupportedValueType {
        index: usize,
        expected: String,
        actual: String,
    },
    #[error("grid mismatch: expected {expected}, actual {actual}")]
    GridMismatch { expected: String, actual: String },
    #[error("encoding mismatch: expected {expected}, actual {actual}")]
    EncodingMismatch { expected: String, actual: String },
    #[error("value length mismatch at index {index}: expected {expected}, actual {actual}")]
    ValueLength {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("non-finite values at index {index}: count {count}")]
    NonFinite { index: usize, count: usize },
    #[error("out of range values at index {index}: count {count}")]
    OutOfRange { index: usize, count: usize },
    #[error("unsupported pixel quality at index {index}: missing {missing}, outside_coverage {outside_coverage}")]
    UnsupportedQuality {
        index: usize,
        missing: usize,
        outside_coverage: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
pub enum OutputViolation {
    #[error("shape mismatch")]
    Shape,
    #[error("non-finite count {count}")]
    NonFinite { count: usize },
    #[error("out of range count {count}")]
    OutOfRange { count: usize },
}
