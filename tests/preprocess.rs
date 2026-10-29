mod common;

use common::read_npy_f32;
use common::read_npy_u8;
use preludio::manifest::{
    CheckpointMeta, GridMeta, Manifest, ModelForecast, ModelMeta, ModelObservation, ModelOrigin,
    PreprocessingStep, TensorMeta,
};
use preludio::observation::{FrameValues, GridDescriptor, ObservationFrame, ObservationSequence};
use preludio::preprocess::preprocess;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Deserialize)]
struct Meta {
    cases: Vec<TestCase>,
}

#[derive(Deserialize)]
struct TestCase {
    name: String,
    input: String,
    reference: String,
    input_shape: Vec<usize>,
    output_size: Vec<usize>,
}

fn create_test_manifest(in_shape: &[usize], out_size: &[usize]) -> Manifest {
    let in_frames = in_shape[0];
    let in_h = in_shape[1];
    let in_w = in_shape[2];
    let out_h = out_size[0];
    let out_w = out_size[1];

    Manifest {
        schema: "preludio.model-package/1".to_string(),
        id: "test-model".to_string(),
        model: ModelMeta {
            file: "model.onnx".to_string(),
            sha256: "0".repeat(64),
            format: "onnx".to_string(),
            opset: 18,
        },
        origin: ModelOrigin {
            checkpoint: CheckpointMeta {
                file: "ckpt".to_string(),
                step: 0,
                sha256: "0".repeat(64),
                run: "test".to_string(),
            },
            source_sha256: "0".repeat(64),
            config_sha256: "0".repeat(64),
            exporter: "test".to_string(),
        },
        observation: ModelObservation {
            frames: in_frames,
            cadence_seconds: 360,
            cadence_tolerance_seconds: 0,
            encoding: "training-grayscale-u8".to_string(),
            grid: GridMeta {
                id: "test-grid".to_string(),
                shape: [in_h, in_w],
                georeferenced: false,
                crs: None,
                affine: None,
            },
            missing_policy: "reject".to_string(),
        },
        tensor_input: TensorMeta {
            name: "radar".to_string(),
            dtype: "float32".to_string(),
            shape: vec![1, in_frames, out_h, out_w],
            axes: vec![
                "batch".to_string(),
                "time".to_string(),
                "y".to_string(),
                "x".to_string(),
            ],
            range: [0.0, 1.0],
        },
        tensor_output: TensorMeta {
            name: "forecast".to_string(),
            dtype: "float32".to_string(),
            shape: vec![1, 20, out_h, out_w],
            axes: vec![
                "batch".to_string(),
                "lead_time".to_string(),
                "y".to_string(),
                "x".to_string(),
            ],
            range: [0.0, 1.0],
        },
        preprocessing: vec![
            PreprocessingStep::U8Div255,
            PreprocessingStep::ResizeBilinearAa {
                height: out_h,
                width: out_w,
                align_corners: false,
            },
            PreprocessingStep::RoundF16,
        ],
        forecast: ModelForecast {
            lead_minutes: vec![6],
            encoding: "training-grayscale-normalized".to_string(),
            grid: GridMeta {
                id: "test-grid-out".to_string(),
                shape: [out_h, out_w],
                georeferenced: false,
                crs: None,
                affine: None,
            },
        },
        supported_targets: vec![],
    }
}

#[test]
fn test_preprocess_bit_exact_against_pytorch() {
    let fixture_dir = Path::new("tests/fixtures/preprocess");
    let meta_bytes = fs::read(fixture_dir.join("meta.json")).expect("read meta.json");
    let meta: Meta = serde_json::from_slice(&meta_bytes).expect("parse meta.json");

    for case in meta.cases {
        let (in_shape, in_bytes) =
            read_npy_u8(fixture_dir.join(&case.input)).expect("read input npy");
        let (_ref_shape, ref_f32) =
            read_npy_f32(fixture_dir.join(&case.reference)).expect("read reference npy");
        assert_eq!(
            in_shape, case.input_shape,
            "input shape mismatch for {}",
            case.name
        );

        let manifest = create_test_manifest(&case.input_shape, &case.output_size);
        let frames_count = case.input_shape[0];
        let h = case.input_shape[1];
        let w = case.input_shape[2];
        let frame_size = h * w;

        let mut frames = Vec::with_capacity(frames_count);
        for f in 0..frames_count {
            let slice = &in_bytes[f * frame_size..(f + 1) * frame_size];
            frames.push(ObservationFrame {
                id: format!("frame-{}", f),
                source: "fixture".to_string(),
                valid_time: 1700000000 + (f as i64) * 360,
                values: FrameValues::U8(slice.to_vec()),
                quality: None,
            });
        }

        let seq = ObservationSequence {
            grid: GridDescriptor {
                id: "test-grid".to_string(),
                shape: [h, w],
                georeference: None,
            },
            encoding: "training-grayscale-u8".to_string(),
            zero_threshold: None,
            frames,
        };

        let result = preprocess(&manifest, &seq).expect("preprocess failed");
        assert_eq!(
            result.len(),
            ref_f32.len(),
            "output length mismatch for {}",
            case.name
        );

        let mut mismatches = 0;
        let mut first_mismatch = None;
        for (i, (&actual, &expected)) in result.iter().zip(ref_f32.iter()).enumerate() {
            if actual.to_bits() != expected.to_bits() {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((i, actual, expected));
                }
            }
        }

        if let Some((idx, actual, expected)) = first_mismatch {
            panic!(
                "Case '{}' failed bit-exact comparison! {} mismatches out of {}. First at {}: actual={:?} ({:#x}) vs expected={:?} ({:#x})",
                case.name,
                mismatches,
                result.len(),
                idx,
                actual,
                actual.to_bits(),
                expected,
                expected.to_bits()
            );
        }
    }
}
fn sequence_with_pixels(
    shape: [usize; 2],
    pixels: &[u8],
    zero_threshold: Option<u8>,
) -> ObservationSequence {
    let frames = (0..4)
        .map(|index| ObservationFrame {
            id: format!("frame-{index}"),
            source: "test".to_string(),
            valid_time: 1_700_000_000 + index as i64 * (index as i64 + 1),
            values: FrameValues::U8(pixels.to_vec()),
            quality: None,
        })
        .collect();
    ObservationSequence {
        grid: GridDescriptor {
            id: "arbitrary-input-grid".to_string(),
            shape,
            georeference: None,
        },
        encoding: "training-grayscale-u8".to_string(),
        zero_threshold,
        frames,
    }
}

#[test]
fn zero_threshold_clears_values_below_boundary_before_normalizing() {
    let manifest = create_test_manifest(&[4, 2, 3], &[2, 3]);
    let seq = sequence_with_pixels([2, 3], &[9, 10, 11, 0, 128, 255], Some(10));
    let result = preprocess(&manifest, &seq).expect("preprocess should succeed");

    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], preludio::preprocess::round_f16(10.0 / 255.0));
    assert_eq!(result[2], preludio::preprocess::round_f16(11.0 / 255.0));
}

#[test]
fn zero_threshold_is_applied_before_downsampling() {
    let manifest = create_test_manifest(&[4, 2, 2], &[1, 1]);
    let seq = sequence_with_pixels([2, 2], &[0, 10, 10, 10], Some(10));
    let result = preprocess(&manifest, &seq).expect("preprocess should succeed");

    assert_eq!(result[0], preludio::preprocess::round_f16(7.5 / 255.0));
    assert!(result[0] > 0.0);
}

#[test]
fn zero_threshold_zero_has_no_effect_and_rectangular_inputs_resize_to_tensor() {
    let manifest = create_test_manifest(&[4, 3, 2], &[3, 5]);
    let values = &[0, 1, 10, 128, 254, 255];
    let none = sequence_with_pixels([2, 3], values, None);
    let zero = sequence_with_pixels([2, 3], values, Some(0));
    let none_result = preprocess(&manifest, &none).expect("preprocess without threshold");
    let zero_result = preprocess(&manifest, &zero).expect("preprocess with zero threshold");

    assert_eq!(zero_result, none_result);
    assert_eq!(zero_result.len(), 4 * 3 * 5);
}

#[test]
fn preprocess_rejects_zero_input_dimensions_without_panicking() {
    let manifest = create_test_manifest(&[4, 2, 3], &[2, 3]);
    let mut seq = sequence_with_pixels([0, 3], &[], None);
    for frame in &mut seq.frames {
        frame.values = FrameValues::U8(Vec::new());
    }

    let error = preprocess(&manifest, &seq).expect_err("zero dimensions must be rejected");
    assert!(matches!(
        error,
        preludio::PreludioError::IncompatibleInput(_)
    ));
}

#[test]
fn preprocess_rejects_f32_grayscale_frames() {
    let manifest = create_test_manifest(&[4, 2, 3], &[2, 3]);
    let mut seq = sequence_with_pixels([2, 3], &[0, 1, 2, 3, 4, 5], None);
    seq.frames[0].values = FrameValues::F32(vec![0.5; 6]);

    let error = preprocess(&manifest, &seq).expect_err("raw grayscale preprocessing is u8 only");
    assert!(matches!(
        error,
        preludio::PreludioError::IncompatibleInput(violations)
            if violations.contains(&preludio::error::InputViolation::UnsupportedValueType {
                index: 0,
                expected: "u8 grayscale values".to_string(),
                actual: "f32".to_string(),
            })
    ));
}
