use preludio::error::{InputViolation, PreludioError};
use preludio::manifest::Manifest;
use preludio::observation::{
    FrameValues, GridDescriptor, ObservationFrame, ObservationSequence, PixelQuality,
};
use preludio::validate::validate;
use std::fs;

fn load_manifest() -> Manifest {
    let content = fs::read_to_string("tests/fixtures/manifest-v1.json")
        .expect("read tests/fixtures/manifest-v1.json");
    serde_json::from_str(&content).expect("parse manifest-v1.json")
}

fn valid_sequence(manifest: &Manifest) -> ObservationSequence {
    let h = manifest.observation.grid.shape[0];
    let w = manifest.observation.grid.shape[1];
    let count = h * w;
    let base_time = 1701336240; // 09:24:00Z

    let mut frames = Vec::new();
    for i in 0..manifest.observation.frames {
        frames.push(ObservationFrame {
            id: format!("frame-{}", i),
            source: "test".to_string(),
            valid_time: base_time + (i as i64) * manifest.observation.cadence_seconds,
            values: FrameValues::U8(vec![10; count]),
            quality: None,
        });
    }

    ObservationSequence {
        grid: GridDescriptor {
            id: manifest.observation.grid.id.clone(),
            shape: [h, w],
            georeference: None,
        },
        encoding: manifest.observation.encoding.clone(),
        zero_threshold: None,
        frames,
    }
}

#[test]
fn test_valid_sequence_passes() {
    let manifest = load_manifest();
    let seq = valid_sequence(&manifest);
    assert!(validate(&manifest, &seq).is_ok());
}

#[test]
fn test_frame_count_mismatch() {
    let manifest = load_manifest();
    let mut seq = valid_sequence(&manifest);
    seq.frames.pop(); // 3 frames instead of 4
    let err = validate(&manifest, &seq).expect_err("should fail");
    match err {
        PreludioError::IncompatibleInput(violations) => {
            assert!(violations.contains(&InputViolation::FrameCount {
                expected: 4,
                actual: 3,
            }));
        }
        _ => panic!("wrong error variant"),
    }
}

#[test]
fn test_times_not_increasing() {
    let manifest = load_manifest();
    let mut seq = valid_sequence(&manifest);
    seq.frames[2].valid_time = seq.frames[1].valid_time; // duplicate
    let err = validate(&manifest, &seq).expect_err("should fail");
    match err {
        PreludioError::IncompatibleInput(violations) => {
            assert!(violations.contains(&InputViolation::NotIncreasing { index: 2 }));
        }
        _ => panic!("wrong error variant"),
    }

    let mut seq2 = valid_sequence(&manifest);
    seq2.frames[2].valid_time = seq2.frames[1].valid_time - 360; // backwards
    let err2 = validate(&manifest, &seq2).expect_err("should fail");
    match err2 {
        PreludioError::IncompatibleInput(violations) => {
            assert!(violations.contains(&InputViolation::NotIncreasing { index: 2 }));
        }
        _ => panic!("wrong error variant"),
    }
}

#[test]
fn test_irregular_increasing_times_are_accepted() {
    let manifest = load_manifest();
    let mut seq = valid_sequence(&manifest);
    let base = seq.frames[0].valid_time;
    seq.frames[1].valid_time = base + 5;
    seq.frames[2].valid_time = base + 7;
    seq.frames[3].valid_time = base + 100;
    assert!(validate(&manifest, &seq).is_ok());
}

#[test]
fn test_arbitrary_input_grid_id_and_shape_are_accepted() {
    let manifest = load_manifest();
    let mut seq = valid_sequence(&manifest);
    seq.grid.id = "arbitrary-rectangular-grid".to_string();
    seq.grid.shape = [2, 3];
    for frame in &mut seq.frames {
        frame.values = FrameValues::U8(vec![10; 6]);
    }
    assert!(validate(&manifest, &seq).is_ok());
}

#[test]
fn test_zero_grid_dimension_is_rejected() {
    let manifest = load_manifest();
    let mut seq = valid_sequence(&manifest);
    seq.grid.shape = [0, 3];
    for frame in &mut seq.frames {
        frame.values = FrameValues::U8(Vec::new());
    }
    let err = validate(&manifest, &seq).expect_err("zero dimensions should fail");
    match err {
        PreludioError::IncompatibleInput(violations) => {
            assert!(violations.contains(&InputViolation::InvalidShape {
                name: "observation grid".to_string(),
                shape: vec![0, 3],
            }))
        }
        _ => panic!("wrong error variant"),
    }
}

#[test]
fn test_grid_element_count_overflow_is_rejected() {
    let manifest = load_manifest();
    let mut seq = valid_sequence(&manifest);
    seq.grid.shape = [usize::MAX, 2];
    let err = validate(&manifest, &seq).expect_err("overflowing dimensions should fail");
    match err {
        PreludioError::IncompatibleInput(violations) => {
            assert!(violations.contains(&InputViolation::ShapeOverflow {
                name: "observation grid".to_string(),
            }))
        }
        _ => panic!("wrong error variant"),
    }
}

#[test]
fn test_raw_grayscale_rejects_f32_values() {
    let manifest = load_manifest();
    let mut seq = valid_sequence(&manifest);
    seq.frames[0].values = FrameValues::F32(vec![0.5; 732 * 732]);

    let err = validate(&manifest, &seq).expect_err("raw grayscale must use u8 values");
    match err {
        PreludioError::IncompatibleInput(violations) => {
            assert!(violations.contains(&InputViolation::UnsupportedValueType {
                index: 0,
                expected: "u8 grayscale values".to_string(),
                actual: "f32".to_string(),
            }));
        }
        _ => panic!("wrong error variant"),
    }
}

#[test]
fn test_encoding_mismatch() {
    let manifest = load_manifest();
    let mut seq = valid_sequence(&manifest);
    seq.encoding = "dbz-f32".to_string();
    let err = validate(&manifest, &seq).expect_err("should fail");
    match err {
        PreludioError::IncompatibleInput(violations) => {
            assert!(violations.contains(&InputViolation::EncodingMismatch {
                expected: "training-grayscale-u8".to_string(),
                actual: "dbz-f32".to_string(),
            }));
        }
        _ => panic!("wrong error variant"),
    }
}

#[test]
fn test_value_length_mismatch() {
    let manifest = load_manifest();
    let mut seq = valid_sequence(&manifest);
    seq.frames[1].values = FrameValues::U8(vec![0; 100]); // wrong length
    let err = validate(&manifest, &seq).expect_err("should fail");
    match err {
        PreludioError::IncompatibleInput(violations) => {
            assert!(violations.contains(&InputViolation::ValueLength {
                index: 1,
                expected: 732 * 732,
                actual: 100,
            }));
        }
        _ => panic!("wrong error variant"),
    }
}

#[test]
fn test_unsupported_quality() {
    let manifest = load_manifest();
    let mut seq = valid_sequence(&manifest);
    let mut q = vec![PixelQuality::Valid; 732 * 732];
    q[10] = PixelQuality::Missing;
    q[20] = PixelQuality::OutsideCoverage;
    seq.frames[3].quality = Some(q);

    let err = validate(&manifest, &seq).expect_err("should fail");
    match err {
        PreludioError::IncompatibleInput(violations) => {
            assert!(violations.contains(&InputViolation::UnsupportedQuality {
                index: 3,
                missing: 1,
                outside_coverage: 1,
            }));
        }
        _ => panic!("wrong error variant"),
    }
}

#[test]
fn test_multiple_faults_all_returned() {
    let manifest = load_manifest();
    let mut seq = valid_sequence(&manifest);
    seq.frames.pop(); // frame count
    seq.encoding = "wrong-enc".to_string();
    seq.frames[1].values = FrameValues::U8(Vec::new());

    let err = validate(&manifest, &seq).expect_err("should fail");
    match err {
        PreludioError::IncompatibleInput(violations) => {
            assert!(violations.len() >= 3);
            assert!(violations
                .iter()
                .any(|violation| matches!(violation, InputViolation::FrameCount { .. })));
            assert!(violations
                .iter()
                .any(|violation| matches!(violation, InputViolation::EncodingMismatch { .. })));
            assert!(violations.iter().any(|violation| matches!(
                violation,
                InputViolation::ValueLength { index: 1, .. }
            )));
        }
        _ => panic!("wrong error variant"),
    }
}

#[test]
fn zero_threshold_is_optional_in_sequence_json() {
    let manifest = load_manifest();
    let sequence = valid_sequence(&manifest);
    let serialized = serde_json::to_value(&sequence).expect("serialize sequence");
    assert!(serialized.get("zero_threshold").is_none());

    let decoded: ObservationSequence = serde_json::from_value(serialized.clone())
        .expect("missing zero_threshold defaults to None");
    assert_eq!(decoded.zero_threshold, None);

    let mut with_threshold = serialized;
    with_threshold["zero_threshold"] = serde_json::json!(10);
    let decoded: ObservationSequence =
        serde_json::from_value(with_threshold).expect("decode threshold");
    assert_eq!(decoded.zero_threshold, Some(10));
}

#[test]
fn wide_timestamp_span_is_accepted_without_overflow() {
    let manifest = load_manifest();
    let mut seq = valid_sequence(&manifest);
    seq.frames[0].valid_time = i64::MIN;
    seq.frames[1].valid_time = -1;
    seq.frames[2].valid_time = 0;
    seq.frames[3].valid_time = i64::MAX;

    assert!(validate(&manifest, &seq).is_ok());
}
