use preludio::error::{OutputViolation, PreludioError};
use preludio::forecast::{
    finish_forecast, finish_forecast_from_frames, Engine, ExecutionInfo, Provider,
};
use preludio::manifest::{Manifest, ModelPackage};
use preludio::observation::{
    FrameMeta, FrameValues, GridDescriptor, ObservationFrame, ObservationSequence,
};
use std::fs;

fn load_package() -> ModelPackage {
    let content = fs::read_to_string("tests/fixtures/manifest-v1.json")
        .expect("read tests/fixtures/manifest-v1.json");
    let manifest: Manifest = serde_json::from_str(&content).expect("parse manifest");
    ModelPackage {
        manifest,
        model_bytes: vec![],
        package_dir: None,
    }
}

fn sample_frames() -> Vec<FrameMeta> {
    vec![
        FrameMeta {
            id: "AZ9002_092400".to_string(),
            source: "radar-az".to_string(),
            valid_time: 1701336240, // 09:24:00Z
        },
        FrameMeta {
            id: "obs-1".to_string(),
            source: "radar-az".to_string(),
            valid_time: 1701336340, // 09:25:40Z
        },
        FrameMeta {
            id: "obs-2".to_string(),
            source: "radar-az".to_string(),
            valid_time: 1701336740, // 09:32:20Z
        },
        FrameMeta {
            id: "obs-3".to_string(),
            source: "radar-az".to_string(),
            valid_time: 1701337240, // 09:40:40Z
        },
    ]
}

fn sequence_from_frames(frames: &[FrameMeta]) -> ObservationSequence {
    ObservationSequence {
        grid: GridDescriptor {
            id: "ignored-input-grid".to_string(),
            shape: [1, 1],
            georeference: None,
        },
        encoding: "training-grayscale-u8".to_string(),
        zero_threshold: None,
        frames: frames
            .iter()
            .map(|frame| ObservationFrame {
                id: frame.id.clone(),
                source: frame.source.clone(),
                valid_time: frame.valid_time,
                values: FrameValues::U8(vec![0]),
                quality: None,
            })
            .collect(),
    }
}

fn sample_execution() -> ExecutionInfo {
    ExecutionInfo {
        target_id: Some("test-target".to_string()),
        verified: true,
        engine: Engine::OnnxRuntime,
        engine_version: "1.31.0".to_string(),
        provider: Provider::Cpu,
        threads: 4,
        platform: "macos-arm64".to_string(),
        cross_origin_isolated: None,
    }
}

#[test]
fn test_forecast_metadata_and_provenance() {
    let mut pkg = load_package();
    // Deliberately differ from the tensor output lead dimension.
    pkg.manifest.forecast.lead_minutes = vec![1, 2, 3];
    let frames = sample_frames();
    let exec = sample_execution();
    let raw = vec![0.5f32; 20 * 448 * 448];
    let generated_at = 1701338000; // independent generated_at

    let forecast = finish_forecast_from_frames(&pkg, &frames, raw, exec.clone(), generated_at)
        .expect("finish_forecast should succeed");

    assert_eq!(forecast.metadata.kind, "forecast");
    assert_eq!(forecast.metadata.encoding, "training-grayscale-normalized");
    assert_eq!(forecast.metadata.grid.id, "onix-station-png-732/448");
    assert_eq!(forecast.metadata.grid.shape, [448, 448]);
    assert!(!forecast.metadata.grid.georeferenced);
    assert_eq!(forecast.metadata.grid.crs, None);
    assert_eq!(forecast.metadata.grid.affine, None);
    assert_eq!(forecast.metadata.lead_minutes.len(), 20);
    assert_eq!(forecast.metadata.lead_minutes[0], 333.0 / 60.0);
    assert_eq!(forecast.metadata.lead_minutes[1], 667.0 / 60.0);

    let valid_times = &forecast.metadata.valid_times;
    assert_eq!(valid_times.len(), 20);
    assert_eq!(valid_times[0], "2023-11-30T09:46:13Z");
    assert_eq!(valid_times[1], "2023-11-30T09:51:47Z");
    assert_eq!(valid_times[2], "2023-11-30T09:57:20Z");
    assert_eq!(valid_times[19], "2023-11-30T11:31:47Z");

    let prov = &forecast.metadata.provenance;
    assert_eq!(prov.last_observation_time, "2023-11-30T09:40:40Z");
    assert_eq!(prov.generated_at, "2023-11-30T09:53:20Z");
    assert_eq!(prov.input_frames.len(), 4);
    assert_eq!(prov.input_frames[0].id, "AZ9002_092400");
    assert_eq!(prov.input_frames[3].valid_time, "2023-11-30T09:40:40Z");
    assert_eq!(prov.model.id, pkg.manifest().id);
    assert_eq!(prov.model.sha256, pkg.manifest().model.sha256);
    assert_eq!(prov.execution.platform, "macos-arm64");
}

#[test]
fn test_forecast_output_validation_errors() {
    let pkg = load_package();
    let frames = sample_frames();
    let exec = sample_execution();

    // 1. Wrong shape length
    let wrong_len_raw = vec![0.5f32; 100];
    let err_shape =
        finish_forecast_from_frames(&pkg, &frames, wrong_len_raw, exec.clone(), 1700000000)
            .expect_err("should fail with shape");
    match err_shape {
        PreludioError::InvalidOutput { reason, .. } => {
            assert_eq!(reason, OutputViolation::Shape);
        }
        _ => panic!("wrong error variant"),
    }

    // 2. NaN / non-finite
    let mut nan_raw = vec![0.5f32; 20 * 448 * 448];
    nan_raw[0] = f32::NAN;
    let err_nan = finish_forecast_from_frames(&pkg, &frames, nan_raw, exec.clone(), 1700000000)
        .expect_err("should fail with non-finite");
    match err_nan {
        PreludioError::InvalidOutput { reason, .. } => {
            assert_eq!(reason, OutputViolation::NonFinite { count: 1 });
        }
        _ => panic!("wrong error variant"),
    }

    // 3. Out of range (range is [0.0, 1.0])
    let mut out_raw = vec![0.5f32; 20 * 448 * 448];
    out_raw[10] = 1.5;
    let err_out = finish_forecast_from_frames(&pkg, &frames, out_raw, exec.clone(), 1700000000)
        .expect_err("should fail with out of range");
    match err_out {
        PreludioError::InvalidOutput { reason, .. } => {
            assert_eq!(reason, OutputViolation::OutOfRange { count: 1 });
        }
        _ => panic!("wrong error variant"),
    }
}

#[test]
fn test_public_forecast_finishers_reject_non_increasing_times() {
    let pkg = load_package();
    let mut frames = sample_frames();
    frames[2].valid_time = frames[1].valid_time;

    let frames_error =
        finish_forecast_from_frames(&pkg, &frames, Vec::new(), sample_execution(), 1701338000)
            .expect_err("frame finisher must validate timestamps");
    assert!(matches!(
        frames_error,
        PreludioError::IncompatibleInput(violations)
            if violations.contains(&preludio::error::InputViolation::NotIncreasing { index: 2 })
    ));

    let seq = sequence_from_frames(&frames);
    let sequence_error = finish_forecast(&pkg, &seq, Vec::new(), sample_execution(), 1701338000)
        .expect_err("sequence finisher must validate timestamps");
    assert!(matches!(
        sequence_error,
        PreludioError::IncompatibleInput(violations)
            if violations.contains(&preludio::error::InputViolation::NotIncreasing { index: 2 })
    ));
}

#[test]
fn forecast_time_addition_overflow_is_reported() {
    let mut pkg = load_package();
    pkg.manifest.tensor_output.shape = vec![1, 1, 1, 1];
    let mut frames = sample_frames();
    let last_time = i64::MAX;
    for (frame, offset) in frames.iter_mut().zip([6, 4, 2, 0]) {
        frame.valid_time = last_time - offset;
    }

    let error =
        finish_forecast_from_frames(&pkg, &frames, vec![0.5], sample_execution(), 1701338000)
            .expect_err("forecast valid time must not overflow");
    assert!(matches!(
        error,
        PreludioError::IncompatibleInput(violations)
            if violations.contains(&preludio::error::InputViolation::TimeOverflow { index: 0 })
    ));
}
