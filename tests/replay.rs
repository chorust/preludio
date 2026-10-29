#![cfg(all(feature = "native", not(target_arch = "wasm32")))]

mod common;

use common::{model_package, read_npy_f32};
use preludio::error::{ModelUnavailable, PreludioError};
use preludio::forecast::Provider;
use preludio::manifest::{Manifest, ModelPackage, SupportedTarget, TargetThreads};
use preludio::observation::FrameMeta;
use preludio::session::{Session, SessionOptions, TargetPolicy};
use std::fs;
use std::path::{Path, PathBuf};
#[cfg(feature = "cli")]
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

struct TemporaryModelPackage(PathBuf);

impl TemporaryModelPackage {
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TemporaryModelPackage {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn temporary_model_package(source_dir: &Path, manifest: &Manifest) -> TemporaryModelPackage {
    static NEXT_PACKAGE: AtomicUsize = AtomicUsize::new(0);
    let package_dir = std::env::temp_dir().join(format!(
        "preludio-replay-{}-{}",
        std::process::id(),
        NEXT_PACKAGE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&package_dir).expect("create temporary model package directory");
    let temp_package = TemporaryModelPackage(package_dir);

    let model_source = source_dir.join(&manifest.model.file);
    let model_destination = temp_package.path().join(&manifest.model.file);
    if let Some(parent) = model_destination.parent() {
        fs::create_dir_all(parent).expect("create temporary model subdirectory");
    }
    if fs::hard_link(&model_source, &model_destination).is_err() {
        fs::copy(&model_source, &model_destination).expect("copy model into temporary package");
    }
    let manifest_json = serde_json::to_vec(manifest).expect("serialize temporary package manifest");
    fs::write(temp_package.path().join("manifest.json"), manifest_json)
        .expect("write temporary package manifest");

    temp_package
}

fn current_platform() -> String {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "macos-arm64".to_string()
    } else {
        format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
    }
}

fn supported_cpu_target(id: &str, platform: &str, threads: u32) -> SupportedTarget {
    SupportedTarget {
        id: id.to_string(),
        platform: platform.to_string(),
        engine: "onnxruntime".to_string(),
        engine_version: "1.31.0".to_string(),
        provider: "cpu".to_string(),
        threads: TargetThreads::Fixed(threads),
        requires_cross_origin_isolation: false,
        validation: "test".to_string(),
        measurement: "test".to_string(),
    }
}

fn change_output_axes_preserving_element_count(manifest: &mut Manifest) {
    let shape = &mut manifest.tensor_output.shape;
    let axes = &mut manifest.tensor_output.axes;
    let mut unequal_axes = None;
    for left in 1..shape.len() {
        for right in left + 1..shape.len() {
            if shape[left] != shape[right] {
                unequal_axes = Some((left, right));
                break;
            }
        }
        if unequal_axes.is_some() {
            break;
        }
    }

    if let Some((left, right)) = unequal_axes {
        shape.swap(left, right);
        axes.swap(left, right);
        return;
    }

    for source in 1..shape.len() {
        if shape[source] > 1 {
            for destination in 1..shape.len() {
                if destination != source && shape[destination] <= usize::MAX / shape[source] {
                    let factor = shape[source];
                    shape[source] = 1;
                    shape[destination] *= factor;
                    axes.swap(source, destination);
                    return;
                }
            }
        }
    }
    panic!("output shape has no alternate positive shape with the same element count");
}

#[test]
fn test_replay_parity_and_integrity() {
    let pkg_dir = match model_package() {
        Some(dir) => dir,
        None => return,
    };

    let manifest_path = pkg_dir.join("manifest.json");
    let manifest_json = fs::read_to_string(&manifest_path).expect("read manifest.json");

    let pkg = ModelPackage::open(&pkg_dir).expect("open model package");
    let model_bytes = fs::read(pkg_dir.join(&pkg.manifest().model.file)).expect("read model.onnx");

    // 1. Unsupported schema check
    let bad_schema_manifest =
        manifest_json.replace("preludio.model-package/1", "preludio.model-package/2");
    let res = ModelPackage::from_parts(&bad_schema_manifest, model_bytes.clone());
    match res {
        Err(PreludioError::ModelUnavailable(ModelUnavailable::UnsupportedSchema { schema })) => {
            assert_eq!(schema, "preludio.model-package/2");
        }
        other => panic!("expected UnsupportedSchema error, got {:?}", other),
    }

    // 2. Flipped model byte check (HashMismatch)
    let mut corrupted_bytes = model_bytes.clone();
    corrupted_bytes[0] ^= 0x01;
    let res = ModelPackage::from_parts(&manifest_json, corrupted_bytes);
    match res {
        Err(PreludioError::ModelUnavailable(ModelUnavailable::HashMismatch {
            expected,
            actual,
        })) => {
            assert_eq!(expected, pkg.manifest().model.sha256);
            assert_ne!(expected, actual);
        }
        other => panic!("expected HashMismatch error, got {:?}", other),
    }

    // 3. TargetPolicy checks
    // If supported_targets is empty (before registration in US4):
    if pkg.manifest().supported_targets.is_empty() {
        let res = Session::open(
            pkg.clone(),
            SessionOptions {
                policy: TargetPolicy::SupportedOnly,
                threads: None,
            },
        );
        match res {
            Err(PreludioError::ModelUnavailable(ModelUnavailable::TargetNotSupported {
                ..
            })) => {}
            other => panic!("expected TargetNotSupported, got {:?}", other),
        }
    }

    // Under Validation policy, Session::open succeeds with verified = false (if not in supported_targets)
    let session = Session::open(
        pkg.clone(),
        SessionOptions {
            policy: TargetPolicy::Validation,
            threads: Some(4),
        },
    )
    .expect("open session in Validation policy");

    assert_eq!(session.execution().provider, Provider::Cpu);
    if pkg.manifest().supported_targets.is_empty() {
        assert!(!session.execution().verified);
        assert_eq!(session.execution().target_id, None);
    }

    // 4. Parity check on reference/input.npy vs reference/reference.npy
    let input_npy_path = pkg_dir.join("reference/input.npy");
    let reference_npy_path = pkg_dir.join("reference/reference.npy");

    if !input_npy_path.exists() || !reference_npy_path.exists() {
        eprintln!(
            "[SKIP] reference/input.npy or reference/reference.npy missing under {}",
            pkg_dir.display()
        );
        return;
    }

    let (in_shape, in_f32) = read_npy_f32(&input_npy_path).expect("read input.npy");
    let (ref_shape, ref_f32) = read_npy_f32(&reference_npy_path).expect("read reference.npy");
    assert_eq!(in_shape, pkg.manifest().tensor_input.shape);
    assert_eq!(ref_shape, pkg.manifest().tensor_output.shape);

    let frames = vec![
        FrameMeta {
            id: "AZ9002_20231130_092400Z".to_string(),
            source: "fixture".to_string(),
            valid_time: 1701336240, // 09:24:00Z
        },
        FrameMeta {
            id: "AZ9002_20231130_093000Z".to_string(),
            source: "fixture".to_string(),
            valid_time: 1701336600, // 09:30:00Z
        },
        FrameMeta {
            id: "AZ9002_20231130_093600Z".to_string(),
            source: "fixture".to_string(),
            valid_time: 1701336960, // 09:36:00Z
        },
        FrameMeta {
            id: "AZ9002_20231130_094200Z".to_string(),
            source: "fixture".to_string(),
            valid_time: 1701337320, // 09:42:00Z
        },
    ];

    let forecast = session
        .predict_tensor(&in_f32, &frames)
        .expect("predict_tensor failed");

    assert_eq!(
        forecast.values.len(),
        ref_f32.len(),
        "forecast output length mismatch"
    );

    let mut failures = 0;
    let mut max_abs = 0.0f32;
    for (i, (&actual, &expected)) in forecast.values.iter().zip(ref_f32.iter()).enumerate() {
        if !actual.is_finite() || !expected.is_finite() {
            failures += 1;
            if failures <= 5 {
                eprintln!(
                    "Parity failure at {}: actual={} expected={} (non-finite value)",
                    i, actual, expected
                );
            }
            continue;
        }
        let abs = (actual - expected).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        let tol = 1e-5 + 1e-4 * expected.abs();
        if abs > tol {
            failures += 1;
            if failures <= 5 {
                eprintln!(
                    "Parity failure at {}: actual={} expected={} diff={} tol={}",
                    i, actual, expected, abs, tol
                );
            }
        }
    }

    assert_eq!(
        failures,
        0,
        "Parity check failed with {} failures out of {}! max_abs={:e}",
        failures,
        forecast.values.len(),
        max_abs
    );
}

#[test]
fn test_replay_target_verification_includes_thread_and_engine_version() {
    let pkg_dir = match model_package() {
        Some(dir) => dir,
        None => return,
    };
    let package = ModelPackage::open(&pkg_dir).expect("open model package");
    let mut manifest = package.manifest().clone();
    let platform = current_platform();
    let mut outdated_version = supported_cpu_target("outdated-version", &platform, 4);
    outdated_version.engine_version = "1.30.0".to_string();
    manifest.supported_targets = vec![
        outdated_version,
        supported_cpu_target("matching-target", &platform, 4),
    ];
    let temp_package = temporary_model_package(&pkg_dir, &manifest);
    let package = ModelPackage::open(temp_package.path()).expect("open adjusted model package");

    let unsupported = Session::open(
        package.clone(),
        SessionOptions {
            policy: TargetPolicy::SupportedOnly,
            threads: Some(5),
        },
    );
    match unsupported {
        Err(PreludioError::ModelUnavailable(ModelUnavailable::TargetNotSupported { .. })) => {}
        other => panic!(
            "expected TargetNotSupported for unmatched thread count, got {:?}",
            other
        ),
    }

    let overridden = Session::open(
        package.clone(),
        SessionOptions {
            policy: TargetPolicy::Validation,
            threads: Some(5),
        },
    )
    .expect("open validation session with unmatched thread count");
    assert_eq!(overridden.execution().threads, 5);
    assert!(!overridden.execution().verified);
    assert_eq!(overridden.execution().target_id, None);
    drop(overridden);

    let matching = Session::open(
        package,
        SessionOptions {
            policy: TargetPolicy::Validation,
            threads: Some(4),
        },
    )
    .expect("open validation session with matching target configuration");
    assert!(matching.execution().verified);
    assert_eq!(
        matching.execution().target_id.as_deref(),
        Some("matching-target")
    );
    assert_eq!(matching.execution().engine_version, "1.31.0");
}

#[test]
fn test_replay_rejects_runtime_output_shape_mismatch() {
    let pkg_dir = match model_package() {
        Some(dir) => dir,
        None => return,
    };
    let package = ModelPackage::open(&pkg_dir).expect("open model package");
    let mut manifest = package.manifest().clone();
    let original_output_count = manifest
        .tensor_output
        .shape
        .iter()
        .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
        .expect("output element count");
    change_output_axes_preserving_element_count(&mut manifest);
    assert_ne!(
        manifest.tensor_output.shape,
        package.manifest().tensor_output.shape
    );
    assert_eq!(
        manifest
            .tensor_output
            .shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension)),
        Some(original_output_count)
    );

    let temp_package = temporary_model_package(&pkg_dir, &manifest);
    let package = ModelPackage::open(temp_package.path()).expect("open adjusted model package");
    match Session::open(
        package,
        SessionOptions {
            policy: TargetPolicy::Validation,
            threads: None,
        },
    ) {
        Err(PreludioError::ModelUnavailable(ModelUnavailable::TensorMismatch { .. })) => {}
        other => panic!(
            "expected TensorMismatch when opening session with altered output axes, got {:?}",
            other
        ),
    }
}

#[cfg(feature = "cli")]
#[test]
fn test_replay_raw_frames_match_recorded_preprocessed_tensor() {
    let pkg_dir = match model_package() {
        Some(dir) => dir,
        None => return,
    };
    let package = ModelPackage::open(&pkg_dir).expect("open model package");
    let reference_input_path = pkg_dir.join("reference/input.npy");
    let frame_paths = [
        "AZ9002_20231130_092400Z.png",
        "AZ9002_20231130_093000Z.png",
        "AZ9002_20231130_093600Z.png",
        "AZ9002_20231130_094200Z.png",
    ]
    .map(|name| pkg_dir.join("reference/frames").join(name));

    if !reference_input_path.exists() || !frame_paths.iter().all(|path| path.exists()) {
        eprintln!(
            "[SKIP] recorded input or reference/frames PNGs missing under {}",
            pkg_dir.display()
        );
        return;
    }

    let (reference_shape, reference_input) =
        read_npy_f32(&reference_input_path).expect("read recorded reference/input.npy");
    let frame_ids = [
        "AZ9002_20231130_092400Z",
        "AZ9002_20231130_093000Z",
        "AZ9002_20231130_093600Z",
        "AZ9002_20231130_094200Z",
    ];
    let valid_times = [1701336240, 1701336600, 1701336960, 1701337320];
    let frames = frame_paths
        .iter()
        .zip(frame_ids)
        .zip(valid_times)
        .map(|((path, id), valid_time)| {
            let bytes = fs::read(path).expect("read reference PNG");
            let decoder = png::Decoder::new(bytes.as_slice());
            let mut reader = decoder.read_info().expect("decode reference PNG header");
            let (height, width, color_type, bit_depth) = {
                let info = reader.info();
                (
                    info.height as usize,
                    info.width as usize,
                    info.color_type,
                    info.bit_depth,
                )
            };
            assert_eq!(color_type, png::ColorType::Grayscale);
            assert_eq!(bit_depth, png::BitDepth::Eight);
            assert_eq!(
                [height, width],
                package.manifest().observation.grid.shape,
                "reference frame grid mismatch"
            );
            let mut values = vec![0; reader.output_buffer_size()];
            let output_info = reader
                .next_frame(&mut values)
                .expect("decode reference PNG");
            values.truncate(output_info.buffer_size());
            preludio::observation::ObservationFrame {
                id: id.to_string(),
                source: "fixture".to_string(),
                valid_time,
                values: preludio::observation::FrameValues::U8(values),
                quality: None,
            }
        })
        .collect();
    let sequence = preludio::observation::ObservationSequence {
        grid: preludio::observation::GridDescriptor {
            id: package.manifest().observation.grid.id.clone(),
            shape: package.manifest().observation.grid.shape,
            georeference: None,
        },
        encoding: package.manifest().observation.encoding.clone(),
        zero_threshold: None,
        frames,
    };

    assert_eq!(reference_shape, package.manifest().tensor_input.shape);
    let preprocessed =
        preludio::preprocess(package.manifest(), &sequence).expect("preprocess reference frames");
    assert_eq!(preprocessed.len(), reference_input.len());
    let first_mismatch = preprocessed
        .iter()
        .zip(&reference_input)
        .position(|(actual, expected)| actual.to_bits() != expected.to_bits());
    assert_eq!(
        first_mismatch, None,
        "preprocessed input differs at {first_mismatch:?}"
    );
}

#[cfg(feature = "cli")]
#[test]
fn test_replay_record_rejects_fewer_than_six_total_runs_before_package_load() {
    static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);
    let temp_dir = std::env::temp_dir().join(format!(
        "preludio-replay-runs-{}-{}",
        std::process::id(),
        NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&temp_dir).expect("create temporary replay directory");
    let _temp_dir = TemporaryModelPackage(temp_dir.clone());
    let record_dir = temp_dir.join("record");
    let missing_package = temp_dir.join("missing-package");

    let output = Command::new(env!("CARGO_BIN_EXE_preludio"))
        .args(["replay", "--package"])
        .arg(&missing_package)
        .args(["--tensor", "missing-input.npy", "--runs", "5", "--record"])
        .arg(&record_dir)
        .output()
        .expect("run replay CLI");
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert_eq!(output.status.code(), Some(2), "stderr: {stderr}");
    assert!(stderr.contains("at least 6 total runs"), "stderr: {stderr}");
    assert!(!stderr.contains("missing-package"), "stderr: {stderr}");
    assert!(!record_dir.exists(), "record directory was created early");
}
