use preludio::manifest::{Manifest, TargetThreads};
use serde_json::{json, Value};

fn valid_manifest() -> Value {
    serde_json::from_str(include_str!("fixtures/manifest-v1.json"))
        .expect("parse valid manifest fixture")
}

#[test]
fn schema_default_threads_round_trip_and_flat_georeference_are_supported() {
    let mut value = valid_manifest();
    value["supported_targets"][0]["threads"] = json!("default");
    value["observation"]["grid"]["georeferenced"] = json!(true);
    value["observation"]["grid"]["crs"] = json!("EPSG:4326");
    value["observation"]["grid"]["affine"] = json!([1.0, 0.0, 0.0, 0.0, -1.0, 0.0]);
    value["origin"]["checkpoint"]["step"] = json!(-1);

    let manifest: Manifest = serde_json::from_value(value).expect("valid schema manifest");
    assert_eq!(
        manifest.supported_targets[0].threads,
        TargetThreads::Default
    );
    assert_eq!(manifest.origin.checkpoint.step, -1);

    let serialized = serde_json::to_value(manifest).expect("serialize manifest");
    assert_eq!(
        serialized["supported_targets"][0]["threads"],
        json!("default")
    );
    assert_eq!(serialized["observation"]["grid"]["crs"], json!("EPSG:4326"));
    assert!(serialized["observation"]["grid"]
        .get("georeference")
        .is_none());
}

#[test]
fn schema_constraints_are_rejected_during_manifest_deserialization() {
    type InvalidCase = (&'static str, fn(&mut Value));
    let cases: &[InvalidCase] = &[
        ("missing required schema", |value| {
            value.as_object_mut().unwrap().remove("schema");
        }),
        ("unknown top-level property", |value| {
            value["unexpected"] = json!(true);
        }),
        ("unsupported schema version", |value| {
            value["schema"] = json!("preludio.model-package/2");
        }),
        ("empty id", |value| value["id"] = json!("")),
        ("non-ONNX model format", |value| {
            value["model"]["format"] = json!("pytorch");
        }),
        ("zero opset", |value| value["model"]["opset"] = json!(0)),
        ("malformed model hash", |value| {
            value["model"]["sha256"] = json!("A".repeat(64));
        }),
        ("zero observation frames", |value| {
            value["observation"]["frames"] = json!(0);
        }),
        ("zero observation cadence", |value| {
            value["observation"]["cadence_seconds"] = json!(0);
        }),
        ("negative cadence tolerance", |value| {
            value["observation"]["cadence_tolerance_seconds"] = json!(-1);
        }),
        ("unsupported observation encoding", |value| {
            value["observation"]["encoding"] = json!("dbz");
        }),
        ("non-positive grid dimension", |value| {
            value["observation"]["grid"]["shape"] = json!([0, 732]);
        }),
        ("georeferenced grid without CRS and affine", |value| {
            value["observation"]["grid"]["georeferenced"] = json!(true);
        }),
        ("unsupported missing policy", |value| {
            value["observation"]["missing_policy"] = json!("zero");
        }),
        ("unsupported tensor dtype", |value| {
            value["tensor_input"]["dtype"] = json!("float16");
        }),
        ("zero tensor dimension", |value| {
            value["tensor_output"]["shape"] = json!([1, 0, 448, 448]);
        }),
        ("unsupported tensor axis", |value| {
            value["tensor_input"]["axes"][0] = json!("channel");
        }),
        ("inverted tensor range", |value| {
            value["tensor_output"]["range"] = json!([1.0, 0.0]);
        }),
        ("empty preprocessing", |value| {
            value["preprocessing"] = json!([])
        }),
        ("unknown preprocessing property", |value| {
            value["preprocessing"][0]["unexpected"] = json!(true);
        }),
        ("zero resize dimension", |value| {
            value["preprocessing"][1]["height"] = json!(0);
        }),
        ("align corners enabled", |value| {
            value["preprocessing"][1]["align_corners"] = json!(true);
        }),
        ("empty forecast leads", |value| {
            value["forecast"]["lead_minutes"] = json!([])
        }),
        ("zero forecast lead", |value| {
            value["forecast"]["lead_minutes"][0] = json!(0);
        }),
        ("unsupported forecast encoding", |value| {
            value["forecast"]["encoding"] = json!("rain-rate");
        }),
        ("unsupported target engine", |value| {
            value["supported_targets"][0]["engine"] = json!("tensorflow");
        }),
        ("unsupported target provider", |value| {
            value["supported_targets"][0]["provider"] = json!("gpu");
        }),
        ("zero target thread count", |value| {
            value["supported_targets"][0]["threads"] = json!(0);
        }),
        ("invalid target thread sentinel", |value| {
            value["supported_targets"][0]["threads"] = json!("auto");
        }),
        ("unknown nested model property", |value| {
            value["model"]["unexpected"] = json!(true);
        }),
    ];

    for (name, mutate) in cases {
        let mut value = valid_manifest();
        mutate(&mut value);
        assert!(
            serde_json::from_value::<Manifest>(value).is_err(),
            "{name} must be rejected"
        );
    }
}

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
#[test]
fn session_open_rejects_manifest_input_shape_mismatch() {
    use preludio::error::{ModelUnavailable, PreludioError};
    use preludio::manifest::ModelPackage;
    use preludio::session::{Session, SessionOptions, TargetPolicy};
    use std::path::PathBuf;

    let package_dir = std::env::var_os("PRELUDIO_MODEL_PACKAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("artifacts/ghost-unet3d-step-22000")
        });
    let mut package = ModelPackage::open(package_dir).expect("open ONNX model package");
    package.manifest.tensor_input.shape[3] += 1;

    let result = Session::open(
        package,
        SessionOptions {
            policy: TargetPolicy::Validation,
            threads: None,
        },
    );
    assert!(
        matches!(
            result,
            Err(PreludioError::ModelUnavailable(
                ModelUnavailable::TensorMismatch { .. }
            ))
        ),
        "Session::open must reject a manifest input shape that differs from ONNX metadata"
    );
}
