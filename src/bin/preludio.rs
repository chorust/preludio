use preludio::error::{InputViolation, ModelUnavailable, PreludioError};

use preludio::manifest::ModelPackage;
use preludio::observation::{
    FrameMeta, FrameValues, GridDescriptor, ObservationFrame, ObservationSequence,
};
use preludio::session::{Session, SessionOptions, TargetPolicy};
use preludio::time::parse_rfc3339_utc;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const REQUIRED_RECORDED_WARM_RUNS: usize = 5;
const DEFAULT_RUNS: usize = REQUIRED_RECORDED_WARM_RUNS + 1;

const DEFAULT_UNRECORDED_RUNS: usize = 1;

fn resolve_runs(explicit_runs: Option<usize>, record_requested: bool) -> usize {
    explicit_runs.unwrap_or(if record_requested {
        DEFAULT_RUNS
    } else {
        DEFAULT_UNRECORDED_RUNS
    })
}

fn should_compare_preprocessed_input(
    reference_path: Option<&Path>,
    recorded_input_exists: bool,
) -> bool {
    reference_path.is_some() && recorded_input_exists
}

fn decode_png_grayscale_u8(bytes: &[u8]) -> Result<(usize, usize, Vec<u8>), PreludioError> {
    let decoder = png::Decoder::new(bytes);
    let mut reader = decoder.read_info().map_err(|e| {
        PreludioError::IncompatibleInput(vec![InputViolation::EncodingMismatch {
            expected: "8-bit grayscale PNG".to_string(),
            actual: e.to_string(),
        }])
    })?;
    let (height, width, color_type, bit_depth) = {
        let info = reader.info();
        (
            info.height as usize,
            info.width as usize,
            info.color_type,
            info.bit_depth,
        )
    };
    if color_type != png::ColorType::Grayscale || bit_depth != png::BitDepth::Eight {
        return Err(PreludioError::IncompatibleInput(vec![
            InputViolation::EncodingMismatch {
                expected: "8-bit grayscale PNG".to_string(),
                actual: format!("{:?} {:?}", color_type, bit_depth),
            },
        ]));
    }
    let mut buf = vec![0; reader.output_buffer_size()];
    let output_info = reader.next_frame(&mut buf).map_err(|e| {
        PreludioError::IncompatibleInput(vec![InputViolation::EncodingMismatch {
            expected: "valid PNG frame".to_string(),
            actual: e.to_string(),
        }])
    })?;
    buf.truncate(output_info.buffer_size());
    Ok((height, width, buf))
}

fn parse_npy_header(bytes: &[u8]) -> Result<(Vec<usize>, usize), String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("invalid NPY file".to_string());
    }
    let major = bytes[6];
    let (header_len, header_start): (usize, usize) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("truncated NPY header".to_string());
            }
            (
                u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
                12,
            )
        }
        _ => return Err(format!("unsupported NPY version {major}")),
    };
    let data_start = header_start
        .checked_add(header_len)
        .ok_or_else(|| "NPY header length overflows usize".to_string())?;
    let header_bytes = bytes
        .get(header_start..data_start)
        .ok_or_else(|| "truncated NPY header".to_string())?;
    let header = std::str::from_utf8(header_bytes).map_err(|e| e.to_string())?;
    let shape_key = header.find("shape").ok_or("no shape in NPY header")?;
    let shape_tail = &header[shape_key + "shape".len()..];
    let open = shape_tail.find('(').ok_or("malformed NPY shape")?;
    let close = shape_tail[open + 1..]
        .find(')')
        .ok_or("unterminated NPY shape")?
        + open
        + 1;
    let shape: Vec<usize> = shape_tail[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if data_start > bytes.len() {
        return Err("truncated NPY payload".to_string());
    }
    Ok((shape, data_start))
}

fn npy_element_count(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| "NPY element count overflows usize".to_string())
    })
}

fn parse_npy_f32(bytes: &[u8]) -> Result<(Vec<usize>, Vec<f32>), String> {
    let (shape, data_start) = parse_npy_header(bytes)?;
    let data = bytes
        .get(data_start..)
        .ok_or_else(|| "truncated NPY payload".to_string())?;
    let expected_bytes = npy_element_count(&shape)?
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "NPY byte length overflows usize".to_string())?;
    if data.len() != expected_bytes {
        return Err("NPY byte length mismatch".to_string());
    }
    let values = data
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    Ok((shape, values))
}

fn read_npy_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    parse_npy_f32(&bytes)
}

fn read_npy_u8(path: &Path) -> Result<(Vec<usize>, Vec<u8>), String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let (shape, data_start) = parse_npy_header(&bytes)?;
    let data = bytes
        .get(data_start..)
        .ok_or_else(|| "truncated NPY payload".to_string())?;
    let expected_bytes = npy_element_count(&shape)?;
    if data.len() != expected_bytes {
        return Err("NPY byte length mismatch".to_string());
    }
    Ok((shape, data.to_vec()))
}

fn get_peak_rss_mib() -> f64 {
    #[cfg(target_os = "macos")]
    {
        let rusage = nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_SELF);
        match rusage {
            Ok(u) => (u.max_rss() as f64) / (1024.0 * 1024.0),
            Err(_) => 0.0,
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let rusage = nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_SELF);
        match rusage {
            Ok(u) => (u.max_rss() as f64) / 1024.0,
            Err(_) => 0.0,
        }
    }
}

fn compute_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, PartialEq)]
struct TensorComparison {
    max_abs_error: f64,
    mean_abs_error: f64,
    failures: usize,
    elements: usize,
    pass: bool,
}

fn compare_tensors(
    actual_shape: &[usize],
    actual: &[f32],
    expected_shape: &[usize],
    expected: &[f32],
) -> TensorComparison {
    let elements = actual.len().max(expected.len());
    let shape_matches = actual_shape == expected_shape;
    let lengths_match = actual.len() == expected.len();
    let mut failures = 0;
    let mut max_abs_error = 0.0f64;
    let mut sum_abs_error = 0.0f64;
    let mut finite_pairs = 0;

    if !shape_matches {
        failures = elements;
    } else {
        for index in 0..elements {
            let (Some(&actual), Some(&expected)) = (actual.get(index), expected.get(index)) else {
                failures += 1;
                continue;
            };
            if !actual.is_finite() || !expected.is_finite() {
                failures += 1;
                continue;
            }

            let abs_error = (actual as f64 - expected as f64).abs();
            max_abs_error = max_abs_error.max(abs_error);
            sum_abs_error += abs_error;
            finite_pairs += 1;
            let tolerance = 1.0e-5 + 1.0e-4 * (expected as f64).abs();
            if abs_error > tolerance {
                failures += 1;
            }
        }
    }

    let mean_abs_error = if finite_pairs == 0 {
        0.0
    } else {
        sum_abs_error / finite_pairs as f64
    };
    TensorComparison {
        max_abs_error,
        mean_abs_error,
        failures,
        elements,
        pass: shape_matches && lengths_match && failures == 0,
    }
}

fn tensors_match_exactly(
    actual_shape: &[usize],
    actual: &[f32],
    expected_shape: &[usize],
    expected: &[f32],
) -> bool {
    actual_shape == expected_shape
        && actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
}

#[derive(Serialize)]
struct ValidationRecord {
    kind: String,
    target: RecordTarget,
    model_sha256: String,
    input: RecordInput,
    tolerance: Tolerance,
    result: ValidationResult,
    created_at: String,
}

#[derive(Serialize)]
struct MeasurementRecord {
    kind: String,
    target: RecordTarget,
    model_sha256: String,
    load_s: f64,
    cold_s: f64,
    warm_s: Vec<f64>,
    peak_memory_mib: f64,
    memory_method: String,
    budget: Budget,
    pass: bool,
    created_at: String,
}

#[derive(Serialize)]
struct RecordTarget {
    platform: String,
    device: String,
    os: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    browser: Option<String>,
    engine: String,
    engine_version: String,
    provider: String,
    threads: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    cross_origin_isolated: Option<bool>,
}

#[derive(Serialize)]
struct RecordInput {
    entry: String,
    sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    preprocessed_matches_reference: Option<bool>,
}

#[derive(Serialize)]
struct Tolerance {
    rtol: f64,
    atol: f64,
}

#[derive(Serialize)]
struct ValidationResult {
    max_abs_error: f64,
    mean_abs_error: f64,
    failures: usize,
    elements: usize,
    pass: bool,
}

#[derive(Serialize)]
struct Budget {
    warm_max_s: f64,
    memory_max_mib: f64,
}

fn handle_replay() -> Result<i32, PreludioError> {
    let args: Vec<String> = env::args().collect();
    let mut package_path: Option<PathBuf> = None;
    let mut frames_args: Vec<String> = Vec::new();
    let mut tensor_path: Option<PathBuf> = None;
    let mut times_str: Option<String> = None;
    let mut ids_str: Option<String> = None;
    let mut source: String = "preludio-cli".to_string();
    let mut reference_path: Option<PathBuf> = None;
    let mut threads: Option<u32> = None;
    let mut policy = TargetPolicy::SupportedOnly;
    let mut runs: Option<usize> = None;
    let mut record_dir: Option<PathBuf> = None;
    let mut encoding_override: Option<String> = None;
    let mut grid_id_override: Option<String> = None;
    let mut zero_threshold: Option<u8> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--package" => {
                i += 1;
                package_path = Some(PathBuf::from(&args[i]));
            }
            "--frames" => {
                i += 1;
                while i < args.len() && !args[i].starts_with("--") {
                    for part in args[i].split(',') {
                        if !part.trim().is_empty() {
                            frames_args.push(part.trim().to_string());
                        }
                    }
                    i += 1;
                }
                continue;
            }
            "--tensor" => {
                i += 1;
                tensor_path = Some(PathBuf::from(&args[i]));
            }
            "--times" => {
                i += 1;
                times_str = Some(args[i].clone());
            }
            "--ids" => {
                i += 1;
                ids_str = Some(args[i].clone());
            }
            "--source" => {
                i += 1;
                source = args[i].clone();
            }
            "--reference" => {
                i += 1;
                reference_path = Some(PathBuf::from(&args[i]));
            }
            "--threads" => {
                i += 1;
                threads = Some(args[i].parse().unwrap_or(4));
            }
            "--policy" => {
                i += 1;
                match args[i].as_str() {
                    "validation" => policy = TargetPolicy::Validation,
                    "supported" => policy = TargetPolicy::SupportedOnly,
                    _ => {}
                }
            }
            "--runs" => {
                i += 1;
                let value = args.get(i).ok_or_else(|| {
                    PreludioError::IncompatibleInput(vec![InputViolation::EncodingMismatch {
                        expected: "positive integer for --runs".to_string(),
                        actual: "missing value".to_string(),
                    }])
                })?;
                runs = Some(value.parse::<usize>().map_err(|_| {
                    PreludioError::IncompatibleInput(vec![InputViolation::EncodingMismatch {
                        expected: "positive integer for --runs".to_string(),
                        actual: value.clone(),
                    }])
                })?);
            }
            "--record" => {
                i += 1;
                record_dir = Some(PathBuf::from(&args[i]));
            }
            "--encoding" => {
                i += 1;
                encoding_override = Some(args[i].clone());
            }
            "--grid-id" => {
                i += 1;
                grid_id_override = Some(args[i].clone());
            }
            "--zero-threshold" => {
                i += 1;
                let value = args.get(i).ok_or_else(|| {
                    PreludioError::IncompatibleInput(vec![InputViolation::EncodingMismatch {
                        expected: "integer from 0 through 255 for --zero-threshold".to_string(),
                        actual: "missing value".to_string(),
                    }])
                })?;
                zero_threshold = Some(value.parse::<u8>().map_err(|_| {
                    PreludioError::IncompatibleInput(vec![InputViolation::EncodingMismatch {
                        expected: "integer from 0 through 255 for --zero-threshold".to_string(),
                        actual: value.clone(),
                    }])
                })?);
            }
            _ => {}
        }
        i += 1;
    }
    let runs = resolve_runs(runs, record_dir.is_some());

    if runs == 0 {
        return Err(PreludioError::IncompatibleInput(vec![
            InputViolation::EncodingMismatch {
                expected: "positive integer for --runs".to_string(),
                actual: "0".to_string(),
            },
        ]));
    }
    if record_dir.is_some() && runs < DEFAULT_RUNS {
        return Err(PreludioError::IncompatibleInput(vec![
            InputViolation::EncodingMismatch {
                expected: format!(
                    "at least {} total runs for --record (one cold and {} warm)",
                    DEFAULT_RUNS, REQUIRED_RECORDED_WARM_RUNS
                ),
                actual: runs.to_string(),
            },
        ]));
    }

    let pkg_path = package_path.ok_or_else(|| {
        PreludioError::ModelUnavailable(ModelUnavailable::FileMissing {
            path: "--package required".to_string(),
        })
    })?;

    let load_start = Instant::now();
    let package = ModelPackage::open(&pkg_path)?;
    let session = Session::open(package.clone(), SessionOptions { policy, threads })?;
    let load_s = load_start.elapsed().as_secs_f64();

    // Parse times
    let mut times: Vec<i64> = Vec::new();
    if let Some(ref t_str) = times_str {
        for s in t_str.split(',') {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                times.push(parse_rfc3339_utc(trimmed)?);
            }
        }
    }

    let mut ids: Vec<String> = Vec::new();
    if let Some(ref id_str) = ids_str {
        for s in id_str.split(',') {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                ids.push(trimmed.to_string());
            }
        }
    }

    let encoding =
        encoding_override.unwrap_or_else(|| package.manifest.observation.encoding.clone());
    let grid_id = grid_id_override.unwrap_or_else(|| package.manifest.observation.grid.id.clone());

    let (forecast, cold_s, warm_s, input_entry, input_sha256, preprocessed_matches) =
        if let Some(t_path) = &tensor_path {
            if zero_threshold.is_some() {
                return Err(PreludioError::IncompatibleInput(vec![
                    InputViolation::EncodingMismatch {
                        expected: "--zero-threshold with raw frame sequences".to_string(),
                        actual: "--tensor input".to_string(),
                    },
                ]));
            }
            let input_bytes = fs::read(t_path).map_err(|error| {
                PreludioError::IncompatibleInput(vec![InputViolation::GridMismatch {
                    expected: "valid npy tensor".to_string(),
                    actual: error.to_string(),
                }])
            })?;
            let (in_shape, in_f32) = parse_npy_f32(&input_bytes).map_err(|error| {
                PreludioError::IncompatibleInput(vec![InputViolation::GridMismatch {
                    expected: "valid npy tensor".to_string(),
                    actual: error,
                }])
            })?;
            if in_shape != package.manifest.tensor_input.shape {
                return Err(PreludioError::IncompatibleInput(vec![
                    InputViolation::GridMismatch {
                        expected: format!("tensor shape {:?}", package.manifest.tensor_input.shape),
                        actual: format!("tensor shape {:?}", in_shape),
                    },
                ]));
            }
            let input_sha = compute_sha256(&input_bytes);

            let mut frame_metas = Vec::new();
            for (idx, &t) in times.iter().enumerate() {
                let id = ids
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("frame-{}", idx));
                frame_metas.push(FrameMeta {
                    id,
                    source: source.clone(),
                    valid_time: t,
                });
            }

            let cold_start = Instant::now();
            let first_fc = session.predict_tensor(&in_f32, &frame_metas)?;
            let cold = cold_start.elapsed().as_secs_f64();

            let mut warm_times = Vec::new();
            for _ in 1..runs {
                let start = Instant::now();
                session.predict_tensor(&in_f32, &frame_metas)?;
                warm_times.push(start.elapsed().as_secs_f64());
            }

            (
                first_fc,
                cold,
                warm_times,
                "tensor".to_string(),
                input_sha,
                None,
            )
        } else if !frames_args.is_empty() {
            let mut frames = Vec::new();
            let mut sequence_shape: Option<[usize; 2]> = None;
            let mut raw_bytes_for_hash = Vec::new();

            if frames_args.len() == 1 && frames_args[0].ends_with(".npy") {
                let (shape, u8_data) = read_npy_u8(Path::new(&frames_args[0])).map_err(|e| {
                    PreludioError::IncompatibleInput(vec![InputViolation::EncodingMismatch {
                        expected: "npy uint8".to_string(),
                        actual: e,
                    }])
                })?;
                raw_bytes_for_hash = u8_data.clone();
                let (frame_count, input_shape) = match shape.as_slice() {
                    [height, width] => (1, [*height, *width]),
                    [frame_count, height, width] => (*frame_count, [*height, *width]),
                    _ => {
                        return Err(PreludioError::IncompatibleInput(vec![
                            InputViolation::GridMismatch {
                                expected:
                                    "2D [height, width] or 3D [frames, height, width] uint8 NPY"
                                        .to_string(),
                                actual: format!("shape {:?}", shape),
                            },
                        ]));
                    }
                };
                let frame_size = input_shape[0] * input_shape[1];
                sequence_shape = Some(input_shape);
                for idx in 0..frame_count {
                    let slice = &u8_data[idx * frame_size..(idx + 1) * frame_size];
                    let vt = times.get(idx).copied().unwrap_or(0);
                    let id = ids
                        .get(idx)
                        .cloned()
                        .unwrap_or_else(|| format!("frame-{}", idx));
                    frames.push(ObservationFrame {
                        id,
                        source: source.clone(),
                        valid_time: vt,
                        values: FrameValues::U8(slice.to_vec()),
                        quality: None,
                    });
                }
            } else {
                // PNGs
                for (idx, png_path_str) in frames_args.iter().enumerate() {
                    let path = Path::new(png_path_str);
                    let bytes = fs::read(path).map_err(|e| {
                        PreludioError::IncompatibleInput(vec![InputViolation::EncodingMismatch {
                            expected: format!("reading {}", path.display()),
                            actual: e.to_string(),
                        }])
                    })?;
                    raw_bytes_for_hash.extend_from_slice(&bytes);
                    let (height, width, u8_buf) = decode_png_grayscale_u8(&bytes)?;
                    let frame_shape = [height, width];
                    if let Some(expected_shape) = sequence_shape {
                        if frame_shape != expected_shape {
                            return Err(PreludioError::IncompatibleInput(vec![
                                InputViolation::GridMismatch {
                                    expected: format!(
                                        "{}x{} PNG frames",
                                        expected_shape[0], expected_shape[1]
                                    ),
                                    actual: format!("{}x{} in {}", height, width, path.display()),
                                },
                            ]));
                        }
                    } else {
                        sequence_shape = Some(frame_shape);
                    }
                    let vt = times.get(idx).copied().unwrap_or(0);
                    let id = ids.get(idx).cloned().unwrap_or_else(|| {
                        path.file_stem()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_else(|| format!("frame-{}", idx))
                    });
                    frames.push(ObservationFrame {
                        id,
                        source: source.clone(),
                        valid_time: vt,
                        values: FrameValues::U8(u8_buf),
                        quality: None,
                    });
                }
            }

            let input_sha = compute_sha256(&raw_bytes_for_hash);
            let seq = ObservationSequence {
                grid: GridDescriptor {
                    id: grid_id,
                    shape: sequence_shape.ok_or_else(|| {
                        PreludioError::IncompatibleInput(vec![InputViolation::GridMismatch {
                            expected: "non-empty raw frame sequence".to_string(),
                            actual: "no frame dimensions".to_string(),
                        }])
                    })?,
                    georeference: None,
                },
                encoding,
                zero_threshold,
                frames,
            };

            let recorded_input_path = pkg_path.join("reference/input.npy");
            let preprocessed_matches = if should_compare_preprocessed_input(
                reference_path.as_deref(),
                recorded_input_path.is_file(),
            ) {
                let (recorded_shape, recorded_input) =
                    read_npy_f32(&recorded_input_path).map_err(|error| {
                        PreludioError::IncompatibleInput(vec![InputViolation::GridMismatch {
                            expected: "valid reference/input.npy tensor".to_string(),
                            actual: error,
                        }])
                    })?;
                let preprocessed_input = preludio::preprocess(&package.manifest, &seq)?;
                Some(tensors_match_exactly(
                    &package.manifest.tensor_input.shape,
                    &preprocessed_input,
                    &recorded_shape,
                    &recorded_input,
                ))
            } else {
                None
            };

            let cold_start = Instant::now();
            let first_fc = session.predict(&seq)?;
            let cold = cold_start.elapsed().as_secs_f64();

            let mut warm_times = Vec::new();
            for _ in 1..runs {
                let start = Instant::now();
                session.predict(&seq)?;
                warm_times.push(start.elapsed().as_secs_f64());
            }

            (
                first_fc,
                cold,
                warm_times,
                "raw_frames".to_string(),
                input_sha,
                preprocessed_matches,
            )
        } else {
            return Err(PreludioError::IncompatibleInput(vec![
                InputViolation::FrameCount {
                    expected: package.manifest.observation.frames,
                    actual: 0,
                },
            ]));
        };

    let mut parity_passed = true;
    let mut val_result = None;
    if let Some(matches) = preprocessed_matches {
        eprintln!("Preprocessed input matches reference/input.npy: {matches}");
    }

    if let Some(ref_path) = &reference_path {
        let (ref_shape, ref_f32) =
            read_npy_f32(ref_path).map_err(|_e| PreludioError::InvalidOutput {
                execution: session.execution().clone(),
                reason: preludio::error::OutputViolation::Shape,
            })?;

        let comparison = compare_tensors(
            &package.manifest.tensor_output.shape,
            &forecast.values,
            &ref_shape,
            &ref_f32,
        );
        let preprocessing_pass = preprocessed_matches != Some(false);
        let pass = comparison.pass && preprocessing_pass;
        if !pass {
            parity_passed = false;
        }
        if package.manifest.tensor_output.shape != ref_shape {
            eprintln!(
                "Reference tensor shape mismatch: expected {:?}, actual {:?}",
                package.manifest.tensor_output.shape, ref_shape
            );
        }

        eprintln!(
            "Parity check: max_abs_err={:.9e} mean_abs_err={:.9e} failures={}/{} output_pass={} validation_pass={}",
            comparison.max_abs_error,
            comparison.mean_abs_error,
            comparison.failures,
            comparison.elements,
            comparison.pass,
            pass
        );

        val_result = Some(ValidationResult {
            max_abs_error: comparison.max_abs_error,
            mean_abs_error: comparison.mean_abs_error,
            failures: comparison.failures,
            elements: comparison.elements,
            pass,
        });
    }

    // Print stdout one-line ForecastMetadata JSON
    let stdout_json =
        serde_json::to_string(&forecast.metadata).map_err(|e| PreludioError::ExecutionFailed {
            execution: session.execution().clone(),
            message: e.to_string(),
        })?;
    println!("{}", stdout_json);

    // If --record was specified
    if let Some(ref r_dir) = record_dir {
        fs::create_dir_all(r_dir).map_err(|e| PreludioError::ExecutionFailed {
            execution: session.execution().clone(),
            message: format!("cannot create record dir: {}", e),
        })?;

        let now =
            preludio::time::format_rfc3339_utc(time::OffsetDateTime::now_utc().unix_timestamp())?;
        let peak_mem = get_peak_rss_mib();

        let target_name = session
            .execution()
            .target_id
            .clone()
            .unwrap_or_else(|| session.execution().platform.clone());

        let target = RecordTarget {
            platform: session.execution().platform.clone(),
            device: "Apple M4 Pro".to_string(),
            os: "macOS 25.6.0".to_string(),
            browser: None,
            engine: "onnxruntime".to_string(),
            engine_version: session.execution().engine_version.clone(),
            provider: "cpu".to_string(),
            threads: session.execution().threads,
            cross_origin_isolated: None,
        };

        if let Some(res) = val_result {
            let val_record = ValidationRecord {
                kind: "validation".to_string(),
                target: RecordTarget {
                    platform: target.platform.clone(),
                    device: target.device.clone(),
                    os: target.os.clone(),
                    browser: None,
                    engine: target.engine.clone(),
                    engine_version: target.engine_version.clone(),
                    provider: target.provider.clone(),
                    threads: target.threads,
                    cross_origin_isolated: None,
                },
                model_sha256: package.manifest.model.sha256.clone(),
                input: RecordInput {
                    entry: input_entry,
                    sha256: input_sha256.clone(),
                    preprocessed_matches_reference: preprocessed_matches,
                },
                tolerance: Tolerance {
                    rtol: 0.0001,
                    atol: 0.00001,
                },
                result: res,
                created_at: now.clone(),
            };
            let val_json = serde_json::to_string_pretty(&val_record).unwrap();
            fs::write(
                r_dir.join(format!("validation-{}.json", target_name)),
                val_json,
            )
            .unwrap();
        }

        let max_warm = warm_s.iter().copied().fold(0.0f64, f64::max);
        let meas_pass = max_warm <= 8.0 && peak_mem <= 1740.8;

        let meas_record = MeasurementRecord {
            kind: "measurement".to_string(),
            target,
            model_sha256: package.manifest.model.sha256.clone(),
            load_s,
            cold_s,
            warm_s,
            peak_memory_mib: peak_mem,
            memory_method: "rusage_max_rss".to_string(),
            budget: Budget {
                warm_max_s: 8.0,
                memory_max_mib: 1740.8,
            },
            pass: meas_pass,
            created_at: now,
        };
        let meas_json = serde_json::to_string_pretty(&meas_record).unwrap();
        fs::write(
            r_dir.join(format!("measurement-{}.json", target_name)),
            meas_json,
        )
        .unwrap();
    }

    if !parity_passed {
        Ok(5)
    } else {
        Ok(0)
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: preludio replay [options]");
        return ExitCode::from(2);
    }

    if args[1] == "replay" {
        match handle_replay() {
            Ok(code) => ExitCode::from(code as u8),
            Err(err) => {
                let err_json = serde_json::to_string(&err).unwrap_or_else(|_| "{}".to_string());
                eprintln!("{}", err_json);
                match err {
                    PreludioError::IncompatibleInput(_) => ExitCode::from(2),
                    PreludioError::ModelUnavailable(_) => ExitCode::from(3),
                    PreludioError::ExecutionFailed { .. } | PreludioError::InvalidOutput { .. } => {
                        ExitCode::from(4)
                    }
                }
            }
        }
    } else {
        eprintln!("Unknown subcommand: {}", args[1]);
        ExitCode::from(2)
    }
}

#[cfg(test)]
mod tests {
    use super::{compare_tensors, tensors_match_exactly};

    #[test]
    fn reference_comparison_rejects_shape_and_length_mismatches() {
        let shape_mismatch = compare_tensors(&[1, 2], &[0.25, 0.75], &[2, 1], &[0.25, 0.75]);
        assert!(!shape_mismatch.pass);
        assert_eq!(shape_mismatch.failures, 2);

        let length_mismatch = compare_tensors(&[2], &[0.25, 0.75], &[2], &[0.25]);
        assert!(!length_mismatch.pass);
        assert_eq!(length_mismatch.failures, 1);
        assert_eq!(length_mismatch.elements, 2);
    }

    #[test]
    fn reference_comparison_rejects_non_finite_values_on_either_side() {
        let comparison = compare_tensors(&[2], &[f32::NAN, 1.0], &[2], &[0.0, f32::INFINITY]);
        assert!(!comparison.pass);
        assert_eq!(comparison.failures, 2);
        assert_eq!(comparison.max_abs_error, 0.0);
        assert_eq!(comparison.mean_abs_error, 0.0);
    }

    #[test]
    fn preprocessing_reference_comparison_requires_shape_and_bit_exact_values() {
        assert!(tensors_match_exactly(&[2], &[0.0, 0.5], &[2], &[0.0, 0.5]));
        assert!(!tensors_match_exactly(
            &[1, 2],
            &[0.0, 0.5],
            &[2],
            &[0.0, 0.5]
        ));
        assert!(!tensors_match_exactly(&[2], &[0.0, 0.5], &[2], &[0.0]));
        assert!(!tensors_match_exactly(
            &[2],
            &[0.0, 0.5],
            &[2],
            &[-0.0, 0.5]
        ));
    }
}
