#![forbid(unsafe_code)]

use anyhow::{bail, Context, Result};
use ndarray::{ArrayD, IxDyn};
use native_spike::{compare, read_npy_f32};
use ort::{
    ep::CPU,
    logging::LogLevel,
    session::{builder::{AutoDevicePolicy, GraphOptimizationLevel, SessionBuilder}, Session},
    value::TensorRef,
};
use std::{env, time::Instant};

fn option<'a>(value: &'a str, name: &str) -> Result<bool> {
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => bail!("{name} must be 0 or 1"),
    }
}

fn set_config(builder: SessionBuilder, key: &str, value: &str) -> Result<SessionBuilder> {
    builder
        .with_config_entry(key, value)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

fn main() -> Result<()> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 12 && args.len() != 13 {
        bail!("usage: ort-sweep <disable|basic|extended|layout|all> <default|1|4> <denormal-zero:0|1> <disable-kleidiai:0|1> <fastmath-bfloat16:0|1> <arena:0|1> <mem-pattern:0|1> <run-count> <model.onnx> <input.npy> <reference.npy> [profile.json]");
    }
    let optimization = match args[1].as_str() {
        "disable" => GraphOptimizationLevel::Disable,
        "basic" => GraphOptimizationLevel::Level1,
        "extended" => GraphOptimizationLevel::Level2,
        "layout" => GraphOptimizationLevel::Level3,
        "all" => GraphOptimizationLevel::All,
        _ => bail!("unknown graph optimization level"),
    };
    let denormal_zero = option(&args[3], "denormal-zero")?;
    let disable_kleidiai = option(&args[4], "disable-kleidiai")?;
    let fastmath_bfloat16 = option(&args[5], "fastmath-bfloat16")?;
    let arena = option(&args[6], "arena")?;
    let mem_pattern = option(&args[7], "mem-pattern")?;
    let prefer_cpu = env::var("ORT_SWEEP_PREFER_CPU").map_or(true, |value| value != "0");
    let run_count = args[8].parse::<usize>()?;
    if run_count == 0 {
        bail!("run-count must be at least 1");
    }
    if !matches!(args[2].as_str(), "default" | "1" | "4") {
        bail!("threads must be default, 1, or 4");
    }

    let input_data = read_npy_f32(&args[10])?;
    let reference = read_npy_f32(&args[11])?;
    let input = ArrayD::from_shape_vec(IxDyn(&input_data.shape), input_data.values)?;
    let env = ort::init().with_name("preludio-native-sweep").build()?;
    let mut builder = Session::builder(&env)?;
    builder = builder
        .with_optimization_level(optimization)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    builder = builder
        .with_log_level(LogLevel::Warning)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    if args[2] != "default" {
        builder = builder
            .with_intra_threads(args[2].parse()?)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    }
    if denormal_zero {
        builder = builder
            .with_flush_to_zero()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    }
    if disable_kleidiai {
        builder = set_config(builder, "mlas.disable_kleidiai", "1")?;
    }
    if fastmath_bfloat16 {
        builder = set_config(builder, "mlas.enable_gemm_fastmath_arm64_bfloat16", "1")?;
    }
    builder = builder
        .with_memory_pattern(mem_pattern)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    builder = builder
        .with_execution_providers([CPU::default().with_arena_allocator(arena).build()])
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    if prefer_cpu {
        builder = builder
            .with_auto_device(AutoDevicePolicy::PreferCPU)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    }
    if args.len() == 13 {
        builder = builder
            .with_profiling(&args[12])
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    }

    let session_start = Instant::now();
    let mut session = builder.commit_from_file(&args[9])?;
    let session_s = session_start.elapsed().as_secs_f64();
    println!("ort_info={}", ort::info());
    println!("opt={} threads={} denormal_zero={} disable_kleidiai={} fastmath_bfloat16={} arena={} mem_pattern={} prefer_cpu={} runs={}", args[1], args[2], denormal_zero, disable_kleidiai, fastmath_bfloat16, arena, mem_pattern, prefer_cpu, run_count);
    println!("input_shape={:?} reference_shape={:?} session_s={session_s:.6}", input.shape(), reference.shape);

    let mut seconds = Vec::with_capacity(run_count);
    let mut summaries = Vec::with_capacity(run_count);
    for index in 0..run_count {
        let started = Instant::now();
        let outputs = session.run(ort::inputs! {
            "radar" => TensorRef::from_array_view(&input)?
        })?;
        let elapsed = started.elapsed().as_secs_f64();
        let output = outputs["forecast"].try_extract_array::<f32>()?;
        let actual = output.as_slice().context("output is not contiguous")?;
        let (max_abs, mean_abs, failures) = compare(actual, &reference.values)?;
        let label = if index == 0 { "cold".to_string() } else { format!("run_{}", index) };
        println!("{label}_s={elapsed:.6} max_abs_err={max_abs:.9e} mean_abs_err={mean_abs:.9e} failures={failures}");
        seconds.push(elapsed);
        summaries.push((max_abs, mean_abs, failures));
    }
    if run_count > 1 {
        let warm = &seconds[1..];
        println!("warm_mean_s={:.6} warm_min_s={:.6} warm_max_s={:.6}", warm.iter().sum::<f64>() / warm.len() as f64, warm.iter().copied().fold(f64::INFINITY, f64::min), warm.iter().copied().fold(0.0, f64::max));
    }
    let max_abs = summaries.iter().map(|x| x.0).fold(0.0, f64::max);
    let mean_abs = summaries.iter().map(|x| x.1).sum::<f64>() / summaries.len() as f64;
    let failures = summaries.iter().map(|x| x.2).sum::<usize>();
    println!("overall_max_abs_err={max_abs:.9e} mean_abs_err={mean_abs:.9e} total_failures={failures}");
    if args.len() == 13 {
        println!("profile_path={}", session.end_profiling()?);
    }
    Ok(())
}
