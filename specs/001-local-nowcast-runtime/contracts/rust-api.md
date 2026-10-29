# Contract: Rust public API (`preludio` crate)

**Plan**: [../plan.md](../plan.md) | **Data model**: [../data-model.md](../data-model.md)

签名级契约，不是实现。字段语义与校验规则以 data-model 为准。`native` 特性（默认，非 wasm32）提供推理；`wasm32` 目标只编译共享核心并经 wasm-bindgen 导出（见 [web-api.md](web-api.md)）。

```rust
// ---- 共享核心（native 与 wasm32 都编译） ----

pub struct ModelPackage { /* parsed manifest + verified model bytes/path */ }

impl ModelPackage {
    /// 读取 `dir/manifest.json`，校验 schema、计算并比对 `model.sha256`。
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(dir: impl AsRef<std::path::Path>) -> Result<Self, PreludioError>;
    /// Web 与测试用：manifest 文本 + 模型字节；同样校验哈希。
    pub fn from_parts(manifest_json: &str, model: Vec<u8>) -> Result<Self, PreludioError>;
    pub fn manifest(&self) -> &Manifest;
}

pub struct ObservationSequence {
    pub grid: GridDescriptor,           // shape = 实际输入 [h, w]；manifest grid id/shape 不作接受限制
    pub encoding: String,                // 当前模型：training-grayscale-u8，duckomo-compatible raw u8 灰度
    pub frames: Vec<ObservationFrame>,   // 固定四帧，oldest → newest
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zero_threshold: Option<u8>,      // None 默认不变换；严格小于阈值的像素在预处理前归零
}


pub struct ObservationFrame {
    pub id: String,
    pub source: String,
    pub valid_time: i64,               // UTC Unix seconds
    pub values: FrameValues,
    pub quality: Option<Vec<PixelQuality>>,
}

pub enum FrameValues { U8(Vec<u8>), F32(Vec<f32>) }

#[repr(u8)]
pub enum PixelQuality { Valid = 0, Missing = 1, OutsideCoverage = 2 }

pub struct GridDescriptor {
    pub id: String,                      // 来源描述信息；不与 manifest grid id 比较
    pub shape: [usize; 2],               // [h, w]，均为正，且所有输入帧同形
    pub georeference: Option<GeoReference>, // 可省略；无地理参照要求
}

/// 推理前校验；返回全部违规（V-01..V-08）。
pub fn validate(manifest: &Manifest, seq: &ObservationSequence) -> Result<(), PreludioError>;

/// 按模型 preprocessing 生成 `[1, frames, H_tensor, W_tensor]` f32；先应用可选 u8 阈值，再 /255、resize 与舍入。
pub fn preprocess(manifest: &Manifest, seq: &ObservationSequence) -> Result<Vec<f32>, PreludioError>;

/// 输出校验 + 有效时间 + 来源，组装 Forecast。`raw` 为引擎输出。
pub fn finish_forecast(
    package: &ModelPackage,
    seq: &ObservationSequence,
    raw: Vec<f32>,
    execution: ExecutionInfo,
    generated_at: i64,
) -> Result<Forecast, PreludioError>;

pub struct Forecast {
    pub values: Vec<f32>,              // [leads, h_out, w_out]
    pub metadata: ForecastMetadata,    // serde → forecast-metadata.schema.json
}

pub struct ExecutionInfo {
    pub target_id: Option<String>,
    pub verified: bool,
    pub engine: Engine,                // OnnxRuntime | OnnxRuntimeWeb
    pub engine_version: String,
    pub provider: Provider,            // Cpu | Wasm
    pub threads: u32,
    pub platform: String,
    pub cross_origin_isolated: Option<bool>,
}

#[derive(Debug, thiserror::Error)]
pub enum PreludioError {
    #[error("model unavailable: {0}")]
    ModelUnavailable(ModelUnavailable),
    #[error("incompatible input: {} violation(s)", .0.len())]
    IncompatibleInput(Vec<InputViolation>),
    #[error("execution failed on {execution:?}: {message}")]
    ExecutionFailed { execution: ExecutionInfo, message: String },
    #[error("invalid output: {reason:?}")]
    InvalidOutput { execution: ExecutionInfo, reason: OutputViolation },
}
// ModelUnavailable / InputViolation / OutputViolation 变体见 data-model §2、§5。
// 所有错误类型实现 serde::Serialize，供 Web 端转为结构化 JS 错误。

// ---- native 特性 ----

pub enum TargetPolicy { SupportedOnly, Validation }

pub struct SessionOptions {
    pub policy: TargetPolicy,          // 默认 SupportedOnly
    pub threads: Option<u32>,          // None 遵从目标声明；Some(n) 请求实际线程数 n
}

pub struct Session { /* ort::Session + ModelPackage + matched target */ }

impl Session {
    /// 匹配当前平台、引擎版本、provider 与线程数的 supported_targets；SupportedOnly 下无匹配 → TargetNotSupported。
    /// ORT 会话：AutoDevicePolicy::PreferCPU、CPU arena on、memory pattern off。
    /// 加载后比对 ONNX 输入输出名称/shape 与 manifest。
    pub fn open(package: ModelPackage, options: SessionOptions) -> Result<Self, PreludioError>;
    pub fn execution(&self) -> &ExecutionInfo;
    /// 同步阻塞：validate → preprocess → run → finish_forecast。
    /// 调用方负责放到后台线程（例如 tokio::task::spawn_blocking）。`&self`，可跨线程共享会话。
    pub fn predict(&self, seq: &ObservationSequence) -> Result<Forecast, PreludioError>;
    /// 开发者重放：跳过预处理；只要求帧数正确且时间严格递增，不要求固定 cadence。
    pub fn predict_tensor(&self, input: &[f32], frames: &[FrameMeta]) -> Result<Forecast, PreludioError>;
}

pub struct FrameMeta { pub id: String, pub source: String, pub valid_time: i64 }

// Session: Send + Sync（由 ort::Session 保证，crate 内无 unsafe）。
```

原生 manifest 中 `supported_targets[].threads = "default"` 表示 runtime 固定默认线程数 4；Python registry 保留符号值 `"default"`，validation/measurement records 则记录实际线程数，二者必须保持一致。`SessionOptions.threads = None` 遵从目标声明；显式 `Some(n)` 只有在目标声明数值线程数 `n` 时才验证匹配，不能匹配 `"default"`。`Validation` 可运行未匹配的线程 override，但此时 `execution.verified = false`、`target_id = None`。原生与 Web 推理均在组装预测前逐维比较实际输出 tensor shape 与 manifest；元素数量相同不能替代形状相同。

## CLI（`cli` 特性，二进制 `preludio`）

```text
preludio replay --package <dir> (--frames <frames.npy|png,png,png,png> | --tensor <input.npy>)
                --times <rfc3339,...> [--ids <id,...>] [--source <s>]
                [--reference <reference.npy>] [--threads <n>] [--policy supported|validation]
                [--zero-threshold <u8>] [--runs <n>] [--record <dir>]
```

- 退出码：0 成功且（若给出 `--reference`）一致性通过；2 输入不兼容；3 模型不可用；4 执行失败或输出无效；5 一致性未通过。
- stdout：一行 JSON `ForecastMetadata`；`--record` 写入 ValidationRecord 与 MeasurementRecord（[validation-record.schema.json](validation-record.schema.json)）。
- `--frames` 为 PNG 时用纯 Rust 解码为 8 位灰度，非灰度或非 8 位 PNG 视为 `EncodingMismatch`。

- 未使用 `--record` 时，`--runs` 默认执行 1 次推理；使用 `--record` 时默认执行 6 次（1 次冷运行、5 次热运行）。显式传入 `--runs` 会覆盖任一默认值；启用 `--record` 后若次数小于 6，会在打开模型包前以退出码 2 拒绝。
- 使用原始 `--frames` 并显式提供 `--reference` 时，若包中存在 `reference/input.npy`，CLI 在校验 manifest/reference 形状后，将预处理 tensor 与它按位比较。不匹配时验证失败并以退出码 5 退出，与是否启用 `--record` 无关。

`--zero-threshold <u8>` 对每帧 raw u8 像素严格按 `< threshold` 置零，发生在 `/255` 与 resize 之前；省略时不置零，0 无效果。
