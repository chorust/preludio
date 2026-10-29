# Contract: Web API (`@chorust/preludio` npm package)

**Plan**: [../plan.md](../plan.md) | **Data model**: [../data-model.md](../data-model.md)

TypeScript 封装 = 共享 Rust 核心（wasm-bindgen，负责哈希、校验、预处理、输出校验与来源）+ `onnxruntime-web` 1.30.0 WASM（负责推理，`proxy = true` 在 Worker 中运行）。签名级契约：

```ts
export interface CreateOptions {
  /** 模型包根 URL；读取 `${base}/manifest.json` 与 manifest 中的 model.file。 */
  packageUrl: string;
  /** onnxruntime-web 的 .wasm 资源目录（同源或满足 COEP 的 CORP 头）。 */
  ortWasmPaths?: string;
  /** 默认 SupportedOnly。Validation 仅用于产生验证/测量记录。 */
  policy?: "supported" | "validation";
  /** 覆盖目标声明的线程数（仅 validation 策略生效）。 */
  threads?: number;
  signal?: AbortSignal;              // 中止下载与会话创建；初始化完成后发现已取消时先释放引擎，再拒绝
}

export type PixelQuality = 0 | 1 | 2; // Valid | Missing | OutsideCoverage

export interface ObservationFrame {
  id: string;
  source: string;
  /** RFC 3339，必须带 `Z` 或数值时区偏移；无时区 → IncompatibleInput。 */
  validTime: string;
  values: Uint8Array | Float32Array;
  quality?: Uint8Array;               // PixelQuality per pixel
}

export interface ObservationSequence {
  grid: { id: string; shape: [number, number]; georeference?: { crs: string; affine: number[] } };
  encoding: string;                    // 当前模型：training-grayscale-u8，duckomo-compatible raw u8 灰度
  frames: ObservationFrame[];          // 恰好四帧，oldest → newest，四帧尺寸相同且为正
  /** None 时不置零；严格小于阈值的像素在 /255 与 resize 前置零；0 无效果。 */
  zeroThreshold?: number;              // 整数 0..255
}

export interface Forecast {
  values: Float32Array;               // [leads, h, w]，长度 = leads·h·w
  metadata: ForecastMetadata;         // = forecast-metadata.schema.json
}

export type RunState = "idle" | "running" | "succeeded" | "failed";

export interface NowcastSession {
  readonly execution: ExecutionInfo;  // 实际 engine/provider/threads/crossOriginIsolated
  readonly state: RunState;
  onStateChange(listener: (state: RunState) => void): () => void;
  /** 每次调用内部新建输入 Tensor（proxy 会转移 buffer）。并发调用排队执行。 */
  predict(sequence: ObservationSequence): Promise<Forecast>;
  /** 开发者重放：直接用已预处理张量，推理前校验帧数与时间严格递增，不要求固定 cadence。 */
  predictTensor(input: Float32Array, frames: { id: string; source: string; validTime: string }[]): Promise<Forecast>;
  dispose(): Promise<void>;
}

export function createSession(options: CreateOptions): Promise<NowcastSession>;

export class PreludioError extends Error {
  readonly category: "ModelUnavailable" | "IncompatibleInput" | "ExecutionFailed" | "InvalidOutput";
  /** 与 Rust 错误的 serde 序列化相同结构，例如 { HashMismatch: { expected, actual } }。 */
  readonly detail: unknown;
}
```

## 行为约束

- 当匹配目标要求隔离而 `globalThis.crossOriginIsolated !== true` 时，`createSession` 抛出 `ModelUnavailable` / `IsolationRequired`。实际线程数大于 1 的 WASM 执行始终要求隔离；未登记的 validation 执行也不会降为单线程。若 validation 配置没有精确匹配项，仍遵守该平台支持声明中的隔离要求。
- `supported` 策略要求目标与实际平台、引擎、引擎版本、provider 和线程数完全匹配；同一平台可有多个声明，运行时选择精确匹配项。当前 Ghost 模型包只声明 macOS 原生目标，因此 Chrome、Safari、Firefox 在默认策略下均得到 `TargetNotSupported`。
- 不尝试 WebGPU，不访问除 `packageUrl` 与 `ortWasmPaths` 以外的网络地址；资源缓存由页面（Service Worker / HTTP 缓存）负责，缓存后断网可运行。
- 托管页面必须发送 `Cross-Origin-Opener-Policy: same-origin` 与 `Cross-Origin-Embedder-Policy: require-corp`；跨源资源需要 `Cross-Origin-Resource-Policy`。
- 推理期间主线程不执行引擎计算；`state` 变化在 `predict` 开始与结束时同步发出。
- 当前 Ghost 输入是单通道 `Uint8Array`，值域 `[0, 255]`。`Float32Array` 即使只含 0..255 整数也以 `IncompatibleInput` 拒绝。灰度含义须与 duckomo 灰度一致；本契约不定义或实现从 dBZ 等物理值的转换。
- `zeroThreshold` 省略时保持输入值；给出整数阈值 `t` 时，仅将 `< t` 的像素设为 0，再除以 255 并 resize。`0` 不改变任何像素。
- 四帧有效时间必须严格递增，但相邻时间无需相等间隔，也不与 manifest cadence 比较。运行时用 `(last - first)/(N - 1)` 的平均秒间隔计算输出时效；第 `k` 个输出（从 1 开始）分别取 `round_half_up(k × average_interval_seconds)`，避免逐帧累积舍入误差。`valid_times` 由最后观测时间加这些整秒 lead 得到，`lead_minutes` 为 lead 秒数除以 60，可为小数。
- 输出数量取 `tensor_output.shape[1]`；manifest `forecast.lead_minutes` 是现有固定模型的训练/reference 元数据，不决定运行时时效。Manifest observation cadence、grid id 与 grid shape 同样保留为参考元数据，不作为输入接受条件。
- 输入高度和宽度可为任意正整数，但四帧必须相同尺寸。运行时缩放到模型输入 tensor 的空间尺寸。输入 georeference 可缺省，输出继续 `georeferenced = false` 且不提供 CRS/affine。
- 推理输出的实际维度必须与 `tensor_output.shape` 逐维相同；相同元素数量的不同形状也以 `InvalidOutput` 拒绝。
- `validation` 可运行未声明或配置不同的 Web 目标。只有平台、引擎、版本、provider 和实际线程数都与某条 manifest 声明匹配时，才返回 `verified = true` 和 `target_id`；否则继续运行并返回 `verified = false`、`target_id = null`。
- `bun run build` 同时生成 JavaScript、Wasm 与 `dist/index.d.ts` 及其依赖声明，可直接供 TypeScript 消费端使用。
