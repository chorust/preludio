<h1 align="center">Preludio</h1>

<p align="center">
  <a href="README.md">简体中文</a> · <a href="README.en.md">English</a>
</p>

**端侧雷达短临预测。** 在你的设备上，提前预报即将到来的降雨。

## 项目定位与目标

Preludio 是面向消费设备的气象雷达短临外推（Nowcasting）推理组件。它接收一段连续的雷达观测序列，在本地设备（原生桌面或浏览器）上完成预处理与模型推理，返回带有有效时间、空间网格和来源信息的预测场。

- 模型与观测数据准备就绪后，计算链路在设备本地完成，可断网运行，不会自动把推理请求转发到云端。
- 职责划分：
  - 上游训练侧（如 PyTorch 训练流程）：负责网络定义、训练、Checkpoint 与气象效果评估。
  - Preludio（本仓库）：负责版本化模型包校验、输入契约检查、模型专用预处理、本地推理（ONNX Runtime Native / Web WASM），以及预测元数据组装与数值验证。
  - Radiust / 消费端应用：负责雷达数据获取、科学解码、地理投影、通用重网格与时间轴 UI 展示（[Radiust](https://github.com/chorust/radiust) 是首个接入消费者）。

```mermaid
flowchart LR
    A["消费端 (Radiust)<br/>四帧观测序列 + 时间/质量"] --> B["Preludio 契约校验<br/>帧数 / 时间递增 / 尺寸 / 质量掩码"]
    B --> C["模型专用预处理<br/>阈值置零 → /255 → 抗锯齿缩放 → fp16 舍入"]
    C --> D["端侧引擎推理<br/>ORT Native CPU / ORT Web WASM"]
    D --> E["输出校验与组装<br/>Forecast 值 + 有效时间 + 来源追溯"]
```

## Features

### 1. 跨平台端侧运行时（共享核心）
- Rust 原生运行时 (`preludio` crate)：基于 ONNX Runtime Native (`ort`) 执行本地 CPU 推理，无 `unsafe` 代码，会话支持跨线程共享 (`Send + Sync`)。
- Web 浏览器运行时 (`@chorust/preludio`)：通过 `wasm-bindgen` 复用 Rust 核心的校验、预处理与结果组装逻辑，配合 `onnxruntime-web` 在 Web Worker 中执行多线程 WASM 推理，避免阻塞页面主线程。
- CLI 工具 (`preludio replay`)：支持从 8 位灰度 PNG 帧或 `.npy` 张量重放预测、比对 PyTorch/ORT 参考输出，并写入验证记录 (`ValidationRecord`) 与设备测量记录 (`MeasurementRecord`)。

### 2. 版本化模型包与支持目标策略
- 模型包契约 (`preludio.model-package/1`)：加载时校验 `manifest.json` Schema 并核对 `model.onnx` 的 SHA-256 哈希，拒绝损坏或不兼容的模型产物。
- 执行目标策略 (`TargetPolicy`)：
  - `SupportedOnly`（默认）：要求当前平台、引擎版本、Execution Provider、线程数及 Web 跨源隔离状态与已登记的目标一致，匹配后输出标记 `verified = true`。验证与测量记录由 manifest 引用，运行时不读取或校验这些记录文件。
  - `Validation`：允许在未登记的配置下运行基准测试与验证，输出标记 `verified = false`。

### 3. 输入校验与模型预处理
- 四帧 `raw-u8` 输入契约：要求恰好 4 帧单通道 `u8` 灰度序列（值域 `[0, 255]`，符合 duckomo 灰度约定），四帧具有相同的任意正尺寸 `[H, W]`，且时间戳严格递增（间隔可以不同；Web 端在预处理前拒绝浮点观测缓冲区）。
- 像素级质量掩码：区分有效像素（`Valid = 0`，无回波用 `0` 值表示）与异常掩码（`Missing = 1`、`OutsideCoverage = 2` 按当前策略拒绝）。
- 预处理顺序：可选序列级阈值置零（将严格 `< zero_threshold` 的像素置零，省略或设为 0 时不变）→ `/255` 归一化 → 抗锯齿双线性缩放 (`align_corners = false`) 至模型张量空间尺寸（当前 `448×448`）→ `float16` 舍入转回 `float32`。

### 4. 动态时效计算与来源追溯
- 输出张量维度校验：组装预测前逐维核对实际输出张量形状与 Manifest 声明（当前 `[1, 20, 448, 448]`）。
- 动态预测有效时间 (`valid_times`)：按四帧输入的平均间隔 `(last_time - first_time) / 3`，对每个输出索引 $k \in [1, N]$ 分别计算 $k \times \text{平均间隔}$ 并四舍五入至整秒（半秒向上取整），区分天气有效时间 (`valid_times`) 与计算完成时间 (`generated_at`)。
- 来源追溯 (`provenance`)：记录输入帧 ID 与时间戳、最后观测时间、模型与权重哈希，以及实际运行的引擎、Provider 与线程数。当前不执行物理量（如 dBZ）转换或地理坐标还原，输出固定为 `georeferenced = false`。

## 当前支持目标与浏览器历史测量

`GhostUNet3D`（`step_22000`，输入 `[1, 4, 448, 448]`，输出 `[1, 20, 448, 448]`）的固定参考比较覆盖 4,014,080 个输出值，容差为 `rtol=1e-4, atol=1e-5`：

| 目标 ID | 平台与后端 | 线程数 | 热推理耗时 | 峰值内存 | 内存预算 | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `macos-arm64-ort-cpu` | macOS arm64 (Apple M4 Pro, ORT CPU) | 4 | ~1.40 s | 1700.9 MiB RSS | ≤ 1740.8 MiB | 已验证通过 |

当前模型包不再声明 Chrome 为支持目标。[浏览器验证历史摘要](docs/roadmap.md#7-m4web-wasm-运行时与浏览器验证)保留了早期固定参考检查的结果：0 次数值超差及 1725.04 MiB 峰值内存；当时记录的 `pass = true` 只反映数值与内存预算。原始 JSON 记录属于本地模型产物，未随仓库分发。后续无头 Chromium 四线程验收的热推理耗时为 21.15–28.60 s，最大 rAF 间隔为 1064.54 ms，未达到热推理 ≤ 8 s 和 rAF < 100 ms 的预算。因此 Chrome 当前不在支持列表，完整浏览器验收仍未通过。

WebAssembly 多线程仍要求托管页启用跨源隔离：
 - `Cross-Origin-Opener-Policy: same-origin`
 - `Cross-Origin-Embedder-Policy: require-corp`

Web `supported` 策略要求平台、引擎、版本、provider 与实际线程数全部匹配 manifest；当前 Chrome 可用 `validation` 策略进行未验证的本地试跑。Safari 暂不支持。

## Usage

### 1. Rust 原生接入

进程启动时加载并校验一次 `ModelPackage`，创建 `Session` 后可通过 `Arc<Session>` 跨线程复用。`Session::predict` 是同步阻塞调用，在异步运行时（如 Tokio）中应放到阻塞工作线程（`spawn_blocking`）中执行。

```rust
use preludio::{
    Forecast, FrameValues, GridDescriptor, ModelPackage, ObservationFrame,
    ObservationSequence, PixelQuality, PreludioError, Session, SessionOptions,
    TargetPolicy,
};

fn open_session() -> Result<Session, PreludioError> {
    let package = ModelPackage::open("artifacts/ghost-unet3d-step-22000")?;
    Session::open(
        package,
        SessionOptions {
            policy: TargetPolicy::SupportedOnly,
            threads: None, // 使用目标声明的默认线程数 (4)
        },
    )
}

fn run_forecast(session: &Session, frames: Vec<ObservationFrame>, shape: [usize; 2]) -> Result<Forecast, PreludioError> {
    let sequence = ObservationSequence {
        grid: GridDescriptor {
            id: "radar-grid".to_owned(),
            shape,
            georeference: None,
        },
        encoding: "training-grayscale-u8".to_owned(),
        frames, // 恰好 4 帧，按 valid_time 从旧到新严格递增
        zero_threshold: None, // 可选：设置 Some(t) 将严格 < t 的像素在预处理前归零
    };
    session.predict(&sequence)
}
```

### 2. Web / TypeScript 接入 (`@chorust/preludio`)

`packageUrl` 需提供模型包目录下的 `manifest.json` 及其声明的模型文件。会话初始化支持传入 `AbortSignal`；若在初始化期间取消，会先释放已创建的引擎再返回失败。
当前模型 manifest 只声明 macOS 原生目标；浏览器示例使用 `validation` 策略供本地试跑，执行结果为未验证。默认 `supported` 策略会拒绝未登记的浏览器组合。

```ts
import {
  createSession,
  type Forecast,
  type NowcastSession,
  type ObservationSequence,
} from "@chorust/preludio";

async function runWebNowcast(observations: ObservationSequence): Promise<Forecast> {
  const session: NowcastSession = await createSession({
    packageUrl: "/models/ghost-unet3d-step-22000",
    ortWasmPaths: "/ort/",
    policy: "validation",
  });

  try {
    session.onStateChange((state) => {
      console.log("Nowcast state:", state); // "idle" | "running" | "succeeded" | "failed"
    });
    return await session.predict(observations);
  } finally {
    await session.dispose();
  }
}
```

### 3. CLI 重放、验证与基准测试

启用 `cli` feature 编译并运行 `preludio replay`，可重放四帧 PNG 或预处理张量、核对参考输出并记录目标设备测量结果：

不使用 `--record` 时，`--runs` 默认执行 1 次推理；使用 `--record` 时默认执行 6 次（1 次冷运行、5 次热运行）。显式传入次数会覆盖默认值；录制时少于 6 次会在打开模型包前以退出码 2 拒绝。原始 `--frames` 搭配显式 `--reference` 时，若包中存在 `reference/input.npy`，会校验形状并将预处理 tensor 与它按位比较；不匹配时以退出码 5 失败，与是否录制无关。

以下示例要求本地已有模型包及其 `reference/input.npy`、`reference/reference.npy`（导出步骤见 [GhostUNet3D 技术验证](docs/ghost-unet3d.md#export-and-verify)）。示例直接重放预处理张量。原始四帧 PNG 未随仓库分发；若使用 `--frames`，需自行取得对应时刻的原始灰度 PNG，并传入实际文件路径。

```sh
# 1. 运行全套单元与契约测试
cargo test

# 2. 使用模型包中的预处理张量执行重放并与参考输出比对
export PKG="artifacts/ghost-unet3d-step-22000"
export TIMES="2023-11-30T09:24:00Z,2023-11-30T09:30:00Z,2023-11-30T09:36:00Z,2023-11-30T09:42:00Z"

cargo run --release --features cli -- replay \
  --package "$PKG" \
  --tensor "$PKG/reference/input.npy" \
  --times "$TIMES" \
  --reference "$PKG/reference/reference.npy"

# 3. 构建并测试 Web 包与本地 Smoke 服务
cd web && bun install && bun run build && bun test
bun run smoke
```

## Roadmap

按里程碑与验证证据推进，详细退出条件见 [docs/roadmap.md](docs/roadmap.md)：

| 里程碑 | 阶段目标 | 当前状态 | 说明与下一步工作 |
| --- | --- | --- | --- |
| M0 | 模型部署 Spike | 已完成 | 从 PyTorch Checkpoint (`step_22000.ckpt`) 导出 ONNX opset 18，处理 GroupNorm 导出误差，在 CPU 上与 PyTorch 逐元素对齐（最大绝对误差 `9.61e-7`）。 |
| M1 | 运行时输入契约与来源兼容 | 进行中 | 四帧 `raw-u8` 契约、任意同形正尺寸缩放、`zero_threshold` 与动态时效已在 Rust/Web 实现，固定参考数值比对通过。待完成：确认具体 Radiust 数据源能否提供符合 duckomo 灰度约定的原始字节。 |
| M2 | 原生运行时与 macOS 验证 | 已完成 | 无需 Python 训练环境的 Rust 运行时已可用；`macos-arm64-ort-cpu` 在 Apple M4 Pro 上通过参考样本与设备预算验证（热推理约 1.40s，峰值 RSS 1700.9 MiB）。 |
| M3 | Radiust 首次端到端接入 | 待开展 | 依赖 M1 来源确认；完成真实数据源、本地推理与 Radiust 时间轴展示联调，区分观测与预测边界（US5）。 |
| M4 | Web WASM 运行时与浏览器验证 | 部分完成 | `@chorust/preludio` 构建（含 `.d.ts` 类型声明）、固定张量参考与真实模型浏览器阈值等价性检查已通过。Chrome 未登记为支持目标；无头 Chromium 四线程验收的内存为 1725.06 MiB，但热推理耗时 21.15–28.60s、最大 rAF 间隔 1064.54 ms，未满足性能与响应预算。标准四线程浏览器验收及连续 10 次断网验收仍未通过。 |
| M5 | 气象效果评估与多端扩展 | 规划中 | 建立独立天气过程评估集与逐时效基线指标；后续按需求评估 Core ML / WebGPU 算子适配及 iOS / Android / Watch 目标。 |

## 文档导航

| 文档 | 内容说明 |
| --- | --- |
| [Quickstart](specs/001-local-nowcast-runtime/quickstart.md) | 构建、样本重放、不兼容输入测试、断网验证与浏览器测量命令 |
| [Rust API 契约](specs/001-local-nowcast-runtime/contracts/rust-api.md) | `preludio` crate 的类型签名、错误结构与 CLI 参数规范 |
| [Web API 契约](specs/001-local-nowcast-runtime/contracts/web-api.md) | `@chorust/preludio` 的 TypeScript 接口、生命周期与跨源隔离约束 |
| [Radiust 接入指南](docs/radiust-integration.md) | 消费端 `RadarField` 字段映射、质量掩码转换及 Rust/Web 调用范式 |
| [项目规格](docs/spec.md) | 产品范围、数据契约、功能/非功能需求与关键验收场景 |
| [架构设计](docs/architecture.md) | 模块职责划分、模型包目录结构、预处理流水线与技术决策记录 |
| [路线图](docs/roadmap.md) | 各里程碑依赖关系、实验数据记录与推进退出条件 |
| [GhostUNet3D 技术验证](docs/ghost-unet3d.md) | 初始模型的复现命令、GroupNorm 精度对齐与平台后端边界探针 |

## 许可证

Apache-2.0
