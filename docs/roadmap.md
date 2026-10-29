# Preludio Roadmap

更新日期：2026-10-27。按里程碑和退出条件推进，不预设交付日期。产品要求见 [Spec](spec.md)，职责与技术决策见 [Architecture](architecture.md)。

## 1. 推进原则

先用一个模型、一组观测和本地后端建立可复现链路，再接入 Radiust。macOS 原生已通过固定 reference 与设备预算验证；Chrome WASM 的固定 reference 数值与旧内存记录仅是历史证据，后续浏览器验收未通过性能/响应预算，因此当前模型包不登记 Chrome。当前工作还包括确认是否有 Radiust 来源能提供符合 duckomo 灰度约定的字节数据。

原讨论中的 Rust + TypeScript、Core ML、Tauri 和跨平台目录树作为候选保留。原生与 WASM 两条路径共用同一组参考样本；稳定以后再决定是否提取共享核心。

## 2. 里程碑总览

| 里程碑 | 状态 | 依赖 | 退出条件 |
| --- | --- | --- | --- |
| M0：模型部署 spike | 已完成，限固定窗口数值验证 | 临时 checkpoint 与训练配置 | 导出产物、参考输入输出、误差报告和复现说明齐全 |
| M1：运行时输入契约与来源兼容 | 未完成：T047 实现验证已完成，具体来源兼容性未证实 | M0、契约实现、Radiust 候选来源 | 四帧 raw-u8 契约与预测时序有实现验证；确定具体来源可提供符合 duckomo 灰度约定的 raw-u8 字节，并记录兼容性证据 |
| M2：原生运行时与 macOS 验证 | 已完成；测量记录通过 | M0 | 一个无需训练环境的本地调用路径，通过参考样本、输入校验与目标设备预算 |
| M3：Radiust 首次完整接入 | 未完成；已有 adapter/test 的范围不证明 duckomo raw-u8 来源或端到端场景 | M1，以及 M2 或 M4 中先通过的路径；另一条路径随后用同一来源验证 | 一种真实来源完成本地预测与时间轴展示，观测/预测区分和失败状态正确；不要求地理参照或物理量映射 |
| M4：Web WASM 运行时与浏览器验证 | 更新契约的真实模型 threshold-equivalence 检查已通过；Chrome 后续四线程浏览器验收热推理为 21.15–28.60 s、最大 rAF 间隔 1064.54 ms，未达预算且未登记支持；10 次断网验收仍未完成，Safari 暂不支持 | 张量级只需 M0；来源接入需要 M1 | 选定浏览器通过契约、数值、内存、性能与断网运行检查 |
| M5：移动端与手表 | 延后 | 至少一个消费端闭环及设备需求 | 单独定义设备、模型和资源预算后再排期 |

M1 的运行时契约实现验证已完成；M1 仍需具体来源兼容性证据，不要求证明 dBZ 映射、站点投影或地理参照。完整来源到时间轴的端到端接入由 M3 验收。M2 的原生固定 tensor 路径已完成；M4 的更新契约 threshold-equivalence 浏览器检查已通过，但 Chrome 四线程性能/响应验收未通过，当前无 Chrome 支持声明，完整断网验收也未完成。

首批预算（2026-10-24 用户确认）：目标设备上单次热推理 ≤ 8 秒；内存以 WASM 实测约 1.7 GiB 进程常驻为可接受量级。下载体积和最低设备要求仍待确定。

## 3. M0：已经取得的证据

- [x] 从 Onix `step_22000.ckpt` 和 Hydra 配置重建 GhostUNet3D，严格加载权重。
- [x] 导出固定形状、opset 18、float32 ONNX，记录 checkpoint、配置、源码与产物哈希。
- [x] 用 AZ9002 的四帧真实灰度观测构造输入，保存 PyTorch 与 ONNX 结果。
- [x] 定位 GroupNorm 默认导出误差，用逐轴归约导出保持 CPU 精度，并保留回归测试。
- [x] 留下导出工具、四项 Python 测试和[复现文档](ghost-unet3d.md)。

| 2026-10-24 的测量 | 结果 |
| --- | --- |
| 环境 | Linux ARM64，4 CPU 线程 |
| ONNX 体积 | 31,072,725 字节，29.63 MiB |
| PyTorch / ONNX 最大绝对误差 | 9.61125e-7 |
| 平均绝对误差 | 3.85843e-9 |
| ONNX session 初始化 | 单次 0.029 秒 |
| ONNX 首次推理 | 单次 4.097 秒 |

M0 的完成仅代表该窗口、该模型和该 Linux CPU 环境的部署验证通过。后续 macOS 原生与 Chrome WASM 的目标设备证据分别记入 M2 和 M4；这些测量不证明 Duckomo 灰度转换实现、任意输入尺寸/阈值行为、具体 Radiust 来源兼容性或气象效果。

## 4. M1：落实 raw-u8 输入契约并确认来源

需要交付的内容：

1. 当前 Ghost 输入定义为 duckomo-compatible 单通道 raw u8 灰度 `[0,255]`；不在 Preludio 中推导 dBZ 或其他物理转换。
2. 接受恰好四帧、任意相同的正 `[H,W]`，并 resize 到模型输入 tensor shape。manifest 中的 observation grid id/shape 与 cadence 是固定训练/reference 元数据，不构成接受限制；有效时间只需严格递增。
3. 记录 no-echo 为 `Valid` + u8 零，Missing 与 OutsideCoverage 保持可区分并按当前策略拒绝；可选 sequence-level `zero_threshold` 在 `/255` 和 resize 前只将严格小于阈值的值置零。
4. 由 `tensor_output.shape[1]` 决定输出数量；按四帧首末时间差得到平均间隔，对每个 lead 分别按最近秒、半秒向上取整；`lead_minutes` 可为小数。
5. 为任意正尺寸和 threshold 行为补齐实现验证，并记录具体 Radiust 来源是否能提供符合 Duckomo 灰度约定的 raw-u8 字节；来源到时间轴的端到端联调由 M3 完成。

**当前状态**：T047 的运行时输入/预测契约实现验证已完成。最终验证有 27 项 native tests、26 项 core-only tests、9 项 Python tests、9 项 Web tests（57 assertions），并通过 clippy 与 TypeScript 检查；native CLI smoke 使用四个 `73×61` raw-u8 帧、`zero_threshold = 16` 并 resize 至 `448×448`，与独立 PyTorch/ORT forecast reference 的 `4,014,080` 个输出值完全一致。输入时刻 `09:24`、`09:27`、`09:32`、`09:38 UTC` 对应首末预测有效时间 `09:42:40`、`11:11:20 UTC`。真实模型浏览器 threshold-equivalence 检查也对 `4,014,080` 个值全部匹配，并验证输入偏移 `0、1、3、7` 秒对应的动态时效（首 lead 2 秒，末 lead 47 秒）。完整四线程浏览器验收和 10 次断网验收尚未全部重跑。M1 尚未完成，因为具体 Radiust 来源兼容性无证据；固定 reference 仅证明原窗口、未设置 threshold 的路径，也没有证据证明仓库内存在 Duckomo 灰度转换实现。物理映射、投影和输入 georeference 不属于 M1。US5 端到端联调属于 M3，仍未完成。

**T047** 已完成 runtime contract 实现与验证：raw-u8 灰度输入、任意同形正尺寸、严格递增但无 cadence 限制、no-echo 零、可选阈值及按输出 tensor 和平均间隔计算 forecast times。验证包括 `73×61`、阈值 16 的 native CLI smoke、真实模型浏览器 threshold-equivalence 检查和尺寸/长度/质量边界测试；这些用例验证代表性输入与边界，不声称穷举所有尺寸。T047 不证明 Duckomo 转换、具体来源兼容或 US5 端到端完成。

## 5. M2：最小原生运行时与目标设备测量

在固定参考配置下，原生 Rust 运行时已实现，无需 Python / Onix 训练环境即可打开模型包、校验观测、预处理、推理并组装带来源的预测结果。首批目标 `macos-arm64-ort-cpu` 已在 Apple M4 Pro 上通过固定参考样本验证，4 线程热推理约 1.40 秒，峰值 RSS 为 1700.9 MiB，低于 1740.8 MiB 预算。

可复现记录位于 [`measurement-macos-arm64.json`](../artifacts/ghost-unet3d-step-22000/reference/measurement-macos-arm64.json) 与 [`validation-macos-arm64.json`](../artifacts/ghost-unet3d-step-22000/reference/validation-macos-arm64.json)。测量记录 `pass = true`。这项固定 tensor 证据不证明一般尺寸/阈值处理或具体 Radiust 来源兼容性；新的尺寸与阈值实现证据记录在 M1/T047。

Core ML 保留为后续加速实验。首次探针中 CoreML EP 仅覆盖部分节点、未通过容差且峰值 5.4 GiB，因此不进入首批。模型裁剪、量化或算子转换都要产生新的验证记录。

M2 退出条件已满足：macOS arm64 本地运行路径通过参考、输入校验和首批设备预算。

## 6. M3：接入 Radiust

Radiust 侧使用适配层，不将演进中的 `RadarField` 作为 Preludio 公共依赖。US5 需要一个能提供 duckomo-compatible raw u8 灰度的具体来源；现有通用 F32 adapter/test 不等同于这一来源兼容或完整接入。

```text
Radiust 来源提供 raw-u8 灰度 → 四帧同尺寸序列 → Preludio 本地预测
    → 含有效时间、来源和模型身份的非地理参照预测 → Radiust 时间轴
```

展示应包括观测与预测的边界、最后观测时间、每帧有效时间、模型身份及失败状态，并保留输出 `georeferenced = false`。UI 技术栈在该阶段选定；Tauri + TypeScript 是原讨论中的候选，尚未实施。

完成条件：一种真实来源提供契约要求的灰度字节，四帧提交并可本地重放，预测时间与来源可追溯，观测/预测标识和失败状态正确。当前不要求地图位置对齐或物理量解释；气象效果尚未验收，不能据此宣传两小时有效预报。

## 7. M4：Web WASM 运行时与浏览器验证

当前 GhostUNet3D 含 Conv3D，ONNX Runtime Web 的 WebGPU 路径不可用，实验性 Web 路径采用 WASM CPU。`@chorust/preludio` 已实现 TypeScript 入口，并可在跨源隔离页面中通过 WASM Worker 执行推理。

Chrome Desktop 的历史测量与验证记录保留为本地模型产物，未随仓库分发；此处保留结果摘要：固定 reference 零数值超差，峰值内存 1725.04 MiB，记录 `pass = true`。后续无头 Chromium 四线程验收虽然数值检查通过、内存 1725.06 MiB，但热推理耗时 21.15–28.60 秒、最大 rAF 间隔 1064.54 毫秒，超过性能与响应预算；因此当前 manifest 不登记 Chrome 支持目标。页面运行多线程仍必须启用 COOP/COEP。

Safari 26.6.2 暂不支持：WebContent RSS 在两轮后达到 4,810 MiB，且当前没有可用的页面内存测量 API。待 Safari 提供可用内存测量 API，或引擎降低内存用量后，再重新验证。

浏览器 threshold-equivalence 与旧固定 reference 数值/内存记录不代表设备性能验收通过。Chrome 目标需在热推理与 rAF 预算均达标后重新运行标准四线程与断网验收；Safari 的内存测量仍是后续工作。WebGPU 适配属于单独优化实验。

## 8. 效果评估与后续扩展

M1 起建立独立的天气过程评估集，覆盖无回波、不同强度、发展和消散等情况。先定义逐时效指标与对照基线，再评估发布候选模型；数值对齐报告与气象效果报告分别保存。模型仍在训练，后续 checkpoint 必须重新导出、比较和评估，不沿用 step 22000 的结论。

M5 先选择一个有明确用户需求的移动平台，验证模型转换、资源预算和效果；Swift / Core ML、Kotlin / LiteRT 等仍是候选。手表可以先探索手机计算后展示结果，是否需要独立小模型由设备实测和产品需求决定。降雨概率、开始时间与通知都需要额外定义和评估。

## 9. 下一次工作的具体起点

剩余工作是完整重跑四线程浏览器验收与 10 次断网验收、确认是否有 Radiust 来源能提供符合 duckomo 灰度约定的 raw u8 字节，并完成 US5 端到端时间轴联调。Chrome 当前未登记为支持目标：历史固定 reference 检查通过，但后续四线程浏览器运行超出热推理与 rAF 预算。真实模型 threshold-equivalence 检查及旧测量均不能证明完整浏览器性能、具体来源兼容性或完整 Radiust 接入。
