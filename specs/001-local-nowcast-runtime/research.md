# Phase 0 Research: 端侧雷达短临预测运行时

**Date**: 2026-10-24 | **Plan**: [plan.md](plan.md) | **Spec**: [spec.md](spec.md)

所有测量均在 Apple M4 Pro（14 核，macOS darwin 25.6.0 arm64）上完成，使用 step 22000 的 `model.onnx` 与 AZ9002 固定输入，参考为 `reference.npy`（PyTorch）。容差 `|a-ref| ≤ 1e-5 + 1e-4·|ref|`，输出共 4,014,080 元素。每个测量行为 1 次冷推理 + 5 次热推理。探针源码见 [spikes/](spikes/README.md)。

## R1. 原生推理引擎

**Decision**: `ort` 2.0.0-rc.15（封装 ONNX Runtime 1.31.0），CPU 执行，固定 `=2.0.0-rc.15`；会话显式设置 `AutoDevicePolicy::PreferCPU`、开启 CPU arena、关闭 memory pattern、使用默认线程数。默认 `download-binaries` 静态链接，不分发额外 dylib。

**Rationale**:

| 后端 | 线程 | 会话 s | 冷推理 s | 热推理均值（最小–最大）s | 峰值 RSS MiB | 最大 / 平均绝对误差 | 超差元素 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ORT CPU（PreferCPU） | 1 | 0.025 | 4.096 | 4.006（3.959–4.047） | 1,687.0 | 9.31e-7 / 3.89e-9 | 0 |
| ORT CPU（PreferCPU） | 4 | 0.024 | 1.451 | 1.415（1.407–1.425） | 1,687.5 | 9.54e-7 / 3.88e-9 | 0 |
| ORT CPU（PreferCPU） | 默认 | 0.025 | 1.029 | 0.967（0.951–0.995） | 1,688.4 | 9.54e-7 / 3.88e-9 | 0 |
| ORT CoreML | 默认 | 0.641 | 1.362 | 1.191（1.157–1.303） | 5,550.8 | 1.16e-3 / 4.75e-6 | 310,403 |
| tract-onnx 0.23.9 | 默认 | 0.110 | 1.934 | 1.790（1.780–1.806） | 2,566.3 | 1.44e-3 / 1.02e-5 | 366,483 |

- 未设置设备策略时，macOS 上的 `ort` 会话会把 103/629 个节点分给 CoreML（profile：31 个 CoreML 分区 + 526 个 CPU 节点），导致与上表 CoreML 行相同的超差。设置 `PreferCPU` 后 628 个 CPU 事件、0 个 CoreML 事件，误差回到 1e-6 量级。Python `onnxruntime` 1.30.0 / 1.31.0 显式 `CPUExecutionProvider` 同样通过（热推理 0.955 / 0.959 s）。结论：问题来自执行提供方选择，不是 ORT 或绑定的数值缺陷。
- 开启 memory pattern 时默认线程峰值为 2.65 GiB；关闭后约 1.65 GiB，满足预算。
- 静态链接产物：探针 release 二进制 26.39 MiB，`otool -L` 无 ORT dylib。
- 所有探针在 `#![forbid(unsafe_code)]` 下编译通过（unsafe 仅在依赖内部）。

**Alternatives considered**:
- CoreML EP：只覆盖部分节点，CPU 回退，超差且 5.4 GiB；排除出首批支持组合，保留为后续加速实验。
- tract-onnx：纯 Rust、能运行 Conv3D，但超差（1.44e-3）且 2.5 GiB；排除。
- candle-onnx 0.11：求值器不支持 3D Conv/MaxPool 与 5 维 Resize；排除。
- burn-onnx 0.22：算子表缺 MaxPool3D；排除。

## R2. MSRV 与 edition

**Decision**: `rust-version` 从 1.70 升至 1.92，保留 edition 2021。

**Rationale**: `ort` 2.0.0-rc.15 要求 1.92；Radiust 也固定 1.92（edition 2024）。crate 只发布过占位版本 0.0.1，没有下游兼容负担。edition 无需变更。

**Alternatives considered**: 保持 1.70 并换用更旧的 ORT 绑定——会失去 `AutoDevicePolicy` 与 ORT 1.31，且无收益。

## R3. 预处理复现与共享核心

**Decision**: 用纯 Rust 实现 PyTorch CPU 的抗锯齿双线性缩放（`align_corners=false`）与 float16 舍入，作为唯一实现；同一 crate 编译为原生库和 `wasm32-unknown-unknown`（wasm-bindgen）供浏览器使用。契约校验、预处理、有效时间计算、哈希与来源记录全部放在共享核心中。

**Rationale**: 在 3 类合成输入（稀疏回波电平 {0,4,7,11,15,18}、均匀随机 uint8、全零；各 4×732×732）上：

| 实现 | fp16 舍入前差异元素 | 舍入后完全一致 | 单窗口耗时（中位） |
| --- | --- | --- | --- |
| Rust 原生（release） | 0 | 802,816 / 802,816（3 类全部） | 3.4–6.4 ms |
| Rust → WASM（Bun） | 0 | 802,816 / 802,816（3 类全部），与原生逐位相同 | 9.3–11.4 ms |
| `image` 0.25.5 Triangle | 最多 416,151 | 均匀随机 802,752 / 802,816 | — |
| `fast_image_resize` 5.1.4 Bilinear | 最多 778,997 | 均匀随机 798,799 / 802,816 | — |

参考使用 PyTorch 2.5.1 CPU 与 torchvision 0.20.1；二者在 3 类输入上逐位一致。WASM 文件 27,997 字节。

算法要点（写入契约）：`scale = in/out`；输出像元中心 `center = scale·(i+0.5)`；三角核支撑按缩小倍数放大；源区间裁剪到图像边界后重新归一化权重；先水平后垂直；权重、中间值与累加均为 float32，首项乘积后的累加使用 fused multiply-add；最后按 IEEE 754 binary16 就近舍入再转回 float32。

**Alternatives considered**: 通用缩放库（不逐位一致，排除）；TypeScript 单独实现（两套实现需双重验证，排除）；在浏览器中把预处理也交给 ONNX Resize（训练时由 torchvision 完成，需单独证明等价，排除）。

**Limit**: 原始训练 PNG 不在本机，PNG 解码与灰度提取未验证；不同 PyTorch 版本的参考需重新比对（`tools/` 增加固定夹具生成脚本，R7）。

## R4. 浏览器执行

**历史实验决策**：当时选择 `onnxruntime-web` 1.30.0 WASM 执行提供方、4 线程和 `ort.env.wasm.proxy = true`（Worker 推理），要求页面跨源隔离（`Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp`）。这些结果使 Chrome 成为候选浏览器，但后续正式四线程验收未达性能/响应预算；当前模型包不登记 Chrome 支持目标。Safari 在可比内存测量通过前也不支持。

**早期候选实验记录**（Chrome 150.0.7871.24、Safari 26.6.2；这些运行均通过数值一致性，最大 / 平均误差 9.41e-7 / 4.95e-9，0 超差，但不代表后续设备验收通过）：

| 浏览器 / 模式 | 隔离；请求→实际线程 | 会话 / 冷 / 热均值 s | 首个结果 s | 内存 |
| --- | --- | --- | --- | --- |
| Chrome 隔离 | 是；1→1 | 0.353 / 9.163 / 8.416 | 9.561 | UA 1,710.7 MiB |
| Chrome 隔离 | 是；4→4 | 0.100 / 2.271 / 2.139 | 2.412 | UA 1,711.1 MiB |
| Chrome 隔离 + proxy | 是；4→4 | 0.108 / 2.245 / 2.136 | 2.391 | JS heap 62 MiB |
| Chrome 未隔离 | 否；4→**1** | 0.295 / 8.009 / 7.902 | 8.362 | — |
| Safari 隔离 | 是；4→4 | 0.249 / 2.350 / 2.134 | 2.655 | WebContent RSS 4,810 MiB（两轮后，无内存 API） |

- 单线程热推理 7.9–8.4 s，处于或超出 8 s 预算，因此跨源隔离是硬性条件。
- 不用 proxy 时主线程最大 rAF 间隔 12,975 ms；proxy 后 16.7 ms。proxy 会转移输入 buffer，每次推理须新建输入 Tensor，否则出现 `DataCloneError`。
- 浏览器需下载 `ort-wasm-simd-threaded.jsep.wasm` 28,312,028 字节、ORT JS 368,025 字节、模型 31,072,725 字节。
- WebGPU：会话创建成功，首次运行失败于 `MaxPool3d`（“Pooling with kernelShape.length > 2 is not supported for NHWC format”）。
- 后续无头 Chromium 四线程验收的数值检查通过、内存为 1725.06 MiB，但热推理耗时 21.15–28.60 秒、最大 rAF 间隔 1064.54 毫秒，超过性能与响应预算。因此这些早期记录只保留为历史证据，Chrome 不列入当前支持目标。

**Alternatives considered**: WebGPU（不可用）；单线程无隔离（超预算）；主线程执行（阻塞交互，违反 FR-012）。

## R5. 模型包与哈希

**Decision**: 模型包为一个目录：`manifest.json`（schema `preludio.model-package/1`）+ `model.onnx` + `reference/`（`input.npy`、`reference.npy`、`validation.json`）。manifest 由 `tools/export_ghost_unet3d.py` 生成，以现有实验 manifest 为基础补齐契约版本、观测编码与缺测策略、网格标识、时间规则、支持组合及验证记录。模型身份 = 可读 id + `model.onnx` 的 SHA-256；加载时在共享核心内（`sha2` crate，原生与 WASM 同一实现）校验。

**Rationale**: 现有 manifest 已有哈希、tensor 与预处理字段，`physical_dbz_mapping` / `georeferencing` 为 `null`；增量扩展即可满足 FR-001/FR-014。浏览器端用同一 Rust 哈希实现避免 SubtleCrypto 与原生实现分叉。

**Alternatives considered**: 把 manifest 嵌入 ONNX metadata（需修改图且浏览器端解析成本高）；单文件打包格式（首版无多文件下载需求）。

## R6. 消费者接入（Radiust）

**Decision**: Preludio 只暴露自身契约类型。当前运行时契约规定 Ghost 输入为 duckomo-compatible 单通道 raw u8 灰度；Radiust 的通用 `RadarField.values: Vec<f32>` 不能直接满足该输入契约，适配器只能使用另行提供且语义符合的字节数据。该契约不要求 dBZ 映射或输入地理参照。US5 端到端实现仍需具体来源与联调；现有 F32 adapter/test 不构成该证据。

**Rationale**（`/Users/blizhan/data/code/github/radiust`，0.1.8-alpha）：
- Radiust 是 Rust workspace（edition 2024，MSRV 1.92，Tokio 1.48）；没有桌面 UI、Web 前端或 WASM 构建，且其核心因 netcdf/rusqlite/tokio 无法编译为 wasm32。
- 质量位掩码区分 `MISSING (0x1)`、`OUTSIDE_COVERAGE (0x2)`、`BELOW_DETECTION (0x20)`（当时记录于 `crates/radiust-core/src/raster.rs:491-506`）。按当前契约，Missing/OutsideCoverage 拒绝；仅有 BELOW_DETECTION 映射为 Valid + u8 零回波。
- 该次 2026-10-24、Radiust 0.1.8-alpha 清单盘点中，25 个已实现来源没有中国大陆 CINRAD/AZ9002 或 732×732、电平 {0,4,7,11,15,18} 灰度产品。`gray-dbz-v1`（`dBZ = min(gray,224)·5/16`）与训练 PNG 是否一致当时未经证实。此历史盘点不证明现有来源可提供 duckomo-compatible raw u8，也不要求恢复物理映射。
- 该次盘点时，`CatalogProduct.forecast` 字段全部为 `false`，时间轴没有预测帧概念。

**Alternatives considered**: Preludio 依赖 `radiust-core`（违反 D-05，且把 Tokio/netcdf 带入运行时，排除）；CLI 子进程交换文件（延迟和部署复杂度更高，排除）。

## R7. 测试策略

**Decision**:
- Rust：`cargo test` 覆盖校验规则、有效时间、fp16 舍入与小尺寸预处理夹具（已提交，`tests/fixtures/`，由 `tools/make_preprocess_fixtures.py` 用 PyTorch 生成）；全尺寸重放与参数一致性测试读取 `PRELUDIO_MODEL_PACKAGE` 指向的模型包，未设置时跳过并打印原因（模型包 ~30 MiB 不入库）。
- Web：`bun test` 覆盖 TypeScript 封装与 WASM 核心的一致性；浏览器重放用 `web/smoke/`（带 COOP/COEP 的本地服务器）+ 真实浏览器手动或自动化执行，输出与 Rust 相同格式的验证记录。
- 设备测量记录为 JSON，与参考比较记录一起写入模型包 `reference/`，作为支持声明的依据。

**Rationale**: 小夹具保证 CI 中逐位预处理回归；大产物通过显式路径加载，避免仓库膨胀且保持测试可隔离。

## 已解决的 Technical Context 未知项

| 未知项 | 结论 |
| --- | --- |
| 原生引擎 / 绑定 | R1：`ort` 2.0.0-rc.15 CPU + PreferCPU |
| MSRV | R2：1.92 |
| 预处理跨语言一致性 | R3：纯 Rust 共享核心，逐位一致 |
| 浏览器后端与线程 | R4：ORT Web WASM 4 线程 + 跨源隔离 + Worker |
| 模型包格式 | R5：目录 + `preludio.model-package/1` manifest |
| Radiust 接缝 | R6：适配器属于 Radiust；raw-u8 来源兼容与 US5 端到端仍未证实 |
| 测试方式 | R7 |

## 仍未关闭的风险

- Chrome 后续四线程浏览器验收未满足热推理与 rAF 预算；旧固定 reference 及内存记录不能作为当前支持声明依据，修复后需重跑完整验收。
- `ort` 仍是预发布版本（rc.15），升级需重新验证。
- 尚无具体 Radiust 来源兼容 duckomo-compatible raw u8 的证据，也无仓库内 Duckomo 灰度转换实现证据；固定 reference 不验证任意尺寸或非默认阈值。物理映射与 georeferencing 不属于当前输入契约。
