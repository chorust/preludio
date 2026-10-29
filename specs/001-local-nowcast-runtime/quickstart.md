# Quickstart: 验证端侧短临运行时

**Plan**: [plan.md](plan.md) | **Contracts**: [contracts/](contracts/) | **Data model**: [data-model.md](data-model.md)

本文列出验证模型加载、预处理、本地推理、输入校验、输出时间和设备行为的场景。命令形态以 [rust-api.md](contracts/rust-api.md) 的 CLI 与 [web-api.md](contracts/web-api.md) 为准；记录格式见 [validation-record.schema.json](contracts/validation-record.schema.json)。这些固定 reference 场景不代表 Duckomo 转换或 US5 来源集成已完成。

## 前置条件

- Rust 1.92（`rustup target add wasm32-unknown-unknown`）、`wasm-bindgen-cli` 0.2.100、Bun ≥ 1.4、Chrome ≥ 150。
- 模型包 `PKG=artifacts/ghost-unet3d-step-22000`，由 `tools/export_ghost_unet3d.py` 生成 `preludio.model-package/1` manifest（[schema](contracts/model-package.schema.json)），并包含 `reference/input.npy`、`reference/reference.npy`。
- 本仓库当前没有 `reference/frames/` 下的原始 PNG；场景 1、3 的原始帧路径和按位预处理对比需先从来源处取得这些文件。`--tensor` 路径只验证固定 tensor 推理，不能替代原始帧验收。
- 此固定样本的原图为 732×732、时间间隔为 6 分钟且未使用 threshold；这些是 reference 事实，不是运行时输入限制，也不证明任意尺寸或 threshold 行为。

```sh
export PRELUDIO_MODEL_PACKAGE="$PWD/artifacts/ghost-unet3d-step-22000"
TIMES=2023-11-30T09:24:00Z,2023-11-30T09:30:00Z,2023-11-30T09:36:00Z,2023-11-30T09:42:00Z
```

## 1. 固定样本重放（US1，SC-001）

```sh
cargo test                                   # 校验、时间、预处理夹具；有模型包时含全尺寸一致性
cargo run --release --features cli -- replay --package "$PRELUDIO_MODEL_PACKAGE" \
  --frames "$PRELUDIO_MODEL_PACKAGE"/reference/frames/AZ9002_20231130_{092400,093000,093600,094200}Z.png \
  --times "$TIMES" --reference "$PRELUDIO_MODEL_PACKAGE/reference/reference.npy" \
  --record "$PRELUDIO_MODEL_PACKAGE/reference"
```

预期：退出码 0；验证记录 `input.preprocessed_matches_reference = true`，`result.failures = 0`，`max_abs_error` ≈ 1e-6 量级；`execution.target_id = "macos-arm64-ort-cpu"`、`provider = "cpu"`、`verified = true`。
原始 `--frames` 与显式 `--reference` 一起使用时，若包中存在 `reference/input.npy`，CLI 会将预处理 tensor 与它按位比较，并检查 manifest/reference 形状；不匹配时以退出码 5 验证失败，与是否启用 `--record` 无关。不使用 `--record` 时 `--runs` 默认 1 次，录制时默认 6 次（1 冷 + 5 热）；显式 `--runs` 覆盖默认值，录制时少于 6 次会在打开模型包前以退出码 2 拒绝。

1b. 无原始帧时用 `--tensor "$PRELUDIO_MODEL_PACKAGE/reference/input.npy"`，记录 `input.entry = "tensor"`。

共享核心可用 `cargo test --no-default-features` 独立验证；原生重放测试仅在非 wasm32 目标且启用 `native` feature 时编译运行。验证原生路径时，设置上述 `PRELUDIO_MODEL_PACKAGE` 后运行 `cargo test --features cli --test replay`，避免因缺少模型包而跳过。

## 2. 断网运行（US1 场景 2，SC-003）

关闭网络（系统设置或 `networksetup -setairportpower en0 off`），用 `--runs 10` 重复场景 1。预期 10 次成功；推理期间 `nettop -p <pid>` 无连接。

也可只禁止预测进程联网，无需修改系统网络设置。已验证的固定 tensor 场景：

```sh
sandbox-exec -p '(version 1)(allow default)(deny network*)' target/release/preludio replay \
  --package "$PRELUDIO_MODEL_PACKAGE" --tensor "$PRELUDIO_MODEL_PACKAGE/reference/input.npy" \
  --times "$TIMES" --reference "$PRELUDIO_MODEL_PACKAGE/reference/reference.npy" \
  --runs 10 --record /tmp/preludio-offline
```

本次 10 次预测全部成功，4,014,080 个输出元素零失败；记录保存在模型包 `reference/{validation,measurement}-macos-arm64-offline.json`。这是原生固定 tensor 断网证据；原始 PNG 和浏览器断网验收仍需各自的数据与环境。

## 3. 不兼容输入（US2，SC-002）

先运行验证测试；它覆盖网格尺寸、值长度、非有限数值及质量掩码等无法由 PNG CLI 参数表达的情况：

```sh
cargo test --test validate
cargo build --release --features cli
```

准备好 `reference/frames/` 中的四个 PNG 后，以下 CLI 命令会以退出码 2 在 stderr 返回结构化违规 JSON：

```sh
PKG=artifacts/ghost-unet3d-step-22000
FRAME_DIR="$PKG/reference/frames"

# FrameCount：只传 3 帧和对应的 3 个时间
target/release/preludio replay --package "$PKG" \
  --frames "$FRAME_DIR/AZ9002_20231130_092400Z.png" "$FRAME_DIR/AZ9002_20231130_093000Z.png" "$FRAME_DIR/AZ9002_20231130_093600Z.png" \
  --times 2023-11-30T09:24:00Z,2023-11-30T09:30:00Z,2023-11-30T09:36:00Z

# NotIncreasing：交换第 2、3 帧时间
target/release/preludio replay --package "$PKG" \
  --frames "$FRAME_DIR/AZ9002_20231130_092400Z.png" "$FRAME_DIR/AZ9002_20231130_093000Z.png" "$FRAME_DIR/AZ9002_20231130_093600Z.png" "$FRAME_DIR/AZ9002_20231130_094200Z.png" \
  --times 2023-11-30T09:24:00Z,2023-11-30T09:36:00Z,2023-11-30T09:30:00Z,2023-11-30T09:42:00Z


# Strictly increasing unequal intervals are accepted; this span has 9-minute average interval
target/release/preludio replay --package "$PKG" \
  --frames "$FRAME_DIR/AZ9002_20231130_092400Z.png" "$FRAME_DIR/AZ9002_20231130_093000Z.png" "$FRAME_DIR/AZ9002_20231130_093600Z.png" "$FRAME_DIR/AZ9002_20231130_094200Z.png" \
  --times 2023-11-30T09:24:00Z,2023-11-30T09:29:00Z,2023-11-30T09:37:00Z,2023-11-30T09:51:00Z

# EncodingMismatch: dBZ float values do not satisfy the raw-u8 gray contract
target/release/preludio replay --package "$PKG" \
  --frames "$FRAME_DIR/AZ9002_20231130_092400Z.png" "$FRAME_DIR/AZ9002_20231130_093000Z.png" "$FRAME_DIR/AZ9002_20231130_093600Z.png" "$FRAME_DIR/AZ9002_20231130_094200Z.png" \
  --times "$TIMES" --encoding dbz-f32

# Optional threshold: values < 4 become zero before /255 and resize; 0 is a no-op
target/release/preludio replay --package "$PKG" \
  --frames "$FRAME_DIR/AZ9002_20231130_092400Z.png" "$FRAME_DIR/AZ9002_20231130_093000Z.png" "$FRAME_DIR/AZ9002_20231130_093600Z.png" "$FRAME_DIR/AZ9002_20231130_094200Z.png" \
  --times "$TIMES" --zero-threshold 4
```

预期：三帧输入以 `FrameCount { expected: 4, actual: 3 }` 失败，重复/乱序以 `NotIncreasing` 失败，`dbz-f32` 以 `EncodingMismatch` 失败；严格递增的不等间隔时间通过。任意正且四帧同形尺寸通过尺寸规则；非正或尺寸不一致由验证样本覆盖。manifest grid id/shape 与 cadence 不作为拒绝条件。


## 4. 模型完整性（US1 场景 3）

复制模型包，改动 `model.onnx` 任意一个字节后重放。预期退出码 3，`HashMismatch { expected, actual }`。把 manifest 的 `schema` 改为 `preludio.model-package/2`，预期 `UnsupportedSchema`。

## 5. 有效时间与来源（US3，SC-004、SC-005）

检查场景 1 的 stdout：固定样本的 `valid_times[0] = 2023-11-30T09:48:00Z`、`valid_times[19] = 2023-11-30T11:42:00Z`，`generated_at` 为运行时刻，`encoding = training-grayscale-normalized`，`grid.georeferenced = false`，`provenance.input_frames` 列出四帧 id。一般情况下输出数量取 `tensor_output.shape[1]`；以四帧首末时间差除以 3 得到平均秒间隔，对每个输出索引 k（从 1 开始）分别按最近秒、半秒向上取整。`lead_minutes` 是这些秒数除以 60 的浮点值。

## 6. 设备测量与支持声明（US4，SC-006）

```sh
cargo build --release --features cli
target/release/preludio replay --package artifacts/ghost-unet3d-step-22000 --policy validation --runs 6 --record artifacts/ghost-unet3d-step-22000/reference --tensor artifacts/ghost-unet3d-step-22000/reference/input.npy --times 2023-11-30T09:24:00Z,2023-11-30T09:30:00Z,2023-11-30T09:36:00Z,2023-11-30T09:42:00Z --reference artifacts/ghost-unet3d-step-22000/reference/reference.npy
```

预期：`reference/measurement-macos-arm64.json` 中 `max(warm_s) ≤ 8`、`peak_memory_mib ≤ 1740.8`、`pass = true`。当前记录的 4 线程热推理约 1.40 秒，峰值 RSS 为 1700.9 MiB。

未声明组合：在 Linux 上以默认 `supported` 策略重放，预期退出码 3、`TargetNotSupported`；加 `--policy validation` 可运行并输出 `verified = false`。

## 7. 浏览器（US1/US4 Web，SC-001、SC-006）

从仓库根目录执行以下命令：

```sh
cd web
bun install
bun run build
bun test
bun run smoke
```

在 smoke server 输出的 URL 打开 Chrome。服务器从仓库内 `artifacts/ghost-unet3d-step-22000/` 提供模型包，并默认发送 COOP/COEP 头。Smoke 使用 `validation` 策略，因此可以执行尚未登记的 Chrome 组合并生成未验证记录；模型包的默认 `supported` 策略目前会拒绝 Chrome。页面执行 1 次冷推理和 5 次热推理，显示并下载验证记录与测量记录。

标准验收要求 `crossOriginIsolated = true`、`threads = 4`、`failures = 0`、热推理 ≤ 8 秒、`measureUserAgentSpecificMemory` 峰值 ≤ 1740.8 MiB，且 rAF 最大间隔 < 100 ms。最近报告的无头 Chromium 运行通过数值检查并测得 1725.06 MiB，但热推理耗时 21.15–28.60 秒、最大 rAF 间隔 1064.54 毫秒，未通过性能与响应验收。留存的早期 Chrome 记录是历史证据，其 `pass = true` 不代表后续验收通过；需保留其验证/测量文件并另行重跑。

从 `web/` 执行 `bun run smoke -- --no-isolation` 后，validation 模式的 4 线程配置预期抛出 `ModelUnavailable` / `IsolationRequired`。Chrome 在默认 `supported` 策略下预期 `TargetNotSupported`，直至有完全匹配且通过验收的声明。Safari 26.6.2 也预期 `TargetNotSupported`；WebContent RSS 曾测得 4,810 MiB，Safari 暂不支持，等待可比内存测量或引擎降低内存用量后再评估。

## 8. Radiust 接入（US5）

US5 尚未完成：当前没有证据表明一个具体 Radiust 来源能提供 duckomo-compatible raw u8 灰度，也没有完整端到端时间轴联调记录。M1 后在 Radiust 侧使用符合契约的数据运行序列并检查预测时间、观测/预测区分和失败状态；不要求物理量转换或地图位置对齐。
