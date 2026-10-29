# Phase 0 spike sources

Throwaway probes behind [research.md](../research.md), kept here as reproducible evidence. They are not part of the crate build and are not production code. They read model artifacts from `artifacts/ghost-unet3d-step-22000/` by absolute path.

| Directory | Contents | Research item |
| --- | --- | --- |
| `preprocess/` | Pure Rust antialiased bilinear resize + fp16 rounding (`lib.rs`), PyTorch reference generator (`reference.py`), WASM bench (`bench_wasm.js`) | R3 |
| `native/` | `ort` 2.0.0-rc.15 sweep binary (`ort_sweep.rs`) with `AutoDevicePolicy::PreferCPU` and memory-pattern switch; NPY helpers (`lib.rs`) | R1, R2 |
| `browser/` | onnxruntime-web test page (`index.html`) and Bun static server with/without COOP/COEP (`server.js`) | R4 |

The native `Cargo.toml` uses `[[bin]]` paths from the original scratch layout (`src/bin/*.rs`, `src/lib.rs`); restore that layout before building.
