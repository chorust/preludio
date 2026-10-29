# Radiust integration

The adapter belongs in Radiust; Preludio exposes its own `ObservationSequence` contract and does not depend on `radiust-core`.

This guide describes the structural mapping from Radiust `RadarField` / `FrameRef` and shows native and browser call patterns. The contract requires raw single-channel u8 gray values whose meaning matches duckomo gray; the repository has no evidence of a Duckomo conversion implementation or a concrete Radiust source that supplies those bytes. The generic F32 adapter and its tests do not establish US5 source compatibility or end-to-end completion.

## Field mapping

A Radiust `RadarField` contains `name`, `values: Vec<f32>`, `shape`, `quality: Vec<u16>`, optional `units`, `valid_time`, `grid`, and `provenance`. Its associated `FrameRef` provides `source`, `product`, optional `station`, `valid_time`, and `logical_id` (among other locator and revision fields).

| Radiust input | Preludio field | Mapping rule |
| --- | --- | --- |
| `FrameRef.logical_id` | `ObservationFrame.id` | Copy exactly. |
| `FrameRef.source`, `.product`, optional `.station` | `ObservationFrame.source` | Use a stable source identifier. This guide's convention is `radiust:<source>/<product>/<station>` when station is present and `radiust:<source>/<product>` otherwise; for example `radiust:rdcap/reflectivity/RCHL`. Keep the same convention across frames from the same source. |
| `RadarField.valid_time` | Rust `ObservationFrame.valid_time`; Web `ObservationFrame.validTime` | Require it to represent the same instant as `FrameRef.valid_time`. Rust parses RFC 3339 with a timezone to UTC Unix seconds via `preludio::time::parse_rfc3339_utc`; Web passes a normalized RFC 3339 UTC string such as `2023-11-30T09:42:00Z`. The Rust API stores whole seconds, so reject sub-second source times unless the adapter has an explicit rounding policy. |
| `RadarField.values` | `FrameValues::U8` / `Uint8Array` | The contract accepts raw single-channel u8 gray bytes in `[0,255]` only. `RadarField.values: Vec<f32>` cannot be cast or treated as compatible by shape or value range; the source adapter must already have contract-compatible bytes. No physical-value conversion is defined here. |
| `RadarField.quality` | `Some(Vec<PixelQuality>)` / `Uint8Array` | Convert each per-pixel `u16` bit mask using the rules below. Preludio quality codes are `Valid = 0`, `Missing = 1`, `OutsideCoverage = 2`. |
| `RadarField.shape` and `RadarField.grid.shape` | `GridDescriptor.shape: [h, w]` | Both Radiust shapes must agree, be positive, and describe the same dimensions for all four frames. Map to `[height, width]`; `RadarField::validate()` checks field/grid shape and value/quality lengths. Input may have any positive dimensions because Preludio resizes to the model tensor. |
| No direct `Grid.id` field | `GridDescriptor.id` | Supply a stable source label if available. The value is descriptive and is not compared with the model manifest. |
| Optional `RadarField.grid.crs` and `.affine` | Optional `GridDescriptor.georeference` | Georeference may be omitted. Do not invent CRS or affine values; the current forecast remains `georeferenced = false` and no spatial restoration is performed. |
| `RadarField.name`, `.units`, `.provenance`, and remaining `FrameRef` fields | No `ObservationFrame` field | Retain these in Radiust's own provenance/logging where needed; do not overload `id` or `source` with mutable locator details. |
Frames in `ObservationSequence.frames` must contain exactly four frames ordered oldest to newest, with a common positive shape and the model's declared encoding. Times must be strictly increasing, but need not have equal intervals. Manifest cadence, grid ID, and grid shape are reference metadata, not acceptance restrictions. Input georeference is optional.
### Quality bit rules

Interpret the flags as bit masks, not as enum values or equality tests:

| Radiust mask condition | Preludio quality | Value handling |
| --- | --- | --- |
| `bits & 0x1 != 0` | `PixelQuality::Missing` (`1`) | Keep the supplied raw byte; Missing remains invalid and prediction is rejected. This takes precedence if `0x1` and `0x2` are both set. |
| Otherwise, `bits & 0x2 != 0` | `PixelQuality::OutsideCoverage` (`2`) | Keep the supplied raw byte; OutsideCoverage remains invalid and prediction is rejected. |
| Otherwise, `bits & 0x20 != 0` | `PixelQuality::Valid` (`0`) | Set the raw u8 value to `0`: below-detection is a valid no-echo value. |
| None of `0x1`, `0x2`, `0x20` | `PixelQuality::Valid` (`0`) | Preserve the supplied raw gray byte. |

The current package declares `missing_policy = reject`, so any `Missing` or `OutsideCoverage` pixel makes prediction fail with `IncompatibleInput`. Other Radiust quality bits have no corresponding Preludio quality variant; decide their source-specific meaning before mapping and do not silently discard unknown or meaningful flags. An optional sequence-level `zero_threshold` additionally zeros raw values strictly below the threshold before normalization and resize.

This Rust fragment illustrates quality/time mapping only after a caller has obtained raw u8 bytes matching the duckomo gray contract. `gray_u8` is an explicit input; this code does not convert `RadarField.values` or prove that a Radiust source can supply such bytes. It checks that the raw byte count matches the validated field:

```rust
use std::{error::Error, io::{Error as IoError, ErrorKind}};

use preludio::{
    time::parse_rfc3339_utc, FrameValues, ObservationFrame, PixelQuality,
};
use radiust_core::model::{FrameRef, RadarField};

fn map_frame(
    field: &RadarField,
    frame_ref: &FrameRef,
    gray_u8: &[u8],
) -> Result<ObservationFrame, Box<dyn Error + Send + Sync>> {
    field
        .validate()
        .map_err(|error| IoError::new(ErrorKind::InvalidData, error.to_string()))?;
    if field.shape.len() != 2 {
        return Err(IoError::new(ErrorKind::InvalidData, "expected a 2-D radar field").into());
    }

    if gray_u8.len() != field.values.len() {
        return Err(IoError::new(ErrorKind::InvalidData, "raw gray byte count does not match field shape").into());
    }

    let valid_time = parse_rfc3339_utc(&field.valid_time)?;
    if valid_time != parse_rfc3339_utc(&frame_ref.valid_time)? {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "FrameRef and RadarField valid_time identify different instants",
        )
        .into());
    }

    const MAPPED_FLAGS: u16 = 0x1 | 0x2 | 0x20;
    if field.quality.iter().any(|&bits| bits & !MAPPED_FLAGS != 0) {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "RadarField contains quality bits without a Preludio mapping",
        )
        .into());
    }

    let mut values = gray_u8.to_vec();
    let quality = field
        .quality
        .iter()
        .enumerate()
        .map(|(index, &bits)| {
            if bits & 0x1 != 0 {
                PixelQuality::Missing
            } else if bits & 0x2 != 0 {
                PixelQuality::OutsideCoverage
            } else if bits & 0x20 != 0 {
                values[index] = 0;
                PixelQuality::Valid
            } else {
                PixelQuality::Valid
            }
        })
        .collect();

    let source = match frame_ref.station.as_deref() {
        Some(station) => format!(
            "radiust:{}/{}/{}",
            frame_ref.source, frame_ref.product, station
        ),
        None => format!("radiust:{}/{}", frame_ref.source, frame_ref.product),
    };

    Ok(ObservationFrame {
        id: frame_ref.logical_id.clone(),
        source,
        valid_time,
        values: FrameValues::U8(values),
        quality: Some(quality),
    })
}
```

Build the sequence only when all four frames have the same positive two-dimensional shape. Do not compare the source grid ID or dimensions with the model manifest, require a cadence, or require georeference. The example leaves the optional georeference unset; it does not perform a spatial conversion:

```rust
use preludio::{GridDescriptor, ObservationSequence};

let sequence = ObservationSequence {
    grid: GridDescriptor {
        id: source_grid_label.to_owned(),
        shape: [field.shape[0], field.shape[1]],
        georeference: None,
    },
    encoding: "training-grayscale-u8".to_owned(),
    frames, // exactly four same-shape frames in oldest-to-newest order
    zero_threshold: None,
};
```

The example only constructs the Preludio contract from a `gray_u8` buffer supplied by another layer. The current generic `Vec<f32>` adapter is not evidence that the source has or computes duckomo-compatible gray bytes.

## Native Rust call pattern

Open and verify the model package once during process startup, create one session, then share that session through `Arc<Session>`. `Session::predict` is synchronous and blocking; run it with `tokio::task::spawn_blocking` so it does not occupy an async runtime worker thread.

```rust
use std::{error::Error, path::Path, sync::Arc};

use preludio::{
    Forecast, ModelPackage, ObservationSequence, PreludioError, Session, SessionOptions,
};

fn open_nowcast(package_path: &Path) -> Result<Arc<Session>, PreludioError> {
    let package = ModelPackage::open(package_path)?; // once at startup; verifies package hash
    let session = Session::open(package, SessionOptions::default())?;
    Ok(Arc::new(session))
}

fn report_preludio_error(error: &PreludioError) {
    match error {
        PreludioError::ModelUnavailable(reason) => eprintln!("model unavailable: {reason}"),
        PreludioError::IncompatibleInput(violations) => {
            eprintln!("incompatible observations: {violations:?}")
        }
        PreludioError::ExecutionFailed { execution, message } => {
            eprintln!("inference failed on {execution:?}: {message}")
        }
        PreludioError::InvalidOutput { execution, reason } => {
            eprintln!("invalid forecast from {execution:?}: {reason:?}")
        }
    }
}

async fn predict(
    session: Arc<Session>,
    sequence: ObservationSequence,
) -> Result<Forecast, Box<dyn Error + Send + Sync>> {
    let worker_session = Arc::clone(&session);
    let result = tokio::task::spawn_blocking(move || worker_session.predict(&sequence)).await?;
    result.map_err(|error| {
        report_preludio_error(&error);
        Box::new(error) as Box<dyn Error + Send + Sync>
    })
}
```

Store the returned `Arc<Session>` in Radiust application state and clone the `Arc` per update. Handle `ModelUnavailable`, `IncompatibleInput`, `ExecutionFailed`, and `InvalidOutput` at the application boundary as shown; a Tokio task join failure is a separate task/runtime error. `Session::predict` returns no partial forecast on any Preludio error.

## Radiust Web call pattern

Serve the application with these **HTTP response headers** on the page response:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

These headers make the page cross-origin isolated, which is required for the model target's WebAssembly threading. Serve `/ort/` resources same-origin or with compatible resource policy headers as well; HTML `<meta>` tags do not replace the response headers.

The TypeScript API uses camel-case `validTime`, numeric quality codes (`0`, `1`, `2`), and typed arrays. Normalize timestamps to RFC 3339 UTC before creating the sequence. Then create the session once, listen for state changes, and await `predict`:

```ts
import {
  createSession,
  PreludioError,
  type Forecast,
  type NowcastSession,
  type ObservationSequence,
  type RunState,
} from "@chorust/preludio";

const stateLabel = document.querySelector<HTMLElement>("#run-state")!;
const failureLabel = document.querySelector<HTMLElement>("#failure")!;
const staleLabel = document.querySelector<HTMLElement>("#stale")!;
const timeline = document.querySelector<HTMLElement>("#timeline")!;
const modelLabel = document.querySelector<HTMLElement>("#model-id")!;
const executionLabel = document.querySelector<HTMLElement>("#execution")!;

function showFailure(error: unknown): void {
  failureLabel.textContent = error instanceof PreludioError
    ? `${error.category}: ${error.message} (${JSON.stringify(error.detail)})`
    : error instanceof Error ? error.message : String(error);
}

function renderTimeline(forecast: Forecast): void {
  const { metadata } = forecast;
  const observations = document.createElement("section");
  const observationHeading = document.createElement("h3");
  observationHeading.textContent = "OBSERVATIONS";
  observations.append(observationHeading);

  const observedTimes = document.createElement("ol");
  for (const frame of metadata.provenance.input_frames) {
    const item = document.createElement("li");
    item.textContent = frame.valid_time;
    observedTimes.append(item);
  }
  observations.append(observedTimes);

  const boundary = document.createElement("p");
  boundary.className = "observation-forecast-boundary";
  boundary.textContent = `LAST OBSERVATION · ${metadata.provenance.last_observation_time} · FORECAST BEGINS`;

  const forecasts = document.createElement("section");
  const forecastHeading = document.createElement("h3");
  forecastHeading.textContent = "FORECAST";
  forecasts.append(forecastHeading);
  const forecastTimes = document.createElement("ol");
  metadata.valid_times.forEach((validTime, index) => {
    const item = document.createElement("li");
    item.textContent = `+${metadata.lead_minutes[index]} min · ${validTime}`;
    forecastTimes.append(item);
  });
  forecasts.append(forecastTimes);

  // Replacing the list happens only after a successful prediction.
  timeline.replaceChildren(observations, boundary, forecasts);
  modelLabel.textContent = metadata.provenance.model.id;
  executionLabel.textContent = JSON.stringify(metadata.provenance.execution);
}

async function startNowcast(sequence: ObservationSequence): Promise<void> {
  let session: NowcastSession;
  try {
    session = await createSession({
      packageUrl: "/models/ghost-unet3d-step-22000",
      ortWasmPaths: "/ort/",
    });
  } catch (error) {
    showFailure(error);
    return;
  }

  session.onStateChange((state: RunState) => {
    stateLabel.textContent = state;
  });

  let lastSuccessfulForecast: Forecast | undefined;
  async function update(sequence: ObservationSequence): Promise<void> {
    try {
      const next = await session.predict(sequence);
      lastSuccessfulForecast = next;
      failureLabel.textContent = "";
      staleLabel.textContent = "";
      renderTimeline(next);
    } catch (error) {
      showFailure(error);
      if (lastSuccessfulForecast) {
        staleLabel.textContent =
          `Showing the last successful forecast, generated at ${lastSuccessfulForecast.metadata.provenance.generated_at}. ` +
          "Its displayed valid times are retained until a new prediction succeeds.";
      } else {
        staleLabel.textContent = "No successful forecast is available yet.";
      }
      // Leave the current timeline and its valid-time labels in place on failure.
    }
  }

  await update(sequence);
  // Call update(nextSequence) for each later observation refresh.
}
```

`renderTimeline` displays all observation times, marks the last observation / forecast boundary, then lists every `valid_times[i]` with `lead_minutes[i]`. The output count comes from `tensor_output.shape[1]`; each lead is computed from the average input interval and rounded separately to the nearest second, with half seconds rounded up. `lead_minutes` is a floating-point value and may be fractional. The recorded fixed model reference uses four six-minute intervals and 20 outputs, so it still yields 6, 12, 18, …, 120 minutes; that sample does not restrict runtime cadence. The metadata formatter emits UTC times in `YYYY-MM-DDTHH:MM:SSZ` form. Show the model ID from `metadata.provenance.model.id` (currently `ghost-unet3d-step-22000`) and execution details from `metadata.provenance.execution`. On update failure, show the error and mark the previous forecast stale while retaining its timestamps and timeline until a later prediction succeeds.

For the full sequence type and validation rules, see [the Web API contract](../specs/001-local-nowcast-runtime/contracts/web-api.md) and [the data model](../specs/001-local-nowcast-runtime/data-model.md).