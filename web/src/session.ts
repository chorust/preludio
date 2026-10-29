import * as ort from "onnxruntime-web";
import { fromWasmError, PreludioError } from "./errors.js";
import * as core from "./core.js";
import { detectPlatform, isCrossOriginIsolated } from "./platform.js";

export interface CreateOptions {
  packageUrl: string;
  ortWasmPaths?: string;
  policy?: "supported" | "validation";
  threads?: number;
  signal?: AbortSignal;
}

export type PixelQuality = 0 | 1 | 2;

export interface ObservationFrame {
  id: string;
  source: string;
  validTime: string;
  values: Uint8Array | Float32Array;
  quality?: Uint8Array;
}

export interface ObservationSequence {
  grid: {
    id: string;
    shape: [number, number];
    georeference?: { crs: string; affine: number[] };
  };
  encoding: string;
  zeroThreshold?: number;
  frames: ObservationFrame[];
}

export interface ExecutionInfo {
  target_id: string | null;
  verified: boolean;
  engine: "onnxruntime-web";
  engine_version: string;
  provider: "wasm";
  threads: number;
  platform: string;
  cross_origin_isolated?: boolean;
}

export interface ForecastMetadata {
  kind: "forecast";
  encoding: "training-grayscale-normalized";
  grid: {
    id: string;
    shape: [number, number];
    georeferenced: boolean;
    crs?: string;
    affine?: number[];
  };
  lead_minutes: number[];
  valid_times: string[];
  provenance: {
    input_frames: { id: string; source: string; valid_time: string }[];
    last_observation_time: string;
    model: { id: string; sha256: string; schema: "preludio.model-package/1" };
    generated_at: string;
    execution: ExecutionInfo;
    preludio_version: string;
  };
}

export interface Forecast {
  values: Float32Array;
  metadata: ForecastMetadata;
}

export type RunState = "idle" | "running" | "succeeded" | "failed";

export interface NowcastSession {
  readonly execution: ExecutionInfo;
  readonly state: RunState;
  onStateChange(listener: (state: RunState) => void): () => void;
  predict(sequence: ObservationSequence): Promise<Forecast>;
  predictTensor(
    input: Float32Array,
    frames: { id: string; source: string; validTime: string }[],
  ): Promise<Forecast>;
  dispose(): Promise<void>;
}

interface ModelTarget {
  id: string;
  platform: string;
  engine: string;
  engine_version: string;
  provider: string;
  threads: number | "default";
  requires_cross_origin_isolation: boolean;
}

interface Manifest {
  schema: "preludio.model-package/1";
  id: string;
  model: { file: string; sha256: string; format: "onnx"; opset: number };
  tensor_input: { name: string; shape: [number, number, number, number] };
  tensor_output: { name: string; shape: [number, number, number, number] };
  forecast: {
    encoding: string;
    grid: {
      id: string;
      shape: [number, number];
      georeferenced: boolean;
      crs?: string;
      affine?: number[];
    };
  };
  supported_targets: ModelTarget[];
  [key: string]: unknown;
}

interface CorePackage {
  manifest: Manifest;
}

type OrtSession = ort.InferenceSession;

const RFC3339_WITH_ZONE = /^(\d{4})-(\d{2})-(\d{2})[Tt]([01]\d|2[0-3]):([0-5]\d):([0-5]\d)(?:\.\d+)?([Zz]|[+-](?:[01]\d|2[0-3]):[0-5]\d)$/;
const PIXEL_QUALITY_VARIANTS = ["Valid", "Missing", "OutsideCoverage"] as const;
const ORT_ENGINE_VERSION = "1.30.0";

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function abortIfNeeded(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  if (signal.reason !== undefined) throw signal.reason;
  throw new DOMException("The operation was aborted", "AbortError");
}

function joinPackageUrl(packageUrl: string, path: string): string {
  return `${packageUrl.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}

async function fetchPackageResource(
  url: string,
  path: string,
  signal?: AbortSignal,
): Promise<Response> {
  try {
    const response = await fetch(url, { signal });
    if (response.ok) return response;
    throw new PreludioError(
      "ModelUnavailable",
      { FileMissing: { path } },
      `Model package resource not found: ${path}`,
    );
  } catch (error) {
    if (signal?.aborted) abortIfNeeded(signal);
    if (error instanceof PreludioError) throw error;
    throw new PreludioError(
      "ModelUnavailable",
      { FileMissing: { path } },
      `Could not load model package resource ${path}: ${messageOf(error)}`,
    );
  }
}

function parseManifest(manifestText: string): { model: { file: string } } {
  let value: unknown;
  try {
    value = JSON.parse(manifestText);
  } catch (error) {
    throw new PreludioError(
      "ModelUnavailable",
      { ManifestInvalid: { message: messageOf(error) } },
      "The model package manifest is not valid JSON.",
    );
  }

  if (typeof value !== "object" || value === null || !("model" in value)) {
    throw new PreludioError(
      "ModelUnavailable",
      { ManifestInvalid: { message: "manifest.model.file must be a non-empty string" } },
      "The model package manifest does not declare a model file.",
    );
  }
  const model = value.model;
  if (
    typeof model !== "object" || model === null || !("file" in model) ||
    typeof model.file !== "string" || model.file.length === 0
  ) {
    throw new PreludioError(
      "ModelUnavailable",
      { ManifestInvalid: { message: "manifest.model.file must be a non-empty string" } },
      "The model package manifest does not declare a model file.",
    );
  }

  return { model: { file: model.file } };
}

function validTimeToUnixSeconds(value: string, index: number): number {
  const match = RFC3339_WITH_ZONE.exec(value);
  if (match === null) {
    throw new PreludioError(
      "IncompatibleInput",
      [{ InvalidTime: { index, value } }],
      `Frame ${index} validTime must be RFC 3339 with an explicit time zone.`,
    );
  }

  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const daysInMonth = new Date(Date.UTC(year, month, 0)).getUTCDate();
  if (month < 1 || month > 12 || day < 1 || day > daysInMonth) {
    throw new PreludioError(
      "IncompatibleInput",
      [{ InvalidTime: { index, value } }],
      `Frame ${index} validTime is not a valid calendar time.`,
    );
  }

  const milliseconds = Date.parse(value.toUpperCase());
  if (!Number.isFinite(milliseconds)) {
    throw new PreludioError(
      "IncompatibleInput",
      [{ InvalidTime: { index, value } }],
      `Frame ${index} validTime is not a valid RFC 3339 time.`,
    );
  }
  return Math.floor(milliseconds / 1000);
}

function toCoreFrames(frames: readonly { id: string; source: string; validTime: string }[]) {
  return frames.map((frame, index) => ({
    id: frame.id,
    source: frame.source,
    valid_time: validTimeToUnixSeconds(frame.validTime, index),
  }));
}

function toCoreSequence(sequence: ObservationSequence) {
  const zeroThreshold = sequence.zeroThreshold;
  if (
    zeroThreshold !== undefined &&
    (!Number.isInteger(zeroThreshold) || zeroThreshold < 0 || zeroThreshold > 255)
  ) {
    throw new PreludioError(
      "IncompatibleInput",
      { field: "zeroThreshold", value: zeroThreshold },
      "zeroThreshold must be an integer from 0 through 255.",
    );
  }
  return {
    grid: {
      id: sequence.grid.id,
      shape: sequence.grid.shape,
      ...(sequence.grid.georeference === undefined
        ? {}
        : { georeference: sequence.grid.georeference }),
    },
    encoding: sequence.encoding,
    ...(sequence.zeroThreshold === undefined ? {} : { zero_threshold: sequence.zeroThreshold }),
    frames: sequence.frames.map((frame, index) => {
      if (sequence.encoding === "training-grayscale-u8" && frame.values instanceof Float32Array) {
        throw new PreludioError(
          "IncompatibleInput",
          { field: "frames.values", index, expected: "Uint8Array" },
          `Frame ${index} values for training-grayscale-u8 must be a Uint8Array.`,
        );
      }
      return {
        id: frame.id,
        source: frame.source,
        valid_time: validTimeToUnixSeconds(frame.validTime, index),
        values: Array.from(frame.values),
        ...(frame.quality === undefined
          ? {}
          : { quality: Array.from(frame.quality, (quality) => PIXEL_QUALITY_VARIANTS[quality] ?? quality) }),
      };
    }),
  };
}


function asForecast(value: unknown, execution: ExecutionInfo, manifest: Manifest): Forecast {
  const invalidForecast = () => new PreludioError(
    "InvalidOutput",
    { execution, reason: { Shape: null } },
    "The model returned an invalid forecast.",
  );
  if (typeof value !== "object" || value === null || !("values" in value) || !("metadata" in value)) {
    throw invalidForecast();
  }
  const values = value.values;
  const metadata = value.metadata;
  if (
    (!(values instanceof Float32Array) && !Array.isArray(values)) ||
    typeof metadata !== "object" || metadata === null ||
    !("grid" in metadata) || !("provenance" in metadata) ||
    typeof metadata.provenance !== "object" || metadata.provenance === null
  ) {
    throw invalidForecast();
  }

  const outputGrid = manifest.forecast.grid;
  return {
    values: values instanceof Float32Array ? values : Float32Array.from(values),
    metadata: {
      ...metadata,
      grid: {
        id: outputGrid.id,
        shape: outputGrid.shape,
        georeferenced: outputGrid.georeferenced,
        ...(outputGrid.crs === undefined ? {} : { crs: outputGrid.crs }),
        ...(outputGrid.affine === undefined ? {} : { affine: outputGrid.affine }),
      },
      provenance: {
        ...metadata.provenance,
        execution,
      },
    } as ForecastMetadata,
  };
}

export async function createSession(options: CreateOptions): Promise<NowcastSession> {
  abortIfNeeded(options.signal);

  const packageUrl = options.packageUrl.replace(/\/+$/, "");
  const manifestResponse = await fetchPackageResource(
    joinPackageUrl(packageUrl, "manifest.json"),
    "manifest.json",
    options.signal,
  );
  const manifestText = await manifestResponse.text();
  abortIfNeeded(options.signal);

  const parsedManifest = parseManifest(manifestText);
  const modelPath = parsedManifest.model.file;
  const modelResponse = await fetchPackageResource(
    joinPackageUrl(packageUrl, modelPath),
    modelPath,
    options.signal,
  );
  let modelBytes: Uint8Array | undefined = new Uint8Array(await modelResponse.arrayBuffer());
  abortIfNeeded(options.signal);

  await core.initCore();
  abortIfNeeded(options.signal);

  let packageValue: CorePackage;
  try {
    packageValue = core.packageFromParts(manifestText, modelBytes) as CorePackage;
  } catch (error) {
    throw fromWasmError(error);
  }
  const manifest = packageValue.manifest;
  const platform = detectPlatform();
  const policy = options.policy ?? "supported";
  const platformTargets = manifest.supported_targets.filter((candidate) => candidate.platform === platform);
  const supported = [...new Set(manifest.supported_targets.map((candidate) => candidate.platform))];
  const runtimeTarget = platformTargets.find((candidate) =>
    candidate.engine === "onnxruntime-web" &&
    candidate.engine_version === ORT_ENGINE_VERSION &&
    candidate.provider === "wasm"
  );

  const requestedThreads = policy === "validation"
    ? options.threads ?? (typeof runtimeTarget?.threads === "number" ? runtimeTarget.threads : undefined)
    : typeof runtimeTarget?.threads === "number"
    ? runtimeTarget.threads
    : undefined;
  const runtimeThreads = requestedThreads ?? ort.env.wasm.numThreads ?? 1;
  const executionTarget = platformTargets.find((candidate) =>
    candidate.engine === "onnxruntime-web" &&
    candidate.engine_version === ORT_ENGINE_VERSION &&
    candidate.provider === "wasm" &&
    (typeof candidate.threads === "number"
      ? candidate.threads === runtimeThreads
      : candidate.threads === "default" && requestedThreads === undefined)
  );

  if (policy === "supported" && executionTarget === undefined) {
    throw new PreludioError(
      "ModelUnavailable",
      { TargetNotSupported: { platform, supported } },
      `The model does not declare a supported target matching platform ${platform} and the current Web runtime configuration.`,
    );
  }

  const crossOriginIsolated = isCrossOriginIsolated();
  const requiresCrossOriginIsolation = runtimeThreads > 1 ||
    (executionTarget?.requires_cross_origin_isolation ??
      platformTargets.some((candidate) => candidate.requires_cross_origin_isolation));
  if (requiresCrossOriginIsolation && !crossOriginIsolated) {
    throw new PreludioError(
      "ModelUnavailable",
      { IsolationRequired: null },
      "This WebAssembly configuration requires cross-origin isolation.",
    );
  }

  ort.env.wasm.proxy = true;
  if (options.ortWasmPaths !== undefined) ort.env.wasm.wasmPaths = options.ortWasmPaths;
  if (requestedThreads !== undefined) ort.env.wasm.numThreads = requestedThreads;

  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders: ["wasm"],
    enableCpuMemArena: true,
    enableMemPattern: false,
    ...(requestedThreads === undefined ? {} : { intraOpNumThreads: requestedThreads }),
  };
  abortIfNeeded(options.signal);
  let ortSession: OrtSession;
  try {
    ortSession = await ort.InferenceSession.create(modelBytes.buffer, sessionOptions);
    modelBytes = undefined;
  } catch (error) {
    if (options.signal?.aborted) abortIfNeeded(options.signal);
    const message = messageOf(error);
    throw new PreludioError(
      "ModelUnavailable",
      { EngineLoadFailed: { message } },
      `Could not create the ONNX Runtime Web session: ${message}`,
    );
  }
  if (options.signal?.aborted) {
    try {
      await ortSession.release();
    } catch {
      // Preserve the caller's abort reason when cleanup also fails.
    }
    abortIfNeeded(options.signal);
  }

  const execution: ExecutionInfo = {
    target_id: executionTarget?.id ?? null,
    verified: executionTarget !== undefined,
    engine: "onnxruntime-web",
    engine_version: ORT_ENGINE_VERSION,
    provider: "wasm",
    threads: runtimeThreads,
    platform,
    cross_origin_isolated: crossOriginIsolated,
  };

  let state: RunState = "idle";
  let disposed = false;
  let disposePromise: Promise<void> | undefined;
  let queue: Promise<void> = Promise.resolve();
  const listeners = new Set<(state: RunState) => void>();

  const setState = (next: RunState): void => {
    state = next;
    for (const listener of [...listeners]) listener(next);
  };

  const run = async (
    input: Float32Array,
    frames: { id: string; source: string; valid_time: number }[],
  ): Promise<Forecast> => {
    try {
      const tensorData = new Float32Array(input);
      const tensor = new ort.Tensor("float32", tensorData, manifest.tensor_input.shape);
      const output = await ortSession.run({ [manifest.tensor_input.name]: tensor });
      const outputTensor = output[manifest.tensor_output.name];
      if (
        outputTensor === undefined ||
        !(outputTensor.data instanceof Float32Array) ||
        outputTensor.dims.length !== manifest.tensor_output.shape.length ||
        outputTensor.dims.some((dimension, index) => dimension !== manifest.tensor_output.shape[index])
      ) {
        throw new PreludioError(
          "InvalidOutput",
          { execution, reason: { Shape: null } },
          `The model did not return float32 output ${manifest.tensor_output.name} with the manifest shape.`,
        );
      }
      const completed = core.finishForecast(
        packageValue,
        frames,
        outputTensor.data,
        execution,
        Math.floor(Date.now() / 1000),
      );
      const forecast = asForecast(completed, execution, manifest);
      setState("succeeded");
      return forecast;
    } catch (error) {
      setState("failed");
      if (error instanceof PreludioError) throw error;
      const message = messageOf(error);
      throw new PreludioError(
        "ExecutionFailed",
        { execution, message },
        message,
      );
    }
  };

  const enqueue = <T>(operation: () => Promise<T>): Promise<T> => {
    if (disposed) {
      return Promise.reject(new PreludioError(
        "ExecutionFailed",
        { message: "The nowcast session has been disposed." },
        "The nowcast session has been disposed.",
      ));
    }

    const task = queue.then(operation, operation);
    queue = task.then(() => undefined, () => undefined);
    return task;
  };

  const session: NowcastSession = {
    get execution() {
      return execution;
    },
    get state() {
      return state;
    },
    onStateChange(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    predict(sequence) {
      return enqueue(async () => {
        setState("running");
        try {
          const normalized = toCoreSequence(sequence);
          const prepared = core.preprocess(manifest, normalized);
          const frames = normalized.frames.map(({ id, source, valid_time }) => ({ id, source, valid_time }));
          return await run(prepared, frames);
        } catch (error) {
          if (state !== "failed") setState("failed");
          throw error instanceof PreludioError ? error : fromWasmError(error);
        }
      });
    },
    predictTensor(input, frames) {
      return enqueue(async () => {
        setState("running");
        try {
          const normalizedFrames = toCoreFrames(frames);
          core.validateTimes(manifest, normalizedFrames);
          return await run(input, normalizedFrames);
        } catch (error) {
          if (state !== "failed") setState("failed");
          throw error instanceof PreludioError ? error : fromWasmError(error);
        }
      });
    },
    dispose() {
      if (disposePromise !== undefined) return disposePromise;
      disposed = true;
      disposePromise = queue.then(async () => {
        try {
          await ortSession.release();
        } catch (error) {
          throw new PreludioError(
            "ExecutionFailed",
            { execution, message: messageOf(error) },
            `Could not release the ONNX Runtime Web session: ${messageOf(error)}`,
          );
        }
      });
      return disposePromise;
    },
  };

  return session;
}
