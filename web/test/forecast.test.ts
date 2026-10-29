import { createHash } from "node:crypto";
import { expect, mock, test } from "bun:test";
import { initCore } from "../src/core.js";
import { detectPlatform } from "../src/platform.js";
import type { NowcastSession } from "../src/session.js";

interface FixtureManifest {
  id: string;
  model: { file: string; sha256: string };
  observation: {
    frames: number;
    encoding: string;
    grid: { id: string; shape: [number, number] };
  };
  tensor_input: { shape: [number, number, number, number] };
  tensor_output: { shape: [number, number, number, number] };
  forecast: {
    lead_minutes: number[];
    encoding: string;
    grid: { id: string; shape: [number, number]; georeferenced: boolean };
  };
  supported_targets: unknown[];
}

let outputValues = new Float32Array(0);
let outputDims: number[] = [];
let lastInputTensor: { data: Float32Array; dims: number[] } | undefined;
const ortEnvironment = { wasm: { proxy: false, numThreads: 1, wasmPaths: undefined as string | undefined } };
let releaseOrtSession: () => Promise<void> = async () => {};
const ortSession = {
  run: mock(async () => ({ forecast: { data: outputValues, dims: outputDims } })),
  release: mock(async () => releaseOrtSession()),
};
let createOrtSession: () => Promise<typeof ortSession> = async () => ortSession;
mock.module("onnxruntime-web", () => ({
  env: ortEnvironment,
  Tensor: class {
    constructor(readonly type: string, readonly data: Float32Array, readonly dims: number[]) {
      lastInputTensor = { data, dims };
    }
  },
  InferenceSession: { create: mock(async () => createOrtSession()) },
}));

// Load after mock.module so the session binds to the fake inference runtime.
const { createSession } = await import("../src/session.js");
await initCore();

function record(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("Expected an object in forecast output");
  }
  return value as Record<string, unknown>;
}

function installPackageFetch(manifestText: string, modelBytes: Uint8Array<ArrayBuffer>): () => void {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith("/manifest.json")) return new Response(manifestText);
    if (url.endsWith("/model.onnx")) return new Response(modelBytes);
    return new Response("not found", { status: 404 });
  }) as typeof fetch;
  return () => {
    globalThis.fetch = originalFetch;
  };
}

test("forecast output contains the metadata contract fields and provenance", async () => {
  const source = await Bun.file(new URL("../../tests/fixtures/manifest-v1.json", import.meta.url)).text();
  const manifest = JSON.parse(source) as FixtureManifest;
  const modelBytes = new Uint8Array([0x50, 0x52, 0x45, 0x4c, 0x55, 0x44, 0x49, 0x4f]);
  manifest.model.sha256 = createHash("sha256").update(modelBytes).digest("hex");
  manifest.forecast.lead_minutes = [6];
  manifest.supported_targets = [];
  outputValues = new Float32Array(
    manifest.tensor_output.shape[1] * manifest.tensor_output.shape[2] * manifest.tensor_output.shape[3],
  );
  outputDims = [...manifest.tensor_output.shape];

  const session = await (async () => {
    const restoreFetch = installPackageFetch(JSON.stringify(manifest), modelBytes);
    try {
      return await createSession({ packageUrl: "https://models.test/radar", policy: "validation" });
    } finally {
      restoreFetch();
    }
  })();
  const firstTime = 1_700_000_000;
  const frameOffsets = [0, 13, 42, 60];
  const frames = Array.from({ length: manifest.observation.frames }, (_, index) => ({
    id: `frame-${index}`,
    source: "fixture:radar",
    validTime: new Date((firstTime + frameOffsets[index]!) * 1000).toISOString(),
  }));
  const inputShape = manifest.tensor_input.shape;

  try {
    expect(session.execution.target_id).toBeNull();
    expect(session.execution.verified).toBe(false);
    const forecast = await session.predictTensor(
      new Float32Array(inputShape[1] * inputShape[2] * inputShape[3]),
      frames,
    );
    const metadata = forecast.metadata;

    expect(forecast.values).toBeInstanceOf(Float32Array);
    expect(Object.keys(metadata).sort()).toEqual([
      "encoding",
      "grid",
      "kind",
      "lead_minutes",
      "provenance",
      "valid_times",
    ]);
    expect(metadata.kind).toBe("forecast");
    expect(metadata.encoding).toBe("training-grayscale-normalized");
    expect(metadata.grid).toEqual({
      id: manifest.forecast.grid.id,
      shape: manifest.forecast.grid.shape,
      georeferenced: false,
    });
    const intervalSeconds = 60 / (frames.length - 1);
    const leadSeconds = Array.from(
      { length: manifest.tensor_output.shape[1] },
      (_, index) => Math.floor((index + 1) * intervalSeconds + 0.5),
    );
    expect(metadata.lead_minutes).toEqual(leadSeconds.map((seconds) => seconds / 60));
    expect(metadata.lead_minutes).toHaveLength(manifest.tensor_output.shape[1]);
    expect(metadata.valid_times.map(Date.parse)).toEqual(
      leadSeconds.map((seconds) => (firstTime + 60 + seconds) * 1000),
    );
    expect(metadata.provenance.input_frames).toHaveLength(manifest.observation.frames);
    expect(metadata.provenance.last_observation_time).toBeDefined();
    expect(metadata.provenance.generated_at).toBeDefined();
    expect(metadata.provenance.model).toEqual({
      id: manifest.id,
      sha256: manifest.model.sha256,
      schema: "preludio.model-package/1",
    });
    expect(metadata.provenance.execution).toEqual(session.execution);
    const wrongOutputDims = [...manifest.tensor_output.shape];
    wrongOutputDims[1] = wrongOutputDims[1]! * 2;
    wrongOutputDims[2] = wrongOutputDims[2]! / 2;
    expect(wrongOutputDims.reduce((product, dimension) => product * dimension, 1)).toBe(
      outputDims.reduce((product, dimension) => product * dimension, 1),
    );
    outputDims = wrongOutputDims;
    await expect(session.predictTensor(
      new Float32Array(inputShape[1] * inputShape[2] * inputShape[3]),
      frames,
    )).rejects.toMatchObject({ category: "InvalidOutput" });
    outputDims = [...manifest.tensor_output.shape];
    for (const zeroThreshold of [-1, 1.5, 256]) {
      await expect(session.predict({
        grid: { id: "arbitrary-grid", shape: [1, 1] },
        encoding: manifest.observation.encoding,
        zeroThreshold,
        frames: [],
      })).rejects.toMatchObject({ category: "IncompatibleInput" });
    }
    const runsBeforeFloatInput = ortSession.run.mock.calls.length;
    await expect(session.predict({
      grid: { id: "arbitrary-grid", shape: [1, 1] },
      encoding: manifest.observation.encoding,
      frames: [0, 1, 2, 3].map((value, index) => ({
        id: `float-${index}`,
        source: "fixture:float",
        validTime: new Date((firstTime + frameOffsets[index]!) * 1000).toISOString(),
        values: Float32Array.of(value),
      })),
    })).rejects.toMatchObject({ category: "IncompatibleInput" });
    expect(ortSession.run.mock.calls.length).toBe(runsBeforeFloatInput);
    const sequenceForecast = await session.predict({
      grid: { id: "arbitrary-grid", shape: [1, 1] },
      encoding: manifest.observation.encoding,
      zeroThreshold: 128,
      frames: [127, 128, 255, 1].map((value, index) => ({
        id: `raw-${index}`,
        source: "fixture:raw",
        validTime: new Date((firstTime + frameOffsets[index]!) * 1000).toISOString(),
        values: Uint8Array.of(value),
      })),
    });
    expect(sequenceForecast.values).toBeInstanceOf(Float32Array);
    expect(lastInputTensor?.dims).toEqual(manifest.tensor_input.shape);
    const frameSize = manifest.tensor_input.shape[2] * manifest.tensor_input.shape[3];
    expect(lastInputTensor?.data[0]).toBe(0);
    expect(lastInputTensor?.data[frameSize]).toBe(0.501953125); // 128/255 rounded to binary16.
    expect(lastInputTensor?.data[frameSize * 2]).toBe(1);
    expect(lastInputTensor?.data[frameSize * 3]).toBe(0);
  } finally {
    await session.dispose();
  }
});

test("validation only marks an exact runtime target as verified", async () => {
  const source = await Bun.file(new URL("../../tests/fixtures/manifest-v1.json", import.meta.url)).text();
  const manifest = JSON.parse(source) as FixtureManifest;
  const modelBytes = new Uint8Array([0x50, 0x52, 0x45, 0x4c, 0x55, 0x44, 0x49, 0x4f]);
  manifest.model.sha256 = createHash("sha256").update(modelBytes).digest("hex");
  manifest.supported_targets = [{
    id: "web-target",
    platform: detectPlatform(),
    engine: "onnxruntime-web",
    engine_version: "1.30.0",
    provider: "wasm",
    threads: 4,
    requires_cross_origin_isolation: false,
    validation: "reference/validation-web-chrome-ort-wasm.json",
    measurement: "reference/measurement-web-chrome-ort-wasm.json",
  }];
  const originalThreads = ortEnvironment.wasm.numThreads;
  const originalIsolation = Object.getOwnPropertyDescriptor(globalThis, "crossOriginIsolated");
  Object.defineProperty(globalThis, "crossOriginIsolated", { configurable: true, value: true });
  const restoreFetch = installPackageFetch(JSON.stringify(manifest), modelBytes);
  let matchingSession: NowcastSession | undefined;
  let overriddenSession: NowcastSession | undefined;

  try {
    matchingSession = await createSession({
      packageUrl: "https://models.test/radar",
      policy: "validation",
      threads: 4,
    });
    expect(matchingSession.execution).toMatchObject({ target_id: "web-target", verified: true, threads: 4 });
    await matchingSession.dispose();
    matchingSession = undefined;

    overriddenSession = await createSession({
      packageUrl: "https://models.test/radar",
      policy: "validation",
      threads: 2,
    });
    expect(overriddenSession.execution).toMatchObject({ target_id: null, verified: false, threads: 2 });
  } finally {
    await overriddenSession?.dispose();
    await matchingSession?.dispose();
    ortEnvironment.wasm.numThreads = originalThreads;
    if (originalIsolation === undefined) Reflect.deleteProperty(globalThis, "crossOriginIsolated");
    else Object.defineProperty(globalThis, "crossOriginIsolated", originalIsolation);
    restoreFetch();
  }
});

test("cancellation during ORT creation releases the session and preserves the abort reason", async () => {
  const source = await Bun.file(new URL("../../tests/fixtures/manifest-v1.json", import.meta.url)).text();
  const manifest = JSON.parse(source) as FixtureManifest;
  const modelBytes = new Uint8Array([0x50, 0x52, 0x45, 0x4c, 0x55, 0x44, 0x49, 0x4f]);
  manifest.model.sha256 = createHash("sha256").update(modelBytes).digest("hex");
  manifest.supported_targets = [];

  const createResult = Promise.withResolvers<typeof ortSession>();
  const createStarted = Promise.withResolvers<void>();
  createOrtSession = () => {
    createStarted.resolve(undefined);
    return createResult.promise;
  };
  let releaseCalls = 0;
  releaseOrtSession = async () => {
    releaseCalls += 1;
    throw new Error("release failed");
  };
  const controller = new AbortController();
  const abortReason = new Error("cancelled during runtime setup");
  const restoreFetch = installPackageFetch(JSON.stringify(manifest), modelBytes);

  try {
    const pendingSession = createSession({
      packageUrl: "https://models.test/radar",
      policy: "validation",
      signal: controller.signal,
    });
    await createStarted.promise;
    controller.abort(abortReason);
    createResult.resolve(ortSession);

    await expect(pendingSession).rejects.toBe(abortReason);
    expect(releaseCalls).toBe(1);
  } finally {
    createOrtSession = async () => ortSession;
    releaseOrtSession = async () => {};
    restoreFetch();
  }
});
