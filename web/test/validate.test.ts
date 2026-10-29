import { createHash } from "node:crypto";
import { expect, mock, test } from "bun:test";
import { initCore } from "../src/core.js";

let outputDims: number[] = [];
const ortEnvironment = { wasm: { proxy: false, numThreads: 1, wasmPaths: undefined as string | undefined } };
const ortSession = {
  inputNames: ["radar"],
  outputNames: ["forecast"],
  run: mock(async () => ({ forecast: { data: new Float32Array(0), dims: outputDims } })),
  release: mock(async () => {}),
};
mock.module("onnxruntime-web", () => ({
  env: ortEnvironment,
  Tensor: class {
    constructor(readonly type: string, readonly data: Float32Array, readonly dims: number[]) {}
  },
  InferenceSession: { create: mock(async () => ortSession) },
}));

// Load after mock.module so the session binds to the fake inference runtime.
const { createSession } = await import("../src/session.js");
await initCore();

interface FixtureManifest {
  model: { sha256: string };
  observation: {
    frames: number;
    encoding: string;
    grid: { id: string; shape: [number, number] };
  };
  supported_targets: unknown[];
  tensor_input: { shape: [number, number, number, number] };
  tensor_output: { shape: [number, number, number, number] };
}

interface SequenceFixture {
  grid: { id: string; shape: [number, number] };
  encoding: string;
  frames: {
    id: string;
    source: string;
    validTime: string;
    values: Uint8Array;
    quality?: Uint8Array;
  }[];
}

async function packageFixture() {
  const source = await Bun.file(new URL("../../tests/fixtures/manifest-v1.json", import.meta.url)).text();
  const manifest = JSON.parse(source) as FixtureManifest;
  const modelBytes = new Uint8Array([0x50, 0x52, 0x45, 0x4c, 0x55, 0x44, 0x49, 0x4f]);
  outputDims = [...manifest.tensor_output.shape];
  manifest.model.sha256 = createHash("sha256").update(modelBytes).digest("hex");
  manifest.supported_targets = [];
  return { manifest, manifestText: JSON.stringify(manifest), modelBytes };
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

function validSequence(manifest: FixtureManifest, frameCount = manifest.observation.frames): SequenceFixture {
  const height = 1;
  const width = 1;
  const start = 1_700_000_000;
  return {
    grid: { id: "independent-grid", shape: [height, width] },
    encoding: manifest.observation.encoding,
    frames: Array.from({ length: frameCount }, (_, index) => ({
      id: `frame-${index}`,
      source: "fixture",
      validTime: new Date((start + index * 13 + index * index * 7) * 1000).toISOString(),
      values: new Uint8Array(height * width),
    })),
  };
}

async function openValidationSession(manifestText: string, modelBytes: Uint8Array<ArrayBuffer>) {
  const restoreFetch = installPackageFetch(manifestText, modelBytes);
  try {
    return await createSession({ packageUrl: "https://models.test/radar", policy: "validation" });
  } finally {
    restoreFetch();
  }
}

test("a sequence with three frames is rejected as incompatible input", async () => {
  const { manifest, manifestText, modelBytes } = await packageFixture();
  const session = await openValidationSession(manifestText, modelBytes);
  const sequence = validSequence(manifest, 3);

  try {
    await expect(session.predict(sequence)).rejects.toMatchObject({ category: "IncompatibleInput" });
    expect(session.state).toBe("failed");
  } finally {
    await session.dispose();
  }
});

test("validTime without a zone is rejected as incompatible input", async () => {
  const { manifest, manifestText, modelBytes } = await packageFixture();
  const session = await openValidationSession(manifestText, modelBytes);
  const sequence = validSequence(manifest);
  sequence.frames[1]!.validTime = "2023-11-14T22:19:20";

  try {
    await expect(session.predict(sequence)).rejects.toMatchObject({ category: "IncompatibleInput" });
    expect(session.state).toBe("failed");
  } finally {
    await session.dispose();
  }
});

test("a sequence with independent grid dimensions and irregular times reaches inference", async () => {
  const { manifest, manifestText, modelBytes } = await packageFixture();
  const session = await openValidationSession(manifestText, modelBytes);

  try {
    await expect(session.predict(validSequence(manifest))).rejects.toMatchObject({ category: "InvalidOutput" });
  } finally {
    await session.dispose();
  }
});

test("invalid dimensions, frame lengths and missing pixels fail before inference", async () => {
  const { manifest, manifestText, modelBytes } = await packageFixture();
  const session = await openValidationSession(manifestText, modelBytes);
  try {
    const invalidShape = validSequence(manifest);
    invalidShape.grid.shape = [0, 3];
    const invalidLength = validSequence(manifest);
    invalidLength.frames[1]!.values = new Uint8Array(2);
    const missing = validSequence(manifest);
    missing.frames[1]!.quality = new Uint8Array(missing.frames[1]!.values.length).fill(1);
    for (const sequence of [invalidShape, invalidLength, missing]) {
      const callsBefore = ortSession.run.mock.calls.length;
      await expect(session.predict(sequence)).rejects.toMatchObject({ category: "IncompatibleInput" });
      expect(ortSession.run.mock.calls.length).toBe(callsBefore);
    }
  } finally {
    await session.dispose();
  }
});

test("predictTensor rejects invalid frame count and time ordering before inference", async () => {
  const { manifest, manifestText, modelBytes } = await packageFixture();
  const session = await openValidationSession(manifestText, modelBytes);
  const validFrames = validSequence(manifest).frames.map(({ id, source, validTime }) => ({
    id,
    source,
    validTime,
  }));
  const inputShape = manifest.tensor_input.shape;
  const input = new Float32Array(inputShape[1] * inputShape[2] * inputShape[3]);

  try {
    const callsBeforeFrameCount = ortSession.run.mock.calls.length;
    await expect(session.predictTensor(input, validFrames.slice(1))).rejects.toMatchObject({
      category: "IncompatibleInput",
    });
    expect(ortSession.run.mock.calls.length).toBe(callsBeforeFrameCount);

    const unorderedFrames = validFrames.map((frame) => ({ ...frame }));
    unorderedFrames[2]!.validTime = unorderedFrames[1]!.validTime;
    const callsBeforeTimeOrder = ortSession.run.mock.calls.length;
    await expect(session.predictTensor(input, unorderedFrames)).rejects.toMatchObject({
      category: "IncompatibleInput",
    });
    expect(ortSession.run.mock.calls.length).toBe(callsBeforeTimeOrder);
  } finally {
    await session.dispose();
  }
});
