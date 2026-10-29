import { createHash } from "node:crypto";
import { beforeAll, expect, test } from "bun:test";
import { initCore, packageFromParts, preprocess } from "../src/core.js";
import { PreludioError } from "../src/errors.js";

type FixtureCase = {
  name: string;
  input: string;
  reference: string;
  input_shape: number[];
  output_size: number[];
};

type NpyArray = {
  shape: number[];
  values: Uint8Array | Float32Array;
};

const fixtureDirectory = new URL("../../tests/fixtures/preprocess/", import.meta.url);

async function readNpy(filename: string): Promise<NpyArray> {
  const bytes = new Uint8Array(await Bun.file(new URL(filename, fixtureDirectory)).arrayBuffer());
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  if (!magic.every((byte, index) => bytes[index] === byte)) {
    throw new Error(`${filename} is not a NumPy .npy file`);
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const majorVersion = bytes[6];
  const headerLengthOffset = majorVersion === 1 ? 8 : 8;
  const headerOffset = majorVersion === 1 ? 10 : 12;
  const headerLength = majorVersion === 1
    ? view.getUint16(headerLengthOffset, true)
    : view.getUint32(headerLengthOffset, true);
  const header = new TextDecoder().decode(bytes.subarray(headerOffset, headerOffset + headerLength));
  const dtypeMatch = header.match(/[\"']descr[\"']\s*:\s*[\"']([^\"']+)[\"']/);
  const shapeMatch = header.match(/[\"']shape[\"']\s*:\s*\(([^)]*)\)/);
  if (!dtypeMatch || !shapeMatch) throw new Error(`${filename} has an unsupported .npy header`);

  const dtype = dtypeMatch[1];
  const shape = Array.from(shapeMatch[1].matchAll(/\d+/g), ([dimension]) => Number(dimension));
  const elementCount = shape.reduce((count, dimension) => count * dimension, 1);
  const dataOffset = headerOffset + headerLength;

  if (/^[|<>=]?u1$/.test(dtype)) {
    return { shape, values: bytes.subarray(dataOffset, dataOffset + elementCount) };
  }
  if (/^[|<>=]?f4$/.test(dtype)) {
    const littleEndian = !dtype.startsWith(">");
    const data = new DataView(bytes.buffer, bytes.byteOffset + dataOffset, elementCount * 4);
    const values = new Float32Array(elementCount);
    for (let index = 0; index < elementCount; index += 1) {
      values[index] = data.getFloat32(index * 4, littleEndian);
    }
    return { shape, values };
  }
  throw new Error(`${filename} uses unsupported .npy dtype ${dtype}`);
}

function createManifest(inputShape: number[], outputSize: number[], modelHash: string) {
  const [frameCount, inputHeight, inputWidth] = inputShape;
  const [outputHeight, outputWidth] = outputSize;
  const grid = (id: string, height: number, width: number) => ({
    id,
    shape: [height, width],
    georeferenced: false,
    georeference: null,
  });

  return {
    schema: "preludio.model-package/1",
    id: "test-model",
    model: { file: "model.onnx", sha256: modelHash, format: "onnx", opset: 18 },
    origin: {
      checkpoint: { file: "ckpt", step: 0, sha256: "0".repeat(64), run: "test" },
      source_sha256: "0".repeat(64),
      config_sha256: "0".repeat(64),
      exporter: "test",
    },
    observation: {
      frames: frameCount,
      cadence_seconds: 360,
      cadence_tolerance_seconds: 0,
      encoding: "training-grayscale-u8",
      grid: grid("test-grid", inputHeight, inputWidth),
      missing_policy: "reject",
    },
    tensor_input: {
      name: "radar",
      dtype: "float32",
      shape: [1, frameCount, outputHeight, outputWidth],
      axes: ["batch", "time", "y", "x"],
      range: [0, 1],
    },
    tensor_output: {
      name: "forecast",
      dtype: "float32",
      shape: [1, 20, outputHeight, outputWidth],
      axes: ["batch", "lead_time", "y", "x"],
      range: [0, 1],
    },
    preprocessing: [
      { op: "u8_div_255" },
      { op: "resize_bilinear_aa", height: outputHeight, width: outputWidth, align_corners: false },
      { op: "round_f16" },
    ],
    forecast: {
      lead_minutes: [6],
      encoding: "training-grayscale-normalized",
      grid: grid("test-grid-out", outputHeight, outputWidth),
    },
    supported_targets: [],
  };
}

beforeAll(async () => {
  await initCore();
});

test("preprocess matches every reference fixture bit-for-bit", async () => {
  const metadata = JSON.parse(await Bun.file(new URL("meta.json", fixtureDirectory)).text()) as {
    cases: FixtureCase[];
  };

  for (const fixture of metadata.cases) {
    const input = await readNpy(fixture.input);
    const reference = await readNpy(fixture.reference);
    expect(input.shape).toEqual(fixture.input_shape);

    const [frameCount, height, width] = fixture.input_shape;
    const [outputHeight, outputWidth] = fixture.output_size;
    const frameSize = height * width;
    const inputValues = input.values as Uint8Array;
    const sequence = {
      grid: { id: "test-grid", shape: [height, width], georeference: null },
      encoding: "training-grayscale-u8",
      zero_threshold: null,
      frames: Array.from({ length: frameCount }, (_, frameIndex) => ({
        id: `frame-${frameIndex}`,
        source: "fixture",
        valid_time: 1_700_000_000 + frameIndex * 360,
        values: Array.from(inputValues.subarray(frameIndex * frameSize, (frameIndex + 1) * frameSize)),
      })),
    };
    const manifest = createManifest(fixture.input_shape, fixture.output_size, "0".repeat(64));
    const actual = preprocess(manifest, sequence);
    const expected = reference.values as Float32Array;

    expect(actual.length).toBe(expected.length);
    expect(actual.length).toBe(outputHeight * outputWidth * frameCount);
    const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
    const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
    for (let index = 0; index < actualBits.length; index += 1) {
      if (actualBits[index] !== expectedBits[index]) {
        throw new Error(`Case '${fixture.name}' differs at float ${index}: ${actual[index]} != ${expected[index]}`);
      }
    }
  }
});

test("packageFromParts reports model hash mismatches as PreludioError", () => {
  const originalBytes = new Uint8Array([1, 2, 3]);
  const alteredBytes = new Uint8Array([1, 2, 4]);
  const hash = createHash("sha256").update(originalBytes).digest("hex");
  const manifest = createManifest([1, 2, 3], [2, 2], hash);

  let thrown: unknown;
  try {
    packageFromParts(JSON.stringify(manifest), alteredBytes);
  } catch (err) {
    thrown = err;
  }

  expect(thrown).toBeInstanceOf(PreludioError);
  expect((thrown as PreludioError).category).toBe("ModelUnavailable");
  expect(JSON.stringify((thrown as PreludioError).detail)).toContain("HashMismatch");
});
