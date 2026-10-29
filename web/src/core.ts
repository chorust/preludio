import initWasm, {
  finishForecast as wasmFinishForecast,
  packageFromParts as wasmPackageFromParts,
  preprocess as wasmPreprocess,
  validate as wasmValidate,
  validateTimes as wasmValidateTimes,
  type InitInput,
} from "../pkg/preludio.js";
import { fromWasmError, PreludioError } from "./errors.js";

let initialization: Promise<void> | undefined;
let initialized = false;

export function initCore(wasmSource?: InitInput): Promise<void> {
  if (initialized) return Promise.resolve();
  if (initialization) return initialization;

  const wasmInitialization = wasmSource === undefined
    ? initWasm()
    : initWasm({ module_or_path: wasmSource });
  initialization = wasmInitialization.then(
    () => {
      initialized = true;
    },
    (err: unknown) => {
      initialization = undefined;
      throw fromWasmError(err);
    },
  );
  return initialization;
}

function callWasm<T>(callback: () => T): T {
  if (!initialized) {
    throw new PreludioError(
      "ExecutionFailed",
      { message: "The Preludio Wasm core is not initialized; await initCore() first." },
    );
  }

  try {
    return callback();
  } catch (err) {
    throw fromWasmError(err);
  }
}

export function packageFromParts(manifestJson: string, modelBytes: Uint8Array): unknown {
  return callWasm(() => wasmPackageFromParts(manifestJson, modelBytes));
}

export function validate(manifest: unknown, sequence: unknown): void {
  callWasm(() => wasmValidate(manifest, sequence));
}

export function validateTimes(manifest: unknown, frames: unknown): void {
  callWasm(() => wasmValidateTimes(manifest, frames));
}

export function preprocess(manifest: unknown, sequence: unknown): Float32Array {
  return callWasm(() => wasmPreprocess(manifest, sequence));
}

export function finishForecast(
  pkg: unknown,
  frames: unknown,
  raw: Float32Array,
  execution: unknown,
  generatedAt: number,
): unknown {
  return callWasm(() => wasmFinishForecast(pkg, frames, raw, execution, BigInt(generatedAt)));
}
