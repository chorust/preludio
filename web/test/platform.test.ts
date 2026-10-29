import { createHash } from "node:crypto";
import { expect, mock, test } from "bun:test";
import { initCore } from "../src/core.js";
import { detectPlatform } from "../src/platform.js";
import type { NowcastSession } from "../src/session.js";

interface FixtureTarget {
  id: string;
  platform: string;
  engine: string;
  engine_version: string;
  provider: string;
  threads: number | "default";
  requires_cross_origin_isolation: boolean;
  validation: string;
  measurement: string;
}

interface FixtureManifest {
  model: { sha256: string };
  supported_targets: FixtureTarget[];
}

const ortEnvironment = { wasm: { proxy: false, numThreads: 1, wasmPaths: undefined as string | undefined } };
const ortSession = { release: mock(async () => {}) };
let createOrtSession: () => Promise<typeof ortSession> = async () => {
  throw new Error("InferenceSession.create should not be reached in rejection tests");
};
mock.module("onnxruntime-web", () => ({
  env: ortEnvironment,
  Tensor: class {},
  InferenceSession: {
    create: mock(async () => createOrtSession()),
  },
}));

// The mock must be registered before loading the session module that imports it.
const { createSession } = await import("../src/session.js");
await initCore();

function overrideGlobal(name: string, value: unknown): () => void {
  const original = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  return () => {
    if (original === undefined) Reflect.deleteProperty(globalThis, name);
    else Object.defineProperty(globalThis, name, original);
  };
}

async function packageFixture(targets: FixtureTarget[] = [{
  id: "web-chrome-test",
  platform: "web-chrome",
  engine: "onnxruntime-web",
  engine_version: "1.30.0",
  provider: "wasm",
  threads: 4,
  requires_cross_origin_isolation: true,
  validation: "reference/validation-web-chrome.json",
  measurement: "reference/measurement-web-chrome.json",
}]) {
  const manifestText = await Bun.file(new URL("../../tests/fixtures/manifest-v1.json", import.meta.url)).text();
  const manifest = JSON.parse(manifestText) as FixtureManifest;
  const modelBytes = new Uint8Array([0x50, 0x52, 0x45, 0x4c, 0x55, 0x44, 0x49, 0x4f]);
  manifest.model.sha256 = createHash("sha256").update(modelBytes).digest("hex");
  manifest.supported_targets = targets;
  return { manifestText: JSON.stringify(manifest), modelBytes };
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
function chromeTarget(id: string, overrides: Partial<FixtureTarget> = {}): FixtureTarget {
  return {
    id,
    platform: "web-chrome",
    engine: "onnxruntime-web",
    engine_version: "1.30.0",
    provider: "wasm",
    threads: 4,
    requires_cross_origin_isolation: true,
    validation: `reference/${id}-validation.json`,
    measurement: `reference/${id}-measurement.json`,
    ...overrides,
  };
}

test("supported policy selects an exact runtime target among same-platform entries", async () => {
  const { manifestText, modelBytes } = await packageFixture([
    chromeTarget("web-chrome-old-version", { engine_version: "1.29.0" }),
    chromeTarget("web-chrome-native-provider", { engine: "onnxruntime", provider: "cpu" }),
    chromeTarget("web-chrome-current"),
  ]);
  const restoreFetch = installPackageFetch(manifestText, modelBytes);
  const restoreNavigator = overrideGlobal("navigator", {
    userAgent: "Mozilla/5.0 Chrome/129.0.0.0 Safari/537.36",
  });
  const restoreIsolation = overrideGlobal("crossOriginIsolated", true);
  const originalThreads = ortEnvironment.wasm.numThreads;
  createOrtSession = async () => ortSession;
  let session: NowcastSession | undefined;

  try {
    session = await createSession({ packageUrl: "https://models.test/radar" });
    expect(session.execution).toMatchObject({
      target_id: "web-chrome-current",
      verified: true,
      engine: "onnxruntime-web",
      engine_version: "1.30.0",
      provider: "wasm",
      threads: 4,
    });
    await session.dispose();
    session = undefined;
  } finally {
    await session?.dispose();
    createOrtSession = async () => {
      throw new Error("InferenceSession.create should not be reached in rejection tests");
    };
    ortEnvironment.wasm.numThreads = originalThreads;
    restoreIsolation();
    restoreNavigator();
    restoreFetch();
  }
});

test("supported policy rejects same-platform entries with no exact runtime match", async () => {
  const { manifestText, modelBytes } = await packageFixture([
    chromeTarget("web-chrome-old-version", { engine_version: "1.29.0" }),
  ]);
  const restoreFetch = installPackageFetch(manifestText, modelBytes);
  const restoreNavigator = overrideGlobal("navigator", {
    userAgent: "Mozilla/5.0 Chrome/129.0.0.0 Safari/537.36",
  });
  const restoreIsolation = overrideGlobal("crossOriginIsolated", true);

  try {
    await expect(createSession({ packageUrl: "https://models.test/radar" })).rejects.toMatchObject({
      category: "ModelUnavailable",
      detail: { TargetNotSupported: { platform: "web-chrome", supported: ["web-chrome"] } },
    });
  } finally {
    restoreIsolation();
    restoreNavigator();
    restoreFetch();
  }
});

test("validation matches exact threads across same-platform entries and permits unverified runs", async () => {
  const { manifestText, modelBytes } = await packageFixture([
    chromeTarget("web-chrome-four-threads"),
    chromeTarget("web-chrome-two-threads", { threads: 2 }),
  ]);
  const restoreFetch = installPackageFetch(manifestText, modelBytes);
  const restoreNavigator = overrideGlobal("navigator", {
    userAgent: "Mozilla/5.0 Chrome/129.0.0.0 Safari/537.36",
  });
  const restoreIsolation = overrideGlobal("crossOriginIsolated", true);
  const originalThreads = ortEnvironment.wasm.numThreads;
  createOrtSession = async () => ortSession;
  let session: NowcastSession | undefined;

  try {
    session = await createSession({
      packageUrl: "https://models.test/radar",
      policy: "validation",
      threads: 2,
    });
    expect(session.execution).toMatchObject({
      target_id: "web-chrome-two-threads",
      verified: true,
      threads: 2,
    });
    await session.dispose();
    session = await createSession({
      packageUrl: "https://models.test/radar",
      policy: "validation",
      threads: 3,
    });
    expect(session.execution).toMatchObject({ target_id: null, verified: false, threads: 3 });
  } finally {
    await session?.dispose();
    createOrtSession = async () => {
      throw new Error("InferenceSession.create should not be reached in rejection tests");
    };
    ortEnvironment.wasm.numThreads = originalThreads;
    restoreIsolation();
    restoreNavigator();
    restoreFetch();
  }
});

test("validation still requires isolation for multiple threads without a registered target", async () => {
  const { manifestText, modelBytes } = await packageFixture([]);
  const restoreFetch = installPackageFetch(manifestText, modelBytes);
  const restoreNavigator = overrideGlobal("navigator", {
    userAgent: "Mozilla/5.0 Chrome/129.0.0.0 Safari/537.36",
  });
  const restoreIsolation = overrideGlobal("crossOriginIsolated", false);

  try {
    await expect(createSession({
      packageUrl: "https://models.test/radar",
      policy: "validation",
      threads: 4,
    })).rejects.toMatchObject({
      category: "ModelUnavailable",
      detail: { IsolationRequired: null },
    });
  } finally {
    restoreIsolation();
    restoreNavigator();
    restoreFetch();
  }
});

test("Safari is rejected when the package only supports Chrome", async () => {
  const { manifestText, modelBytes } = await packageFixture();
  const restoreFetch = installPackageFetch(manifestText, modelBytes);
  const restoreNavigator = overrideGlobal("navigator", {
    userAgent: "Mozilla/5.0 (Macintosh) Version/17.0 Safari/605.1.15",
  });
  const restoreIsolation = overrideGlobal("crossOriginIsolated", true);

  try {
    expect(detectPlatform()).toBe("web-safari");
    await expect(createSession({ packageUrl: "https://models.test/radar" })).rejects.toMatchObject({
      category: "ModelUnavailable",
      detail: {
        TargetNotSupported: {
          platform: "web-safari",
          supported: ["web-chrome"],
        },
      },
    });
  } finally {
    restoreIsolation();
    restoreNavigator();
    restoreFetch();
  }
});

test("Chrome target requiring isolation is rejected when isolation is unavailable", async () => {
  const { manifestText, modelBytes } = await packageFixture();
  const restoreFetch = installPackageFetch(manifestText, modelBytes);
  const restoreNavigator = overrideGlobal("navigator", {
    userAgent: "Mozilla/5.0 Chrome/129.0.0.0 Safari/537.36",
  });
  const restoreIsolation = overrideGlobal("crossOriginIsolated", false);

  try {
    expect(detectPlatform()).toBe("web-chrome");
    await expect(createSession({ packageUrl: "https://models.test/radar" })).rejects.toMatchObject({
      category: "ModelUnavailable",
      detail: { IsolationRequired: null },
    });
  } finally {
    restoreIsolation();
    restoreNavigator();
    restoreFetch();
  }
});
