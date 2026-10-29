const fs = require("fs");
const path = require("path");
const { preprocess_wasm } = require("./wasm-pkg/preprocess_parity.js");

const root = __dirname;
const dataDir = path.join(root, "data");
const outputLen = 4 * 448 * 448;
const cases = ["sparse_blobs", "uniform_u8", "all_zero"];

for (const name of cases) {
  const input = new Uint8Array(fs.readFileSync(path.join(dataDir, `${name}_input.raw`)));
  let result = preprocess_wasm(input); // warm-up and compile the hot WASM path
  const times = [];
  for (let run = 0; run < 5; run++) {
    const start = performance.now();
    result = preprocess_wasm(input);
    times.push(performance.now() - start);
  }
  const prehalf = result.subarray(0, outputLen);
  const rounded = result.subarray(outputLen);
  fs.writeFileSync(path.join(dataDir, `${name}_wasm_prehalf.f32`), Buffer.from(prehalf.buffer, prehalf.byteOffset, prehalf.byteLength));
  fs.writeFileSync(path.join(dataDir, `${name}_wasm_output.f32`), Buffer.from(rounded.buffer, rounded.byteOffset, rounded.byteLength));
  console.log(`case=${name} run_ms=${times.map((x) => x.toFixed(6)).join(",")} warmup=1 measured_runs=${times.length}`);
}
