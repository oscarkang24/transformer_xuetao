// Verifies that the browser-side model (docs/js) reproduces the PyTorch
// forward pass. Compares the JS logits against the reference logits that
// export_toy_model.py baked into weights.js.
//
// Usage: node scripts/verify_js_model.mjs

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const docsJs = join(dirname(fileURLToPath(import.meta.url)), "..", "docs", "js");
const src = ["weights.js", "tensor.js", "tokenizer.js", "model.js"]
  .map((f) => readFileSync(join(docsJs, f), "utf8"))
  .join("\n");

const { trace, expected } = new Function(
  src +
    `\nconst t = Model.forward(MODEL_WEIGHTS.test.input_ids);
     return { trace: t, expected: MODEL_WEIGHTS.test.expected_last_logits };`
)();

const got = trace.logits[trace.logits.length - 1];
let maxErr = 0;
for (let i = 0; i < expected.length; i++) {
  maxErr = Math.max(maxErr, Math.abs(got[i] - expected[i]));
}

// Weights are rounded to 5 decimals on export, so allow a small tolerance.
const TOL = 2e-3;
console.log(`max |JS - PyTorch| over last-position logits: ${maxErr.toExponential(2)}`);
if (maxErr > TOL) {
  console.error(`FAIL: exceeds tolerance ${TOL}`);
  process.exit(1);
}
console.log("PASS: JS forward pass matches PyTorch");
