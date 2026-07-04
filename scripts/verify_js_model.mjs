// Verifies that the browser-side models (docs/js) reproduce their PyTorch
// forward passes, using the reference logits the export scripts baked into
// weights.js (causal GPT) and weights_diffusion.js (bidirectional twin).
//
// Usage: node scripts/verify_js_model.mjs

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const docsJs = join(dirname(fileURLToPath(import.meta.url)), "..", "docs", "js");
const src = ["weights.js", "weights_diffusion.js", "tensor.js", "tokenizer.js", "model.js"]
  .map((f) => readFileSync(join(docsJs, f), "utf8"))
  .join("\n");

const results = new Function(
  src +
    `
    const ar = Model.forward(MODEL_WEIGHTS.test.input_ids);
    const dt = DIFFUSION_WEIGHTS.test;
    const diff = Model.forward(dt.input_ids, DIFFUSION_WEIGHTS);
    return [
      { name: "AR (causal GPT), last-position logits",
        got: ar.logits[ar.logits.length - 1],
        want: MODEL_WEIGHTS.test.expected_last_logits },
      { name: "diffusion (bidirectional), logits at pos 1",
        got: diff.logits[1], want: dt.expected_logits_pos1 },
      { name: "diffusion (bidirectional), logits at pos 4",
        got: diff.logits[4], want: dt.expected_logits_pos4 },
    ];`
)();

// Weights are rounded to 5 decimals on export, so allow a small tolerance.
const TOL = 2e-3;
let failed = false;
for (const { name, got, want } of results) {
  let maxErr = 0;
  for (let i = 0; i < want.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(got[i] - want[i]));
  }
  const ok = maxErr <= TOL;
  failed ||= !ok;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}: max |JS - PyTorch| = ${maxErr.toExponential(2)}`);
}
process.exit(failed ? 1 : 0);
