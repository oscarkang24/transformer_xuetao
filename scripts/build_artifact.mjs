// Bundles docs/ into a single self-contained HTML file with everything
// inlined — the app makes no external requests.
//
// Default output is a fragment (no <html>/<head>/<body>) for publishing as
// a claude.ai Artifact, which supplies its own wrapper. Pass --standalone
// for a complete document you can open directly in a browser.
//
// Usage: node scripts/build_artifact.mjs <output-path> [--standalone]

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const docs = join(dirname(fileURLToPath(import.meta.url)), "..", "docs");
const out = process.argv[2];
const standalone = process.argv.includes("--standalone");
if (!out) {
  console.error("usage: node scripts/build_artifact.mjs <output-path> [--standalone]");
  process.exit(1);
}

const read = (p) => readFileSync(join(docs, p), "utf8");

const html = read("index.html");
// keep only what's inside <body>, minus the script tags (we inline them)
const body = html
  .match(/<body>([\s\S]*)<\/body>/)[1]
  .replace(/\s*<script src="[^"]+"><\/script>/g, "");

const css = read("css/style.css");
const scripts = ["js/weights.js", "js/tensor.js", "js/tokenizer.js",
  "js/model.js", "js/viz.js", "js/stages.js", "js/app.js"]
  .map(read)
  .join("\n;\n");

const title = "Transformer Builder — a GPT, step by step";
const core = (hideToggle) => `<style>
${css}${hideToggle ? `
/* The artifact viewer provides its own theme toggle (it stamps data-theme
   on the root); hide the in-app one to avoid two competing switches. */
#theme-toggle { display: none; }
` : ""}</style>
${body}
<script>
${scripts}
</script>`;

const page = standalone
  ? `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${title}</title>
</head>
<body>
${core(false)}
</body>
</html>
`
  : `<title>${title}</title>\n${core(true)}\n`;

writeFileSync(out, page);
console.log(`wrote ${out} (${(page.length / 1024).toFixed(0)} KB${standalone ? ", standalone" : ", artifact fragment"})`);
