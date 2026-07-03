// The six visualization stages, one per step of the build.
// Each stage exposes render(root) and reads the shared App.state
// ({ text, tokens, ids, trace }). All numbers shown are the real values
// from the trained toy model's forward pass — nothing is faked.

const Stages = [];

// Small helpers ------------------------------------------------------------

function stageHeader(num, title, ref, lede) {
  const frag = document.createDocumentFragment();
  frag.append(
    Viz.el("h2", { text: `${num}. ${title}` }),
    Viz.el("div", { class: "chapter-ref", text: ref }),
    Viz.el("p", { class: "lede", text: lede })
  );
  return frag;
}

function footnote(html) {
  return Viz.el("div", { class: "footnote", html });
}

function tokenLabels() {
  return App.state.tokens.map((t) => t.token);
}

// ===========================================================================
// Stage 1 — Tokenize
// ===========================================================================

Stages.push({
  name: "Tokenize",
  chapter: "Ch. 2",
  render(root) {
    root.append(stageHeader(1, "Tokenize the text", "Chapter 2, §2.2–2.6 — Working with text data",
      "A language model never sees words — it sees integer token IDs. The text is split " +
      "into tokens, and each token is looked up in a fixed vocabulary. Words the model " +
      "was never trained on fall back to <|unk|>. (GPT-2 avoids unknown tokens entirely " +
      "by using byte-pair encoding; this toy model uses a word-level vocabulary of " +
      Model.vocab().length + " tokens so every step stays inspectable.)"));

    const controls = Viz.el("div", { class: "controls" },
      Viz.el("button", { class: "btn", text: "▶ Replay", onclick: () => play() }));
    const row = Viz.el("div", { class: "chip-row" });
    root.append(controls, Viz.el("div", { class: "viz" }, row));

    // vocabulary panel
    const vocabWrap = Viz.el("div");
    vocabWrap.append(Viz.el("div", { class: "mat-title", style: { marginTop: "18px" }, text: "The entire vocabulary" }));
    const grid = Viz.el("div", { class: "vocab-grid" });
    const usedIds = new Set(App.state.ids);
    Model.vocab().forEach((w, i) => {
      grid.append(Viz.el("span", {
        class: "v" + (usedIds.has(i) ? " used" : ""),
        text: `${w} · ${i}`,
      }));
    });
    vocabWrap.append(grid);
    root.append(vocabWrap, footnote(
      "PyTorch twin: <code>SimpleTokenizerV2</code> in <code>transformer_builder/ch02_data.py</code> " +
      "(listing 2.4). Try typing a word that isn't highlighted below — it becomes <code>&lt;|unk|&gt;</code>."));

    function play() {
      row.replaceChildren();
      App.state.tokens.forEach((t, i) => {
        const c = Viz.chip(t, "pop");
        c.style.animationDelay = `${i * 90}ms`;
        row.append(c);
      });
    }
    play();
  },
});

// ===========================================================================
// Stage 2 — Embeddings
// ===========================================================================

Stages.push({
  name: "Embed",
  chapter: "Ch. 2",
  render(root) {
    root.append(stageHeader(2, "Token + positional embeddings", "Chapter 2, §2.7–2.8 — Encoding word positions",
      "Each token ID selects a row of the embedding matrix — a learned 32-dimensional vector. " +
      "Because attention alone has no sense of order, a second embedding indexed by position " +
      "is added, so identical tokens at different positions get different input vectors."));

    const controls = Viz.el("div", { class: "controls" },
      Viz.el("button", { class: "btn", text: "▶ Replay", onclick: () => play() }));
    const viz = Viz.el("div", { class: "viz" });
    root.append(controls, viz);

    const trace = App.state.trace;
    const vmax = Math.max(Viz.maxAbs(trace.tokEmb), Viz.maxAbs(trace.posEmb), Viz.maxAbs(trace.x0));

    function play() {
      viz.replaceChildren();
      const tbl = Viz.el("table", { class: "emb-table" });
      tbl.append(Viz.el("tr", {},
        Viz.el("th"), Viz.el("th", { text: "token embedding  tok_emb[id]" }),
        Viz.el("th"), Viz.el("th", { text: "positional  pos_emb[pos]" }),
        Viz.el("th"), Viz.el("th", { text: "input vector  x = tok + pos" })));

      App.state.tokens.forEach((t, i) => {
        const tr = Viz.el("tr");
        const chipCell = Viz.el("td");
        chipCell.append(Viz.chip(t));
        const tdTok = Viz.el("td");
        const tdPos = Viz.el("td");
        const tdSum = Viz.el("td");
        tr.append(chipCell, tdTok, Viz.el("td", { class: "op", text: "+" }), tdPos,
          Viz.el("td", { class: "op", text: "=" }), tdSum);
        tbl.append(tr);

        const base = i * 260;
        setTimeout(() => tdTok.append(Viz.strip(trace.tokEmb[i], vmax, { label: `tok_emb[${t.id}]`, animate: true })), base);
        setTimeout(() => tdPos.append(Viz.strip(trace.posEmb[i], vmax, { label: `pos_emb[${i}]`, animate: true })), base + 110);
        setTimeout(() => tdSum.append(Viz.strip(trace.x0[i], vmax, { label: `x[${i}]`, animate: true })), base + 220);
      });
      viz.append(tbl);
    }
    play();

    root.append(Viz.legendRamp("div", "negative", "positive"));
    root.append(footnote(
      "Each strip is one real 32-dimensional vector from the trained model — hover any cell " +
      "for its value. PyTorch twin: <code>tok_emb</code> / <code>pos_emb</code> lookups in " +
      "<code>GPTModel.forward</code> (<code>transformer_builder/ch04_gpt.py</code>)."));
  },
});

// ===========================================================================
// Stage 3 — Self-attention, step by step
// ===========================================================================

Stages.push({
  name: "Attention",
  chapter: "Ch. 3",
  render(root) {
    root.append(stageHeader(3, "Causal self-attention, step by step",
      "Chapter 3, §3.4–3.5 — From simple to causal attention",
      "The heart of the transformer. Every token builds a query, a key, and a value; " +
      "query·key dot products decide how much each token attends to the others, a causal " +
      "mask hides the future, softmax turns scores into weights, and the output is a " +
      "weight-blended mix of value vectors."));

    let head = 0;
    let step = 0;
    let timer = null;

    const trace = App.state.trace;
    const labels = tokenLabels();
    const nSteps = 6;

    const captions = [
      "Step 0 — The inputs. Each row is one token's 32-d vector (after layer norm), the x that attention reads.",
      "Step 1 — Project. Multiplying x by the learned matrices W_q, W_k, W_v gives each token a query, key, and value (8 dimensions per head). The query asks; the key advertises; the value carries the content.",
      "Step 2 — Score. Every query is dotted with every key: scores = Q·Kᵀ/√d_k. Cell [i][j] says how relevant token j looks to token i. The √d_k keeps dot products from swamping softmax.",
      "Step 3 — Mask the future. Position i may not look at positions j > i — otherwise training on “predict the next token” would be cheating. Future cells are set to −∞ so softmax gives them exactly zero.",
      "Step 4 — Softmax. Each row becomes a probability distribution: non-negative, summing to 1. These are the attention weights — hover a cell to read “how much token i attends to token j”.",
      "Step 5 — Blend. context = weights · V: each token's output is a weighted average of the value vectors it attends to. The arcs below show where the selected token gathers its information from.",
    ];

    const formula = Viz.el("div", { class: "formula" });
    const caption = Viz.el("div", { class: "step-caption" });
    const dots = Viz.el("span", { class: "step-dots" });
    for (let i = 0; i < nSteps; i++) dots.append(Viz.el("i"));

    const btnPrev = Viz.el("button", { class: "btn secondary", text: "◀ Back", onclick: () => show(step - 1) });
    const btnNext = Viz.el("button", { class: "btn", text: "Step ▶", onclick: () => show(step + 1) });
    const btnPlay = Viz.el("button", { class: "btn secondary", text: "▶ Play all", onclick: playAll });

    const headSel = Viz.el("select", {
      onchange: (e) => { head = +e.target.value; show(step); },
    });
    for (let h = 0; h < Model.cfg().n_heads; h++) {
      headSel.append(Viz.el("option", { value: h, text: `head ${h + 1}` }));
    }

    const viz = Viz.el("div", { class: "viz" });
    root.append(
      formula,
      Viz.el("div", { class: "controls" },
        btnPrev, btnNext, btnPlay, dots,
        Viz.el("span", { class: "spacer" }),
        Viz.el("label", { text: "block 1, " }), headSel),
      caption,
      viz,
      footnote("PyTorch twin: <code>CausalAttention</code> / <code>MultiHeadAttention</code> in " +
        "<code>transformer_builder/ch03_attention.py</code> (listings 3.3 and 3.5). " +
        "The matrices shown are block 1's real Q/K/V for the head selected above."));

    function playAll() {
      clearTimeout(timer);
      show(0);
      let s = 0;
      const tick = () => {
        if (s < nSteps - 1) {
          s += 1;
          show(s);
          timer = setTimeout(tick, 2300);
        }
      };
      timer = setTimeout(tick, 1600);
    }

    function fPart(i, txt) {
      return step === i ? `<span class="hot">${txt}</span>` : txt;
    }

    function show(s) {
      clearTimeout(timer);
      step = Math.min(Math.max(s, 0), nSteps - 1);
      btnPrev.disabled = step === 0;
      btnNext.disabled = step === nSteps - 1;
      [...dots.children].forEach((d, i) => d.classList.toggle("on", i <= step));
      caption.textContent = captions[step];
      formula.innerHTML =
        `context = ${fPart(4, "softmax")}( ${fPart(2, "Q·Kᵀ / √d_k")} ${fPart(3, "+ mask")} ) ${fPart(5, "· V")}` +
        `   where   ${fPart(1, "Q = x·W_q,  K = x·W_k,  V = x·W_v")}` +
        (step === 0 ? `   <span class="hot">x = layer-normed input</span>` : "");

      viz.replaceChildren();
      const b = trace.blocks[0];
      const hd = b.heads[head];
      const grid = Viz.el("div", { class: "attn-grid" });
      viz.append(grid);

      if (step === 0) {
        grid.append(Viz.matrix(b.norm1, {
          title: "x — layer-normed input", dims: `${labels.length}×32`,
          rowLabels: labels, cellSize: 13, name: "x",
        }));
      }

      if (step === 1) {
        for (const [nm, M] of [["Q — queries", hd.q], ["K — keys", hd.k], ["V — values", hd.v]]) {
          const m = Viz.matrix(M, {
            title: nm, dims: `${labels.length}×8`, rowLabels: labels,
            cellSize: 22, name: nm[0],
          });
          grid.append(m);
          Viz.revealCells(m, 8);
        }
      }

      if (step >= 2 && step <= 4) {
        const opts = {
          dims: `${labels.length}×${labels.length}`,
          rowLabels: labels, colLabels: labels, showValues: true, cellSize: 40, name: "scores",
        };
        let m;
        if (step === 2) {
          m = Viz.matrix(hd.scores, { ...opts, title: "attention scores = Q·Kᵀ/√8" });
          Viz.revealCells(m, 45);
        } else if (step === 3) {
          m = Viz.matrix(hd.scores, { ...opts, title: "scores + causal mask", maskedFrom: 0 });
        } else {
          m = Viz.matrix(hd.weights, {
            ...opts, mode: "seq", name: "weights",
            title: "attention weights = softmax(masked scores)",
            valueFmt: (v) => (v >= 0.995 ? "1" : v.toFixed(2).slice(1)),
          });
          Viz.revealCells(m, 25);
        }
        grid.append(m);
        if (step === 4) {
          const lg = Viz.legendRamp("seq", "0", "1  (each row sums to 1)");
          grid.append(Viz.el("div", {}, lg));
        }
      }

      if (step === 5) {
        const m = Viz.matrix(hd.context, {
          title: "context = weights · V", dims: `${labels.length}×8`,
          rowLabels: labels, cellSize: 22, name: "ctx",
        });
        grid.append(m);
        Viz.revealCells(m, 10);
        grid.append(arcsView(hd.weights, labels));
      }
    }

    // Attention arcs: pick a query token, arcs show its weights over keys.
    function arcsView(weights, labels) {
      const wrap = Viz.el("div");
      wrap.append(Viz.el("div", { class: "mat-title", text: "where does each token look? (click a token)" }));
      let qi = labels.length - 1;

      const W = Math.max(labels.length * 86, 300), H = 150;
      const svg = Viz.el("svg", { class: "arcs", width: W, height: H, viewBox: `0 0 ${W} ${H}` });
      wrap.append(svg);

      function draw() {
        svg.replaceChildren();
        const xs = labels.map((_, i) => 43 + i * 86);
        const baseY = H - 28;
        for (let j = 0; j <= qi; j++) {
          const w = weights[qi][j];
          if (j !== qi) {
            const x1 = xs[j], x2 = xs[qi];
            const mid = (x1 + x2) / 2;
            const lift = Math.min(90, 24 + Math.abs(x2 - x1) * 0.28);
            svg.append(Viz.el("path", {
              d: `M ${x1} ${baseY - 14} Q ${mid} ${baseY - 14 - lift} ${x2} ${baseY - 14}`,
              "stroke-width": Math.max(0.6, w * 9),
              opacity: Math.max(0.18, w),
            }));
          }
        }
        labels.forEach((l, i) => {
          const t = Viz.el("text", {
            x: xs[i], y: baseY, "text-anchor": "middle",
            class: i === qi ? "q-label" : "",
            style: { cursor: "pointer", opacity: i > qi ? 0.3 : 1 },
            text: l,
          });
          t.addEventListener("click", () => { qi = i; draw(); });
          svg.append(t);
          if (i <= qi) {
            svg.append(Viz.el("text", {
              x: xs[i], y: baseY + 17, "text-anchor": "middle",
              style: { fontSize: "10.5px", fontWeight: 400, fill: "var(--text-muted)" },
              text: (weights[qi][i] * 100).toFixed(0) + "%",
            }));
          }
        });
      }
      draw();
      return wrap;
    }

    show(0);
  },
});

// ===========================================================================
// Stage 4 — Multi-head attention
// ===========================================================================

Stages.push({
  name: "Multi-head",
  chapter: "Ch. 3",
  render(root) {
    root.append(stageHeader(4, "Multi-head attention", "Chapter 3, §3.6 — Extending single-head to multi-head",
      "Instead of one attention pattern, the 32 dimensions are split into 4 heads of 8, " +
      "each with its own Q/K/V slice — so each head can learn to look at the sequence " +
      "differently. Their outputs are concatenated back together and mixed by an output " +
      "projection. Compare the four weight maps: they are genuinely different."));

    const trace = App.state.trace;
    const labels = tokenLabels();
    const H = Model.cfg().n_heads;

    let blockIdx = 0;
    const blockSel = Viz.el("select", {
      onchange: (e) => { blockIdx = +e.target.value; draw(); },
    });
    trace.blocks.forEach((_, i) => blockSel.append(Viz.el("option", { value: i, text: `block ${i + 1}` })));

    const controls = Viz.el("div", { class: "controls" },
      Viz.el("button", { class: "btn", text: "▶ Replay", onclick: draw }),
      Viz.el("span", { class: "spacer" }),
      Viz.el("label", { text: "layer:" }), blockSel);
    const viz = Viz.el("div", { class: "viz" });
    root.append(controls, viz, footnote(
      "PyTorch twin: <code>MultiHeadAttention</code> with weight splitting in " +
      "<code>transformer_builder/ch03_attention.py</code> (listing 3.5) — one big Q/K/V " +
      "projection reshaped into heads, exactly as visualized here."));

    function draw() {
      viz.replaceChildren();
      const b = trace.blocks[blockIdx];

      const row = Viz.el("div", { class: "heads-row" });
      b.heads.forEach((hd, h) => {
        const card = Viz.el("div", {
          class: "head-card",
          style: { "--hc": Viz.headColor(h), opacity: 0, transition: "opacity 0.4s" },
        });
        card.append(Viz.el("div", { class: "hname" }, Viz.el("i"), `head ${h + 1} — attention weights`));
        card.append(Viz.matrix(hd.weights, {
          rowLabels: labels, colLabels: labels, mode: "seq", cellSize: 20, name: `w(h${h + 1})`,
        }));
        row.append(card);
        setTimeout(() => { card.style.opacity = 1; }, 250 + h * 350);
      });
      viz.append(row);

      // concat + projection for the last token
      const ti = labels.length - 1;
      const concatWrap = Viz.el("div", { style: { marginTop: "22px" } });
      concatWrap.append(Viz.el("div", {
        class: "mat-title",
        text: `concatenate the heads for "${labels[ti]}" (last token), then project`,
      }));

      const vmax = Viz.maxAbs(b.concat);
      const strip = Viz.el("div", { class: "concat-strip" });
      const segLabels = Viz.el("div", { class: "seg-label-row" });
      b.heads.forEach((hd, h) => {
        const seg = Viz.el("span", {
          class: "seg",
          style: { borderBottom: `3px solid ${Viz.headColor(h)}` },
        });
        hd.context[ti].forEach((v, k) => {
          const cell = Viz.el("i", { style: { background: Viz.div(v, vmax) } });
          Viz.hoverable(cell, () => `head ${h + 1} ctx[${k}] = ${v.toFixed(3)}`);
          seg.append(cell);
        });
        strip.append(seg);
        segLabels.append(Viz.el("span", {
          text: `head ${h + 1}`,
          style: { width: `${8 * 7 + 2}px`, textAlign: "center" },
        }));
      });

      const outRow = Viz.el("div", { class: "emb-table-like", style: { display: "flex", alignItems: "center", gap: "10px", marginTop: "8px", flexWrap: "wrap" } });
      outRow.append(
        Viz.el("div", {}, strip, segLabels),
        Viz.el("span", { class: "op", style: { color: "var(--text-muted)", fontWeight: 600 }, text: "· W_out =" }),
        Viz.strip(b.attnOut[ti], Viz.maxAbs(b.attnOut), { label: "attn_out", animate: true }));
      concatWrap.append(outRow);
      viz.append(concatWrap);
    }
    draw();
  },
});

// ===========================================================================
// Stage 5 — The transformer block
// ===========================================================================

Stages.push({
  name: "Block",
  chapter: "Ch. 4",
  render(root) {
    root.append(stageHeader(5, "Inside a transformer block", "Chapter 4, §4.5–4.6 — Shortcut connections and the block",
      "A block wires the pieces together: layer norm → masked multi-head attention → " +
      "add the shortcut, then layer norm → feed-forward network (expand 32→128, GELU, " +
      "contract back) → add the shortcut again. The GPT stacks 2 of these blocks " +
      "(GPT-2 small stacks 12). Press Run to follow one token through the block."));

    const trace = App.state.trace;
    const labels = tokenLabels();
    let ti = labels.length - 1;
    let running = false;

    const tokSel = Viz.el("select", { onchange: (e) => { ti = +e.target.value; drawStrips(-1); } });
    labels.forEach((l, i) => tokSel.append(Viz.el("option", { value: i, text: `"${l}" (pos ${i})`, selected: i === ti ? "" : null })));

    const runBtn = Viz.el("button", { class: "btn", text: "▶ Run the block", onclick: run });
    root.append(Viz.el("div", { class: "controls" },
      runBtn, Viz.el("span", { class: "spacer" }),
      Viz.el("label", { text: "follow token:" }), tokSel));

    // --- flow diagram ---
    const W = 560, Hh = 470;
    const nodes = [
      { id: "in",    y: 26,  label: "input x", small: "(from embeddings / previous block)" },
      { id: "ln1",   y: 88,  label: "LayerNorm 1" },
      { id: "mha",   y: 150, label: "Masked multi-head attention", small: "4 heads × 8 dims" },
      { id: "add1",  y: 212, label: "⊕  + shortcut" },
      { id: "ln2",   y: 274, label: "LayerNorm 2" },
      { id: "ffn",   y: 336, label: "Feed forward", small: "Linear 32→128 · GELU · Linear 128→32" },
      { id: "add2",  y: 398, label: "⊕  + shortcut" },
      { id: "out",   y: 448, label: "output → next block" },
    ];
    const cx = 240, nodeW = 300, nodeH = 40;

    const svg = Viz.el("svg", { class: "flow", width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}` });
    // main wire
    svg.append(Viz.el("line", { class: "wire", x1: cx, y1: nodes[0].y, x2: cx, y2: nodes[nodes.length - 1].y }));
    // residual wires (right side bypasses)
    const res1 = `M ${cx + nodeW / 2 - 40} ${nodes[0].y + 14} H ${cx + nodeW / 2 + 46} V ${nodes[3].y} H ${cx + nodeW / 2}`;
    const res2 = `M ${cx + nodeW / 2 - 40} ${nodes[3].y + 14} H ${cx + nodeW / 2 + 46} V ${nodes[6].y} H ${cx + nodeW / 2}`;
    svg.append(Viz.el("path", { class: "wire", d: res1, "stroke-dasharray": "4 4" }));
    svg.append(Viz.el("path", { class: "wire", d: res2, "stroke-dasharray": "4 4" }));
    svg.append(Viz.el("text", { class: "small", x: cx + nodeW / 2 + 52, y: (nodes[0].y + nodes[3].y) / 2, text: "shortcut" }));
    svg.append(Viz.el("text", { class: "small", x: cx + nodeW / 2 + 52, y: (nodes[3].y + nodes[6].y) / 2, text: "shortcut" }));

    const nodeEls = {};
    nodes.forEach((n) => {
      const g = Viz.el("g");
      const isOp = n.id.startsWith("add") || n.id === "in" || n.id === "out";
      const w = isOp ? 190 : nodeW;
      const rect = Viz.el("rect", {
        class: "node", x: cx - w / 2, y: n.y - nodeH / 2 + 7, width: w, height: nodeH - (n.small ? 0 : 8),
        rx: 9,
      });
      g.append(rect);
      g.append(Viz.el("text", { x: cx, y: n.y + (n.small ? 3 : 8), "text-anchor": "middle", text: n.label }));
      if (n.small) g.append(Viz.el("text", { class: "small", x: cx, y: n.y + 18, "text-anchor": "middle", text: n.small }));
      svg.append(g);
      nodeEls[n.id] = rect;
    });
    const pulse = Viz.el("circle", { class: "pulse", r: 6, cx, cy: nodes[0].y, opacity: 0 });
    svg.append(pulse);

    root.append(Viz.el("div", { class: "viz" }, svg));

    // --- checkpoint strips ---
    const strips = Viz.el("div", { style: { marginTop: "6px" } });
    root.append(strips);
    root.append(footnote(
      "Without the shortcut connections, gradients vanish as blocks stack (§4.4); with them, " +
      "each block only has to learn a <em>refinement</em> of its input. PyTorch twin: " +
      "<code>TransformerBlock</code> in <code>transformer_builder/ch04_gpt.py</code> (listing 4.6)."));

    const checkpoints = [
      { id: "in",   label: "x in", get: (b) => b.input },
      { id: "add1", label: "after attention + shortcut", get: (b) => b.afterAttn },
      { id: "add2", label: "after feed-forward + shortcut (block 1 out)", get: (b) => b.output },
      { id: "out2", label: "after block 2", get: () => trace.blocks[1].output },
    ];

    function drawStrips(upto) {
      strips.replaceChildren();
      const b = trace.blocks[0];
      const vmax = Math.max(...checkpoints.map((c) => Viz.maxAbs(c.get(b))));
      checkpoints.forEach((c, i) => {
        const row = Viz.el("div", { class: "checkpoint-row" });
        const lbl = Viz.el("span", { style: { minWidth: "300px" }, text: `"${labels[ti]}" — ${c.label}:` });
        row.append(lbl);
        if (upto < 0 || i <= upto) {
          row.append(Viz.strip(c.get(b)[ti], vmax, { label: c.label, animate: upto >= 0 && i === upto }));
        } else {
          row.append(Viz.el("span", { style: { color: "var(--text-muted)" }, text: "…" }));
        }
        strips.append(row);
      });
    }

    function run() {
      if (running) return;
      running = true;
      runBtn.disabled = true;
      drawStrips(0);
      const order = ["in", "ln1", "mha", "add1", "ln2", "ffn", "add2", "out"];
      const cpAt = { in: 0, add1: 1, add2: 2, out: 3 };
      let i = 0;
      pulse.setAttribute("opacity", 1);

      const stepDur = 620;
      const tick = () => {
        Object.values(nodeEls).forEach((r) => r.classList.remove("hot"));
        if (i >= order.length) {
          pulse.setAttribute("opacity", 0);
          running = false;
          runBtn.disabled = false;
          return;
        }
        const id = order[i];
        const node = nodes.find((n) => n.id === id);
        nodeEls[id].classList.add("hot");
        animatePulse(+pulse.getAttribute("cy"), node.y, stepDur * 0.7);
        if (id in cpAt) drawStrips(cpAt[id]);
        i += 1;
        setTimeout(tick, stepDur);
      };
      tick();
    }

    function animatePulse(fromY, toY, dur) {
      const t0 = performance.now();
      const frame = (t) => {
        const k = Math.min((t - t0) / dur, 1);
        pulse.setAttribute("cy", fromY + (toY - fromY) * (k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2));
        if (k < 1) requestAnimationFrame(frame);
      };
      requestAnimationFrame(frame);
    }

    drawStrips(-1);
  },
});

// ===========================================================================
// Stage 6 — Generate
// ===========================================================================

Stages.push({
  name: "Generate",
  chapter: "Ch. 5",
  render(root) {
    root.append(stageHeader(6, "Generate, one token at a time",
      "Chapter 5, §5.1 & 5.3 — Decoding strategies: temperature and top-k",
      "The full loop: run the sequence through the model, read the last position's logits, " +
      "turn them into probabilities, pick the next token, append it, repeat. Temperature " +
      "reshapes the distribution (0 = always take the argmax; higher = more adventurous), " +
      "and top-k throws away everything but the k most likely tokens before sampling."));

    let seq = App.state.tokens.map((t) => ({ ...t }));
    const promptLen = seq.length;
    let temperature = 1.0;
    let topK = Model.cfg().vocab_size;
    let autoTimer = null;

    // --- controls ---
    const tempVal = Viz.el("span", { class: "val", text: "1.0" });
    const tempSlider = Viz.el("input", {
      type: "range", min: 0, max: 2, step: 0.1, value: 1,
      oninput: (e) => { temperature = +e.target.value; tempVal.textContent = temperature.toFixed(1); refresh(); },
    });
    const kVal = Viz.el("span", { class: "val", text: "off" });
    const kSlider = Viz.el("input", {
      type: "range", min: 1, max: Model.cfg().vocab_size, step: 1, value: Model.cfg().vocab_size,
      oninput: (e) => {
        topK = +e.target.value;
        kVal.textContent = topK >= Model.cfg().vocab_size ? "off" : String(topK);
        refresh();
      },
    });

    const stepBtn = Viz.el("button", { class: "btn", text: "Sample next ▶", onclick: () => stepOnce() });
    const autoBtn = Viz.el("button", { class: "btn secondary", text: "▶▶ Auto-generate", onclick: toggleAuto });
    const resetBtn = Viz.el("button", { class: "btn secondary", text: "↺ Reset", onclick: reset });

    root.append(Viz.el("div", { class: "controls" },
      stepBtn, autoBtn, resetBtn,
      Viz.el("span", { class: "spacer" }),
      Viz.el("div", { class: "slider-group" }, Viz.el("label", { text: "temperature" }), tempSlider, tempVal),
      Viz.el("div", { class: "slider-group" }, Viz.el("label", { text: "top-k" }), kSlider, kVal)));

    // --- layout: sequence | distribution ---
    const seqTitle = Viz.el("div", { class: "mat-title", text: "sequence" });
    const chipRow = Viz.el("div", { class: "chip-row" });
    const note = Viz.el("div", { class: "gen-note" });
    const left = Viz.el("div", {}, seqTitle, chipRow, note);

    const barsTitle = Viz.el("div", { class: "mat-title", text: "P(next token) — top 10 of 37" });
    const bars = Viz.el("div", { class: "bars" });
    const right = Viz.el("div", {}, barsTitle, bars);

    root.append(Viz.el("div", { class: "gen-layout" }, left, right));
    root.append(footnote(
      "The bars are the model's real output distribution — drag the sliders and watch it " +
      "reshape. PyTorch twin: <code>generate()</code> in <code>transformer_builder/ch05_train.py</code> " +
      "(listing 5.4): top-k filter → divide logits by temperature → softmax → multinomial sample."));

    let lastProbs = null;
    let lastLogits = null;

    function currentDistribution() {
      const ctx = seq.slice(-Model.cfg().context_length).map((t) => t.id);
      const trace = Model.forward(ctx);
      lastLogits = trace.logits[trace.logits.length - 1];
      lastProbs = Model.nextTokenDistribution(lastLogits, temperature, topK);
      return lastProbs;
    }

    function drawChips(newIdx = -1) {
      chipRow.replaceChildren();
      seq.forEach((t, i) => {
        const c = Viz.chip(t, i >= promptLen ? "gen" : "");
        if (i === newIdx) c.classList.add("pop", "flash");
        chipRow.append(c);
      });
      const over = seq.length - Model.cfg().context_length;
      note.textContent = over > 0
        ? `Context window is ${Model.cfg().context_length} tokens — the first ${over} token${over > 1 ? "s are" : " is"} cropped, exactly like idx[:, -context_size:] in the book.`
        : `Generated tokens are underlined in blue. Context used: ${seq.length}/${Model.cfg().context_length} tokens.`;
    }

    function drawBars(sampledId = -1) {
      const probs = lastProbs;
      const vocab = Model.vocab();
      const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 10);
      // keep the sampled token visible even if it fell out of the top 10
      if (sampledId >= 0 && !order.some(([, i]) => i === sampledId)) {
        order[order.length - 1] = [probs[sampledId], sampledId];
      }
      bars.replaceChildren();
      const kth = topK < vocab.length
        ? lastLogits.slice().sort((a, b) => b - a)[topK - 1]
        : -Infinity;
      order.forEach(([p, i]) => {
        const cut = lastLogits[i] < kth;
        const row = Viz.el("div", { class: `bar-row${cut ? " cut" : ""}${i === sampledId ? " sampled" : ""}` });
        const fill = Viz.el("div", { class: "fill" });
        row.append(
          Viz.el("span", { class: "tok", text: vocab[i] }),
          Viz.el("div", { class: "track" }, fill),
          Viz.el("span", { class: "p", text: (p * 100).toFixed(1) + "%" }));
        Viz.hoverable(row, () =>
          `"${vocab[i]}"\nlogit = ${lastLogits[i].toFixed(3)}\nP = ${(p * 100).toFixed(2)}%${cut ? "\n(removed by top-k)" : ""}`);
        bars.append(row);
        requestAnimationFrame(() => { fill.style.width = (p * 100).toFixed(2) + "%"; });
      });
    }

    function refresh() {
      currentDistribution();
      drawBars();
    }

    function stepOnce() {
      currentDistribution();
      const id = Model.sample(lastProbs);
      drawBars(id);
      setTimeout(() => {
        seq.push({ word: Model.vocab()[id], token: Model.vocab()[id], id, unk: false });
        drawChips(seq.length - 1);
        refresh();
      }, 620);
    }

    function toggleAuto() {
      if (autoTimer) {
        clearInterval(autoTimer);
        autoTimer = null;
        autoBtn.textContent = "▶▶ Auto-generate";
        return;
      }
      autoBtn.textContent = "⏸ Stop";
      autoTimer = setInterval(() => {
        if (seq.length - promptLen >= 24) { toggleAuto(); return; }
        stepOnce();
      }, 1250);
    }

    function reset() {
      if (autoTimer) toggleAuto();
      seq = App.state.tokens.map((t) => ({ ...t }));
      drawChips();
      refresh();
    }

    drawChips();
    refresh();

    this.cleanup = () => { if (autoTimer) toggleAuto(); };
  },
});
