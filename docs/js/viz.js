// Shared visualization primitives: color scales, matrices, vector strips,
// token chips, and the tooltip layer.
//
// Color roles (see docs/css/style.css for the palette source):
//   signed values (embeddings, Q/K/V, hidden states) -> diverging blue<->red
//   attention weights / probabilities in [0,1]       -> sequential blue
//   head identity                                     -> categorical slots 1-4

const Viz = {
  isDark() {
    const forced = document.documentElement.dataset.theme;
    if (forced === "dark") return true;
    if (forced === "light") return false;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  },

  // --- color scales -------------------------------------------------------

  _hex(h) {
    return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  },

  _lerp(a, b, t) {
    return a.map((v, i) => Math.round(v + (b[i] - v) * t));
  },

  _rgb(c) {
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  },

  _ramp(stops, t) {
    const n = stops.length - 1;
    const x = Math.min(Math.max(t, 0), 1) * n;
    const i = Math.min(Math.floor(x), n - 1);
    return Viz._rgb(Viz._lerp(Viz._hex(stops[i]), Viz._hex(stops[i + 1]), x - i));
  },

  // Sequential blue, light->dark on light surfaces; flipped so that
  // "more" reads as brighter on the dark surface.
  seqStops() {
    return Viz.isDark()
      ? ["#16233a", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"]
      : ["#e8f1fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"];
  },

  seq(t) {
    return Viz._ramp(Viz.seqStops(), t);
  },

  // Diverging blue (negative) <-> gray (zero) <-> red (positive).
  divStops() {
    return Viz.isDark()
      ? ["#3987e5", "#383835", "#e66767"]
      : ["#2a78d6", "#f0efec", "#e34948"];
  },

  div(v, vmax) {
    const t = Math.min(Math.max(v / (2 * vmax) + 0.5, 0), 1);
    return Viz._ramp(Viz.divStops(), t);
  },

  headColor(h) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue(`--head-${h + 1}`).trim();
  },

  maxAbs(mat) {
    let m = 1e-9;
    for (const row of mat) for (const v of row) m = Math.max(m, Math.abs(v));
    return m;
  },

  // --- DOM helper ---------------------------------------------------------

  el(tag, attrs = {}, ...children) {
    const ns = ["svg", "path", "rect", "circle", "text", "line", "g", "defs",
      "marker", "polygon"].includes(tag);
    const node = ns
      ? document.createElementNS("http://www.w3.org/2000/svg", tag)
      : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.setAttribute("class", v);
      else if (k === "text") node.textContent = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
      else node.setAttribute(k, v);
    }
    for (const c of children) if (c != null) node.append(c);
    return node;
  },

  // --- tooltip -------------------------------------------------------------

  _tip: null,

  tip() {
    if (!Viz._tip) {
      Viz._tip = Viz.el("div", { class: "tooltip" });
      document.body.append(Viz._tip);
    }
    return Viz._tip;
  },

  hoverable(node, textFn) {
    node.addEventListener("mousemove", (e) => {
      const tip = Viz.tip();
      tip.textContent = textFn();
      tip.classList.add("show");
      const pad = 12;
      let x = e.clientX + pad, y = e.clientY + pad;
      const r = tip.getBoundingClientRect();
      if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - pad;
      if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - pad;
      tip.style.left = x + "px";
      tip.style.top = y + "px";
    });
    node.addEventListener("mouseleave", () => Viz.tip().classList.remove("show"));
    return node;
  },

  // --- components ----------------------------------------------------------

  chip(tokenInfo, extraClass = "") {
    const c = Viz.el(
      "div",
      { class: `chip ${tokenInfo.unk ? "unk" : ""} ${extraClass}`.trim() },
      Viz.el("span", { text: tokenInfo.token }),
      Viz.el("span", { class: "id", text: `#${tokenInfo.id}` })
    );
    if (tokenInfo.unk) {
      Viz.hoverable(c, () => `"${tokenInfo.word}" is not in the vocabulary\n-> mapped to <|unk|> (id ${tokenInfo.id})`);
    }
    return c;
  },

  // A 1-D vector as a compact color strip (signed -> diverging scale).
  strip(vec, vmax, { label = "v", animate = false } = {}) {
    const s = Viz.el("div", { class: `strip${animate ? " appear" : ""}` });
    vec.forEach((v, i) => {
      const cell = Viz.el("i", { style: { background: Viz.div(v, vmax) } });
      Viz.hoverable(cell, () => `${label}[${i}] = ${v.toFixed(3)}`);
      s.append(cell);
    });
    return s;
  },

  // Matrix as a table of colored cells with row/col labels.
  // mode: "div" (signed) or "seq" ([0,1] weights).
  // showValues: print the number inside each cell.
  matrix(data, {
    title = "",
    dims = "",
    rowLabels = null,
    colLabels = null,
    mode = "div",
    showValues = false,
    cellSize = 30,
    maskedFrom = null,   // if set, cells with col > row render as masked
    valueFmt = (v) => v.toFixed(1),
    name = "M",
  } = {}) {
    const vmax = mode === "div" ? Viz.maxAbs(data) : 1;
    const wrap = Viz.el("div", { class: "mat-wrap" });
    if (title) {
      wrap.append(Viz.el("div", { class: "mat-title", html: `${title} <span class="dim">${dims}</span>` }));
    }
    const tbl = Viz.el("table", { class: "mat" });
    if (colLabels) {
      const tr = Viz.el("tr");
      tr.append(Viz.el("th"));
      colLabels.forEach((l) => tr.append(Viz.el("th", { text: l })));
      tbl.append(tr);
    }
    data.forEach((row, i) => {
      const tr = Viz.el("tr");
      if (rowLabels) tr.append(Viz.el("th", { class: "rowh", text: rowLabels[i] }));
      row.forEach((v, j) => {
        const masked = maskedFrom !== null && j > i;
        const td = Viz.el("td", {
          class: "cell" + (masked ? " masked-cell" : ""),
          style: { width: cellSize + "px", height: cellSize + "px" },
        });
        if (masked) {
          td.textContent = "-∞";
        } else {
          td.style.background = mode === "seq" ? Viz.seq(v) : Viz.div(v, vmax);
          if (showValues) {
            td.textContent = valueFmt(v);
            const t = mode === "seq" ? v : Math.abs(v) / vmax;
            // strong cells get inverted ink; the dark-mode seq ramp runs
            // dark->light, so its strong cells need dark text instead
            const strongInk = mode === "seq" && Viz.isDark() ? "#0b0b0b" : "#fff";
            td.style.color = t > 0.55 ? strongInk : "var(--text-primary)";
          }
        }
        Viz.hoverable(td, () =>
          masked
            ? `${name}[${i}][${j}] = -∞ (future token, masked)`
            : `${name}[${i}][${j}] = ${v.toFixed(4)}`
        );
        tr.append(td);
      });
      tbl.append(tr);
    });
    wrap.append(tbl);
    return wrap;
  },

  // Staggered cell reveal for a matrix built by Viz.matrix.
  revealCells(matWrap, perCell = 18) {
    const cells = [...matWrap.querySelectorAll("td.cell")];
    cells.forEach((c) => c.classList.add("pending"));
    cells.forEach((c, i) => setTimeout(() => c.classList.remove("pending"), i * perCell));
    return cells.length * perCell;
  },

  legendRamp(kind, leftLabel, rightLabel) {
    const stops = kind === "seq" ? Viz.seqStops() : Viz.divStops();
    const ramp = Viz.el("span", { class: "ramp" });
    for (let i = 0; i < 7; i++) {
      ramp.append(Viz.el("i", { style: { background: Viz._ramp(stops, i / 6) } }));
    }
    return Viz.el("div", { class: "legend" },
      Viz.el("span", { text: leftLabel }), ramp, Viz.el("span", { text: rightLabel }));
  },
};
