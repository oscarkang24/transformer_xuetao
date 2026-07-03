// App shell: shared state, prompt input, stage navigation, theme toggle.

const App = {
  state: {
    text: "every effort moves you",
    tokens: [],
    ids: [],
    trace: null,
    stage: 0,
  },

  recompute() {
    const cfg = Model.cfg();
    let tokens = Tokenizer.encode(App.state.text, Model.vocab());
    if (tokens.length === 0) {
      tokens = Tokenizer.encode("every effort moves you", Model.vocab());
    }
    tokens = tokens.slice(0, cfg.context_length);
    App.state.tokens = tokens;
    App.state.ids = tokens.map((t) => t.id);
    App.state.trace = Model.forward(App.state.ids);
  },

  renderStage() {
    const root = document.getElementById("stage-root");
    for (const s of Stages) if (s.cleanup) { s.cleanup(); s.cleanup = null; }
    root.replaceChildren();
    Stages[App.state.stage].render(root);

    document.querySelectorAll("nav.stepper button").forEach((b, i) => {
      b.classList.toggle("active", i === App.state.stage);
    });
    document.getElementById("btn-prev").disabled = App.state.stage === 0;
    document.getElementById("btn-next").disabled = App.state.stage === Stages.length - 1;
  },

  goto(i) {
    App.state.stage = Math.min(Math.max(i, 0), Stages.length - 1);
    App.renderStage();
  },

  init() {
    // stepper
    const nav = document.getElementById("stepper");
    Stages.forEach((s, i) => {
      nav.append(Viz.el("button", {
        onclick: () => App.goto(i),
      },
        Viz.el("span", { class: "n", text: `${i + 1} · ${s.chapter}` }),
        Viz.el("span", { class: "t", text: s.name })));
    });

    document.getElementById("btn-prev").addEventListener("click", () => App.goto(App.state.stage - 1));
    document.getElementById("btn-next").addEventListener("click", () => App.goto(App.state.stage + 1));

    // prompt input
    const input = document.getElementById("prompt-input");
    input.value = App.state.text;
    let debounce = null;
    input.addEventListener("input", () => {
      clearTimeout(debounce);
      debounce = setTimeout(() => {
        App.state.text = input.value;
        App.recompute();
        App.renderStage();
      }, 350);
    });

    // clickable vocabulary hint
    const hint = document.getElementById("vocab-hint");
    const samples = ["every effort moves you", "the dog chased the cat", "the bird flew over the house", "you build the model step by step"];
    hint.append("Try: ");
    samples.forEach((s, i) => {
      hint.append(Viz.el("span", {
        class: "w", text: `“${s}”`,
        onclick: () => {
          input.value = s;
          App.state.text = s;
          App.recompute();
          App.renderStage();
        },
      }));
      if (i < samples.length - 1) hint.append("  ·  ");
    });
    hint.append(` — the model knows ${Model.vocab().length} words (full list in stage 1).`);

    // theme toggle: auto -> light -> dark
    const themeBtn = document.getElementById("theme-toggle");
    const labels = { "": "theme: auto", light: "theme: light", dark: "theme: dark" };
    themeBtn.addEventListener("click", () => {
      const cur = document.documentElement.dataset.theme || "";
      const next = cur === "" ? "light" : cur === "light" ? "dark" : "";
      if (next) document.documentElement.dataset.theme = next;
      else delete document.documentElement.dataset.theme;
      themeBtn.textContent = labels[next];
      App.renderStage(); // re-render so JS-computed colors follow the theme
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      if (!document.documentElement.dataset.theme) App.renderStage();
    });

    App.recompute();
    App.renderStage();
  },
};

document.addEventListener("DOMContentLoaded", App.init);
