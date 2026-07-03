// The toy GPT's forward pass, mirroring transformer_builder/ch04_gpt.py
// step for step. forward() records every intermediate tensor into a trace
// object so the visualizations can show the real numbers at each stage.

const Model = {
  cfg() {
    return MODEL_WEIGHTS.config;
  },

  vocab() {
    return MODEL_WEIGHTS.vocab;
  },

  // ids: array of token ids, length <= context_length.
  // Returns the full trace of intermediates.
  forward(ids) {
    const W = MODEL_WEIGHTS;
    const cfg = W.config;
    const n = ids.length;

    const tokEmb = ids.map((id) => W.tok_emb[id].slice());
    const posEmb = ids.map((_, i) => W.pos_emb[i].slice());
    const x0 = T.addMat(tokEmb, posEmb);

    const trace = { ids: ids.slice(), tokEmb, posEmb, x0, blocks: [] };

    let x = x0;
    for (const bw of W.blocks) {
      const b = { input: x };

      // --- attention sublayer: LN -> MHA -> residual ---
      b.norm1 = x.map((row) => T.layerNorm(row, bw.norm1_scale, bw.norm1_shift));
      b.q = T.matmul(b.norm1, bw.W_query);
      b.k = T.matmul(b.norm1, bw.W_key);
      b.v = T.matmul(b.norm1, bw.W_value);

      const H = cfg.n_heads;
      const hd = cfg.emb_dim / H;
      b.heads = [];
      for (let h = 0; h < H; h++) {
        const sl = (M) => M.map((row) => row.slice(h * hd, (h + 1) * hd));
        const q = sl(b.q), k = sl(b.k), v = sl(b.v);
        const scores = T.scale(T.matmul(q, T.transpose(k)), 1 / Math.sqrt(hd));
        const masked = scores.map((row, i) =>
          row.map((s, j) => (j > i ? -Infinity : s))
        );
        const weights = T.softmax(masked);
        const context = T.matmul(weights, v);
        b.heads.push({ q, k, v, scores, masked, weights, context });
      }

      // concat heads back to (n, emb_dim), then output projection
      b.concat = [];
      for (let i = 0; i < n; i++) {
        b.concat.push([].concat(...b.heads.map((head) => head.context[i])));
      }
      b.attnOut = T.matmul(b.concat, bw.out_proj_w).map((row) =>
        T.addVec(row, bw.out_proj_b)
      );
      b.afterAttn = T.addMat(b.attnOut, x);

      // --- feed-forward sublayer: LN -> FFN -> residual ---
      b.norm2 = b.afterAttn.map((row) =>
        T.layerNorm(row, bw.norm2_scale, bw.norm2_shift)
      );
      b.ffHidden = T.matmul(b.norm2, bw.ff_w1).map((row) =>
        T.addVec(row, bw.ff_b1)
      );
      b.ffGelu = b.ffHidden.map(T.geluVec);
      b.ffOut = T.matmul(b.ffGelu, bw.ff_w2).map((row) => T.addVec(row, bw.ff_b2));
      b.output = T.addMat(b.ffOut, b.afterAttn);

      trace.blocks.push(b);
      x = b.output;
    }

    trace.finalNorm = x.map((row) =>
      T.layerNorm(row, W.final_norm_scale, W.final_norm_shift)
    );
    trace.logits = T.matmul(trace.finalNorm, W.out_head);
    return trace;
  },

  // Next-token distribution from the last position's logits, with the
  // chapter-5 decoding controls: top-k filtering then temperature scaling.
  nextTokenDistribution(logits, temperature, topK) {
    const V = logits.length;
    let filtered = logits.slice();
    if (topK && topK < V) {
      const kth = logits.slice().sort((a, b) => b - a)[topK - 1];
      filtered = filtered.map((l) => (l < kth ? -Infinity : l));
    }
    if (temperature > 0) {
      return T.softmaxRow(filtered.map((l) => l / temperature));
    }
    // temperature 0 -> greedy: probability 1 on the argmax
    const probs = T.zeros(V);
    probs[filtered.indexOf(Math.max(...filtered))] = 1;
    return probs;
  },

  sample(probs, rand = Math.random) {
    let r = rand();
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return probs.length - 1;
  },
};
