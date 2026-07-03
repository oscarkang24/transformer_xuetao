// Minimal tensor math for in-browser GPT inference.
// Matrices are plain arrays of row arrays: A[i][j]. Vectors are arrays.

const T = {
  zeros(n) {
    return new Array(n).fill(0);
  },

  // (n,k) @ (k,m) -> (n,m)
  matmul(A, B) {
    const n = A.length, k = B.length, m = B[0].length;
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      const row = new Array(m).fill(0);
      const a = A[i];
      for (let p = 0; p < k; p++) {
        const av = a[p], b = B[p];
        if (av === 0) continue;
        for (let j = 0; j < m; j++) row[j] += av * b[j];
      }
      out[i] = row;
    }
    return out;
  },

  transpose(A) {
    const n = A.length, m = A[0].length;
    const out = new Array(m);
    for (let j = 0; j < m; j++) {
      const row = new Array(n);
      for (let i = 0; i < n; i++) row[i] = A[i][j];
      out[j] = row;
    }
    return out;
  },

  addVec(a, b) {
    return a.map((v, i) => v + b[i]);
  },

  addMat(A, B) {
    return A.map((row, i) => T.addVec(row, B[i]));
  },

  scale(A, s) {
    return A.map((row) => row.map((v) => v * s));
  },

  softmaxRow(row) {
    const max = Math.max(...row.filter((v) => v !== -Infinity));
    const exps = row.map((v) => (v === -Infinity ? 0 : Math.exp(v - max)));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  },

  softmax(A) {
    return A.map(T.softmaxRow);
  },

  // Matches the PyTorch LayerNorm in ch04_gpt.py: biased variance, eps 1e-5.
  layerNorm(x, scale, shift, eps = 1e-5) {
    const n = x.length;
    const mean = x.reduce((a, b) => a + b, 0) / n;
    let variance = 0;
    for (const v of x) variance += (v - mean) ** 2;
    variance /= n;
    const inv = 1 / Math.sqrt(variance + eps);
    return x.map((v, i) => ((v - mean) * inv) * scale[i] + shift[i]);
  },

  // Tanh approximation, matching the book's GELU (listing 4.3).
  gelu(x) {
    const c = Math.sqrt(2 / Math.PI);
    return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x ** 3)));
  },

  geluVec(v) {
    return v.map(T.gelu);
  },
};
