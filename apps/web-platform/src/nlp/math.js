export const unit = (v) => { let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n) || 1; return Float32Array.from(v, (x) => x / n); };
export const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
export const softmax = (a) => { const m = Math.max(...a), e = a.map((v) => Math.exp(v - m)), s = e.reduce((x, y) => x + y, 0); return e.map((v) => v / s); };
// Sinusoidal position encoding added so identical tokens at different positions differ.
const posEnc = (i, d) => Float32Array.from({ length: d }, (_, j) => (j % 2 ? Math.cos : Math.sin)(i / 10000 ** ((2 * Math.floor(j / 2)) / d)));
// Single-head scaled dot-product attention weights: softmax_j( cos(q_i, k_j) / tau ), with Q = K = hidden + position.
export function attention(H, tau) {
  const X = H.map((h, i) => { const p = posEnc(i, h.length), v = unit(h); return unit(Float32Array.from(v, (x, j) => x + 0.15 * p[j] / Math.sqrt(h.length / 2))); });
  return X.map((q) => softmax(X.map((k) => dot(q, k) / tau)));
}
// 2-D PCA by power iteration with deflation (deterministic start vectors).
export function pca2(X) {
  const n = X.length, d = X[0].length, mean = new Float32Array(d);
  X.forEach((x) => x.forEach((v, i) => (mean[i] += v / n)));
  let R = X.map((x) => Float32Array.from(x, (v, i) => v - mean[i])); const comps = [];
  for (let c = 0; c < 2; c++) {
    let v = Float32Array.from({ length: d }, (_, i) => Math.sin(i * 12.9898 + c * 78.233));
    for (let it = 0; it < 60; it++) {
      const w = new Float32Array(d);
      for (const r of R) { const s = dot(r, v); for (let i = 0; i < d; i++) w[i] += s * r[i]; }
      v = unit(w);
    }
    comps.push(v); R = R.map((r) => { const s = dot(r, v); return Float32Array.from(r, (x, i) => x - s * v[i]); });
  }
  return { project: (x) => comps.map((c) => { let s = 0; for (let i = 0; i < d; i++) s += (x[i] - mean[i]) * c[i]; return s; }) };
}
